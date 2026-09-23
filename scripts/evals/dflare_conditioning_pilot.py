#!/usr/bin/env python3
"""Bounded offline conditioning screen; never deploys or claims serving speed.

Use the separate GPU lease wrapper for --phase train. All data/checkpoints stay
in the explicitly supplied, ignored artifact directory. No network calls.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / '.marathon/drafter-training'
SOURCE = Path('/home/deforest/AI/backends/qwen38-runtime-cleanup/source')
sys.path[:0] = [str(TRAINING / 'SpecForge'), str(SOURCE / 'gguf-py')]
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
import gguf

DRAFT = Path('/home/deforest/AI/models/gguf/qwen3.8-27b-dflash2/Qwen3.8-27B-DFlash2-Marathon-R32-Q4_K_M.gguf')
TARGET = Path('/home/deforest/AI/models/gguf/swift-qwen3.8-27b-uncensored-merge/Swift-Qwen3.8-27B-Uncensored-Merge-IQ4_XS.gguf')


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def prepare(out):
    from transformers import Qwen3Config
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel
    config = Qwen3Config.from_json_file(TRAINING / 'stock/config.json')
    with torch.device('meta'):
        draft = DFlash2DraftModel(config)
    reader = gguf.GGUFReader(DRAFT)
    tensors = {t.name: t for t in reader.tensors}
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.DFLASH, 5)
    imported = {}
    for name, original in draft.state_dict().items():
        key = mapping.get_name('model.' + name + ('.weight' if name.endswith('_codebook') else ''), try_suffixes=('.weight', '.bias'))
        t = tensors[key]
        imported[name] = torch.from_numpy(gguf.dequantize(t.data, t.tensor_type).copy()).reshape(original.shape).to(torch.bfloat16)
    assert len(imported) == len(tensors) == 81
    save_file(imported, str(out / 'draft.bf16.safetensors'))
    del imported, tensors, reader, draft
    reader = gguf.GGUFReader(TARGET)
    for name in ('output.weight', 'token_embd.weight'):
        t = next(t for t in reader.tensors if t.name == name)
        rows, width = tuple(reversed(t.shape.tolist()))
        dest = torch.empty(rows, width, dtype=torch.bfloat16)
        data = t.data.reshape(rows, -1)
        for row in range(0, rows, 512):
            values = gguf.dequantize(data[row:row+512], t.tensor_type)
            dest[row:row+512] = torch.from_numpy(values.copy()).reshape(-1, width)
        save_file({'weight': dest}, str(out / (name + '.safetensors')))
        del dest
        print('prepared', name, flush=True)
    save(out / 'weight-provenance.json', {'draft':str(DRAFT),'target':str(TARGET),'draft_tensors':81,'initialization':'dequantized deployed GGUF, BF16 arithmetic; not original BF16 weights'})


def install_conditioning(draft, mode, rank=0):
    """Zero-gated residual. This is DFlare-inspired, not native DFlare."""
    draft.conditioning_mode = mode
    draft.conditioning_enabled = True
    draft.feature_gain = torch.nn.Parameter(torch.zeros(5, 5120, device='cuda', dtype=torch.float32))
    draft.fusion_logits = torch.nn.Parameter(torch.zeros(5, 5, device='cuda', dtype=torch.float32), requires_grad=mode == 'per_layer')
    draft.conditioning_rank = rank
    if rank:
        draft.feature_gain.requires_grad_(False)
        draft.conditioning_down = torch.nn.ModuleList([
            torch.nn.Linear(5120, rank, bias=False, device='cuda', dtype=torch.float32) for _ in range(5)])
        draft.conditioning_up = torch.nn.ModuleList([
            torch.nn.Linear(rank, 5120, bias=False, device='cuda', dtype=torch.float32) for _ in range(5)])
        for layer in draft.conditioning_up: torch.nn.init.zeros_(layer.weight)
    def forward(self, position_ids, attention_mask=None, noise_embedding=None,
                target_hidden=None, past_key_values=None, use_cache=False, **kwargs):
        hidden = noise_embedding
        base = self.hidden_norm(self.fc(target_hidden))
        raw = target_hidden.reshape(*target_hidden.shape[:-1], 5, 5120)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        for i, (layer_type, layer) in enumerate(zip(self.layer_types, self.layers)):
            context = base
            if self.conditioning_enabled:
                if self.conditioning_mode == 'per_layer':
                    weights = self.fusion_logits[i].softmax(-1).to(raw.dtype)
                    mixed = (raw * weights[None, None, :, None]).sum(-2)
                    residual = F.rms_norm(mixed.float(), (5120,), eps=1e-6).to(base.dtype)
                else:
                    residual = base
                delta = (self.conditioning_up[i](self.conditioning_down[i](residual))
                    if self.conditioning_rank else residual * self.feature_gain[i].to(base.dtype))
                context = base + delta
            mask = attention_mask[layer_type] if isinstance(attention_mask, dict) else attention_mask
            hidden = layer(hidden_states=hidden, target_hidden=context, attention_mask=mask,
                position_ids=position_ids, past_key_value=past_key_values, use_cache=use_cache,
                position_embeddings=position_embeddings, **kwargs)
        return self.norm(hidden)
    draft.forward = types.MethodType(forward, draft)


def cases(out):
    result = []
    for path in sorted((out / 'features').glob('*.json')):
        meta = json.loads(path.read_text())
        ids = torch.from_numpy(np.fromfile(path.with_suffix('.i32'), np.int32).astype(np.int64))
        hidden = torch.from_numpy(np.memmap(path.with_suffix('.bf16'), mode='c', dtype=np.uint16, shape=(len(ids),25600))).view(torch.bfloat16)
        if len(ids) - meta['prompt_tokens'] < 16:
            continue
        mask = torch.zeros(len(ids)); mask[meta['prompt_tokens']:] = 1
        result.append((meta, {'input_ids':ids[None], 'hidden_states':hidden[None], 'loss_mask':mask[None]}))
    return result


def train(out, mode, steps, lr, rank=0, audit=False):
    from transformers import Qwen3Config
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
    run = out / ('audit' if audit else mode + (f'-r{rank}' if rank else ''))
    run.mkdir(exist_ok=False)
    config = Qwen3Config.from_json_file(TRAINING / 'stock/config.json')
    draft = DFlash2DraftModel(config).to(torch.bfloat16)
    draft.load_state_dict(load_file(str(out/'draft.bf16.safetensors')), strict=True)
    draft.requires_grad_(False).cuda()
    draft.block_size = 7
    for module in draft.modules():
        if module.__class__.__name__ == 'DFlashGroupedConv':module.block_size = 7
    original_forward = draft.forward
    install_conditioning(draft, mode, rank)
    draft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    head = torch.nn.Linear(5120,248320,bias=False,device='meta')
    head.load_state_dict(load_file(str(out/'output.weight.safetensors')), assign=True)
    embed = torch.nn.Embedding(248320,5120,device='meta')
    embed.load_state_dict(load_file(str(out/'token_embd.weight.safetensors')), assign=True)
    head.requires_grad_(False).cuda();embed.requires_grad_(False).cuda()
    model = OnlineDFlashModel(draft,head,embed,248070,block_size=7,attention_backend='sdpa',num_anchors=8,objective_chunk_blocks=2,selector_stop_gradient=False,selector_loss_alpha=1.0,loss_type='dflash')
    data = cases(out)
    train_cases = [c for c in data if c[0]['split']=='train']
    val = [c for c in data if c[0]['split']=='validation']
    test = [c for c in data if c[0]['split']=='test']
    assert len(train_cases)>=12 and len(val)>=6 and len(test)>=6
    assert not ({c[0]['thread'] for c in train_cases+val} & {c[0]['thread'] for c in test})
    def batch(c):return {k:v.cuda() for k,v in c[1].items()}
    def evaluate(split, step):
        model.eval();model.num_anchors=32
        rows=[];n=d=0.;losses=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i,c in enumerate(split):
                torch.manual_seed(61000+i)
                loss,_,metrics=model(**batch(c),collect_detailed_metrics=True)
                numerator,denominator=metrics['ratio_metrics']['dflash2/selector/serving_accepted_length']
                num,den=numerator.item(),denominator.item();n+=num;d+=den
                rows.append({'id':c[0]['id'],'accepted_proposals':num/den-1,'anchors':den});losses.append(loss.item())
        r={'step':step,'accepted_proposals':n/d-1,'advanced_tokens_proxy':n/d,'anchors':d,'loss':sum(losses)/len(losses),'cases':rows,'serving_measurement':False}
        print(json.dumps({'mode':mode,'evaluation':step,'accepted_proposals':r['accepted_proposals'],'loss':r['loss']}),flush=True)
        model.train();model.num_anchors=8
        return r
    if audit:
        # Every eligible complete block in the held-out captures, not a fresh
        # random anchor draw. This is still an offline teacher-forced screen.
        def all_anchors(self, seq_len, loss_mask, device, max_valid_anchors=None):
            del max_valid_anchors
            valid=torch.arange(seq_len-self.block_size+1,device=device)
            valid=valid[(loss_mask[0,valid]>0.5) & (loss_mask[0,valid+self.block_size-1]>0.5)]
            positions=valid[None]
            return positions,torch.ones_like(positions,dtype=torch.bool)
        model._sample_anchor_positions=types.MethodType(all_anchors,model)
        draft.conditioning_enabled=False
        results={'baseline':evaluate(test,'all-baseline')}
        for variant in ('per_layer','control'):
            state=load_file(str(out/(variant+'-r16')/'conditioning.safetensors'))
            with torch.no_grad():
                for k,v in state.items():draft.get_parameter(k).copy_(v)
            draft.conditioning_mode=variant;draft.conditioning_enabled=True
            results[variant]=evaluate(test,'all-'+variant)
        save(run/'results.json',results)
        return
    # Verify exact no-op against the original implementation at identical anchors.
    model.eval();model.num_anchors=8
    sample=batch(val[0])
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        torch.manual_seed(817);a=model(**sample,collect_detailed_metrics=True)
        modified_forward=draft.forward;draft.forward=original_forward
        torch.manual_seed(817);b=model(**sample,collect_detailed_metrics=True)
        draft.forward=modified_forward
    parity=torch.equal(a[0],b[0]) and all(torch.equal(x,y) for key,pair in a[2]['ratio_metrics'].items() for x,y in zip(pair,b[2]['ratio_metrics'][key]))
    save(run/'parity.json',{'loss_and_all_ratio_metrics_identical':parity})
    if not parity:raise RuntimeError('Zero-gate baseline parity failed')
    del sample,a,b
    # This isolates PyTorch draft-forward overhead, not llama.cpp serving speed.
    # Keep the same short block and streamed feature rows in both conditions.
    draft.eval()
    micro = dict(position_ids=torch.arange(14, device='cuda')[None],
        noise_embedding=torch.randn(1, 7, 5120, device='cuda', dtype=torch.bfloat16),
        target_hidden=torch.randn(1, 7, 25600, device='cuda', dtype=torch.bfloat16))
    timing = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for enabled in (False, True, True, False):
            draft.conditioning_enabled = enabled
            for _ in range(3): draft(**micro)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(20): draft(**micro)
            end.record(); end.synchronize()
            timing.append({'enabled': enabled, 'draft_forward_ms': begin.elapsed_time(end)/20})
    draft.conditioning_enabled = True
    save(run/'overhead.json', {'order': timing, 'limitation': 'BF16 PyTorch microbenchmark, not serving latency or 196K memory qualification'})
    del micro
    baseline=evaluate(val,0);save(run/'validation-000.json',baseline)
    baseline_test=evaluate(test,'baseline-test');save(run/'test-baseline.json',baseline_test)
    trainable=[p for p in draft.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=lr,weight_decay=0.0)
    best=baseline['advanced_tokens_proxy'];best_step=0
    state_names=[k for k,p in draft.named_parameters() if p.requires_grad]
    def branch_state():return {k:v.detach().cpu().clone() for k,v in draft.named_parameters() if k in state_names}
    best_state=branch_state()
    started=time.monotonic();order=[]
    with (run/'steps.jsonl').open('w') as log:
        for step in range(1,steps+1):
            if (step-1)%len(train_cases)==0:
                order=list(range(len(train_cases)));random.Random(910+(step-1)//len(train_cases)).shuffle(order)
            c=train_cases[order[(step-1)%len(order)]]
            torch.manual_seed(72000+step)
            model.train();model.num_anchors=8;optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_=model(**batch(c),collect_detailed_metrics=False)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
            row={'step':step,'loss':loss.item(),'grad_norm':norm.item(),'seconds':time.monotonic()-started,'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
            log.write(json.dumps(row)+'\n');log.flush()
            if step==1 or step%16==0:print(json.dumps({'mode':mode,**row}),flush=True)
            if step%32==0 or step==steps:
                r=evaluate(val,step);save(run/f'validation-{step:03}.json',r)
                if r['advanced_tokens_proxy']>best:
                    best=r['advanced_tokens_proxy'];best_step=step
                    best_state=branch_state()
    with torch.no_grad():
        for k,v in best_state.items():draft.get_parameter(k).copy_(v)
    result=evaluate(test,'selected-test');save(run/'test-selected.json',result)
    save_file(best_state,str(run/'conditioning.safetensors'))
    save(run/'summary.json',{'mode':mode,'steps':steps,'learning_rate':lr,'best_step':best_step,'trainable_parameters':sum(p.numel() for p in trainable),'baseline_test':baseline_test['advanced_tokens_proxy'],'selected_test':result['advanced_tokens_proxy'],'relative_advance_change':result['advanced_tokens_proxy']/baseline_test['advanced_tokens_proxy']-1,'elapsed_training_seconds':time.monotonic()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'limitation':'Offline greedy suffix screen; no serving speed, quality or near-capacity qualification. Candidate is residual per-layer conditioning, not full DFlare.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prepare','train'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['control','per_layer'],default='per_layer')
    p.add_argument('--steps',type=int,default=96)
    p.add_argument('--lr',type=float,default=0.003)
    p.add_argument('--rank',type=int,choices=[0,16],default=0)
    p.add_argument('--audit',action='store_true')
    a=p.parse_args()
    if a.audit and (a.phase!='train' or a.rank!=16):
        p.error('--audit requires --phase train --rank 16 and both completed rank-16 arms')
    torch.set_num_threads(6);torch.manual_seed(910);random.seed(910)
    a.output.mkdir(parents=True,exist_ok=True)
    if not a.output.resolve().is_relative_to(ROOT/'.marathon'):raise ValueError('Private artifacts must remain under ignored .marathon')
    if a.phase=='prepare':prepare(a.output)
    else:train(a.output,a.mode,a.steps,a.lr,a.rank,a.audit)
if __name__=='__main__':main()
