// API doubles for the real DFlash process() and draft-batch construction bodies.
// The Python driver inserts both source versions below. No llama library is linked.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using llama_pos = int32_t;
using llama_seq_id = int32_t;
using llama_token = int32_t;
#define GGML_ASSERT(x) assert(x)
#define GGML_ABORT(...) throw std::runtime_error("missing fixture layer")
#define LOG_ERR(...) ((void) 0)

struct llama_batch {
    int32_t n_tokens;
    llama_token * token;
    float * embd;
    llama_pos * pos;
    int32_t * n_seq_id;
    llama_seq_id ** seq_id;
    int8_t * logits;
};

struct Storage {
    std::vector<llama_token> tokens;
    std::vector<float> embeddings;
    std::vector<llama_pos> positions;
    std::vector<int32_t> counts;
    std::vector<llama_seq_id> sequences;
    std::vector<llama_seq_id *> sequence_ptrs;
    std::vector<int8_t> logits;
    llama_batch batch;
    Storage(int rows, int width) : tokens(rows), embeddings(rows * width), positions(rows * 4),
            counts(rows, 1), sequences(rows), sequence_ptrs(rows), logits(rows) {
        for (int i = 0; i < rows; ++i) sequence_ptrs[i] = &sequences[i];
        batch = {0, width ? nullptr : tokens.data(), width ? embeddings.data() : nullptr,
                 positions.data(), counts.data(), sequence_ptrs.data(), logits.data()};
    }
};

struct Event {
    int kind;
    bool fused;
    std::vector<float> embeddings;
    std::vector<int32_t> positions, sequences, logits;
    bool operator==(const Event & other) const {
        return kind == other.kind && fused == other.fused && embeddings == other.embeddings &&
            positions == other.positions && sequences == other.sequences && logits == other.logits;
    }
};

struct Context {
    int ubatch;
    bool mrope, fused = false;
    int fail_encode = 0, fail_decode = 0;
    std::vector<std::vector<float>> layers = std::vector<std::vector<float>>(5);
    std::vector<float> encoded;
    std::vector<Event> events;
};

constexpr int TARGET_WIDTH = 4, ENCODER_WIDTH = 12, DECODER_WIDTH = 5;
int llama_n_ubatch(Context * ctx) { return ctx->ubatch; }
const float * llama_get_embeddings_layer_inp(Context * ctx, uint32_t layer) { return ctx->layers.at(layer).data(); }
const float * llama_get_embeddings_nextn(Context * ctx) { return ctx->encoded.data(); }
void llama_set_dflash_fused_injection(Context * ctx, bool enabled) { ctx->fused = enabled; }

void record(Context * ctx, const llama_batch & batch, int kind) {
    assert(batch.n_tokens > 0 && batch.n_tokens <= ctx->ubatch);
    const int width = kind == 0 || ctx->fused ? ENCODER_WIDTH : DECODER_WIDTH;
    Event event{kind, ctx->fused, {}, {}, {}, {}};
    event.embeddings.assign(batch.embd, batch.embd + batch.n_tokens * width);
    if (batch.pos) event.positions.assign(batch.pos, batch.pos + batch.n_tokens * (ctx->mrope ? 4 : 1));
    if (kind == 1) {
        for (int i = 0; i < batch.n_tokens; ++i) {
            assert(batch.n_seq_id[i] == 1);
            event.sequences.push_back(batch.seq_id[i][0]);
            event.logits.push_back(batch.logits[i]);
        }
    }
    ctx->events.push_back(event);
}

int llama_encode(Context * ctx, const llama_batch & batch) {
    record(ctx, batch, 0);
    ctx->encoded.resize(batch.n_tokens * DECODER_WIDTH);
    for (int i = 0; i < batch.n_tokens; ++i)
        for (int j = 0; j < DECODER_WIDTH; ++j)
            ctx->encoded[i * DECODER_WIDTH + j] = batch.embd[i * ENCODER_WIDTH + j] * 2;
    return ctx->fail_encode;
}

int llama_decode(Context * ctx, const llama_batch & batch) {
    record(ctx, batch, 1);
    return ctx->fail_decode;
}

void common_sampler_reset(int *) {}
void common_batch_clear(llama_batch & batch) { batch.n_tokens = 0; }
void common_batch_add(llama_batch & batch, llama_token token, llama_pos pos,
                      const std::vector<llama_seq_id> & seqs, bool logits) {
    const int i = batch.n_tokens++;
    batch.token[i] = token; batch.pos[i] = pos; batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = seqs.at(0); batch.logits[i] = logits;
}

struct Draft { bool drafting; int n_past; llama_token id_last; };
using common_speculative_draft_params_vec = std::vector<Draft>;

struct Stage {
    int n_seq, n_max;
    int n_embd_tgt = TARGET_WIDTH, n_embd_enc = ENCODER_WIDTH, n_embd_dec = DECODER_WIDTH;
    bool is_mrope, media = false, is_dspark = false, sample_from_anchor = true, is_dflash2 = true;
    int mask_token_id = 100;
    int32_t target_layer_ids[3] = {0, 2, 4};
    uint32_t target_layer_ids_n = 3;
    Context target, draft_ctx;
    struct { Context * ctx_tgt; Context * ctx_dft; int n_max; } params;
    Storage small, fused, noise;
    llama_batch batch_inject, batch_inject_fused, batch;
    std::vector<float> features_buf;
    std::vector<int32_t> i_batch_beg, i_batch_end, i_block_beg, n_block;
    std::vector<std::unique_ptr<int>> smpls;
    Stage(int seqs, int ubatch, int max, bool mrope, bool compact)
        : n_seq(seqs), n_max(max), is_mrope(mrope), target{ubatch, mrope}, draft_ctx{ubatch, mrope},
          params{&target, &draft_ctx, max}, small(compact ? ubatch : 4096, DECODER_WIDTH),
          fused(ubatch, ENCODER_WIDTH), noise(seqs * (max + 1), 0),
          batch_inject(small.batch), batch_inject_fused(fused.batch), batch(noise.batch),
          i_batch_beg(seqs, -1), i_batch_end(seqs, -1), i_block_beg(seqs, -1), n_block(seqs), smpls(seqs) {}
    bool supports_media() const { return media; }
    void prepare(int rows, int seed) {
        for (int l = 0; l < 5; ++l) {
            target.layers[l].resize(rows * TARGET_WIDTH);
            for (int i = 0; i < rows * TARGET_WIDTH; ++i) target.layers[l][i] = float(l * 1024 + i + seed);
        }
        draft_ctx.events.clear();
    }
};

struct Before : Stage {
    using Stage::Stage;
    // BEFORE_METHODS
};
struct After : Stage {
    using Stage::Stage;
    // AFTER_METHODS
};

int main() {
    std::mt19937 random(61029);
    int checked = 0;
    for (int seqs : {1, 3}) for (int ubatch : {1, 7, 32, 256})
    for (int max : {0, 1, 6, 7, 16}) for (bool mrope : {false, true}) {
        Before before(seqs, ubatch, max, mrope, false);
        After after(seqs, ubatch, max, mrope, true);
        for (int round = 0; round < 80; ++round) {
            const int count = round % 10 == 0 ? 0 : int(random() % (ubatch * 3 + 1));
            Storage input(count * seqs, TARGET_WIDTH);
            input.batch.n_tokens = count * seqs;
            input.batch.token = input.tokens.data();
            input.batch.embd = nullptr;
            for (int i = 0; i < count * seqs; ++i) {
                input.sequences[i] = i / count;
                input.positions[i] = 100 + round * 17 + i * 3;
                input.tokens[i] = i + 1;
            }
            if (count && round % 11 == 0) input.sequences[count - 1] = -1;
            before.media = after.media = round % 5 == 0;
            if (round % 7 == 0) {
                input.batch.token = nullptr;
                input.batch.embd = input.embeddings.data();
            }
            before.prepare(count * seqs, round);
            after.prepare(count * seqs, round);
            before.draft_ctx.fail_encode = after.draft_ctx.fail_encode = round % 19 == 0 ? 1 : 0;
            before.draft_ctx.fail_decode = after.draft_ctx.fail_decode = round % 23 == 0 ? 1 : 0;
            assert(before.process(input.batch) == after.process(input.batch));
            assert(before.draft_ctx.events == after.draft_ctx.events);
            assert(!before.draft_ctx.fused && !after.draft_ctx.fused);
            common_speculative_draft_params_vec requests;
            for (int seq = 0; seq < seqs; ++seq) requests.push_back({random() % 2 == 0, round * 53, seq + 10});
            before.is_dspark = after.is_dspark = round % 3 == 0;
            before.sample_from_anchor = after.sample_from_anchor = round % 4 == 0;
            before.build_draft(requests);
            after.build_draft(requests);
            assert(before.batch.n_tokens == after.batch.n_tokens);
            for (int i = 0; i < before.batch.n_tokens; ++i) {
                assert(before.batch.token[i] == after.batch.token[i]);
                assert(before.batch.pos[i] == after.batch.pos[i]);
                assert(before.batch.seq_id[i][0] == after.batch.seq_id[i][0]);
                assert(before.batch.logits[i] == after.batch.logits[i]);
            }
            ++checked;
        }
    }
    std::cout << checked << " paired staging and draft-batch cases passed\n";
}
