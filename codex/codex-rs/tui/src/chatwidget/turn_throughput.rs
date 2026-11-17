use std::collections::HashSet;
use std::time::Duration;
use std::time::Instant;

use codex_app_server_protocol::MarathonResponseMetrics;

pub(super) const GENERATION_ACTIVITY_INTERVAL: Duration = Duration::from_millis(100);
const GENERATION_ACTIVITY_FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GenerationStage {
    Prefill,
    Thinking,
    Answering,
    PreparingAction,
    DraftingCode,
    ApplyingChanges,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ThroughputSample {
    pub(super) tokens_per_second: f64,
}

/// Backend decode totals and client-observed latency have different clocks.
/// Never infer decode throughput from streamed byte counts or tool-loop wall time.
#[derive(Debug, Default)]
pub(super) struct TurnThroughput {
    turn_started_at: Option<Instant>,
    turn_finished_at: Option<Instant>,
    seen_responses: HashSet<String>,
    decode_tokens: u64,
    decode_microseconds: u64,
    missing_timings: bool,
    latest_sample: Option<ThroughputSample>,
    generation_stage: Option<GenerationStage>,
    patch_call_id: Option<String>,
}

impl TurnThroughput {
    pub(super) fn start_turn(&mut self, now: Instant) {
        *self = Self::default();
        self.turn_started_at = Some(now);
        self.generation_stage = Some(GenerationStage::Prefill);
    }

    pub(super) fn record_output_delta(&mut self, delta: &str, generation_stage: GenerationStage) {
        if delta.is_empty() {
            return;
        }
        self.generation_stage = Some(generation_stage);
    }

    pub(super) fn complete_response(
        &mut self,
        response_id: &str,
        metrics: Option<&MarathonResponseMetrics>,
    ) {
        if self.turn_started_at.is_none() || !self.seen_responses.insert(response_id.to_string()) {
            return;
        }
        self.generation_stage = None;
        self.latest_sample = None;
        let Some(metrics) = metrics else {
            self.missing_timings = true;
            return;
        };
        if metrics.backend_calls == 0
            || metrics.timed_backend_calls != metrics.backend_calls
            || (metrics.decode_tokens > 0 && metrics.decode_microseconds == 0)
        {
            self.missing_timings = true;
            return;
        }
        if metrics.decode_tokens > 0 && metrics.decode_microseconds > 0 {
            self.latest_sample = Some(ThroughputSample {
                tokens_per_second: metrics.decode_tokens as f64 * 1_000_000.0
                    / metrics.decode_microseconds as f64,
            });
        }
        self.decode_tokens = self.decode_tokens.saturating_add(metrics.decode_tokens);
        self.decode_microseconds = self
            .decode_microseconds
            .saturating_add(metrics.decode_microseconds);
    }

    pub(super) fn finish_turn(&mut self, now: Instant) {
        if self.generation_stage.is_some() {
            self.missing_timings = true;
        }
        self.generation_stage = None;
        self.patch_call_id = None;
        self.turn_finished_at.get_or_insert(now);
    }

    pub(super) fn generation_stage(&self) -> Option<GenerationStage> {
        self.generation_stage
    }

    pub(super) fn generation_activity_frame(&self, now: Instant) -> Option<&'static str> {
        if !self.is_active() {
            return None;
        }
        let started_at = self.turn_started_at?;
        let frame_index = (now.saturating_duration_since(started_at).as_millis()
            / GENERATION_ACTIVITY_INTERVAL.as_millis())
            % GENERATION_ACTIVITY_FRAMES.len() as u128;
        GENERATION_ACTIVITY_FRAMES
            .get(frame_index as usize)
            .copied()
    }

    pub(super) fn elapsed(&self, now: Instant) -> Option<Duration> {
        self.turn_started_at.map(|start| {
            self.turn_finished_at
                .unwrap_or(now)
                .saturating_duration_since(start)
        })
    }

    pub(super) fn set_generation_stage(&mut self, generation_stage: GenerationStage) {
        self.generation_stage = Some(generation_stage);
        if !matches!(
            generation_stage,
            GenerationStage::DraftingCode | GenerationStage::ApplyingChanges
        ) {
            self.patch_call_id = None;
        }
    }

    pub(super) fn record_patch_progress(&mut self, call_id: &str) {
        self.patch_call_id = Some(call_id.to_string());
        self.generation_stage = Some(GenerationStage::DraftingCode);
    }

    pub(super) fn begin_patch_apply(&mut self, call_id: &str) {
        if self
            .patch_call_id
            .as_deref()
            .is_none_or(|active_call_id| active_call_id != call_id)
        {
            self.record_patch_progress(call_id);
        }
        self.generation_stage = Some(GenerationStage::ApplyingChanges);
    }

    pub(super) fn finish_patch_apply(&mut self, call_id: &str) {
        if self
            .patch_call_id
            .as_deref()
            .is_some_and(|active_call_id| active_call_id == call_id)
        {
            self.patch_call_id = None;
        }
        if matches!(
            self.generation_stage,
            Some(GenerationStage::DraftingCode | GenerationStage::ApplyingChanges)
        ) {
            self.generation_stage = Some(GenerationStage::Prefill);
        }
    }

    pub(super) fn is_finished(&self) -> bool {
        self.turn_finished_at.is_some()
    }

    pub(super) fn is_active(&self) -> bool {
        self.turn_started_at.is_some() && !self.is_finished()
    }

    // During tool loops report the latest response; only summarize at turn end.
    pub(super) fn sample(&self) -> Option<ThroughputSample> {
        if !self.is_finished() {
            return self.latest_sample;
        }
        if self.missing_timings || self.decode_tokens == 0 || self.decode_microseconds == 0 {
            return None;
        }
        Some(ThroughputSample {
            tokens_per_second: self.decode_tokens as f64 * 1_000_000.0
                / self.decode_microseconds as f64,
        })
    }
}

#[cfg(test)]
#[path = "turn_throughput_tests.rs"]
mod tests;
