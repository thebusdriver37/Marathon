use std::collections::HashSet;
use std::time::Duration;
use std::time::Instant;

use codex_app_server_protocol::MarathonResponseMetrics;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GenerationStage {
    Prefill,
    Thinking,
    Answering,
}

impl GenerationStage {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Prefill => "Prefill",
            Self::Thinking => "Thinking",
            Self::Answering => "Answering",
        }
    }
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
    time_to_first_output: Option<Duration>,
    seen_responses: HashSet<String>,
    decode_tokens: u64,
    decode_microseconds: u64,
    missing_timings: bool,
    latest_sample: Option<ThroughputSample>,
    generation_stage: Option<GenerationStage>,
}

impl TurnThroughput {
    pub(super) fn start_turn(&mut self, now: Instant) {
        *self = Self::default();
        self.turn_started_at = Some(now);
        self.generation_stage = Some(GenerationStage::Prefill);
    }

    pub(super) fn record_output_delta(
        &mut self,
        delta: &str,
        generation_stage: GenerationStage,
        now: Instant,
    ) {
        if delta.is_empty() {
            return;
        }
        self.record_first_output(now);
        self.generation_stage = Some(generation_stage);
    }

    pub(super) fn record_first_output(&mut self, now: Instant) {
        if self.time_to_first_output.is_none()
            && let Some(turn_started_at) = self.turn_started_at
        {
            self.time_to_first_output = Some(now.saturating_duration_since(turn_started_at));
        }
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
        self.turn_finished_at.get_or_insert(now);
    }

    pub(super) fn generation_stage(&self) -> Option<GenerationStage> {
        self.generation_stage
    }

    pub(super) fn time_to_first_output(&self) -> Option<Duration> {
        self.time_to_first_output
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
    }

    pub(super) fn is_finished(&self) -> bool {
        self.turn_finished_at.is_some()
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
