//! Per-turn output throughput derived from live deltas and exact response usage.

use std::time::Duration;
use std::time::Instant;

use codex_utils_string::approx_tokens_from_byte_count;

const MIN_SAMPLE_DURATION: Duration = Duration::from_millis(1);

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ThroughputSample {
    pub(super) tokens_per_second: f64,
    pub(super) estimated: bool,
}

#[derive(Debug)]
struct ActiveResponse {
    started_at: Instant,
    output_bytes: usize,
}

#[derive(Debug, Default)]
pub(super) struct TurnThroughput {
    active_response: Option<ActiveResponse>,
    completed_output_tokens: u64,
    completed_generation_duration: Duration,
    last_sample: Option<ThroughputSample>,
}

impl TurnThroughput {
    pub(super) fn start_turn(&mut self) {
        *self = Self::default();
    }

    pub(super) fn record_output_delta(&mut self, delta: &str, now: Instant) {
        let response = self.active_response.get_or_insert(ActiveResponse {
            started_at: now,
            output_bytes: 0,
        });
        response.output_bytes = response.output_bytes.saturating_add(delta.len());
    }

    pub(super) fn complete_response(&mut self, output_tokens: Option<i64>, now: Instant) {
        let live_sample = self.sample(now);
        let Some(response) = self.active_response.take() else {
            return;
        };
        let elapsed = now.saturating_duration_since(response.started_at);
        let Some(output_tokens) = output_tokens.and_then(|tokens| u64::try_from(tokens).ok())
        else {
            self.last_sample = live_sample;
            return;
        };
        if output_tokens == 0 || elapsed < MIN_SAMPLE_DURATION {
            self.last_sample = live_sample;
            return;
        }

        self.completed_output_tokens = self.completed_output_tokens.saturating_add(output_tokens);
        self.completed_generation_duration =
            self.completed_generation_duration.saturating_add(elapsed);
        self.last_sample = sample_from_totals(
            self.completed_output_tokens,
            self.completed_generation_duration,
            /*estimated*/ false,
        );
    }

    pub(super) fn finish_turn(&mut self, now: Instant) {
        if self.active_response.is_some() {
            self.last_sample = self.sample(now);
            self.active_response = None;
        }
    }

    pub(super) fn sample(&self, now: Instant) -> Option<ThroughputSample> {
        let Some(response) = self.active_response.as_ref() else {
            return self.last_sample;
        };
        let elapsed = now.saturating_duration_since(response.started_at);
        let live_tokens = approx_tokens_from_byte_count(response.output_bytes);
        sample_from_totals(
            self.completed_output_tokens.saturating_add(live_tokens),
            self.completed_generation_duration.saturating_add(elapsed),
            /*estimated*/ true,
        )
        .or(self.last_sample)
    }
}

fn sample_from_totals(
    output_tokens: u64,
    generation_duration: Duration,
    estimated: bool,
) -> Option<ThroughputSample> {
    if output_tokens == 0 || generation_duration < MIN_SAMPLE_DURATION {
        return None;
    }
    Some(ThroughputSample {
        tokens_per_second: output_tokens as f64 / generation_duration.as_secs_f64(),
        estimated,
    })
}

#[cfg(test)]
#[path = "turn_throughput_tests.rs"]
mod tests;
