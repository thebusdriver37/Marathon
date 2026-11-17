use std::time::Duration;
use std::time::Instant;

use codex_app_server_protocol::MarathonResponseMetrics;
use pretty_assertions::assert_eq;

use super::GenerationStage;
use super::ThroughputSample;
use super::TurnThroughput;

fn metrics(tokens: u64, seconds: u64) -> MarathonResponseMetrics {
    MarathonResponseMetrics {
        decode_tokens: tokens,
        decode_microseconds: seconds * 1_000_000,
        backend_calls: 1,
        timed_backend_calls: 1,
        ..MarathonResponseMetrics::default()
    }
}

#[test]
fn backend_decode_excludes_prefill_and_tool_waits() {
    let start = Instant::now();
    let mut throughput = TurnThroughput::default();
    throughput.start_turn(start);
    throughput.record_output_delta(
        "first output",
        GenerationStage::Thinking,
        start + Duration::from_secs(5),
    );
    throughput.complete_response("response", Some(&metrics(/*tokens*/ 2600, /*seconds*/ 40)));
    throughput.finish_turn(start + Duration::from_secs(80));
    assert_eq!(
        throughput.sample(),
        Some(ThroughputSample {
            tokens_per_second: 65.0
        })
    );
    assert_eq!(
        throughput.time_to_first_output(),
        Some(Duration::from_secs(5))
    );
    assert_eq!(
        throughput.elapsed(start + Duration::from_secs(100)),
        Some(Duration::from_secs(80))
    );
}

#[test]
fn completed_calls_have_a_decode_time_weighted_average() {
    let mut throughput = TurnThroughput::default();
    throughput.start_turn(Instant::now());
    throughput.complete_response("first", Some(&metrics(/*tokens*/ 200, /*seconds*/ 2)));
    throughput.complete_response("second", Some(&metrics(/*tokens*/ 200, /*seconds*/ 6)));
    assert_eq!(
        throughput.sample(),
        Some(ThroughputSample {
            tokens_per_second: 50.0
        })
    );
}

#[test]
fn duplicate_completion_does_not_count_twice() {
    let mut throughput = TurnThroughput::default();
    throughput.start_turn(Instant::now());
    throughput.complete_response("first", Some(&metrics(/*tokens*/ 200, /*seconds*/ 2)));
    throughput.complete_response("first", Some(&metrics(/*tokens*/ 200, /*seconds*/ 2)));
    throughput.complete_response("second", Some(&metrics(/*tokens*/ 200, /*seconds*/ 6)));
    assert_eq!(
        throughput.sample(),
        Some(ThroughputSample {
            tokens_per_second: 50.0
        })
    );
}

#[test]
fn incomplete_timing_coverage_is_not_reported_as_turn_decode() {
    for missing in [
        None,
        Some(MarathonResponseMetrics {
            backend_calls: 2,
            ..metrics(/*tokens*/ 200, /*seconds*/ 2)
        }),
        Some(metrics(/*tokens*/ 200, /*seconds*/ 0)),
    ] {
        let mut throughput = TurnThroughput::default();
        throughput.start_turn(Instant::now());
        throughput.complete_response("measured", Some(&metrics(/*tokens*/ 200, /*seconds*/ 2)));
        throughput.complete_response("missing", missing.as_ref());
        assert_eq!(throughput.sample(), None);
    }
}

#[test]
fn a_new_turn_clears_previous_measurements() {
    let start = Instant::now();
    let mut throughput = TurnThroughput::default();
    throughput.start_turn(start);
    throughput.complete_response("first", Some(&metrics(/*tokens*/ 200, /*seconds*/ 2)));
    throughput.finish_turn(start + Duration::from_secs(4));
    throughput.start_turn(start + Duration::from_secs(10));
    assert_eq!(throughput.sample(), None);
    assert_eq!(throughput.time_to_first_output(), None);
    assert_eq!(
        throughput.generation_stage(),
        Some(GenerationStage::Prefill)
    );
    assert_eq!(
        throughput.elapsed(start + Duration::from_secs(12)),
        Some(Duration::from_secs(2))
    );
}

#[test]
fn interrupted_generation_does_not_publish_a_partial_turn_average() {
    let start = Instant::now();
    let mut throughput = TurnThroughput::default();
    throughput.start_turn(start);
    throughput.complete_response("first", Some(&metrics(/*tokens*/ 200, /*seconds*/ 2)));
    throughput.record_output_delta(
        "unfinished",
        GenerationStage::Thinking,
        start + Duration::from_secs(5),
    );
    throughput.finish_turn(start + Duration::from_secs(6));
    assert_eq!(throughput.sample(), None);
}

#[test]
fn first_output_is_recorded_once_and_stage_tracks_the_stream() {
    let start = Instant::now();
    let mut throughput = TurnThroughput::default();
    throughput.start_turn(start);
    throughput.record_output_delta(
        "thinking",
        GenerationStage::Thinking,
        start + Duration::from_secs(2),
    );
    assert_eq!(
        throughput.generation_stage(),
        Some(GenerationStage::Thinking)
    );
    throughput.record_output_delta(
        "answer",
        GenerationStage::Answering,
        start + Duration::from_secs(8),
    );
    assert_eq!(
        throughput.generation_stage(),
        Some(GenerationStage::Answering)
    );
    assert_eq!(
        throughput.time_to_first_output(),
        Some(Duration::from_secs(2))
    );
    throughput.finish_turn(start + Duration::from_secs(10));
    assert_eq!(throughput.generation_stage(), None);
}
