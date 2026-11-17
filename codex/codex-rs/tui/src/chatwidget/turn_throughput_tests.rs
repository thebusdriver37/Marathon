use std::time::Duration;
use std::time::Instant;

use codex_app_server_protocol::MarathonResponseMetrics;
use pretty_assertions::assert_eq;

use super::GENERATION_ACTIVITY_INTERVAL;
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
            tokens_per_second: 200.0 / 6.0
        })
    );
    throughput.finish_turn(Instant::now());
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
            tokens_per_second: 200.0 / 6.0
        })
    );
    throughput.finish_turn(Instant::now());
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

#[test]
fn generation_activity_frame_advances_and_stops_with_the_turn() {
    let start = Instant::now();
    let mut throughput = TurnThroughput::default();
    throughput.start_turn(start);

    assert_eq!(throughput.generation_activity_frame(start), Some("⠋"));
    assert_eq!(
        throughput.generation_activity_frame(start + GENERATION_ACTIVITY_INTERVAL),
        Some("⠙")
    );
    assert_eq!(
        throughput.generation_activity_frame(start + GENERATION_ACTIVITY_INTERVAL * 10),
        Some("⠋")
    );

    throughput.finish_turn(start + Duration::from_secs(2));
    assert_eq!(
        throughput.generation_activity_frame(start + Duration::from_secs(3)),
        None
    );
}

#[test]
fn latest_call_recovers_after_missing_timings_but_turn_average_does_not() {
    let mut throughput = TurnThroughput::default();
    throughput.start_turn(Instant::now());
    throughput.complete_response("first", Some(&metrics(/*tokens*/ 6700, /*seconds*/ 100)));
    throughput.complete_response("missing", None);
    assert_eq!(throughput.sample(), None);
    throughput.complete_response("latest", Some(&metrics(/*tokens*/ 105, /*seconds*/ 1)));
    assert_eq!(
        throughput.sample(),
        Some(ThroughputSample {
            tokens_per_second: 105.0
        })
    );
    throughput.complete_response("first", Some(&metrics(/*tokens*/ 6700, /*seconds*/ 100)));
    assert_eq!(
        throughput.sample(),
        Some(ThroughputSample {
            tokens_per_second: 105.0
        })
    );
    throughput.finish_turn(Instant::now());
    assert_eq!(throughput.sample(), None);
}

#[test]
fn tool_activity_precedes_later_text_and_empty_deltas_do_not_start_the_clock() {
    let start = Instant::now();
    let mut throughput = TurnThroughput::default();
    throughput.start_turn(start);
    throughput.record_output_delta(
        "",
        GenerationStage::Thinking,
        start + Duration::from_secs(1),
    );
    assert_eq!(throughput.time_to_first_output(), None);
    throughput.record_first_output(start + Duration::from_secs(5));
    throughput.record_output_delta(
        "Done",
        GenerationStage::Answering,
        start + Duration::from_secs(37),
    );
    throughput.finish_turn(start + Duration::from_secs(40));
    assert_eq!(
        throughput.time_to_first_output(),
        Some(Duration::from_secs(5))
    );
    assert_eq!(
        throughput.elapsed(start + Duration::from_secs(100)),
        Some(Duration::from_secs(40))
    );
}
