use std::time::Duration;
use std::time::Instant;

use pretty_assertions::assert_eq;

use super::GenerationStage;
use super::ThroughputSample;
use super::TurnThroughput;

#[test]
fn live_sample_is_estimated_from_streamed_bytes() {
    let started_at = Instant::now();
    let mut throughput = TurnThroughput::default();

    throughput.start_turn();
    throughput.record_output_delta(
        "1234567890123456789012345678901234567890",
        GenerationStage::Thinking,
        started_at,
    );

    assert_eq!(
        throughput.sample(started_at + Duration::from_secs(2)),
        Some(ThroughputSample {
            tokens_per_second: 5.0,
            estimated: true,
        })
    );
}

#[test]
fn completed_response_replaces_estimate_with_exact_usage() {
    let started_at = Instant::now();
    let completed_at = started_at + Duration::from_secs(2);
    let mut throughput = TurnThroughput::default();

    throughput.start_turn();
    throughput.record_output_delta("streamed text", GenerationStage::Answering, started_at);
    throughput.complete_response(Some(30), completed_at);

    assert_eq!(
        throughput.sample(completed_at),
        Some(ThroughputSample {
            tokens_per_second: 15.0,
            estimated: false,
        })
    );
}

#[test]
fn fast_completed_response_still_reports_throughput() {
    let started_at = Instant::now();
    let completed_at = started_at + Duration::from_millis(100);
    let mut throughput = TurnThroughput::default();

    throughput.start_turn();
    throughput.record_output_delta("READY", GenerationStage::Answering, started_at);
    throughput.complete_response(Some(4), completed_at);

    assert_eq!(
        throughput.sample(completed_at),
        Some(ThroughputSample {
            tokens_per_second: 40.0,
            estimated: false,
        })
    );
}

#[test]
fn completed_responses_form_a_generation_time_weighted_turn_average() {
    let started_at = Instant::now();
    let mut throughput = TurnThroughput::default();

    throughput.start_turn();
    throughput.record_output_delta("first", GenerationStage::Thinking, started_at);
    throughput.complete_response(Some(20), started_at + Duration::from_secs(2));
    throughput.record_output_delta(
        "second",
        GenerationStage::Answering,
        started_at + Duration::from_secs(10),
    );
    throughput.complete_response(Some(40), started_at + Duration::from_secs(14));

    assert_eq!(
        throughput.sample(started_at + Duration::from_secs(14)),
        Some(ThroughputSample {
            tokens_per_second: 10.0,
            estimated: false,
        })
    );
}

#[test]
fn starting_a_turn_clears_the_previous_sample() {
    let started_at = Instant::now();
    let mut throughput = TurnThroughput::default();
    throughput.record_output_delta("streamed text", GenerationStage::Answering, started_at);
    throughput.complete_response(Some(30), started_at + Duration::from_secs(2));

    throughput.start_turn();

    assert_eq!(throughput.sample(started_at + Duration::from_secs(3)), None);
    assert_eq!(
        throughput.generation_stage(),
        Some(GenerationStage::Prefill)
    );
}

#[test]
fn generation_stage_tracks_prefill_thinking_and_answering() {
    let started_at = Instant::now();
    let mut throughput = TurnThroughput::default();

    throughput.start_turn();
    assert_eq!(
        throughput.generation_stage(),
        Some(GenerationStage::Prefill)
    );

    throughput.record_output_delta("thought", GenerationStage::Thinking, started_at);
    assert_eq!(
        throughput.generation_stage(),
        Some(GenerationStage::Thinking)
    );

    throughput.record_output_delta("answer", GenerationStage::Answering, started_at);
    assert_eq!(
        throughput.generation_stage(),
        Some(GenerationStage::Answering)
    );

    throughput.finish_turn(started_at + Duration::from_secs(1));
    assert_eq!(throughput.generation_stage(), None);
}
