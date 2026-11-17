//! Per-response usage metadata reported by the upstream service, without aggregation.

use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use ts_rs::TS;

/// Usage metadata reported for one upstream response.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq, TS, JsonSchema)]
pub struct ResponseUsageMetadata {
    pub amount: Option<String>,
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marathon: Option<MarathonResponseMetrics>,
}

/// Local backend measurements, kept separate from cumulative response usage.
#[derive(Debug, Default, Clone, Deserialize, Serialize, PartialEq, Eq, TS, JsonSchema)]
pub struct MarathonResponseMetrics {
    pub context_usage: Option<crate::protocol::TokenUsage>,
    pub decode_tokens: u64,
    pub decode_microseconds: u64,
    pub prefill_tokens: u64,
    pub prefill_microseconds: u64,
    pub backend_calls: u64,
    pub timed_backend_calls: u64,
}
