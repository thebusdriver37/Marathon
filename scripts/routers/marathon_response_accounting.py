"""Separate cumulative inference usage, occupied context, and backend timing."""

import copy
import math
from dataclasses import dataclass, field
from typing import Any


def _count(value: Any) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _microseconds(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return round(value * 1000)


@dataclass
class ResponseAccounting:
    """Bounded totals for one logical response, including managed-tool retries."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    context_usage: dict[str, int] = field(default_factory=dict)
    decode_tokens: int = 0
    decode_microseconds: int = 0
    prefill_tokens: int = 0
    prefill_microseconds: int = 0
    draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    backend_calls: int = 0
    timed_backend_calls: int = 0

    def add(self, response: dict[str, Any]) -> None:
        self.backend_calls += 1
        usage = response.get("usage") or {}
        inputs = usage.get("input_tokens_details") or {}
        outputs = usage.get("output_tokens_details") or {}
        self.context_usage = {
            "input_tokens": _count(usage.get("input_tokens")),
            "cached_input_tokens": _count(inputs.get("cached_tokens")),
            "cache_write_input_tokens": _count(inputs.get("cache_write_tokens")),
            "output_tokens": _count(usage.get("output_tokens")),
            "reasoning_output_tokens": _count(outputs.get("reasoning_tokens")),
        }
        for name, value in self.context_usage.items():
            setattr(self, name, getattr(self, name) + value)
        self.context_usage["total_tokens"] = (
            self.context_usage["input_tokens"] + self.context_usage["output_tokens"]
        )
        timings = response.get("timings")
        if not isinstance(timings, dict):
            return
        decode_us = _microseconds(timings.get("predicted_ms"))
        prefill_us = _microseconds(timings.get("prompt_ms"))
        predicted = timings.get("predicted_n")
        if decode_us is None or prefill_us is None or not isinstance(predicted, int):
            return
        if (
            predicted < 0
            or isinstance(predicted, bool)
            or (predicted > 1 and decode_us == 0)
        ):
            return
        self.timed_backend_calls += 1
        # llama.cpp accounts for the first generated token in the prompt phase.
        self.decode_tokens += max(0, predicted - 1)
        self.decode_microseconds += decode_us
        self.prefill_tokens += _count(timings.get("prompt_n"))
        self.prefill_microseconds += prefill_us
        self.draft_tokens += _count(timings.get("draft_n"))
        self.accepted_draft_tokens += _count(timings.get("draft_n_accepted"))

    def apply(self, response: dict[str, Any]) -> None:
        response["usage"] = {
            **copy.deepcopy(response.get("usage") or {}),
            "input_tokens": self.input_tokens,
            "input_tokens_details": {
                "cached_tokens": self.cached_input_tokens,
                "cache_write_tokens": self.cache_write_input_tokens,
            },
            "output_tokens": self.output_tokens,
            "output_tokens_details": {"reasoning_tokens": self.reasoning_output_tokens},
            "total_tokens": self.input_tokens + self.output_tokens,
        }
        metadata = copy.deepcopy(response.get("usage_metadata") or {})
        metadata["marathon"] = {
            "context_usage": copy.deepcopy(self.context_usage),
            "decode_tokens": self.decode_tokens,
            "decode_microseconds": self.decode_microseconds,
            "prefill_tokens": self.prefill_tokens,
            "prefill_microseconds": self.prefill_microseconds,
            "draft_tokens": self.draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "backend_calls": self.backend_calls,
            "timed_backend_calls": self.timed_backend_calls,
        }
        response["usage_metadata"] = metadata
