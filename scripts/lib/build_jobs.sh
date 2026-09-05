#!/usr/bin/env bash
# Conservative defaults keep first-time builds usable on everyday machines.
# Explicit JOBS / CARGO_BUILD_JOBS still take precedence.
if [[ -z "${JOBS:-}" ]]; then
  JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
  (( JOBS <= 8 )) || JOBS=8
fi
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$JOBS}"
