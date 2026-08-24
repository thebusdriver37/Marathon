#!/usr/bin/env bash
set -euo pipefail

# Manage the Marathon SearXNG Docker stack used to back the local web_search
# tool exposed by the Codex local router.
#
# Subcommands: up, down, restart, status, logs, url, env-init, pull
#
# Usage: scripts/ops/searxng.sh <subcommand>

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_DIR="$ROOT_DIR/docker/searxng"
ENV_FILE="$COMPOSE_DIR/.env"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"
LEGACY_DEFAULT_IMAGE="searxng/searxng:2026.5.2-aefc3c316"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is not installed or not on PATH" >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "error: 'docker compose' v2 plugin is required" >&2
  exit 1
fi

ensure_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    migrate_legacy_defaults
    if grep -q '^MARATHON_SEARXNG_SECRET=$' "$ENV_FILE" 2>/dev/null \
      || ! grep -q '^MARATHON_SEARXNG_SECRET=' "$ENV_FILE" 2>/dev/null; then
      generate_secret_into_env
    fi
    return 0
  fi
  cp "$COMPOSE_DIR/.env.example" "$ENV_FILE"
  generate_secret_into_env
  echo "wrote $ENV_FILE (with freshly generated secret)" >&2
}

migrate_legacy_defaults() {
  if ! grep -Fxq "MARATHON_SEARXNG_IMAGE=$LEGACY_DEFAULT_IMAGE" "$ENV_FILE" 2>/dev/null; then
    return 0
  fi

  local tmp
  tmp="$(mktemp)"
  awk -v legacy_image="MARATHON_SEARXNG_IMAGE=$LEGACY_DEFAULT_IMAGE" '
    $0 == legacy_image { next }
    $0 == "MARATHON_SEARXNG_WORKERS=2" { next }
    $0 == "MARATHON_SEARXNG_THREADS=4" { next }
    { print }
  ' "$ENV_FILE" >"$tmp"
  mv "$tmp" "$ENV_FILE"
  echo "updated generated SearXNG defaults in $ENV_FILE" >&2
}

generate_secret_into_env() {
  local secret
  secret="$(openssl rand -hex 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_hex(32))')"
  if grep -q '^MARATHON_SEARXNG_SECRET=' "$ENV_FILE" 2>/dev/null; then
    local tmp
    tmp="$(mktemp)"
    awk -v sec="$secret" '
      /^MARATHON_SEARXNG_SECRET=/ { print "MARATHON_SEARXNG_SECRET=" sec; next }
      { print }
    ' "$ENV_FILE" >"$tmp"
    mv "$tmp" "$ENV_FILE"
  else
    printf 'MARATHON_SEARXNG_SECRET=%s\n' "$secret" >>"$ENV_FILE"
  fi
}

compose() {
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

resolve_url() {
  local host port
  host="$(grep -E '^MARATHON_SEARXNG_BIND=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)"
  port="$(grep -E '^MARATHON_SEARXNG_PORT=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)"
  host="${host:-127.0.0.1}"
  port="${port:-18093}"
  printf 'http://%s:%s\n' "$host" "$port"
}

configured_url() {
  if [[ -n "${MARATHON_SEARXNG_URL:-}" ]]; then
    printf '%s\n' "${MARATHON_SEARXNG_URL%/}"
  else
    resolve_url
  fi
}

health_check() {
  local url="$1"
  local curl_command=(curl -fsS)
  if [[ "$url" =~ ^https?://(127\.0\.0\.1|localhost|\[::1\])([:/]|$) ]]; then
    curl_command+=(-H 'X-Forwarded-For: 127.0.0.1')
  fi
  "${curl_command[@]}" --max-time 2 "$url/healthz" >/dev/null 2>&1
}

wait_for_health() {
  local url
  url="$(configured_url)"
  for _attempt in {1..45}; do
    if health_check "$url"; then
      return 0
    fi
    sleep 1
  done
  echo "error: SearXNG did not become healthy at $url within 45 seconds" >&2
  return 1
}

check_search() {
  local url response_file summary
  local curl_command=(curl -fsS -G)
  url="$(configured_url)"
  if [[ "$url" =~ ^https?://(127\.0\.0\.1|localhost|\[::1\])([:/]|$) ]]; then
    curl_command+=(-H 'X-Forwarded-For: 127.0.0.1')
  fi
  response_file="$(mktemp)"
  if ! "${curl_command[@]}" --max-time 20 \
    --data-urlencode 'q=SearXNG documentation official' \
    --data-urlencode 'format=json' \
    "$url/search" >"$response_file"; then
    rm -f "$response_file"
    echo "error: SearXNG search request failed at $url" >&2
    return 1
  fi

  if ! summary="$(python3 - "$response_file" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception as exc:
    print(f"error: SearXNG returned invalid JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)

results = payload.get("results") if isinstance(payload, dict) else None
if isinstance(results, list) and results:
    engines = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        names = result.get("engines")
        if isinstance(names, list):
            engines.update(str(name) for name in names if name)
        elif result.get("engine"):
            engines.add(str(result["engine"]))
    suffix = f" via {', '.join(sorted(engines))}" if engines else ""
    print(f"SearXNG search works: {len(results)} results{suffix}")
    raise SystemExit(0)

failures = payload.get("unresponsive_engines") if isinstance(payload, dict) else None
details = []
if isinstance(failures, list):
    for failure in failures:
        if isinstance(failure, list) and failure:
            name = str(failure[0])
            reason = str(failure[1]) if len(failure) > 1 else "failed"
            details.append(f"{name}: {reason}")
detail = f" ({'; '.join(details)})" if details else ""
print(f"error: SearXNG returned no usable results{detail}", file=sys.stderr)
raise SystemExit(1)
PY
)"; then
    rm -f "$response_file"
    return 1
  fi
  rm -f "$response_file"
  printf '%s\n' "$summary"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  up|start)
    ensure_env_file
    compose up -d "$@"
    wait_for_health
    check_search
    echo
    echo "SearXNG is ready at $(configured_url)"
    echo "Tail logs with: ./bin/marathon search logs"
    ;;
  down|stop)
    if [[ -f "$ENV_FILE" ]]; then
      compose down "$@"
    else
      echo "no .env file present; nothing to stop" >&2
    fi
    ;;
  restart)
    ensure_env_file
    compose up -d --force-recreate "$@"
    wait_for_health
    check_search
    ;;
  status|ps)
    ensure_env_file
    compose ps
    if ! health_check "$(configured_url)"; then
      echo "error: SearXNG is not healthy at $(configured_url)" >&2
      exit 1
    fi
    check_search
    ;;
  check)
    ensure_env_file
    if ! health_check "$(configured_url)"; then
      echo "error: SearXNG is not healthy at $(configured_url)" >&2
      exit 1
    fi
    check_search
    ;;
  logs)
    ensure_env_file
    compose logs --tail=200 -f "$@"
    ;;
  url)
    ensure_env_file
    resolve_url
    ;;
  env-init)
    ensure_env_file
    echo "$ENV_FILE"
    ;;
  pull)
    ensure_env_file
    compose pull
    ;;
  help|--help|-h|"")
    cat <<USAGE
Usage: scripts/ops/searxng.sh <subcommand>

Subcommands:
  up         Bring the stack up and wait for a working search
  down       Stop and remove the SearXNG stack
  restart    Recreate the stack and wait for a working search
  status     Show container status and run a search check
  check      Run a functional search check
  logs       Tail container logs
  url        Print the resolved http://host:port URL
  env-init   Create docker/searxng/.env if missing and print its path
  pull       docker compose pull (refresh the image)

The .env file is generated from .env.example with a freshly-rolled
MARATHON_SEARXNG_SECRET on first run. Edit .env to change bind/port.
USAGE
    ;;
  *)
    echo "unknown subcommand: $cmd" >&2
    "$0" help >&2
    exit 1
    ;;
esac
