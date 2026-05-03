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

cmd="${1:-help}"
shift || true

case "$cmd" in
  up|start)
    ensure_env_file
    compose up -d "$@"
    echo
    echo "SearXNG is starting at $(resolve_url)"
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
    compose restart "$@"
    ;;
  status|ps)
    ensure_env_file
    compose ps
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
  up         Bring the SearXNG stack up (creates .env on first run)
  down       Stop and remove the SearXNG stack
  restart    Restart the running stack
  status     Show container status
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
