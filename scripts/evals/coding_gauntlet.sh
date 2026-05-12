#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUTER_HOST="${MARATHON_PROXY_HOST:-127.0.0.1}"
ROUTER_PORT="${MARATHON_PROXY_PORT:-18111}"
TIMEOUT_SECONDS="${MARATHON_EVAL_TIMEOUT_SECONDS:-720}"
BASE_DIR="${MARATHON_EVAL_BASE_DIR:-/tmp}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
WORK_DIR="${MARATHON_EVAL_WORK_DIR:-$BASE_DIR/marathon-eval-coding-$RUN_ID}"
LOG_FILE="$WORK_DIR/marathon-eval.log"
TEST_LOG="$WORK_DIR/tests.log"

usage() {
  cat <<USAGE
Usage: marathon eval coding

Runs a temporary Codex-style coding task against the active Marathon backend.
The gauntlet lives under /tmp by default and does not modify this repo.

Environment:
  MARATHON_EVAL_TIMEOUT_SECONDS   Timeout for the model run (default: 720)
  MARATHON_EVAL_BASE_DIR          Parent directory for temporary eval workspaces (default: /tmp)
  MARATHON_EVAL_WORK_DIR          Exact workspace path to use
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    help|--help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown eval option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

router_url="http://$ROUTER_HOST:$ROUTER_PORT"
health_json="$(curl -fsS --max-time 2 "$router_url/health" 2>/dev/null || true)"
if [[ -z "$health_json" ]]; then
  echo "error: Marathon backend is not running." >&2
  echo "run: marathon backend start 128k-single" >&2
  exit 1
fi

backend_status="$(
  HEALTH_JSON="$health_json" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["HEALTH_JSON"])
print((payload.get("backend_health") or {}).get("status") or "")
PY
)"
active_model="$(
  HEALTH_JSON="$health_json" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["HEALTH_JSON"])
print(payload.get("current_model") or "")
PY
)"

if [[ "$backend_status" != "ok" ]]; then
  echo "error: Marathon backend is not ready: ${backend_status:-unknown}" >&2
  echo "run: marathon backend status" >&2
  exit 1
fi

mkdir -p "$WORK_DIR/miniboard" "$WORK_DIR/tests"

cat >"$WORK_DIR/brief.md" <<'EOF'
# Coding Gauntlet Brief

Make this repository production-ready enough for the included tests and normal
CLI use.

Constraints:

- Keep the implementation small and idiomatic.
- Preserve the public API exposed from `miniboard/__init__.py`.
- Do not add third-party dependencies.
- Use the standard library only.
- Run the tests before you finish.

The main user workflow is:

1. Parse Markdown backlog cards.
2. Schedule them into sprints by dependency order and sprint capacity.
3. Render either Markdown or JSON output from the CLI.
EOF

cat >"$WORK_DIR/README.md" <<'EOF'
# MiniBoard

MiniBoard turns a lightweight Markdown backlog into sprint planning reports.

Run tests with:

```bash
python3 -m unittest discover -s tests -v
```
EOF

cat >"$WORK_DIR/miniboard/__init__.py" <<'EOF'
from .core import Card
from .core import parse_cards
from .core import render_report
from .core import schedule_cards
from .core import slugify

__all__ = [
    "Card",
    "parse_cards",
    "render_report",
    "schedule_cards",
    "slugify",
]
EOF

cat >"$WORK_DIR/miniboard/__main__.py" <<'EOF'
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
EOF

cat >"$WORK_DIR/miniboard/cli.py" <<'EOF'
import argparse
import json
from pathlib import Path

from .core import parse_cards
from .core import render_report
from .core import schedule_cards


def main(argv=None):
    parser = argparse.ArgumentParser(prog="miniboard")
    parser.add_argument("backlog", help="Markdown backlog file")
    parser.add_argument("--capacity", type=int, default=5)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    text = Path(args.backlog).read_text()
    cards = parse_cards(text)
    sprints = schedule_cards(cards, args.capacity)

    if args.as_json:
        payload = [[card.__dict__ for card in sprint] for sprint in sprints]
        print(json.dumps(payload))
    else:
        print(render_report(sprints))
    return 0
EOF

cat >"$WORK_DIR/miniboard/core.py" <<'EOF'
from dataclasses import dataclass


@dataclass
class Card:
    slug: str
    title: str
    owner: str | None
    estimate: int
    tags: list[str]
    depends_on: list[str]
    body: str


def slugify(title: str, existing=None) -> str:
    existing = existing or set()
    slug = title.lower().replace(" ", "-")
    if slug in existing:
        slug = slug + "-1"
    return slug


def parse_cards(markdown: str) -> list[Card]:
    cards = []
    current_title = None
    body = []
    meta = {}

    def flush():
        if current_title is None:
            return
        slug = slugify(current_title, {card.slug for card in cards})
        estimate = int(meta.get("estimate", "1"))
        tags = meta.get("tags", "").split(",") if meta.get("tags") else []
        depends_on = meta.get("depends_on", "").split(",") if meta.get("depends_on") else []
        cards.append(
            Card(
                slug=slug,
                title=current_title,
                owner=meta.get("owner"),
                estimate=estimate,
                tags=tags,
                depends_on=depends_on,
                body="\n".join(body).strip(),
            )
        )

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            flush()
            current_title = line[3:]
            body = []
            meta = {}
        elif line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            meta[key] = value.strip()
        elif current_title:
            body.append(raw_line)

    flush()
    return cards


def schedule_cards(cards: list[Card], capacity: int) -> list[list[Card]]:
    sprints = []
    current = []
    total = 0
    for card in cards:
        if total + card.estimate > capacity:
            sprints.append(current)
            current = []
            total = 0
        current.append(card)
        total += card.estimate
    if current:
        sprints.append(current)
    return sprints


def render_report(sprints: list[list[Card]]) -> str:
    lines = ["# Sprint Plan"]
    for index, sprint in enumerate(sprints):
        lines.append("")
        lines.append(f"## Sprint {index}")
        for card in sprint:
            lines.append(f"- {card.title} ({card.estimate})")
    return "\n".join(lines)
EOF

cat >"$WORK_DIR/tests/test_miniboard.py" <<'EOF'
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from miniboard import parse_cards
from miniboard import render_report
from miniboard import schedule_cards
from miniboard import slugify


BACKLOG = textwrap.dedent(
    """
    # Product backlog

    ## API Pagination
    - owner: @ana
    - estimate: 2
    - tags: api, performance

    Return stable page tokens for large result sets.

    ## Search Filters!
    - owner: @sam
    - estimate: 3
    - tags: search, ui
    - depends_on: api-pagination

    Users need to filter by owner and tag.

    ## Search Filters
    - estimate: 1

    Follow-up copy tweaks.
    """
)


class SlugTests(unittest.TestCase):
    def test_slugify_normalizes_and_deduplicates(self):
        existing = {"api-pagination", "search-filters"}
        self.assertEqual(slugify(" API Pagination!! ", existing), "api-pagination-2")
        self.assertEqual(slugify("Search Filters", existing), "search-filters-2")
        self.assertEqual(slugify("!!!", set()), "item")


class ParserTests(unittest.TestCase):
    def test_parse_cards_extracts_metadata_body_and_unique_slugs(self):
        cards = parse_cards(BACKLOG)
        self.assertEqual([card.slug for card in cards], ["api-pagination", "search-filters", "search-filters-2"])
        self.assertEqual(cards[0].owner, "@ana")
        self.assertEqual(cards[0].estimate, 2)
        self.assertEqual(cards[0].tags, ["api", "performance"])
        self.assertEqual(cards[1].depends_on, ["api-pagination"])
        self.assertIn("filter by owner", cards[1].body)
        self.assertIsNone(cards[2].owner)
        self.assertEqual(cards[2].tags, [])

    def test_parse_cards_rejects_invalid_estimates_with_context(self):
        with self.assertRaisesRegex(ValueError, "Bad Estimate"):
            parse_cards("## Bad Estimate\n- estimate: nope\n")
        with self.assertRaisesRegex(ValueError, "Positive"):
            parse_cards("## Positive\n- estimate: 0\n")


class SchedulerTests(unittest.TestCase):
    def test_schedule_respects_capacity_and_dependency_order(self):
        cards = parse_cards(BACKLOG)
        sprints = schedule_cards(cards, capacity=3)
        self.assertEqual([[card.slug for card in sprint] for sprint in sprints], [["api-pagination"], ["search-filters"], ["search-filters-2"]])

    def test_schedule_detects_unknown_or_cyclic_dependencies(self):
        cards = parse_cards("## A\n- depends_on: missing\n")
        with self.assertRaisesRegex(ValueError, "missing"):
            schedule_cards(cards, capacity=3)

        cycle = parse_cards("## A\n- depends_on: b\n\n## B\n- depends_on: a\n")
        with self.assertRaisesRegex(ValueError, "cycle|cyclic|dependency"):
            schedule_cards(cycle, capacity=3)

    def test_schedule_rejects_bad_capacity_and_oversized_cards(self):
        cards = parse_cards("## A\n- estimate: 4\n")
        with self.assertRaisesRegex(ValueError, "capacity"):
            schedule_cards(cards, capacity=0)
        with self.assertRaisesRegex(ValueError, "capacity"):
            schedule_cards(cards, capacity=3)


class ReportAndCliTests(unittest.TestCase):
    def test_render_report_is_readable_markdown(self):
        cards = parse_cards(BACKLOG)
        report = render_report(schedule_cards(cards, 5))
        self.assertIn("# Sprint Plan", report)
        self.assertIn("## Sprint 1", report)
        self.assertIn("- [ ] API Pagination `api-pagination` - 2 pts - @ana - api, performance", report)
        self.assertIn("depends on: api-pagination", report)

    def test_cli_outputs_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backlog.md"
            path.write_text(BACKLOG)
            json_run = subprocess.run(
                [sys.executable, "-m", "miniboard", str(path), "--capacity", "5", "--json"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            payload = json.loads(json_run.stdout)
            self.assertEqual(payload[0][0]["slug"], "api-pagination")

            text_run = subprocess.run(
                [sys.executable, "-m", "miniboard", str(path), "--capacity", "5"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertIn("Sprint Plan", text_run.stdout)


if __name__ == "__main__":
    unittest.main()
EOF

cat >"$WORK_DIR/.gitignore" <<'EOF'
__pycache__/
*.pyc
EOF

git -C "$WORK_DIR" init -q
git -C "$WORK_DIR" add .
git -C "$WORK_DIR" -c user.name=Marathon -c user.email=marathon@example.invalid commit -q -m "seed coding gauntlet"

prompt="Read brief.md and the repository. Implement the smallest production-quality fix that satisfies the brief and included tests. Preserve the public API, do not add dependencies, and run the test suite before finishing."

printf 'Marathon coding eval\n'
printf 'model: %s\n' "$active_model"
printf 'workspace: %s\n' "$WORK_DIR"
printf 'timeout: %ss\n\n' "$TIMEOUT_SECONDS"

set +e
start_time="$(date +%s)"
(
  cd "$WORK_DIR"
  MARATHON_WEB_SEARCH_MODE=disabled timeout "$TIMEOUT_SECONDS" "$ROOT_DIR/bin/marathon" exec --sandbox workspace-write "$prompt"
) 2>&1 | tee "$LOG_FILE"
exec_rc="${PIPESTATUS[0]}"
end_time="$(date +%s)"

(
  cd "$WORK_DIR"
  python3 -m unittest discover -s tests -v
) >"$TEST_LOG" 2>&1
test_rc="$?"
set -e

find "$WORK_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +

changed_files="$(git -C "$WORK_DIR" status --short | wc -l | tr -d ' ')"
diffstat="$(git -C "$WORK_DIR" diff --stat || true)"
tokens_used="$(sed -n '/tokens used/{n;p;}' "$LOG_FILE" | tail -n1 | tr -d '[:space:]')"

printf '\nResult\n'
printf 'model: %s\n' "$active_model"
printf 'exec_exit: %s\n' "$exec_rc"
printf 'test_exit: %s\n' "$test_rc"
printf 'duration_seconds: %s\n' "$((end_time - start_time))"
printf 'changed_files: %s\n' "$changed_files"
if [[ -n "$tokens_used" ]]; then
  printf 'tokens_used: %s\n' "$tokens_used"
fi
printf 'log: %s\n' "$LOG_FILE"
printf 'test_log: %s\n' "$TEST_LOG"

printf '\nTest summary\n'
tail -n 40 "$TEST_LOG"

if [[ -n "$diffstat" ]]; then
  printf '\nDiffstat\n%s\n' "$diffstat"
fi

printf '\nWorkspace kept for inspection: %s\n' "$WORK_DIR"

if [[ "$exec_rc" != "0" || "$test_rc" != "0" ]]; then
  exit 1
fi
