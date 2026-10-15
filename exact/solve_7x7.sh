#!/usr/bin/env bash
set -euo pipefail
if (( $# > 2 )); then
  echo 'Usage: bash solve_7x7.sh [POSITION] [SECONDS]' >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
POSITION="${1:-}"
SECONDS_LIMIT="${2:-60}"
if [[ ! "$POSITION" =~ ^([123][A-G])*$ ]]; then
  echo 'POSITION must use uppercase tile+column tokens, e.g. 2A1G2A1G2A1F.' >&2
  exit 2
fi
if [[ ! "$SECONDS_LIMIT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo 'SECONDS must be non-negative; 0 means no time limit.' >&2
  exit 2
fi
make -C "$SCRIPT_DIR" release
mkdir -p "$SCRIPT_DIR/results"
RUN_DIR="$(mktemp -d "$SCRIPT_DIR/results/run-XXXXXXXX")"
echo "Output directory: $RUN_DIR"
set +e
"$SCRIPT_DIR/Make7DFPNv9_8_3" \
  --rows 7 --threads 1 --perspective green \
  --position "$POSITION" --time-limit "$SECONDS_LIMIT" \
  --memory-gb 0.25 --shared-tt-gb 0.25 --worker-tt-mb 64 \
  --child-cache-mb 8 --worker-cache-mb 8 \
  --no-deep-reroot --no-frontier --no-deep-frontier \
  --log "$RUN_DIR/progress.csv" --summary "$RUN_DIR/summary.csv" \
  2>&1 | tee "$RUN_DIR/console.txt"
solver_status=${PIPESTATUS[0]}
set -e
if [[ "$solver_status" == 4 ]]; then
  echo 'UNKNOWN: the position was not solved within this run; it is not a draw.'
fi
exit "$solver_status"
