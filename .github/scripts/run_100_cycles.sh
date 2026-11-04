#!/usr/bin/env bash
# Run one complete paper-trading cycle and persist the full learning state.
set -euo pipefail

CYCLES="${CYCLES_PER_WORKFLOW:-1}"
if ! [[ "$CYCLES" =~ ^[0-9]+$ ]] || [ "$CYCLES" -lt 1 ]; then
  echo "CYCLES_PER_WORKFLOW must be a positive integer" >&2
  exit 2
fi

STATE_BACKUP="$RUNNER_TEMP/zied-state-backup"
mkdir -p "$STATE_BACKUP"
cp "$GITHUB_WORKSPACE/state_chunks.py" "$RUNNER_TEMP/state_chunks.py"
cp "$GITHUB_WORKSPACE/dashboard_export.py" "$RUNNER_TEMP/dashboard_export.py"

persist_state() {
  local cycle="$1"
  rm -rf "$STATE_BACKUP/zied_data"
  if [ ! -d zied_data ]; then
    echo "ERROR: zied_data is missing after cycle $cycle; refusing to overwrite durable state." >&2
    return 1
  fi
  cp -a zied_data "$STATE_BACKUP/zied_data"

  git config user.name "github-actions[bot]"
  git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

  local orphan="state-data-cycle-$cycle"
  git checkout --orphan "$orphan"
  git rm -rf . >/dev/null 2>&1 || true
  rm -rf zied_data zied_data_store
  python "$RUNNER_TEMP/state_chunks.py" pack "$STATE_BACKUP/zied_data" zied_data_store
  # Fetch only the compact Strategy Lab summary; dashboard still publishes one JSON file.
  if ! curl -fsSL --retry 2 --connect-timeout 10 \
    "https://raw.githubusercontent.com/ziedb1132-cpu/Zied-Strategy-Lab/main/data/dashboard_summary.json" \
    -o "$RUNNER_TEMP/strategy_lab_summary.json"; then
    printf '%s\\n' '{"updated_at":null,"cycle":0,"strategies_discovered":0,"successful_buy_strategies":0,"strategies":[]}' > "$RUNNER_TEMP/strategy_lab_summary.json"
    echo "WARNING: Strategy Lab summary unavailable; preserving the single-file dashboard format with empty summary."
  fi
  python "$RUNNER_TEMP/dashboard_export.py" "$STATE_BACKUP/zied_data" "$RUNNER_TEMP/dashboard_data.json" "$RUNNER_TEMP/strategy_lab_summary.json"
  cp "$RUNNER_TEMP/dashboard_data.json" dashboard_data.json
  git add -f zied_data_store dashboard_data.json
  if git diff --cached --quiet; then
    echo "ERROR: no state chunks produced after cycle $cycle." >&2
    git checkout --detach -f "$GITHUB_SHA"
    cp -a "$STATE_BACKUP/zied_data" ./zied_data
    git branch -D "$orphan" >/dev/null 2>&1 || true
    return 1
  fi
  git commit -m "Persist quant state after cycle $cycle [skip ci]"
  git push --force origin HEAD:state-data

  git checkout --detach -f "$GITHUB_SHA"
  rm -rf zied_data zied_data_store
  cp -a "$STATE_BACKUP/zied_data" ./zied_data
  git branch -D "$orphan" >/dev/null 2>&1 || true
  echo "Durable state saved after cycle $cycle/$CYCLES."
}

for cycle in $(seq 1 "$CYCLES"); do
  echo "========== COMPLETE QUANT CYCLE $cycle/$CYCLES =========="
  python evolved_runner.py once
  persist_state "$cycle"
done

echo "Completed $CYCLES complete cycle(s); durable state was saved after each cycle."
