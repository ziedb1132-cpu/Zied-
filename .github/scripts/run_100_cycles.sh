#!/usr/bin/env bash
# Run 100 sequential paper-trading cycles, persisting the full learning state
# to state-data after every successful cycle.
set -euo pipefail

CYCLES="${CYCLES_PER_WORKFLOW:-30}"
if ! [[ "$CYCLES" =~ ^[0-9]+$ ]] || [ "$CYCLES" -lt 1 ]; then
  echo "CYCLES_PER_WORKFLOW must be a positive integer" >&2
  exit 2
fi

STATE_BACKUP="$RUNNER_TEMP/zied-state-backup"
mkdir -p "$STATE_BACKUP"
cp "$GITHUB_WORKSPACE/state_chunks.py" "$RUNNER_TEMP/state_chunks.py"

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
  git add -f zied_data_store
  if git diff --cached --quiet; then
    echo "ERROR: no state chunks produced after cycle $cycle." >&2
    git checkout --detach -f "$GITHUB_SHA"
    cp -a "$STATE_BACKUP/zied_data" ./zied_data
    git branch -D "$orphan" >/dev/null 2>&1 || true
    return 1
  fi
  git commit -m "Persist quant state after cycle $cycle [skip ci]"
  git push --force origin HEAD:state-data

  # Return to the original code commit so the next cycle runs the same code.
  git checkout --detach -f "$GITHUB_SHA"
  rm -rf zied_data zied_data_store
  cp -a "$STATE_BACKUP/zied_data" ./zied_data
  git branch -D "$orphan" >/dev/null 2>&1 || true
  echo "Durable state saved after cycle $cycle/$CYCLES."
}

for cycle in $(seq 1 "$CYCLES"); do
  echo "========== QUANT CYCLE $cycle/$CYCLES =========="
  python evolved_runner.py once
  persist_state "$cycle"
done

echo "Completed $CYCLES successful cycles; durable state was saved after each cycle."
