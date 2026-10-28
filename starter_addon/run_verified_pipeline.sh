#!/usr/bin/env bash
# Vessels of Cosmos: starter pipeline + second-camera verification.
# Same stages as run_sort_pipeline.sh (unchanged), plus:
#   1b. capture the calibrated SIDE camera of the same static scene
#   3b. cross-check the plan in the side view; block on conflict
#       (VERIFY_MODE=--strict blocks unless every cube is confirmed)
#   5b. audit the finished tower from the side camera (vision only)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaacsim}"
: "${COSMOS:?Set COSMOS to the Python interpreter in your Cosmos environment}"
HOST_PYTHON="${HOST_PYTHON:-$HERE/.venv/bin/python}"
for executable in "$ISAAC/python.sh" "$COSMOS" "$HOST_PYTHON"; do
    if [ ! -x "$executable" ]; then
        echo "Python launcher missing or not executable: $executable" >&2
        exit 1
    fi
done
RUN="${1:-$HERE/runs/verified_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
if [ -n "$(ls -A "$RUN")" ]; then
    echo "Use an empty run directory: $RUN" >&2
    exit 1
fi
"$ISAAC/python.sh" "$HERE/capture_scene.py" --out "$RUN"
"$ISAAC/python.sh" "$HERE/capture_scene.py" --out "$RUN/side_view" --camera side
"$COSMOS" "$HERE/perceive.py" --run "$RUN" --stack-cubes
"$HOST_PYTHON" "$HERE/plan_sort.py" --run "$RUN"
"$HOST_PYTHON" "$HERE/verify_views.py" picks --run "$RUN" --side "$RUN/side_view" ${VERIFY_MODE:---enforce}
"$ISAAC/python.sh" "$HERE/sort_scene.py" --run "$RUN"
"$HOST_PYTHON" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
report = Path(sys.argv[1]) / 'execution_report.json'
if not report.is_file():
    raise SystemExit('Execution report missing; inspect the simulator output.')
result = json.loads(report.read_text())
if not result.get('task_complete'):
    raise SystemExit(f"Stack incomplete: {result.get('abort_reason')}; inspect {report}")
PY
# Vision-only audit of the finished tower (does not read simulator state).
"$HOST_PYTHON" "$HERE/verify_views.py" stack --run "$RUN" --side "$RUN/side_view" || \
    echo "WARNING: side-camera stack audit did not confirm the tower; see $RUN/stack_audit.json"
"$HOST_PYTHON" "$HERE/encode_video.py" --frames "$RUN/frames" --out "$RUN/sort_table.mp4" --fps 30
"$HOST_PYTHON" "$HERE/encode_video.py" --frames "$RUN/side_frames" --out "$RUN/sort_side.mp4" --fps 30
echo "Review $RUN: view_verification.json, side_verification_overlay.png, stack_audit.json, stack_audit_overlay.png, plus the usual starter outputs"
