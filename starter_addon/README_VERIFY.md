# Second-camera verification add-on (Vessels of Cosmos)

Adds three files to the organiser starter. **No starter file is modified.**

| File | Purpose |
| --- | --- |
| `verify_views.py` | Cross-checks the table-camera plan in the calibrated side camera; audits the finished tower |
| `test_verify_views.py` | 15 tests using synthetic side-camera renders (no Isaac/Cosmos needed) |
| `run_verified_pipeline.sh` | `run_sort_pipeline.sh` plus side capture, pre-execution check and post-execution audit |

## What it adds

The starter plans every grasp from a single table camera. Before the robot moves,
`verify_views.py picks` projects each planned cube (position + yaw) into the
side camera and checks the pixels there for the expected colour:

- `confirmed`: the side view agrees, and the lateral (x) disagreement between views is reported in mm
- `conflict`: a different cube colour sits there, so with `--enforce` the pipeline stops (exit 2)
- `occluded`: hidden by a nearer cube or a larger object such as a cap, so it is unverified and falls back to the starter gates
- `not_seen`: no cube colour there (phantom detection, mislocalisation or a clear-tube occluder)
- `out_of_view`

Blocking policy: `--enforce` blocks only on `conflict` (the default, a fallback policy). `--strict` blocks unless **every** cube is `confirmed` (full two-camera confirmation).

After execution, `verify_views.py stack` reads each tower layer's colour from the
last side-camera frame. This is a vision-only check of the result that does not use simulator object poses, so it
can be reused with a calibrated camera on the physical cell.

## Run

```bash
# tests (host Python)
python -m unittest test_verify_views -v

# full pipeline with verification (Isaac + Cosmos machine)
./run_verified_pipeline.sh runs/verified_01

# or add to an existing run
"$ISAAC/python.sh" capture_scene.py --out runs/X/side_view --camera side
python verify_views.py picks --run runs/X --side runs/X/side_view --enforce
python verify_views.py stack --run runs/X --side runs/X/side_view
```

Outputs in the run folder: `view_verification.json`, `side_verification_overlay.png`,
`stack_audit.json`, `stack_audit_overlay.png`.

## Limits

The tests use synthetic renders. Real Isaac side captures still need checking
(lighting, the parked arm in view). Colour gates match the starter's HSV gates.
The side camera is about 14 cm above the table, so it measures x (image-horizontal)
well and y (depth) poorly. Only the lateral offset is reported.
