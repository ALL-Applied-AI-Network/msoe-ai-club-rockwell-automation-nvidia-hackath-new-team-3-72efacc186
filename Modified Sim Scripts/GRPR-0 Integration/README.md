# GRPR-0 Isaac Sim integration

This package converts the two metre-scale SolidWorks STL finger meshes into the
starter's triangle-mesh JSON format and adds them as a selectable finger pair on
the existing SMC MHZL2-16D2 actuator.

## Select the variant

The calibrated starter remains the default.  Select the new fingers for any
Isaac/host command with:

```bash
export FANUC_GRIPPER_VARIANT=grpr_0
```

Unset the variable (or set it to `baseline`) to restore the original fingers.
On the configured Spark, a complete run is:

```bash
FANUC_GRIPPER_VARIANT=grpr_0 \
  /home/yt-newteam3b12c/workspace/spark-launchers/run-fanuc-pipeline.sh \
  runs/grpr0_next_run
```

## Geometry-derived initial values

- Finger envelope: 41.0 x 16.18 x 14.4 mm
- Closed gap: 2.9 mm
- Open gap with the existing 6 mm/jaw actuator stroke: 14.9 mm
- Wrist-centre-to-tip length: 258 mm
- Initial 10 mm cube closure target: 9.4 mm gap

The updated source is one STEP finger and the opposite finger is its mirror.
The checked-in `assets/gripper/grpr_0_right.stl` is millimetre-scale. Rebuild
the paired asset with:

```powershell
python .\convert_gripper_stl.py .\assets\gripper\grpr_0_right.stl `
  --scale 0.001 --grpr0-local-frame --mirror-pair `
  --names finger_l finger_r `
  --output .\assets\gripper\grpr_0.json
```

The fingers are provisionally modeled as PLA at 1,240 kg/m3.  The STEP volume
gives 6.081 g per finger if the prints are completely solid.  Isaac uses initial
PLA contact coefficients of 0.30 static and 0.25 dynamic.  Infill, print
orientation, surface finish and the cube material can change all three values,
so measured finger masses and a simple slip test should replace these estimates.
The total jaw-body mass remains the existing 32 g estimate because the SMC
carriage mass is not known separately.  The high-detail meshes are rendered
directly; PhysX cooks separate invisible `convexDecomposition` collision meshes.

## Files

- `assets/gripper/grpr_0.json`: converted jaw-local meshes
- `source/scene_config.py`: selectable dimensions and TCP
- `source/cell.py`: mesh loading and collision integration
- `source/package_hackathon.py`: package allowlist update
- `convert_gripper_stl.py`: dependency-free conversion/inspection tool
- `launchers/`: wrappers updated to forward `FANUC_GRIPPER_VARIANT` into Docker

## Spark validation

`runs/grpr0_validation_01` replayed seed `1458622090` with a completely fresh
capture and Cosmos perception pass.  Results:

- 33 baseline host tests passed
- Isaac scene and 4K capture completed without GRPR-0/PhysX collision errors
- Cosmos detected exactly six cubes; the planner accepted six picks
- Physics-only execution completed 6/6 pickups and 6/6 verified placements
- `execution_report.json` recorded `task_complete: true`

The pre-integration Spark files and launchers are backed up under
`fanuc_er4ia_sim/backups/grpr0-20261029T031011Z/`.

Validate scale, handedness, mounting position and collision cooking in Isaac
before treating any simulation comparison as meaningful.  Physical deployment
also requires a measured TCP, measured finger/jaw mass, PLA-to-part friction
calibration and a controller-side UTOOL update.
