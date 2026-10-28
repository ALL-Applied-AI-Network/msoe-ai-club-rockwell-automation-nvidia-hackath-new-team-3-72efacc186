"""Tests for verify_views.py using synthetic side-camera renders.

No Isaac Sim or Cosmos needed. A tiny painter's-algorithm renderer draws
cubes and caps through the starter's calibrated SIDE camera model, so the
verification logic is exercised with exact, known answers.

    python -m unittest test_verify_views -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import scene_config as C
from backproject import CameraModel
import verify_views as V

BGR = {
    "yellow": (70, 195, 215),  # pale yellow like the sim cubes
    "blue": (190, 85, 35),
    "red": (40, 40, 200),
    "cap": (25, 200, 225),  # saturated yellow cap
}
BACKGROUND = (215, 215, 215)


def camera_json_from_config(cfg, name):
    """Same transform cell.camera() builds in USD (Gf SetLookAt, inverted)."""
    eye, tgt, up = (np.array(cfg[k], float) for k in ("eye", "look_at", "up"))
    z = eye - tgt
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3], M[3, :3] = x, y, z, eye
    w, h = cfg["resolution"]
    return {
        "camera": name,
        "resolution": [w, h],
        "focal_length_mm": cfg["focal_length"],
        "horizontal_aperture_mm": cfg["horiz_aperture"],
        "vertical_aperture_mm": cfg["horiz_aperture"] * h / w,
        "camera_to_world": M.tolist(),
        "table_top_z": C.TABLE_TOP_Z + C.SURFACE_DZ,
        "robot_base_xy": [0.0, 0.0],
    }


SIDE_JSON = camera_json_from_config(C.SIDE_CAM, "side")


def cap_points(xy, z0, radius=C.CAP_RADIUS, height=C.CAP_HEIGHT, n=24):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([xy[0] + radius * np.cos(a), xy[1] + radius * np.sin(a)], 1)
    return np.array([[*p, z] for z in (z0, z0 + height) for p in ring])


def render(cam, objects):
    """objects: list of (kind, colour, xy, yaw_deg, level). Far objects drawn first."""
    img = np.full((cam.H, cam.W, 3), BACKGROUND, np.uint8)
    drawn = []
    for kind, colour, xy, yaw, level in objects:
        z0 = cam.table_z + level * C.CUBE_EDGE
        pts = cap_points(xy, z0) if kind == "cap" else V.cube_corners(xy, yaw, z0)
        depth = np.linalg.norm(pts.mean(0) - cam.eye)
        drawn.append((depth, pts, BGR["cap" if kind == "cap" else colour]))
    for _, pts, bgr in sorted(drawn, key=lambda d: -d[0]):
        uv = np.array([cam.project(p) for p in pts], np.float32)
        hull = cv2.convexHull(uv)
        cv2.fillConvexPoly(img, np.round(hull * 16).astype(np.int32), bgr, cv2.LINE_AA, 4)
    return img


def pick(colour, xy, yaw=0.0, did=0, level=0):
    return {"detection_id": did, "category": "cube", "colour": colour,
            "xy": list(xy), "yaw_deg": yaw, "stack_level": level,
            "place_xy": [0.28, 0.10], "place_yaw_deg": 0.0}


class VerifyPicksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = CameraModel(SIDE_JSON)

    def test_projection_lands_in_side_image(self):
        uv = self.cam.project([0.40, 0.05, self.cam.table_z + 0.005])
        self.assertTrue(0 < uv[0] < self.cam.W and 0 < uv[1] < self.cam.H)

    def test_correct_plan_is_confirmed(self):
        world = [("cube", "yellow", (0.30, 0.05), 10, 0),
                 ("cube", "blue", (0.40, 0.05), -30, 0),
                 ("cube", "red", (0.50, 0.05), 20, 0)]
        plan = [pick(c, xy, yaw, i) for i, (_, c, xy, yaw, _) in enumerate(world)]
        res = V.verify_picks(render(self.cam, world), self.cam, plan)
        self.assertEqual([r["status"] for r in res], ["confirmed"] * 3)
        for r in res:
            self.assertLess(abs(r["side_lateral_offset_mm"]), 0.5)
        self.assertTrue(V.summarise_picks(res)["safe_to_execute"])

    def test_wrong_colour_is_conflict(self):
        world = [("cube", "red", (0.40, 0.05), 0, 0)]
        res = V.verify_picks(render(self.cam, world), self.cam, [pick("blue", (0.40, 0.05))])
        self.assertEqual(res[0]["status"], "conflict")
        self.assertEqual(res[0]["observed_colour"], "red")
        self.assertFalse(V.summarise_picks(res)["safe_to_execute"])

    def test_large_cap_in_front_is_occluded_not_conflict(self):
        # A yellow cap (not in the cube plan) directly in front of a blue cube.
        # The side camera is ~14 cm above the table, so the occluder has to be
        # close to the cube to hide it.
        world = [("cube", "blue", (0.40, 0.05), 0, 0), ("cap", "yellow", (0.40, 0.08), 0, 0)]
        res = V.verify_picks(render(self.cam, world), self.cam, [pick("blue", (0.40, 0.05))])
        self.assertEqual(res[0]["status"], "occluded")

    def test_cap_in_front_of_yellow_cube_is_not_false_confirmation(self):
        world = [("cube", "yellow", (0.40, 0.05), 0, 0), ("cap", "yellow", (0.40, 0.08), 0, 0)]
        res = V.verify_picks(render(self.cam, world), self.cam, [pick("yellow", (0.40, 0.05))])
        self.assertEqual(res[0]["status"], "occluded")

    def test_missing_cube_is_not_seen(self):
        res = V.verify_picks(render(self.cam, []), self.cam, [pick("red", (0.40, 0.05))])
        self.assertEqual(res[0]["status"], "not_seen")

    def test_nearer_planned_cube_occludes_farther(self):
        world = [("cube", "red", (0.40, 0.062), 0, 0), ("cube", "blue", (0.40, 0.05), 0, 0)]
        plan = [pick("red", (0.40, 0.062), did=0), pick("blue", (0.40, 0.05), did=1)]
        res = V.verify_picks(render(self.cam, world), self.cam, plan)
        self.assertEqual(res[0]["status"], "confirmed")
        self.assertEqual(res[1]["status"], "occluded")

    def test_lateral_position_error_is_measured(self):
        world = [("cube", "yellow", (0.403, 0.05), 0, 0)]  # 3 mm off in x
        res = V.verify_picks(render(self.cam, world), self.cam, [pick("yellow", (0.40, 0.05))])
        self.assertEqual(res[0]["status"], "confirmed")
        self.assertAlmostEqual(abs(res[0]["side_lateral_offset_mm"]), 3.0, delta=0.6)

    def test_out_of_view(self):
        res = V.verify_picks(render(self.cam, []), self.cam, [pick("red", (1.5, 0.05))])
        self.assertEqual(res[0]["status"], "out_of_view")


class StackAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = CameraModel(SIDE_JSON)
        cls.order = ["yellow", "yellow", "blue", "blue", "red", "red"]

    def tower(self, colours):
        return [("cube", c, (0.28, 0.10), 0, k) for k, c in enumerate(colours)]

    def test_correct_tower_verified(self):
        img = render(self.cam, self.tower(self.order))
        audit = V.audit_stack(img, self.cam, (0.28, 0.10), self.order)
        self.assertTrue(audit["stack_verified_by_vision"])
        self.assertEqual(audit["layers_matched"], 6)

    def test_swapped_layers_detected(self):
        wrong = ["yellow", "yellow", "red", "blue", "blue", "red"]
        audit = V.audit_stack(render(self.cam, self.tower(wrong)), self.cam, (0.28, 0.10), self.order)
        self.assertFalse(audit["stack_verified_by_vision"])
        self.assertEqual([l["match"] for l in audit["layers"]], [True, True, False, True, False, True])

    def test_incomplete_tower_detected(self):
        audit = V.audit_stack(render(self.cam, self.tower(self.order[:4])), self.cam, (0.28, 0.10), self.order)
        self.assertFalse(audit["stack_verified_by_vision"])
        self.assertEqual(audit["layers"][4]["observed"], "empty_or_unknown")


class CommandLineTest(unittest.TestCase):
    def test_enforce_blocks_conflicting_plan(self):
        cam = CameraModel(SIDE_JSON)
        with tempfile.TemporaryDirectory() as d:
            run, side = Path(d), Path(d) / "side_view"
            side.mkdir()
            (side / "camera.json").write_text(json.dumps(SIDE_JSON))
            cv2.imwrite(str(side / "world_camera.png"),
                        render(cam, [("cube", "red", (0.40, 0.05), 0, 0)]))
            plan = {"task": "cube_stack", "picks": [pick("blue", (0.40, 0.05))]}
            (run / "sort_plan.json").write_text(json.dumps(plan))
            code = V.main(["picks", "--run", str(run), "--side", str(side), "--enforce"])
            self.assertEqual(code, 2)
            report = json.loads((run / "view_verification.json").read_text())
            self.assertEqual(report["summary"]["conflict"], 1)
            self.assertTrue((run / "side_verification_overlay.png").is_file())


    def _run_cli(self, world, plan_picks, *flags):
        cam = CameraModel(SIDE_JSON)
        with tempfile.TemporaryDirectory() as d:
            run, side = Path(d), Path(d) / "side_view"
            side.mkdir()
            (side / "camera.json").write_text(json.dumps(SIDE_JSON))
            cv2.imwrite(str(side / "world_camera.png"), render(cam, world))
            (run / "sort_plan.json").write_text(json.dumps({"task": "cube_stack", "picks": plan_picks}))
            return V.main(["picks", "--run", str(run), "--side", str(side), *flags])

    def test_strict_blocks_unconfirmed_but_enforce_allows(self):
        world = [("cube", "blue", (0.40, 0.05), 0, 0), ("cap", "yellow", (0.40, 0.08), 0, 0)]
        plan = [pick("blue", (0.40, 0.05))]  # occluded by the cap
        self.assertEqual(self._run_cli(world, plan, "--enforce"), 0)
        self.assertEqual(self._run_cli(world, plan, "--strict"), 2)

    def test_strict_passes_when_all_confirmed(self):
        world = [("cube", "red", (0.40, 0.05), 0, 0)]
        self.assertEqual(self._run_cli(world, [pick("red", (0.40, 0.05))], "--strict"), 0)


if __name__ == "__main__":
    unittest.main()
