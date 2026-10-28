"""
Second-camera verification for the cube-stack pipeline (host Python).

Team: Vessels of Cosmos. This is an add-on to the organiser starter. It does
not change any starter file.

The starter plans every grasp from ONE camera (the 4K table camera). This
module uses the starter's second calibrated camera (the side camera) to check
that plan independently:

  picks  - BEFORE execution. Each planned cube pose from sort_plan.json is
           projected into the side image. The pixels where the cube should
           appear are checked for the expected colour. Each cube gets a verdict:
             confirmed    side view sees the expected colour at that spot
             conflict     side view sees a DIFFERENT cube colour there
             occluded     hidden behind a nearer cube or a larger object (cap)
             not_seen     nothing coloured where the cube should be
             out_of_view  projects outside the side image
           With --enforce, any conflict stops the pipeline (exit code 2)
           so the robot never moves on a contradicted observation.
           With --strict, the robot moves only if EVERY cube is confirmed
           by both cameras (full two-camera confirmation).

  stack  - AFTER execution. The expected tower layers are projected into a
           side-camera frame and the colour of each layer is read back from
           pixels, giving a vision-only audit of the finished stack.

Only images, camera calibration (camera.json written by capture_scene.py)
and sort_plan.json are used. Ground truth / simulator object poses are never
read, so the same checks can run on a physical cell with a calibrated camera.

Examples (run from the starter folder):
    python verify_views.py picks --run runs/stack_01 --side runs/stack_01/side_view --enforce
    python verify_views.py stack --run runs/stack_01 --side runs/stack_01/side_view \
        --image runs/stack_01/side_frames/LAST
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backproject import CameraModel  # noqa: E402
import scene_config as C  # noqa: E402

COLOURS = ("yellow", "blue", "red")

# Thresholds. Fractions are of the cube's VISIBLE projected pixels.
CONFIRM_FRACTION = 0.50  # expected colour must cover at least this much
CONFLICT_FRACTION = 0.50  # another cube colour covering this much -> conflict
MIN_VISIBLE_FRACTION = 0.35  # less visible than this -> occluded
MIN_VISIBLE_PIXELS = 25  # too few pixels to judge -> occluded
LARGE_OBJECT_FACTOR = 3.0  # colour blob > 3x cube area is not a cube (e.g. cap)
LAYER_CORE = 0.5  # stack audit samples the central 50% of each layer


# ---------------------------------------------------------------- colour
def colour_masks(img_bgr):
    """Same HSV gates as the starter's cube_pose.refine_pose."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sat = (s >= 45) & (v >= 35)
    return {
        "yellow": sat & (h >= 12) & (h <= 42),
        "blue": sat & (h >= 95) & (h <= 135),
        "red": sat & ((h <= 11) | (h >= 170)),
    }


# ---------------------------------------------------------------- geometry
def cube_corners(xy, yaw_deg, z_bottom, edge=C.CUBE_EDGE):
    """8 world corners of an upright cube with its bottom face at z_bottom."""
    t = np.deg2rad(yaw_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = []
    for z in (z_bottom, z_bottom + edge):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = np.asarray(xy, float) + rot @ np.array([sx, sy]) * edge / 2
            pts.append([x, y, z])
    return np.array(pts)


def projected_mask(cam, corners, shape):
    """Filled convex silhouette of 3D points in the image; None if behind camera."""
    uv = [cam.project(p) for p in corners]
    if any(p is None for p in uv):
        return None
    hull = cv2.convexHull(np.asarray(uv, np.float32))
    mask = np.zeros(shape[:2], np.uint8)
    # 4 fractional bits keep sub-pixel accuracy for small 10 mm cubes
    cv2.fillConvexPoly(mask, np.round(hull * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    return mask.astype(bool)


def shrink_mask(mask, keep):
    """Keep the central `keep` fraction of a mask's bounding box (stack layers)."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return mask
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    hw, hh = (xs.max() - xs.min()) * keep / 2, (ys.max() - ys.min()) * keep / 2
    core = np.zeros_like(mask)
    core[
        int(np.floor(cy - hh)) : int(np.ceil(cy + hh)) + 1,
        int(np.floor(cx - hw)) : int(np.ceil(cx + hw)) + 1,
    ] = True
    return mask & core


def focal_px(cam):
    return cam.W / (2.0 * cam.tan_h)


# ---------------------------------------------------------------- loading
def load_camera(side_dir_or_json):
    p = Path(side_dir_or_json)
    if p.is_dir():
        p = p / "camera.json"
    return CameraModel(str(p))


def load_image(path, cam):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    if img.shape[1] != cam.W or img.shape[0] != cam.H:
        raise SystemExit(
            f"Image {path} is {img.shape[1]}x{img.shape[0]} but camera.json says "
            f"{cam.W}x{cam.H}; use the image captured with that camera.json"
        )
    return img


def cube_pose(pick):
    """Planned pose from sort_plan.json (x, y in robot frame; yaw in degrees)."""
    return np.asarray(pick["xy"], float), float(pick.get("yaw_deg", 0.0))


# ---------------------------------------------------------------- picks check
def verify_picks(img, cam, picks):
    """Cross-check planned cube poses against an independent camera view."""
    masks = colour_masks(img)
    H, W = img.shape[:2]
    z0 = cam.table_z
    fpx = focal_px(cam)

    # Silhouette and depth of every planned cube (used for mutual occlusion).
    entries = []
    for i, pick in enumerate(picks):
        xy, yaw = cube_pose(pick)
        sil = projected_mask(cam, cube_corners(xy, yaw, z0), img.shape)
        centre = np.array([*xy, z0 + C.CUBE_EDGE / 2])
        entries.append(
            {
                "sil": sil,
                "depth": float(np.linalg.norm(centre - cam.eye)),
                "centre": centre,
            }
        )

    results = []
    for i, (pick, e) in enumerate(zip(picks, entries)):
        expected = pick.get("colour")
        res = {
            "detection_id": pick.get("detection_id", i),
            "expected_colour": expected,
            "planned_xy_m": [round(float(v), 5) for v in pick["xy"]],
            "stack_level": pick.get("stack_level"),
        }
        sil = e["sil"]
        if sil is None or not sil.any():
            res.update(status="out_of_view")
            results.append(res)
            continue
        uv = cam.project(e["centre"])
        if not (0 <= uv[0] < W and 0 <= uv[1] < H):
            res.update(status="out_of_view")
            results.append(res)
            continue

        # Remove pixels covered by nearer planned cubes.
        visible = sil.copy()
        for j, other in enumerate(entries):
            if j != i and other["sil"] is not None and other["depth"] < e["depth"]:
                visible &= ~other["sil"]
        area = int(sil.sum())
        n_vis = int(visible.sum())
        res["projected_px"] = area
        res["visible_fraction"] = round(n_vis / max(area, 1), 3)
        if n_vis < MIN_VISIBLE_PIXELS or n_vis / area < MIN_VISIBLE_FRACTION:
            res.update(status="occluded", reason="behind another planned cube")
            results.append(res)
            continue

        fractions = {c: float((masks[c] & visible).sum()) / n_vis for c in COLOURS}
        res["colour_fractions"] = {c: round(f, 3) for c, f in fractions.items()}
        best_other = max(
            (c for c in COLOURS if c != expected), key=lambda c: fractions[c]
        )

        if fractions.get(expected, 0.0) >= CONFIRM_FRACTION:
            blob = _blob_touching(masks[expected], visible)
            if blob is not None and blob.sum() > LARGE_OBJECT_FACTOR * area:
                # e.g. a yellow cube hidden behind a (much larger) yellow cap
                res.update(
                    status="occluded",
                    reason=f"{expected} region is {blob.sum() / area:.1f}x cube size "
                    "(larger object in front)",
                )
            else:
                res["status"] = "confirmed"
                if blob is not None and res["visible_fraction"] >= 0.9:
                    # Lateral (image-x) disagreement between the two cameras.
                    obs_u = np.nonzero(blob)[1].mean()
                    pred_u = np.nonzero(sil)[1].mean()
                    res["side_lateral_offset_mm"] = round(
                        float((obs_u - pred_u) * e["depth"] / fpx * 1000), 2
                    )
        elif fractions[best_other] >= CONFLICT_FRACTION:
            blob = _blob_touching(masks[best_other], visible)
            if blob is not None and blob.sum() > LARGE_OBJECT_FACTOR * area:
                res.update(
                    status="occluded",
                    reason=f"{best_other} object {blob.sum() / area:.1f}x cube size in front",
                )
            else:
                res.update(
                    status="conflict",
                    observed_colour=best_other,
                    reason=f"side view sees a {best_other} cube-sized object, "
                    f"plan says {expected}",
                )
        else:
            res.update(
                status="not_seen",
                reason="no cube colour where the planned cube should be "
                "(transparent occluder, mislocalised or phantom detection)",
            )
        results.append(res)
    return results


def _blob_touching(colour_mask, region):
    """Union of connected colour components that overlap `region`."""
    n, labels = cv2.connectedComponents(colour_mask.astype(np.uint8), connectivity=8)
    ids = np.unique(labels[region & colour_mask])
    ids = ids[ids > 0]
    if not len(ids):
        return None
    return np.isin(labels, ids)


def summarise_picks(results):
    counts = {
        s: sum(r["status"] == s for r in results)
        for s in ("confirmed", "conflict", "occluded", "not_seen", "out_of_view")
    }
    offsets = [
        abs(r["side_lateral_offset_mm"])
        for r in results
        if "side_lateral_offset_mm" in r
    ]
    return {
        "planned_cubes": len(results),
        **counts,
        "cross_view_confirmed_rate": round(counts["confirmed"] / max(len(results), 1), 3),
        "max_side_lateral_offset_mm": round(max(offsets), 2) if offsets else None,
        "safe_to_execute": counts["conflict"] == 0,
        "all_confirmed": counts["confirmed"] == len(results) and len(results) > 0,
    }


# ---------------------------------------------------------------- stack audit
def audit_stack(img, cam, place_xy, expected_colours, place_yaw_deg=0.0):
    """Read the colour of each tower layer from a side-camera image."""
    masks = colour_masks(img)
    layers = []
    for level, expected in enumerate(expected_colours):
        corners = cube_corners(place_xy, place_yaw_deg, cam.table_z + level * C.CUBE_EDGE)
        sil = projected_mask(cam, corners, img.shape)
        if sil is None or not sil.any():
            layers.append({"level": level, "expected": expected, "status": "out_of_view"})
            continue
        core = shrink_mask(sil, LAYER_CORE)
        n = max(int(core.sum()), 1)
        fr = {c: float((masks[c] & core).sum()) / n for c in COLOURS}
        observed = max(fr, key=fr.get)
        if fr[observed] < 0.4:
            observed = "empty_or_unknown"
        layers.append(
            {
                "level": level,
                "expected": expected,
                "observed": observed,
                "colour_fractions": {c: round(v, 3) for c, v in fr.items()},
                "match": observed == expected,
            }
        )
    # A 7th position above the tower should be empty (no extra cube).
    above = projected_mask(
        cam,
        cube_corners(place_xy, place_yaw_deg, cam.table_z + len(expected_colours) * C.CUBE_EDGE),
        img.shape,
    )
    extra = None
    if above is not None and above.any():
        core = shrink_mask(above, LAYER_CORE)
        n = max(int(core.sum()), 1)
        extra = max(float((masks[c] & core).sum()) / n for c in COLOURS) >= 0.4
    return {
        "place_xy_m": [float(v) for v in place_xy],
        "layers": layers,
        "unexpected_cube_above_top": extra,
        "layers_matched": sum(l.get("match", False) for l in layers),
        "stack_verified_by_vision": all(l.get("match", False) for l in layers)
        and not extra,
    }


# ---------------------------------------------------------------- drawing
STATUS_BGR = {
    "confirmed": (0, 190, 0),
    "conflict": (0, 0, 255),
    "occluded": (0, 165, 255),
    "not_seen": (200, 0, 200),
    "out_of_view": (128, 128, 128),
}


def draw_picks(img, cam, picks, results, scale_up=True):
    out = img.copy()
    for pick, r in zip(picks, results):
        xy, yaw = cube_pose(pick)
        sil = projected_mask(cam, cube_corners(xy, yaw, cam.table_z), img.shape)
        if sil is None:
            continue
        colour = STATUS_BGR[r["status"]]
        cnts, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, colour, 2)
        ys, xs = np.nonzero(sil)
        if len(xs):
            label = f'{r["expected_colour"]}:{r["status"]}'
            cv2.putText(out, label, (int(xs.min()), int(ys.min()) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return out


def draw_stack(img, cam, audit, place_yaw_deg=0.0):
    out = img.copy()
    xy = audit["place_xy_m"]
    for layer in audit["layers"]:
        sil = projected_mask(
            cam, cube_corners(xy, place_yaw_deg, cam.table_z + layer["level"] * C.CUBE_EDGE), img.shape
        )
        if sil is None:
            continue
        colour = (0, 190, 0) if layer.get("match") else (0, 0, 255)
        cnts, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, colour, 1)
        ys, xs = np.nonzero(sil)
        cv2.putText(out, f'L{layer["level"]} {layer.get("observed", "?")}',
                    (int(xs.max()) + 6, int(ys.mean()) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- CLI
def _last_frame(folder):
    frames = sorted(Path(folder).glob("f*.png"))
    if not frames:
        raise SystemExit(f"No side frames found in {folder}")
    return frames[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("picks", help="cross-check sort_plan.json with the side camera")
    p.add_argument("--run", required=True, help="run folder containing sort_plan.json")
    p.add_argument("--side", required=True, help="folder (or camera.json) from capture_scene.py --camera side")
    p.add_argument("--image", help="side image (default: <side>/world_camera.png)")
    p.add_argument("--enforce", action="store_true", help="exit 2 if any planned cube is contradicted")
    p.add_argument("--strict", action="store_true",
                   help="exit 2 unless EVERY planned cube is confirmed by the side camera "
                        "(occluded / not_seen / out_of_view also block)")

    s = sub.add_parser("stack", help="audit the finished tower from a side-camera image")
    s.add_argument("--run", required=True)
    s.add_argument("--side", required=True, help="folder (or camera.json) with the side camera model")
    s.add_argument("--image", help="side image after execution (default: last side_frames/f*.png)")

    a = ap.parse_args(argv)
    run = Path(a.run)
    plan = json.loads((run / "sort_plan.json").read_text())
    cam = load_camera(a.side)
    picks = [q for q in plan.get("picks", []) if q.get("category", "cube") == "cube"]

    if a.cmd == "picks":
        image_path = Path(a.image) if a.image else Path(a.side) / "world_camera.png"
        img = load_image(image_path, cam)
        results = verify_picks(img, cam, picks)
        summary = summarise_picks(results)
        report = {"image": str(image_path), "summary": summary, "cubes": results}
        (run / "view_verification.json").write_text(json.dumps(report, indent=2) + "\n")
        cv2.imwrite(str(run / "side_verification_overlay.png"), draw_picks(img, cam, picks, results))
        for r in results:
            extra = f' ({r["reason"]})' if "reason" in r else ""
            off = f'  dx={r["side_lateral_offset_mm"]:+.2f} mm' if "side_lateral_offset_mm" in r else ""
            print(f'[verify] cube {r["detection_id"]} {r["expected_colour"]:<6} -> {r["status"]}{off}{extra}')
        print(f"[verify] summary {json.dumps(summary)}")
        if a.strict and not summary["all_confirmed"]:
            print("[verify] BLOCKED (strict): not every cube is confirmed by both cameras; re-observe before moving the robot.")
            return 2
        if a.enforce and not summary["safe_to_execute"]:
            print("[verify] BLOCKED: second camera contradicts the plan; re-observe before moving the robot.")
            return 2
        return 0

    # stack
    image_path = Path(a.image) if a.image else _last_frame(run / "side_frames")
    img = load_image(image_path, cam)
    if not picks:
        raise SystemExit("sort_plan.json has no cube picks")
    place_xy = picks[0].get("place_xy", [0.28, 0.10])
    colours = plan.get("stack_colours_bottom_to_top") or [q["colour"] for q in sorted(picks, key=lambda q: q["stack_level"])]
    place_yaw = float(picks[0].get("place_yaw_deg", 0.0))
    audit = audit_stack(img, cam, place_xy, colours, place_yaw)
    audit["image"] = str(image_path)
    (run / "stack_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    cv2.imwrite(str(run / "stack_audit_overlay.png"), draw_stack(img, cam, audit, place_yaw))
    for l in audit["layers"]:
        print(f'[audit] level {l["level"]}: expected {l["expected"]:<6} observed {l.get("observed", l["status"])}')
    print(f'[audit] stack_verified_by_vision = {audit["stack_verified_by_vision"]}')
    return 0 if audit["stack_verified_by_vision"] else 3


if __name__ == "__main__":
    sys.exit(main())
