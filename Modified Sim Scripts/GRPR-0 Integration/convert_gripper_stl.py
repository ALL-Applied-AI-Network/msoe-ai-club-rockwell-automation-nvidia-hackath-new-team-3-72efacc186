#!/usr/bin/env python3
"""Inspect binary/ASCII STL meshes and convert them to the starter's JSON schema.

The converter intentionally has no third-party dependencies.  Coordinates are
scaled but otherwise preserved so that CAD-axis alignment remains an explicit,
reviewable integration step in the simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path


def _is_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    return 84 + triangle_count * 50 == len(data)


def _load_binary(data: bytes):
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    triangles = []
    normals = []
    offset = 84
    for _ in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        normals.append(values[0:3])
        triangles.append((values[3:6], values[6:9], values[9:12]))
        offset += 50
    return triangles, normals


def _load_ascii(data: bytes):
    triangles = []
    normals = []
    vertices = []
    normal = (0.0, 0.0, 0.0)
    for raw_line in data.decode("utf-8", errors="strict").splitlines():
        fields = raw_line.strip().split()
        if fields[:2] == ["facet", "normal"] and len(fields) >= 5:
            normal = tuple(float(value) for value in fields[2:5])
        elif fields[:1] == ["vertex"] and len(fields) >= 4:
            vertices.append(tuple(float(value) for value in fields[1:4]))
            if len(vertices) == 3:
                triangles.append(tuple(vertices))
                normals.append(normal)
                vertices = []
    if not triangles:
        raise ValueError("No triangles found in ASCII STL")
    return triangles, normals


def load_stl(path: Path):
    data = path.read_bytes()
    if _is_binary_stl(data):
        triangles, normals = _load_binary(data)
        encoding = "binary"
    else:
        triangles, normals = _load_ascii(data)
        encoding = "ascii"
    return triangles, normals, encoding


def indexed_mesh(triangles, scale: float):
    points = []
    point_index = {}
    indices = []
    for triangle in triangles:
        for vertex in triangle:
            point = tuple(round(float(value) * scale, 10) for value in vertex)
            index = point_index.get(point)
            if index is None:
                index = len(points)
                point_index[point] = index
                points.append(point)
            indices.append(index)
    return points, indices


def mesh_stats(points, indices):
    axes = list(zip(*points))
    bounds_min = [min(axis) for axis in axes]
    bounds_max = [max(axis) for axis in axes]
    size = [hi - lo for lo, hi in zip(bounds_min, bounds_max)]
    centroid = [sum(axis) / len(axis) for axis in axes]

    edges = Counter()
    adjacency = [set() for _ in points]
    for i in range(0, len(indices), 3):
        tri = indices[i : i + 3]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (a, b) if a < b else (b, a)
            edges[edge] += 1
            adjacency[a].add(b)
            adjacency[b].add(a)

    visited = set()
    components = 0
    for start in range(len(points)):
        if start in visited:
            continue
        components += 1
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

    degenerate = 0
    area = 0.0
    signed_volume = 0.0
    for i in range(0, len(indices), 3):
        a, b, c = (points[index] for index in indices[i : i + 3])
        ab = tuple(b[j] - a[j] for j in range(3))
        ac = tuple(c[j] - a[j] for j in range(3))
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        triangle_area = 0.5 * math.sqrt(sum(value * value for value in cross))
        area += triangle_area
        signed_volume += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) / 6.0
        if triangle_area <= 1e-16:
            degenerate += 1

    return {
        "vertices": len(points),
        "triangles": len(indices) // 3,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "size": size,
        "vertex_centroid": centroid,
        "connected_components": components,
        "boundary_edges": sum(count == 1 for count in edges.values()),
        "nonmanifold_edges": sum(count > 2 for count in edges.values()),
        "degenerate_triangles": degenerate,
        "surface_area": area,
        "signed_volume": signed_volume,
        "volume": abs(signed_volume),
    }


def grpr0_local_frame(points):
    """Map SolidWorks assembly axes to one jaw's local Isaac frame.

    Source assembly: spread +/-X, thickness Y, tool extension -Z.
    Isaac jaw local: tool +X, spread Y, thickness Z.  The source mesh begins
    8 mm below its assembly origin; normalizing mapped X to zero places the
    mesh's mounting face at the existing moving-jaw origin.
    """
    mapped = [(-z, -x, y) for x, y, z in points]
    x_origin = min(point[0] for point in mapped)
    spread_center = 0.5 * (
        min(point[1] for point in mapped) + max(point[1] for point in mapped)
    )
    local = [
        (
            round(x - x_origin, 10),
            round(y - spread_center, 10),
            round(z, 10),
        )
        for x, y, z in mapped
    ]
    return local, {
        "source_mount_offset_m": x_origin,
        "assembly_spread_center_m": spread_center,
        "axis_map": {"x": "-source_z", "y": "-source_x", "z": "source_y"},
    }


def mirrored_local_mesh(points, indices):
    """Mirror one jaw-local mesh across spread Y and preserve outward normals."""
    mirrored_points = [(x, round(-y, 10), z) for x, y, z in points]
    mirrored_indices = []
    for offset in range(0, len(indices), 3):
        a, b, c = indices[offset : offset + 3]
        mirrored_indices.extend((a, c, b))
    return mirrored_points, mirrored_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="STL units to metres (GRPR-0 is already metre-scale)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--names", nargs="*")
    parser.add_argument(
        "--grpr0-local-frame",
        action="store_true",
        help="map the GRPR-0 SolidWorks assembly axes into jaw-local Isaac axes",
    )
    parser.add_argument(
        "--mirror-pair",
        action="store_true",
        help="treat one positive-X/right input as finger_r and mirror finger_l",
    )
    args = parser.parse_args()

    if args.mirror_pair:
        if len(args.inputs) != 1:
            parser.error("--mirror-pair requires exactly one input STL")
        if not args.grpr0_local_frame:
            parser.error("--mirror-pair requires --grpr0-local-frame")
        if args.names and len(args.names) != 2:
            parser.error("--mirror-pair --names must contain left and right names")
    elif args.names and len(args.names) != len(args.inputs):
        parser.error("--names must contain one value per input STL")

    parts = []
    reports = []
    for number, path in enumerate(args.inputs):
        triangles, _normals, encoding = load_stl(path)
        points, indices = indexed_mesh(triangles, args.scale)
        frame = None
        if args.grpr0_local_frame:
            points, frame = grpr0_local_frame(points)
        if args.mirror_pair:
            left_name, right_name = args.names or ("finger_l", "finger_r")
            left_points, left_indices = mirrored_local_mesh(points, indices)
            left_frame = dict(frame)
            left_frame["assembly_spread_center_m"] = -frame[
                "assembly_spread_center_m"
            ]
            left_frame["mirrored_from"] = str(path)
            for name, part_points, part_indices, part_frame in (
                (left_name, left_points, left_indices, left_frame),
                (right_name, points, indices, frame),
            ):
                stats = mesh_stats(part_points, part_indices)
                reports.append(
                    {"file": str(path), "encoding": encoding, "name": name, **stats}
                )
                parts.append(
                    {
                        "name": name,
                        "points": part_points,
                        "indices": part_indices,
                        "frame": part_frame,
                    }
                )
            continue

        name = args.names[number] if args.names else path.stem
        stats = mesh_stats(points, indices)
        reports.append({"file": str(path), "encoding": encoding, "name": name, **stats})
        part = {"name": name, "points": points, "indices": indices}
        if frame is not None:
            part["frame"] = frame
        parts.append(part)

    print(json.dumps(reports, indent=2))
    if args.output:
        payload = {
            "format": "fanuc_gripper_triangle_mesh_v1",
            "coordinate_units": "metres",
            "source_scale": args.scale,
            "source_files": [str(path) for path in args.inputs],
            "grpr0_local_frame": args.grpr0_local_frame,
            "parts": parts,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
