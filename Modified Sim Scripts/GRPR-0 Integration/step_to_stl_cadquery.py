#!/usr/bin/env python3
"""Tessellate a STEP solid to a millimetre-scale binary STL with CadQuery/OCCT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cadquery as cq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--linear-tolerance", type=float, default=0.025)
    parser.add_argument("--angular-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output.resolve()
    if not source.is_file():
        parser.error(f"STEP file not found: {source}")

    # CadQuery/OCCT convention is millimetres for model and STL coordinates.
    imported = cq.importers.importStep(str(source), unit="MM")
    shapes = imported.vals()
    if len(shapes) != 1:
        raise RuntimeError(f"expected one imported shape, found {len(shapes)}")
    shape = shapes[0]
    solids = shape.Solids()
    if len(solids) != 1:
        raise RuntimeError(f"expected one closed solid, found {len(solids)}")

    bounds = shape.BoundingBox()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not shape.exportStl(
        str(destination),
        tolerance=args.linear_tolerance,
        angularTolerance=args.angular_tolerance,
        ascii=False,
        relative=False,
        parallel=True,
    ):
        raise RuntimeError("CadQuery reported that STL export failed")

    print(
        json.dumps(
            {
                "input": str(source),
                "output": str(destination),
                "solids": len(solids),
                "bounds_min_mm": [bounds.xmin, bounds.ymin, bounds.zmin],
                "bounds_max_mm": [bounds.xmax, bounds.ymax, bounds.zmax],
                "size_mm": [bounds.xlen, bounds.ylen, bounds.zlen],
                "volume_mm3": shape.Volume(),
                "linear_tolerance_mm": args.linear_tolerance,
                "angular_tolerance_rad": args.angular_tolerance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
