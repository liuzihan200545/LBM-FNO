"""Create a simple periodic 3-D pore cell for COMSOL.

Eight cube corners each contain one eighth of the same periodic sphere.
A second, complete sphere is centered in the cube. The exported COMSOL
geometry is the cube minus these nine exact Sphere primitives.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .export_comsol_java import export_comsol_java
except ImportError:
    from export_comsol_java import export_comsol_java


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "simple_sphere_cell"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corner-radius", type=float, default=0.28,
                        help="corner-sphere radius divided by cube edge")
    parser.add_argument("--center-radius", type=float, default=0.30,
                        help="central-sphere radius divided by cube edge")
    parser.add_argument("--length", type=float, default=256.0)
    parser.add_argument("--unit", choices=("um", "mm", "m"), default="um")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    corner, center = args.corner_radius, args.center_radius
    if not 0 < corner < 0.5 or not 0 < center < 0.5:
        parser.error("Both radius fractions must be between 0 and 0.5")
    gap = np.sqrt(3) / 2 - corner - center
    if gap <= 0:
        parser.error("The corner and center spheres must not overlap")

    # The one corner sphere has eight periodic images, one at each cube vertex.
    # Its eight pieces inside the cube add up to one complete sphere.
    centers = np.array(((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)))
    radii = np.array((corner, center))
    args.output.mkdir(parents=True, exist_ok=True)
    java_file = args.output / "SimpleSphereCell.java"
    count = export_comsol_java(
        centers, radii, java_file, boundary="periodic",
        length=args.length, unit=args.unit, class_name="SimpleSphereCell",
    )
    np.savez_compressed(args.output / "spheres.npz", centers=centers, radii=radii)
    metadata = {
        "length": args.length, "unit": args.unit, "boundary": "periodic",
        "corner_radius_fraction": corner, "center_radius_fraction": center,
        "minimum_corner_center_gap_fraction": float(gap),
        "sphere_primitives": count,
        "porosity": float(1 - 4 * np.pi / 3 * (corner**3 + center**3)),
        "fluid_geometry": "cube minus eight corner sphere pieces and one center sphere",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"COMSOL Java model: {java_file.resolve()}")


if __name__ == "__main__":
    main()
