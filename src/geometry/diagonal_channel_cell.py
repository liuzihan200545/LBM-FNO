"""Generate a simple periodic 3-D pore cell with a large K_xy response.

Two inclined cylinder pieces join across periodic x/y faces to form a channel
along (1, 1, 0). A vertical cylinder intersects it and opens the z faces.
COMSOL receives exact Cylinder/Block/Union/Intersection geometry primitives.
"""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "diagonal_channel_cell"


def estimated_fluid_fraction(diagonal_radius, vertical_radius, samples=96):
    """Voxel sampling is used only for metadata, never for the COMSOL geometry."""
    xy = (np.arange(samples) + 0.5) / samples
    x, y = np.meshgrid(xy, xy, indexing="ij")
    z = xy
    diagonal_distance_sq = np.minimum(
        (x - y - 0.5) ** 2, (x - y + 0.5) ** 2
    ) / 2
    diagonal = diagonal_distance_sq[:, :, None] + (z[None, None, :] - 0.5) ** 2
    vertical = (x - 0.75) ** 2 + (y - 0.25) ** 2
    return float(np.mean((diagonal <= diagonal_radius**2) |
                         (vertical[:, :, None] <= vertical_radius**2)))


def generate_java(length, unit, diagonal_radius, vertical_radius):
    """Return a COMSOL Java model whose final domain is fluid, not solid."""
    return f"""// Periodic inclined pore channel: expect K_xy and K_yx to be appreciable.
// Two tilted cylinders are periodic images; a z cylinder adds vertical flow.
import com.comsol.model.*;
import com.comsol.model.util.*;

public class DiagonalChannelCell {{
  public static void main(String[] args) {{ run(); }}

  public static Model run() {{
    Model model = ModelUtil.create("Model");
    model.label("Periodic diagonal channel pore cell");
    model.param().set("L", "{length:.17g}[{unit}]");
    model.param().set("r_diag", "{diagonal_radius:.17g}*L");
    model.param().set("r_z", "{vertical_radius:.17g}*L");
    model.component().create("comp1", true);
    GeomSequence g = model.component("comp1").geom().create("geom1", 3);
    g.lengthUnit("{unit}");

    g.create("blk1", "Block");
    g.feature("blk1").set("size", new String[]{{"L", "L", "L"}});

    // The lines x-y=+0.5L and x-y=-0.5L are the two images that
    // intersect the unit cube. Their cut faces match periodically.
    g.create("diag1", "Cylinder");
    g.feature("diag1").set("pos", new String[]{{"-2*L", "-2.5*L", "0.5*L"}});
    g.feature("diag1").set("axistype", "cartesian");
    g.feature("diag1").set("ax3", new double[]{{1, 1, 0}});
    g.feature("diag1").set("r", "r_diag");
    g.feature("diag1").set("h", "8*L");

    g.create("diag2", "Cylinder");
    g.feature("diag2").set("pos", new String[]{{"-2*L", "-1.5*L", "0.5*L"}});
    g.feature("diag2").set("axistype", "cartesian");
    g.feature("diag2").set("ax3", new double[]{{1, 1, 0}});
    g.feature("diag2").set("r", "r_diag");
    g.feature("diag2").set("h", "8*L");

    g.create("vert1", "Cylinder");
    g.feature("vert1").set("pos", new String[]{{"0.75*L", "0.25*L", "-L"}});
    g.feature("vert1").set("r", "r_z");
    g.feature("vert1").set("h", "3*L");

    g.create("uni1", "Union");
    g.feature("uni1").selection("input").set(
        new String[]{{"diag1", "diag2", "vert1"}});
    g.feature("uni1").set("intbnd", "off");
    g.create("int1", "Intersection");
    g.feature("int1").selection("input").set(new String[]{{"blk1", "uni1"}});
    g.feature("int1").set("intbnd", "off");
    g.run();
    return model;
  }}
}}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagonal-radius", type=float, default=0.20,
                        help="inclined channel radius / cube edge")
    parser.add_argument("--vertical-radius", type=float, default=0.11,
                        help="z channel radius / cube edge")
    parser.add_argument("--length", type=float, default=256.0)
    parser.add_argument("--unit", choices=("um", "mm", "m"), default="um")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not np.isfinite(args.length) or args.length <= 0:
        parser.error("length must be finite and positive")
    if not 0 < args.diagonal_radius < 0.25:
        parser.error("diagonal radius must be between 0 and 0.25 of the edge")
    if not 0 < args.vertical_radius < 0.25:
        parser.error("vertical radius must be between 0 and 0.25 of the edge")

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "DiagonalChannelCell.java"
    target.write_text(generate_java(args.length, args.unit, args.diagonal_radius,
                                    args.vertical_radius), encoding="utf-8")
    metadata = {
        "length": args.length, "unit": args.unit, "periodic_axes": ["x", "y", "z"],
        "diagonal_radius_fraction": args.diagonal_radius,
        "vertical_radius_fraction": args.vertical_radius,
        "expected_large_components": ["Kxy", "Kyx"],
        "estimated_fluid_fraction": estimated_fluid_fraction(
            args.diagonal_radius, args.vertical_radius
        ),
        "geometry": "cube intersect (two x+y diagonal cylinders union z cylinder)",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"COMSOL Java model: {target.resolve()}")


if __name__ == "__main__":
    main()
