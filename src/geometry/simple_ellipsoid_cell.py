"""Generate a periodic COMSOL cell with corner spheres and a central ellipsoid.

The eight one-eighth corner spheres are unchanged from simple_sphere_cell.py.
The central solid is a 45-degree tilted ellipsoid by default, so the resulting
fluid domain has an xy coupling that an axis-aligned ellipsoid would not have.
"""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "simple_ellipsoid_cell"


def generate_java(length, unit, corner_radius, axes, angle):
    a, b, c = axes
    return f'''// Exact COMSOL geometry: cube minus eight corner spheres and one ellipsoid.
// The central ellipsoid is rotated about z; its center stays at (L/2,L/2,L/2).
import com.comsol.model.*;
import com.comsol.model.util.*;

public class SimpleEllipsoidCell {{
  public static void main(String[] args) {{ run(); }}

  public static Model run() {{
    Model model = ModelUtil.create("Model");
    model.label("Periodic corner spheres and central ellipsoid");
    model.param().set("L", "{length:.17g}[{unit}]");
    model.param().set("r_corner", "{corner_radius:.17g}*L");
    model.param().set("a_center", "{a:.17g}*L");
    model.param().set("b_center", "{b:.17g}*L");
    model.param().set("c_center", "{c:.17g}*L");
    model.component().create("comp1", true);
    GeomSequence g = model.component("comp1").geom().create("geom1", 3);
    g.lengthUnit("{unit}");
    g.create("blk1", "Block");
    g.feature("blk1").set("size", new String[]{{"L", "L", "L"}});

    String[] solids = new String[9];
    int k = 0;
    for (int ix = 0; ix <= 1; ix++) {{
      for (int iy = 0; iy <= 1; iy++) {{
        for (int iz = 0; iz <= 1; iz++) {{
          String tag = "corner" + k;
          solids[k++] = tag;
          g.create(tag, "Sphere");
          g.feature(tag).set("pos", new String[]{{
              Integer.toString(ix) + "*L",
              Integer.toString(iy) + "*L",
              Integer.toString(iz) + "*L"}});
          g.feature(tag).set("r", "r_corner");
        }}
      }}
    }}
    g.create("center", "Ellipsoid");
    g.feature("center").set("pos", new String[]{{"0.5*L", "0.5*L", "0.5*L"}});
    g.feature("center").set("semiaxes", new String[]{{
        "a_center", "b_center", "c_center"}});
    g.feature("center").set("rot", {angle:.17g});
    solids[8] = "center";

    g.create("dif1", "Difference");
    g.feature("dif1").selection("input").set(new String[]{{"blk1"}});
    g.feature("dif1").selection("input2").set(solids);
    g.run();
    return model;
  }}
}}
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corner-radius", type=float, default=0.28,
                        help="corner-sphere radius / cube edge")
    parser.add_argument("--center-axes", type=float, nargs=3, metavar=("A", "B", "C"),
                        default=(0.43, 0.18, 0.30),
                        help="ellipsoid semiaxes / cube edge")
    parser.add_argument("--angle", type=float, default=45.0,
                        help="rotation of ellipsoid about z, in degrees")
    parser.add_argument("--length", type=float, default=256.0)
    parser.add_argument("--unit", choices=("um", "mm", "m"), default="um")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    axes = np.asarray(args.center_axes, dtype=float)
    if not np.isfinite(args.length) or args.length <= 0:
        parser.error("length must be finite and positive")
    if not np.isfinite(args.corner_radius) or not 0 < args.corner_radius < 0.5:
        parser.error("corner radius must be between 0 and 0.5")
    if not np.isfinite(axes).all() or np.any(axes <= 0) or np.any(axes >= 0.5):
        parser.error("all ellipsoid semiaxes must be between 0 and 0.5")
    if not np.isfinite(args.angle):
        parser.error("angle must be finite")
    # This conservative bound ensures that the center ellipsoid never touches
    # any periodic corner sphere, regardless of its orientation.
    gap_bound = np.sqrt(3) / 2 - args.corner_radius - float(axes.max())
    if gap_bound <= 0:
        parser.error("corner spheres may overlap the center ellipsoid")

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "SimpleEllipsoidCell.java"
    target.write_text(generate_java(args.length, args.unit, args.corner_radius,
                                    axes, args.angle), encoding="utf-8")
    metadata = {
        "length": args.length,
        "unit": args.unit,
        "boundary": "periodic",
        "corner_radius_fraction": args.corner_radius,
        "center_semiaxes_fraction": axes.tolist(),
        "center_rotation_deg": args.angle,
        "minimum_corner_ellipsoid_gap_lower_bound_fraction": gap_bound,
        "porosity": float(1 - 4 * np.pi / 3 *
                          (args.corner_radius**3 + np.prod(axes))),
        "geometry": "cube minus eight corner sphere pieces and one center ellipsoid",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"COMSOL Java model: {target.resolve()}")


if __name__ == "__main__":
    main()
