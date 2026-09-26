"""Export a periodic sphere packing as native COMSOL geometry (no STL import)."""

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np


def sphere_images(centers, radii, boundary):
    """Return original balls and only the periodic copies intersecting the cell."""
    for center, radius in zip(centers, radii):
        shifts = []
        for coordinate in center:
            choices = [0]
            if boundary == 'periodic':
                if coordinate - radius < 0:
                    choices.append(1)
                if coordinate + radius > 1:
                    choices.append(-1)
            shifts.append(choices)
        for shift in itertools.product(*shifts):
            yield center + shift, radius


def export_comsol_java(centers, radii, path, *, boundary='periodic',
                       length=256.0, unit='um', class_name='PeriodicSphereCell'):
    """Write a COMSOL model file for Java containing exact Sphere primitives."""
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or radii.shape != (len(centers),):
        raise ValueError('Expected centers (N, 3) and radii (N,)')
    if not np.isfinite(centers).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('Sphere data must be finite with positive radii')
    if boundary not in ('periodic', 'clip', 'wall'):
        raise ValueError('Invalid boundary type')
    if not np.isfinite(length) or length <= 0 or unit not in ('um', 'mm', 'm'):
        raise ValueError('Length must be positive and unit must be um, mm, or m')

    path = Path(path)
    if not re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', class_name):
        raise ValueError('Invalid Java class name')
    if path.stem != class_name or path.suffix != '.java':
        raise ValueError(f'The Java file must be named {class_name}.java')
    images = list(sphere_images(centers, radii, boundary))
    rows = []
    for center, radius in images:
        values = ', '.join(f'{float(value):.17g}' for value in (*center, radius))
        rows.append(f'      {{{values}}},')
    source = f'''// Generated from spheres.npz; all coordinates are fractions of the cell length.
// Original spheres: {len(centers)}; sphere objects including periodic images: {len(images)}.
// Compile with comsolcompile, then open the .class file in COMSOL Desktop.
import com.comsol.model.*;
import com.comsol.model.util.*;

public class {class_name} {{
  public static void main(String[] args) {{ run(); }}

  public static Model run() {{
    Model model = ModelUtil.create("Model");
    model.label("Periodic sphere pore cell");
    model.param().set("L", "{length:.17g}[{unit}]");
    // Keep spatial, material, geometry, and mesh frames explicitly defined.
    model.component().create("comp1", true);
    GeomSequence g = model.component("comp1").geom().create("geom1", 3);
    g.lengthUnit("{unit}");
    g.create("blk1", "Block");
    g.feature("blk1").set("size", new String[]{{"L", "L", "L"}});

    // Exact solid spheres; out-of-cell portions are removed by Difference.
    double[][] balls = new double[][]{{
{chr(10).join(rows)}
    }};
    String[] sphereTags = new String[balls.length];
    for (int i = 0; i < balls.length; i++) {{
      String tag = "s" + (i + 1);
      sphereTags[i] = tag;
      g.create(tag, "Sphere");
      g.feature(tag).set("pos", new String[]{{
        Double.toString(balls[i][0]) + "*L",
        Double.toString(balls[i][1]) + "*L",
        Double.toString(balls[i][2]) + "*L"
      }});
      g.feature(tag).set("r", Double.toString(balls[i][3]) + "*L");
    }}
    g.create("dif1", "Difference");
    g.feature("dif1").selection("input").init().set(new String[]{{"blk1"}});
    g.feature("dif1").selection("input2").init().set(sphereTags);
    g.run();
    return model;
  }}
}}
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
    return len(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent /
                        'output' / 'spheres_3d' / 'spheres.npz')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--length', type=float, default=256.0)
    parser.add_argument('--unit', choices=('um', 'mm', 'm'), default='um')
    parser.add_argument('--boundary', choices=('periodic', 'clip', 'wall'))
    args = parser.parse_args()
    with np.load(args.input) as data:
        centers, radii = data['centers'], data['radii']
    metadata = args.input.with_name('metadata.json')
    boundary = args.boundary or (json.loads(metadata.read_text(encoding='utf-8'))['boundary']
                                 if metadata.exists() else 'periodic')
    output = args.output or args.input.with_name('PeriodicSphereCell.java')
    count = export_comsol_java(centers, radii, output, boundary=boundary,
                               length=args.length, unit=args.unit)
    print(f'Wrote {output} with {count} native spheres ({len(centers)} originals).')


if __name__ == '__main__':
    main()
