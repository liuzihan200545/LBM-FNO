"""Export 3-D LBM flow fields to ParaView VTI and orthogonal slice PNGs.

Example:
    uv run --locked python src/solver/3D_DARCY/visualize.py \
        --output src/solver/3D_DARCY/output/G1_seed42
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "G1_seed42"


def load_flow(path):
    with np.load(path, allow_pickle=False) as data:
        required = ("solid", "rho", "ux", "uy", "uz", "pressure")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        fields = {name: data[name] for name in required}
    shape = fields["solid"].shape
    if len(shape) != 3 or any(value.shape != shape for value in fields.values()):
        raise ValueError(f"{path}: all arrays must have matching (z, y, x) shape")
    if not np.isin(fields["solid"], (0, 1)).all():
        raise ValueError(f"{path}: solid must contain only 0 and 1")
    return fields


def to_vti(fields, path, spacing):
    nz, ny, nx = fields["solid"].shape
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1),
                        spacing=(spacing,) * 3, origin=(0, 0, 0))

    # VTK cell order is x-fastest. Input arrays have shape (z, y, x).
    def cell_values(array, dtype):
        return np.asarray(array, dtype=dtype).transpose(2, 1, 0).ravel(order="F")

    grid.cell_data["solid"] = cell_values(fields["solid"], np.uint8)
    grid.cell_data["velocity"] = np.column_stack([
        cell_values(fields[name], np.float32) for name in ("ux", "uy", "uz")
    ])
    speed = np.sqrt(fields["ux"]**2 + fields["uy"]**2 + fields["uz"]**2)
    grid.cell_data["speed"] = cell_values(speed, np.float32)
    grid.cell_data["pressure"] = cell_values(fields["pressure"], np.float32)
    grid.cell_data["rho"] = cell_values(fields["rho"], np.float32)
    grid.save(path)


def plot_slices(fields, path, spacing, unit, direction, converged):
    solid = fields["solid"].astype(bool)
    speed = np.sqrt(fields["ux"]**2 + fields["uy"]**2 + fields["uz"]**2)
    nz, ny, nx = solid.shape
    iz, iy, ix = nz // 2, ny // 2, nx // 2
    extent_xy = (0, nx * spacing, 0, ny * spacing)
    extent_xz = (0, nx * spacing, 0, nz * spacing)
    extent_yz = (0, ny * spacing, 0, nz * spacing)
    panels = (
        (speed[iz], solid[iz], extent_xy, "XY", "x", "y", iz),
        (speed[:, iy, :], solid[:, iy, :], extent_xz, "XZ", "x", "z", iy),
        (speed[:, :, ix], solid[:, :, ix], extent_yz, "YZ", "y", "z", ix),
    )
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")
    vmax = float(speed[~solid].max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, (values, mask, extent, plane, horizontal, vertical, index) in zip(axes, panels):
        image = ax.imshow(np.ma.array(values, mask=mask), origin="lower",
                          extent=extent, cmap=cmap, vmin=0, vmax=vmax,
                          interpolation="nearest")
        ax.set(title=f"{plane} at cell {index}",
               xlabel=f"{horizontal} ({unit})", ylabel=f"{vertical} ({unit})")
        ax.set_aspect("equal")
    state = "converged" if converged else "NOT CONVERGED"
    fig.suptitle(f"{direction.upper()}-drive speed | {state}")
    fig.colorbar(image, ax=axes, label="Speed (lattice units / time step)",
                 shrink=0.75)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directory containing summary.json and flow_*.npz")
    parser.add_argument("--direction", choices=("all", "x", "y", "z"),
                        default="all")
    parser.add_argument("--length-um", type=float,
                        help="override physical cube edge; otherwise use summary.json")
    parser.add_argument("--vti", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--png", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.vti and not args.png:
        parser.error("enable at least one of --vti and --png")
    summary_path = args.output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    length_um = args.length_um if args.length_um is not None else summary.get("length_um")
    if length_um is not None and (not np.isfinite(length_um) or length_um <= 0):
        parser.error("physical cube length must be finite and positive")
    directions = ("x", "y", "z") if args.direction == "all" else (args.direction,)
    for direction in directions:
        flow_path = args.output / f"flow_{direction}.npz"
        if not flow_path.is_file():
            if args.direction == "all":
                continue
            parser.error(f"missing {flow_path}")
        fields = load_flow(flow_path)
        nz, ny, nx = fields["solid"].shape
        if length_um is not None:
            if nx != ny or ny != nz:
                parser.error("--length-um requires a cubic grid")
            spacing, unit = length_um / nx, "um"
        else:
            spacing, unit = 1.0, "lattice cells"
        converged = bool(summary.get("runs", {}).get(direction, {}).get("converged"))
        if args.vti:
            target = args.output / f"flow_{direction}.vti"
            to_vti(fields, target, spacing)
            print(f"VTI: {target.resolve()} ({unit})", flush=True)
        if args.png:
            target = args.output / f"flow_{direction}_slices.png"
            plot_slices(fields, target, spacing, unit, direction, converged)
            print(f"Slices: {target.resolve()}", flush=True)


if __name__ == "__main__":
    main()
