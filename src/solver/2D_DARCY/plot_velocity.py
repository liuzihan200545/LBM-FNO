"""Plot the two saved LBM velocity fields with a shared lattice-speed scale."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output",
                        help="directory containing flow_x.npz and flow_y.npz")
    args = parser.parse_args()

    fields = {}
    for direction in ("x", "y"):
        with np.load(args.output / f"flow_{direction}.npz") as data:
            fields[direction] = {name: data[name] for name in ("solid", "ux", "uy")}
    vmax = max(float(np.hypot(field["ux"], field["uy"]).max())
               for field in fields.values())
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for ax, direction in zip(axes, ("x", "y")):
        field = fields[direction]
        solid, ux, uy = field["solid"].astype(bool), field["ux"], field["uy"]
        speed = np.ma.array(np.hypot(ux, uy), mask=solid)
        image = ax.imshow(speed, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                          interpolation="nearest")

        ax.set(title=f"{direction.upper()}-direction body force",
               xlabel="x (lattice cells)", ylabel="y (lattice cells)")
        ax.set_aspect("equal")

    fig.colorbar(image, ax=axes, label="Speed (lattice units / time step)", shrink=0.82)
    target = args.output / "velocity_fields.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target.resolve())


if __name__ == "__main__":
    main()
