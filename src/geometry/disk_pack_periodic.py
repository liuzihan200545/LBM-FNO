"""二维周期 RSA 圆盘堆积：越界圆盘从对侧回绕，输出实心体素和预览。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parent / "output" / "disks_periodic"


def generate_disks(nx=256, ny=256, n_disks=100, log_mean=2.0,
                   log_variance=0.35, r_min=2.0, r_max=18.0,
                   max_attempts_per_disk=5000, seed=42):
    """在二维环面上放置互不重叠的圆盘，坐标单位为像素。"""
    if nx < 2 or ny < 2 or n_disks < 1 or max_attempts_per_disk < 1:
        raise ValueError("nx, ny >= 2 and n_disks, max_attempts_per_disk >= 1")
    if not 0 < r_min <= r_max < min(nx, ny)/2:
        raise ValueError("Require 0 < r_min <= r_max < min(nx, ny)/2")
    if log_variance < 0 or seed < 0:
        raise ValueError("log_variance and seed must be nonnegative")

    rng = np.random.default_rng(seed)
    centers = np.empty((n_disks, 2), dtype=float)
    radii = np.empty(n_disks, dtype=float)
    count = 0
    for disk in range(n_disks):
        for _ in range(max_attempts_per_disk):
            radius = float(np.clip(rng.lognormal(log_mean, np.sqrt(log_variance)),
                                   r_min, r_max))
            center = rng.uniform((0, 0), (nx, ny))
            delta = np.abs(centers[:count]-center)
            delta = np.minimum(delta, (nx, ny)-delta)
            if np.all(np.sum(delta**2, axis=1) >= (radii[:count]+radius)**2):
                centers[count], radii[count] = center, radius
                count += 1
                break
        else:
            print(f"Stopped at {count}/{n_disks} disks: no nonoverlapping position found.")
            break
    return centers[:count], radii[:count]


def rasterize(centers, radii, nx, ny):
    """solid[x, y]；像素中心取样，圆盘跨边界时在相对边接续。"""
    solid = np.zeros((nx, ny), dtype=np.uint8)
    x = np.arange(nx, dtype=float)+0.5
    y = np.arange(ny, dtype=float)+0.5
    for (cx, cy), radius in zip(centers, radii):
        dx = np.abs(x-cx)
        dy = np.abs(y-cy)
        dx = np.minimum(dx, nx-dx)
        dy = np.minimum(dy, ny-dy)
        solid[(dx[:, None]**2+dy[None, :]**2) <= radius**2] = 1
    return solid


def save_preview(solid, path, tiled=False):
    image = np.tile(solid, (2, 2)) if tiled else solid
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image.T, origin="lower", cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title("2 × 2 periodic tiling" if tiled else "Periodic disk packing")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    if tiled:
        ax.axvline(solid.shape[0]-0.5, color="tab:red", linewidth=0.8)
        ax.axhline(solid.shape[1]-0.5, color="tab:red", linewidth=0.8)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--n-disks", type=int, default=100)
    parser.add_argument("--log-mean", type=float, default=2.0)
    parser.add_argument("--log-variance", type=float, default=0.35)
    parser.add_argument("--r-min", type=float, default=2.0)
    parser.add_argument("--r-max", type=float, default=18.0)
    parser.add_argument("--max-attempts-per-disk", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    parameters = vars(args).copy()
    output = parameters.pop("output")
    centers, radii = generate_disks(**parameters)
    solid = rasterize(centers, radii, args.nx, args.ny)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/"solid.npy", solid)
    np.save(output/"pore.npy", 1-solid)
    np.savez_compressed(output/"disks.npz", centers=centers, radii=radii)
    save_preview(solid, output/"preview.png")
    save_preview(solid, output/"tiled_preview.png", tiled=True)
    crosses = np.any((centers-radii[:, None] < 0) |
                     (centers+radii[:, None] > (args.nx, args.ny)), axis=1)
    metadata = dict(parameters, placed_disks=len(radii),
                    boundary="periodic", crossing_disks=int(crosses.sum()),
                    porosity=float(1-solid.mean()), array_order="solid[x, y]")
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
