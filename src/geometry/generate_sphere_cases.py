"""Generate one 128³ periodic sphere-packing sample for each G1–G9 case.

Run from the repository root:
    uv run --locked python src/geometry/generate_sphere_cases.py

Existing complete cases with matching parameters are reused, never overwritten.
"""

import json
from pathlib import Path

import numpy as np

from disk_pack3d import generate


OUTPUT = Path(__file__).resolve().parent / "output" / "sphere_cases"
RESOLUTION = 128
SEED = 42
CASES = (
    ("G1", 100, 0.16, 0.55),
    ("G2", 50, 0.16, 0.55),
    ("G3", 200, 0.16, 0.55),
    ("G4", 100, 0.04, 0.55),
    ("G5", 100, 0.36, 0.55),
    ("G6", 100, 0.16, 0.45),
    ("G7", 100, 0.16, 0.65),
    ("G8", 50, 0.36, 0.65),
    ("G9", 200, 0.04, 0.45),
)
EXPECTED_FILES = (
    "spheres.npz", "PeriodicSphereCell.java", "solid.npy", "solid.vti",
    "preview.png", "surface_mesh.npz", "surface_mesh.vtp",
    "surface_mesh.stl", "preview.html", "metadata.json",
)


def verify_case(folder, n, variance, porosity):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    expected = {"n": n, "variance": variance, "porosity": porosity,
                "seed": SEED, "resolution": RESOLUTION, "boundary": "periodic",
                "variance_mode": "radius"}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing case has different parameters: {folder}")
    missing = [name for name in EXPECTED_FILES if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete case {folder}; missing {missing}")
    solid = np.load(folder / "solid.npy", allow_pickle=False, mmap_mode="r")
    if solid.shape != (RESOLUTION,) * 3 or not np.isin(solid, (0, 1)).all():
        raise ValueError(f"Invalid solid.npy in {folder}")
    with np.load(folder / "spheres.npz", allow_pickle=False) as spheres:
        radii = spheres["radii"]
        centers = spheres["centers"]
    if radii.shape != (n,) or centers.shape != (n, 3):
        raise ValueError(f"Invalid spheres.npz in {folder}")
    return {
        "directory": str(folder.resolve()),
        "n": n, "target_porosity": porosity,
        "voxel_porosity": float(1 - solid.mean()),
        "target_relative_radius_std": float(np.sqrt(variance)),
        "actual_relative_radius_std": float(radii.std() / radii.mean()),
        "mean_radius_voxels": float(radii.mean() * RESOLUTION),
        "minimum_radius_voxels": float(radii.min() * RESOLUTION),
        "minimum_sphere_gap_voxels": float(metadata["minimum_gap"] * RESOLUTION),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = {}
    for label, n, variance, porosity in CASES:
        folder = OUTPUT / f"{label}_seed{SEED}"
        if folder.exists():
            print(f"Reusing {folder}", flush=True)
        else:
            print(f"Generating {label}: n={n}, variance={variance}, "
                  f"porosity={porosity}", flush=True)
            generate(n=n, variance=variance, porosity=porosity, seed=SEED,
                     resolution=RESOLUTION, boundary="periodic", output=folder)
        records[label] = verify_case(folder, n, variance, porosity)
        (OUTPUT / "cases.json").write_text(
            json.dumps({"resolution": RESOLUTION, "seed": SEED,
                        "boundary": "periodic", "cases": records}, indent=2),
            encoding="utf-8",
        )
    print(f"Validated {len(records)} cases; summary: {OUTPUT / 'cases.json'}", flush=True)


if __name__ == "__main__":
    main()
