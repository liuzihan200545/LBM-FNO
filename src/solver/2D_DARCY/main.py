"""IFEDC202621768 §3.2: periodic 2-D pore-cell LBM reference cases.

solid.npy has shape (nx, ny), with 1 for solid and 0 for pore.
The paper uses 512x512, D2Q9/BGK, omega=1, halfway bounce-back,
periodic outer boundaries and x/y lattice forcing of 4e-4 each.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.boundary_condition import HalfwayBounceBackBC
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.precision_policy import PrecisionPolicy
from xlb.velocity_set import D2Q9


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOLID = ROOT / "geometry" / "output" / "disks_periodic" / "solid.npy"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"


def load_solid(path):
    solid = np.load(path, allow_pickle=False)
    if solid.ndim != 2 or min(solid.shape) < 3:
        raise ValueError("solid.npy must be 2-D with both dimensions >= 3")
    if not np.isin(solid, (0, 1)).all():
        raise ValueError("solid.npy must contain only 0 (pore) and 1 (solid)")
    solid = solid.astype(bool, copy=False)
    if not solid.any() or solid.all():
        raise ValueError("Geometry must contain both pore and solid pixels")
    return solid


class DarcyFlow2D:
    """One driving direction; XLB arrays have (q, x, y) order."""

    def __init__(self, solid, direction="x", omega=1.0, force=4.0e-4):
        if direction not in ("x", "y"):
            raise ValueError("direction must be x or y")
        if not 0.0 < omega < 2.0 or force <= 0.0:
            raise ValueError("Require 0 < omega < 2 and force > 0")
        self.solid = np.asarray(solid, dtype=bool)
        self.nx, self.ny = self.solid.shape
        self.direction, self.omega, self.force = direction, omega, force
        self.nu = (1.0 / omega - 0.5) / 3.0  # rho_ref=1; mu=nu in lattice units

        backend = ComputeBackend.JAX
        precision = PrecisionPolicy.FP32FP32
        self.velocity_set = D2Q9(precision_policy=precision, compute_backend=backend)
        xlb.init(velocity_set=self.velocity_set, default_backend=backend,
                 default_precision_policy=precision)
        grid = grid_factory((self.nx, self.ny), compute_backend=backend)

        # The XLB masker pads the exterior. We replace its masks below so that
        # solid-fluid links across periodic seams also receive bounce-back.
        surface = self.solid & np.logical_or.reduce([
            ~np.roll(self.solid, (cx, cy), axis=(0, 1))
            for cx in (-1, 0, 1) for cy in (-1, 0, 1)
            if cx or cy
        ])
        boundary = HalfwayBounceBackBC(indices=np.array(np.where(surface)).tolist())
        vector = np.array((force, 0.0) if direction == "x" else (0.0, force),
                          dtype=np.float32)
        self.stepper = IncompressibleNavierStokesStepper(
            grid=grid, boundary_conditions=[boundary], collision_type="BGK",
            forcing_scheme="exact_difference", force_vector=vector,
        )
        self.f0, self.f1, _, _ = self.stepper.prepare_fields()
        c = np.asarray(self.velocity_set.c)
        missing = np.stack([
            np.roll(self.solid, (int(cx), int(cy)), axis=(0, 1)) & ~self.solid
            for cx, cy in c.T
        ])
        bc_mask = np.where(np.any(missing, axis=0), boundary.id, 0).astype(np.uint8)
        self.bc_mask = jnp.asarray(bc_mask[None, ...])
        self.missing_mask = jnp.asarray(missing)

    def fields(self):
        """Return rho, ux, uy and fluid-mean-centered pressure, in (y,x) order."""
        rho, u = self.stepper.macroscopic(self.f0)
        rho, u = np.asarray(rho[0]), np.array(u)
        if not np.isfinite(rho).all() or not np.isfinite(u).all() or np.any(rho <= 0):
            raise FloatingPointError("Non-finite fields or non-positive density")
        u[:, self.solid] = 0.0
        pressure = rho / 3.0
        pressure -= pressure[~self.solid].mean()
        pressure[self.solid] = 0.0
        return rho.T, u[0].T, u[1].T, pressure.T

    def run(self, steps=30000, check_interval=500, min_steps=1000, rtol=1e-6):
        if steps < 1 or check_interval < 1 or min_steps < 0 or rtol <= 0:
            raise ValueError("Invalid step or convergence settings")
        previous = None
        relative_change = None
        converged = False
        print(f"{self.direction}-drive: grid={self.nx}x{self.ny}, "
              f"omega={self.omega:g}, nu={self.nu:g}, force={self.force:g}; "
              f"JAX devices={jax.devices()}", flush=True)
        for step in range(1, steps + 1):
            self.f0, self.f1 = self.stepper(
                self.f0, self.f1, self.bc_mask, self.missing_mask, self.omega, step - 1
            )
            self.f0, self.f1 = self.f1, self.f0
            if step % check_interval != 0 and step != steps:
                continue
            _, ux, uy, _ = self.fields()
            velocity = np.stack((ux.T, uy.T))[:, ~self.solid]
            if previous is not None:
                relative_change = float(
                    np.linalg.norm(velocity - previous) /
                    max(np.linalg.norm(velocity), np.finfo(float).eps)
                )
            previous = velocity
            max_speed = float(np.hypot(ux, uy).max())
            print(f"  step {step}/{steps}: max|u|={max_speed:.6g}, "
                  f"relative change={relative_change}", flush=True)
            if step >= min_steps and relative_change is not None and relative_change < rtol:
                converged = True
                break
        self.f0.block_until_ready()
        return {"steps": step, "converged": converged,
                "relative_change": relative_change, "max_speed": max_speed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solid", type=Path, default=DEFAULT_SOLID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--direction", choices=("both", "x", "y"), default="both")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--force", type=float, default=4.0e-4,
                        help="lattice-unit acceleration, not SI N/m^3")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--check-interval", type=int, default=500)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args(argv)

    solid = load_solid(args.solid)
    if solid.shape != (512, 512):
        print(f"Note: {solid.shape} differs from the paper's 512x512 lattice; "
              "this is a solver-format example, not its reported sample.", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    directions = ("x", "y") if args.direction == "both" else (args.direction,)
    summary = {
        "source_paper": "IFEDC202621768, section 3.2",
        "solid_file": str(args.solid.resolve()), "shape_xy": list(solid.shape),
        "porosity": float((~solid).mean()),
        "paper_resolution_match": solid.shape == (512, 512),
        "boundary": "periodic xy",
        "solid_bc": "halfway bounce-back", "model": "D2Q9 BGK",
        "omega": args.omega, "force_lattice": args.force,
        "convergence_rtol": args.rtol, "convergence_check_interval": args.check_interval,
        "runs": {},
    }
    means = {}
    for direction in directions:
        simulation = DarcyFlow2D(solid, direction, args.omega, args.force)
        info = simulation.run(args.steps, args.check_interval, args.min_steps, args.rtol)
        rho, ux, uy, pressure = simulation.fields()
        # Darcy velocity averages over the whole cell, with solids set to zero.
        mean_velocity = np.array((ux.mean(), uy.mean()), dtype=float)
        means[direction] = mean_velocity
        info["mean_velocity_xy"] = mean_velocity.tolist()
        summary["runs"][direction] = info
        np.savez_compressed(args.output / f"flow_{direction}.npz", solid=solid.T,
                            rho=rho, ux=ux, uy=uy, pressure=pressure)
        print(f"  <u>_cell={mean_velocity.tolist()} (lattice units)", flush=True)

    if len(means) == 2:
        mu = (1.0 / args.omega - 0.5) / 3.0
        tensor = mu / args.force * np.column_stack((means["x"], means["y"]))
        summary["permeability_lattice_squared"] = tensor.tolist()
        summary["symmetry_residual"] = float(
            abs(tensor[0, 1] - tensor[1, 0]) /
            max(np.linalg.norm(tensor), np.finfo(float).eps)
        )
        print("K (lattice length^2):\n", tensor, flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
