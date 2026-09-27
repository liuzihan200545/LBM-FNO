"""Periodic 3-D pore-scale LBM and the full permeability tensor.

Input solid.npy has shape (nz, ny, nx): 1 = solid, 0 = pore. The outer
faces are periodic in x, y, z; solid surfaces use halfway bounce-back.
Three independent body-force runs supply the columns of the 3x3 tensor.
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
from xlb.velocity_set import D3Q19


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOLID = (ROOT / "geometry" / "output" / "sphere_cases" /
                 "G1_seed42" / "solid.npy")
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
DIRECTIONS = ("x", "y", "z")


def load_solid(path):
    solid = np.load(path, allow_pickle=False)
    if solid.ndim != 3 or min(solid.shape) < 3:
        raise ValueError("solid.npy must have shape (nz, ny, nx), each dimension >= 3")
    if not np.isin(solid, (0, 1)).all():
        raise ValueError("solid.npy must contain only 0 (pore) and 1 (solid)")
    solid = solid.astype(bool, copy=False)
    if not solid.any() or solid.all():
        raise ValueError("Geometry must contain both pore and solid voxels")
    return solid


class DarcyFlow3D:
    """One forcing direction. XLB fields use (q, x, y, z) axis order."""

    def __init__(self, solid_zyx, direction="x", omega=1.0, force=4e-6,
                 precision="fp64"):
        if direction not in DIRECTIONS:
            raise ValueError("direction must be x, y or z")
        if not 0 < omega < 2 or not np.isfinite(force) or force <= 0:
            raise ValueError("Require 0 < omega < 2 and finite force > 0")
        if precision not in ("fp32", "fp64"):
            raise ValueError("precision must be fp32 or fp64")

        # Geometry writes solid[z,y,x]; XLB grid and velocity vectors use x,y,z.
        self.solid = np.ascontiguousarray(solid_zyx.transpose(2, 1, 0))
        self.nx, self.ny, self.nz = self.solid.shape
        self.direction, self.omega, self.force = direction, omega, force
        self.precision = precision
        self.nu = (1 / omega - 0.5) / 3  # rho_ref=1, so mu=nu in lattice units

        if precision == "fp64":
            jax.config.update("jax_enable_x64", True)
        policy = (PrecisionPolicy.FP64FP64 if precision == "fp64"
                  else PrecisionPolicy.FP32FP32)
        backend = ComputeBackend.JAX
        self.velocity_set = D3Q19(precision_policy=policy, compute_backend=backend)
        xlb.init(velocity_set=self.velocity_set, default_backend=backend,
                 default_precision_policy=policy)
        grid = grid_factory((self.nx, self.ny, self.nz), compute_backend=backend)

        # np.roll makes both the solid mask and solid-fluid links periodic.
        # Replace XLB's default exterior mask, just as in the 2-D solver.
        c = np.asarray(self.velocity_set.c)
        shifts = [tuple(map(int, vector)) for vector in c.T]
        surface = self.solid & np.logical_or.reduce([
            ~np.roll(self.solid, shift, axis=(0, 1, 2))
            for shift in shifts if any(shift)
        ])
        boundary = HalfwayBounceBackBC(indices=np.array(np.where(surface)).tolist())
        force_vector = np.zeros(3, dtype=np.float64 if precision == "fp64"
                                else np.float32)
        force_vector[DIRECTIONS.index(direction)] = force
        self.stepper = IncompressibleNavierStokesStepper(
            grid=grid, boundary_conditions=[boundary], collision_type="BGK",
            forcing_scheme="exact_difference", force_vector=force_vector,
        )
        self.f0, self.f1, _, _ = self.stepper.prepare_fields()
        missing = np.stack([
            np.roll(self.solid, shift, axis=(0, 1, 2)) & ~self.solid
            for shift in shifts
        ])
        bc_mask = np.where(np.any(missing, axis=0), boundary.id, 0).astype(np.uint8)
        self.bc_mask = jnp.asarray(bc_mask[None, ...])
        self.missing_mask = jnp.asarray(missing)

    def fields(self):
        """Return rho, ux, uy, uz, pressure in (z, y, x) array order."""
        rho, velocity = self.stepper.macroscopic(self.f0)
        rho, velocity = np.asarray(rho[0]), np.array(velocity)
        if (not np.isfinite(rho).all() or not np.isfinite(velocity).all()
                or np.any(rho <= 0)):
            raise FloatingPointError("Non-finite fields or non-positive density")
        velocity[:, self.solid] = 0
        pressure = rho / 3
        pressure -= pressure[~self.solid].mean()
        pressure[self.solid] = 0
        return (rho.transpose(2, 1, 0),
                *(velocity[i].transpose(2, 1, 0) for i in range(3)),
                pressure.transpose(2, 1, 0))

    def run(self, steps=30000, check_interval=500, min_steps=1000, rtol=1e-6):
        if steps < 1 or check_interval < 1 or min_steps < 0 or rtol <= 0:
            raise ValueError("Invalid step or convergence settings")
        previous = None
        relative_change = None
        converged = False
        max_speed = None
        print(f"{self.direction}-drive: grid={self.nx}x{self.ny}x{self.nz}, "
              f"omega={self.omega:g}, nu={self.nu:g}, force={self.force:g}, "
              f"precision={self.precision}; JAX devices={jax.devices()}", flush=True)
        for step in range(1, steps + 1):
            self.f0, self.f1 = self.stepper(
                self.f0, self.f1, self.bc_mask, self.missing_mask, self.omega,
                step - 1,
            )
            self.f0, self.f1 = self.f1, self.f0
            if step % check_interval != 0 and step != steps:
                continue
            _, ux, uy, uz, _ = self.fields()
            velocity = np.stack((ux, uy, uz))[:, ~self.solid.transpose(2, 1, 0)]
            if previous is not None:
                relative_change = float(
                    np.linalg.norm(velocity - previous) /
                    max(np.linalg.norm(velocity), np.finfo(float).eps)
                )
            previous = velocity
            max_speed = float(np.sqrt(ux**2 + uy**2 + uz**2).max())
            print(f"  step {step}/{steps}: max|u|={max_speed:.6g}, "
                  f"relative change={relative_change}", flush=True)
            if (step >= min_steps and relative_change is not None
                    and relative_change < rtol):
                converged = True
                break
        self.f0.block_until_ready()
        return {"steps": step, "converged": converged,
                "relative_change": relative_change, "max_speed": max_speed,
                "max_mach": max_speed * np.sqrt(3)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solid", type=Path, default=DEFAULT_SOLID)
    parser.add_argument("--output", type=Path,
                        help="default: output/<name of solid.npy parent directory>")
    parser.add_argument("--direction", choices=("all", *DIRECTIONS), default="all")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--force", type=float, default=4e-6,
                        help="lattice force, not SI N/m^3; check force independence")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp64")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--check-interval", type=int, default=500)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--length-um", type=float,
                        help="physical cube edge in micrometers; enables K in um^2")
    parser.add_argument("--save-fields", action=argparse.BooleanOptionalAction,
                        default=True, help="save rho, velocity and pressure NPZ files")
    args = parser.parse_args(argv)

    if not np.isfinite(args.omega) or not 0 < args.omega < 2:
        parser.error("--omega must be finite and between 0 and 2")
    if not np.isfinite(args.force) or args.force <= 0:
        parser.error("--force must be finite and positive")
    if args.steps < 1 or args.check_interval < 1 or args.min_steps < 0:
        parser.error("step counts and check interval must be positive")
    if not np.isfinite(args.rtol) or args.rtol <= 0:
        parser.error("--rtol must be finite and positive")
    solid = load_solid(args.solid)
    if args.length_um is not None:
        if not np.isfinite(args.length_um) or args.length_um <= 0:
            parser.error("--length-um must be finite and positive")
        if len(set(solid.shape)) != 1:
            parser.error("--length-um requires a cubic grid")
    output = args.output or OUTPUT_ROOT / args.solid.parent.name
    output.mkdir(parents=True, exist_ok=True)
    directions = DIRECTIONS if args.direction == "all" else (args.direction,)
    summary = {
        "solid_file": str(args.solid.resolve()), "shape_zyx": list(solid.shape),
        "porosity": float((~solid).mean()), "boundary": "periodic xyz",
        "solid_bc": "halfway bounce-back", "model": "D3Q19 BGK",
        "omega": args.omega, "kinematic_viscosity_lattice": (1 / args.omega - 0.5) / 3,
        "force_lattice": args.force, "precision": args.precision,
        "length_um": args.length_um, "convergence_rtol": args.rtol,
        "convergence_check_interval": args.check_interval, "runs": {},
    }
    means = {}
    for direction in directions:
        simulation = DarcyFlow3D(solid, direction, args.omega, args.force,
                                 args.precision)
        info = simulation.run(args.steps, args.check_interval,
                              args.min_steps, args.rtol)
        rho, ux, uy, uz, pressure = simulation.fields()
        # Darcy (superficial) velocity averages over the full cube; solids are 0.
        mean_velocity = np.array((ux.mean(), uy.mean(), uz.mean()), dtype=float)
        means[direction] = mean_velocity
        info["mean_velocity_xyz"] = mean_velocity.tolist()
        summary["runs"][direction] = info
        if args.save_fields:
            np.savez_compressed(output / f"flow_{direction}.npz", solid=solid,
                                rho=rho, ux=ux, uy=uy, uz=uz,
                                pressure=pressure)
        print(f"  <u>_cell={mean_velocity.tolist()} (lattice units)", flush=True)
        summary["permeability_columns_lattice_squared"] = {
            key: (simulation.nu / args.force * vector).tolist()
            for key, vector in means.items()
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
        del simulation

    summary["all_directions_converged"] = all(
        summary["runs"][direction]["converged"] for direction in directions
    )
    if len(means) == 3:
        nu = (1 / args.omega - 0.5) / 3
        tensor = nu / args.force * np.column_stack(tuple(means[d] for d in DIRECTIONS))
        summary["permeability_lattice_squared"] = tensor.tolist()
        summary["symmetry_residual"] = float(
            np.linalg.norm(tensor - tensor.T) /
            max(np.linalg.norm(tensor), np.finfo(float).eps)
        )
        if args.length_um is not None:
            voxel_size_um = args.length_um / solid.shape[0]
            summary["voxel_size_um"] = voxel_size_um
            summary["permeability_um_squared"] = (tensor * voxel_size_um**2).tolist()
        print("K (lattice length^2):\n", tensor, flush=True)
        if args.length_um is not None:
            print("K (um^2):\n", tensor * voxel_size_um**2, flush=True)
    if not summary["all_directions_converged"]:
        print("WARNING: at least one run did not converge; K is provisional.",
              flush=True)
    if any(run["max_mach"] >= 0.1 for run in summary["runs"].values()):
        print("WARNING: max Mach >= 0.1; reduce --force and check K independence.",
              flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2),
                                           encoding="utf-8")
    print(f"Saved results to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
