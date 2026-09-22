"""用 XLB/JAX 实现 my_lbm.py 的周期域圆柱绕流演示。

运行：uv run python src/solver/demo-xlb/my_lbm_xlb.py
运行参数在 main() 中设置；无窗口运行时将 plot 改为 False。
保留原始几何、周期边界和 tau；采用低马赫数平衡初始化及 FullwayBounceBackBC，
不再像原程序那样在反弹后对固体节点执行 BGK 碰撞。
这仍是初始动量驱动的瞬态流动，没有持续入口或外力。
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.boundary_condition import FullwayBounceBackBC
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.precision_policy import PrecisionPolicy
from xlb.velocity_set import D2Q9


class CylinderFlow2D:
    """D2Q9 + BGK；XLB 数组为 (q, Nx, Ny)，绘图转为 (Ny, Nx)。"""

    def __init__(self, seed=0):
        self.nx, self.ny = 400, 100
        self.tau = 0.53
        backend = ComputeBackend.JAX
        precision = PrecisionPolicy.FP32FP32
        self.velocity_set = D2Q9(precision_policy=precision, compute_backend=backend)
        xlb.init(
            velocity_set=self.velocity_set,
            default_backend=backend,
            default_precision_policy=precision,
        )
        self.grid = grid_factory((self.nx, self.ny), compute_backend=backend)
        x, y = np.indices((self.nx, self.ny))
        self.solid = (x - self.nx // 4) ** 2 + (y - self.ny // 2) ** 2 < 13**2
        boundary = FullwayBounceBackBC(indices=np.array(np.where(self.solid)).tolist())
        # 不设置外边界 BC：JAX Stream 的 roll 操作保持上下、左右周期性。
        self.stepper = IncompressibleNavierStokesStepper(
            grid=self.grid, boundary_conditions=[boundary], collision_type="BGK"
        )
        self.f0, self.f1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields()

        # 使用平衡分布和低马赫数初速度，避免原始初始化的数值发散。
        rng = np.random.default_rng(seed)
        rho = jnp.ones((1, self.nx, self.ny), dtype=jnp.float32)
        u = np.zeros((2, self.nx, self.ny), dtype=np.float32)
        u[0] = 0.05
        u += rng.normal(0.0, 0.0001, u.shape).astype(np.float32)
        u[:, self.solid] = 0.0
        self.f0 = self.stepper.equilibrium(rho, jnp.asarray(u))
        self.f1 = self.f0.copy()

    def fields(self):
        rho, u = self.stepper.macroscopic(self.f0)
        rho, u = np.asarray(rho[0]), np.asarray(u)
        if not np.isfinite(rho).all() or not np.isfinite(u).all() or np.any(rho <= 0):
            raise FloatingPointError("流场出现非有限值或非正密度；请检查速度和松弛时间。")
        # 固体内的矩不代表流体速度，仅在输出时清零。
        u = np.where(self.solid[None, ...], 0.0, u)
        return rho.T, u[0].T, u[1].T

    def run(self, steps=3000, interval=100, plot=True):
        if steps < 1 or interval < 1:
            raise ValueError("steps 和 interval 必须为正整数")
        if plot:
            from matplotlib import pyplot as plt

            fig, ax = plt.subplots()
            _, ux, uy = self.fields()
            display = ax.imshow(np.hypot(ux, uy), origin="upper", vmin=0, vmax=0.2)
            fig.colorbar(display, ax=ax, label="Speed (lattice units)")
            ax.set(xlabel="x", ylabel="y (array index)")

        print(f"XLB/JAX devices: {jax.devices()}; tau={self.tau}; nu={(self.tau - 0.5) / 3:.4f}")
        for step in range(1, steps + 1):
            self.f0, self.f1 = self.stepper(
                self.f0, self.f1, self.bc_mask, self.missing_mask, 1.0 / self.tau, step - 1
            )
            # stepper 的第二个返回值是新的分布函数。
            self.f0, self.f1 = self.f1, self.f0
            if step % interval == 0 or step == steps:
                _, ux, uy = self.fields()
                speed = np.hypot(ux, uy)
                print(f"Step {step}/{steps}: max speed={speed.max():.6f}")
                if plot:
                    if not plt.fignum_exists(fig.number):
                        break
                    display.set_data(speed)
                    ax.set_title(f"Periodic cylinder flow — step {step}")
                    plt.pause(0.01)
        self.f0.block_until_ready()
        if plot:
            plt.show()


def main():
    steps = 30000
    interval = 100
    seed = 0
    plot = True

    simulation = CylinderFlow2D(seed=seed)
    simulation.run(steps=steps, interval=interval, plot=plot)


if __name__ == "__main__":
    main()
