import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import time

@dataclass
class Circle:
    x: float
    y: float
    r: float

class RSADiskPackGenerator:
    """
    RSA (Random Sequential Addition) disk-pack porous media generator.

    Output:
        solid: 1 = solid, 0 = pore
        pore : 1 = pore, 0 = solid
    """

    def __init__(
        self,
        nx=512,
        ny=512,
        n_disks=200,
        log_mean=2.0,
        log_variance=0.35,
        r_min=2.0,
        r_max=40.0,
        max_attempts_per_disk=5000,
        seed=None
    ):
        self.nx = nx
        self.ny = ny
        self.n_disks = n_disks

        # numpy.lognormal(mean, sigma) 里的 sigma 是底层正态分布标准差
        self.log_mean = log_mean
        self.log_variance = log_variance
        self.log_sigma = np.sqrt(log_variance)

        self.r_min = r_min
        self.r_max = r_max
        self.max_attempts_per_disk = max_attempts_per_disk

        self.rng = np.random.default_rng(seed)

        self.circles = []

    def sample_radius(self):
        """Sample disk radius from a lognormal distribution and clip it."""
        r = self.rng.lognormal(mean=self.log_mean, sigma=self.log_sigma)
        r = np.clip(r, self.r_min, self.r_max)
        return float(r)

    def is_valid_circle(self, x, y, r):
        """
        Check if a circle:
        1) stays inside the domain
        2) does not overlap with existing circles
        """
        # boundary check
        if x - r < 0 or x + r >= self.nx:
            return False
        if y - r < 0 or y + r >= self.ny:
            return False

        # overlap check
        for c in self.circles:
            dx = x - c.x
            dy = y - c.y
            dist2 = dx * dx + dy * dy
            min_dist = r + c.r
            if dist2 < min_dist * min_dist:
                return False

        return True

    def generate(self, verbose=True):
        """
        Generate non-overlapping disks by RSA.
        """
        self.circles = []

        for i in range(self.n_disks):
            placed = False

            for _ in range(self.max_attempts_per_disk):
                r = self.sample_radius()

                x = self.rng.uniform(r, self.nx - r)
                y = self.rng.uniform(r, self.ny - r)

                if self.is_valid_circle(x, y, r):
                    self.circles.append(Circle(x, y, r))
                    placed = True
                    break

            if not placed:
                if verbose:
                    print(
                        f"[Warning] Disk {i+1}/{self.n_disks} could not be placed. "
                        f"Stopped early. Total placed = {len(self.circles)}"
                    )
                break

        if verbose:
            print(f"Requested disks : {self.n_disks}")
            print(f"Placed disks    : {len(self.circles)}")

        return self.circles

    def rasterize(self):
        """
        Convert circles to binary solid/pore images.
        solid: 1 = solid, 0 = pore
        pore : 1 = pore, 0 = solid
        """
        solid = np.zeros((self.nx, self.ny), dtype=np.uint8)

        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")

        for c in self.circles:
            mask = (X - c.x) ** 2 + (Y - c.y) ** 2 <= c.r ** 2
            solid[mask] = 1

        pore = 1 - solid
        return solid, pore

    def porosity(self, pore):
        return pore.mean()

    def plot(self, solid, figsize=(6, 6), title="RSA Disk-Pack Porous Media"):
        fig = plt.figure(figsize=figsize)
        plt.imshow(solid, cmap="gray", origin="lower")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)

    def plot_with_circles(self, figsize=(6, 6), title="Placed Circles (Vector View)"):
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlim(0, self.nx)
        ax.set_ylim(0, self.ny)
        ax.set_aspect("equal")
        ax.set_title(title)

        # white background = pore
        ax.set_facecolor("white")

        for c in self.circles:
            circ = plt.Circle((c.y, c.x), c.r, fill=True)
            ax.add_patch(circ)

        # 注意这里坐标和imshow不同，仅用于看圆盘摆放
        ax.set_xlim(0, self.ny)
        ax.set_ylim(0, self.nx)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show(block=True)
        plt.close(fig)

    def save(self, solid, pore, prefix="rock"):
        np.save(f"{prefix}_solid.npy", solid)
        np.save(f"{prefix}_pore.npy", pore)
        print(f"Saved: {prefix}_solid.npy and {prefix}_pore.npy")


# =========================
# 论文里的 6 类 rock type
# =========================
ROCK_TYPES = {
    1: {"n_disks": 100, "log_variance": 0.05},
    2: {"n_disks": 300, "log_variance": 0.45},
    3: {"n_disks": 100, "log_variance": 0.15},
    4: {"n_disks": 200, "log_variance": 0.05},
    5: {"n_disks": 100, "log_variance": 0.35},
    6: {"n_disks": 200, "log_variance": 0.35},
}


def generate_one_rock_type(
    rock_type=4,
    nx=512,
    ny=512,
    log_mean=2.0,
    r_min=2.0,
    r_max=25.0,
    max_attempts_per_disk=5000,
    seed=42,
    save=True
):
    params = ROCK_TYPES[rock_type]

    gen = RSADiskPackGenerator(
        nx=nx,
        ny=ny,
        n_disks=params["n_disks"],
        log_mean=log_mean,
        log_variance=params["log_variance"],
        r_min=r_min,
        r_max=r_max,
        max_attempts_per_disk=max_attempts_per_disk,
        seed=seed
    )

    gen.generate(verbose=True)
    solid, pore = gen.rasterize()
    phi = gen.porosity(pore)

    print(f"Rock type     : {rock_type}")
    print(f"Porosity      : {phi:.4f}")
    print(f"Image shape   : {solid.shape}")

    gen.plot(solid, title=f"Rock Type {rock_type} | Porosity={phi:.4f}")

    if save:
        gen.save(solid, pore, prefix=f"rock_type_{rock_type}")

    return gen, solid, pore


def generate_all_rock_types(
    nx=512,
    ny=512,
    log_mean=2.0,
    r_min=2.0,
    r_max=25.0,
    max_attempts_per_disk=5000,
    base_seed=100
):
    results = {}

    for rock_type in ROCK_TYPES.keys():
        print("=" * 60)
        print(f"Generating Rock Type {rock_type}")
        print("=" * 60)

        gen = RSADiskPackGenerator(
            nx=nx,
            ny=ny,
            n_disks=ROCK_TYPES[rock_type]["n_disks"],
            log_mean=log_mean,
            log_variance=ROCK_TYPES[rock_type]["log_variance"],
            r_min=r_min,
            r_max=r_max,
            max_attempts_per_disk=max_attempts_per_disk,
            seed=base_seed + rock_type
        )

        gen.generate(verbose=True)
        solid, pore = gen.rasterize()
        phi = gen.porosity(pore)

        print(f"Porosity: {phi:.4f}")

        results[rock_type] = {
            "generator": gen,
            "solid": solid,
            "pore": pore,
            "porosity": phi
        }

    return results


def plot_rock_overview(results, rock_types=(3, 4, 5, 6), i = 0, show=True):
    """把选定类别放到同一张图中，每行两张，并保存总览。"""
    if not rock_types:
        raise ValueError("请选择至少一种岩石类型")
    fig, axes = plt.subplots(
        (len(rock_types) + 1) // 2, 2,
        figsize=(10, 5 * ((len(rock_types) + 1) // 2)),
        squeeze=False, layout="constrained",
    )
    try:
        for ax, rock_type in zip(axes.flat, rock_types):
            data = results[rock_type]
            ax.imshow(data["solid"], cmap="gray_r", vmin=0, vmax=1,
                      origin="lower", interpolation="nearest")
            ax.set_title(f"Rock Type {rock_type} | Porosity={data['porosity']:.4f}")
            ax.set_axis_off()
        for ax in list(axes.flat)[len(rock_types):]:
            ax.set_axis_off()
        output = Path(__file__).resolve().parent / "output" / f"rsa_overview{i}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, facecolor="white")
        print(f"Saved overview: {output}")
        if show:
            plt.show(block=True)
    finally:
        plt.close(fig)
    return fig


def main():
    N_samples = 100  # 生成几组不同的岩石类型组合

    start = time.perf_counter()

    # 示例1：生成单个岩石类型
    # gen, solid, pore = generate_one_rock_type(
    #     rock_type=1,   # 改成 1~6
    #     nx=256,        # 先用 256 测试更快
    #     ny=256,
    #     log_mean=2.0,
    #     r_min=2.0,
    #     r_max=18.0,
    #     max_attempts_per_disk=5000,
    #     seed=42,
    #     save=True
    # )

    # 如果你想生成全部 6 类，把上面注释掉，改用这个：
    for i in range(N_samples):
        results = generate_all_rock_types(
                nx=256,
                ny=256,
                log_mean=2.0,
                r_min=2.0,
                r_max=18.0,
                max_attempts_per_disk=5000,
                base_seed=100
        )
        # 在这里指定要拼到一起的四类；也可改为 (1, 2, 3, 4, 5, 6)。
        plot_rock_overview(results, rock_types=(1, 2, 3, 4, 5, 6), i = i, show=False)

        end = time.perf_counter()

        print(f"程序总耗时: {end-start:.3f} s")
        print(f"平均每组耗时: {(end-start)/(N_samples):.3f} s")

if __name__ == "__main__":
    main()
