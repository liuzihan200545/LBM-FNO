# LBM-FNO

## 环境（WSL Ubuntu）

项目使用 Python 3.11 和 uv，已有虚拟环境位于 `.venv`。在 WSL 终端执行：

```bash
cd /home/liuzihan/LBM-FNO
uv sync --locked
uv run --locked python src/geometry/disk_pack3d.py
```

首次从源码构建 RCPGenerator 需要 Git、C++ 编译器和 OpenMP（Ubuntu 可用 `sudo apt install git build-essential`）。该依赖固定到官方仓库的提交，地址见 `pyproject.toml`，完整依赖见 `uv.lock`。

PyCharm 请选择 **WSL / Ubuntu-24.04.3** 中的现有解释器：
`/home/liuzihan/LBM-FNO/.venv/bin/python`，然后运行 `src/geometry/disk_pack3d.py`。

## 三维球堆积

默认生成 300 个球、目标孔隙率 0.50、128³ 体素、周期边界。可修改脚本顶部默认值或使用参数：

```bash
uv run --locked python src/geometry/disk_pack3d.py --n 300 --porosity 0.5 --resolution 128
uv run --locked python src/geometry/disk_pack3d.py --boundary wall --porosity 0.6 --output tmp/wall
```

结果默认保存在 `src/geometry/output/spheres_3d/`：

- `spheres.npz`：球心 `centers` 和半径 `radii`，单位为盒子边长。
- `solid.npy`：体素数组，顺序 `[z, y, x]`，固体为 1、孔隙为 0。
- `preview.png`：三维预览。
- `metadata.json`：参数、最小间隙和体素孔隙率。

`--resolution 0` 跳过体素生成。连续几何孔隙率与离散体素孔隙率有离散误差；目标孔隙率低于本次密堆积能达到的值时会报错。

## GPU 验证范围

三维球堆积脚本使用 CPU，不需要 CUDA。当前环境已验证 JAX 可以在 RTX 4060 上执行计算；为避免默认预分配过多显存，运行 JAX 程序时可设置：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

现有 XLB/JAX 示例已在代码中设置此项。Warp 1.12.1 在当前驱动环境下仅检测到 CPU，Warp GPU 后端尚不可用；不影响上述球堆积脚本。未验证所有求解器的完整仿真。
