# LBM-FNO

用于构建二维、三维多孔介质几何，并探索基于孔隙流场的渗透率计算。仓库已有几何生成脚本和部分 LBM/XLB 示例；论文中的完整三维流场数据集与 FNO 训练流程尚未实现。

## 环境与快速运行

建议在 WSL Ubuntu 中使用 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。进入项目目录后安装锁定的依赖：

```bash
cd /home/liuzihan/LBM-FNO
uv sync --locked
```

首次构建 RCPGenerator 需要 Git、C++ 编译器和 OpenMP；Ubuntu 可安装 `git` 和 `build-essential`。该依赖的源码提交已固定在 `pyproject.toml` 中，完整依赖版本见 `uv.lock`。

PyCharm 使用 WSL 解释器 `/home/liuzihan/LBM-FNO/.venv/bin/python`。直接运行脚本时，输出路径相对于脚本位置确定，不依赖当前工作目录。

| 脚本 | 用途 | 默认边界 |
| --- | --- | --- |
| `src/geometry/disk_pack3d.py` | 三维球堆积、体素和表面网格 | 周期边界 |
| `src/geometry/disk_pack_periodic.py` | 二维周期 RSA 圆盘堆积 | 周期边界 |
| `src/geometry/disk_pack.py` | 原有二维 RSA 圆盘堆积 | 圆盘必须完整位于盒内 |

## 三维周期球堆积

```bash
uv run --locked python src/geometry/disk_pack3d.py
```

默认生成 300 个球，目标孔隙率为 0.50，体素分辨率为 `128³`。`periodic` 模式会把穿过单位立方体边界的球体接续到对侧，适合构建周期计算单元。可以用命令行参数覆盖默认值：

```bash
uv run --locked python src/geometry/disk_pack3d.py --n 300 --porosity 0.5 --resolution 128 --boundary periodic
uv run --locked python src/geometry/disk_pack3d.py --boundary wall --porosity 0.6 --output tmp/wall
```

边界模式的区别：

| `--boundary` | 几何处理 |
| --- | --- |
| `periodic` | 越界球体在对侧接续；默认模式。 |
| `clip` | 仅保留立方体内的部分，不在对侧接续。 |
| `wall` | 每个球完整位于立方体内。 |

默认输出目录为 `src/geometry/output/spheres_3d/`：

| 文件 | 内容和用途 |
| --- | --- |
| `solid.npy` | 实心体素数组，轴顺序为 `[z, y, x]`；`1` 为固体，`0` 为孔隙。 |
| `solid.vti` | 与 `solid.npy` 相同的体素数据，供 ParaView 使用；标量名为 `solid`。 |
| `surface_mesh.vtp` | 已在立方体边界截断并封口的球体表面，可在 ParaView 中直接显示。 |
| `surface_mesh.npz` | 同一闭合表面网格的 NumPy 文件：`vertices` 为坐标，`faces` 为三角面索引。 |
| `spheres.npz` | 原始球心 `centers` 和半径 `radii`，坐标按盒子边长归一化。边界球仍以完整球参数记录。 |
| `preview.html` | 离线交互式三维预览，可旋转、缩放并切换透明度。 |
| `preview.png` | 固定视角预览图。 |
| `metadata.json` | 参数、最小球间隙和体素孔隙率。 |

`--resolution 0` 可跳过体素生成。`clip` 模式的 `--porosity` 按完整球体积设定；裁切后的实际孔隙率应查看 `metadata.json` 中的 `voxel_porosity`。表面网格使用有限数量的三角形近似球面，因此由网格计算的体积与体素估计可能略有差异。

### 在 ParaView 中查看

- **直接看球体表面**：打开 `surface_mesh.vtp`，点击 **Apply**，显示方式选 **Surface**；若视图空白，点 **Reset Camera**。
- **查看实心体素**：打开 `solid.vti` 并点击 **Apply**。如果只看到白色外框，先在 Pipeline Browser 中选中该文件，然后添加 **Threshold**；选择 **Cell Data → solid**，范围设为 **1 到 1**，再点击 **Apply**。选中生成的过滤结果，将显示方式设为 **Surface**。白框是原始数据的 **Outline** 显示方式。

## 二维周期圆盘堆积

```bash
uv run --locked python src/geometry/disk_pack_periodic.py
```

脚本使用周期最短距离判断圆盘是否重叠，并把越界部分回绕到对侧。默认在 `256 × 256` 网格中尝试放置 100 个圆盘。可用 `--nx`、`--ny`、`--n-disks`、`--seed`、`--r-min` 和 `--r-max` 等参数调整。

结果保存在 `src/geometry/output/disks_periodic/`：

| 文件 | 内容 |
| --- | --- |
| `solid.npy` / `pore.npy` | 二维掩膜，轴顺序为 `[x, y]`；分别以 `1` 表示固体和孔隙。 |
| `disks.npz` | 圆心 `centers` 与半径 `radii`，坐标单位为像素。 |
| `preview.png` | 单个周期单元。 |
| `tiled_preview.png` | `2 × 2` 平铺图；红线标记单元边界，便于检查跨边界接续。 |
| `metadata.json` | 生成参数、实际放置数、跨边界圆盘数和孔隙率。 |
| `unit_cell.dxf` | COMSOL 二维几何使用的矩形单元。 |
| `periodic_disks.dxf` | 原始圆盘及跨边界的周期镜像，保留精确圆弧。 |

圆盘密度过高时，RSA 可能无法放满指定数量并提前停止。实际放置数见 `placed_disks`。原有的 `disk_pack.py` 使用不同的边界规则：圆盘必须完整留在盒内，不会跨边界回绕。

### 导入 COMSOL

重新生成圆盘后，在 WSL 中执行 `./copy_comsol_dxf.sh`，可将输出目录下的全部 DXF 复制到 `E:\LBM-FNO-3D\COMSOL\resource`。脚本会检查 E: 盘是否挂载，并在复制后比对文件。

1. 新建 **2D** 组件，在 **Geometry** 中用两个 **Import** 节点分别导入 `unit_cell.dxf` 和 `periodic_disks.dxf`，将导入对象转换为 **Solid**。
2. 添加 **Difference**：保留 `unit_cell.dxf` 的矩形，减去 `periodic_disks.dxf` 的全部圆盘，得到孔隙流体域。越界圆盘及其周期镜像会由这个运算截断在单元内。
3. **Build All** 后检查左右、上下边界；求解时还需在物理接口中分别设置两组周期流动边界，并把圆盘表面设为无滑移。DXF 文件只提供几何，不包含物理边界条件。

DXF 坐标使用像素单位：默认矩形大小为 `256 × 256`。在 COMSOL 中根据实际样品尺寸设置几何长度单位或缩放比例；例如若每个像素代表 `1 µm`，整个单元宽度便是 `256 µm`。这组 DXF 是二维连续圆弧几何，与 `solid.npy` 的像素化边缘可能存在离散误差。

## 论文方法与三维扩展

仓库中的相关论文见 `docs/`。IFEDC 论文讨论的是**二维周期孔隙单元**：先用 LBM 计算流场，再对速度做全域平均，通过达西关系得到渗透率。网络预测的是速度场和压力场，渗透率由预测速度进一步计算。

对于同一个二维几何，分别沿 $x$、$y$ 方向施加大小为 $f_0$ 的均匀体积力。外边界为周期边界，固体表面为无滑移边界。每次求解得到两个速度分量。以 $x$ 方向驱动为例，全域平均速度为

$$
\langle u_x^{(x)}\rangle_\Omega
=\frac{1}{|\Omega|}\int_{\Omega_f}u_x^{(x)}\,\mathrm d\Omega.
$$

积分只在孔隙区域进行，但分母是**整个计算单元的面积**。在均匀网格中，若 `pore` 为孔隙取 `1`、固体取 `0` 的掩膜，可以直接计算 `np.mean(ux * pore)`。仅对孔隙速度取平均得到的是另一种平均量。

二维渗透率张量由两次方向驱动的响应组成：

$$
\mathbf K=\frac{\mu}{f_0}
\begin{pmatrix}
\langle u_x^{(x)}\rangle_\Omega & \langle u_x^{(y)}\rangle_\Omega \\
\langle u_y^{(x)}\rangle_\Omega & \langle u_y^{(y)}\rangle_\Omega
\end{pmatrix}.
$$

其中 $\mu$ 为动力黏度，上标表示驱动方向，下标表示速度分量。压力场用于流场预测和训练，但此处的渗透率计算直接使用平均速度。

扩展到三维球体时，应在相同几何上分别沿 $x$、$y$、$z$ 方向驱动，计算三个速度分量，并组成 $3 × 3$ 的渗透率张量。这是对二维论文方法的扩展；三维 LBM 格子、收敛设置和物理单位换算需要另行确定。格子单位下的数值不能未经换算就标成 Darcy。

## 在Comsol中进行二维情况的思路跑通



## 运行环境说明

几何生成脚本在 CPU 上运行，不需要 CUDA。现有 XLB/JAX 示例使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 限制显存预分配；运行其他 JAX 程序时也可先设置：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

仓库中的求解器示例尚未作为完整的三维论文复现流程验证。
