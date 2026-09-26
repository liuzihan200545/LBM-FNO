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

当前脚本默认生成 100 个球，目标孔隙率为 0.40，体素分辨率为 `128³`。`periodic` 模式会把穿过单位立方体边界的球体接续到对侧，适合构建周期计算单元。可以用命令行参数覆盖默认值：

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

直接运行脚本或在 PyCharm 中运行时，可修改 `disk_pack3d.py` 顶部的 `SAVE_*` 开关，分别控制 `spheres.npz`、COMSOL Java、两种体素文件、两种预览和三种表面网格文件；`SAVE_METADATA_JSON` 控制参数摘要。将某项设为 `False` 会跳过该文件及不必要的体素化、表面网格计算，**不会删除目录中上一次运行留下的同名文件**。`metadata.json` 的 `saved_files` 记录本次实际生成的文件。

| 文件 | 内容和用途 |
| --- | --- |
| `solid.npy` | 实心体素数组，轴顺序为 `[z, y, x]`；`1` 为固体，`0` 为孔隙。 |
| `solid.vti` | 与 `solid.npy` 相同的体素数据，供 ParaView 使用；标量名为 `solid`。 |
| `surface_mesh.vtp` | 已在立方体边界截断并封口的球体表面，可在 ParaView 中直接显示。 |
| `surface_mesh.stl` | 同一球体表面的三角网格，供 COMSOL 三维几何导入；坐标范围为 0 到 1，文件不含物理单位。 |
| `PeriodicSphereCell.java` | 使用 COMSOL 原生球体和差集生成孔隙域的模型源文件；无需导入 STL。 |
| `surface_mesh.npz` | 同一闭合表面网格的 NumPy 文件：`vertices` 为坐标，`faces` 为三角面索引。 |
| `spheres.npz` | 原始球心 `centers` 和半径 `radii`，坐标按盒子边长归一化。边界球仍以完整球参数记录。 |
| `preview.html` | 离线交互式三维预览，可旋转、缩放并切换透明度。 |
| `preview.png` | 固定视角预览图。 |
| `metadata.json` | 参数、最小球间隙、本次生成的文件；生成体素时还包含体素孔隙率。 |

`--resolution 0` 可跳过体素生成。`clip` 模式的 `--porosity` 按完整球体积设定；裁切后的实际孔隙率应查看 `metadata.json` 中的 `voxel_porosity`。表面网格使用有限数量的三角形近似球面，因此由网格计算的体积与体素估计可能略有差异。

### 在 ParaView 中查看

- **直接看球体表面**：打开 `surface_mesh.vtp`，点击 **Apply**，显示方式选 **Surface**；若视图空白，点 **Reset Camera**。
- **查看实心体素**：打开 `solid.vti` 并点击 **Apply**。如果只看到白色外框，先在 Pipeline Browser 中选中该文件，然后添加 **Threshold**；选择 **Cell Data → solid**，范围设为 **1 到 1**，再点击 **Apply**。选中生成的过滤结果，将显示方式设为 **Surface**。白框是原始数据的 **Outline** 显示方式。

### 在 COMSOL 中导入三维球堆积

**推荐：原生球体几何。** 三维脚本会同时生成 `PeriodicSphereCell.java`，其中包含球心、半径及跨边界镜像球，COMSOL 会创建精确的球体、边长为 `L` 的立方体和差集。默认 `L=256[um]`，可在 COMSOL 的“全局定义 → 参数”中修改。生成的模型仅包含几何，流动物理场、周期条件和求解设置仍需在 COMSOL 中添加。

已有 `spheres.npz` 时，无需重新堆积球体，可直接重新生成 Java 文件并指定边长：

```bash
uv run --locked python src/geometry/export_comsol_java.py --length 256 --unit um
```

在 WSL 的项目根目录执行以下命令，脚本会将当前 Java 文件复制到 `E:\LBM-FNO-3D\COMSOL\resource`，使用 `D:\COMSOL63\Multiphysics\bin\win64\comsolcompile.exe` 编译，并检查新生成的 `.class`：

```bash
bash compile_comsol_java.sh
```

该脚本**只编译现有 Java 文件**。若更改了球堆积参数，请先重新运行 `disk_pack3d.py`；若只想更改物理边长，可先运行上面的 `export_comsol_java.py --length ... --unit ...`，然后再执行编译脚本。在 COMSOL Desktop 中通过 **文件 → 打开** 选择 `E:\LBM-FNO-3D\COMSOL\resource\PeriodicSphereCell.class`。打开时 COMSOL 会执行模型代码并构建差集；完成后另存为 `.mph`。该方式**不需要 CAD Import Module**。运行 300 个原生球体的差集仍可能耗时，但不会再处理 STL 的约 22 万个三角面。参考 [COMSOL Java 模型编译与打开说明](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_running.38.09.html)。

**备选：STL 表面网格。**

运行三维脚本后，使用 `src/geometry/output/spheres_3d/surface_mesh.stl`；在 Windows 版 COMSOL 中可将该文件复制到 `E:\LBM-FNO-3D\COMSOL\resource\surface_mesh.stl` 后导入。已有旧输出而缺少 STL 时，重新运行脚本即可生成。`solid.npy` 和 `solid.vti` 是体素数据；STL 是适合导入 COMSOL 几何序列的球体表面网格，不是已挖去球体的流体域。

1. 新建 **3D** 组件。在 **几何 1** 设置长度单位及物理边长。STL 的原始坐标位于 `[0, 1]^3` 且不含单位：例如要让计算单元边长为 `256 µm`，先将几何长度单位设为 `µm`，导入后对球体对象使用 **Scale（缩放）**、各方向比例因子为 `256`；随后创建尺寸为 `256 × 256 × 256 µm`、起点为 `(0, 0, 0)` 的 **Block（块）**。若选择其他物理边长，球体和块必须使用相同的比例。
2. 在 **几何 1 → Import（导入）** 中选择 **Mesh or 3D printing file (STL, 3MF, PLY)**，指定 `surface_mesh.stl`，并启用 **Form solids from surface objects（从表面对象形成实体）**。导入后应检查球体是实体对象，而不是只有外壳。
3. 添加 **Difference（差集）**：以块作为保留对象，减去导入并缩放的球体实体，得到孔隙流体域。构建几何并检查是否能对所得域生成自由四面体网格；若导入、差集或网格步骤失败，先检查 STL 修复信息、极窄孔喉和缩放是否一致。
4. 在得到的流体域上设置蠕动流动（Stokes）物理场：球体表面无滑移，立方体三组相对外表面分别设置周期流动条件，并沿 `x`、`y`、`z` 各施加一次体积力，分别求解以构造 `3 × 3` 渗透率张量。STL 只包含几何，不包含边界条件、材料或体积力。

STL 将球面离散为三角形；若需要精确球面，可使用 `spheres.npz` 中的球心和半径，在 COMSOL 中创建 Sphere 几何基元及相应周期镜像，再从块中做差集。不要把 `surface_mesh.vtp` 或 `solid.vti` 当成 COMSOL 可直接布尔运算的实体几何。

### 角球与中心椭球测试单元

`src/geometry/simple_ellipsoid_cell.py` 保留八个立方体角上的 $1/8$ 球，将中心球换成绕 $z$ 轴旋转的椭球。默认角球半径为 `0.28*L`，椭球三条半轴为 `(0.43, 0.18, 0.30)*L`，长轴在 $xy$ 平面内与 $x$ 轴成 $45^\circ$；默认 `L=256[um]`。原来的 `simple_sphere_cell.py` 保留，便于对照实验。

```bash
uv run --locked python src/geometry/simple_ellipsoid_cell.py
```

可用 `--center-axes A B C`、`--angle`、`--corner-radius`、`--length` 和 `--unit` 调整参数。脚本生成 `src/geometry/output/simple_ellipsoid_cell/SimpleEllipsoidCell.java` 和 `metadata.json`。COMSOL 6.3 编译并构建成功的几何模型位于 `E:\LBM-FNO-3D\COMSOL\resource\simple_ellipsoid_cell\SimpleEllipsoidCell_Model.mph`；它仅包含孔隙域几何，尚未包含物理场与求解结果。

中心椭球的 $45^\circ$ 倾斜使 $x$、$y$ 方向不再各自具有镜面对称性，预期张量具有 $K_{xx}\approx K_{yy}$、$K_{xy}\approx K_{yx}\ne0$，且 $K_{xz},K_{yz}$ 接近零。若把椭球改成与坐标轴对齐，非对角项理论上应接近零。单个椭球带来的非对角项未必像下文的斜向贯通孔道那样大，具体数值仍需使用相同边界条件、黏度和体积力计算。

### 非对角渗透率较大的简单三维单元

运行 `src/geometry/diagonal_channel_cell.py` 可生成一个用于验证渗透率张量的周期孔隙单元。它在立方体中保留沿 $(1,1,0)$ 方向贯通的斜圆柱孔道，并用一条竖直孔道连通 $z$ 面；其余部分视为固体。斜孔道穿过相对的 $x$、$y$ 面时轮廓对应，竖直孔道穿过相对的 $z$ 面时轮廓对应。这样沿 $x$ 或 $y$ 施力都会产生明显的横向流速，预期 $K_{xy}$、$K_{yx}$ 与 $K_{xx}$、$K_{yy}$ 同量级；实际数值仍以流动求解结果为准。

```bash
uv run --locked python src/geometry/diagonal_channel_cell.py
```

默认边长 `L=256[um]`，斜孔道半径 `0.20*L`，竖直孔道半径 `0.11*L`。可用 `--length`、`--unit`、`--diagonal-radius`、`--vertical-radius` 调整。输出位于 `src/geometry/output/diagonal_channel_cell/`，其中 `DiagonalChannelCell.java` 使用 COMSOL 原生 Block、Cylinder 和布尔运算生成**孔隙流体域**，`metadata.json` 记录参数和估算孔隙率。在 WSL 中复制并编译：

```bash
target=/mnt/e/LBM-FNO-3D/COMSOL/resource/diagonal_channel_cell
mkdir -p "$target"
cp src/geometry/output/diagonal_channel_cell/DiagonalChannelCell.java "$target/"
/mnt/d/COMSOL63/Multiphysics/bin/win64/comsolcompile.exe "$(wslpath -w "$target/DiagonalChannelCell.java")"
```

在 COMSOL 中通过 **文件 → 打开** 选择 `E:\LBM-FNO-3D\COMSOL\resource\diagonal_channel_cell\DiagonalChannelCell.class`；也可以直接打开已生成的 `E:\LBM-FNO-3D\COMSOL\resource\diagonal_channel_cell\DiagonalChannelCell_Model.mph`。此文件只有几何，还需给孔道内壁设置无滑移，给三组相对外表面分别设置周期流动条件，并按照下文的三维公式计算渗透率张量。

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

### 从圆盘参数生成 DXF

运行 `disk_pack_periodic.py` 时会**同时生成 DXF**，不需要对 `solid.npy` 做额外转换。导出使用 `disks.npz` 中同一组连续几何参数，即每个圆盘的圆心 `(x, y)` 和半径 `r`：

1. 用 `ezdxf` 将 `(0, 0)` 到 `(nx, ny)` 的闭合矩形写入 `unit_cell.dxf`。
2. 将每个原始圆盘写成 DXF 的 `CIRCLE` 实体。若 `x-r < 0`，再写一个圆心为 `(x+nx, y)` 的镜像；若 `x+r > nx`，再写 `(x-nx, y)`。上下边界同理，分别将圆心的 `y` 加上或减去 `ny`。
3. 对同时跨越两条边的圆盘组合 x、y 两个方向的平移，补齐角落镜像。把所有原始圆盘和镜像写入 `periodic_disks.dxf`。例如 `nx=256`、`x=2`、`r=5` 时，还会写入圆心横坐标为 `258` 的圆；它伸入单元右侧的部分与左侧圆盘相接。

DXF 保存的是**精确圆和矩形轮廓**，不是 `solid.npy` 中的像素方块。生成阶段不裁掉单元外的圆弧；后续在 COMSOL 中用 **Difference** 将几何限制在矩形内。DXF 本身不含周期流动边界条件。

生成并复制到 Windows 目录可连续执行：

```bash
uv run --locked python src/geometry/disk_pack_periodic.py
./copy_comsol_dxf.sh
```

复制目标是 `E:\LBM-FNO-3D\COMSOL\resource`。复制脚本会检查 E: 盘是否挂载、源 DXF 是否存在，并在复制后比对文件。

### 在 COMSOL 中构造孔隙域

1. 新建 **2D** 组件，在 **Geometry** 中用两个 **Import** 节点分别导入 `unit_cell.dxf` 和 `periodic_disks.dxf`，将导入对象转换为 **Solid**。
2. 添加 **Difference**：保留 `unit_cell.dxf` 的矩形，减去 `periodic_disks.dxf` 的全部圆盘，得到孔隙流体域。越界圆盘及其周期镜像会由这个运算截断在单元内。
3. **Build All** 后检查左右、上下边界；求解时还需在物理接口中分别设置两组周期流动边界，并把圆盘表面设为无滑移。DXF 文件只提供几何，不包含物理边界条件。

DXF 文件未指定物理单位，一个绘图单位对应一个像素长度；默认矩形大小为 `256 × 256`。在 COMSOL 中根据实际样品尺寸设置几何长度单位或缩放比例；例如若每个像素代表 `1 µm`，整个单元宽度便是 `256 µm`。连续圆弧几何与 `solid.npy` 的像素化边缘可能存在离散误差。

## 论文方法与三维扩展

仓库中的相关论文见 `docs/`。IFEDC 论文讨论的是**二维周期孔隙单元**：先用 LBM 计算流场，再对速度做全域平均，通过达西关系得到渗透率。网络预测的是速度场和压力场，渗透率由预测速度进一步计算。

### 为什么使用 EDT、MIS 和双向 ToF 特征

另一篇论文 *A Deep-Learning-Assisted Homogenization Framework for Permeability Upscaling in Porous Media* 将二值孔隙掩膜与 EDT、MIS、左→右 ToF、右→左 ToF 一起作为网络的五个输入通道。这些特征从不同角度描述孔隙几何及其对流场的影响：

| 特征 | 提供的信息 | 对流场预测的帮助 |
| --- | --- | --- |
| EDT（欧氏距离变换） | 每个孔隙点到最近固体壁面的距离 | 标出狭窄喉道和远离壁面的区域；近壁速度通常受黏性阻力影响更强。 |
| MIS（最大内接球） | 局部孔隙能容纳的最大球体尺度 | 描述孔体和喉道的大小，补充单点壁面距离难以表达的空间尺度。 |
| 左→右 ToF | 从左侧到各处的几何路径代价 | 提示左向右驱动时的连通通道、绕行路径和死端。 |
| 右→左 ToF | 从右侧到各处的几何路径代价 | 补充反向可达性，使方向性和几何不对称性更明显。 |

这里的 ToF 是描述几何可达性或传播路径的特征，不能直接当成流体实际通过时间。该论文未给出 ToF 的完整计算细节，也未报告逐项去除特征的消融实验，因此上表解释的是各特征可能提供的物理信息，并非论文已经证实的独立贡献。

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

二维 LBM 示例 `src/solver/2D_DARCY/main.py` 现在默认使用 FP64 计算。它每 500 步比较一次孔隙内完整速度场，相对变化低于 `1e-6` 才报告收敛；`--steps 30000` 是安全上限。可用 `--precision fp32` 切换回单精度，但在当前 `256×256` 周期孔隙样例中，FP32 的速度场残差未能降到该阈值。论文对应的格子力默认仍为 `4e-4`；在 `4e-5` 下虽然也能收敛，求得的渗透率与 `4e-4` 下相差约 5%，因此用于线性 Darcy 渗透率前还应做驱动力独立性检查。

其中 $\mu$ 为动力黏度，上标表示驱动方向，下标表示速度分量。压力场用于流场预测和训练，但此处的渗透率计算直接使用平均速度。

扩展到三维球体时，在同一周期立方体上分别施加 $\mathbf f^{(x)}=(f_0,0,0)$、$\mathbf f^{(y)}=(0,f_0,0)$、$\mathbf f^{(z)}=(0,0,f_0)$，每次求解得到速度分量 $(u_x,u_y,u_z)$。三维渗透率张量由三次响应按**列**组成：

$$
\mathbf K=\frac{\mu}{f_0}
\begin{pmatrix}
\langle u_x^{(x)}\rangle_\Omega & \langle u_x^{(y)}\rangle_\Omega & \langle u_x^{(z)}\rangle_\Omega \\
\langle u_y^{(x)}\rangle_\Omega & \langle u_y^{(y)}\rangle_\Omega & \langle u_y^{(z)}\rangle_\Omega \\
\langle u_z^{(x)}\rangle_\Omega & \langle u_z^{(y)}\rangle_\Omega & \langle u_z^{(z)}\rangle_\Omega
\end{pmatrix},
\qquad
\langle u_i^{(j)}\rangle_\Omega
=\frac{1}{|\Omega|}\int_{\Omega_f}u_i^{(j)}\,\mathrm dV.
$$

其中 $i$ 表示速度分量，$j$ 表示体积力方向，$\mu$ 是动力黏度，$f_0$ 是物理体积力密度。积分只覆盖孔隙流体域，但分母是**整个立方体体积**，因此在边长为 $L$ 的 COMSOL 模型中为 $L^3$。若 `intop1` 选择全部流体域，一次参数化扫描的三个结果表达式可写为 `mu_calc*intop1(u)/(L^3*F0)`、`mu_calc*intop1(v)/(L^3*F0)`、`mu_calc*intop1(w)/(L^3*F0)`；扫描结果的 x、y、z 三行分别对应 $\mathbf K$ 的第一、第二、第三列。

这是对二维论文方法的三维扩展；三维 LBM 格子、收敛设置和物理单位换算需要另行确定。格子单位下的数值不能未经换算就标成 Darcy。

## 运行环境说明

几何生成脚本在 CPU 上运行，不需要 CUDA。现有 XLB/JAX 示例使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 限制显存预分配；运行其他 JAX 程序时也可先设置：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```
仓库中的求解器示例尚未作为完整的三维论文复现流程验证。

## 2026-9-24 TODO:
- [x] 完成x方向体积力的速度场计算
- [x] 完成y方向体积力的速度场计算
- [x] 根据上面计算结果，计算出渗透率张量
- [x] 更换更大的体积力，判断之前结果都正确性
- [x] 更换更大的基质面积，判断渗透率与基质面积关系
- [x] 调研学习3d情况下如何使用comsol

## 2026-9-25 TODO:
- [x] 跑通COMSOL3D模拟
- [x] 计算三位渗透率张量
- [x] 完成2d情况下的xlb gpu并行
- [x] 改善2d情况下的不收敛问题，准确算出了稳态情况

## 2026-9-26 TODO:
- [x] 研究不规则情况下的渗透率张量
- [ ] 确定物理单位设计
- [ ] 研究二维情况下的个性化岩心生成
- [ ] 研究三维情况下的个性化岩心生成
- [ ] 研究如何可视化三维结果
