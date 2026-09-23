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

默认生成 300 个球、目标孔隙率 0.50、128³ 体素，并采用周期边界：穿过立方体边界的球体会在对侧接续。可修改脚本顶部默认值或使用参数：

```bash
uv run --locked python src/geometry/disk_pack3d.py --n 300 --porosity 0.5 --resolution 128
uv run --locked python src/geometry/disk_pack3d.py --boundary wall --porosity 0.6 --output tmp/wall
```

结果默认保存在 `src/geometry/output/spheres_3d/`：

- `spheres.npz`：球心 `centers` 和半径 `radii`，单位为盒子边长。
- `solid.npy`：体素数组，顺序 `[z, y, x]`，固体为 1、孔隙为 0。
- `solid.vti`：与 `solid.npy` 相同的体素模型，坐标轴为 x/y/z，可在 ParaView 中直接打开。选择 `solid` 后显示 `Volume`，或使用 `Threshold` 过滤出值为 1 的固体。
- `preview.png`：三维预览。
- `preview.html`：离线交互式三维预览，用浏览器打开；拖动旋转、滚轮缩放、右键拖动平移，顶部按钮切换透明度。
- `surface_mesh.npz`：实际生成的实心球表面三角网格，含 `vertices` 和 `faces`。周期模式会把越界球体在对侧补齐，再裁切到单位立方体并封口。它是闭合网格，可用于体积计算；由于表面为有限三角形近似，体积与 `solid.npy` 的体素估计会有差异。
- `surface_mesh.vtp`：同一表面网格的 ParaView 格式，用 ParaView 打开后选择 `Surface` 即可查看球体。
- `metadata.json`：参数、最小间隙和体素孔隙率。

`--resolution 0` 跳过体素生成。默认 `periodic` 模式在对侧回绕越界球体；`--boundary clip` 仅截断盒外部分，不在对侧接续；`--boundary wall` 则让完整球体位于盒内。`clip` 模式的 `porosity` 参数仍按完整球体积设定，裁切后盒内实际孔隙率请看 `metadata.json` 中的 `voxel_porosity`。目标孔隙率低于本次密堆积能达到的值时会报错。

## GPU 验证范围

三维球堆积脚本使用 CPU，不需要 CUDA。当前环境已验证 JAX 可以在 RTX 4060 上执行计算；为避免默认预分配过多显存，运行 JAX 程序时可设置：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

现有 XLB/JAX 示例已在代码中设置此项。Warp 1.12.1 在当前驱动环境下仅检测到 CPU，Warp GPU 后端尚不可用；不影响上述球堆积脚本。未验证所有求解器的完整仿真。

文件	用途
preview.html	交互式查看：用浏览器打开，可旋转、缩放和切换透明度。
preview.png	固定视角的图片，适合快速查看。
solid.npy	实心体素模型，形状为 (128, 128, 128)，轴顺序是 [z, y, x]；1 表示球体，0 表示孔隙。用于栅格计算最方便。
surface_mesh.npz	裁切并封口后的闭合三角网格。vertices 是顶点坐标，faces 是三角面顶点索引；适合三维建模或网格分析。
spheres.npz	原始球的参数：centers 是球心，radii 是半径。边界球在这里仍以完整球表示；要使用裁切后的实体，请读 solid.npy 或 surface_mesh.npz。
metadata.json	生成参数和统计值，包括球数、随机种子、边界模式、最小间隙和实际体素孔隙率。


论文的核心思路是：**先得到岩石孔隙中的速度场，再对速度场求平均，利用达西定律计算渗透率。神经网络负责预测流场，不直接输出渗透率。**

以你后来发的 IFEDC 论文为例，流程如下。

**1. 对同一个岩石结构，分别施加两个方向的驱动力**

在孔隙中求解稳态流动，固体表面满足无滑移条件，计算区域外边界采用周期条件。

分别做两次计算：

- x 方向施加体力：\(\mathbf f^{(x)}=(f_0,0)\)；
- y 方向施加体力：\(\mathbf f^{(y)}=(0,f_0)\)。

每次都得到两个速度分量。即使沿 x 驱动，水流绕过颗粒时也会产生 y 方向速度。

**2. 计算整个岩石单元内的平均速度**

以 x 驱动为例：

\[
\left\langle u_x^{(x)}\right\rangle_\Omega
=\frac{1}{|\Omega|}
\int_{\Omega_f}u_x^{(x)}\,d\Omega
\]

这里积分只来自孔隙，但**分母是包含固体的整个计算区域面积**。这是达西速度对应的平均方式。

对均匀网格，可以将固体速度设为零后直接求平均：

```python
mean_ux = np.mean(ux * pore)
```

其中 `pore` 是**孔隙为 1、固体为 0**的掩膜。

如果使用 `ux[pore == 1].mean()`，得到的是孔隙内平均速度，还需要乘以孔隙率才能得到上述全域平均值。

**3. 用平均速度和驱动力计算渗透率**

论文第 4 页式（3）给出：

\[
\mathbf K=
\frac{\mu}{f_0}
\begin{pmatrix}
\left\langle u_x^{(x)}\right\rangle_\Omega &
\left\langle u_x^{(y)}\right\rangle_\Omega\\
\left\langle u_y^{(x)}\right\rangle_\Omega &
\left\langle u_y^{(y)}\right\rangle_\Omega
\end{pmatrix}
\]

其中：

- \(\mu\)：动力黏度；
- \(f_0\)：体力幅值，按论文方程具有压力梯度的量纲；
- 第一列来自 x 驱动，第二列来自 y 驱动；
- \(K_{xx},K_{yy}\) 表示主方向渗透能力；
- \(K_{xy},K_{yx}\) 表示两个方向之间的耦合响应。

对应代码为：

```python
K = mu / f0 * np.array([
    [(ux_x * pore).mean(), (ux_y * pore).mean()],
    [(uy_x * pore).mean(), (uy_y * pore).mean()],
])
```

这里 `ux_y` 表示“**y 驱动时的 x 速度**”。

**4. 神经网络替代耗时的流场求解**

训练阶段，论文用 LBM 生成参考速度、压力场，让网络学习：

```text
岩石几何特征 + 驱动方向 → 速度场、压力场
```

预测新岩石时，网络分别预测 x、y 驱动下的流场，再代入同一个平均公式计算 \(\mathbf K\)。压力用于场预测和训练，但上述渗透率公式直接使用的是速度。

旧论文重点计算**指定驱动方向的渗透率分量**；新论文扩展为**完整二维渗透率张量**，并检查 \(K_{xy}\approx K_{yx}\) 的物理一致性。

对于你现在的三维球模型，按同样思路需要做 **x、y、z 三次驱动**，得到 \(3\times3\) 渗透率张量。另外，若使用格子单位，结果不能直接标成 Darcy；必须根据网格的实际物理长度完成单位换算。
