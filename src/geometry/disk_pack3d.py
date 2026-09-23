"""三维多粒径球堆积。使用项目 .venv 解释器运行，可直接修改下方默认参数。"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import rcpgenerator

# PyCharm 中直接运行时使用这些参数。
N_SPHERES = 300
RADIUS_VARIANCE = 0.35   # 归一化半径 E[R]=1 时的分布方差，非最终像素半径方差
POROSITY = 0.50         # 最终孔隙体积分数，越大越疏松
SEED = 42
RESOLUTION = 128        # 体素模型边长；设为 0 可跳过体素生成
BOUNDARY = 'periodic'   # periodic：周期边界；wall：球完整地位于盒子内部
OUTPUT = Path(__file__).resolve().parent/'output'/'spheres_3d'


def minimum_gap(centers, radii, boundary):
    """独立逐球检查，避免创建 N×N×3 大数组。长度单位为盒子边长。"""
    gap = float(np.min(1-2*radii)) if boundary == 'periodic' else float(
        np.min(np.minimum(centers, 1-centers)-radii[:, None]))
    for i in range(len(radii)-1):
        delta = centers[i+1:]-centers[i]
        if boundary == 'periodic':
            delta -= np.rint(delta)
        gap = min(gap, float(np.min(np.linalg.norm(delta, axis=1)-radii[i+1:]-radii[i])))
    return gap


def voxelize(centers, radii, size, boundary):
    """像素中心判定；数组顺序 solid[z,y,x]，固体=1。"""
    solid = np.zeros((size, size, size), dtype=np.uint8)
    for center, radius in zip(centers, radii):
        axes = []
        for coordinate in center:
            indices = np.arange(int(np.floor((coordinate-radius)*size)),
                                int(np.ceil((coordinate+radius)*size)))
            if boundary == 'wall':
                indices = indices[(indices >= 0) & (indices < size)]
            axes.append(indices)
        ix, iy, iz = axes
        x = (ix+.5)/size-center[0]
        y = (iy+.5)/size-center[1]
        z = (iz+.5)/size-center[2]
        inside = z[:, None, None]**2+y[None, :, None]**2+x[None, None, :]**2 <= radius**2
        zz, yy, xx = np.nonzero(inside)
        solid[iz[zz] % size, iy[yy] % size, ix[xx] % size] = 1
    return solid


def preview(centers, radii, path, porosity):
    # 原始球心显示一次，周期边缘球可伸出盒子；体素模型会正确跨边界延续。
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 17), np.linspace(0, np.pi, 13))
    sx, sy, sz = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    colors = plt.get_cmap('viridis')
    for (x, y, z), r in zip(centers, radii):
        color = colors((r-radii.min())/max(float(np.ptp(radii)), 1e-12))
        ax.plot_surface(x+r*sx, y+r*sy, z+r*sz, color=color, linewidth=0, shade=True)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel='x', ylabel='y', zlabel='z')
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f'{len(radii)} spheres | porosity={porosity:.3f}')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def generate(n=N_SPHERES, variance=RADIUS_VARIANCE, porosity=POROSITY,
             seed=SEED, resolution=RESOLUTION, boundary=BOUNDARY,
             output=OUTPUT, variance_mode='radius'):
    if not isinstance(n, int) or n < 2 or seed < 0:
        raise ValueError('Use integer n >= 2 and seed >= 0')
    if not np.isfinite(variance) or variance < 0 or not np.isfinite(porosity) or not 0 < porosity < 1:
        raise ValueError('Use finite variance >= 0 and 0 < porosity < 1')
    if boundary not in ('periodic', 'wall') or variance_mode not in ('radius', 'log'):
        raise ValueError('Invalid boundary or variance_mode')
    if resolution != 0 and resolution < 16:
        raise ValueError('resolution must be 0 or >= 16')
    sigma2 = np.log1p(variance) if variance_mode == 'radius' else variance
    weights = np.random.default_rng(seed).lognormal(-sigma2/2, np.sqrt(sigma2), n)
    rcpgenerator.set_num_threads(1)
    print(f'Generating {n} spheres with RCPGenerator...', flush=True)
    packing = rcpgenerator.Packing(phi=.08, N=n, Ndim=3, box=[1., 1., 1.],
        walls=[0 if boundary == 'periodic' else 1]*3,
        dist={'type': 'custom', 'custom': weights.tolist()}, seed=seed, neighbor_max=0)
    result = packing.pack()
    centers = np.asarray(result['positions'], dtype=float)
    raw_radii = np.asarray(result['diameters'], dtype=float)/2
    if centers.shape != (n, 3) or raw_radii.shape != (n,) or not np.isfinite(centers).all() or not np.isfinite(raw_radii).all() or np.any(raw_radii <= 0):
        raise RuntimeError('Invalid backend output')
    if boundary == 'periodic':
        centers %= 1.0
    packed_fraction = float(4*np.pi/3*np.sum(raw_radii**3))
    scale = ((1-porosity)/packed_fraction)**(1/3)
    if scale > 1+1e-8:
        raise ValueError(f'Requested porosity is too low; packing supports >= {1-packed_fraction:.6f}')
    radii = raw_radii*scale
    gap = minimum_gap(centers, radii, boundary)
    if gap < -1e-7:
        raise RuntimeError(f'Packing overlaps or crosses walls: minimum gap={gap}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'spheres.npz', centers=centers, radii=radii)
    metadata = dict(n=n, variance=variance, variance_mode=variance_mode,
                    porosity=porosity, seed=seed, resolution=resolution,
                    boundary=boundary, minimum_gap=gap,
                    rcpgenerator_version=importlib.metadata.version('rcpgenerator'))
    if resolution:
        solid = voxelize(centers, radii, resolution, boundary)
        np.save(output/'solid.npy', solid)
        metadata['voxel_porosity'] = float(1-solid.mean())
    preview(centers, radii, output/'preview.png', porosity)
    interactive_preview(centers, radii, output/'preview.html', porosity, boundary)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print(f'Saved to: {output.resolve()}')
    return centers, radii


def interactive_preview(centers, radii, path, porosity, boundary):
    """离线 HTML 三维视图：拖动旋转、滚轮缩放，按钮切换透明度。"""
    # 合并所有球的三角网格，避免为每个球创建独立绘图对象。
    segments, rings = 24, 16
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, segments, endpoint=False),
                             np.linspace(0, np.pi, rings))
    unit = np.column_stack((np.sin(phi).ravel()*np.cos(theta).ravel(),
                            np.sin(phi).ravel()*np.sin(theta).ravel(),
                            np.cos(phi).ravel()))
    faces = []
    for row in range(rings-1):
        for col in range(segments):
            a = row*segments+col
            b = row*segments+(col+1) % segments
            faces.extend(((a, b, a+segments), (b, b+segments, a+segments)))
    vertices = (centers[:, None, :]+radii[:, None, None]*unit).reshape(-1, 3)
    triangles = (np.asarray(faces)[None, :, :]
                 + np.arange(len(radii))[:, None, None]*len(unit)).reshape(-1, 3)
    fig = go.Figure(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        intensity=np.repeat(radii, len(unit)), colorscale='Viridis',
        colorbar=dict(title='Radius'), showscale=True,
        hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>',
        lighting=dict(ambient=0.35, diffuse=0.8, specular=0.25, roughness=0.5)))
    edges = []
    for axis in range(3):
        for first in (0, 1):
            for second in (0, 1):
                start = [first, second]
                start.insert(axis, 0)
                end = start.copy()
                end[axis] = 1
                edges.extend((start, end, [None]*3))
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in edges], y=[p[1] for p in edges], z=[p[2] for p in edges],
        mode='lines', line=dict(color='#334155', width=3), hoverinfo='skip', showlegend=False))
    fig.update_layout(
        title=f'{len(radii)} spheres | porosity={porosity:.3f} | {boundary}',
        template='plotly_white', margin=dict(l=0, r=0, t=100, b=55),
        scene=dict(aspectmode='data', dragmode='orbit',
                   xaxis=dict(title='x'), yaxis=dict(title='y'), zaxis=dict(title='z')),
        updatemenus=[dict(type='buttons', direction='left', x=0, y=1.08,
                         buttons=[dict(label=label, method='restyle', args=[{'opacity': value}, [0]])
                                  for label, value in [('Opaque', 1), ('Transparent', 0.3)]])],
        annotations=[dict(text='Drag: rotate | Scroll: zoom | Right-drag: pan'
                          '<br>Periodic view shows each original sphere once; edge spheres may extend outside the box.'
                          if boundary == 'periodic' else 'Drag: rotate | Scroll: zoom | Right-drag: pan',
                          x=0.5, y=-0.06, xref='paper', yref='paper', showarrow=False)])
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={'scrollZoom': True, 'displaylogo': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=N_SPHERES)
    parser.add_argument('--variance', type=float, default=RADIUS_VARIANCE)
    parser.add_argument('--porosity', type=float, default=POROSITY)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--resolution', type=int, default=RESOLUTION)
    parser.add_argument('--boundary', choices=['periodic', 'wall'], default=BOUNDARY)
    parser.add_argument('--variance-mode', choices=['radius', 'log'], default='radius')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    generate(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
