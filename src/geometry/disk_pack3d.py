"""三维多粒径球堆积。使用项目 .venv 解释器运行，可直接修改下方默认参数。"""
import argparse
import importlib.metadata
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
import rcpgenerator
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# PyCharm 中直接运行时使用这些参数。
N_SPHERES = 300
RADIUS_VARIANCE = 0.35   # 归一化半径 E[R]=1 时的分布方差，非最终像素半径方差
POROSITY = 0.50         # 最终孔隙体积分数，越大越疏松
SEED = 42
RESOLUTION = 128        # 体素模型边长；设为 0 可跳过体素生成
BOUNDARY = 'periodic'   # periodic：周期边界；clip：截断盒外球体；wall：整球位于盒内
OUTPUT = Path(__file__).resolve().parent/'output'/'spheres_3d'


def minimum_gap(centers, radii, boundary):
    """独立逐球检查，避免创建 N×N×3 大数组。长度单位为盒子边长。"""
    gap = float(np.min(1-2*radii)) if boundary in ('periodic', 'clip') else float(
        np.min(np.minimum(centers, 1-centers)-radii[:, None]))
    for i in range(len(radii)-1):
        delta = centers[i+1:]-centers[i]
        if boundary in ('periodic', 'clip'):
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
            if boundary != 'periodic':
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


def periodic_images(centers, radii):
    """为越界球添加其在相对边及棱角处的周期镜像。"""
    image_centers, image_radii = [], []
    for center, radius in zip(centers, radii):
        shifts = []
        for coordinate in center:
            choices = [0]
            if coordinate-radius < 0:
                choices.append(1)
            if coordinate+radius > 1:
                choices.append(-1)
            shifts.append(choices)
        for shift in itertools.product(*shifts):
            image_centers.append(center+shift)
            image_radii.append(radius)
    return np.asarray(image_centers), np.asarray(image_radii)


def preview(centers, radii, path, porosity, boundary):
    if boundary == 'periodic':
        centers, radii = periodic_images(centers, radii)
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 17), np.linspace(0, np.pi, 13))
    sx, sy, sz = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    colors = plt.get_cmap('viridis')
    for (x, y, z), r in zip(centers, radii):
        color = colors((r-radii.min())/max(float(np.ptp(radii)), 1e-12))
        ax.plot_surface(x+r*sx, y+r*sy, z+r*sz, color=color,
                        linewidth=0, shade=True, axlim_clip=True)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel='x', ylabel='y', zlabel='z')
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f'porosity={porosity:.3f} | {boundary}')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def generate(n=N_SPHERES, variance=RADIUS_VARIANCE, porosity=POROSITY,
             seed=SEED, resolution=RESOLUTION, boundary=BOUNDARY,
             output=OUTPUT, variance_mode='radius'):
    if not isinstance(n, int) or n < 2 or seed < 0:
        raise ValueError('Use integer n >= 2 and seed >= 0')
    if not np.isfinite(variance) or variance < 0 or not np.isfinite(porosity) or not 0 < porosity < 1:
        raise ValueError('Use finite variance >= 0 and 0 < porosity < 1')
    if boundary not in ('clip', 'periodic', 'wall') or variance_mode not in ('radius', 'log'):
        raise ValueError('Invalid boundary or variance_mode')
    if resolution != 0 and resolution < 16:
        raise ValueError('resolution must be 0 or >= 16')
    sigma2 = np.log1p(variance) if variance_mode == 'radius' else variance
    weights = np.random.default_rng(seed).lognormal(-sigma2/2, np.sqrt(sigma2), n)
    rcpgenerator.set_num_threads(1)
    print(f'Generating {n} spheres with RCPGenerator...', flush=True)
    packing = rcpgenerator.Packing(phi=.08, N=n, Ndim=3, box=[1., 1., 1.],
        walls=[0 if boundary in ('periodic', 'clip') else 1]*3,
        dist={'type': 'custom', 'custom': weights.tolist()}, seed=seed, neighbor_max=0)
    result = packing.pack()
    centers = np.asarray(result['positions'], dtype=float)
    raw_radii = np.asarray(result['diameters'], dtype=float)/2
    if centers.shape != (n, 3) or raw_radii.shape != (n,) or not np.isfinite(centers).all() or not np.isfinite(raw_radii).all() or np.any(raw_radii <= 0):
        raise RuntimeError('Invalid backend output')
    if boundary in ('periodic', 'clip'):
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
    try:
        from .export_comsol_java import export_comsol_java
    except ImportError:
        from export_comsol_java import export_comsol_java
    export_comsol_java(centers, radii, output/'PeriodicSphereCell.java',
                       boundary=boundary)
    metadata = dict(n=n, variance=variance, variance_mode=variance_mode,
                    porosity=porosity, seed=seed, resolution=resolution,
                    boundary=boundary, minimum_gap=gap,
                    rcpgenerator_version=importlib.metadata.version('rcpgenerator'))
    if boundary == 'clip':
        metadata['porosity_note'] = ('porosity is computed from full sphere volumes; '
                                    'voxel_porosity measures the cropped cube')
    if resolution:
        solid = voxelize(centers, radii, resolution, boundary)
        np.save(output/'solid.npy', solid)
        grid = pv.ImageData(dimensions=(resolution+1,)*3,
                            spacing=(1/resolution,)*3)
        grid.cell_data['solid'] = solid.transpose(2, 1, 0).ravel(order='F')
        grid.save(output/'solid.vti')
        metadata['voxel_porosity'] = float(1-solid.mean())
    display_porosity = metadata.get('voxel_porosity', porosity)
    preview(centers, radii, output/'preview.png', display_porosity, boundary)
    interactive_preview(centers, radii, output/'preview.html', display_porosity, boundary,
                        mesh_path=output/'surface_mesh.npz')
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print(f'Saved to: {output.resolve()}')
    return centers, radii


def solid_sphere_mesh(centers, radii):
    """生成与单位立方体相交的实心球表面，裁切面由平面三角形封口。"""
    planes = vtk.vtkPlaneCollection()
    for axis in range(3):
        for coordinate, direction in ((0, 1), (1, -1)):
            plane = vtk.vtkPlane()
            origin, normal = [0]*3, [0]*3
            origin[axis], normal[axis] = coordinate, direction
            plane.SetOrigin(origin)
            plane.SetNormal(normal)
            planes.AddItem(plane)
    all_vertices, all_faces, all_values = [], [], []
    offset = 0
    for center, radius in zip(centers, radii):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*center)
        sphere.SetRadius(float(radius))
        sphere.SetThetaResolution(24)
        sphere.SetPhiResolution(16)
        if np.any(center-radius < 0) or np.any(center+radius > 1):
            clipper = vtk.vtkClipClosedSurface()
            clipper.SetInputConnection(sphere.GetOutputPort())
            clipper.SetClippingPlanes(planes)
            clipper.GenerateFacesOn()
            surface = clipper
        else:
            surface = sphere
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(surface.GetOutputPort())
        triangles.Update()
        mesh = triangles.GetOutput()
        vertices = vtk_to_numpy(mesh.GetPoints().GetData()).copy()
        faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3).copy()
        all_vertices.append(np.clip(vertices, 0, 1))
        all_faces.append(faces+offset)
        all_values.append(np.full(len(vertices), radius))
        offset += len(vertices)
    return (np.concatenate(all_vertices), np.concatenate(all_faces),
            np.concatenate(all_values))


def interactive_preview(centers, radii, path, porosity, boundary, mesh_path=None):
    """离线 HTML 三维视图：拖动旋转、滚轮缩放，按钮切换透明度。"""
    if boundary == 'periodic':
        centers, radii = periodic_images(centers, radii)
    vertices, triangles, intensity = solid_sphere_mesh(centers, radii)
    if mesh_path is not None:
        np.savez_compressed(mesh_path, vertices=vertices, faces=triangles)
        faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
        surface = pv.PolyData(vertices, faces)
        surface.point_data['radius'] = intensity
        surface.save(mesh_path.with_suffix('.vtp'))
        surface.save(mesh_path.with_suffix('.stl'), binary=True)
    fig = go.Figure(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        intensity=intensity, colorscale='Viridis',
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
        title=f'porosity={porosity:.3f} | {boundary}',
        template='plotly_white', margin=dict(l=0, r=0, t=100, b=55),
        scene=dict(aspectmode='data', dragmode='orbit',
                   xaxis=dict(title='x'), yaxis=dict(title='y'), zaxis=dict(title='z')),
        updatemenus=[dict(type='buttons', direction='left', x=0, y=1.08,
                         buttons=[dict(label=label, method='restyle', args=[{'opacity': value}, [0]])
                                  for label, value in [('Opaque', 1), ('Transparent', 0.3)]])],
        annotations=[dict(text='Drag: rotate | Scroll: zoom | Right-drag: pan',
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
    parser.add_argument('--boundary', choices=['clip', 'periodic', 'wall'], default=BOUNDARY)
    parser.add_argument('--variance-mode', choices=['radius', 'log'], default='radius')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    generate(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
