#!/usr/bin/env python3
"""
glb_to_ply.py  –  Convert a GLB mesh to a Gaussian-Splatting PLY.

Key improvement over naive sphere-Gaussians:
  Each Gaussian is a flat disc aligned with the mesh surface normal,
  giving much sharper rendering than isotropic spheres.

Usage:
    python glb_to_ply.py data/yellowcar/sample_car.glb data/yellowcar/static_0.ply
    python glb_to_ply.py data/mug/model.glb data/mug/static_0.ply --n_samples 80000

The output PLY is directly compatible with simulate.py (GaussianModel.load_ply).
Once static_0.ply exists, simulate.py will prefer it over the GLB automatically.
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh

# SH DC:  rendered_color = C0 * f_dc + 0.5
C0 = 0.28209479177387814


def _normal_to_quaternion(normals: np.ndarray) -> np.ndarray:
    """
    Vectorised: compute quaternions that rotate [0,0,1] → each surface normal.
    Returns (N, 4) float32 [w, x, y, z].
    """
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    dot = np.clip(normals[:, 2], -1.0, 1.0)          # cos(angle) = z-component

    # cross([0,0,1], n) = (-ny, nx, 0)
    cross = np.stack([-normals[:, 1], normals[:, 0],
                      np.zeros(len(normals), dtype=np.float32)], axis=1)
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)
    safe_cross = cross / (cross_norm + 1e-8)

    half = np.arccos(dot) / 2.0
    quats = np.zeros((len(normals), 4), dtype=np.float32)
    quats[:, 0] = np.cos(half)
    quats[:, 1:] = safe_cross * np.sin(half)[:, None]

    # Degenerate cases
    quats[dot >  0.9999] = [1, 0, 0, 0]
    quats[dot < -0.9999] = [0, 1, 0, 0]   # 180° flip around X
    return quats


def glb_to_ply(glb_path: str, ply_path: str, n_samples: int = 50000) -> None:
    # ── Load mesh ──────────────────────────────────────────────────────────────
    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = loaded

    print(f"Mesh : {len(mesh.vertices):,} verts  {len(mesh.faces):,} faces")
    print(f"Area : {mesh.area:.4f} m²")

    # ── Sample surface ─────────────────────────────────────────────────────────
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    pts = pts.astype(np.float32)

    # GLTF Y-up → MPM Z-up:  (x, y, z) → (x, -z, y)
    pts = pts[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0], dtype=np.float32)

    # Surface normals (per face, then same rotation)
    normals = mesh.face_normals[face_idx].astype(np.float32)         # (N, 3)
    normals = normals[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0], dtype=np.float32)

    N = len(pts)

    # ── Colors from texture / vertex colors ───────────────────────────────────
    try:
        vc = mesh.visual.to_color().vertex_colors   # (V, 4) uint8
        fv = mesh.faces[face_idx]                   # (N, 3)
        colors = vc[fv, :3].mean(axis=1).astype(np.float32) / 255.0
    except Exception:
        print("Warning: could not extract colors; using neutral grey.")
        colors = np.full((N, 3), 0.5, dtype=np.float32)

    f_dc = ((colors - 0.5) / C0).astype(np.float32)

    # ── Gaussian scale: flat disc aligned with surface normal ──────────────────
    # Tangent radius ≈ average spacing on surface
    avg_area_per_sample = mesh.area / n_samples
    tangent_sigma = float(np.sqrt(avg_area_per_sample / np.pi))
    normal_sigma  = tangent_sigma / 3.0          # 3× thinner along normal

    log_tangent = float(np.log(max(tangent_sigma, 1e-6)))
    log_normal  = float(np.log(max(normal_sigma,  1e-6)))

    print(f"Tangent σ : {tangent_sigma:.5f}  →  log = {log_tangent:.3f}")
    print(f"Normal  σ : {normal_sigma:.5f}  →  log = {log_normal:.3f}")

    # ── Quaternions: align Gaussian z-axis with surface normal ─────────────────
    quats = _normal_to_quaternion(normals)   # (N, 4) [w, x, y, z]

    # ── Write PLY ──────────────────────────────────────────────────────────────
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('opacity', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0',   'f4'), ('rot_1',   'f4'), ('rot_2',   'f4'), ('rot_3', 'f4'),
    ]
    vd = np.zeros(N, dtype=dtype)
    vd['x'],      vd['y'],      vd['z']      = pts[:, 0], pts[:, 1], pts[:, 2]
    vd['opacity']                             = 4.0          # sigmoid(4) ≈ 0.982
    vd['f_dc_0'], vd['f_dc_1'], vd['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    vd['scale_0'] = log_tangent
    vd['scale_1'] = log_tangent
    vd['scale_2'] = log_normal
    vd['rot_0'],  vd['rot_1'],  vd['rot_2'],  vd['rot_3'] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    )

    PlyData([PlyElement.describe(vd, 'vertex')]).write(ply_path)
    print(f"Wrote {N:,} Gaussians → {ply_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GLB mesh to Gaussian-Splatting PLY for simulate.py"
    )
    parser.add_argument("glb",          help="Input GLB file")
    parser.add_argument("ply",          help="Output PLY file (e.g. static_0.ply)")
    parser.add_argument("--n_samples",  type=int, default=50000,
                        help="Surface sample count (default: 50000)")
    args = parser.parse_args()

    glb_to_ply(args.glb, args.ply, args.n_samples)
