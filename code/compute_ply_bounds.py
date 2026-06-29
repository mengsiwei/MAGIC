"""
compute_ply_bounds.py
=====================

Compute the object's bounding box in MPM space by replicating the exact
same preprocessing that simulate.py applies before running the solver:

  1. Load Gaussian PLY  →  filter by opacity threshold (sigmoid > thresh)
  2. Apply rotation matrices (from config rotation_degree / rotation_axis)
  3. transform2origin  (normalise to [-0.5, 0.5] in dominant axis)
  4. shift2center111   (translate to ~[0.5, 1.5]^3 centred at [1,1,1])

Returns the tight axis-aligned bounding box in MPM space, which is used
by phymagic_pipeline.py to place boundary-condition regions correctly on
the actual object geometry rather than hardcoded unit-cube coordinates.

Usage (standalone)
------------------
  python compute_ply_bounds.py --data_path data/yellowcar

Usage (library)
---------------
  from compute_ply_bounds import compute_mpm_bounds
  bounds = compute_mpm_bounds("data/yellowcar", bottom_quantile=0.05)
  print(bounds)
  # {"xmin": 0.52, "xmax": 1.48, "ymin": 0.62, "ymax": 1.38,
  #  "zmin": 0.55, "zmax": 1.45, "cx": 1.0, "cy": 1.0, "cz": 1.0,
  #  "sx": 0.48, "sy": 0.38, "sz": 0.45,
  #  "hx": 0.96, "hy": 0.76, "hz": 0.90}
  # sx/sy/sz = half-extents; hx/hy/hz = full extents
"""

import os
import math
import argparse
import json
import numpy as np

try:
    from plyfile import PlyData
    _HAVE_PLYFILE = True
except ImportError:
    _HAVE_PLYFILE = False


# ── Sigmoid helper (no torch required) ──────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


# ── Rotation helpers (numpy, matches transformation_utils.py) ────────────────────

def _rot_matrix_np(degree_rad, axis):
    """3×3 rotation matrix around the given axis (0=X, 1=Y, 2=Z)."""
    c, s = math.cos(degree_rad), math.sin(degree_rad)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    elif axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    elif axis == 2:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError(f"Invalid axis: {axis}")


def _apply_rotations_np(xyz, degrees_deg, axes):
    """Apply a sequence of rotations (list of degrees + axes) to (N,3) array."""
    result = xyz.copy()
    for deg, ax in zip(degrees_deg, axes):
        rad = deg / 180.0 * math.pi
        R = _rot_matrix_np(rad, ax)
        result = result @ R.T
    return result


# ── Transform2origin + shift2center111 (numpy, matches transformation_utils.py) ─

def _transform2origin_np(xyz):
    """
    Normalise points so that the largest coordinate range = 1,
    centred at the origin.  Returns (transformed_xyz, scale, mean).
    """
    mn   = xyz.min(axis=0)
    mx   = xyz.max(axis=0)
    mean = (mn + mx) / 2.0
    max_diff = (mx - mn).max()
    if max_diff == 0:
        max_diff = 1.0
    scale = 1.0 / max_diff
    return (xyz - mean) * scale, scale, mean


def _shift2center111_np(xyz):
    return xyz + np.array([1.0, 1.0, 1.0])


# ── Initial rotation applied in load_checkpoint (simulate.py) ───────────────────
# simulate.py always applies R.from_euler('xyz', [-90, 0, 0]) BEFORE the
# config rotation when loading a PLY (see load_checkpoint).

_LOAD_CHECKPOINT_R = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
], dtype=np.float64)   # R = Rx(-90°): y→-z, z→y


# ── Main function ────────────────────────────────────────────────────────────────

def compute_mpm_bounds(
    data_path,
    ply_name="static_0.ply",
    opacity_threshold=0.02,
    rotation_degrees=None,
    rotation_axes=None,
    apply_load_checkpoint_rotation=True,
    bottom_quantile=0.05,
):
    """
    Compute tight object bounding box in MPM space.

    Parameters
    ----------
    data_path : str
        Directory containing the PLY file (e.g. "data/yellowcar").
    ply_name : str
        Name of the PLY file (default "static_0.ply").
    opacity_threshold : float
        Keep Gaussians with sigmoid(opacity_raw) > threshold (default 0.02).
    rotation_degrees : list[float] or None
        Config rotation_degree list (default [0.0] = no rotation).
    rotation_axes : list[int] or None
        Config rotation_axis list (default [0]).
    apply_load_checkpoint_rotation : bool
        If True, apply the Rx(-90°) rotation from load_checkpoint() first.
        Set False for custom PLYs that don't go through load_checkpoint.
    bottom_quantile : float
        Robust bottom plane quantile in MPM space (e.g. 0.05 ignores lowest 5%).

    Returns
    -------
    dict with keys:
        xmin, xmax, ymin, ymax, zmin, zmax  – tight bounds in MPM space
        zmin_majority                       – robust bottom (quantile) in MPM space
        cx, cy, cz                           – centre of bounding box
        sx, sy, sz                           – half-extents (size/2)
        hx, hy, hz                           – full extents
        n_points                             – number of Gaussians kept
    """
    if rotation_degrees is None:
        rotation_degrees = [0.0]
    if rotation_axes is None:
        rotation_axes = [0]

    ply_path = os.path.join(data_path, ply_name)
    if not os.path.exists(ply_path):
        raise FileNotFoundError(
            f"PLY not found: {ply_path}. "
            f"Run TRELLIS first (--from_origin) or provide {ply_name}."
        )

    if not _HAVE_PLYFILE:
        raise ImportError("plyfile is required: pip install plyfile")

    # ── 1. Load PLY ────────────────────────────────────────────────────────────
    plydata = PlyData.read(ply_path)
    vertex  = plydata["vertex"]

    xyz = np.column_stack([
        np.array(vertex["x"], dtype=np.float64),
        np.array(vertex["y"], dtype=np.float64),
        np.array(vertex["z"], dtype=np.float64),
    ])

    # Raw opacity logits; sigmoid → actual opacity
    if "opacity" in vertex.data.dtype.names:
        opacity_raw = np.array(vertex["opacity"], dtype=np.float64)
        opacity_act = _sigmoid(opacity_raw)
        mask = opacity_act > opacity_threshold
    else:
        # No opacity field → keep all points
        mask = np.ones(len(xyz), dtype=bool)

    xyz = xyz[mask]
    if len(xyz) == 0:
        raise ValueError(
            f"No points above opacity threshold {opacity_threshold} in {ply_path}."
        )

    # ── 2. Apply load_checkpoint Rx(-90°) (simulate.py does this for all PLYs) ─
    if apply_load_checkpoint_rotation:
        xyz = xyz @ _LOAD_CHECKPOINT_R.T

    # ── 3. Apply config rotation (rotation_degree / rotation_axis) ────────────
    if any(d != 0.0 for d in rotation_degrees):
        xyz = _apply_rotations_np(xyz, rotation_degrees, rotation_axes)

    # ── 4. transform2origin + shift2center111 ─────────────────────────────────
    xyz, _scale, _mean = _transform2origin_np(xyz)
    xyz = _shift2center111_np(xyz)

    # ── 5. Compute bounding box ────────────────────────────────────────────────
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    cx, cy, cz = ((mn + mx) / 2.0).tolist()
    hx, hy, hz = (mx - mn).tolist()
    sx, sy, sz = hx / 2.0, hy / 2.0, hz / 2.0
    zmin_majority = float(np.quantile(xyz[:, 2], bottom_quantile))

    return {
        "xmin": float(mn[0]), "xmax": float(mx[0]),
        "ymin": float(mn[1]), "ymax": float(mx[1]),
        "zmin": float(mn[2]), "zmax": float(mx[2]),
        "zmin_majority": zmin_majority,
        "cx": cx, "cy": cy, "cz": cz,
        "sx": sx, "sy": sy, "sz": sz,
        "hx": hx, "hy": hy, "hz": hz,
        "n_points": int(mask.sum()),
    }


def make_behavior_bcs(bounds, behavior_key, grid_lim=2.0):
    """
    Build boundary-condition list for ``behavior_key`` adapted to the
    actual object bounding box in MPM space.

    The BCs are placed relative to the real object geometry rather than
    hard-coded cube coordinates:

      • floor         – global sticky plane at z ∈ [0, 0.05] (grid bottom)
      • pin_bottom    – 10 % of object height from bottom
      • compress_top  – particle_impulse on top 50 % of object
      • push_top      – enforce_particle_translation on top 20 % (slow press)
      • pull_top      – enforce_particle_translation on top 10 % (stretch)
      • side_impulse  – particle_impulse covering full object extent

    Parameters
    ----------
    bounds : dict
        Output of compute_mpm_bounds().
    behavior_key : str
        One of: free_fall, side_push, compress, spin, gravity_push,
                stretch, top_push.
    grid_lim : float
        MPM grid upper bound (default 2.0).

    Returns
    -------
    dict with keys "g" (gravity vector) and "boundary_conditions" (list).
    """
    xmin, xmax = bounds["xmin"], bounds["xmax"]
    ymin, ymax = bounds["ymin"], bounds["ymax"]
    zmin, zmax = bounds["zmin"], bounds["zmax"]
    z_floor = bounds.get("zmin_majority", zmin)
    cx,  cy,  cz  = bounds["cx"],  bounds["cy"],  bounds["cz"]
    sx,  sy,  sz  = bounds["sx"],  bounds["sy"],  bounds["sz"]
    hz            = bounds["hz"]          # full height

    # Add a small padding around the object for BC regions
    pad = 0.02

    # ── Shared primitives ──────────────────────────────────────────────────────
    BB = {"type": "bounding_box"}

    # Global sticky floor at z ∈ [0, 0.05]  (grid bottom boundary)
    FLOOR = {
        "type": "cuboid",
        "point": [grid_lim / 2, grid_lim / 2, 0.0],
        "size":  [grid_lim / 2, grid_lim / 2, 0.05],
        "velocity": [0, 0, 0],
        "start_time": 0, "end_time": 1e3, "reset": 1,
    }

    # Pin bottom 10 % of object height (anchored at object's actual base)
    pin_z_cen = z_floor + hz * 0.05       # centre of bottom 10 %
    pin_z_sz  = hz * 0.05 + pad
    PIN_BOTTOM = {
        "type": "cuboid",
        "point": [cx, cy, pin_z_cen],
        "size":  [sx + pad, sy + pad, pin_z_sz],
        "velocity": [0, 0, 0],
        "start_time": 0.0, "end_time": 1e3, "reset": 1,
    }

    # Thin floor at object base — reaction surface for "placed on a plane" behaviors
    FLOOR_OBJ = {
        "type": "cuboid",
        "point": [cx, cy, z_floor],
        "size":  [sx + 0.1, sy + 0.1, 0.01],
        "velocity": [0, 0, 0], "start_time": 0, "end_time": 1e3, "reset": 1,
    }

    if behavior_key == "free_fall":
        # Object falls under gravity and impacts the sticky floor.
        # No pinning — whole object is free.
        return {
            "g": [0, 0, -9.8],
            "boundary_conditions": [BB, FLOOR],
        }

    elif behavior_key == "side_push":
        # Object assumed already on plane; lateral impulse at t=0 covers whole object.
        SIDE_IMPULSE = {
            "type": "particle_impulse",
            "force": [0, -6, 0],
            "num_dt": 20,
            "start_time": 0.0,
            "point": [cx, cy, cz],
            "size":  [sx + pad, sy + pad, sz + pad],
        }
        return {
            "g": [0, 0, -9.8],
            "boundary_conditions": [BB, FLOOR_OBJ, SIDE_IMPULSE],
        }

    elif behavior_key == "compress":
        # Floor at object base provides reaction surface (like a table).
        # Top surface pressed DOWN at −0.3 m/s for 0.5 s (12.5 frames), then released.
        # Elastic springs back; plasticine stays flat.
        COMPRESS_PRESS = {
            "type": "enforce_particle_translation",
            "use_object_top": True,
            "top_fraction": 0.0,    # top surface
            "top_thickness": 0.05,  # top 5% thickness
            "velocity": [0, 0, -0.3],
            "start_time": 0.0, "end_time": 0.5,
        }
        return {
            "g": [0, 0, 0],
            "boundary_conditions": [BB, FLOOR_OBJ, PIN_BOTTOM, COMPRESS_PRESS],
        }

    elif behavior_key == "top_push":
        # Object on plane; top 20 % slowly pushed down; floor provides reaction.
        top_z_cen = zmax - hz * 0.10      # centre of top 20 %
        top_z_sz  = hz * 0.10 + pad
        TOP_PUSH_VEL = {
            "type": "enforce_particle_translation",
            "point": [cx, cy, top_z_cen],
            "size":  [sx + pad, sy + pad, top_z_sz],
            "velocity": [0, 0, -0.2],
            "start_time": 0.0, "end_time": 1.0,
        }
        return {
            "g": [0, 0, 0],
            "boundary_conditions": [BB, FLOOR_OBJ, TOP_PUSH_VEL],
        }

    elif behavior_key == "spin":
        # Object on plane; rotates around Z axis passing through (cx, cy).
        lateral_radius = max(sx, sy) + 0.1
        SPIN = {
            "type": "enforce_particle_velocity_rotation",
            "point":  [cx, cy, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "half_height_and_radius": [grid_lim / 2, lateral_radius],
            "rotation_scale":    3.0,
            "translation_scale": 0.0,
            "start_time": 0.0,
            "end_time":   0.8,
        }
        return {
            "g": [0, 0, 0],
            "boundary_conditions": [BB, FLOOR_OBJ, SPIN],
        }

    elif behavior_key == "gravity_push":
        # Object on plane; tilted gravity (~11° slope in Y) causes lateral sliding.
        return {
            "g": [0, -2, -9.8],
            "boundary_conditions": [BB, FLOOR_OBJ],
        }

    elif behavior_key == "stretch":
        # Object on plane; floor holds base (reset=1 pins contact layer);
        # top 10 % pulled upward — reveals elastic spring-back vs permanent elongation.
        pull_z_cen = zmax - hz * 0.05     # centre of top 10 %
        pull_z_sz  = hz * 0.05 + pad
        PULL_TOP = {
            "type": "enforce_particle_translation",
            "point": [cx, cy, pull_z_cen],
            "size":  [sx + pad, sy + pad, pull_z_sz],
            "velocity": [0, 0, 0.5],
            "start_time": 0.0, "end_time": 0.8,
        }
        return {
            "g": [0, 0, 0],
            "boundary_conditions": [BB, FLOOR_OBJ, PULL_TOP],
        }

    else:
        raise ValueError(
            f"Unknown behavior_key '{behavior_key}'. "
            f"Valid: free_fall, side_push, compress, top_push, spin, gravity_push, stretch"
        )


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute object bounding box in MPM space from PLY file"
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Path to data directory containing static_0.ply (e.g. data/yellowcar)",
    )
    parser.add_argument(
        "--ply_name", default="static_0.ply",
    )
    parser.add_argument(
        "--opacity_threshold", type=float, default=0.02,
    )
    parser.add_argument(
        "--bottom_quantile", type=float, default=0.05,
        help="Quantile to define robust bottom plane (e.g. 0.05 ignores lowest 5%%)",
    )
    parser.add_argument(
        "--no_load_checkpoint_rotation", action="store_true",
        help="Skip the Rx(-90°) rotation applied by load_checkpoint()",
    )
    parser.add_argument(
        "--behavior", default=None,
        choices=["free_fall","side_push","compress","top_push","spin","gravity_push","stretch"],
        help="Also print the adapted BC config for this behavior",
    )
    args = parser.parse_args()

    bounds = compute_mpm_bounds(
        data_path=args.data_path,
        ply_name=args.ply_name,
        opacity_threshold=args.opacity_threshold,
        apply_load_checkpoint_rotation=not args.no_load_checkpoint_rotation,
        bottom_quantile=args.bottom_quantile,
    )

    print("\nObject bounds in MPM space:")
    print(f"  X: [{bounds['xmin']:.4f}, {bounds['xmax']:.4f}]  width={bounds['hx']:.4f}")
    print(f"  Y: [{bounds['ymin']:.4f}, {bounds['ymax']:.4f}]  depth={bounds['hy']:.4f}")
    print(f"  Z: [{bounds['zmin']:.4f}, {bounds['zmax']:.4f}]  height={bounds['hz']:.4f}")
    print(f"  Z majority (q={args.bottom_quantile:.3f}): {bounds['zmin_majority']:.4f}")
    print(f"  centre: ({bounds['cx']:.4f}, {bounds['cy']:.4f}, {bounds['cz']:.4f})")
    print(f"  half-extents: ({bounds['sx']:.4f}, {bounds['sy']:.4f}, {bounds['sz']:.4f})")
    print(f"  n_points: {bounds['n_points']}")

    if args.behavior:
        print(f"\nAdapted BCs for '{args.behavior}':")
        bc = make_behavior_bcs(bounds, args.behavior)
        print(json.dumps(bc, indent=2))


if __name__ == "__main__":
    main()
