"""
llm_to_mpm_config.py

Converts LLM-inferred physical properties (physics.yaml from infer_physics_0124.py)
into a simulator config JSON accepted by dynamic.py / decode_param_json.

Supported scene types (auto-detected or manually specified via --scene_type):
  elastic_freefall  – elastic object falls under gravity, bounces
  rigid_freefall    – rigid object falls under gravity
  rigid_move        – rigid object driven by initial velocity / external force
  plastic           – plasticine / foam / snow squishes on impact
  granular          – sand pile collapses under gravity
  fluid             – viscous fluid flows under gravity

Usage
-----
  # Auto-detect scene from LLM output:
  python llm_to_mpm_config.py \\
      --physics_yaml data/yellowcar/physics.yaml \\
      --output_config configs/yellowcar_config.json

  # Force a specific scene type:
  python llm_to_mpm_config.py \\
      --physics_yaml data/car/physics.yaml \\
      --output_config configs/car_move_config.json \\
      --scene_type rigid_move

  # Chain after infer_physics_0124.py inside a script:
  from llm_to_mpm_config import llm_yaml_to_mpm_config
  config = llm_yaml_to_mpm_config("data/yellowcar/physics.yaml",
                                   "configs/yellowcar_config.json")
"""

import json
import argparse
from ruamel.yaml import YAML

# ── LLM material string → MPM material enum string ─────────────────────────────
# MPM solver accepts: "jelly"/"elastic", "metal", "sand", "foam", "snow",
#                     "plasticine", "non_newtonian"
MATERIAL_MAP = {
    "elastic":             "jelly",
    "jelly":               "jelly",
    "rubber":              "jelly",
    "soft":                "jelly",
    "plasticine":          "plasticine",
    "clay":                "plasticine",
    "putty":               "plasticine",
    "foam":                "foam",
    "sponge":              "foam",
    "snow":                "snow",
    "ice":                 "snow",
    "rigid":               "metal",
    "metal":               "metal",
    "hard":                "metal",
    "glass":               "metal",
    "wood":                "metal",
    "plastic":             "metal",    # rigid plastic
    "ceramic":             "metal",
    "sand":                "sand",
    "soil":                "sand",
    "granular":            "sand",
    "powder":              "sand",
    "newtonian fluid":     "non_newtonian",
    "non-newtonian fluid": "non_newtonian",
    "non_newtonian fluid": "non_newtonian",
    "fluid":               "non_newtonian",
    "liquid":              "non_newtonian",
    "water":               "non_newtonian",
}

# E fallbacks when LLM does not provide a value
_E_DEFAULT = {
    "jelly":         3e4,
    "plasticine":    5e4,
    "foam":          1e3,
    "snow":          1.4e5,
    "metal":         1e7,
    "sand":          1e6,
    "non_newtonian": 1e5,
}

# nu fallbacks
_NU_DEFAULT = {
    "jelly":         0.45,
    "plasticine":    0.40,
    "foam":          0.20,
    "snow":          0.20,
    "metal":         0.30,
    "sand":          0.30,
    "non_newtonian": 0.40,
}

# Cap E for rigid ("metal") materials to prevent MPM blow-up.
# Real steel is ~2e11 Pa which is unphysically stiff for MPM particles.
_RIGID_E_CAP = 1e8


def _get(d, *keys, default=None):
    """Return the first key found in dict d, else default."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def _to_vec3(v, default=None):
    """Normalise a scalar / 2-list / 3-list value to [x, y, z]."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return [float(v), 0.0, 0.0]
    lst = list(v)
    if len(lst) == 3:
        return [float(x) for x in lst]
    if len(lst) == 1:
        return [float(lst[0]), 0.0, 0.0]
    return default


def _nonzero(vec):
    return vec is not None and any(x != 0 for x in vec)


# ── Scene type classification ────────────────────────────────────────────────────

def classify_scene(mpm_material, initial_velocity, external_force):
    """
    Determine the scene type from material and dynamic properties.

    Returns one of:
        "elastic_freefall"   – elastic drop/bounce
        "rigid_freefall"     – rigid free-fall
        "rigid_move"         – rigid object pushed / given initial velocity
        "plastic"            – plastically deforming material
        "granular"           – sand/soil collapse
        "fluid"              – viscous flow
    """
    has_motion = _nonzero(initial_velocity) or _nonzero(external_force)

    if mpm_material == "jelly":
        return "elastic_freefall"
    elif mpm_material == "metal":
        return "rigid_move" if has_motion else "rigid_freefall"
    elif mpm_material in ("plasticine", "foam", "snow"):
        return "plastic"
    elif mpm_material == "sand":
        return "granular"
    elif mpm_material == "non_newtonian":
        return "fluid"
    return "elastic_freefall"


# ── Boundary condition builders per scene type ──────────────────────────────────

def _append_dynamics(bcs, initial_velocity, external_force, frame_dt):
    """Add velocity / impulse BCs whenever the LLM provided non-zero values."""
    if _nonzero(initial_velocity):
        bcs.append({
            "type":       "enforce_particle_translation",
            "point":      [0.0, 0.0, 0.0],
            "size":       [2.0, 2.0, 2.0],
            "velocity":   initial_velocity,
            "start_time": 0.0,
            "end_time":   float(frame_dt),  # enforce for one frame then release
        })
    if _nonzero(external_force):
        bcs.append({
            "type":       "particle_impulse",
            "force":      external_force,
            "num_dt":     1,
            "start_time": 0.0,
        })
    return bcs


def _floor_bc(grid_lim):
    return {
        "type": "cuboid",
        "point": [grid_lim / 2, grid_lim / 2, 0.0],
        "size":  [grid_lim / 2, grid_lim / 2, 0.05],
        "velocity": [0, 0, 0],
        "start_time": 0,
        "end_time": 1e3,
        "reset": 1,
    }


def _bcs_elastic_freefall(initial_velocity, external_force, frame_dt, grid_lim):
    """Elastic object drops under gravity and bounces off the floor."""
    bcs = [
        {"type": "bounding_box"},
        _floor_bc(grid_lim),
        {
            # Add a brief upward push on the top layer after impact to ensure rebound.
            "type": "enforce_particle_translation",
            "use_object_top": True,
            "top_fraction": 0.1,
            "top_thickness": 0.1,
            "velocity": [0, 0, 0.8],
            "start_time": 8 * frame_dt,
            "end_time": 11 * frame_dt,
        },
    ]
    return _append_dynamics(bcs, initial_velocity, external_force, frame_dt)


def _bcs_rigid_freefall(initial_velocity, external_force, frame_dt, grid_lim):
    """Rigid object drops under gravity."""
    bcs = [{"type": "bounding_box"}, _floor_bc(grid_lim)]
    return _append_dynamics(bcs, initial_velocity, external_force, frame_dt)


def _bcs_rigid_move(initial_velocity, external_force, frame_dt, grid_lim):
    """Rigid object driven by initial velocity and / or external force."""
    return _append_dynamics([{"type": "bounding_box"}], initial_velocity, external_force, frame_dt)


def _bcs_plastic(initial_velocity, external_force, frame_dt, grid_lim):
    """Plasticine / foam / snow falls and squishes."""
    return _append_dynamics([{"type": "bounding_box"}], initial_velocity, external_force, frame_dt)


def _bcs_granular(initial_velocity, external_force, frame_dt, grid_lim):
    """Sand pile collapses; sticky floor so grains pile up."""
    bcs = [
        {"type": "bounding_box"},
        {
            "type":       "surface_collider",
            "point":      [0.0, 0.0, 0.0],
            "normal":     [0.0, 0.0, 1.0],
            "surface":    "sticky",
            "friction":   0.5,
            "start_time": 0.0,
            "end_time":   1e3,
        },
    ]
    return _append_dynamics(bcs, initial_velocity, external_force, frame_dt)


def _bcs_fluid(initial_velocity, external_force, frame_dt, grid_lim):
    """Fluid flows and pools; sticky floor to contain it."""
    bcs = [
        {"type": "bounding_box"},
        {
            "type":       "surface_collider",
            "point":      [0.0, 0.0, 0.0],
            "normal":     [0.0, 0.0, 1.0],
            "surface":    "sticky",
            "friction":   0.0,
            "start_time": 0.0,
            "end_time":   1e3,
        },
        {
            # brief upward push on the bottom layer to simulate splash after impact
            "type": "enforce_particle_translation",
            "use_object_bottom": True,
            "bottom_fraction": 0.1,
            "bottom_thickness": 0.1,
            "velocity": [0, 0, 1.2],
            "start_time": 8 * frame_dt,
            "end_time": 11 * frame_dt,
        },
        {
            # short spin to spread fluid radially after splash
            "type": "enforce_particle_velocity_rotation",
            "use_object_spin": True,
            "rotation_scale": 6.0,
            "translation_scale": 0.0,
            "start_time": 11 * frame_dt,
            "end_time": 20 * frame_dt,
        },
    ]
    return _append_dynamics(bcs, initial_velocity, external_force, frame_dt)


_BC_BUILDERS = {
    "elastic_freefall": _bcs_elastic_freefall,
    "rigid_freefall":   _bcs_rigid_freefall,
    "rigid_move":       _bcs_rigid_move,
    "plastic":          _bcs_plastic,
    "granular":         _bcs_granular,
    "fluid":            _bcs_fluid,
}


# ── Main conversion function ────────────────────────────────────────────────────

def llm_yaml_to_mpm_config(
    physics_yaml_path,
    output_config_path=None,
    scene_type="auto",
    frame_dt=4e-2,
    frame_num=100,
    substep_dt=1e-4,
    n_grid=50,
    grid_lim=2.0,
):
    """
    Convert a physics.yaml (output of infer_physics_0124.py) into a
    MPM simulator config dict compatible with dynamic.py / decode_param_json.

    Parameters
    ----------
    physics_yaml_path : str
        Path to the YAML produced by infer_physics_0124.py.
    output_config_path : str or None
        Write the JSON config here (if given).
    scene_type : str
        "auto" to infer from LLM output, or one of the explicit scene
        type strings listed in the module docstring.
    frame_dt, frame_num, substep_dt : float / int
        Simulation time parameters.
    n_grid, grid_lim : int / float
        MPM grid resolution and physical domain size.

    Returns
    -------
    dict  – the complete MPM config dict.
    """
    yaml_parser = YAML()
    with open(physics_yaml_path, "r") as fh:
        physics = yaml_parser.load(fh)

    # Support both list-of-objects and single-object YAML
    if isinstance(physics, list):
        obj = physics[0]
    else:
        obj = physics

    # ── Extract & normalize LLM fields ────────────────────────────────────────
    llm_material = str(_get(obj, "material", default="elastic")).lower().strip()
    mpm_material = MATERIAL_MAP.get(llm_material, "jelly")

    density = float(_get(obj, "density", default=200.0))

    E = float(_get(
        obj,
        "youngsModulus", "E", "young_modulus", "youngModulus",
        default=_E_DEFAULT[mpm_material],
    ))
    nu = float(_get(
        obj,
        "poissonsRatio", "nu", "poisson_ratio",
        default=_NU_DEFAULT[mpm_material],
    ))
    yield_stress     = _get(obj, "yieldStress",       "yield_stress")
    friction_angle   = _get(obj, "frictionAngle",     "friction_angle")
    plastic_viscosity = _get(obj, "plasticViscosity", "plastic_viscosity")

    # Clamp E for rigid objects (MPM blows up with real-world steel stiffness)
    if mpm_material == "metal" and E > _RIGID_E_CAP:
        E = _RIGID_E_CAP

    # nu must be strictly inside (0, 0.5)
    nu = max(0.01, min(0.49, nu))

    # Dynamic properties
    initial_velocity = _to_vec3(_get(
        obj, "initial velocity", "initialVelocity", "initial_velocity"
    ))
    external_force = _to_vec3(_get(
        obj, "external force", "externalForce", "external_force"
    ))

    # ── Determine scene type ──────────────────────────────────────────────────
    if scene_type == "auto":
        scene_type = classify_scene(mpm_material, initial_velocity, external_force)

    # ── Build boundary conditions ─────────────────────────────────────────────
    bcs = _BC_BUILDERS[scene_type](initial_velocity, external_force, frame_dt, grid_lim)

    # ── Gravity direction (MPM convention: z-up, gravity in -z) ──────────────
    g = [0.0, 0.0, -9.8]

    # ── Assemble config ───────────────────────────────────────────────────────
    config = {
        # Material
        "material":             mpm_material,
        "E":                    E,
        "nu":                   nu,
        "density":              density,
        "g":                    g,
        # Grid
        "n_grid":               n_grid,
        "grid_lim":             grid_lim,
        # Time
        "substep_dt":           substep_dt,
        "frame_dt":             frame_dt,
        "frame_num":            frame_num,
        # Pre-processing defaults
        "opacity_threshold":    0.02,
        "rotation_degree":      [0.0],
        "rotation_axis":        [0],
        "grid_v_damping_scale": 0.9999,
        "rpic_damping":         0.0,
        # Boundary conditions
        "boundary_conditions":  bcs,
        # Camera defaults (centred, slightly elevated view)
        "mpm_space_vertical_upward_axis": [0, 0, 1],
        "mpm_space_viewpoint_center":     [1.0, 1.0, 1.0],
        "default_camera_index": -1,
        "show_hint":            False,
        "init_azimuthm":        -36.7,
        "init_elevation":       8.96,
        "init_radius":          4.11,
        "move_camera":          False,
        "delta_a":              0.0,
        "delta_e":              0.0,
        "delta_r":              0.0,
        # Debug metadata (ignored by decode_param_json, useful for inspection)
        "_scene_type":          scene_type,
        "_llm_material":        llm_material,
    }

    # Per-material optional params
    if yield_stress is not None:
        config["yield_stress"] = float(yield_stress)
    if friction_angle is not None and mpm_material == "sand":
        config["friction_angle"] = float(friction_angle)
    if plastic_viscosity is not None:
        config["plastic_viscosity"] = float(plastic_viscosity)

    # ── Write JSON ────────────────────────────────────────────────────────────
    if output_config_path is not None:
        with open(output_config_path, "w") as fh:
            json.dump(config, fh, indent=4)
        print(f"[llm_to_mpm] scene_type  : {scene_type}")
        print(f"[llm_to_mpm] mpm_material: {mpm_material}  E={E:.3g}  nu={nu:.3f}  density={density:.1f}")
        print(f"[llm_to_mpm] config saved: {output_config_path}")

    return config


# ── CLI entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LLM physics.yaml → MPM simulator config JSON"
    )
    parser.add_argument(
        "--physics_yaml", required=True,
        help="Path to physics.yaml produced by infer_physics_0124.py",
    )
    parser.add_argument(
        "--output_config", required=True,
        help="Path to write the output JSON config (consumed by dynamic.py)",
    )
    parser.add_argument(
        "--scene_type", default="auto",
        choices=["auto", "elastic_freefall", "rigid_freefall",
                 "rigid_move", "plastic", "granular", "fluid"],
        help="Override scene type (default: auto-detect from LLM output)",
    )
    parser.add_argument("--frame_dt",   type=float, default=4e-2)
    parser.add_argument("--frame_num",  type=int,   default=100)
    parser.add_argument("--substep_dt", type=float, default=1e-4)
    parser.add_argument("--n_grid",     type=int,   default=50)
    parser.add_argument("--grid_lim",   type=float, default=2.0)

    args = parser.parse_args()
    llm_yaml_to_mpm_config(
        physics_yaml_path=args.physics_yaml,
        output_config_path=args.output_config,
        scene_type=args.scene_type,
        frame_dt=args.frame_dt,
        frame_num=args.frame_num,
        substep_dt=args.substep_dt,
        n_grid=args.n_grid,
        grid_lim=args.grid_lim,
    )
