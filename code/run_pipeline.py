#!/usr/bin/env python3
"""
run_pipeline.py — MPM Foreground Simulation Compositing Pipeline
(implements renew.md spec, integrates with existing phymagic MPM pipeline)

Usage examples:
  # External particle mode (particle NPZ/PLY files from any MPM solver):
  python run_pipeline.py --image assets/input.png --mpm_dir mpm/ --mode particles --fps 24 --out out/

  # Phymagic integration: run full simulation then composite:
  python run_pipeline.py --image data/yellowcar/origin.png \
      --data_path data/yellowcar --config_path configs/ \
      --fps 24 --out out/yellowcar/

  # Phymagic integration with Blender shadow rendering:
  python run_pipeline.py --image data/yellowcar/origin.png \
      --data_path data/yellowcar --config_path configs/ \
      --fps 24 --out out/yellowcar/ --use_blender

  # External particles with Blender rendering:
  python run_pipeline.py --image assets/input.png --mpm_dir mpm/ --mode particles \
      --fps 24 --out out/ --use_blender --fov 50 --light_dir "0.2,-0.8,0.6"

Outputs (in --out directory):
  fg_rgba/%04d.png    — foreground RGBA frames
  fg_depth/%04d.exr   — foreground depth (Blender mode only)
  shadow/%04d.png     — shadow pass frames
  final_frames/%04d.png — composited frames
  final.mp4           — final video
  metadata.json       — camera + alignment parameters
"""

import argparse
import os
import sys
import json
import math
import shutil
import subprocess
import glob as _glob
from pathlib import Path

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MPM foreground simulation compositing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Input ---
    p.add_argument("--image", type=str, default=None,
                   help="Background plate image (e.g. assets/input.png or data/yellowcar/origin.png)")
    p.add_argument("--mpm_dir", type=str, default=None,
                   help="Directory with MPM particle/mesh files per frame "
                        "(frame_0000.npz, frame_0000.ply, or frame_0000.obj)")
    p.add_argument("--mode", type=str, default="particles",
                   choices=["particles", "mesh", "gaussian"],
                   help="MPM data format: particles (NPZ/PLY), mesh (OBJ/PLY), "
                        "or gaussian (use existing simulate.py RGBA output)")

    # ---- Phymagic integration ---
    p.add_argument("--data_path", type=str, default=None,
                   help="Phymagic data path (e.g. data/yellowcar). "
                        "Used with --mode gaussian to locate existing renders.")
    p.add_argument("--config_path", type=str, default="configs/",
                   help="Config path for simulate.py (used with --run_simulate)")
    p.add_argument("--run_simulate", action="store_true",
                   help="Run simulate.py first to generate/export frames. "
                        "Requires --data_path and a config JSON in --config_path.")
    p.add_argument("--instruction", type=str, default="",
                   help="Physics instruction (only used when --run_phymagic_pipeline is set)")
    p.add_argument("--run_phymagic_pipeline", action="store_true",
                   help="Run the full phymagic_pipeline.py before compositing.")

    # ---- Output ---
    p.add_argument("--out", type=str, default="out/",
                   help="Output directory. Subdirs fg_rgba/, shadow/, final_frames/ "
                        "are created automatically.")
    p.add_argument("--fps", type=int, default=24, help="Output video FPS")

    # ---- Camera / alignment ---
    p.add_argument("--fov", type=float, default=50.0,
                   help="Camera horizontal FOV in degrees [35-60]")
    p.add_argument("--cam_height", type=float, default=2.5,
                   help="Camera height above ground (meters)")
    p.add_argument("--cam_tilt", type=float, default=-20.0,
                   help="Camera tilt angle in degrees (negative = tilting down)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Object scale factor (world units per MPM unit)")
    p.add_argument("--translate", type=str, default="0,0,0",
                   help="Object translation offset (x,y,z)")
    p.add_argument("--rotate", type=str, default="0,0,0",
                   help="Object rotation Euler angles in degrees (x,y,z)")

    # ---- Light ---
    p.add_argument("--light_dir", type=str, default="0.2,-0.8,0.6",
                   help="Sun light direction vector (x,y,z) for Blender rendering")

    # ---- Compositing ---
    p.add_argument("--use_shadow", type=int, default=1,
                   help="Enable shadow pass (1=yes, 0=no)")
    p.add_argument("--shadow_strength", type=float, default=0.45,
                   help="Shadow darkening strength [0,1]")
    p.add_argument("--shadow_blur", type=int, default=31,
                   help="Shadow blur kernel radius (pixels)")
    p.add_argument("--use_occlusion", type=int, default=0,
                   help="Enable depth-based occlusion (0 or 1, requires Blender depth pass)")

    # ---- Blender ---
    p.add_argument("--use_blender", action="store_true",
                   help="Use Blender for rendering particles + shadow catcher. "
                        "Requires Blender 3.6+ installed.")
    p.add_argument("--blender_exec", type=str, default="blender",
                   help="Path to Blender executable")
    p.add_argument("--particle_radius", type=float, default=0.03,
                   help="Particle sphere radius for Blender rendering")
    p.add_argument("--blender_samples", type=int, default=64,
                   help="Number of Cycles samples for Blender rendering")
    p.add_argument("--blender_engine", type=str, default="BLENDER_EEVEE",
                   choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"],
                   help="Blender render engine for particle rendering")
    p.add_argument("--particle_color", type=str, default="0.6,0.6,0.6",
                   help="Particle material color (r,g,b) for Blender rendering")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parse_vec3(s):
    """Parse 'x,y,z' string to float list."""
    return [float(v) for v in s.split(",")]


def estimate_camera(image_path, fov_deg=50.0, cam_height=2.5, cam_tilt_deg=-20.0):
    """
    Simple camera approximation.
    Returns a metadata dict compatible with the Blender script.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]

    tilt_rad = math.radians(cam_tilt_deg)  # negative = looking down
    # Distance from camera to origin on the ground
    if abs(tilt_rad) > 1e-4:
        dist = cam_height / abs(math.tan(tilt_rad))
    else:
        dist = cam_height * 5.0

    cam_location = [0.0, -dist, cam_height]
    # Rotation: tilt around X so camera looks down onto object
    # Blender convention: looking along -Y at rest; tilt down adds positive X rotation
    cam_rx = math.radians(90.0 + cam_tilt_deg)   # e.g. tilt=-20 → 70°
    cam_rotation_euler = [cam_rx, 0.0, 0.0]

    return {
        "fov_deg": fov_deg,
        "image_width": W,
        "image_height": H,
        "cam_location": cam_location,
        "cam_rotation_euler": cam_rotation_euler,
        "ground_plane_z": 0.0,
        "scale_m_per_unit": 1.0,
    }


def find_frame_files(mpm_dir, mode="particles"):
    """
    Return sorted list of per-frame files from an MPM output directory.
    Supports NPZ, PLY, and OBJ formats.
    """
    npz_files = sorted(_glob.glob(os.path.join(mpm_dir, "frame_*.npz")))
    ply_files = sorted(_glob.glob(os.path.join(mpm_dir, "frame_*.ply")))
    obj_files = sorted(_glob.glob(os.path.join(mpm_dir, "frame_*.obj")))

    if mode == "particles":
        if npz_files:
            return npz_files, "npz"
        if ply_files:
            return ply_files, "ply"
    elif mode == "mesh":
        if obj_files:
            return obj_files, "obj"
        if ply_files:
            return ply_files, "ply"

    # Fallback: anything found
    for files, fmt in [(npz_files, "npz"), (ply_files, "ply"), (obj_files, "obj")]:
        if files:
            return files, fmt

    return [], None


def load_particle_positions(frame_file, fmt):
    """Load (N,3) particle positions from a frame file."""
    if fmt == "npz":
        data = np.load(frame_file)
        if "pos" in data:
            return data["pos"].astype(np.float32)
        # fallback key names
        for key in ["positions", "x", "points"]:
            if key in data:
                return data[key].astype(np.float32)
        raise KeyError(f"No position key found in {frame_file}. Keys: {list(data.keys())}")
    elif fmt in ("ply", "obj"):
        try:
            import trimesh
            mesh = trimesh.load(frame_file, process=False)
            if hasattr(mesh, "vertices"):
                return np.array(mesh.vertices, dtype=np.float32)
            return np.array(mesh.vertices, dtype=np.float32)
        except ImportError:
            raise ImportError("trimesh is required to load PLY/OBJ particle files. "
                              "Install with: pip install trimesh")
    else:
        raise ValueError(f"Unsupported frame format: {fmt}")


# ---------------------------------------------------------------------------
# Python-based shadow generation (no Blender required)
# ---------------------------------------------------------------------------

def generate_shadow_from_rgba(rgba_path, out_path, shadow_blur=31,
                               ground_y_frac=0.75, shear_amount=0.3):
    """
    Generate an approximate drop shadow image from an RGBA foreground frame.

    Strategy:
    1. Extract alpha channel from RGBA image.
    2. Project it downward (shear + translate) to simulate ground shadow.
    3. Apply Gaussian blur for soft shadow.
    4. Save as grayscale shadow image (bright = more shadow).

    Args:
        rgba_path:    Path to RGBA foreground PNG.
        out_path:     Output shadow PNG path.
        shadow_blur:  Gaussian blur kernel size (odd integer).
        ground_y_frac: Fraction of image height where the ground is (0=top, 1=bottom).
        shear_amount: Horizontal shear for directional shadow.
    """
    img = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Create a blank shadow if the frame is missing
        cv2.imwrite(out_path, np.zeros((100, 100), dtype=np.uint8))
        return
    H, W = img.shape[:2]

    if img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
    else:
        # Derive alpha from RGB: luminance-based estimate
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        alpha = gray

    # Flatten (project) the alpha silhouette onto the ground.
    # We squish the alpha vertically: compress toward the ground line,
    # with a slight downward and horizontal offset (directional shadow).
    ground_y = int(H * ground_y_frac)
    shadow_scale_y = 0.35      # compress height to 35% → flat shadow on ground
    shadow_offset_y = 20       # shift down by N pixels
    shadow_offset_x = int(shear_amount * W * 0.1)  # slight directional shift

    # Create a shadow silhouette via affine transform
    # Scale vertically toward ground line, then shift
    M_scale = np.array([
        [1.0, 0.0, shadow_offset_x],
        [0.0, shadow_scale_y, ground_y * (1.0 - shadow_scale_y) + shadow_offset_y],
    ], dtype=np.float32)
    shadow = cv2.warpAffine(alpha, M_scale, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Blur for soft shadow
    k = shadow_blur if shadow_blur % 2 == 1 else shadow_blur + 1
    shadow = cv2.GaussianBlur(shadow, (k, k), 0)
    shadow = np.clip(shadow, 0, 1)

    cv2.imwrite(out_path, (shadow * 255).astype(np.uint8))


def generate_all_shadows_python(fg_rgba_dir, shadow_dir, shadow_blur=31):
    """Generate shadow images for all RGBA frames using Python approximation."""
    os.makedirs(shadow_dir, exist_ok=True)
    rgba_files = sorted(_glob.glob(os.path.join(fg_rgba_dir, "*.png")))
    if not rgba_files:
        print(f"[shadow] No RGBA frames found in {fg_rgba_dir}")
        return 0
    for f in rgba_files:
        fname = os.path.basename(f)
        out_path = os.path.join(shadow_dir, fname)
        generate_shadow_from_rgba(f, out_path, shadow_blur=shadow_blur)
    print(f"[shadow] Generated {len(rgba_files)} shadow frames → {shadow_dir}")
    return len(rgba_files)


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def composite_one_frame(bg, fg_rgba_path, shadow_path,
                        shadow_strength=0.45, use_shadow=True):
    """
    Composite one frame: bg + shadow + fg RGBA → final RGB image.

    Formula:
      bg_shadowed = bg * (1 - shadow_strength * shadow_mask)
      final = bg_shadowed * (1 - fg_alpha) + fg_rgb * fg_alpha

    Returns composited uint8 BGR image.
    """
    H, W = bg.shape[:2]

    # Load foreground RGBA
    fg_rgba = cv2.imread(fg_rgba_path, cv2.IMREAD_UNCHANGED)
    if fg_rgba is None:
        # Frame missing: return plain background
        return (bg * 255).astype(np.uint8)

    # Resize fg to match bg if needed
    if fg_rgba.shape[0] != H or fg_rgba.shape[1] != W:
        fg_rgba = cv2.resize(fg_rgba, (W, H), interpolation=cv2.INTER_LINEAR)

    if fg_rgba.shape[2] == 4:
        fg_bgr = fg_rgba[:, :, :3].astype(np.float32) / 255.0
        fg_alpha = fg_rgba[:, :, 3:].astype(np.float32) / 255.0  # [H,W,1]
    else:
        fg_bgr = fg_rgba.astype(np.float32) / 255.0
        fg_alpha = np.ones((H, W, 1), dtype=np.float32)

    # Apply shadow to background
    bg_f = bg.copy()
    if use_shadow and shadow_path and os.path.exists(shadow_path):
        shadow = cv2.imread(shadow_path, cv2.IMREAD_GRAYSCALE)
        if shadow is not None:
            if shadow.shape[0] != H or shadow.shape[1] != W:
                shadow = cv2.resize(shadow, (W, H), interpolation=cv2.INTER_LINEAR)
            shadow_f = shadow.astype(np.float32) / 255.0
            shadow_mask = shadow_f[:, :, np.newaxis]  # [H,W,1]
            bg_f = bg_f * (1.0 - shadow_strength * shadow_mask)

    # Alpha composite: fg over shadowed bg
    result = bg_f * (1.0 - fg_alpha) + fg_bgr * fg_alpha
    result = np.clip(result, 0.0, 1.0)
    return (result * 255).astype(np.uint8)


def composite_all_frames(bg_image_path, fg_rgba_dir, shadow_dir, out_dir,
                          shadow_strength=0.45, use_shadow=True):
    """Composite all frames and save to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    bg = cv2.imread(bg_image_path)
    if bg is None:
        raise FileNotFoundError(f"Background image not found: {bg_image_path}")
    bg = bg.astype(np.float32) / 255.0

    rgba_files = sorted(_glob.glob(os.path.join(fg_rgba_dir, "*.png")))
    if not rgba_files:
        raise RuntimeError(f"No RGBA frames found in {fg_rgba_dir}")

    count = 0
    for rgba_path in rgba_files:
        fname = os.path.basename(rgba_path)
        # Normalise filename to 4-digit zero-padded
        stem = os.path.splitext(fname)[0]
        try:
            idx = int(stem)
            out_name = f"{idx:04d}.png"
        except ValueError:
            out_name = fname

        shadow_path = os.path.join(shadow_dir, fname) if shadow_dir else None
        result = composite_one_frame(bg, rgba_path, shadow_path,
                                     shadow_strength=shadow_strength,
                                     use_shadow=use_shadow)
        cv2.imwrite(os.path.join(out_dir, out_name), result)
        count += 1

    print(f"[composite] {count} frames composited → {out_dir}")
    return count


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------

def encode_video(frames_dir, output_path, fps, width=None, height=None):
    """Encode a directory of PNG frames to MP4 using ffmpeg."""
    # Detect resolution from first frame if not given
    if width is None or height is None:
        frames = sorted(_glob.glob(os.path.join(frames_dir, "*.png")))
        if frames:
            img = cv2.imread(frames[0])
            if img is not None:
                height, width = img.shape[:2]
                width = width // 2 * 2
                height = height // 2 * 2

    size_arg = []
    if width and height:
        size_arg = ["-s", f"{width}x{height}"]

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        *size_arg,
        output_path,
    ]
    print(f"[encode] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[encode] ffmpeg stderr:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}")
    print(f"[encode] Video saved → {output_path}")


# ---------------------------------------------------------------------------
# Blender rendering (optional)
# ---------------------------------------------------------------------------

def run_blender_render(args, particles_dir, metadata_json, out_rgba, out_shadow,
                       out_depth=None):
    """
    Invoke scripts/render_mpm_composite.py via Blender.

    args: parsed argparse namespace
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "scripts", "render_mpm_composite.py")
    if not os.path.exists(script):
        raise FileNotFoundError(f"Blender script not found: {script}")

    cmd = [
        args.blender_exec, "--background",
        "--python-exit-code", "1",
        "--python", script,
        "--",
        "--image", args.image,
        "--particles_dir", particles_dir,
        "--camera_json", metadata_json,
        "--out_rgba", out_rgba,
        "--out_shadow", out_shadow,
        "--fps", str(args.fps),
        "--mode", args.mode,
        "--particle_radius", str(args.particle_radius),
        "--particle_color", args.particle_color,
        "--samples", str(args.blender_samples),
        "--render_engine", args.blender_engine,
        "--light_dir", args.light_dir,
    ]
    if out_depth:
        cmd += ["--out_depth", out_depth]
    if not args.use_shadow:
        cmd += ["--no_shadow"]

    blender_env = os.environ.copy()
    for k in [
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
    ]:
        blender_env.pop(k, None)
    blender_env["PYTHONNOUSERSITE"] = "1"

    print(f"[blender] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=blender_env)
    if result.returncode != 0:
        raise RuntimeError(f"Blender script failed with code {result.returncode}")


# ---------------------------------------------------------------------------
# Phymagic integration helpers
# ---------------------------------------------------------------------------

def run_simulate(args, export_rgba_dir, export_particles_dir):
    """
    Run simulate.py with --export_rgba_dir and --export_particles_npz flags.
    Requires --data_path and corresponding config JSON.
    """
    image_name = os.path.basename(args.data_path.rstrip("/"))
    data_root = os.path.dirname(args.data_path.rstrip("/"))
    if not data_root:
        data_root = "."

    cmd = [
        sys.executable, "simulate.py",
        "--config_path", args.config_path,
        "--data_path", data_root + "/",
        "--image_path", image_name,
        "--render_img",
        "--compile_video",
        "--export_rgba_dir", export_rgba_dir,
        "--export_particles_npz", export_particles_dir,
    ]
    if args.image:
        cmd += ["--bg_image", args.image]

    print(f"[simulate] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"simulate.py failed with code {result.returncode}")


def collect_gaussian_rgba_frames(data_path, out_rgba_dir):
    """
    Copy existing RGBA frames from data_path/output/fg_rgba/ (if exported)
    or try to locate rendered frames from simulate.py output.

    Returns number of frames found, or 0 if none.
    """
    os.makedirs(out_rgba_dir, exist_ok=True)

    # First check: data_path/output/fg_rgba/ (from --export_rgba_dir)
    candidate_dirs = [
        os.path.join(data_path, "output", "fg_rgba"),
        os.path.join(data_path, "output_rgba"),
    ]
    for src_dir in candidate_dirs:
        files = sorted(_glob.glob(os.path.join(src_dir, "*.png")))
        if files:
            for f in files:
                shutil.copy2(f, os.path.join(out_rgba_dir, os.path.basename(f)))
            print(f"[rgba] Copied {len(files)} RGBA frames from {src_dir}")
            return len(files)

    print(f"[rgba] No RGBA frames found in {data_path}. "
          f"Run with --run_simulate to generate them.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Validate inputs ----
    if args.image is None and args.data_path is None:
        print("ERROR: Provide --image (background plate) and either "
              "--mpm_dir (particles) or --data_path (phymagic data dir).")
        sys.exit(1)

    # If only data_path given, infer background image
    if args.image is None and args.data_path is not None:
        candidate = os.path.join(args.data_path, "origin.png")
        if os.path.exists(candidate):
            args.image = candidate
            print(f"[info] Using background image: {args.image}")
        else:
            print(f"WARNING: --image not given and {candidate} not found. "
                  "Compositing will be skipped (no background).")

    # ---- Output directories ----
    out_dir = args.out.rstrip("/")
    fg_rgba_dir  = os.path.join(out_dir, "fg_rgba")
    fg_depth_dir = os.path.join(out_dir, "fg_depth")
    shadow_dir   = os.path.join(out_dir, "shadow")
    final_dir    = os.path.join(out_dir, "final_frames")
    particles_export_dir = os.path.join(out_dir, "_particles_export")

    for d in [out_dir, fg_rgba_dir, fg_depth_dir, shadow_dir, final_dir]:
        os.makedirs(d, exist_ok=True)

    # ---- Camera metadata ----
    metadata_json = os.path.join(out_dir, "metadata.json")
    if args.image and os.path.exists(args.image):
        metadata = estimate_camera(
            args.image,
            fov_deg=args.fov,
            cam_height=args.cam_height,
            cam_tilt_deg=args.cam_tilt,
        )
    else:
        metadata = {
            "fov_deg": args.fov,
            "image_width": 1920,
            "image_height": 1080,
            "cam_location": [0.0, -6.0, 2.5],
            "cam_rotation_euler": [math.radians(70), 0.0, 0.0],
            "ground_plane_z": 0.0,
            "scale_m_per_unit": 1.0,
        }

    metadata["scale"] = args.scale
    metadata["translate"] = parse_vec3(args.translate)
    metadata["rotate_deg"] = parse_vec3(args.rotate)
    metadata["light_dir"] = parse_vec3(args.light_dir)
    metadata["fps"] = args.fps
    metadata["shadow_strength"] = args.shadow_strength

    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[metadata] Saved → {metadata_json}")

    # ===========================================================
    # Stage 1: Obtain foreground RGBA frames
    # ===========================================================
    n_rgba_frames = 0

    if args.mode in ("particles", "mesh") and args.mpm_dir:
        # ---- External particle / mesh mode ----
        frame_files, fmt = find_frame_files(args.mpm_dir, mode=args.mode)
        if not frame_files:
            print(f"ERROR: No frame files found in {args.mpm_dir} for mode '{args.mode}'")
            sys.exit(1)
        print(f"[input] Found {len(frame_files)} {fmt.upper()} frames in {args.mpm_dir}")

        if args.use_blender:
            # Blender renders RGBA + shadow from particle files
            print("[stage1] Running Blender to render particles + shadow...")
            run_blender_render(
                args,
                particles_dir=args.mpm_dir,
                metadata_json=metadata_json,
                out_rgba=fg_rgba_dir,
                out_shadow=shadow_dir if args.use_shadow else None,
                out_depth=fg_depth_dir if args.use_occlusion else None,
            )
            n_rgba_frames = len(sorted(_glob.glob(os.path.join(fg_rgba_dir, "*.png"))))
        else:
            # Python-based 2D projection rendering of particles
            print("[stage1] Projecting particles to 2D (Python fallback, no Blender)...")
            n_rgba_frames = render_particles_python(
                frame_files, fmt, metadata, fg_rgba_dir, args
            )

    elif args.mode == "gaussian" or args.data_path:
        # ---- Phymagic Gaussian mode ----
        if args.run_phymagic_pipeline:
            # Run the full phymagic pipeline first
            image_name = os.path.basename(args.data_path.rstrip("/"))
            phymagic_cmd = [
                sys.executable, "phymagic_pipeline.py",
                "--data_path", args.data_path,
                "--instruction", args.instruction or "The object falls and deforms",
                "--bg_composite",
            ]
            print(f"[phymagic] {' '.join(phymagic_cmd)}")
            subprocess.run(phymagic_cmd, check=True)

        if args.run_simulate:
            # Run simulate.py with RGBA + particle export
            print("[stage1] Running simulate.py with RGBA export...")
            run_simulate(args, fg_rgba_dir, particles_export_dir)
            n_rgba_frames = len(sorted(_glob.glob(os.path.join(fg_rgba_dir, "*.png"))))
        else:
            # Try to collect pre-existing RGBA frames
            if args.data_path:
                n_rgba_frames = collect_gaussian_rgba_frames(args.data_path, fg_rgba_dir)
            if n_rgba_frames == 0:
                print("[stage1] No RGBA frames found. "
                      "Add --run_simulate to generate them, or provide --mpm_dir.")
                if not os.path.exists(os.path.join(out_dir, "final.mp4")):
                    sys.exit(1)
    else:
        print("ERROR: Specify --mpm_dir (particles/mesh mode) or "
              "--data_path (gaussian/phymagic mode).")
        sys.exit(1)

    # ===========================================================
    # Stage 2: Generate shadow pass
    # ===========================================================
    if args.use_shadow:
        n_shadow = len(sorted(_glob.glob(os.path.join(shadow_dir, "*.png"))))
        if n_shadow == 0:
            # Blender didn't produce shadow (or wasn't used) → Python approximation
            print("[stage2] Generating shadow frames via Python approximation...")
            generate_all_shadows_python(fg_rgba_dir, shadow_dir,
                                         shadow_blur=args.shadow_blur)
        else:
            print(f"[stage2] Using {n_shadow} existing shadow frames from {shadow_dir}")
    else:
        print("[stage2] Shadow disabled (--use_shadow 0)")

    # ===========================================================
    # Stage 3: Composite
    # ===========================================================
    if args.image and os.path.exists(args.image):
        print(f"[stage3] Compositing {n_rgba_frames} frames onto {args.image}...")
        n_final = composite_all_frames(
            bg_image_path=args.image,
            fg_rgba_dir=fg_rgba_dir,
            shadow_dir=shadow_dir if args.use_shadow else None,
            out_dir=final_dir,
            shadow_strength=args.shadow_strength,
            use_shadow=bool(args.use_shadow),
        )
        print(f"[stage3] {n_final} composited frames → {final_dir}")
    else:
        # No background image: just copy RGBA frames as final
        print("[stage3] No background image — copying RGBA frames as final output...")
        for f in sorted(_glob.glob(os.path.join(fg_rgba_dir, "*.png"))):
            shutil.copy2(f, os.path.join(final_dir, os.path.basename(f)))

    # ===========================================================
    # Stage 4: Encode video
    # ===========================================================
    final_video = os.path.join(out_dir, "final.mp4")
    n_final_frames = len(sorted(_glob.glob(os.path.join(final_dir, "*.png"))))
    if n_final_frames == 0:
        print("[encode] WARNING: No frames in final_frames/. Skipping video encoding.")
    else:
        print(f"[encode] Encoding {n_final_frames} frames at {args.fps} fps...")
        encode_video(final_dir, final_video, fps=args.fps)

    # ===========================================================
    # Done
    # ===========================================================
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Background image  : {args.image}")
    print(f"  RGBA frames       : {fg_rgba_dir}/")
    print(f"  Shadow frames     : {shadow_dir}/")
    print(f"  Final frames      : {final_dir}/")
    print(f"  Final video       : {final_video}")
    print(f"  Metadata          : {metadata_json}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Python-based particle 2D rendering (no Blender)
# ---------------------------------------------------------------------------

def render_particles_python(frame_files, fmt, metadata, out_rgba_dir, args):
    """
    Render particles as colored circles projected with a pinhole camera.
    This is the no-Blender fallback for particle mode.

    Returns number of frames rendered.
    """
    os.makedirs(out_rgba_dir, exist_ok=True)

    W = metadata["image_width"]
    H = metadata["image_height"]
    fov_h = metadata["fov_deg"]  # horizontal FOV in degrees
    fov_rad = math.radians(fov_h)
    fx = (W / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0

    # Camera extrinsics (camera-to-world)
    cam_loc = np.array(metadata["cam_location"], dtype=np.float64)
    cam_rot_euler = np.array(metadata["cam_rotation_euler"], dtype=np.float64)

    # Build rotation matrix from Euler angles (XYZ order)
    def euler_to_matrix(rx, ry, rz):
        Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
        Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
        Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
        return Rz @ Ry @ Rx

    R_cw = euler_to_matrix(*cam_rot_euler)  # camera-to-world rotation
    R_wc = R_cw.T  # world-to-camera

    # Object transform
    scale = args.scale
    translate = np.array(parse_vec3(args.translate), dtype=np.float64)
    rot_deg = parse_vec3(args.rotate)
    R_obj = euler_to_matrix(*[math.radians(d) for d in rot_deg])

    radius_px = max(2, int(args.particle_radius * fx))  # approx radius in pixels
    color_rgb = [int(c * 255) for c in parse_vec3(args.particle_color)]
    color_bgr = color_rgb[::-1]  # cv2 uses BGR

    print(f"[render2D] Rendering {len(frame_files)} frames as 2D projected particles...")

    for i, fpath in enumerate(frame_files):
        positions = load_particle_positions(fpath, fmt)  # (N, 3)

        # Apply object transforms: scale + rotate + translate
        positions = (positions * scale) @ R_obj.T + translate

        # World to camera
        pts_cam = (positions - cam_loc) @ R_wc.T  # (N, 3) in camera space

        # Filter points in front of camera (positive Z in camera space)
        # Blender uses -Z as forward, but let's use +Z for simplicity here
        # In the convention set up: camera looks along +Y in world → camera -Z
        # We'll use standard pin-hole: z_cam > 0 means in front
        in_front = pts_cam[:, 2] < 0  # looking down -Z
        z_cam = -pts_cam[:, 2]

        # Project to image
        valid = in_front & (z_cam > 0.01)
        if not valid.any():
            # Write blank frame
            blank = np.zeros((H, W, 4), dtype=np.uint8)
            cv2.imwrite(os.path.join(out_rgba_dir, f"{i:04d}.png"), blank)
            continue

        pts_v = pts_cam[valid]
        z_v = z_cam[valid]
        u = (fx * pts_v[:, 0] / z_v + cx).astype(np.float32)
        v = (fy * pts_v[:, 1] / z_v + cy).astype(np.float32)

        # Sort back-to-front for painter's algorithm
        order = np.argsort(-z_v)
        u = u[order]
        v_pix = v[order]
        z_sorted = z_v[order]

        img = np.zeros((H, W, 4), dtype=np.uint8)
        for j in range(len(u)):
            xi = int(round(u[j]))
            yi = int(round(v_pix[j]))
            r = max(1, int(radius_px / z_sorted[j] * z_sorted[0]
                          if z_sorted[0] > 0 else radius_px))
            r = min(r, radius_px * 3)
            if 0 <= xi < W and 0 <= yi < H:
                cv2.circle(img, (xi, yi), r, (*color_bgr, 255), -1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(out_rgba_dir, f"{i:04d}.png"), img)

    print(f"[render2D] {len(frame_files)} frames → {out_rgba_dir}")
    return len(frame_files)


if __name__ == "__main__":
    main()
