import os
import argparse
import json
import numpy as np
import trimesh as tm
import torch
import taichi as ti
from tqdm import tqdm
from PIL import Image
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.system_utils import mkdir_p, write_particles
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from simulator import Simulator
import subprocess
import torchvision
import sys
import time
import json

# 时间统计类
class Timer:
    """简单的时间统计类"""
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0.0
        
    def start(self):
        """开始计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        
    def stop(self):
        """停止计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        
    def get_elapsed_time(self):
        """获取经过的时间"""
        return self.elapsed_time
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class TimeProfiler:
    """时间分析器，用于管理多个计时器"""
    def __init__(self):
        self.timers = {}
        self.total_times = {}
        
    def add_timer(self, name):
        """添加一个计时器"""
        if name not in self.timers:
            self.timers[name] = Timer(name)
            self.total_times[name] = 0.0
            
    def start_timer(self, name):
        """开始指定名称的计时器"""
        if name not in self.timers:
            self.add_timer(name)
        self.timers[name].start()
        
    def stop_timer(self, name):
        """停止指定名称的计时器"""
        if name in self.timers:
            self.timers[name].stop()
            self.total_times[name] += self.timers[name].get_elapsed_time()
            
    def get_total_time(self, name):
        """获取指定计时器的总时间"""
        return self.total_times.get(name, 0.0)
        
    def get_all_times(self):
        """获取所有计时器的时间"""
        return self.total_times.copy()
        
    def print_summary(self):
        """打印时间统计摘要"""
        print("\n" + "="*50)
        print("时间统计摘要")
        print("="*50)
        total_overall = sum(self.total_times.values())
        for name, time_val in self.total_times.items():
            percentage = (time_val / total_overall * 100) if total_overall > 0 else 0
            print(f"{name:<30}: {time_val:.4f}s ({percentage:.2f}%)")
        print(f"{'总计':<30}: {total_overall:.4f}s (100.00%)")
        print("="*50)
        
    def save_to_json(self, filepath):
        """保存时间统计到JSON文件"""
        summary = {
            "timers": self.total_times,
            "total_time": sum(self.total_times.values()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"时间统计已保存到: {filepath}")

# 创建全局时间分析器实例
profiler = TimeProfiler()

sys.path.append("gaussian-splatting")

from argparse import ArgumentParser, Namespace
from utils.load_config import load_config
import math
import cv2
import json
import imageio
from PIL import Image
from scipy.spatial.transform import Rotation as R


# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov, geom_transform_points


# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from utils.rotate_utils import *


wp.init()
wp.config.verify_cuda = True

# ti.init(arch=ti.cuda)
# ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.4)
ti.init(arch=ti.cpu)

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def _compute_bounds_from_positions(pos_tensor):
    """Compute AABB bounds from (N,3) positions in MPM space."""
    mn = torch.min(pos_tensor, dim=0).values
    mx = torch.max(pos_tensor, dim=0).values
    center = (mn + mx) / 2.0
    half = (mx - mn) / 2.0
    # Robust bottom to ignore outliers
    z_vals = pos_tensor[:, 2]
    if hasattr(torch, "quantile"):
        zmin_majority = torch.quantile(z_vals, 0.05)
    else:
        z_sorted, _ = torch.sort(z_vals)
        idx = int(0.05 * (z_sorted.numel() - 1))
        zmin_majority = z_sorted[idx]
    return {
        "xmin": float(mn[0].item()),
        "ymin": float(mn[1].item()),
        "zmin": float(mn[2].item()),
        "zmin_majority": float(zmin_majority.item()),
        "xmax": float(mx[0].item()),
        "ymax": float(mx[1].item()),
        "zmax": float(mx[2].item()),
        "cx": float(center[0].item()),
        "cy": float(center[1].item()),
        "cz": float(center[2].item()),
        "sx": float(half[0].item()),
        "sy": float(half[1].item()),
        "sz": float(half[2].item()),
    }


def _pca_align_to_ground(pos_tensor):
    """Return rotation matrix (3x3) that aligns the smallest PCA axis to +Z."""
    center = pos_tensor.mean(dim=0, keepdim=True)
    centered = pos_tensor - center
    cov = centered.t().mm(centered) / max(centered.shape[0], 1)
    # eigenvalues in ascending order
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # order: largest -> X, middle -> Y, smallest -> Z
    order = torch.tensor([2, 1, 0], device=pos_tensor.device)
    R = eigvecs[:, order]
    # ensure right-handed
    if torch.det(R) < 0:
        R[:, 2] *= -1.0
    return R


def _pca_align_axes(pos_tensor, x_sign=None, y_sign=None, z_sign=None):
    """Align largest axis to X, smallest to Z, with optional sign constraints."""
    center = pos_tensor.mean(dim=0, keepdim=True)
    centered = pos_tensor - center
    cov = centered.t().mm(centered) / max(centered.shape[0], 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
    order = torch.tensor([2, 1, 0], device=pos_tensor.device)
    R = eigvecs[:, order]

    x_dir = torch.tensor([1.0, 0.0, 0.0], device=pos_tensor.device)
    z_dir = torch.tensor([0.0, 0.0, 1.0], device=pos_tensor.device)
    if x_sign is not None:
        x_dir = x_dir * float(x_sign)
        if torch.dot(R[:, 0], x_dir) < 0:
            R[:, 0] *= -1.0
    if y_sign is not None:
        y_dir = torch.tensor([0.0, 1.0, 0.0], device=pos_tensor.device) * float(y_sign)
        if torch.dot(R[:, 1], y_dir) < 0:
            R[:, 1] *= -1.0
    if z_sign is not None:
        z_dir = z_dir * float(z_sign)
        if torch.dot(R[:, 2], z_dir) < 0:
            R[:, 2] *= -1.0

    if torch.det(R) < 0:
        if y_sign is None:
            R[:, 1] *= -1.0
        elif x_sign is None:
            R[:, 0] *= -1.0
        elif z_sign is None:
            R[:, 2] *= -1.0
        else:
            # All signs specified; fallback to flipping Y to keep right-handed.
            R[:, 1] *= -1.0
    return R


def _project_points_to_image(points, camera):
    """Project 3D points to 2D pixel coordinates using the camera."""
    ndc = geom_transform_points(points, camera.full_proj_transform)
    x = (ndc[:, 0] + 1.0) * 0.5 * float(camera.image_width)
    y = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * float(camera.image_height)
    return torch.stack([x, y], dim=1)

def _apply_object_bounds_to_bcs(bc_params, bounds):
    """Fill BC point/size using object bounds when requested."""
    if not bc_params:
        return
    cx, cy, cz = bounds["cx"], bounds["cy"], bounds["cz"]
    sx, sy, sz = bounds["sx"], bounds["sy"], bounds["sz"]
    z_floor = bounds.get("zmin_majority", bounds["zmin"])
    zmin = bounds["zmin"]
    zmax = bounds["zmax"]
    hz = zmax - zmin
    spin_radius = max(sx, sy) + float(bounds.get("spin_radius_pad", 0.1))
    for bc in bc_params:
        if not isinstance(bc, dict):
            continue
        if bc.get("type") == "translate_object_box":
            bc["type"] = "enforce_particle_translation"
            bc["point"] = [cx, cy, cz]
            bc["size"] = [sx, sy, sz]
        if bc.get("use_object_bounds"):
            bc["point"] = [cx, cy, cz]
            bc["size"] = [sx, sy, sz]
        if bc.get("use_object_top"):
            # Push/pull the top portion of the object
            top_fraction = float(bc.get("top_fraction", 0.1))
            top_thickness = float(bc.get("top_thickness", 0.1))
            top_z_cen = zmax - hz * top_fraction
            top_z_sz = hz * top_thickness
            bc["point"] = [cx, cy, top_z_cen]
            bc["size"] = [sx, sy, top_z_sz]
        if bc.get("use_object_bottom"):
            # Push/pull the bottom portion of the object
            bottom_fraction = float(bc.get("bottom_fraction", 0.1))
            bottom_thickness = float(bc.get("bottom_thickness", 0.1))
            bottom_z_cen = zmin + hz * bottom_fraction
            bottom_z_sz = hz * bottom_thickness
            bc["point"] = [cx, cy, bottom_z_cen]
            bc["size"] = [sx, sy, bottom_z_sz]
        if bc.get("use_object_spin"):
            # Spin around Z to encourage radial spread
            bc["point"] = [cx, cy, bc.get("spin_center_z", 0.0)]
            bc["normal"] = [0.0, 0.0, 1.0]
            half_height = float(bc.get("spin_half_height", max(hz * 0.6, 0.1)))
            radius = float(bc.get("spin_radius", spin_radius))
            bc["half_height_and_radius"] = [half_height, radius]
        if bc.get("use_object_floor"):
            floor_offset = float(bc.get("floor_offset", 0.0))
            point = bc.get("point")
            if isinstance(point, list) and len(point) == 3:
                point[2] = z_floor + floor_offset
                bc["point"] = point
            else:
                bc["point"] = [0.0, 0.0, z_floor + floor_offset]


def _glb_to_gaussians(gaussians, glb_path, n_samples=50000):
    """Convert a GLB mesh to Gaussian splatting representation in-place."""
    import glob as _glob

    loaded = tm.load(glb_path)
    if isinstance(loaded, tm.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, tm.Trimesh)]
        mesh = tm.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = loaded

    pts, face_idx = tm.sample.sample_surface(mesh, n_samples)
    pts = pts.astype(np.float32)

    # GLTF/GLB uses Y-up; MPM uses Z-up.  Apply -90° around X to align:
    #   new_x = x,  new_y = -z,  new_z = y
    pts = pts[:, [0, 2, 1]] * np.array([1, -1, 1], dtype=np.float32)

    N = len(pts)

    # Extract per-sample colors (average of face vertex colors)
    try:
        vc = mesh.visual.to_color().vertex_colors  # (V, 4) uint8
        face_verts = mesh.faces[face_idx]           # (N, 3)
        colors = vc[face_verts, :3].mean(axis=1).astype(np.float32) / 255.0
    except Exception:
        colors = np.full((N, 3), 0.5, dtype=np.float32)

    # Colors [0,1] → SH DC coefficients  (inverse of: color = C0*dc + 0.5)
    C0 = 0.28209479177387814
    f_dc = ((colors - 0.5) / C0).astype(np.float32)  # (N, 3)

    xyz          = torch.tensor(pts,  dtype=torch.float, device="cuda")
    features_dc  = torch.tensor(f_dc, dtype=torch.float, device="cuda").unsqueeze(1)  # (N,1,3)
    features_rest = torch.zeros((N, 0, 3), dtype=torch.float, device="cuda")
    opacity      = torch.full((N, 1), 4.0,  dtype=torch.float, device="cuda")
    scaling      = torch.full((N, 3), -5.0, dtype=torch.float, device="cuda")
    rotation     = torch.zeros((N, 4),      dtype=torch.float, device="cuda")
    rotation[:, 0] = 1.0  # identity quaternion [w, x, y, z]

    gaussians._xyz           = torch.nn.Parameter(xyz.requires_grad_(True))
    gaussians._features_dc   = torch.nn.Parameter(features_dc.requires_grad_(True))
    gaussians._features_rest = torch.nn.Parameter(features_rest.requires_grad_(True))
    gaussians._opacity       = torch.nn.Parameter(opacity.requires_grad_(True))
    gaussians._scaling       = torch.nn.Parameter(scaling.requires_grad_(True))
    gaussians._rotation      = torch.nn.Parameter(rotation.requires_grad_(True))
    gaussians.active_sh_degree = 0
    print(f"[load_checkpoint] GLB → {N} Gaussians from {glb_path}")


def load_checkpoint(model_path, sh_degree=0, iteration=-1):
    import glob as _glob
    gaussians = GaussianModel(sh_degree)
    ply_path = os.path.join(model_path, "static_0.ply")
    if os.path.exists(ply_path):
        gaussians.load_ply(ply_path)
    else:
        glb_files = sorted(_glob.glob(os.path.join(model_path, "*.glb")))
        if not glb_files:
            raise FileNotFoundError(
                f"No static_0.ply or *.glb found in {model_path}"
            )
        _glb_to_gaussians(gaussians, glb_files[0])
    return gaussians


def load_pcd_file(model_path):
    path = os.path.join(model_path + "sample.glb")
    pcd = tm.load_mesh(path) # mesh_file_path
    np_pcd = np.array(pcd.vertices)
    vol = torch.from_numpy(np_pcd).to('cuda',dtype=torch.float32).contiguous()    
    return vol


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/")
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--output_file", type=str, default="output.mp4")
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--from_origin", action="store_true", help="have no 3D input, generated from input image.")
    parser.add_argument("--profile_time", action="store_true", help="启用时间统计功能")
    parser.add_argument("--bg_image", type=str, default=None,
                        help="Background image to composite simulated object onto. "
                             "Composited frames are saved to <output_path>_bg/ and "
                             "compiled to output_bg.mp4.")
    parser.add_argument("--export_particles_npz", type=str, default=None,
                        help="Directory to save per-frame world-space particle positions "
                             "as NPZ files (frame_0000.npz with key 'pos'[N,3]).")
    parser.add_argument("--export_rgba_dir", type=str, default=None,
                        help="Directory to save per-frame RGBA foreground images with "
                             "alpha channel (requires black+white dual render).")

    args = parser.parse_args()
    
    # 开始总体计时
    if args.profile_time:
        profiler.start_timer("总体执行时间")

    # io = Namespace(**config.io)
    # gs = Namespace(**config.gs)
    # preprocessing_params = Namespace(**config.preprocessing_params)
    # data_path = args.data_path
    image_path = os.path.join(args.data_path, args.image_path)
    image_file = os.path.join(image_path, "origin.png")
    output_path = os.path.join(image_path, args.output_path)
    config_data = os.path.join(args.config_path, f"{str(args.image_path)}_config.json")
    config = Namespace(**load_config(config_data))


    if not os.path.exists(image_path):
        AssertionError("Model path does not exist!")
    # if not os.path.exists(args.config):
    #     AssertionError("Scene config does not exist!")
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        # Remove stale frames from previous runs to avoid including them in video
        for _f in os.listdir(output_path):
            if _f.endswith(".png"):
                os.remove(os.path.join(output_path, _f))

    # load scene config
    # print(decode_param_json(args.config))
    print("Loading scene config...")
    if args.profile_time:
        profiler.start_timer("场景配置加载")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(config_data)
    if args.profile_time:
        profiler.stop_timer("场景配置加载")

    # Load an image
    # print(os.getcwd())
    if args.from_origin:
        os.environ['ATTN_BACKEND'] = 'xformers'
        os.environ['SPCONV_ALGO'] = 'native'
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import render_utils, postprocessing_utils
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
        if args.profile_time:
            profiler.start_timer("TRELLIS图像处理")
        image = Image.open(image_file)

        outputs = pipeline.run(
            image,
            seed=1,
        )

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export(os.path.join(image_path + "/sample.glb"))

        # 1. load gaussians .ply file from GLB file
        print("Loading gaussians...")
        gaussian = outputs['gaussian'][0]
        mesh = outputs['mesh'][0]
        outputs['gaussian'][0].save_ply(os.path.join(image_path + "/static_0.ply"))
        if args.profile_time:
            profiler.stop_timer("TRELLIS图像处理")


    # 2. transfer gaussian into point cloud pcd
    if args.profile_time:
        profiler.start_timer("高斯模型加载")
    gaussians = load_checkpoint(image_path)
    if args.profile_time:
        profiler.stop_timer("高斯模型加载")

    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    # Background compositing setup
    bg_tensor = None
    bg_output_path = None
    if args.bg_image:
        _bg_raw = cv2.imread(args.bg_image)
        if _bg_raw is None:
            raise FileNotFoundError(f"Background image not found: {args.bg_image}")
        _bg_raw = cv2.cvtColor(_bg_raw, cv2.COLOR_BGR2RGB)
        bg_tensor = torch.from_numpy(_bg_raw.astype(np.float32) / 255.0).cuda()  # [H,W,3]
        bg_output_path = output_path.rstrip("/") + "_bg"
        os.makedirs(bg_output_path, exist_ok=True)
        # Remove stale composited frames from previous runs
        for _f in os.listdir(bg_output_path):
            if _f.endswith(".png"):
                os.remove(os.path.join(bg_output_path, _f))

    # 3. init the scene
    print("Initializing scene and pre-processing...")
    if args.profile_time:
        profiler.start_timer("场景初始化")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]


    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]


    ## debug
    print("init_pos after opacity filtering:", init_pos.shape)
    if init_pos.shape[0] == 0:
        raise ValueError("No points remaining after opacity filtering. Adjust the opacity threshold.")

    x_min, x_max, x_mean = torch.min(init_pos[:, 0]), torch.max(init_pos[:, 0]), torch.mean(init_pos[:, 0])
    y_min, y_max, y_mean = torch.min(init_pos[:, 1]), torch.max(init_pos[:, 1]), torch.mean(init_pos[:, 1])
    z_min, z_max, z_mean = torch.min(init_pos[:, 2]), torch.max(init_pos[:, 2]), torch.mean(init_pos[:, 2])

    print(f"Point cloud bounds: x=[{x_min}, {x_max}, {x_mean}], y=[{y_min}, {y_max}, {x_mean}], z=[{z_min}, {z_max}, {x_mean}]")

    # rotate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(
            init_pos,
            "./log/init_particles.ply",
        )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)
    
    ## debug
    print("rotated_pos after rotation:", rotated_pos.shape)
    if rotated_pos.shape[0] == 0:
        raise ValueError("No points remaining after rotation. Check rotation logic.")

    # Optional: auto-orient so the smallest PCA axis aligns with +Z (flat on ground)
    if preprocessing_params.get("auto_orient", False):
        mode = preprocessing_params.get("auto_orient_mode", "pca_z")
        if mode == "pca_full":
            x_sign = preprocessing_params.get("auto_orient_x_sign", None)
            y_sign = preprocessing_params.get("auto_orient_y_sign", None)
            z_sign = preprocessing_params.get("auto_orient_z_sign", None)
            R_auto = _pca_align_axes(rotated_pos, x_sign=x_sign, y_sign=y_sign, z_sign=z_sign)
        else:
            R_auto = _pca_align_to_ground(rotated_pos)
        rotation_matrices.append(R_auto)
        rotated_pos = apply_rotation(rotated_pos, R_auto)
        print(f"Applied auto-orient rotation (mode={mode}).")

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        ## debug
        if mask.sum().item() == 0:
            raise ValueError("No points within the simulation area. Adjust the boundary.")
        
        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]
        
        rotated_pos = rotated_pos[mask, :]

        ## debug
        print("rotated_pos after boundary filtering:", rotated_pos.shape)

        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    transformed_pos = shift2center111(transformed_pos)
    

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )
    
    if args.profile_time:
        profiler.stop_timer("场景初始化")

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    print("gs_num", gs_num)
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        if args.profile_time:
            profiler.start_timer("粒子填充")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)
        if args.profile_time:
            profiler.stop_timer("粒子填充")

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    obj_bounds = _compute_bounds_from_positions(mpm_init_pos)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    if args.profile_time:
        profiler.start_timer("MPM求解器初始化")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)


    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    _apply_object_bounds_to_bcs(bc_params, obj_bounds)
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    mpm_solver.finalize_mu_lam()
    if args.profile_time:
        profiler.stop_timer("MPM求解器初始化")

    # camera setting
    if args.profile_time:
        profiler.start_timer("相机设置")
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    if preprocessing_params.get("keep_sim_rotation", False):
        viewpoint_center_worldspace = (
            mpm_space_viewpoint_center.clone().detach().cpu().numpy().squeeze(0)
        )
        vertical = (
            mpm_space_vertical_upward_axis.clone().detach().cpu().numpy().squeeze(0)
        )
        vertical, h1, h2 = generate_local_coord(vertical)
        observant_coordinates = np.column_stack((h1, h2, vertical))
    else:
        (
            viewpoint_center_worldspace,
            observant_coordinates,
        ) = get_center_view_worldspace_and_observant_coordinate(
            mpm_space_viewpoint_center,
            mpm_space_vertical_upward_axis,
            rotation_matrices,
            scale_origin,
            original_mean_pos,
        )
    if args.profile_time:
        profiler.stop_timer("相机设置")

    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None

    # 逐帧渲染
    if args.profile_time:
        profiler.start_timer("仿真计算")
    for frame in tqdm(range(frame_num)):
        if args.profile_time:
            profiler.start_timer("相机设置")
        current_camera = get_camera_view(
            model_path=image_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],   # 相机初始方位
            init_elevation=camera_params["init_elevation"], # ....仰角
            init_radius=camera_params["init_radius"],       # ....距离
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],               # 相机增量值，更新相机位置
            delta_e=camera_params["delta_e"],               # 相机增量值，更新相机位置
            delta_r=camera_params["delta_r"],               # 相机增量值，更新相机位置
        )
        if args.profile_time:
            profiler.stop_timer("相机设置")

        # 返回resterize，将3D场景转回2D图像
        if args.profile_time:
            profiler.start_timer("光栅化初始化")
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )
        if args.profile_time:
            profiler.stop_timer("光栅化初始化")

        if args.profile_time:
            profiler.start_timer("MPM计算")
        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
        if args.profile_time:
            profiler.stop_timer("MPM计算")

        if args.output_ply or args.output_h5:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )

        if args.render_img:
            if args.profile_time:
                profiler.start_timer("渲染")
            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)

            if not preprocessing_params.get("keep_sim_rotation", False):
                pos = apply_inverse_rotations(
                    undotransform2origin(
                        undoshift2center111(pos), scale_origin, original_mean_pos
                    ),
                    rotation_matrices,
                )
                cov3D = cov3D / (scale_origin * scale_origin)
                cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            # Export world-space particle positions as NPZ (sim particles only)
            if args.export_particles_npz:
                os.makedirs(args.export_particles_npz, exist_ok=True)
                np.savez_compressed(
                    os.path.join(args.export_particles_npz, f"frame_{frame:04d}.npz"),
                    pos=pos.detach().cpu().numpy(),
                )

            opacity = opacity_render
            shs = shs_render
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            _raster_out = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )
            if isinstance(_raster_out, tuple):
                rendering = _raster_out[0]
                raddi = _raster_out[-1] if len(_raster_out) >= 2 else None
            else:
                rendering = _raster_out
                raddi = None
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            local_h, local_w = cv2_img.shape[0], cv2_img.shape[1]
            if height is None or width is None:
                height = local_h // 2 * 2
                width = local_w // 2 * 2
            if preprocessing_params.get("debug_draw_bbox", False):
                mn = torch.min(pos, dim=0).values
                mx = torch.max(pos, dim=0).values
                corners = torch.stack(
                    [
                        torch.stack([mn[0], mn[1], mn[2]]),
                        torch.stack([mn[0], mn[1], mx[2]]),
                        torch.stack([mn[0], mx[1], mn[2]]),
                        torch.stack([mn[0], mx[1], mx[2]]),
                        torch.stack([mx[0], mn[1], mn[2]]),
                        torch.stack([mx[0], mn[1], mx[2]]),
                        torch.stack([mx[0], mx[1], mn[2]]),
                        torch.stack([mx[0], mx[1], mx[2]]),
                    ],
                    dim=0,
                )
                uv = _project_points_to_image(corners, current_camera)
                uv = uv.detach().cpu().numpy()
                x0 = int(max(0, min(uv[:, 0])))
                x1 = int(min(local_w - 1, max(uv[:, 0])))
                y0 = int(max(0, min(uv[:, 1])))
                y1 = int(min(local_h - 1, max(uv[:, 1])))
                cv2.rectangle(cv2_img, (x0, y0), (x1, y1), (255, 0, 0), 2)
            assert output_path is not None
            cv2.imwrite(
                os.path.join(output_path, f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )

            # Background compositing / RGBA export: render with black bg, derive alpha
            _needs_black_render = (
                (bg_tensor is not None and bg_output_path is not None)
                or args.export_rgba_dir is not None
            )
            if _needs_black_render:
                background_black = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
                rasterize_black = initialize_resterize(
                    current_camera, gaussians, pipeline, background_black
                )
                screen_pts_black = torch.zeros_like(pos, requires_grad=False)
                _raster_black_out = rasterize_black(
                    means3D=pos,
                    means2D=screen_pts_black,
                    shs=None,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D,
                )
                rendering_black = (
                    _raster_black_out[0]
                    if isinstance(_raster_black_out, tuple)
                    else _raster_black_out
                )
                # W = white render [3,H,W], B = black render [3,H,W], both in [0,1]
                W = rendering          # [3, H, W]
                B = rendering_black    # [3, H, W]
                # (W - B) per channel = 1 - true_alpha (opaque pixels: W≈B → diff≈0 → alpha≈1)
                fg_alpha = 1.0 - (W - B).clamp(0, 1).mean(dim=0, keepdim=True)  # [1, H, W]
                h_f, w_f = W.shape[1], W.shape[2]

                # Save RGBA foreground (straight alpha PNG)
                if args.export_rgba_dir is not None:
                    os.makedirs(args.export_rgba_dir, exist_ok=True)
                    # Straight alpha: recover unblended color where alpha > 0
                    fg_rgb_straight = (B / fg_alpha.clamp(min=1e-4)).clamp(0, 1)  # [3,H,W]
                    # Build BGRA image for cv2 (H, W, 4)
                    fg_rgb_np = fg_rgb_straight.permute(1, 2, 0).detach().cpu().numpy()
                    fg_a_1hw = fg_alpha.squeeze(0).detach().cpu().numpy()  # [H, W]
                    fg_bgr_np = cv2.cvtColor(fg_rgb_np, cv2.COLOR_RGB2BGR)
                    rgba_out = np.concatenate(
                        [fg_bgr_np, fg_a_1hw[:, :, np.newaxis]], axis=2
                    )
                    cv2.imwrite(
                        os.path.join(args.export_rgba_dir, f"{frame:04d}.png"),
                        (rgba_out * 255).astype(np.uint8),
                    )

                # Background compositing
                if bg_tensor is not None and bg_output_path is not None:
                    bg_h, bg_w = bg_tensor.shape[0], bg_tensor.shape[1]
                    if bg_h != h_f or bg_w != w_f:
                        bg_resized = torch.nn.functional.interpolate(
                            bg_tensor.permute(2, 0, 1).unsqueeze(0),
                            size=(h_f, w_f),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)  # [3, H, W]
                    else:
                        bg_resized = bg_tensor.permute(2, 0, 1)  # [3, H, W]
                    # composite = fg_premul + bg * (1 - fg_alpha)
                    composite = (B + (1.0 - fg_alpha) * bg_resized).clamp(0, 1)
                    comp_np = composite.permute(1, 2, 0).detach().cpu().numpy()
                    comp_bgr = cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(bg_output_path, f"{frame}.png".rjust(8, "0")),
                        255 * comp_bgr,
                    )

            if args.profile_time:
                profiler.stop_timer("渲染")

    if args.profile_time:
        profiler.stop_timer("仿真计算")
    
    if args.render_img and args.compile_video:
        if args.profile_time:
            profiler.start_timer("视频编译")
        fps = int(1.0 / time_params["frame_dt"])
        print("fps:",fps)
        os.system(
            f"ffmpeg -framerate {fps} -i {output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {output_path}/{args.output_file}"
        )
        # Compile composited video if background was provided
        if bg_output_path is not None:
            bg_video_name = args.output_file.replace(".mp4", "_bg.mp4")
            if bg_video_name == args.output_file:
                bg_video_name = "output_bg.mp4"
            os.system(
                f"ffmpeg -framerate {fps} -i {bg_output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {bg_output_path}/{bg_video_name}"
            )
            print(f"Composited video: {bg_output_path}/{bg_video_name}")
        if args.profile_time:
            profiler.stop_timer("视频编译")
    
    # 停止总体计时并打印统计信息
    if args.profile_time:
        profiler.stop_timer("总体执行时间")
        
        # 打印时间统计摘要
        profiler.print_summary()
        
        # 保存时间统计到JSON文件
        timing_file = os.path.join(output_path, "timing_stats.json")
        profiler.save_to_json(timing_file)
