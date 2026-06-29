# PhyMAGIC — Inference & Simulation Scripts

Core scripts for the PhyMAGIC pipeline: **single image → physics inference → MPM simulation → rendered video**.

These are copied from the main `phymagic` repository for reference. Running them end-to-end additionally requires the simulator backend (`mpm_solver_warp`), the Gaussian renderer, TRELLIS, and the trained/3rd-party model weights present in the full repo.

## Environment

```bash
conda activate physicdynamic     # or: conda env create -f environment.yml
pip install -r requirements.txt
```

Put your OpenAI API key in `configs/openai_apikey` (used by the LLM physics reasoning stage).

## Files

| File | Role |
|------|------|
| `infer_physics.py` | **Stage 1.** Image → video (CogVideoX) → GPT-4o physics reasoning with confidence scores → iterative prompt refinement. Outputs `physics.yaml`. |
| `i2v.py` | Image-to-video generation (CogVideoX1.5-5B) used to enrich motion cues. |
| `generate_new_text.py` | LLM prompt-generation / refinement helper. |
| `llm_to_mpm_config.py` | **Stage 2 setup.** Converts inferred `physics.yaml` → MPM config JSON. |
| `select_mpm_behavior.py` | Maps inferred material type to MPM constitutive behavior. |
| `compute_ply_bounds.py`, `glb_to_ply.py` | Geometry utilities (bounds computation, mesh → point cloud). |
| `simulate.py` | **Stage 2.** MPM simulation + Gaussian rasterization → rendered video. |
| `run_pipeline.py` | Foreground simulation + scene compositing pipeline (RGBA frames, optional Blender shadows, final MP4). |

## End-to-end pipeline

### Step 1 — LLM physics inference

Infers material properties (density, elasticity, friction, yield, …) for the object via image-to-video + GPT reasoning.

```bash
python infer_physics.py --data_path data/yellowcar --generate_video
```

Outputs `data/yellowcar/physics.yaml`.
(Omit `--generate_video` to reason over an existing frame sequence instead of generating one.)

Key arguments:
- `--data_path` — folder containing the input image / frames.
- `--my_apikey` — path to OpenAI API key file (default `configs/openai_apikey`).
- `--query_txt` — prompt template (default `configs/prompts_multi_v4.txt`).
- `--save_file` — output YAML name (default `physics.yaml`).

### Step 2 — Convert physics YAML → MPM config

```bash
python llm_to_mpm_config.py \
    --physics_yaml data/yellowcar/physics.yaml \
    --output_config configs/yellowcar_config.json
```

Tunable simulation defaults: `--frame_dt`, `--frame_num`, `--substep_dt`, `--n_grid`, `--grid_lim`.

### Step 3 — Simulate & render

```bash
python simulate.py \
    --config_path configs/ \
    --data_path data/ \
    --image_path yellowcar \
    --render_img --compile_video
```

Useful flags:
- `--from_origin` — run TRELLIS image→3D first (requires `data/<name>/origin.png`).
- `--white_bg` — render on a white background.
- `--bg_image` — composite over a background image.
- `--output_ply` / `--output_h5` — export simulated particles.
- `--export_particles_npz`, `--export_rgba_dir` — export for external compositing.

### (Optional) Scene compositing

```bash
python run_pipeline.py --image data/yellowcar/origin.png \
    --data_path data/yellowcar --config_path configs/ \
    --fps 24 --out out/yellowcar/ [--use_blender]
```

Produces `fg_rgba/`, `shadow/`, `final_frames/`, and `final.mp4` with camera/alignment metadata.

## Pipeline diagram

```
Image
  → [CogVideoX i2v]              (i2v.py)
  → [GPT-4o physics reasoning]   (infer_physics.py)        → physics.yaml
  → [confidence-guided refine]   (generate_new_text.py)
  → [YAML → MPM config]          (llm_to_mpm_config.py)    → *_config.json
  → [TRELLIS i23d + particles]   (simulate.py --from_origin)
  → [Differentiable MPM]         (simulate.py)             → trajectories
  → [Gaussian rasterizer]        (simulate.py)             → frames → video
  → [scene compositing]          (run_pipeline.py)         → final.mp4
```
