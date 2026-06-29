# PhyMAGIC

**Physical Motion-Aware Generative Inference with Confidence-guided VLM**

A training-free framework that generates physically consistent 3D motion from a **single image**, by coupling an image-to-video diffusion model, confidence-guided VLM physics reasoning, and a differentiable Material Point Method (MPM) simulator in a closed loop.

> 📄 [Paper](https://arxiv.org/abs/2505.16456) · 🌐 [Project website](https://mengsiwei.github.io/MAGIC/)  · Accepted to ECCV 2026
>
> Siwei Meng, Yawei Luo, Ping Liu


## ✨ Method at a glance

```
Single image + instruction
   │
   ├─ (1) Motion Probe Generation   I2V  →  probe videos
   ├─ (2) VLM Physical Reasoning     parameters + confidence  →  instruction-refinement loop
   ├─ (3.1) Hybrid Physical Spec     HPP = object-level params + simulator-native descriptors
   └─ (3.2) MPM Simulation           I23D  →  3D Gaussians  →  differentiable MPM  →  rendered video
```

The VLM reasons over probe videos to infer **Hybrid Physical Parameters (HPP)** — object-level parameters (material, density, Young's modulus, Poisson ratio, yield stress, …) plus simulator-native descriptors (motion, boundary condition, rotation, velocity, surface type). These feed the MPM simulator, and simulation feedback closes the loop with the perception stage.

## 🎞️ Results

The site features:

- **Real-object results** — toy car, horse, basketball, cola can, water cube.
- **Material diversity** — jelly, elastic, sand, snow, metal, fluid, foam, plasticine under varied actions.
- **Multi-material collisions** — water + rigid, water + elastic.

## 🔧 Running inference

See [`code/README.md`](code/README.md) for the full image → physics → simulation → video pipeline.

## 🙏 Acknowledgements

PhyMAGIC is built upon several excellent open-source projects. We thank the authors for their great work:

- [TRELLIS](https://github.com/microsoft/TRELLIS) — image-to-3D Gaussian generation
- [CogVideoX](https://github.com/THUDM/CogVideo) — image-to-video diffusion for motion probes
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — Gaussian scene representation and differentiable rasterization
- [PhysGaussian](https://github.com/XPandora/PhysGaussian) — MPM-based physical simulation on 3D Gaussians
