# **TRELLIS.2-Text-to-3D**

> A Gradio-based demonstration for Microsoft's TRELLIS.2-4B model, enabling text-to-3D and image-to-3D generation. The workflow generates 2D images via Tongyi-MAI/Z-Image-Turbo (text-to-image) or accepts direct uploads, then lifts them to textured 3D meshes exportable as GLB files. Supports advanced sampler controls for sparse structure, shape, and material stages, with preprocessing (background removal, cropping) for optimal results.

> [!IMPORTANT]
> Device Support and Testing (Hopper Architecture): NVIDIA H200, Kernel (Memory-Efficient): FlashAttention 3

## Features

- **Text-to-Image-to-3D**: Enter prompts (e.g., "A realistic Cat 3D model") to auto-generate images via Z-Image-Turbo, then convert to 3D.
- **Direct Image-to-3D**: Upload RGBA/PNG images for immediate 3D lifting; auto-preprocesses with background removal (BRIA-RMBG-2.0) and cropping.
- **Advanced Controls**: Tune resolutions (512/1024/1536), sampler guidance/rescale/steps for three stages (sparse structure, shape, material), decimation (faces), and texture size.
- **Export Options**: Outputs interactive GLB models viewable in-browser; downloadable with timestamps for session management.
- **Session Handling**: Per-user temp directories for outputs; auto-cleanup on unload.
- **Custom Theme**: Storj theme with CSS for responsive layout; progress tracking via Gradio.
- **Examples**: 70+ pre-loaded image/text prompts for objects like cats, planes, cars, shoes, and more.

---
<img width="1918" height="1028" alt="Screenshot 2025-12-20 at 10-39-49 TRELLIS 2-Text-to-3D - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/76c33511-f2fb-4ecd-9ff3-1e0d905facd4" />

https://github.com/user-attachments/assets/436cf082-b04f-4590-82e9-10b643b7118f

---
## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (required for bfloat16; low_vram mode available).
- pip >= 23.0.0 (see pre-requirements.txt).
- Stable internet for initial model downloads (~4B for TRELLIS, Turbo for Z-Image).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/TRELLIS.2-Text-to-3D-FA3.git
   cd TRELLIS.2-Text-to-3D-FA3
   ```

2. Install pre-requirements (for pip version):
   Create a `pre-requirements.txt` file with the following content, then run:
   ```
   pip install -r pre-requirements.txt
   ```

   **pre-requirements.txt content:**
   ```
   pip>=23.0.0
   ```

3. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   --extra-index-url https://download.pytorch.org/whl/cu124
   git+https://github.com/huggingface/diffusers.git@refs/pull/12790/head
   torch==2.6.0
   torchvision==0.21.0
   triton==3.2.0
   pillow==12.0.0
   matplotlib
   rembg
   imageio==2.37.2
   imageio-ffmpeg==0.6.0
   tqdm==4.67.1
   easydict==1.13
   opencv-python-headless==4.12.0.88
   trimesh==4.10.1
   zstandard==0.25.0
   kornia==0.8.2
   timm==1.0.22
   transformers==4.57.3
   git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/cumesh-0.0.1-cp310-cp310-linux_x86_64.whl
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/nvdiffrast-0.4.0-cp310-cp310-linux_x86_64.whl
   https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/nvdiffrec_render-0.0.0-cp310-cp310-linux_x86_64.whl
   ```

4. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or the provided URL if using Spaces).

## Usage

1. **Text-to-Image-to-3D Tab**:
   - Enter a prompt (e.g., "A cyberpunk Cat 3D").
   - Click "Generate Image" to create a 2D base via Z-Image-Turbo.
   - Proceed to 3D generation.

2. **Image-to-3D Tab**:
   - Upload an RGBA/PNG image (auto-preprocessed).

3. **Configure Settings**:
   - Resolution: 512 (fast), 1024/1536 (detailed, cascade mode).
   - Sampler Params: Expand accordion for stage-specific guidance (1-10), rescale (0-1), steps (1-50), rescale_t (1-6).
   - Export: Target faces (50k-500k), texture size (512-4096).

4. **Generate 3D**: Click "Generate 3D"; monitor progress (geometry ~10%, mesh ~70%, export ~90%).
5. **Output**: View interactive GLB in 3D viewer; download via button.

### Examples

#### Text-to-3D Prompts
| Category | Examples |
|----------|----------|
| Cats    | "A Cat 3D model", "A realistic Cat 3D model", "A cartoon Cat 3D model", "A low poly Cat 3D", "A cyberpunk Cat 3D" |
| Planes  | "A Plane 3D model", "A commercial Plane 3D", "A fighter jet Plane 3D", "A low poly Plane 3D", "A vintage Plane 3D" |
| Cars    | "A Car 3D model", "A sports Car 3D", "A luxury Car 3D", "A low poly Car 3D", "A racing Car 3D" |
| Shoes   | "A Shoe 3D model", "A sneaker Shoe 3D", "A running Shoe 3D", "A leather Shoe 3D", "A high heel Shoe 3D" |
| Furniture | "A Chair 3D model", "A Table 3D model", "A Sofa 3D model", "A Lamp 3D model" |
| Others  | "A Watch 3D model", "A Backpack 3D model", "A Drone 3D model", "A Robot 3D model", "A Smartphone 3D model" |

#### Image-to-3D
Upload from 70+ example images (e.g., "example-images/A (1).webp" to "A (71).webp") for objects like animals, vehicles, furniture.

## Troubleshooting

- **Model Loading Errors**: Ensure torch 2.6.0 and diffusers PR #12790; check CUDA with `torch.cuda.is_available()`. Use `low_vram=True` if OOM.
- **Z-Image Fails**: Guidance fixed at 0.0; verify prompt is descriptive. Empty cache with `torch.cuda.empty_cache()`.
- **TRELLIS Errors**: Flash Attn 3 and Flex GEMM wheels required; verbose autotune in env vars. Reduce resolution/steps for low VRAM.
- **Preprocessing Issues**: BRIA-RMBG client needs internet; fallback for alpha channels. Crop uses 80% alpha threshold.
- **GLB Export Fails**: o_voxel handles remeshing; simplify to 16M faces max. Check tmp dir permissions.
- **Gradio Session**: 120s cache; restart for cleanup. Set `ssr_mode=True` if rendering issues.
- **Wheels Missing**: Download from provided URLs; match Python 3.10+ and CUDA 12.4.

## Contributing

Contributions encouraged! Fork the repo, enhance samplers or add workflows (e.g., multi-view), and submit PRs with tests. Focus areas:
- Video-to-3D support.
- Custom preprocessors.
- Batch generation.

Repository: [https://github.com/PRITHIVSAKTHIUR/TRELLIS.2-Text-to-3D-FA3.git](https://github.com/PRITHIVSAKTHIUR/TRELLIS.2-Text-to-3D-FA3.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
