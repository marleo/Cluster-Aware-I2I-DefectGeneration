# Cluster-Aware Img2Img Defect Generation

This project generates SD 1.5 img2img variants from real wind-turbine blade
images selected across DINOv3 feature clusters. It then filters the generated
images with the same global and local DINOv3 similarity logic used in
`dinov3_global_local_window.ipynb`.

The intended pipeline is:

```text
real YOLO images
    -> DINOv3 global/local feature bank
    -> global-feature clusters
    -> balanced source-image selection
    -> SD 1.5 img2img + crack LoRA through ComfyUI
    -> DINOv3 global/local filtering
    -> accepted/rejected/review outputs and distribution report
```

Sampling-time DINO guidance is deliberately not required. The source image and
its cluster control the generated image's feature-space anchor. DINO is used
after generation to verify realism, reject outliers and duplicates, and measure
coverage.

## Files

- `config.toml`: all paths, models, generation settings, and filter thresholds.
- `main.py`: command-line entry point.
- `cluster_defects/`: reusable pipeline implementation.
- `workflows/SD15_Cluster_Aware_Img2Img_Crack.workflow.json`: editable ComfyUI
  UI workflow.
- `workflows/SD15_Cluster_Aware_Img2Img_Crack.api.json`: documented API graph
  with placeholder values.
- `examples/cluster_assignments.csv`: format for optional precomputed clusters.
- `tests/`: lightweight tests that do not load DINO or Stable Diffusion.

## Environment

Generation runs in the already-running ComfyUI process. The command-line
project only needs Python for DINO feature extraction and orchestration.

Create a dedicated environment with a CUDA-enabled PyTorch build when possible.
CPU works, but feature-bank construction is substantially slower.

```powershell
cd C:\Users\Mario\Documents\Paper02-Defect\cluster_aware_img2img_dino
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the correct CUDA build of `torch` and `torchvision` from the PyTorch
index first if the default packages do not provide CUDA on the machine.

The existing `C:\Users\Mario\Documents\Paper02-Defect\.venv-clip` environment
already has the required libraries, but its PyTorch build is CPU-only.

## Configuration

Review `config.toml` before the first run. Its defaults point to the currently
used Craze dataset, DINOv3 weights, local DINOv3 repository, ComfyUI server,
SD 1.5 checkpoint, and crack LoRA.

By default, the project computes six clusters from real training-image global
DINO features. To use existing cluster assignments instead, set
`paths.cluster_manifest` to a CSV with these columns:

```text
image_path,cluster_id
```

Relative `image_path` values are resolved below the dataset root.

## Run

Make sure ComfyUI is running on `http://127.0.0.1:8000`.

```powershell
# 1. Build the real feature bank, thresholds, and cluster assignments.
python main.py build-bank

# 2. Generate a balanced batch from all clusters.
python main.py generate --sources-per-cluster 3 --variants-per-source 4

# 3. Filter every generated candidate.
python main.py filter

# 4. Plot real train/validation and accepted synthetic distributions.
python main.py plot
```

Or run all stages:

```powershell
python main.py run --sources-per-cluster 3 --variants-per-source 4
```

For a small end-to-end smoke test:

```powershell
python main.py run --sources-per-cluster 1 --variants-per-source 1
```

Outputs are written under `outputs/`:

```text
outputs/
  feature_bank/
    real_features.npz
    global_metadata.csv
    local_metadata.csv
    bank_summary.json
  candidates/
    images/
    candidate_manifest.csv
  filtered/
    accepted/
    rejected/
    manual_review/
    filter_results.csv
  reports/
    feature_distribution.png
    mmd_summary.csv
```

## ComfyUI Workflow

Open `workflows/SD15_Cluster_Aware_Img2Img_Crack.workflow.json` in ComfyUI.
The workflow uses:

- `v1-5-pruned.safetensors`
- full `LoraLoader` for both model and CLIP
- `wtbsd_sd15_crack_defects-step00002250.safetensors`
- a real source image encoded by the VAE
- `denoise = 0.42`, which preserves framing while allowing a visible defect
- DPM++ 2M with a Karras schedule

For interactive use, choose a source image in `LoadImage` and adjust:

- `denoise 0.25-0.35`: close replication of camera/framing and surface.
- `denoise 0.40-0.50`: recommended defect variants with moderate change.
- `denoise 0.55-0.65`: more diversity, with a larger realism risk.
- LoRA model/CLIP strength `0.75-0.95`: crack visibility and token influence.

The automated generator constructs the same graph through the ComfyUI API and
changes the source image, seed, and output prefix per candidate.

## Filtering Details

The filter matches the notebook in the following ways:

- DINOv3 ViT-S/16 at 512x512 with ImageNet normalization.
- normalized CLS token for global features.
- normalized pooled patch tokens for local defect features.
- mean cosine similarity to the five nearest real references.
- lower thresholds calibrated with the fifth percentile of leave-one-out real
  similarities.
- review margin around calibrated thresholds.

For generated img2img candidates, the source YOLO box is mapped to the output
and scored directly. Sliding-window search is used only when no source box is
available. This is stricter than always searching for the most defect-like
window.

The notebook's real-reference duplicate threshold is recorded but disabled by
default. Img2img candidates are supposed to resemble a real source. Duplicate
control instead compares each candidate with its source and with already
accepted synthetic images.

## Tests

```powershell
python -m unittest discover -s tests -v
```

