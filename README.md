# fruit_zipnerf

ZipNeRF base model instead of Nerfacto for FruitNeRF.

FruitNeRF's binary semantic head on a **ZipNeRF** backbone, registered as a
nerfstudio method (`fruit_zipnerf`).

This replaces FruitNeRF's Nerfacto backbone with ZipNeRF's anti-aliased,
multisampled hash-grid geometry, while keeping the semantic supervision and the
export contract byte-compatible with the existing InvNeRF-Seg pipeline.

`fruit_nerf` itself is **not modified**. It remains available and reproducible,
so the Nerfacto-vs-ZipNeRF comparison is a controlled A/B rather than a
before/after that invalidates the existing `cherry_sift_ds2` baseline runs.

## Why not the official `camp_zipnerf`?

The official implementation is JAX/Flax. A Flax module cannot participate in
PyTorch autograd, so it cannot be used as a nerfstudio model — and training it
standalone would put the reconstruction in ZipNeRF's own scene normalization
rather than the frozen dataparser contract, which would break the shared
canonical frame the InvNeRF-Seg occupancy/IoU comparison depends on.

The approach taken instead is to use `zipnerf-pytorch` as the PyTorch skeleton
and port the official release's recipe into it. See `CAMP_DELTAS.md` for the
audit of what the PyTorch port is missing relative to `camp_zipnerf`.

## Architecture

The semantic head hangs off ZipNeRF's density bottleneck (`internal/models.py`,
`MLP.forward`) — the structural analog of the `density_embedding` tap in
`fruit_nerf/fruit_field.py:262-268`. It is read **before** the view-direction and
GLO stages, so semantics stay appearance-independent, matching FruitNeRF.

Only the final NeRF level carries a semantic head; the proposal MLPs exist to
place samples and are left untouched.

Outputs that exist specifically to keep InvNeRF-Seg working unchanged:

| Output | Purpose |
|---|---|
| `semantics` | Weight-composited binary fruit logits, BCE-supervised against `batch["fruit_mask"]` |
| `density` | Per-sample density of the final level, in the `[..., S, 1]` shape `generate_inv_nerf_radiance_fields_cloud` expects |
| `semantics_colormap` | Thresholded mask for the viewer |

## Data

Uses FruitNeRF's `FruitDataManager`/`FruitDataset` unchanged: it already reads
the frozen dataparser contract (`load_frozen_contract=True`) and exposes the
contract's `binary_img` modality as `batch["fruit_mask"]`. The only addition is
`batch["rgb"]`, which ZipNeRF's loss expects.

`downscale_factor` is **pinned to 2**, not auto-selected — the contract pins
image paths, and auto-selection can freeze full resolution and OOM.

## Install

Requires, all installed from source in the same environment:

- a nerfstudio with the frozen-contract dataparser patch (`Nerfstudio_Patch`)
- `zipnerf-pytorch` (provides `zipnerf_ns`, `internal`, `gridencoder`)
- `FruitNeRF_115` (provides the Fruit data stack)
- `torch_scatter` matching your torch build — required, not optional
  (`scale_featurization` needs `segment_coo`)

```bash
# CUDA backend for zipnerf-pytorch
cd zipnerf-pytorch/extensions/cuda
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 pip install -e . --no-build-isolation

pip install -e . --no-deps --no-build-isolation   # this repo
```

`--no-deps` is deliberate: the target environment has a pinned torch/nerfstudio
stack that pip must not resolve over.

## Train

```bash
export PYTHONPATH=/path/to/Nerfstudio_Patch
ns-train fruit_zipnerf \
  --data /path/to/cherrytree \
  --output-dir OUTPUTS/cherry_fruitzipnerf \
  --experiment-name cherry_sift_ds2 \
  --max-num-iterations 100000
```

## Notable knobs

| Flag | Default | Note |
|---|---|---|
| `--pipeline.model.semantic-loss-weight` | 1.0 | Weight on the BCE term |
| `--pipeline.model.pass-semantic-gradients` | False | If False, semantics do not shape geometry (FruitNeRF's default) |
| `--pipeline.model.semantic-threshold` | 0.5 | Matches the export gate, so training metrics and exported clouds agree |

## Upstream changes this depends on

Small, gated changes in `zipnerf-pytorch`, all no-ops when the semantic head is
disabled so stock `zipnerf` reproduces bit-identically:

- `internal/models.py` — `MLP` gains `num_semantic_classes` (default `0` =
  disabled), builds the head and emits a per-sample `semantics` key only when enabled.
- `zipnerf_ns/zipnerf_model.py` — `gin_bindings()` hook so subclasses can
  configure the MLPs without duplicating gin parsing.

Three separate upstream bugs also had to be fixed for `zipnerf-pytorch` to run
against nerfstudio 1.1.5 / torch 2.1.2 at all:

1. `extensions/cuda/setup.py` — C++ sources compiled with `-std=c++14` while
   torch 2.1.2 requires C++17; the extension could not build.
2. `zipnerf_ns/zipnerf_config.py` — the default `ColmapDataParserConfig` did not
   match the registered `colmap` subcommand, so tyro could not build the parser
   and the shipped method could never launch.
3. `zipnerf_ns/zipnerf_pipeline.py` — called `datamanager.to(device)`, a stale
   copy of an older `VanillaPipeline`; nerfstudio ≥1.1 DataManagers are not
   `nn.Module`s.
