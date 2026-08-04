# Official `camp_zipnerf` vs `zipnerf-pytorch`

Audit performed 2026-08-04 against:

- `camp_zipnerf` — the official JAX/Flax release (`jax==0.4.23`, `flax==0.7.5`),
  config `configs/zipnerf/360.gin`.
- `zipnerf-pytorch` — the PyTorch port, which implements the **arXiv-v1** ZipNeRF.

> **Correction notice.** The first version of this document was produced by
> grepping for the official config *names*. That method over-reported the gap:
> two of the "missing" features ship in the port under different names. Every
> claim below is now backed by reading both implementations, and the two
> equivalence claims are backed by numerical tests in
> `zipnerf-pytorch/tests/test_camp_equivalence.py`.

## Why the official cannot be used directly

`camp_zipnerf` is JAX/Flax. A Flax module cannot participate in PyTorch autograd,
so it cannot serve as a nerfstudio model; bridging VJPs across dlpack would cost
more than the compute it wraps.

More decisively, `camp_zipnerf` performs its own scene normalization in
`internal/datasets.py` and never touches nerfstudio's dataparser. A JAX-trained
ZipNeRF would live in a *different* coordinate frame from the Nerfacto and
InvNeRF checkpoints, breaking the frozen dataparser contract that makes the
InvNeRF-Seg occupancy/IoU comparison meaningful. Measured IoU deltas already sit
near the Sim(3) noise floor, so an extra frame mismatch is fatal to the measurement.

Approach taken: **PyTorch skeleton, official recipe ported in.**

## A trap in the official config

`configs/zipnerf/360.gin` **sets the distortion loss twice**. The first block
(`p=-1`, `premult=20000`, `mult=0.005`) is superseded by a later block in the same
file, and gin takes the last assignment. The effective official values are:

```
Config.distortion_loss_target = 'tdist'
Config.distortion_loss_mult   = 0.01
Config.distortion_loss_curve_fn = (@math.power_ladder, {'p': -0.25, 'premult': 10000.})
```

Reading only the first block gives the wrong recipe.

## Status of each delta

### Ported

| Feature | Notes |
|---|---|
| `distortion_loss_target = 'tdist'` | Was hardcoded to `sdist`. `tdist` also had to be stored in `ray_results`, which the port did not do. |
| `distortion_loss_curve_fn` = `power_ladder` | `math.power_ladder` ported; verified against the closed form to 1.2e-07 (float32 eps), incl. saturation limits, special cases (`p=1`, `p=0`, `p=±inf`), monotonicity and gradient flow. |
| `net_depth_viewdirs = 3`, `skip_layer_dir = 2` | Official calls this the change that "decreases floaters substantially". Set on `NerfMLP` — the port only gin-registers `NerfMLP`/`PropMLP`, not the base `MLP`, and `PropMLP` has `disable_rgb=True` so its view branch is never built. |
| `bg_intensity_range = (0, 1)` | Port defaults to `(1, 1)`, a fixed white background. |
| grid weight decay strength | camp's `param_regularizers` `(0.1, mean, 2, 1)` evaluates to `0.05 * mean(grid**2)`; the port's `hash_decay` omits the `0.5`, so its `0.1` regularizes 2× harder. `360_camp.gin` sets `hash_decay_mults = 0.05`. Not exact — see "Approximate" below. |
| `enable_grid_c2f` + resolution schedule | Coarse-to-fine hash-grid annealing. Implemented in `internal/grid_c2f.py`. Two subtleties preserved: the weight scales the optimizer **update** (camp chains it after Adam — scaling the raw gradient would be near-inert under Adam's normalization), and this repo packs all levels into one `embeddings` tensor, so the blend is applied per level slice rather than by masking separate parameters. `scale_supersample` is derived as `1/log2(per_level_scale)` = 1.0, matching camp. Off by default; enabled in `360_camp.gin`. |

All of the above live in `zipnerf-pytorch/configs/360_camp.gin`, kept separate
from `360.gin` so arXiv-v1 behaviour stays reproducible. Code-level defaults are
unchanged, so existing runs reproduce bit-identically.

### Already present — no port needed

| Feature | Evidence |
|---|---|
| `spline_interlevel_params` | The port's `anti_interlevel_loss` **is** camp's `spline_interlevel_loss`. Both blur the NeRF histogram, integrate to a piecewise-quadratic CDF, resample into the proposal intervals, and apply the same truncated chi-squared; they differ only in evaluating the quadratic (camp: `linspline.compute_integral`+`interpolate_integral`; port: `math.sorted_interp_quad`). Measured max difference **0.0 – 5.96e-08** across three interval/blur configurations. Parameters already match exactly: `anti_interlevel_loss_mult = 0.01` == camp `mults = 0.01`; `pulse_width = [0.03, 0.003]` == camp `blurs = (0.03, 0.003)`. |
| `bottleneck_width = 256`, `net_width_viewdirs = 256` | Already the port's defaults. |
| `HashEncoding.max_grid_size = 8192`, `hash_map_size = 2097152` | Port defaults `grid_disired_resolution = 8192`, `grid_log2_hashmap_size = 21` (2²¹ = 2097152). |
| `Model.raydist_fn` power ladder `p = -1.5` | Port uses `raydist_fn='power_transformation'` with `power_lambda = -1.5`, matching on `p`. The official's `premult=2` has no equivalent knob and is unported. |

### Not applicable to this recipe

| Feature | Why |
|---|---|
| affine GLO (`glo_mlp_arch`, `glo_premultiplier`) | The official 360 config sets `Model.num_glo_features = 0` — GLO is off. It is only used by `360_aglo128.gin`. |

### Approximate, not exact

- **Grid decay grouping.** camp takes a flat `mean(param**2)` per grid; the port
  takes `segment_coo(param**2, idx, reduce='mean').mean()`, a mean of
  per-hash-entry means. These agree when hash entries are evenly populated and
  diverge when they are not. Only the multiplier has been matched.

### Still missing

| Feature | Notes |
|---|---|
| `unscented_mip_basis = 'hexify'`, `unscented_scale_mult = 0.5` | Alternative multisampling basis. Absent. |
| `scene_bbox` | Absent. |
| `rad_mult_min` / `rad_mult_max` | Absent. |
| `param_regularizers` as a general mechanism | Only the grid case is approximated via `hash_decay_mults`; there is no general per-parameter-prefix regularizer. |

## Reproduction quality of the port (upstream's own numbers)

mip-NeRF 360, PSNR:

| | bicycle | garden | stump | room | counter | kitchen | bonsai |
|---|---|---|---|---|---|---|---|
| Paper | 25.80 | 28.20 | 27.55 | 32.65 | 29.38 | 32.50 | 34.46 |
| Port | 25.44 | 27.98 | 26.75 | 32.13 | 29.10 | 32.63 | 34.20 |

Within 0.1–0.8 dB, above the paper on kitchen.

## Verification

`zipnerf-pytorch/tests/test_camp_equivalence.py` (6 tests, all passing):

- `power_ladder` against its closed form (max error 1.2e-07), saturation limits,
  special cases `p ∈ {1, 0, ±inf}`, monotonicity, gradient flow.
- `anti_interlevel_loss` == camp's `spline_interlevel_loss` (max difference
  0.0 – 5.96e-08 over three interval/blur configurations).
- Coarse-to-fine weight schedule against camp's `cosine_sequential` window,
  including clamping and monotonicity in step.
- Coarse-to-fine scales the update, not the gradient: a zero-weight level does
  not move through a simulated optimizer step while an active one moves fully.

End-to-end on cherry ds2 (seed 42, 200 iterations, c2f enabled): between the
step-50 and step-199 checkpoints, grid levels 4096 and 8192 — exactly the two
whose c2f weight is 0 at that point — are **bit-identical** (`max|Δ| = 0.0`),
while all eight active levels changed.

## Caveat on measurement

Everything above establishes that the ports are *faithful* and that the config
*trains* — smoke runs of 150–300 iterations, no errors, losses finite and
decreasing. None of it shows improved reconstruction quality; a few hundred
iterations measures nothing. Any quality claim needs a full-length run against
the `cherry_sift_ds2/fruit_nerf/run_100k` baseline.
