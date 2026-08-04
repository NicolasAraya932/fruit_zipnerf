# Official `camp_zipnerf` vs `zipnerf-pytorch`

Audit performed 2026-08-04 against:

- `camp_zipnerf` — the official JAX/Flax release (`jax==0.4.23`, `flax==0.7.5`,
  gin configs under `configs/zipnerf/`).
- `zipnerf-pytorch` — the unofficial PyTorch port, which implements the **arXiv v1**
  ZipNeRF, not the later official release.

## Why the official cannot be used directly

`camp_zipnerf` is JAX/Flax. A Flax module cannot participate in PyTorch autograd,
so it cannot serve as a nerfstudio model; bridging VJPs across dlpack would cost
more than the compute it wraps.

More decisively, `camp_zipnerf` performs its own scene normalization in
`internal/datasets.py` and never touches nerfstudio's dataparser. A JAX-trained
ZipNeRF would therefore live in a *different* coordinate frame from the Nerfacto
and InvNeRF checkpoints, breaking the frozen dataparser contract that makes the
InvNeRF-Seg per-bbox occupancy/IoU comparison meaningful. Since measured IoU
deltas already sit near the Sim(3) noise floor, an extra frame mismatch is fatal
to the measurement.

The chosen approach is therefore: **PyTorch skeleton, official recipe ported in.**

## What the PyTorch port is missing

Checked by name across `internal/` and `configs/`:

| Official feature | Present in `zipnerf-pytorch`? |
|---|---|
| `distortion_loss_target = 'tdist'` | absent |
| `distortion_loss_curve_fn = power_ladder(p=-1, premult=20000)`, mult `0.005` | absent — uses older normalized distortion |
| `spline_interlevel_params` (spline interlevel loss) | absent — has the older `anti_interlevel_loss` |
| `enable_grid_c2f` + `grid_c2f_resolution_schedule_def` (coarse-to-fine hash grid) | absent |
| `param_regularizers` | absent — only the narrower `hash_decay_mults` |
| `scene_bbox` | absent |
| `rad_mult_min` / `rad_mult_max` | absent |
| affine GLO (`glo_mlp_arch`, `glo_mlp_act`, `glo_premultiplier`) | absent |
| `isotropize_gaussians` | absent |
| unscented multisampling (`unscented_mip_basis`, `unscented_scale_mult`) | absent |

Present in both: `anti_interlevel_loss`, `hash_decay_mults`, the multisampled
hash-grid featurization, and `scale_featurization`.

## Reproduction quality of the port (upstream's own numbers)

mip-NeRF 360, PSNR:

| | bicycle | garden | stump | room | counter | kitchen | bonsai |
|---|---|---|---|---|---|---|---|
| Paper | 25.80 | 28.20 | 27.55 | 32.65 | 29.38 | 32.50 | 34.46 |
| Port | 25.44 | 27.98 | 26.75 | 32.13 | 29.10 | 32.63 | 34.20 |

Within 0.1–0.8 dB, and above the paper on kitchen. So the port is a sound
skeleton; the table above is the gap worth closing.

## Port status

None of these are ported yet — this document is the plan of record, not a
completion claim. Each ported term should be validated numerically against the
JAX reference where feasible, since several (notably `power_ladder`) change the
loss surface rather than merely adding a term.
