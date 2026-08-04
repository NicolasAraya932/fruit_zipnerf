# RESUME — instructions for the next session

Written 2026-08-04, at the point where the 100k `fruit_zipnerf` run was about to
start. Read this **before** touching anything.

## Read these first

1. `~/.claude` memory, especially `feedback-always-commit` (commit as each step
   finishes, never batched, never wait to be asked; pushes still need confirming)
   and `feedback-repos-justify-changes`.
2. `~/Desktop/Repos/InvNeRF-Seg/memories.json` and
   `cherry_cluster_occupancy_replication.json` — the real pipeline memory.
3. `CAMP_DELTAS.md` in this repo — the audited state of the camp port. Note it
   carries a correction notice; the *first* version of that audit was wrong.

## Where things stand

**Done and committed.** `fruit_zipnerf` exists as a registered nerfstudio method:
FruitNeRF's binary semantic head on a ZipNeRF backbone, trained through the
frozen dataparser contract. The camp_zipnerf official-recipe deltas are ported.

`fruit_zipnerf` (github.com/NicolasAraya932/fruit_zipnerf, branch `main`):

| commit | what |
|---|---|
| `531bdbc` | Document the grid_c2f port and its verification |
| `c65d6fb` | Correct the camp deltas audit with verified findings |
| `0190990` | Merge GitHub repo initialization (LICENSE, .gitignore) |
| `1697df5` | Add fruit_zipnerf |

`~/Desktop/Repos/zipnerf-pytorch` (local only, **not pushed** — fork of
SuLvXiangXin/zipnerf-pytorch; `3d06463` and earlier are the user's):

| commit | what |
|---|---|
| `5fcb802` | Port coarse-to-fine hash-grid annealing (`enable_grid_c2f`) |
| `a786a0e` | Match official grid decay strength in `360_camp.gin` |
| `749864d` | Prove `anti_interlevel_loss` already IS camp's spline interlevel loss |
| `672e99d` | Port distortion loss: `power_ladder` on metric `tdist` |
| `ad0fa21` | Make `zipnerf_ns` run on nerfstudio 1.1.5; gated semantic head |

`FruitNeRF_115` and `Nerfstudio_Patch` were **not modified**. `fruit_nerf` stays
reproducible so the comparison is a controlled A/B.

## The run that should be finishing

```
ns-train fruit_zipnerf --data /workspace/Desktop/DATASETS/IMAGES/cherrytree
  --output-dir /workspace/Desktop/DATASETS/OUTPUTS/cherry_fruitzipnerf
  --experiment-name cherry_sift_ds2 --timestamp run_100k --vis wandb
  --max-num-iterations 100000 --optimizers.model.scheduler.max-steps 100000
  --pipeline.model.gin-file configs/360_camp.gin
log: /workspace/Desktop/DATASETS/OUTPUTS/fruitzipnerf_100k.log
```

Baseline to compare against (exists, `step-000099999.ckpt`):
`/workspace/Desktop/DATASETS/OUTPUTS/cherry_default_fruitnerf/cherry_sift_ds2/fruit_nerf/run_100k`

### Check these before believing any result

1. **Did it actually finish?** `tail` the log; confirm
   `nerfstudio_models/step-000099999.ckpt` exists.
2. **Was the LR schedule right?** `config.yml` must show
   `scheduler.max_steps: 100000`. If it says 25000, the LR flatlined at 25k and
   the run is invalid — rerun, do not report the number.
3. **`semantic_f1` in W&B, not just PSNR.** It was 0.0 at 300 iterations —
   majority-class collapse, since cherry masks cover only **0.53%** of each
   image. If it is still ~0 at 100k, the semantic head collapsed and the fix is
   `pos_weight ≈ 190` on the BCE — but it must then be applied to **both**
   `fruit_nerf` and `fruit_zipnerf`, or the A/B is no longer fair.
   `fruit_nerf.py:189` uses plain `BCEWithLogitsLoss(reduction="mean")`; that was
   matched deliberately.

## Remaining work (task 6)

Validate against the InvNeRF-Seg export and occupancy pipeline:

- The model emits per-sample `density` in the `[..., S, 1]` shape
  `generate_inv_nerf_radiance_fields_cloud` expects, plus `semantics`. Confirm
  the export path actually consumes the ZipNeRF checkpoint.
- Run occupancy against the Nerfacto baseline (`per_bbox` + 100k point budget).
- **Deltas must clear the Sim(3) noise floor** recorded in
  `cherry_cluster_occupancy_replication.json` before any claim is made.
- Then update `memories.json` and the registries **in the same commit as the
  code**, per the standing rule.

Also still unported from camp (all smaller than what is done): `unscented
'hexify'`, `scene_bbox`, `rad_mult`, and a general `param_regularizers`
mechanism. `hash_decay`'s grouping is approximate — only the multiplier matches.

## Traps that already cost time — do not rediscover these

- **`--pipeline.model.gin-file` takes a list.** Repeating the flag *replaces*
  instead of appending, silently dropping `360.gin` and producing a confusing
  autograd error. Pass space-separated in one flag.
- **`--optimizers.model.scheduler.max-steps` defaults to 25000**, independent of
  `--max-num-iterations`. Always set it for long runs.
- **`git` on `zipnerf-pytorch` needs** `git config --global --add safe.directory
  /home/sir3po/Desktop/Repos/zipnerf-pytorch`; it is root-owned. Without it git
  reports nothing and the repo *looks* like it has no history. Commit from
  inside the container (`docker exec angry_hawking`) — the host user cannot
  write `.git`.
- **`Backend.set_backend('cuda')` is only called inside ZipNeRF's `Model.__init__`**
  (`internal/models.py:65`). Anything touching `gridencoder` outside that — an
  export script, a direct field query — must set it explicitly first.
- **`import torch` before `import _cuda_backend`**, or you get a bare
  `libc10.so: cannot open shared object file`.
- **The frozen contract pins image paths.** `downscale_factor` is pinned to 2 in
  the config on purpose; auto-select can freeze full-res and OOM.
- Cherry's `dataparser_scale` is **0.0740**. The `0.23033` in memory is the
  *outdoor* rig, a different capture.
- **camp's `configs/zipnerf/360.gin` sets the distortion loss twice.** gin takes
  the last assignment: `p=-0.25, premult=10000, mult=0.01`. The earlier
  `p=-1/premult=20000/mult=0.005` block is superseded — reading only that gives
  the wrong recipe.

## Standing rules

- Commit after each meaningful step. Confirm before `git push`; push with the
  `GIT_SSH_COMMAND` + `SUPER_OBVIOUS_GITHUB_SSH_KEY` pattern.
- Send the two notification emails via `~/Desktop/Repos/EmailSender`
  (`email-sender --action` / `--results`) before presenting a response.
- Many files under `DATASETS/` are root-owned — write from inside the container.
- Never print or commit the W&B key (`Repos/KEYS/wandb_api.yml`).

## Honest status line

Everything so far establishes that the ports are **faithful** (6 passing tests,
`power_ladder` to 1.2e-07, spline-loss equivalence to 5.96e-08, frozen c2f levels
bit-identical across checkpoints) and that the config **trains**. Nothing yet
shows ZipNeRF reconstructs cherry ds2 **better** than Nerfacto/`fruit_nerf`.
The longest run before this one was 300 iterations. Do not let the volume of
green checkmarks read as a quality result.
