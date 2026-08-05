import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
# psnr_data.json comes from extract_wandb_metrics.py (run inside the container)
SP = os.environ.get("PSNR_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
D = json.load(open(f"{SP}/psnr_data.json"))

BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1,2
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
SURF = "#fcfcfb"

# method -> hue, front end -> line style (composite encoding, 2 hues only)
STYLE = {
    "fruit_zipnerf / SIFT":   dict(color=BLUE,   ls="-",  label="fruit_zipnerf · SIFT"),
    "fruit_zipnerf / ALIKED": dict(color=BLUE,   ls="--", label="fruit_zipnerf · ALIKED+LG"),
    "fruit_nerf / SIFT":      dict(color=ORANGE, ls="-",  label="fruit_nerf · SIFT"),
    "fruit_nerf / ALIKED":    dict(color=ORANGE, ls="--", label="fruit_nerf · ALIKED+LG"),
}
ORDER = ["fruit_zipnerf / SIFT", "fruit_zipnerf / ALIKED",
         "fruit_nerf / SIFT", "fruit_nerf / ALIKED"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False,
})

def finish(ax):
    ax.grid(axis="y", color=MUTED, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

# ---------- Fig 1: eval PSNR vs step ----------
fig, ax = plt.subplots(figsize=(9, 4.6))
for name in ORDER:
    s = np.array(D[name]["series"]["Eval Metrics Dict/psnr"], dtype=float)
    st = STYLE[name]
    # light raw trace + smoothed line (eval batch metric is noisy)
    ax.plot(s[:, 0], s[:, 1], color=st["color"], ls=st["ls"], lw=0.7, alpha=0.22)
    k = 9
    sm = np.convolve(s[:, 1], np.ones(k) / k, mode="valid")
    ax.plot(s[k - 1:, 0], sm, color=st["color"], ls=st["ls"], lw=2.0, label=st["label"])
    ax.annotate(f"{s[-1,1]:.2f}", xy=(s[-1, 0], sm[-1]), xytext=(6, -3),
                textcoords="offset points", color=INK2, fontsize=9)
ax.set_xlabel("training step"); ax.set_ylabel("eval PSNR (dB)")
ax.set_xlim(0, 114000); ax.set_ylim(14.0, 17.4)
ax.set_title("Eval PSNR over training — cherry ds2, 100k steps", color=INK, fontsize=12, pad=12, loc="left")
ax.legend(frameon=False, loc="lower left", fontsize=9.5, labelcolor=INK2, ncol=2)
finish(ax); fig.tight_layout(); fig.savefig(f"{OUT}/psnr_curves.pdf"); plt.close(fig)

# ---------- Fig 2: final full-eval-set PSNR ----------
K = "Eval Images Metrics Dict (all images)/psnr"
fig, ax = plt.subplots(figsize=(7.4, 4.2))
groups = ["SIFT", "ALIKED+LightGlue"]
zn = [D["fruit_zipnerf / SIFT"]["summary"][K], D["fruit_zipnerf / ALIKED"]["summary"][K]]
fn = [D["fruit_nerf / SIFT"]["summary"][K],    D["fruit_nerf / ALIKED"]["summary"][K]]
x = np.arange(2); w = 0.34; gap = 0.012      # 2px-equivalent surface gap
for off, vals, c, lab in [(-w/2 - gap, fn, ORANGE, "fruit_nerf"),
                          (+w/2 + gap, zn, BLUE,   "fruit_zipnerf")]:
    b = ax.bar(x + off, vals, w, color=c, label=lab, zorder=3)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 0.12, f"{v:.2f}",
                ha="center", color=INK, fontsize=10.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("PSNR (dB), full eval set"); ax.set_ylim(0, 20.5)
ax.set_title("Final PSNR — all 337 frames, matched ds2", color=INK, fontsize=12, pad=12, loc="left")
ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor=INK2, ncol=2)
finish(ax); fig.tight_layout(); fig.savefig(f"{OUT}/psnr_final.pdf"); plt.close(fig)

# ---------- Fig 3: semantic F1 (the metric that did NOT move) ----------
KF = "Eval Images Metrics Dict (all images)/semantic_f1"
fig, ax = plt.subplots(figsize=(7.4, 4.2))
zn = [D["fruit_zipnerf / SIFT"]["summary"][KF], D["fruit_zipnerf / ALIKED"]["summary"][KF]]
fn = [D["fruit_nerf / SIFT"]["summary"][KF],    D["fruit_nerf / ALIKED"]["summary"][KF]]
for off, vals, c, lab in [(-w/2 - gap, fn, ORANGE, "fruit_nerf"),
                          (+w/2 + gap, zn, BLUE,   "fruit_zipnerf")]:
    b = ax.bar(x + off, vals, w, color=c, label=lab, zorder=3)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 0.006, f"{v:.3f}",
                ha="center", color=INK, fontsize=10.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("semantic F1, full eval set"); ax.set_ylim(0, 0.78)
ax.set_title("Semantic F1 — the backbone gain does not reach the semantic head",
             color=INK, fontsize=12, pad=12, loc="left")
ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor=INK2, ncol=2)
finish(ax); fig.tight_layout(); fig.savefig(f"{OUT}/semantic_f1.pdf"); plt.close(fig)

print("figures written")
