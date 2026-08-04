"""
FruitZipNerf model: ZipNeRF geometry with FruitNeRF's binary semantic head.

This is FruitNeRF with its Nerfacto backbone replaced by ZipNeRF. The semantic
head hangs off ZipNeRF's density bottleneck -- the structural analog of the
`density_embedding` tap in `fruit_nerf/fruit_field.py` -- so the supervision
signal and the export contract are unchanged while the geometry comes from
ZipNeRF's anti-aliased multisampled hash grid.

Two outputs exist purely to keep the InvNeRF-Seg pipeline working unchanged:
  * ``semantics``  -- weight-composited binary fruit logits (BCE-supervised).
  * ``density``    -- the per-sample density of the final NeRF level, matching
                      the ``[N, S, 1]`` key that FruitModel emits for
                      ``generate_inv_nerf_radiance_fields_cloud``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
from torch import nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils import colormaps

from zipnerf_ns.zipnerf_model import ZipNerfModel, ZipNerfModelConfig


@dataclass
class FruitZipNerfModelConfig(ZipNerfModelConfig):
    """Config for FruitZipNerf."""

    _target: Type = field(default_factory=lambda: FruitZipNerfModel)

    semantic_loss_weight: float = 1.0
    """Weight on the binary cross-entropy fruit/background loss."""
    num_semantic_classes: int = 1
    """Binary foreground/background, matching FruitNeRF."""
    num_layers_semantic: int = 2
    """Hidden layers in the semantic head."""
    hidden_dim_semantics: int = 64
    """Width of the semantic head's hidden layers."""
    pass_semantic_gradients: bool = False
    """If False (default, as in FruitNeRF) semantics do not shape geometry."""
    semantic_threshold: float = 0.5
    """Sigmoid threshold used for the colormap and reported semantic metrics.

    Matches the export gate (``--semantic-threshold``) so training-time metrics
    and the exported cloud agree on what counts as fruit.
    """


class FruitZipNerfModel(ZipNerfModel):
    """ZipNeRF with FruitNeRF's semantic head."""

    config: FruitZipNerfModelConfig

    def gin_bindings(self) -> List[str]:
        """Enable the semantic head on the NerfMLP only.

        The proposal MLPs stay untouched: they exist to place samples, and
        FruitNeRF likewise reads semantics only from the main field.
        """
        return [
            f"NerfMLP.num_semantic_classes = {self.config.num_semantic_classes}",
            f"NerfMLP.num_layers_semantic = {self.config.num_layers_semantic}",
            f"NerfMLP.hidden_dim_semantics = {self.config.hidden_dim_semantics}",
            f"NerfMLP.pass_semantic_gradients = {self.config.pass_semantic_gradients}",
        ]

    def populate_modules(self):
        super().populate_modules()
        self.binary_cross_entropy_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)

        # The last entry of ray_history is the final NeRF level; the proposal
        # levels carry no semantics and no meaningful colour.
        final = outputs["ray_history"][-1]
        weights = final["weights"]

        # Per-sample density, kept in the [..., S, 1] shape InvNeRF-Seg's cloud
        # export expects (see the NOTE in fruit_nerf.py's get_outputs).
        density = final["density"]
        outputs["density"] = density if density.dim() == weights.dim() + 1 else density[..., None]

        semantics = final["semantics"]
        semantic_weights = weights if self.config.pass_semantic_gradients else weights.detach()
        # Alpha-composite the per-sample logits with the same weights used for RGB.
        outputs["semantics"] = (semantic_weights[..., None] * semantics).sum(dim=-2)

        labels = torch.sigmoid(outputs["semantics"].detach())
        outputs["semantics_colormap"] = torch.heaviside(
            labels - self.config.semantic_threshold, torch.tensor(0.0, device=labels.device)
        ).float()
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.binary_cross_entropy_loss(
            outputs["semantics"], batch["fruit_mask"].to(self.device)
        )
        return loss_dict

    @staticmethod
    def _semantic_metrics(logits, target, threshold: float = 0.5, prefix: str = "semantic"):
        """Precision/recall/F1/IoU on the binary fruit mask at `threshold`."""
        pred = (torch.sigmoid(logits) > threshold).float()
        gt = (target > 0.5).float()
        tp = (pred * gt).sum()
        fp = (pred * (1 - gt)).sum()
        fn = ((1 - pred) * gt).sum()
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        return {
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1": 2 * precision * recall / (precision + recall + eps),
            f"{prefix}_iou": tp / (tp + fp + fn + eps),
        }

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if "fruit_mask" in batch and "semantics" in outputs:
            fruit_mask = batch["fruit_mask"].to(self.device)
            metrics_dict.update(
                self._semantic_metrics(outputs["semantics"], fruit_mask, self.config.semantic_threshold)
            )
            # Logged unweighted beside the weighted term in loss_dict, so a moving
            # semantic curve can be attributed to the head rather than the weight.
            metrics_dict["semantic_bce_unweighted"] = self.binary_cross_entropy_loss(
                outputs["semantics"], fruit_mask
            )
        return metrics_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Chunked full-image inference.

        Overridden because ZipNeRF returns ``renderings`` and ``ray_history`` as
        lists of dicts, which nerfstudio's base implementation would try to
        ``torch.cat`` across chunks. Those keys are training-only, so they are
        dropped here.
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists: Dict[str, List[torch.Tensor]] = {}
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx, end_idx = i, i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # forward(), not get_outputs(): the collider populates nears/fars, which
            # ZipNeRF's ray warp requires. get_outputs() alone leaves them None.
            outputs = self.forward(ray_bundle=ray_bundle)
            for key, value in outputs.items():
                if not isinstance(value, torch.Tensor):
                    continue
                # Per-sample tensors (e.g. density) do not tile back into an image.
                if value.shape[0] != ray_bundle.origins.shape[0]:
                    continue
                outputs_lists.setdefault(key, []).append(value)
        return {
            key: torch.cat(vals).view(image_height, image_width, -1)
            for key, vals in outputs_lists.items()
        }

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        images_dict["semantics_colormap"] = colormaps.apply_colormap(
            torch.sigmoid(outputs["semantics"])
        )
        if "fruit_mask" in batch:
            for key, value in self._semantic_metrics(
                outputs["semantics"], batch["fruit_mask"].to(self.device), self.config.semantic_threshold
            ).items():
                metrics_dict[key] = float(value)
        return metrics_dict, images_dict
