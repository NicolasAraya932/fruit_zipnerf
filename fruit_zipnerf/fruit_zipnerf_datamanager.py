"""
FruitZipNerf DataManager.

Reuses FruitNeRF's data stack unchanged -- it already reads the frozen dataparser
contract (``load_frozen_contract=True``) and exposes the ``binary_img`` modality as
``batch["fruit_mask"]`` -- and only adds the one key ZipNeRF's loss expects
(``batch["rgb"]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from nerfstudio.cameras.rays import RayBundle

from fruit_nerf.data.fruit_datamanager import FruitDataManager, FruitDataManagerConfig


@dataclass
class FruitZipNerfDataManagerConfig(FruitDataManagerConfig):
    """Config for the FruitZipNerf DataManager."""

    _target: Type = field(default_factory=lambda: FruitZipNerfDataManager)


class FruitZipNerfDataManager(FruitDataManager):
    """FruitNeRF's data manager with ZipNeRF's expected ``rgb`` key added."""

    config: FruitZipNerfDataManagerConfig

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)
        batch["rgb"] = batch["image"].to(self.device)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_eval(step)
        batch["rgb"] = batch["image"].to(self.device)
        return ray_bundle, batch
