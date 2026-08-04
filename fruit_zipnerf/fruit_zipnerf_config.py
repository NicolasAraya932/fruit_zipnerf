"""
FruitZipNerf method registration.

Registers `fruit_zipnerf` with the nerfstudio CLI: FruitNeRF's semantic
supervision on a ZipNeRF backbone, trained through the frozen dataparser
contract so it shares one canonical normalized frame with Nerfacto/InvNeRF and
stays commensurable with the InvNeRF-Seg occupancy comparison.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from fruit_zipnerf.fruit_zipnerf_datamanager import FruitZipNerfDataManagerConfig
from fruit_zipnerf.fruit_zipnerf_model import FruitZipNerfModelConfig
from fruit_zipnerf.fruit_zipnerf_pipeline import FruitZipNerfPipelineConfig

# The patched nerfstudio dataparser loads metadata/dataparser/contract.json with
# load_frozen_contract=True and exposes image_modalities['binary_img'] as the
# fruit mask. downscale_factor is pinned rather than auto-selected: the contract
# pins image paths, and auto-select can freeze full-res and OOM.
fruit_zipnerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="fruit_zipnerf",
        steps_per_eval_batch=500,
        steps_per_eval_image=5000,
        steps_per_save=5000,
        max_num_iterations=25000,
        mixed_precision=True,
        log_gradients=False,
        pipeline=FruitZipNerfPipelineConfig(
            datamanager=FruitZipNerfDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    downscale_factor=2,
                    orientation_method="none",
                    center_method="poses",
                ),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=8192,
            ),
            model=FruitZipNerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                gin_file=["configs/360.gin"],
                proposal_weights_anneal_max_num_iters=1000,
            ),
        ),
        optimizers={
            "model": {
                "optimizer": AdamOptimizerConfig(lr=8e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    warmup_steps=1000, lr_final=1e-3, max_steps=25000
                ),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="FruitNeRF's binary semantic head on a ZipNeRF backbone, trained through the frozen dataparser contract.",
)
