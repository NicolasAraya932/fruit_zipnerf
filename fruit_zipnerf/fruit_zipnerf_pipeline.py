"""FruitZipNerf pipeline: ZipNeRF's pipeline with the Fruit datamanager/model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManagerConfig
from nerfstudio.models.base_model import ModelConfig

from zipnerf_ns.zipnerf_pipeline import ZipNerfPipeline, ZipNerfPipelineConfig

from fruit_zipnerf.fruit_zipnerf_datamanager import FruitZipNerfDataManagerConfig
from fruit_zipnerf.fruit_zipnerf_model import FruitZipNerfModelConfig


@dataclass
class FruitZipNerfPipelineConfig(ZipNerfPipelineConfig):
    """Config for the FruitZipNerf pipeline."""

    _target: Type = field(default_factory=lambda: FruitZipNerfPipeline)
    datamanager: DataManagerConfig = field(default_factory=FruitZipNerfDataManagerConfig)
    model: ModelConfig = field(default_factory=FruitZipNerfModelConfig)


class FruitZipNerfPipeline(ZipNerfPipeline):
    """Inherits ZipNerfPipeline unchanged; only the configured types differ."""

    config: FruitZipNerfPipelineConfig
