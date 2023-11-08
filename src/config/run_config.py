"""This module indicates the detailed configurations of our framework"""

from __future__ import annotations
from typing import Dict, Any
from config.base_config import BaseConfig
from config.data_config import DataConfig
from config.model_config import ModelConfig
from config.train_config import TrainConfig


class RunConfig(BaseConfig):
    """The CDR Configuration class"""
    # chemical_string = "Chemical"
    # disease_string = "Disease"
    # adjacency_rel = "node"
    # root_rel = "root"

    required_arguments = {"experiment", "experiment_dir", "model"}

    def __init__(self,
                 model: ModelConfig,
                 experiment: str,
                 experiment_dir: str):
        self.model = model
        self.experiment = experiment
        self.experiment_dir = experiment_dir

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        cls.check_required(d)
        model = ModelConfig.from_dict(d['model'])
        config = RunConfig(model=model,
                           experiment=d['experiment'],
                           experiment_dir=d['experiment_dir'])
        return config
