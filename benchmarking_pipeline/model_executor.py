"""
Model executor for isolated benchmarking runs.

This module provides the ModelExecutor class that runs individual models
in isolated conda environments to avoid dependency conflicts.
It handles model loading, hyperparameter tuning, and evaluation.
"""

import os
import pdb
import textwrap
import argparse
import importlib
import subprocess
import numpy as np
import pickle, csv, json, yaml

from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from benchmarking_pipeline.utils.logger import Logger
from benchmarking_pipeline.models.base_model import BaseModel
from benchmarking_pipeline.models.model_router import ModelRouter
from benchmarking_pipeline.utils.config_validator import load_config
from benchmarking_pipeline.pipeline.hyperparameter_tuning import HyperparameterTuner

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelExecutor:
    def __init__(self, config_path: str, run_dir: str, datasets_dir: str, logger=None, tf_logger=None):
        self.config = load_config(config_path)
        self.config_path = config_path
        self.run_dir = run_dir
        self.datasets_dir = datasets_dir
        self.logger = logger
        self.tf_logger = tf_logger

    def _generate_hyperparameter_tuning_script(self,
            model_name: str,
            context_steps: int,
            train_steps: int,
            validate_steps: int,
            hyper_grid: list[dict]):

        script = textwrap.dedent(f"""
            from benchmarking_pipeline.models.model_router import ModelRouter
            from benchmarking_pipeline.pipeline.data_loader import DataLoader

            data_loader = DataLoader(
                config_path={self.config_path},
                datasets_dir={self.datasets_dir},
                run_dir={run_dir}
            )

            config = data_loader.config
            model_config = config['model'][{model_name}]

            for dataset in data_loader.generate_dataset_split(
                {context_steps},{train_steps},{validate_steps}):
                timestamps = dataset.timestamps
                target = dataset.target

                # Extract start and end indices for context, train, and validate from dataset
                cstart = dataset.context.start;    cend = dataset.context.end
                tstart = dataset.train.start;      tend = dataset.train.end
                vstart = dataset.validation.start; vend = dataset.validation.end
                freq = dataset.metadata.freq

                model = ModelRouter(name={model_name}, ndim=context_set.ndim)

                for params in {hyper_grid}:
                    trained_model = model.train(
                        y_context: target[:, cstart:cend], # context
                        y_target: target[:, tstart:tend], # train
                        timestamps_context: timestamps[cstart:cend],
                        timestamps_target: timestamps[tstart:tend],
                        freq: freq,
                    )

                    results = trained_model.predict(
                        y_context: target[:, cstart:tend], # context + train
                        timestamps_context: timestamps[cstart:tend],
                        timestamps_target: timestamps[vstart:vend], # validate
                        freq: freq,
                    )

                    loss = trained_model.compute_loss(results, target[:, vstart:vend])
        """)
        return script

    def optimize_hyperparameters(self,
            context_steps: int,
            train_steps: int,
            validate_steps: int):
        """
        Run model evaluation with the provided hyperparameters.

        Args:
            hyperparameters: Dictionary containing hyperparameters for each model
        """
        logging = self.logger is not None

        # This method needs to be implemented based on the actual requirements
        # For now, it's a placeholder that handles the basic structure
        for model_name, hyperparameters in self.config["model"].items():
            # INSERT_YOUR_CODE
            hyper_grid = []
            if hyperparameters:
                keys = list(hyperparameters.keys())
                values_lists = [hyperparameters[k] for k in keys]
                # Cartesian product: build list of dicts for each combination
                from itertools import product
                hyper_grid = [dict(zip(keys, values)) for values in product(*values_lists)]
            else:
                hyper_grid = [None]

            if logging:
                self.logger.info(f"Preparing to run model: {model_name}")
                self.logger.debug(f"Hyperparameters for {model_name}: {hyperparameters}")
                if not hyperparameters:
                    self.logger.warning(f"{model_name} has no parameters, using empty dict, probably a foundation model")

            # Create Conda Env
            conda_env = CondaEnvManager(
                name = f"benchmark.{model_name}",
                python = "3.11",
                requirements_path = model.requirements_path
            )

            conda_env.run(
                script=self._generate_hyperparameter_tuning_script(
                    model_name=model_name,
                    context_steps=context_steps,
                    train_steps=train_steps,
                    validate_steps=validate_steps,
                    hyper_grid=hyper_grid
                )
            )