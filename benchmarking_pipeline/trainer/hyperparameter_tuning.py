import itertools
from ..models.base_model import BaseModel
from typing import Dict, List
import numpy as np
import pandas as pd
import re


class HyperparameterTuner:
    def __init__(
        self,
        model_class: BaseModel,
        hyperparameter_ranges: Dict[str, List],
        use_exog: str,
    ):
        """
        Initialize the hyperparameter tuner with a model class and a hyperparameter search space.

        Args:
            model_class: A model class that inherits from BaseModel
            hyperparameter_ranges: A dictionary where keys are hyperparameter names and values are lists of values to search over.
            use_exog: Whether to use exogeneous variables or not
            is_statsmodel: Whether this model is a statsmodel or not. Defaults to
                           True
        """
        self.model_class = model_class
        self.hyperparameter_ranges = hyperparameter_ranges
        self.use_exog = use_exog

    def _extract_number_before_capital(self, freq_str):
        match = re.match(r"(\d+)?[A-Z]", freq_str)
        if match:
            return int(match.group(1)) if match.group(1) else 1
        else:
            raise ValueError(f"Invalid frequency string: {freq_str}")

    def hyperparameter_grid_search(self, time_series_dataset):
        """
        Perform grid search over hyperparameter combinations for multiple time series datasets,
        then identify the model with the best average validation performance across the datasets.

        Args:
            time_series_dataset: A Dataset object, containing 'train', 'validation', and 'test' splits.

        Returns:
            best_arima_model_overall: The model instance that achieved the best average validation loss across datasets.
            best_hyperparameters_overall: The hyperparameter values (as a tuple) that correspond to the best model.

        """
        list_of_validation_scores = []
        list_of_hyperparameters_per_validation_score = []

        # Get validation losses over every chunk
        current_train_loss = 0

        opt_hyperparams = None
        opt_valid_loss = float("inf")

        print("Starting hyperparameter grid search!")

        # List of hyperparameter names
        hyperparameter_names = list(self.hyperparameter_ranges.keys())
        # We iterate over all possible hyperparameter value combinations
        for hyperparameter_setting in itertools.product(
            *self.hyperparameter_ranges.values()
        ):
            current_hyperparameter_dict = dict(
                zip(hyperparameter_names, hyperparameter_setting)
            )

            print(f"Current hyperparameter dict: {current_hyperparameter_dict}")

            # Change the model's hyperparameter values
            self.model_class.set_params(**current_hyperparameter_dict)
            # Train a new model

            y_train = time_series_dataset.train.targets
            y_valid = time_series_dataset.validation.targets

            timestamps_train = time_series_dataset.train.timestamps
            timestamps_valid = time_series_dataset.validation.timestamps

            timestamps_train = np.expand_dims(timestamps_train, axis=1)
            timestamps_valid = np.expand_dims(timestamps_valid, axis=1)

            start_date = time_series_dataset.metadata["start"]
            freq_str = time_series_dataset.metadata["freq"]
            first_capital_letter_finder = re.search(r"[A-Z]", freq_str)
            freq = first_capital_letter_finder.group()
            freq_coefficient = self._extract_number_before_capital(freq_str)

            # Handle deprecated frequency 'T' -> 'min'
            if freq == "T":
                freq = "min"
            freq_offset = pd.tseries.frequencies.to_offset(freq)
            x_start_date = pd.to_datetime(start_date)
            y_start_date = x_start_date + (
                freq_coefficient * len(y_train) * freq_offset
            )

            ############# TRAINING  #############
            trained_model = self.model_class.train(
                y_context=y_train,
                y_target=y_valid,
                timestamps_context=timestamps_train,
                timestamps_target=timestamps_valid,
                freq=freq,
            )

            # All other models (including Prophet) use y_target_timestamps
            y_pred = trained_model.predict(
                y_context=y_train,
                timestamps_context=timestamps_train,
                timestamps_target=timestamps_valid,
                freq=freq,
            )

            # Ensure predictions align with requested validation horizon via timestamps
            # Predictions should already match the provided `timestamps_valid` length.

            # Pass training data for MASE calculation
            train_loss = trained_model.compute_loss(
                y_valid,
                y_pred,
                y_train=y_train,
            )

            print("TRAINING LOSSES", self.model_class.training_loss)
            current_train_loss = train_loss[self.model_class.training_loss]

            # Average validation losses over the chunks
            if (current_train_loss < opt_valid_loss) or (opt_hyperparams is None):
                # For this hyper parameter setting, we have a lower average validation loss
                opt_valid_loss = current_train_loss
                opt_hyperparams = current_hyperparameter_dict
                print(f"Lowest average train loss so far {opt_valid_loss}")
                print(f"New best hyperparameter dict: {opt_hyperparams}")

        # After every possible hyperparameter setting, for the model trained on this chunk,
        # we choose the hyperparameters that give us the lowest validation scores across all chunks
        return opt_valid_loss, opt_hyperparams

    def final_evaluation(
        self, best_hyperparamters: Dict[str, int], time_series_dataset
    ):
        """
            Train a model using the best hyperparameters on the combined train and validation splits,
            then evaluate its performance on the test split for each dataset.

            Args:
                best_hyperparamters: A dictionary mapping hyperparameter names to their best-tuned values.
                time_series_dataset: A Dataset, containing 'train', 'validation', and 'test' splits.

        Returns:
            results_dict: A dictionary mapping each loss metric name to its average value across datasets on the test split.
        """

        print(f"Best hyperparameters: {best_hyperparamters}")

        self.model_class.set_params(**best_hyperparamters)

        results_dict = None

        y_train = time_series_dataset.train.targets
        y_val = time_series_dataset.validation.targets

        y_context = np.concatenate([y_train, y_val], axis=0)
        y_test = time_series_dataset.test.targets

        y_target_start_date = time_series_dataset.test.timestamps[0]

        timestamps_train = np.expand_dims(time_series_dataset.train.timestamps, axis=1)
        timestamps_val = np.expand_dims(
            time_series_dataset.validation.timestamps, axis=1
        )

        # Combine train and validation timestamps for context
        timestamps_context = np.concatenate([timestamps_train, timestamps_val], axis=0)
        timestamps_test = np.expand_dims(time_series_dataset.test.timestamps, axis=1)

        freq_str = time_series_dataset.metadata["freq"]
        first_capital_letter_finder = re.search(r"[A-Z]", freq_str)
        freq = first_capital_letter_finder.group()

        # Handle deprecated frequency 'T' -> 'min'
        if freq == "T":
            freq = "min"

        ############### TRAIN ###############

        # Default case for other models
        trained_model = self.model_class.train(
            y_context=y_context,
            y_target=y_test,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_test,
            freq=freq,
        )

        ############### TEST ###############

        y_pred = trained_model.predict(
            y_context=y_context,
            timestamps_context=timestamps_context,
            timestamps_target=timestamps_test,
            freq=freq,
        )

        forecast_horizon = y_pred.shape[0]
        y_test = y_test[:forecast_horizon, :]

        # Pass training data for MASE calculation
        final_metrics = trained_model.compute_loss(y_pred, y_test, y_train=y_context)

        return y_context, y_test, y_pred, final_metrics
