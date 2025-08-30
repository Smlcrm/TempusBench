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

    def hyperparameter_grid_search_several_time_series(
        self, list_of_time_series_datasets
    ):
        """
        Perform grid search over hyperparameter combinations for multiple time series datasets,
        then identify the model with the best average validation performance across the datasets.

        Args:
            list_of_time_series_datasets: A list of Dataset objects, each containing 'train', 'validation', and 'test' splits.

        Returns:
            best_arima_model_overall: The model instance that achieved the best average validation loss across datasets.
            best_hyperparameters_overall: The hyperparameter values (as a tuple) that correspond to the best model.

        """
        list_of_validation_scores = []
        list_of_hyperparameters_per_validation_score = []

        for time_series_dataset in list_of_time_series_datasets:
            best_hyperparameters = 0
            lowest_train_loss = float("inf")
            print("Starting hyperparameter grid search on single time series!")

            # List of hyperparameter names
            hyperparameter_names = list(self.hyperparameter_ranges.keys())
            # We iterate over all possible hyperparameter value combinations
            for hyperparameter_setting in itertools.product(
                *self.hyperparameter_ranges.values()
            ):
                current_hyperparameter_dict = dict()
                for key_value_index in range(len(hyperparameter_names)):

                    # For the current chosen hyperparameter combination,
                    # Create a dictionary associating hyperparameter names to the
                    # Appropriate hyperparameter value
                    current_hyperparameter_dict[
                        hyperparameter_names[key_value_index]
                    ] = hyperparameter_setting[key_value_index]
                    print(f"Current hyperparameter dict: {current_hyperparameter_dict}")

                # Change the model's hyperparameter values
                self.model_class.set_params(**current_hyperparameter_dict)
                # Train a new model
                # Debug info for numpy arrays
                if hasattr(time_series_dataset.train.targets, "columns"):
                    print(
                        f"DEBUG: train.targets columns: {time_series_dataset.train.targets.columns}"
                    )
                    print(
                        f"DEBUG: train.targets head:\n{time_series_dataset.train.targets.head()}"
                    )
                else:
                    print(
                        f"DEBUG: train.targets shape: {time_series_dataset.train.targets.shape}"
                    )
                    print(
                        f"DEBUG: train.targets first few values: {time_series_dataset.train.targets[:5]}"
                    )

                # Handle multivariate vs univariate data - work with raw target arrays
                if (
                    hasattr(time_series_dataset.train.targets, "shape")
                    and len(time_series_dataset.train.targets.shape) > 1
                ):
                    # Multivariate case - use all targets
                    target = time_series_dataset.train.targets
                    validation_series = time_series_dataset.validation.targets
                else:
                    # Univariate case - use targets directly (raw arrays)
                    target = time_series_dataset.train.targets
                    validation_series = time_series_dataset.validation.targets



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
                    freq_coefficient * len(target) * freq_offset
                )
                y_context_timestamps = time_series_dataset.train.timestamps
                y_target_timestamps = time_series_dataset.validation.timestamps
                # Handle different model types with different train method signatures
                if (
                    hasattr(self.model_class, "__class__")
                    and "Prophet" in self.model_class.__class__.__name__
                ):
                    # Prophet model only accepts y_context, x_context, y_start_date
                    trained_model = self.model_class.train(
                        y_context=target, y_context_timestamps=y_context_timestamps
                    )
                elif (
                    hasattr(self.model_class, "__class__")
                    and "DeepAR" in self.model_class.__class__.__name__
                ):
                    # DeepAR model doesn't accept timestamp parameters
                    trained_model = self.model_class.train(
                        y_context=target,
                        y_target=validation_series,
                        x_start_date=x_start_date,
                    )
                elif (
                    hasattr(self.model_class, "__class__")
                    and "RandomForest" in self.model_class.__class__.__name__
                ):
                    # Random Forest model uses timestamp features
                    trained_model = self.model_class.train(
                        y_context=target,
                        y_target=validation_series,
                        y_context_timestamps=y_context_timestamps,
                        y_target_timestamps=y_target_timestamps,
                    )
                else:
                    # Default case for other models
                    trained_model = self.model_class.train(
                        y_context=target,
                        y_target=validation_series,
                        y_context_timestamps=y_context_timestamps,
                        y_target_timestamps=y_target_timestamps,
                        x_start_date=x_start_date,
                    )

                # Get validation losses over every chunk
                current_train_loss = 0
                for time_series_dataset_from_all in list_of_time_series_datasets:
                    # Handle multivariate vs univariate data for validation - work with raw arrays
                    if (
                        hasattr(time_series_dataset_from_all.train.targets, "shape")
                        and len(time_series_dataset_from_all.train.targets.shape) > 1
                    ):
                        # Multivariate case - use all targets
                        target = time_series_dataset_from_all.train.targets
                        validation_series = (
                            time_series_dataset_from_all.validation.targets
                        )
                    else:
                        # Univariate case - use targets directly (raw arrays)
                        target = time_series_dataset_from_all.train.targets
                        validation_series = (
                            time_series_dataset_from_all.validation.targets
                        )

                    # print(f"Time series dataset from all datasets\n\n:{time_series_dataset_from_all.metadata}")
                    # Handle different model types with different predict method signatures
                    if (
                        hasattr(self.model_class, "__class__")
                        and "DeepAR" in self.model_class.__class__.__name__
                    ):
                        # DeepAR model doesn't accept timestamp parameters
                        model_predictions = trained_model.predict(
                            y_context=target, y_target=validation_series
                        )
                    elif (
                        hasattr(self.model_class, "__class__")
                        and "RandomForest" in self.model_class.__class__.__name__
                    ):
                        # Random Forest model uses timestamp features
                        model_predictions = trained_model.predict(
                            y_context=target,
                            y_target=validation_series,
                            y_context_timestamps=time_series_dataset_from_all.train.timestamps,
                            y_target_timestamps=time_series_dataset_from_all.validation.timestamps,
                        )
                    else:
                        # All other models (including Prophet) use y_target_timestamps
                        model_predictions = trained_model.predict(
                            y_context=target,
                            y_target=validation_series,
                            y_target_timestamps=time_series_dataset_from_all.validation.timestamps,
                        )
                    # Handle multivariate vs univariate loss computation - work with raw arrays
                    # Extract only the first forecast_horizon values from validation data for proper comparison
                    validation_targets = time_series_dataset_from_all.validation.targets
                    forecast_horizon = (
                        model_predictions.shape[0]
                        if hasattr(model_predictions, "shape")
                        and len(model_predictions.shape) > 1
                        else len(model_predictions)
                    )

                    # Extract only the first forecast_horizon values from validation data
                    if (
                        hasattr(validation_targets, "shape")
                        and len(validation_targets.shape) > 1
                    ):
                        # Multivariate case - use first forecast_horizon targets
                        validation_subset = validation_targets[:forecast_horizon]
                    else:
                        # Univariate case - extract first forecast_horizon values
                        validation_subset = validation_targets[:forecast_horizon]

                    # Convert to numpy array to ensure compatibility with metrics
                    if hasattr(validation_subset, "values"):
                        validation_subset = validation_subset.values
                    elif hasattr(validation_subset, "to_numpy"):
                        validation_subset = validation_subset.to_numpy()

                    # Ensure validation_subset is 1D for univariate case
                    if validation_subset.ndim == 2 and validation_subset.shape[1] == 1:
                        validation_subset = validation_subset.flatten()

                    # Pass training data for MASE calculation
                    train_loss = trained_model.compute_loss(
                        validation_subset,
                        model_predictions,
                        y_train=time_series_dataset.train.targets,
                    )
                    current_train_loss += train_loss[self.model_class.training_loss]
                # Average validation losses over the chunks
                current_train_loss /= len(list_of_time_series_datasets)
                if current_train_loss < lowest_train_loss:
                    # For this hyper parameter setting, we have a lower average validation loss
                    print(f"Lowest average train loss so far {current_train_loss}")
                    lowest_train_loss = current_train_loss
                    best_hyperparameters = hyperparameter_setting
            # After every possible hyperparameter setting, for the model trained on this chunk,
            # we choose the hyperparameters that give us the lowest validation scores across all chunks
            validation_score, hyperparameters = lowest_train_loss, best_hyperparameters
            list_of_validation_scores.append(validation_score)
            list_of_hyperparameters_per_validation_score.append(hyperparameters)
        print(f"List of validation scores: {list_of_validation_scores}")
        print(
            f"List of hyperparameters per validation score: {list_of_hyperparameters_per_validation_score}"
        )

        min_score = min(list_of_validation_scores)
        min_index = list_of_validation_scores.index(min_score)

        # Select the corresponding hyperparameters
        best_hyperparameters_overall = list_of_hyperparameters_per_validation_score[
            min_index
        ]

        return min_score, best_hyperparameters_overall

    def final_evaluation(
        self, best_hyperparamters: Dict[str, int], list_of_time_series_datasets
    ):
        """
        Train a model using the best hyperparameters on the combined train and validation splits,
        then evaluate its performance on the test split for each dataset.

        Args:
            best_hyperparamters: A dictionary mapping hyperparameter names to their best-tuned values.
            list_of_time_series_datasets: A list of Datasets, each containing 'train', 'validation', and 'test' splits.
            use_exog: Whether to use exogeneous variables or not

        Returns:
            results_dict: A dictionary mapping each loss metric name to its average value across datasets on the test split.
        """
        print(f"Best hyperparameters: {best_hyperparamters}")

        # Convert numpy array to dictionary if needed
        if isinstance(best_hyperparamters, np.ndarray):
            hyperparameter_names = list(self.hyperparameter_ranges.keys())
            best_hyperparamters = {
                hyperparameter_names[i]: best_hyperparamters[i]
                for i in range(len(hyperparameter_names))
            }

        self.model_class.set_params(**best_hyperparamters)
        results_dict = None
        for time_series_dataset in list_of_time_series_datasets:
            # Handle multivariate vs univariate data for final evaluation - work with raw arrays
            if (
                hasattr(time_series_dataset.train.targets, "shape")
                and len(time_series_dataset.train.targets.shape) > 1
            ):
                # Multivariate case - ensure DataFrame then concatenate along time axis
                def to_df(x):
                    if isinstance(x, pd.DataFrame):
                        return x
                    arr = np.asarray(x)
                    cols = (
                        [f"target_{i}" for i in range(arr.shape[1])]
                        if arr.ndim == 2
                        else ["target_0"]
                    )
                    return pd.DataFrame(arr, columns=cols)

                train_df = to_df(time_series_dataset.train.targets)
                val_df = to_df(time_series_dataset.validation.targets)
                train_val_split = pd.concat([train_df, val_df], axis=0)
                target = train_val_split
            else:
                # Univariate case - concatenate raw arrays
                train_val_split = np.concatenate(
                    [
                        time_series_dataset.train.targets,
                        time_series_dataset.validation.targets,
                    ]
                )
                target = (
                    train_val_split.flatten()
                    if hasattr(train_val_split, "flatten")
                    else train_val_split
                )
            y_target_start_date = time_series_dataset.test.timestamps[0]
            # Combine train and validation timestamps for context
            train_val_timestamps = np.concatenate(
                [
                    time_series_dataset.train.timestamps,
                    time_series_dataset.validation.timestamps,
                ]
            )
            # Handle different model types with different train method signatures
            if (
                hasattr(self.model_class, "__class__")
                and "Prophet" in self.model_class.__class__.__name__
            ):
                # Prophet model only accepts y_context, x_context, y_start_date
                trained_model = self.model_class.train(
                    y_context=target, y_context_timestamps=train_val_timestamps
                )
            elif (
                hasattr(self.model_class, "__class__")
                and "DeepAR" in self.model_class.__class__.__name__
            ):
                # DeepAR model doesn't accept timestamp parameters
                trained_model = self.model_class.train(
                    y_context=target, y_target=time_series_dataset.test.targets
                )
            elif (
                hasattr(self.model_class, "__class__")
                and "RandomForest" in self.model_class.__class__.__name__
            ):
                # Random Forest model uses timestamp features
                trained_model = self.model_class.train(
                    y_context=target,
                    y_target=time_series_dataset.test.targets,
                    y_context_timestamps=train_val_timestamps,
                    y_target_timestamps=time_series_dataset.test.timestamps,
                )
            else:
                # Default case for other models
                trained_model = self.model_class.train(
                    y_context=target,
                    y_target=time_series_dataset.test.targets,
                    y_context_timestamps=train_val_timestamps,
                    y_target_timestamps=time_series_dataset.test.timestamps,
                )

            # Handle different model types with different predict method signatures
            if (
                hasattr(self.model_class, "__class__")
                and "DeepAR" in self.model_class.__class__.__name__
            ):
                # DeepAR model doesn't accept timestamp parameters
                predictions = trained_model.predict(
                    y_context=target, y_target=time_series_dataset.test.targets
                )
            elif (
                hasattr(self.model_class, "__class__")
                and "RandomForest" in self.model_class.__class__.__name__
            ):
                # Random Forest model uses timestamp features
                predictions = trained_model.predict(
                    y_context=target,
                    y_target=time_series_dataset.test.targets,
                    y_context_timestamps=train_val_timestamps,
                    y_target_timestamps=time_series_dataset.test.timestamps,
                )
            else:
                # All other models (including Prophet) use both y_context_timestamps and y_target_timestamps
                predictions = trained_model.predict(
                    y_context=target,
                    y_target=time_series_dataset.test.targets,
                    y_context_timestamps=train_val_timestamps,
                    y_target_timestamps=time_series_dataset.test.timestamps,
                )

            # Handle multivariate vs univariate loss computation for final evaluation - work with raw arrays
            # Extract only the first forecast_horizon values from test data for proper comparison
            test_targets = time_series_dataset.test.targets
            forecast_horizon = (
                predictions.shape[0]
                if hasattr(predictions, "shape") and len(predictions.shape) > 1
                else len(predictions)
            )

            # Extract only the first forecast_horizon values from test data
            if hasattr(test_targets, "shape") and len(test_targets.shape) > 1:
                # Multivariate case - use first forecast_horizon targets
                test_subset = test_targets[:forecast_horizon]
            else:
                # Univariate case - extract first forecast_horizon values
                test_subset = test_targets[:forecast_horizon]

            # Convert to numpy array to ensure compatibility with metrics
            if hasattr(test_subset, "values"):
                test_subset = test_subset.values
            elif hasattr(test_subset, "to_numpy"):
                test_subset = test_subset.to_numpy()

            # Ensure test_subset is 1D for univariate case
            if test_subset.ndim == 2 and test_subset.shape[1] == 1:
                test_subset = test_subset.flatten()

            # Pass training data for MASE calculation
            train_loss_dict = trained_model.compute_loss(
                test_subset, predictions, y_train=target
            )
            if results_dict is None:
                results_dict = train_loss_dict
            else:
                # aggregates dict
                results_dict = {
                    key: results_dict[key] + train_loss_dict[key]
                    for key in results_dict
                }

        results_dict = {
            key: float(results_dict[key] / len(list_of_time_series_datasets))
            for key in results_dict
        }
        return results_dict
