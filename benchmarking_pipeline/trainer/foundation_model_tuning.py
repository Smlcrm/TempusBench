# Need to rework this for any foundation model - we don't train, so that's
# an important rewrite.

import itertools
from ..models.base_model import BaseModel
from typing import Dict, List
import numpy as np
import pandas as pd
import re

class FoundationModelTuner:
  def __init__(self, model_class, hyperparameter_ranges: Dict[str, List], use_exog: str):
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
    match = re.match(r'(\d+)?[A-Z]', freq_str)
    if match:
        return int(match.group(1)) if match.group(1) else 1
    else:
        raise ValueError(f"Invalid frequency string: {freq_str}")

  def _get_target_data(self, dataset_split):
        """
        Extract target data from a dataset split.
        
        Args:
            dataset_split: Dataset split containing targets and timestamps
            
        Returns:
            numpy.ndarray: Target data as a 2D array with shape (time_steps, num_targets)
        """
        if dataset_split.targets is not None:
            # In the new structure, targets is a pandas DataFrame
            # with rows as time steps and columns as target series
            if hasattr(dataset_split.targets, 'values'):
                # Convert DataFrame to numpy array
                return dataset_split.targets.values
            elif isinstance(dataset_split.targets, np.ndarray):
                # Already a numpy array
                return dataset_split.targets
            elif isinstance(dataset_split.targets, list):
                # Convert list to numpy array
                return np.array(dataset_split.targets)
            else:
                raise ValueError(f"Unexpected targets format: {type(dataset_split.targets)}")
        else:
            raise ValueError("No targets available in dataset split")

  def hyperparameter_grid_search_several_time_series(self, list_of_time_series_datasets):
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
      for hyperparameter_setting in itertools.product(*self.hyperparameter_ranges.values()):
        current_hyperparameter_dict = dict()
        for key_value_index in range(len(hyperparameter_names)):

          # For the current chosen hyperparameter combination,
          # Create a dictionary associating hyperparameter names to the 
          # Appropriate hyperparameter value
          current_hyperparameter_dict[hyperparameter_names[key_value_index]] = hyperparameter_setting[key_value_index]
          print(f"Current hyperparameter dict: {current_hyperparameter_dict}")
          
        # Change the model's hyperparameter values
        self.model_class.set_params(**current_hyperparameter_dict)
        # Train a new model
        
        # Handle datasets with or without exogenous features
        if time_series_dataset.train.features is not None:
            print(f"DEBUG: train.features columns: {time_series_dataset.train.features.columns}")
            print(f"DEBUG: train.features head:\n{time_series_dataset.train.features.head()}")
        else:
            print(f"DEBUG: train.features is None, using train.targets")
            print(f"DEBUG: train.targets shape: {time_series_dataset.train.targets.shape if time_series_dataset.train.targets is not None else 'None'}")
        
        # Get target data from raw arrays (no column names needed)
        target = self._get_target_data(time_series_dataset.train)
        validation_series = self._get_target_data(time_series_dataset.validation)
        
        start_date = time_series_dataset.metadata["start"]
        freq_str = time_series_dataset.metadata["freq"]
        first_capital_letter_finder = re.search(r'[A-Z]', freq_str)
        freq = first_capital_letter_finder.group()
        freq_coefficient = self._extract_number_before_capital(freq_str)
        freq_offset = pd.tseries.frequencies.to_offset(freq)
        x_start_date = pd.to_datetime(start_date)
        y_start_date = x_start_date + (freq_coefficient * len(target) * freq_offset)
        y_context_timestamps = time_series_dataset.train.timestamps
        y_target_timestamps = time_series_dataset.validation.timestamps
        
        # For foundation models, we need to call train() to initialize the model
        # First update the model parameters for this hyperparameter setting
        trained_model = self.model_class
        trained_model.set_params(**current_hyperparameter_dict)
        trained_model.train(y_context=target, y_target=validation_series, y_start_date=y_start_date)
        
        # Get validation losses over every chunk
        current_train_loss = 0
        for time_series_dataset_from_all in list_of_time_series_datasets:
          target = self._get_target_data(time_series_dataset_from_all.train)
          validation_series = self._get_target_data(time_series_dataset_from_all.validation)
          #print(f"Time series dataset from all datasets\n\n:{time_series_dataset_from_all.metadata}")
          # Handle different model types with different predict method signatures
          
          # Get frequency from the dataset
          freq = time_series_dataset_from_all.metadata['freq']
          model_predictions = trained_model.predict(y_context=target, y_target=validation_series, y_context_timestamps=time_series_dataset_from_all.train.timestamps, y_target_timestamps=time_series_dataset_from_all.validation.timestamps, freq=freq)
          
          # Get validation targets for loss computation
          validationum_targets = self._get_target_data(time_series_dataset_from_all.validation)
          train_targets = self._get_target_data(time_series_dataset_from_all.train)
          train_loss = trained_model.compute_loss(validationum_targets, model_predictions, y_train=train_targets)
          loss_val = train_loss[trained_model.training_loss]
          # Aggregate per-target arrays to scalar
          if isinstance(loss_val, np.ndarray):
            loss_val = float(np.mean(loss_val))
          current_train_loss += loss_val
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
    print(f"List of hyperparameters per validation score: {list_of_hyperparameters_per_validation_score}")

    min_score = min(list_of_validation_scores)
    min_index = list_of_validation_scores.index(min_score)

    # Select the corresponding hyperparameters
    best_hyperparameters_overall = list_of_hyperparameters_per_validation_score[min_index]

    return min_score, best_hyperparameters_overall

  def final_evaluation(self, best_hyperparamters: Dict[str, int], list_of_time_series_datasets):
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
    self.model_class.set_params(**best_hyperparamters)
    results_dict = None
    for time_series_dataset in list_of_time_series_datasets:
      # Get train and validation data using helper method
      train_data = self._get_target_data(time_series_dataset.train)
      validation_data = self._get_target_data(time_series_dataset.validation)
      
      train_val_split = np.concatenate([train_data, validation_data])
      # Preserve 2D shape: rows=time steps, columns=targets
      target = train_val_split
      y_start_date = time_series_dataset.test.timestamps[0]
      # Combine train and validation timestamps for context
      train_val_timestamps = np.concatenate([
            time_series_dataset.train.timestamps,
            time_series_dataset.validation.timestamps
        ])
      # Handle different model types with different train method signatures
      trained_model = self.model_class
      # For foundation models, we need to call train() to initialize the model
      trained_model.train(y_context=target, y_target=validation_data, y_start_date=y_start_date)
            
      # Handle different model types with different predict method signatures
      test_targets = self._get_target_data(time_series_dataset.test)
      # Get frequency from the dataset
      freq = time_series_dataset.metadata['freq']
      predictions = trained_model.predict(y_context=target, y_target=test_targets, y_context_timestamps=train_val_timestamps, y_target_timestamps=time_series_dataset.test.timestamps, freq=freq)

      train_loss_dict = trained_model.compute_loss(test_targets, predictions, y_train=train_data)
      if results_dict is None:
        # Reduce arrays to scalars via mean for consistent aggregation
        results_dict = {k: (float(np.mean(v)) if isinstance(v, np.ndarray) else float(v)) for k, v in train_loss_dict.items()}
      else:
        # aggregates dict with reduction of arrays
        for key in results_dict:
          val = train_loss_dict[key]
          if isinstance(val, np.ndarray):
            val = float(np.mean(val))
          else:
            val = float(val)
          results_dict[key] += val

    results_dict = {key: float(results_dict[key]/len(list_of_time_series_datasets)) for key in results_dict}
    return results_dict
      
    