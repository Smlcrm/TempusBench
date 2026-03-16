# TODO - Benchmarking Pipeline Improvements

This document tracks future issues and improvements that need to be addressed in the benchmarking pipeline.

## Model Testing & Evaluation

### 1. Test Models Using Data Including Covariates
- **Status**: Not Started
- **Priority**: Medium
- **Description**: Currently, most models are tested on univariate time series data. Need to extend testing to include datasets with exogenous variables (covariates).
- **Tasks**:
  - [ ] Identify datasets with covariates/exogenous variables
  - [ ] Test if data loading pipeline properly handle covariate data
  - [ ] Modify model interfaces to properly handle exogenous variables
  - [ ] Test all models with covariate data
  - [ ] Compare performance with and without covariates

### 2. Test Multivariate Models
- **Status**: Not Started
- **Priority**: High
- **Description**: Extend the pipeline to support multivariate time series forecasting where multiple target variables are predicted simultaneously.
- **Tasks**:
  - [ ] Research and implement multivariate versions of existing models
  - [ ] Update data structures to handle multiple target variables
  - [ ] Modify evaluation metrics for multivariate predictions
  - [ ] Extend all models for multivariate forecasting

  ### 3. Runaway Predictions for Random Forest
- **Status**: Not Started
- **Priority**: Medium
- **Description**: The predictions for random forest increase over time due to iterative prediction.
- **Tasks**:
  - [ ] Store scaler and inverse transform
  - [ ] Clip predictions to keep within normalized space

### 3. Error Handling and Robustness
- **Status**: Not Started
- **Priority**: Medium
- **Description**: Improve error handling and make the pipeline more robust.
- **Tasks**:
  - [ ] Implement graceful degradation for model failures
  - [ ] Add input validation for all model parameters

### 4. Documentation and Examples
- **Status**: Not Started
- **Priority**: Low
- **Description**: Improve documentation and provide more examples.
- **Tasks**:
  - [ ] Add comprehensive API documentation
  - [ ] Create tutorials for different use cases
  - [ ] Add code examples for each model type
  - [ ] Document best practices and common pitfalls

## Infrastructure

### 5. Configuration Management
- **Status**: Not Started
- **Priority**: Medium
- **Description**: Improve configuration management for experiments.
- **Tasks**:
  - [ ] Create standardized configuration templates
  - [ ] Add configuration validation
  - [ ] Update configuration docs

---

## Notes

- Priority levels: High, Medium, Low
- Status: Not Started, In Progress, Completed, Blocked
- This document should be updated as new issues are identified and existing ones are resolved. 