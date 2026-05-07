# Time Series Tasks

This directory contains a comprehensive collection of time series tasks for benchmarking forecasting models. The tasks are organized into two main categories: univariate and multivariate time series.

**Dataset CSV schema:** each task CSV uses columns `variable_name`, `timestamps`, `values`, and `variable_type` (`target` or `covariate`). The `timestamps` and `values` cells hold JSON arrays; timestamps are ISO 8601 strings in UTC with a `Z` suffix. The `DataLoader` accepts only this format.

## Task Categories

### Univariate Time Series (24 tasks)
Univariate time series contain a single target variable over time, making them ideal for testing models that predict one variable at a time.

**Healthcare & Medical:**
- `chickenpox_dense_univariate/` - Weekly chickenpox case counts (2005-2013)
- `patient_sparse_univariate/` - Sparse patient data with irregular patterns
- `employees_healthcare_univariate/` - Healthcare employee metrics over time

**Financial & Economics:**
- `coinbase_days_univariate/` - Daily Bitcoin prices from Coinbase (2020-2025)
- `coinbase_economics_univariate/` - Economic indicators related to cryptocurrency
- `german_quarterly_univariate/` - German quarterly economic data
- `german_houses_sales_univariate/` - German house sales data
- `federal_funds_weeks_univariate/` - Weekly federal funding data

**Energy & Environment:**
- `electricity_energy_univariate/` - Monthly electricity energy consumption (1978-2023)
- `power_consumption_years_univariate/` - Annual power consumption data
- `delhi_climate_univariate/` - Climate data from Delhi
- `soil_nature_univariate/` - Soil quality measurements over time

**Transportation & Infrastructure:**
- `madrid_transport_univariate/` - Madrid transportation metrics
- `occupancy_count_univariate/` - Building occupancy counts

**Technology & Software:**
- `software_nonstationary_univariate/` - Non-stationary software metrics
- `sw_job_postings_software_univariate/` - Software job posting trends
- `web_traffic_univariate/` - Web traffic data

**Business & Retail:**
- `retail_categorical_univariate/` - Retail sales with categorical patterns
- `inventories_months_univariate/` - Monthly inventory levels
- `inventories_manufacturing_univariate/` - Manufacturing inventory data

**Synthetic Datasets:**
- `synthetic_additive2_univariate/` - Additive synthetic time series
- `synthetic_cyclic_univariate/` - Cyclical synthetic patterns
- `synthetic_multiplicative_univariate/` - Multiplicative synthetic data
- `synthetic_nonstationary_univariate/` - Non-stationary synthetic series

**Special Cases:**
- `absent_binary_univariate/` - Binary data with missing values

### Multivariate Time Series (22 tasks)
Multivariate time series contain multiple related variables, allowing models to leverage cross-variable dependencies for improved forecasting.

**Financial Markets:**
- `gold_india_continuous_multivariate/` - Indian gold market data (price, volume, OHLC)
- `gold_india_dense_multivariate/` - Dense gold market observations
- `gold_india_economics_multivariate/` - Gold market with economic indicators
- `gold_india_real_multivariate/` - Real gold market data
- `india_gold_days_multivariate/` - Daily Indian gold prices
- `lt_stock_longest_multivariate/` - Long-term stock data
- `lt_stock_minutes_multivariate/` - High-frequency stock data (minute-level)

**Transportation & Logistics:**
- `baggage_100_multivariate/` - Airline baggage data (100 variables)
- `baggage_months_multivariate/` - Monthly baggage statistics
- `baggage_sales_multivariate/` - Baggage sales metrics
- `madrid_transport_multivariate/` - Madrid transportation system data
- `madrid_count_multivariate/` - Madrid traffic counts
- `madrid_cyclical_multivariate/` - Cyclical Madrid transport patterns
- `madrid_hours_multivariate/` - Hourly Madrid transport data
- `madrid_noisy_multivariate/` - Noisy Madrid transport measurements

**Environmental & Air Quality:**
- `batadal_software_multivariate/` - Environmental software metrics
- `soil_500_multivariate/` - Soil data with 500 variables
- `soil_nature_multivariate/` - Natural soil measurements

**Healthcare & Public Health:**
- `nyc_covid_healthcare_multivariate/` - NYC COVID-19 healthcare data

**Energy & Utilities:**
- `split_smart_energy_multivariate/` - Smart energy grid data
- `utah_manufacturing_multivariate/` - Utah manufacturing energy consumption

## Dataset Characteristics

### Temporal Resolution
- **Daily**: Most financial and economic datasets
- **Weekly**: Healthcare and some economic indicators
- **Monthly**: Energy consumption, climate data
- **Hourly**: High-frequency transport and environmental data
- **Minute-level**: Ultra-high-frequency financial data

### Data Quality
- **Dense**: Regular observations with minimal missing values
- **Sparse**: Irregular patterns with significant missing data
- **Continuous**: Smooth, continuous value ranges
- **Categorical**: Discrete categories or binary values
- **Noisy**: High measurement noise requiring robust models

### Complexity Levels
- **Stationary**: Stable statistical properties over time
- **Non-stationary**: Changing statistical properties
- **Cyclical**: Regular seasonal patterns
- **Additive**: Linear combination of components
- **Multiplicative**: Non-linear component interactions

## Usage Notes

1. **File Structure**: Each dataset contains a main CSV file and chunk files for efficient processing
2. **Date Formats**: Various date formats are used - check individual files for specifics
3. **Missing Values**: Some datasets contain missing values indicated by empty cells
4. **Scaling**: Consider normalizing/standardizing data before model training
5. **Validation**: Use appropriate train/validation/test splits based on temporal order

## Benchmarking Guidelines

- Use consistent evaluation metrics across all datasets
- Respect temporal ordering in train/test splits
- Consider dataset-specific characteristics when selecting models
- Report results separately for different data quality levels (dense vs sparse)
- Account for seasonality and trend components appropriately
