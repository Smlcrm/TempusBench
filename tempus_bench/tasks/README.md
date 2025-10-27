# Time Series Tasks

This directory contains a comprehensive collection of time series tasks for benchmarking forecasting models. The tasks are organized into two main categories: univariate and multivariate time series.

## Task Categories

### Univariate Time Series (25 tasks)
Univariate time series contain a single target variable over time, making them ideal for testing models that predict one variable at a time.

**Healthcare & Medical:**
- `chickenpox_dense_univariate/` - Weekly chickenpox case counts (2005-2013)
- `patient_sparse_univariate/` - Sparse patient data with irregular patterns
- `employees_healthcare_univariate/` - Healthcare employee metrics over time

**Financial & Economics:**
- `coinbase_days_univariate/` - Daily Bitcoin prices from Coinbase (2020-2025)
- `coinbase_economics_univariate/` - Economic indicators related to cryptocurrency
- `german_quaterly_univariate/` - German quarterly economic data
- `germanhouses_sales_univariate/` - German house sales data
- `federalfuns_weeks_univariate/` - Weekly federal funding data

**Energy & Environment:**
- `electricity_energy_univariate/` - Monthly electricity energy consumption (1978-2023)
- `pconsumption_years_univariate/` - Annual power consumption data
- `delhi_climate_univariate/` - Climate data from Delhi
- `forestfires_continuous_univariate/` - Continuous forest fire occurrence data
- `soil_nature_univariate/` - Soil quality measurements over time

**Transportation & Infrastructure:**
- `madrid_transport_univariate/` - Madrid transportation metrics
- `occupancy_count_univariate/` - Building occupancy counts

**Technology & Software:**
- `software_nonstationary_univariate/` - Non-stationary software metrics
- `swjobpostings_software_univariate/` - Software job posting trends
- `wtraffic_web_univariate/` - Web traffic data

**Business & Retail:**
- `retail_categorical_univariate/` - Retail sales with categorical patterns
- `inventories_months_univariate/` - Monthly inventory levels
- `invetories_manufacturing_univariate/` - Manufacturing inventory data

**Synthetic Datasets:**
- `synthetic_additive2_univariate/` - Additive synthetic time series
- `synthetic_cyclic_univariate/` - Cyclical synthetic patterns
- `synthetic_multiplticative_univariate/` - Multiplicative synthetic data
- `synthetic_nonstationary_univariate/` - Non-stationary synthetic series

**Special Cases:**
- `absent_binary_univariate/` - Binary data with missing values

### Multivariate Time Series (23 tasks)
Multivariate time series contain multiple related variables, allowing models to leverage cross-variable dependencies for improved forecasting.

**Financial Markets:**
- `goldindia_continuous_multivariate/` - Indian gold market data (price, volume, OHLC)
- `goldindia_dense_multivariate/` - Dense gold market observations
- `goldindia_economics_multivariate/` - Gold market with economic indicators
- `goldindia_real_multivariate/` - Real gold market data
- `indiagold_days_multivariate/` - Daily Indian gold prices
- `ltstock_longest_multivariate/` - Long-term stock data
- `ltstock_minutes_multivariate/` - High-frequency stock data (minute-level)

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
- `batadal_nonstationary_multivariate/` - Non-stationary environmental data
- `batadal_software_multivariate/` - Environmental software metrics
- `soil_500_multivariate/` - Soil data with 500 variables
- `soil_nature_multivariate/` - Natural soil measurements

**Healthcare & Public Health:**
- `nyccovid_healthcare_multivariate/` - NYC COVID-19 healthcare data

**Energy & Utilities:**
- `splitsmart_energy_multivariate/` - Smart energy grid data
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
