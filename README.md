# Forecasting System - Project Demand of Products at a Retail Outlet Based on Historical Data

## Description

The primary objective of this project is to develop a comprehensive forecasting system that predicts product demand across retail outlets using historical sales data. The system employs advanced time series analysis techniques, specifically **ARIMA** (AutoRegressive Integrated Moving Average) and **Prophet** models, to generate accurate demand forecasts.

This system provides a robust foundation for retail demand forecasting, enabling better inventory management and business planning through data-driven predictions.

## Features

- **Demand Prediction**: Build a reliable system to forecast product demand at retail outlets
- **Multi-Model Approach**: Implement and compare both ARIMA and Prophet forecasting methods
- **Visualization**: Create clear visual representations of forecasts for decision-making
- **Scalable Solution**: Design a system that can handle multiple product-outlet combinations
- **Data Processing**: Clean and preprocess historical sales data, handle missing values and outliers
- **Output Generation**: CSV files containing forecasts and visual plots for each product-outlet combination

## Installation

### Prerequisites
- Python 3.x

### Dependencies
- pandas
- numpy
- statsmodels
- prophet
- matplotlib
- seaborn

Install the required packages using pip:
```sh
pip install pandas numpy statsmodels prophet matplotlib seaborn
```

## Usage

Run the Python scripts in the following order:

1. **Data Cleaning**: `python Data_cleaner.py` - Cleans and preprocesses the raw sales data.
2. **Exploratory Data Analysis**: `python exploratory_data_analysis.py` - Performs exploratory data analysis on the cleaned data.
3. **Generate Forecasts**: `python generate_forecasts.py` - Generates forecasts using both ARIMA and Prophet models for all product-outlet pairs.
4. **Plot Forecasts**:
    - `python plot_arima_forecast.py` - Creates ARIMA forecast plots.
    - `python plot_prophet_forecast.py` - Creates Prophet forecast plots.
5. **Evaluate and Combine**: `python evaluate_and_combine_forecasts.py` - Evaluates model performance and combines forecasts.

## Project Structure

- `Data_cleaner.py`: Script for cleaning and preprocessing raw sales data.
- `exploratory_data_analysis.py`: Script for performing exploratory data analysis.
- `generate_forecasts.py`: Main script for generating ARIMA and Prophet forecasts.
- `plot_arima_forecast.py`: Script to generate ARIMA forecast plots.
- `plot_prophet_forecast.py`: Script to generate Prophet forecast plots.
- `evaluate_and_combine_forecasts.py`: Script to evaluate and combine forecast results.
- `Retail_Product_Demand_Forecasting.py`: Main project file (if applicable).
- `arima_plots/`: Directory containing ARIMA forecast plot images.
- `prophet_plots/`: Directory containing Prophet forecast plot images.
- `cleaned_sales_data.csv`: Cleaned sales data file.
- `Retail_dataset_200k.csv`: Raw retail sales dataset.
- `product_outlet_arima_forecasts.csv`: ARIMA forecast results.
- `product_outlet_prophet_forecasts.csv`: Prophet forecast results.
- `forecast_metrics_comparison.csv`: Comparison of forecast metrics.
- `PROJECT_LOG.md`: Project log and documentation.
- **`final_predictions_part1.csv`**: First part of the combined prediction output.
- **`final_predictions_part2.csv`**: Second part of the combined prediction output.
- **`final_predictions_part3.csv`**
- **`final_predictions_part4.csv`**
- **`final_predictions_part5.csv`**
- **`final_predictions_part6.csv`**
- **`final_predictions_part7.csv`**
- **`final_predictions_part8.csv`**
- **`final_predictions_part9.csv`**
- **`final_predictions_part10.csv`**
- **`final_predictions_part11.csv`**
- **`final_predictions_part12.csv`**
- **`final_predictions_part13.csv`**
- **`final_predictions_part14.csv`**
- **`final_predictions_part15.csv`**
- **`final_predictions_part16.csv`**

## Models Used

- **ARIMA (AutoRegressive Integrated Moving Average)**: Traditional time series forecasting model for handling trends and seasonality.
- **Prophet**: Robust forecasting model developed by Facebook, excellent for handling seasonality, holidays, and trend changes.

## Visualization

The system generates three types of visualizations:
- Individual ARIMA forecast plots (stored in `arima_plots/`)
- Individual Prophet forecast plots (stored in `prophet_plots/`)

Plots include time series data, prediction intervals, and clear representations of forecasts.

## Output

- **CSV Files**:
  - `product_outlet_arima_forecasts.csv`: ARIMA forecasts for each product-outlet pair.
  - `product_outlet_prophet_forecasts.csv`: Prophet forecasts for each product-outlet pair.
  - **`final_predictions_part1.csv`**: First part of combined predictions.
  - **`final_predictions_part2.csv`**: Second part of combined predictions.
  - **`final_predictions_part3.csv`**: Third part of combined predictions.
  - `forecast_metrics_comparison.csv`: Performance metrics comparison between models.

> **Note:** The final predictions are split into **16 files** (approximately **20 MB** each) and saved directly to the root directory due to GitHub file size limits.

- **Visualization Plots**:
  - ARIMA plots in `arima_plots/` directory.
  - Prophet plots in `prophet_plots/` directory.

## Benefits

1. Better inventory management through accurate demand predictions.
2. Reduced stockouts and overstocking.
3. Data-driven decision making for retail operations.
4. Comparative analysis of different forecasting methods.
5. Scalable solution for multiple products and outlets.

## Future Scope

1. Integration of external factors (holidays, promotions, economic indicators, etc.).
2. Real-time forecast updates and monitoring.
3. Enhanced visualization dashboard for interactive analysis.
4. Automated model selection based on performance metrics.
5. Machine learning integration for improved accuracy.
6. Web-based interface for easier access and deployment.

## License

This project was developed as part of the TCS iON RIO internship program. The code is available for educational and research purposes.

## Contact

For any queries regarding this project, please contact: 
- Project Maintainer: Bushra Abdul 
- Email: bushra14@amityonline.com