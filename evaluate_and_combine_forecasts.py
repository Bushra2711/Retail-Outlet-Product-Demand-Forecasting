import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os
import math
warnings.filterwarnings('ignore')

# Load cleaned data
print("Loading cleaned sales data...")
# NOTE: 'cleaned_sales_data.csv' must exist in the current directory for the script to run.
df = pd.read_csv('cleaned_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Get unique product-outlet pairs
pairs = df[['Product_Name', 'Outlet']].drop_duplicates()
print(f"Found {len(pairs)} unique product-outlet pairs")

# Store metrics
metrics_list = []

# --- MODEL TRAINING AND METRICS COLLECTION (Unchanged) ---
for i, row in pairs.iterrows():
    product = row['Product_Name']
    outlet = row['Outlet']
    # print(f"\nProcessing pair {i+1}/{len(pairs)}: {product} at {outlet}") # Muted for cleaner output
    
    # Filter and sort data
    pair_df = df[(df['Product_Name'] == product) & (df['Outlet'] == outlet)].sort_values('Date')
    
    if len(pair_df) < 24:  # Need at least 24 months for train/test
        # print(f"Skipping: not enough data ({len(pair_df)} months)") # Muted for cleaner output
        continue
    
    # Split into train and test (last 12 months as test)
    train = pair_df[:-12]
    test = pair_df[-12:]
    
    
    # ARIMA
    try:
        ts_train = train.set_index('Date')['Monthly_Sales']
        ts_train.index = pd.DatetimeIndex(ts_train.index).to_period('M')
        model_arima = ARIMA(ts_train, order=(1,1,1))
        model_arima_fit = model_arima.fit()
        forecast_arima = model_arima_fit.forecast(steps=12)
        arima_rmse = np.sqrt(mean_squared_error(test['Monthly_Sales'], forecast_arima))
        arima_mae = mean_absolute_error(test['Monthly_Sales'], forecast_arima)
        arima_mape = mean_absolute_percentage_error(test['Monthly_Sales'], forecast_arima)
    except Exception as e:
        # print(f"ARIMA failed: {e}") # Muted for cleaner output
        arima_rmse = arima_mae = arima_mape = np.nan
    
    
    # Prophet
    try:
        prophet_train = train[['Date', 'Monthly_Sales']].rename(columns={'Date': 'ds', 'Monthly_Sales': 'y'})
        model_prophet = Prophet()
        model_prophet.fit(prophet_train)
        future = model_prophet.make_future_dataframe(periods=12, freq='ME')
        forecast_prophet = model_prophet.predict(future)
        prophet_test_forecast = forecast_prophet[-12:]['yhat'].values
        prophet_rmse = np.sqrt(mean_squared_error(test['Monthly_Sales'], prophet_test_forecast))
        prophet_mae = mean_absolute_error(test['Monthly_Sales'], prophet_test_forecast)
        prophet_mape = mean_absolute_percentage_error(test['Monthly_Sales'], prophet_test_forecast)
    except Exception as e:
        # print(f"Prophet failed: {e}") # Muted for cleaner output
        prophet_rmse = prophet_mae = prophet_mape = np.nan

    # Store metrics
    metrics_list.append({
        'Product': product,
        'Outlet': outlet,
        'ARIMA_RMSE': arima_rmse,
        'ARIMA_MAE': arima_mae,
        'ARIMA_MAPE': arima_mape,
        'Prophet_RMSE': prophet_rmse,
        'Prophet_MAE': prophet_mae,
        'Prophet_MAPE': prophet_mape
    })

# Create metrics DataFrame
metrics_df = pd.DataFrame(metrics_list)
print("\nMetrics Summary:")
print(metrics_df.describe())

# Print comparison
print("\nAverage Metrics Comparison:")
print(f"ARIMA - RMSE: {metrics_df['ARIMA_RMSE'].mean():.2f}, MAE: {metrics_df['ARIMA_MAE'].mean():.2f}, MAPE: {metrics_df['ARIMA_MAPE'].mean():.2f}")
print(f"Prophet - RMSE: {metrics_df['Prophet_RMSE'].mean():.2f}, MAE: {metrics_df['Prophet_MAE'].mean():.2f}, MAPE: {metrics_df['Prophet_MAPE'].mean():.2f}")

# Save metrics
metrics_df.to_csv('forecast_metrics_comparison.csv', index=False)
print("\nMetrics saved to forecast_metrics_comparison.csv")

# --- DATA MERGING AND STANDARDIZATION ---
print("\nLoading forecast CSVs...")
# NOTE: 'product_outlet_arima_forecasts.csv' and 'product_outlet_prophet_forecasts.csv' must exist.
arima_df = pd.read_csv('product_outlet_arima_forecasts.csv')
prophet_df = pd.read_csv('product_outlet_prophet_forecasts.csv')

# Standardize column names robustly
def standardize_forecast_frame(df, model_name):
    df = df.copy()
    if 'Date' not in df.columns and 'ds' in df.columns:
        df = df.rename(columns={'ds': 'Date'})
    if 'Product' not in df.columns and 'Product_Name' in df.columns:
        df = df.rename(columns={'Product_Name': 'Product'})
    
    if model_name == 'ARIMA':
        if 'ARIMA_Forecast' not in df.columns and 'Predicted_Sales' in df.columns:
            df = df.rename(columns={'Predicted_Sales': 'ARIMA_Forecast'})
    elif model_name == 'Prophet':
        if 'Prophet_Forecast' not in df.columns and 'yhat' in df.columns:
            df = df.rename(columns={'yhat': 'Prophet_Forecast'})
    
    keep_cols = ['Date', 'Product', 'Outlet']
    if model_name == 'ARIMA' and 'ARIMA_Forecast' in df.columns:
        keep_cols.append('ARIMA_Forecast')
    if model_name == 'Prophet' and 'Prophet_Forecast' in df.columns:
        keep_cols.append('Prophet_Forecast')
        
    df = df[[c for c in keep_cols if c in df.columns]]
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
    return df

arima_df = standardize_forecast_frame(arima_df, 'ARIMA')
prophet_df = standardize_forecast_frame(prophet_df, 'Prophet')

# Merge on Date, Product, Outlet using outer join
merged_df = pd.merge(arima_df, prophet_df, on=['Date', 'Product', 'Outlet'], how='outer')

# Ensure Date is in consistent string format for CSV
if 'Date' in merged_df.columns:
    merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.strftime('%Y-%m-%d')

print(f"Total rows in merged data: {len(merged_df)}")
print("Columns:", merged_df.columns.tolist())

# --- FILE SPLITTING FOR GITHUB (MODIFIED SECTION) ---

print("\n**Splitting final predictions into parts of ~20 MB and saving to current directory for GitHub upload...**")

# Convert DataFrame to CSV lines
csv_string = merged_df.to_csv(index=False)
lines = csv_string.split('\n')
header = lines[0]
data_lines = lines[1:]

# Calculate average line size for chunking
sample_size = min(1000, len(data_lines))
if len(data_lines) == 0:
    print("No data lines to split.")
    num_parts = 0
else:
    sample_bytes = sum(len(line.encode('utf-8')) for line in data_lines[:sample_size])
    avg_line_size = sample_bytes / sample_size if sample_size else 0
    
    # Calculate lines per part (20 MB limit to be safe for GitHub's 100MB max)
    lines_per_part = int((20 * 1024 * 1024) / avg_line_size) if avg_line_size else len(data_lines)
    
    total_lines = len(data_lines)
    num_parts = math.ceil(total_lines / lines_per_part)
    
    # Perform splitting and save directly to current directory
    for i in range(num_parts):
        start = i * lines_per_part
        end = min((i + 1) * lines_per_part, total_lines)
        chunk = data_lines[start:end]
        
        # Save directly to the current working directory
        part_filename = f'final_predictions_part{i+1}.csv'
        
        with open(part_filename, 'w', encoding='utf-8') as f:
            f.write(header + '\n')
            f.write('\n'.join(chunk))
        
        # Print confirmation for the file being saved
        print(f"  -> Saved: {part_filename} (Approx. {len(chunk)} rows)")

    print("\n✅ **Split complete.** All parts are saved directly in the current directory and are sized appropriately for GitHub.")