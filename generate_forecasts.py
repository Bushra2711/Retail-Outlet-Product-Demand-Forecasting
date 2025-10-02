import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Load cleaned data
df = pd.read_csv('cleaned_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Get all unique product-outlet pairs
pairs = df[['Product_Name', 'Outlet']].drop_duplicates()

# Store forecasts
prophet_forecasts = []
arima_forecasts = []

for _, row in pairs.iterrows():
    product = row['Product_Name']
    outlet = row['Outlet']
    pair_df = df[(df['Product_Name'] == product) & (df['Outlet'] == outlet)].sort_values('Date')
    dates = pair_df['Date']
    sales = pair_df['Monthly_Sales']

    # Prophet for historical dates
    prophet_df = pair_df[['Date', 'Monthly_Sales']].rename(columns={'Date': 'ds', 'Monthly_Sales': 'y'})
    if len(prophet_df) < 10:
        print(f"Skipping Prophet for {product} at {outlet}: not enough data points ({len(prophet_df)})")
        continue
    model = Prophet()
    model.fit(prophet_df)
    forecast = model.predict(prophet_df[['ds']])
    temp = pd.DataFrame({
        'Date': prophet_df['ds'],
        'Product_Name': product,
        'Outlet': outlet,
        'Prophet_Forecast': forecast['yhat'].values
    })
    prophet_forecasts.append(temp)

    # ARIMA for historical dates
    if len(sales) < 10:
        print(f"Skipping ARIMA for {product} at {outlet}: not enough data ({len(sales)})")
        continue
    try:
        model = ARIMA(sales, order=(1,1,1))
        model_fit = model.fit()
        arima_pred = model_fit.predict(start=0, end=len(sales)-1)
        temp = pd.DataFrame({
            'Date': dates,
            'Product_Name': product,
            'Outlet': outlet,
            'ARIMA_Forecast': arima_pred.values
        })
        arima_forecasts.append(temp)
    except Exception as e:
        print(f"ARIMA failed for {product}-{outlet}: {e}")

# Combine all forecasts
prophet_result = pd.concat(prophet_forecasts, ignore_index=True)
arima_result = pd.concat(arima_forecasts, ignore_index=True)

# Save to CSV
prophet_result.to_csv('product_outlet_prophet_forecasts.csv', index=False)
arima_result.to_csv('product_outlet_arima_forecasts.csv', index=False)

print('Forecasts for all product-outlet pairs saved to product_outlet_prophet_forecasts.csv and product_outlet_arima_forecasts.csv')
