import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory for ARIMA plots
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arima_plots')
print(f"Creating output directory at: {output_dir}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load ARIMA forecast results
print("Loading ARIMA forecast files...")
try:
    df_arima = pd.read_csv('product_outlet_arima_forecasts.csv')
    df_arima['Date'] = pd.to_datetime(df_arima['Date'])
    print("Successfully loaded ARIMA forecasts")
except Exception as e:
    print(f"Error loading ARIMA forecasts: {e}")
    df_arima = pd.DataFrame()

def plot_arima_forecast(product, outlet):
    """Plot ARIMA forecast for the given product and outlet"""
    print(f"\nPlotting ARIMA forecast for {product} at {outlet}")
    arima = df_arima[(df_arima['Product_Name'] == product) & (df_arima['Outlet'] == outlet)]
    if arima.empty:
        print(f"No ARIMA forecasts found for {product} at {outlet}.")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(arima['Date'], arima['ARIMA_Forecast'], marker='o', label='ARIMA Forecast', linestyle='--')
    plt.title(f'ARIMA Forecast for {product} at {outlet}')
    plt.xlabel('Date')
    plt.ylabel('Predicted Sales')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{product}_{outlet}_arima.png'.replace(" ", "_"))
    print(f"Saving figure to {filename}")
    try:
        plt.savefig(filename, dpi=300)
        print("Successfully saved figure")
    except Exception as e:
        print(f"Error saving figure: {e}")
    plt.close()

# Generate ARIMA-only plots for all product-outlet pairs
for _, row in df_arima[['Product_Name', 'Outlet']].drop_duplicates().iterrows():
    plot_arima_forecast(row['Product_Name'], row['Outlet'])

print(f"All ARIMA forecast plots saved in {output_dir}")