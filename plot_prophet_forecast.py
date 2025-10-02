import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory for Prophet plots
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prophet_plots')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load Prophet forecast results
df_prophet = pd.read_csv('product_outlet_prophet_forecasts.csv')
df_prophet['Date'] = pd.to_datetime(df_prophet['Date'])

# Load cleaned actual sales data
cleaned_df = pd.read_csv('cleaned_sales_data.csv')
cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])

def plot_prophet_forecast(product, outlet):
    # Filter data for the product-outlet pair
    actual = cleaned_df[(cleaned_df['Product_Name'] == product) & (cleaned_df['Outlet'] == outlet)]
    prophet = df_prophet[(df_prophet['Product_Name'] == product) & (df_prophet['Outlet'] == outlet)]

    plt.figure(figsize=(12, 6))
    if not actual.empty:
        plt.plot(actual['Date'], actual['Monthly_Sales'], marker='o', label='Actual Sales', color='black')
    if not prophet.empty:
        plt.plot(prophet['Date'], prophet['Prophet_Forecast'], marker='^', label='Prophet Forecast', linestyle='-')
        # Optional: confidence interval if columns exist
        if 'yhat_lower' in prophet.columns and 'yhat_upper' in prophet.columns:
            plt.fill_between(prophet['Date'], prophet['yhat_lower'], prophet['yhat_upper'], alpha=0.2, label='Prophet 95% Interval')

    plt.title(f'Prophet Sales Forecast for {product} at {outlet}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(output_dir, f'{product}_{outlet}_prophet.png'.replace(" ", "_"))
    plt.savefig(filename, dpi=300)
    plt.close()

# Generate Prophet-only plots for all product-outlet pairs in Prophet forecasts
for _, row in df_prophet[['Product_Name', 'Outlet']].drop_duplicates().iterrows():
    plot_prophet_forecast(row['Product_Name'], row['Outlet'])

print(f"All Prophet forecast plots saved in {output_dir}")