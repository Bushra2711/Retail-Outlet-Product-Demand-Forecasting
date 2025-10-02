import pandas as pd
import numpy as np
import random
from datetime import datetime

def generate_sales_dataset(num_entries=200000):
    """
    Generate a retail dataset with the following columns:
    - Outlet
    - Product name
    - Cost
    - Year
    - Month
    - Date (optional)
    - Monthly sales
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Product names ( List of product )
    product_names = [
        "Laptop Pro", "Smartphone X", "Tablet Air", "Wireless Headphones", "Smart Watch",
        "Gaming Console", "Bluetooth Speaker", "Digital Camera", "Fitness Tracker", "VR Headset",
        "Wireless Mouse", "Mechanical Keyboard", "USB Drive", "Power Bank", "Webcam",
        "Microphone", "Monitor", "Printer", "Scanner", "Projector"
    ]
    
    
    # Outlet names (add as many as needed)
    outlet_names = [
        "Outlet_A", "Outlet_B", "Outlet_C", "Outlet_D", "Outlet_E",
        "Outlet_F", "Outlet_G", "Outlet_H", "Outlet_I", "Outlet_J"
    ]
    
    # Cost ranges for different product categories
    cost_ranges = { 
        "Laptop Pro": (800, 2500),
        "Smartphone X": (400, 1200),
        "Tablet Air": (300, 900),
        "Wireless Headphones": (50, 300),
        "Smart Watch": (100, 500),
        "Gaming Console": (300, 600),
        "Bluetooth Speaker": (30, 200),
        "Digital Camera": (200, 1500),
        "Fitness Tracker": (50, 300),
        "VR Headset": (200, 800),
        "Wireless Mouse": (20, 100),
        "Mechanical Keyboard": (80, 300),
        "USB Drive": (10, 100),
        "Power Bank": (20, 150),
        "Webcam": (30, 200),
        "Microphone": (40, 300),
        "Monitor": (150, 800),
        "Printer": (100, 500),
        "Scanner": (80, 400),
        "Projector": (300, 2000)
    }
    
    # Years (4 years)
    years = [2021, 2022, 2023, 2024]

    # Generate data
    data = []
    
    for _ in range(num_entries):
        # Randomly select outlet
        outlet = random.choice(outlet_names)
        
        # Randomly select product
        product = random.choice(product_names)
        
        # Generate cost based on product range
        min_cost, max_cost = cost_ranges[product]
        cost = round(random.uniform(min_cost, max_cost), 2)
        
        # Random year and month
        year = random.choice(years)
        month = random.randint(1, 12)
        
        # Optional: Generate a date (use first day of month)
        date = datetime(year, month, 1).strftime("%Y-%m-%d")
        
        # Generate monthly sales (considering seasonality and trends)
        base_sales = random.uniform(10, 1000)
        
        # Add some seasonality (higher sales in Q4 due to holidays)
        seasonal_factor = 1.0
        if month in [11, 12]:  # Holiday season
            seasonal_factor = 1.5
        elif month in [6, 7, 8]:  # Summer months
            seasonal_factor = 1.2
        
        # Add some year-over-year growth
        year_factor = 1.0 + (year - 2021) * 0.1  # 10% growth per year
        
        # Add some randomness
        random_factor = random.uniform(0.8, 1.2)
        
        monthly_sales = int(base_sales * seasonal_factor * year_factor * random_factor)
        
        data.append({
            'Outlet': outlet,
            'Product_Name': product,
            'Cost': cost,
            'Year': year,
            'Month': month,
            'Date': date,
            'Monthly_Sales': monthly_sales
        })
    
    return pd.DataFrame(data)

def main():
    print("Generating  sales dataset with 200,000 entries...")
    
    # Generate the dataset
    df = generate_sales_dataset(200000)
    
    
    # Display basic information
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display sample data
    print("\nSample data:")
    print(df.head(10))
    
    # Display summary statistics
    print("\nSummary statistics:")
    print(df.describe())
    
    # Display year-wise summary
    print("\nYear-wise summary:")
    print(df.groupby('Year')['Monthly_Sales'].agg(['count', 'mean', 'sum']))
    
    # Display product-wise summary
    print("\nTop 5 products by total sales:")
    product_sales = df.groupby('Product_Name')['Monthly_Sales'].sum().sort_values(ascending=False)
    print(product_sales.head())
    
    # Save to CSV
    output_file = 'Retail_dataset_200k.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    
    # Display file size
    import os
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
    print(f"File size: {file_size:.2f} MB")

if __name__ == "__main__":
    main() 
    