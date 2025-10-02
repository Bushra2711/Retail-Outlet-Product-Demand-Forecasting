import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Data cleaning and sanitization class for sales forecasting project."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.cleaned_data = None
        
    def load_data(self):
        """Load the raw sales dataset."""
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully! Shape: {self.raw_data.shape}")
            return self.raw_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Display basic information about the dataset."""
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\nDATASET EXPLORATION")
        print(f"Shape: {self.raw_data.shape}")
        print(f"Columns: {list(self.raw_data.columns)}")
        print(f"Missing values:\n{self.raw_data.isnull().sum()}")
        print(f"\nSample data:\n{self.raw_data.head()}")
        print(f"\nSummary statistics:\n{self.raw_data.describe()}")
    
    def clean_data(self):
        """Clean and sanitize the dataset."""
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return None
        
        print("\nDATA CLEANING & SANITIZATION")
        
        # Create a copy for cleaning
        self.cleaned_data = self.raw_data.copy()
        
        # Remove any rows with missing values
        initial_rows = len(self.cleaned_data)
        self.cleaned_data = self.cleaned_data.dropna()
        if len(self.cleaned_data) < initial_rows:
            print(f"Removed {initial_rows - len(self.cleaned_data)} rows with missing values")
        
        # Remove any negative values
        initial_rows = len(self.cleaned_data)
        self.cleaned_data = self.cleaned_data[
            (self.cleaned_data['Monthly_Sales'] >= 0) & 
            (self.cleaned_data['Cost'] >= 0)
        ]
        if len(self.cleaned_data) < initial_rows:
            print(f"Removed {initial_rows - len(self.cleaned_data)} rows with negative values")
        
        # Remove outliers using IQR method for sales
        Q1 = self.cleaned_data['Monthly_Sales'].quantile(0.25)
        Q3 = self.cleaned_data['Monthly_Sales'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_rows = len(self.cleaned_data)
        self.cleaned_data = self.cleaned_data[
            (self.cleaned_data['Monthly_Sales'] >= lower_bound) & 
            (self.cleaned_data['Monthly_Sales'] <= upper_bound)
        ]
        if len(self.cleaned_data) < initial_rows:
            print(f"Removed {initial_rows - len(self.cleaned_data)} outlier rows using IQR method")
        
        # Ensure data types are correct
        self.cleaned_data['Year'] = self.cleaned_data['Year'].astype(int)
        self.cleaned_data['Cost'] = self.cleaned_data['Cost'].astype(float)
        self.cleaned_data['Monthly_Sales'] = self.cleaned_data['Monthly_Sales'].astype(int)
        
         # Ensure Month and Date columns are correct if present
        if 'Month' in self.cleaned_data.columns:
            self.cleaned_data['Month'] = self.cleaned_data['Month'].astype(int)
        if 'Date' in self.cleaned_data.columns:
            self.cleaned_data['Date'] = pd.to_datetime(self.cleaned_data['Date'])

        print(f"Data types corrected")
        print(f"Final cleaned dataset shape: {self.cleaned_data.shape}")
        
        return self.cleaned_data
    
    def save_cleaned_data(self, output_path='cleaned_sales_data.csv'):
        """Save the cleaned dataset."""
        if self.cleaned_data is None:
            print("No cleaned data available. Please clean data first.")
            return
        
        self.cleaned_data.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")

def main():
    """Main function to demonstrate the data cleaning process."""
    print("TCS iON RIO - Sales Forecasting Data Cleaning System")
    
    # Initialize the data cleaner
    cleaner = DataCleaner('Retail_dataset_200k.csv')
    
    # Load data
    cleaner.load_data()
    
    # Explore data
    cleaner.explore_data()
    
    # Clean data
    cleaner.clean_data()
    
    # Save cleaned data
    cleaner.save_cleaned_data()
    