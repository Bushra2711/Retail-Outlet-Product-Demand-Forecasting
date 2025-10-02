import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_data_analysis(cleaned_data_path='cleaned_sales_data.csv'):
    """Perform basic EDA on the cleaned dataset."""
    df = pd.read_csv(cleaned_data_path)
    print("\n--- Exploratory Data Analysis (EDA) ---")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(df.describe())

    # Boxplot for outliers in Monthly_Sales
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df['Monthly_Sales'])
    plt.title('Monthly Sales Distribution')
    plt.show()

    # Time series plot for a sample product-outlet pair
    sample = df[(df['Product_Name'] == df['Product_Name'].iloc[0]) & (df['Outlet'] == df['Outlet'].iloc[0])]
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(sample['Date']), sample['Monthly_Sales'], marker='o')
    plt.title(f"Sales Over Time: {sample['Product_Name'].iloc[0]} at {sample['Outlet'].iloc[0]}")
    plt.xlabel('Date')
    plt.ylabel('Monthly Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    exploratory_data_analysis()
