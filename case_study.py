import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf # Import yfinance

# Set a style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

# --- 1. Simulate Economic Indicators Data (as yfinance doesn't provide this) ---
# In a real scenario, you'd fetch this from a separate API (e.g., FRED)
months = pd.date_range(start='2023-01-01', end='2024-12-31', freq='MS') # Month start
np.random.seed(42) # for reproducibility
interest_rate = np.linspace(0.04, 0.055, len(months)) + np.random.normal(0, 0.002, len(months))
inflation_rate = np.linspace(0.06, 0.03, len(months)) + np.random.normal(0, 0.003, len(months))

df_economic = pd.DataFrame({
    'Month': months.strftime('%Y-%m'), # Format to 'YYYY-MM'
    'InterestRate': interest_rate.round(4),
    'InflationRate': inflation_rate.round(4)
})
df_economic.to_csv('economic_indicators.csv', index=False)
print("Simulated 'economic_indicators.csv' created.\n")


# --- 2. Data Loading & Initial Inspection (Using yfinance) ---

# Define start and end dates for historical data
start_date = "2023-01-01"
end_date = "2024-12-31" # You can set this to pd.Timestamp.today().strftime('%Y-%m-%d') for current data

print(f"Fetching AAPL data from {start_date} to {end_date} using yfinance...")
df_aapl = yf.download("AAPL", start=start_date, end=end_date)
print(f"Fetching MSFT data from {start_date} to {end_date} using yfinance...")
df_msft = yf.download("MSFT", start=start_date, end=end_date)

# Load economic indicators
df_economic = pd.read_csv('economic_indicators.csv')

print("\n--- AAPL Data Info (from yfinance) ---")
df_aapl.info()
print("\n--- MSFT Data Info (from yfinance) ---")
df_msft.info()
print("\n--- Economic Indicators Data Info ---")
df_economic.info()

print("\nAAPL Head:\n", df_aapl.head())
print("\nMSFT Head:\n", df_msft.head())
print("\nEconomic Indicators Head:\n", df_economic.head())

# --- 3. Data Cleaning & Preprocessing (Time Series Setup) ---
# yfinance usually returns data with 'Date' as index and correct dtypes.
# We'll just ensure 'Month' in economic indicators is datetime for merging.

# Convert 'Month' in economic indicators to datetime objects
df_economic['Month'] = pd.to_datetime(df_economic['Month'])

# Check for missing values (usually minimal with yfinance historical data)
print("\nMissing values in AAPL:\n", df_aapl.isnull().sum().sum()) # Total missing values
print("\nMissing values in MSFT:\n", df_msft.isnull().sum().sum()) # Total missing values
print("\nMissing values in Economic Indicators:\n", df_economic.isnull().sum().sum())


# --- 4. Merging DataFrames ---

# Resample daily stock data to monthly average close prices
# 'MS' means Month Start frequency, which aligns with our economic data
df_aapl_monthly = df_aapl['Close'].resample('MS').mean().reset_index()
df_msft_monthly = df_msft['Close'].resample('MS').mean().reset_index()

# Rename columns for clarity before merging
df_aapl_monthly.rename(columns={'Close': 'AAPL_Avg_Close', 'Date': 'Month'}, inplace=True)
df_msft_monthly.rename(columns={'Close': 'MSFT_Avg_Close', 'Date': 'Month'}, inplace=True)

# Merge AAPL and MSFT monthly data
df_merged_stocks = pd.merge(df_aapl_monthly, df_msft_monthly, on='Month', how='inner')

# Merge combined stock data with economic indicators
# Ensure 'Month' column in economic indicators is formatted identically (already done above)
df_final_merged = pd.merge(df_merged_stocks, df_economic, on='Month', how='inner')

print("\n--- Merged Monthly Stock & Economic Data (Head) ---")
print(df_final_merged.head())
print("\n--- Merged Data Info ---")
df_final_merged.info()


# --- 5. Time Series Analysis ---

# Plotting Close Prices
plt.figure(figsize=(14, 7))
plt.plot(df_aapl.index, df_aapl['Close'], label='AAPL Close Price', color='blue')
plt.plot(df_msft.index, df_msft['Close'], label='MSFT Close Price', color='red')
plt.title(f'AAPL and MSFT Daily Closing Prices ({start_date} to {end_date})')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate Daily Returns
df_aapl['Daily_Return'] = df_aapl['Adj Close'].pct_change()
df_msft['Daily_Return'] = df_msft['Adj Close'].pct_change()

print("\nAAPL Daily Returns Head:\n", df_aapl['Daily_Return'].head())
print("\nMSFT Daily Returns Head:\n", df_msft['Daily_Return'].head())

# Calculate Rolling Moving Averages
df_aapl['MA_30_Day'] = df_aapl['Close'].rolling(window=30).mean()
df_aapl['MA_90_Day'] = df_aapl['Close'].rolling(window=90).mean()

df_msft['MA_30_Day'] = df_msft['Close'].rolling(window=30).mean()
df_msft['MA_90_Day'] = df_msft['Close'].rolling(window=90).mean()

plt.figure(figsize=(14, 7))
plt.plot(df_aapl.index, df_aapl['Close'], label='AAPL Close')
plt.plot(df_aapl.index, df_aapl['MA_30_Day'], label='AAPL 30-Day MA', linestyle='--')
plt.plot(df_aapl.index, df_aapl['MA_90_Day'], label='AAPL 90-Day MA', linestyle=':')
plt.title('AAPL Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(df_msft.index, df_msft['Close'], label='MSFT Close')
plt.plot(df_msft.index, df_msft['MA_30_Day'], label='MSFT 30-Day MA', linestyle='--')
plt.plot(df_msft.index, df_msft['MA_90_Day'], label='MSFT 90-Day MA', linestyle=':')
plt.title('MSFT Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Resample Volume to Monthly Total
aapl_monthly_volume = df_aapl['Volume'].resample('M').sum()
msft_monthly_volume = df_msft['Volume'].resample('M').sum()

plt.figure(figsize=(14, 7))
plt.bar(aapl_monthly_volume.index, aapl_monthly_volume.values, width=20, label='AAPL Monthly Volume', alpha=0.7, color='teal')
plt.bar(msft_monthly_volume.index, msft_monthly_volume.values, width=20, label='MSFT Monthly Volume', alpha=0.7, color='orange')
plt.title('AAPL and MSFT Monthly Total Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Correlation between daily returns
# Create a DataFrame with daily returns for both stocks, aligning by index (Date)
df_returns_combined = pd.DataFrame({
    'AAPL_Daily_Return': df_aapl['Daily_Return'],
    'MSFT_Daily_Return': df_msft['Daily_Return']
}).dropna() # Drop NaN values introduced by pct_change() or misalignment

correlation_returns = df_returns_combined['AAPL_Daily_Return'].corr(df_returns_combined['MSFT_Daily_Return'])
print(f"\nCorrelation between AAPL and MSFT daily returns: {correlation_returns:.4f}")
# A positive correlation indicates they tend to move in the same direction.

# --- 6. Pivot Tables ---

# Combine daily data for both stocks for easy pivot table creation
# Add a 'Ticker' column to each DataFrame
df_aapl['Ticker'] = 'AAPL'
df_msft['Ticker'] = 'MSFT'

# Ensure the index (Date) is reset before concatenating if you want 'Date' as a column
# Or ensure it's handled properly if you want to keep it as index
df_combined_daily = pd.concat([df_aapl.reset_index(), df_msft.reset_index()])

# Re-set 'Date' as index for the combined DataFrame for consistent time series operations later
df_combined_daily.set_index('Date', inplace=True)

# Extract Year and Month from the index
df_combined_daily['Year'] = df_combined_daily.index.year
df_combined_daily['Month'] = df_combined_daily.index.month_name() # Get month name for better readability

pivot_avg_monthly_close = pd.pivot_table(
    df_combined_daily,
    values='Close',
    index=['Year', 'Month'],
    columns='Ticker',
    aggfunc='mean'
)
print("\n--- Average Monthly Closing Price by Ticker (Pivot Table) ---")
print(pivot_avg_monthly_close.head())

# Create a pivot table to analyze daily returns by month
pivot_monthly_returns_stats = pd.pivot_table(
    df_combined_daily,
    values='Daily_Return',
    index='Month',
    columns='Ticker',
    aggfunc=['mean', 'std', 'max', 'min']
)
print("\n--- Monthly Return Statistics by Ticker (Pivot Table) ---")
print(pivot_monthly_returns_stats)


# --- 7. Feature Engineering for Prediction (Conceptual) ---

# Example: Create lagged features for AAPL
# (Operating on the df_aapl DataFrame with Date as index)
df_aapl['Close_Lag1'] = df_aapl['Close'].shift(1) # Previous day's close
df_aapl['Close_Lag5'] = df_aapl['Close'].shift(5) # Close 5 days ago
df_aapl['Volume_Lag1'] = df_aapl['Volume'].shift(1) # Previous day's volume
df_aapl['Daily_Return_Lag1'] = df_aapl['Daily_Return'].shift(1) # Previous day's return

# Example of a new feature: difference between close and a moving average
df_aapl['Close_Minus_MA30'] = df_aapl['Close'] - df_aapl['MA_30_Day']

print("\n--- AAPL Data with Engineered Features (Head) ---")
# Display only relevant columns to avoid overwhelming output
print(df_aapl[['Close', 'Close_Lag1', 'Daily_Return', 'Daily_Return_Lag1', 'Close_Minus_MA30', 'MA_30_Day']].head(10))

print("\n**Discussion on Stock Price Prediction:**")
print("These engineered features (lagged prices, returns, volume, and derived indicators like 'Close_Minus_MA30', moving averages) serve as potential independent variables (X) for a predictive model.")
print("The goal is often to predict the next day's 'Close' price or 'Daily_Return' (the target variable, Y).")
print("\n**Next Steps for Prediction:**")
print("1.  **Define Target Variable (Y):** E.g., `df_aapl['Next_Day_Close'] = df_aapl['Close'].shift(-1)`.")
print("2.  **Handle Missing Values:** Features will have NaN values at the beginning due to `shift()` and `rolling()` operations. These rows would need to be dropped or imputed.")
print("3.  **Split Data:** Divide the dataset into training and testing sets (crucially, a time-series split, not random, to avoid data leakage).")
print("4.  **Choose a Model:** Select an appropriate machine learning model (e.g., Linear Regression, Random Forest Regressor, Gradient Boosting, or for more complex time series, LSTM/ARIMA).")
print("5.  **Train & Evaluate:** Train the model on the training data and evaluate its performance on the unseen testing data using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), or R-squared.")
print("6.  **Incorporate Macroeconomic Data:** The `df_final_merged` DataFrame demonstrates how macroeconomic factors (InterestRate, InflationRate) could be integrated as additional features to potentially improve prediction accuracy, especially for longer-term forecasting or broader market analysis.")