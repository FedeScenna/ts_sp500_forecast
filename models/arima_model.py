#!/usr/bin/env python3
"""
ARIMA Model for Stock Price Prediction
Converted from Jupyter notebook 1_ARIMA.ipynb

This script implements ARIMA(1,0,0) model for stock price forecasting
using a rolling window approach with one-step ahead predictions.
"""

import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm, trange

# Import constants (you may need to create this file or define end_date here)
try:
    from models.constants import end_date
except ImportError:
    # Define end_date if constants.py doesn't exist
    end_date = "2024-01-01"
    print(f"Warning: constants.py not found. Using default end_date: {end_date}")

def main():
    """Main function to run the ARIMA model"""
    
    # Cell 1: Load data
    print("Loading price data...")
    df = pd.read_csv("../data/price_data.csv")
    smoke_test = False
    if smoke_test:
        df = df[df["ticker"] == "AAPL"]
    
    # Cell 2: Pivot data
    print("Preparing time series data...")
    df_ts = df.pivot(index='date', columns='ticker', values='close')
    
    # Cell 3: Get stock list
    stocks = df_ts.columns.tolist()
    print(f"Processing {len(stocks)} stocks...")
    
    # Cell 4: Split train/test
    train, test = df_ts[df_ts.index <= end_date], df_ts[df_ts.index > end_date]
    print(f"Training period: {train.index.min()} to {train.index.max()}")
    print(f"Testing period: {test.index.min()} to {test.index.max()}")
    
    # Cell 5: Run ARIMA model
    print("Running ARIMA(1,0,0) model...")
    
    # Suppress statsmodels warnings about date index frequency
    warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
    warnings.filterwarnings('ignore', message='.*date index.*frequency.*', module='statsmodels')
    warnings.filterwarnings('ignore', message='.*No supported index.*', module='statsmodels')
    
    all_stock_predictions = {}
    
    for s in tqdm(stocks, desc="Processing stocks", position=0):
        all_predictions = []
        current_data = train[s].copy()
        
        # Set frequency if it's a datetime index
        if hasattr(current_data.index, 'freq') and current_data.index.freq is None:
            current_data = current_data.asfreq('D')  # Adjust 'D' to your actual frequency
       
        for i in tqdm(range(len(test.index)), desc=f"Stock: {s}", position=1, leave=False):
            # Fit model on current available data
            model = ARIMA(current_data, order=(1, 0, 0)).fit()
            
            # Make one-step ahead prediction
            prediction = model.forecast(steps=1)
            all_predictions.append(prediction.values[0])
            
            # Update current_data with the ACTUAL observed value for next iteration
            if i < len(test.index) - 1:
                actual_value = test[s].iloc[i]
                new_data = pd.Series([actual_value], index=[test.index[i]])
                if hasattr(current_data.index, 'freq') and current_data.index.freq is not None:
                    new_data = new_data.asfreq(current_data.index.freq)
                current_data = pd.concat([current_data, new_data])
       
        # Store predictions for this stock
        all_stock_predictions[s] = all_predictions
    
    # Create DataFrame after the loop finishes
    predictions_df = pd.DataFrame(data=all_stock_predictions, index=test.index)
    
    # Re-enable warnings if needed
    warnings.resetwarnings()
    
    # Cell 6: Save predictions
    print("Saving predictions...")
    predictions_df.to_csv("data/models/preds_arima.csv", index=False)
    print(f"Predictions saved to data/models/preds_arima.csv")
    print(f"Shape of predictions: {predictions_df.shape}")
    
    return predictions_df

if __name__ == "__main__":
    try:
        predictions = main()
        print("ARIMA model execution completed successfully!")
    except Exception as e:
        print(f"Error during execution: {e}")
        raise
