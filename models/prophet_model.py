#!/usr/bin/env python3
"""
Prophet Model for Stock Price Prediction
Converted from Jupyter notebook 1_Prophet.ipynb

This script implements Facebook Prophet model for stock price forecasting.
Prophet is designed for forecasting time series data with strong seasonal patterns.
"""

import pandas as pd
from prophet import Prophet
from tqdm import tqdm

# Import constants (you may need to create this file or define end_date here)
try:
    from models.constants import end_date
except ImportError:
    # Define end_date if constants.py doesn't exist
    end_date = "2024-01-01"
    print(f"Warning: constants.py not found. Using default end_date: {end_date}")

def main():
    """Main function to run the Prophet model"""
    
    # Cell 1: Load data
    print("Loading price data...")
    df = pd.read_csv("./data/price_data.csv")
    
    # Cell 2: Pivot data
    print("Preparing time series data...")
    df_ts = df.pivot(index='date', columns='ticker', values='close')
    
    # Cell 3: Get stock list
    stocks = df_ts.columns
    print(f"Processing {len(stocks)} stocks...")
    
    # Cell 4: Split train/test
    train, test = df_ts[df_ts.index <= end_date], df_ts[df_ts.index > end_date]
    print(f"Training period: {train.index.min()} to {train.index.max()}")
    print(f"Testing period: {test.index.min()} to {test.index.max()}")
    
    # Cell 5: Run Prophet model
    print("Running Prophet model...")
    
    df_predsprophet = pd.DataFrame()
    
    for s in tqdm(stocks, desc="Processing stocks with Prophet"):
        # Create Prophet model
        m = Prophet()
        
        # Prepare data for Prophet (requires 'ds' for dates and 'y' for values)
        temp_train = train.reset_index()[["date", s]].copy()
        temp_train.rename(columns={"date": "ds", s: "y"}, inplace=True)
        
        # Fit the model
        m.fit(temp_train)
        
        # Make future predictions
        future_prices = m.make_future_dataframe(periods=test.shape[0])
        forecast = m.predict(future_prices)[["yhat"]]
        forecast.rename(columns={"yhat": s}, inplace=True)
        
        # Concatenate predictions
        df_predsprophet = pd.concat([df_predsprophet, forecast], axis=1)
    
    # Add date column
    df_predsprophet["date"] = df_ts.index
    
    # Save predictions
    print("Saving predictions...")
    df_predsprophet.to_csv("data/models/preds_prophet_simple.csv", index=False)
    print(f"Predictions saved to data/models/preds_prophet_simple.csv")
    print(f"Shape of predictions: {df_predsprophet.shape}")
    
    return df_predsprophet

if __name__ == "__main__":
    try:
        predictions = main()
        print("Prophet model execution completed successfully!")
    except Exception as e:
        print(f"Error during execution: {e}")
        raise
