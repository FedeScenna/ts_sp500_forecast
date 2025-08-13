"""
Constants file for the ARIMA model
Defines common parameters used across the project
"""

# Date to split training and testing data
end_date = "2024-01-01"

# Model parameters
ARIMA_ORDER = (1, 0, 0)  # (p, d, q) for ARIMA model

# File paths
PRICE_DATA_PATH = "data/price_data.csv"
PREDICTIONS_PATH = "data/models/preds_arima.csv"

# Data processing parameters
SMOKE_TEST = False  # Set to True for testing with a single stock