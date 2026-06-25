#!/usr/bin/env python
# coding: utf-8

# In[10]:


# !pip install scikit-learn yfinance


# In[11]:


import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[12]:


# STEP 1: Fetch Stock Data using yfinance
# ==========================================
print("--- STEP 1: FETCHING STOCK DATA ---")
ticker = "AAPL"  # Apple Inc.
start_date = "2021-01-01"
end_date = "2026-01-01"

print(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")
df = yf.download(ticker, start=start_date, end=end_date)

# Clean multi-level columns if returned by yfinance (recent versions sometimes do this)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"Data retrieved! Shape: {df.shape}")
print("First 5 rows of fetched data:")
print(df.head(), "\n")


# In[13]:


# ==========================================
# STEP 2: Feature Engineering (Shift for Forecasting)
# ==========================================
print("--- STEP 2: FEATURE ENGINEERING ---")
# To predict the "next day's" close, we must shift the closing prices back by 1 day.
# Thus, today's Open, High, Low, Close, and Volume will be mapped to tomorrow's Close.
df['Next_Close'] = df['Close'].shift(-1)

# Drop any rows with missing values (specifically the last row, which doesn't have a "tomorrow")
df_cleaned = df.dropna().copy()

# Features (X) are today's Open, High, Low, Close, and Volume
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
X = df_cleaned[feature_cols]

# Target (y) is the next day's close price
y = df_cleaned['Next_Close']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}\n")



# In[ ]:


# ==========================================
# STEP 3: Train / Test Split
# ==========================================
print("--- STEP 3: TEMPORAL TRAIN-TEST SPLIT ---")
# Since stock prices are time-series data, a random split can cause data leakage.
# We will split chronologically: the first 80% of days for training, the last 20% for testing.
split_idx = int(len(df_cleaned) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training samples: {len(X_train)} (from {X_train.index[0].strftime('%Y-%m-%d')} to {X_train.index[-1].strftime('%Y-%m-%d')})")
print(f"Testing samples:  {len(X_test)} (from {X_test.index[0].strftime('%Y-%m-%d')} to {X_test.index[-1].strftime('%Y-%m-%d')})\n")




# In[16]:


# ==========================================
# STEP 4: Model Training and Predictions
# ==========================================
# 1. Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
print("Linear Regression model trained.")

# 2. Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest Regressor model trained.\n")


# In[17]:


# ==========================================
# STEP 5: Performance Evaluation
# ==========================================
print("--- STEP 5: MODEL EVALUATION ---")

def evaluate_predictions(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{model_name} Metrics]")
    print(f"  Mean Absolute Error (MAE):     ${mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:.4f}")
    print(f"  R-squared (R2) Score:          {r2:.4f}\n")
    return mae, rmse, r2

lr_metrics = evaluate_predictions(y_test, lr_preds, "Linear Regression")
rf_metrics = evaluate_predictions(y_test, rf_preds, "Random Forest")


# In[18]:


print("--- STEP 6: SAVING VISUALIZATIONS ---")
sns.set_theme(style="darkgrid")
plt.figure(figsize=(14, 7))

# Plot actual prices
plt.plot(y_test.index, y_test.values, label="Actual Next-Day Close", color="black", linewidth=1.5, alpha=0.8)

# Plot Linear Regression predictions
plt.plot(y_test.index, lr_preds, label="Linear Regression Predictions", color="royalblue", linestyle="--", linewidth=1.2)

# Plot Random Forest predictions
plt.plot(y_test.index, rf_preds, label="Random Forest Predictions", color="darkorange", linestyle=":", linewidth=1.2)

plt.title(f"{ticker} Stock Price Prediction (Short-Term: Next-Day Close)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Stock Price (USD)", fontsize=12)
plt.legend(fontsize=11, loc="upper left")
plt.tight_layout()

# Save the plot
plt.savefig("stock_predictions.png", dpi=300)
plt.close()

print("- Saved: stock_predictions.png")


# In[ ]:




