### Task 2: Predict Future Stock Prices (Short-Term)

#### Objective

The objective of this project is to build a machine learning pipeline that predicts the next business day's closing price of a stock using historical stock market data. Utilizing daily open, high, low, close, and volume metrics (OHLCV) from Apple Inc. (AAPL), we train and compare two predictive models: a statistical baseline (Linear Regression) and an ensemble-based machine learning model (Random Forest Regressor).

#### Dataset Overview

Data Source: Yahoo Finance (yfinance API)

Ticker Symbol: AAPL (Apple Inc.)

Historical Range: January 1, 2021, to January 1, 2026

Input Features ($X$): Today's values for:

Open: Opening price

High: High price

Low: Low price

Close: Closing price

Volume: Total shares traded

Target Variable ($y$): Next_Close (Tomorrow's Closing Price, engineered via a temporal shift of $-1$)

#### Installation & Environment Setup

Clone this project directory:

git clone https://github.com/numanabubakar/AI-ML-Internship-Task.git
cd AI-ML-Internship-Task/Task 2


Install dependencies:

pip install yfinance scikit-learn pandas numpy matplotlib seaborn


Run the prediction pipeline:

python task2_stock_prediction.py


#### Methodology and Technical Choices

##### 1. Feature Engineering (Temporal Lag Mapping)

To frame the time-series forecasting problem as a supervised learning task, we apply a daily shift to the target variable:


$$y_t = \text{Close}_{t+1}$$


This maps the feature matrix representing market dynamics on day $t$ directly to the closing price of day $t+1$. We drop the final row because there is no known future date to assign as its target label.

##### 2. Temporal Train-Test Split (Data Leakage Avoidance)

Traditional random splits (train_test_split with shuffling) must never be used on time-series stock data. Doing so causes chronological data leakage (shuffled future prices leaking into the past).

Training Partition (First 80%): Represents historical data used to learn trends and coefficients.

Testing Partition (Last 20%): Represents unseen future data used to evaluate the model's forward-generalization capabilities.

##### 3. Modeling Strategies

Linear Regression: Serves as a strong baseline. Since stock prices exhibit high day-to-day correlation (autocorrelation), a linear relationship between today's boundary prices and tomorrow's close holds a lot of predictive weight.

Random Forest Regressor: A non-linear, ensemble-based decision tree approach that aggregates predictions from 100 estimators to capture intricate market fluctuations.

#### Key Evaluation Metrics and Results

The models are assessed on the test set (which represents the most recent 20% of chronological stock history) using three key statistical metrics:

##### Mean Absolute Error (MAE): The average absolute difference between predicted and actual closing prices.


$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

##### Root Mean Squared Error (RMSE): Panelizes larger errors more heavily to reflect severe mispredictions.


$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

##### R-squared ($R^2$) Score: Displays the proportion of variance in tomorrow's close price that is predictable from our feature set.

```text
[Linear Regression Metrics]
  Mean Absolute Error (MAE):     $3.1620
  Root Mean Squared Error (RMSE): $4.5313
  R-squared (R2) Score:          0.9707

[Random Forest Metrics]
  Mean Absolute Error (MAE):     $6.8266
  Root Mean Squared Error (RMSE): $9.7414
  R-squared (R2) Score:          0.8648
```

#### Visualization Output

The pipeline automatically generates a visualization file saved as stock_predictions.png.

The black line represents actual price movements over the testing period.

The dashed blue line represents Linear Regression predictions.

The dotted orange line represents Random Forest predictions.

This visualization allows immediate visual inspection of how well both models capture stock volatility, pullbacks, and upward market swings.

#### Key Skills Demonstrated

Real-world financial data retrieval using APIs (yfinance)

Time-series preprocessing and safe temporal-lag feature engineering

Strategic train-test partitioning designed for historical time-stamped sequences

Regression benchmarking, error metric profiling (MAE, RMSE, $R^2$), and complex line chart visualization