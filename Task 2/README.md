# DevelopersHub AI/ML Engineering Internship

## Task 2: Predict Future Stock Prices (Short-Term)

### Objective
The objective of this project is to build a machine learning pipeline that predicts the next business day's closing price of a stock using historical stock market data. Utilizing daily open, high, low, close, and volume metrics (OHLCV) from Apple Inc. (AAPL), we train and compare two predictive models: a statistical baseline (**Linear Regression**) and an ensemble-based machine learning model (**Random Forest Regressor**).

---

### Dataset Overview
* **Data Source:** Yahoo Finance (`yfinance` API)
* **Ticker Symbol:** `AAPL` (Apple Inc.)
* **Historical Range:** January 1, 2021, to January 1, 2026
* **Input Features ($X$):** Today's values for:
  * `Open`: Opening price
  * `High`: High price
  * `Low`: Low price
  * `Close`: Closing price
  * `Volume`: Total shares traded
* **Target Variable ($y$):** `Next_Close` (Tomorrow's Closing Price, engineered via a temporal shift of $-1$)

---

### Installation & Environment Setup

1. **Clone this project directory:**
   ```bash
  git clone https://github.com/numanabubakar/AI-ML-Internship-Task.git
cd AI-ML-Internship-Task/Task 2
   ```

2. **Install dependencies:**
   ```bash
   pip install yfinance scikit-learn pandas numpy matplotlib seaborn
   ```


### Methodology and Technical Choices

#### 1. Feature Engineering (Temporal Lag Mapping)
To frame the time-series forecasting problem as a supervised learning task, we apply a daily shift to the target variable:
$$y_t = \text{Close}_{t+1}$$
This maps the feature matrix representing market dynamics on day $t$ directly to the closing price of day $t+1$.

#### 2. Temporal Train-Test Split (Data Leakage Avoidance)
Traditional random splits must **never** be used on time-series stock data. Doing so causes chronological data leakage (shuffled future prices leaking into the past). 
* **Training Partition (First 80%):** Represents historical data used to learn trends and coefficients.
* **Testing Partition (Last 20%):** Represents unseen future data used to evaluate the model's forward-generalization capabilities.

---

### Key Skills Demonstrated
* Real-world financial data retrieval using **APIs** (`yfinance`)
* Time-series preprocessing and safe temporal-lag feature engineering
* Strategic train-test partitioning designed for historical time-stamped sequences
* Regression benchmarking, error metric profiling (MAE, RMSE, $R^2$), and complex line chart visualization