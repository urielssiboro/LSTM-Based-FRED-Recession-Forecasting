# Macroeconomic Recession Prediction Using LSTM

This project builds a machine learning model to estimate the probability of a U.S. recession **12 months ahead** using macroeconomic indicators from the Federal Reserve Economic Data (FRED) database.

The model applies a **Long Short-Term Memory (LSTM)** neural network to capture temporal patterns in macroeconomic data such as inflation, industrial production, unemployment, interest rates, and financial market expectations.

The objective is to explore how **sequence-based deep learning models can be applied to macroeconomic forecasting**, particularly in identifying early warning signals of economic downturns.

---

# Project Objective

Economic recessions emerge from interactions between **monetary policy, labor markets, financial conditions, and real economic activity**.

Traditional econometric models often rely on linear assumptions, while macroeconomic systems frequently exhibit **nonlinear and time-dependent dynamics**.

This project investigates whether LSTM networks can learn these temporal relationships and detect patterns in macroeconomic indicators that precede recessions.

The model predicts whether the economy will enter a **recession within the next 12 months**.

---

# Dataset

All data is sourced from the **Federal Reserve Economic Data (FRED)** database.

| Indicator | FRED Code | Economic Component |
|---|---|---|
| Federal Funds Rate | FEDFUNDS | Monetary Policy |
| Consumer Price Index | CPIAUCSL | Inflation |
| Unemployment Rate | UNRATE | Labor Market |
| 10-Year Treasury Yield | DGS10 | Financial Market Expectations |
| Industrial Production Index | INDPRO | Real Economic Activity |
| Recession Indicator | USREC | Business Cycle |

---

# Feature Engineering

Macroeconomic variables often contain strong long-term trends. To emphasize economic cycles rather than levels, the data is transformed into **year-over-year changes and spreads**.

### Year-over-Year Changes

| Feature | Transformation |
|---|---|
| Inflation | pct_change(12) |
| Industrial Production Growth | pct_change(12) |

### One-Year Absolute Changes

| Feature | Transformation |
|---|---|
| Fed Funds Rate Change | diff(12) |
| Unemployment Change | diff(12) |
| 10-Year Yield Change | diff(12) |

### Yield Curve Signal

yield_spread = ten_year_yield - fed_funds

This measures the slope of the yield curve, which is widely used as a recession indicator.

---

# Target Variable

The model predicts **recession probability 12 months ahead**.

recession_target = recession.shift(-12)

This means macroeconomic conditions today are used to predict whether the economy will be in recession **one year in the future**.

---

# Modeling Approach

Macroeconomic indicators evolve gradually, so the dataset is converted into **time sequences** before training.

Each observation contains:

12 months of macroeconomic history → recession probability

Input tensor structure:

(samples, timesteps, features)

Example:

(732, 12, 6)

Meaning:
- 732 sequences
- 12 months of macroeconomic history
- 6 engineered macro features

---

# Model Architecture

The LSTM network architecture:

Input Layer (12 timesteps, 6 features)  
↓  
LSTM (32 units)  
↓  
Dropout (0.2)  
↓  
LSTM (16 units)  
↓  
Dropout (0.2)  
↓  
Dense Layer (sigmoid)

The final output represents:

recession probability ∈ [0,1]

---

# Evaluation Metrics

Because recessions are rare events, multiple evaluation metrics are used.

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Reliability of recession predictions |
| Recall | Ability to detect recessions |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Ability to rank recession risk |

---

# Results

The model successfully detects recession signals but produces some false positives due to **extreme class imbalance**.

Key observations:

- High recall means the model identifies recession periods effectively.
- Precision is low because some non-recession periods are flagged as recession.
- ROC-AUC indicates moderate predictive ability.

This behavior is typical for **early-warning economic models**, where detecting potential recessions is often prioritized over avoiding false alarms.

---

# Limitations

Several limitations affect the model:

- Recessions are rare events, leading to severe class imbalance.
- The feature set is limited to a small number of macro indicators.
- Only a single chronological train/test split is used.
- Structural economic changes may alter historical relationships.

---

# Possible Improvements

Future work could include:

- Adding additional macro-financial indicators
- Using the 10Y–3M Treasury spread
- Applying walk-forward time series validation
- Hyperparameter tuning
- Comparing performance with tree-based models such as XGBoost or Random Forest

---

# Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
TensorFlow / Keras  
Matplotlib  
FRED API  

---


# Key Takeaway

This project demonstrates how **deep learning models can be applied to macroeconomic time-series data** to estimate recession probabilities.

Machine learning does not replace traditional macroeconomic analysis, but it can complement it by identifying **nonlinear temporal patterns in economic data**.
