# Model Documentation

## Statistical Models

### 1. ARIMA (Auto-Regressive Integrated Moving Average)
- **Purpose**: Forecasts time-series data using past values, differences, and lagged errors.
- **Input**:
  - Time series data (e.g., `Smf`).
  - Order parameters `(p, d, q)`:
    - `p`: Number of lag observations.
    - `d`: Degree of differencing.
    - `q`: Size of moving average window.
- **Output**: Predicted time series values.
- **Implementation**: Defined in `statistical.py`.

### 2. SARIMA (Seasonal ARIMA)
- **Purpose**: Extends ARIMA to handle seasonality.
- **Input**:
  - Time series data.
  - Order parameters `(p, d, q)`.
  - Seasonal parameters `(P, D, Q, m)`:
    - `m`: Seasonal period.
- **Output**: Predicted time series values.
- **Implementation**: Defined in `statistical.py`.

---

## Machine Learning Models

### 1. Random Forest Regressor
- **Purpose**: Predicts numerical outcomes using decision trees.
- **Input**:
  - Feature-engineered dataset (e.g., weather, demand).
  - Hyperparameters:
    - `n_estimators`: Number of trees (default: 100).
    - `max_depth`: Maximum tree depth (default: None).
- **Output**: Predicted numerical values.
- **Implementation**: Defined in `ml_models.py`.

### 2. Support Vector Regressor (SVR)
- **Purpose**: Models complex relationships using kernel-based regression.
- **Input**:
  - Feature-engineered dataset.
  - Hyperparameters:
    - `C`: Regularization parameter (default: 1.0).
    - `kernel`: Type of kernel (e.g., `linear`, `rbf`).
- **Output**: Predicted numerical values.
- **Implementation**: Defined in `ml_models.py`.

### 3. XGBoost Regressor
- **Purpose**: Uses gradient boosting for robust predictions.
- **Input**:
  - Feature-engineered dataset.
  - Hyperparameters:
    - `learning_rate`: Step size shrinkage (default: 0.1).
    - `n_estimators`: Number of boosting rounds (default: 100).
- **Output**: Predicted numerical values.
- **Implementation**: Defined in `ml_models.py`.

---

## Deep Learning Models

### 1. LSTM (Long Short-Term Memory)
- **Purpose**: Captures long-term dependencies in sequential data.
- **Input**:
  - Feature-engineered sequential data (e.g., `Smf` values over time).
  - Hyperparameters:
    - `units`: Number of LSTM units (default: 50).
    - `dropout`: Fraction of units to drop (default: 0.2).
- **Output**: Predicted time-series values.
- **Implementation**: Defined in `dl_models.py`.

### 2. GRU (Gated Recurrent Unit)
- **Purpose**: Similar to LSTM but computationally simpler.
- **Input**:
  - Feature-engineered sequential data.
  - Hyperparameters:
    - `units`: Number of GRU units (default: 50).
    - `dropout`: Fraction of units to drop (default: 0.2).
- **Output**: Predicted time-series values.
- **Implementation**: Defined in `dl_models.py`.

### 3. Temporal Fusion Transformer (TFT) [Placeholder]
- **Purpose**: Advanced transformer-based model for time-series forecasting.
- **Status**: Pending implementation.

---

## Hyperparameter Optimization

### Techniques Used
- **GridSearchCV** (Machine Learning Models):
  - Optimized parameters for `RandomForestRegressor` and `SVR`.
- **Manual Tuning** (Deep Learning Models):
  - Adjusted `units`, `dropout`, and `learning_rate` for LSTM and GRU.

### Example (Random Forest)
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
```

---

## Notes
- Each model is tested interactively in `02_data_exploration.ipynb`.
- Further hyperparameter tuning can be performed for specific use cases.

