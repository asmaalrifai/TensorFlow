# Project Documentation: System Marginal Price Forecasting

## 1. Project Overview
This project focuses on developing a forecasting system for Turkey's electricity market using machine learning techniques. The objective is to predict System Marginal Price (SMF) and Market Clearing Price (PTF) through data processing, feature engineering, and model development.

### Objectives:
- Apply data science and machine learning techniques for electricity price forecasting.
- Develop a structured pipeline from data preprocessing to model evaluation.
- Compare statistical models, machine learning models, and deep learning models.
- Deliver a final dataset and comprehensive analysis report.

---

## 2. Data Collection and Processing
### Raw Data Sources:
- EPİAŞ Market Data: Electricity prices (SMF and PTF).
- Weather Data (API-based): Temperature and humidity from major cities.
- Economic Data: Exchange rates (USD, EUR).

### Data Preprocessing Steps:
- Loaded data from `smfdb_cleaned.csv`.
- Checked missing values and handled NaNs.
- Converted `Tarih` (Date) column to datetime format.
- Saved processed dataset as `smfdb_final.csv`.

---

## 3. Feature Engineering
### Time-Based Features:
- Extracted hour, day, month, year, weekday, and weekend indicators.
- Assigned correct `Mevsim` (Season) values.
- Mapped `Mevsim` to Winter, Spring, Summer, and Autumn.

### Rolling Statistics:
- Computed 30-day and weekly moving averages for SMF and PTF.
- Plotted rolling trends for better visualization.

### Lag Features:
- Created lagged values for SMF and PTF:
  - `Smf_Lag1`, `Smf_Lag7`, `Smf_Lag30`
  - `Ptf_Lag1`, `Ptf_Lag7`, `Ptf_Lag30`
- Addressed missing values due to lag calculations by either dropping or filling NaNs.

### Weather-Derived Features:
- Integrated temperature and humidity data for Istanbul, Bursa, Izmir, Adana, and Gaziantep.

### Economic Indicators:
- Included Dolar, Euro, Ptfdolar, Ptfeuro, Smfdolar, and Smfeuro.
- Plotted exchange rate trends over time.

---

## 4. Feature Selection
### Techniques Applied:
- Applied mRMR (Minimum Redundancy Maximum Relevance) for feature importance ranking.
- Conducted correlation heatmaps for different feature groups:
  - Electricity Prices (SMF, PTF, USD, EUR)
  - Weather Conditions (Temperature, Humidity)
  - Energy Production (Wind, Solar, Hydro, Lignite, etc.)
- Evaluated feature importance using Random Forest.

---

## 5. Model Preparation and Next Steps
### Immediate Next Steps:
1. Train baseline models (ARIMA, SARIMA, PROPHET) for statistical forecasting.
2. Implement machine learning models (SVR, XGBoost, Random Forest).
3. Train deep learning models (LSTM, Transformers).
4. Evaluate models using MAPE, MAE, and RMSE.

### Final Deliverables:
- Final processed dataset: `smfdb_final_clean.csv`
- Complete Jupyter Notebook: `02_data_exploration.ipynb`
- Final report with model comparisons and insights
- Presentation summarizing results

---

The next step is to proceed with model training. Would you like to begin with statistical models (ARIMA, SARIMA) or move directly to machine learning models?

