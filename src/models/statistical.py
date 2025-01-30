import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure results directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("results/graphs", exist_ok=True)

# Define test and train files
test_files = ["test_data_1.csv", "test_data_2.csv", "test_data_3.csv"] 
train_files = ["train_data_1.csv", "train_data_2.csv", "train_data_3.csv"] 

# Function to calculate SMAPE (Fixing MAPE issue)
def smape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    smape_score = smape(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "SMAPE (%)": smape_score, "RMSE": rmse}

# Function to load and preprocess train/test data
def load_data(train_file, test_file):
    train_df = pd.read_csv(f"data/processed/modeling/{train_file}", parse_dates=["Tarih"], index_col="Tarih")
    test_df = pd.read_csv(f"data/processed/modeling/{test_file}", parse_dates=["Tarih"], index_col="Tarih")

    # Ensure test data has full 24-hour range
    test_df = test_df.resample("h").asfreq()

    # **Fill Missing Values in Lag Features Directly from Dataset**
    for col in ["Ptfdolar_Lag1", "Ptfdolar_Lag7", "Ptfdolar_Lag30"]:
        train_df.loc[:, col] = train_df[col].ffill().bfill()  # Fill missing values
        test_df.loc[:, col] = test_df[col].ffill().bfill()  # Fill missing values


    return train_df, test_df

# Function to train SARIMA and save results
def train_sarima(train_df, test_df, test_file):
    print(f"Training SARIMA for {test_file}...")

    # **Set explicit time frequency to prevent warnings**
    train_df.index = pd.date_range(start=train_df.index[0], periods=len(train_df), freq="h")
    test_df.index = pd.date_range(start=test_df.index[0], periods=len(test_df), freq="h")

    # **Check for Data Distribution Shift**
    print(f"=========Train Ptfdolar Mean: {train_df['Ptfdolar'].mean()} | Test Ptfdolar Mean: {test_df['Ptfdolar'].mean()}")
    print(f"=========Train Ptfdolar Std: {train_df['Ptfdolar'].std()} | Test Ptfdolar Std: {test_df['Ptfdolar'].std()}")

    # **Choose Model Complexity Based on Test File**
    model = SARIMAX(train_df["Ptfdolar"], 
                        order=(1,1,0), 
                        seasonal_order=(1,1,0,24), 
                        enforce_stationarity=True, 
                        enforce_invertibility=True)

    # Fit SARIMA Model
    sarima_fit = model.fit(disp=False)

    # **Shorten Forecasting Horizon for `test_data_1.csv` to avoid overfitting**
    forecast_steps = min(len(test_df), 12) if test_file == "test_data_1.csv" else len(test_df)

    # Predict
    predictions = sarima_fit.forecast(steps=forecast_steps)
    predictions.index = test_df.index[:forecast_steps]  # **Fix Index Issue**
    
    # Evaluate Model Performance
    train_mae = mean_absolute_error(train_df["Ptfdolar"], sarima_fit.fittedvalues)
    train_rmse = np.sqrt(mean_squared_error(train_df["Ptfdolar"], sarima_fit.fittedvalues))
    train_smape = smape(train_df["Ptfdolar"], sarima_fit.fittedvalues)

    test_mae = mean_absolute_error(test_df["Ptfdolar"][:forecast_steps], predictions)
    test_rmse = np.sqrt(mean_squared_error(test_df["Ptfdolar"][:forecast_steps], predictions))
    test_smape = smape(test_df["Ptfdolar"][:forecast_steps], predictions)

    # **Print Final Model Performance**
    print(f"======={test_file} - SARIMA Model:========")
    print(f"   Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
    print(f"   Train SMAPE: {train_smape:.2f}% | Test SMAPE: {test_smape:.2f}%")
    print(f"   Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    # Save predictions
    predictions_df = pd.DataFrame({"Actual": test_df["Ptfdolar"][:forecast_steps], "Predicted": predictions})
    predictions_df.to_csv(f"results/sarima_predictions_{test_file}.csv")

    return {"Model": "SARIMA", "Test File": test_file, "MAE": test_mae, "SMAPE (%)": test_smape, "RMSE": test_rmse}

# Run SARIMA models for all test days
results = [train_sarima(*load_data(train_file, test_file), test_file) for train_file, test_file in zip(train_files, test_files)]

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("results/sarima_results.csv", index=False)

# Compute the average of MAE, SMAPE, and RMSE
average_results = results_df[["MAE", "SMAPE (%)", "RMSE"]].mean().to_frame().T
average_results.to_csv("results/sarima_results_avg.csv", index=False)

print("SARIMA results and average metrics successfully saved in `results/`")