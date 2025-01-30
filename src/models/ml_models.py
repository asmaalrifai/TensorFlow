import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

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

    # **Select Features and Target**
    features = ["Ptfdolar_Lag1", "Ptfdolar_Lag7", "Ptfdolar_Lag30",
                "Istanbul_anadolu_temp", "Istanbul_anadolu_humidity",
                "Istanbul_avrupa_temp", "Istanbul_avrupa_humidity",
                "Bursa_temp", "Bursa_humidity", "Izmir_temp", "Izmir_humidity",
                "Adana_temp", "Adana_humidity", "Gaziantep_temp", "Gaziantep_humidity",
                "Hour", "Is_Weekend", "Season"]
    
    target = "Ptfdolar"

    # **Fill Missing Values in Features**
    train_df[features] = train_df[features].ffill().bfill()
    test_df[features] = test_df[features].ffill().bfill()

    
    # Encode categorical features
    categorical_features = ["Season", "Is_Weekend"]
    encoder = LabelEncoder()

    for cat in categorical_features:
        train_df[cat] = encoder.fit_transform(train_df[cat])
        test_df[cat] = encoder.transform(test_df[cat])

    # Normalize the features (keep target unchanged)
    scaler = MinMaxScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    return train_df[features], train_df[target], test_df[features], test_df[target]

# Function to generate and save plots
def save_plot(test_file, actual, predicted, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label="Actual", marker="o", linestyle="-", color="blue")
    plt.plot(predicted.index, predicted, label=f"Predicted ({model_name})", marker="s", linestyle="dashed", color="red")
    
    plt.title(f"{model_name} Forecast - {test_file}")
    plt.xlabel("Time")
    plt.ylabel("Ptfdolar")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()

    # Save the graph
    plt.savefig(f"results/graphs/{model_name.lower()}_{test_file}.png")
    plt.close()

# Function to train and evaluate SVR
def train_svr(X_train, y_train, X_test, y_test, test_file):
    print(f"Training SVR for {test_file}...")

    # Train SVR model
    svr_model = SVR(kernel="rbf", C=1.0, epsilon=0.05)  # Reduced complexity

    svr_model.fit(X_train, y_train)

    # Predict on train & test data
    train_predictions = svr_model.predict(X_train)
    test_predictions = svr_model.predict(X_test)

    # Evaluate on both Train and Test
    train_metrics = evaluate_model(y_train, train_predictions)
    test_metrics = evaluate_model(y_test, test_predictions)

    # **Print Final Model Performance**
    print(f"====={test_file} - SVR Model:======")
    print(f"   Train MAE: {train_metrics['MAE']:.4f} | Test MAE: {test_metrics['MAE']:.4f}")
    print(f"   Train SMAPE: {train_metrics['SMAPE (%)']:.2f}% | Test SMAPE: {test_metrics['SMAPE (%)']:.2f}%")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f} | Test RMSE: {test_metrics['RMSE']:.4f}")

    # Save predictions
    predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": test_predictions}, index=y_test.index)
    predictions_df.to_csv(f"results/svr_predictions_{test_file}.csv")

    # Generate and save plot
    save_plot(test_file, y_test, pd.Series(test_predictions, index=y_test.index), "SVR")

    return {"Model": "SVR", "Test File": test_file, **test_metrics}

# Function to train and evaluate XGBoost
def train_xgboost(X_train, y_train, X_test, y_test, test_file):
    print(f"Training XGBoost for {test_file}...")

    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", 
                             n_estimators=50, 
                             learning_rate=0.1, 
                             n_jobs=-1) 

    xgb_model.fit(X_train, y_train)

    # Predict on train & test data
    train_predictions = xgb_model.predict(X_train)
    test_predictions = xgb_model.predict(X_test)

    # Evaluate on both Train and Test
    train_metrics = evaluate_model(y_train, train_predictions)
    test_metrics = evaluate_model(y_test, test_predictions)

    # **Print Final Model Performance**
    print(f"======{test_file} - XGBoost Model:======")
    print(f"   Train MAE: {train_metrics['MAE']:.4f} | Test MAE: {test_metrics['MAE']:.4f}")
    print(f"   Train SMAPE: {train_metrics['SMAPE (%)']:.2f}% | Test SMAPE: {test_metrics['SMAPE (%)']:.2f}%")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f} | Test RMSE: {test_metrics['RMSE']:.4f}")

    # Save predictions
    predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": test_predictions}, index=y_test.index)
    predictions_df.to_csv(f"results/xgb_predictions_{test_file}.csv")

    # Generate and save plot
    save_plot(test_file, y_test, pd.Series(test_predictions, index=y_test.index), "XGBoost")

    return {"Model": "XGBoost", "Test File": test_file, **test_metrics}

# Run SVR and XGBoost models in parallel
results = Parallel(n_jobs=-1)(
    delayed(train_svr)(*load_data(train_file, test_file), test_file) for train_file, test_file in zip(train_files, test_files)
) + Parallel(n_jobs=-1)(
    delayed(train_xgboost)(*load_data(train_file, test_file), test_file) for train_file, test_file in zip(train_files, test_files)
)

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("results/ml_results.csv", index=False)

# Compute the average of MAE, SMAPE, and RMSE
average_results = results_df.groupby("Model")[["MAE", "SMAPE (%)", "RMSE"]].mean().reset_index()
average_results.to_csv("results/ml_results_avg.csv", index=False)

print("ML results and average metrics successfully saved in `results/`")
