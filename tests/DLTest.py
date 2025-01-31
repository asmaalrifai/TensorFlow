import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load your trained model
#model = load_model('gru_model.keras')
#model = load_model('transformer_model.keras')
model = load_model('lstm_electricity_model.keras')

# Load the cleaned data
data = pd.read_csv('C:\\Users\\amust\\Desktop\\learning-from-data-asma-ahmet\\learning-from-data-asma-ahmet\\pre\\pre2.csv')
realdata = pd.read_csv('C:\\Users\\amust\\Desktop\\learning-from-data-asma-ahmet\\learning-from-data-asma-ahmet\\\\pre\\act2.csv')

# Initial lag values (fallback if missing in data)
initial_ptfdolar_lag1 = data['Ptfdolar_Lag1'].iloc[-1] if not pd.isna(data['Ptfdolar_Lag1'].iloc[-1]) else 0.5
initial_ptfdolar_lag7 = data['Ptfdolar_Lag7'].iloc[-1] if not pd.isna(data['Ptfdolar_Lag7'].iloc[-1]) else 0.5
initial_ptfdolar_lag30 = data['Ptfdolar_Lag30'].iloc[-1] if not pd.isna(data['Ptfdolar_Lag30'].iloc[-1]) else 0.5

# Prepare for recursive prediction
predictions = []
data['Predicted_Ptfdolar'] = None  # Add a column to store predictions

# Check if data and realdata are valid
if data is None or realdata is None:
    raise ValueError("Failed to load one or more datasets. Check file paths.")

# Modelin beklediği sütunlar
expected_columns =     [
    "Ptfdolar_Lag1", "Ptfdolar_Lag7", "Ptfdolar_Lag30","Dolar",
    "Istanbul_anadolu_temp", "Istanbul_anadolu_humidity",
    "Istanbul_avrupa_temp", "Istanbul_avrupa_humidity",
    "Bursa_temp", "Bursa_humidity",
    "Izmir_temp", "Izmir_humidity",
    "Adana_temp", "Adana_humidity",
    "Gaziantep_temp", "Gaziantep_humidity",
    "Hour", "Is_Weekend", "Day/Night"
] # Modelde kullanılan sütunlar

# Iterate over each row for recursive prediction
for i in range(len(data)):
    row = data.iloc[i].copy()

    # Fill missing lag values
    row['Ptfdolar_Lag1'] = row['Ptfdolar_Lag1'] if not pd.isna(row['Ptfdolar_Lag1']) else initial_ptfdolar_lag1
    row['Ptfdolar_Lag7'] = row['Ptfdolar_Lag7'] if not pd.isna(row['Ptfdolar_Lag7']) else initial_ptfdolar_lag7
    row['Ptfdolar_Lag30'] = row['Ptfdolar_Lag30'] if not pd.isna(row['Ptfdolar_Lag30']) else initial_ptfdolar_lag30

    # Sadece modelin kullandığı sütunları seç
    row = row[expected_columns]

    # Veriyi reshape et (samples, timesteps, features)
    row_reshaped = np.array(row, dtype='float32').reshape((1, 1, len(row)))

    # Tahmin yap
    predicted_value = model.predict(row_reshaped)[0][0]
    predictions.append(predicted_value)

    # Tahmin edilen sütunu güncelle
    data.at[i, 'Predicted_Ptfdolar'] = predicted_value

    # Lag değerlerini güncelle
    initial_ptfdolar_lag1 = predicted_value
    if i >= 6:
        initial_ptfdolar_lag7 = predictions[-7]
    if i >= 29:
        initial_ptfdolar_lag30 = predictions[-30]

# Add real Ptfdolar values to the data
data['Actual_Ptfdolar'] = realdata['Ptfdolar']

# Calculate error metrics
mse = mean_squared_error(data['Actual_Ptfdolar'], data['Predicted_Ptfdolar'])
mape = mean_absolute_percentage_error(data['Actual_Ptfdolar'], data['Predicted_Ptfdolar'])

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# Save results
results = data[['Tarih', 'Actual_Ptfdolar', 'Predicted_Ptfdolar']]
results.to_csv('transformer_prediction_results.csv', index=False)

# Plot Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(data['Tarih'], data['Actual_Ptfdolar'], label='Actual Ptfdolar', marker='o')
plt.plot(data['Tarih'], data['Predicted_Ptfdolar'], label='Predicted Ptfdolar', marker='x')
plt.xlabel('Tarih')
plt.ylabel('Ptfdolar')
plt.title('Actual vs Predicted Ptfdolar')
plt.legend()
plt.xticks(ticks=data.index[::10], labels=data['Tarih'].iloc[::10], rotation=45)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Predictions and comparison saved to 'prediction_results.csv'")
