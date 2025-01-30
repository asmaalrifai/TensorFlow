import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Gru Model
# Load the dataset
data = pd.read_csv('actualdata.csv')

# Drop unnecessary columns
unnecessary_columns = [
    "Ptf", "Ptfeuro", "Smf", 
    "Minalisfiyati", "Minsatisfiyati", 
    "Maxalisfiyati", "Maxsatisfiyati"
]
data = data.drop(columns=unnecessary_columns, errors='ignore')

# Drop rows with NaN values
data = data.dropna()

# Define features and target variable
target_column = "Ptfdolar"
feature_columns = [
    "Ptfdolar_Lag1", "Ptfdolar_Lag7", "Ptfdolar_Lag30", "Dolar",
    "Istanbul_anadolu_temp", "Istanbul_anadolu_humidity",
    "Istanbul_avrupa_temp", "Istanbul_avrupa_humidity",
    "Bursa_temp", "Bursa_humidity",
    "Izmir_temp", "Izmir_humidity",
    "Is_Weekend", "Mevsim", "Day/Night"
]

X = data[feature_columns].values
y = data[target_column].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

# Reshape data for GRU input
timesteps = 1
X_train_reshaped = X_train.reshape((X_train.shape[0], timesteps, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))

# Define the GRU model
model = Sequential([
    GRU(64, activation='relu', return_sequences=True, input_shape=(timesteps, X_train_reshaped.shape[2])),
    Dropout(0.2),
    GRU(32, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model with EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train_reshaped, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test_reshaped, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Save the model
model.save('gru_model.keras')

# Make predictions
y_pred = model.predict(X_test_reshaped)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")

# Save predictions
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred.flatten()
})
results.to_csv('gru_predictions.csv', index=False)

# Plot training history
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training History')
plt.show()

# Plot actual vs predicted values
plt.figure()
plt.plot(y_test[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Ptfdolar')
plt.title('Actual vs Predicted Values (First 100 Samples)')
plt.show()
