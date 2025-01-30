import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
import matplotlib.pyplot as plt
#Transformer model
# Load the dataset
data = pd.read_csv('22-23.csv')


# Define features and target variable
target_column = "Ptfdolar"
feature_columns = [
    "Ptfdolar_Lag1", "Ptfdolar_Lag7", "Ptfdolar_Lag30",
    "Dolar", "Istanbul_anadolu_temp", "Istanbul_anadolu_humidity",
    "Istanbul_avrupa_temp", "Istanbul_avrupa_humidity",
    "Bursa_temp", "Bursa_humidity",
    "Izmir_temp", "Izmir_humidity",
    "Adana_temp", "Adana_humidity",
    "Gaziantep_temp", "Gaziantep_humidity",
    "Hour", "Is_Weekend", "Mevsim", "Day/Night", 
    "Year", "Month", "Day"
]

X = data[feature_columns].values
y = data[target_column].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

# Reshape data for Transformer input
timesteps = 1
X_train_reshaped = X_train.reshape((X_train.shape[0], timesteps, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))

# Define the Transformer model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-Head Attention
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = Add()([attention, inputs])
    attention = LayerNormalization(epsilon=1e-6)(attention)

    # Feed Forward Network
    outputs = Dense(ff_dim, activation="relu")(attention)
    outputs = Dropout(dropout)(outputs)
    outputs = Dense(inputs.shape[-1])(outputs)
    outputs = Add()([outputs, attention])
    outputs = LayerNormalization(epsilon=1e-6)(outputs)
    return outputs

input_shape = (timesteps, X_train_reshaped.shape[2])
inputs = Input(shape=input_shape)
x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
x = Dense(1)(x)
outputs = x[:, -1, :]  # Use only the last output

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train the model
history = model.fit(
    X_train_reshaped, y_train,
    validation_data=(X_test_reshaped, y_test),
    epochs=100,
    batch_size=64,
    verbose=1
)

# Save the model
model.save('transformer_model.keras')

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
results.to_csv('transformer_predictions.csv', index=False)

# Plot training history
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training History')
plt.show()

# Plot actual vs predicted values (last 100 samples)
plt.figure()
plt.plot(y_test[-100:], label='Actual')
plt.plot(y_pred[-100:], label='Predicted')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Ptfdolar')
plt.title('Actual vs Predicted Values (Last 100 Samples)')
plt.show()
