import pandas as pd

# Preprocessing
file_path = "data/raw/smfdb_with_weather19-20.csv"
output_path = "data/processed/cleaned_data.csv"
data = pd.read_csv(file_path)
print("Loaded raw data")

# Clean data
data['Tarih'] = pd.to_datetime(data['Tarih'])
data.fillna(method='ffill', inplace=True)
data.drop_duplicates(inplace=True)
print("Cleaned data. Rows:", len(data))

# Save cleaned data
data.to_csv(output_path, index=False)
print("Cleaned data saved.")

# Feature engineering
cleaned_path = "data/processed/cleaned_data.csv"
processed_output = "data/processed/processed_data.csv"
cleaned_data = pd.read_csv(cleaned_path)

# Feature engineering steps
cleaned_data['hour'] = pd.to_datetime(cleaned_data['Tarih']).dt.hour
cleaned_data['Smf_rolling_mean'] = cleaned_data['Smf'].rolling(window=7).mean()
cleaned_data['Smf_lag1'] = cleaned_data['Smf'].shift(1)

# Save processed data
cleaned_data.to_csv(processed_output, index=False)
print("Processed data saved.")
