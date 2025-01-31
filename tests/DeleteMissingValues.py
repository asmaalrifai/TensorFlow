import pandas as pd

# Load the processed data
file_path = "data/processed/processed_data.csv"
data = pd.read_csv(file_path)

# Drop rows with missing values in critical columns
data_cleaned = data.dropna(subset=['Smf_rolling_mean', 'Smf_lag1'])

# Save the cleaned data
output_path = "data/processed/processed_data_cleaned.csv"
data_cleaned.to_csv(output_path, index=False)

print(f"Cleaned data saved to {output_path}.")
print("Number of rows in the cleaned data:", len(data_cleaned))
