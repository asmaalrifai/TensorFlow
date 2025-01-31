import pandas as pd

# Load cleaned data
data_cleaned = pd.read_csv("data/processed/processed_data_cleaned.csv")

# Split the data
train_data = data_cleaned[data_cleaned['Tarih'] < "2023-02-20"]
test_data = data_cleaned[(data_cleaned['Tarih'] >= "2023-02-21") & (data_cleaned['Tarih'] <= "2023-02-27")]

# Save train and test sets
train_data.to_csv("data/processed/train_data.csv", index=False)
test_data.to_csv("data/processed/test_data.csv", index=False)

print("Training data saved to data/processed/train_data.csv")
print("Testing data saved to data/processed/test_data.csv")
