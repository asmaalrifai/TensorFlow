import pandas as pd

# Load the processed dataset with lag features
df = pd.read_csv("data/processed/smfdb_final_with_lags.csv", parse_dates=["Tarih"], index_col="Tarih")

# Define test start and end dates
test_dates = [
    ("2023-02-20 23:00", "2023-02-21 23:00"),
    ("2023-02-21 23:00", "2023-02-22 23:00"),
    ("2023-02-22 23:00", "2023-02-23 23:00"),
    ("2023-02-23 23:00", "2023-02-24 23:00"),
    ("2023-02-24 23:00", "2023-02-25 23:00"),
    ("2023-02-25 23:00", "2023-02-26 23:00"),
    ("2023-02-26 23:00", "2023-02-27 23:00"),
]

# Loop through each test set and create corresponding train-test splits
for i, (train_end, test_end) in enumerate(test_dates):
    train_df = df.loc[:train_end]
    test_df = df.loc[pd.to_datetime(train_end) + pd.Timedelta(hours=1) : test_end]
    
    # Save files
    train_df.to_csv(f"data/processed/modeling/train_data_{i+1}.csv")
    test_df.to_csv(f"data/processed/modeling/test_data_{i+1}.csv")

print("Training and Testing Data Prepared for 7-Day Evaluation with Ptfdolar lag features.")
