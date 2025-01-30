import pandas as pd

# Function to add time-based features
def add_time_features(df):
    """Extracts time-based features and ensures 'Mevsim' is correctly mapped."""

    # Convert 'Tarih' to datetime
    df["Tarih"] = pd.to_datetime(df["Tarih"])

    # Extract time-based features
    df["Hour"] = df["Tarih"].dt.hour
    df["Day"] = df["Tarih"].dt.day
    df["Month"] = df["Tarih"].dt.month
    df["Year"] = df["Tarih"].dt.year
    df["Weekday"] = df["Tarih"].dt.weekday  # 0 = Monday, 6 = Sunday
    df["Is_Weekend"] = df["Weekday"].isin([5, 6]).astype(int)  # 1 if Saturday/Sunday

    # ✅ Overwrite Mevsim based on Month (Fixing Incorrect Values)
    month_to_season = {
        12: 3, 1: 3, 2: 3,  # Winter
        3: 2, 4: 2, 5: 2,  # Spring
        6: 1, 7: 1, 8: 1,  # Summer
        9: 4, 10: 4, 11: 4  # Autumn
    }
    df["Mevsim"] = df["Month"].map(month_to_season)

    # ✅ Correct Season Mapping
    season_mapping = {1: "Summer", 2: "Spring", 3: "Winter", 4: "Autumn"}
    df["Season"] = df["Mevsim"].map(season_mapping)

    return df

# Function to add lag features for electricity prices and Ptfdolar
def add_lag_features(df):
    """Creates lag features for electricity prices (SMF & PTF) and Ptfdolar to improve forecasting."""
    
    # Sort by date (Tarih)
    df = df.sort_values("Tarih")
    
    # Create lag features for SMF and PTF
    df["Smf_Lag1"] = df["Smf"].shift(1)
    df["Smf_Lag7"] = df["Smf"].shift(7)
    df["Smf_Lag30"] = df["Smf"].shift(30)

    df["Ptf_Lag1"] = df["Ptf"].shift(1)
    df["Ptf_Lag7"] = df["Ptf"].shift(7)
    df["Ptf_Lag30"] = df["Ptf"].shift(30)

    # Create lag features for Ptfdolar
    df["Ptfdolar_Lag1"] = df["Ptfdolar"].shift(1)
    df["Ptfdolar_Lag7"] = df["Ptfdolar"].shift(7)
    df["Ptfdolar_Lag30"] = df["Ptfdolar"].shift(30)
    
    return df

# Load dataset
df = pd.read_csv("data/processed/smfdb_cleaned.csv", parse_dates=["Tarih"])

# Apply time feature extraction
df = add_time_features(df)

# Save intermediate dataset with time features
df.to_csv("data/processed/smfdb_time_features.csv", index=False)
print("Time-based features added and saved!")

# Apply lag feature extraction
df = add_lag_features(df)

# Save final processed dataset with lag features
df.to_csv("data/processed/smfdb_final_with_lags.csv", index=False)
print("Lag features added and final dataset saved!")
