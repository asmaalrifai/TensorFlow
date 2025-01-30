import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath, parse_dates=["Tarih"])

def clean_data(df):
    """Handle missing values, format date, and preprocess."""
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing numerical values
    df.fillna(df.mode().iloc[0], inplace=True)  # Fill missing categorical values
    df["Year"] = df["Tarih"].dt.year
    df["Month"] = df["Tarih"].dt.month
    df["Day"] = df["Tarih"].dt.day
    df["Hour"] = df["Tarih"].dt.hour
    df["Is_Weekend"] = df["Tarih"].dt.weekday.isin([5, 6]).astype(int)
    return df

def scale_data(df):
    """Normalize numerical features using MinMaxScaler, handling infinity and large values."""
    scaler = MinMaxScaler()
    
    # Select only numerical columns (excluding 'Tarih')
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(["Tarih"])

    # Replace infinite values with NaN
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

    # Fill any remaining NaN values with the column mean
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    # Apply MinMax Scaling
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df


def save_processed_data(df, output_path):
    """Save the cleaned dataset."""
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

if __name__ == "__main__":
    # File paths
    raw_file_path = "data/raw/5city-5year_day_mevsim.csv"
    processed_file_path = "data/processed/smfdb_cleaned.csv"
    
    # Execute preprocessing
    df = load_data(raw_file_path)
    df = clean_data(df)
    df = scale_data(df)
    save_processed_data(df, processed_file_path)
