import pandas as pd

# Load the processed data
data = pd.read_csv("C:\Users\Asmz\github-classroom\FSMUNIV\learning-from-data-asma-ahmet\data\raw\combined_weather_data.csv")

# Check for missing values
print(data.isnull().sum())
