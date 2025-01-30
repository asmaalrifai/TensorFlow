import pandas as pd

# Load the dataset
file_path = 'C:\\Users\\amust\\Desktop\\learning-from-data-asma-ahmet\\learning-from-data-asma-ahmet\\data\\processed\\smfdb_final.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

data['Ptfdolar_Lag1'] = data['Ptfdolar'].shift(1)
data['Ptfdolar_Lag7'] = data['Ptfdolar'].shift(7)
data['Ptfdolar_Lag30'] = data['Ptfdolar'].shift(30)

season_mapping = {'Winter': 4, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
data['Season'] = data['Season'].map(season_mapping)


data['Is_Weekend'] = data['Is_Weekend'].apply(lambda x: 1 if x == 'Yes' else 0)


# Drop unnecessary columns
unnecessary_columns = [
    "Ptf", "Ptfeuro", "Smf", 
    "Minalisfiyati", "Minsatisfiyati", 
    "Maxalisfiyati", "Maxsatisfiyati",
    "Ptf_Lag1", "Ptf_Lag7", "Ptf_Lag30",
    "Blokeslesmemiktari", "Saatlikeslesmemiktari", "Mineslesmefiyati", "Maxeslesmefiyati",
    "Talepislemhacmi", "Arzislemhacmi", "Gerceklesendogalgaz", "Gerceklesenbarajli",
    "Gerceklesenlinyit", "Gerceklesenakarsu", "Gerceklesenithalkomur", "Gerceklesenruzgar",
    "Gerceklesengunes", "Gerceklesenfueloil", "Gerceklesenjeotermal", "Gerceklesenasfaltitkomur",
    "Gerceklesentaskomur", "Gerceklesenbiyokutle", "Gerceklesennafta", "Gerceklesenlng",
    "Gerceklesenuluslararasi", "Gerceklesentoplam", "Euro", "Smfeuro", "Smf_Lag1","Smf_Lag7","Smf_Lag30"
]
data = data.drop(columns=unnecessary_columns, errors='ignore')

# Drop rows with missing values
data = data.dropna()

# Create new lag columns using 'Ptfdolar'
data["Ptfdolar_Lag1"] = data["Ptfdolar"].shift(1)
data["Ptfdolar_Lag7"] = data["Ptfdolar"].shift(7)
data["Ptfdolar_Lag30"] = data["Ptfdolar"].shift(30)

# Parse 'Tarih' column as datetime if not already done
data['Tarih'] = pd.to_datetime(data['Tarih'])

# Remove rows after 20.02.2023:23:00 for 'Ptfdolar' and lag columns
#cutoff_time = pd.Timestamp("2023-02-20 23:00:00")
#cutoff_time1 = pd.Timestamp("2023-02-21 00:00:00")
#cutoff_time2 = pd.Timestamp("2023-02-21 06:00:00")
#cutoff_time3 = pd.Timestamp("2023-02-22 05:00:00")
#data.loc[data['Tarih'] > cutoff_time, ['Ptfdolar','Smfdolar']] = 0
#data.loc[data['Tarih'] > cutoff_time1, ['Ptfdolar_Lag1']] = 0
#data.loc[data['Tarih'] > cutoff_time2, ['Ptfdolar_Lag7']] = 0
#data.loc[data['Tarih'] > cutoff_time3, ['Ptfdolar_Lag30']] = 0
# Save the updated dataset
updated_file_path = 'actualdata.csv'
data.to_csv(updated_file_path, index=False)

print(f"Updated dataset saved to: {updated_file_path}")
