import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression

def feature_selection(df):
    """Performs feature selection using SelectKBest and Random Forest."""

    # Drop non-numeric and target columns
    feature_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(["Smf", "Ptf"])
    X = df[feature_cols]
    y = df["Smf"]

    # SelectKBest Feature Selection
    selector = SelectKBest(score_func=f_regression, k=10)
    X_new = selector.fit_transform(X, y)
    selected_feature_names = [feature_cols[i] for i in selector.get_support(indices=True)]

    print("Top 10 Features from SelectKBest:", selected_feature_names)

    # Feature Importance with Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    print("\nFeature Importance (Random Forest):")
    print(feature_importances.head(10))

    return selected_feature_names, feature_importances

# Load dataset with time features
df = pd.read_csv("data/processed/smfdb_time_features.csv")

# Run feature selection
selected_features, feature_importance = feature_selection(df)


# Save feature importance results to CSV
feature_importance.to_csv("data/processed/feature_importance.csv", index=True)
print("Feature importance saved to data/processed/feature_importance.csv")
