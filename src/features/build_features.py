# build_features.py
import pandas as pd

feature_columns = []  # Global placeholder for consistent columns

def build_features(df):
    global feature_columns
    df_encoded = pd.get_dummies(df, columns=['basement', 'property_type'], drop_first=True)
    if not feature_columns:
        feature_columns = df_encoded.columns.tolist()
    else:
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_columns]
    return df_encoded