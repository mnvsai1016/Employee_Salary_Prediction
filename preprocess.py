import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Define base directory to safely locate model assets relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_files"


def load_model_asset(filename: str):
    """Safely load a pickle file from the model_files directory."""
    filepath = MODEL_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Required model asset '{filename}' not found in {MODEL_DIR}. "
            "Please extract model_files.zip before running."
        )
    with open(filepath, "rb") as f:
        return pickle.load(f)


def Preprocess(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw input survey data for salary prediction model inference.
    
    Parameters:
        input_data (pd.DataFrame): DataFrame containing raw user input values.
        
    Returns:
        pd.DataFrame: Fully preprocessed and feature-aligned DataFrame ready for model input.
    """
    df = input_data.copy()

    # 1. Age Mapping
    age_map = {
        'Under 18 years old': 0,
        '18-24 years old': 1,
        '25-34 years old': 2,
        '35-44 years old': 3,
        '45-54 years old': 4,
        '55-64 years old': 5,
        '65 years or older': 6,
        'Prefer not to say': np.nan
    }
    df['AgeEncoded'] = df['Age'].map(age_map)

    # 2. Education Level Mapping
    education_map = {
        'Primary/elementary school': 0,
        'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 1,
        'Some college/university study without earning a degree': 2,
        'Associate degree (A.A., A.S., etc.)': 3,
        "Bachelor's degree (B.A., B.S., B.Eng., etc.)": 4,
        'Bachelor’s degree (B.A., B.S., B.Eng., etc.)': 4,
        "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": 5,
        'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)': 5,
        'Professional degree (JD, MD, Ph.D, Ed.D, etc.)': 6,
        'Something else': 0
    }
    df['EdLevelEncoded'] = df['EdLevel'].map(education_map)

    # Drop raw object columns
    df.drop(columns=['Age', 'EdLevel'], errors='ignore', inplace=True)

    # 3. Country Label Encoding
    le_country = load_model_asset("country_encoder.pkl")
    known_countries = set(le_country.classes_)
    df['Country'] = df['Country'].apply(lambda x: x if x in known_countries else le_country.classes_[0])
    df['CountryEncoded'] = le_country.transform(df['Country'])
    df.drop(columns=['Country'], errors='ignore', inplace=True)

    # 4. Job Role / DevType One-hot Encoding
    df_dummies = pd.get_dummies(df['DevType'], prefix='DevType').astype(int)
    df = pd.concat([df.drop(columns=['DevType'], errors='ignore'), df_dummies], axis=1)

    # 5. Employment Multi-hot Encoding
    df['EmploymentList'] = df['Employment']
    mlb = MultiLabelBinarizer()
    employment_dummies = pd.DataFrame(
        mlb.fit_transform(df['EmploymentList']),
        columns=mlb.classes_,
        index=df.index
    )
    df = pd.concat([df.drop(columns=['Employment', 'EmploymentList'], errors='ignore'), employment_dummies], axis=1)

    # 6. RemoteWork One-hot Encoding
    remote_dummies = pd.get_dummies(df['RemoteWork'], prefix='RemoteWork')
    df = pd.concat([df.drop(columns=['RemoteWork'], errors='ignore'), remote_dummies], axis=1)

    # 7. Technology Multi-hot Encodings
    tech_columns = [
        'LanguageHaveWorkedWith',
        'DatabaseHaveWorkedWith',
        'PlatformHaveWorkedWith',
        'WebframeHaveWorkedWith',
        'EmbeddedHaveWorkedWith',
        'MiscFrameworks',
        'ToolsHaveWorkedWith'
    ]

    for col_name in tech_columns:
        if col_name in df.columns:
            mlb = MultiLabelBinarizer()
            tech_dummies = pd.DataFrame(
                mlb.fit_transform(df[col_name]),
                columns=mlb.classes_,
                index=df.index
            )
            df = pd.concat([df.drop(columns=[col_name]), tech_dummies], axis=1)

    # 8. Work Experience Log Scaling
    if 'WorkExp' in df.columns:
        df['WorkExp'] = np.log1p(df['WorkExp'].fillna(0))

    # 9. Feature Alignment with Saved Feature List using reindex (avoids fragmentation)
    saved_features = load_model_asset("features.pkl")
    df = df.reindex(columns=saved_features, fill_value=0)

    return df
