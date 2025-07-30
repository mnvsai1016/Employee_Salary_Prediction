import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import pickle
import os

def Preprocess(input_data):
    df = input_data.copy()

    # Age mapping
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

    # Education mapping
    education_map = {
        'Primary/elementary school': 0,
        'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 1,
        'Some college/university study without earning a degree': 2,
        'Associate degree (A.A., A.S., etc.)': 3,
        'Bachelor’s degree (B.A., B.S., B.Eng., etc.)': 4,
        'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)': 5,
        'Professional degree (JD, MD, Ph.D, Ed.D, etc.)': 6,
        'Something else': 0
    }
    df['EdLevelEncoded'] = df['EdLevel'].map(education_map)
        # Drop unused object columns
    df.drop(columns=['Age', 'EdLevel'], errors='ignore', inplace=True)

    with open(os.path.join("model_files", "country_encoder.pkl"), "rb") as f:
        le_country = pickle.load(f)
    df['CountryEncoded'] = le_country.transform(df['Country'])
    df.drop(columns = ['Country'],inplace=True)

    # DevType One-hot
    df_dummies = pd.get_dummies(df['DevType'], prefix='DevType')
    df_dummies = df_dummies.astype(int)
    df = pd.concat([df.drop(columns=['DevType']), df_dummies], axis=1)

    # Employment multi-hot
    df['EmploymentList'] = df['Employment']
    mlb = MultiLabelBinarizer()
    employment_dummies = pd.DataFrame(
        mlb.fit_transform(df['EmploymentList']),
        columns=mlb.classes_,
        index=df.index
    )
    df = pd.concat([df.drop(columns=['Employment', 'EmploymentList']), employment_dummies], axis=1)

    # RemoteWork one-hot
    remote_dummies = pd.get_dummies(df['RemoteWork'], prefix='RemoteWork')
    df = pd.concat([df.drop(columns=['RemoteWork']), remote_dummies], axis=1)

    # Languages multi-hot
    mlb = MultiLabelBinarizer()
    lang_dummies = pd.DataFrame(mlb.fit_transform(df['LanguageHaveWorkedWith']), columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop(columns=['LanguageHaveWorkedWith']), lang_dummies], axis=1)

    # Database multi-hot
    mlb = MultiLabelBinarizer()
    db_dummies = pd.DataFrame(mlb.fit_transform(df['DatabaseHaveWorkedWith']), columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop(columns=['DatabaseHaveWorkedWith']), db_dummies], axis=1)

    # Platform multi-hot
    mlb = MultiLabelBinarizer()
    plat_dummies = pd.DataFrame(mlb.fit_transform(df['PlatformHaveWorkedWith']), columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop(columns=['PlatformHaveWorkedWith']), plat_dummies], axis=1)

    # Web Frameworks
    mlb = MultiLabelBinarizer()
    web_dummies = pd.DataFrame(mlb.fit_transform(df['WebframeHaveWorkedWith']), columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop(columns=['WebframeHaveWorkedWith']), web_dummies], axis=1)

    # Embedded
    mlb = MultiLabelBinarizer()
    emb_dummies = pd.DataFrame(mlb.fit_transform(df['EmbeddedHaveWorkedWith']), columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop(columns=['EmbeddedHaveWorkedWith']), emb_dummies], axis=1)

    # MiscFrameworks
    mlb = MultiLabelBinarizer()
    misc_dummies = pd.DataFrame(mlb.fit_transform(df['MiscFrameworks']), columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop(columns=['MiscFrameworks']), misc_dummies], axis=1)

    # Tools
    mlb = MultiLabelBinarizer()
    tools_dummies = pd.DataFrame(mlb.fit_transform(df['ToolsHaveWorkedWith']), columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop(columns=['ToolsHaveWorkedWith']), tools_dummies], axis=1)

    df['WorkExp'] = np.log1p(df['WorkExp'])

    with open(os.path.join("model_files","features.pkl"), "rb") as f:
        saved_features = pickle.load(f)

    # 2. Add any missing columns in the current input
    for col in saved_features:
        if col not in df.columns:
            df[col] = 0

    # 3. Drop any extra columns that are not in saved features
    df = df[saved_features]

    # 4. Return the aligned dataframe
    return df
