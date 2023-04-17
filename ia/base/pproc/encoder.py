from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def lencod(df):

    for col in df.select_dtypes('object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


def lecond_2(df_original, df_dados_manual):  # Fit original e Transform na atual

    for coluna in df_original.select_dtypes('object'):
        le = LabelEncoder()
        le.fit(df_original[coluna])
        df_dados_manual[coluna] = le.transform(df_dados_manual[coluna])
    return df_dados_manual


def lecond_2_rev(df_original, df_dados_manual):  # Reversivo Fit original e Transform na atual

    for coluna in df_original.select_dtypes('object'):
        le = LabelEncoder()
        le.fit(df_original[coluna])
        df_dados_manual[coluna] = le.inverse_transform(df_dados_manual[coluna])
    return df_dados_manual


def lencond_3(db):  # Tabelar as codificações

    for col in db.select_dtypes('object'):
        a = pd.DataFrame(db[col].value_counts().sort_index().reset_index())

        le = LabelEncoder()
        db[col] = pd.DataFrame(le.fit_transform(db[col]))

        b = db[col].value_counts().sort_index().reset_index(name=col)

        merge = pd.merge(a, b, on=col)
        dff = merge.drop(col, axis=1)

    return dff


def onehot(df):

    for col in df_prev.select_dtypes(df):
        one_hot_encoded_data = pd.get_dummies(df, df[col])
        return one_hot_encoded_data
