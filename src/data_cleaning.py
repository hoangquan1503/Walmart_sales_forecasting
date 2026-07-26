import pandas as pd

def check_missing_value(df):
    df_na = pd.DataFrame({
        "counts": df.isna().sum(),
        "ratio": df.isna().sum() / df.shape[0]
    })
    return df_na

def fill_missing_promote(df, cols):
    df_fill = df.copy()
    df_fill[cols] = df_fill[cols].fillna(0)
    return df_fill

def corrected_outlier(df, factor=3):
    df_corrected = df.copy()
    z_score = (df_corrected['Weekly_Sales'] - df_corrected['Weekly_Sales'].mean()) / df_corrected['Weekly_Sales'].std()
    outlier = z_score > factor
    df_corrected.loc[outlier, 'Weekly_Sales'] = df_corrected.median()
    return df_corrected