import pandas as pd

def check_missing_value(df):
    df_na = pd.DataFrame({
        "counts": df.isna().sum(),
        "ratio": df.isna().sum() / df.shape[0]
    })
    return df_na
def fill_missing_promote(cols):
    df_fill = df.copy()
    df_fill = df_fill.fillna(0)
    return df_fill
