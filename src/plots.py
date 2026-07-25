import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_sales(df, store_id=1, dept_id=1):
    df_plot = df.copy()
    df_plot = df_plot[(df_plot['Store'] == store_id) & (df_plot['Dept'] == dept_id)]
    fig, ax = plt.subplots(figsize=(8,4))
    df_plot[["Date", "Weekly_Sales"]].plot(x="Date",y="Weekly_Sales", ax=ax, legend=True)
    nan_indice = df_plot[df_plot["Weekly_Sales"].isna()].index
    if len(nan_indice) >=1 :
        df_plot["Weekly_Sales"] = df_plot["Weekly_Sales"].fillna(method="ffill")
        nan_date = df_plot.loc[nan_indice, "Date"]
        nan_sale = df_plot.loc[nan_indice, "Weekly_Sales"]
        for date, sale in zip(nan_date, nan_sale):
            ax.annotate(
                "-",
                xy=(date, sale),
                color="red",  # Set text color to red
                size=20,
            )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Store ID: {store_id} - Dept ID: {dept_id}")
    ax.legend()
    plt.show()
