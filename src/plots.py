import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_sales(df, store_id=1, item_id=1):
    df_plot = df.copy()
    df_plot = df_plot.query("((store_id=@store_id)&(item_id=@item_id))")
    fig, ax = plt.subplot(figsize=(8,4))
    df_plot[["Date", "Weekly_Sales"]].plot(x="Date",y="Weekly_Sales", ax=ax, legend=True)
    nan_indice = df[df["Weekly_Sales"].isna()].index
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
    ax.set_title("Store ID: {store_id} - Item ID: {item_id}")
    ax.legend()
    plt.show()
