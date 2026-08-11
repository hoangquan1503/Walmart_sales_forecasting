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
                color="red",  
                size=20,
            )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title(f"Store ID: {store_id} - Dept ID: {dept_id}")
    ax.legend()
    plt.show()


def visualize_pred(pred_dict):
    final_df = []
    for store_dept, pred in pred_dict.items():
        pred = pred.copy()
        pred['store_dept'] = store_dept
        final_df.append(pred)
    return pd.concat(final_df, ignore_index=True)


def plot_prophet_comparison(pred, store_dept):
    df = pred[pred['store_dept'].str.strip() == store_dept]
    if df.empty:
        print('No data to plot')
        return
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df, x="ds", y="y", label="Actual", color="black")
    sns.lineplot(data=df, x="ds", y="yhat", label="Forecast", color="blue")
    plt.fill_between(df['ds'], df['yhat_lower'], df['yhat_upper'], color='blue', alpha=0.2, label='Confidence Interval')
    plt.title(f'{store_dept} Prediction vs True Value')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_sales_prediction(df_pred, store=1, nrows=6, ncols=5, figsize=(20,20)):
    """
    Plots actual vs predicted sales for items in a given store.

    Parameters:
        df_prediction (DataFrame): Must include ['Store', 'Dept', 'Date', 'Weekly_Sales', 'pred']
        store (int): Store to filter on
        nrows (int): Rows of subplots
        ncols (int): Columns of subplots
        figsize (tuple): Size of the full figure
    """
    
    sample_store = df_pred[df_pred['Store'] == store]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    dept_list = df_pred['Dept'].unique()
    
    for idx, ax in enumerate(axes.flatten()):
        if idx >= len(dept_list): #out of range, skip
            ax.axis('off')
            continue
        
        dept = dept_list[idx]
        sample_dept = sample_store[sample_store['Dept'] == dept]
        
        # no data, skip
        if sample_dept.empty:
            ax.axis('off')
            continue
        
        ax.plot(sample_dept['Date'], sample_dept['Weekly_Sales'], label='True', color='blue')
        ax.plot(sample_dept['Date'], sample_dept['pred'], label='Prediction', color='red', linestyle='--', marker='.')
        
        ax.set_title(f'True vs Predict - store {store} dept {dept}')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the legend
    fig.suptitle(
        f"Sales Forecast vs Actual - Store {store}", fontsize=16, fontweight="bold"
    )
    plt.show()
        
    
