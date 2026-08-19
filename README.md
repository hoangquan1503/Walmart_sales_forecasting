# Walmart Sales Forecasting

A time-series forecasting project for predicting Walmart weekly sales using exploratory data analysis, data preprocessing, feature engineering, and machine learning models.

## Overview

This project focuses on forecasting Walmart weekly sales using historical sales data and additional store, department, calendar, and economic features.

The project implements a complete forecasting workflow:

- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- Time-series forecasting
- Machine learning forecasting
- Model evaluation
- Model interpretation
- Forecast visualization

The main models explored in this project are:

- **Prophet**
- **LightGBM**
- **Optuna** for hyperparameter optimization

---

## Project Structure

```text
Walmart_sales_forecasting/
│
├── figures/
│   └── Generated figures and visualizations
│
├── notebook/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   ├── feature_engineering.ipynb
│   ├── model.ipynb
│   └── explain.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── metrics.py
│   └── plots.py
│
└── README.md

Dataset

The project is based on the Walmart Store Sales Forecasting dataset.

The main target variable is:

Weekly_Sales

which represents the weekly sales of a specific store and department.

The dataset contains variables related to:

Store
Department
Date
Weekly Sales
Store Size
Temperature
Fuel Price
CPI
Unemployment
Markdown variables
Holiday information

Additional time-series features are generated during the feature engineering stage.

The dataset itself is not included in this repository.

Workflow
Raw Walmart Dataset
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ├──────────────────┐
        │                  │
        ▼                  ▼
     Prophet           LightGBM
        │                  │
        └────────┬─────────┘
                 │
                 ▼
          Model Evaluation
                 │
                 ▼
       Visualization & Analysis
Notebooks
1. Exploratory Data Analysis

File:

notebook/EDA.ipynb

This notebook is used to understand the Walmart sales dataset and identify important patterns.

The analysis includes:

Dataset structure
Data distributions
Missing values
Sales trends
Store-level analysis
Department-level analysis
Holiday effects
Time-series behavior
Relationships between sales and external variables
2. Data Preprocessing

File:

notebook/preprocessing.ipynb

This notebook prepares the raw dataset for downstream modeling.

Main tasks include:

Data cleaning
Date conversion
Missing-value handling
Data type conversion
Dataset preparation
Train/validation data preparation
3. Feature Engineering

File:

notebook/feature_engineering.ipynb

This notebook generates additional features from historical sales data.

The project uses time-series features such as:

Lag features
Rolling statistics
Historical mean sales
Historical maximum sales
Historical minimum sales
Historical standard deviation
Exponential moving statistics
Store-level statistics
Calendar/time-based features

These features are particularly useful for tree-based forecasting models such as LightGBM.

4. Model Training

File:

notebook/model.ipynb

This notebook contains the main model training and evaluation pipeline.

The project currently experiments with:

Prophet
LightGBM
Optuna
Scikit-learn

The general workflow is:

Prepare Data
     │
     ▼
Train Model
     │
     ▼
Generate Predictions
     │
     ▼
Evaluate Predictions
     │
     ▼
Compare Models
5. Model Explanation

File:

notebook/explain.ipynb

This notebook is used to analyze and interpret the forecasting results.

It can be used for:

Feature importance
Model interpretation
Prediction analysis
Error analysis
Forecast visualization
Models
Prophet

Prophet is used as a dedicated time-series forecasting model.

It is designed to model time-dependent patterns such as:

Trend
Seasonality
Holidays
Temporal effects
Additional regressors

In this project, Prophet is used to forecast Walmart sales for individual time-series groups.

LightGBM

LightGBM is used as a gradient-boosting model for tabular forecasting.

The model can utilize engineered features such as:

Lag values
Rolling statistics
Calendar features
Store information
Department information
Historical sales statistics
External variables

LightGBM is particularly useful when a large number of engineered forecasting features are available.

Optuna

Optuna is used for hyperparameter optimization.

It can automatically search for better model configurations instead of relying entirely on manually selected hyperparameters.

Evaluation Metrics

The project contains reusable evaluation functions in:

src/metrics.py

The main metrics include:

MAE

Mean Absolute Error:

MAE = mean(|y_true - y_pred|)

MAE measures the average absolute difference between actual and predicted sales.

RMSE

Root Mean Squared Error:

RMSE = sqrt(mean((y_true - y_pred)^2))

RMSE gives a larger penalty to large prediction errors.

WAPE

Weighted Absolute Percentage Error:

WAPE = sum(|y_true - y_pred|) / sum(|y_true|)

WAPE is useful for evaluating forecasting performance across sales series with different scales.

Installation

Install the required Python packages:

pip install numpy pandas matplotlib seaborn scikit-learn
pip install prophet cmdstanpy lightgbm optuna
pip install joblib pyarrow shap

For reproducible experiments, package versions should ideally be pinned in a requirements.txt file.

Running the Project

Clone the repository:

git clone https://github.com/hoangquan1503/Walmart_sales_forecasting.git

Enter the project directory:

cd Walmart_sales_forecasting

Run the notebooks in the following order:

1. EDA.ipynb
2. preprocessing.ipynb
3. feature_engineering.ipynb
4. model.ipynb
5. explain.ipynb

Depending on the environment, the dataset path may need to be modified.

Performance Considerations

One of the main computational challenges of this project is Prophet training.

If Prophet is trained independently for a large number of store/department combinations and every model is trained sequentially, the total training time can become very large.

For example:

Store 1 / Department 1
        ↓
Train Prophet
        ↓
Store 1 / Department 2
        ↓
Train Prophet
        ↓
Store 1 / Department 3
        ↓
Train Prophet
        ↓
...

A more scalable architecture is to train independent time-series models in parallel:

                 ┌── Store 1 / Dept 1
                 │
                 ├── Store 1 / Dept 2
                 │
Input Data ──────┼── Store 2 / Dept 1
                 │
                 ├── Store 2 / Dept 2
                 │
                 └── Store N / Dept N
                           │
                           ▼
                  Combine Predictions

Potential optimization techniques include:

Parallel Prophet training
Multiprocessing
Joblib-based parallelization
Model caching
Reusing preprocessed data
Avoiding unnecessary retraining
Separating preprocessing from model training
Using LightGBM for large-scale tabular forecasting
Reproducibility

The notebooks currently install and check dependencies inside the notebook environment.

For a more reproducible project, it is recommended to create a:

requirements.txt

file containing pinned package versions.

For example:

numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
lightgbm==4.3.0
optuna==3.6.1
joblib==1.4.2

The exact Prophet and CmdStan versions should be pinned according to the environment used for the final experiment.

Prophet uses CmdStan through cmdstanpy. The first execution may take additional time because the Stan backend may need to be initialized or compiled.

Future Improvements

Potential future improvements include:

 Add requirements.txt
 Add .gitignore
 Move model training from notebooks into Python modules
 Implement parallel Prophet training
 Add model caching
 Add experiment tracking
 Add automated evaluation
 Add reproducible training scripts
 Add CI tests
 Add Streamlit deployment
 Compare Prophet and LightGBM systematically
 Add ensemble forecasting
 Optimize memory usage for large-scale training
Project Goals

The main goals of this project are:

Understand Walmart sales patterns.
Build a complete sales forecasting pipeline.
Compare statistical and machine-learning forecasting approaches.
Engineer meaningful time-series features.
Evaluate models using multiple forecasting metrics.
Improve training efficiency for large numbers of forecasting series.
Provide an extensible structure for future experiments.
Author

Hoang Quan

GitHub:

https://github.com/hoangquan1503

Repository:

https://github.com/hoangquan1503/Walmart_sales_forecasting