import numpy as np

def weighted_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
   
    if np.sum(np.abs(y_true)) == 0:
        return 0.0
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))