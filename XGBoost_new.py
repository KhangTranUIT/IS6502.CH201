import os
import glob

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')   # chạy headless trên Colab + pyenv

import matplotlib.pyplot as plt

from utils import *
from model import walk_forward_validation

# =========================================================
# CREATE OUTPUT DIR
# =========================================================
os.makedirs("plots_xgboost", exist_ok=True)

# =========================================================
# HELPER FUNCTION
# =========================================================
plot_counter = 0

def save_plot():
    global plot_counter

    filename = f"plots_xgboost/figure_{plot_counter}.png"

    plt.savefig(filename, bbox_inches='tight')

    print(f"Saved: {filename}")

    plt.close()

    plot_counter += 1

# =========================================================
# LOAD DATA
# =========================================================
data = pd.read_csv('601988.SH.csv')

data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')

data = data.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]

close = data.pop('close')

data.insert(5, 'close', close)

data1 = data.iloc[3501:, 5]

# =========================================================
# LOAD RESIDUALS
# =========================================================
residuals = pd.read_csv('ARIMA_residuals1.csv')

residuals.index = pd.to_datetime(residuals['trade_date'])

residuals.pop('trade_date')

merge_data = pd.merge(data, residuals, on='trade_date')

time = pd.Series(data.index[3501:])

# =========================================================
# LOAD ARIMA PREDICTION
# =========================================================
Lt = pd.read_csv('ARIMA.csv')

Lt = Lt.drop('trade_date', axis=1)

Lt = np.array(Lt)

Lt = Lt.flatten().tolist()

# =========================================================
# PREPARE DATA
# =========================================================
train, test = prepare_data(
    merge_data,
    n_test=180,
    n_in=6,
    n_out=1
)

# =========================================================
# WALK FORWARD VALIDATION
# =========================================================
y, yhat = walk_forward_validation(train, test)

# =========================================================
# RESIDUALS PREDICTION PLOT
# =========================================================
plt.figure(figsize=(10, 6))

plt.plot(time, y, label='Residuals')
plt.plot(time, yhat, label='Predicted Residuals')

plt.title('ARIMA+XGBoost: Residuals Prediction')

plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Residuals', fontsize=14, horizontalalignment='center')

plt.legend()

save_plot()

# =========================================================
# FINAL STOCK PRICE PREDICTION
# =========================================================
finalpredicted_stock_price = [
    i + j for i, j in zip(Lt, yhat)
]

evaluation_metric(data1, finalpredicted_stock_price)

# =========================================================
# FINAL PRICE PLOT
# =========================================================
plt.figure(figsize=(10, 6))

plt.plot(time, data1, label='Stock Price')
plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')

plt.title('ARIMA+XGBoost: Stock Price Prediction')

plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')

plt.legend()

save_plot()

print("\nALL FIGURES SAVED SUCCESSFULLY")
print("OUTPUT DIR: plots_xgboost/")


# =========================================================
# SAVE PREDICTION RESULTS
# =========================================================
os.makedirs("prediction_results_xgboost", exist_ok=True)

# ---------------------------------------------------------
# FINAL STOCK PRICE PREDICTION
# ---------------------------------------------------------
prediction_df = pd.DataFrame({
    'trade_date': time.values,
    'real_close': data1.values,
    'predicted_close': finalpredicted_stock_price
})

prediction_df.to_csv(
    "prediction_results_xgboost/xgboost_prediction.csv",
    index=False
)

print("\nSaved:")
print("prediction_results_xgboost/xgboost_prediction.csv")

print("\nPrediction Preview:")
print(prediction_df.head())

# ---------------------------------------------------------
# RESIDUAL PREDICTION
# ---------------------------------------------------------
residual_df = pd.DataFrame({
    'trade_date': time.values,
    'real_residual': y,
    'predicted_residual': yhat
})

residual_df.to_csv(
    "prediction_results_xgboost/xgboost_residual_prediction.csv",
    index=False
)

print("\nSaved:")
print("prediction_results_xgboost/xgboost_residual_prediction.csv")

print("\nResidual Prediction Preview:")
print(residual_df.head())

print("\nALL PREDICTION FILES SAVED SUCCESSFULLY")