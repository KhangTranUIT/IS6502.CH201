import os
import glob

import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Headless backend cho Colab + pyenv

import matplotlib.pyplot as plt
from IPython.display import Image, display

from datetime import datetime
import numpy as np
from pandas.plotting import autocorrelation_plot
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import metrics

from utils import *

# =========================================================
# CREATE OUTPUT DIR
# =========================================================
os.makedirs("plots", exist_ok=True)

# =========================================================
# HELPER FUNCTION
# =========================================================
plot_counter = 0

def save_and_show():
    global plot_counter

    filename = f"plots/figure_{plot_counter}.png"

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filename}")
    display(Image(filename))

    plot_counter += 1

# =========================================================
# LOAD DATA, đổi tên file csv mã cổ phiếu ở đây
# =========================================================
### *** ###
data = pd.read_csv('HPG.csv')

# =========================================================
# CONFIG
# =========================================================

TOTAL_DAYS = 3000        # số ngày muốn sử dụng
TRAIN_RATIO = 3480 / 3661   



data = data.tail(TOTAL_DAYS).reset_index(drop=True)

test_start = int(TOTAL_DAYS * TRAIN_RATIO)

test_set2 = data.iloc[test_start:, :]


data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')

data = data.drop(['ts_code', 'trade_date'], axis=1)

data = pd.DataFrame(data, dtype=np.float64)

#training_set = data.loc['2007-01-04':'2021-06-21', :]
#test_set = data.loc['2021-06-22':, :]

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

train_size = int(len(data) * TRAIN_RATIO)

training_set = data.iloc[:train_size].copy()

test_set = data.iloc[train_size:].copy()

print("="*60)
print("Total samples :", len(data))
print("Train samples :", len(training_set))
print("Test samples  :", len(test_set))
print("="*60)


# =========================================================
# CLOSE PRICE
# =========================================================
plt.figure(figsize=(10, 6))

plt.plot(training_set['close'], label='training_set')
plt.plot(test_set['close'], label='test_set')

plt.title('Close price')

plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('close', fontsize=14, horizontalalignment='center')

plt.legend()

save_and_show()

# =========================================================
# FIRST ORDER DIFF
# =========================================================
temp = np.array(training_set['close'])

training_set.loc[:, 'diff_1'] = training_set['close'].diff(1)

plt.figure(figsize=(10, 6))

training_set['diff_1'].plot()

plt.title('First-order diff')

plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')

save_and_show()

# =========================================================
# SECOND ORDER DIFF
# =========================================================
training_set.loc[:, 'diff_2'] = training_set['diff_1'].diff(1)

plt.figure(figsize=(10, 6))

training_set['diff_2'].plot()

plt.title('Second-order diff')

plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_2', fontsize=14, horizontalalignment='center')

save_and_show()

# =========================================================
# WHITE NOISE TEST
# =========================================================
temp1 = np.diff(training_set['close'], n=1)

training_data1 = training_set['close'].diff(1)

temp2 = np.diff(training_set['close'], n=1)

print(acorr_ljungbox(temp2, lags=2, boxpierce=True))

# =========================================================
# ACF PACF
# =========================================================
acf_pacf_plot(training_set['close'], acf_lags=160)

price = list(temp2)

data2 = {
    'trade_date': training_set['diff_1'].index[1:],
    'close': price
}

df = pd.DataFrame(data2)

df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

training_data_diff = df.set_index(['trade_date'], drop=True)

print('&', training_data_diff)

acf_pacf_plot(training_data_diff)

# =========================================================
# ARIMA MODEL
# =========================================================
model = sm.tsa.ARIMA(endog=training_set['close'], order=(2, 1, 0)).fit()

history = [x for x in training_set['close']]

predictions = list()

for t in range(test_set.shape[0]):

    model1 = sm.tsa.ARIMA(history, order=(2, 1, 0))

    model_fit = model1.fit()

    yhat = model_fit.forecast()

    yhat = np.float(yhat[0])

    predictions.append(yhat)

    obs = test_set['close'].iloc[t]

    history.append(obs)

# =========================================================
# PREDICTION RESULT
# =========================================================
predictions1 = {
    'trade_date': test_set.index[:],
    'close': predictions
}

predictions1 = pd.DataFrame(predictions1)

predictions1 = predictions1.set_index(['trade_date'], drop=True)

predictions1.to_csv('ARIMA.csv')

plt.figure(figsize=(10, 6))

plt.plot(test_set['close'], label='Stock Price')
plt.plot(predictions1, label='Predicted Stock Price')

plt.title('ARIMA: Stock Price Prediction')

plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')

plt.legend()

save_and_show()

# =========================================================
# RESIDUALS
# =========================================================
model2 = sm.tsa.ARIMA(endog=data['close'], order=(2, 1, 0)).fit()

residuals = pd.DataFrame(model2.resid)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])

save_and_show()

residuals.to_csv('ARIMA_residuals1.csv')

# =========================================================
# EVALUATION
# =========================================================
evaluation_metric(test_set['close'], predictions)

adf_test(temp)

adf_test(temp1)

# =========================================================
# DIFF FIT
# =========================================================
predictions_ARIMA_diff = pd.Series(model.fittedvalues, copy=True)

#predictions_ARIMA_diff = predictions_ARIMA_diff[3479:]
predictions_ARIMA_diff = predictions_ARIMA_diff[-len(training_data_diff):]


print('#', predictions_ARIMA_diff)

plt.figure(figsize=(10, 6))

plt.plot(training_data_diff, label="diff_1")
plt.plot(predictions_ARIMA_diff, label="prediction_diff_1")

plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')

plt.title('DiffFit')

plt.legend()

save_and_show()

print("\nALL FIGURES SAVED SUCCESSFULLY")