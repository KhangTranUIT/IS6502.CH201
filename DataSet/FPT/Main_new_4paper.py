import os
import glob

from keras.optimizers import Adam

import matplotlib
matplotlib.use('Agg')   # headless backend cho Colab + pyenv

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn import metrics

from utils import *
from model import *

# =========================================================
# CREATE OUTPUT DIR
# =========================================================
os.makedirs("plots_lstm", exist_ok=True)

# =========================================================
# HELPER FUNCTION
# =========================================================
plot_counter = 0

def save_plot():
    global plot_counter

    filename = f"plots_lstm/figure_{plot_counter}.png"

    plt.savefig(filename, bbox_inches='tight')

    print(f"Saved: {filename}")

    plt.close()

    plot_counter += 1

# =========================================================
# LOAD DATA
# =========================================================
data1 = pd.read_csv("FPT.csv")

data1.index = pd.to_datetime(
    data1['trade_date'],
    format='%Y%m%d'
)

data1 = data1.loc[
    :,
    ['open', 'high', 'low', 'close', 'vol', 'amount']
]

data_yuan = data1

# =========================================================
# LOAD RESIDUALS
# =========================================================
residuals = pd.read_csv('ARIMA_residuals1.csv')

residuals.index = pd.to_datetime(
    residuals['trade_date']
)

residuals.pop('trade_date')

data1 = pd.merge(
    data1,
    residuals,
    on='trade_date'
)

# =========================================================
# SPLIT DATA
# =========================================================
#data = data1.iloc[1:3500, :]

#data2 = data1.iloc[3500:, :]
# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

TRAIN_RATIO = 3480 / 3661     # giống paper

train_end = int(len(data1) * TRAIN_RATIO)
data = data1.iloc[1:train_end, :]

data2 = data1.iloc[train_end:, :]


TIME_STEPS = 20

# =========================================================
# NORMALIZE
# =========================================================
data, normalize = NormalizeMult(data)

print('#', normalize)

pollution_data = data[:, 3].reshape(len(data), 1)

# =========================================================
# CREATE DATASET
# =========================================================
train_X, _ = create_dataset(
    data,
    TIME_STEPS
)

_, train_Y = create_dataset(
    pollution_data,
    TIME_STEPS
)

print(train_X.shape, train_Y.shape)

# =========================================================
# BUILD MODEL
# =========================================================
m = attention_model(INPUT_DIMS=7)

m.summary()

adam = Adam(learning_rate=0.01)

m.compile(
    optimizer=adam,
    loss='mse'
)

# =========================================================
# TRAIN MODEL
# =========================================================
history = m.fit(
    [train_X],
    train_Y,
    epochs=50,
    batch_size=32,
    validation_split=0.1
)

# =========================================================
# SAVE MODEL
# =========================================================
m.save("./stock_model.h5")

np.save("stock_normalize.npy", normalize)

# =========================================================
# TRAINING LOSS PLOT
# =========================================================
plt.figure(figsize=(10, 6))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Training and Validation Loss')

plt.legend()

save_plot()

# =========================================================
# CONFIG
# =========================================================
class Config:
    def __init__(self):
        self.dimname = 'close'

config = Config()

name = config.dimname

# =========================================================
# PREDICTION
# =========================================================
y_hat, y_test = PredictWithData(
    data2,
    data_yuan,
    name,
    'stock_model.h5',
    7
)

y_hat = np.array(
    y_hat,
    dtype='float64'
)

y_test = np.array(
    y_test,
    dtype='float64'
)

# =========================================================
# EVALUATION
# =========================================================
evaluation_metric(
    y_test,
    y_hat
)

print("len(y_test) =", len(y_test))
print("len(y_hat)  =", len(y_hat))

# =========================================================
# TIME INDEX
# =========================================================
time = pd.Series(
    data1.index[-len(y_test):]
)

# =========================================================
# FINAL PREDICTION PLOT
# =========================================================
plt.figure(figsize=(10, 6))

plt.plot(
    time,
    y_test,
    label='True'
)

plt.plot(
    time,
    y_hat,
    label='Prediction'
)

plt.title('Hybrid model prediction')

plt.xlabel(
    'Time',
    fontsize=12,
    verticalalignment='top'
)

plt.ylabel(
    'Price',
    fontsize=14,
    horizontalalignment='center'
)

plt.legend()

save_plot()

print("\nALL FIGURES SAVED SUCCESSFULLY")
print("OUTPUT DIR: plots_lstm/")


# =========================================================
# SAVE PREDICTION RESULTS
# =========================================================
os.makedirs("prediction_results_lstm", exist_ok=True)

# ---------------------------------------------------------
# MAIN PREDICTION RESULT
# ---------------------------------------------------------
prediction_df = pd.DataFrame({
    'trade_date': time.values,
    'real_close': y_test.flatten(),
    'predicted_close': y_hat.flatten()
})

prediction_df.to_csv(
    "prediction_results_lstm/lstm_prediction.csv",
    index=False
)

print("\nSaved:")
print("prediction_results_lstm/lstm_prediction.csv")

print("\nPrediction Preview:")
print(prediction_df.head())

# ---------------------------------------------------------
# SAVE TRAINING HISTORY
# ---------------------------------------------------------
history_df = pd.DataFrame({
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})

history_df.to_csv(
    "prediction_results_lstm/lstm_training_history.csv",
    index=False
)

print("\nSaved:")
print("prediction_results_lstm/lstm_training_history.csv")

print("\nTraining History Preview:")
print(history_df.head())

print("\nALL PREDICTION FILES SAVED SUCCESSFULLY")