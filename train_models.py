import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers, models, callbacks

# =========================
# 1) تنظیمات کلی پروژه
# =========================
DATA_PATH = "Shiraz_RNN_Normalized.xlsx"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

WINDOW = 30
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

FEATURE_COLS = ["Tmean","Tmax","Tmin","RHmean","RHmax","RHmin","Rain"]
TARGET_COL = "Tmean_t+1"

# 🔹 لیست seed ها برای پایداری
SEEDS = [7, 21, 42, 99, 202]

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# =========================
# 2) خواندن داده
# =========================
df = pd.read_excel(DATA_PATH)

train_df = df[df["Split"] == "train"].reset_index(drop=True)
val_df   = df[df["Split"] == "val"].reset_index(drop=True)
test_df  = df[df["Split"] == "test"].reset_index(drop=True)

def make_sequences(dataframe, window, feature_cols, target_col):
    X = dataframe[feature_cols].values
    y = dataframe[target_col].values
    Xs, ys = [], []
    for i in range(window-1, len(dataframe)-1):
        Xs.append(X[i-window+1:i+1, :])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train, y_train = make_sequences(train_df, WINDOW, FEATURE_COLS, TARGET_COL)
X_val, y_val     = make_sequences(val_df, WINDOW, FEATURE_COLS, TARGET_COL)
X_test, y_test   = make_sequences(test_df, WINDOW, FEATURE_COLS, TARGET_COL)

# =========================
# 3) ساخت مدل‌ها
# =========================
def build_simple_rnn(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.SimpleRNN(64, activation="tanh"),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

def build_lstm(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

def build_gru(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

def build_bilstm(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

MODELS = {
    "SimpleRNN": build_simple_rnn,
    "LSTM": build_lstm,
    "GRU": build_gru,
    "BiLSTM": build_bilstm
}

# =========================
# 4) آموزش چندباره (Multi-Seed)
# =========================
all_runs = []

for seed in SEEDS:
    print(f"\n===== Running with seed {seed} =====")
    set_seed(seed)

    for name, build_fn in MODELS.items():
        print(f"Training {name}")

        model = build_fn((WINDOW, len(FEATURE_COLS)))
        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        model.compile(optimizer=opt, loss="mse")

        es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es, rl],
            verbose=0
        )

        y_pred = model.predict(X_test, verbose=0).reshape(-1)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        all_runs.append([seed, name, mae, rmse, r2])

        # فقط برای seed اول نمودار ذخیره شود
        if seed == SEEDS[0]:
            plt.figure()
            plt.plot(hist.history["loss"], label="train")
            plt.plot(hist.history["val_loss"], label="val")
            plt.legend()
            plt.title(f"{name} - Loss")
            plt.savefig(os.path.join(RESULT_DIR, f"{name}_loss.png"), dpi=200)
            plt.close()

# =========================
# 5) محاسبه پایداری
# =========================
runs_df = pd.DataFrame(all_runs, columns=["Seed","Model","MAE","RMSE","R2"])
runs_df.to_csv(os.path.join(RESULT_DIR, "all_runs.csv"), index=False)

summary = runs_df.groupby("Model").agg(
    MAE_mean=("MAE","mean"),
    MAE_std=("MAE","std"),
    RMSE_mean=("RMSE","mean"),
    RMSE_std=("RMSE","std"),
    R2_mean=("R2","mean"),
    R2_std=("R2","std"),
).reset_index()

summary.to_csv(os.path.join(RESULT_DIR, "stability_summary.csv"), index=False)

print("\n=== Stability Results ===")
print(summary)