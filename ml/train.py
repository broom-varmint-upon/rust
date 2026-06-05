import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss

CSV_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

WINDOWS = [1, 5, 10, 30, 60, 300]
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TARGET_FORWARD = 1


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=CSV_COLS, dtype=np.float64)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="us")
    df = df.sort_values("open_time").reset_index(drop=True)
    df.drop(columns=["close_time", "ignore"], inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    returns = df["close"].pct_change()

    for w in WINDOWS:
        df[f"ret_{w}s"] = returns.rolling(w).sum()
        df[f"vol_{w}s"] = returns.rolling(w).std()
        df[f"vwap_{w}s"] = df["quote_volume"].rolling(w).sum() / df["volume"].rolling(w).sum()
        df[f"vol_mean_{w}s"] = df["volume"].rolling(w).mean()
        df[f"trades_{w}s"] = df["trade_count"].rolling(w).sum()

    df["taker_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["open_close_range"] = (df["close"] - df["open"]) / df["open"]
    df["close_to_vwap"] = df["close"] / (df["quote_volume"] / df["volume"].replace(0, np.nan))

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    future_close = df["close"].shift(-TARGET_FORWARD)
    df["target"] = (future_close > df["close"]).astype(np.int8)
    return df


def build_dataset(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("open_time", "close", "target", "open", "high", "low", "volume", "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote")]
    df = df.dropna(subset=feature_cols + ["target"])

    X = df[feature_cols].values
    y = df["target"].values
    n = len(X)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:],
        "feature_cols": feature_cols,
    }


def train(ds: dict):
    dtrain = lgb.Dataset(ds["X_train"], label=ds["y_train"])
    dval = lgb.Dataset(ds["X_val"], label=ds["y_val"], reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(50)],
    )

    return model


def evaluate(model, ds: dict):
    for split in ["val", "test"]:
        X = ds[f"X_{split}"]
        y = ds[f"y_{split}"]
        preds = model.predict(X)
        labels = (preds > 0.5).astype(int)
        acc = accuracy_score(y, labels)
        ll = log_loss(y, preds)
        print(f"\n--- {split.upper()} ---")
        print(f"accuracy:  {acc:.4f}")
        print(f"log_loss:  {ll:.4f}")
        print(classification_report(y, labels, target_names=["down", "up"]))


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT-1s-2026-05.csv"
    if not Path(csv_path).exists():
        print(f"error: {csv_path} not found")
        sys.exit(1)

    print(f"loading {csv_path}...")
    df = load(csv_path)
    print(f"  {len(df):,} rows")

    print("engineering features...")
    df = add_features(df)

    print("building dataset...")
    ds = add_target(df)
    ds = build_dataset(ds)
    print(f"  features: {len(ds['feature_cols'])}")
    print(f"  train: {len(ds['X_train']):,}  val: {len(ds['X_val']):,}  test: {len(ds['X_test']):,}")

    print("training...")
    model = train(ds)

    evaluate(model, ds)


if __name__ == "__main__":
    main()
