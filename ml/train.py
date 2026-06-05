import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss

# ── Config ──────────────────────────────────────────────────────────────────

MONTH = "2026-05"
SYMBOL = "BTCUSDT"
BASE_INTERVAL = "1s"
HIGHER_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]
DATA_DIR = Path("data")

URL = (
    "https://data.binance.vision/data/spot/monthly/klines"
    f"/{SYMBOL}/{{interval}}/{SYMBOL}-{{interval}}-{MONTH}.zip"
)

CSV_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

WINDOWS = [1, 5, 10, 30, 60, 300]
TF_WINDOWS = [3, 6, 12]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TARGET_FORWARD = 1

EXCLUDE_COLS = {
    "open_time", "close", "target", "open", "high", "low",
    "volume", "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote",
}

# ── Data ────────────────────────────────────────────────────────────────────

def ensure_data(interval: str) -> Path:
    csv_path = DATA_DIR / f"{SYMBOL}-{interval}-{MONTH}.csv"
    if csv_path.exists():
        return csv_path

    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / f"{SYMBOL}-{interval}-{MONTH}.zip"
    print(f"  downloading {interval}...")
    urlretrieve(URL.format(interval=interval), zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()
    return csv_path


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=CSV_COLS, dtype=np.float64)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="us")
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["close_time", "ignore"], inplace=True)
    return df

# ── Features ────────────────────────────────────────────────────────────────

def base_features(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["close"].pct_change()

    for w in WINDOWS:
        df[f"ret_{w}s"] = ret.rolling(w).sum()
        df[f"vol_{w}s"] = ret.rolling(w).std()
        df[f"vwap_{w}s"] = (
            df["quote_volume"].rolling(w).sum()
            / df["volume"].rolling(w).sum()
        )
        df[f"vm_{w}s"] = df["volume"].rolling(w).mean()
        df[f"tc_{w}s"] = df["trade_count"].rolling(w).sum()

    df["taker_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]
    df["close_to_vwap"] = df["close"] / (
        df["quote_volume"] / df["volume"].replace(0, np.nan)
    )
    return df


def higher_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    h = df[["open_time", "open", "high", "low", "close", "volume",
            "quote_volume", "trade_count", "taker_buy_base"]].copy()

    ret = h["close"].pct_change()

    for w in TF_WINDOWS:
        h[f"{prefix}_ret_{w}"] = ret.rolling(w).sum()
        h[f"{prefix}_vol_{w}"] = ret.rolling(w).std()

    h[f"{prefix}_vwap"] = (
        h["quote_volume"].rolling(6).sum() / h["volume"].rolling(6).sum()
    )
    h[f"{prefix}_taker"] = h["taker_buy_base"] / h["volume"].replace(0, np.nan)
    h[f"{prefix}_hl"] = (h["high"] - h["low"]) / h["close"]
    h[f"{prefix}_vm"] = h["volume"].rolling(6).mean()
    h[f"{prefix}_tc"] = h["trade_count"].rolling(6).sum()

    feat_cols = [c for c in h.columns if c.startswith(prefix)]
    h = h[["open_time"] + feat_cols].shift(1)
    h["open_time"] = df["open_time"].values

    return h

# ── Dataset ─────────────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    target = (df["close"].shift(-TARGET_FORWARD) > df["close"]).astype(np.int8)
    df = pd.concat([df, target.rename("target")], axis=1)
    df = df.dropna(subset=["target"])

    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values
    n = len(X)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    return {
        "X_train": X[:train_end], "y_train": y[:train_end],
        "X_val": X[train_end:val_end], "y_val": y[train_end:val_end],
        "X_test": X[val_end:], "y_test": y[val_end:],
        "feature_cols": feature_cols,
    }

# ── Train ───────────────────────────────────────────────────────────────────

def train_model(ds: dict):
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

    return lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(50)],
    )


def evaluate(model, ds: dict):
    for split in ["val", "test"]:
        X, y = ds[f"X_{split}"], ds[f"y_{split}"]
        preds = model.predict(X)
        labels = (preds > 0.5).astype(int)
        print(f"\n--- {split.upper()} ---")
        print(f"accuracy:  {accuracy_score(y, labels):.4f}")
        print(f"log_loss:  {log_loss(y, preds):.4f}")
        print(classification_report(y, labels, target_names=["down", "up"]))

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("downloading data...")
    base_path = ensure_data(BASE_INTERVAL)

    higher = {}
    for iv in HIGHER_INTERVALS:
        try:
            higher[iv] = ensure_data(iv)
        except Exception as e:
            print(f"  skipping {iv}: {e}")

    print(f"loading {base_path.name}...")
    base = load(base_path)
    print(f"  {len(base):,} rows")

    print("engineering base features...")
    base = base_features(base)

    for iv, path in higher.items():
        prefix = iv.replace("/", "")
        print(f"merging {iv} features...")
        h = load(path)
        hf = higher_features(h, prefix)
        base = pd.merge_asof(base, hf, on="open_time", direction="backward")

    print("building dataset...")
    ds = build_dataset(base)
    print(f"  features: {len(ds['feature_cols'])}")
    print(f"  train: {len(ds['X_train']):,}  val: {len(ds['X_val']):,}  test: {len(ds['X_test']):,}")

    print("training...")
    model = train_model(ds)
    evaluate(model, ds)


if __name__ == "__main__":
    main()
