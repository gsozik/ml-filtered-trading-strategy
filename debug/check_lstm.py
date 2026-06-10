import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ta import TechnicalAnalysisPipeline
from ml import LSTMDirectionFilter
from ml.dataset import DirectionTargetBuilder


DATA_PATH = PROJECT_ROOT / "storage" / "BTC_USDT_4h_2024_2025.csv"
SAVE_DIR = PROJECT_ROOT / "storage" / "metrics"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SIZE = 1000

HORIZON = 10
LONG_THRESHOLD = 0.05
SHORT_THRESHOLD = -0.05

SEQUENCE_LENGTH = 50
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
USE_RAW_OHLCV = False

# 1. Load data
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.set_index("timestamp").sort_index()

df = TechnicalAnalysisPipeline().transform(df)

train_df = df.iloc[:TRAIN_SIZE].copy()
test_df = df.iloc[TRAIN_SIZE:].copy()

print("\nDATA")
print("=" * 80)
print(f"Full rows:  {len(df)}")
print(f"Train rows: {len(train_df)}")
print(f"Test rows:  {len(test_df)}")
print(f"Train: {train_df.index.min()} -> {train_df.index.max()}")
print(f"Test:  {test_df.index.min()} -> {test_df.index.max()}")


# 2. Build true target
target_builder = DirectionTargetBuilder(
    horizon=HORIZON,
    long_threshold=LONG_THRESHOLD,
    short_threshold=SHORT_THRESHOLD,
)

y_train_true = target_builder.build(train_df).dropna().astype(int)
y_test_true = target_builder.build(test_df).dropna().astype(int)

print("\nTARGET DISTRIBUTION")
print("=" * 80)
print("Train:")
print(y_train_true.value_counts().sort_index())
print("\nTest:")
print(y_test_true.value_counts().sort_index())


# 3. Train LSTM
lstm = LSTMDirectionFilter(
    horizon=HORIZON,
    long_threshold=LONG_THRESHOLD,
    short_threshold=SHORT_THRESHOLD,
    sequence_length=SEQUENCE_LENGTH,
    use_raw_ohlcv=USE_RAW_OHLCV,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=1e-4,
    random_state=42,
    verbose=True,
)

lstm.fit(train_df)
test_df["lstm_filter"] = lstm.predict_filter(test_df)


# 4. Evaluate predictions
y_pred = test_df.loc[y_test_true.index, "lstm_filter"].astype(int)

acc = accuracy_score(y_test_true, y_pred)
f1_macro = f1_score(y_test_true, y_pred, average="macro", zero_division=0)

report = classification_report(
    y_test_true,
    y_pred,
    labels=[-1, 0, 1],
    target_names=["short", "stay", "long"],
    zero_division=0,
    output_dict=True,
)

cm = confusion_matrix(y_test_true, y_pred, labels=[-1, 0, 1])

print("\nLSTM PARAMETERS")
print("=" * 80)
print(f"Features:        {len(lstm.feature_columns)}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Hidden size:     {lstm.hidden_size}")
print(f"Num layers:      {lstm.num_layers}")
print(f"Dropout:         {lstm.dropout}")
print(f"Epochs:          {lstm.epochs}")
print(f"Batch size:      {lstm.batch_size}")
print(f"Learning rate:   {lstm.learning_rate}")
print(f"Device:          {lstm.device}")

total_params = sum(p.numel() for p in lstm.model.parameters())
trainable_params = sum(p.numel() for p in lstm.model.parameters() if p.requires_grad)

print(f"Total params:    {total_params}")
print(f"Trainable params:{trainable_params}")

print("\nPRED DISTRIBUTION")
print("=" * 80)
print(y_pred.value_counts().sort_index())

print("\nQUALITY")
print("=" * 80)
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1_macro:.4f}")

print("\nCONFUSION MATRIX")
print("=" * 80)
print("labels = [-1, 0, 1]")
print(cm)

print("\nCLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(
    y_test_true,
    y_pred,
    labels=[-1, 0, 1],
    target_names=["short", "stay", "long"],
    zero_division=0,
))


# 5. Baselines
always_stay = pd.Series(0, index=y_test_true.index)
prev_return = test_df["close"].pct_change()
prev_direction = pd.Series(0, index=test_df.index)
prev_direction[prev_return > 0] = 1
prev_direction[prev_return < 0] = -1
prev_direction = prev_direction.loc[y_test_true.index].astype(int)

baseline_rows = [
    {
        "model": "lstm",
        "accuracy": acc,
        "f1_macro": f1_macro,
    },
    {
        "model": "always_stay",
        "accuracy": accuracy_score(y_test_true, always_stay),
        "f1_macro": f1_score(y_test_true, always_stay, average="macro", zero_division=0),
    },
    {
        "model": "previous_return",
        "accuracy": accuracy_score(y_test_true, prev_direction),
        "f1_macro": f1_score(y_test_true, prev_direction, average="macro", zero_division=0),
    },
]

metrics_df = pd.DataFrame(baseline_rows)

print("\nBASELINE COMPARISON")
print("=" * 80)
print(metrics_df.round(4).to_string(index=False))


# 6. Save metrics
params = {
    "train_size": TRAIN_SIZE,
    "horizon": HORIZON,
    "long_threshold": LONG_THRESHOLD,
    "short_threshold": SHORT_THRESHOLD,
    "sequence_length": SEQUENCE_LENGTH,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "use_raw_ohlcv": USE_RAW_OHLCV,
    "features": len(lstm.feature_columns),
    "hidden_size": lstm.hidden_size,
    "num_layers": lstm.num_layers,
    "dropout": lstm.dropout,
    "device": str(lstm.device),
    "total_params": total_params,
    "trainable_params": trainable_params,
    "accuracy": acc,
    "f1_macro": f1_macro,
}

metrics_df.to_csv(SAVE_DIR / "lstm_baseline_comparison.csv", index=False)

pd.DataFrame(cm, index=["true_short", "true_stay", "true_long"], columns=["pred_short", "pred_stay", "pred_long"]).to_csv(
    SAVE_DIR / "lstm_confusion_matrix.csv"
)

pd.DataFrame(report).T.to_csv(SAVE_DIR / "lstm_classification_report.csv")

test_df[["close", "lstm_filter"]].to_csv(SAVE_DIR / "lstm_predictions.csv")

with open(SAVE_DIR / "lstm_params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False, indent=4)

print("\nSAVED")
print("=" * 80)
print(f"Saved to: {SAVE_DIR}")