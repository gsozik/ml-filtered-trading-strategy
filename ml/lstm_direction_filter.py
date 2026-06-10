import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

from ml.base import BaseMLFilter
from ml.dataset import DirectionTargetBuilder, get_feature_columns
from ml.scalers import create_scaler


class LSTMDirectionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMDirectionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


class LSTMDirectionFilter(BaseMLFilter):
    name = "lstm"

    def __init__(
        self,
        horizon: int = 32,
        long_threshold: float = 0.02,
        short_threshold: float = -0.02,
        sequence_length: int = 50,
        use_raw_ohlcv: bool = False,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        normalizer: str = "standard",
        tune: bool = False,
        n_trials: int = 20,
        val_size: float = 0.2,
        save_weights: bool = False,
        model_dir: str = "models",
        model_name: str = "lstm_direction_filter",
        load_weights_path: str | None = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        if normalizer == "auto" and not tune and load_weights_path is None:
            raise ValueError("normalizer='auto' можно использовать только при tune=True")

        self.horizon = horizon
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.sequence_length = sequence_length
        self.use_raw_ohlcv = use_raw_ohlcv

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.normalizer = normalizer
        self.tune = tune
        self.n_trials = n_trials
        self.val_size = val_size

        self.save_weights = save_weights
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.load_weights_path = load_weights_path

        self.random_state = random_state
        self.verbose = verbose
        self.device = self._get_device()

        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.train_tail = None
        self.train_losses = []
        self.best_params = None

        self.target_builder = DirectionTargetBuilder(
            horizon=self.horizon,
            long_threshold=self.long_threshold,
            short_threshold=self.short_threshold,
        )

        self.model_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, df: pd.DataFrame):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        df = df.copy().sort_index()

        if self.load_weights_path is not None:
            self.load_weights(self.load_weights_path)
            self.train_tail = df.tail(self.sequence_length - 1).copy()
            return self

        if self.tune:
            self.best_params = self._tune_hyperparameters(df)

            for key, value in self.best_params.items():
                setattr(self, key, value)

        self.scaler = create_scaler(self.normalizer)

        self.feature_columns = get_feature_columns(
            df,
            use_raw_ohlcv=self.use_raw_ohlcv,
        )

        y_raw = self.target_builder.build(df)
        valid_mask = y_raw.notna()

        X_raw = df[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

        if self.scaler is not None:
            self.scaler.fit(X_raw.loc[valid_mask])
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_raw),
                index=df.index,
                columns=self.feature_columns,
            )
        else:
            X_scaled = X_raw.copy()

        X_seq, y_seq = self._build_sequences(X_scaled, y_raw)

        if len(y_seq) == 0:
            raise ValueError("No training samples. Check train size, horizon and sequence_length.")

        self.model = LSTMDirectionModel(
            input_size=len(self.feature_columns),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        self._train_model(X_seq, y_seq)

        self.train_tail = df.tail(self.sequence_length - 1).copy()

        if self.save_weights:
            self.save_model()

        return self

    def predict_filter(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit(train_df) first.")

        if self.train_tail is None:
            raise ValueError("train_tail is empty. Call fit(train_df) first.")

        df = df.copy().sort_index()

        context_df = pd.concat([self.train_tail, df])
        context_df = context_df[~context_df.index.duplicated(keep="last")].sort_index()

        X_raw = context_df[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_raw),
                index=context_df.index,
                columns=self.feature_columns,
            )
        else:
            X_scaled = X_raw.copy()

        preds = pd.Series(0, index=df.index, name=self.name, dtype=int)

        self.model.eval()

        values = X_scaled.values
        index = X_scaled.index

        with torch.no_grad():
            for i in range(self.sequence_length - 1, len(X_scaled)):
                idx = index[i]

                if idx not in df.index:
                    continue

                seq = values[i - self.sequence_length + 1:i + 1]
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

                logits = self.model(x)
                class_id = int(torch.argmax(logits, dim=1).item())

                preds.loc[idx] = self._class_to_signal(class_id)

        return preds.astype(int)

    def _build_sequences(self, X_scaled: pd.DataFrame, y_raw: pd.Series):
        values = X_scaled.values
        index = X_scaled.index

        X_seq = []
        y_seq = []

        for i in range(self.sequence_length - 1, len(X_scaled)):
            idx = index[i]

            if pd.isna(y_raw.loc[idx]):
                continue

            signal = int(y_raw.loc[idx])
            X_seq.append(values[i - self.sequence_length + 1:i + 1])
            y_seq.append(self._signal_to_class(signal))

        return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.int64)

    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        dataset = LSTMDirectionDataset(X_train, y_train)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        class_counts = np.bincount(y_train, minlength=3).astype(float)
        class_counts[class_counts == 0] = 1.0

        class_weights = class_counts.sum() / class_counts
        class_weights = class_weights / class_weights.mean()

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.train_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(loader), 1)
            self.train_losses.append(avg_loss)

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} | loss={avg_loss:.6f}")

    def _tune_hyperparameters(self, df: pd.DataFrame) -> dict:
        import optuna

        split = int(len(df) * (1 - self.val_size))
        inner_train_df = df.iloc[:split].copy()
        val_df = df.iloc[split:].copy()

        y_val = self.target_builder.build(val_df).dropna().astype(int)

        normalizers = (
            ["standard", "robust", "maxabs", "none"]
            if self.normalizer == "auto"
            else [self.normalizer]
        )

        def objective(trial):
            params = {
                "sequence_length": trial.suggest_categorical("sequence_length", [30, 50, 80, 100]),
                "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "epochs": trial.suggest_categorical("epochs", [10, 20, 30, 50]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 3e-4, 1e-4]),
                "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3]),
                "normalizer": trial.suggest_categorical("normalizer", normalizers),
            }

            model = LSTMDirectionFilter(
                horizon=self.horizon,
                long_threshold=self.long_threshold,
                short_threshold=self.short_threshold,
                use_raw_ohlcv=self.use_raw_ohlcv,
                tune=False,
                save_weights=False,
                random_state=self.random_state,
                verbose=False,
                **params,
            )

            try:
                model.fit(inner_train_df)
                pred = model.predict_filter(val_df).loc[y_val.index].astype(int)
            except Exception:
                return -1.0

            raw_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
            score = raw_f1

            if pred.nunique() == 1:
                score -= 0.15

            if pred.nunique() >= 2:
                score += 0.03

            trial.set_user_attr("raw_f1_macro", raw_f1)
            trial.set_user_attr("pred_classes", int(pred.nunique()))
            trial.set_user_attr(
                "pred_distribution",
                pred.value_counts().sort_index().to_dict(),
            )

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        trials_path = self.model_dir / f"{self.model_name}_optuna_trials.csv"
        best_params_path = self.model_dir / f"{self.model_name}_best_params.json"

        study.trials_dataframe().to_csv(trials_path, index=False)

        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, ensure_ascii=False, indent=4)

        if self.verbose:
            print("\nLSTM OPTUNA BEST PARAMS")
            print("=" * 80)
            print(study.best_params)
            print(f"Best score: {study.best_value:.4f}")

        return study.best_params

    def save_model(self, path: str | None = None):
        if self.model is None:
            raise ValueError("Model is not fitted. Nothing to save.")

        path = self.model_dir / f"{self.model_name}.pt" if path is None else Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "feature_columns": self.feature_columns,
            "sequence_length": self.sequence_length,
            "horizon": self.horizon,
            "long_threshold": self.long_threshold,
            "short_threshold": self.short_threshold,
            "use_raw_ohlcv": self.use_raw_ohlcv,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "normalizer": self.normalizer,
            "best_params": self.best_params,
            "train_losses": self.train_losses,
        }

        torch.save(checkpoint, path)

        if self.scaler is not None:
            joblib.dump(self.scaler, path.with_suffix(".scaler.pkl"))

        if self.verbose:
            print(f"Saved LSTM weights to: {path}")

    def load_weights(self, path: str):
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.feature_columns = checkpoint["feature_columns"]
        self.sequence_length = checkpoint["sequence_length"]
        self.horizon = checkpoint["horizon"]
        self.long_threshold = checkpoint["long_threshold"]
        self.short_threshold = checkpoint["short_threshold"]
        self.use_raw_ohlcv = checkpoint["use_raw_ohlcv"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.normalizer = checkpoint["normalizer"]
        self.best_params = checkpoint.get("best_params")
        self.train_losses = checkpoint.get("train_losses", [])

        scaler_path = path.with_suffix(".scaler.pkl")
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else create_scaler(self.normalizer)

        self.target_builder = DirectionTargetBuilder(
            horizon=self.horizon,
            long_threshold=self.long_threshold,
            short_threshold=self.short_threshold,
        )

        self.model = LSTMDirectionModel(
            input_size=len(self.feature_columns),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        if self.verbose:
            print(f"Loaded LSTM weights from: {path}")

    @staticmethod
    def _signal_to_class(signal: int) -> int:
        if signal == -1:
            return 0
        if signal == 0:
            return 1
        if signal == 1:
            return 2

        raise ValueError(f"Unknown signal: {signal}")

    @staticmethod
    def _class_to_signal(class_id: int) -> int:
        if class_id == 0:
            return -1
        if class_id == 1:
            return 0
        if class_id == 2:
            return 1

        raise ValueError(f"Unknown class_id: {class_id}")

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")