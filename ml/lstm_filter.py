import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


class LSTMPriceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPriceModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


class WalkForwardLSTMPriceFilter:
    """
    Walk-forward LSTM-фильтр прогноза следующей цены.

    Логика:
        1. На каждом окне модель обучается на прошлом.
        2. LSTM получает sequence_length последних свечей.
        3. Прогнозирует close через horizon свечей.
        4. Если predicted_close > current_close -> ml_filter = 1.
        5. Если predicted_close < current_close -> ml_filter = -1.
        6. Если разница меньше return_threshold -> ml_filter = 0.

    В backtester использовать:
        use_ml_filter=True
        ml_filter_mode="confirm"

    Тогда:
        TA long + LSTM long -> вход разрешен
        TA short + LSTM short -> вход разрешен
        TA и LSTM расходятся -> вход запрещен
    """

    def __init__(
        self,
        train_window: int = 1000,
        test_window: int = 50,
        sequence_length: int = 24,
        horizon: int = 1,
        return_threshold: float = 0.0,
        window_mode: str = "expanding",
        allow_long: bool = True,
        allow_short: bool = True,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        random_state: int = 42,
        verbose: bool = True,
    ):
        if window_mode not in {"expanding", "rolling"}:
            raise ValueError("window_mode must be 'expanding' or 'rolling'")

        self.train_window = train_window
        self.test_window = test_window
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.return_threshold = return_threshold
        self.window_mode = window_mode
        self.allow_long = allow_long
        self.allow_short = allow_short

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.verbose = verbose

        self.feature_columns: list[str] = []
        self.windows_: list[dict] = []
        self.metrics_: dict = {}

        self.device = self._get_device()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        df = df.copy().sort_index()

        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        self.feature_columns = self._get_lstm_price_features(df)

        if len(df) < self.train_window + self.test_window:
            raise ValueError(
                f"Not enough data: len(df)={len(df)}, "
                f"required at least {self.train_window + self.test_window}"
            )

        df["ml_filter"] = 0
        df["predicted_close"] = np.nan
        df["predicted_return"] = np.nan
        df["is_ml_predicted"] = 0

        self.windows_ = []

        start = self.train_window

        while start < len(df):
            if self.window_mode == "expanding":
                train_start = 0
            else:
                train_start = start - self.train_window

            train_end = start
            test_start = start
            test_end = min(start + self.test_window, len(df))

            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()

            if self.verbose:
                print(
                    f"\nLSTM price window {len(self.windows_) + 1}: "
                    f"train={train_df.index.min()} -> {train_df.index.max()}, "
                    f"test={test_df.index.min()} -> {test_df.index.max()}"
                )

            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()

            train_features = self._prepare_features(train_df)
            train_close = train_df[["close"]].astype(float)

            feature_scaler.fit(train_features)
            target_scaler.fit(train_close)

            train_features_scaled = pd.DataFrame(
                feature_scaler.transform(train_features),
                index=train_df.index,
                columns=self.feature_columns,
            )

            train_close_scaled = pd.Series(
                target_scaler.transform(train_close).ravel(),
                index=train_df.index,
                name="close_scaled",
            )

            X_train, y_train = self._build_price_dataset(
                features_scaled=train_features_scaled,
                close_scaled=train_close_scaled,
            )

            if len(y_train) < 30:
                if self.verbose:
                    print("Skipped: not enough train samples")

                df.iloc[test_start:test_end, df.columns.get_loc("is_ml_predicted")] = 1

                self.windows_.append(
                    self._make_window_info(
                        train_df=train_df,
                        test_df=test_df,
                        skipped=True,
                        train_samples=len(y_train),
                        long_predictions=0,
                        short_predictions=0,
                    )
                )

                start += self.test_window
                continue

            model = LSTMPriceModel(
                input_size=len(self.feature_columns),
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)

            self._train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
            )

            # Берем хвост train перед test, чтобы первые test-свечи тоже имели sequence_length истории.
            context_start = max(0, test_start - self.sequence_length + 1)
            context_df = df.iloc[context_start:test_end].copy()

            context_features = self._prepare_features(context_df)

            context_features_scaled = pd.DataFrame(
                feature_scaler.transform(context_features),
                index=context_df.index,
                columns=self.feature_columns,
            )

            predicted_close = self._predict_window(
                model=model,
                features_scaled=context_features_scaled,
                target_scaler=target_scaler,
                test_index=test_df.index,
            )

            for idx, pred_close in predicted_close.items():
                current_close = float(df.loc[idx, "close"])
                pred_return = pred_close / current_close - 1

                df.loc[idx, "predicted_close"] = pred_close
                df.loc[idx, "predicted_return"] = pred_return
                df.loc[idx, "is_ml_predicted"] = 1

                if pred_return > self.return_threshold and self.allow_long:
                    df.loc[idx, "ml_filter"] = 1
                elif pred_return < -self.return_threshold and self.allow_short:
                    df.loc[idx, "ml_filter"] = -1
                else:
                    df.loc[idx, "ml_filter"] = 0

            # На всякий случай помечаем весь test как out-of-sample,
            # даже если для некоторых строк не было прогноза.
            df.loc[test_df.index, "is_ml_predicted"] = 1

            long_predictions = int((df.loc[test_df.index, "ml_filter"] == 1).sum())
            short_predictions = int((df.loc[test_df.index, "ml_filter"] == -1).sum())

            self.windows_.append(
                self._make_window_info(
                    train_df=train_df,
                    test_df=test_df,
                    skipped=False,
                    train_samples=len(y_train),
                    long_predictions=long_predictions,
                    short_predictions=short_predictions,
                )
            )

            start += self.test_window

        self._calculate_metrics(df)

        return df

    def _get_lstm_price_features(self, df: pd.DataFrame) -> list[str]:
        """
        Признаки для LSTM price prediction.

        Здесь специально оставляем open/high/low/close/volume,
        потому что задача — прогнозировать следующую цену на основе прошлых цен,
        как в исходном ноутбуке.
        """

        forbidden = {
            "target",
            "future_return",
            "ml_filter",
            "predicted_close",
            "predicted_return",
            "is_ml_predicted",
            "base_signal",
            "final_signal",
        }

        return [
            col
            for col in df.columns
            if col not in forbidden
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        return X.astype(float)

    def _build_price_dataset(
        self,
        features_scaled: pd.DataFrame,
        close_scaled: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        X: последние sequence_length свечей до момента t включительно.
        y: close на t + horizon.

        При horizon=1:
            последние 24 свечи -> close следующей свечи.
        """

        X_values = features_scaled.values
        y_values = close_scaled.values

        X_sequences = []
        y_targets = []

        max_i = len(features_scaled) - self.horizon

        for i in range(self.sequence_length - 1, max_i):
            seq = X_values[i - self.sequence_length + 1:i + 1]
            target = y_values[i + self.horizon]

            X_sequences.append(seq)
            y_targets.append(target)

        if not X_sequences:
            return (
                np.empty((0, self.sequence_length, len(self.feature_columns))),
                np.array([]),
            )

        return (
            np.asarray(X_sequences, dtype=np.float32),
            np.asarray(y_targets, dtype=np.float32),
        )

    def _predict_window(
        self,
        model: nn.Module,
        features_scaled: pd.DataFrame,
        target_scaler: MinMaxScaler,
        test_index: pd.Index,
    ) -> pd.Series:
        """
        Для каждой свечи test:
            берем sequence_length свечей до текущей включительно;
            прогнозируем close следующей свечи;
            возвращаем predicted_close, привязанный к текущей свече.
        """

        model.eval()

        predictions = {}

        X_values = features_scaled.values
        index = features_scaled.index

        with torch.no_grad():
            for i in range(self.sequence_length - 1, len(features_scaled)):
                idx = index[i]

                if idx not in test_index:
                    continue

                seq = X_values[i - self.sequence_length + 1:i + 1]

                X_tensor = torch.tensor(
                    seq,
                    dtype=torch.float32,
                ).unsqueeze(0).to(self.device)

                pred_scaled = model(X_tensor).cpu().numpy().reshape(-1, 1)
                pred_close = target_scaler.inverse_transform(pred_scaled)[0, 0]

                predictions[idx] = float(pred_close)

        return pd.Series(predictions, name="predicted_close")

    def _train_model(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        dataset = LSTMPriceDataset(X_train, y_train)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                prediction = model(X_batch)
                loss = criterion(prediction, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            if self.verbose and (epoch + 1 == self.epochs):
                avg_loss = epoch_loss / max(len(loader), 1)
                print(f"LSTM price final train loss: {avg_loss:.6f}")

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def _make_window_info(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        skipped: bool,
        train_samples: int,
        long_predictions: int,
        short_predictions: int,
    ) -> dict:
        return {
            "window_id": len(self.windows_) + 1,
            "train_start": train_df.index.min(),
            "train_end": train_df.index.max(),
            "test_start": test_df.index.min(),
            "test_end": test_df.index.max(),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_samples": train_samples,
            "long_predictions": long_predictions,
            "short_predictions": short_predictions,
            "skipped": skipped,
        }

    def _calculate_metrics(self, df: pd.DataFrame) -> None:
        predicted_df = df[df["is_ml_predicted"] == 1]

        self.metrics_ = {
            "filter_type": "lstm_price_prediction",
            "train_window": self.train_window,
            "test_window": self.test_window,
            "sequence_length": self.sequence_length,
            "horizon": self.horizon,
            "return_threshold": self.return_threshold,
            "window_mode": self.window_mode,
            "allow_long": self.allow_long,
            "allow_short": self.allow_short,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": str(self.device),
            "n_windows": len(self.windows_),
            "predicted_rows": len(predicted_df),
            "long_predictions": int((predicted_df["ml_filter"] == 1).sum()),
            "short_predictions": int((predicted_df["ml_filter"] == -1).sum()),
            "mean_predicted_return": float(predicted_df["predicted_return"].mean()),
            "std_predicted_return": float(predicted_df["predicted_return"].std()),
        }

    def get_metrics(self) -> dict:
        return self.metrics_

    def get_windows(self) -> pd.DataFrame:
        return pd.DataFrame(self.windows_)