import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml.dataset import DirectionTargetBuilder, get_feature_columns
from ml.models import create_model


class WalkForwardMLFilter:
    """
    Walk-forward ML-фильтр.

    ml_filter:
        1  = long
        0  = stay
        -1 = short

    window_mode:
        "expanding" — модель каждый раз обучается на всей доступной истории:
                      [0:start] -> [start:start+test_window]

        "rolling"   — модель каждый раз обучается только на последних train_window свечах:
                      [start-train_window:start] -> [start:start+test_window]
    """

    def __init__(
        self,
        model_name: str = "random_forest",
        train_window: int = 2000,
        test_window: int = 200,
        horizon: int = 1,
        long_threshold: float = 0.003,
        short_threshold: float = -0.003,
        window_mode: str = "expanding",
        random_state: int = 42,
    ):
        if window_mode not in {"expanding", "rolling"}:
            raise ValueError("window_mode must be 'expanding' or 'rolling'")

        self.model_name = model_name
        self.train_window = train_window
        self.test_window = test_window
        self.window_mode = window_mode
        self.random_state = random_state

        self.target_builder = DirectionTargetBuilder(
            horizon=horizon,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
        )

        self.feature_columns: list[str] = []
        self.metrics_: dict[str, float] = {}
        self.windows_: list[dict] = []

    def _create_model(self):
        return create_model(
            model_name=self.model_name,
            random_state=self.random_state,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_index()

        self.feature_columns = get_feature_columns(df)

        if len(df) < self.train_window + self.test_window:
            raise ValueError(
                f"Not enough data: len(df)={len(df)}, "
                f"required at least {self.train_window + self.test_window}"
            )

        df["ml_filter"] = 0
        df["is_ml_predicted"] = 0

        true_targets = []
        predicted_targets = []
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

            # target строится только внутри ML-блока
            y_train = self.target_builder.build(train_df)

            # последние horizon строк train не имеют честного target
            valid_train_mask = y_train.notna()

            X_train = train_df.loc[valid_train_mask, self.feature_columns]
            y_train = y_train.loc[valid_train_mask].astype(int)

            X_test = test_df[self.feature_columns]

            model = self._create_model()
            model.fit(X_train, y_train)

            y_pred = np.asarray(model.predict(X_test)).ravel()

            df.iloc[test_start:test_end, df.columns.get_loc("ml_filter")] = y_pred
            df.iloc[test_start:test_end, df.columns.get_loc("is_ml_predicted")] = 1

            # y_test нужен только для оценки качества модели
            # наружу future_return / target не возвращаем
            y_test = self.target_builder.build(test_df)
            valid_test_mask = y_test.notna()

            y_pred_series = pd.Series(y_pred, index=test_df.index)

            true_targets.extend(y_test.loc[valid_test_mask].astype(int).tolist())
            predicted_targets.extend(
                y_pred_series.loc[valid_test_mask].tolist()
            )

            self.windows_.append(
                {
                    "window_id": len(self.windows_) + 1,
                    "train_start": train_df.index.min(),
                    "train_end": train_df.index.max(),
                    "test_start": test_df.index.min(),
                    "test_end": test_df.index.max(),
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                }
            )

            start += self.test_window

        self._calculate_metrics(true_targets, predicted_targets)

        return df

    def _calculate_metrics(self, y_true: list[int], y_pred: list[int]) -> None:
        if not y_true:
            self.metrics_ = {}
            return

        self.metrics_ = {
            "model_name": self.model_name,
            "window_mode": self.window_mode,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "predicted_rows": len(y_pred),
            "train_window": self.train_window,
            "test_window": self.test_window,
            "n_windows": len(self.windows_),
        }

    def get_metrics(self) -> dict[str, float]:
        return self.metrics_

    def get_windows(self) -> pd.DataFrame:
        return pd.DataFrame(self.windows_)