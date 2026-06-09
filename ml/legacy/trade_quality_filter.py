import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml.dataset import get_feature_columns
from ml.models import create_model
from strategy.base import BaseStrategy


class WalkForwardTradeQualityFilter:
    """
    Walk-forward ML-фильтр качества входов стратегии.

    Идея:
        - стратегия генерирует потенциальные входы;
        - модель обучается отличать прибыльные входы от убыточных;
        - на test-окне модель разрешает или запрещает входы.

    ml_filter:
        1  = разрешить long
        0  = запретить вход
        -1 = разрешить short
    """

    def __init__(
        self,
        model_name: str = "random_forest",
        train_window: int = 2000,
        test_window: int = 200,
        profit_horizon: int = 6,
        min_profit: float = 0.003,
        window_mode: str = "expanding",
        random_state: int = 42,
    ):
        if window_mode not in {"expanding", "rolling"}:
            raise ValueError("window_mode must be 'expanding' or 'rolling'")

        self.model_name = model_name
        self.train_window = train_window
        self.test_window = test_window
        self.profit_horizon = profit_horizon
        self.min_profit = min_profit
        self.window_mode = window_mode
        self.random_state = random_state

        self.feature_columns: list[str] = []
        self.metrics_: dict[str, float] = {}
        self.windows_: list[dict] = []

    def _create_model(self):
        return create_model(
            model_name=self.model_name,
            random_state=self.random_state,
        )

    def transform(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
    ) -> pd.DataFrame:
        df = df.copy().sort_index()

        self.feature_columns = get_feature_columns(df)

        if len(df) < self.train_window + self.test_window:
            raise ValueError(
                f"Not enough data: len(df)={len(df)}, "
                f"required at least {self.train_window + self.test_window}"
            )

        df["ml_filter"] = 0
        df["is_ml_predicted"] = 0

        y_true_all = []
        y_pred_all = []
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

            X_train, y_train = self._build_trade_quality_dataset(
                df=train_df,
                strategy=strategy,
            )

            # Если в окне слишком мало сделок или только один класс — пропускаем обучение
            if len(y_train) < 10 or y_train.nunique() < 2:
                df.iloc[test_start:test_end, df.columns.get_loc("is_ml_predicted")] = 1

                self.windows_.append(
                    self._make_window_info(
                        train_df=train_df,
                        test_df=test_df,
                        train_samples=len(y_train),
                        skipped=True,
                    )
                )

                start += self.test_window
                continue

            model = self._create_model()
            model.fit(X_train, y_train)

            test_orders = strategy.generate_orders(test_df)

            test_long_entries = test_orders.entries
            test_short_entries = test_orders.short_entries
            test_entry_mask = test_long_entries | test_short_entries

            if test_entry_mask.sum() > 0:
                X_test_entries = test_df.loc[test_entry_mask, self.feature_columns]
                y_pred = np.asarray(model.predict(X_test_entries)).ravel().astype(int)

                entry_index = X_test_entries.index

                for idx, pred in zip(entry_index, y_pred):
                    if pred != 1:
                        continue

                    if bool(test_long_entries.loc[idx]):
                        df.loc[idx, "ml_filter"] = 1

                    elif bool(test_short_entries.loc[idx]):
                        df.loc[idx, "ml_filter"] = -1

                # Для оценки качества на test строим фактический target только по входам test-окна
                X_test_quality, y_test_quality = self._build_trade_quality_dataset(
                    df=test_df,
                    strategy=strategy,
                )

                if len(y_test_quality) > 0:
                    # Берем только пересекающиеся индексы, чтобы длины совпали
                    common_index = X_test_entries.index.intersection(X_test_quality.index)

                    if len(common_index) > 0:
                        X_eval = test_df.loc[common_index, self.feature_columns]
                        y_eval_true = y_test_quality.loc[common_index].astype(int)
                        y_eval_pred = np.asarray(model.predict(X_eval)).ravel().astype(int)

                        y_true_all.extend(y_eval_true.tolist())
                        y_pred_all.extend(y_eval_pred.tolist())

            df.iloc[test_start:test_end, df.columns.get_loc("is_ml_predicted")] = 1

            self.windows_.append(
                self._make_window_info(
                    train_df=train_df,
                    test_df=test_df,
                    train_samples=len(y_train),
                    skipped=False,
                )
            )

            start += self.test_window

        self._calculate_metrics(y_true_all, y_pred_all)

        return df

    def _build_trade_quality_dataset(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Строит обучающую выборку только по точкам входа стратегии.

        y = 1, если вход был бы прибыльным через profit_horizon свечей
        y = 0, если вход был бы неприбыльным
        """

        orders = strategy.generate_orders(df)

        long_entries = orders.entries.astype(bool)
        short_entries = orders.short_entries.astype(bool)

        entry_mask = long_entries | short_entries

        if entry_mask.sum() == 0:
            return (
                pd.DataFrame(columns=self.feature_columns),
                pd.Series(dtype=int, name="trade_quality_target"),
            )

        future_close = df["close"].shift(-self.profit_horizon)

        long_return = future_close / df["close"] - 1
        short_return = df["close"] / future_close - 1

        target = pd.Series(np.nan, index=df.index, name="trade_quality_target")

        target[long_entries] = (long_return[long_entries] > self.min_profit).astype(int)
        target[short_entries] = (short_return[short_entries] > self.min_profit).astype(int)

        valid_mask = entry_mask & target.notna()

        X = df.loc[valid_mask, self.feature_columns]
        y = target.loc[valid_mask].astype(int)

        return X, y

    def _calculate_metrics(self, y_true: list[int], y_pred: list[int]) -> None:
        if not y_true:
            self.metrics_ = {
                "model_name": self.model_name,
                "filter_type": "trade_quality",
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "predicted_trade_samples": 0,
                "train_window": self.train_window,
                "test_window": self.test_window,
                "profit_horizon": self.profit_horizon,
                "min_profit": self.min_profit,
                "window_mode": self.window_mode,
                "n_windows": len(self.windows_),
            }
            return

        self.metrics_ = {
            "model_name": self.model_name,
            "filter_type": "trade_quality",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "predicted_trade_samples": len(y_pred),
            "train_window": self.train_window,
            "test_window": self.test_window,
            "profit_horizon": self.profit_horizon,
            "min_profit": self.min_profit,
            "window_mode": self.window_mode,
            "n_windows": len(self.windows_),
        }

    def _make_window_info(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_samples: int,
        skipped: bool,
    ) -> dict:
        return {
            "window_id": len(self.windows_) + 1,
            "train_start": train_df.index.min(),
            "train_end": train_df.index.max(),
            "test_start": test_df.index.min(),
            "test_end": test_df.index.max(),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_trade_samples": train_samples,
            "skipped": skipped,
        }

    def get_metrics(self) -> dict:
        return self.metrics_

    def get_windows(self) -> pd.DataFrame:
        return pd.DataFrame(self.windows_)