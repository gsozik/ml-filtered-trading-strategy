import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml.dataset import get_feature_columns
from ml.models import create_model
from ml.legacy.optuna_tuning import tune_model_params


class WalkForwardRegimeFilter:
    """
    Walk-forward ML-фильтр рыночного режима.

    target = close > close.shift(regime_lookback)

    ml_filter:
        1  = разрешить long
        -1 = разрешить short, если allow_short=True
        0  = запретить вход

    При use_optuna=True:
        в каждом walk-forward окне гиперпараметры подбираются
        только на train-части через внутренний time validation.
    """

    def __init__(
        self,
        model_name: str = "catboost",
        train_window: int = 2000,
        test_window: int = 200,
        regime_lookback: int = 50,
        window_mode: str = "expanding",
        allow_short: bool = False,
        random_state: int = 42,
        use_optuna: bool = False,
        optuna_trials: int = 20,
        optuna_validation_size: float = 0.25,
        optuna_metric: str = "f1_macro",
    ):
        if window_mode not in {"expanding", "rolling"}:
            raise ValueError("window_mode must be 'expanding' or 'rolling'")

        self.model_name = model_name
        self.train_window = train_window
        self.test_window = test_window
        self.regime_lookback = regime_lookback
        self.window_mode = window_mode
        self.allow_short = allow_short
        self.random_state = random_state

        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.optuna_validation_size = optuna_validation_size
        self.optuna_metric = optuna_metric

        self.feature_columns: list[str] = []
        self.metrics_: dict[str, float] = {}
        self.windows_: list[dict] = []

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

            y_train = self._build_regime_target(train_df)
            valid_train_mask = y_train.notna()

            X_train = train_df.loc[valid_train_mask, self.feature_columns]
            y_train = y_train.loc[valid_train_mask].astype(int)

            if len(y_train) < 20 or y_train.nunique() < 2:
                df.iloc[test_start:test_end, df.columns.get_loc("is_ml_predicted")] = 1

                self.windows_.append(
                    self._make_window_info(
                        train_df=train_df,
                        test_df=test_df,
                        train_samples=len(y_train),
                        skipped=True,
                        best_params={},
                    )
                )

                start += self.test_window
                continue

            best_params = {}

            if self.use_optuna:
                best_params = tune_model_params(
                    model_name=self.model_name,
                    X=X_train,
                    y=y_train,
                    random_state=self.random_state,
                    n_trials=self.optuna_trials,
                    validation_size=self.optuna_validation_size,
                    metric=self.optuna_metric,
                )

            model = create_model(
                model_name=self.model_name,
                random_state=self.random_state,
                params=best_params,
            )

            model.fit(X_train, y_train)

            X_test = test_df[self.feature_columns]
            y_pred = np.asarray(model.predict(X_test)).ravel().astype(int)

            ml_values = self._predictions_to_filter(y_pred)

            df.iloc[test_start:test_end, df.columns.get_loc("ml_filter")] = ml_values
            df.iloc[test_start:test_end, df.columns.get_loc("is_ml_predicted")] = 1

            y_test = self._build_regime_target(test_df)
            valid_test_mask = y_test.notna()

            y_pred_series = pd.Series(y_pred, index=test_df.index)

            y_true_all.extend(y_test.loc[valid_test_mask].astype(int).tolist())
            y_pred_all.extend(
                y_pred_series.loc[valid_test_mask].astype(int).tolist()
            )

            self.windows_.append(
                self._make_window_info(
                    train_df=train_df,
                    test_df=test_df,
                    train_samples=len(y_train),
                    skipped=False,
                    best_params=best_params,
                )
            )

            start += self.test_window

        self._calculate_metrics(y_true_all, y_pred_all)

        return df

    def _build_regime_target(self, df: pd.DataFrame) -> pd.Series:
        past_close = df["close"].shift(self.regime_lookback)

        target = pd.Series(np.nan, index=df.index, name="regime_target")

        target[df["close"] > past_close] = 1
        target[df["close"] <= past_close] = -1

        return target

    def _predictions_to_filter(self, y_pred: np.ndarray) -> np.ndarray:
        if self.allow_short:
            return y_pred.astype(int)

        return np.where(y_pred == 1, 1, 0).astype(int)

    def _calculate_metrics(self, y_true: list[int], y_pred: list[int]) -> None:
        if not y_true:
            self.metrics_ = {
                "model_name": self.model_name,
                "filter_type": "regime",
                "accuracy": None,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": None,
                "predicted_rows": 0,
                "train_window": self.train_window,
                "test_window": self.test_window,
                "regime_lookback": self.regime_lookback,
                "window_mode": self.window_mode,
                "allow_short": self.allow_short,
                "use_optuna": self.use_optuna,
                "optuna_trials": self.optuna_trials,
                "n_windows": len(self.windows_),
            }
            return

        self.metrics_ = {
            "model_name": self.model_name,
            "filter_type": "regime",
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
            "regime_lookback": self.regime_lookback,
            "window_mode": self.window_mode,
            "allow_short": self.allow_short,
            "use_optuna": self.use_optuna,
            "optuna_trials": self.optuna_trials,
            "n_windows": len(self.windows_),
        }

    def _make_window_info(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_samples: int,
        skipped: bool,
        best_params: dict,
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
            "skipped": skipped,
            "best_params": str(best_params),
        }

    def get_metrics(self) -> dict:
        return self.metrics_

    def get_windows(self) -> pd.DataFrame:
        return pd.DataFrame(self.windows_)