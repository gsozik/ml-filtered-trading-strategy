from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.base import BaseMLFilter
from ml.dataset import DirectionTargetBuilder, get_feature_columns
from ml.models import create_model
from ml.scalers import create_scaler
from ml.tuning import tune_filter, get_direction_search_space


class DirectionMLFilter(BaseMLFilter):
    """
    ML-фильтр направления для:
    - logistic_regression
    - random_forest
    - catboost

    fit(train_df)
    predict_filter(test_df) -> pd.Series[-1, 0, 1]
    """

    def __init__(
        self,
        model_name: str,
        horizon: int = 32,
        long_threshold: float = 0.02,
        short_threshold: float = -0.02,
        use_raw_ohlcv: bool = False,
        normalizer: str = "standard",
        tune: bool = False,
        n_trials: int = 20,
        val_size: float = 0.2,
        save_model: bool = False,
        model_dir: str = "models",
        model_save_name: str | None = None,
        load_model_path: str | None = None,
        random_state: int = 42,
        params: dict | None = None,
        verbose: bool = True,
    ):
        if normalizer == "auto" and not tune and load_model_path is None:
            raise ValueError("normalizer='auto' можно использовать только при tune=True")

        self.name = model_name
        self.model_name = model_name

        self.horizon = horizon
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.use_raw_ohlcv = use_raw_ohlcv

        self.normalizer = normalizer
        self.tune = tune
        self.n_trials = n_trials
        self.val_size = val_size

        self.save_model_flag = save_model
        self.model_dir = Path(model_dir)
        self.model_save_name = model_save_name or f"{model_name}_direction_filter"
        self.load_model_path = load_model_path

        self.random_state = random_state
        self.params = params or {}
        self.verbose = verbose

        self.target_builder = DirectionTargetBuilder(
            horizon=self.horizon,
            long_threshold=self.long_threshold,
            short_threshold=self.short_threshold,
        )

        self.scaler = None
        self.model = None
        self.feature_columns: list[str] = []
        self.best_params: dict | None = None
        self.is_fitted = False

        self.model_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, df: pd.DataFrame):
        df = df.copy().sort_index()

        if self.load_model_path is not None:
            self.load_model(self.load_model_path)
            return self

        if self.tune:
            self.best_params = self._tune(df)
        
            self.normalizer = self.best_params.get("normalizer", self.normalizer)
            self.params = {
                key: value
                for key, value in self.best_params.items()
                if key != "normalizer"
            }

        self.feature_columns = get_feature_columns(
            df,
            use_raw_ohlcv=self.use_raw_ohlcv,
        )

        y = self.target_builder.build(df)
        valid_mask = y.notna()

        X = self._prepare_X(df)

        X_train = X.loc[valid_mask]
        y_train = y.loc[valid_mask].astype(int)

        if len(y_train) == 0:
            raise ValueError("No training samples. Check horizon and thresholds.")

        self.scaler = create_scaler(self.normalizer)

        if self.scaler is not None:
            self.scaler.fit(X_train)
            X_train = pd.DataFrame(
                self.scaler.transform(X_train),
                index=X_train.index,
                columns=self.feature_columns,
            )

        self.model = create_model(
            model_name=self.model_name,
            random_state=self.random_state,
            params=self.params,
        )

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        if self.save_model_flag:
            self.save_model()

        return self

    def predict_filter(self, df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit(train_df) first.")

        df = df.copy().sort_index()

        X = self._prepare_X(df)

        if self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                index=X.index,
                columns=self.feature_columns,
            )

        pred = self.model.predict(X)

        return pd.Series(
            np.asarray(pred).ravel(),
            index=df.index,
            name=self.name,
        ).astype(int)

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[self.feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    def _tune(self, df: pd.DataFrame) -> dict:
        search_space = get_direction_search_space(
            model_name=self.model_name,
            normalizer=self.normalizer,
        )

        def filter_factory(tuned_params: dict):
            return DirectionMLFilter(
                model_name=self.model_name,
                horizon=self.horizon,
                long_threshold=self.long_threshold,
                short_threshold=self.short_threshold,
                use_raw_ohlcv=self.use_raw_ohlcv,
                normalizer=tuned_params["normalizer"],
                tune=False,
                save_model=False,
                random_state=self.random_state,
                params=tuned_params["model_params"],
                verbose=False,
            )

        return tune_filter(
            filter_factory=filter_factory,
            train_df=df,
            target_builder=self.target_builder,
            search_space_func=search_space,
            n_trials=self.n_trials,
            val_size=self.val_size,
            model_dir=str(self.model_dir),
            model_name=self.model_save_name,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def save_model(self, path: str | None = None):
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Nothing to save.")

        path = Path(path) if path is not None else self.model_dir / f"{self.model_save_name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model_name": self.model_name,
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "horizon": self.horizon,
            "long_threshold": self.long_threshold,
            "short_threshold": self.short_threshold,
            "use_raw_ohlcv": self.use_raw_ohlcv,
            "normalizer": self.normalizer,
            "params": self.params,
            "best_params": self.best_params,
        }

        joblib.dump(bundle, path)

        if self.verbose:
            print(f"Saved ML model to: {path}")

    def load_model(self, path: str):
        path = Path(path)
        bundle = joblib.load(path)

        self.model_name = bundle["model_name"]
        self.name = self.model_name

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_columns = bundle["feature_columns"]

        self.horizon = bundle["horizon"]
        self.long_threshold = bundle["long_threshold"]
        self.short_threshold = bundle["short_threshold"]
        self.use_raw_ohlcv = bundle["use_raw_ohlcv"]
        self.normalizer = bundle["normalizer"]
        self.params = bundle.get("params", {})
        self.best_params = bundle.get("best_params")

        self.target_builder = DirectionTargetBuilder(
            horizon=self.horizon,
            long_threshold=self.long_threshold,
            short_threshold=self.short_threshold,
        )

        self.is_fitted = True

        if self.verbose:
            print(f"Loaded ML model from: {path}")