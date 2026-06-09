import warnings
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

from ml.dataset import get_feature_columns
from ml.models import create_model


@dataclass
class StrategyOrdersLike:
    entries: pd.Series
    exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series


class WalkForwardTradingOptimizedFilter:
    """
    Walk-forward ML-фильтр, оптимизированный по торговому результату.

    Главная идея:
        1. Стратегия дает потенциальные входы.
        2. ML учится отличать хорошие входы от плохих.
        3. Optuna подбирает параметры модели и порог вероятности.
        4. Целевая функция Optuna — не F1/accuracy, а результат backtest на validation.

    В каждом walk-forward окне:

        train_window
        ├── inner_train  -> обучение модели
        └── validation   -> подбор параметров по backtest score

        test_window      -> честная будущая проверка
    """

    def __init__(
        self,
        model_name: str = "catboost",
        train_window: int = 1000,
        test_window: int = 100,
        validation_size: float = 0.25,
        window_mode: str = "expanding",
        n_trials: int = 30,
        random_state: int = 42,
        min_train_samples: int = 30,
        min_validation_trades: int = 1,
        score_metric: str = "return_minus_drawdown",
        verbose: bool = True,
    ):
        if window_mode not in {"expanding", "rolling"}:
            raise ValueError("window_mode must be 'expanding' or 'rolling'")

        if not 0.1 <= validation_size <= 0.5:
            raise ValueError("validation_size must be between 0.1 and 0.5")

        self.model_name = model_name
        self.train_window = train_window
        self.test_window = test_window
        self.validation_size = validation_size
        self.window_mode = window_mode
        self.n_trials = n_trials
        self.random_state = random_state
        self.min_train_samples = min_train_samples
        self.min_validation_trades = min_validation_trades
        self.score_metric = score_metric
        self.verbose = verbose

        self.base_feature_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.windows_: list[dict] = []
        self.metrics_: dict = {}

    def transform(
        self,
        df: pd.DataFrame,
        strategy,
        backtester,
    ) -> pd.DataFrame:
        """
        Возвращает df с колонками:
            ml_filter:
                1  = разрешить long entry
                -1 = разрешить short entry
                0  = запретить вход

            ml_probability:
                вероятность хорошего входа

            is_ml_predicted:
                1 = строка относится к out-of-sample test-окну
        """

        df = df.copy().sort_index()

        self.base_feature_columns = get_feature_columns(df)
        self.feature_columns = self.base_feature_columns + ["signal_direction"]

        if len(df) < self.train_window + self.test_window:
            raise ValueError(
                f"Not enough data: len(df)={len(df)}, "
                f"required at least {self.train_window + self.test_window}"
            )

        df["ml_filter"] = 0
        df["ml_probability"] = np.nan
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
                    f"\nTrading-optimized window {len(self.windows_) + 1}: "
                    f"train={train_df.index.min()} -> {train_df.index.max()}, "
                    f"test={test_df.index.min()} -> {test_df.index.max()}"
                )

            best_config = self._optimize_window(
                train_df=train_df,
                strategy=strategy,
                backtester=backtester,
            )

            if best_config is None:
                df.iloc[test_start:test_end, df.columns.get_loc("is_ml_predicted")] = 1

                self.windows_.append(
                    self._make_window_info(
                        train_df=train_df,
                        test_df=test_df,
                        skipped=True,
                        best_score=None,
                        best_config=None,
                        train_samples=0,
                        allowed_long=0,
                        allowed_short=0,
                    )
                )

                start += self.test_window
                continue

            X_train, y_train = self._build_trade_quality_dataset(
                df=train_df,
                strategy=strategy,
                horizon=best_config["target_horizon"],
                min_profit=best_config["min_profit"],
                allow_long=best_config["allow_long"],
                allow_short=best_config["allow_short"],
            )

            if len(y_train) < self.min_train_samples or y_train.nunique() < 2:
                df.iloc[test_start:test_end, df.columns.get_loc("is_ml_predicted")] = 1

                self.windows_.append(
                    self._make_window_info(
                        train_df=train_df,
                        test_df=test_df,
                        skipped=True,
                        best_score=best_config.get("best_score"),
                        best_config=best_config,
                        train_samples=len(y_train),
                        allowed_long=0,
                        allowed_short=0,
                    )
                )

                start += self.test_window
                continue

            model = create_model(
                model_name=self.model_name,
                random_state=self.random_state,
                params=best_config["model_params"],
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            filtered_test_df = self._apply_model_filter(
                df_part=test_df,
                strategy=strategy,
                model=model,
                probability_threshold=best_config["probability_threshold"],
                allow_long=best_config["allow_long"],
                allow_short=best_config["allow_short"],
            )

            df.loc[filtered_test_df.index, "ml_filter"] = filtered_test_df["ml_filter"]
            df.loc[filtered_test_df.index, "ml_probability"] = filtered_test_df["ml_probability"]
            df.loc[filtered_test_df.index, "is_ml_predicted"] = 1

            allowed_long = int((filtered_test_df["ml_filter"] == 1).sum())
            allowed_short = int((filtered_test_df["ml_filter"] == -1).sum())

            self.windows_.append(
                self._make_window_info(
                    train_df=train_df,
                    test_df=test_df,
                    skipped=False,
                    best_score=best_config.get("best_score"),
                    best_config=best_config,
                    train_samples=len(y_train),
                    allowed_long=allowed_long,
                    allowed_short=allowed_short,
                )
            )

            start += self.test_window

        self._calculate_metrics(df)

        return df

    def _optimize_window(
        self,
        train_df: pd.DataFrame,
        strategy,
        backtester,
    ) -> dict | None:
        split_idx = int(len(train_df) * (1 - self.validation_size))

        if split_idx <= 0 or split_idx >= len(train_df):
            return None

        inner_train_df = train_df.iloc[:split_idx].copy()
        validation_df = train_df.iloc[split_idx:].copy()

        def objective(trial: optuna.Trial) -> float:
            try:
                model_params = self._suggest_model_params(trial)

                probability_threshold = trial.suggest_float(
                    "probability_threshold",
                    0.50,
                    0.85,
                    step=0.05,
                )

                target_horizon = trial.suggest_int(
                    "target_horizon",
                    3,
                    24,
                    step=3,
                )

                min_profit = trial.suggest_float(
                    "min_profit",
                    0.0,
                    0.02,
                    step=0.002,
                )

                allow_long = trial.suggest_categorical(
                    "allow_long",
                    [True],
                )

                allow_short = trial.suggest_categorical(
                    "allow_short",
                    [False, True],
                )

                X_inner_train, y_inner_train = self._build_trade_quality_dataset(
                    df=inner_train_df,
                    strategy=strategy,
                    horizon=target_horizon,
                    min_profit=min_profit,
                    allow_long=allow_long,
                    allow_short=allow_short,
                )

                if len(y_inner_train) < self.min_train_samples:
                    return -1_000_000.0

                if y_inner_train.nunique() < 2:
                    return -1_000_000.0

                model = create_model(
                    model_name=self.model_name,
                    random_state=self.random_state,
                    params=model_params,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_inner_train, y_inner_train)

                filtered_validation_df = self._apply_model_filter(
                    df_part=validation_df,
                    strategy=strategy,
                    model=model,
                    probability_threshold=probability_threshold,
                    allow_long=allow_long,
                    allow_short=allow_short,
                )

                validation_result = self._run_backtest(
                    backtester=backtester,
                    df=filtered_validation_df,
                    strategy=strategy,
                    use_ml_filter=True,
                )

                score = self._score_backtest_result(validation_result)

                metrics = validation_result["metrics"]
                trades = metrics.get("total_trades", 0)

                if trades is None:
                    trades = 0

                if trades < self.min_validation_trades:
                    score -= 100_000.0

                return float(score)

            except Exception:
                return -1_000_000.0

        sampler = optuna.samplers.TPESampler(seed=self.random_state)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
            n_jobs=1,
        )

        if study.best_value <= -999_999:
            return None

        best_trial_params = study.best_params.copy()

        model_param_names = self._model_param_names()
        model_params = {
            key: value
            for key, value in best_trial_params.items()
            if key in model_param_names
        }

        best_config = {
            "model_name": self.model_name,
            "model_params": model_params,
            "probability_threshold": best_trial_params["probability_threshold"],
            "target_horizon": best_trial_params["target_horizon"],
            "min_profit": best_trial_params["min_profit"],
            "allow_long": best_trial_params["allow_long"],
            "allow_short": best_trial_params["allow_short"],
            "best_score": float(study.best_value),
        }

        return best_config

    def _build_trade_quality_dataset(
        self,
        df: pd.DataFrame,
        strategy,
        horizon: int,
        min_profit: float,
        allow_long: bool,
        allow_short: bool,
    ) -> tuple[pd.DataFrame, pd.Series]:
        orders = self._get_strategy_orders(strategy, df)

        long_entries = orders.entries.astype(bool)
        short_entries = orders.short_entries.astype(bool)

        if not allow_long:
            long_entries = pd.Series(False, index=df.index)

        if not allow_short:
            short_entries = pd.Series(False, index=df.index)

        entry_mask = long_entries | short_entries

        if entry_mask.sum() == 0:
            return (
                pd.DataFrame(columns=self.feature_columns),
                pd.Series(dtype=int, name="trade_quality_target"),
            )

        future_close = df["close"].shift(-horizon)

        long_return = future_close / df["close"] - 1
        short_return = df["close"] / future_close - 1

        target = pd.Series(np.nan, index=df.index, name="trade_quality_target")

        target[long_entries] = (long_return[long_entries] > min_profit).astype(int)
        target[short_entries] = (short_return[short_entries] > min_profit).astype(int)

        valid_mask = entry_mask & target.notna()

        if valid_mask.sum() == 0:
            return (
                pd.DataFrame(columns=self.feature_columns),
                pd.Series(dtype=int, name="trade_quality_target"),
            )

        X = df.loc[valid_mask, self.base_feature_columns].copy()
        X["signal_direction"] = 0
        X.loc[long_entries.loc[valid_mask], "signal_direction"] = 1
        X.loc[short_entries.loc[valid_mask], "signal_direction"] = -1

        y = target.loc[valid_mask].astype(int)

        return X, y

    def _apply_model_filter(
        self,
        df_part: pd.DataFrame,
        strategy,
        model,
        probability_threshold: float,
        allow_long: bool,
        allow_short: bool,
    ) -> pd.DataFrame:
        df_part = df_part.copy()

        df_part["ml_filter"] = 0
        df_part["ml_probability"] = np.nan

        orders = self._get_strategy_orders(strategy, df_part)

        long_entries = orders.entries.astype(bool)
        short_entries = orders.short_entries.astype(bool)

        if not allow_long:
            long_entries = pd.Series(False, index=df_part.index)

        if not allow_short:
            short_entries = pd.Series(False, index=df_part.index)

        candidate_mask = long_entries | short_entries

        if candidate_mask.sum() == 0:
            return df_part

        X_candidates = df_part.loc[candidate_mask, self.base_feature_columns].copy()
        X_candidates["signal_direction"] = 0
        X_candidates.loc[long_entries.loc[candidate_mask], "signal_direction"] = 1
        X_candidates.loc[short_entries.loc[candidate_mask], "signal_direction"] = -1

        probabilities = self._predict_positive_probability(model, X_candidates)

        probability_series = pd.Series(
            probabilities,
            index=X_candidates.index,
            name="ml_probability",
        )

        allowed_mask = probability_series >= probability_threshold

        allowed_index = probability_series.index[allowed_mask]

        df_part.loc[probability_series.index, "ml_probability"] = probability_series

        allowed_long_index = allowed_index.intersection(long_entries[long_entries].index)
        allowed_short_index = allowed_index.intersection(short_entries[short_entries].index)

        df_part.loc[allowed_long_index, "ml_filter"] = 1
        df_part.loc[allowed_short_index, "ml_filter"] = -1

        return df_part

    @staticmethod
    def _predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            pred = np.asarray(model.predict(X)).ravel()
            return pred.astype(float)

        proba = model.predict_proba(X)

        classes = getattr(model, "classes_", None)

        if classes is None and hasattr(model, "named_steps"):
            final_model = model.named_steps.get("model")
            classes = getattr(final_model, "classes_", None)

        if classes is None:
            return proba[:, -1]

        classes = list(classes)

        if 1 in classes:
            positive_idx = classes.index(1)
            return proba[:, positive_idx]

        return np.zeros(len(X), dtype=float)

    def _score_backtest_result(self, result: dict) -> float:
        metrics = result["metrics"]

        total_return = metrics.get("total_return_pct", 0.0)
        max_drawdown = metrics.get("max_drawdown_pct", 0.0)
        sharpe = metrics.get("sharpe_ratio", 0.0)
        profit_factor = metrics.get("profit_factor", 0.0)

        total_return = self._safe_number(total_return)
        max_drawdown = abs(self._safe_number(max_drawdown))
        sharpe = self._safe_number(sharpe)
        profit_factor = self._safe_number(profit_factor)

        if self.score_metric == "return_minus_drawdown":
            return total_return - max_drawdown

        if self.score_metric == "return_plus_sharpe_minus_drawdown":
            return total_return + 5.0 * sharpe - max_drawdown

        if self.score_metric == "profit_factor":
            return min(profit_factor, 10.0)

        raise ValueError(
            "score_metric must be one of: "
            "'return_minus_drawdown', "
            "'return_plus_sharpe_minus_drawdown', "
            "'profit_factor'"
        )

    @staticmethod
    def _safe_number(value) -> float:
        if value is None:
            return 0.0

        try:
            value = float(value)
        except Exception:
            return 0.0

        if np.isnan(value) or np.isinf(value):
            return 0.0

        return value

    def _suggest_model_params(self, trial: optuna.Trial) -> dict:
        model_name = self.model_name.lower()

        if model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=100),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80),
                "min_samples_split": trial.suggest_int("min_samples_split", 10, 120),
                "max_features": trial.suggest_categorical(
                    "max_features",
                    ["sqrt", "log2", None],
                ),
            }

        if model_name == "logistic_regression":
            return {
                "C": trial.suggest_float("C", 0.001, 100.0, log=True),
                "penalty": "l2",
                "solver": "lbfgs",
            }

        if model_name == "catboost":
            return {
                "iterations": trial.suggest_int("iterations", 100, 700, step=100),
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.005,
                    0.15,
                    log=True,
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg",
                    1.0,
                    20.0,
                    log=True,
                ),
                "random_strength": trial.suggest_float(
                    "random_strength",
                    0.1,
                    10.0,
                    log=True,
                ),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature",
                    0.0,
                    5.0,
                ),
            }

        raise ValueError(f"Unknown model_name='{self.model_name}'")

    def _model_param_names(self) -> set[str]:
        model_name = self.model_name.lower()

        if model_name == "random_forest":
            return {
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "min_samples_split",
                "max_features",
            }

        if model_name == "logistic_regression":
            return {
                "C",
                "penalty",
                "solver",
            }

        if model_name == "catboost":
            return {
                "iterations",
                "depth",
                "learning_rate",
                "l2_leaf_reg",
                "random_strength",
                "bagging_temperature",
            }

        return set()

    def _get_strategy_orders(self, strategy, df: pd.DataFrame) -> StrategyOrdersLike:
        """
        Поддерживает два варианта стратегии:
            1. Новый интерфейс: generate_orders(df)
            2. Старый интерфейс: generate_signal(df)
        """

        if hasattr(strategy, "generate_orders"):
            orders = strategy.generate_orders(df)

            return StrategyOrdersLike(
                entries=orders.entries.astype(bool),
                exits=orders.exits.astype(bool),
                short_entries=orders.short_entries.astype(bool),
                short_exits=orders.short_exits.astype(bool),
            )

        if hasattr(strategy, "generate_signal"):
            signal = strategy.generate_signal(df).astype(int)
            prev_signal = signal.shift(1).fillna(0).astype(int)

            entries = (signal == 1) & (prev_signal != 1)
            exits = (prev_signal == 1) & (signal != 1)

            short_entries = (signal == -1) & (prev_signal != -1)
            short_exits = (prev_signal == -1) & (signal != -1)

            return StrategyOrdersLike(
                entries=entries.astype(bool),
                exits=exits.astype(bool),
                short_entries=short_entries.astype(bool),
                short_exits=short_exits.astype(bool),
            )

        raise ValueError(
            "Strategy must have either generate_orders(df) or generate_signal(df)."
        )

    @staticmethod
    def _run_backtest(
        backtester,
        df: pd.DataFrame,
        strategy,
        use_ml_filter: bool,
    ) -> dict:
        """
        Совместимость с двумя версиями backtester:
            - новая: run(..., ml_filter_mode='confirm', save_plots=False)
            - старая: run(..., use_ml_filter=True)
        """

        try:
            return backtester.run(
                df=df,
                strategy=strategy,
                use_ml_filter=use_ml_filter,
                ml_filter_mode="confirm",
                save_plots=False,
            )
        except TypeError:
            return backtester.run(
                df=df,
                strategy=strategy,
                use_ml_filter=use_ml_filter,
            )

    def _make_window_info(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        skipped: bool,
        best_score: float | None,
        best_config: dict | None,
        train_samples: int,
        allowed_long: int,
        allowed_short: int,
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
            "allowed_long": allowed_long,
            "allowed_short": allowed_short,
            "skipped": skipped,
            "best_score": best_score,
            "best_config": str(best_config),
        }

    def _calculate_metrics(self, df: pd.DataFrame) -> None:
        predicted_df = df[df["is_ml_predicted"] == 1]

        self.metrics_ = {
            "model_name": self.model_name,
            "filter_type": "trading_optimized",
            "train_window": self.train_window,
            "test_window": self.test_window,
            "validation_size": self.validation_size,
            "window_mode": self.window_mode,
            "n_trials": self.n_trials,
            "score_metric": self.score_metric,
            "n_windows": len(self.windows_),
            "predicted_rows": len(predicted_df),
            "allowed_long": int((predicted_df["ml_filter"] == 1).sum()),
            "allowed_short": int((predicted_df["ml_filter"] == -1).sum()),
        }

    def get_metrics(self) -> dict:
        return self.metrics_

    def get_windows(self) -> pd.DataFrame:
        return pd.DataFrame(self.windows_)