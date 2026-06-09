import warnings

import optuna
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score

from ml.models import create_model


def suggest_params(
    trial: optuna.Trial,
    model_name: str,
) -> dict:
    model_name = model_name.lower()

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
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        }

    raise ValueError(f"Unknown model_name='{model_name}'")


def tune_model_params(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_trials: int = 20,
    validation_size: float = 0.25,
    metric: str = "f1_macro",
) -> dict:
    """
    Подбор гиперпараметров без нарушения временной структуры.

    ВАЖНО:
    Данные НЕ перемешиваются.
    Последняя часть train-окна используется как validation.
    Optuna не видит test-окно walk-forward.
    """

    if len(X) < 100:
        return {}

    split_idx = int(len(X) * (1 - validation_size))

    if split_idx <= 0 or split_idx >= len(X):
        return {}

    X_inner_train = X.iloc[:split_idx]
    y_inner_train = y.iloc[:split_idx]

    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]

    if y_inner_train.nunique() < 2 or y_val.nunique() < 2:
        return {}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)

        model = create_model(
            model_name=model_name,
            random_state=random_state,
            params=params,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_inner_train, y_inner_train)

        y_pred = model.predict(X_val)

        if metric == "accuracy":
            return accuracy_score(y_val, y_pred)

        if metric == "f1_macro":
            return f1_score(y_val, y_pred, average="macro", zero_division=0)

        raise ValueError("metric must be 'accuracy' or 'f1_macro'")

    sampler = optuna.samplers.TPESampler(seed=random_state)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        n_jobs=1,
    )

    return study.best_params