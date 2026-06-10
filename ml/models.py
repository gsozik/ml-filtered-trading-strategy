from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def create_model(
    model_name: str,
    random_state: int = 42,
    params: dict | None = None,
):
    model_name = model_name.lower()
    params = params or {}

    if model_name == "logistic_regression":
        default_params = {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 3000,
            "class_weight": "balanced",
            "random_state": random_state,
        }
        default_params.update(params)

        return LogisticRegression(**default_params)

    if model_name == "random_forest":
        default_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "min_samples_leaf": 20,
            "min_samples_split": 20,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(params)

        return RandomForestClassifier(**default_params)

    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("CatBoost is not installed. Run: pip install catboost") from exc

        default_params = {
            "iterations": 300,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "loss_function": "MultiClass",
            "auto_class_weights": "Balanced",
            "random_seed": random_state,
            "verbose": False,
        }
        default_params.update(params)

        return CatBoostClassifier(**default_params)

    raise ValueError(
        f"Unknown model_name='{model_name}'. "
        "Available: logistic_regression, random_forest, catboost"
    )