from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_model(
    model_name: str,
    random_state: int = 42,
):
    model_name = model_name.lower()

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state
                    ),
                ),
            ]
        )

    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "CatBoost is not installed. Run: pip install catboost"
            ) from exc

        return CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.03,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=False,
        )

    raise ValueError(
        f"Unknown model_name='{model_name}'. "
        "Available: random_forest, logistic_regression, catboost"
    )