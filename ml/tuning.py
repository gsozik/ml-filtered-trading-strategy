import json
from pathlib import Path

import optuna
from sklearn.metrics import f1_score


def tune_filter(
    filter_factory,
    train_df,
    target_builder,
    search_space_func,
    n_trials: int = 20,
    val_size: float = 0.2,
    model_dir: str = "models",
    model_name: str = "model",
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    train_df = train_df.copy().sort_index()

    split = int(len(train_df) * (1 - val_size))
    inner_train_df = train_df.iloc[:split].copy()
    val_df = train_df.iloc[split:].copy()

    y_val = target_builder.build(val_df).dropna().astype(int)

    if len(y_val) == 0:
        raise ValueError("Validation target is empty. Check val_size and horizon.")

    def objective(trial):
        params = search_space_func(trial)

        model = filter_factory(params)

        try:
            model.fit(inner_train_df)
            pred = model.predict_filter(val_df).loc[y_val.index].astype(int)
        except Exception as e:
            if verbose:
                print(f"Trial failed: {e}")
            return -1.0

        raw_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        score = raw_f1

        pred_classes = pred.nunique()

        if pred_classes == 1:
            score -= 0.15

        if pred_classes >= 2:
            score += 0.03

        trial.set_user_attr("raw_f1_macro", raw_f1)
        trial.set_user_attr("pred_classes", int(pred_classes))
        trial.set_user_attr(
            "pred_distribution",
            pred.value_counts().sort_index().to_dict(),
        )

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    study.optimize(objective, n_trials=n_trials)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    trials_path = model_dir / f"{model_name}_optuna_trials.csv"
    best_params_path = model_dir / f"{model_name}_best_params.json"

    study.trials_dataframe().to_csv(trials_path, index=False)

    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=4)

    if verbose:
        print("\nOPTUNA BEST PARAMS")
        print("=" * 80)
        print(study.best_params)
        print(f"Best score: {study.best_value:.4f}")

    return study.best_params


def get_direction_search_space(model_name: str, normalizer: str):
    model_name = model_name.lower()

    def search_space(trial):
        params = {}

        if normalizer == "auto":
            params["normalizer"] = trial.suggest_categorical(
                "normalizer",
                ["standard", "robust", "maxabs", "none"],
            )
        else:
            params["normalizer"] = normalizer

        if model_name == "logistic_regression":
            params["model_params"] = {
                "C": trial.suggest_float("C", 0.001, 10.0, log=True),
                "solver": "lbfgs",
                "penalty": "l2",
                "max_iter": 3000,
                "class_weight": "balanced",
            }

        elif model_name == "random_forest":
            params["model_params"] = {
                "n_estimators": trial.suggest_categorical(
                    "n_estimators", [100, 300, 500]
                ),
                "max_depth": trial.suggest_categorical(
                    "max_depth", [3, 5, 8, 12, None]
                ),
                "min_samples_leaf": trial.suggest_categorical(
                    "min_samples_leaf", [5, 10, 20, 50]
                ),
                "min_samples_split": trial.suggest_categorical(
                    "min_samples_split", [10, 20, 50]
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "class_weight": "balanced",
            }

        elif model_name == "catboost":
            params["model_params"] = {
                "iterations": trial.suggest_categorical(
                    "iterations", [100, 300, 500]
                ),
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 1.0, 10.0
                ),
                "loss_function": "MultiClass",
                "auto_class_weights": "Balanced",
                "verbose": False,
            }

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        return params

    return search_space