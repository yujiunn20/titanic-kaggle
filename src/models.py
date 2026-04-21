from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_random_forest_model(random_state=42):
    """Create the baseline Random Forest model."""
    return RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
    )


def build_lightgbm_model(random_state=42):
    """Create a LightGBM classifier."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    )


def build_xgboost_model(random_state=42):
    """Create an XGBoost classifier."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
    )


def build_mlp_model(random_state=42):
    """Create an MLP classifier with feature scaling."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_logistic_model(random_state=42):
    """Create a Logistic Regression classifier with feature scaling."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_voting_model(random_state=42):
    """Create a hard-voting ensemble from all baseline models."""
    return VotingClassifier(
        estimators=[
            ("random_forest", build_random_forest_model(random_state=random_state)),
            ("lightgbm", build_lightgbm_model(random_state=random_state)),
            ("xgboost", build_xgboost_model(random_state=random_state)),
            ("mlp", build_mlp_model(random_state=random_state)),
            ("logistic", build_logistic_model(random_state=random_state)),
        ],
        voting="hard",
    )


MODEL_BUILDERS = {
    "random_forest": build_random_forest_model,
    "randomforest": build_random_forest_model,
    "rf": build_random_forest_model,
    "lightgbm": build_lightgbm_model,
    "lgbm": build_lightgbm_model,
    "xgboost": build_xgboost_model,
    "xgb": build_xgboost_model,
    "mlp": build_mlp_model,
    "logistic": build_logistic_model,
    "logistic_regression": build_logistic_model,
    "voting": build_voting_model,
    "ensemble": build_voting_model,
    "majority_vote": build_voting_model,
}


def build_model(model_name="random_forest", random_state=42):
    """Create a model by name."""
    normalized_name = model_name.lower().strip()

    if normalized_name not in MODEL_BUILDERS:
        available_models = ", ".join(MODEL_BUILDERS)
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models}"
        )

    return MODEL_BUILDERS[normalized_name](random_state=random_state)
