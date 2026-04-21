from src.data_loader import load_titanic_data
from src.features import prepare_features
from src.models import build_model
from src.submission import create_submission
from src.trainer import cross_validate_model, train_full_model


MODEL_NAMES = [
    "random_forest",
    "lightgbm",
    "xgboost",
    "mlp",
    "logistic",
    "voting",
]
N_SPLITS = 5


def compare_models(model_names, X, y):
    results = []

    for model_name in model_names:
        try:
            model = build_model(model_name)
            scores = cross_validate_model(model, X, y, n_splits=N_SPLITS)
            mean_accuracy = sum(scores) / len(scores)

            results.append({
                "name": model_name,
                "accuracy": mean_accuracy,
                "scores": scores,
            })

            fold_scores = ", ".join(f"{score:.4f}" for score in scores)
            print(f"{model_name}: {mean_accuracy:.4f} [{fold_scores}]")
        except Exception as error:
            print(f"{model_name}: skipped ({error})")

    if not results:
        raise RuntimeError("No models were trained successfully.")

    return max(results, key=lambda result: result["accuracy"])


def main():
    train, test = load_titanic_data()
    X, y, X_test = prepare_features(train, test)

    print("Model comparison:")
    best_result = compare_models(MODEL_NAMES, X, y)
    best_model = build_model(best_result["name"])
    best_model = train_full_model(best_model, X, y)

    print(
        f"Best model: {best_result['name']} "
        f"({best_result['accuracy']:.4f})"
    )

    create_submission(best_model, X_test, test)
    print("submission.csv saved with best model")


if __name__ == "__main__":
    main()
