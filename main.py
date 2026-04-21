from pathlib import Path

from src.data_loader import load_titanic_data
from src.features import prepare_features
from src.models import build_model
from src.submission import create_submission
from src.trainer import cross_validate_model, train_full_model


FEATURE_SETS = [
    "basic",
    "title",
    "family",
    "embarked",
    "title_family",
    "title_family_embarked",
    "advanced",
]
MODEL_NAMES = [
    "random_forest",
    "lightgbm",
    "xgboost",
    "logistic",
]
N_SPLITS = 5
TOP_N_SUBMISSIONS = 7
SUBMISSIONS_DIR = Path("submissions")
FOCUS_FEATURE_SET = "title"
FOCUS_MODEL_NAMES = [
    "random_forest",
    "lightgbm",
    "xgboost",
    "xgboost_tuned",
    "logistic",
]


def compare_models(feature_set, model_names, X, y):
    results = []

    for model_name in model_names:
        try:
            model = build_model(model_name)
            scores = cross_validate_model(model, X, y, n_splits=N_SPLITS)
            mean_accuracy = sum(scores) / len(scores)

            results.append({
                "feature_set": feature_set,
                "model_name": model_name,
                "accuracy": mean_accuracy,
                "scores": scores,
            })

            fold_scores = ", ".join(f"{score:.4f}" for score in scores)
            print(f"  {model_name}: {mean_accuracy:.4f} [{fold_scores}]")
        except Exception as error:
            print(f"  {model_name}: skipped ({error})")

    if not results:
        raise RuntimeError(f"No models trained successfully for {feature_set}.")

    return max(results, key=lambda result: result["accuracy"])


def run_experiment_matrix(train, test):
    results = []

    for feature_set in FEATURE_SETS:
        print(f"\nFeature set: {feature_set}")
        X, y, X_test = prepare_features(train, test, feature_set=feature_set)
        best_result = compare_models(feature_set, MODEL_NAMES, X, y)
        best_result["X"] = X
        best_result["y"] = y
        best_result["X_test"] = X_test
        results.append(best_result)

    return sorted(results, key=lambda result: result["accuracy"], reverse=True)


def print_leaderboard(results):
    print("\nExperiment leaderboard:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['feature_set']} + {result['model_name']}: "
            f"{result['accuracy']:.4f}"
        )


def create_experiment_submission(result, test, rank=None, output_dir=SUBMISSIONS_DIR):
    model = build_model(result["model_name"])
    model = train_full_model(model, result["X"], result["y"])

    rank_prefix = f"{rank:02d}_" if rank is not None else ""
    filename = (
        f"{rank_prefix}{result['feature_set']}_"
        f"{result['model_name']}_"
        f"{result['accuracy']:.4f}.csv"
    )
    output_path = output_dir / filename

    print(
        f"Saving {output_path}: {result['feature_set']} + "
        f"{result['model_name']} ({result['accuracy']:.4f})"
    )

    create_submission(model, result["X_test"], test, output_path=output_path)


def create_top_submissions(results, test, top_n=TOP_N_SUBMISSIONS):
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    print(f"\nSaving top {top_n} submissions:")
    for rank, result in enumerate(results[:top_n], start=1):
        create_experiment_submission(result, test, rank)


def create_focus_submissions(train, test):
    output_dir = SUBMISSIONS_DIR / f"focus_{FOCUS_FEATURE_SET}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving focus submissions for feature set: {FOCUS_FEATURE_SET}")
    X, y, X_test = prepare_features(train, test, feature_set=FOCUS_FEATURE_SET)

    for model_name in FOCUS_MODEL_NAMES:
        try:
            model = build_model(model_name)
            scores = cross_validate_model(model, X, y, n_splits=N_SPLITS)
            mean_accuracy = sum(scores) / len(scores)
            result = {
                "feature_set": FOCUS_FEATURE_SET,
                "model_name": model_name,
                "accuracy": mean_accuracy,
                "scores": scores,
                "X": X,
                "y": y,
                "X_test": X_test,
            }
            create_experiment_submission(result, test, output_dir=output_dir)
        except Exception as error:
            print(f"Skipping {FOCUS_FEATURE_SET} + {model_name}: {error}")


def main():
    train, test = load_titanic_data()

    print("Experiment matrix:")
    results = run_experiment_matrix(train, test)
    print_leaderboard(results)

    create_top_submissions(results, test)
    create_focus_submissions(train, test)


if __name__ == "__main__":
    main()
