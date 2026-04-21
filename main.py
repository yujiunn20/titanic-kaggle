from src.data_loader import load_titanic_data
from src.evaluation import evaluate_accuracy
from src.features import prepare_features
from src.models import build_random_forest_model
from src.submission import create_submission
from src.trainer import train_with_validation


def main():
    train, test = load_titanic_data()
    X, y, X_test = prepare_features(train, test)

    model = build_random_forest_model()
    model, X_valid, y_valid = train_with_validation(model, X, y)

    acc = evaluate_accuracy(model, X_valid, y_valid)
    print("Validation Accuracy:", acc)

    create_submission(model, X_test, test)
    print("submission.csv saved")


if __name__ == "__main__":
    main()
