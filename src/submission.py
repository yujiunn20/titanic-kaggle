import pandas as pd


def create_submission(model, X_test, test, output_path="submission.csv"):
    """Generate and save a Kaggle submission file."""
    test_preds = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_preds,
    })

    submission.to_csv(output_path, index=False)
    return submission
