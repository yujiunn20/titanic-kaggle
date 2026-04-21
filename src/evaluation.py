from sklearn.metrics import accuracy_score


def evaluate_accuracy(model, X_valid, y_valid):
    """Calculate validation accuracy."""
    preds = model.predict(X_valid)
    return accuracy_score(y_valid, preds)
