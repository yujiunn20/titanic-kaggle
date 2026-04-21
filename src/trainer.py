from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def train_with_validation(model, X, y, test_size=0.2, random_state=42):
    """Train a model and return the fitted model plus validation data."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    return model, X_valid, y_valid


def cross_validate_model(model, X, y, n_splits=5, random_state=42):
    """Evaluate a model with stratified K-fold validation."""
    kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    scores = []

    for train_index, valid_index in kfold.split(X, y):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        preds = fold_model.predict(X_valid)
        scores.append(accuracy_score(y_valid, preds))

    return scores


def train_full_model(model, X, y):
    """Train a model on the full training dataset."""
    model.fit(X, y)
    return model
