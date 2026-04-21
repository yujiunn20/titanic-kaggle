from sklearn.model_selection import train_test_split


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
