FEATURE_COLUMNS = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]


def prepare_features(train, test):
    """Apply basic Titanic feature preprocessing and split inputs/targets."""
    train = train.copy()
    test = test.copy()

    train["Age"] = train["Age"].fillna(train["Age"].median())
    test["Age"] = test["Age"].fillna(test["Age"].median())
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

    X = train[FEATURE_COLUMNS]
    y = train["Survived"]
    X_test = test[FEATURE_COLUMNS]

    return X, y, X_test
