from sklearn.ensemble import RandomForestClassifier


def build_random_forest_model(random_state=42):
    """Create the baseline Random Forest model."""
    return RandomForestClassifier(random_state=random_state)
