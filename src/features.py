import pandas as pd


NUMERIC_FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "FarePerPerson",
    "AgeMissing",
    "FareMissing",
    "HasCabin",
]
CATEGORICAL_FEATURES = ["Embarked", "Title"]
BASIC_FEATURES = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
FEATURE_SETS = {
    "basic": {
        "numeric": BASIC_FEATURES,
        "categorical": [],
        "age_strategy": "median",
    },
    "title": {
        "numeric": BASIC_FEATURES,
        "categorical": ["Title"],
        "age_strategy": "group",
    },
    "family": {
        "numeric": BASIC_FEATURES + ["FamilySize", "IsAlone"],
        "categorical": [],
        "age_strategy": "median",
    },
    "embarked": {
        "numeric": BASIC_FEATURES,
        "categorical": ["Embarked"],
        "age_strategy": "median",
    },
    "title_family": {
        "numeric": BASIC_FEATURES + ["FamilySize", "IsAlone"],
        "categorical": ["Title"],
        "age_strategy": "group",
    },
    "title_family_embarked": {
        "numeric": BASIC_FEATURES + ["FamilySize", "IsAlone"],
        "categorical": ["Title", "Embarked"],
        "age_strategy": "group",
    },
    "advanced": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "age_strategy": "group",
    },
}


def extract_title(data):
    """Extract and normalize passenger titles from Name."""
    titles = data["Name"].str.extract(r",\s*([^.]*)\.", expand=False)

    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    titles = titles.replace(title_mapping)

    common_titles = {"Mr", "Mrs", "Miss", "Master"}
    return titles.where(titles.isin(common_titles), "Rare")


def add_family_features(data):
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    return data


def add_cabin_features(data):
    data["HasCabin"] = data["Cabin"].notna().astype(int)
    return data


def fill_age_by_group(data, group_medians, fallback_age):
    age_values = data["Age"].copy()
    missing_age = age_values.isna()

    def get_group_age(row):
        group_age = group_medians.get(
            (row["Sex"], row["Pclass"], row["Title"]),
            fallback_age,
        )
        if pd.isna(group_age):
            return fallback_age
        return group_age

    grouped_age = data.loc[missing_age].apply(
        get_group_age,
        axis=1,
    )
    age_values.loc[missing_age] = grouped_age
    return age_values


def engineer_features(data):
    data = data.copy()

    data["Title"] = extract_title(data)
    data = add_family_features(data)
    data = add_cabin_features(data)

    data["AgeMissing"] = data["Age"].isna().astype(int)
    data["FareMissing"] = data["Fare"].isna().astype(int)
    data["Embarked"] = data["Embarked"].fillna("S")

    return data


def encode_features(train, test, categorical_features):
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    if categorical_features:
        combined = pd.get_dummies(
            combined,
            columns=categorical_features,
            drop_first=False,
            dtype=int,
        )

    X = combined.iloc[: len(train)].copy()
    X_test = combined.iloc[len(train) :].copy()

    return X, X_test


def prepare_basic_features(train, test):
    """Create the original compact baseline feature set."""
    train = train.copy()
    test = test.copy()

    train["Age"] = train["Age"].fillna(train["Age"].median())
    test["Age"] = test["Age"].fillna(test["Age"].median())
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

    X = train[BASIC_FEATURES]
    y = train["Survived"]
    X_test = test[BASIC_FEATURES]

    return X, y, X_test


def prepare_advanced_features(train, test):
    """Create Titanic model features and split inputs/targets."""
    train = engineer_features(train)
    test = engineer_features(test)

    fallback_age = train["Age"].median()
    age_group_medians = train.groupby(["Sex", "Pclass", "Title"])["Age"].median()

    train["Age"] = fill_age_by_group(train, age_group_medians, fallback_age)
    test["Age"] = fill_age_by_group(test, age_group_medians, fallback_age)

    fallback_fare = train["Fare"].median()
    train["Fare"] = train["Fare"].fillna(fallback_fare)
    test["Fare"] = test["Fare"].fillna(fallback_fare)

    train["FarePerPerson"] = train["Fare"] / train["FamilySize"]
    test["FarePerPerson"] = test["Fare"] / test["FamilySize"]

    train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

    y = train["Survived"]
    train_features = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    test_features = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X, X_test = encode_features(train_features, test_features, CATEGORICAL_FEATURES)

    return X, y, X_test


def prepare_features(train, test, feature_set="basic"):
    """Create Titanic model features by feature set name."""
    if feature_set not in FEATURE_SETS:
        available_sets = ", ".join(FEATURE_SETS)
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. Available sets: {available_sets}"
        )

    if feature_set == "basic":
        return prepare_basic_features(train, test)

    config = FEATURE_SETS[feature_set]
    train = engineer_features(train)
    test = engineer_features(test)

    if config["age_strategy"] == "group":
        fallback_age = train["Age"].median()
        age_group_medians = train.groupby(["Sex", "Pclass", "Title"])["Age"].median()
        train["Age"] = fill_age_by_group(train, age_group_medians, fallback_age)
        test["Age"] = fill_age_by_group(test, age_group_medians, fallback_age)
    else:
        train["Age"] = train["Age"].fillna(train["Age"].median())
        test["Age"] = test["Age"].fillna(test["Age"].median())

    fallback_fare = train["Fare"].median()
    train["Fare"] = train["Fare"].fillna(fallback_fare)
    test["Fare"] = test["Fare"].fillna(fallback_fare)

    if "FarePerPerson" in config["numeric"]:
        train["FarePerPerson"] = train["Fare"] / train["FamilySize"]
        test["FarePerPerson"] = test["Fare"] / test["FamilySize"]

    train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

    selected_columns = config["numeric"] + config["categorical"]
    y = train["Survived"]
    train_features = train[selected_columns]
    test_features = test[selected_columns]
    X, X_test = encode_features(
        train_features,
        test_features,
        config["categorical"],
    )

    return X, y, X_test
