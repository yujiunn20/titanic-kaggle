import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]

train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

X = train[features]
y = train["Survived"]
X_test = test[features]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_valid)
acc = accuracy_score(y_valid, preds)
print("Validation Accuracy:", acc)

test_preds = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved")