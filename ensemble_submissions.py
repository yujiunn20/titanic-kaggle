from pathlib import Path
import csv


ENSEMBLES = {
    "top3_public": [
        "submissions/02_title_xgboost_0.8440.csv",
        "submissions/01_title_family_xgboost_0.8496.csv",
        "submissions/focus_title/title_xgboost_tuned_0.8372.csv",
    ],
    "title_xgb_family_basic": [
        "submissions/02_title_xgboost_0.8440.csv",
        "submissions/01_title_family_xgboost_0.8496.csv",
        "submissions/04_basic_xgboost_0.8428.csv",
    ],
    "xgb_feature_sweep": [
        "submissions/02_title_xgboost_0.8440.csv",
        "submissions/01_title_family_xgboost_0.8496.csv",
        "submissions/04_basic_xgboost_0.8428.csv",
        "submissions/05_family_xgboost_0.8395.csv",
        "submissions/03_title_family_embarked_xgboost_0.8429.csv",
    ],
}
OUTPUT_DIR = Path("submissions/ensembles")


def majority_vote(paths):
    votes_by_passenger = {}

    for path in paths:
        with open(path, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                passenger_id = row["PassengerId"]
                votes_by_passenger.setdefault(passenger_id, []).append(
                    int(row["Survived"])
                )

    rows = []
    for passenger_id, votes in votes_by_passenger.items():
        survived = int(sum(votes) >= (len(votes) / 2))
        rows.append({
            "PassengerId": passenger_id,
            "Survived": survived,
        })

    return rows


def save_ensembles():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, paths in ENSEMBLES.items():
        missing_paths = [path for path in paths if not Path(path).exists()]
        if missing_paths:
            print(f"Skipping {name}; missing files: {missing_paths}")
            continue

        submission = majority_vote(paths)
        output_path = OUTPUT_DIR / f"{name}.csv"
        with open(output_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["PassengerId", "Survived"])
            writer.writeheader()
            writer.writerows(submission)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    save_ensembles()
