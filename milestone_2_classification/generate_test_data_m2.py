"""Create a full-feature 10,000-row classification test CSV."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
INPUT_FILE = PROJECT_DIR / "milestone_1_linear_regression" / "train_data.csv"
OUTPUT_FILE = BASE_DIR / "test_data_m2.csv"
SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def popularity_to_level(popularity: pd.Series) -> pd.Series:
    return pd.cut(
        popularity,
        bins=[-float("inf"), 0.9999, 1.9999, 9.9999, float("inf")],
        labels=["Very Low", "Low", "Medium", "High"],
    ).astype(str)


def main() -> None:
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"{INPUT_FILE.name} has only {len(df)} rows; need at least {SAMPLE_SIZE}."
        )

    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()
    popularity_index = sample.columns.get_loc("popularity")
    popularity_level = popularity_to_level(sample["popularity"])

    sample = sample.drop(columns=["popularity"])
    sample.insert(popularity_index, "popularityLevel", popularity_level)
    sample.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(sample)} rows to {OUTPUT_FILE}")
    print(sample["popularityLevel"].value_counts().to_string())


if __name__ == "__main__":
    main()
