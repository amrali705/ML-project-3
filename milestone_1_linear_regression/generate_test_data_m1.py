"""Create a random 10,000-row test CSV from train_data.csv."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "train_data.csv"
OUTPUT_FILE = BASE_DIR / "test_data_m1.csv"
SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"{INPUT_FILE.name} has only {len(df)} rows; need at least {SAMPLE_SIZE}."
        )

    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    sample.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(sample)} random rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
