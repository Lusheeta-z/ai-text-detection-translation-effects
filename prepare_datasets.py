from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "raw_data"
OUTPUT_PATH = PROJECT_DIR / "data" / "dataset.csv"

SOURCE_DATASETS = {
    "AI_Human.csv": {
        "slug": "shanegerami/ai-vs-human-text",
        "url": "https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data",
    },
    "balanced_ai_human_prompts.csv": {
        "slug": "navjotkaushal/human-vs-ai-generated-essays",
        "url": "https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays/data",
    },
}


def source_path(filename: str) -> Path:
    root_path = PROJECT_DIR / filename
    raw_path = RAW_DATA_DIR / filename
    if root_path.exists():
        return root_path
    return raw_path


def missing_sources() -> list[str]:
    return [filename for filename in SOURCE_DATASETS if not source_path(filename).exists()]


def print_source_instructions(missing: list[str]) -> None:
    print("\nMissing required source CSVs:")
    for filename in missing:
        details = SOURCE_DATASETS[filename]
        print(f"- {filename}: {details['url']}")
    print(
        "\nThese original Kaggle CSVs are not included in this submission because they are large. "
        "Download them from the links above and place them either in the project root or in raw_data/, "
        "then run this script again."
    )
    print(
        "\nAutomatic download is available if the Kaggle API is installed and configured "
        "with your kaggle.json credentials."
    )


def kaggle_command() -> list[str] | None:
    if shutil.which("kaggle"):
        return ["kaggle"]

    try:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None

    return [sys.executable, "-m", "kaggle"]


def download_sources(missing: list[str]) -> bool:
    command = kaggle_command()
    if command is None:
        print("\nKaggle API was not found. Install it with: pip install kaggle")
        return False

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in missing:
        details = SOURCE_DATASETS[filename]
        print(f"\nDownloading {filename} from Kaggle dataset {details['slug']}...")
        result = subprocess.run(
            [
                *command,
                "datasets",
                "download",
                "-d",
                details["slug"],
                "-p",
                str(RAW_DATA_DIR),
                "--unzip",
            ],
            cwd=PROJECT_DIR,
        )
        if result.returncode != 0:
            print(f"Could not download {filename}. Please download it manually: {details['url']}")
            return False

        if not source_path(filename).exists():
            print(f"Download finished, but {filename} was not found in {RAW_DATA_DIR}.")
            print("Please check the extracted files and rename/place the CSV as needed.")
            return False

    return True


def ensure_sources(auto_download: bool | None) -> None:
    missing = missing_sources()
    if not missing:
        return

    print_source_instructions(missing)
    should_download = auto_download
    if should_download is None and sys.stdin.isatty():
        answer = input("\nTry automatic Kaggle download now? [y/N]: ").strip().lower()
        should_download = answer in {"y", "yes"}

    if should_download:
        if download_sources(missing):
            missing = missing_sources()
            if not missing:
                return
        print_source_instructions(missing_sources())

    raise SystemExit("\nCannot prepare dataset until the missing source CSVs are available.")


def word_count(text) -> int:
    if isinstance(text, str):
        return len(text.split())
    return 0


def long_mask(df: pd.DataFrame, min_words: int = 70) -> pd.Series:
    return df["text"].apply(word_count) >= min_words


def safe_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    replace = len(df) < n
    return df.sample(n=n, random_state=seed, replace=replace)


def build_dataset() -> pd.DataFrame:
    ai_human = pd.read_csv(source_path("AI_Human.csv"))
    balanced_ai_human_prompts = pd.read_csv(source_path("balanced_ai_human_prompts.csv"))

    ai_generated_essays = ai_human[ai_human["generated"] == 1.0]
    ai_human_essays = ai_human[ai_human["generated"] == 0.0]
    balanced_generated_essays = balanced_ai_human_prompts[balanced_ai_human_prompts["generated"] == 1]
    balanced_human_essays = balanced_ai_human_prompts[balanced_ai_human_prompts["generated"] == 0]

    ai_generated_essays = ai_generated_essays[long_mask(ai_generated_essays, min_words=70)]
    ai_human_essays = ai_human_essays[long_mask(ai_human_essays, min_words=70)]
    balanced_generated_essays = balanced_generated_essays[long_mask(balanced_generated_essays, min_words=70)]
    balanced_human_essays = balanced_human_essays[long_mask(balanced_human_essays, min_words=70)]

    generated_sample = pd.concat(
        [
            safe_sample(ai_generated_essays, 375, seed=42),
            safe_sample(balanced_generated_essays, 375, seed=43),
        ],
        ignore_index=True,
    )
    human_sample = pd.concat(
        [
            safe_sample(ai_human_essays, 375, seed=44),
            safe_sample(balanced_human_essays, 375, seed=45),
        ],
        ignore_index=True,
    )

    final_df = pd.concat([generated_sample, human_sample], ignore_index=True)
    final_df = final_df[["text", "generated"]]
    final_df["generated"] = final_df["generated"].astype(int)

    short_rows = final_df[final_df["text"].apply(word_count) < 70]
    if not short_rows.empty:
        print(f"Warning: {len(short_rows)} rows have fewer than 70 words after sampling.")

    final_ai = safe_sample(final_df[final_df["generated"] == 1], 750, seed=99)
    final_human = safe_sample(final_df[final_df["generated"] == 0], 750, seed=100)
    final_df = pd.concat([final_ai, final_human], ignore_index=True)
    return final_df.sample(frac=1.0, random_state=123).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the balanced base dataset from the two Kaggle source CSVs.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true", help="Try to download missing Kaggle CSVs automatically.")
    group.add_argument("--no-download", action="store_true", help="Do not prompt or try automatic download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auto_download = True if args.download else (False if args.no_download else None)
    ensure_sources(auto_download)

    final_df = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Prepared dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
