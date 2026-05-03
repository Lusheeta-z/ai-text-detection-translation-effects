from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import pandas as pd


DETECTOR_THRESHOLDS = {
    "perplexity": ("<=", 20.48153305),
    "detectgpt": (">=", 1.647826771),
    "roberta": (">=", 0.246669412),
    "xlmr": (">=", 0.030704854),
    "binoculars": ("<=", 0.9015310749276843),
}

CONDITIONS = {
    "Google DE": {
        "perplexity": "ai_de_en_perplexity",
        "detectgpt": "ai_de_en_detectgpt",
        "roberta": "ai_de_en_roberta",
        "xlmr": "ai_de_en_xlmr",
        "binoculars": "ai_de_en_binoculars",
    },
    "Google UR": {
        "perplexity": "ai_ur_en_perplexity",
        "detectgpt": "ai_ur_en_detectgpt",
        "roberta": "ai_ur_en_roberta",
        "xlmr": "ai_ur_en_xlmr",
        "binoculars": "ai_ur_en_binoculars",
    },
    "Libre DE": {
        "perplexity": "ai_lt_de_en_perplexity",
        "detectgpt": "ai_lt_de_en_detectgpt",
        "roberta": "ai_lt_de_en_roberta",
        "xlmr": "ai_lt_de_en_xlmr",
        "binoculars": "ai_lt_de_en_binoculars",
    },
    "Libre UR": {
        "perplexity": "ai_lt_ur_en_perplexity",
        "detectgpt": "ai_lt_ur_en_detectgpt",
        "roberta": "ai_lt_ur_en_roberta",
        "xlmr": "ai_lt_ur_en_xlmr",
        "binoculars": "ai_lt_ur_en_binoculars",
    },
}


def predict(scores: pd.Series, detector: str) -> pd.Series:
    direction, threshold = DETECTOR_THRESHOLDS[detector]
    scores = scores.astype(float)
    if direction == ">=":
        return (scores >= threshold).astype(int)
    return (scores <= threshold).astype(int)


def exact_mcnemar_pvalue(correct_to_wrong: int, wrong_to_correct: int) -> float:
    n = correct_to_wrong + wrong_to_correct
    statistic = min(correct_to_wrong, wrong_to_correct)
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(statistic + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def build_mcnemar_flip_table(
    input_path: str | Path = "data/dataset_with_scores.csv",
    output_path: str | Path = "data/mcnemar_flip_summary.csv",
    label_col: str = "generated",
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    labels = df[label_col].astype(int)
    rows = []

    for condition, detector_columns in CONDITIONS.items():
        for detector, translated_col in detector_columns.items():
            original_col = f"ai_original_{detector}"
            if original_col not in df.columns or translated_col not in df.columns:
                continue

            mask = df[original_col].notna() & df[translated_col].notna()
            original_pred = predict(df.loc[mask, original_col], detector)
            translated_pred = predict(df.loc[mask, translated_col], detector)
            y = labels[mask]

            original_correct = original_pred.to_numpy() == y.to_numpy()
            translated_correct = translated_pred.to_numpy() == y.to_numpy()

            correct_to_wrong = int(((original_correct == 1) & (translated_correct == 0)).sum())
            wrong_to_correct = int(((original_correct == 0) & (translated_correct == 1)).sum())
            unchanged_correct = int(((original_correct == 1) & (translated_correct == 1)).sum())
            unchanged_wrong = int(((original_correct == 0) & (translated_correct == 0)).sum())
            p_value = exact_mcnemar_pvalue(correct_to_wrong, wrong_to_correct)

            rows.append(
                {
                    "Detector": detector,
                    "Condition": condition,
                    "N": int(mask.sum()),
                    "Correct to Wrong": correct_to_wrong,
                    "Wrong to Correct": wrong_to_correct,
                    "Unchanged Correct": unchanged_correct,
                    "Unchanged Wrong": unchanged_wrong,
                    "McNemar Statistic": min(correct_to_wrong, wrong_to_correct),
                    "p-value": p_value,
                }
            )

    result = pd.DataFrame(rows)
    result = result.sort_values(["Correct to Wrong", "Wrong to Correct"], ascending=[False, True])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"McNemar flip summary saved to {output_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate McNemar before-vs-after translation flip counts.")
    parser.add_argument("--input", default="data/dataset_with_scores.csv", help="Input scored dataset CSV.")
    parser.add_argument("--output", default="data/mcnemar_flip_summary.csv", help="Output CSV path.")
    parser.add_argument("--label-col", default="generated", help="Binary label column.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_mcnemar_flip_table(args.input, args.output, args.label_col)
