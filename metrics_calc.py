from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


DETECTORS = ["perplexity", "detectgpt", "roberta", "xlmr", "binoculars"]
TRANSLATIONS = {
    "original": "ai_original_{}",
    "google_de": "ai_de_en_{}",
    "google_ur": "ai_ur_en_{}",
    "libre_de": "ai_lt_de_en_{}",
    "libre_ur": "ai_lt_ur_en_{}",
}
BINOCULARS_THRESHOLD = 0.9015310749276843


def transform_scores(name: str, scores: pd.Series) -> pd.Series:
    if name in ["perplexity", "binoculars"]:
        return -scores.astype(float)
    return scores.astype(float)


def transform_threshold(name: str, threshold: float) -> float:
    if name in ["perplexity", "binoculars"]:
        return -threshold
    return threshold


def valid_pair_mask(index: pd.Index, *series: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=index)
    for values in series:
        mask &= values.notna()
    return mask


def evaluate(y_true: pd.Series, y_scores: pd.Series, threshold: float) -> dict[str, float]:
    y_pred = (y_scores >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y_true, y_scores),
    }


def find_optimal_threshold(y_true: pd.Series, y_scores: pd.Series) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def mcnemar_test(y_true: pd.Series, score_a: pd.Series, score_b: pd.Series, threshold: float) -> tuple[int, float]:
    pred_a = (score_a >= threshold).astype(int)
    pred_b = (score_b >= threshold).astype(int)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b01 = int(np.sum((correct_a == 0) & (correct_b == 1)))
    b10 = int(np.sum((correct_a == 1) & (correct_b == 0)))
    n = b01 + b10
    statistic = min(b01, b10)
    if n == 0:
        return statistic, 1.0
    tail = sum(comb(n, i) for i in range(statistic + 1)) / (2 ** n)
    return statistic, min(1.0, 2.0 * tail)


def bootstrap_ci(y_true: np.ndarray, y_scores: np.ndarray, n: int = 1000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        scores.append(roc_auc_score(y_true[idx], y_scores[idx]))
    return np.percentile(scores, [2.5, 97.5])


def calculate_metrics(
    input_path: str | Path = "data/dataset_with_scores.csv",
    output_path: str | Path = "metrics_summary_fixed_thresholds.csv",
    label_col: str = "generated",
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    y_true = df[label_col].astype(int)
    results = []

    for detector in DETECTORS:
        base_col = TRANSLATIONS["original"].format(detector)
        if base_col not in df.columns:
            continue

        base_raw = df[base_col]
        base_mask = valid_pair_mask(df.index, base_raw)
        base_scores = transform_scores(detector, base_raw[base_mask])
        y_base = y_true[base_mask]

        try:
            if detector == "binoculars":
                threshold = transform_threshold(detector, BINOCULARS_THRESHOLD)
            else:
                threshold = find_optimal_threshold(y_base, base_scores)
        except Exception:
            threshold = 0.5

        for variant, fmt in TRANSLATIONS.items():
            col = fmt.format(detector)
            if col not in df.columns:
                continue

            raw_scores = df[col]
            mask = valid_pair_mask(df.index, base_raw, raw_scores)
            y_condition = y_true[mask]
            scores = transform_scores(detector, raw_scores[mask])
            paired_base_scores = transform_scores(detector, base_raw[mask])
            paired_base_metrics = evaluate(y_condition, paired_base_scores, threshold)
            metrics = evaluate(y_condition, scores, threshold)
            stat, pval = mcnemar_test(y_condition, paired_base_scores, scores, threshold)
            ci_auc = bootstrap_ci(y_condition.values, scores.values)

            results.append({
                "Detector": detector,
                "Variant": variant,
                "Optimal Threshold": threshold,
                "N": int(mask.sum()),
                **metrics,
                "Delta Accuracy": metrics["Accuracy"] - paired_base_metrics["Accuracy"],
                "Delta AUROC": metrics["AUROC"] - paired_base_metrics["AUROC"],
                "McNemar statistic": stat,
                "McNemar p-value": pval,
                "AUROC 95% CI": ci_auc,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Evaluation complete. Results saved to {output_path}.")
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate detector metrics for translated text experiments.")
    parser.add_argument("--input", default="data/dataset_with_scores.csv", help="Input CSV with detector scores.")
    parser.add_argument("--output", default="metrics_summary_fixed_thresholds.csv", help="Output metrics CSV.")
    parser.add_argument("--label-col", default="generated", help="Binary label column.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calculate_metrics(args.input, args.output, args.label_col)
