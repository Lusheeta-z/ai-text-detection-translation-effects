from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


DETECTOR_THRESHOLDS = {
    # direction, threshold, higher_score_means_ai
    "perplexity": ("<=", 20.48153305, False),
    "detectgpt": (">=", 1.647826771, True),
    "roberta": (">=", 0.246669412, True),
    "xlmr": (">=", 0.030704854, True),
    # Matches the fixed-threshold Binoculars results used in the thesis tables.
    "binoculars": ("<=", 1.0, False),
}

CONDITIONS = {
    "Single Google DE": {
        "perplexity": "ai_score_perplexity_gt_de",
        "detectgpt": "ai_score_detectgpt_gt_de",
        "roberta": "ai_score_roberta_gt_de",
        "xlmr": "ai_score_xlmr_gt_de",
        "binoculars": "ai_score_binoculars_gt_de",
    },
    "Round Google DE": {
        "perplexity": "ai_de_en_perplexity",
        "detectgpt": "ai_de_en_detectgpt",
        "roberta": "ai_de_en_roberta",
        "xlmr": "ai_de_en_xlmr",
        "binoculars": "ai_de_en_binoculars",
    },
    "Single Google UR": {
        "perplexity": "ai_score_perplexity_gt_ur",
        "detectgpt": "ai_score_detectgpt_gt_ur",
        "roberta": "ai_score_roberta_gt_ur",
        "xlmr": "ai_score_xlmr_gt_ur",
        "binoculars": "ai_score_binoculars_gt_ur",
    },
    "Round Google UR": {
        "perplexity": "ai_ur_en_perplexity",
        "detectgpt": "ai_ur_en_detectgpt",
        "roberta": "ai_ur_en_roberta",
        "xlmr": "ai_ur_en_xlmr",
        "binoculars": "ai_ur_en_binoculars",
    },
    "Single Libre DE": {
        "perplexity": "ai_score_perplexity_lt_de",
        "detectgpt": "ai_score_detectgpt_lt_de",
        "roberta": "ai_score_roberta_lt_de",
        "xlmr": "ai_score_xlmr_lt_de",
        "binoculars": "ai_score_binoculars_lt_de",
    },
    "Round Libre DE": {
        "perplexity": "ai_lt_de_en_perplexity",
        "detectgpt": "ai_lt_de_en_detectgpt",
        "roberta": "ai_lt_de_en_roberta",
        "xlmr": "ai_lt_de_en_xlmr",
        "binoculars": "ai_lt_de_en_binoculars",
    },
    "Single Libre UR": {
        "perplexity": "ai_score_perplexity_lt_ur",
        "detectgpt": "ai_score_detectgpt_lt_ur",
        "roberta": "ai_score_roberta_lt_ur",
        "xlmr": "ai_score_xlmr_lt_ur",
        "binoculars": "ai_score_binoculars_lt_ur",
    },
    "Round Libre UR": {
        "perplexity": "ai_lt_ur_en_perplexity",
        "detectgpt": "ai_lt_ur_en_detectgpt",
        "roberta": "ai_lt_ur_en_roberta",
        "xlmr": "ai_lt_ur_en_xlmr",
        "binoculars": "ai_lt_ur_en_binoculars",
    },
}


def evaluate_condition(df: pd.DataFrame, label_col: str, detector: str, score_col: str) -> dict[str, float]:
    direction, threshold, higher_score_means_ai = DETECTOR_THRESHOLDS[detector]
    subset = df[[label_col, score_col]].dropna()
    labels = subset[label_col].astype(int)
    scores = subset[score_col].astype(float)

    predictions = scores >= threshold if direction == ">=" else scores <= threshold
    roc_scores = scores if higher_score_means_ai else -scores

    return {
        "n": float(len(subset)),
        "accuracy": accuracy_score(labels, predictions),
        "auroc": roc_auc_score(labels, roc_scores),
    }


def build_summary(df: pd.DataFrame, label_col: str = "generated") -> pd.DataFrame:
    rows = []
    for condition, detector_columns in CONDITIONS.items():
        for detector, score_col in detector_columns.items():
            metrics = evaluate_condition(df, label_col, detector, score_col)
            rows.append({"condition": condition, "detector": detector, **metrics})

    metrics_df = pd.DataFrame(rows)
    summary_rows = []
    for system in ("Google", "Libre"):
        for language in ("DE", "UR"):
            single = metrics_df[metrics_df["condition"] == f"Single {system} {language}"]
            round_trip = metrics_df[metrics_df["condition"] == f"Round {system} {language}"]
            summary_rows.append(
                {
                    "Path": f"{system} {language}",
                    "Single Accuracy": single["accuracy"].mean(),
                    "Round-trip Accuracy": round_trip["accuracy"].mean(),
                    "Single AUROC": single["auroc"].mean(),
                    "Round-trip AUROC": round_trip["auroc"].mean(),
                }
            )

    return pd.DataFrame(summary_rows)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = ["#356f9f", "#c05f42"]

    for ax, metric in zip(axes, ["Accuracy", "AUROC"]):
        x_positions = range(len(summary))
        bar_width = 0.36
        ax.bar(
            [x - bar_width / 2 for x in x_positions],
            summary[f"Single {metric}"],
            bar_width,
            label="Single-pass",
            color=colors[0],
        )
        ax.bar(
            [x + bar_width / 2 for x in x_positions],
            summary[f"Round-trip {metric}"],
            bar_width,
            label="Round-trip",
            color=colors[1],
        )
        ax.set_title(metric)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(summary["Path"], rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean score across detectors")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Single-pass versus round-trip translation impact", y=1.02)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create the single-pass versus round-trip translation impact diagram."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "data" / "dataset_with_scores.csv",
        help="Path to scored dataset CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "figures" / "single_vs_round_translation_impact.png",
        help="Path where the PNG diagram should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    summary = build_summary(df)
    plot_summary(summary, args.output)

    print(summary.round(4).to_string(index=False))
    print(f"\nSaved diagram to: {args.output}")


if __name__ == "__main__":
    main()
