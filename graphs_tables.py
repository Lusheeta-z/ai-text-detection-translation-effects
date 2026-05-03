from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def normalize_metrics_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Delta Accuracy" not in df.columns:
        df = df.rename(columns={"Î” Accuracy": "Delta Accuracy", "Î” AUROC": "Delta AUROC"})
    df = df.copy()
    df["Variant"] = df["Variant"].str.replace("google_", "Google ").str.replace("libre_", "Libre ").str.replace("original", "Original")
    df["Variant"] = df["Variant"].str.replace("_", " ").str.title()
    df["Detector"] = df["Detector"].str.upper()
    return df


def _ci_bounds(series: pd.Series) -> pd.DataFrame:
    return series.str.strip("[]").str.split(expand=True).astype(float).rename(columns={0: "CI_lower", 1: "CI_upper"})


def generate_graphs(
    metrics_path: str | Path = "metrics_summary_fixed_thresholds.csv",
    output_dir: str | Path = ".",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = normalize_metrics_columns(pd.read_csv(metrics_path))

    original = df[df["Variant"] == "Original"]
    print("\n=== Detector Performance on Original Text ===\n")
    print(original[["Detector", "Accuracy", "F1", "AUROC"]].to_string(index=False))

    delta_acc = df[df["Variant"] != "Original"][["Detector", "Variant", "Delta Accuracy"]].pivot(index="Variant", columns="Detector")
    print("\n=== Delta Accuracy per Language and Detector ===\n")
    print(delta_acc)

    df["System"] = df["Variant"].apply(lambda x: "Google" if "Google" in x else ("Libre" if "Libre" in x else "Original"))
    google_vs_libre = df[df["Variant"] != "Original"].groupby(["System", "Detector"])[["Delta Accuracy", "Delta AUROC"]].mean().reset_index()
    print("\n=== Avg Delta Accuracy and Delta AUROC: Google vs Libre ===\n")
    print(google_vs_libre.to_string(index=False))

    df["Language"] = df["Variant"].apply(lambda x: x.split(" ")[-1] if x != "Original" else "Original")
    print("\n=== Avg Accuracy and AUROC per Language ===\n")
    print(df.groupby("Language")[["Accuracy", "AUROC"]].mean().round(4).to_string())
    print("\n=== Avg Accuracy and AUROC per System ===\n")
    print(df.groupby("System")[["Accuracy", "AUROC"]].mean().round(4).to_string())

    outputs = {
        "lineplot": output_dir / "lineplot_auroc_variants.png",
        "delta_barplot": output_dir / "barplot_delta_accuracy.png",
        "heatmap": output_dir / "heatmap_auroc_scores.png",
        "perplexity_scatter": output_dir / "scatter_perplexity_auroc_shift.png",
        "auroc_ci": output_dir / "auroc_with_ci.png",
    }

    plt.figure(figsize=(10, 6))
    for detector in df["Detector"].unique():
        vals = df[df["Detector"] == detector].sort_values("Variant")
        if len(vals) > 1:
            plt.plot(vals["Variant"], vals["AUROC"], marker="o", label=detector)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("AUROC")
    plt.title("AUROC Across Translation Variants")
    plt.legend(title="Detector")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(outputs["lineplot"])
    plt.close()

    delta_df = df[df["Variant"] != "Original"]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=delta_df, x="Variant", y="Delta Accuracy", hue="Detector")
    plt.title("Delta Accuracy by Detector and Variant")
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(outputs["delta_barplot"])
    plt.close()

    heatmap_df = df.pivot(index="Detector", columns="Variant", values="AUROC")
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Heatmap of AUROC Scores")
    plt.tight_layout()
    plt.savefig(outputs["heatmap"])
    plt.close()

    orig = df[(df["Detector"] == "PERPLEXITY") & (df["Variant"] == "Original")]["AUROC"].values[0]
    scatter_data = df[(df["Detector"] == "PERPLEXITY") & (df["Variant"] != "Original")]
    plt.figure(figsize=(8, 5))
    plt.scatter(scatter_data["Variant"], scatter_data["AUROC"], label="Translated", color="orange")
    plt.axhline(orig, color="blue", linestyle="--", label="Original AUROC")
    plt.xticks(rotation=45, ha="right")
    plt.title("Perplexity Detector: AUROC After Translation")
    plt.ylabel("AUROC")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(outputs["perplexity_scatter"])
    plt.close()

    ci_df = _ci_bounds(df["AUROC 95% CI"])
    df = pd.concat([df, ci_df], axis=1)
    df["CI_low_err"] = df["AUROC"].astype(float) - df["CI_lower"]
    df["CI_high_err"] = df["CI_upper"] - df["AUROC"].astype(float)

    plt.figure(figsize=(12, 6))
    for detector in df["Detector"].unique():
        det_data = df[df["Detector"] == detector]
        plt.errorbar(
            det_data["Variant"],
            det_data["AUROC"],
            yerr=[det_data["CI_low_err"], det_data["CI_high_err"]],
            fmt="o-",
            label=detector,
            capsize=5,
        )
    plt.xlabel("Variant")
    plt.ylabel("AUROC")
    plt.title("AUROC with 95% Confidence Intervals")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(True)
    plt.legend(title="Detector")
    plt.savefig(outputs["auroc_ci"])
    plt.close()

    print(f"\nAll plots saved in {output_dir}.")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate graphs and summary tables from metrics CSV.")
    parser.add_argument("--metrics", default="metrics_summary_fixed_thresholds.csv", help="Metrics CSV path.")
    parser.add_argument("--output-dir", default=".", help="Directory for generated PNG files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_graphs(args.metrics, args.output_dir)
