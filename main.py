from __future__ import annotations

import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else SCRIPT_DIR / path


def cmd_translate(args: argparse.Namespace) -> None:
    from translation_pipeline import run_translation_pipeline

    run_translation_pipeline(
        input_path=resolve_path(args.input),
        output_path=resolve_path(args.output),
        providers=args.providers,
        google_backend=args.google_backend,
        google_project_id=args.google_project_id,
        google_location=args.google_location,
        libre_endpoint=args.libre_endpoint,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )


def cmd_detect(args: argparse.Namespace) -> None:
    from detection import run_detection_pipeline

    run_detection_pipeline(
        input_path=str(resolve_path(args.input)),
        output_path=str(resolve_path(args.output)),
        batch_size=args.batch_size,
        save_every=args.save_every,
        providers=args.providers,
        enable_binoculars=not args.disable_binoculars,
        enable_perplexity=not args.disable_perplexity,
        enable_detectgpt=not args.disable_detectgpt,
    )


def cmd_metrics(args: argparse.Namespace) -> None:
    from metrics_calc import calculate_metrics

    calculate_metrics(
        input_path=resolve_path(args.input),
        output_path=resolve_path(args.output),
        label_col=args.label_col,
    )


def cmd_mcnemar_flips(args: argparse.Namespace) -> None:
    from mcnemar_flips import build_mcnemar_flip_table

    build_mcnemar_flip_table(
        input_path=resolve_path(args.input),
        output_path=resolve_path(args.output),
        label_col=args.label_col,
    )


def cmd_graphs(args: argparse.Namespace) -> None:
    from graphs_tables import generate_graphs

    generate_graphs(
        metrics_path=resolve_path(args.metrics),
        output_dir=resolve_path(args.output_dir),
    )


def cmd_single_round(args: argparse.Namespace) -> None:
    from single_pass_vs_round_trip import build_summary, plot_summary
    import pandas as pd

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    df = pd.read_csv(input_path)
    summary = build_summary(df, label_col=args.label_col)
    plot_summary(summary, output_path)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved diagram to: {output_path}")


def cmd_avg_lengths(args: argparse.Namespace) -> None:
    import pandas as pd

    input_path = resolve_path(args.input)
    df = pd.read_csv(input_path)
    text_cols = {
        "Original": "text",
        "Google DE->EN": "gt_de_en",
        "Google UR->EN": "gt_ur_en",
        "Libre DE->EN": "lt_de_en",
        "Libre UR->EN": "lt_ur_en",
    }

    missing = [col for col in text_cols.values() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    averages = {
        label: df[col].astype(str).str.split().str.len().mean()
        for label, col in text_cols.items()
    }
    print("Average Lengths:")
    for label, value in averages.items():
        print(f"{label}: {value:.2f}")
    print("\nAverage Length Change vs Original:")
    for label, value in averages.items():
        if label != "Original":
            print(f"{label}: {value - averages['Original']:+.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the FYP thesis pipeline: translation, detection, metrics, McNemar flips, and figures."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    translate = subparsers.add_parser("translate", help="Create/resume Google and LibreTranslate columns.")
    translate.add_argument("--input", default="data/dataset.csv")
    translate.add_argument("--output", default="data/dataset_translated.csv")
    translate.add_argument("--providers", nargs="+", choices=["google", "libre"], default=["google", "libre"])
    translate.add_argument("--google-backend", choices=["cloud", "googletrans"], default="cloud")
    translate.add_argument("--google-project-id", default=None)
    translate.add_argument("--google-location", default="global")
    translate.add_argument("--libre-endpoint", default="https://api.libretranslate.texttechnologylab.org/translate")
    translate.add_argument("--batch-size", type=int, default=20)
    translate.add_argument("--save-every", type=int, default=50)
    translate.set_defaults(func=cmd_translate)

    detect = subparsers.add_parser("detect", help="Run AI-text detectors over translated columns.")
    detect.add_argument("--input", default="data/dataset_translated.csv")
    detect.add_argument("--output", default="data/dataset_with_scores.csv")
    detect.add_argument("--providers", nargs="+", choices=["google", "libre"], default=["google", "libre"])
    detect.add_argument("--batch-size", type=int, default=16)
    detect.add_argument("--save-every", type=int, default=10)
    detect.add_argument("--disable-binoculars", action="store_true")
    detect.add_argument("--disable-perplexity", action="store_true")
    detect.add_argument("--disable-detectgpt", action="store_true")
    detect.set_defaults(func=cmd_detect)

    metrics = subparsers.add_parser("metrics", help="Calculate accuracy, F1, AUROC, CI, and McNemar tests.")
    metrics.add_argument("--input", default="data/dataset_with_scores.csv")
    metrics.add_argument("--output", default="metrics_summary_fixed_thresholds.csv")
    metrics.add_argument("--label-col", default="generated")
    metrics.set_defaults(func=cmd_metrics)

    flips = subparsers.add_parser("mcnemar-flips", help="Generate paired McNemar prediction flip counts.")
    flips.add_argument("--input", default="data/dataset_with_scores.csv")
    flips.add_argument("--output", default="data/mcnemar_flip_summary.csv")
    flips.add_argument("--label-col", default="generated")
    flips.set_defaults(func=cmd_mcnemar_flips)

    graphs = subparsers.add_parser("graphs", help="Generate thesis graphs from metrics CSV.")
    graphs.add_argument("--metrics", default="metrics_summary_fixed_thresholds.csv")
    graphs.add_argument("--output-dir", default="figures")
    graphs.set_defaults(func=cmd_graphs)

    single_round = subparsers.add_parser("single-round", help="Create single-pass vs round-trip summary figure.")
    single_round.add_argument("--input", default="data/dataset_with_scores.csv")
    single_round.add_argument("--output", default="figures/single_vs_round_translation_impact.png")
    single_round.add_argument("--label-col", default="generated")
    single_round.set_defaults(func=cmd_single_round)

    avg_lengths = subparsers.add_parser("avg-lengths", help="Print average source and back-translation lengths.")
    avg_lengths.add_argument("--input", default="data/dataset_with_scores.csv")
    avg_lengths.set_defaults(func=cmd_avg_lengths)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
