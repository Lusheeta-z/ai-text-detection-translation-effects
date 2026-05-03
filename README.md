# Final Year Project - Leila Assim 6790482

This folder contains the runnable code for the thesis experiment on machine translation and AI-generated text detection.

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Google Cloud Translate, set credentials and a project ID:

Windows PowerShell:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
$env:GOOGLE_CLOUD_PROJECT_ID="your-project-id"
```

macOS/Linux:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT_ID="your-project-id"
```

## Main Commands

Translate the base dataset:

```bash
python main.py translate --input data/dataset.csv --output data/dataset_translated.csv --providers google libre
```

Run detectors on translated data:

```bash
python main.py detect --input data/dataset_translated.csv --output data/dataset_with_scores.csv
```

Calculate metrics:

```bash
python main.py metrics --input data/dataset_with_scores.csv --output metrics_summary_fixed_thresholds.csv
```

Generate graphs:

```bash
python main.py graphs --metrics metrics_summary_fixed_thresholds.csv --output-dir figures
```

Generate McNemar prediction flip counts:

```bash
python main.py mcnemar-flips --input data/dataset_with_scores.csv --output data/mcnemar_flip_summary.csv
```

Generate the single-pass versus round-trip figure:

```bash
python main.py single-round --input data/dataset_with_scores.csv --output figures/single_vs_round_translation_impact.png
```

Print average text lengths:

```bash
python main.py avg-lengths --input data/dataset_with_scores.csv
```

## Notes

- The translation and detection scripts are resumable: if the output CSV already exists, the pipeline continues from it.
- The final scored dataset is stored at `data/dataset_with_scores.csv`.
- Missing detector scores are excluded pairwise in metric calculations.
- Binoculars is optional in code. To reproduce the full detector set, install it from its source repository as noted in `requirements.txt`.
