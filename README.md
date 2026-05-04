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

Prepare the base dataset from the original Kaggle CSVs:

```bash
python prepare_datasets.py
```

If the source CSVs are missing, the script will show the required Kaggle links and ask whether to try an automatic download. Automatic download requires the Kaggle API and configured `kaggle.json` credentials:

```bash
pip install kaggle
python prepare_datasets.py --download
```

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

- The original Kaggle source CSVs (`AI_Human.csv` and `balanced_ai_human_prompts.csv`) are not included because they are large. Download them from the links printed by `prepare_datasets.py` and place them in the project root or `raw_data/`.
- The dataset contains English, German, and Urdu text. All CSV files are saved using UTF-8 encoding. To view the Urdu text correctly, open the files in an editor that supports UTF-8 and right-to-left scripts, such as Visual Studio Code, Jupyter Notebook, LibreOffice Calc, or a modern terminal with a Unicode font.
- If the file is opened in Microsoft Excel and the Urdu text looks misaligned or displayed in the wrong direction, this is usually a display/font issue and not a problem with the data. The Urdu text is stored correctly in the CSV files. For best results, import the CSV using UTF-8 encoding or view it through Python/Pandas or VS Code.
- The translation and detection scripts are resumable: if the output CSV already exists, the pipeline continues from it.
- Some missing translation or detector scores may occur due to API/model failures.
- The final scored dataset is stored at `data/dataset_with_scores.csv`.
- Missing detector scores are excluded pairwise in metric calculations.
- Binoculars is optional in code. To reproduce the full detector set, install it from its source repository as noted in `requirements.txt`.
