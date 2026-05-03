# Data

`dataset.csv` is the dataset before translation with just the text content and AI flag.

`dataset_with_scores.csv` is the final scored dataset used for the thesis analysis. It includes the original text, translation columns, round-trip translation columns, single-pass detector scores, and round-trip detector scores.

`mcnemar_flip_summary.csv` is the generated paired prediction flip summary used for McNemar before-vs-after translation analysis.

Expected intermediate files, if reproducing the pipeline from scratch:

- `dataset.csv`: balanced base dataset before translation.
- `dataset_translated.csv`: base dataset after Google/LibreTranslate translation.
- `dataset_with_scores.csv`: final translated dataset with detector scores.
