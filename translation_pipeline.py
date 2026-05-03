from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import requests


GOOGLE_COLUMNS = ["gt_de", "gt_de_en", "gt_ur", "gt_ur_en"]
LIBRE_COLUMNS = ["lt_de", "lt_de_en", "lt_ur", "lt_ur_en"]


def is_missing(value) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = ""


def google_cloud_translator(project_id: str | None = None, location: str = "global") -> Callable[[list[str], str, str], list[str]]:
    from google.api_core import retry as g_retry
    from google.api_core.exceptions import DeadlineExceeded, ServiceUnavailable
    from google.cloud import translate_v3 as translate

    project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    if not project_id:
        raise ValueError("Google Cloud translation requires --google-project-id or GOOGLE_CLOUD_PROJECT_ID.")

    client = translate.TranslationServiceClient()
    parent = f"projects/{project_id}/locations/{location}"
    retry_policy = g_retry.Retry(
        predicate=g_retry.if_exception_type(ServiceUnavailable, DeadlineExceeded),
        initial=1.0,
        maximum=20.0,
        multiplier=2.0,
        deadline=120.0,
    )

    def translate_batch(texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        if not texts:
            return []
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": texts,
                "mime_type": "text/plain",
                "source_language_code": source_lang,
                "target_language_code": target_lang,
            },
            retry=retry_policy,
            timeout=120,
        )
        return [item.translated_text for item in response.translations]

    return translate_batch


def googletrans_translator() -> Callable[[list[str], str, str], list[str]]:
    from googletrans import Translator

    translator = Translator()

    def translate_batch(texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        outputs = []
        for text in texts:
            result = translator.translate(text, src=source_lang, dest=target_lang)
            outputs.append(result.text or "")
        return outputs

    return translate_batch


def libre_translator(endpoint: str) -> Callable[[list[str], str, str], list[str]]:
    def translate_batch(texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        outputs = []
        for text in texts:
            response = requests.post(
                endpoint,
                data={"q": text, "source": source_lang, "target": target_lang, "format": "text"},
                timeout=120,
            )
            response.raise_for_status()
            outputs.append(str(response.json().get("translatedText", "")))
        return outputs

    return translate_batch


def safe_translate_batch(
    translate_batch: Callable[[list[str], str, str], list[str]],
    texts: list[str],
    source_lang: str,
    target_lang: str,
    max_attempts: int = 3,
) -> list[str]:
    for attempt in range(1, max_attempts + 1):
        try:
            outputs = translate_batch(texts, source_lang, target_lang)
            return [output if output and output.strip() else "" for output in outputs]
        except Exception as exc:
            if attempt >= max_attempts:
                print(f"[ERROR] Failed {source_lang}->{target_lang} for {len(texts)} rows: {exc}")
                return [""] * len(texts)
            sleep_s = min(2 ** attempt, 10)
            print(f"[WARN] {source_lang}->{target_lang} failed, retrying in {sleep_s}s: {exc}")
            time.sleep(sleep_s)
    return [""] * len(texts)


def translate_stage(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    source_lang: str,
    target_lang: str,
    translate_batch: Callable[[list[str], str, str], list[str]],
    output_path: Path,
    batch_size: int,
    save_every: int,
) -> None:
    pending = []
    for idx, row in df.iterrows():
        if source_col not in df.columns or target_col not in df.columns:
            continue
        if is_missing(row.get(source_col)) or not is_missing(row.get(target_col)):
            continue
        pending.append((idx, str(row[source_col])))

    if not pending:
        print(f"[INFO] Nothing to translate for {target_col}.")
        return

    print(f"[INFO] Translating {len(pending)} rows for {target_col} ({source_lang}->{target_lang})")
    processed = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        idxs = [idx for idx, _ in batch]
        texts = [text for _, text in batch]
        outputs = safe_translate_batch(translate_batch, texts, source_lang, target_lang)

        for idx, translated in zip(idxs, outputs):
            df.at[idx, target_col] = translated

        processed += len(batch)
        if processed % save_every == 0 or processed >= len(pending):
            df.to_csv(output_path, index=False)
            print(f"[INFO] Saved {target_col}: {processed}/{len(pending)}")


def run_translation_pipeline(
    input_path: str | Path = "data/dataset.csv",
    output_path: str | Path = "data/dataset_translated.csv",
    providers: list[str] | None = None,
    google_backend: str = "cloud",
    google_project_id: str | None = None,
    google_location: str = "global",
    libre_endpoint: str = "https://api.libretranslate.texttechnologylab.org/translate",
    batch_size: int = 20,
    save_every: int = 50,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)
    providers = providers or ["google", "libre"]

    if output_path.exists():
        print(f"[INFO] Resuming from {output_path}")
        df = pd.read_csv(output_path)
    else:
        df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    if "google" in providers:
        ensure_columns(df, GOOGLE_COLUMNS)
        google_translate = (
            google_cloud_translator(google_project_id, google_location)
            if google_backend == "cloud"
            else googletrans_translator()
        )
        translate_stage(df, "text", "gt_de", "en", "de", google_translate, output_path, batch_size, save_every)
        translate_stage(df, "gt_de", "gt_de_en", "de", "en", google_translate, output_path, batch_size, save_every)
        translate_stage(df, "text", "gt_ur", "en", "ur", google_translate, output_path, batch_size, save_every)
        translate_stage(df, "gt_ur", "gt_ur_en", "ur", "en", google_translate, output_path, batch_size, save_every)

    if "libre" in providers:
        ensure_columns(df, LIBRE_COLUMNS)
        libre_translate = libre_translator(libre_endpoint)
        translate_stage(df, "text", "lt_de", "en", "de", libre_translate, output_path, batch_size, save_every)
        translate_stage(df, "lt_de", "lt_de_en", "de", "en", libre_translate, output_path, batch_size, save_every)
        translate_stage(df, "text", "lt_ur", "en", "ur", libre_translate, output_path, batch_size, save_every)
        translate_stage(df, "lt_ur", "lt_ur_en", "ur", "en", libre_translate, output_path, batch_size, save_every)

    df.to_csv(output_path, index=False)
    print(f"[DONE] Translated dataset saved to {output_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate the thesis dataset with Google and/or LibreTranslate.")
    parser.add_argument("--input", default="data/dataset.csv", help="Input CSV path.")
    parser.add_argument("--output", default="data/dataset_translated.csv", help="Output/resume CSV path.")
    parser.add_argument("--providers", nargs="+", choices=["google", "libre"], default=["google", "libre"])
    parser.add_argument("--google-backend", choices=["cloud", "googletrans"], default="cloud")
    parser.add_argument("--google-project-id", default=None)
    parser.add_argument("--google-location", default="global")
    parser.add_argument("--libre-endpoint", default="https://api.libretranslate.texttechnologylab.org/translate")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_translation_pipeline(
        input_path=args.input,
        output_path=args.output,
        providers=args.providers,
        google_backend=args.google_backend,
        google_project_id=args.google_project_id,
        google_location=args.google_location,
        libre_endpoint=args.libre_endpoint,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )
