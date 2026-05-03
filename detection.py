from __future__ import annotations

import os
import time
import pandas as pd
import atexit, signal, sys
import time
from typing import Optional, List, Dict
import numpy as np
import torch
from torch import nn
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    AutoTokenizer, AutoModelForSequenceClassification
)
import random, torch
from contextlib import nullcontext

# Random seeds for reproducibility
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# CUDA settings
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Automatic mixed precision context manager
def amp_ctx():
    if torch.cuda.is_available():
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dt)
    return nullcontext()

# Chunking
def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

# Device selection
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Safe import of Binoculars - optional dependency
def _safe_import_binoculars():
    try:
        import binoculars
        return binoculars
    except Exception:
        return None

# raw PPL score: lower often means more AI-like
class PerplexityScorer:
    # Initialization: GPT2 model and tokenizer
    def __init__(self, model_name: str = "gpt2", device: Optional[torch.device] = None, max_length: int = 256):
        self.device = device or _device()
        self.max_length = max_length
        self.tok = GPT2TokenizerFast.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

    # Single text scoring
    @torch.no_grad()
    def score(self, text: str) -> float:
        return self.score_many([text])[0]

    # Batch text scoring
    @torch.no_grad()
    def score_many(self, texts: List[str], batch_size: int = 32) -> List[float]:
        print("Scoring perplexity for", len(texts), "texts...")
        out: List[float] = []
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Iterate over batches
        for i in range(0, len(texts), batch_size):
            # Process batch
            batch = texts[i:i+batch_size]
            print(f"  Processing batch {i//batch_size + 1} / {(len(texts)-1)//batch_size + 1}...")
            # Scoring
            with torch.cuda.amp.autocast():
                # Encoding
                enc = self.tok(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                ).to(self.device)

                # Model forward pass
                logits = self.model(**enc).logits
                
                # Shift logits and labels for loss calculation
                logits = logits[:, :-1, :].contiguous()
                labels = enc["input_ids"][:, 1:].contiguous()
                attn   = enc["attention_mask"][:, 1:].contiguous()

                # Loss
                vocab = logits.size(-1)
                nll = loss_fct(logits.view(-1, vocab), labels.view(-1)).view(labels.size())
                
                # Perplexity calculation
                token_sums = (nll * attn).sum(dim=1)
                token_counts = attn.sum(dim=1).clamp(min=1)
                seq_loss = token_sums / token_counts
                seq_ppl = torch.exp(seq_loss).tolist()
                # Collect results
                out.extend([float(p) for p in seq_ppl])

        print(f"[DONE] Scored perplexity for", len(texts), f"texts: {out}")
        return out


# higher means more AI-like
class DetectGPTScorer:
    # Initialization: base LM (GPT2) and perturber (t5 Large) models
    def __init__(
        self,
        base_lm: str = "gpt2",
        perturber: str = "t5-large",
        device: Optional[torch.device] = None,
        max_length: int = 128,
        num_perturbations: int = 32,
        mask_ratio: float = 0.30,
        batch_perturb: int = 16,
        max_new_tokens: int = 64,
    ):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.device = device or _device()
        self.max_length = max_length
        self.num_perturbations = num_perturbations
        self.mask_ratio = mask_ratio
        self.batch_perturb = batch_perturb
        self.max_new_tokens = max_new_tokens

        self.base_tok = GPT2TokenizerFast.from_pretrained(base_lm)
        self.base_tok.pad_token = self.base_tok.eos_token
        self.base_lm = GPT2LMHeadModel.from_pretrained(base_lm, torch_dtype=torch.float16).to(self.device).eval()

        self.t5_tok = T5Tokenizer.from_pretrained(perturber)
        self.t5 = T5ForConditionalGeneration.from_pretrained(perturber, torch_dtype=torch.float16).to(self.device).eval()

    # Log probability calculation
    @torch.no_grad()
    def _logprob(self, texts: List[str]) -> List[float]:
        # Calculate log probabilities for a list of texts

        with amp_ctx():

            # Encoding
            enc = self.base_tok(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)

            # Model forward pass
            logits = self.base_lm(**enc).logits
            logits = logits[:, :-1, :].contiguous()
            labels = enc["input_ids"][:, 1:].contiguous()
            attn   = enc["attention_mask"][:, 1:].contiguous()

            # Loss calculation
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            vocab = logits.size(-1)
            nll = loss_fct(logits.view(-1, vocab), labels.view(-1)).view(labels.size())

            # Sequence NLL calculation
            seq_nll = (nll * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
            
            # Convert to list
            return (-seq_nll).tolist()


    # Masking spans in text for perturbation
    def _mask_spans_many(self, text: str, n: int, num_spans: int = 3) -> List[str]:
        # Mask spans in text
        words = text.split()
        if len(words) < 12:
            # Not enough words
            return [text] * n
        out = []
        L = len(words)
        # Generate n masked versions
        for _ in range(n):
            spans = []
            # Generate num_spans random spans
            for s in range(num_spans):
                span_len = max(1, int(0.05 * L))
                start = np.random.randint(0, max(1, L - span_len))
                spans.append((start, start + span_len))
            spans.sort()
            w = []
            cur = 0
            # Build masked text
            for si, (a,b) in enumerate(spans):
                w.extend(words[cur:a])
                w.append(f"<extra_id_{si}>")
                cur = b
            # Add rest
            w.extend(words[cur:])
            # Join
            out.append(" ".join(w))
        return out


    # Infilling using T5 perturber
    @torch.no_grad()
    def _infilling_many(self, masked_list: List[str]) -> List[str]:
        # Infilling
        with amp_ctx():
            # Encoding
            enc = self.t5_tok(masked_list, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True).to(self.device)
            # Generation of outputs from perturber
            out = self.t5.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            return self.t5_tok.batch_decode(out, skip_special_tokens=True)
        
    # Infilling using T5 perturber in chunks
    def _infilling_many_chunked(self, masked_list: List[str], gen_bs: int = 64) -> List[str]:
        outs = []
        for chunk in _chunked(masked_list, gen_bs):
            outs.extend(self._infilling_many(chunk))
        return outs

    # Single text scoring
    def score(self, text: str) -> float:
        print('Scoring DetectGPT for text...')
        # Original log probability
        logp_orig = self._logprob([text])[0]
        perts = []
        remaining = self.num_perturbations
        # Generate perturbations and compute log probabilities
        while remaining > 0:
            # Batch size: min between remaining and batch_perturb
            b = min(self.batch_perturb, remaining)
            # Mask spans
            masked = self._mask_spans_many(text, b)
            # Infill using T5
            pert_texts = self._infilling_many(masked)
            # Log probabilities of perturbed texts
            perts.extend(self._logprob(pert_texts))
            remaining -= b
        
        # Mean of perturbed log probabilities
        mean_pert = float(np.mean(perts)) if perts else logp_orig
        # Standard deviation
        std_pert = float(np.std(perts)) if len(perts) > 1 else 1.0
        # Score
        scored = float((logp_orig - mean_pert) / (std_pert if std_pert > 1e-8 else 1.0))
        print(f'[DONE] Scored DetectGPT for text: {scored}')
        return scored
    
    # Batch text scoring
    def score_many(self,
               texts: List[str],
               texts_bs: int = 8,
               gen_bs: int = 64,
               lp_bs: int = 64
              ) -> List[float]:

        n = len(texts)
        if n == 0: return []

        orig_logp: List[float] = []

        # Original log probabilities
        for tblock in _chunked(texts, texts_bs):
            orig_logp.extend(self._logprob(tblock))

        # Initialize statistics
        counts = np.zeros(n, dtype=np.int32)
        means  = np.zeros(n, dtype=np.float64)
        M2     = np.zeros(n, dtype=np.float64)

        # Update statistics function
        def update_stats(idx, values):
            for v in values:
                counts[idx] += 1
                delta = v - means[idx]
                means[idx] += delta / counts[idx]
                delta2 = v - means[idx]
                M2[idx] += delta2 * delta

        remaining = self.num_perturbations
        while remaining > 0:
            # Batch size: min between remaining and batch_perturb
            b = min(self.batch_perturb, remaining)

            offset = 0
            # Process texts in chunks
            for tblock in _chunked(texts, texts_bs):
                block_size = len(tblock)

                masked_flat = []
                # Mask spans
                for t in tblock:
                    masked_flat.extend(self._mask_spans_many(t, b))

                # Infill
                perts_texts = self._infilling_many_chunked(masked_flat, gen_bs=gen_bs)

                lp_flat: List[float] = []
                # Compute log probabilities
                for lp_chunk in _chunked(perts_texts, lp_bs):
                    lp_flat.extend(self._logprob(lp_chunk))

                # Validate length of log probabilities list
                assert len(lp_flat) == block_size * b

                # Update statistics
                for j in range(block_size):
                    start = j * b
                    end   = start + b
                    idx   = offset + j
                    update_stats(idx, lp_flat[start:end])

                offset += block_size

            remaining -= b

        scores: List[float] = []
        # Compute scores
        for i in range(n):
            # Mean and std deviation of perturbed log probabilities
            mean_pert = means[i] if counts[i] > 0 else orig_logp[i]
            std_pert  = float(np.sqrt(M2[i] / (counts[i]-1))) if counts[i] > 1 else 1.0
            # Score
            z = float((orig_logp[i] - mean_pert) / (std_pert if std_pert > 1e-8 else 1.0))
            scores.append(z)
        return scores

# returns probability of AI
class HFClassifierDetector:
    # Initialization: load HuggingFace classifier model
    def __init__(self, ckpt: str, device: Optional[torch.device] = None, max_length: int = 256, ai_positive_label: Optional[str] = None):
        self.device = device or _device()
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, torch_dtype=torch.float16).to(self.device).eval()

        # Load labels
        self.id2label = getattr(self.model.config, "id2label", None)
        print(f"Loaded classifier ({ckpt}) with labels: {self.id2label}")
        self.ai_index = None
        # Determine AI label index
        if isinstance(self.id2label, dict):
            norm = {int(k): str(v).lower() for k,v in self.id2label.items()}
            for i, name in norm.items():
                if ai_positive_label is not None and name == ai_positive_label.lower():
                    self.ai_index = i
                    break
                if "ai" in name or "gpt" in name or "machine" in name or "chatgpt" in name:
                    self.ai_index = i
                    break
        if self.ai_index is None and self.model.config.num_labels == 2:
            self.ai_index = 1


    # Score text
    @torch.no_grad()
    def score(self, text: str) -> float:
        return self.score_many([text])[0]

    # Batch text scoring
    @torch.no_grad()
    def score_many(self, texts: List[str], batch_size: int = 32) -> List[float]:
        probs = []
        print("Scoring classifier for", len(texts), "texts...")
        # Iterate over batches
        for i in range(0, len(texts), batch_size):
            # Process batch
            batch = texts[i:i+batch_size]
            # Scoring
            with torch.cuda.amp.autocast():
                # Encoding
                enc = self.tok(batch, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True).to(self.device)
                # Model forward pass
                logits = self.model(**enc).logits
                # Probabilities
                probs_all = torch.softmax(logits, dim=1)
                # AI probability
                p = probs_all[:, self.ai_index].tolist()
                # Collect results
                probs.extend([float(x) for x in p])
        print(f"[DONE] Scored classifier for", len(texts), f"texts: {probs}")
        return probs


class BinocularsScorer:
    # Initialization: load Binoculars model
    def __init__(self):
        bino_mod = _safe_import_binoculars()
        if bino_mod is None:
            # Optional dependency not found
            raise RuntimeError("Binoculars package not found. Install from https://github.com/ahans30/Binoculars")
        self._impl = bino_mod.Binoculars()

    # Score text
    def score(self, text: str) -> float:
        print('Scoring Binoculars for text...')
        binscore = float(self._impl.compute_score(text))
        print(f'[DONE] Scored Binoculars for text: {binscore}')
        return binscore
    
    # Batch text scoring
    def score_many(self, texts: List[str]) -> List[float]:
        print("Scoring Binoculars for", len(texts), "texts...")
        scores = self._impl.compute_score(texts)
        print(f"[DONE] Scored Binoculars for", len(texts), f"texts: {scores}")
        return scores


class DetectorsSuite:
    def __init__(
        self,
        
        ppl_model: str = "gpt2",
        ppl_max_length: int = 256,

        detectgpt_base: str = "gpt2",
        detectgpt_perturber: str = "t5-large",
        detectgpt_max_length: int = 128,
        detectgpt_num_perturb: int = 32,
        detectgpt_mask_ratio: float = 0.30,

        roberta_ckpt: Optional[str] = None,
        roberta_ai_positive_label: Optional[str] = None,
        xlmr_ckpt: Optional[str] = None,
        xlmr_ai_positive_label: Optional[str] = None,
        clf_max_length: int = 512,

        enable_binoculars: bool = True,
        enable_perplexity: bool = True,
        enable_detectgpt: bool = True,

      

        device: Optional[torch.device] = None
    ):
        """
        Initializes the DetectorsSuite object.

        Args:
            ppl_model (str): the name of the Perplexity model to use for scoring (default: "gpt2").
            ppl_max_length (int): the maximum length of text to process with the PPL model (default: 256).

            detectgpt_base (str): the name of the base model to use for DetectGPT (default: "gpt2").
            detectgpt_perturber (str): the name of the perturbation model to use for DetectGPT (default: "t5-large").
            detectgpt_max_length (int): the maximum length of text to process with DetectGPT (default: 128).
            detectgpt_num_perturb (int): the number of perturbations to generate for DetectGPT (default: 32).
            detectgpt_mask_ratio (float): the ratio of tokens to mask for DetectGPT (default: 0.30).

            roberta_ckpt (Optional[str]): the name of the checkpoint to use for RoBERTa (default: None).
            roberta_ai_positive_label (Optional[str]): the label to use for the AI class in RoBERTa (default: None).
            xlmr_ckpt (Optional[str]): the name of the checkpoint to use for XLM-R (default: None).
            xlmr_ai_positive_label (Optional[str]): the label to use for the AI class in XLM-R (default: None).
            clf_max_length (int): the maximum length of text to process with the classifiers (default: 512).

            enable_binoculars (bool): whether to enable the Binoculars detector (default: True).
            enable_perplexity (bool): whether to enable the Perplexity detector (default: True).
            enable_detectgpt (bool): whether to enable the DetectGPT detector (default: True).

            device (Optional[torch.device]): the device to use for computations (default: None).
        """
        self.device = device or _device()
        self.enabled_detectors = {"roberta", "xlmr"}

        if enable_perplexity:
            self.enabled_detectors.add("perplexity")
            self.perplexity = PerplexityScorer(model_name=ppl_model, device=self.device, max_length=ppl_max_length)

        if enable_detectgpt:
            self.enabled_detectors.add("detectgpt")
            self.detectgpt = DetectGPTScorer(
                base_lm=detectgpt_base,
                perturber=detectgpt_perturber,
                device=self.device,
                max_length=detectgpt_max_length,
                num_perturbations=detectgpt_num_perturb,
                mask_ratio=detectgpt_mask_ratio
            )

        self.roberta = HFClassifierDetector(roberta_ckpt, device=self.device, max_length=clf_max_length, ai_positive_label=roberta_ai_positive_label) if roberta_ckpt else None
        self.xlmr = HFClassifierDetector(xlmr_ckpt, device=self.device, max_length=clf_max_length, ai_positive_label=xlmr_ai_positive_label) if xlmr_ckpt else None
        
        self.binoculars = None
        if enable_binoculars:
            bino_mod = _safe_import_binoculars()
            if bino_mod is not None:
                self.enabled_detectors.add("binoculars")
                self.binoculars = BinocularsScorer()
            else:
                print("[Binoculars] package not found; skipping. Install from https://github.com/ahans30/Binoculars")

    def score_batch_light(self, texts: List[str]) -> Dict[str, List[Optional[float]]]:
        """
        Scores a batch of texts using the Perplexity, RoBERTa, XLM-R, and Binoculars detectors.

        Args:
            texts (List[str]): The list of texts to score.

        Returns:
            Dict[str, List[Optional[float]]]: A dictionary containing the scores for each detector.
                The keys are the detector names, and the values are lists of scores for each text.
        """
        out = {
            "perplexity_ppl": [None] * len(texts),
            "roberta_prob_ai": [None] * len(texts),
            "xlmr_prob_ai": [None] * len(texts),
            "binoculars": [None] * len(texts),
        }
        if texts:
            if "perplexity" in self.enabled_detectors:
                print("Detection Started: Perplexity")
                out["perplexity_ppl"] = self.perplexity.score_many(texts)
                print("Detection Completed: Perplexity")
            print("Detection Started: RoBERTa")
            out["roberta_prob_ai"] = self.roberta.score_many(texts) if self.roberta else [None]*len(texts)
            print("Detection Completed: RoBERTa")
            print("Detection Started: XLM-R")
            out["xlmr_prob_ai"] = self.xlmr.score_many(texts) if self.xlmr else [None]*len(texts)
            print("Detection Completed: XLM-R")
            if "binoculars" in self.enabled_detectors:
                print("Detection Started: Binoculars")
                out["binoculars"] = self.binoculars.score_many(texts) if self.binoculars else [None]*len(texts)
                print("Detection Completed: Binoculars")
        return out

    def score_detectgpt_list(self, texts: List[str]) -> List[Optional[float]]:
        """
        Scores a list of texts using the DetectGPT detector.

        Args:
            texts (List[str]): The list of texts to score.

        Returns:
            List[Optional[float]]: A list of scores for each text.
        """
        if "detectgpt" not in self.enabled_detectors:
            return [None] * len(texts)

        print("Detection Started: DetectGPT")

        # Handle empty or invalid texts
        clean_idx = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
        # Score valid texts
        clean_txt = [texts[i] for i in clean_idx]
        batched   = self.detectgpt.score_many(clean_txt, texts_bs=8, gen_bs=64, lp_bs=64)

        # Prepare output with None for invalid texts
        returned = [None]*len(texts)
        # Fill in scores for valid texts
        for i, v in zip(clean_idx, batched):
            returned[i] = v
        print("Detection Completed: DetectGPT")
        return returned
    
    def score_binoculars_list(self, texts: List[str]) -> List[float]:
        """
        Scores a list of texts using the Binoculars detector.

        Args:
            texts (List[str]): The list of texts to score.

        Returns:
            List[float]: A list of scores for each text.
        """
        print("Scoring Binoculars for", len(texts), "texts:" , [f"{t[:10]}..." if isinstance(t, str) else None for t in texts])
        current_time = time.time()
        scores = self.binoculars.score_many(texts) if self.binoculars else [None]*len(texts)
        print(f"[DONE] Scored Binoculars for", len(texts), f"texts in {time.time() - current_time} sec: {scores}")
        return scores

    @torch.no_grad()
    def score_text(self, text: str) -> Dict[str, Optional[float]]:
        """
        Scores a single text using the Perplexity, DetectGPT, RoBERTa, XLM-R, and Binoculars detectors.

        Args:
            text (str): The text to score.

        Returns:
            Dict[str, Optional[float]]: A dictionary containing the scores for each detector.
                The keys are the detector names, and the values are the scores for the text.
        """
        out = {}

        print("Detection Started: Perplexity")
        try:
            out["perplexity_ppl"] = self.perplexity.score(text)
            print(f"Detection Completed: Perplexity ({out['perplexity_ppl']})")
        except Exception as e:
            out["perplexity_ppl"] = None
            print(f"[Perplexity] error: {e}")

        print("Detection Started: DetectGPT")
        try:
            out["detectgpt"] = self.detectgpt.score(text)
            print(f"Detection Completed: DetectGPT ({out['detectgpt']})")
        except Exception as e:
            out["detectgpt"] = None
            print(f"[DetectGPT] error: {e}")

        print("Detection Started: RoBERTa")
        try:
            out["roberta_prob_ai"] = self.roberta.score(text) if self.roberta else None
            print(f"Detection Completed: RoBERTa ({out['roberta_prob_ai']})")
        except Exception as e:
            out["roberta_prob_ai"] = None
            print(f"[RoBERTa] error: {e}")
            
        print("Detection Started: XLM-R")
        try:
            out["xlmr_prob_ai"] = self.xlmr.score(text) if self.xlmr else None
            print(f"Detection Completed: XLM-R ({out['xlmr_prob_ai']})")
        except Exception as e:
            out["xlmr_prob_ai"] = None
            print(f"[XLM-R] error: {e}")

        print("Detection Started: Binoculars")
        try:
            out["binoculars"] = self.binoculars.score(text) if self.binoculars else None
            print(f"Detection Completed: Binoculars ({out['binoculars']})")
        except Exception as e:
            out["binoculars"] = None
            print(f"[Binoculars] error: {e}")

      

        return out

def initialize_detectors(**kwargs) -> DetectorsSuite:
    return DetectorsSuite(**kwargs)

def all_detectors(text: str, suite: DetectorsSuite) -> Dict[str, Optional[float]]:
    return suite.score_text(text)


GOOGLE_SCORE_COLUMNS = {
    "text": "ai_original",
    "gt_de": "ai_score_{}_gt_de",
    "gt_ur": "ai_score_{}_gt_ur",
    "gt_de_en": "ai_de_en",
    "gt_ur_en": "ai_ur_en",
}

LIBRE_SCORE_COLUMNS = {
    "lt_de": "ai_score_{}_lt_de",
    "lt_ur": "ai_score_{}_lt_ur",
    "lt_de_en": "ai_lt_de_en",
    "lt_ur_en": "ai_lt_ur_en",
}


def _required_output_columns() -> list[str]:
    columns = []
    for prefix in ["ai_original", "ai_de_en", "ai_ur_en", "ai_lt_de_en", "ai_lt_ur_en"]:
        columns.extend([
            f"{prefix}_perplexity",
            f"{prefix}_detectgpt",
            f"{prefix}_roberta",
            f"{prefix}_xlmr",
            f"{prefix}_binoculars",
        ])
    for source in ["gt_de", "gt_ur", "lt_de", "lt_ur"]:
        columns.extend([
            f"ai_score_perplexity_{source}",
            f"ai_score_detectgpt_{source}",
            f"ai_score_roberta_{source}",
            f"ai_score_xlmr_{source}",
            f"ai_score_binoculars_{source}",
        ])
    return columns


def _is_missing(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _score_prefix_to_columns(prefix: str) -> dict[str, str]:
    if "{}" in prefix:
        return {
            "perplexity": prefix.format("perplexity"),
            "detectgpt": prefix.format("detectgpt"),
            "roberta": prefix.format("roberta"),
            "xlmr": prefix.format("xlmr"),
            "binoculars": prefix.format("binoculars"),
        }
    return {
        "perplexity": f"{prefix}_perplexity",
        "detectgpt": f"{prefix}_detectgpt",
        "roberta": f"{prefix}_roberta",
        "xlmr": f"{prefix}_xlmr",
        "binoculars": f"{prefix}_binoculars",
    }


def _enabled_detector_names(
    enable_binoculars: bool,
    enable_perplexity: bool,
    enable_detectgpt: bool,
) -> list[str]:
    detectors = ["roberta", "xlmr"]
    if enable_perplexity:
        detectors.append("perplexity")
    if enable_detectgpt:
        detectors.append("detectgpt")
    if enable_binoculars:
        detectors.append("binoculars")
    return detectors


def _score_and_store(
    df: pd.DataFrame,
    suite: DetectorsSuite,
    batch_idx,
    text_col: str,
    prefix: str,
    enabled_detectors: list[str],
) -> None:
    columns = _score_prefix_to_columns(prefix)
    texts = df.loc[batch_idx, text_col].fillna("").astype(str).tolist()
    light_scores = suite.score_batch_light(texts)
    detectgpt_scores = suite.score_detectgpt_list(texts)
    for j, idx in enumerate(batch_idx):
        if "perplexity" in enabled_detectors:
            df.at[idx, columns["perplexity"]] = light_scores["perplexity_ppl"][j]
        if "detectgpt" in enabled_detectors:
            df.at[idx, columns["detectgpt"]] = detectgpt_scores[j]
        if "roberta" in enabled_detectors:
            df.at[idx, columns["roberta"]] = light_scores["roberta_prob_ai"][j]
        if "xlmr" in enabled_detectors:
            df.at[idx, columns["xlmr"]] = light_scores["xlmr_prob_ai"][j]
        if "binoculars" in enabled_detectors:
            df.at[idx, columns["binoculars"]] = light_scores["binoculars"][j]


def run_detection_pipeline(
    input_path: str = "data/dataset_translated.csv",
    output_path: str = "data/dataset_with_scores.csv",
    batch_size: int = 16,
    save_every: int = 10,
    providers: Optional[List[str]] = None,
    enable_binoculars: bool = True,
    enable_perplexity: bool = True,
    enable_detectgpt: bool = True,
) -> pd.DataFrame:
    providers = providers or ["google", "libre"]
    enabled_detectors = _enabled_detector_names(
        enable_binoculars=enable_binoculars,
        enable_perplexity=enable_perplexity,
        enable_detectgpt=enable_detectgpt,
    )
    score_maps = {}
    if "google" in providers:
        score_maps.update(GOOGLE_SCORE_COLUMNS)
    if "libre" in providers:
        score_maps.update(LIBRE_SCORE_COLUMNS)

    if os.path.exists(output_path):
        print(f"Resuming from existing file: {output_path}")
        df = pd.read_csv(output_path)
    else:
        print(f"Starting new scoring process from {input_path}")
        df = pd.read_csv(input_path)
        for col in _required_output_columns():
            if col not in df.columns:
                df[col] = None
        df.to_csv(output_path, index=False)
        df = pd.read_csv(output_path)

    print(f"Loaded {len(df)} rows from {output_path}")

    def find_todo(detectors: list[str]) -> list[int]:
        todo = []
        for i, row in df.iterrows():
            for text_col, prefix in score_maps.items():
                if text_col not in df.columns or _is_missing(row.get(text_col, "")):
                    continue
                columns = _score_prefix_to_columns(prefix)
                if any(_is_missing(row.get(columns[detector], None)) for detector in detectors):
                    todo.append(i)
                    break
        return todo

    todo_idx = find_todo(enabled_detectors)
    if todo_idx and enable_binoculars:
        without_binoculars = [detector for detector in enabled_detectors if detector != "binoculars"]
        if not find_todo(without_binoculars):
            print("Only optional Binoculars scores are missing; skipping Binoculars rerun.")
            return df
    print(f"Total rows to process: {len(todo_idx)}")
    if not todo_idx:
        print("No missing detector scores found; nothing to do.")
        return df

    def save_progress():
        print("\nAuto-saving current progress...")
        df.to_csv(output_path, index=False)

    atexit.register(save_progress)
    signal.signal(signal.SIGINT, lambda s, f: (save_progress(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (save_progress(), sys.exit(0)))

    print("Initializing detectors...")
    start_time = time.time()
    suite = initialize_detectors(
        roberta_ckpt="Hello-SimpleAI/chatgpt-detector-roberta",
        roberta_ai_positive_label="ChatGPT",
        xlmr_ckpt="daalft/xlmr-ai-text-detection",
        xlmr_ai_positive_label="LABEL_1",
        enable_binoculars=enable_binoculars,
        enable_perplexity=enable_perplexity,
        enable_detectgpt=enable_detectgpt,
    )
    print(f"Initialization time: {time.time() - start_time:.2f} seconds")

    for start in range(0, len(todo_idx), batch_size):
        batch_idx = todo_idx[start:start + batch_size]
        for text_col, prefix in score_maps.items():
            if text_col not in df.columns:
                print(f"[WARN] Missing text column skipped: {text_col}")
                continue
            _score_and_store(df, suite, batch_idx, text_col, prefix, enabled_detectors)

        if (start + batch_size) % save_every == 0 or (start + batch_size) >= len(todo_idx):
            df.to_csv(output_path, index=False)
            print(f"Saved progress at row {start + len(batch_idx)}/{len(todo_idx)}")

    df.to_csv(output_path, index=False)
    print(f"All done! Results saved to {output_path}")
    return df


if __name__ == "__main__":
    run_detection_pipeline()
