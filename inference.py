"""CPU extractive QA inference module using fine-tuned DistilBERT."""
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
# The served model can be switched without code changes: QA_MODEL_NAME=<folder under weights/>.
# The calibrated no-answer threshold is read from <model folder>/metrics.json, so the model
# folder is self-contained; the older logs/<name>/metrics.json location is the fallback.
MODEL_NAME = os.environ.get('QA_MODEL_NAME', 'modelo-final-v3')
MODEL_DIR = BASE_DIR / 'weights' / MODEL_NAME
METRICS_PATH = (MODEL_DIR / 'metrics.json') if (MODEL_DIR / 'metrics.json').exists() \
    else BASE_DIR / 'logs' / MODEL_NAME / 'metrics.json'

# Questions are short in practice; cap them so a huge question can never leave no room for the context.
MAX_QUESTION_TOKENS = 96


class QAInference:
    """CPU inference pipeline for extractive Question Answering."""

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        metrics_path: Path = METRICS_PATH,
        max_length: int = 384,
        stride: int = 128,
        num_threads: Optional[int] = None,
    ):
        self.device = torch.device('cpu')
        threads = num_threads or min(4, max(1, os.cpu_count() or 1))
        torch.set_num_threads(threads)

        self.max_length = max_length
        self.stride = stride

        # Load tokenizer and model strictly on CPU in eval mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(self.device)
        self.model.eval()

        # Load calibrated no-answer threshold from metrics
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        self.no_answer_threshold = float(metrics['no_answer_threshold'])
        self.base_model_id = str(metrics.get('model', 'distilbert/distilbert-base-uncased'))

    def _clip_question(self, question: str) -> str:
        """Keep at most MAX_QUESTION_TOKENS tokens of the question (avoids tokenizer truncation errors)."""
        ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        if len(ids) <= MAX_QUESTION_TOKENS:
            return question
        return self.tokenizer.decode(ids[:MAX_QUESTION_TOKENS])

    def _tokenize_windows(
        self, context: str, question: str
    ) -> Tuple[Dict[str, torch.Tensor], List[List[Tuple[int, int]]], List[List[Optional[int]]]]:
        """Tokenize context and question with sliding windows.

        Returns model input tensors, per-window character offset mappings,
        and per-window sequence_ids to distinguish context tokens.
        """
        question = self._clip_question(question.lstrip())
        encodings = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,  # pad to the longest window, not to max_length: same logits, less CPU work
            return_tensors="pt",
        )

        offset_mapping = encodings.pop("offset_mapping").tolist()
        num_windows = encodings["input_ids"].shape[0]
        sequence_ids = [encodings.sequence_ids(i) for i in range(num_windows)]

        # Keep only tensors expected by the model
        model_inputs = {
            "input_ids": encodings["input_ids"].to(self.device),
            "attention_mask": encodings["attention_mask"].to(self.device),
        }
        return model_inputs, offset_mapping, sequence_ids

    def predict(self, context: str, question: str) -> Dict[str, Any]:
        """Extract best answer span from context or abstain based on calibrated threshold."""
        if not isinstance(context, str):
            raise ValueError("Context must be a string.")
        if len(context) < 1 or len(context) > 5000:
            raise ValueError(f"Context length must be between 1 and 5000 characters (got {len(context)}).")

        if not isinstance(question, str):
            raise ValueError("Question must be a string.")
        if len(question) < 1 or len(question) > 500:
            raise ValueError(f"Question length must be between 1 and 500 characters (got {len(question)}).")

        # Whitespace-only input is valid per the API contract (minLength 1) but has nothing to answer from.
        if not context.strip() or not question.strip():
            return {"answer": None, "start": None, "score": 1.0}

        inputs, offset_mapping, sequence_ids = self._tokenize_windows(context, question)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits.cpu()
            end_logits = outputs.end_logits.cpu()

        num_windows = inputs["input_ids"].shape[0]
        window_candidates = []

        for w in range(num_windows):
            offsets = offset_mapping[w]
            seq = sequence_ids[w]
            sl = start_logits[w]
            el = end_logits[w]

            # Context tokens only
            valid_indices = {
                k for k, sid in enumerate(seq)
                if sid == 1 and offsets[k][1] > offsets[k][0]
            }

            best_span = ("", None)
            best_score = -1e9

            if valid_indices:
                top_s = torch.topk(sl, min(20, sl.shape[0])).indices.tolist()
                top_e = torch.topk(el, min(20, el.shape[0])).indices.tolist()

                for s in top_s:
                    if s not in valid_indices:
                        continue
                    for e in top_e:
                        if e not in valid_indices:
                            continue
                        if e < s or (e - s) > 30:
                            continue

                        score = float(sl[s] + el[e])
                        if score > best_score:
                            best_score = score
                            cs, ce = offsets[s][0], offsets[e][1]
                            best_span = (context[cs:ce], cs)

            null_score = float(sl[0] + el[0])
            margin = best_score - null_score
            window_candidates.append((best_span[0], best_span[1], margin))

        # Select window candidate maximizing the margin (best_score - null_score)
        best_answer, best_start, best_margin = max(window_candidates, key=lambda x: x[2]) if window_candidates else ("", None, -1e9)

        # Abstain if margin does not exceed calibrated threshold
        if best_margin <= self.no_answer_threshold or not best_answer:
            answer = None
            start = None
            diff = self.no_answer_threshold - best_margin
            confidence = 1.0 / (1.0 + math.exp(-diff))
        else:
            answer = best_answer
            start = best_start
            diff = best_margin - self.no_answer_threshold
            confidence = 1.0 / (1.0 + math.exp(-diff))

        return {
            "answer": answer,
            "start": start,
            "score": round(confidence, 4),
        }
