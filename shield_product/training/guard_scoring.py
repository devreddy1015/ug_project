import importlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib import error as urlerror
from urllib import request

import numpy as np

from env_loader import load_env_files

from .guard_taxonomy import CATEGORY_SPECS, category_descriptions, category_keywords


_LOADED_ENV_FILES = tuple(load_env_files(Path(__file__).resolve().parents[1]))


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(value, upper))


def _similarity_to_score(similarity: float, floor: float = 0.14, ceiling: float = 0.80) -> float:
    """Convert cosine similarity into a calibrated 0-100 risk score.

    Raw cosine values from sentence embeddings often sit around 0.1-0.3 for weak matches.
    Mapping directly with (sim + 1) / 2 can overstate weak evidence. This mapping suppresses
    low-similarity noise and keeps high-similarity signals expressive.
    """
    sim = _clamp(float(similarity), -1.0, 1.0)
    if sim <= floor:
        return 0.0
    span = max(ceiling - floor, 1e-6)
    scaled = _clamp((sim - floor) / span, 0.0, 1.0)
    return (scaled**1.35) * 100.0


@dataclass(frozen=True)
class DualScoreResult:
    llm_scores: Dict[str, float]
    vector_scores: Dict[str, float]
    combined_scores: Dict[str, float]
    llm_verdict: str
    used_groq: bool
    used_embeddings: bool
    used_chromadb: bool
    llm_error: str


class VectorEmbeddingScorer:
    def __init__(self) -> None:
        self._descriptions = category_descriptions()
        self._keywords = category_keywords()
        self._category_names = [spec.name for spec in CATEGORY_SPECS]

        self._token_keywords: Dict[str, set[str]] = {}
        self._substring_keywords: Dict[str, Tuple[str, ...]] = {}
        for category in self._category_names:
            token_keywords: set[str] = set()
            substring_keywords = []
            seen_keywords: set[str] = set()

            for raw_keyword in self._keywords.get(category, []):
                keyword = str(raw_keyword).strip().lower()
                if not keyword or keyword in seen_keywords:
                    continue

                seen_keywords.add(keyword)
                if re.fullmatch(r"[a-z0-9_]+", keyword):
                    token_keywords.add(keyword)
                else:
                    substring_keywords.append(keyword)

            self._token_keywords[category] = token_keywords
            self._substring_keywords[category] = tuple(substring_keywords)

        self._sentence_model = None
        self._category_embeddings: Optional[np.ndarray] = None
        self._chroma_collection = None
        self._embedding_similarity_floor = 0.14
        self._embedding_similarity_ceiling = 0.80
        self._keyword_backstop_threshold = 20.0
        self.used_embeddings = False
        self.used_chromadb = False

        sentence_transformers = _optional_import("sentence_transformers")
        chromadb = _optional_import("chromadb")

        if sentence_transformers is not None:
            try:
                self._sentence_model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = self._sentence_model.encode(
                    [self._descriptions[name] for name in self._category_names],
                    normalize_embeddings=True,
                )
                self._category_embeddings = np.asarray(embeddings, dtype=np.float32)
                self.used_embeddings = True
            except Exception:
                self._sentence_model = None
                self._category_embeddings = None
                self.used_embeddings = False

        if chromadb is not None and self._category_embeddings is not None:
            try:
                client = chromadb.Client()
                self._chroma_collection = client.get_or_create_collection("guard_categories")
                if self._chroma_collection.count() == 0:
                    self._chroma_collection.add(
                        ids=self._category_names,
                        documents=[self._descriptions[name] for name in self._category_names],
                        embeddings=self._category_embeddings.tolist(),
                    )
                self.used_chromadb = True
            except Exception:
                self._chroma_collection = None
                self.used_chromadb = False

    def score(self, content_text: str) -> Dict[str, float]:
        keyword_scores = self._keyword_scores(content_text)
        if self._sentence_model is not None and self._category_embeddings is not None:
            embedding_scores = self._embedding_similarity_scores(content_text)
            if embedding_scores:
                return self._blend_embedding_with_keywords(embedding_scores, keyword_scores)
        return keyword_scores

    def _blend_embedding_with_keywords(
        self,
        embedding_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
    ) -> Dict[str, float]:
        blended_scores: Dict[str, float] = {}
        for category in self._category_names:
            emb = float(embedding_scores.get(category, 0.0))
            key = float(keyword_scores.get(category, 0.0))
            blended = 0.82 * emb + 0.18 * key

            # Keep explicit lexical cues from being washed out by embedding smoothness.
            if key >= self._keyword_backstop_threshold:
                blended = max(blended, key * 0.9)

            blended_scores[category] = round(_clamp(blended), 4)
        return blended_scores

    def _embedding_similarity_scores(self, content_text: str) -> Dict[str, float]:
        normalized_text = content_text.strip()
        if not normalized_text:
            return {name: 0.0 for name in self._category_names}

        token_count = len(re.findall(r"[a-z0-9_]+", normalized_text.lower()))
        if token_count < 3:
            return {name: 0.0 for name in self._category_names}

        try:
            query_embedding = self._sentence_model.encode(
                [normalized_text], normalize_embeddings=True
            )[0]
            query = np.asarray(query_embedding, dtype=np.float32)
            similarities = np.dot(self._category_embeddings, query)
            scores = {
                name: round(
                    _clamp(
                        _similarity_to_score(
                            float(sim),
                            floor=self._embedding_similarity_floor,
                            ceiling=self._embedding_similarity_ceiling,
                        )
                    ),
                    4,
                )
                for name, sim in zip(self._category_names, similarities)
            }
            return scores
        except Exception:
            return {}

    def _keyword_scores(self, content_text: str) -> Dict[str, float]:
        lowered = content_text.lower()
        tokens = re.findall(r"[a-z0-9_]+", lowered)
        token_set = set(tokens)
        token_count = max(len(tokens), 1)
        scores: Dict[str, float] = {}

        for category in self._category_names:
            token_keywords = self._token_keywords.get(category, set())
            substring_keywords = self._substring_keywords.get(category, ())

            token_hits = token_set.intersection(token_keywords)
            substring_hit_count = sum(1 for keyword in substring_keywords if keyword in lowered)
            matched_count = len(token_hits) + substring_hit_count

            keyword_total = len(token_keywords) + len(substring_keywords)
            coverage = matched_count / max(keyword_total, 1)
            density = min(matched_count / token_count * 40.0, 1.0)
            score = (0.75 * coverage + 0.25 * density) * 100.0
            scores[category] = round(max(0.0, min(score, 100.0)), 4)

        return scores


class GroqLLMScorer:
    def __init__(self) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.model = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
        self.api_base = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        self._category_names = [spec.name for spec in CATEGORY_SPECS]
        self.last_error = ""

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def score(self, content_text: str) -> Tuple[Dict[str, float], str, bool]:
        self.last_error = ""
        if not self.is_configured() or not content_text.strip():
            if not self.is_configured():
                self.last_error = "missing GROQ_API_KEY"
            elif not content_text.strip():
                self.last_error = "empty content"
            return ({name: 0.0 for name in self._category_names}, "", False)

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Guard, a short-form video evaluator. "
                        "Return strict JSON with keys: category_scores and verdict. "
                        "category_scores must include every provided category and a score 0-100."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(content_text),
                },
            ],
        }

        endpoint = self.api_base.rstrip("/") + "/chat/completions"
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=45) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            content = parsed["choices"][0]["message"]["content"]
        except urlerror.HTTPError as exc:
            message = ""
            try:
                body = exc.read().decode("utf-8", errors="ignore")
                message = body.strip().replace("\n", " ")[:220]
            except Exception:
                message = ""
            self.last_error = f"http {exc.code}: {message}".strip()
            return ({name: 0.0 for name in self._category_names}, "", False)
        except urlerror.URLError as exc:
            self.last_error = f"network error: {exc.reason}"
            return ({name: 0.0 for name in self._category_names}, "", False)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ({name: 0.0 for name in self._category_names}, "", False)

        body = _extract_json_block(content)
        if body is None:
            self.last_error = "model response was not JSON"
            return ({name: 0.0 for name in self._category_names}, "", False)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.last_error = "model response JSON parse failed"
            return ({name: 0.0 for name in self._category_names}, "", False)

        raw_scores = payload.get("category_scores", {})
        scores: Dict[str, float] = {}
        for category in self._category_names:
            value = raw_scores.get(category, 0.0)
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = 0.0
            scores[category] = round(max(0.0, min(number, 100.0)), 4)

        verdict = str(payload.get("verdict", "")).strip()
        return scores, verdict, True

    def _build_prompt(self, content_text: str) -> str:
        categories = "\n".join(f"- {spec.name}: {spec.description}" for spec in CATEGORY_SPECS)
        return (
            "Analyze this reel evidence and score all categories from 0 to 100.\n"
            "Categories:\n"
            f"{categories}\n\n"
            "Evidence:\n"
            f"{content_text}\n\n"
            "Return JSON exactly like: "
            '{"category_scores": {"explicit_content": 0}, "verdict": "..."}'
        )


class DualScoringEngine:
    def __init__(self, llm_weight: float = 0.65, vector_weight: float = 0.35) -> None:
        self._vector = VectorEmbeddingScorer()
        self._llm = GroqLLMScorer()
        self._llm_weight = llm_weight
        self._vector_weight = vector_weight
        self._category_names = [spec.name for spec in CATEGORY_SPECS]

    def score(self, content_text: str) -> DualScoreResult:
        vector_scores = self._vector.score(content_text)
        llm_scores, llm_verdict, used_groq = self._llm.score(content_text)

        combined: Dict[str, float] = {}
        for category in self._category_names:
            vector_value = vector_scores.get(category, 0.0)
            llm_value = llm_scores.get(category, 0.0)
            if used_groq:
                value = self._llm_weight * llm_value + self._vector_weight * vector_value
            else:
                value = vector_value
            combined[category] = round(max(0.0, min(value, 100.0)), 4)

        return DualScoreResult(
            llm_scores=llm_scores,
            vector_scores=vector_scores,
            combined_scores=combined,
            llm_verdict=llm_verdict,
            used_groq=used_groq,
            used_embeddings=self._vector.used_embeddings,
            used_chromadb=self._vector.used_chromadb,
            llm_error=self._llm.last_error,
        )

    def score_vector_only(self, content_text: str) -> Dict[str, float]:
        vector_scores = self._vector.score(content_text)
        scores: Dict[str, float] = {}
        for category in self._category_names:
            scores[category] = round(float(vector_scores.get(category, 0.0)), 4)
        return scores


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return None
