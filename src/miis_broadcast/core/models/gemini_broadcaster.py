# src/miis_broadcast/core/models/gemini_broadcaster.py

import os
import re
import logging
import threading
from collections import Counter
from typing import Dict, Any, Generator, List

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from miis_broadcast.core.utils.config import load_app_config, load_system_prompts

load_dotenv()

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_gemini_cfg = load_app_config().get("gemini", {})
_prompts_cfg = load_system_prompts()


# ---------------------------------------------------------------------------
# Lightweight TF-IDF retriever for game context RAG
# ---------------------------------------------------------------------------

class _ContextRetriever:
    """
    In-memory TF-IDF retriever for splitting game context into chunks
    and retrieving the most relevant ones given a visual query.

    No external dependencies beyond numpy.
    """

    _SPLIT_RE = re.compile(r"\n{2,}|(?<=[.!?])\s{2,}")
    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, top_k: int = 3, min_chunk_len: int = 20) -> None:
        self.top_k = top_k
        self.min_chunk_len = min_chunk_len
        self._chunks: List[str] = []
        self._idf: np.ndarray = np.array([])
        self._tf_matrix: np.ndarray = np.array([])   # shape (n_chunks, vocab)
        self._vocab: Dict[str, int] = {}

    # ---- public ----

    def build(self, text: str) -> None:
        """Chunk text and build TF-IDF index. Call once after loading context."""
        raw = self._SPLIT_RE.split(text.strip())
        self._chunks = [c.strip() for c in raw if len(c.strip()) >= self.min_chunk_len]
        if not self._chunks:
            return
        self._fit(self._chunks)
        logging.info("[RAG] Built index: %d chunks, vocab=%d", len(self._chunks), len(self._vocab))

    def retrieve(self, query: str) -> str:
        """Return top-k relevant chunks joined as a single string."""
        if not self._chunks:
            return ""
        if len(self._chunks) <= self.top_k:
            return "\n".join(self._chunks)

        q_vec = self._query_vec(query)
        if q_vec is None:
            return "\n".join(self._chunks[: self.top_k])

        scores = self._tf_matrix.dot(q_vec)  # cosine numerator (vectors are L2-normed)
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return "\n".join(self._chunks[i] for i in sorted(top_idx))

    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    # ---- private ----

    def _tokenize(self, text: str) -> List[str]:
        return self._TOKEN_RE.findall(text.lower())

    def _fit(self, docs: List[str]) -> None:
        tokenized = [self._tokenize(d) for d in docs]
        # Build vocab
        all_terms = {t for toks in tokenized for t in toks}
        self._vocab = {t: i for i, t in enumerate(sorted(all_terms))}
        V = len(self._vocab)
        N = len(docs)

        # TF matrix (raw counts → L2-normed)
        tf = np.zeros((N, V), dtype=np.float32)
        for row, toks in enumerate(tokenized):
            for t, cnt in Counter(toks).items():
                col = self._vocab.get(t)
                if col is not None:
                    tf[row, col] = cnt

        # IDF = log((N+1) / (df+1)) + 1  (sklearn smooth)
        df = (tf > 0).sum(axis=0).astype(np.float32)
        self._idf = np.log((N + 1) / (df + 1)) + 1.0

        tfidf = tf * self._idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._tf_matrix = tfidf / norms  # L2-normed rows

    def _query_vec(self, query: str) -> "np.ndarray | None":
        toks = self._tokenize(query)
        if not toks:
            return None
        V = len(self._vocab)
        vec = np.zeros(V, dtype=np.float32)
        for t, cnt in Counter(toks).items():
            col = self._vocab.get(t)
            if col is not None:
                vec[col] = cnt * self._idf[col]
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            return None
        return vec / norm

_GEMINI_STYLES: Dict[str, str] = _prompts_cfg.get("gemini_broadcaster", {})
_DEFAULT_STYLE: str = "objective"
_SYSTEM_PROMPT: str = (
    _GEMINI_STYLES.get(_DEFAULT_STYLE, "")
    if isinstance(_GEMINI_STYLES, dict)
    else str(_GEMINI_STYLES)
)

_client: "genai.Client | None" = None
_client_lock = threading.Lock()


def set_style(style_key: str) -> None:
    """Switch Gemini broadcaster style at runtime. style_key must match a key in system_prompts.yml."""
    global _SYSTEM_PROMPT
    if not isinstance(_GEMINI_STYLES, dict):
        logging.warning("[GeminiBroadcaster] gemini_broadcaster in config is not a dict, cannot switch style")
        return
    prompt = _GEMINI_STYLES.get(style_key, "")
    if not prompt:
        logging.warning("[GeminiBroadcaster] Style '%s' not found, keeping current prompt", style_key)
        return
    _SYSTEM_PROMPT = prompt.strip()
    logging.info("[GeminiBroadcaster] Style switched to '%s' (%d chars)", style_key, len(_SYSTEM_PROMPT))
_retriever: _ContextRetriever = _ContextRetriever(top_k=3)
_raw_context: str = ""
_RAG_THRESHOLD: int = int(_gemini_cfg.get("rag_threshold", 600))


def set_game_context(text: str) -> None:
    global _raw_context
    _raw_context = text.strip()
    _retriever.build(_raw_context)
    logging.info("[GeminiBroadcaster] Game context set (%d chars, RAG=%s)",
                 len(_raw_context), len(_raw_context) > _RAG_THRESHOLD)


def clear_game_context() -> None:
    global _raw_context
    _raw_context = ""
    _retriever.build("")
    logging.info("[GeminiBroadcaster] Game context cleared")


def load_game_context_file(path: str) -> str:
    """Load game context from a text file. Returns the loaded text."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    set_game_context(text)
    return _raw_context


def _get_context_for_query(visual: str) -> str:
    """Return the context string to inject into the prompt.

    Short contexts (<= _RAG_THRESHOLD chars) are used verbatim.
    Longer ones go through the TF-IDF retriever to pick the most
    relevant chunks so we don't bloat the prompt with irrelevant info.
    """
    if not _raw_context:
        return ""
    if len(_raw_context) <= _RAG_THRESHOLD or _retriever.is_empty():
        return _raw_context
    return _retriever.retrieve(visual)


def _get_match_state() -> str:
    """Return current match state string from MatchTracker, or empty string."""
    try:
        from miis_broadcast.core.match_tracker import match_tracker
        return match_tracker.get_state_string()
    except Exception:
        return ""


def _get_client() -> "genai.Client":
    global _client
    with _client_lock:
        if _client is None:
            if not _GEMINI_API_KEY:
                raise RuntimeError("[GeminiBroadcaster] GEMINI_API_KEY not found in environment")
            _client = genai.Client(api_key=_GEMINI_API_KEY)
            logging.info("[GeminiBroadcaster] Gemini client initialized")
    return _client


_MODEL_NAME: str = _gemini_cfg.get("model_name", "gemini-3.1-flash-lite-preview")
_TEMPERATURE: float = float(_gemini_cfg.get("temperature", 0.4))
_MAX_OUTPUT_TOKENS: int = int(_gemini_cfg.get("max_output_tokens", 50))


def _make_generate_config() -> Any:
    return genai_types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        temperature=_TEMPERATURE,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )



def _fallback(event_data: Dict[str, Any]) -> Dict[str, Any]:
    event = event_data.get("event", "unknown_event")
    urgency = int(event_data.get("urgency", 5))
    return {
        "priority": urgency,
        "broadcast_text": event.replace("_", " "),
        "action_label": event,
        "should_speak": urgency <= 3,
    }


def _generate_from_rag(visual: str) -> Dict[str, Any]:
    """Generate broadcast output from RAG context when Gemini output is not relevant.

    This fallback is used when:
    - LiveCC sends possibly hallucinated output (all outputs now pass through)
    - Gemini assigns P5 (no active play) or doesn't provide meaningful output
    - We use RAG to pull relevant game context as the broadcast text instead
    """
    ctx = _get_context_for_query(visual)
    if not ctx:
        return {
            "priority": 5,
            "broadcast_text": "Continuing play.",
            "action_label": "fill_rag",
            "should_speak": False,
        }

    lines = ctx.strip().split("\n")
    broadcast_text = lines[0][:50] if lines else "Continuing play."

    return {
        "priority": 4,
        "broadcast_text": broadcast_text,
        "action_label": "fill_rag",
        "should_speak": False,
    }


class StreamEvent:
    """Emitted incrementally as Gemini streams the response."""
    __slots__ = ("priority", "broadcast_text", "action_label", "should_speak", "complete")

    def __init__(self) -> None:
        self.priority: int | None = None
        self.broadcast_text: str | None = None
        self.action_label: str | None = None
        self.should_speak: bool | None = None
        self.complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "broadcast_text": self.broadcast_text,
            "action_label": self.action_label,
            "should_speak": self.should_speak,
        }


def _extract_text(event_data: Dict[str, Any]) -> str:
    """Extract the plain-text visual description from a LiveCC event dict."""
    return event_data.get("metadata", {}).get("raw") or event_data.get("event", "")


_PLINE_RE = re.compile(r"^P([1-5]):\s*(.+)$")


def stream_gemini(event_data: Dict[str, Any]) -> Generator[StreamEvent, None, None]:
    """
    Stream Gemini response for a LiveCC event dict.

    Output format (2 lines):
      P<1-5>: <broadcast text>   → sets priority + broadcast_text simultaneously
      LABEL: <label>             → sets action_label, complete=True

    Yields StreamEvent instances as fields become available.
    """
    client = _get_client()
    visual = _extract_text(event_data)
    ctx = _get_context_for_query(visual)
    match_state = _get_match_state()

    parts = []
    if ctx:
        parts.append(f"[Game context: {ctx}]")
    if match_state:
        parts.append(f"[Match state: {match_state}]")
    parts.append(visual)
    prompt = "\n".join(parts)

    ev = StreamEvent()
    buffer = ""

    def _parse_line(line: str) -> bool:
        """Parse one line into ev. Returns True if ev was updated."""
        nonlocal ev
        line = line.strip()
        if not line:
            return False

        m = _PLINE_RE.match(line)
        if m and ev.priority is None:
            ev.priority = int(m.group(1))
            ev.should_speak = ev.priority <= 3
            ev.broadcast_text = m.group(2).strip()
            return True

        if line.startswith("LABEL:") and ev.action_label is None:
            ev.action_label = line.split(":", 1)[1].strip()
            ev.complete = True
            return True

        return False

    try:
        for chunk in client.models.generate_content_stream(
            model=_MODEL_NAME,
            contents=prompt,
            config=_make_generate_config(),
        ):
            chunk_text = chunk.text or ""
            buffer += chunk_text

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if _parse_line(line):
                    yield ev

        # Handle any remaining buffer without trailing newline
        if buffer.strip() and _parse_line(buffer):
            yield ev

    except Exception:
        logging.exception("[GeminiBroadcaster] stream_gemini failed")
        raise
    finally:
        if not ev.complete:
            ev.action_label = ev.action_label or "unknown"
            ev.priority = ev.priority or 5
            ev.should_speak = ev.should_speak if ev.should_speak is not None else False
            ev.complete = True


def call_gemini(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Blocking call — collects the full stream and returns a result dict.

    Strategy for handling LiveCC "hallucinations":
    1. All LiveCC outputs are sent to Gemini (no filtering at LiveCC layer)
    2. If Gemini assigns P5 (no active play) or fails to provide meaningful output
    3. Use RAG to substitute: fetch relevant context from game context instead
    """
    ev = StreamEvent()
    try:
        for ev in stream_gemini(event_data):
            pass
        result = ev.to_dict()
        if not result.get("broadcast_text"):
            visual = _extract_text(event_data)
            return _generate_from_rag(visual)
        # If Gemini assigned P5 and it came from a non-empty visual input,
        # consider using RAG as well (optional; currently accept Gemini's judgment)
        if result.get("priority") == 5 and event_data.get("metadata", {}).get("raw"):
            logging.debug("[GeminiBroadcaster] P5 assigned to non-empty input, accepting Gemini judgment")
        return result
    except Exception:
        logging.exception("[GeminiBroadcaster] call_gemini failed")
        visual = _extract_text(event_data)
        return _generate_from_rag(visual)
