# src/miis_broadcast/core/prompts/prompt_manager.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml


@dataclass(frozen=True)
class StyleItem:
    key: str
    label: str
    description: str


class PromptManager:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._raw: Dict[str, Any] = {}
        self._styles: Dict[str, StyleItem] = {}
        self._default_style: str = ""
        self._livecc_query: str = ""
        self._livecc_response_prefix: str = ""
        self.reload()

    def reload(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Prompt config not found: {self.config_path}")

        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self._raw = data

        self._default_style = (data.get("default_style") or "").strip()
        self._livecc_query = (data.get("livecc_query") or "").strip()
        self._livecc_response_prefix = (data.get("livecc_response_prefix") or "").strip()

        styles = data.get("styles") or {}
        parsed: Dict[str, StyleItem] = {}
        for key, v in styles.items():
            key = str(key)
            label = str(v.get("label", key))
            desc = str(v.get("description", ""))
            parsed[key] = StyleItem(key=key, label=label, description=desc)

        self._styles = parsed

        if self._default_style not in self._styles and self._styles:
            self._default_style = next(iter(self._styles.keys()))

    def list_styles(self) -> Tuple[StyleItem, ...]:
        return tuple(self._styles.values())

    def default_style_key(self) -> str:
        return self._default_style

    def livecc_query(self) -> str:
        """Return the fixed LiveCC objective description query."""
        return self._livecc_query

    def livecc_response_prefix(self) -> str:
        """Forced opening stem for every LiveCC line (e.g. 'The player'). Keeps the
        fine-tuned model in grounded third-person commentary instead of drifting into
        first-person YouTube-narration hallucination on this VR-gameplay footage."""
        return self._livecc_response_prefix
