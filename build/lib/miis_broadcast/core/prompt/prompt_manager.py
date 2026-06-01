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
    query: str


class PromptManager:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._raw: Dict[str, Any] = {}
        self._styles: Dict[str, StyleItem] = {}
        self._default_style: str = ""
        self._common_rules: str = ""
        self.reload()

    def reload(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Prompt config not found: {self.config_path}")

        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self._raw = data

        self._default_style = (data.get("default_style") or "").strip()
        self._common_rules = (data.get("common_rules") or "").strip()

        styles = data.get("styles") or {}
        parsed: Dict[str, StyleItem] = {}
        for key, v in styles.items():
            key = str(key)
            label = str(v.get("label", key))
            desc = str(v.get("description", ""))
            query = str(v.get("query", "")).strip()
            parsed[key] = StyleItem(key=key, label=label, description=desc, query=query)

        self._styles = parsed

        # 兜底：如果 default_style 不存在，就選第一個
        if self._default_style not in self._styles and self._styles:
            self._default_style = next(iter(self._styles.keys()))

    def list_styles(self) -> Tuple[StyleItem, ...]:
        return tuple(self._styles.values())

    def default_style_key(self) -> str:
        return self._default_style

    def build_query(self, style_key: str) -> str:
        style = self._styles.get(style_key)
        if style is None:
            style_key = self._default_style
            style = self._styles.get(style_key)

        if style is None:
            return self._common_rules  # 真的完全沒 styles 時兜底

        parts = []
        if self._common_rules:
            parts.append(self._common_rules)
        if style.query:
            parts.append(style.query)

        return "\n\n".join(parts).strip()
