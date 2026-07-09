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
    def __init__(self, config_path: str | Path, sport: str | None = None):
        self.config_path = Path(config_path)
        self._raw: Dict[str, Any] = {}
        self._styles: Dict[str, StyleItem] = {}
        self._default_style: str = ""
        self._sports: Dict[str, Dict[str, str]] = {}
        self._default_sport: str = ""
        self._sport: str = ""
        self._livecc_response_prefix: str = ""
        self.reload()
        self.set_sport(sport or self._default_sport)

    def reload(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Prompt config not found: {self.config_path}")

        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self._raw = data

        self._default_style = (data.get("default_style") or "").strip()
        self._livecc_response_prefix = (data.get("livecc_response_prefix") or "").strip()

        # 解析各運動的 prompt。向後相容：若沒有 sports 區塊，就把頂層的
        # livecc_query / livecc_query_splitscreen 當成單一 default 運動。
        sports = data.get("sports") or {}
        parsed_sports: Dict[str, Dict[str, str]] = {}
        for key, v in sports.items():
            v = v or {}
            parsed_sports[str(key)] = {
                "label": str(v.get("label", key)),
                "livecc_query": (v.get("livecc_query") or "").strip(),
                "livecc_query_splitscreen": (v.get("livecc_query_splitscreen") or "").strip(),
            }
        if not parsed_sports:
            parsed_sports["default"] = {
                "label": "Default",
                "livecc_query": (data.get("livecc_query") or "").strip(),
                "livecc_query_splitscreen": (data.get("livecc_query_splitscreen") or "").strip(),
            }
        self._sports = parsed_sports

        self._default_sport = (data.get("default_sport") or "").strip()
        if self._default_sport not in self._sports:
            self._default_sport = next(iter(self._sports.keys()))

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

    def set_sport(self, sport: str | None) -> None:
        """選擇要使用哪一組運動 prompt；未知的名稱會退回 default。"""
        if sport and sport in self._sports:
            self._sport = sport
        else:
            self._sport = self._default_sport

    def current_sport(self) -> str:
        return self._sport

    def list_sports(self) -> Tuple[str, ...]:
        return tuple(self._sports.keys())

    def list_styles(self) -> Tuple[StyleItem, ...]:
        return tuple(self._styles.values())

    def default_style_key(self) -> str:
        return self._default_style

    def livecc_query(self) -> str:
        """Return the fixed LiveCC objective description query for the selected sport."""
        return self._sports.get(self._sport, {}).get("livecc_query", "")

    def livecc_query_splitscreen(self) -> str:
        """Return the split-screen fallback query for the selected sport."""
        return self._sports.get(self._sport, {}).get("livecc_query_splitscreen", "")

    def livecc_response_prefix(self) -> str:
        """Forced opening stem for every LiveCC line (e.g. 'The player'). Keeps the
        fine-tuned model in grounded third-person commentary instead of drifting into
        first-person YouTube-narration hallucination on this VR-gameplay footage."""
        return self._livecc_response_prefix
