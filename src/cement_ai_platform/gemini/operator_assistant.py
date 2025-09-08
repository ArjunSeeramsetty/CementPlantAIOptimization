"""Operator assistant backed by Google Gemini API.

Usage:
  export GEMINI_API_KEY=...
  from cement_ai_platform.gemini.operator_assistant import OperatorAssistant
  assistant = OperatorAssistant()
  print(assistant.ask("What are kiln optimization levers?"))
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AssistantConfig:
    model: str = "gemini-1.5-pro"
    safety_mode: bool = True
    system_context: Optional[str] = None


class OperatorAssistant:
    def __init__(self, config: AssistantConfig | None = None):
        self.config = config or AssistantConfig()
        self._model = None
        self._ensure_client()

    def _ensure_client(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Defer hard failure until first call to ask(); allow construction for tests
            return
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "google-generativeai is not installed. Add it to requirements.txt"
            ) from exc
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.config.model)

    def ask(self, prompt: str, plant_context: Optional[dict] = None) -> str:
        if self._model is None:
            # Retry lazy init once in case env var became available
            self._ensure_client()
        if self._model is None:
            return "[Gemini not configured: set GEMINI_API_KEY]"

        system_prompt = self._build_cement_plant_context(plant_context)
        request = f"{system_prompt}\n{prompt}" if system_prompt else prompt
        try:
            response = self._model.generate_content(request)
            # google-generativeai returns an object with `text` property
            return getattr(response, "text", "") or "[No response]"
        except Exception as exc:  # pragma: no cover
            return f"[Gemini error: {exc}]"

    def suggest_actions(self, context: str) -> List[str]:
        return [
            "Review kiln fuel feed",
            "Check ID fan speed",
            "Inspect temperature sensors",
        ]

    def _build_cement_plant_context(self, plant_context: Optional[dict]) -> str:
        base = self.config.system_context or (
            "You are an expert assistant for cement plant operations. "
            "Provide concise, actionable recommendations."
        )
        if not plant_context:
            return base
        lines = [base, "Context:"]
        for key, value in plant_context.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)



