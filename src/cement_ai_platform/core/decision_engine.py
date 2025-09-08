"""Unified decision engine combining optimization with LLM explanations."""

from __future__ import annotations

from typing import Any, Dict

from ..gemini.operator_assistant import OperatorAssistant
from ..models.optimization import CementNSGA2Optimizer, ObjectiveFn


class UnifiedDecisionEngine:
    def __init__(self, optimizer: CementNSGA2Optimizer, assistant: OperatorAssistant):
        self.optimizer = optimizer
        self.assistant = assistant

    def optimize_plant_operations(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        result = self.optimizer.optimize(current_state)
        explanation = self.assistant.ask(
            "Explain the recommended actions for these results:",
            plant_context={"optimization_result": result},
        )
        return {"recommendations": result, "explanations": explanation}


