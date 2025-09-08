"""NSGA-II based multi-objective optimizer with pymoo fallback.

This module provides a production-ready wrapper around a multi-objective
optimization workflow. If :mod:`pymoo` is available, we run a true NSGA-II
algorithm; otherwise, we degrade gracefully to a lightweight heuristic so the
rest of the system keeps functioning in constrained environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class ObjectiveFn:
    """Represents a single objective function.

    Attributes:
        name: Human readable name of the objective.
        direction: "min" to minimize, "max" to maximize.
        fn: Callable accepting a state dict and returning a float score.
    """

    name: str
    direction: str  # "min" or "max"
    fn: Callable[[Dict[str, Any]], float]


@dataclass
class DecisionVariable:
    """Metadata describing a decision variable optimized by NSGA-II."""

    name: str
    lower_bound: float
    upper_bound: float
    initial_value: Optional[float] = None


class CementNSGA2Optimizer:
    """Run multi-objective optimization with NSGA-II if available.

    Example usage:
        optimizer = CementNSGA2Optimizer(
            decision_variables=[
                DecisionVariable("kiln_id_fan_speed", 50.0, 75.0, 60.0),
                DecisionVariable("primary_air_percent", 5.0, 12.0, 8.0),
            ],
            objectives=[
                ObjectiveFn("energy_consumption", "min", lambda s: s["kiln_id_fan_speed"] * 1.0),
                ObjectiveFn("quality_penalty", "min", lambda s: abs(9.0 - s["primary_air_percent"]))
            ],
        )
        result = optimizer.optimize(initial_state={})
    """

    def __init__(
        self,
        decision_variables: List[DecisionVariable],
        objectives: List[ObjectiveFn],
    ) -> None:
        if not objectives:
            raise ValueError("At least one objective is required")
        if not decision_variables:
            raise ValueError("At least one decision variable is required")

        self.objectives = objectives
        self.decision_variables = decision_variables

    # ------------------------- pymoo-backed optimization -------------------------
    def _optimize_with_pymoo(
        self,
        initial_state: Dict[str, Any],
        num_generations: int,
        population_size: int,
        seed: int,
    ) -> Dict[str, Any]:
        try:
            import numpy as np  # type: ignore
            from pymoo.algorithms.moo.nsga2 import NSGA2  # type: ignore
            from pymoo.core.problem import Problem  # type: ignore
            from pymoo.optimize import minimize  # type: ignore
            from pymoo.termination import get_termination  # type: ignore
        except Exception:
            # If pymoo or numpy is unavailable, fall back
            return self._optimize_with_fallback(initial_state)

        objective_names: List[str] = [o.name for o in self.objectives]
        objective_signs: List[int] = [1 if o.direction == "min" else -1 for o in self.objectives]

        lower_bounds = np.array([v.lower_bound for v in self.decision_variables], dtype=float)
        upper_bounds = np.array([v.upper_bound for v in self.decision_variables], dtype=float)

        name_by_index = [v.name for v in self.decision_variables]

        class _CementProblem(Problem):  # type: ignore
            def __init__(self) -> None:
                super().__init__(n_var=len(name_by_index), n_obj=len(objective_names), xl=lower_bounds, xu=upper_bounds)

            def _evaluate(self, X, out, *args, **kwargs):  # noqa: N802 - pymoo API
                # X shape: (pop_size, n_var)
                F = np.zeros((X.shape[0], len(objective_names)), dtype=float)
                for i in range(X.shape[0]):
                    state = {name_by_index[j]: float(X[i, j]) for j in range(len(name_by_index))}
                    # Include any provided baseline context
                    state.update(initial_state)
                    for k, obj in enumerate(self_outer.objectives):
                        raw_value = float(obj.fn(state))
                        F[i, k] = raw_value if obj.direction == "min" else -raw_value
                out["F"] = F

        # capture outer self for inner class
        self_outer = self
        problem = _CementProblem()

        algorithm = NSGA2(pop_size=population_size)
        termination = get_termination("n_gen", num_generations)
        result = minimize(problem, algorithm, termination, seed=seed, verbose=False)

        return {
            "algorithm": "pymoo.NSGA2",
            "objective_names": objective_names,
            "objective_directions": [o.direction for o in self.objectives],
            "pareto_X": result.X.tolist() if hasattr(result, "X") and result.X is not None else [],
            "pareto_F": result.F.tolist() if hasattr(result, "F") and result.F is not None else [],
            "best": {
                "decision_vector": (result.X[0].tolist() if hasattr(result, "X") and result.X is not None else []),
                "objectives": (result.F[0].tolist() if hasattr(result, "F") and result.F is not None else []),
            },
        }

    # ------------------------------ heuristic fallback -----------------------------
    def _optimize_with_fallback(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        # Evaluate only the initial state and return a single-point "front".
        scores = []
        state_for_eval = initial_state.copy()
        for var in self.decision_variables:
            if var.initial_value is not None:
                state_for_eval[var.name] = var.initial_value
        for obj in self.objectives:
            raw = float(obj.fn(state_for_eval))
            scores.append(raw if obj.direction == "min" else -raw)
        return {
            "algorithm": "heuristic",
            "objective_names": [o.name for o in self.objectives],
            "objective_directions": [o.direction for o in self.objectives],
            "pareto_X": [[state_for_eval.get(v.name, v.initial_value or v.lower_bound) for v in self.decision_variables]],
            "pareto_F": [scores],
            "best": {"decision_vector": None, "objectives": scores},
        }

    # ----------------------------------- public API ----------------------------------
    def optimize(
        self,
        initial_state: Dict[str, Any],
        num_generations: int = 40,
        population_size: int = 40,
        seed: int = 1,
    ) -> Dict[str, Any]:
        """Run optimization returning a serializable summary dict."""
        return self._optimize_with_pymoo(
            initial_state=initial_state,
            num_generations=num_generations,
            population_size=population_size,
            seed=seed,
        )


