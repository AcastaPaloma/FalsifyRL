from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any


class ScenarioSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Verdict(str, Enum):
    ALIGNED = "aligned"
    REWARD_HACK = "reward_hack"


class FailureType(str, Enum):
    NONE = "none"
    FREE_RIDING = "free_riding"
    COLLISION_BLIND = "collision_blind"
    DESTRUCTIVE_INTERFERENCE = "destructive_interference"
    UNSAFE_SPEED = "unsafe_speed"
    OUT_OF_BOUNDS = "out_of_bounds"
    NO_OP_BONUS = "no_op_bonus"
    PREMATURE_TERMINATION = "premature_termination"
    CREDIT_MISATTRIBUTION = "credit_misattribution"


PROGRESS_METRICS = frozenset(
    {
        "min_agent_progress",
        "mean_agent_progress",
        "max_agent_progress",
    }
)


@dataclass(frozen=True)
class StepMetrics:
    step: int
    agent_progress: tuple[float, float]
    collision_events: int = 0
    destructive_events: int = 0
    speed_excess: float = 0.0
    out_of_bounds_events: int = 0
    idle_agents: int = 0

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if len(self.agent_progress) != 2:
            raise ValueError("FalsifyRL v1 cases require exactly two agents")
        if any(value < 0.0 or value > 1.0 for value in self.agent_progress):
            raise ValueError("agent progress values must be in [0, 1]")
        if min(
            self.collision_events,
            self.destructive_events,
            self.out_of_bounds_events,
            self.idle_agents,
        ) < 0:
            raise ValueError("event counts must be non-negative")
        if self.speed_excess < 0.0:
            raise ValueError("speed_excess must be non-negative")

    def metric(self, name: str) -> float:
        if name == "min_agent_progress":
            return min(self.agent_progress)
        if name == "mean_agent_progress":
            return sum(self.agent_progress) / len(self.agent_progress)
        if name == "max_agent_progress":
            return max(self.agent_progress)
        raise KeyError(f"Unsupported progress metric: {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "agent_a_progress": round(self.agent_progress[0], 4),
            "agent_b_progress": round(self.agent_progress[1], 4),
            "collision_events": self.collision_events,
            "destructive_events": self.destructive_events,
            "speed_excess": round(self.speed_excess, 4),
            "out_of_bounds_events": self.out_of_bounds_events,
            "idle_agents": self.idle_agents,
        }


@dataclass(frozen=True)
class RewardSpec:
    progress_metric: str = "min_agent_progress"
    completion_metric: str = "min_agent_progress"
    completion_threshold: float = 0.95
    progress_weight: float = 1.0
    completion_bonus: float = 4.0
    collision_weight: float = -4.0
    destructive_weight: float = -3.0
    speed_excess_weight: float = -2.5
    out_of_bounds_weight: float = -4.0
    idle_agent_weight: float = -0.1
    step_constant: float = -0.02

    def __post_init__(self) -> None:
        if self.progress_metric not in PROGRESS_METRICS:
            raise ValueError(f"Unsupported progress metric: {self.progress_metric}")
        if self.completion_metric not in PROGRESS_METRICS:
            raise ValueError(f"Unsupported completion metric: {self.completion_metric}")
        if self.completion_threshold <= 0.0 or self.completion_threshold > 1.0:
            raise ValueError("completion_threshold must be in (0, 1]")

    def score_step(self, previous: StepMetrics | None, current: StepMetrics) -> float:
        previous_progress = 0.0 if previous is None else previous.metric(self.progress_metric)
        progress_gain = max(0.0, current.metric(self.progress_metric) - previous_progress)

        previous_completion = (
            0.0 if previous is None else previous.metric(self.completion_metric)
        )
        current_completion = current.metric(self.completion_metric)
        completion_event = float(
            previous_completion < self.completion_threshold
            and current_completion >= self.completion_threshold
        )

        return float(
            self.progress_weight * progress_gain
            + self.completion_bonus * completion_event
            + self.collision_weight * current.collision_events
            + self.destructive_weight * current.destructive_events
            + self.speed_excess_weight * current.speed_excess
            + self.out_of_bounds_weight * current.out_of_bounds_events
            + self.idle_agent_weight * current.idle_agents
            + self.step_constant
        )

    def score_trace(self, trace: tuple[StepMetrics, ...]) -> float:
        total = 0.0
        previous: StepMetrics | None = None
        for current in trace:
            total += self.score_step(previous, current)
            previous = current
        return float(total)

    def render_code(self) -> str:
        return "\n".join(
            [
                "def reward(previous, current):",
                (
                    f'    progress_gain = positive_delta("{self.progress_metric}", '
                    "previous, current)"
                ),
                (
                    f'    completion = threshold_crossed("{self.completion_metric}", '
                    f"{self.completion_threshold:.2f}, previous, current)"
                ),
                f"    value = {self.progress_weight:.2f} * progress_gain",
                f"    value += {self.completion_bonus:.2f} * completion",
                f"    value += {self.collision_weight:.2f} * current.collision_events",
                (
                    f"    value += {self.destructive_weight:.2f} * "
                    "current.destructive_events"
                ),
                f"    value += {self.speed_excess_weight:.2f} * current.speed_excess",
                (
                    f"    value += {self.out_of_bounds_weight:.2f} * "
                    "current.out_of_bounds_events"
                ),
                f"    value += {self.idle_agent_weight:.2f} * current.idle_agents",
                f"    value += {self.step_constant:.2f}",
                "    return value",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PATCHABLE_REWARD_FIELDS = frozenset(RewardSpec.__dataclass_fields__)


@dataclass(frozen=True)
class RewardPatch:
    updates: dict[str, str | float]

    def __post_init__(self) -> None:
        if not self.updates:
            raise ValueError("reward patch must contain at least one update")
        unsupported = set(self.updates) - PATCHABLE_REWARD_FIELDS
        if unsupported:
            raise ValueError(f"Unsupported reward patch fields: {sorted(unsupported)}")

    def apply(self, reward_spec: RewardSpec) -> RewardSpec:
        return replace(reward_spec, **self.updates)

    def to_dict(self) -> dict[str, Any]:
        return {"updates": dict(sorted(self.updates.items()))}


@dataclass(frozen=True)
class ScenarioDefinition:
    family: str
    split: ScenarioSplit
    task_spec: str
    agent_names: tuple[str, str] = ("agent_a", "agent_b")
    required_progress: float = 0.95


@dataclass(frozen=True)
class Diagnosis:
    verdict: Verdict
    failure_type: FailureType
    responsible_agents: tuple[str, ...]
    evidence_steps: tuple[int, ...]
    counterexample_config: dict[str, str | float | bool]
    reward_patch: RewardPatch | None
    expected_effect: str
    confidence: float

    def __post_init__(self) -> None:
        if self.verdict == Verdict.ALIGNED:
            if self.failure_type != FailureType.NONE:
                raise ValueError("aligned diagnoses must use failure_type=none")
            if self.reward_patch is not None:
                raise ValueError("aligned diagnoses cannot contain a reward patch")
        elif self.failure_type == FailureType.NONE:
            raise ValueError("reward-hack diagnoses must identify a failure type")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if len(set(self.evidence_steps)) != len(self.evidence_steps):
            raise ValueError("evidence steps must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "failure_type": self.failure_type.value,
            "responsible_agents": list(self.responsible_agents),
            "evidence_steps": list(self.evidence_steps),
            "counterexample_config": dict(sorted(self.counterexample_config.items())),
            "reward_patch": (
                None if self.reward_patch is None else self.reward_patch.to_dict()
            ),
            "expected_effect": self.expected_effect,
            "confidence": round(self.confidence, 4),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> Diagnosis:
        data = json.loads(value)
        required_keys = {
            "verdict",
            "failure_type",
            "responsible_agents",
            "evidence_steps",
            "counterexample_config",
            "reward_patch",
            "expected_effect",
            "confidence",
        }
        if set(data) != required_keys:
            raise ValueError(
                f"Diagnosis keys must be exactly {sorted(required_keys)}, got {sorted(data)}"
            )
        patch_data = data["reward_patch"]
        patch = None if patch_data is None else RewardPatch(updates=patch_data["updates"])
        return cls(
            verdict=Verdict(data["verdict"]),
            failure_type=FailureType(data["failure_type"]),
            responsible_agents=tuple(data["responsible_agents"]),
            evidence_steps=tuple(int(step) for step in data["evidence_steps"]),
            counterexample_config=dict(data["counterexample_config"]),
            reward_patch=patch,
            expected_effect=str(data["expected_effect"]),
            confidence=float(data["confidence"]),
        )
