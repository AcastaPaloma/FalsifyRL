"""Deterministic reward-falsification tools for embodied reinforcement learning."""

from falsifyrl.scenarios import (
    GENERATOR_VERSION,
    SCENARIO_DEFINITIONS,
    GeneratedCase,
    generate_case,
    generate_cases,
    generate_paired_cases,
)
from falsifyrl.schema import (
    Diagnosis,
    FailureType,
    RewardPatch,
    RewardSpec,
    ScenarioDefinition,
    ScenarioSplit,
    StepMetrics,
    Verdict,
)
from falsifyrl.verifier import CaseVerification, verify_case

__all__ = [
    "CaseVerification",
    "Diagnosis",
    "FailureType",
    "GENERATOR_VERSION",
    "GeneratedCase",
    "RewardPatch",
    "RewardSpec",
    "SCENARIO_DEFINITIONS",
    "ScenarioDefinition",
    "ScenarioSplit",
    "StepMetrics",
    "Verdict",
    "generate_case",
    "generate_cases",
    "generate_paired_cases",
    "verify_case",
]
