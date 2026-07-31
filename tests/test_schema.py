from __future__ import annotations

import json

import pytest

from falsifyrl import Diagnosis, FailureType, RewardPatch, RewardSpec, StepMetrics, Verdict


def test_diagnosis_json_round_trip_is_strict_and_stable() -> None:
    diagnosis = Diagnosis(
        verdict=Verdict.REWARD_HACK,
        failure_type=FailureType.COLLISION_BLIND,
        responsible_agents=("agent_a", "agent_b"),
        evidence_steps=(3,),
        counterexample_config={"collision_step": 3},
        reward_patch=RewardPatch(updates={"collision_weight": -4.0}),
        expected_effect="Safe trajectories outrank collision trajectories.",
        confidence=0.97,
    )

    serialized = diagnosis.to_json()

    assert Diagnosis.from_json(serialized) == diagnosis
    assert serialized == json.dumps(
        diagnosis.to_dict(), sort_keys=True, separators=(",", ":")
    )


def test_diagnosis_rejects_extra_json_fields() -> None:
    value = Diagnosis(
        verdict=Verdict.ALIGNED,
        failure_type=FailureType.NONE,
        responsible_agents=(),
        evidence_steps=(),
        counterexample_config={},
        reward_patch=None,
        expected_effect="No patch needed.",
        confidence=0.99,
    ).to_dict()
    value["hidden_reasoning"] = "not allowed"

    with pytest.raises(ValueError, match="keys must be exactly"):
        Diagnosis.from_json(json.dumps(value))


def test_reward_patch_is_declarative_and_field_limited() -> None:
    reward = RewardSpec(collision_weight=0.0)
    patched = RewardPatch(updates={"collision_weight": -5.0}).apply(reward)

    assert reward.collision_weight == 0.0
    assert patched.collision_weight == -5.0
    with pytest.raises(ValueError, match="Unsupported reward patch fields"):
        RewardPatch(updates={"arbitrary_python": "import os"})


def test_step_metrics_enforce_two_agent_bounds() -> None:
    with pytest.raises(ValueError, match="exactly two agents"):
        StepMetrics(step=0, agent_progress=(0.5,))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        StepMetrics(step=0, agent_progress=(0.5, 1.2))

