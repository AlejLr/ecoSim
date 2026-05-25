"""Shared population-coexistence scoring helpers.

These helpers are used both for reward shaping and checkpoint ranking so
training and evaluation optimize the same objective.
"""

from __future__ import annotations

from config.config import (
    COEXISTENCE_EXTINCTION_PENALTY,
    COEXISTENCE_PREDATOR_FLOOR,
    COEXISTENCE_PREY_FLOOR,
    COEXISTENCE_STEP_BONUS,
    COEXISTENCE_TARGET_PREY_TO_PREDATOR_RATIO,
)


def coexistence_score(prey_count: int, predator_count: int) -> float:
    """Return a normalized coexistence score in [0, 1]."""

    if prey_count <= 0 or predator_count <= 0:
        return 0.0

    prey_health = min(prey_count / COEXISTENCE_PREY_FLOOR, 1.0)
    predator_health = min(predator_count / COEXISTENCE_PREDATOR_FLOOR, 1.0)
    ratio = (prey_count + 1.0) / (predator_count + 1.0)
    balance = max(
        0.0,
        1.0 - abs(ratio - COEXISTENCE_TARGET_PREY_TO_PREDATOR_RATIO) / COEXISTENCE_TARGET_PREY_TO_PREDATOR_RATIO,
    )

    return float(prey_health * predator_health * balance)


def coexistence_reward(prey_count: int, predator_count: int) -> float:
    """Return a small shaping reward for long-term coexistence."""

    if prey_count <= 0 or predator_count <= 0:
        return COEXISTENCE_EXTINCTION_PENALTY
    return COEXISTENCE_STEP_BONUS * coexistence_score(prey_count, predator_count)