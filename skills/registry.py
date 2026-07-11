"""Skill registry — registration and lookup."""

from __future__ import annotations

from models.schemas import IntentType
from skills.analysis_skill import AnalysisSkill
from skills.base import BaseSkill
from skills.qa_skill import QASkill
from skills.summary_skill import SummarySkill


class SkillRegistry:
    """Register and resolve Skills by intent."""

    def __init__(self) -> None:
        self._skills: dict[IntentType, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        self._skills[skill.intent] = skill

    def get(self, intent: IntentType) -> BaseSkill:
        try:
            return self._skills[intent]
        except KeyError as exc:
            raise KeyError(f"No skill registered for intent: {intent}") from exc

    def list_skills(self) -> list[dict[str, str]]:
        return [
            {
                "name": skill.name,
                "intent": skill.intent.value,
                "description": skill.description,
            }
            for skill in self._skills.values()
        ]


def build_default_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(QASkill())
    registry.register(SummarySkill())
    registry.register(AnalysisSkill())
    return registry


_default_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_skill_registry()
    return _default_registry
