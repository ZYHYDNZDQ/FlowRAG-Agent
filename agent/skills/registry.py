"""Backward-compatible re-export of ``skills.registry``."""

from skills.registry import SkillRegistry, build_default_skill_registry, get_skill_registry

__all__ = ["SkillRegistry", "build_default_skill_registry", "get_skill_registry"]
