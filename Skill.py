from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Optional

from .ToolNormal import ToolNormal


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    path: Path
    metadata: dict[str, str]
    content: str


class Skill(ToolNormal):
    toolName = "skill"
    toolDescription = """
Load reusable instructions from the configured project skill directory by name.
Use this only when one of the advertised skills is directly relevant to the current task.
""".strip()
    paramSchema = {
        "name": (str, "the skill name to load, for example 'cwe'"),
    }

    _logger = logging.getLogger(__name__)
    _VALID_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

    def __init__(self, skill_folder_path: str):
        self.skill_folder_path = Path(skill_folder_path).resolve()
        self.skills = self._discover_skills()

    def extra_system_prompt(self) -> Optional[str]:
        if not self.skills:
            return (
                f"No project skills were found in {self.skill_folder_path}. "
                "Do not call the skill tool unless the user adds skills later."
            )

        skill_lines = [
            f"- {skill.name}: {skill.description}" for skill in self.skills.values()
        ]
        return "\n".join(
            [
                "You can load reusable project-specific skills with the `skill` tool.",
                "Only call it when one of the available skills is clearly relevant.",
                "Available skills:",
                *skill_lines,
            ]
        )

    def run(self, tool_call_arg: dict[str, Any]) -> str:
        skill_name = str(tool_call_arg.get("name", "")).strip()
        if not skill_name:
            return self._format_available_skills(
                "Missing required argument `name` for the skill tool."
            )

        skill = self.skills.get(skill_name)
        if skill is None:
            return self._format_available_skills(
                f"Skill `{skill_name}` was not found in {self.skill_folder_path}."
            )

        return "\n".join(
            [
                f"# Skill: {skill.name}",
                f"Source: {skill.path}",
                f"Description: {skill.description}",
                "",
                skill.content.strip(),
            ]
        ).strip()

    def _format_available_skills(self, prefix: Optional[str] = None) -> str:
        lines: list[str] = []
        if prefix:
            lines.append(prefix)
            lines.append("")
        lines.append(f"Configured skill directory: {self.skill_folder_path}")
        if not self.skills:
            lines.append("No valid skills are currently available.")
            return "\n".join(lines)

        lines.append("Available skills:")
        for skill in self.skills.values():
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)

    def _discover_skills(self) -> dict[str, SkillSpec]:
        if not self.skill_folder_path.exists():
            self._logger.warning(
                "Skill directory does not exist: %s", self.skill_folder_path
            )
            return {}
        if not self.skill_folder_path.is_dir():
            self._logger.warning(
                "Skill path is not a directory: %s", self.skill_folder_path
            )
            return {}

        skills: dict[str, SkillSpec] = {}
        for child in sorted(self.skill_folder_path.iterdir()):
            skill_file = child / "SKILL.md"
            if not child.is_dir() or not skill_file.is_file():
                continue
            try:
                skill = self._load_skill_file(skill_file)
            except Exception as exc:
                self._logger.warning("Skipping invalid skill %s: %s", skill_file, exc)
                continue
            skills[skill.name] = skill
        return skills

    def _load_skill_file(self, skill_file: Path) -> SkillSpec:
        raw_text = skill_file.read_text(encoding="utf-8")
        metadata, content = self._parse_frontmatter(raw_text, skill_file)

        name = metadata.get("name", "").strip()
        description = metadata.get("description", "").strip()

        if not self._VALID_SKILL_NAME.fullmatch(name):
            raise ValueError(
                "frontmatter field `name` must match ^[a-z0-9][a-z0-9_-]{0,63}$"
            )
        if not description:
            raise ValueError("frontmatter field `description` is required")
        if not content.strip():
            raise ValueError("SKILL.md content must not be empty")

        return SkillSpec(
            name=name,
            description=description,
            path=skill_file,
            metadata=metadata,
            content=content,
        )

    def _parse_frontmatter(
        self, raw_text: str, skill_file: Path
    ) -> tuple[dict[str, str], str]:
        lines = raw_text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError(f"{skill_file} must start with YAML frontmatter")

        end_idx: Optional[int] = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_idx = idx
                break
        if end_idx is None:
            raise ValueError("frontmatter is missing the closing --- delimiter")

        metadata: dict[str, str] = {}
        for line in lines[1:end_idx]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"invalid frontmatter line: {line}")
            key, value = stripped.split(":", 1)
            metadata[key.strip()] = self._strip_yaml_scalar(value.strip())

        content = "\n".join(lines[end_idx + 1 :]).strip()
        return metadata, content

    @staticmethod
    def _strip_yaml_scalar(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value
