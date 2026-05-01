"""Smoke tests for ml-automation-llm — validate plugin layout invariants."""
import json
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_validity():
    """Verify .cortex-plugin/plugin.json is valid and required fields exist."""
    manifest_path = PLUGIN_ROOT / ".cortex-plugin" / "plugin.json"

    assert manifest_path.exists(), "plugin.json manifest not found"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Verify required fields
    required_fields = ["name", "version", "description", "cortex"]
    for field in required_fields:
        assert field in manifest, f"Missing required field: {field}"

    # Verify cortex subdirectories are specified
    cortex = manifest.get("cortex", {})
    assert "agents_dir" in cortex, "cortex.agents_dir not defined"
    assert "skills_dir" in cortex, "cortex.skills_dir not defined"
    assert "commands_dir" in cortex, "cortex.commands_dir not defined"


def test_agents_md_referential_integrity():
    """Verify AGENTS.md entries have matching agent/skill files."""
    agents_md_path = PLUGIN_ROOT / "AGENTS.md"

    assert agents_md_path.exists(), "AGENTS.md not found"

    with open(agents_md_path, "r") as f:
        agents_md_content = f.read()

    # Extract agent names from "Available Agents" section
    agents_section = agents_md_content.split("## Available Agents")[1].split("## Available Skills")[0]
    agents_lines = agents_section.strip().split("\n")[2:]  # Skip header lines

    agents_dir = PLUGIN_ROOT / "agents"

    for line in agents_lines:
        if line.startswith("|") and "`" in line:
            # Extract agent name from markdown table
            agent_name = line.split("`")[1]
            agent_file = agents_dir / f"{agent_name}.md"
            assert agent_file.exists(), f"Agent file not found: agents/{agent_name}.md"

    # Extract skill names from "Available Skills" section
    skills_section = agents_md_content.split("## Available Skills")[1].split("## Routing")[0]
    skills_lines = skills_section.strip().split("\n")[2:]  # Skip header lines

    skills_dir = PLUGIN_ROOT / "skills"

    for line in skills_lines:
        if line.startswith("|") and "`/" in line:
            # Extract skill name from markdown table (e.g., `/llm-evaluate`)
            skill_slash = line.split("`")[1]
            skill_name = skill_slash.lstrip("/")
            skill_dir = skills_dir / skill_name
            assert skill_dir.exists(), f"Skill directory not found: skills/{skill_name}"

            skill_file = skill_dir / "SKILL.md"
            assert skill_file.exists(), f"Skill definition not found: skills/{skill_name}/SKILL.md"
