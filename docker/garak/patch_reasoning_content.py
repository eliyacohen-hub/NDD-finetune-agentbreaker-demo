"""Teach pinned Garak to accept reasoning-only OpenAI-compatible responses."""

from pathlib import Path

import garak.generators.openai

old = "Message(c.message.content) for c in response.choices"
new = (
    "Message(c.message.content or getattr(c.message, "
    "'reasoning_content', None)) for c in response.choices"
)
paths = {
    Path(garak.generators.openai.__file__),
    Path("/opt/garak/garak/generators/openai.py"),
}
for path in paths:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one Garak response extraction site in {path}")
    path.write_text(source.replace(old, new), encoding="utf-8")

probe_paths = {
    Path("/opt/garak/garak/probes/agent_breaker.py"),
}
for path in probe_paths:
    source = path.read_text(encoding="utf-8")
    old_analysis = "        analysis = self._get_model_response(analysis_prompt)\n"
    new_analysis = (
        "        self._setup_parse_model()\n"
        "        analysis = self._get_model_response(\n"
        "            analysis_prompt, model=self.parse_model\n"
        "        )\n"
    )
    if source.count(old_analysis) != 1:
        raise RuntimeError(f"Expected one AgentBreaker analysis call in {path}")
    source = source.replace(old_analysis, new_analysis)
    old_priority = (
        '                result["priority_targets"] = '
        'parsed.get("priority_targets", [])\n'
    )
    new_priority = old_priority + (
        '                for tool_analysis in result["tool_analyses"].values():\n'
        "                    if isinstance(tool_analysis, dict):\n"
        '                        tool_analysis["attack_prompts"] = '
        'tool_analysis.get("attack_prompts", [])[:1]\n'
    )
    if source.count(old_priority) != 1:
        raise RuntimeError(f"Expected one AgentBreaker priority assignment in {path}")
    path.write_text(source.replace(old_priority, new_priority), encoding="utf-8")
