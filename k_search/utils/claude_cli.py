"""Subprocess wrapper around `claude -p` for indirect LLM calls.

Lets K-Search use a logged-in Claude Code CLI (subscription auth) instead of
an OpenAI-compatible HTTP endpoint. The prompt is piped via stdin to avoid
argv length limits; tools are disabled so the call is pure chat-completion.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Optional


class ClaudeCLIError(RuntimeError):
    pass


def call_claude_cli(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout: float = 900.0,
    cwd: Optional[str] = None,
) -> str:
    """Invoke `claude -p` and return the stripped text response.

    Tools are disabled (`--tools ""`) so Claude responds purely with text and
    cannot invoke Bash/Edit/etc. Permission/workspace-trust prompts are skipped
    automatically in -p mode. Raises ClaudeCLIError on non-zero exit.
    """
    if shutil.which("claude") is None:
        raise ClaudeCLIError(
            "`claude` CLI not found in PATH. Install Claude Code "
            "(https://docs.claude.com/en/docs/claude-code/) to use --use-claude-cli."
        )

    args = ["claude", "-p", "--tools", "", "--output-format", "text"]
    if model:
        args += ["--model", model]

    try:
        proc = subprocess.run(
            args,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as e:
        raise ClaudeCLIError(f"claude -p timed out after {timeout:.0f}s") from e

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:1000]
        raise ClaudeCLIError(
            f"claude -p exited with {proc.returncode}: {stderr or '(no stderr)'}"
        )

    return (proc.stdout or "").strip()
