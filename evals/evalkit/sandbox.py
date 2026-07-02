"""Subprocess sandbox for executing model-generated Python.

Isolation profile: `python -I` (isolated mode — no user site-packages, no
env-var injection, cwd not on sys.path), a scratch temp directory as cwd, a
minimal environment, and a hard wall-clock timeout. That is adequate for a
personal eval box grading its own model's code; it is NOT a security
boundary against a deliberately adversarial model.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool

    def summary(self) -> str:
        if self.timed_out:
            return "timed out"
        if self.ok:
            return "ok"
        tail = self.stderr.strip().splitlines()[-1] if self.stderr.strip() else ""
        return f"exit {self.exit_code}: {tail}"


def run_python(code: str, timeout: float = 10.0) -> SandboxResult:
    with tempfile.TemporaryDirectory(prefix="evalsbx_") as tmp:
        script = Path(tmp) / "main.py"
        script.write_text(code, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmp,
            "TMPDIR": tmp,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as err:
            return SandboxResult(
                ok=False, exit_code=-1,
                stdout=(err.stdout or b"").decode() if isinstance(err.stdout, bytes)
                else (err.stdout or ""),
                stderr="", timed_out=True,
            )
        return SandboxResult(
            ok=proc.returncode == 0,
            exit_code=proc.returncode,
            stdout=proc.stdout[-4000:],
            stderr=proc.stderr[-4000:],
            timed_out=False,
        )
