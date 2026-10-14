"""Public compiler entry points for turning a SKILL.md into a typed recipe.

Source generation and repair live in :mod:`calcifier.repair`; the deterministic
acceptance checks live in :mod:`calcifier.gate`. Artifact preparation validates
the complete candidate before the destination is written.
"""

from __future__ import annotations

import json
from pathlib import Path

from .agent_events import AgentActivity
from .artifact import (
    PreparedArtifact as PreparedArtifact,
)
from .artifact import (
    _atomic_write_text,
)
from .artifact import (
    calcify_file as calcify_file,
)
from .artifact import (
    prepare_artifact as prepare_artifact,
)
from .artifact import (
    render_summary as render_summary,
)
from .catalog import load_skill_catalog as load_skill_catalog
from .compiler_activity import (
    compiler_activity_payload as compiler_activity_payload,
)
from .compiler_activity import (
    render_compiler_activity as render_compiler_activity,
)
from .diagnostics import (
    CalcifierDefectError as CalcifierDefectError,
)
from .diagnostics import (
    CalcifyError as CalcifyError,
)
from .diagnostics import (
    SkillDeclarationError as SkillDeclarationError,
)
from .repair import calcify_source as calcify_source
from .repair import render_dry_run as render_dry_run


def write_compiler_trace(path: Path, activity: AgentActivity) -> None:
    """Atomically persist compiler progress and normalized telemetry."""
    _atomic_write_text(path, json.dumps(compiler_activity_payload(activity), indent=2) + "\n")
