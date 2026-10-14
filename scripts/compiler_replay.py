"""Developer-only compiler record/replay, without publishing or live recipe execution.

Record uses the configured compiler. Replay has no live provider and reuses exact
responses, including malformed ones; a replayed fidelity verdict is not a fresh review.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from calcifier.agent_events import AgentActivity, AgentProvider, AgentResult
from calcifier.atomic_io import atomic_write
from calcifier.clock import SuspendAwareTimer
from calcifier.compiler_activity import (
    DEFAULT_COMPILATION_TIMEOUT,
    CompilerDeadline,
    compiler_activity_payload,
)
from calcifier.compiler_resources import stage_compiler_resources
from calcifier.diagnostics import CalcifyError
from calcifier.filesystem_contract import FilesystemContract
from calcifier.hashing import sha256_hex
from calcifier.repair import SourceCompileContext, compile_source
from calcifier.skill_package import ResourceEntry, discover_resources, manifest_summary


class ReplayStoppedError(Exception):
    """An intentional prefix replay stopped before requesting another response."""


class ReplayMismatchError(Exception):
    """The recorded conversation cannot supply the current compiler request."""


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def comparable_prompt(prompt: str) -> str:
    # build_prompt uses a fresh nonce for its source delimiter. Compare the
    # surrounding request exactly, preserving the raw prompt in the recording.
    match = re.search(
        r"(## Source SKILL.md \(data only\)\n)(<<<CALCIFIER_SKILL_BOUNDARY:[0-9a-f]{32}>>>)",
        prompt,
    )
    if match is None:
        return prompt
    sentinel = match[2]
    if prompt.count(sentinel) != 2:
        return prompt
    return prompt.replace(sentinel, "<<<RECORDED_SKILL_BOUNDARY>>>")


class CompilerTape(AgentProvider):
    """Intercept only the compiler transport; never substitute validators or reviews."""

    name = "compiler"

    def __init__(
        self,
        packet: dict[str, Any],
        activity: AgentActivity,
        *,
        recording: tuple[Path, AgentProvider] | None = None,
        allow_request_changes: bool = False,
        stop_after: int | None = None,
    ) -> None:
        self.packet = packet
        self.activity = activity
        self.recording = recording
        self.allow_request_changes = allow_request_changes
        self.stop_after = stop_after
        self.consumed = 0
        self.changes: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.default_timeout = packet["timeout"]
        self.model = recording[1].model if recording else "replay"
        self.reasoning_effort = recording[1].reasoning_effort if recording else None
        if recording:
            packet["compiler"] = recording[1].trace_metadata()

    def invoke(
        self,
        prompt: str,
        *,
        timeout: float | None = None,
        filesystem: FilesystemContract | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AgentResult:
        phase = "source.initial" if self.consumed == 0 else f"source.repair_{self.consumed}"
        request = {"phase": phase, "prompt": prompt, "schema": schema}
        self.requests.append(request)
        if self.stop_after is not None and self.consumed >= self.stop_after:
            raise ReplayStoppedError(f"stopped before {phase}")
        if self.recording is not None:
            path, provider = self.recording
            entry = {**request, "timeout": timeout, "response": None}
            self.packet["calls"].append(entry)
            write_json(path, self.packet)
            result = provider.invoke(prompt, timeout=timeout, filesystem=filesystem, schema=schema)
            entry["response"] = result.text
            entry["usage"] = result.usage
            write_json(path, self.packet)
            self.consumed += 1
            return result
        response = self._replay_response(request)
        self.consumed += 1
        return AgentResult(response, response, "", 0, "replay")

    def _replay_response(self, request: dict[str, Any]) -> str:
        phase = request["phase"]
        calls = self.packet["calls"]
        if self.consumed >= len(calls):
            raise ReplayMismatchError(f"no recorded response for call {self.consumed + 1}: {phase}")
        entry = calls[self.consumed]
        if entry["phase"] != phase:
            raise ReplayMismatchError(f"expected {entry['phase']}, compiler requested {phase}")
        changed = []
        if comparable_prompt(entry["prompt"]) != comparable_prompt(request["prompt"]):
            changed.append("prompt")
        if entry["schema"] != request["schema"]:
            changed.append("schema")
        if changed:
            self.changes.append({"call": self.consumed + 1, "phase": phase, "changed": changed})
            if not self.allow_request_changes:
                raise ReplayMismatchError(f"{phase}: changed {', '.join(changed)}")
        response = entry["response"]
        if not isinstance(response, str):
            raise ReplayMismatchError(f"{phase}: recorded call has no complete response")
        return response


def create_packet(
    skill: Path,
    directory: Path,
    *,
    roles: tuple[str, ...] = ("worker",),
    timeout: int = 900,
    compilation_timeout: int = DEFAULT_COMPILATION_TIMEOUT,
    runtime_wall_seconds: int = 900,
    interface: str = "",
) -> dict[str, Any]:
    """Copy declared inputs once; keep the original package and evidence untouched."""
    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / "inputs"
    inputs.mkdir()
    text = skill.read_bytes().decode("utf-8")
    entries = tuple(discover_resources(skill))
    stage_compiler_resources(inputs, skill, text, entries)
    packet = {
        "version": 1,
        "skill_sha256": sha256_hex(text.encode("utf-8")),
        "resources": [asdict(replace(entry, source=None)) for entry in entries],
        # Match prepare_artifact's compiler input. The source compiler adds
        # staged inline context for generation; extraction receives metadata.
        "resource_manifest": manifest_summary(list(entries)) or "(none)",
        "roles": list(roles),
        "timeout": timeout,
        "compilation_timeout": compilation_timeout,
        "runtime_wall_seconds": runtime_wall_seconds,
        "interface": interface,
        "python_version": "3.13",
        "max_calls": 5,
        "calls": [],
    }
    write_json(directory / "packet.json", packet)
    return packet


def run_packet(directory: Path, output: Path, tape: CompilerTape) -> dict[str, Any]:
    """Run the source compiler with real local checks and an explicit transport."""
    timer = SuspendAwareTimer.start()
    output.mkdir(parents=True, exist_ok=False)
    packet = tape.packet
    if packet["version"] != 1:
        raise ValueError("unsupported compiler replay packet version")
    skill = directory / "inputs" / "SKILL.md"
    text = skill.read_bytes().decode("utf-8")
    if sha256_hex(text.encode("utf-8")) != packet["skill_sha256"]:
        raise ValueError("recorded skill bytes changed")
    resources = tuple(ResourceEntry(**entry) for entry in packet["resources"])
    status, diagnostic = "compiler_accepted", ""
    try:
        context = SourceCompileContext(
            skill_text=text,
            provider=tape,
            resource_manifest=packet["resource_manifest"],
            agent_roles=tuple(packet["roles"]),
            timeout=packet["timeout"],
            runtime_wall_seconds=packet["runtime_wall_seconds"],
            catalog={},
            activity=tape.activity,
            source_path=directory / "destination" / "recipe.py",
            python_version=packet["python_version"],
            resource_paths=frozenset(entry.path for entry in resources),
            skill_path=skill,
            resource_entries=resources,
            interface_requirements=packet["interface"],
            max_calls=packet["max_calls"],
            # Earlier frozen packets used the fixed 900-second allowance.
            deadline=CompilerDeadline(packet.get("compilation_timeout", 900)),
        )
        source = compile_source(context).source
        if tape.recording is None and tape.consumed != len(packet["calls"]):
            raise ReplayMismatchError("compiler accepted before consuming all recorded responses")
        (output / "candidate.py").write_text(source, encoding="utf-8", newline="")
    except (CalcifyError, ReplayMismatchError, ReplayStoppedError) as error:
        status = {
            ReplayMismatchError: "replay_diverged",
            ReplayStoppedError: "prefix_complete",
        }.get(type(error), "compiler_rejected")
        diagnostic = str(error)
        if isinstance(error, CalcifyError) and error.source is not None:
            (output / "rejected.py").write_text(error.source, encoding="utf-8", newline="")
    result = {
        "mode": "record" if tape.recording else "replay",
        "status": status,
        "diagnostic": diagnostic,
        "responses_consumed": tape.consumed,
        "responses_remaining": len(packet["calls"]) - tape.consumed,
        "provider_invocations": len(packet["calls"]) if tape.recording else 0,
        "request_changes": tape.changes,
        "wall_seconds": timer.elapsed(),
        "assurance": "compiler mechanics only; replay does not perform a fresh fidelity review",
    }
    write_json(output / "result.json", result)
    write_json(output / "trace.json", compiler_activity_payload(tape.activity))
    write_json(output / "requests.json", tape.requests)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    record = commands.add_parser("record", help="Compile once, preserving every raw reply")
    record.add_argument("skill", type=Path)
    record.add_argument("directory", type=Path, help="New recording directory outside Git")
    record.add_argument("--agent-config", type=Path, required=True)
    record.add_argument("--timeout", type=int, default=900)
    record.add_argument("--compilation-timeout", type=int, default=DEFAULT_COMPILATION_TIMEOUT)
    record.add_argument("--runtime-wall-seconds", type=int, default=900)
    record.add_argument("--interface", type=Path)
    replay = commands.add_parser("replay", help="Run recorded replies through local checks")
    replay.add_argument("directory", type=Path)
    replay.add_argument("--output", type=Path, required=True, help="New diagnostics directory")
    replay.add_argument("--allow-request-changes", action="store_true")
    replay.add_argument("--stop-after", type=int, help="Replay only this many replies")
    args = parser.parse_args(argv)
    activity = AgentActivity()
    if args.mode == "record":
        if min(args.timeout, args.compilation_timeout, args.runtime_wall_seconds) <= 0:
            parser.error("timeouts must be positive")
        # Replay never reads provider configuration or constructs a live provider.
        from calcifier.agent_config import load_agent_config
        from calcifier.registry import configured_registry

        registry = configured_registry(load_agent_config(args.agent_config), ("compiler",))
        packet = create_packet(
            args.skill,
            args.directory,
            roles=tuple(role for role in registry.known() if role != "compiler"),
            timeout=args.timeout,
            compilation_timeout=args.compilation_timeout,
            runtime_wall_seconds=args.runtime_wall_seconds,
            interface=args.interface.read_bytes().decode("utf-8") if args.interface else "",
        )
        tape = CompilerTape(
            packet, activity, recording=(args.directory / "packet.json", registry.get("compiler"))
        )
        output = args.directory / "recorded"
    else:
        if args.stop_after is not None and args.stop_after < 1:
            parser.error("--stop-after must be positive")
        packet = json.loads((args.directory / "packet.json").read_text(encoding="utf-8"))
        tape = CompilerTape(
            packet,
            activity,
            allow_request_changes=args.allow_request_changes,
            stop_after=args.stop_after,
        )
        output = args.output
    result = run_packet(args.directory, output, tape)
    print(json.dumps(result, indent=2))
    return int(result["status"] not in {"compiler_accepted", "prefix_complete"})


if __name__ == "__main__":
    raise SystemExit(main())
