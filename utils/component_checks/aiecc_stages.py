# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""What the component checks share: compile stages, measured runs and rows.

An aiecc ``--profile-json`` file's edges group into compile stages.

The JIT writes one beside every build it runs aiecc for. Stages:

- ``place``: tile placement (``placed.mlir``, ``aie-place-tiles``, which is
  where the SA placer runs).
- ``route``: routing the placed design's flows (``input_physical.mlir``).
- ``lower``: the rest of getting from the parsed input to a placed, routed,
  addressed module (objectFifo lowering, buffer addresses, each core's
  module), and the runtime-sequence lowering.
- ``kernels``: aiecc blocked until the external kernel objects the cores
  link are compiled (``awaitLinkFiles``); compiling them overlaps the lowering
  before it, so only the wait shows here.
- ``codegen``: each core's program to an object (LLVM IR, Peano).
- ``link``: core ELFs, linker scripts, and the stack/data sizes and bank
  checks read back from them, plus the device cache that restores them.
- ``package``: CDO, PDI, xclbin and instruction stream.
- ``other``: an edge no rule names, so a new one shows up rather than
  vanishing.

Edges run in parallel, so a stage's time (its edges' summed time) can add up,
over the stages, to more than ``wall_s``, the graph's span. An aiecc that
records ``process_ms`` (main() to the profile's writing) also gives its own
time outside the graph: option parsing, building the graph, writing outputs.

Every subprocess a check runs goes through ``run_measured``, which streams its
output as it comes and kills it past a timeout, and, for a compile, measures
its whole process tree's peak memory in a user systemd scope of its own.
Every check writes
``write_unfinished``'s placeholder before its first device run, so a check
stopped part way (a hang, a step timeout) still publishes as a failure.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable
from typing import NamedTuple

STAGES = ("place", "route", "lower", "kernels", "codegen", "link", "package", "other")
# What the JIT names the file, in each build's ``<xclbin stem>.prj`` dir.
PROFILE_NAME = "aiecc_profile.json"
# A compile's rows: each stage's time, aiecc's graph span, and the rest of the
# compile's (Python's import and trace, the dispatch bridge's build, and
# aiecc's own time outside its graph when the profile can't tell it apart).
STAGE_METRICS = (
    *(f"compile_{stage}_s" for stage in STAGES),
    "compile_aiecc_wall_s",
    "compile_outside_aiecc_s",
)
# aiecc's own time outside its graph, from a profile with process_ms.
AIECC_OWN_METRIC = "compile_aiecc_outside_graph_s"
# The wall time of compiling the external kernel objects, from the JIT's
# kernel profile: the "kernels" stage shows only what aiecc waited of it.
KERNEL_OBJECTS_METRIC = "compile_kernel_objects_s"
# How many of the slowest kernel objects a compile's detail names.
SLOWEST_KERNELS = 5
# A cold then warm compile's rows, as every check names them.
COMPILE_METRICS = (
    "compile_s",
    "compile_peak_mib",
    "compile_peak_rss_mb",
    "compile_s_warm",
    "compile_peak_mib_warm",
    "compile_peak_rss_mb_warm",
    *STAGE_METRICS,
    *(f"{m}_warm" for m in STAGE_METRICS),
    AIECC_OWN_METRIC,
    f"{AIECC_OWN_METRIC}_warm",
    KERNEL_OBJECTS_METRIC,
    f"{KERNEL_OBJECTS_METRIC}_warm",
)
# xrt-smi's power modes, and "any" to accept whichever is set.
PMODES = ("any", "default", "powersaver", "balanced", "performance", "turbo")
# The --profile-json format this reads.
PROFILE_VERSION = 1
# What a command killed for running past its timeout exits with, as coreutils'
# timeout(1) does.
TIMED_OUT = 124
# What a warm compile keeps of the cold one's NPU_CACHE_HOME: the kernel
# objects and the precompiled headers they were built with, the two caches an
# edit to the design, not its kernels, leaves valid.
WARM_KEEPS = ("objects", "pch")
# A command in a transient user scope of its own, so its cgroup holds its whole
# process tree and nothing else; --scope runs it in this environment and cwd.
SCOPE = ("systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryAccounting=yes")
# Run from inside the scope: the cgroup's memory.peak, read before the scope
# (and its cgroup) goes away with its last process.
_SCOPE_PEAK = 'cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.peak"'

# First match wins; output names chosen on aiecc's command line (xclbin, pdi,
# instructions) are matched by extension.
_RULES = [
    ("place", r"placed\.mlir"),
    ("route", r"input_physical\.mlir"),
    ("kernels", r"awaitLinkFiles"),
    ("link", r"deviceCacheLookup|preBakedCores|placedCoreCompile"),
    (
        "link",
        r"(probeElfs|elfs|preBakedElfs)_\{0\}\.elf|(probeScripts|ldScripts)_\{0\}\.ld\.script"
        r"|placedCore_\{0\}\.mlir|measured_(stack|data)_sizes\.mlir"
        r"|checked_(bank_placement|lut_banks)\.mlir|physical_with_elfs\.mlir|image_check\.txt"
        r"|linkwith_\{0\}\.txt|\{0\}\.bcf|xchesscc_wrapper",
    ),
    (
        "codegen",
        r"perCoreInDevice|perCoreCompile|(llvmIR|peano-compat|peano-linked|chess-compat|chesslinked|opted)_\{0\}\.ll"
        r"|objects_\{0\}\.o",
    ),
    (
        "lower",
        r"(input|default_stack_size|traced|input_with_symbols|input_with_addresses)\.mlir"
        r"|(perCore|lowered|perDevice|npu_seq|npu_seq_full_elf|ctrlpkt_lowered|ctrlpkt_expanded_seq)_\{0\}\.mlir"
        r"|perCore(StackSpace|Arches|IRLinkFiles)_\{0\}\.txt|npu_[a-z_]+\.mlir|ctrlpkt_[a-z_]+\.mlir"
        r"|perDeviceMatching|perSeq(FullElf)?Matching|sequencePlacement"
        r"|perDevice(NPULowered|NPULoweredFullElf|Compile)(_\{0\}\.mlir|Matching)|perDeviceArches_\{0\}\.txt"
        r"|(noC|c)trlPktSeqs|fullElfCtrlpktNonEmpty",
    ),
    (
        "package",
        r"cdo_\{0\}|.*\.(bif|pdi|xclbin|bin|elf|cpp|h)|(memTopology|kernels|partition|input_aie_partition"
        r"|input_aie_partition_parsed|merged_partition|ctrlpkt_extbuf)_\{0\}\.json|full_elf.*"
        r"|\{0\}_transaction\.mlir|.*\.bin\.locmap\.json",
    ),
]
_COMPILED = [(stage, re.compile(pattern)) for stage, pattern in _RULES]


def stage_of(edge: str) -> str:
    """Return the stage of the aiecc edge named ``edge``."""
    for stage, pattern in _COMPILED:
        if pattern.fullmatch(edge):
            return stage
    return "other"


class ProfileError(Exception):
    """A profile this cannot read as the compile's own."""


def stage_seconds(profile_path: str) -> dict:
    """Return seconds per stage plus ``wall_s``.

    Args:
        profile_path: an aiecc ``--profile-json`` file.

    Returns:
        ``{stage: seconds for each of STAGES, "wall_s": seconds}``, and
        ``"process_s"`` when the profile records aiecc's whole run.

    Raises:
        ProfileError: The file is missing (an aiecc without
            ``--profile-json``) or in a format other than ``PROFILE_VERSION``.
    """
    if not os.path.exists(profile_path):
        raise ProfileError(f"no profile at {profile_path}")
    with open(profile_path) as f:
        profile = json.load(f)
    if profile.get("version") != PROFILE_VERSION:
        raise ProfileError(
            f"{profile_path} is version {profile.get('version')}, "
            f"this reads {PROFILE_VERSION}"
        )
    out: dict[str, float] = dict.fromkeys(STAGES, 0.0)
    for edge in profile["edges"]:
        out[stage_of(edge["name"])] += edge["ms"] / 1000.0
    out["wall_s"] = profile["wall_ms"] / 1000.0
    if "process_ms" in profile:
        out["process_s"] = profile["process_ms"] / 1000.0
    return out


def stage_rows(profile_path: str, compile_s: float | None, suffix: str) -> dict:
    """Return one compile's STAGE_METRICS values, each name ending in ``suffix``.

    Args:
        profile_path: the compile's aiecc ``--profile-json`` file.
        compile_s: the whole compile's wall time, or None if it failed.
        suffix: ``""`` for a cold compile, ``"_warm"`` for a warm one.

    Returns:
        ``{metric + suffix: seconds}``; empty when the compile failed. When
        the profile can't be the compile's (missing, an unknown format, or
        longer than the whole compile, so from another one), no stage rows
        but ``{"profile_problem" + suffix: why}``, which a check records.
    """
    if compile_s is None:
        return {}
    try:
        stages = stage_seconds(profile_path)
        aiecc_s = stages.get("process_s", stages["wall_s"])
        if aiecc_s > compile_s:
            raise ProfileError(
                f"{profile_path} spans {aiecc_s:.2f} s of a "
                f"{compile_s:.2f} s compile: not this compile's"
            )
    except ProfileError as e:
        print(f"no compile stages{suffix}: {e}")
        return {f"profile_problem{suffix}": str(e)}
    seconds = {f"compile_{stage}_s": stages[stage] for stage in STAGES}
    seconds["compile_aiecc_wall_s"] = stages["wall_s"]
    seconds["compile_outside_aiecc_s"] = compile_s - aiecc_s
    if "process_s" in stages:
        seconds[AIECC_OWN_METRIC] = stages["process_s"] - stages["wall_s"]
    return {f"{k}{suffix}": round(v, 2) for k, v in seconds.items()}


def kernel_rows(profile_path: str, compile_s: float | None, suffix: str) -> dict:
    """Return one compile's kernel-object compile time and what took it.

    Args:
        profile_path: the JIT's kernel profile for the compile
            (``aie.utils.compile.utils.KERNEL_PROFILE_NAME``, beside aiecc's).
        compile_s: the whole compile's wall time, or None if it failed.
        suffix: ``""`` for a cold compile, ``"_warm"`` for a warm one.

    Returns:
        ``{KERNEL_OBJECTS_METRIC + suffix: wall seconds, "kernel_objects" +
        suffix: {jobs, compiled, cached, slowest}}``, ``slowest`` naming the
        ``SLOWEST_KERNELS`` objects that took longest; empty when the compile
        failed, and ``{"kernel_profile_problem" + suffix: why}`` without a
        profile.
    """
    if compile_s is None:
        return {}
    if not os.path.exists(profile_path):
        print(f"no kernel compile time{suffix}: no profile at {profile_path}")
        return {f"kernel_profile_problem{suffix}": f"no profile at {profile_path}"}
    with open(profile_path) as f:
        profile = json.load(f)
    kernels = sorted(profile["kernels"], key=lambda k: -k["seconds"])
    return {
        f"{KERNEL_OBJECTS_METRIC}{suffix}": round(profile["wall_s"], 2),
        f"kernel_objects{suffix}": {
            "jobs": profile["jobs"],
            "compiled": sum(not k["cached"] for k in kernels),
            "cached": sum(k["cached"] for k in kernels),
            "slowest": [
                {"object": k["object"], "seconds": round(k["seconds"], 2)}
                for k in kernels[:SLOWEST_KERNELS]
            ],
        },
    }


class Measured(NamedTuple):
    """A finished command and what running it took."""

    # Its merged stdout and stderr in ``stdout``.
    process: subprocess.CompletedProcess
    seconds: float
    # The largest single process of its tree, not their sum.
    peak_rss_mib: float
    # User plus system CPU over its tree.
    cpu_s: float
    # Its whole tree's peak memory (resident and page cache, as its cgroup's
    # memory.peak counts it); None unless asked for and measurable.
    tree_peak_mib: float | None = None
    # Why tree_peak_mib was asked for and is None.
    tree_peak_unavailable: str | None = None


def scope_unavailable(env: dict | None) -> str | None:
    """Return why a command can't run in a memory-accounted user scope, or None.

    Args:
        env: The environment the command would run in; ``systemd-run --user``
            finds the user's manager through its XDG_RUNTIME_DIR.

    Returns:
        None when a probe in such a scope reads its cgroup's memory.peak,
        else what stopped it.
    """
    if shutil.which("systemd-run", path=(env or os.environ).get("PATH")) is None:
        return "systemd-run is not on PATH"
    probe = subprocess.run(
        [*SCOPE, "--", "sh", "-c", _SCOPE_PEAK],
        env=env,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip().isdigit():
        said = (probe.stderr or probe.stdout).strip() or f"exit {probe.returncode}"
        return f"{' '.join(SCOPE)}: {said}"
    return None


def run_measured(
    cmd: list[str],
    cwd: str | None,
    env: dict | None,
    timeout: float,
    echo: bool = True,
    tree_peak: bool = False,
) -> Measured:
    """Run ``cmd`` with a timeout, echoing its output as it comes.

    stdout and stderr are merged into one pipe, read as the command writes
    it and, with ``echo``, written to this process's stdout line by line, so
    a check's log shows how far a command got even if the check is killed.
    Past ``timeout`` the command's whole process group is killed, its status
    is ``TIMED_OUT`` and its output ends with a line saying so.

    Its peak RSS and CPU time are os.wait4's, so they are this command's:
    resource.getrusage(RUSAGE_CHILDREN) keeps a running max of RSS over every
    child this process ever reaped. With ``tree_peak`` it runs in a user
    systemd scope of its own, whose cgroup's memory.peak is the whole tree's
    peak; where no such scope can be made, it runs as without and says why.

    Args:
        cmd: The command line.
        cwd: Where to run it; None for here.
        env: Its environment; None for this process's.
        timeout: Seconds before it is killed.
        echo: Whether to echo its output.
        tree_peak: Whether to measure its whole tree's peak memory.

    Returns:
        The finished command and its wall seconds, peak RSS, CPU time and,
        with ``tree_peak``, its tree's peak memory or why there is none.
    """
    unavailable = scope_unavailable(env) if tree_peak else None
    if unavailable:
        print(f"no whole-tree peak memory: {unavailable}", flush=True)
    peak_file = None
    run = cmd
    if tree_peak and not unavailable:
        peak_file = tempfile.NamedTemporaryFile(prefix="tree-peak-", delete=False).name
        run = [
            *SCOPE,
            "--",
            "sh",
            "-c",
            f'f=$1; shift; "$@"; rc=$?; {_SCOPE_PEAK} > "$f"; exit $rc',
            "sh",
            peak_file,
            *cmd,
        ]
    start = time.perf_counter()
    proc = subprocess.Popen(
        run,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    lines: list[str] = []

    def drain():
        for line in proc.stdout:
            lines.append(line)
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()

    fired = threading.Event()

    def kill():
        fired.set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    killer = threading.Timer(timeout, kill)
    killer.start()
    _, status, ru = os.wait4(proc.pid, 0)
    seconds = time.perf_counter() - start
    killer.cancel()
    returncode = os.waitstatus_to_exitcode(status)
    timed_out = fired.is_set() and returncode == -signal.SIGKILL
    reader.join(10)
    if reader.is_alive():
        # A child it left behind still holds the pipe open.
        kill()
        reader.join()
    if timed_out:
        returncode = TIMED_OUT
        note = f"{' '.join(map(str, cmd))}: timed out after {timeout:g} s\n"
        lines.append(note)
        print(note, end="", flush=True)
    tree_peak_mib = None
    if peak_file is not None:
        with open(peak_file) as f:
            peak = f.read().strip()
        os.remove(peak_file)
        if peak.isdigit():
            tree_peak_mib = int(peak) / 2**20
        else:
            unavailable = "the command ended before its scope's memory.peak was read"
            print(f"no whole-tree peak memory: {unavailable}", flush=True)
    return Measured(
        subprocess.CompletedProcess(cmd, returncode, "".join(lines), ""),
        seconds,
        ru.ru_maxrss / 1024.0,
        ru.ru_utime + ru.ru_stime,
        tree_peak_mib,
        unavailable,
    )


def compile_stats(cold: Measured, warm: Measured) -> dict:
    """Return a cold and a warm compile's times and memory, None where one failed.

    Args:
        cold: ``run_measured``'s result for the cold compile, with ``tree_peak``.
        warm: Its result for the warm compile.

    Returns:
        compile_s, compile_peak_mib (the whole tree's peak, None where it
        couldn't be measured), compile_peak_rss_mb (the largest process's)
        and their ``_warm`` twins, and ``compile_peak_unavailable``: why a
        compile that ran has no compile_peak_mib, else None.
    """
    stats: dict = {"compile_peak_unavailable": None}
    for suffix, m in (("", cold), ("_warm", warm)):
        ok = m.process.returncode == 0
        stats[f"compile_s{suffix}"] = (
            round(m.seconds, 2 if suffix else 1) if ok else None
        )
        stats[f"compile_peak_mib{suffix}"] = (
            round(m.tree_peak_mib, 1) if ok and m.tree_peak_mib is not None else None
        )
        stats[f"compile_peak_rss_mb{suffix}"] = round(m.peak_rss_mib, 1) if ok else None
        if ok and m.tree_peak_mib is None:
            stats["compile_peak_unavailable"] = (
                m.tree_peak_unavailable or "the compile ran without tree_peak"
            )
    return stats


def row(case: str, metric: str, value, unit: str, spread: str | None = None) -> dict:
    """Return one perf.json row, ``case/metric``, with an optional ``range`` note."""
    out = {"name": f"{case}/{metric}", "unit": unit, "value": value}
    if spread is not None:
        out["range"] = spread
    return out


def compile_rows(case: str, stats: dict) -> list[dict]:
    """Return ``case``'s rows for each COMPILE_METRICS value ``stats`` holds.

    Args:
        case: The row names' prefix.
        stats: The compile's values, by metric name; a missing or None one
            gets no row.

    Returns:
        The rows, in COMPILE_METRICS order.
    """
    return [
        row(case, metric, stats[metric], "MiB" if "_peak_" in metric else "s")
        for metric in COMPILE_METRICS
        if stats.get(metric) is not None
    ]


def clear_for_warm(cache_dir: str, outputs: Iterable[str] = ()) -> None:
    """Clear what a warm compile must redo: everything but ``WARM_KEEPS``.

    Args:
        cache_dir: The cold compile's NPU_CACHE_HOME.
        outputs: Files or directories the compile wrote outside it (an
            explicit output and its work dir), removed too.
    """
    doomed = [
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name not in WARM_KEEPS
    ]
    for path in [*doomed, *outputs]:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.lexists(path):
            os.remove(path)


def write(path: str, data) -> None:
    """Write ``data`` to ``path`` as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def write_unfinished(meta: dict, case: str, out_path: str, meta_path: str) -> None:
    """Record ``case`` as failed, for the check's results to overwrite.

    Written before a check's first device run: if the check never finishes
    (a hang, a step timeout), its record publishes as a failure that says
    so, not as a night with no run.

    Args:
        meta: The run's meta so far.
        case: The case to fail.
        out_path: The perf.json to write, empty.
        meta_path: The meta.json to write.
    """
    write(out_path, [])
    write(
        meta_path,
        {**meta, "failed": [case], "exitstatus": 1, "reason": "did not finish"},
    )


def cpu_model() -> str:
    """Return the CPU's model name, from /proc/cpuinfo, or "unknown"."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() == "model name":
                    return value.strip()
    except OSError:
        pass
    return "unknown"


def add_pmode_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--pmode``, the power mode a check refuses to measure outside of."""
    parser.add_argument(
        "--pmode",
        choices=PMODES,
        default="any",
        help="refuse to measure unless the NPU is in this power mode (default: any)",
    )


def refuse(reason: str, meta: dict, out_path: str, meta_path: str) -> int:
    """Record why a check did not measure, write empty results, and fail.

    Args:
        reason: Why, for the meta's ``refused``.
        meta: The run's meta so far.
        out_path: The perf.json to write, empty.
        meta_path: The meta.json to write.

    Returns:
        The exit status, 1.
    """
    print(f"refusing to measure: {reason}")
    write(out_path, [])
    write(meta_path, {**meta, "refused": reason, "failed": [], "exitstatus": 1})
    return 1
