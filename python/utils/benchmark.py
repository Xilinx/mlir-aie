# benchmark.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Benchmarking helpers for NPU kernel callables.

``Stats`` keeps the raw sample list and robust statistics (median, MAD,
p95, coefficient of variation, the median's standard error) beside
avg/min/max, so callers can report jitter, not just central tendency. The
numbers are numpy's.

``run_iters`` times one callable; ``run_interleaved`` times several against
each other, round by round, so a drift in the host or the NPU lands on all of
them alike. ``wait_idle`` holds a measurement until no other process uses the
NPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Hashable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Callable

import numpy as np
from aie.utils import config
from aie.utils.probe import xrt_smi_path


@dataclass
class Stats:
    avg_us: float
    min_us: float
    max_us: float
    median_us: float = 0.0
    mad_us: float = 0.0  # median absolute deviation (robust spread)
    p95_us: float = 0.0
    cov: float = 0.0  # stdev / mean; 0 for a single sample
    n: int = 0
    # sqrt(pi / 2) * stdev / sqrt(n), the median's under normal noise; 0 for a single sample
    median_se_us: float = 0.0
    samples_us: list[float] = field(default_factory=list, repr=False)

    @classmethod
    def from_samples(cls, samples_us: list[float]) -> "Stats":
        if not samples_us:
            raise ValueError("Stats.from_samples needs at least one sample")
        s = np.asarray(samples_us, dtype=np.float64)
        med = float(np.median(s))
        mean = float(s.mean())
        # p95 is a sample, not an interpolation between two: at benchmark
        # sample counts the interpolating percentile methods report a duration
        # no run took. inverted_cdf is numpy's name for that (nearest-rank)
        # definition, and every published chart was recorded with it.
        return cls(
            avg_us=mean,
            min_us=float(s.min()),
            max_us=float(s.max()),
            median_us=med,
            mad_us=float(np.median(np.abs(s - med))),
            p95_us=float(np.percentile(s, 95, method="inverted_cdf")),
            cov=float(s.std(ddof=1) / mean) if len(s) > 1 and mean > 0 else 0.0,
            n=len(s),
            median_se_us=(
                math.sqrt(math.pi / 2) * float(s.std(ddof=1)) / math.sqrt(len(s))
                if len(s) > 1
                else 0.0
            ),
            samples_us=list(samples_us),
        )

    def as_dict(self, prefix: str = "") -> dict:
        """Flat dict for JSON emission (samples omitted)."""
        return {
            f"{prefix}median_us": self.median_us,
            f"{prefix}mad_us": self.mad_us,
            f"{prefix}p95_us": self.p95_us,
            f"{prefix}min_us": self.min_us,
            f"{prefix}max_us": self.max_us,
            f"{prefix}avg_us": self.avg_us,
            f"{prefix}cov": self.cov,
            f"{prefix}n": self.n,
        }


@dataclass
class BenchmarkResult:
    e2e: Stats
    npu: Stats | None  # None when the callable does not expose NPU time


def _stats(samples_us: list[float]) -> Stats:
    return Stats.from_samples(samples_us)


def _extract_npu_time_ns(ret) -> int | None:
    """Pull npu_time (ns) from a kernel callable's return value.

    Supports the ``(handle, KernelResult)`` tuple returned by
    ``HostRuntime.load_and_run`` (and therefore by ``NPUKernel.__call__`` and
    ``@iron.jit`` callables). Returns ``None`` if the shape doesn't match,
    so the helper degrades to e2e-only timing for non-kernel callables.
    """
    candidate = ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else ret
    return getattr(candidate, "npu_time", None)


def run_iters(
    fn: Callable,
    *args,
    warmup: int = 0,
    iters: int = 1,
    arg_sets: list[tuple] | None = None,
    **kwargs,
) -> BenchmarkResult:
    """Invoke ``fn`` ``warmup + iters`` times, reporting timings.

    End-to-end latency is measured around the Python call. If the return
    value carries an ``npu_time`` (nanoseconds, captured by the runtime
    around ``kernel.wait()``), it is reported separately so callers can see
    the host-side overhead delta.

    ``arg_sets``: an optional list of positional-argument tuples to
    rotate through, one per iteration, so consecutive runs do not hit the
    same host buffers (buffer rotation; see the CUTLASS measurement
    guidelines). When given, ``*args`` must be empty.
    """
    if iters < 1:
        raise ValueError(f"iters must be >= 1 (got {iters})")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0 (got {warmup})")
    if arg_sets is not None and args:
        raise ValueError("pass either *args or arg_sets, not both")
    if arg_sets is not None and not arg_sets:
        raise ValueError("arg_sets must hold at least one argument tuple")
    rotation = arg_sets if arg_sets is not None else [args]

    e2e_samples: list[float] = []
    npu_samples: list[float] = []
    for i in range(warmup + iters):
        call_args = rotation[i % len(rotation)]
        start = time.perf_counter()
        ret = fn(*call_args, **kwargs)
        e2e_us = (time.perf_counter() - start) * 1_000_000
        if i < warmup:
            continue
        e2e_samples.append(e2e_us)
        npu_ns = _extract_npu_time_ns(ret)
        if npu_ns is not None:
            npu_samples.append(npu_ns / 1_000.0)

    return BenchmarkResult(
        e2e=_stats(e2e_samples),
        npu=_stats(npu_samples) if npu_samples else None,
    )


# Kernel work delays the timing thread tens of microseconds; another process,
# under lazy preemption, hundreds.
CPU_WAIT_US = 100.0


def cpu_wait_us() -> float | None:
    """How long the calling thread has waited, runnable, for a CPU, microseconds in all.

    Read from ``/proc/thread-self/schedstat``.

    Returns:
        float | None: The run delay so far, or None where the kernel does not
        report it (not Linux, or schedstats off).
    """
    try:
        return int(Path("/proc/thread-self/schedstat").read_text().split()[1]) / 1e3
    except OSError:
        return None


def other_npu_users(holding: bool = False) -> set[int]:
    """Return the pids of the other processes holding a context on the NPU.

    Read from ``xrt-smi examine -r aie-partitions``, which reports pids in the
    host's namespace. On the host this process finds its own pid there; in a
    container it cannot, and none of the reported pids is a process it sees.
    That test reads ``/proc``, so it assumes ``/proc`` is not mounted with
    ``hidepid``, which would hide other users' processes on the host too.

    Args:
        holding: Whether this process holds a context. In a container, a lone
            reported pid is then taken to be its own; several are all
            returned.

    Returns:
        The other holders' pids, in the host's namespace.

    Raises:
        FileNotFoundError: This host has no ``xrt-smi``.
        subprocess.CalledProcessError: ``xrt-smi`` failed.
    """
    with tempfile.TemporaryDirectory() as d:
        report = Path(d) / "partitions.json"
        subprocess.run(
            [xrt_smi_path() or "xrt-smi", "examine", "-r", "aie-partitions"]
            + ["-f", "JSON", "-o", str(report)],
            capture_output=True,
            check=True,
        )
        devices = json.loads(report.read_text())["devices"]
    pids = {
        int(context["pid"])
        for device in devices
        for partition in device["aie_partitions"].get("partitions", [])
        for context in partition.get("hw_contexts", [])
    }
    if os.getpid() in pids:
        return pids - {os.getpid()}
    namespaced = not any(Path(f"/proc/{pid}").exists() for pid in pids)
    if holding and namespaced and len(pids) == 1:
        return set()
    return pids


# Seconds between looks at an NPU another process holds, and the most waited.
IDLE_POLL = 5.0
IDLE_WAIT = 1800.0


def wait_idle(log: Callable[[str], None] = print, holding: bool = False) -> None:
    """Return once no other process holds a context on the NPU (``other_npu_users``).

    Args:
        log: Where the wait is reported.
        holding: Whether this process holds a context, as ``other_npu_users``
            takes it.

    Raises:
        TimeoutError: Another has held one for ``IDLE_WAIT`` seconds. Two
            processes measuring at once would each wait for the other.
        FileNotFoundError: This host has no ``xrt-smi``.
        subprocess.CalledProcessError: ``xrt-smi`` failed.
    """
    held, told = other_npu_users(holding), set()
    start = time.monotonic()
    while held:
        if held != told:
            log(f"the NPU is in use by pid {sorted(held)}: waiting")
            told = held
        if time.monotonic() - start > IDLE_WAIT:
            raise TimeoutError(
                f"pid {sorted(held)} has held the NPU for {IDLE_WAIT:.0f} s; "
                "measure once it is idle"
            )
        time.sleep(IDLE_POLL)
        held = other_npu_users(holding)


class CompletionCores(AbstractContextManager):
    """While entered, the calling thread held on the CPUs the NPU's completions interrupted.

    A core in deep idle adds its exit latency to long runs but not short ones.
    A nice-19 spinner on each of those CPUs' SMT siblings keeps the core
    shallow. Linux only; elsewhere nothing is held.

    Args:
        before (Mapping[str, int]): ``interrupts()`` before the calls whose
            completions choose the CPUs.
    """

    IRQ = "xdna_mailbox"

    def __init__(self, before: Mapping[str, int]):
        self.cpus: set[int] = set()
        for irq, count in self.interrupts().items():
            if count > before.get(irq, 0):
                self.cpus |= self._cpus(f"/proc/irq/{irq}/effective_affinity_list")
        # A container's cpuset may leave out the cores the completions
        # interrupt: only the ones the process may run on can be held.
        allowed = os.sched_getaffinity(0) if self.cpus else set()
        self.cpus &= allowed
        self.siblings: set[int] = set()
        for cpu in self.cpus:
            self.siblings |= self._cpus(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            )
        self.siblings = (self.siblings & allowed) - self.cpus
        self._affinity: set[int] | None = None
        self._spinners: list[subprocess.Popen] = []

    @classmethod
    def interrupts(cls) -> dict[str, int]:
        """Each NPU completion interrupt's count over every CPU, by IRQ number.

        Returns:
            dict[str, int]: Empty where ``/proc/interrupts`` does not exist.
        """
        path = Path("/proc/interrupts")
        if not path.is_file():
            return {}
        counts = {}
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[-1] == cls.IRQ:
                counts[fields[0].rstrip(":")] = sum(
                    int(f) for f in fields[1:] if f.isdigit()
                )
        return counts

    @staticmethod
    def _cpus(path: str) -> set[int]:
        cpus = set()
        for part in Path(path).read_text().strip().split(","):
            first, _, last = part.partition("-")
            cpus.update(range(int(first), int(last or first) + 1))
        return cpus

    def __enter__(self) -> CompletionCores:
        try:
            if self.cpus:
                self._affinity = os.sched_getaffinity(0)
                os.sched_setaffinity(0, self.cpus)
            for cpu in sorted(self.siblings):
                spinner = subprocess.Popen([sys.executable, "-c", "while True: pass"])
                self._spinners.append(spinner)
                os.sched_setaffinity(spinner.pid, {cpu})
                os.setpriority(os.PRIO_PROCESS, spinner.pid, 19)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        for spinner in self._spinners:
            spinner.kill()
            spinner.wait()
        self._spinners.clear()
        if self._affinity is not None:
            os.sched_setaffinity(0, self._affinity)
            self._affinity = None


def run_interleaved(
    fns: Sequence[Callable[[], object]],
    *,
    rounds: int = 8,
    calls: int = 50,
    settle: int = 2,
    cutoff: float = 1.5,
    groups: Sequence[Hashable | None] | None = None,
    fastest_us: float = math.inf,
    completion_cores: bool = False,
    max_cpu_wait_us: float = CPU_WAIT_US,
    idle: bool = False,
    log: Callable[[str], None] = print,
) -> list[Stats]:
    """Time callables against each other in interleaved rounds.

    Each round runs every callable still timed ``calls`` times in turn and
    keeps each one's median call. A call's time is the ``npu_time`` its
    return value carries (``run_iters``'s rule), else the time around it.
    Each callable is called twice first, untimed, so first-run setup lands on
    nobody's figure.

    Args:
        fns: The callables, each taking no arguments.
        rounds: Rounds to time.
        calls: Calls per callable per round.
        settle: Rounds after which `cutoff` applies.
        cutoff: With `groups`, a group whose slowest callable takes more than
            `cutoff` times the fastest group's is timed no further; its
            figure is from the rounds it had.
        groups: Per callable, the setting it measures (callables of one group
            are stopped together), or None for one never stopped. Every
            callable is timed every round if not given.
        fastest_us: The fastest group's slowest median from callables timed
            before these, which `cutoff` also measures against.
        completion_cores: Hold the timing thread on the CPUs the callables'
            NPU completions interrupt (`CompletionCores`), for callables that
            dispatch to the NPU.
        max_cpu_wait_us: The most a callable's median call may wait for a CPU
            in its median round before the figures are refused; ``math.inf``
            accepts figures from a busy host.
        idle: Time only while no other process holds the NPU: wait for it to
            be idle first (`wait_idle`), and time a round again if another
            holds it when the round ends. A use that starts and ends within
            one round goes unseen, and one held but idle context stops the
            measurement with `wait_idle`'s ``TimeoutError``.
        log: Where `idle` reports waiting or timing a round again.

    Returns:
        list[Stats]: Per callable, the statistics of its per-round medians.

    Raises:
        TimeoutError: With `idle`, another process held the NPU for
            ``IDLE_WAIT`` seconds.
        RuntimeError: A callable's median call waited for a CPU past
            `max_cpu_wait_us` (`cpu_wait_us`) in its median round: the host was
            busy enough to hold the timing thread off its core, so the
            figures are refused. Measure with the host otherwise idle.
        ValueError: `rounds` or `calls` is below one, or `groups` is not one
            per callable.
    """
    if rounds < 1 or calls < 1:
        raise ValueError(f"{rounds} rounds of {calls} calls: both must be positive")
    if groups is not None and len(groups) != len(fns):
        raise ValueError(f"{len(groups)} groups for {len(fns)} callables")
    for fn in fns:
        fn()
    before = CompletionCores.interrupts()
    for fn in fns:
        fn()
    medians: list[list[float]] = [[] for _ in fns]
    waited: list[list[float]] = [[] for _ in fns]
    live = set(range(len(fns)))
    if idle:
        wait_idle(log, holding=True)
    r = 0
    with CompletionCores(before) if completion_cores else nullcontext():
        while r < rounds:
            timed = {}
            for i in sorted(live):
                times, waits = [], []
                for _ in range(calls):
                    wait = cpu_wait_us()
                    start = time.perf_counter()
                    ret = fns[i]()
                    elapsed_us = (time.perf_counter() - start) * 1e6
                    waited_after = cpu_wait_us()
                    if wait is not None and waited_after is not None:
                        waits.append(waited_after - wait)
                    npu_ns = _extract_npu_time_ns(ret)
                    times.append(elapsed_us if npu_ns is None else npu_ns / 1e3)
                timed[i] = (
                    float(np.median(times)),
                    float(np.median(waits)) if waits else 0.0,
                )
            if idle and (held := other_npu_users(holding=True)):
                log(
                    f"    pid {sorted(held)} used the NPU during a round: timing it again"
                )
                wait_idle(log, holding=True)
                continue
            for i, (us, wait) in timed.items():
                medians[i].append(us)
                waited[i].append(wait)
            r += 1
            if groups is None or r < settle:
                continue
            slowest: dict[Hashable | None, float] = {}
            for i in live:
                us = float(np.median(medians[i]))
                slowest[groups[i]] = max(slowest.get(groups[i], 0.0), us)
            bar = cutoff * min([fastest_us, *slowest.values()])
            live = {i for i in live if groups[i] is None or slowest[groups[i]] <= bar}
    late = max(float(np.median(w)) for w in waited)
    if late > max_cpu_wait_us:
        raise RuntimeError(
            f"a callable's median call waited {late:.0f} us for a CPU in its "
            "median round; time with the host otherwise idle"
        )
    return [Stats.from_samples(m) for m in medians]


def print_benchmark(result: BenchmarkResult) -> None:
    """Print a BenchmarkResult in the canonical 'avg/min/max us' format."""
    if result.npu is not None:
        s = result.npu
        print(
            f"NPU time     (avg/min/max us): {s.avg_us:.1f} / {s.min_us:.1f} / {s.max_us:.1f}"
            f"   [median {s.median_us:.1f}, MAD {s.mad_us:.1f}, p95 {s.p95_us:.1f}]"
        )
    s = result.e2e
    print(
        f"End-to-end   (avg/min/max us): {s.avg_us:.1f} / {s.min_us:.1f} / {s.max_us:.1f}"
        f"   [median {s.median_us:.1f}, MAD {s.mad_us:.1f}, p95 {s.p95_us:.1f}]"
    )


def peano_version() -> str | None:
    """Return the Peano that compiles the kernels, as ``21.0.0+c9c5ecb7``.

    Asked of the compiler itself rather than read from the ``llvm-aie``
    distribution: ``PEANO_INSTALL_DIR`` can select any build, and the
    installed wheel then names a compiler that never ran. ``None`` when no
    compiler is found.
    """
    try:
        out = subprocess.run(
            [config.peano_cxx_path(), "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        return None
    m = re.match(r"clang version (\S+) \(\S+ ([0-9a-f]{8})", out)
    if m:
        return f"{m[1]}+{m[2]}"
    return out.splitlines()[0] if out else None


def kernel_tree_digest() -> str | None:
    """Return a 12-hex digest of the kernel sources the library factories compile.

    Every file under ``aie_kernels_dir()`` and ``aie_runtime_lib_dir()``, by
    relative path and content. The commit alone cannot say which kernels
    ran: ``MLIR_AIE_KERNEL_SOURCES`` can name another tree, and a checkout
    can carry uncommitted edits. ``None`` when neither directory exists.
    """
    h = hashlib.sha256()
    found = False
    for top in (config.aie_kernels_dir(), config.aie_runtime_lib_dir()):
        root = Path(top)
        if not root.is_dir():
            continue
        found = True
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            h.update(f"{root.name}/{path.relative_to(root)}\0".encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:12] if found else None


def provenance(**extra: str | None) -> str:
    """Return a one-line description of what produced a measurement.

    The git commit (``GITHUB_SHA`` or ``git rev-parse HEAD``), the Peano that
    compiles the kernels (``peano_version()``), the installed ``mlir_aie``
    version, the kernel tree
    (``MLIR_AIE_KERNEL_SOURCES`` when set, and ``kernel_tree_digest()``),
    the selected runtime's provenance, the runner the job ran
    on (``RUNNER_NAME``, under GitHub Actions: nightly runs move between
    hosts of a generation, and a host swap moves ``npu_us`` as a regression
    would), and any ``extra`` fields (``device="NPU Strix"``,
    ``pmode="performance"``) as ``key value`` pairs. A benchmark row records
    it so a number can be traced to a toolchain, the kernel sources and
    the machine.

    A package that is not installed is left out rather than recorded as
    unknown. CI builds ``mlir_aie`` from source and puts it on ``PYTHONPATH``,
    so there is no distribution to read a version from, and every published
    row would otherwise carry a word that reads like a lookup failure. The
    commit already identifies that build.
    """
    import aie.utils as aie_utils

    def pkg(name: str) -> str | None:
        try:
            return _pkg_version(name)
        except PackageNotFoundError:
            return None

    commit = os.environ.get("GITHUB_SHA")
    if not commit:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
    runtime_fields = {}
    try:
        runtime = aie_utils.DefaultNPURuntime
        if runtime is not None:
            runtime_fields = runtime.provenance()
    except Exception:  # noqa: BLE001 - optional metadata must not fail a measurement
        pass
    fields = {
        "commit": commit[:10],
        "peano": peano_version(),
        "mlir_aie": pkg("mlir_aie"),
        "kernel_sources": os.environ.get("MLIR_AIE_KERNEL_SOURCES"),
        "kernels": kernel_tree_digest(),
        **runtime_fields,
        "host": os.environ.get("RUNNER_NAME"),
        **extra,
    }
    return " | ".join(f"{k} {v}" for k, v in fields.items() if v)


@dataclass(frozen=True)
class Preflight:
    """What the runtime says about the device a run is about to use."""

    npu: str  # "npu1" / "npu2"
    arch: str  # "aie2" / "aie2p"
    device: str  # the device as the runtime describes it
    pmode: str | None  # power mode, or None if it could not be read


def preflight() -> Preflight:
    """Describe the device through the host runtime (any backend)."""
    import aie.utils as aie_utils
    from aie.utils.compile.utils import resolve_target_arch

    runtime = aie_utils.DefaultNPURuntime
    if runtime is None:
        raise RuntimeError("no NPU runtime is available (XRT, HRX or HSA)")
    device = runtime.device()
    arch = resolve_target_arch(device)
    return Preflight(
        npu="npu2" if arch == "aie2p" else "npu1",
        arch=arch,
        device=runtime.device_name() or type(device).__name__,
        pmode=runtime.power_mode(),
    )
