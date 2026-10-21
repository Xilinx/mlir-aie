#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Run a router hazard case on an NPU and record what each routing did.

    python utils/router_testing/router_hw.py run utils/router_testing/hw_cases/<case>.py
        [--variant NAME ...] [--runs 20] [--device npu1]
        [--verdicts test/create-packet-flows/nightly/Inputs/hw_verdicts]

A case (hw_case.HwCase) is one design and the ways to route it: as the
router does, and pinned onto shared arbiters or links that should or should
not deadlock. Cases target npu2; --device npu1 runs one on an NPU1, its
variants named <variant>_npu1. Each variant is routed with aie-opt, compiled with aiecc, and
run with NPUKernel and IRON tensors, --runs times, each run in a process of
its own. A run that completes has its outputs checked against the case's
reference; one the driver ends with a timeout (amdxdna's TDR) is a HANG, and
one it ends in any other state an ERROR. The variant's verdict is ERROR if any
run erred, HANG if any hung, FAIL if any output was wrong, and PASS
otherwise, and the command exits 1 when a verdict differs from what the
variant expects.

A hang can wedge the driver (every later open of the device blocks), so the
variants expected to hang run last, and every variant stops at its first
hang or error, expected or not. A run whose
process outlives --process-timeout-s is taken for a wedged device: the
command names it and stops, exit 2, leaving the reload or reboot that
recovers the device to whoever runs it.

With --verdicts, each variant is written as <case>_<variant>.mlir in the
format test/create-packet-flows/nightly/hw_verdicts.py checks the deadlock
rules against: the HW line, the design as the router gets it, then the
switchboxes the run used.

Needs an NPU with the IRON host runtime, aie-opt and aiecc on PATH, and
Peano (--peano, default $PEANO_INSTALL_DIR).
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

from aie import iron
from aie.ir import Context, Location, Module
import aie.dialects.aie  # noqa: F401
import aie.dialects.aiex  # noqa: F401
import aie.utils
import pyxrt
from aie.utils import NPUKernel
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTKernelResult

from hw_case import HwCase, Variant

ROUTING_OPS = ("aie.tile", "aie.switchbox", "aie.shim_mux", "aie.end")

# Repro commands are pasted into the platform's shell: cmd.exe on Windows,
# which does not take POSIX quoting, nor a bare cd to another drive.
SHELL_JOIN = subprocess.list2cmdline if os.name == "nt" else shlex.join
CD = "cd /d" if os.name == "nt" else "cd"


def load_case(path: Path) -> HwCase:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Case()


def route(variant: Variant, out: Path, timeout_s: float) -> tuple[str, str]:
    """Route the variant as the router would; returns (routed, diagnostics)."""
    text = variant.pinned if variant.pinned is not None else variant.design
    # Written before aie-opt runs, so a hang or a crash still leaves the
    # input and the exact command that reproduces it, not just a message.
    (out / "input.mlir").write_text(text)
    opt = f"--aie-create-pathfinder-flows={variant.options}"
    repro = SHELL_JOIN(["aie-opt", opt, str(out / "input.mlir")])
    try:
        p = subprocess.run(
            ["aie-opt", opt, "-"],
            input=text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{variant.name}: aie-opt did not finish within {timeout_s} s: {repro}"
        ) from None
    if p.returncode:
        raise RuntimeError(f"{variant.name}: the router failed:\n{p.stderr}\n{repro}")
    (out / "routed.mlir").write_text(p.stdout)
    return p.stdout, p.stderr


def routing_only(routed: str) -> str:
    """The routed module with every op but the tiles, switchboxes and shim
    muxes taken out, as hw_verdicts files keep it."""
    with Context(), Location.unknown():
        module = Module.parse(routed)
        for device in module.body.operations:
            block = device.regions[0].blocks[0]
            for op in reversed(list(block.operations)):
                if op.operation.name not in ROUTING_OPS:
                    op.operation.erase()
        return str(module)


def compile_variant(out: Path, peano: str, timeout_s: float) -> None:
    # routed.mlir is already on disk (route() wrote it), so this is enough on
    # its own to reproduce a hang or a crash.
    args = [
        "aiecc",
        f"--peano={peano}",
        "--get-xclbin",
        "--get-npu-insts",
        "--xclbin-name=aie.xclbin",
        "--npu-insts-name=insts.bin",
        "routed.mlir",
    ]
    repro = f"{CD} {SHELL_JOIN([str(out)])} && {SHELL_JOIN(args)}"
    try:
        p = subprocess.run(
            args, cwd=out, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"aiecc did not finish within {timeout_s} s: {repro}"
        ) from None
    if p.returncode:
        raise RuntimeError(f"aiecc failed:\n{p.stderr[-4000:]}\n{repro}")


def run_once(case: HwCase, out: Path) -> dict:
    """One run of the compiled variant in `out`: its outcome and mismatches."""
    # Created here, in the run's own process, so `run` never opens the device.
    runtime = aie.utils.DefaultNPURuntime
    handle = runtime.load(NPUKernel(out / "aie.xclbin", out / "insts.bin"))
    outputs = [iron.full(count, -1, dtype=dtype) for count, dtype in case.outputs]
    result = runtime.run(handle, outputs, fail_on_error=False)
    device = runtime.device_name()
    if not result.is_success():
        state = result.ret if isinstance(result, XRTKernelResult) else None
        # amdxdna's TDR ends a hung run as a timeout; any other end is no hang.
        hung = state == pyxrt.ert_cmd_state.ERT_CMD_STATE_TIMEOUT
        outcome = "HANG" if hung else "ERROR"
        return {"outcome": outcome, "state": f"ended {state}", "device": device}
    problems = case.check([t.numpy() for t in outputs])
    outcome = "FAIL" if problems else "PASS"
    return {"outcome": outcome, "problems": problems[:5], "device": device}


def run_isolated(case_path: Path, out: Path, timeout_s: int) -> dict:
    """run_once in a process of its own. One that does not exit is stuck in
    the driver, where it cannot be killed, so it is not waited for."""
    proc = subprocess.Popen(
        [sys.executable, __file__, "once", str(case_path), str(out)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"outcome": "WEDGED", "state": f"process {proc.pid} did not exit"}
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    if not lines:
        return {"outcome": "ERROR", "state": (stderr or stdout)[-2000:]}
    return json.loads(lines[-1])


def verdict_of(results: list[dict]) -> str:
    outcomes = {r["outcome"] for r in results}
    for outcome in ("ERROR", "HANG", "FAIL"):
        if outcome in outcomes:
            return outcome
    return "PASS"


def verdict_file(
    case: HwCase,
    variant: Variant,
    routed: str,
    verdict: str,
    counts: Counter,
    device: str | None,
) -> str:
    observed = ", ".join(f"{n} {o}" for o, n in sorted(counts.items()))
    head = [f"// HW: {verdict}"]
    if "circuit-switch-hops=false" in variant.options:
        head.append("// hops: off")
    head.append(
        f"// router_hw.py {case.name}, {variant.name}: {variant.note} "
        f"Observed on {device or 'an NPU of unknown model'}: {observed}."
    )
    return (
        "\n".join(head)
        + "\n"
        + variant.design.strip()
        + "\n// -----\n\n"
        + routing_only(routed).strip()
        + "\n"
    )


def run_case(args) -> int:
    case = load_case(args.case)
    all_variants = case.variants()
    unknown = sorted(set(args.variant) - {v.name for v in all_variants})
    if unknown:
        print(f"{case.name}: no such variant {', '.join(unknown)}", file=sys.stderr)
        return 2
    variants = [v for v in all_variants if not args.variant or v.name in args.variant]
    if args.device != "npu2":
        target = f"aie.device({args.device})"
        variants = [
            dataclasses.replace(
                v,
                name=f"{v.name}_{args.device}",
                design=v.design.replace("aie.device(npu2)", target),
                pinned=(
                    None
                    if v.pinned is None
                    else v.pinned.replace("aie.device(npu2)", target)
                ),
            )
            for v in variants
        ]
    variants.sort(key=lambda v: v.expect == "HANG")
    wrong = 0
    for variant in variants:
        out = args.work / case.name / variant.name
        out.mkdir(parents=True, exist_ok=True)
        routed, diagnostics = route(variant, out, args.compile_timeout_s)
        for line in diagnostics.splitlines():
            if re.search(r"\b(warning|error):", line):
                print(f"  {variant.name}: router: {line.split(': ', 2)[-1][:200]}")
        compile_variant(out, args.peano, args.compile_timeout_s)
        results = []
        for _ in range(args.runs):
            results.append(run_isolated(args.case, out, args.process_timeout_s))
            if results[-1]["outcome"] == "WEDGED":
                print(
                    f"{case.name} {variant.name}: {results[-1]['state']}; the NPU "
                    "driver looks wedged. Stopping: reload amdxdna or reboot."
                )
                return 2
            if results[-1]["outcome"] in ("HANG", "ERROR"):
                break
        counts = Counter(r["outcome"] for r in results)
        verdict = verdict_of(results)
        mark = "as expected" if verdict == variant.expect else "UNEXPECTED"
        print(
            f"{case.name} {variant.name}: {verdict} ({dict(counts)}), "
            f"expected {variant.expect}: {mark}",
            flush=True,
        )
        for r in results:
            if r["outcome"] != "PASS":
                print(f"    {r.get('state') or r.get('problems')}")
                break
        if verdict != variant.expect:
            wrong += 1
        if args.verdicts and verdict in ("PASS", "HANG"):
            args.verdicts.mkdir(parents=True, exist_ok=True)
            path = args.verdicts / f"{case.name}_{variant.name}.mlir"
            device = results[-1]["device"]
            path.write_text(
                verdict_file(case, variant, routed, verdict, counts, device)
            )
    return 1 if wrong else 0


def main() -> int:
    cli = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = cli.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="route, compile and run every variant")
    run.add_argument("case", type=Path, help="a hw_cases/*.py module")
    run.add_argument("--variant", action="append", default=[])
    run.add_argument("--device", choices=["npu1", "npu2"], default="npu2")
    run.add_argument("--runs", type=int, default=20)
    run.add_argument("--process-timeout-s", type=int, default=120)
    run.add_argument("--compile-timeout-s", type=int, default=300)
    run.add_argument("--work", type=Path, default=Path("router_hw_work"))
    run.add_argument("--verdicts", type=Path, default=None)
    run.add_argument("--peano", default=os.environ.get("PEANO_INSTALL_DIR", ""))
    once = sub.add_parser("once", help="one run of a compiled variant (internal)")
    once.add_argument("case", type=Path)
    once.add_argument("out", type=Path)
    args = cli.parse_args()
    if args.command == "once":
        result = run_once(load_case(args.case), args.out)
        print(json.dumps(result), flush=True)
        return 0
    if not args.peano:
        cli.error("--peano or $PEANO_INSTALL_DIR is required")
    if args.runs < 1:
        cli.error("--runs must be at least 1")
    return run_case(args)


if __name__ == "__main__":
    sys.exit(main())
