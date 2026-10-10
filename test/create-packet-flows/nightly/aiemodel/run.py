#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Running aie-opt and reading its diagnostics."""

import os
import re
import shutil
import subprocess
from pathlib import Path

# Running the router.


def aie_opt_path():
    near = (p / "build" / "bin" / "aie-opt" for p in Path(__file__).resolve().parents)
    here = next((p for p in near if p.exists()), "aie-opt")
    return os.environ.get("AIE_OPT") or shutil.which("aie-opt") or str(here)


def aie_opt(text, hops_on=True, split=False, extra=(), timeout=None, strict=False):
    """Run the router, letting flows that can deadlock route with a warning
    unless strict. A timeout comes back as returncode -1000."""
    opts = [] if hops_on else ["circuit-switch-hops=false"]
    if not strict:
        opts.append("allow-deadlock-prone=true")
    opt = "--aie-create-pathfinder-flows" + ("=" + " ".join(opts) if opts else "")
    cmd = (
        [aie_opt_path(), opt, *extra]
        + (["--split-input-file"] if split else [])
        + ["-"]
    )
    for _ in range(2):
        try:
            p = subprocess.run(
                cmd, input=text, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                cmd, -1000, "", f"timeout after {timeout}s"
            )
        # The binary may be relinked under us; a crash is worth one retry.
        if p.returncode >= 0:
            break
    return p


MODULE_TAG_RE = re.compile(r"^module @s(\d+)\b", re.M)


def route_batch(tagged, hops_on, extra=(), strict=False):
    """Route many tagged designs in one run. A failure drops that design's
    output, so designs missing from it are rerun alone. Returns ({tag:
    output}, {tag: (returncode, stderr)}, {tag: debug output})."""
    p = aie_opt(
        "\n// -----\n".join(m for _, m in tagged),
        hops_on,
        split=True,
        extra=extra,
        strict=strict,
    )
    out, errs, debug = {}, {}, {}
    for chunk in p.stdout.split("\n// -----\n"):
        if m := MODULE_TAG_RE.search(chunk):
            out[int(m.group(1))] = chunk.removeprefix("// -----\n")
    logs = p.stderr.split(PASS_BEGIN)[1:]
    if len(logs) == len(tagged):
        debug = {tag: log for (tag, _), log in zip(tagged, logs)}
    for tag, text in tagged:
        if tag not in out:
            q = aie_opt(text, hops_on, extra=extra, strict=strict)
            debug[tag] = q.stderr
            if q.returncode == 0:
                out[tag] = q.stdout
            else:
                errs[tag] = (q.returncode, q.stderr)
    return out, errs, debug


PASS_BEGIN = "---Begin AIEPathfinderPass---"
_debug_ok = []


def debug_flags():
    """DEBUG_ONLY when aie-opt was built with assertions, else nothing."""
    if not _debug_ok:
        p = aie_opt("module {}", extra=(DEBUG_ONLY,))
        _debug_ok.append(p.returncode == 0)
    return (DEBUG_ONLY,) if _debug_ok[0] else ()


def unavoidable_warning(an):
    """The warning the router gives for pairs no routing keeps from deadlocking."""
    pairs = an.unavoidable()
    if not pairs:
        return None
    s = "Flows can deadlock however they are routed: " + an.explain(*pairs[0])
    if len(pairs) > 1:
        s += f" So can {len(pairs) - 1} other pair{'s' if len(pairs) > 2 else ''} of flows."
    return s


SHARED_RECEIVER_WARNING = (
    "Packet flows into receivers they share can deadlock holding arbiters across "
    "switchboxes, and no routing found avoids it: "
)
ALLOW_HINT = (
    " Set allow-deadlock-prone (aiecc --allow-deadlock-prone-routing) to "
    "route them anyway."
)


def router_warning(stderr):
    return next(
        (
            l.split(" warning: ", 1)[1].strip()
            for l in stderr.splitlines()
            if "however they are routed" in l
        ),
        None,
    )


def shared_receiver_warning(stderr):
    return next(
        (
            l.split(SHARED_RECEIVER_WARNING, 1)[1].strip()
            for l in stderr.splitlines()
            if SHARED_RECEIVER_WARNING in l
        ),
        None,
    )


def first_error(stderr):
    return next(
        (l.strip() for l in stderr.splitlines() if "error" in l), stderr.strip()[:300]
    )


DEBUG_ONLY = "-debug-only=aie-create-pathfinder-flows,aie-pathfinder"
