# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Delay a stale-entry writer until another writer has repaired the cache."""

import contextlib
import fcntl
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
llc = Path(shutil.which(sys.argv[3])).resolve()
aiecc = sys.argv[4:]
cache = root / "cache"
bin_dir = root / "peano" / "bin"
bin_dir.mkdir(parents=True)
for tool in llc.parent.iterdir():
    if tool.name != "llc":
        (bin_dir / tool.name).symlink_to(tool)

# The wrapper and command line stay identical across builds so keys match.
# Only the delayed build's environment enables the compilation barrier.
wrapper = bin_dir / "llc"
wrapper.write_text(
    f"#!/bin/sh\n"
    'if [ -n "$DEVICE_CACHE_GATE" ]; then\n'
    '  touch "$DEVICE_CACHE_GATE/ready"\n'
    "  for attempt in $(seq 1 1200); do\n"
    '    [ -f "$DEVICE_CACHE_GATE/release" ] && break\n'
    "    sleep 0.1\n"
    "  done\n"
    '  [ -f "$DEVICE_CACHE_GATE/release" ] || exit 1\n'
    "fi\n"
    f'exec {shlex.quote(str(llc))} "$@"\n'
)
wrapper.chmod(0o755)
command = aiecc + [
    "-v",
    "--get-pdi",
    f"--peano={bin_dir.parent}",
    f"--device-cache={cache}",
    str(source),
]


def stop_build(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(name):
    work = root / name
    work.mkdir()
    process = subprocess.Popen(
        command,
        cwd=work,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=120)
        assert process.returncode == 0, output
        return output
    finally:
        stop_build(process)


def invalidate():
    for elf in cache.glob("*/*.elf"):
        elf.unlink()


assert "device cache store:" in run("populate")
invalidate()
gate = root / "delayed"
gate.mkdir()
with (gate / "log").open("w+") as log:
    delayed = subprocess.Popen(
        command,
        cwd=gate,
        env={**os.environ, "DEVICE_CACHE_GATE": str(gate)},
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not (gate / "ready").exists():
            assert delayed.poll() is None, "delayed build exited before barrier"
            assert time.monotonic() < deadline, "compilation barrier timed out"
            time.sleep(0.05)
        assert "device cache store:" in run("repair")
        # Mark the repaired directories: a delayed stale writer must not
        # replace them, even if it would publish byte-identical ELF files.
        markers = [entry / "preserve" for entry in cache.iterdir() if entry.is_dir()]
        assert len(markers) == 2, markers
        for marker in markers:
            marker.touch()
        (gate / "release").touch()
        assert delayed.wait(timeout=120) == 0
        log.seek(0)
        output = log.read()
        assert "device cache miss:" in output, output
        assert "device cache store:" not in output, output
        assert all(marker.exists() for marker in markers)
    finally:
        stop_build(delayed)

output = run("hit")
assert "device cache miss:" not in output, output
assert "device cache hit:" in output, output
for device in ("a", "b"):
    assert (root / "repair" / f"{device}.pdi").read_bytes() == (
        gate / f"{device}.pdi"
    ).read_bytes()

# Busy locks must not wait indefinitely, damage entries, or fail the build.
# Cover both POSIX record locks and flock, independent of LLVM's choice.
invalidate()
with contextlib.ExitStack() as stack:
    locks = list(cache.glob("*.lock"))
    assert len(locks) == 2, locks
    for path in locks:
        lock = stack.enter_context(path.open("r+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = run("busy")
    assert "device cache store:" not in output, output
    assert all(marker.exists() for marker in markers)
    assert not list(cache.glob("*.tmp*"))
assert "device cache store:" in run("unlocked")

# A lock released while the build is still waiting is plain contention, not a
# lock failure: the build must wait it out and store silently, with no
# "cannot lock" warning.
invalidate()
hold_seconds = 2
locks = list(cache.glob("*.lock"))
assert len(locks) == 2, locks
holder_script = (
    "import fcntl, sys, time\n"
    "f = open(sys.argv[1], 'r+')\n"
    "fcntl.lockf(f, fcntl.LOCK_EX)\n"
    "open(sys.argv[2], 'w').close()\n"
    "time.sleep(float(sys.argv[3]))\n"
    "fcntl.lockf(f, fcntl.LOCK_UN)\n"
)
holders = []
readies = []
for path in locks:
    ready = path.with_suffix(".ready")
    readies.append(ready)
    holders.append(
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                holder_script,
                str(path),
                str(ready),
                str(hold_seconds),
            ]
        )
    )
try:
    deadline = time.monotonic() + 30
    while not all(r.exists() for r in readies):
        assert all(h.poll() is None for h in holders), "a holder exited early"
        assert time.monotonic() < deadline, "holders never took their locks"
        time.sleep(0.02)
    output = run("contended")
finally:
    for holder in holders:
        assert holder.wait(timeout=30) == 0
    for ready in readies:
        ready.unlink(missing_ok=True)
assert "cannot lock" not in output, output
assert "No locks available" not in output, output
assert "device cache store:" in output, output
