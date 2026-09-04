# probe.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

"""Structured diagnosis of the NPU stack, from silicon up to this interpreter.

Callers previously answered "can I use the NPU?" with a single bool derived from
``import pyxrt``. That conflates conditions which fail independently and have
different fixes, so the resulting error could only ever report the device string
the caller passed in. Each stage here returns a :class:`Check` carrying the
reason and a remedy, so a caller can say *why* the NPU is unavailable.

Cheap stages (platform, hardware, driver) are pure filesystem lookups and run
eagerly. Stages that load the XRT stack or spawn a process are deferred, keeping
the property the previous probe was right to protect.
"""

from __future__ import annotations

import functools
import glob
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass

_logger = logging.getLogger(__name__)

# Linux exposes a bound NPU as a DRM accel node plus sysfs attributes. Windows
# reaches the same device through WDDM and has neither, so the Linux-only checks
# report "unknown" there rather than "absent" -- see Check.ok.
_ACCEL_GLOB = "/dev/accel/accel*"
_SYSFS_ACCEL_GLOB = "/sys/class/accel/accel*/device"


@dataclass(frozen=True)
class Check:
    """One stage of the diagnosis.

    ``ok`` is tri-state on purpose. ``None`` means the stage could not be
    evaluated here (typically a Linux-only probe on another platform); reporting
    that as ``False`` is how a portability gap turns into a wrong answer about
    the machine.
    """

    name: str
    ok: bool | None
    detail: str
    remedy: str | None = None
    exc: BaseException | None = None

    @property
    def failed(self) -> bool:
        return self.ok is False

    def __str__(self) -> str:
        mark = {True: "ok", False: "FAIL", None: "?"}[self.ok]
        line = f"[{mark:>4}] {self.name}: {self.detail}"
        return (
            f"{line}\n       -> {self.remedy}" if self.remedy and not self.ok else line
        )


def _xrt_smi() -> str | None:
    """Locate xrt-smi on PATH or under XILINX_XRT, honouring the platform suffix."""
    found = shutil.which("xrt-smi")
    if found:
        return found
    base = os.environ.get("XILINX_XRT")
    if base:
        candidate = os.path.join(base, "bin", _executable_name("xrt-smi"))
        if os.path.exists(candidate):
            return candidate
    return None


@functools.cache
def _examine() -> str | None:
    """Return `xrt-smi examine` output, or None if it cannot be run.

    Used where sysfs is unavailable. Deferred and memoised: this spawns a process,
    unlike the sysfs reads that serve the same purpose on Linux.
    """
    binary = _xrt_smi()
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, "examine"], timeout=20, capture_output=True, text=True
        )
    except (OSError, subprocess.SubprocessError) as e:
        _logger.debug("xrt-smi examine failed: %s", e)
        return None
    return result.stdout or None


def _examine_devices() -> list[str]:
    """Names from the `Device(s) Present` table of `xrt-smi examine`."""
    text = _examine()
    if not text:
        return []
    names, in_table = [], False
    for line in text.splitlines():
        if line.startswith("Device(s) Present"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                if names:
                    break
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 2 and cells[1] and not set(cells[1]) <= {"-"}:
                if cells[1] != "Name":
                    names.append(f"{cells[1]} {cells[0]}")
    return names


def _examine_field(label: str) -> str | None:
    text = _examine()
    if not text:
        return None
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == label:
            return value.strip()
    return None


def _undetermined(name: str) -> Check:
    # Reached only when sysfs is unavailable (non-Linux) AND xrt-smi cannot be run,
    # so neither source can answer. Reported as unknown rather than absent: a
    # platform-specific path check that returns False here is how a portability gap
    # turns into a wrong claim about the machine.
    # TODO: native enumeration for this case (SetupAPI/WMI on Windows) would remove
    # the dependency on XRT being installed before the device can be seen at all.
    return Check(
        name,
        None,
        f"not determined on {platform.system()}: no sysfs, and xrt-smi is unavailable",
        remedy="Install XRT so the device can be enumerated with xrt-smi examine.",
    )


@functools.cache
def check_platform() -> Check:
    impl = f"{platform.python_implementation()} {platform.python_version()}"
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return Check(
        "platform",
        True,
        f"{platform.system()} {platform.machine()}, {impl} (wheel ABI tag {tag})",
    )


@functools.cache
def check_hardware() -> Check:
    """Report whether an NPU is present, without requiring a userspace runtime."""
    if sys.platform == "linux":
        nodes = sorted(glob.glob(_ACCEL_GLOB))
        if nodes:
            name = _sysfs_attr("vbnv") or "unknown model"
            return Check("hardware", True, f"{name} ({', '.join(nodes)})")
        if _xrt_smi() is None:
            return Check(
                "hardware",
                False,
                "no /dev/accel/accel* node",
                remedy=(
                    "No NPU is exposed. Check that the part has one, that it is enabled "
                    "in BIOS, and that the amdxdna driver is loaded."
                ),
            )

    devices = _examine_devices()
    if devices:
        return Check("hardware", True, ", ".join(devices))
    if _examine() is None:
        return _undetermined("hardware")
    return Check(
        "hardware",
        False,
        "xrt-smi examine lists no devices",
        remedy="Check that the NPU is enabled in BIOS and that its driver is loaded.",
    )


@functools.cache
def check_driver() -> Check:
    """Report whether a driver is bound, and its firmware level."""
    if sys.platform == "linux" and glob.glob(_SYSFS_ACCEL_GLOB):
        fw = _sysfs_attr("fw_version")
        return Check(
            "driver", True, "amdxdna bound" + (f", firmware {fw}" if fw else "")
        )

    fw = _examine_field("NPU Firmware Version")
    if fw:
        return Check("driver", True, f"driver bound, firmware {fw}")
    if _examine() is None:
        if sys.platform == "linux":
            return Check(
                "driver",
                False,
                "no accel device bound in sysfs",
                remedy="Load the amdxdna driver (modprobe amdxdna) or install xdna-driver.",
            )
        return _undetermined("driver")
    return Check(
        "driver",
        False,
        "xrt-smi examine reports no NPU firmware version",
        remedy="Install or load the NPU driver for this platform.",
    )


@functools.cache
def check_runtime() -> Check:
    """Report XRT userspace, independent of whether its bindings are importable.

    Deliberately does not read the XRT version. That costs an ``xrt-smi examine``
    subprocess, it cannot change this verdict, and this stage sits on the path of
    an error message -- the module's rule is that a stage spawns a process only
    when the answer depends on it. ``xrt-smi examine`` reports the version.
    """
    binary = _xrt_smi()
    if binary is None:
        return Check(
            "runtime",
            False,
            "xrt-smi not on PATH and XILINX_XRT is unset or does not contain it",
            remedy="Install XRT, or set XILINX_XRT to an existing install.",
        )
    return Check("runtime", True, f"XRT userspace found ({binary})")


@functools.cache
def check_bindings() -> Check:
    """Report whether pyxrt imports *in this interpreter*.

    Deferred: importing pyxrt pulls in the whole XRT stack.
    """
    try:
        import pyxrt  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        return Check(
            "bindings",
            False,
            f"pyxrt is not importable: {e}",
            remedy=(
                "pyxrt ships with XRT, not on PyPI, and is built against one Python "
                f"version. Install/build XRT for {platform.python_version()}, or run "
                "under the interpreter its pyxrt was built for."
            ),
            exc=e,
        )
    except Exception as e:  # noqa: BLE001 - the stack this diagnoses is the broken one
        # Reached when pyxrt is found but its initialisation raises, which the
        # import machinery propagates unchanged. Letting that out of a probe
        # would replace the diagnosis with the failure it exists to report.
        return Check(
            "bindings",
            False,
            f"pyxrt failed to initialise: {type(e).__name__}: {e}",
            remedy="Check the XRT install this pyxrt was built against.",
            exc=e,
        )
    return Check("bindings", True, "pyxrt imports")


@functools.cache
def check_toolchain() -> Check:
    versions = {}
    for dist in ("mlir_aie", "llvm-aie"):
        try:
            from importlib.metadata import version

            versions[dist] = version(dist)
        except Exception:  # noqa: BLE001 - absence is the signal, not the exception
            versions[dist] = None

    missing = [d for d, v in versions.items() if v is None]
    detail = ", ".join(f"{d} {v or 'not installed'}" for d, v in versions.items())
    if missing:
        return Check(
            "toolchain",
            False,
            detail,
            remedy=f"Install {' and '.join(missing)} (see mlir-aie's README).",
        )
    return Check("toolchain", True, detail)


def _executable_name(name: str) -> str:
    # Same rule as aie.utils.config._executable_name; repeated rather than imported
    # to keep this module dependency-free, since it must work when the install is broken.
    return f"{name}.exe" if os.name == "nt" else name


def _sysfs_attr(attr: str) -> str | None:
    for dev in sorted(glob.glob(_SYSFS_ACCEL_GLOB)):
        try:
            with open(os.path.join(dev, attr)) as f:
                return f.read().strip()
        except OSError:
            continue
    return None


# Ordered from silicon upwards, so the first failure is the one to fix first.
_STAGES = (
    check_platform,
    check_hardware,
    check_driver,
    check_runtime,
    check_bindings,
    check_toolchain,
)

# Stages that load the XRT stack, and so are skippable by a caller that only
# wants what can be answered without it.
_DEFERRED = frozenset({check_bindings})


def _selected(include_deferred: bool):
    return (s for s in _STAGES if include_deferred or s not in _DEFERRED)


def probe(include_deferred: bool = True) -> list[Check]:
    """Run the diagnosis. Set ``include_deferred=False`` to skip stages that load XRT."""
    return [s() for s in _selected(include_deferred)]


def failures(include_deferred: bool = True) -> list[Check]:
    """Return every stage that positively failed; ``None`` stages are not failures."""
    return [c for c in probe(include_deferred) if c.failed]


def first_actionable(include_deferred: bool = True) -> Check | None:
    """Return the lowest failing stage; fixing it precedes the ones above.

    Stops there rather than running the whole probe: the stages above cannot
    change which one to fix first, and this is the path an error message takes.
    """
    for stage in _selected(include_deferred):
        check = stage()
        if check.failed:
            return check
    return None


def npu_unavailable_reason() -> str | None:
    """One line explaining why the NPU is unusable, or None if it looks usable.

    Intended for error messages that would otherwise only be able to echo back
    the device string they rejected.
    """
    first = first_actionable()
    if first is None:
        return None
    return f"{first.detail}. {first.remedy}" if first.remedy else first.detail


def summary() -> str:
    lines = [str(c) for c in probe()]
    first = first_actionable()
    lines.append("")
    lines.append(
        "NPU looks usable."
        if first is None
        else f"NPU unavailable. Fix this first -- {first.name}: {first.detail}"
    )
    return "\n".join(lines)


def as_dict() -> dict:
    """Return a machine-readable form, for consumers outside Python."""
    return {
        "checks": [{k: v for k, v in asdict(c).items() if k != "exc"} for c in probe()],
        "usable": first_actionable() is None,
    }
