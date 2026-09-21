# device.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Opening the NPU, once per process.

``pyxrt.device(index)`` is an open, not a lookup. XRT calls into the shim on
every construction and hands back a handle whose deleter closes the device
again, while the shim refcounts the file descriptor: the last handle going out
of scope closes ``/dev/accel/accel0``, and the next construction reopens it. So
a caller that keeps one handle per object turns allocation into open/close
churn, and it is the reopen that fails, with ``ENODEV``, under an unrelated
transient on the device.

One handle per process removes both halves of that. It is also the ownership
order the runtime needs on Windows, where an XRT object outliving what it was
allocated from is an access violation rather than a leak (#3100): a handle held
for the life of the process cannot be outlived by the buffers derived from it.
"""

import gc
import logging
import os
import shutil
import subprocess
import time

import pyxrt  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# Handles by device index, kept for the life of the process. Deliberately not
# released by ``cleanup_npu_runtime()``: that releases hw contexts and insts
# BOs, which are derived from a device and must not outlive it.
_DEVICES: dict[int, "pyxrt.device"] = {}

# A failed open is retried this many times before it is allowed to propagate.
# Retrying at all is what #2814 established, from a CI run where the device was
# briefly unopenable and no state of this process explained it.
_MAX_RETRIES = 5


def _log_device_state():
    """Report what the host thinks of the NPU, after an open has failed.

    Best-effort and informational: it names the single-NPU device node rather
    than resolving the one behind a given index, and it must not raise, since
    it runs on the way to reporting a different failure.
    """
    try:
        if os.path.exists("/dev/accel/accel0"):
            logger.debug("/dev/accel/accel0 exists")
            logger.debug("Stat: %s", os.stat("/dev/accel/accel0"))
        else:
            logger.debug("/dev/accel/accel0 does not exist")

        xrt_bin = shutil.which("xrt-smi")
        if xrt_bin is None:
            xrt_base = os.environ.get("XILINX_XRT", "/opt/xilinx/xrt")
            xrt_bin = xrt_base + "/bin/xrt-smi"
        if os.path.exists(xrt_bin):
            logger.debug("Running %s examine", xrt_bin)
            result = subprocess.run(
                [xrt_bin, "examine"],
                timeout=5,
                capture_output=True,
                text=True,
            )
            logger.debug("xrt-smi stdout:\n%s", result.stdout)
            logger.debug("xrt-smi stderr:\n%s", result.stderr)
    except Exception as debug_e:
        logger.debug("Failed to run debug checks: %s", debug_e)


def _open_device(index: int):
    """Open device ``index``, retrying a failure that may not be permanent."""
    for attempt in range(_MAX_RETRIES):
        try:
            return pyxrt.device(index)
        except RuntimeError as e:
            logger.warning(
                "Failed to acquire NPU device %d (attempt %d/%d): %s",
                index,
                attempt + 1,
                _MAX_RETRIES,
                e,
            )
            _log_device_state()
            if attempt + 1 == _MAX_RETRIES:
                raise
            gc.collect()  # a handle awaiting collection still holds the device
            time.sleep(1.0 * (attempt + 1))  # linear backoff
    raise AssertionError("_MAX_RETRIES must be at least 1")


def acquire_device(index: int = 0):
    """Return this process's handle on NPU device ``index``.

    The device is opened on the first call and the handle is reused after it,
    so a process holds one however many buffers it allocates.

    Args:
        index (int, optional): Device index to open. Defaults to 0.

    Returns:
        pyxrt.device: The handle for ``index``.

    Raises:
        RuntimeError: If the device could not be opened, after retries.
    """
    device = _DEVICES.get(index)
    if device is None:
        device = _open_device(index)
        _DEVICES[index] = device
    return device
