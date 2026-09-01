# _dispatch_artifacts.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""On-disk layout of a compiled dispatch bridge.

Both the compile side (which writes ``dispatch.so`` and this sidecar) and the
dispatch side (which loads them) need to agree on where these live and what
they contain. Kept in its own stdlib-only module so every consumer --
``compilabledesign``, ``callabledesign``, ``_dispatch_compile`` and
``_dispatch_bridge`` -- can import it at module scope without pulling in the
host runtime and creating an import cycle.
"""

from __future__ import annotations

import json
from pathlib import Path

# The built dispatch.so's call ABI, recorded next to it at compile time.
# Dispatch reads this instead of re-parsing the generated C++ header: the ABI
# is decided once, when the .so is built, and a cached kernel_dir should not
# depend on generated-source formatting staying stable forever.
DISPATCH_ABI_NAME = "dispatch_abi.json"


def dispatch_abi_path(kernel_dir: Path) -> Path:
    """Path of the ABI sidecar for the dispatch.so built into *kernel_dir*."""
    return Path(kernel_dir) / DISPATCH_ABI_NAME


def write_dispatch_abi(
    kernel_dir: Path,
    func_name: str,
    dispatch_params: list[str],
    param_ctypes: list[str],
) -> Path:
    """Record the built bridge's ABI beside its ``.so``."""
    path = dispatch_abi_path(kernel_dir)
    path.write_text(
        json.dumps(
            {
                "func_name": func_name,
                "dispatch_params": list(dispatch_params),
                "param_ctypes": list(param_ctypes),
            },
            indent=2,
        )
    )
    return path


def read_dispatch_abi(kernel_dir: Path) -> dict:
    """Return the recorded ABI, or raise ``FileNotFoundError`` if absent."""
    return json.loads(dispatch_abi_path(kernel_dir).read_text())
