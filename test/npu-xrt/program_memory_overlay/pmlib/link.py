# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Link a kernel into a program-memory slot, and refuse anything unusable.

Promoted to `aie.iron.overlay._link` (the production `iron.overlay` API's
build pipeline links every overlay the same way); this module re-exports it,
adding back the one thing a CLI tool wants that a build-pipeline library
function should not do unconditionally: printing the result. `pm.py`'s own
tests (e.g. build/reject_overlay.lit's `FULL:` check) depend on this exact
line.
"""

import inspect

from aie.iron.overlay._link import OverlayError, resident_syms_script
from aie.iron.overlay._link import link as _link
from aie.iron.overlay._link import verify as _verify

__all__ = ["OverlayError", "resident_syms_script", "link", "verify"]

_link_sig = inspect.signature(_link)
_verify_sig = inspect.signature(_verify)


def link(*args, **kwargs):
    bound = _link_sig.bind(*args, **kwargs)
    bound.apply_defaults()
    size = _link(*args, **kwargs)
    name = bound.arguments["output"].split("/")[-1]
    print(f"{name}: {size} bytes at 0x{bound.arguments['slot_base']:x}")
    return size


def verify(*args, **kwargs):
    bound = _verify_sig.bind(*args, **kwargs)
    bound.apply_defaults()
    size = _verify(*args, **kwargs)
    name = bound.arguments["path"].split("/")[-1]
    print(f"{name}: {size} bytes at 0x{bound.arguments['slot_base']:x}")
    return size
