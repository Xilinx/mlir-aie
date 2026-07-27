# _log_setup.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Logging setup for the ``aie`` namespace.

Imported first (for its side effect) by :mod:`aie.utils` so the level is
configured before any ``aie.utils`` submodule import can itself emit a log
record.
"""

import logging
import os

# Prevent "No handlers could be found" warnings when aie is used as a library.
logging.getLogger("aie").addHandler(logging.NullHandler())

# Honour AIE_LOG_LEVEL env var (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL).
_log_level_str = os.environ.get("AIE_LOG_LEVEL", "").upper()
if _log_level_str:
    _log_level = getattr(logging, _log_level_str, None)
    if _log_level is not None:
        logging.getLogger("aie").setLevel(_log_level)
