# _object_cache.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Compiled kernel objects, shared across JIT cache entries.

A design's cache entry is keyed on the whole design, so a kernel used by two
designs -- or by one design at two sizes -- was compiled once per design
directory. What determines an object's bytes is much narrower: the
``KernelObject`` recipe, the target, the design's include paths, whether IR is
retained, and the compiler. Objects are keyed on exactly that, built once under
``<root>/<key>/``, and copied into each design directory that links them.

An entry records the inputs its compile read (see ``_manifest``) and is rebuilt
when one changes, so a header edit reaches every object that included it. The
copied depfile lets the design's own manifest record the same inputs.

**Peano only.** xchesscc reports no inputs, so a Chess object's headers cannot
be checked; one shared across designs would stay stale for all of them. Chess
kernels keep compiling into the design directory.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
from pathlib import Path

from aie.utils import config
from aie.utils.compile.cache.utils import file_lock
from aie.utils.compile.utils import (
    _cleanup_failed_compilation,
    _compile_external_kernel,
    _copy_source,
)

from . import _manifest
from ._hash import _tool_identity

logger = logging.getLogger(__name__)


def _key(func, target_arch, include_dirs, embed_bitcode) -> str:
    identity = (
        func.object_file_name,
        func.object_file._source,
        target_arch,
        tuple(str(d) for d in include_dirs or ()),
        embed_bitcode,
        str(config.cxx_header_path()),
        _tool_identity("peano", config.peano_cxx_path),
    )
    return hashlib.sha256(repr(identity).encode()).hexdigest()[:24]


class KernelObjectCache:
    """Content-addressed store of compiled kernel objects under ``root``."""

    def __init__(self, root: Path, lock_timeout_seconds: int):
        self.root = Path(root).absolute()
        self.lock_timeout_seconds = lock_timeout_seconds

    def fetch(self, func, kernel_dir, target_arch, include_dirs, embed_bitcode) -> bool:
        """Copy ``func``'s object into ``kernel_dir``, compiling it on a miss.

        Returns False, touching nothing, for a kernel the cache does not hold:
        a Chess kernel, or one with no source recipe.
        """
        recipe = getattr(getattr(func, "object_file", None), "_source", None)
        if recipe is None or recipe.use_chess:
            return False
        entry = self.root / _key(func, target_arch, include_dirs, embed_bitcode)
        obj = entry / func.object_file_name
        with file_lock(entry / ".lock", timeout_seconds=self.lock_timeout_seconds):
            if obj.is_file() and _manifest.is_valid(entry):
                logger.debug("Kernel object cache hit for %s (%s)", obj.name, entry)
            else:
                logger.debug("Kernel object cache miss for %s (%s)", obj.name, entry)
                _cleanup_failed_compilation(entry)
                try:
                    _compile_external_kernel(
                        func, str(entry), target_arch, include_dirs, embed_bitcode
                    )
                    _manifest.record(entry, [func], [])
                except BaseException:
                    _cleanup_failed_compilation(entry)
                    raise
            stamps = entry.glob(f"{glob.escape(obj.name)}.prefix_state.*.json")
            for artifact in (obj, entry / f"{obj.name}.d", *stamps):
                if artifact.is_file():
                    _copy_source(os.path.join(kernel_dir, artifact.name), str(artifact))
        return True
