# _explicit_builds.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Finished builds whose outputs the caller places, shared across output paths.

A build to caller-named paths is found again only where it was written: its
``<stem>.prj`` records the outputs and the key that made them (see
``_manifest.record_outputs``). The same design built to a new path -- a fresh
build directory, a second checkout of an application -- reran aiecc in full,
though nothing it reads had changed.

Entries live under ``<root>/<build key>/``. Each holds the outputs, stored by
role, and the files from the work directory that callers read after a build:
``params.txt``, ``input_with_addresses.mlir``, ``full_elf_config.json`` and the
PDIs. The design's manifest is copied last, so an entry without one is
incomplete and never fetched; a fetch also requires every recorded input to be
unchanged (see ``_manifest.is_valid``), exactly as a hit in place does.

A fetched work directory holds no kernel objects or other intermediates.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aie.utils.compile.cache.utils import file_lock
from aie.utils.compile.utils import _cleanup_failed_compilation, _copy_source

from . import _manifest

logger = logging.getLogger(__name__)

_COMPANION_NAMES = ("params.txt", "input_with_addresses.mlir", "full_elf_config.json")


def _companions(directory: Path) -> list[Path]:
    named = (directory / name for name in _COMPANION_NAMES)
    return [p for p in named if p.is_file()] + sorted(directory.glob("*.pdi"))


class BuildCache:
    """Content-keyed store of explicit-path build outputs under ``root``."""

    def __init__(self, root: Path, lock_timeout_seconds: int):
        self.root = Path(root).absolute()
        self.lock_timeout_seconds = lock_timeout_seconds

    def fetch(self, key: str, kernel_dir: Path, outputs: dict[str, Path]) -> bool:
        """Copy the build ``key`` into ``outputs`` and ``kernel_dir``, if stored.

        ``outputs`` maps each role to where the caller wants it, in the order
        the caller checks them. On a hit ``kernel_dir`` is replaced by the
        stored work files and records the outputs as written by ``key``.
        """
        entry = self.root / key
        if not entry.is_dir():
            return False
        with file_lock(entry / ".lock", timeout_seconds=self.lock_timeout_seconds):
            if not all((entry / role).is_file() for role in outputs):
                return False
            if not _manifest.is_valid(entry):
                return False
            _cleanup_failed_compilation(kernel_dir)
            for path in _companions(entry):
                _copy_source(str(kernel_dir / path.name), str(path))
            for role, dest in outputs.items():
                _copy_source(str(dest), str(entry / role))
            _copy_source(
                str(kernel_dir / _manifest.MANIFEST_NAME),
                str(entry / _manifest.MANIFEST_NAME),
            )
        _manifest.record_outputs(kernel_dir, key, list(outputs.values()))
        logger.debug("Build cache hit for %s (%s)", kernel_dir, entry)
        return True

    def store(self, key: str, kernel_dir: Path, outputs: dict[str, Path]) -> None:
        """Keep the build ``key`` that just wrote ``outputs`` and ``kernel_dir``."""
        manifest = kernel_dir / _manifest.MANIFEST_NAME
        if not manifest.is_file():
            return
        entry = self.root / key
        with file_lock(entry / ".lock", timeout_seconds=self.lock_timeout_seconds):
            _cleanup_failed_compilation(entry)
            try:
                for path in _companions(kernel_dir):
                    _copy_source(str(entry / path.name), str(path))
                for role, path in outputs.items():
                    _copy_source(str(entry / role), str(path))
                _copy_source(str(entry / _manifest.MANIFEST_NAME), str(manifest))
            except OSError as e:
                logger.warning("Could not store build %s in %s: %s", key, entry, e)
                _cleanup_failed_compilation(entry)
