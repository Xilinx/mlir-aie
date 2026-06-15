# _hash.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Content-addressed hashing for the JIT cache.

Two halves so callers can distinguish "recipe changed" from "rebuild needed":

* :func:`_compute_recipe_hash`   — generator bytecode + compile_kwargs +
  aiecc/compile flags.  Pure function of the design specification.
* :func:`_compute_artifact_hash` — source / object mtimes + tool mtimes +
  target arch.  Captures things that change the *output* of compilation
  without changing the *recipe*.

:func:`_compute_hash` composes both into the 24-hex cache-key
``CompilableDesign`` uses to address ``$NPU_CACHE_HOME``.

Carved out of ``compilabledesign.py`` to keep the main file focused on the
``CompilableDesign`` class itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


def _compute_recipe_hash(
    generator: Callable | Path,
    compile_kwargs: Mapping[str, Any],
    aiecc_flags: list[str] | tuple[str, ...],
    compile_flags: list[str] | tuple[str, ...],
) -> str:
    """Hash of the "recipe": generator bytecode + CompileTime[T] kwargs + flags.

    Pure function of the design specification; does not touch the filesystem
    or environment.  Two CompilableDesigns with the same recipe_hash will
    produce identical MLIR (modulo nondeterminism in the generator body).
    """
    h = hashlib.sha256()

    if isinstance(generator, Path):
        h.update(str(generator).encode())
        try:
            h.update(str(generator.stat().st_mtime).encode())
        except FileNotFoundError:
            pass
    else:
        code = generator.__code__
        h.update(code.co_code)
        h.update(repr(code.co_consts).encode())
        h.update(getattr(generator, "__qualname__", "").encode())
        h.update(getattr(generator, "__module__", "").encode())

    def _kwarg_repr(v):
        if callable(v) and hasattr(v, "__code__"):
            code = v.__code__
            closure = (
                tuple(c.cell_contents for c in v.__closure__) if v.__closure__ else None
            )
            try:
                closure_repr = repr(closure)
            except Exception:
                closure_repr = "<unhashable closure>"
            return (
                "fn:",
                bytes(code.co_code).hex(),
                repr(code.co_consts),
                repr(getattr(v, "__defaults__", None)),
                closure_repr,
            )
        return str(v)

    try:
        kwargs_json = json.dumps(
            {k: _kwarg_repr(v) for k, v in sorted(compile_kwargs.items())}
        ).encode()
    except (TypeError, ValueError):
        kwargs_json = repr(sorted(compile_kwargs.items())).encode()
    h.update(kwargs_json)

    h.update(repr(sorted(aiecc_flags)).encode())
    h.update(repr(sorted(compile_flags)).encode())

    return h.hexdigest()


def _compute_artifact_hash(
    generator: Callable | Path,
    source_files: list[Path] | tuple[Path, ...],
    object_files: list[Path] | tuple[Path, ...],
) -> str:
    """Hash of the "artifacts": source/object mtimes + tool mtimes + target arch.

    Captures everything that can change the *output* of compilation without
    changing the *recipe*: edited C++ kernels, swapped object files, upgraded
    Peano / aiecc, retargeted device.
    """
    h = hashlib.sha256()

    for sf in sorted(source_files, key=str):
        h.update(str(sf).encode())
        try:
            h.update(str(Path(sf).stat().st_mtime).encode())
        except (FileNotFoundError, OSError):
            pass

    for of in sorted(object_files, key=str):
        h.update(str(of).encode())
        try:
            h.update(str(Path(of).stat().st_mtime).encode())
        except (FileNotFoundError, OSError):
            pass

    # Static .mlir is arch-agnostic; compiled kernels need a target identifier.
    # Missing components collapse to a constant + WARNING log so cross-arch cache
    # collisions surface instead of silently aliasing.
    if not isinstance(generator, Path):
        try:
            import aie.iron as _iron
            from aie.utils.compile.utils import resolve_target_arch

            try:
                device = _iron.get_current_device()
            except AttributeError:
                # Older/minimal iron imports may not expose get_current_device();
                # only then fall back to DefaultNPURuntime.  Importing
                # DefaultNPURuntime eagerly probes XRT via aie.utils.__getattr__,
                # which breaks hardware-less CI and ignores an explicit
                # set_current_device(...) override.
                from aie.utils import DefaultNPURuntime

                device = (
                    DefaultNPURuntime.device()
                    if DefaultNPURuntime is not None
                    else None
                )
            target_arch = resolve_target_arch(device)
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            logger.warning(
                "_compute_artifact_hash: target_arch unresolved (%s); using 'unknown'",
                exc,
            )
            target_arch = "unknown"

        try:
            from aie.utils import config as _config

            peano_cxx = _config.peano_cxx_path()
            peano_mtime = str(Path(peano_cxx).stat().st_mtime)
        except (
            ImportError,
            AttributeError,
            FileNotFoundError,
            OSError,
            RuntimeError,
        ) as exc:
            try:
                from aie.utils import config as _config

                peano_mtime = f"path:{_config.peano_install_dir()}"
                logger.warning(
                    "_compute_artifact_hash: peano cxx unavailable (%s); "
                    "keying on install dir path only",
                    exc,
                )
            except (ImportError, AttributeError, RuntimeError) as exc2:
                logger.warning("_compute_artifact_hash: peano absent (%s)", exc2)
                peano_mtime = "absent"

        try:
            import shutil as _shutil

            _aiecc_path = _shutil.which("aiecc")
            aiecc_mtime = (
                str(Path(_aiecc_path).stat().st_mtime) if _aiecc_path else "absent"
            )
        except (FileNotFoundError, OSError) as exc:
            logger.warning("_compute_artifact_hash: aiecc absent (%s)", exc)
            aiecc_mtime = "absent"

        h.update(
            f"target_arch={target_arch}|peano_mtime={peano_mtime}|aiecc_mtime={aiecc_mtime}".encode()
        )

    return h.hexdigest()


def _compute_hash(
    generator: Callable | Path,
    compile_kwargs: Mapping[str, Any],
    source_files: list[Path] | tuple[Path, ...],
    object_files: list[Path] | tuple[Path, ...],
    aiecc_flags: list[str] | tuple[str, ...],
    compile_flags: list[str] | tuple[str, ...],
) -> str:
    """Stable 24-hex SHA-256 cache key combining recipe + artifact hashes."""
    recipe = _compute_recipe_hash(generator, compile_kwargs, aiecc_flags, compile_flags)
    artifact = _compute_artifact_hash(generator, source_files, object_files)
    return hashlib.sha256(f"{recipe}|{artifact}".encode()).hexdigest()[:24]
