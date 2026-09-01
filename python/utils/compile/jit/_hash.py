# _hash.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Content-addressed hashing for the JIT cache.

Two halves so callers can distinguish "recipe changed" from "rebuild needed":

* :func:`_compute_recipe_hash`   — generator identity + compile_kwargs +
  aiecc/compile flags. Target-independent design identity.
* :func:`_compute_artifact_hash` — source / object content + tool mtimes +
  target device.  Captures things that change the *output* of compilation
  without changing the *recipe*.

Design inputs (sources, objects, a `Path` generator) are identified by their
**content**.  mtime is not a property of a file, it is a property of how the
file arrived: a fresh clone, a `pip install`, a `cp` or a `touch` all restamp it
without changing a byte, and restoring one hides a change that did happen.  Tool
identity stays on mtime, which is cheap and moves whenever the toolchain is
rebuilt or reinstalled.

:func:`_compute_hash` composes both into the 24-hex cache-key
``CompilableDesign`` uses to address ``$NPU_CACHE_HOME``.

Carved out of ``compilabledesign.py`` to keep the main file focused on the
``CompilableDesign`` class itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import marshal
from pathlib import Path
from types import CodeType
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

# Read granularity for content digests.  Bounded so a large input is streamed
# rather than materialised, which keeps MemoryError off this path.
_DIGEST_CHUNK = 1 << 20


def _content_digest(path: Path | str) -> str:
    """Digest a file by content, streamed.

    Returns a marker instead of raising: an input we cannot read is still an
    input, and it must not silently collapse onto the same key as an input we
    can.  The marker embeds the error class so "missing" and "unreadable" stay
    distinguishable, which keeps a later fail-closed change a pure policy edit
    rather than a re-plumbing.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(_DIGEST_CHUNK):
                h.update(chunk)
    except (OSError, ValueError) as exc:
        return f"<unreadable:{type(exc).__name__}>"
    return h.hexdigest()


def _device_identity_key(device) -> tuple[str, str, str, str]:
    """Return the cache-relevant identity of an IRON device."""
    if device is None:
        return ("none", "", "", "")
    return (
        f"{type(device).__module__}.{type(device).__qualname__}",
        str(getattr(device, "arch", "")),
        str(getattr(device, "cols", "")),
        str(getattr(device, "rows", "")),
    )


def _without_location(const):
    """Return a constant with file and line info dropped, nested code objects included.

    Applied to every constant, not just code objects, so one cannot keep its
    location by sitting inside a tuple.
    """
    if isinstance(const, tuple):
        return tuple(_without_location(c) for c in const)
    if not isinstance(const, CodeType):
        return const
    return const.replace(
        co_consts=tuple(_without_location(c) for c in const.co_consts),
        co_filename="",
        co_firstlineno=1,
        co_linetable=b"",
    )


def _code_identity(code: CodeType) -> bytes:
    """Stable bytes for a code object: what marshal writes into a ``.pyc``.

    ``repr()`` of a code object embeds its address, so a key built from it moves
    between processes. marshal covers bytecode, names, varnames, flags and
    constants, recursing into nested code, with no addresses and a hash-seed
    independent frozenset order. ``co_names`` matters: ``matmul_bf16(a, b, c)``
    and ``matmul_i8(a, b, c)`` compile to identical bytecode.

    Location is stripped first: it is not part of the design, and keying on it
    would split the cache per checkout. Version 4 is pinned because
    ``marshal.version`` is 4 through 3.13 and 5 from 3.14.
    """
    return marshal.dumps(_without_location(code), 4)


def _compute_recipe_hash(
    generator: Callable | Path,
    compile_kwargs: Mapping[str, Any],
    aiecc_flags: list[str] | tuple[str, ...],
    compile_flags: list[str] | tuple[str, ...],
    full_elf: bool = False,
    include_paths: list[Path] | tuple[Path, ...] = (),
) -> str:
    """Hash of the "recipe": generator bytecode + CompileTime[T] kwargs + flags.

    Captures the target-independent generator and compile configuration. It
    omits device identity, so equal recipe hashes can produce different
    target-specialized MLIR.

    ``full_elf`` is part of the recipe: full-ELF and xclbin+insts builds emit
    different MLIR (the former injects ``npu.load_pdi``) and different
    artifacts, so they must not share a cache entry.

    ``include_paths`` likewise: they are ``-I`` directories forwarded to the
    C++ compiler, so two otherwise identical designs pointed at different
    header trees compile to different objects. Hashed in ORDER, not sorted
    like the flag lists above, because ``-I`` search order decides which
    header wins when two directories provide the same name.
    """
    h = hashlib.sha256()

    if isinstance(generator, Path):
        h.update(str(generator).encode())
        h.update(_content_digest(generator).encode())
    else:
        h.update(_code_identity(generator.__code__))
        h.update(getattr(generator, "__qualname__", "").encode())
        h.update(getattr(generator, "__module__", "").encode())
        # A CompileTime[T] left to its Python default never reaches
        # compile_kwargs, and the default lives outside the code object.
        h.update(repr(getattr(generator, "__defaults__", None)).encode())
        h.update(repr(getattr(generator, "__kwdefaults__", None)).encode())
        # Annotations also live outside the code object, and they decide how
        # every parameter is treated: In vs Out vs InOut, and the wrapped type
        # of a DispatchTime[T] (which is forwarded into the generated MLIR as
        # the runtime scalar's type). Without this, changing only an annotation
        # reuses an artifact built for the old one. Markers repr stably by
        # design (markers.py) so this is constant across processes.
        h.update(repr(getattr(generator, "__annotations__", None)).encode())

    def _kwarg_repr(v):
        if callable(v) and hasattr(v, "__code__"):
            closure = (
                tuple(c.cell_contents for c in v.__closure__) if v.__closure__ else None
            )
            try:
                closure_repr = repr(closure)
            except Exception:
                closure_repr = "<unhashable closure>"
            return (
                "fn:",
                _code_identity(v.__code__).hex(),
                repr(getattr(v, "__defaults__", None)),
                repr(getattr(v, "__kwdefaults__", None)),
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
    h.update(f"full_elf={full_elf}".encode())
    h.update(repr([str(p) for p in include_paths]).encode())

    return h.hexdigest()


def _compute_artifact_hash(
    generator: Callable | Path,
    source_files: list[Path] | tuple[Path, ...],
    object_files: list[Path] | tuple[Path, ...],
    fold_ddr_addr_offset: bool,
    has_dispatch_params: bool = False,
) -> str:
    """Hash of the "artifacts": source/object content + tool mtimes + device.

    Captures everything that can change the *output* of compilation without
    changing the *recipe*: edited C++ kernels, swapped object files, upgraded
    Peano / aiecc, retargeted device.

    ``fold_ddr_addr_offset`` is the active backend's DDR-patch ABI: XRT/CPU emit
    a folded ``insts.bin`` and HRX an unfolded one, so the two must never share a
    cache entry. It is resolved once by the caller and passed in explicitly (no
    silent default) so the cache key and the compilation can never disagree.

    ``has_dispatch_params`` additionally hashes the dynamic-dispatch toolchain
    (aie-opt/aie-translate/host C++ compiler) mtimes, so an upgrade to any of
    them invalidates a design's ``dispatch.so`` the same way an upgraded Peano
    invalidates a design's kernel objects. A no-op for the overwhelming
    majority of (non-DispatchTime[T]) designs.
    """
    h = hashlib.sha256()

    for sf in sorted(source_files, key=str):
        h.update(str(sf).encode())
        h.update(_content_digest(sf).encode())

    for of in sorted(object_files, key=str):
        h.update(str(of).encode())
        h.update(_content_digest(of).encode())

    h.update(f"fold_ddr_addr_offset={fold_ddr_addr_offset}".encode())

    # Static .mlir is target-agnostic; compiled kernels need a device identifier.
    # Missing components collapse to a constant + WARNING log so cross-target
    # cache collisions surface instead of silently aliasing.
    if not isinstance(generator, Path):
        try:
            from aie.utils import get_current_device
            from aie.utils.compile.utils import resolve_target_arch

            device = get_current_device(probe_runtime=False)
            target_arch = resolve_target_arch(device)
            target_device = _device_identity_key(device)
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            logger.warning(
                "_compute_artifact_hash: target_arch unresolved (%s); using 'unknown'",
                exc,
            )
            target_arch = "unknown"
            target_device = ("unknown", "", "", "")

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
            from aie.utils import config as _config

            # Resolve aiecc the way the compile does.  Probing PATH instead
            # misses the bundled bin/aiecc that _run_aiecc actually invokes,
            # and then every aiecc aliases onto the constant below.
            aiecc_mtime = str(Path(_config.aiecc_path()).stat().st_mtime)
        except (
            ImportError,
            AttributeError,
            FileNotFoundError,
            OSError,
            RuntimeError,
        ) as exc:
            logger.warning("_compute_artifact_hash: aiecc absent (%s)", exc)
            aiecc_mtime = "absent"

        h.update(
            f"target_arch={target_arch}|target_device={target_device!r}|"
            f"peano_mtime={peano_mtime}|aiecc_mtime={aiecc_mtime}".encode()
        )

        if has_dispatch_params:
            for tool_name, path_fn in (
                ("aie_opt", "aie_opt_path"),
                ("aie_translate", "aie_translate_path"),
                ("host_cxx", "host_cxx_path"),
            ):
                try:
                    from aie.utils import config as _config

                    tool_mtime = str(Path(getattr(_config, path_fn)()).stat().st_mtime)
                except (
                    ImportError,
                    AttributeError,
                    FileNotFoundError,
                    OSError,
                    RuntimeError,
                ) as exc:
                    logger.warning(
                        "_compute_artifact_hash: %s absent (%s)", tool_name, exc
                    )
                    tool_mtime = "absent"
                h.update(f"{tool_name}_mtime={tool_mtime}".encode())

    return h.hexdigest()


def _compute_hash(
    generator: Callable | Path,
    compile_kwargs: Mapping[str, Any],
    source_files: list[Path] | tuple[Path, ...],
    object_files: list[Path] | tuple[Path, ...],
    aiecc_flags: list[str] | tuple[str, ...],
    compile_flags: list[str] | tuple[str, ...],
    full_elf: bool = False,
    fold_ddr_addr_offset: bool = True,
    has_dispatch_params: bool = False,
    include_paths: list[Path] | tuple[Path, ...] = (),
) -> str:
    """Stable 24-hex SHA-256 cache key combining recipe + artifact hashes."""
    recipe = _compute_recipe_hash(
        generator, compile_kwargs, aiecc_flags, compile_flags, full_elf, include_paths
    )
    artifact = _compute_artifact_hash(
        generator,
        source_files,
        object_files,
        fold_ddr_addr_offset,
        has_dispatch_params,
    )
    return hashlib.sha256(f"{recipe}|{artifact}".encode()).hexdigest()[:24]
