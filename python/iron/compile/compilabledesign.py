# compilabledesign.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""CompilableDesign: bundles an MLIR generator with its compile-time configuration.

``CompilableDesign`` is the central mid-level abstraction.  It pairs an MLIR
generator function (or a path to a ``.mlir`` file) with explicit compile-time
parameters and produces an ``xclbin`` + ``insts.bin`` artifact pair via
``compile()``.

The compile-time parameters (``compile_kwargs``) are bound at construction
time.  No runtime tensors are required to compile; only the ``Compile[T]``-
annotated parameters need to be supplied.

Hashing uses ``SHA256(generator_bytecode + compile_kwargs_json +
source_file_mtimes + flags)`` — no MLIR generation is needed for a cache
lookup, making it significantly faster than the legacy MLIR-string hash.
"""

from __future__ import annotations

import builtins
import hashlib
import inspect
import json
import logging
import os
import pickle
import sys
import types
import typing
from pathlib import Path
from typing import Any, Callable, get_args, get_origin

from aie.utils.compile import (
    NPU_CACHE_HOME,
    compile_external_kernel,
    compile_mlir_module,
)
from aie.utils.compile.cache.utils import file_lock
from aie.utils.compile.utils import _cleanup_failed_compilation
from aie.extras.context import mlir_mod_ctx

from .context import compile_context
from .markers import Compile, In, InOut, Out

logger = logging.getLogger(__name__)

_BUILTIN_NAMES = set(dir(builtins))
_PRIMITIVE_TYPES = (int, float, str, bool, bytes)


def _collect_co_names(code) -> set:
    """Recursively collect all co_names from *code* and nested code objects."""
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _collect_co_names(const)
    return names


def _hash_captured_globals(generator) -> str:
    """Return a SHA-256 hex digest of the primitive globals referenced by *generator*.

    Recursively collects all ``co_names`` from the generator's code object and
    nested code objects, looks each name up in the generator's global namespace,
    and hashes the primitive-scalar values it finds.  Modules, types, builtins,
    and callables are skipped.  Non-primitive values are hashed via pickle; if
    pickle fails the value is skipped with a debug log.
    """
    all_names = _collect_co_names(generator.__code__)
    globs = generator.__globals__
    parts = []
    for name in sorted(all_names):
        val = globs.get(name)
        if val is None:
            continue
        if isinstance(val, (types.ModuleType, type)):
            continue
        if name in _BUILTIN_NAMES:
            continue
        if callable(val):
            continue
        if isinstance(val, _PRIMITIVE_TYPES):
            parts.append(f"{name}={repr(val)}")
            continue
        if isinstance(val, (tuple, list)) and all(
            isinstance(v, _PRIMITIVE_TYPES) for v in val
        ):
            parts.append(f"{name}={repr(val)}")
            continue
        # Non-primitive: try pickle
        try:
            digest = hashlib.sha256(pickle.dumps(val)).hexdigest()
            parts.append(f"{name}=pickle:{digest}")
        except Exception as exc:
            logger.debug("_hash_captured_globals: skipping %r (%s)", name, exc)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _encode_kwarg(value: Any) -> Any:
    """Encode a compile_kwarg value for JSON storage with type information."""
    if isinstance(value, bool):
        return {"__type__": "bool", "__value__": value}
    if isinstance(value, int):
        return {"__type__": "int", "__value__": value}
    if isinstance(value, float):
        return {"__type__": "float", "__value__": value}
    if isinstance(value, str):
        return {"__type__": "str", "__value__": value}
    # Fallback: store as string with a marker so from_json can warn.
    return {"__type__": "unknown", "__value__": str(value)}


def _decode_kwarg(encoded: Any) -> Any:
    """Decode a compile_kwarg value from JSON storage."""
    if not isinstance(encoded, dict) or "__type__" not in encoded:
        # Legacy format (plain string) — return as-is.
        return encoded
    t = encoded["__type__"]
    v = encoded["__value__"]
    if t == "bool":
        return bool(v)
    if t == "int":
        return int(v)
    if t == "float":
        return float(v)
    if t == "str":
        return str(v)
    # Unknown type — return the string value with a warning.
    logger.warning(
        "from_json: compile_kwarg value of unknown type %r decoded as string", t
    )
    return str(v)


class _TensorPlaceholder:
    """Sentinel passed for In/Out/InOut params during MLIR generation.

    Raises a descriptive ``RuntimeError`` if the generator body tries to read
    any attribute (e.g. ``.shape``, ``.dtype``, ``.size``) from a runtime
    tensor parameter.  This enforces the contract that generator bodies must
    not depend on tensor values at compile time — all shape/dtype information
    must come from ``Compile[T]`` parameters instead.
    """

    def __init__(self, param_name: str) -> None:
        object.__setattr__(self, "_param_name", param_name)

    def _raise(self, op: str = "") -> None:
        name = object.__getattribute__(self, "_param_name")
        raise RuntimeError(
            f"Generator parameter {name!r} is an In/Out/InOut runtime tensor "
            f"and is not available at compile time{': ' + op if op else ''}.\n"
            f"Move shape and dtype information into Compile[T] parameters:\n\n"
            f"  # Instead of:\n"
            f"  def gen({name}: In, *, ...):\n"
            f"      N = {name}.shape[0]   # ← fails: {name!r} is a runtime tensor\n\n"
            f"  # Write:\n"
            f"  def gen({name}: In, *, N: Compile[int], dtype: Compile[type] = np.float32):\n"
            f"      tensor_ty = np.ndarray[(N,), np.dtype[dtype]]"
        )

    def __getattr__(self, name: str):
        self._raise(f".{name}")

    def __setattr__(self, name: str, value) -> None:
        self._raise(f".{name} = ...")

    def __getitem__(self, key):
        self._raise(f"[{key!r}]")

    def __len__(self) -> int:
        self._raise("len()")
        return 0  # unreachable

    def __iter__(self):
        self._raise("iter()")

    def __bool__(self) -> bool:
        self._raise("bool()")
        return False  # unreachable

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_param_name")
        return f"<_TensorPlaceholder for {name!r}>"


# Sentinel: annotation origins that represent runtime tensor directions.
_TENSOR_ANNOTATIONS = (In, Out, InOut)


def _is_compile_param(annotation) -> bool:
    """Return True if *annotation* is ``Compile[T]`` (or bare ``Compile``)."""
    if annotation is Compile:
        return True
    origin = get_origin(annotation)
    return origin is Compile


def _is_tensor_param(annotation) -> bool:
    """Return True if *annotation* is ``In``, ``Out``, or ``InOut``."""
    return annotation in _TENSOR_ANNOTATIONS


def _split_params(generator: Callable) -> tuple[list[str], list[str], list[str]]:
    """Inspect *generator* and return (compile_params, tensor_params, scalar_params).

    * ``compile_params``  — names with ``Compile[T]`` annotation
    * ``tensor_params``   — names with ``In``/``Out``/``InOut`` annotation (in order)
    * ``scalar_params``   — names with any other annotation (runtime scalars)

    Uses ``typing.get_type_hints()`` so that stringified annotations (produced
    by ``from __future__ import annotations`` or PEP 563 mode) are evaluated
    correctly.  Falls back to ``inspect.signature`` annotations on any error
    (e.g. when the generator's globals are not resolvable at call time).
    """
    compile_params: list[str] = []
    tensor_params: list[str] = []
    scalar_params: list[str] = []

    # get_type_hints() evaluates string annotations (from __future__ import
    # annotations / PEP 563). Falls back to {} on any resolution error.
    try:
        hints = typing.get_type_hints(generator)
    except Exception as exc:
        logger.debug("get_type_hints failed for %r: %s", generator, exc)
        hints = {}

    sig = inspect.signature(generator)
    for name, param in sig.parameters.items():
        # Prefer the resolved hint; fall back to the raw annotation.
        ann = hints.get(name, param.annotation)
        if ann is inspect.Parameter.empty:
            # Unannotated — treat as scalar.
            scalar_params.append(name)
        elif _is_compile_param(ann):
            compile_params.append(name)
        elif _is_tensor_param(ann):
            tensor_params.append(name)
        else:
            scalar_params.append(name)

    return compile_params, tensor_params, scalar_params


def _compute_hash(
    generator: Callable | Path,
    compile_kwargs: dict[str, Any],
    source_files: list[Path],
    object_files: list[Path],
    aiecc_flags: list[str],
    compile_flags: list[str],
) -> str:
    """Compute a stable SHA-256 cache key without generating MLIR.

    Components:
    1. Generator bytecode (``co_code`` + ``co_consts``) — or file path for .mlir.
    2. Sorted ``compile_kwargs`` JSON.
    3. ``(path, mtime)`` pairs for each source file.
    4. Sorted ``aiecc_flags`` + ``compile_flags``.
    """
    h = hashlib.sha256()

    if isinstance(generator, Path):
        # Static .mlir file: hash the path and its mtime.
        h.update(str(generator).encode())
        try:
            h.update(str(generator.stat().st_mtime).encode())
        except FileNotFoundError:
            pass
    else:
        code = generator.__code__
        h.update(code.co_code)
        h.update(repr(code.co_consts).encode())
        # Include qualname so that two structurally identical functions defined
        # in different scopes (e.g. gen_a vs gen_b in the same module) hash
        # differently when their qualname reflects their definition context.
        h.update(getattr(generator, "__qualname__", "").encode())
        h.update(getattr(generator, "__module__", "").encode())

    # Compile kwargs — sort for determinism.
    try:
        kwargs_json = json.dumps(
            {k: str(v) for k, v in sorted(compile_kwargs.items())}
        ).encode()
    except (TypeError, ValueError):
        kwargs_json = repr(sorted(compile_kwargs.items())).encode()
    h.update(kwargs_json)

    # Source file mtimes.
    for sf in sorted(source_files, key=str):
        h.update(str(sf).encode())
        try:
            h.update(str(Path(sf).stat().st_mtime).encode())
        except (FileNotFoundError, OSError):
            pass

    # Object file mtimes.
    for of in sorted(object_files, key=str):
        h.update(str(of).encode())
        try:
            h.update(str(Path(of).stat().st_mtime).encode())
        except (FileNotFoundError, OSError):
            pass

    # Flags.
    h.update(repr(sorted(aiecc_flags)).encode())
    h.update(repr(sorted(compile_flags)).encode())

    # Platform/hardware identifier — only for callable generators.
    # A static .mlir file is architecture-agnostic; compiled kernels are not.
    if not isinstance(generator, Path):
        try:
            from aie.utils import DefaultNPURuntime

            device = (
                DefaultNPURuntime.device() if DefaultNPURuntime is not None else None
            )
            from aie.utils.compile.utils import resolve_target_arch

            target_arch = resolve_target_arch(device)
        except Exception:
            target_arch = "unknown"

        try:
            from aie.utils import config as _config

            peano_cxx = _config.peano_cxx_path()
            peano_mtime = str(Path(peano_cxx).stat().st_mtime)
        except Exception:
            try:
                from aie.utils import config as _config

                peano_mtime = f"path:{_config.peano_install_dir()}"
            except Exception:
                peano_mtime = "absent"

        try:
            import shutil as _shutil

            _aiecc_path = _shutil.which("aiecc")
            aiecc_mtime = (
                str(Path(_aiecc_path).stat().st_mtime) if _aiecc_path else "absent"
            )
        except Exception:
            aiecc_mtime = "absent"

        h.update(
            f"target_arch={target_arch}|peano_mtime={peano_mtime}|aiecc_mtime={aiecc_mtime}".encode()
        )

    # Captured module-level globals (primitive scalars and homogeneous containers).
    if not isinstance(generator, Path):
        global_hash = _hash_captured_globals(generator)
        h.update(global_hash.encode())

    return h.hexdigest()[:24]


class CompilableDesign:
    """Bundles an MLIR generator with compile-time parameters.

    Args:
        mlir_generator: A callable that accepts ``Compile[T]`` kwargs and
            returns an MLIR module (unplaced style) or ``None`` (placed style),
            OR a ``pathlib.Path`` to a pre-written ``.mlir`` file.
        use_cache: When ``True`` (default), a file-system cache keyed by the
            bytecode+kwargs hash is consulted before recompiling.
        compile_kwargs: Values for the ``Compile[T]``-annotated parameters.
            Validated against the generator signature via ``inspect.Signature.bind``.
        compile_flags: Extra flags forwarded to the Peano C++ compiler.
        source_files: Paths to C++ kernel source files.  Their mtimes are
            included in the cache key so that edits correctly invalidate the cache.
        include_paths: Extra ``-I`` paths forwarded to the C++ compiler.
        aiecc_flags: Extra flags forwarded to ``aiecc``.
        object_files: Pre-compiled ``.o`` files to link with.
    """

    def __init__(
        self,
        mlir_generator: Callable | Path,
        *,
        use_cache: bool = True,
        compile_kwargs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
        source_files: list[str | Path] | None = None,
        include_paths: list[str | Path] | None = None,
        aiecc_flags: list[str] | None = None,
        object_files: list[str | Path] | None = None,
    ):
        self.mlir_generator = mlir_generator
        self.use_cache = use_cache
        self.compile_kwargs: dict[str, Any] = dict(compile_kwargs or {})
        self.compile_flags: list[str] = list(compile_flags or [])
        self.source_files: list[Path] = [Path(sf) for sf in (source_files or [])]
        self.include_paths: list[Path] = [Path(p) for p in (include_paths or [])]
        self.aiecc_flags: list[str] = list(aiecc_flags or [])
        self.object_files: list[Path] = [Path(of) for of in (object_files or [])]

        # Cached artifact paths (set after compile()).
        self._xclbin_path: Path | None = None
        self._inst_path: Path | None = None
        self._expected_tensor_sizes: list[int] | None = None

        # Introspect generator signature to split param categories.
        if callable(mlir_generator):
            (
                self._compile_params,
                self._tensor_params,
                self._scalar_params,
            ) = _split_params(mlir_generator)
        else:
            self._compile_params = []
            self._tensor_params = []
            self._scalar_params = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self) -> tuple[Path, Path]:
        """Compile the generator to ``(xclbin_path, inst_path)``.

        Checks the file-system cache first (when ``use_cache=True``).  On a
        cache miss, calls the generator with ``compile_kwargs``, compiles any
        ``ExternalFunction`` instances discovered, then invokes ``aiecc``.

        Returns:
            ``(xclbin_path, inst_path)`` — paths to the compiled artifacts.
        """
        from aie.iron.kernel import ExternalFunction
        from aie.utils import DefaultNPURuntime

        cache_hash = self._compute_cache_hash()
        kernel_dir = NPU_CACHE_HOME / cache_hash
        lock_file_path = kernel_dir / ".lock"
        xclbin_path = kernel_dir / "final.xclbin"
        inst_path = kernel_dir / "insts.bin"

        with file_lock(lock_file_path):
            os.makedirs(kernel_dir, exist_ok=True)

            xclbin_exists = xclbin_path.exists()
            inst_exists = inst_path.exists()

            if self.use_cache and xclbin_exists and inst_exists:
                logger.debug(
                    "Cache hit for '%s' (hash=%s)", self._generator_name(), cache_hash
                )
                self._xclbin_path = xclbin_path
                self._inst_path = inst_path
                return xclbin_path, inst_path

            logger.debug(
                "Cache miss for '%s' (hash=%s); compiling...",
                self._generator_name(),
                cache_hash,
            )

            try:
                mlir_module = self._generate_mlir(ExternalFunction)

                # Determine target architecture.
                from aie.utils.compile import resolve_target_arch

                device = (
                    DefaultNPURuntime.device()
                    if DefaultNPURuntime is not None
                    else None
                )
                target_arch = resolve_target_arch(device)

                # Compile any ExternalFunction kernels created during generation.
                external_kernels = list(ExternalFunction._instances)
                ExternalFunction._instances.clear()
                for func in external_kernels:
                    if not func._compiled:
                        compile_external_kernel(func, kernel_dir, target_arch)

                compile_mlir_module(
                    mlir_module=mlir_module,
                    insts_path=inst_path,
                    xclbin_path=xclbin_path,
                    work_dir=kernel_dir,
                )

                # Verify that the expected output files were actually created.
                # aiecc may exit with code 0 even when xclbin generation fails
                # silently (e.g. missing xclbinutil or bootgen), so we must
                # check the files exist before treating compilation as a success.
                missing = [
                    p for p in (xclbin_path, inst_path) if not p.exists()
                ]
                if missing:
                    raise RuntimeError(
                        "[aiecc] Compilation appeared to succeed (exit code 0) "
                        "but expected output file(s) were not created: "
                        + ", ".join(str(p) for p in missing)
                    )
            except Exception:
                _cleanup_failed_compilation(kernel_dir)
                raise

        self._xclbin_path = xclbin_path
        self._inst_path = inst_path
        # Parse expected tensor sizes for runtime validation.
        self._expected_tensor_sizes = self._parse_expected_tensor_sizes(kernel_dir)
        return xclbin_path, inst_path

    def get_artifacts(self) -> tuple[Path, Path] | None:
        """Return cached artifact paths without recompiling, or ``None``."""
        if self._xclbin_path is None or self._inst_path is None:
            return None
        return self._xclbin_path, self._inst_path

    def split_runtime_args(
        self, runtime_args: tuple, runtime_kwargs: dict[str, Any]
    ) -> tuple[list, dict[str, Any]]:
        """Split ``runtime_args``/``runtime_kwargs`` into tensor list and scalar dict.

        Uses the ``In``/``Out``/``InOut`` annotation order from the generator
        signature.  Positional ``runtime_args`` are consumed left-to-right to
        fill tensor params (in signature order), then scalar params.

        ``Kernel`` and ``ExternalFunction`` instances are compile-time-only
        objects resolved at link time; they are silently filtered out and
        never forwarded to the NPU kernel as runtime arguments.

        Returns:
            ``(tensor_args, scalar_kwargs)``
        """
        from aie.iron.kernel import ExternalFunction, Kernel

        if not callable(self.mlir_generator):
            # Static .mlir file: pass everything through as tensors,
            # but still filter compile-time-only kernel objects.
            runtime_args = [a for a in runtime_args if not isinstance(a, Kernel)]
            return runtime_args, runtime_kwargs

        tensor_args = []
        scalar_kwargs = dict(runtime_kwargs)

        # Use get_type_hints() to resolve stringified annotations
        # (from __future__ import annotations / PEP 563).
        try:
            hints = typing.get_type_hints(self.mlir_generator)
        except Exception:
            hints = {}

        sig = inspect.signature(self.mlir_generator)
        params = [
            (name, p)
            for name, p in sig.parameters.items()
            if name not in self.compile_kwargs
        ]

        # Walk the non-compile parameters in order, consuming positional args.
        # Kernel/ExternalFunction instances are compile-time only; skip them
        # in the positional stream so they never land in tensor_args or
        # scalar_kwargs.
        def _next_non_kernel(it):
            while True:
                val = next(it)
                if not isinstance(val, Kernel):
                    return val

        pos_iter = iter(runtime_args)
        for name, param in params:
            ann = hints.get(name, param.annotation)
            if _is_tensor_param(ann):
                # Try positional first, then kwargs.
                if name in scalar_kwargs:
                    tensor_args.append(scalar_kwargs.pop(name))
                else:
                    try:
                        tensor_args.append(_next_non_kernel(pos_iter))
                    except StopIteration:
                        pass
            else:
                # Scalar param: leave in scalar_kwargs (already there from kwargs)
                # or consume from positional.
                if name not in scalar_kwargs:
                    try:
                        val = _next_non_kernel(pos_iter)
                        scalar_kwargs[name] = val
                    except StopIteration:
                        pass

        return tensor_args, scalar_kwargs

    def generate_mlir(self):
        """Generate and return the MLIR module without compiling to xclbin.

        Useful for inspecting generated MLIR, debugging, or offline analysis.
        Does not require an NPU or XRT to be present.

        Returns:
            The generated ``mlir.ir.Module``.
        """
        from aie.iron.kernel import ExternalFunction

        return self._generate_mlir(ExternalFunction)

    def validate_tensor_args(self, tensor_args: list) -> None:
        """Validate that *tensor_args* element counts match the compiled kernel.

        Compares each tensor's element count against the DMA transfer sizes
        extracted from the compiled ``aiex.runtime_sequence``.  Raises
        ``RuntimeError`` with a clear message if a mismatch is detected.

        For parallel/distributed kernels, work is split across N AIE columns
        and each logical tensor maps to N DMA ops of size ``total/N``.
        ``_parse_expected_tensor_sizes`` returns all N per-column sizes.  To
        avoid false positives in this case, validation is skipped for a tensor
        whose element count is an exact non-zero multiple of the expected DMA
        size (i.e. ``actual % expected == 0`` and ``actual > 0``).  A true
        mismatch (e.g. 1000 elements vs 128-element DMA) does not divide
        evenly, so the error is still raised.

        No-op when expected sizes are unavailable (e.g. offline compilation
        or when ``input_with_addresses.mlir`` was not produced).
        """
        if not self._expected_tensor_sizes:
            return
        import numpy as np

        for i, (tensor, expected) in enumerate(
            zip(tensor_args, self._expected_tensor_sizes)
        ):
            try:
                actual = int(np.size(tensor))
            except Exception:
                continue
            # Skip if actual is an exact positive multiple of expected — this
            # covers parallel/distributed kernels where one logical tensor maps
            # to multiple per-column DMA ops each of size (total / N).
            if actual > 0 and expected > 0 and actual % expected == 0:
                continue
            if actual != expected:
                param_name = (
                    self._tensor_params[i]
                    if i < len(self._tensor_params)
                    else f"arg[{i}]"
                )
                raise RuntimeError(
                    f"Tensor argument {param_name!r} has {actual} elements but "
                    f"the kernel was compiled for {expected} elements.\n"
                    f"Compile[T] parameters used at compile time: "
                    f"{self.compile_kwargs!r}"
                )

    def to_json(self) -> str:
        """Serialise the non-callable parts of this design to JSON.

        The generator callable itself cannot be serialised; callers must
        supply it back to ``from_json``.
        """
        data = {
            "generator_name": self._generator_name(),
            "use_cache": self.use_cache,
            "compile_kwargs": {
                k: _encode_kwarg(v) for k, v in self.compile_kwargs.items()
            },
            "compile_flags": self.compile_flags,
            "source_files": [str(sf) for sf in self.source_files],
            "include_paths": [str(p) for p in self.include_paths],
            "aiecc_flags": self.aiecc_flags,
            "object_files": [str(of) for of in self.object_files],
            "cache_hash": self._compute_cache_hash(),
        }
        return json.dumps(data)

    @classmethod
    def from_json(
        cls, json_str: str, generator: Callable | None = None
    ) -> CompilableDesign:
        """Deserialise a ``CompilableDesign`` from JSON.

        Args:
            json_str: JSON string produced by ``to_json()``.
            generator: The original callable (required unless ``mlir_generator``
                is a ``.mlir`` file path encoded in the JSON).

        Note:
            ``compile_kwargs`` values for the types ``int``, ``float``,
            ``str``, and ``bool`` are round-tripped exactly.  Values of other
            types are stored as strings and decoded as strings.
        """
        data = json.loads(json_str)
        if generator is None:
            raise ValueError(
                "generator must be supplied to CompilableDesign.from_json() "
                "because callables cannot be serialised."
            )
        return cls(
            mlir_generator=generator,
            use_cache=data.get("use_cache", True),
            compile_kwargs={
                k: _decode_kwarg(v) for k, v in data.get("compile_kwargs", {}).items()
            },
            compile_flags=data.get("compile_flags", []),
            source_files=data.get("source_files", []),
            include_paths=data.get("include_paths", []),
            aiecc_flags=data.get("aiecc_flags", []),
            object_files=data.get("object_files", []),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_expected_tensor_sizes(self, kernel_dir: Path) -> list[int] | None:
        """Parse expected DMA transfer sizes from the lowered MLIR.

        Reads ``input_with_addresses.mlir`` from *kernel_dir* and extracts
        the element counts for each host-side DMA transfer inside the
        ``aie.runtime_sequence`` block.

        The lowered MLIR uses ``aie.dma_bd`` with positional arguments::

            aie.dma_bd(%arg0 : memref<1024xi32>, 0, 1024, [...]) {...}
            #                                       ^^^^
            #                                   element count

        Only ``aie.dma_bd`` lines that reference a runtime sequence parameter
        (``%argN``) are counted; tile-internal buffer ops (which reference
        named SSA values like ``%out_buff_0``) are excluded.

        Returns a list of element counts in transfer order, or ``None`` if the
        file is absent or unparseable.
        """
        mlir_path = kernel_dir / "input_with_addresses.mlir"
        if not mlir_path.exists():
            return None
        try:
            import re

            text = mlir_path.read_text()

            # Isolate the aie.runtime_sequence body so we only look at the
            # host-facing DMA descriptors, not tile-internal aie.mem ops.
            # Use brace counting to find the matching closing brace, because
            # the sequence body contains nested braces (dma_configure_task_for
            # regions, attribute dicts, etc.).
            seq_start = re.search(r"aie\.runtime_sequence\s*\([^)]*\)\s*\{", text)
            if not seq_start:
                return None
            pos = seq_start.end()
            depth = 1
            while pos < len(text) and depth > 0:
                if text[pos] == "{":
                    depth += 1
                elif text[pos] == "}":
                    depth -= 1
                pos += 1
            seq_body = text[seq_start.end() : pos - 1]

            # Match aie.dma_bd lines that start with a %argN parameter
            # (runtime sequence argument), not a named tile buffer.
            # Format: aie.dma_bd(%argN : memref<...>, <offset>, <count>, [...])
            # The element count is the integer after the offset.
            sizes = []
            for m in re.finditer(
                r"aie\.dma_bd\s*\(\s*%arg\d+\s*:[^,]+,\s*\d+\s*,\s*(\d+)",
                seq_body,
            ):
                sizes.append(int(m.group(1)))
            return sizes if sizes else None
        except Exception:
            return None

    def _generator_name(self) -> str:
        if isinstance(self.mlir_generator, Path):
            return str(self.mlir_generator)
        return getattr(self.mlir_generator, "__name__", repr(self.mlir_generator))

    def _compute_cache_hash(self) -> str:
        return _compute_hash(
            self.mlir_generator,
            self.compile_kwargs,
            self.source_files,
            self.object_files,
            self.aiecc_flags,
            self.compile_flags,
        )

    def _generate_mlir(self, ExternalFunction):
        """Call the generator (or read the .mlir file) and return the MLIR module."""
        if isinstance(self.mlir_generator, Path):
            # Static MLIR file.
            mlir_path = self.mlir_generator
            with mlir_mod_ctx() as ctx:
                ctx.module.parse(mlir_path.read_text())
            return ctx.module

        # Validate that all Compile[T] params are supplied.
        try:
            hints = typing.get_type_hints(self.mlir_generator)
        except Exception:
            hints = {}

        # Guard 2-A: compile_kwargs must not contain tensor param names.
        tensor_names = set(self._tensor_params)
        confused_tensor_keys = set(self.compile_kwargs.keys()) & tensor_names
        if confused_tensor_keys:
            raise TypeError(
                f"CompilableDesign for {self._generator_name()!r}: "
                f"compile_kwargs contains name(s) annotated as runtime tensors "
                f"(In/Out/InOut), not Compile[T] parameters: {confused_tensor_keys}.\n"
                f"  Tensor params must be supplied at call time, not compile time.\n"
                f"  Compile[T] params are: {self._compile_params}."
            )

        # Guard 2-B: compile_kwargs must not contain entirely unknown keys.
        known_params = (
            set(self._compile_params)
            | set(self._tensor_params)
            | set(self._scalar_params)
        )
        unknown_keys = set(self.compile_kwargs.keys()) - known_params
        if unknown_keys:
            raise TypeError(
                f"CompilableDesign for {self._generator_name()!r}: "
                f"compile_kwargs contains key(s) not in the generator signature: "
                f"{unknown_keys}.\n"
                f"  Valid Compile[T] params are: {self._compile_params}."
            )

        sig = inspect.signature(self.mlir_generator)
        compile_only_params = {
            name: p
            for name, p in sig.parameters.items()
            if _is_compile_param(hints.get(name, p.annotation))
        }
        compile_only_sig = inspect.Signature(
            parameters=list(compile_only_params.values())
        )
        try:
            compile_only_sig.bind(**self.compile_kwargs)
        except TypeError as exc:
            raise TypeError(
                f"CompilableDesign for '{self._generator_name()}': "
                f"compile_kwargs do not match Compile[T] parameters — {exc}"
            ) from exc

        # Clear stale ExternalFunction instances before generation.
        ExternalFunction._instances.clear()

        # Build the call kwargs: Compile[T] params from compile_kwargs,
        # plus None placeholders for In/Out/InOut params (which are not
        # available at compile time — the generator must not read them).
        _tensor_placeholders = {
            name: _TensorPlaceholder(name) for name in self._tensor_params
        }
        _gen_call_kwargs = {**_tensor_placeholders, **self.compile_kwargs}

        # Re-register any ExternalFunction instances passed as Compile[T] params
        # so that compile() collects them for compilation after generation returns.
        for _v in _gen_call_kwargs.values():
            if isinstance(_v, ExternalFunction):
                ExternalFunction._instances.add(_v)

        with compile_context(**self.compile_kwargs):
            with mlir_mod_ctx() as ctx:
                result = self.mlir_generator(**_gen_call_kwargs)

        module = ctx.module if result is None else result
        if not module.operation.verify():
            raise RuntimeError(
                f"MLIR verification failed for '{self._generator_name()}'"
            )
        return module

    def __hash__(self) -> int:
        # Fold the 96-bit cache hash down to a signed 64-bit Python hash.
        h = int(self._compute_cache_hash(), 16)
        # Wrap to the range [-(2^63), 2^63-1] by taking modulo and adjusting.
        bits = sys.hash_info.width  # typically 64
        mask = (1 << bits) - 1
        h = h & mask
        if h >= (1 << (bits - 1)):
            h -= 1 << bits
        return h if h != -1 else -2  # -1 is reserved by CPython

    def __repr__(self) -> str:
        return (
            f"CompilableDesign(generator={self._generator_name()!r}, "
            f"compile_kwargs={self.compile_kwargs!r})"
        )
