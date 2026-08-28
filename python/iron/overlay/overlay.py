# overlay.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""ProgramMemoryOverlay: one piece of code that can be written into a ProgramMemorySlot."""

from pathlib import Path


class ProgramMemoryOverlay:
    """Code that can be loaded into a [`ProgramMemorySlot`][iron.overlay.ProgramMemorySlot].

    A `ProgramMemoryOverlay` is not itself part of the MLIR design -- it never
    resolves to an op. It is an artifact reference:
    [`ProgramMemoryOverlayDesign`][iron.overlay.ProgramMemoryOverlayDesign]
    links it against the resident's symbols (at the slot it is assigned to)
    during the build's link step, and extracts its `.text` bytes for
    embedding. `slot.load(overlay)`, called from a `Runtime` sequence or a
    tile-sourced loader `Worker`'s `core_fn`, is what actually schedules those
    bytes to be written at run time.

    Only a pre-compiled object file is supported today
    (`object_file_name=`). Compiling an overlay from C++ source at JIT time,
    the way [`ExternalFunction`][iron.ExternalFunction] does for ordinary
    kernels, needs the same instance-registry integration `ExternalFunction`
    uses (`ExternalFunction._instances`, consumed by `@iron.jit`) and is not
    implemented yet -- construct with `object_file_name` and pre-compile with
    Peano directly in the meantime.
    """

    def __init__(
        self,
        name: str,
        slot: "ProgramMemorySlot",  # noqa: F821
        object_file_name: str | None = None,
        source_file: str | None = None,
        source_string: str | None = None,
    ):
        """Construct a ProgramMemoryOverlay.

        Args:
            name: Symbol name of the overlay's entry function. Must match the
                function name in `object_file_name`'s source.
            slot: The [`ProgramMemorySlot`][iron.overlay.ProgramMemorySlot]
                this overlay's `.text` will be linked at and written into.
                Its `arg_types` must match this overlay's signature -- there is
                one call site (the slot's), shared by every overlay assigned
                to it.
            object_file_name: Path to a pre-compiled object file containing
                `name`'s definition. Mutually exclusive with `source_file` /
                `source_string`, and the only one implemented today.
            source_file: Reserved for a future JIT-compiled overlay. Raises
                `NotImplementedError` today.
            source_string: Reserved for a future JIT-compiled overlay. Raises
                `NotImplementedError` today.
        """
        if source_file is not None or source_string is not None:
            raise NotImplementedError(
                f"ProgramMemoryOverlay '{name}': compiling an overlay from C++ "
                f"source at JIT time is not implemented yet. Pre-compile with "
                f"Peano and pass object_file_name= instead."
            )
        if object_file_name is None:
            raise ValueError(
                f"ProgramMemoryOverlay '{name}': object_file_name is required "
                f"(source_file/source_string are not implemented yet)."
            )
        if not name:
            raise ValueError("ProgramMemoryOverlay name cannot be empty.")
        self._name = name
        self._slot = slot
        self._object_file_name = object_file_name
        slot._register_overlay(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def slot(self) -> "ProgramMemorySlot":  # noqa: F821
        return self._slot

    @property
    def object_file_name(self) -> str:
        return self._object_file_name

    def __repr__(self) -> str:
        return f"ProgramMemoryOverlay({self._name!r}, slot={self._slot.name!r})"
