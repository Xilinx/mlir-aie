# scratchpad_parameter.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""ScratchpadParameter: a named runtime value set from the host and read by Workers."""

from typing import cast

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects import aiex
from ..helpers.npdtypes import NpuDType
from ..helpers.sourceloc import SourceSite
from ..helpers.util import mlir_type_to_np_dtype, np_dtype_to_mlir_type
from .resolvable import Resolvable


class ScratchpadParameter(Resolvable):
    """A named runtime parameter communicated from host to AIE cores via the scratchpad mechanism.

    Declare a `ScratchpadParameter` at design time. Pass it to a
    [`Worker`][iron.Worker] via `fn_args` and call
    [`read`][iron.scratchpad_parameter.ScratchpadParameter.read] inside the
    `core_fn` to obtain its current value. The
    `--aie-lower-scratchpad-parameters` pass automatically inserts the
    necessary lock and scratchpad-sync preamble ops.

    Example:
        ```python
        import numpy as np
        from aie.iron import ScratchpadParameter, Worker, Runtime, Program

        seq_len = ScratchpadParameter("seq_len", np.int32)

        def core_body(p):
            v = p.read()
            ...

        worker = Worker(core_body, [seq_len])

        def sequence(out):
            # The compiler automatically inserts the parameter-sync preamble.
            ...

        rt = Runtime(sequence, [output_type])
        ```
    """

    def __init__(self, name: str, dtype: NpuDType):
        """Create a ScratchpadParameter.

        Args:
            name: Symbol name for the parameter (must be unique within the
                  device).
            dtype: The numpy scalar type (e.g. ``np.int32``, ``np.int16``,
                   ``bfloat16``).  ``np.float32`` is not supported -- the
                   scratchpad encoding zeroes the top 2 bits of the value,
                   which clobbers the sign and top exponent bits of an f32.
        """
        self._site = SourceSite.capture()
        self._name = name
        self._dtype = dtype
        self._resolved = False
        self._attributes: dict[str, str] = {}

    @classmethod
    def from_declaration(cls, op) -> "ScratchpadParameter":
        """Return the parameter an ``aiex.scratchpad_parameter`` op declares.

        It is declared again with every attribute of ``op``, so one that
        ``--aie-lower-scratchpad-parameters`` already lowered keeps its
        ``kind``, ``state_table_idx``, bounds and joint bounds.

        Args:
            op: The ``aiex.scratchpad_parameter`` declaration.

        Returns:
            The parameter, named and typed as ``op`` declares it.

        Raises:
            TypeError: The declared type has no NumPy dtype.
        """
        type_ = ir.TypeAttr(op.attributes["type"]).value
        dtype = mlir_type_to_np_dtype(type_)
        if dtype is None:
            raise TypeError(f"No NumPy dtype for scratchpad parameter {type_}.")
        parameter = cls(
            ir.StringAttr(op.attributes["sym_name"]).value, cast(NpuDType, dtype)
        )
        parameter._attributes = {
            name: str(op.attributes[name])
            for name in op.attributes
            if name not in ("sym_name", "type")
        }
        return parameter

    @property
    def name(self) -> str:
        """The symbol name of this parameter."""
        return self._name

    @property
    def dtype(self) -> NpuDType:
        """The numpy scalar type of this parameter."""
        return self._dtype

    @property
    def attributes(self) -> dict[str, str]:
        """The declaration's attributes beyond its name and type, as MLIR text."""
        return dict(self._attributes)

    def read(self) -> "ir.Value":
        """Emit `aiex.read_scratchpad_parameter` inside a core body.

        Must be called within an active MLIR insertion point (i.e. inside a
        Worker's `core_fn`).

        Returns:
            An MLIR SSA value of the parameter's type.
        """
        mlir_type = np_dtype_to_mlir_type(self._dtype)
        return aiex.read_scratchpad_parameter(self._name, mlir_type)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Emit ``aiex.scratchpad_parameter @name : type`` at module scope."""
        if not self._resolved:
            mlir_type = np_dtype_to_mlir_type(self._dtype)
            op = aiex.scratchpad_parameter(  # pyright: ignore[reportAttributeAccessIssue]
                self._name,
                mlir_type,
                loc=loc or self._site.location(self._name),
                ip=ip,
            )
            for name, text in self._attributes.items():
                op.attributes[name] = ir.Attribute.parse(text)
            self._resolved = True
