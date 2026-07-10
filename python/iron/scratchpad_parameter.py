# scratchpad_parameter.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""ScratchpadParameter: a named runtime value set from the host and read by Workers."""

import numpy as np

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects import aiex
from ..helpers.util import np_dtype_to_mlir_type, NpuDType
from .resolvable import Resolvable


class ScratchpadParameter(Resolvable):
    """A named runtime parameter communicated from host to AIE cores via the
    scratchpad mechanism.

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

        rt = Runtime()
        with rt.sequence(output_type) as out:
            # The compiler automatically inserts the parameter-sync preamble.
            ...
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
        self._name = name
        self._dtype = dtype
        self._resolved = False

    @property
    def name(self) -> str:
        """The symbol name of this parameter."""
        return self._name

    @property
    def dtype(self) -> NpuDType:
        """The numpy scalar type of this parameter."""
        return self._dtype

    def read(self):
        """Emit ``aiex.read_scratchpad_parameter`` inside a core body.

        Must be called within an active MLIR insertion point (i.e. inside a
        Worker's ``core_fn``).

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
            aiex.scratchpad_parameter(  # pyright: ignore[reportAttributeAccessIssue]
                self._name, mlir_type, loc=loc, ip=ip
            )
            self._resolved = True
