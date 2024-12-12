# globalbuffer.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
import numpy as np
from typing import Sequence

from .. import ir  # type: ignore
from ..dialects.aie import buffer
from ..helpers.util import (
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
)
from .device import PlacementTile
from .resolvable import Resolvable
from .placeable import Placeable


class GlobalBuffer(Resolvable, Placeable):
    """A buffer that is available both to Workers and to the Runtime for operations.
    This is often used for Runtime Parameters.
    """

    """This is used to generate unique names if none is given during construction"""
    __gbuf_index = 0

    def __init__(
        self,
        type: type[np.ndarray] | None = None,
        initial_value: np.ndarray | None = None,
        name: str | None = None,
        placement: PlacementTile | None = None,
        use_write_rtp: bool = False,
    ):
        """A GlobalBuffer is a memory region declared at the top-level of the design, allowing it to
        be accessed by both Workers and the Runtime.

        Args:
            type (type[np.ndarray] | None, optional): The type of the buffer. Defaults to None.
            initial_value (np.ndarray | None, optional): An initial value to set the buffer to. Should be of same datatype and shape as the buffer. Defaults to None.
            name (str | None, optional): The name of the buffer. If none is given, a unique name will be generated. Defaults to None.
            placement (PlacementTile | None, optional): A placement location for the buffer. Defaults to None.
            use_write_rtp (bool, optional): If use_write_rtp, write_rtp/read_rtp operations will be generated. Otherwise, traditional write/read operations will be used. Defaults to False.

        Raises:
            ValueError: Arguments are validated.
        """
        if type is None and initial_value is None:
            raise ValueError("Must provide either type, initial value, or both.")
        if type is None:
            type = np.ndarray[initial_value.shape, np.dtype[initial_value.dtype]]
        self._initial_value = initial_value
        self._name = name
        self._op = None
        self._arr_type = type
        if not self._name:
            self._name = f"buf_{self.__get_index()}"
        self._use_write_rtp = use_write_rtp
        Placeable.__init__(self, placement)

    @classmethod
    def __get_index(cls) -> int:
        idx = cls.__gbuf_index
        cls.__gbuf_index += 1
        return idx

    @property
    def shape(self) -> Sequence[int]:
        """The shape of the buffer"""
        return np_ndarray_type_get_shape(self._obj_type)

    @property
    def dtype(self) -> np.dtype:
        """The per-element datatype of the buffer."""
        return np_ndarray_type_get_dtype(self._obj_type)

    def __getitem__(self, idx):
        if self._op is None:
            return AttributeError(
                "Cannot index into GlobalBuffer before it has been resolved."
            )
        return self._op[idx]

    def __setitem__(self, idx, source):
        if self._op is None:
            return AttributeError(
                "Cannot index into GlobalBuffer before it has been resolved."
            )
        else:
            self._op[idx] = source

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._op:
            if not self._tile:
                raise ValueError("Cannot resolve buffer until it has been placed.")
            self._op = buffer(
                tile=self._tile.op,
                datatype=self._arr_type,
                name=self._name,
                initial_value=self._initial_value,
                use_write_rtp=self._use_write_rtp,
            )
