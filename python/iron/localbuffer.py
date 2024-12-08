# localbuffer.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
from collections import defaultdict
import numpy as np

from ..dialects.aie import buffer
from .worker import Worker
from .device import Tile


class LocalBuffer(buffer):
    __buf_tile_index = defaultdict(int)

    def __init__(
        self,
        type: type[np.ndarray] | None = None,
        initial_value: np.ndarray | None = None,
        name: str | None = None,
    ):
        if type is None and initial_value is None:
            raise ValueError("Must provide either type, initial value, or both.")
        if type is None:
            type = np.ndarray[initial_value.shape, np.dtype[initial_value.dtype]]

        # Use context to get current core placement
        current_core_placement = Worker.current_core_placement.get()
        if current_core_placement is None:
            raise ValueError("Can only instantiate Buffer() in a worker function.")

        if not name:
            name = f"buf_{current_core_placement.col}_{current_core_placement.row}_{self.__get_index(current_core_placement)}"
        super().__init__(
            tile=current_core_placement,
            datatype=type,
            name=name,
            initial_value=initial_value,
        )

    @classmethod
    def __get_index(cls, placement: Tile) -> int:
        idx = cls.__buf_tile_index[placement]
        cls.__buf_tile_index[placement] += 1
        return idx
