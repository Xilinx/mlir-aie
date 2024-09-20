# dma_transpose/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys
import numpy as np

from aie.api.dataflow.inout.simplefifoinout import SimpleFifoInOutSequence
from aie.api.dataflow.objectfifo import MyObjectFifo
from aie.api.dataflow.objectfifolink import MyObjectFifoLink
from aie.api.phys.device import NPU1Col1
from aie.api.program import MyProgram
from aie.api.worker import MyWorker

N = 4096
M = 64
K = 64

if len(sys.argv) == 3:
    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = M * K

my_dtype = np.uint32
obj_type = np.ndarray[my_dtype, (M, K)]

of_in = MyObjectFifo(2, obj_type, shim_endpoint=(0, 0))
of_out = MyObjectFifo(2, obj_type, shim_endpoint=(0, 0))

worker_program = MyWorker(None, [], coords=(0, 2))
my_link = MyObjectFifoLink([of_in.second], [of_out.first], coords=(0, 2))

inout_sequence = SimpleFifoInOutSequence(
    of_in.first,
    N,
    of_out.second,
    N,
    in_sizes=[1, K, M, 1],
    in_strides=[1, 1, K, 1],
    dtype=my_dtype,
)

my_program = MyProgram(
    NPU1Col1(),
    worker_programs=[worker_program],
    links=[my_link],
    inout_sequence=inout_sequence,
)
my_program.resolve_program()
