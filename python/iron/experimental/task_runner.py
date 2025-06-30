# task_runner.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import numpy as np
from typing import Sequence

from ...compiler.aiecc.main import run as aiecc_run
from ...utils.xrt import setup_aie, execute as execute_on_aie
from ...helpers.taplib import TensorTiler2D
from ..dataflow import ObjectFifo
from ..device import NPU1
from ..placers import SequentialPlacer
from ..program import Program
from ..runtime import Runtime
from ..worker import Worker

from .array import array


class TaskRunner:
    _INSTS = "npu_insts.bin"
    _XCLBIN = "final.xclbin"

    def __init__(
        self, module, input_arrs: Sequence[array], output_arrs: Sequence[array]
    ):
        self._module = module
        self._input_arrs = input_arrs
        self._output_arrs = output_arrs

    @classmethod
    def _aiecc_args(cls, xclbin, insts):
        return [
            "--aie-generate-xclbin",
            f"--xclbin-name={xclbin}",
            "--no-xchesscc",
            "--no-xbridge",
            "--aie-generate-npu",
            f"--npu-insts-name={insts}",
        ]

    def run(self):
        # Compile
        aiecc_run(self._module, self._aiecc_args(self._XCLBIN, self._INSTS))

        MAX_INPUTS = 2
        MAX_OUTPUTS = 1

        if len(self._input_arrs) > MAX_INPUTS:
            raise NotImplementedError(
                f"setup_aie XRT wrapper can only handle {MAX_INPUTS} inputs as present, but got {len(self._input_arrs)}"
            )
        if len(self._output_arrs) > MAX_OUTPUTS:
            raise NotImplementedError(
                f"setup_aie XRT wrapper can only handle {MAX_OUTPUTS} outputs as present, but got {len(self._output_arrs)}"
            )

        # Setup input/output
        kwargs = {}
        for i, arr in enumerate(self._input_arrs):
            kwargs[f"in_{i}_shape"] = arr._shape
            kwargs[f"in_{i}_dtype"] = arr._dtype
        for i in range(len(self._input_arrs), MAX_INPUTS):
            kwargs[f"in_{i}_shape"] = None
            kwargs[f"in_{i}_dtype"] = None

        kwargs[f"out_buf_shape"] = self._output_arrs[0]._shape
        kwargs[f"out_buf_dtype"] = self._output_arrs[0]._dtype

        app = setup_aie(
            self._XCLBIN,
            self._INSTS,
            **kwargs,
        )

        # Execute program and collect output
        aie_output = execute_on_aie(app, *[arr.asnumpy() for arr in self._input_arrs])
        self._output_arrs[0]._array = aie_output


def task_runner(
    task_fn,
    tiled_inputs: Sequence[tuple[array, Sequence[int]]],
    tiled_outputs: Sequence[tuple[array, Sequence[int]]],
    num_workers: int = 1,
) -> TaskRunner:

    tas_ins = []
    of_ins = [[] for _ in range(num_workers)]
    rt_types = []
    input_arrs = []
    for i, tiles_param in enumerate(tiled_inputs):
        arr, tile_shape = tiles_param
        input_arrs.append(arr)
        tile_type = np.ndarray[tile_shape, np.dtype[arr._dtype]]
        tas_ins.append(TensorTiler2D.simple_tiler(arr._shape, tile_shape))
        rt_types.append(np.ndarray[arr._shape, np.dtype[arr._dtype]])
        for w in range(num_workers):
            of_ins[w].append(
                ObjectFifo(tile_type, depth=arr._num_buffs, name=f"in{i}_{w}")
            )

    tas_outs = []
    of_outs = [[] for _ in range(num_workers)]
    output_arrs = []
    for i, tiles_param in enumerate(tiled_outputs):
        arr, tile_shape = tiles_param
        output_arrs.append(arr)
        tile_type = np.ndarray[tile_shape, np.dtype[arr._dtype]]
        tas_outs.append(TensorTiler2D.simple_tiler(arr._shape, tile_shape))
        rt_types.append(np.ndarray[arr._shape, np.dtype[arr._dtype]])
        for w in range(num_workers):
            of_outs[w].append(
                ObjectFifo(tile_type, depth=arr._num_buffs, name=f"out{i}_{w}")
            )

    def worker_wrapper(*args):
        datas = []
        for of in args:
            datas.append(of.acquire(1))
        task_fn(*datas)
        for of in args:
            of.release(1)

    workers = []
    for w in range(num_workers):
        args = [of_in.cons() for of_in in of_ins[w]]
        args += [of_out.prod() for of_out in of_outs[w]]
        workers.append(Worker(worker_wrapper, args))

    for i in range(num_workers):
        of_outs[i] = [of.cons() for of in of_outs[i]]

    rt = Runtime()
    with rt.sequence(*rt_types) as rt_buffers:
        rt.start(*workers)

        taps_idx = 0
        worker_idx = 0
        while taps_idx < len(tas_ins[0]):
            for i, tas in enumerate(tas_ins):
                rt.fill(of_ins[worker_idx][i].prod(), rt_buffers[i], tas[taps_idx])
            for i, tas in enumerate(tas_outs):
                rt.drain(
                    of_outs[worker_idx][i],
                    rt_buffers[i + len(tas_ins)],
                    tas[taps_idx],
                    wait=(i == len(tas_outs) - 1),
                )
            taps_idx += 1
            worker_idx = (worker_idx + 1) % num_workers

    my_program = Program(NPU1(), rt)
    module = my_program.resolve_program(SequentialPlacer())
    return TaskRunner(module, input_arrs, output_arrs)
