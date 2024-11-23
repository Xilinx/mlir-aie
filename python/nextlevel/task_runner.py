import numpy as np
from typing import Sequence

from ..compiler.aiecc.main import run as aiecc_run
from ..utils.xrt import setup_aie, execute as execute_on_aie
from ..helpers.taplib import TensorTiler2D
from ..iron.dataflow import ObjectFifo
from ..iron.phys.device import NPU1Col4
from ..iron.placers import SequentialPlacer
from ..iron.program import Program
from ..iron.runtime import Runtime
from ..iron.worker import Worker

from .array import array


class TaskRunner:
    _INSTS = "npu_insts.txt"
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
            "--aie-generate-cdo",
            f"--xclbin-name={xclbin}",
            "--no-xchesscc",
            "--no-xbridge",
            "--aie-generate-npu",
            f"--npu-insts-name={insts}",
        ]

    def run(self):
        aiecc_run(self._module, self._aiecc_args(self._XCLBIN, self._INSTS))

        args = []
        for arr in self._input_arrs:
            args.append(np.prod(arr._shape))
            args.append(arr._dtype)
        for arr in self._output_arrs:
            args.append(np.prod(arr._shape))
            args.append(arr._dtype)

        app = setup_aie(
            self._XCLBIN,
            self._INSTS,
            *args,
        )
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
            of_ins[w].append(ObjectFifo(arr._num_buffs, tile_type, f"in{i}_{w}"))

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
            of_outs[w].append(ObjectFifo(arr._num_buffs, tile_type, f"out{i}_{w}"))

    workers = []
    for w in range(num_workers):
        args = [of_in.cons for of_in in of_ins[w]]
        args += [of_out.prod for of_out in of_outs[w]]
        workers.append(Worker(task_fn, args))

    rt = Runtime()
    with rt.sequence(*rt_types) as rt_buffers:
        rt.start(*workers)

        taps_idx = 0
        worker_idx = 0
        while taps_idx < len(tas_ins[0]):
            for i, tas in enumerate(tas_ins):
                rt.fill(of_ins[worker_idx][i].prod, tas[taps_idx], rt_buffers[i])
            for i, tas in enumerate(tas_outs):
                rt.drain(
                    of_outs[worker_idx][i].cons,
                    tas[taps_idx],
                    rt_buffers[i + len(tas_ins)],
                    wait=(i == len(tas_outs) - 1),
                )
            taps_idx += 1
            worker_idx = (worker_idx + 1) % num_workers

    my_program = Program(NPU1Col4(), rt)
    module = my_program.resolve_program(SequentialPlacer())
    return TaskRunner(module, input_arrs, output_arrs)
