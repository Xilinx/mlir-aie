# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A memtile->core objectFIFO holding compile-time initValues, read once per
# dispatch, over 1000 dispatches on one hardware context and with no
# dma_channel_reset_for. The MemTile's source locks used to be armed with one
# pass worth of tokens, so the second dispatch hung waiting for weights.
#
# REQUIRES: ryzen_ai_npu2, peano
#
# RUN: %python %S/aie2.py > ./aie2.mlir
# RUN: %aiecc --get-xclbin --get-npu-insts --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: %host_clang %S/../dma_channel_reset_for/test.cpp -o test.exe -std=c++17 -Wall -Wextra %xrt_flags %host_link_flags
# RUN: %run_on_npu2% ./test.exe 1000 | FileCheck %s
#
# CHECK: PASS: 1000 exact dispatches on one hardware context

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint
from aie.iron.device import NPU2, Tile

N = 256
TILE = 16
N_TILES = N // TILE


def build_design():
    vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]

    weights = ObjectFifo(
        vector_ty,
        depth=1,
        name="weights",
        init_values=[np.arange(1, N + 1, dtype=np.int32)],
    )
    # No Worker produces the weights: they start out in the MemTile's buffer.
    weights.prod().endpoint = ObjectFifoEndpoint(Tile(1, 1))
    inputs = ObjectFifo(tile_ty, depth=2, name="inputs")
    outputs = ObjectFifo(tile_ty, depth=2, name="outputs")

    def core_body(weights_in, inputs_in, outputs_out):
        weight = weights_in.acquire(1)
        for tile_index in range_(N_TILES):
            input_tile = inputs_in.acquire(1)
            output_tile = outputs_out.acquire(1)
            for element in range_(TILE):
                index = tile_index * TILE + element
                output_tile[element] = input_tile[element] + weight[index]
            inputs_in.release(1)
            outputs_out.release(1)
        weights_in.release(1)

    worker = Worker(
        core_body,
        fn_args=[weights.cons(), inputs.cons(), outputs.prod()],
        tile=Tile(1, 2),
    )

    def sequence(input_vector, output_vector, inputs_in, outputs_out):
        inputs_in.fill(input_vector)
        outputs_out.drain(output_vector, wait=True)

    rt = Runtime(
        sequence,
        [
            vector_ty,
            vector_ty,
            inputs.prod(tile=Tile(0, 0)),
            outputs.cons(tile=Tile(0, 0)),
        ],
    )

    module = Program(NPU2(), rt, workers=[worker]).resolve_program()
    if not module.operation.verify():
        raise RuntimeError("generated module failed verification")
    print(module)


if __name__ == "__main__":
    build_design()
