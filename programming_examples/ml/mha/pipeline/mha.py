#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import sys
import argparse

from pathlib import Path

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, GlobalBuffer, WorkerRuntimeBarrier, LocalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D

base_dir = Path(__file__).parent

dtype_map = {
    "bf16": bfloat16,
    "f32": np.float32,
}

microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
    },
}

def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--heads", type=int, default=1)
    argparser.add_argument("--S_q", type=int, default=256)
    argparser.add_argument("--S_kv", type=int, default=256)
    argparser.add_argument("-d", type=int, default=64)
    argparser.add_argument("--B_q", type=int, default=64)
    argparser.add_argument("--B_kv", type=int, default=64)
    argparser.add_argument("--num_KV_heads", type=int, default=2, help="Number of heads for Key-Value pairs")
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument("--output_file_path", type=str, default = base_dir / "build" / f"my_mha.mlir", help="Output file path for the generated MLIR module")
    argparser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = argparser.parse_args()
    
    maybe_module = batched_matmul_single_core(
        args.heads,
        args.S_q,
        args.S_kv,
        args.d,
        args.B_q,
        args.B_kv,
        args.num_KV_heads,
        args.emulate_bf16_mmul_with_bfp16,
        args.trace_size,
        args.verbose
    )
    
    output_file_path = Path(args.output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file_path, "w") as f:
        f.write(str(maybe_module))

    if args.verbose:
        print(f"MLIR module written to {output_file_path}")

def batched_matmul_single_core(
    heads: int,
    S_q: int,
    S_kv: int,
    d: int,
    B_q: int,
    B_kv: int,
    num_KV_heads: int,
    emulate_bf16_mmul_with_bfp16: bool,
    trace_size: int = 0,
    verbose: bool = False,
):

    # When false toogle sclar GEMM (for QK)
    vectorized = True
    debug_QK = False
    
    enable_tracing = True if trace_size > 0 else False

    dtype_str = "bf16"
    dev = "npu2"

    # VJUNG: When the number of KV head is 0 we do a regular MHA, otherwise we do GQA.
    if num_KV_heads == 0:
        num_KV_heads = heads
        
    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_str]
    r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]

    if verbose:
        print(f"Device: {dev}")
        print(f"Number of heads: {heads}")
        print(f"MHA Dimensions: S_q={S_q}, S_kv={S_kv}, d={d}, B_q={B_q}, B_kv={B_kv}")
        print(f"Data type: {dtype_str}")
        print(f"Microkernel MAC dimensions: r={r}, s={s}, t={t}")
        print(f"Vectorized: {vectorized}")
        print(f"Enable tracing: {enable_tracing}")
        
    assert num_KV_heads > 0, "Number of KV heads must be greater than 0"
    assert heads > 0, "Number of heads must be greater than 0"
    assert num_KV_heads <= heads, "Number of KV heads must be less than or equal to number of heads"
    assert heads % num_KV_heads == 0, f"Number of KV heads ({num_KV_heads}) must be divisible by number of heads ({heads})"
    assert S_q % B_q == 0, f"S_q must be divisible by B_q ({S_q} % {B_q} != 0)"
    assert S_kv % B_kv == 0, f"S_kv must be divisible by B_kv ({S_kv} % {B_kv} != 0)"
    
    assert B_q % r == 0, f"B_q must be divisible by r ({B_q} % {r} != 0)"
    assert B_kv % t == 0, f"B_kv must be divisible by t ({B_kv} % {t} != 0)"
    assert d % s == 0, f"d must be divisible by s ({d} % {s} != 0)"

    dtype = dtype_map[dtype_str]

    num_q_blocks = S_q // B_q
    num_kv_blocks = S_kv // B_kv
    
    inv_scale = 1 / np.sqrt(d)

    # Tensors living in DRAM
    Q_ty = np.ndarray[(heads * S_q * d,), np.dtype[dtype]]
    KV_ty = np.ndarray[(num_KV_heads * S_kv * d,), np.dtype[dtype]]

    # Tensors living on the AIE-array
    q_ty = np.ndarray[(B_q, d), np.dtype[dtype]]
    k_ty = np.ndarray[(d, B_kv), np.dtype[dtype]]
    qk_ty = np.ndarray[(B_q, B_kv), np.dtype[dtype]]
    s_ty = np.ndarray[(4*B_q,), np.dtype[np.float32]]
    
    # AIE kernel declarations
    func_type = "" if vectorized else "_scalar"
    bin_name = "kernels.a"
    
    
    zero_kernel = Kernel(
        f"zero_{dtype_str}", bin_name, [qk_ty]
    )
    
    memcopy_kernel_scale = Kernel(
        f"passThroughLineScalar", bin_name, [s_ty, s_ty, np.int32]
    )
    
    memcopy_kernel_debug = Kernel(
        f"passThroughLineScalarDebug", bin_name, [qk_ty, qk_ty, np.int32]
    )
    
    scale_buffer_init_kernel = Kernel(
        "init_scale_buffer",
        bin_name,
        [s_ty , np.int32]
    )
    
    partial_softmax_kernel = Kernel(
        "partial_softmax",
        bin_name,
        [qk_ty, qk_ty, s_ty, np.float32, np.int32, np.int32],
    )
    
    matmul_QK = Kernel(
        f"matmul_bf16_bf16_wrapper{func_type}",
        bin_name,
        [q_ty, k_ty, qk_ty],
    )
    
    matmul_PV = Kernel(
        "matmul_PV",
        bin_name,
        [qk_ty, k_ty, qk_ty, s_ty, np.int32, np.int32],
    )
    
    rescale_O = Kernel(
        "rescale_O",
        bin_name,
        [qk_ty, s_ty, np.int32],
    )

    # AIE-array data movement with object fifos
    inQ = ObjectFifo(q_ty, name="inQ")
    q_dims = None
    if vectorized:
        q_dims = [(B_q // r, r * d), (d // s, s), (r, d), (s, 1)]
    memQ = inQ.cons().forward(name="memQ",  dims_to_stream=q_dims) # Forward DRAM -> Mem tile -> L1

    # K is stored in column-major order
    inK = ObjectFifo(k_ty, name="inK")
    k_dims = None
    if vectorized:
        k_dims = [(B_kv // t, t * d), (d // s, s), (t, d), (s, 1)]
    memK = inK.cons().forward(name="memK", dims_to_stream=k_dims)
        
    inV = ObjectFifo(k_ty, name="inV")
    v_dims = None
    if vectorized:
        v_dims = [(B_kv // s, s * B_kv), (B_kv // t, t), (s, B_kv), (t, 1)]
    memV = inV.cons().forward(name="memV", dims_to_stream=v_dims, placement=Tile(col=1, row=1))

    a_dims = None
    if vectorized:
        a_dims = [(B_q // r, r * B_kv), (r, t), (B_kv // t, r * t), (t, 1)]
    memA = ObjectFifo(qk_ty, name="memA")
    outA = memA.cons().forward(name="outA", dims_to_stream=a_dims)
    
    memP = ObjectFifo(qk_ty, name="memP")
    outP = memP.cons().forward(name="outP", dims_to_stream=q_dims, placement=Tile(col=1, row=1))
    
    # Scale buffer for partial softmax
    scaleOF = ObjectFifo(s_ty, name="scaleOF")
    
    memO = ObjectFifo(qk_ty, name="memO")
    o_dims = None
    if vectorized:
        o_dims = [(B_q // r, r * B_kv), (r, t), (B_kv // t, r * t), (t, 1)]
    outO = memO.cons().forward(name="outO", dims_to_stream=o_dims, placement=Tile(col=1, row=1))


    def batched_matmul_qk(of_q, of_k, of_a_out, zero, matmul_QK):
        
        elem_in_q = of_q.acquire(1)
        
        for _ in range_(num_kv_blocks):
            elem_in_k = of_k.acquire(1)
            elem_a_out = of_a_out.acquire(1)
            
            zero(elem_a_out)
            matmul_QK(elem_in_q, elem_in_k, elem_a_out)
            
            of_k.release(1)
            of_a_out.release(1)
        
        of_q.release(1)

    def softmax(of_in_a, of_out_p, of_out_scale, partial_softmax, init_scale_buffer, memcopy_kernel_scale, memcopy_kernel_debug):
        
        scale_buffer = LocalBuffer(initial_value=np.zeros(shape=(4*B_q,), dtype=np.float32))
        
        for _ in range_(sys.maxsize):
            
            for _ in range_(num_q_blocks):
            
                init_scale_buffer(scale_buffer, B_q)
            
                for _ in range_(num_kv_blocks):
                    
                    elt_of_out_p = of_out_p.acquire(1)
                    elt_of_in_a = of_in_a.acquire(1)
                    elt_of_out_scale = of_out_scale.acquire(1)
                
                    if debug_QK:
                        memcopy_kernel_debug(elt_of_in_a, elt_of_out_p, B_q * d) # Debug
                    else:
                        partial_softmax(elt_of_in_a, elt_of_out_p, scale_buffer, inv_scale, B_q, B_q)
                        memcopy_kernel_scale(scale_buffer, elt_of_out_scale, 4*B_q)
                
                    of_in_a.release(1)
                    of_out_p.release(1)
                    of_out_scale.release(1)
    
    def batched_matmul_pv(of_p, of_v, of_scale, of_o_out, zero, matmul_PV, rescale_O, memcopy_kernel):
        
        for _ in range_(num_q_blocks):
            
            elem_o_out = of_o_out.acquire(1)
        
            zero(elem_o_out)
            
            ### First iteration, don't rescale O_{i-1}
            elem_in_p = of_p.acquire(1)
            elem_in_v = of_v.acquire(1)
            elt_of_out_scale = of_scale.acquire(1)
            
            if debug_QK:
                memcopy_kernel(elem_in_p, elem_o_out, B_q * d)
            else:
                matmul_PV(elem_in_p, elem_in_v, elem_o_out, elt_of_out_scale, B_q, 0)
            
            of_p.release(1)
            of_v.release(1)
            of_scale.release(1)
            ###
            
            if num_kv_blocks > 2:
                for _ in range_(num_kv_blocks - 2):
                    elem_in_p = of_p.acquire(1)
                    elem_in_v = of_v.acquire(1)
                    elt_of_out_scale = of_scale.acquire(1)
                    
                    matmul_PV(elem_in_p, elem_in_v, elem_o_out, elt_of_out_scale, B_q, 1)
                    
                    of_p.release(1)
                    of_v.release(1)
                    of_scale.release(1)
            
            
            ### Last iteration, final rescaling
            if num_kv_blocks > 1:
                elem_in_p = of_p.acquire(1)
                elem_in_v = of_v.acquire(1)
                elt_of_out_scale = of_scale.acquire(1)
                
                matmul_PV(elem_in_p, elem_in_v, elem_o_out, elt_of_out_scale, B_q, 1)
                if not debug_QK:
                    rescale_O(elem_o_out, elt_of_out_scale, B_q)
                
                of_p.release(1)
                of_v.release(1)
                of_scale.release(1)
            else:
                rescale_O(elem_o_out, elt_of_out_scale, B_q)
            ###
            
            of_o_out.release(1)

    # Create worker from task
    matmul_worker = Worker(
        batched_matmul_qk,
        fn_args = [
            memQ.cons(), 
            memK.cons(),
            memA.prod(),
            zero_kernel,
            matmul_QK,
        ], 
        stack_size=0xD00,
        placement=Tile(col=0, row=2)
    )
    
    softmax_worker = Worker(
        softmax,
        fn_args = [
            outA.cons(),
            memP.prod(),
            scaleOF.prod(),
            partial_softmax_kernel,
            scale_buffer_init_kernel,
            memcopy_kernel_scale,
            memcopy_kernel_debug,
        ],
        stack_size=0xD00,
        placement=Tile(col=0, row=3),
        while_true=False
    )
    
    matmul_av_worker = Worker(
        batched_matmul_pv,
        fn_args = [
            outP.cons(),
            memV.cons(),
            scaleOF.cons(),
            memO.prod(),
            zero_kernel,
            matmul_PV,
            rescale_O,
            memcopy_kernel_debug,
        ], 
        stack_size=0xD00,
        placement=Tile(col=0, row=4)
    )
    
    # Define tensor access patterns for inputs/outputs
    # A and B are tiled across M and N respectively, while C is tiled across M and N
    Q_tiles = TensorTiler2D.group_tiler((heads * S_q, d), (B_q, d), (1, 1))
    
    K_tiles = TensorTiler2D.group_tiler((heads* S_kv, d), (S_kv, d), (1, 1))
    
    V_tiles = TensorTiler2D.group_tiler((heads* S_kv, d), (S_kv, d), (1, 1))
    
    O_tiles = TensorTiler2D.group_tiler((heads * S_q, d), (B_q, d), (1, 1))
        
    def print_tap_seq_info(tap_seq, name):
        for idx, tap in enumerate(tap_seq):
            print(f"{name} tile {idx}:")
            print(f"  Offset: {tap.offset}")
            print(f"  Sizes: {tap.sizes}")
            print(f"  Strides: {tap.strides}")

    if verbose:
        print(f"DMA Transfer Configuration: DRAM <-> Mem tile")
        print_tap_seq_info(Q_tiles, "Q")
        print_tap_seq_info(K_tiles, "K")
        print_tap_seq_info(V_tiles, "V")
        print_tap_seq_info(O_tiles, "O")
            
    def fixup_tiles(tile_list):
        for tile in tile_list:
            tile._sizes = [1, 1, 512, 128] # [1, 1, 1024, 64]
            tile._strides = [0, 0, 128, 1] # [0, 0, 64, 1]
    
    # Need to use this when one head is larger than 1024x1024, should be done by the compiler
    if S_q == 1024 and S_kv == 1024:
        fixup_tiles(K_tiles)    
        fixup_tiles(V_tiles)        

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(Q_ty, KV_ty, KV_ty, Q_ty) as (Q, K, V, O):
        rt.start(matmul_worker)
        rt.start(softmax_worker)
        rt.start(matmul_av_worker)

        for head_idx in range(heads):
            
            for q_block_idx in range(num_q_blocks):
                rt.fill(inQ.prod(), Q, tap=Q_tiles[head_idx*num_q_blocks + q_block_idx], placement = Tile(col = 0, row = 0))
                
                # for kv_block_idx in range(num_kv_blocks):
                #     rt.fill(inK.prod(), K, tap=K_tiles[head_idx*num_kv_blocks + kv_block_idx], placement = Tile(col = 0, row = 0))
                #     rt.fill(inV.prod(), V, tap=V_tiles[head_idx*num_kv_blocks + kv_block_idx], placement = Tile(col = 1, row = 0))
                
                # Thow on bd containing the full K and V in the object fifo, then does it transfer cunks of inKV size at the time?
                rt.fill(inK.prod(), K, tap=K_tiles[head_idx], placement = Tile(col = 0, row = 0))
                rt.fill(inV.prod(), V, tap=V_tiles[head_idx], placement = Tile(col = 1, row = 0))
                    
                rt.drain(outO.cons(), O, tap=O_tiles[head_idx*num_q_blocks + q_block_idx], wait=True, placement = Tile(col = 0, row = 0))
                

    # Create the program from the device type and runtime
    if dev == "npu":
        dev_ty = NPU1Col1()
    else:
        dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module

if __name__ == "__main__":
    main()
