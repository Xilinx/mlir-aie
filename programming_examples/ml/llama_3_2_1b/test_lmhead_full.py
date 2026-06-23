"""M3a test: fused lm_head GEMM -> 2-memtile logits -> 3-pass sampler, bit-exact.

Builds a final hidden state (int8[D]+scale), runs the device path (final_norm ->
lm_head GEMM over V=128256 -> 2-memtile logits -> streamed full sampler), and
asserts the device token == numpy oracle for greedy / full / top-k.

Oracle: device-faithful final_norm (numpy_rmsnorm_int8_dyn) + int32-accumulate
lm_head dequant -> fp32 logits -> sample_streamed_reference.

Run:
  make lmhead_full
  python test_lmhead_full.py -x build/final_lmhead_full.xclbin \\
      -i build/insts_lmhead_full.bin -k MLIR_AIE
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_lmhead_full import D, VOCAB, N_TILE, WLM_TOTAL
from numpy_llama_ref import _load_qw, _load_bf16, VOCAB_SIZE, EMB_DIM
from numpy_layer_mh import requant
from test_rmsnorm_int8_dyn import numpy_rmsnorm_int8_dyn
from test_ffn_half import pack_perchan_slots
from numpy_sample_streamed import sample_streamed_reference
from numpy_sample import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut

DATA_DIR = Path(__file__).parent / "data"


def oracle_logits(hidden_i8, hidden_scale, gamma, embed_i8, embed_sc):
    normed_i8, norm_scale = numpy_rmsnorm_int8_dyn(hidden_i8, gamma, hidden_scale)
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)
    return acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(np.float32)


def pack_lmw(embed_i8, embed_sc):
    bias = np.zeros(VOCAB, np.int32)
    blob = pack_perchan_slots(
        embed_i8, embed_sc.astype(np.float32), bias, N_TILE, prefix_bytes=b"\x00" * 64
    )
    assert blob.size == WLM_TOTAL, (blob.size, WLM_TOTAL)
    return blob


def run_dev(npu, xin, gamma, lmw, opts):
    x_t = iron.tensor(xin, dtype=np.int8)
    g_t = iron.tensor(gamma, dtype=gamma.dtype)
    w_t = iron.tensor(lmw, dtype=np.int8)
    o_t = iron.zeros([1], dtype=np.int32)
    rc = DefaultNPURuntime.run_test(
        npu, [x_t, g_t, w_t, o_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        return None
    o_t.to("cpu")
    return int(o_t.numpy()[0])


def main():
    opts = test_utils.create_default_argparser().parse_args()
    assert VOCAB == VOCAB_SIZE and D == EMB_DIM

    print(f"loading embed/final_norm from {DATA_DIR} ...", flush=True)
    embed_i8, embed_sc = _load_qw(DATA_DIR, "embed", (VOCAB, D))
    gamma = _load_bf16(DATA_DIR, "final_norm", (D,))
    print("packing lm_head weights (262 MB) ...", flush=True)
    lmw = pack_lmw(embed_i8, embed_sc)
    lut = exp_lut(EXP_QUANT_SCALE).astype(np.float32)

    npu = test_utils.create_npu_kernel(opts).npu_kernel

    rng = np.random.default_rng(0)
    hidden_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    h_scale = np.float32(max(float(np.abs(hidden_fp).max()), 1e-12) / 127.0)
    hidden_i8 = requant(hidden_fp, np.float32(1.0) / h_scale)
    xin = np.zeros(D + 8, dtype=np.int8)
    xin[:D] = hidden_i8
    xin[D : D + 4] = np.frombuffer(np.float32(h_scale).tobytes(), dtype=np.int8)

    logits = oracle_logits(hidden_i8, h_scale, gamma, embed_i8, embed_sc)

    # Params are BAKED into the xclbin (worker-local Buffer initial_value) via
    # the same env vars the design reads. Test compares against that one config.
    import os

    temp = float(os.environ.get("LLAMA_SAMPLE_TEMP", "0.0"))
    topk = int(os.environ.get("LLAMA_SAMPLE_TOPK", "0"))
    seed = int(os.environ.get("LLAMA_SAMPLE_SEED", "0"))

    print(f"lmhead_full  V={VOCAB}  temp={temp} topk={topk} seed={seed}", flush=True)
    a = run_dev(npu, xin, gamma, lmw, opts)
    e = sample_streamed_reference(logits, temp, topk, seed, lut)
    ok = a == e
    tag = "greedy" if temp <= 0 else (f"topk{topk}" if topk > 0 else "full")
    print(f"[{tag}]  dev={a}  ref={e}  {'PASS' if ok else 'FAIL'}")
    print(f"\nlmhead_full: {'BIT-EXACT PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
