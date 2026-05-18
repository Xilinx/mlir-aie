"""Python driver for dma_compression on Phoenix npu1.

Runs the IRON design under iron.jit for each compression config and
validates output. All configs complete in well under one second.

Run with:
    python3 test_dma_compression.py            # all configs + roundtrip
    python3 test_dma_compression.py both       # just one config
    python3 test_dma_compression.py -v         # show first 8 elems of each buffer

The `roundtrip` test runs `cmp_only` (raw -> compressed) then `dcmp_only`
(compressed -> raw) sequentially and asserts the recovered bytes equal the
original input. This is the true lossless test on non-trivial data;
individual `cmp_only` / `dcmp_only` / `both` configs only show partial
matches because the output stream is the compressed form, not the input.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Multi-config NPU test isolation: clearing the iron.jit xclbin cache + the
# in-memory _compiled_kernels cache between every config avoids the case
# where a previous config's hw_context is still bound on the device when the
# next config tries to register a new xclbin (manifests as err=19 ENODEV
# from DRM_IOCTL_AMDXDNA_CREATE_HWCTX).
def _isolate_for_next_config():
    cache = Path.home() / ".npu" / "cache"
    if cache.exists():
        for entry in cache.iterdir():
            shutil.rmtree(entry, ignore_errors=True)
    try:
        from aie.utils.jit import _compiled_kernels
        _compiled_kernels.clear()
    except Exception:
        pass
    # Release any cached XRT hw_contexts so the next config can register a
    # fresh xclbin without hitting err=19 ENODEV from
    # DRM_IOCTL_AMDXDNA_CREATE_HWCTX.
    try:
        from aie.utils import _get_default_npu_runtime
        rt = _get_default_npu_runtime()
        if rt is not None:
            rt.cleanup()
    except Exception:
        pass

import aie.iron as iron

from dma_compression import dma_compression, CONFIGS, RATIOED_N


N = 4096
LINE_SIZE = 1024
SENTINEL = np.uint32(0xDEADBEEF)

# Per-config expected match/mismatch/untouched buckets (Phoenix npu1, arange
# input). All configs are deterministic and complete in milliseconds. The
# first BD (1024 elems) passes through unmodified across configs that engage
# (de)compression, after a state-machine warm-up. The asymmetric configs
# (cmp_only, dcmp_only) use ratio-sized shim BDs so neither side hangs.
EXPECTED = {
    "base":      dict(matches=4096, mismatches=0,    untouched=0),
    "cmp_only":  dict(matches=1024, mismatches=1920, untouched=1152),
    "dcmp_only": dict(matches=1024, mismatches=3072, untouched=0),
    # `both` with arange input + ratioed out_n: shim S2MM gets compressed
    # bytes from the consumer side; with both directions compressed in the
    # link, the first BD's worth of bytes pass through close to identity (a
    # state-machine warm-up artifact, matches=1043 ~ first 1024 elems), then
    # the remainder are compressor output (garbage when interpreted as int32)
    # and the tail of the output buffer is never written.
    "both":      dict(matches=1043, mismatches=1901, untouched=1152),
    # Core-side configs (peano `write_tm` in kernel.cc, host enables the
    # Core_Processor_Bus first). Measured outputs are bit-identical to their
    # host-side counterparts (cmp_only / both), confirming the core-side
    # path drives the same DMA register state as `npu_maskwrite32`.
    "core_cmp_only":  dict(matches=1024, mismatches=1920, untouched=1152),
    "core_dcmp_only": dict(matches=1024, mismatches=3072, untouched=0),
    "core_both":      dict(matches=1043, mismatches=1901, untouched=1152),
    # Memtile configs route through memtile(0,1) instead of compute(0,2);
    # different register addresses (MT_BD4_BASE, MT_S2MM/MM2S_0_CTRL).
    # Measured outputs are bit-identical to compute-tile counterparts —
    # the memtile compression hardware presents the same observable
    # behaviour as the compute-tile one.
    "memtile_base":      dict(matches=4096, mismatches=0,    untouched=0),
    "memtile_cmp_only":  dict(matches=1024, mismatches=1920, untouched=1152),
    "memtile_dcmp_only": dict(matches=1024, mismatches=3072, untouched=0),
    "memtile_both":      dict(matches=1043, mismatches=1901, untouched=1152),
    # Single-dispatch lossless roundtrip: shim -> CT -> memtile -> shim,
    # compress on CT MM2S, decompress on memtile S2MM. Both shim ends see
    # raw int32s; expected matches=4096 if the chained compress->decompress
    # is truly lossless.
    "lossless_roundtrip":       dict(matches=4096, mismatches=0, untouched=0),
    "multi_base":               dict(matches=4096, mismatches=0, untouched=0),
    # multi_cmp_only proves DMA compression engages on the inter-tile link:
    # compress on CT(0,2) MM2S, NO decompress on CT(0,3) S2MM, with the
    # consumer-side BD hand-sized to RATIOED_PER_LINE=736 ints/BD so the
    # DMA doesn't stall. Output pattern matches single-tile cmp_only's
    # asymmetric signature, confirming compression actually engages on the
    # CT-to-CT link.
    "multi_cmp_only":           dict(matches=1024, mismatches=1920, untouched=1152),
    "multi_lossless_roundtrip": dict(matches=4096, mismatches=0,    untouched=0),
}


def run_one(config: str, verbose: bool = False) -> bool:
    input_np = np.arange(N, dtype=np.uint32)
    sentinel_np = np.full(N, SENTINEL, dtype=np.uint32)

    in_tensor = iron.tensor(input_np.copy(), dtype=np.uint32, device="npu")
    out_tensor = iron.tensor(sentinel_np.copy(), dtype=np.uint32, device="npu")

    t0 = time.perf_counter()
    iron.jit(dma_compression)(in_tensor, out_tensor, config=config)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    out = out_tensor.numpy()
    matches = int(np.count_nonzero(out == input_np))
    untouched = int(np.count_nonzero(out == SENTINEL))
    mismatches = N - matches - untouched

    exp = EXPECTED[config]
    if exp is None:
        tag = "MEASURE"
        ok = True
    else:
        ok = (
            matches == exp["matches"]
            and mismatches == exp["mismatches"]
            and untouched == exp["untouched"]
        )
        tag = "PASS" if ok else "FAIL"
    print(
        f"[{config:>15}] {tag}  matches={matches}  mismatches={mismatches}  "
        f"untouched={untouched}  ({elapsed_ms} ms)"
    )
    if exp is not None and not ok:
        print(
            f"                  expected matches={exp['matches']} "
            f"mismatches={exp['mismatches']} untouched={exp['untouched']}"
        )
    if verbose:
        print(f"           in[:8]={input_np[:8].tolist()}")
        print(f"           out[:8]={out[:8].tolist()}")
        print(f"           out[1020:1028]={out[1020:1028].tolist()}")
    return ok


def run_roundtrip(verbose: bool = False) -> bool:
    """Sequential cmp_only -> dcmp_only on arange input. Asserts recovered == input."""
    input_np = np.arange(N, dtype=np.uint32)
    sentinel_np = np.full(N, SENTINEL, dtype=np.uint32)

    # Stage 1: compress raw -> compressed bytes (first RATIOED_N ints of output)
    in_t = iron.tensor(input_np.copy(), dtype=np.uint32, device="npu")
    compressed_t = iron.tensor(sentinel_np.copy(), dtype=np.uint32, device="npu")
    t0 = time.perf_counter()
    iron.jit(dma_compression)(in_t, compressed_t, config="cmp_only")
    t_cmp_ms = int((time.perf_counter() - t0) * 1000)

    # Stage 2: feed the compressed bytes back in as input, decompress to raw
    compressed_bytes = compressed_t.numpy().copy()
    cmp_in_t = iron.tensor(compressed_bytes, dtype=np.uint32, device="npu")
    recovered_t = iron.tensor(sentinel_np.copy(), dtype=np.uint32, device="npu")
    t0 = time.perf_counter()
    iron.jit(dma_compression)(cmp_in_t, recovered_t, config="dcmp_only")
    t_dcmp_ms = int((time.perf_counter() - t0) * 1000)

    recovered = recovered_t.numpy()
    ok = bool(np.array_equal(recovered, input_np))
    tag = "PASS" if ok else "FAIL"
    print(
        f"[{'roundtrip':>15}] {tag}  recovered_matches_input={ok}  "
        f"(cmp {t_cmp_ms} ms, dcmp {t_dcmp_ms} ms)"
    )
    if verbose or not ok:
        print(f"           in[:8]={input_np[:8].tolist()}")
        print(f"           compressed[:8]={compressed_bytes[:8].tolist()}")
        print(f"           recovered[:8]={recovered[:8].tolist()}")
        if not ok:
            mismatches = int(np.count_nonzero(recovered != input_np))
            print(f"           {mismatches} of {N} elements differ from input")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "config",
        nargs="?",
        default=None,
        choices=list(CONFIGS) + ["roundtrip"],
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.config == "roundtrip":
        return 0 if run_roundtrip(verbose=args.verbose) else 1

    configs = [args.config] if args.config else list(CONFIGS)
    all_ok = True
    for cfg in configs:
        _isolate_for_next_config()
        all_ok &= run_one(cfg, verbose=args.verbose)

    if not args.config:
        _isolate_for_next_config()
        all_ok &= run_roundtrip(verbose=args.verbose)

    print()
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
