"""Driver for the dma_compression IRON design.

python3 test_dma_compression.py                  # full sweep + roundtrip
python3 test_dma_compression.py both             # single config
python3 test_dma_compression.py -v               # extra hex dump
python3 test_dma_compression.py --skip both,core_both
"""

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import aie.iron as iron

from dma_compression import dma_compression, CONFIGS, RATIOED_N


def _detect_arch() -> str:
    from aie.utils import _get_default_npu_runtime

    rt = _get_default_npu_runtime()
    if rt is None or not getattr(rt, "npu_str", None):
        raise RuntimeError("could not determine NPU arch from XRT runtime")
    return rt.npu_str


def _isolate_for_next_config():
    # Drop stale xclbin + hw_context so the next dispatch's registration
    # doesn't fail with ENODEV from DRM_IOCTL_AMDXDNA_CREATE_HWCTX.
    cache = Path.home() / ".npu" / "cache"
    if cache.exists():
        for entry in cache.iterdir():
            shutil.rmtree(entry, ignore_errors=True)
    try:
        from aie.utils.jit import _compiled_kernels

        _compiled_kernels.clear()
    except Exception:
        pass
    try:
        from aie.utils import _get_default_npu_runtime

        rt = _get_default_npu_runtime()
        if rt is not None:
            rt.cleanup()
    except Exception:
        pass


N = 4096
SENTINEL = np.uint32(0xDEADBEEF)

# Empirically derived. If a future arch diverges, split into a per-arch dict.
GOLDEN_SHA = {
    "cmp": "f947e931d2e3c530be240f4e622d1c3757b2e1d5e23a05be5d21768332420bc5",
    "dcmp": "1215bcd067bc6097eecdb81fb8ebf45211cb1e9ea00c95674eea4e8d476a7152",
}


def _payload_kind(config: str):
    if config.endswith("dcmp_only"):
        return "dcmp", N
    if config.endswith("cmp_only") or config.endswith("both"):
        return "cmp", RATIOED_N
    return None, None


def _hex_block(arr: np.ndarray, n_int32: int) -> str:
    return arr[:n_int32].tobytes().hex(" ", -4)


# `regdump` has its own assertion path; entry is None to skip the buckets check.
EXPECTED = {
    "base": dict(matches=4096, mismatches=0, untouched=0),
    "cmp_only": dict(matches=1024, mismatches=1920, untouched=1152),
    "dcmp_only": dict(matches=1024, mismatches=3072, untouched=0),
    "both": dict(matches=2944, mismatches=0, untouched=1152),
    "core_cmp_only": dict(matches=1024, mismatches=1920, untouched=1152),
    "core_dcmp_only": dict(matches=1024, mismatches=3072, untouched=0),
    "core_both": dict(matches=2944, mismatches=0, untouched=1152),
    "memtile_base": dict(matches=4096, mismatches=0, untouched=0),
    "memtile_cmp_only": dict(matches=1024, mismatches=1920, untouched=1152),
    "memtile_dcmp_only": dict(matches=1024, mismatches=3072, untouched=0),
    "memtile_both": dict(matches=2944, mismatches=0, untouched=1152),
    "lossless_roundtrip": dict(matches=4096, mismatches=0, untouched=0),
    "multi_base": dict(matches=4096, mismatches=0, untouched=0),
    "multi_cmp_only": dict(matches=1024, mismatches=1920, untouched=1152),
    "multi_lossless_roundtrip": dict(matches=4096, mismatches=0, untouched=0),
    "regdump": None,
}

# `regdump` output buffer layout (see kernel.cc::dump_compress_regs).
COMPRESS_BIT = 0x80000000
REGDUMP_WRITE_BACK = [("BD0_1", 6), ("BD1_1", 7), ("BD2_1", 8), ("BD3_1", 9)]
REGDUMP_INITIAL_INFO = [
    ("BD0_1", 0),
    ("BD1_1", 1),
    ("BD2_1", 2),
    ("BD3_1", 3),
    ("S2MM_0_CTRL", 4),
    ("MM2S_0_CTRL", 5),
]


def run_one(config: str, verbose: bool = False, input_np=None) -> bool:
    if input_np is None:
        input_np = np.arange(N, dtype=np.uint32)
    sentinel_np = np.full(N, SENTINEL, dtype=np.uint32)

    in_tensor = iron.tensor(input_np.copy(), dtype=np.uint32, device="npu")
    out_tensor = iron.tensor(sentinel_np.copy(), dtype=np.uint32, device="npu")

    t0 = time.perf_counter()
    iron.jit(dma_compression)(in_tensor, out_tensor, config=config)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    out = out_tensor.numpy()

    if config == "regdump":
        all_ok = True
        print(
            f"[{config:>23}] core-side write_tm + read_tm self-test "
            f"({elapsed_ms} ms)"
        )
        print("  initial register values (informational; not asserted):")
        for label, idx in REGDUMP_INITIAL_INFO:
            regval = int(out[idx])
            print(f"    {label:>15}  = 0x{regval:08x}")
        print("  post-write_tm(COMPRESS_BIT) readback (each must == 0x80000000):")
        for label, idx in REGDUMP_WRITE_BACK:
            regval = int(out[idx])
            ok_bit = regval == COMPRESS_BIT
            tag = "ok " if ok_bit else "FAIL"
            print(f"    [{tag}] {label:>10}  = 0x{regval:08x}")
            all_ok &= ok_bit
        tag = "PASS" if all_ok else "FAIL"
        print(f"[{config:>23}] {tag}")
        return all_ok
    matches = int(np.count_nonzero(out == input_np))
    untouched = int(np.count_nonzero(out == SENTINEL))
    mismatches = N - matches - untouched

    exp = EXPECTED[config]
    ok = (
        matches == exp["matches"]
        and mismatches == exp["mismatches"]
        and untouched == exp["untouched"]
    )

    kind, plen = _payload_kind(config)
    actual_sha = None
    golden_sha = None
    sha_status = ""
    if kind is not None:
        actual_sha = hashlib.sha256(out[:plen].tobytes()).hexdigest()
        golden_sha = GOLDEN_SHA[kind]
        if actual_sha == golden_sha:
            sha_status = "sha-ok"
        else:
            sha_status = "sha-MISMATCH"
            ok = False

    tag = "PASS" if ok else "FAIL"
    bits = [
        f"matches={matches}",
        f"mismatches={mismatches}",
        f"untouched={untouched}",
    ]
    if kind == "cmp":
        pct = RATIOED_N * 100 / N
        bits.append(f"compressed_to={pct:.1f}%(={N/RATIOED_N:.3f}x)")
    elif kind == "dcmp":
        pct = N * 100 / RATIOED_N
        bits.append(f"expanded_to={pct:.1f}%(={N/RATIOED_N:.3f}x)")
    if sha_status:
        bits.append(sha_status)
    print(f"[{config:>23}] {tag}  " + "  ".join(bits) + f"  ({elapsed_ms} ms)")

    if not ok:
        print(
            f"                          expected matches={exp['matches']} "
            f"mismatches={exp['mismatches']} untouched={exp['untouched']}"
        )
        if sha_status == "sha-MISMATCH":
            print(f"                          expected sha256({kind}) = {golden_sha}")
            print(f"                          actual   sha256({kind}) = {actual_sha}")

    if kind == "cmp" and not config.endswith("both"):
        print(f"                  raw_in [:8]  = {_hex_block(input_np, 8)}")
        print(f"                  cmp_out[:8]  = {_hex_block(out, 8)}")
        print(
            f"                  cmp_out[{RATIOED_N - 8}:{RATIOED_N}] (tail of payload) = "
            f"{_hex_block(out[RATIOED_N - 8 : RATIOED_N], 8)}"
        )
        if actual_sha:
            print(f"                  sha256(out[:{RATIOED_N}]) = {actual_sha}")
    elif kind == "cmp" and config.endswith("both"):
        print(
            f"                  cmp_in [{RATIOED_N - 8}:{RATIOED_N}]  (tail of compressed input) = "
            f"{_hex_block(input_np[RATIOED_N - 8 : RATIOED_N], 8)}"
        )
        print(
            f"                  cmp_out[{RATIOED_N - 8}:{RATIOED_N}]  (tail after dcmp -> cmp)  = "
            f"{_hex_block(out[RATIOED_N - 8 : RATIOED_N], 8)}"
        )
        if actual_sha:
            print(f"                  sha256(out[:{RATIOED_N}]) = {actual_sha}")
    elif kind == "dcmp":
        print(f"                  cmp_in [:8]  = {_hex_block(input_np, 8)}")
        print(f"                  dcmp_out[:8] = {_hex_block(out, 8)}")
        if actual_sha:
            print(f"                  sha256(out[:{N}])    = {actual_sha}")
    elif verbose:
        print(f"                  in [:8] = {_hex_block(input_np, 8)}")
        print(f"                  out[:8] = {_hex_block(out, 8)}")
    return ok


def run_roundtrip(verbose: bool = False) -> bool:
    """Two-dispatch host-orchestrated roundtrip: cmp_only -> dcmp_only -> arange."""
    input_np = np.arange(N, dtype=np.uint32)
    sentinel_np = np.full(N, SENTINEL, dtype=np.uint32)

    in_t = iron.tensor(input_np.copy(), dtype=np.uint32, device="npu")
    compressed_t = iron.tensor(sentinel_np.copy(), dtype=np.uint32, device="npu")
    t0 = time.perf_counter()
    iron.jit(dma_compression)(in_t, compressed_t, config="cmp_only")
    t_cmp_ms = int((time.perf_counter() - t0) * 1000)

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
        f"[{'roundtrip':>23}] {tag}  recovered_matches_input={ok}  "
        f"(cmp {t_cmp_ms} ms, dcmp {t_dcmp_ms} ms)"
    )
    print(f"                  raw_in    [:8]              = {_hex_block(input_np, 8)}")
    print(
        f"                  compressed[{RATIOED_N - 8}:{RATIOED_N}] (tail) = "
        f"{_hex_block(compressed_bytes[RATIOED_N - 8 : RATIOED_N], 8)}"
    )
    print(f"                  recovered [:8]              = {_hex_block(recovered, 8)}")
    if not ok:
        mismatches = int(np.count_nonzero(recovered != input_np))
        print(f"                  {mismatches} of {N} elements differ from input")
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
    p.add_argument(
        "--skip",
        default="",
        help="Comma-separated configs to skip (also accepts 'roundtrip')",
    )
    args = p.parse_args()

    skip = {s for s in args.skip.split(",") if s}
    print(f"detected arch: {_detect_arch()}")

    if args.config == "roundtrip":
        return 0 if run_roundtrip(verbose=args.verbose) else 1

    configs = [args.config] if args.config else list(CONFIGS)

    # `*both` configs feed real compressed bytes (not garbage) into the
    # decompressor; pre-compute them once via cmp_only.
    needs_compressed_input = any(c.endswith("both") and c not in skip for c in configs)
    compressed_input_np = None
    if needs_compressed_input:
        _isolate_for_next_config()
        print("[ pre-compute cmp_only ] capturing compressed-arange for *both configs")
        arange = np.arange(N, dtype=np.uint32)
        in_t = iron.tensor(arange.copy(), dtype=np.uint32, device="npu")
        out_t = iron.tensor(
            np.full(N, SENTINEL, dtype=np.uint32), dtype=np.uint32, device="npu"
        )
        iron.jit(dma_compression)(in_t, out_t, config="cmp_only")
        compressed_input_np = np.zeros(N, dtype=np.uint32)
        compressed_input_np[:RATIOED_N] = out_t.numpy()[:RATIOED_N]

    all_ok = True
    for cfg in configs:
        if cfg in skip:
            print(f"[{cfg:>23}] SKIP")
            continue
        _isolate_for_next_config()
        cfg_input = compressed_input_np if cfg.endswith("both") else None
        all_ok &= run_one(cfg, verbose=args.verbose, input_np=cfg_input)

    if not args.config and "roundtrip" not in skip:
        _isolate_for_next_config()
        all_ok &= run_roundtrip(verbose=args.verbose)

    print()
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
