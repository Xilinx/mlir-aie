"""Host-side runner for the DMA deinterleave test."""

from __future__ import annotations

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

IN_W = 16
IN_C = 8
N_ROWS = 8
ROW_BYTES = IN_W * IN_C  # 128
TOTAL_BYTES = N_ROWS * ROW_BYTES


def expected_deinterleave(in_bytes: np.ndarray) -> np.ndarray:
    """CPU-side reference: per row, even pixels in first half, odd in second."""
    out = np.zeros_like(in_bytes)
    in_rows = in_bytes.reshape(N_ROWS, IN_W, IN_C)
    out_rows = out.reshape(N_ROWS, IN_W, IN_C)
    half = IN_W // 2
    for r in range(N_ROWS):
        out_rows[r, :half, :] = in_rows[r, 0::2, :]  # even pixels -> first half
        out_rows[r, half:, :] = in_rows[r, 1::2, :]  # odd  pixels -> second half
    return out


def main():
    p = test_utils.create_default_argparser()
    opts = p.parse_args()

    # Unique byte per (row, pixel, channel) so we can spot any mis-routing.
    in_bytes = np.arange(TOTAL_BYTES, dtype=np.int32).astype(np.int8)
    expected = expected_deinterleave(in_bytes)

    in_tensor = iron.tensor(in_bytes, dtype=np.int8)
    out_tensor = iron.zeros([TOTAL_BYTES], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in_tensor, out_tensor],
        {1: expected},
        verify=False,
        verbosity=npu_opts.verbosity,
    )

    npu_out = out_tensor.numpy().astype(np.int8)

    mismatches = np.where(npu_out != expected)[0]
    print(f"Total bytes: {TOTAL_BYTES}, mismatches: {len(mismatches)}")
    if len(mismatches) == 0:
        print("BIT-EXACT: dims_to_stream deinterleave layout matches CPU model")
        return 0

    print(f"First 20 mismatches (idx: npu vs expected, row.pixel.byte):")
    for i in mismatches[:20]:
        r = i // ROW_BYTES
        rem = i % ROW_BYTES
        p_idx = rem // IN_C
        b = rem % IN_C
        print(f"  [{i:4d}] row={r} half_pix={p_idx} byte={b}: "
              f"npu={int(npu_out[i]):4d} expected={int(expected[i]):4d} "
              f"in_byte={int(in_bytes[i]):4d}")
    print(f"\nPer-row mismatch count:")
    for r in range(N_ROWS):
        row_start = r * ROW_BYTES
        row_end = row_start + ROW_BYTES
        row_mm = ((npu_out[row_start:row_end] != expected[row_start:row_end])
                  .sum())
        print(f"  row {r}: {row_mm} / {ROW_BYTES}")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
