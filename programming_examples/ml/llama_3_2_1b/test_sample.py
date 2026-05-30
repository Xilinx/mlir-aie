"""Phase 2 sample test: greedy argmax bit-exact vs numpy."""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-V", type=int, default=1024)
    opts = p.parse_args()

    V = opts.V
    rng = np.random.default_rng(0)
    logits = rng.integers(-128, 128, size=V, dtype=np.int8)

    l_t = iron.tensor(logits, dtype=np.int8)
    t_t = iron.zeros([1], dtype=np.int32)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [l_t, t_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    t_t.to("cpu")
    actual = int(t_t.numpy()[0])
    expected = int(np.argmax(logits))

    print(f"sample NPU vs numpy: V={V}  NPU_token={actual}  expected={expected}  "
          f"logit_at_NPU={int(logits[actual])}  logit_at_expected={int(logits[expected])}")

    if actual == expected:
        print("BIT-EXACT PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
