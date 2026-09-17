<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Complete Copy-Paste IRON Examples

Nine runnable designs, from the one-screen library form through hand-built topologies to a full reusable `MLIROperator`. Pick the closest one and adapt. Examples 0–7 run on the base `mlir_aie` wheel; example 8 (`MLIROperator`) needs the downstream **`amd/IRON`** repo — see the note there.

**Start with example 0.** Examples 1–7 build the topology by hand, which is what you need for
custom dataflow — but for a plain "same op over every tile" design, the library form is
shorter, faster to get right, and uses a maintained kernel. Reach for the hand-built forms
when the topology or the dtype actually demands it.

---

## 0. Element-wise add — library form, multi-core, **no C++**

The shortest correct multi-core design. Adapted from
`programming_examples/ml/eltwise/eltwise.py`.

```python
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Out, kernels
from aie.iron.algorithms import transform_parallel_binary
from aie.utils.verify import assert_pass


@iron.jit
def eltwise_add(
    a_in: In,
    b_in: In,
    c_out: Out,
    *,
    size: CompileTime[int] = 65536,
    num_channels: CompileTime[int] = 1,
):
    return transform_parallel_binary(
        kernels.add(tile_size=1024),                     # prebuilt bf16 vector add
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=1024,
        num_channels=num_channels,
        pass_size_to_kernel=False,                       # kernels.add is (in0, in1, out)
    )


if __name__ == "__main__":
    N = 65536
    rng = np.random.default_rng(0)
    a_np = rng.uniform(-1.0, 1.0, size=(N,)).astype(bfloat16)
    b_np = rng.uniform(-1.0, 1.0, size=(N,)).astype(bfloat16)

    a = iron.tensor(a_np, dtype=bfloat16, device="npu")
    b = iron.tensor(b_np, dtype=bfloat16, device="npu")
    c = iron.zeros_like(a)

    eltwise_add(a, b, c, size=N)

    expected = (a_np.astype(np.float32) + b_np.astype(np.float32)).astype(bfloat16)
    assert_pass(c.numpy(), expected, atol=0.00390625, fail_msg="mismatch")
```

Swap `kernels.add` for `kernels.mul`, or switch to `transform_parallel` +
`kernels.relu`/`kernels.softmax`/`kernels.gelu` for a unary op. Full catalog and the dtype /
tile-size limits of each kernel are in [`builtin_kernels.md`](builtin_kernels.md).

---

## 1. Passthrough — single core, int32, `iron.jit`

```python
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_

N = 1024
tile_size = 256
N_div_n = N // tile_size
tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
tile_ty   = np.ndarray[(tile_size,), np.dtype[np.int32]]

@iron.jit
def passthrough(input_tensor, output_tensor):
    of_in  = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_fn(of_in, of_out):
        for _ in range_(N_div_n):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            for i in range_(tile_size):
                eo[i] = ei[i]
            of_in.release(1)
            of_out.release(1)

    w = Worker(core_fn, [of_in.cons(), of_out.prod()])

    def sequence(A, C, in_h, out_h):
        in_h.fill(A)
        out_h.drain(C, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])

    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

inp = iron.arange(N, dtype=np.dtype(np.int32))
out = iron.zeros((N,), dtype=np.dtype(np.int32))
passthrough(inp, out)
```

---

## 2. Element-wise Add — single core, bfloat16, inline `ExternalFunction(source_string=...)`

Inline-source kernels avoid managing a separate `.cc` / `.o` file during prototyping.

```python
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_

N = 1024; tile_size = 256; N_div_n = N // tile_size
tensor_ty = np.ndarray[(N,),        np.dtype[bfloat16]]
tile_ty   = np.ndarray[(tile_size,), np.dtype[bfloat16]]

add_kernel = ExternalFunction(
    "eltwise_add_bf16",
    source_string=r'''
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" {
void eltwise_add_bf16(bfloat16 *__restrict a, bfloat16 *__restrict b,
                      bfloat16 *__restrict c, int32_t n) {
    event0();
    constexpr int V = 32;
    for (int i = 0; i < n; i += V) {
        aie::vector<bfloat16, V> va = aie::load_v<V>(a + i);
        aie::vector<bfloat16, V> vb = aie::load_v<V>(b + i);
        aie::store_v(c + i, aie::add(va, vb));
    }
    event1();
}
}
''',
    arg_types=[tile_ty, tile_ty, tile_ty, np.int32],
)

@iron.jit
def vector_add(a_in, b_in, c_out, kernel):
    of_a = ObjectFifo(tile_ty, name="a")
    of_b = ObjectFifo(tile_ty, name="b")
    of_c = ObjectFifo(tile_ty, name="c")

    def core_fn(of_a, of_b, of_c, kernel):
        for _ in range_(N_div_n):
            ea = of_a.acquire(1); eb = of_b.acquire(1); ec = of_c.acquire(1)
            kernel(ea, eb, ec, tile_size)
            of_a.release(1); of_b.release(1); of_c.release(1)

    w = Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod(), kernel])

    def sequence(A, B, C, a_h, b_h, c_h):
        a_h.fill(A)
        b_h.fill(B)
        c_h.drain(C, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, tensor_ty, of_a.prod(), of_b.prod(), of_c.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

a = iron.rand((N,), dtype=np.dtype(bfloat16))
b = iron.rand((N,), dtype=np.dtype(bfloat16))
c = iron.zeros((N,), dtype=np.dtype(bfloat16))
vector_add(a, b, c, add_kernel)
```

---

## 3. Vector × scalar multiply — single core, precompiled `Kernel(...)`

Uses a pre-built `.o`/`.a` archive (what `MLIROperator`-based designs typically use).

```python
import numpy as np
from ml_dtypes import bfloat16
from aie.iron import ObjectFifo, Worker, Runtime, Program, Kernel
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_

N = 1024; tile_size = 256; N_div_n = N // tile_size
tensor_ty = np.ndarray[(N,),        np.dtype[bfloat16]]
tile_ty   = np.ndarray[(tile_size,), np.dtype[bfloat16]]
scalar_ty = np.ndarray[(1,),        np.dtype[np.int32]]

def vector_scalar_mul_design(dev, kernel_archive):
    of_in     = ObjectFifo(tile_ty,   name="in")
    of_out    = ObjectFifo(tile_ty,   name="out")
    of_factor = ObjectFifo(scalar_ty, name="factor", depth=1)

    scale = Kernel("vector_scalar_mul_aie_scalar", kernel_archive,
                   [tile_ty, tile_ty, scalar_ty, np.int32])

    def core_fn(of_in, of_factor, of_out, scale_fn):
        ef = of_factor.acquire(1)             # weight reuse: acquire once
        for _ in range_(N_div_n):
            ei = of_in.acquire(1); eo = of_out.acquire(1)
            scale_fn(ei, eo, ef, tile_size)
            of_in.release(1); of_out.release(1)
        of_factor.release(1)

    w = Worker(core_fn, [of_in.cons(), of_factor.cons(), of_out.prod(), scale])

    def sequence(A, F, C, in_h, factor_h, out_h):
        factor_h.fill(F)
        in_h.fill(A)
        out_h.drain(C, wait=True)

    rt = Runtime(sequence, [tensor_ty, scalar_ty, tensor_ty, of_in.prod(), of_factor.prod(), of_out.cons()])
    return Program(dev, rt, workers=[w]).resolve_program()
```

---

## 4. Multi-column element-wise add (4 cols, `TensorAccessPattern` + `task_group`)

Each column gets independent ObjectFifos; `TensorAccessPattern` slices the host tensor.

```python
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction, TaskGroup
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

N = 4096; num_cols = 4; chunk = N // num_cols
tile_size = 256; iters_per_col = chunk // tile_size
tensor_ty = np.ndarray[(N,),        np.dtype[bfloat16]]
tile_ty   = np.ndarray[(tile_size,), np.dtype[bfloat16]]

add_kernel = ExternalFunction("eltwise_add_bf16", source_string=r'''
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" {
void eltwise_add_bf16(bfloat16 *__restrict a, bfloat16 *__restrict b,
                      bfloat16 *__restrict c, int32_t n) {
    event0();
    constexpr int V = 32;
    for (int i = 0; i < n; i += V) {
        aie::store_v(c + i, aie::add(aie::load_v<V>(a + i), aie::load_v<V>(b + i)));
    }
    event1();
}
}''', arg_types=[tile_ty, tile_ty, tile_ty, np.int32])

@iron.jit
def multi_core_add(a_in, b_in, c_out, kernel):
    of_as = [ObjectFifo(tile_ty, name=f"a_{i}") for i in range(num_cols)]
    of_bs = [ObjectFifo(tile_ty, name=f"b_{i}") for i in range(num_cols)]
    of_cs = [ObjectFifo(tile_ty, name=f"c_{i}") for i in range(num_cols)]

    def core_fn(of_a, of_b, of_c, kernel):
        for _ in range_(iters_per_col):
            ea = of_a.acquire(1); eb = of_b.acquire(1); ec = of_c.acquire(1)
            kernel(ea, eb, ec, tile_size)
            of_a.release(1); of_b.release(1); of_c.release(1)

    workers = [Worker(core_fn, [of_as[i].cons(), of_bs[i].cons(), of_cs[i].prod(), kernel])
               for i in range(num_cols)]

    taps = [TensorAccessPattern((1, N), chunk*i, [1,1,1,chunk], [0,0,0,1])
            for i in range(num_cols)]

    def sequence(A, B, C, *handles):
        a_hs, b_hs, c_hs = handles[:num_cols], handles[num_cols:2*num_cols], handles[2*num_cols:]
        tg = TaskGroup()               # constructed inside the sequence body; registers with this Runtime
        for i in range(num_cols):
            a_hs[i].fill(A, tap=taps[i], group=tg)
            b_hs[i].fill(B, tap=taps[i], group=tg)
        for i in range(num_cols):
            c_hs[i].drain(C, tap=taps[i], wait=True, group=tg)
        tg.finish()                    # REQUIRED — awaits wait=True tasks, then frees the group

    rt = Runtime(sequence, [
        tensor_ty, tensor_ty, tensor_ty,
        *[of_as[i].prod() for i in range(num_cols)],
        *[of_bs[i].prod() for i in range(num_cols)],
        *[of_cs[i].cons() for i in range(num_cols)],
    ])

    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()
```

---

## 5. Data-parallel with `split`/`join` at the MemTile (4 workers, single ObjectFifo at L3)

Simpler than #4 for true SIMD-on-vector workloads — the runtime sees one big input/output.

```python
import numpy as np
from ml_dtypes import bfloat16
from aie.iron import ObjectFifo, Worker, Runtime, Program, Kernel
from aie.iron.device import NPU1
from aie.iron.controlflow import range_

N = 4096; n_workers = 4; chunk = N // n_workers      # chunk = 1024
data_ty  = np.ndarray[(N,),     np.dtype[bfloat16]]
chunk_ty = np.ndarray[(chunk,), np.dtype[bfloat16]]

def data_parallel_design(dev, kernel_archive):
    of_in  = ObjectFifo(data_ty, name="in")
    of_out = ObjectFifo(data_ty, name="out")

    # The sub-object size MUST equal the offset stride, so the sub-objects tile
    # the parent exactly (4 x 1024 == 4096). See the warning below.
    offsets = [chunk * i for i in range(n_workers)]
    of_ins  = of_in.cons().split (offsets, obj_types=[chunk_ty]*n_workers,
                                  names=[f"in_{i}"  for i in range(n_workers)])
    of_outs = of_out.prod().join (offsets, obj_types=[chunk_ty]*n_workers,
                                  names=[f"out_{i}" for i in range(n_workers)])

    kernel = Kernel("relu_bf16", kernel_archive, [chunk_ty, chunk_ty, np.int32])

    def core_fn(of_in, of_out, kernel):
        ei = of_in.acquire(1); eo = of_out.acquire(1)
        kernel(ei, eo, chunk)
        of_in.release(1); of_out.release(1)

    workers = [Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), kernel])
               for i in range(n_workers)]

    def sequence(A, C, in_h, out_h):
        in_h.fill(A)
        out_h.drain(C, wait=True)

    rt = Runtime(sequence, [data_ty, data_ty, of_in.prod(), of_out.cons()])
    return Program(dev, rt, workers=workers).resolve_program()
```

> **`split`/`join` offsets are positions inside the parent object, and the sub-object sizes
> must tile it exactly.** Pairing a 1024-element offset stride with 256-element `obj_types`
> resolves and compiles cleanly, then moves only 256 of every 1024 elements — silently
> dropping 75% of the data with no error anywhere. Always check
> `len(offsets) * sub_object_size == parent_size`. If you want tile-granular L1 buffers under
> a coarser split, do the split at chunk granularity and `forward()` each sub-fifo down to
> tile-sized objects rather than shrinking `obj_types` under a coarse stride.

---

## 6. ReLU via `iron.jit` + inline `ExternalFunction`

```python
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_

N = 2048; tile_size = 256; N_div_n = N // tile_size
tensor_ty = np.ndarray[(N,),        np.dtype[bfloat16]]
tile_ty   = np.ndarray[(tile_size,), np.dtype[bfloat16]]

relu_kernel = ExternalFunction("relu_bf16", source_string=r'''
#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" {
void relu_bf16(bfloat16 *__restrict a, bfloat16 *__restrict c, int32_t n) {
    event0();
    constexpr int V = 32;
    aie::vector<bfloat16, V> zeros = aie::zeros<bfloat16, V>();
    for (int i = 0; i < n; i += V) {
        aie::vector<bfloat16, V> in = aie::load_v<V>(a + i);
        aie::store_v(c + i, aie::max(in, zeros));
    }
    event1();
}
}''', arg_types=[tile_ty, tile_ty, np.int32])

@iron.jit
def relu_op(inp, out, kernel):
    of_in  = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_fn(of_in, of_out, kernel):
        for _ in range_(N_div_n):
            ei = of_in.acquire(1); eo = of_out.acquire(1)
            kernel(ei, eo, tile_size)
            of_in.release(1); of_out.release(1)

    w = Worker(core_fn, [of_in.cons(), of_out.prod(), kernel])

    def sequence(A, C, in_h, out_h):
        in_h.fill(A)
        out_h.drain(C, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()
```

---

## 7. Two-stage pipeline (producer-consumer on two cores)

```python
import numpy as np
from ml_dtypes import bfloat16
from aie.iron import ObjectFifo, Worker, Runtime, Program, Kernel
from aie.iron.controlflow import range_

N = 1024; tile_size = 256; N_div_n = N // tile_size
tensor_ty = np.ndarray[(N,),        np.dtype[bfloat16]]
tile_ty   = np.ndarray[(tile_size,), np.dtype[bfloat16]]

def two_stage_pipeline(dev, kernel_archive):
    of_in  = ObjectFifo(tile_ty, name="in")
    of_mid = ObjectFifo(tile_ty, name="mid")
    of_out = ObjectFifo(tile_ty, name="out")

    scale = Kernel("scale_bf16", kernel_archive, [tile_ty, tile_ty, np.int32])
    relu  = Kernel("relu_bf16",  kernel_archive, [tile_ty, tile_ty, np.int32])

    def stage1(of_in, of_mid, k):
        for _ in range_(N_div_n):
            ei = of_in.acquire(1); em = of_mid.acquire(1)
            k(ei, em, tile_size)
            of_in.release(1); of_mid.release(1)

    def stage2(of_mid, of_out, k):
        for _ in range_(N_div_n):
            em = of_mid.acquire(1); eo = of_out.acquire(1)
            k(em, eo, tile_size)
            of_mid.release(1); of_out.release(1)

    w1 = Worker(stage1, [of_in.cons(),  of_mid.prod(), scale])
    w2 = Worker(stage2, [of_mid.cons(), of_out.prod(), relu])

    def sequence(A, C, in_h, out_h):
        in_h.fill(A)
        out_h.drain(C, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(dev, rt, workers=[w1, w2]).resolve_program()
```

---

## 8. Reusable operator via `MLIROperator` (op.py + design.py + test.py)

When you want a real, importable operator with a stable API, ship three files:

### `iron/operators/eltwise_add/design.py`

```python
import numpy as np
from ml_dtypes import bfloat16
from aie.iron import ObjectFifo, Worker, Runtime, Program, Kernel
from aie.iron.controlflow import range_

def build_eltwise_add(dev, N, tile_size, dtype, kernel_archive):
    N_div_n = N // tile_size
    tensor_ty = np.ndarray[(N,),        np.dtype[dtype]]
    tile_ty   = np.ndarray[(tile_size,), np.dtype[dtype]]

    of_a = ObjectFifo(tile_ty, name="a")
    of_b = ObjectFifo(tile_ty, name="b")
    of_c = ObjectFifo(tile_ty, name="c")

    add_kernel = Kernel("eltwise_add_bf16", kernel_archive,
                        [tile_ty, tile_ty, tile_ty, np.int32])

    def core_fn(of_a, of_b, of_c, k):
        for _ in range_(N_div_n):
            ea = of_a.acquire(1); eb = of_b.acquire(1); ec = of_c.acquire(1)
            k(ea, eb, ec, tile_size)
            of_a.release(1); of_b.release(1); of_c.release(1)

    w = Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod(), add_kernel])
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(w)
        rt.fill (of_a.prod(), A)
        rt.fill (of_b.prod(), B)
        rt.drain(of_c.cons(), C, wait=True)
    return Program(dev, rt).resolve_program()
```

### `iron/operators/eltwise_add/op.py`

```python
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path
from iron.common import MLIROperator          # amd/IRON only — see note below
from iron.common import AIERuntimeArgSpec     # amd/IRON only
from .design import build_eltwise_add

class EltwiseAdd(MLIROperator):
    """Element-wise add of two bfloat16 vectors on the NPU."""

    def __init__(self, N=1024, tile_size=256):
        self.N = N
        self.tile_size = tile_size
        self.dtype = np.dtype(bfloat16)
        super().__init__()

    def get_operator_name(self) -> str:
        return f"eltwise_add_N{self.N}_t{self.tile_size}"

    def get_mlir_artifact(self, dev):
        archive = self.get_kernel_artifacts()[0]
        return build_eltwise_add(dev, self.N, self.tile_size, self.dtype, archive)

    def get_kernel_artifacts(self):
        return [str(Path(__file__).parent / "kernels" / "eltwise_add_bf16.a")]

    def get_arg_spec(self):
        # Returns list[AIERuntimeArgSpec]; each entry is (role, shape).
        shape = (self.N,)
        return [
            AIERuntimeArgSpec("in",  shape),   # A
            AIERuntimeArgSpec("in",  shape),   # B
            AIERuntimeArgSpec("out", shape),   # C
        ]
```

> **These three files use the `amd/IRON` operator framework** (`MLIROperator`, `AIERuntimeArgSpec`, `op.compile()`, the `iron/operators/<name>/` layout). That framework ships in the downstream **`amd/IRON`** repo, **not** the base `mlir_aie` wheel — examples 1–7 above (plain `@iron.jit` + `aie.iron`) run on the wheel alone; this pattern additionally requires an `amd/IRON` checkout. Use it only when the target environment has `amd/IRON` installed.
>
> **`design.py` intentionally uses the old `Runtime()`/`rt.sequence()`/`rt.start()`/`rt.fill()`/`rt.drain()` API, not the v1.4.0 `Runtime(sequence, fn_args)` style used in examples 1–7.** `amd/IRON` currently pins `mlir_aie==1.3.5.dev20+g167f34d` in its `requirements.txt`, predating the v1.4.0 Runtime rework — every operator design in that repo is written against the old API. If you're generating a design meant to live under `amd/IRON`'s `iron/operators/`, match this old-style pattern; only use the new-style `Runtime` from examples 1–7 for plain `aie.iron`/`iron.jit` code that targets the base wheel directly. Re-check `amd/IRON`'s `requirements.txt` pin before assuming it has caught up.

### `iron/operators/eltwise_add/test.py`

```python
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from iron.common.test_utils import verify_buffer   # amd/IRON only (not in the mlir_aie wheel)
from .op import EltwiseAdd

def test_eltwise_add_basic():
    N = 1024
    op = EltwiseAdd(N=N, tile_size=256); op.compile()

    a = iron.rand ((N,), dtype=np.dtype(bfloat16))
    b = iron.rand ((N,), dtype=np.dtype(bfloat16))
    c = iron.zeros((N,), dtype=np.dtype(bfloat16))
    op(a, b, c)

    ref = (np.array(a, dtype=np.float32) + np.array(b, dtype=np.float32)).astype(bfloat16)
    verify_buffer(c, "c", ref, rel_tol=0.02, abs_tol=1e-6)
```
