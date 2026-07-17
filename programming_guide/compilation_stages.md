<!---//===- compilation_stages.md ---------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Compilation Stages

What actually happens between the moment you call an `@iron.jit` design
and the moment XRT runs it on the NPU.

This page is a reference, not a tutorial. If you only want to *use*
`@iron.jit`, the [landing page](./README.md) and
[Section 3](./section-3/README.md) are enough. Come here when you need to debug
a cache miss, plumb a custom kernel object, or understand which setting
lives at which stage.

## The pipeline

```
   Python function (@iron.jit)
        │
        │  decoration: capture generator + CompileTime[T] params + flags
        ▼
   CompilableDesign           ◄── frozen recipe (immutable inputs)
        │
        │  recipe_hash → cache key
        │  artifact_hash → "is this still fresh on disk?"
        ▼
   _generate_mlir             ◄── runs the generator exactly once
        │                          (memoized via cached MLIR text)
        ▼
   MLIR module
        │
        │  aiecc.main: aie-opt → aie-translate → peano/chess → bootgen
        ▼
   xclbin + insts(.bin|.elf)  ◄── on-disk artifacts (under $NPU_CACHE_HOME)
        │
        ▼
   CallableDesign             ◄── runtime callable
        │
        │  NPUKernel: xrt::hw_context + xrt::kernel + run()
        ▼
   NPU
```

The four objects you may interact with directly:

| Object | Where you get one | What it owns |
|---|---|---|
| `CompilableDesign` | `@compileconfig` decorator, or `CallableDesign.compilable`. | Generator function, `compile_kwargs`, `compile_flags`, `aiecc_flags`, source/include/object file lists, recipe hash. |
| `CallableDesign` | `@iron.jit` decorator, or `CallableDesign(CompilableDesign)`. | A `CompilableDesign` + runtime call protocol (`__call__`, `as_mlir`, `specialize`, `compile`). |
| `xclbin` / insts | Returned by `.compile()`; cached on disk. | Reused across runs whose `recipe_hash` matches and whose source/tool timestamps still match the `artifact_hash`. |
| `NPUKernel` | Built lazily on first `__call__`. | XRT `hw_context` + `kernel` + `run()` plumbing. |

## Stages in detail

### 1. Decoration — `@iron.jit` / `@compileconfig`

The decorator captures four things and returns a `CallableDesign` (or
`CompilableDesign` for `@compileconfig`):

- the **generator function** itself,
- any **pre-bound `CompileTime[T]` values** passed to the decorator
  (`@iron.jit(N=1024)`-style),
- `compile_flags` (passed to `aie-opt` for MLIR-side switches),
- `aiecc_flags` (passed to `aiecc.main` for tool-chain switches like
  `--packet-sw-objFifos`).

No MLIR is built. No tool runs. The decorator is constant-time and the
returned object is safe to capture into closures, pickle into `to_json`,
or hand to another module.

### 2. Recipe freeze — `CompilableDesign`

When `CompilableDesign` materializes, its inputs are frozen:
`compile_kwargs` becomes a `MappingProxyType`, the flag/file lists
become tuples. The recipe **does not change** for the lifetime of the
object — that invariant is what lets the memoization in stage 3 be
safe.

Two hashes drop out of the frozen recipe:

- **`recipe_hash`** — function of generator + `compile_kwargs` +
  compile/aiecc flags. Same across machines. Used as the cache key
  for "have I built this design before?"
- **`artifact_hash`** — function of source-file mtimes, object-file
  mtimes, tool-binary mtimes, and target architecture. Different
  across machines / builds. Used to decide "are the cached artifacts
  still fresh, or do I rebuild?"

`.specialize(**kw)` returns a *new* `CompilableDesign` with the kwargs
merged in; the original is untouched.

### 3. MLIR generation — `_generate_mlir`

Called exactly once per `CompilableDesign` (per generation request),
even across many `.compile()` / `__call__` / `.as_mlir()` invocations.
The first call:

1. Opens a fresh MLIR `Context` and `Location` via `mlir_mod_ctx()`.
2. Builds tensor placeholders for the `In`/`Out`/`InOut` parameters.
3. Calls the generator with `(**placeholders, **compile_kwargs)`.
4. Captures the resulting `Module` (either the generator's return
   value, or `ctx.module` if it returned `None` — "placed" vs
   "unplaced" generator style).
5. Verifies the module and serializes it to text.

The text is stashed on the instance with `@functools.cached_property`.
Subsequent calls re-parse the text into a fresh `Context` each time —
the returned `Module` is always bound to the active `Context`, so it
can never escape. The generator body itself never runs again, so any
side effects in it (kernel registration, `ExternalFunction` instance
tracking) are also captured once and replayed.

### 4. Lowering — `aiecc.main`

The serialized MLIR is handed to `aiecc.main` along with
`aiecc_flags`, the source/include/object file lists, the target
architecture (derived from the device the generator captured), and
the recipe hash (used to name the work directory).

`aiecc` runs the usual sequence — `aie-opt` passes → `aie-translate` →
peano or chess for the kernel `.o` → linker → `bootgen` for the
xclbin — and writes the final `xclbin` and `insts.bin` (and optionally
`insts.elf` when `elf_path=` is set) under `$NPU_CACHE_HOME`.

If the cache directory already has an `xclbin` + insts matching both
the recipe hash *and* the artifact hash, `aiecc` is **not invoked** at
all — the cached paths are returned directly. This is the common case
on a warm run.

To force a rebuild, decorate with `@iron.jit(..., use_cache=False)` or
remove the cache directory:

```bash
rm -rf "${NPU_CACHE_HOME:-$HOME/.npu/cache}"
```

#### Per-design cache directory contents

Each cache directory `$NPU_CACHE_HOME/<recipe_hash>/` keeps every
artifact `aiecc` produced for that design, not just the two the runtime
loads.  This is useful when you need to inspect the lowered MLIR, the
post-placement IR, the per-core LLVM IR, or the actual `.o` files that
linked into the final xclbin.

| File | What it is |
|---|---|
| `.lock` | cross-process flock so concurrent compiles of the same recipe don't collide |
| `aie.mlir` | input MLIR your generator produced (post-`resolve_program`) — the same text `cd.as_mlir(...)` returns |
| `input_with_addresses.mlir` | lowered MLIR with shim / memtile DMA addresses bound; the last MLIR form before `aie-translate` |
| `final.xclbin` | the binary XRT loads onto the NPU |
| `insts.bin` | NPU runtime instruction stream paired with the xclbin |
| `insts.elf` | only present when `.compile(elf_path=...)` is set — `xrt::elf` variant of `insts.bin` |
| `main.pdi` | Programmable Device Image — config data packed by `bootgen`; locate via `get_pdi_path()` or name it with `.compile(pdi_path=...)` |
| `main_aie_partition.json` | AIE partition manifest (which tiles, kernels, memory regions) |
| `main_aie_cdo_init.bin` | CDO blob initialising tile / DMA registers |
| `main_aie_cdo_elfs.bin` | CDO blob carrying the per-core ELFs to the device |
| `main_aie_cdo_enable.bin` | CDO blob enabling tiles after init |
| `main_design.bif` | Bootgen Image Format spec used to assemble `main.pdi` |
| `main_kernels.json` | kernel metadata XRT consumes alongside the xclbin |
| `main_mem_topology.json` | memory-topology descriptor embedded into the xclbin |
| `main_core_<col>_<row>.elf` | per-core final ELF the AIE tile actually runs |
| `main_core_<col>_<row>.ld.script` | linker script Peano used to lay out that ELF |
| `main_core_<col>_<row>.ll` | per-core LLVM IR before `opt` |
| `main_core_<col>_<row>.opt.ll` | per-core LLVM IR after `opt` |
| `main_core_<col>_<row>.peano-compat.ll` | Peano-compatible IR (LLVM-23 features downgraded for Peano's opt/llc) |
| `main_core_<col>_<row>.o` | per-core compiled object that links into the `.elf` |
| `<kernel>.cc` | `ExternalFunction` kernel source, copied in for `clang` |
| `<kernel>_<hash>.o` | `clang`'s compiled `.o` for that `ExternalFunction` |

Designs that pin a single Worker (e.g. `Tile(0, 2)`) produce one set
of `main_core_*` files; multi-Worker designs like the `whole_array`
matmul produce one set per `(col, row)` the placer landed on.

To list what a freshly compiled design wrote out:

```python
from pathlib import Path
from aie.utils.compile import NPU_CACHE_HOME

xclbin, _ = my_design.specialize(N=4096).compile()
kernel_dir = Path(xclbin).parent
for p in sorted(kernel_dir.iterdir()):
    print(p.name)
```

#### Controlling artifact output

By default `.compile()` writes into the cache under `$NPU_CACHE_HOME` with
fixed names (`final.xclbin`, `insts.bin`, `main.pdi`, …). Pass explicit paths
to write named artifacts wherever you want; this **bypasses the cache** (the
caller is presumed to own dependency tracking, e.g. a Makefile):

```python
my_design.specialize(N=4096).compile(
    xclbin_path="out/design.xclbin",   # binary XRT loads
    inst_path="out/design.insts.bin",  # NPU instruction stream
    pdi_path="out/design.pdi",         # Programmable Device Image
    elf_path="out/design.insts.elf",   # xrt::elf-wrapped insts (optional)
)
```

`xclbin_path` and `inst_path` must be given together. `pdi_path` and
`elf_path` each additionally require both of those — the cache path doesn't
track caller-named PDI/ELF artifacts.

If you're using the default cache (no explicit paths) but still need the PDI,
`aiecc` writes a `<device>.pdi` (`main.pdi` for `@iron.jit` designs) into the
cache directory on every compile. Locate it without recompiling via
`get_pdi_path()`, which finds the PDI regardless of the device symbol name:

```python
spec = my_design.specialize(N=4096)
spec.compile()   # cache mode
pdi = spec.get_pdi_path()   # -> Path to the emitted .pdi
```

A multi-device design emits one PDI per `aie.device`. Use `get_pdi_paths()`
for the full list, or `get_pdi_path(device_name="…")` to pick one by its
`aie.device` symbol:

```python
all_pdis = spec.get_pdi_paths()                  # -> [Path, ...]
one = spec.get_pdi_path(device_name="second")    # -> Path to second.pdi
```

The [`vector_scalar_add`](../programming_examples/basic/vector_scalar_add/)
example demonstrates the explicit-path flow end-to-end via its `--aot-dir`
flag.

### 5. Runtime binding — `CallableDesign.__call__`

The first call to a `CallableDesign` does three things:

1. Merges call-time `**compile_kwargs` over the pre-bound recipe
   (call-time wins). Tensor-typed kwargs are forbidden — tensors are
   positional only.
2. Triggers `.compile()` (stage 4) if the merged recipe has no cached
   artifacts.
3. Lazily constructs an `NPUKernel` from the xclbin + insts, with
   `trace_config=` forwarded to `NPUKernel.__init__` (not to the call
   itself).

The `NPUKernel` is cached per-recipe, so subsequent calls with the
same compile kwargs reuse the same XRT `hw_context` and `kernel`
objects.

#### Tensor-arg validation

Before any DMAs run, each positional tensor argument is checked against
the corresponding entry in the compiled `aie.runtime_sequence`'s
signature.  The runtime_sequence in `input_with_addresses.mlir` carries
fully resolved memref types — `parse_dma_sizes` reads the per-arg
element count straight off the signature, so fan-outs, repeated
transfers (matmul B reloaded each tile-row), and InOut fill+drain pairs
on the same arg all give the host-tensor size directly without
re-folding multi-DMA patterns.

A mismatch raises `RuntimeError` *before* the kernel runs, naming the
parameter and the `CompileTime[T]` settings that produced the expected
size:

```python
wrong_size = iron.zeros(99, dtype=np.uint8)
try:
    passthrough(wrong_size, out_t, N=4096)
except RuntimeError as e:
    print(e)
# Tensor argument 'a_in' has 99 elements but the kernel was compiled
# for 4096 elements.
# CompileTime[T] parameters used at compile time: {'N': 4096}
```

The check runs on both fresh compiles and cache hits.  It is silently
skipped when the expected sizes cannot be determined — for example,
modules with several `aie.runtime_sequence`s and no unique call-graph
root (multi-device programs that route through `aiex.run` from several
top-level sequences).  Args wired only to runtime params and never DMA-
transferred (expected entry `0`) are also skipped.

#### Consuming external artifacts (bring your own)

Stages 1–4 exist to *produce* an xclbin + insts.  If you already have that
pair — from a Makefile, a raw `aiecc` invocation, the `.compile(xclbin_path=,
inst_path=)` export above, or any other tool — you can skip straight to stage 5
by constructing an `NPUKernel` directly:

```python
from aie.utils import NPUKernel
import aie.iron as iron

kernel = NPUKernel("design.xclbin", "design.insts.bin")

in_t = iron.arange(1, 1025, dtype=np.int32, device="npu")
out_t = iron.zeros_like(in_t)
kernel(in_t, out_t)          # blocking; output written in place
```

`NPUKernel` does not care how the artifacts were built — it only needs the
xclbin and the instruction binary.  There is **no `CompileTime[T]` recipe and
no tensor-arg validation** on this path (that metadata lives in the JIT
`CompilableDesign`, which you've bypassed), so *you* own matching the tensor
shapes/dtypes to what the artifacts were compiled for.

The run inputs are the **xclbin + insts**. A PDI is not a standalone run input
on the IRON runtime — `bootgen` packs it *inside* the xclbin, and the runtime
loads via `pyxrt.xclbin(...)`.  Use `.compile(pdi_path=...)` / `get_pdi_path()`
to *extract* a PDI for an external consumer, not to feed one back in.

The [`vector_scalar_add`](../programming_examples/basic/vector_scalar_add/)
example demonstrates this via its `--from-xclbin` / `--from-insts` flags.

## Setting → stage cheat-sheet

| Setting | Set at | Affects |
|---|---|---|
| `@iron.jit(N=...)` (pre-bound `CompileTime[T]`) | decoration | recipe_hash (stage 2) |
| Call-time `cd(a, b, N=...)` kwarg | each call | merged recipe for that call (stage 5) |
| `@iron.jit(compile_flags=[...])` | decoration | aie-opt pass options (stage 4) |
| `@iron.jit(aiecc_flags=[...])` | decoration | aiecc.main switches (stage 4) |
| `@iron.jit(source_files=[...])` | decoration | extra kernel `.cc` files compiled into the design (stage 4) |
| `@iron.jit(object_files=[...])` | decoration | pre-built kernel `.o` files linked into the design (stage 4) |
| `@iron.jit(use_cache=False)` | decoration | bypass on-disk cache; always rebuild (stage 4) |
| `.compile(elf_path=...)` | compile call | also emit `insts.elf` for the `xrt::elf` testbench flow (stage 4); requires explicit `xclbin_path` + `inst_path` |
| `.compile(pdi_path=...)` | compile call | write the PDI to a chosen path (stage 4); requires explicit `xclbin_path` + `inst_path` |
| `$NPU_CACHE_HOME` env var | process-wide | where `aiecc` reads/writes artifacts (stage 4) |
| `trace_config=` kwarg at call-time | each call | forwarded to `NPUKernel.__init__` (stage 5); **not** a CompileTime[T] kwarg |
| `iron.set_current_device(...)` | before generator runs | target architecture baked into the generator's `Program` (stage 3) |

## Inspecting an intermediate stage

| You want to see… | Call |
|---|---|
| The serialized MLIR (no NPU touched, no aiecc run) | `cd.as_mlir(**kwargs)` |
| The cached artifact paths | `cd.compile(**kwargs)` returns `(xclbin_path, insts_path)` |
| The PDI path (cache mode) | `cd.get_pdi_path()` (after a `compile()`) |
| Artifacts at chosen paths | `cd.compile(xclbin_path=, inst_path=, pdi_path=, elf_path=)` |
| Run a pre-built xclbin + insts | `NPUKernel(xclbin, insts)(in_t, out_t)` |
| The recipe hash (cache key) | `cd.compilable.recipe_hash` |
| The artifact hash (freshness key) | `cd.compilable.artifact_hash` |
| Whether the next call will rebuild | `cd.compilable.artifact_hash != cached_artifact_hash` |

### Progressive binding: chained `.specialize()`

`specialize()` is *immutable-style* — it returns a new `CallableDesign`
with the kwargs merged in, leaving the original untouched.  Binding
compile params can be split across call sites without losing the
unspecialised handle:

```python
@iron.jit
def add_const(x: In, y: Out, *,
              N: CompileTime[int], dtype: CompileTime[type]):
    ...

half  = add_const.specialize(N=4096)         # bind N now
full  = half.specialize(dtype=np.int32)      # bind dtype later
wider = full.specialize(N=8192)              # override an earlier bind

add_const  # still unspecialised — half/full/wider didn't mutate it
```

Because `as_mlir` / `compile` / `__call__` are plain methods on the
returned object, `functools.partial` works the same way for pinning
*non-compile* kwargs:

```python
import functools
as_mlir_N4096 = functools.partial(add_const.as_mlir, N=4096)
mlir = as_mlir_N4096(dtype=np.int32)
```

### Watching the compile pipeline

All compile-path subprocesses (`clang++` for `ExternalFunction`s,
`aiecc` for the design itself) log through `aie.utils.compile.*` —
setting the package root logger to `DEBUG` cascades to every child.
Each line carries the exact command line invoked plus cache hit / miss
messages, which is enough to copy the `clang++` invocation into a
terminal and iterate on a kernel without running the full design:

```python
import logging
logging.basicConfig(level=logging.WARNING,
                    format="%(name)s | %(message)s", force=True)
logging.getLogger("aie.utils.compile").setLevel(logging.DEBUG)

# use_cache=False forces a fresh compile so every clang++ / aiecc
# invocation lights up (a warm cache hit emits only one DEBUG line).
@iron.jit(use_cache=False)
def design(...): ...

design.specialize(N=4096).compile()

logging.getLogger("aie.utils.compile").setLevel(logging.WARNING)
```

## Appendix A: three ways to bind `CompileTime[T]` parameters

`CompileTime[T]`-annotated parameters are part of the compile recipe (they
participate in the recipe hash), so the value has to be known by
stage 4.  Three places can supply it; later wins.  All three are
keyword-only — `CompileTime[T]` params must sit after a `*,` in the
signature (the framework rejects positional `CompileTime[T]` at decoration
time with a fix-it `TypeError`).

```python
# A — bare decorator; params supplied at call time
@iron.jit
def gemm_a(a: In, b: In, c: Out, *,
           M: CompileTime[int], N: CompileTime[int]):
    ...
gemm_a(a, b, c, M=512, N=512)            # call-time bind

# B — pre-bound at decoration; call-time can still override
@iron.jit(M=512, N=512)
def gemm_b(a: In, b: In, c: Out, *,
           M: CompileTime[int], N: CompileTime[int]):
    ...
gemm_b(a, b, c)                          # uses pre-bound 512×512
gemm_b(a, b, c, M=1024)                  # call-time override wins → recompile

# C — signature defaults; used at lowering time if not bound otherwise
@iron.jit
def gemm_c(a: In, b: In, c: Out, *,
           M: CompileTime[int] = 256,
           N: CompileTime[int] = 256):
    ...
gemm_c(a, b, c)                          # uses 256×256
```

Two fail-fast guards run at decoration time so misuse surfaces
immediately rather than silently running a kernel with the wrong value
baked in:

- **Unknown kwargs to `@iron.jit`** raise `TypeError` listing the valid
  `CompileTime[T]` params and the recognised config keys
  (`aiecc_flags`, `source_files`, `use_cache`, …).  Catches typos like
  `@jit(NN=512)` instead of `@jit(N=512)`.
- **Unannotated scalar params with default values** raise `TypeError`
  with a fix-it message pointing at either `CompileTime[T] = default`
  (recompile on per-call change) or `In / Out / InOut` (tensor).
  Prevents silently baking a default into the compiled MLIR while a
  per-call override is ignored.

A defaulted *and* call-time-overridden `CompileTime[T]` recompiles for the
new value; the cache key includes the resolved value, not the source
default.

## Related reading

- [`aiecc` compiler driver](../tools/aiecc/README.md)
  — the declarative build-graph driver that turns the lowered MLIR into
  core ELFs, CDO/PDI, xclbin and NPU instruction streams, with a developer
  guide to its `Edge`/`Node`/`Item` graph model and `--checkpoint`/`--resume`.
- [`iron_configuration.md`](./iron_configuration.md) — environment
  variables (cache dir, default tensor class, default device, log
  level).
- [`implicit_mlir_context.md`](./implicit_mlir_context.md) — what
  goes wrong when MLIR ops leak across `Context` boundaries.
- [Section 1](./section-1/README.md) — the building blocks the generator
  produces (`Worker`, `ObjectFifo`, `Runtime`, `Program`).
