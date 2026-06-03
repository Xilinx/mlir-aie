<!---//===- compilation_stages.md ---------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Compilation Stages

What actually happens between the moment you call an `@iron.jit` design
and the moment XRT runs it on the NPU.

This page is a reference, not a tutorial. If you only want to *use*
`@iron.jit`, the [landing page](./README.md) and
[Section 3](./section-3/) are enough. Come here when you need to debug
a cache miss, plumb a custom kernel object, or understand which knob
lives at which stage.

## The pipeline

```
   Python function (@iron.jit)
        │
        │  decoration: capture generator + Compile[T] params + flags
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
   xclbin + insts(.bin|.elf)  ◄── on-disk artifacts (under $NPU_CACHE_DIR)
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
- any **pre-bound `Compile[T]` values** passed to the decorator
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
`insts.elf` when `elf_path=` is set) under `$NPU_CACHE_DIR`.

If the cache directory already has an `xclbin` + insts matching both
the recipe hash *and* the artifact hash, `aiecc` is **not invoked** at
all — the cached paths are returned directly. This is the common case
on a warm run.

To force a rebuild, decorate with `@iron.jit(..., use_cache=False)` or
remove the cache directory:

```bash
rm -rf "${NPU_CACHE_DIR:-$HOME/.npu/cache}"
```

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

## Knob → stage cheat-sheet

| Knob | Set at | Affects |
|---|---|---|
| `@iron.jit(N=...)` (pre-bound `Compile[T]`) | decoration | recipe_hash (stage 2) |
| Call-time `cd(a, b, N=...)` kwarg | each call | merged recipe for that call (stage 5) |
| `@iron.jit(compile_flags=[...])` | decoration | aie-opt pass options (stage 4) |
| `@iron.jit(aiecc_flags=[...])` | decoration | aiecc.main switches (stage 4) |
| `@iron.jit(source_files=[...])` | decoration | extra kernel `.cc` files compiled into the design (stage 4) |
| `@iron.jit(object_files=[...])` | decoration | pre-built kernel `.o` files linked into the design (stage 4) |
| `@iron.jit(use_cache=False)` | decoration | bypass on-disk cache; always rebuild (stage 4) |
| `@iron.jit(elf_path=...)` | decoration | also emit `insts.elf` for the `xrt::elf` testbench flow (stage 4) |
| `$NPU_CACHE_DIR` env var | process-wide | where `aiecc` reads/writes artifacts (stage 4) |
| `trace_config=` kwarg at call-time | each call | forwarded to `NPUKernel.__init__` (stage 5); **not** a Compile[T] kwarg |
| `iron.set_current_device(...)` | before generator runs | target architecture baked into the generator's `Program` (stage 3) |

## Inspecting an intermediate stage

| You want to see… | Call |
|---|---|
| The serialized MLIR (no NPU touched, no aiecc run) | `cd.as_mlir(**kwargs)` |
| The cached artifact paths | `cd.compile(**kwargs)` returns `(xclbin_path, insts_path)` |
| The recipe hash (cache key) | `cd.compilable.recipe_hash` |
| The artifact hash (freshness key) | `cd.compilable.artifact_hash` |
| Whether the next call will rebuild | `cd.compilable.artifact_hash != cached_artifact_hash` |

For a tour of these same methods with timings, see the
[`@iron.jit` notebook](./_whats_new_executed.md).

## Related reading

- [`iron_configuration.md`](./iron_configuration.md) — environment
  variables (cache dir, default tensor class, default device, log
  level).
- [`implicit_mlir_context.md`](./implicit_mlir_context.md) — what
  goes wrong when MLIR ops leak across `Context` boundaries.
- [Section 1](./section-1/) — the building blocks the generator
  produces (`Worker`, `ObjectFifo`, `Runtime`, `Program`).
