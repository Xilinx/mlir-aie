<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# C++-linkage kernels: resolve the symbol at the artifact, not in the frontend

Date: 2026-08-25
Branch: `ypapadop-amd/mangled-name-support`

## Problem

An IRON kernel declared as `ExternalFunction("reduce_min_vector", ...)` emits an
MLIR `func.func private @reduce_min_vector`. The linker therefore needs a symbol
literally named `reduce_min_vector` in the object file. Today that is guaranteed
only by writing `extern "C"` trampolines by hand:

```c++
extern "C" {
void reduce_min_vector(int32_t *a, int32_t *c, int32_t n) { _reduce_min_vector(a, c, n); }
}
```

This is pure boilerplate, it forces every kernel to carry a shadow `_`-prefixed
implementation, and it blocks namespaces and overloads in kernel sources.

## Decision

**Resolve the symbol where the artifact is, by demangling — never by mangling in
the frontend.**

After a kernel object is compiled, list its defined global symbols, demangle
them, and find the one whose *base name* (the text before `(`) equals the symbol
IRON expects. Rename that symbol to the expected name with
`llvm-objcopy --redefine-sym`. IRON's MLIR generation is unchanged and never
learns that mangling exists.

### Why not mangle in Python

The alternative — build `_Z17reduce_min_vectorPiS_i` from `(name, arg_types)` —
was rejected. Mangling and demangling are not symmetric in difficulty:

- **Mangling requires reconstructing exact C++ types.** `arg_types` carries numpy
  types, which cannot express `const`. `void f(const int32_t*, int32_t*, int32_t)`
  mangles to `_Z2f1PKiPii`, not `_Z2f1PiS_i`, so a read-only kernel written the
  natural way would fail to link. It also cannot express references, `aie_api`
  vector types, or default arguments.
- **Itanium substitution compression is stateful.** The second `int32_t*` in
  `_Z17reduce_min_vectorPiS_i` is a back-reference, and the index shifts with
  namespace nesting: the same parameters become `S0_` in
  `_ZN2ns1gEPiS0_i`. A mangler that works on flat functions breaks silently the
  first time someone adds a namespace, surfacing only as a link error.
- **Demangling has none of this.** `const`, `__restrict`, namespaces, and
  overload parameter lists all reduce to the same base name, so every one of the
  above cases is handled without the frontend knowing anything about them.

Two variants of frontend mangling were considered and dropped: `libclang`
(adds a versioned `clang==20.1.0` + matching `libclang.so` dependency, and
mangles with the *system* clang rather than the one that built the object), and
a hand-written Itanium mangler in Python (maintaining an ABI implementation for
no benefit over reading the answer off the artifact).

### Tooling

No new Python dependencies. Peano already ships the tools, verified against a
real `aie2` object on 2026-08-25:

| Tool | Source | Verified |
| --- | --- | --- |
| `llvm-nm --defined-only --extern-only --format=just-symbols` | Peano `bin/` | lists exactly the global symbols, excluding `.LBB*` locals |
| `llvm-cxxfilt` | Peano `bin/` | demangles `_Z17reduce_min_vectorPiS_i` → `reduce_min_vector(int*, int*, int)` |
| `llvm-objcopy --redefine-sym` | `config.objcopy_path()` | renames the symbol in an AIE-`e_machine` object; **exit 0 when the symbol is absent**, so it is idempotent |

`config.objcopy_path()` (`python/utils/config.py:81`) already exists and already
documents why GNU binutils `objcopy` cannot be used here.

## Implementation

### Where

`compile_external_kernel()` in `python/utils/compile/utils.py`, at the end of the
compile, **before** the existing `symbol_prefix` rename. The file already has
`_rename_symbol_in_object()` using `--redefine-sym`, already exercised by the
`symbol_prefix` feature.

Both call sites — the fresh compile and the cache-hit early return — were
factored into one `_apply_symbol_renames(func, output_file)` helper, so the two
renames cannot drift apart.

### Ordering with `symbol_prefix`

The two renames compose, and the order is load-bearing. For a C++-linkage source
the defined symbol is `_Z...`, not `_original_name`, so the existing prefix
rename would match nothing and silently no-op. Resolving C++ linkage first
restores the invariant the prefix rename already assumes:

```
_Z17reduce_min_vectorPiS_i  --(new step)-->  reduce_min_vector  --(existing)-->  myprefix_reduce_min_vector
```

### Algorithm

1. Run `llvm-nm --defined-only --extern-only --format=just-symbols` on the object.
2. If `_original_name` is already present verbatim, return — the kernel has C
   linkage. This is the existing-behaviour fast path and costs one `llvm-nm`.
3. Otherwise demangle the list through `llvm-cxxfilt` and collect every symbol
   whose base name equals `_original_name`.
4. Exactly one match: `_rename_symbol_in_object(obj, mangled, _original_name)`.
5. Zero matches: raise, listing the demangled symbols actually present.
6. More than one match: raise as ambiguous, listing the demangled candidates and
   stating that IRON cannot pick between overloads.

Both errors name the object file and the expected symbol.

### Cache-hit path

`compile_external_kernel` returns early when the object already exists, and
already re-applies the prefix rename there for exactly this reason. The C++-linkage resolution must be applied on that path too. It is
idempotent: on a cached object the plain name is already present, so step 2
returns immediately.

## Scope

**In the first cut:** `ExternalFunction`, Peano toolchain, object (non-inline)
mode. This covers the `reduce_min` example and every kernel produced by the
`iron.kernels.*` factories.

**Explicitly excluded:**

- `inline=True` / `link_with_mode="merge"` — the artifact is textual `.ll` fed to
  `llvm-link`; no linker and no objcopy are involved, so neither mechanism
  applies. Symbol fixup there means rewriting the IR module, which is a separate
  design. **No new guard was needed:** `_make_ir_inlinable` (`utils.py:154`)
  already raises when it cannot find a `define` for the expected symbol, and its
  message already names the `extern "C"` fix.
- `use_chess=True` — xchesscc's object format has not been tested with
  `llvm-objcopy` or `llvm-nm`, and the toolchain was not installed on the
  development machine, so it could not be tested. Rather than assume, the chess
  path skips C++-linkage resolution entirely and keeps byte-for-byte the
  behaviour it had before. Inspecting an xchesscc object could otherwise break
  chess kernels that link fine today. Lifting this needs someone with the
  toolchain to confirm `llvm-nm` reads those objects.

**Out of scope for this cut:** prebuilt `Kernel("foo.o")`. Those objects belong
to the user and should not be mutated in place; covering them means either
copying into `kernel_dir` first or passing `--defsym` through aiecc's link. Both
are reasonable follow-ups but neither is needed for the example.

## Signature checking

Binding by base name would let a mismatch between `arg_types` and the real C++
signature link silently. Since the matched symbol is demangled anyway, its
parameter list is free to inspect — so `_check_cxx_signature` compares the two.

This is only possible for C++-linkage kernels: an `extern "C"` symbol demangles
to a bare name with no parameter list. **Dropping the trampolines is what buys
the check** — it cannot be a blanket guarantee, only a benefit that arrives as
kernels stop using `extern "C"`.

Strictness is tiered by what a false positive would cost:

| Tier | Check | On mismatch |
| --- | --- | --- |
| 1 | Parameter count | Hard error — a demangled signature's arity is unambiguous |
| 2 | Pointer vs scalar | Hard error — equally unambiguous, and silently corrupts today |
| 3 | Element type | Error **only when both** the C++ spelling and the numpy dtype are modelled |

Tier 3's restriction is the safety valve: a kernel taking `aie::vector<int32,
16>*`, a struct, or `void*` demangles to a spelling the table does not contain,
and the checker stays silent rather than guessing. A checker that does not
understand a type must not reject it.

Parsing notes, all driven by observed llvm-cxxfilt output: the parameter list is
split on *top-level* commas only (`aie::vector<int, 16>*` contains one of its
own); `const` is emitted **east** (`int const*`); and top-level `__restrict` is
dropped by the mangling before demangling ever sees it.

### Rollout risk: measured, not assumed

Tier 1 is the tier that could break a working build. A sweep of `aie_kernels/`
found that **51 of 55 sources use `extern "C"`**; of the four that do not, two
are `zero.cc` (templates-only, `#include`d rather than compiled standalone). The
other two, `reduce_min.cc` and `reduce_add.cc`, are the ones this change
converts. Tier 1 therefore has no pre-existing kernel to break, and ships as a
hard error.

### How much boilerplate this actually removes

Less than the size of the `extern "C"` blocks suggests, and it is worth being
precise about why. Ranking the shipped kernels by trampoline-block size finds
`bn_conv2dk1_relu.cc` (21 functions), `bn_conv2dk1_i8.cc` (17), and `mm.cc` (20,
generated by an X-macro over five type combinations). **None of these are
boilerplate this change can remove.** They do real work:

- `mm.cc`'s wrappers pin template arguments — `matmul_i8_i8` forwards to
  `matmul_vectorized_4x8x8_i8_i8<DIM_M, DIM_K, DIM_N>`. A template
  instantiation's demangled base name *includes* its arguments
  (`matmul_vectorized_4x8x8_i8_i8<64, 64, 64>`), so it can never match a plain
  expected name of `matmul_i8_i8`.
- `threshold.cc`'s wrappers reorder arguments and supply defaults.
- The bottleneck kernels' wrappers rename deliberately (`bn13_1_conv2dk1_...`
  → `conv2dk1_ui8_ui8_scalar_...`), encoding which network layer a kernel serves.

Deleting any of those would change behaviour, not remove noise. Only a wrapper
that forwards identical arguments to an identically-named implementation is pure
boilerplate; a scan for that shape found exactly two files, `reduce_min.cc` and
`reduce_add.cc`, at two functions each.

The honest summary: this change removes ~16 lines per converted kernel today.
Its value is not in bulk deletion but in what it unblocks — namespaces,
overloads, `const` parameters, and the signature checking below, none of which
`extern "C"` permits.

The genuine regression risk is different, and is what the catalog sweep test
actually covers: symbol resolution now runs `llvm-nm` over *every* compiled
kernel object, including the 52 `extern "C"` ones that take the fast path. All
41 buildable catalog kernels compile and export their expected symbol.

## Reverting the current branch

The frontend approach on this branch is removed:

- `python/iron/kernel.py`: delete `find_mangled_symbol` and `create_mangled_name`,
  restore `Kernel.resolve()` to emit `self._name`, drop the `cxxfilt` /
  `elftools` / `clang.cindex` / `tempfile` imports and the unused
  `peano_install_dir` import.
- `python/requirements.txt`: drop `cxxfilt`, `pyelftools`, `clang==20.1.0`.
- `aie_kernels/aie2/reduce_min.cc`: **keep** the change. Dropping the `extern "C"`
  trampolines is the point of the exercise and becomes the end-to-end test case.

## Testing

**Unit** — against a small C++ source compiled by Peano to an `aie2` object,
covering: a C-linkage symbol (fast path, no rename), a namespaced C++ symbol, a
`const`-parameter symbol, an absent symbol (error names what is present), and two
overloads (ambiguity error names both candidates). Fixtures build the object in a
tmpdir with the same flags `compile_cxx_core_function` uses.

**Integration** — the `reduce_min` design compiles and runs with
`aie_kernels/aie2/reduce_min.cc` carrying no `extern "C"`, matching the numeric
result it produced before the change.

**Regression** — `test/python/test_kernels_specs.py` continues to pass unchanged;
it asserts `expected_name="reduce_min_vector"`, which is exactly the invariant
this design preserves. The catalog sweep in `test_cxx_linkage_symbols.py`
compiles every non-NPU2 kernel factory and asserts it still exports its symbol,
covering the `extern "C"` fast path across the whole shipped catalog.

**Signature checking** — `test/python/test_cxx_signature_check.py` covers the
three tiers plus the parsing edge cases (nested template commas, east const,
unknown C++ types, unknown numpy dtypes) without needing a toolchain.
