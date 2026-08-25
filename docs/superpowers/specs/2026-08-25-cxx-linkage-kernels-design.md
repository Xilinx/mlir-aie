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

`compile_external_kernel()` in `python/utils/compile/utils.py:557`, at the end of
the compile, **before** the existing `symbol_prefix` rename. The file already has
`_rename_symbol_in_object()` (`utils.py:545`) using `--redefine-sym`, already
exercised by the `symbol_prefix` feature.

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

`compile_external_kernel` returns early when the object already exists
(`utils.py:591`), and already re-applies the prefix rename there for exactly this
reason. The C++-linkage resolution must be applied on that path too. It is
idempotent: on a cached object the plain name is already present, so step 2
returns immediately.

## Scope

**In the first cut:** `ExternalFunction`, Peano toolchain, object (non-inline)
mode. This covers the `reduce_min` example and every kernel produced by the
`iron.kernels.*` factories.

**Explicitly excluded, each with a loud error rather than a silent failure:**

- `inline=True` / `link_with_mode="merge"` — the artifact is textual `.ll` fed to
  `llvm-link`; no linker and no objcopy are involved, so neither mechanism
  applies. Symbol fixup there means rewriting the IR module, which is a separate
  design. Raise `NotImplementedError` when a `.ll`/`.bc` artifact has no
  matching plain symbol.
- `use_chess=True` — xchesscc's object format has not been tested with
  `llvm-objcopy`. Whether it works is a question to answer by running it, not by
  reasoning about it. Until then, raise when the plain symbol is missing.

**Out of scope for this cut:** prebuilt `Kernel("foo.o")`. Those objects belong
to the user and should not be mutated in place; covering them means either
copying into `kernel_dir` first or passing `--defsym` through aiecc's link. Both
are reasonable follow-ups but neither is needed for the example.

**Deliberate non-goal:** checking `arg_types` against the C++ signature. Binding
by base name means a mismatch links silently — which is exactly the status quo
under `extern "C"`, so this is not a regression. A later change can demangle the
matched symbol *fully* and compare its parameter list loosely (element types and
pointer-ness, ignoring `const`/`restrict`); that would be strictly better
diagnostics than today, but it is not required to remove the trampolines.

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
this design preserves.
