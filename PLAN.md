# Plan: Replace Python aiecc.py with C++ aiecc Bindings

## Context

The Python `aiecc.py` compiler driver (~2,700 lines) orchestrates AIE compilation but has performance overhead from Python startup and subprocess calls to `aie-opt`/`aie-translate`. The C++ `aiecc` implementation (~3,300 lines) now has near feature parity and uses direct in-memory MLIR APIs, eliminating these subprocess calls.

**Goal:** Deprecate the Python implementation by making `aiecc.py` a thin wrapper that delegates to the C++ `aiecc` binary, similar to how `aie-opt` and `aie-translate` are wrapped in `python/tools/__init__.py`.

## Current State

### Python aiecc.py (`python/compiler/aiecc/main.py`)
- **Lines:** 2,164 (main.py) + 469 (cl_arguments.py) = ~2,633 lines
- **Architecture:** `FlowRunner` class with asyncio-based parallel compilation
- **Features:** 60+ command-line options, host compilation, parallel cores, repeater scripts
- **Entry point:** `aie.compiler.aiecc.main:main` registered in pyproject.toml

### C++ aiecc (`tools/aiecc/aiecc.cpp`)
- **Lines:** ~3,300 lines
- **Architecture:** Sequential compilation with in-memory MLIR passes
- **Features:** All core compilation features, transaction/control packet generation
- **Performance:** Zero subprocess calls for MLIR passes

### Feature Comparison

| Feature | Python | C++ | Notes |
|---------|--------|-----|-------|
| Core compilation (Peano) | ✅ | ✅ | |
| Core compilation (xchesscc) | ✅ | ✅ | |
| NPU instruction generation | ✅ | ✅ | |
| CDO/PDI/xclbin generation | ✅ | ✅ | |
| Transaction generation | ✅ | ✅ | |
| Control packet generation | ✅ | ✅ | |
| Full ELF generation | ✅ | ✅ | |
| xclbin extension | ✅ | ✅ | |
| Multi-device support | ✅ | ✅ | |
| **Parallel compilation (-j)** | ✅ | ❌ | Python uses asyncio.gather() |
| **Host compilation** | ✅ | ❌ | --compile-host, --host-target |
| **Unified compilation** | ✅ | ❌ | --unified flag |
| **Repeater scripts** | ✅ | ❌ | Debug script generation on failure |
| **Progress bar** | ✅ | ❌ | Rich progress display |
| **Profiling** | ✅ | ❌ | --profile timing |
| **HSA linking** | ✅ | ❌ | --link_against_hsa |

## Approach Options

### Option A: Thin Subprocess Wrapper (Recommended)

Make `aiecc.py` call the C++ `aiecc` binary via subprocess, similar to `aie-opt`:

```python
# python/compiler/aiecc/main.py (new implementation)
import os
import subprocess
import sys

def main():
    aiecc_bin = os.path.join(os.path.dirname(__file__), "..", "..", "..", "bin", "aiecc")
    sys.exit(subprocess.call([aiecc_bin, *sys.argv[1:]]))
```

**Pros:**
- Minimal code changes
- Immediate performance gains
- Backwards compatible CLI
- Easy to maintain

**Cons:**
- Features missing in C++ won't work until ported
- No programmatic API (breaks `run()` function usage)

### Option B: Shared Library with Python Bindings

Create `libaiecc.so` with nanobind bindings for in-process calls:

```python
# Python can call C++ directly
from aie._aiecc import compile_module
result = compile_module(mlir_string, options)
```

**Pros:**
- Preserves programmatic API
- Best performance (no subprocess)
- Full integration with Python workflows

**Cons:**
- Significant C++ refactoring needed
- Complex build system changes
- Harder to maintain two interfaces

### Option C: Hybrid Approach

Keep Python for orchestration (parallel, host compilation), delegate core work to C++:

```python
# Python handles parallel scheduling
async def compile_cores():
    tasks = [subprocess.run(['aiecc', '--compile-only', core_mlir]) for core in cores]
    await asyncio.gather(*tasks)
```

**Pros:**
- Preserves parallel compilation
- Incremental migration
- Keeps Python's async benefits

**Cons:**
- Two codebases to maintain
- Complex interaction between Python and C++

## Recommended Implementation Plan

### Phase 1: Feature Parity (Prerequisites)

Before deprecating Python, add missing features to C++ aiecc:

1. **Parallel core compilation** (HIGH PRIORITY)
   - Add `--nthreads` / `-j` option
   - Use C++ `std::thread` or `ThreadPool` for parallel core compilation
   - ~200-300 lines in `aiecc.cpp`

2. **Host compilation** (MEDIUM PRIORITY)
   - Add `--compile-host`, `--no-compile-host`, `--host-target` options
   - Invoke host compiler (clang) with AIE library includes
   - ~150-200 lines in `aiecc.cpp`

3. **Unified compilation** (LOW PRIORITY)
   - Add `--unified` option to compile all cores together
   - ~50-100 lines in `aiecc.cpp`

### Phase 2: Create Wrapper

**File:** `python/compiler/aiecc/main.py`

Replace the 2,164-line implementation with a thin wrapper:

```python
#!/usr/bin/env python3
"""
aiecc.py - AIE Compiler Driver (Python wrapper)

This is a thin wrapper that delegates to the C++ aiecc binary.
The C++ implementation provides better performance through
in-memory MLIR pass execution.
"""

import os
import subprocess
import sys
from pathlib import Path

def _find_aiecc_binary():
    """Find the C++ aiecc binary."""
    # Check relative to this file (installed location)
    bin_dir = Path(__file__).parent.parent.parent.parent / "bin"
    aiecc_path = bin_dir / "aiecc"
    if aiecc_path.exists():
        return str(aiecc_path)

    # Check PATH
    import shutil
    path = shutil.which("aiecc")
    if path:
        return path

    raise FileNotFoundError(
        "Could not find 'aiecc' binary. Ensure mlir-aie is properly installed."
    )

def main():
    """Main entry point - delegates to C++ aiecc."""
    try:
        aiecc_bin = _find_aiecc_binary()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Pass all arguments directly to C++ binary
    result = subprocess.run([aiecc_bin, *sys.argv[1:]])
    sys.exit(result.returncode)

def run(mlir_module_str, args=None):
    """
    Programmatic API for compiling MLIR modules.

    DEPRECATED: This function is deprecated. Use the C++ aiecc binary
    directly or the IRON Python API instead.
    """
    import warnings
    warnings.warn(
        "aiecc.run() is deprecated. Use the C++ aiecc binary or IRON API.",
        DeprecationWarning,
        stacklevel=2
    )

    import tempfile

    # Write MLIR to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_module_str)
        mlir_path = f.name

    try:
        aiecc_bin = _find_aiecc_binary()
        cmd = [aiecc_bin, mlir_path]
        if args:
            cmd.extend(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"aiecc failed: {result.stderr}")
        return result.stdout
    finally:
        os.unlink(mlir_path)

if __name__ == "__main__":
    main()
```

### Phase 3: Update Entry Points

**File:** `utils/mlir_aie_wheels/pyproject.toml`

No changes needed - entry point still points to `aie.compiler.aiecc.main:main`

### Phase 4: Archive Old Implementation

Move the full Python implementation to an archive for reference:

```bash
git mv python/compiler/aiecc/main.py python/compiler/aiecc/_main_legacy.py
# Create new thin wrapper as main.py
```

### Phase 5: Update Documentation

Update `tools/aiecc/README.md` to:
- Note that `aiecc.py` now delegates to `aiecc` binary
- Document deprecation of `run()` API
- Point users to C++ binary for direct usage

## Files to Modify

| File | Action |
|------|--------|
| `tools/aiecc/aiecc.cpp` | Add parallel compilation, host compilation |
| `python/compiler/aiecc/main.py` | Replace with thin wrapper (~80 lines) |
| `python/compiler/aiecc/_main_legacy.py` | Archive old implementation |
| `python/compiler/aiecc/cl_arguments.py` | Can be deleted (C++ has own arg parsing) |
| `tools/aiecc/README.md` | Update documentation |

## Verification

### Test Wrapper Works
```bash
# Verify aiecc.py delegates to C++ binary
aiecc.py --verbose test/aiecc/cpp_basic.mlir 2>&1 | grep "aiecc (C++ version)"
```

### Test Feature Parity
```bash
# Run existing aiecc tests
cd build && ninja check-aie

# Run specific aiecc tests
lit test/aiecc/ -v
```

### Test Programmatic API (if preserved)
```python
from aie.compiler.aiecc import run
result = run(mlir_string, ['--aie-generate-npu-insts'])
```

### Test Programming Examples
```bash
cd programming_examples/basic/passthrough_kernel
make clean && make  # Uses aiecc.py via Makefile
```

## Timeline Estimate

| Phase | Description | Effort |
|-------|-------------|--------|
| Phase 1a | Add parallel compilation to C++ | 2-3 days |
| Phase 1b | Add host compilation to C++ | 1-2 days |
| Phase 2 | Create thin wrapper | 0.5 day |
| Phase 3 | Update entry points | 0.5 day |
| Phase 4 | Archive old implementation | 0.5 day |
| Phase 5 | Update documentation | 0.5 day |
| **Total** | | **5-7 days** |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Feature gaps in C++ break user workflows | Complete Phase 1 before Phase 2 |
| Programmatic API users impacted | Provide deprecation warning with migration path |
| Build system complexity | Keep wrapper simple, test thoroughly |
| Performance regression | C++ is faster; monitor in CI |

## Success Criteria

1. `aiecc.py` invocation produces identical output to direct `aiecc` invocation
2. All `test/aiecc/*.mlir` tests pass
3. All `programming_examples/` that use aiecc.py work correctly
4. No performance regression (expect improvement)
5. Documentation updated with migration guide
