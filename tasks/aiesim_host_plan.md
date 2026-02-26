# PR Plan: aiesim and Host Compilation Support for C++ aiecc

## Overview

This PR adds AIE simulation (`--aiesim`) and basic host compilation (`--compile-host`) support to the C++ aiecc compiler driver. These features are intertwined because:
1. Both require generating `aie_inc.cpp` via `aie-translate --aie-generate-xaie`
2. aiesim builds ps.so which uses host compilation infrastructure
3. aiesim requires xbridge linking (constraint from Python implementation)

## Scope

**In scope:**
- `--aiesim` / `--no-aiesim` - Generate aiesim Work folder
- `--compile-host` / `--no-compile-host` - Basic host compilation
- `--host-target` - Target architecture for host (needed for runtime_lib paths)

**Deferred to future PR:**
- `--link_against_hsa` - ROCm HSA runtime linking (adds significant complexity)
- `--no-materialize` - Pass control flag (unrelated to aiesim)
- `--sysroot` - Cross-compilation support (edge case)

## Base Branch

- Base PR: #2883 (`aiecc-elf-insts-support`)
- New branch: `aiecc-aiesim-host-support`

## Features to Implement

### 1. Command-line Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--aiesim` | bool | false | Generate aiesim Work folder |
| `--no-aiesim` | bool | - | Do not generate aiesim Work folder |
| `--compile-host` | bool | false | Enable compiling of the host program |
| `--no-compile-host` | bool | - | Disable compiling of the host program |
| `--host-target` | string | "x86_64-linux-gnu" | Target architecture of the host program |

### 2. aie_inc.cpp Generation

Both `--compile-host` and `--aiesim` require generating `aie_inc.cpp`:
```cpp
// When compile_host or aiesim is enabled
if (compileHost || aiesim) {
  // Generate aie_inc.cpp via aie-translate --aie-generate-xaie
  executeCommand({"aie-translate", "--aie-generate-xaie",
                  "--aie-device-name", deviceName,
                  inputPhysicalWithElfs, "-o", aieIncCpp});
}
```

### 3. Host Compilation (`--compile-host`)

Basic implementation from `process_host_cgen()` in main.py:

1. Build clang++ command with:
   - `--target=<host_target>`
   - Include paths: xaiengine, tmpdir
   - Library paths: xaiengine
   - Memory allocator: `libmemory_allocator_ion.a`
   - Library: `-lxaienginecdo`
   - AIE target defines

2. Compile user's host files (passed as remaining arguments)

### 4. AIE Simulation (`--aiesim`)

From `gen_sim()` in main.py (lines 1538-1737):

1. **Validation**: aiesim requires xbridge (`--aiesim` without `--xbridge` → error)

2. **Create sim directory structure**:
   ```
   <tmpdir>/sim/
   ├── arch/
   │   └── aieshim_solution.aiesol
   ├── reports/
   │   └── graph.xpe
   ├── config/
   │   └── scsim_config.json
   ├── ps/
   │   └── ps.so
   ├── flows_physical.mlir
   ├── flows_physical.json
   └── .target
   ```

3. **Generate simulation files** (via aie-translate):
   - `--aie-mlir-to-xpe` → graph.xpe
   - `--aie-mlir-to-shim-solution` → aieshim_solution.aiesol
   - `--aie-mlir-to-scsim-config` → scsim_config.json

4. **Run flow analysis**:
   - Run `aie.device(aie-find-flows)` pass on physical module
   - Generate flows_physical.mlir

5. **Generate flows JSON**:
   - `aie-translate --aie-flows-to-json` → flows_physical.json

6. **Build ps.so** (process server shared object):
   - Use clang++ to compile `genwrapper_for_ps.cpp`
   - Compile flags:
     ```
     -fPIC -flto -fpermissive -O2 -shared
     -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR
     -Wno-deprecated-declarations
     -Wno-enum-constexpr-conversion
     -Wno-format-security
     -DSC_INCLUDE_DYNAMIC_PROCESSES
     -D__AIESIM__
     -D__PS_INIT_AIE__
     -Dmain(...)=ps_main(...)
     ```
   - Include paths:
     - `<tmpdir>` (for aie_inc.cpp)
     - `<aietools>/include`
     - `<xaiengine>/include`
     - `<aietools>/data/osci_systemc/include`
     - `<aietools>/include/xtlm/include`
     - `<aietools>/include/common_cpp/common_cpp_v1_0/include`
     - `<runtime_lib>/<arch>/test_lib/include`
   - Link against:
     - `libmemory_allocator_sim_aie.a`
     - `-lxaienginecdo`
     - `-lsystemc`
     - `-lxtlm`
   - Library paths:
     - `<xaiengine>/lib`
     - `<aietools>/lib/lnx64.o`
     - `<aietools>/lib/lnx64.o/Ubuntu`
     - `<aietools>/data/osci_systemc/lib/lnx64`

7. **Generate aiesim.sh script**:
   ```bash
   #!/bin/sh
   prj_name=$(basename $(dirname $(realpath $0)))
   root=$(dirname $(dirname $(realpath $0)))
   vcd_filename=foo
   if [ -n "$1" ]; then
     vcd_filename=$1
   fi
   cd $root
   aiesimulator --pkg-dir=${prj_name}/sim --dump-vcd ${vcd_filename}
   ```

### 5. Path Discovery

Need to discover:
- `install_path()` - mlir-aie installation directory (relative to aiecc binary)
- `runtime_lib/<arch>/xaiengine/` - xaiengine includes/libs
- `runtime_lib/<arch>/test_lib/` - test library and memory allocator
- `aie_runtime_lib/<target>/aiesim/` - simulation runtime files (genwrapper_for_ps.cpp)
- `aietools` - already discovered for xchesscc

## Implementation Steps

- [ ] **1. Add command-line flags**
  - Add `--aiesim`, `--no-aiesim`
  - Add `--compile-host`, `--no-compile-host`
  - Add `--host-target` with default "x86_64-linux-gnu"

- [ ] **2. Add path discovery functions**
  - `discoverInstallPath()` - Find mlir-aie install directory (relative to aiecc binary)
  - Helpers for xaiengine, test_lib, aiesim runtime paths

- [ ] **3. Implement `aie_inc.cpp` generation**
  - Add check: if `compileHost || aiesim`
  - Call aie-translate with `--aie-generate-xaie`

- [ ] **4. Implement basic host compilation**
  - Add `compileHost()` function
  - Build clang++ command with correct flags
  - Use ion memory allocator (no HSA)
  - Pass remaining command-line args to clang++

- [ ] **5. Implement aiesim validation**
  - Check: aiesim requires xbridge, error if not

- [ ] **6. Implement aiesim directory structure**
  - Create sim/, sim/arch/, sim/reports/, sim/config/, sim/ps/
  - Create .target file with "hw\n"

- [ ] **7. Implement aie-translate calls for sim files**
  - `--aie-mlir-to-xpe` → graph.xpe
  - `--aie-mlir-to-shim-solution` → aieshim_solution.aiesol
  - `--aie-mlir-to-scsim-config` → scsim_config.json

- [ ] **8. Implement flow analysis**
  - Run `aie.device(aie-find-flows)` pass via PassManager
  - Write flows_physical.mlir
  - Generate flows_physical.json via aie-translate

- [ ] **9. Implement ps.so compilation**
  - Find genwrapper_for_ps.cpp in aie_runtime_lib
  - Build clang++ command with all sim flags
  - Compile to ps.so

- [ ] **10. Generate aiesim.sh script**
  - Write script content
  - Make executable (chmod +x)

- [ ] **11. Integrate into main flow**
  - Add aie_inc.cpp generation after core compilation
  - Add host compilation step
  - Add aiesim generation step

- [ ] **12. Update README.md**
  - Document new flags
  - Update feature comparison table

- [ ] **13. Add LIT tests**
  - `test/aiecc/cpp_aiesim.mlir` - Test aiesim (dry-run mode with -n)
  - `test/aiecc/cpp_host_compile.mlir` - Test host compilation flags

- [ ] **14. Format and verify**
  - Run `git clang-format origin/main`
  - Build and test locally

## Files to Modify

- `tools/aiecc/aiecc.cpp` - Main implementation
- `tools/aiecc/README.md` - Documentation updates

## New Test Files

- `test/aiecc/cpp_aiesim.mlir`
- `test/aiecc/cpp_host_compile.mlir`

## Complexity Assessment

- **Host compilation**: Low-Medium - straightforward clang++ invocation
- **aiesim**: Medium-High - multiple aie-translate calls, pass execution, ps.so compilation, script generation

## Reference Files

- Python implementation: `python/compiler/aiecc/main.py`
  - `process_host_cgen()`: lines 1417-1536
  - `gen_sim()`: lines 1538-1737
- Command-line args: `python/compiler/aiecc/cl_arguments.py`
- Configuration: `python/compiler/aiecc/configure.py`
