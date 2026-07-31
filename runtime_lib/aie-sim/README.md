# aie-sim

An open simulator for the AIE array. Design, rationale and phasing are in
[docs/AIESimulator.md](../../docs/AIESimulator.md); this file is just orientation.

## What it is

A software model of the AIE2/AIE2P array that defines the seven `ess_*` symbols aie-rt's
`XAIE_IO_BACKEND_SIM` calls. Link a host program against it instead of against the Vitis simulator's
`ps.so`, and the program simulates itself: no `aiesimulator` process, no SystemC, no licence.

The model is driven entirely by the register writes the host makes through aie-rt, so it simulates what
the configuration code actually programmed rather than what the compiler described.

## Building and testing on its own

It depends on a C++17 compiler and the register definitions vendored in `third_party/aie-rt`. Nothing
else: no MLIR, no LLVM, no Vitis, no NPU.

```
cmake -S runtime_lib/aie-sim -B build-sim -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sim -j
ctest --test-dir build-sim --output-on-failure
```

In a full mlir-aie build it is picked up from the top-level `CMakeLists.txt` and its tests run under
`check-aie-sim`.

## Layout

| path | what lives there |
| --- | --- |
| `include/aiesim/Device.h` | array geometry and address decode |
| `include/aiesim/Array.h` | tiles, memories, the register file, DDR, the clock |
| `include/aiesim/Components.h` | the lock, stream-switch and DMA interfaces |
| `include/aiesim/CoreEngine.h` | how an instruction simulator plugs in |
| `include/aiesim/aie_iss_c_abi.h` | the versioned C ABI that engine is loaded through |
| `lib/EssAbi.cpp` | the seven `ess_*` entry points, and the whole external surface |

## Two rules worth knowing before changing anything

**Never guess.** An unmapped access or an unmodelled behaviour must produce a named failure through
`Array::error`, not a plausible default. A simulator that silently invents an answer is worse than no
simulator, because it converts a missing feature into a wrong result.

**Stay deterministic.** One thread, one cycle counter, components stepped in a fixed registration
order. The array advances inside the `ess_*` entry points, which is what lets a host polling loop make
progress without threads. Same input, same cycle counts, every run.
