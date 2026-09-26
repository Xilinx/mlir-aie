window.BENCHMARK_DATA = {
  "lastUpdate": 1790404098319,
  "repoUrl": "https://github.com/Xilinx/mlir-aie",
  "entries": {
    "aie_kernels (npu1, default)": [
      {
        "commit": {
          "author": {
            "name": "Erika Hunhoff",
            "username": "hunhoffe",
            "email": "erika.hunhoff@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "e8d062f9dc8c053c38e022eedf52abc760db6761",
          "message": "Kernel benchmarks: fix publish on dispatch and schedule runs (#3803)\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-09-25T00:24:37Z",
          "url": "https://github.com/Xilinx/mlir-aie/commit/e8d062f9dc8c053c38e022eedf52abc760db6761"
        },
        "date": 1790300317122,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 264,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 128.906,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 195,
            "range": "min 182.4 max 310.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/e2e_us",
            "value": 348.2,
            "range": "min 331.8 max 462.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/compile_s",
            "value": 1.57,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 264,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 128.906,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 1442.45,
            "range": "min 766.7 max 1745.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/e2e_us",
            "value": 1915.48,
            "range": "min 1204.3 max 2218.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/compile_s",
            "value": 1.54,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 264,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 64.453,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 193.44,
            "range": "min 181.7 max 304.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/e2e_us",
            "value": 343.54,
            "range": "min 326.8 max 459.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 136,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 33.203,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 175.15,
            "range": "min 158.0 max 354.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/e2e_us",
            "value": 324.08,
            "range": "min 303.1 max 498.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/compile_s",
            "value": 1.54,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 162.75,
            "range": "min 149.4 max 347.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/e2e_us",
            "value": 320.16,
            "range": "min 296.2 max 1056.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 3048,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 291.46,
            "range": "min 276.1 max 371.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/e2e_us",
            "value": 454.56,
            "range": "min 432.3 max 529.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3048,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 175.6,
            "range": "min 159.5 max 323.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/e2e_us",
            "value": 329.94,
            "range": "min 314.2 max 472.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/compile_s",
            "value": 1.59,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9079,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 3208,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 186.98,
            "range": "min 167.6 max 260.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/e2e_us",
            "value": 330.07,
            "range": "min 305.4 max 412.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/compile_s",
            "value": 1.72,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 8951,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3052,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 1261.93,
            "range": "min 647.1 max 1528.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/e2e_us",
            "value": 1783.78,
            "range": "min 1091.4 max 2245.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/compile_s",
            "value": 1.72,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 8951,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3052,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 179.59,
            "range": "min 167.6 max 347.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/e2e_us",
            "value": 320.71,
            "range": "min 301.4 max 483.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/compile_s",
            "value": 1.74,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 603.01,
            "range": "min 566.6 max 1217.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/e2e_us",
            "value": 931.39,
            "range": "min 815.8 max 1956.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/compile_s",
            "value": 1.74,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 157.51,
            "range": "min 151.9 max 261.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/e2e_us",
            "value": 301.6,
            "range": "min 290.2 max 399.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 666.53,
            "range": "min 319.6 max 1536.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/e2e_us",
            "value": 1309.42,
            "range": "min 572.5 max 2133.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/compile_s",
            "value": 1.6,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 176.28,
            "range": "min 158.1 max 332.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/e2e_us",
            "value": 324.25,
            "range": "min 301.4 max 473.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 3092,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 432.54,
            "range": "min 414.7 max 664.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/e2e_us",
            "value": 678.81,
            "range": "min 612.9 max 1216.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 3092,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 204.36,
            "range": "min 164.4 max 332.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/e2e_us",
            "value": 380.85,
            "range": "min 320.7 max 764.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/compile_s",
            "value": 1.56,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 3108,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 433.74,
            "range": "min 420.2 max 444.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/e2e_us",
            "value": 683.23,
            "range": "min 667.3 max 700.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/compile_s",
            "value": 1.56,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 3108,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 177.96,
            "range": "min 157.8 max 259.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/e2e_us",
            "value": 323.44,
            "range": "min 301.2 max 403.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 3140,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 471.16,
            "range": "min 425.7 max 1277.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/e2e_us",
            "value": 782.68,
            "range": "min 663.5 max 1869.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 3140,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 166.23,
            "range": "min 151.2 max 254.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/e2e_us",
            "value": 312.63,
            "range": "min 290.0 max 664.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 283.04,
            "range": "min 272.4 max 387.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/e2e_us",
            "value": 443.96,
            "range": "min 429.9 max 601.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 291.27,
            "range": "min 252.5 max 474.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/e2e_us",
            "value": 520.64,
            "range": "min 461.5 max 760.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/compile_s",
            "value": 4.1,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 14649,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 9244,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 2224.43,
            "range": "min 1775.3 max 2714.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/e2e_us",
            "value": 2918.46,
            "range": "min 2353.3 max 3415.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/compile_s",
            "value": 4.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 14649,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 9244,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 248.36,
            "range": "min 225.1 max 381.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/e2e_us",
            "value": 475.02,
            "range": "min 435.1 max 1030.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/compile_s",
            "value": 4.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 14441,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 9020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 1559.73,
            "range": "min 1292.6 max 1829.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/e2e_us",
            "value": 1999.83,
            "range": "min 1539.8 max 2776.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/compile_s",
            "value": 4.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 14441,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 9020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 217.97,
            "range": "min 205.5 max 304.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/e2e_us",
            "value": 364.95,
            "range": "min 347.0 max 544.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/compile_s",
            "value": 3.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 14073,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 8632,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 1487.26,
            "range": "min 1205.1 max 1779.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/e2e_us",
            "value": 1970.48,
            "range": "min 1686.3 max 2656.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/compile_s",
            "value": 3.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 14073,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 8632,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 196.05,
            "range": "min 186.5 max 310.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/e2e_us",
            "value": 346.21,
            "range": "min 333.4 max 518.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/compile_s",
            "value": 3.44,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 14249,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 8952,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 1389.91,
            "range": "min 1253.8 max 1696.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/e2e_us",
            "value": 2023.11,
            "range": "min 1696.2 max 2485.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/compile_s",
            "value": 3.43,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 14249,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 8952,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 225.23,
            "range": "min 213.9 max 329.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/e2e_us",
            "value": 451.79,
            "range": "min 423.2 max 546.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/compile_s",
            "value": 3.42,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 14377,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 9088,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 1547.34,
            "range": "min 1077.0 max 1852.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/e2e_us",
            "value": 1996.5,
            "range": "min 1487.1 max 2287.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/compile_s",
            "value": 3.44,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 14377,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 9088,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 265.86,
            "range": "min 241.0 max 359.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/e2e_us",
            "value": 490.13,
            "range": "min 396.9 max 581.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/compile_s",
            "value": 3.29,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 16057,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 11628,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 2166.13,
            "range": "min 1823.0 max 2685.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/e2e_us",
            "value": 2880.73,
            "range": "min 2302.4 max 3508.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/compile_s",
            "value": 3.29,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 16057,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 11628,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 160.75,
            "range": "min 153.8 max 252.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/e2e_us",
            "value": 308.26,
            "range": "min 297.3 max 398.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 2972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 317.09,
            "range": "min 278.2 max 709.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/e2e_us",
            "value": 532.81,
            "range": "min 438.7 max 1150.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 166.69,
            "range": "min 161.0 max 259.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/e2e_us",
            "value": 316.46,
            "range": "min 301.9 max 401.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/compile_s",
            "value": 1.77,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 494.54,
            "range": "min 457.0 max 1456.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/e2e_us",
            "value": 745.97,
            "range": "min 700.8 max 1906.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/compile_s",
            "value": 1.78,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 180.06,
            "range": "min 162.7 max 256.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/e2e_us",
            "value": 326.72,
            "range": "min 310.1 max 401.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 2984,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 450,
            "range": "min 436.5 max 1370.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/e2e_us",
            "value": 682.29,
            "range": "min 611.3 max 2158.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/compile_s",
            "value": 1.64,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 2984,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 163.16,
            "range": "min 149.9 max 251.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/e2e_us",
            "value": 314.18,
            "range": "min 295.8 max 400.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 9111,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 3216,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 174.28,
            "range": "min 155.6 max 359.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/e2e_us",
            "value": 318.72,
            "range": "min 291.4 max 566.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 9352,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 3492,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 160.06,
            "range": "min 146.2 max 257.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/e2e_us",
            "value": 312.55,
            "range": "min 298.9 max 407.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 9111,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 3140,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 183.54,
            "range": "min 167.2 max 364.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/e2e_us",
            "value": 345,
            "range": "min 312.8 max 833.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/compile_s",
            "value": 1.76,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 3524,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 247.13,
            "range": "min 236.0 max 294.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/e2e_us",
            "value": 409.09,
            "range": "min 399.5 max 453.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/compile_s",
            "value": 1.77,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1922.06,
            "range": "min 1612.9 max 2534.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/e2e_us",
            "value": 2864.35,
            "range": "min 2126.0 max 3585.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/compile_s",
            "value": 1.77,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 252.22,
            "range": "min 237.8 max 408.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/e2e_us",
            "value": 412.04,
            "range": "min 390.4 max 566.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 9912,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 231.32,
            "range": "min 211.5 max 258.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/e2e_us",
            "value": 385.35,
            "range": "min 369.4 max 473.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/compile_s",
            "value": 1.79,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 10329,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 4716,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 166.27,
            "range": "min 150.6 max 269.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/e2e_us",
            "value": 320.02,
            "range": "min 299.4 max 563.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/compile_s",
            "value": 3.17,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 10505,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 4688,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 166.54,
            "range": "min 151.7 max 246.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/e2e_us",
            "value": 320.95,
            "range": "min 310.6 max 404.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/compile_s",
            "value": 1.78,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9608,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3768,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 147.68,
            "range": "min 135.3 max 161.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/e2e_us",
            "value": 291.18,
            "range": "min 283.6 max 311.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/compile_s",
            "value": 1.58,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2856,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 152.92,
            "range": "min 142.8 max 185.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/e2e_us",
            "value": 299.63,
            "range": "min 284.7 max 325.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/compile_s",
            "value": 1.6,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2880,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 269.57,
            "range": "min 256.6 max 396.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/e2e_us",
            "value": 479.84,
            "range": "min 457.9 max 602.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/compile_s",
            "value": 3.84,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 14889,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 10196,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 2270.13,
            "range": "min 1882.5 max 2702.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/e2e_us",
            "value": 2987.64,
            "range": "min 2414.7 max 3642.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/compile_s",
            "value": 3.84,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 14889,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 10196,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 191.8,
            "range": "min 175.2 max 237.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/e2e_us",
            "value": 340.49,
            "range": "min 324.3 max 444.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/compile_s",
            "value": 1.57,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 2948,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 190.58,
            "range": "min 173.9 max 270.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/e2e_us",
            "value": 335.61,
            "range": "min 316.6 max 410.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 3136,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 166.29,
            "range": "min 150.6 max 256.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/e2e_us",
            "value": 313.17,
            "range": "min 294.9 max 400.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 9816,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 4804,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 177.37,
            "range": "min 163.1 max 365.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/e2e_us",
            "value": 326.3,
            "range": "min 308.3 max 633.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/compile_s",
            "value": 1.57,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3100,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 171.83,
            "range": "min 156.6 max 288.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/e2e_us",
            "value": 318.8,
            "range": "min 300.0 max 430.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/compile_s",
            "value": 1.56,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3100,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 188.59,
            "range": "min 172.2 max 325.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/e2e_us",
            "value": 338.01,
            "range": "min 321.3 max 781.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3408,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 202.31,
            "range": "min 184.5 max 414.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/e2e_us",
            "value": 359.8,
            "range": "min 338.9 max 579.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 9608,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 4572,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 227.43,
            "range": "min 213.4 max 243.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/e2e_us",
            "value": 381.99,
            "range": "min 364.4 max 401.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/compile_s",
            "value": 3.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 11145,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 5748,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 264.37,
            "range": "min 239.4 max 398.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/e2e_us",
            "value": 479.5,
            "range": "min 395.3 max 895.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/compile_s",
            "value": 1.76,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13785,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 3964,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 285.85,
            "range": "min 228.4 max 1573.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/e2e_us",
            "value": 481.73,
            "range": "min 422.7 max 2068.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13785,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 3968,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 337.07,
            "range": "min 306.6 max 395.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/e2e_us",
            "value": 551.03,
            "range": "min 475.5 max 625.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/compile_s",
            "value": 1.94,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 18489,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 5428,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 302.77,
            "range": "min 279.9 max 394.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/e2e_us",
            "value": 557.41,
            "range": "min 452.0 max 658.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/compile_s",
            "value": 1.89,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 18617,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 8060,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 298.09,
            "range": "min 277.5 max 426.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/e2e_us",
            "value": 475.99,
            "range": "min 452.3 max 656.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/compile_s",
            "value": 1.89,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 18665,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 7196,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 239.52,
            "range": "min 219.1 max 306.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/e2e_us",
            "value": 404.21,
            "range": "min 384.2 max 474.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13785,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 3964,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 1433.75,
            "range": "min 1149.4 max 1748.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/e2e_us",
            "value": 2029.95,
            "range": "min 1721.1 max 2615.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/compile_s",
            "value": 1.82,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 48633,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 7704,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 1324.83,
            "range": "min 522.2 max 1727.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/e2e_us",
            "value": 1909.38,
            "range": "min 887.0 max 2476.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 47945,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 6944,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 188.82,
            "range": "min 167.8 max 366.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/e2e_us",
            "value": 341.56,
            "range": "min 314.2 max 753.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3548,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 188.72,
            "range": "min 174.8 max 269.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/e2e_us",
            "value": 333.86,
            "range": "min 311.8 max 763.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3548,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Erika Hunhoff",
            "username": "hunhoffe",
            "email": "erika.hunhoff@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "e8d062f9dc8c053c38e022eedf52abc760db6761",
          "message": "Kernel benchmarks: fix publish on dispatch and schedule runs (#3803)\n\nCo-authored-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-09-25T00:24:37Z",
          "url": "https://github.com/Xilinx/mlir-aie/commit/e8d062f9dc8c053c38e022eedf52abc760db6761"
        },
        "date": 1790318183176,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 264,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 128.906,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 193.93,
            "range": "min 182.0 max 353.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/e2e_us",
            "value": 344.95,
            "range": "min 329.8 max 510.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/compile_s",
            "value": 1.58,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 264,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 128.906,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 737.09,
            "range": "min 722.6 max 1646.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/e2e_us",
            "value": 1103.98,
            "range": "min 1088.5 max 2414.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 264,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 64.453,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 198.67,
            "range": "min 179.3 max 305.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/e2e_us",
            "value": 354.82,
            "range": "min 322.6 max 619.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 136,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 33.203,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 176.48,
            "range": "min 156.2 max 261.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/e2e_us",
            "value": 329.24,
            "range": "min 313.8 max 414.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 166.51,
            "range": "min 150.7 max 250.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/e2e_us",
            "value": 316.04,
            "range": "min 297.9 max 650.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 3048,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 312.98,
            "range": "min 277.5 max 485.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/e2e_us",
            "value": 486.17,
            "range": "min 437.3 max 989.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3048,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 235.98,
            "range": "min 167.8 max 349.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/e2e_us",
            "value": 455.38,
            "range": "min 318.0 max 674.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/compile_s",
            "value": 1.59,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9079,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 3208,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 186.92,
            "range": "min 174.9 max 224.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/e2e_us",
            "value": 329.64,
            "range": "min 315.5 max 381.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/compile_s",
            "value": 1.73,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 8951,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3052,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 1175.1,
            "range": "min 711.3 max 1602.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/e2e_us",
            "value": 1759.93,
            "range": "min 1134.0 max 2412.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/compile_s",
            "value": 1.73,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 8951,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3052,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 184.64,
            "range": "min 170.9 max 201.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/e2e_us",
            "value": 335.02,
            "range": "min 311.2 max 354.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/compile_s",
            "value": 1.74,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 1363.51,
            "range": "min 633.8 max 1755.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/e2e_us",
            "value": 1882.25,
            "range": "min 1051.1 max 2234.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/compile_s",
            "value": 1.74,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 156.62,
            "range": "min 142.1 max 258.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/e2e_us",
            "value": 299.05,
            "range": "min 286.4 max 401.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 318.3,
            "range": "min 275.6 max 1852.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/e2e_us",
            "value": 526.35,
            "range": "min 435.9 max 2309.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 175.49,
            "range": "min 164.0 max 191.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/e2e_us",
            "value": 322.07,
            "range": "min 313.3 max 347.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 3092,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 432.73,
            "range": "min 417.8 max 543.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/e2e_us",
            "value": 674.1,
            "range": "min 648.1 max 808.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/compile_s",
            "value": 1.56,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 3092,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 173.88,
            "range": "min 161.1 max 320.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/e2e_us",
            "value": 323.42,
            "range": "min 309.3 max 682.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/compile_s",
            "value": 1.56,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 3108,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 427.44,
            "range": "min 419.9 max 602.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/e2e_us",
            "value": 684.78,
            "range": "min 666.6 max 1109.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/compile_s",
            "value": 1.57,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 3108,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 171.22,
            "range": "min 159.5 max 182.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/e2e_us",
            "value": 315.72,
            "range": "min 306.1 max 344.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 3140,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 434.4,
            "range": "min 417.7 max 504.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/e2e_us",
            "value": 682.12,
            "range": "min 660.2 max 750.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 3140,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 160.73,
            "range": "min 147.1 max 261.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/e2e_us",
            "value": 306.25,
            "range": "min 287.9 max 410.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 783.98,
            "range": "min 413.7 max 1512.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/e2e_us",
            "value": 1375.79,
            "range": "min 625.3 max 2251.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 271.84,
            "range": "min 256.7 max 463.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/e2e_us",
            "value": 496.84,
            "range": "min 477.0 max 694.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/compile_s",
            "value": 4.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 14649,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 9244,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 2146.92,
            "range": "min 1857.5 max 2682.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/e2e_us",
            "value": 2928.23,
            "range": "min 2422.2 max 3686.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/compile_s",
            "value": 4.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 14649,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 9244,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 275.4,
            "range": "min 225.4 max 379.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/e2e_us",
            "value": 491.22,
            "range": "min 446.3 max 602.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/compile_s",
            "value": 4.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 14441,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 9020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 1537.17,
            "range": "min 1374.0 max 2201.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/e2e_us",
            "value": 2122.8,
            "range": "min 1852.5 max 3071.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/compile_s",
            "value": 4.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 14441,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 9020,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 222.71,
            "range": "min 212.5 max 332.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/e2e_us",
            "value": 443.91,
            "range": "min 389.7 max 545.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/compile_s",
            "value": 3.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 14073,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 8632,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 1477.46,
            "range": "min 1185.6 max 1776.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/e2e_us",
            "value": 1968.17,
            "range": "min 1595.6 max 2674.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/compile_s",
            "value": 3.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 14073,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 8632,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 196.54,
            "range": "min 183.1 max 312.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/e2e_us",
            "value": 366.68,
            "range": "min 318.5 max 469.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/compile_s",
            "value": 3.43,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 14249,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 8952,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 1319.95,
            "range": "min 922.8 max 1605.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/e2e_us",
            "value": 1917.15,
            "range": "min 1368.6 max 2388.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/compile_s",
            "value": 3.47,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 14249,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 8952,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 226.22,
            "range": "min 200.8 max 349.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/e2e_us",
            "value": 448.5,
            "range": "min 360.9 max 576.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/compile_s",
            "value": 3.42,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 14377,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 9088,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 1406.71,
            "range": "min 1107.6 max 1887.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/e2e_us",
            "value": 1994.7,
            "range": "min 1635.3 max 2587.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/compile_s",
            "value": 3.41,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 14377,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 9088,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 286.5,
            "range": "min 241.0 max 432.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/e2e_us",
            "value": 513.14,
            "range": "min 392.0 max 666.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/compile_s",
            "value": 3.29,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 16057,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 11628,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 2016.3,
            "range": "min 1743.1 max 2601.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/e2e_us",
            "value": 2891.76,
            "range": "min 2176.6 max 3417.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/compile_s",
            "value": 3.28,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 16057,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 11628,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 170.17,
            "range": "min 154.1 max 361.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/e2e_us",
            "value": 319.37,
            "range": "min 295.9 max 703.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 2972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 339.89,
            "range": "min 279.4 max 1324.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/e2e_us",
            "value": 542.04,
            "range": "min 438.1 max 1813.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/compile_s",
            "value": 1.76,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 178.26,
            "range": "min 161.9 max 282.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/e2e_us",
            "value": 324.04,
            "range": "min 306.7 max 429.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/compile_s",
            "value": 1.77,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 433.53,
            "range": "min 416.4 max 1296.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/e2e_us",
            "value": 672.64,
            "range": "min 579.3 max 1553.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/compile_s",
            "value": 1.78,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 177.5,
            "range": "min 163.9 max 347.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/e2e_us",
            "value": 324.01,
            "range": "min 306.6 max 495.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 2984,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 1201.02,
            "range": "min 495.8 max 1361.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/e2e_us",
            "value": 1538.49,
            "range": "min 840.5 max 2157.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/compile_s",
            "value": 1.64,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 2984,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 163.58,
            "range": "min 154.9 max 234.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/e2e_us",
            "value": 313.46,
            "range": "min 300.9 max 399.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 9111,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 3216,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 172.97,
            "range": "min 157.8 max 202.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/e2e_us",
            "value": 313.24,
            "range": "min 296.8 max 401.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 9352,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 3492,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 163.22,
            "range": "min 140.6 max 247.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/e2e_us",
            "value": 310.24,
            "range": "min 292.0 max 397.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 9111,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 3140,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 177.91,
            "range": "min 162.3 max 271.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/e2e_us",
            "value": 329.71,
            "range": "min 315.9 max 431.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/compile_s",
            "value": 1.76,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 3524,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 238.95,
            "range": "min 230.0 max 315.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/e2e_us",
            "value": 400.17,
            "range": "min 387.9 max 473.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/compile_s",
            "value": 1.77,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1717.95,
            "range": "min 1604.6 max 2509.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/e2e_us",
            "value": 2674.86,
            "range": "min 2086.2 max 3344.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/compile_s",
            "value": 1.77,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 251.84,
            "range": "min 238.0 max 321.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/e2e_us",
            "value": 409.45,
            "range": "min 388.6 max 497.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 9912,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 228.69,
            "range": "min 214.1 max 646.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/e2e_us",
            "value": 387.64,
            "range": "min 369.9 max 1196.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/compile_s",
            "value": 1.78,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 10329,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 4716,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 184.44,
            "range": "min 149.8 max 230.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/e2e_us",
            "value": 409.86,
            "range": "min 294.0 max 460.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/compile_s",
            "value": 3.16,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 10505,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 4688,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 170.66,
            "range": "min 151.2 max 269.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/e2e_us",
            "value": 350.14,
            "range": "min 298.6 max 512.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/compile_s",
            "value": 1.78,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9608,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3768,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 151.91,
            "range": "min 140.6 max 247.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/e2e_us",
            "value": 306.51,
            "range": "min 288.6 max 399.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/compile_s",
            "value": 1.6,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2856,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 152.88,
            "range": "min 139.1 max 250.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/e2e_us",
            "value": 289.42,
            "range": "min 276.2 max 385.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/compile_s",
            "value": 1.6,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2880,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 294.26,
            "range": "min 243.6 max 462.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/e2e_us",
            "value": 511.31,
            "range": "min 403.1 max 709.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/compile_s",
            "value": 3.84,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 14889,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 10196,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 2129.52,
            "range": "min 1842.8 max 2594.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/e2e_us",
            "value": 2919.31,
            "range": "min 2328.6 max 3625.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/compile_s",
            "value": 3.84,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 14889,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 10196,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 186.77,
            "range": "min 168.8 max 354.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/e2e_us",
            "value": 330.75,
            "range": "min 316.7 max 495.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/compile_s",
            "value": 1.57,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 2948,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 194.87,
            "range": "min 173.8 max 232.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/e2e_us",
            "value": 337.45,
            "range": "min 317.1 max 491.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 3136,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 164.33,
            "range": "min 153.2 max 333.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/e2e_us",
            "value": 314.46,
            "range": "min 294.3 max 493.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 9816,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 4804,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 180.25,
            "range": "min 166.5 max 353.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/e2e_us",
            "value": 332.71,
            "range": "min 301.7 max 668.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/compile_s",
            "value": 1.57,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3100,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 179.06,
            "range": "min 162.8 max 269.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/e2e_us",
            "value": 324.86,
            "range": "min 308.4 max 419.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3100,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 184.63,
            "range": "min 171.4 max 193.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/e2e_us",
            "value": 332.16,
            "range": "min 316.8 max 346.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/compile_s",
            "value": 1.6,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3408,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 200.88,
            "range": "min 183.0 max 211.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/e2e_us",
            "value": 352.69,
            "range": "min 327.3 max 371.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 9608,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 4572,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 220.89,
            "range": "min 212.7 max 351.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/e2e_us",
            "value": 378.31,
            "range": "min 362.7 max 787.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/compile_s",
            "value": 3.19,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 11145,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 5748,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 243.37,
            "range": "min 228.5 max 322.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/e2e_us",
            "value": 405.94,
            "range": "min 387.7 max 486.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/compile_s",
            "value": 1.76,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13785,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 3964,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 248.92,
            "range": "min 220.1 max 382.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/e2e_us",
            "value": 435.43,
            "range": "min 388.0 max 722.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/compile_s",
            "value": 1.76,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13785,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 3968,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 345.8,
            "range": "min 327.5 max 483.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/e2e_us",
            "value": 587.5,
            "range": "min 530.9 max 667.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/compile_s",
            "value": 1.94,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 18489,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 5428,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 312.31,
            "range": "min 284.9 max 486.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/e2e_us",
            "value": 546.94,
            "range": "min 471.7 max 878.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/compile_s",
            "value": 1.87,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 18617,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 8060,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 293.45,
            "range": "min 280.1 max 564.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/e2e_us",
            "value": 470.33,
            "range": "min 453.1 max 790.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/compile_s",
            "value": 1.89,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 18665,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 7196,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 240.23,
            "range": "min 216.3 max 354.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/e2e_us",
            "value": 404.95,
            "range": "min 377.0 max 565.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13785,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 3964,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 1417.08,
            "range": "min 1079.1 max 1915.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/e2e_us",
            "value": 2025.66,
            "range": "min 1551.2 max 2741.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/compile_s",
            "value": 1.83,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 48633,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 7704,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 467.91,
            "range": "min 430.8 max 1470.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/e2e_us",
            "value": 802.63,
            "range": "min 735.2 max 2552.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/compile_s",
            "value": 1.64,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 47945,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 6944,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 183.78,
            "range": "min 170.8 max 308.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/e2e_us",
            "value": 328.27,
            "range": "min 311.7 max 449.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3548,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 179.68,
            "range": "min 166.4 max 272.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/e2e_us",
            "value": 321.54,
            "range": "min 303.6 max 424.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3548,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Erika Hunhoff",
            "username": "hunhoffe",
            "email": "erika.hunhoff@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "a82bb55c19869f1bdaeb1ca44e61f362349c5e0f",
          "message": "[trace] trace_to_json: one file per trace-buffer slice; parser raises instead of exiting (#3805)\n\nCo-authored-by: Claude <noreply@anthropic.com>\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
          "timestamp": "2026-09-25T19:02:08Z",
          "url": "https://github.com/Xilinx/mlir-aie/commit/a82bb55c19869f1bdaeb1ca44e61f362349c5e0f"
        },
        "date": 1790404095460,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 264,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 128.906,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 186.1,
            "range": "min 181.3 max 196.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/e2e_us",
            "value": 340.05,
            "range": "min 332.5 max 349.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/compile_s",
            "value": 1.58,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 264,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 128.906,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 1412.09,
            "range": "min 865.4 max 1724.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/e2e_us",
            "value": 1948.38,
            "range": "min 1365.5 max 2593.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/compile_s",
            "value": 1.56,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 264,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 64.453,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 192.56,
            "range": "min 178.6 max 289.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/e2e_us",
            "value": 345.7,
            "range": "min 324.1 max 440.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 136,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 33.203,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 177.13,
            "range": "min 162.6 max 249.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/e2e_us",
            "value": 330.54,
            "range": "min 307.6 max 435.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 169.44,
            "range": "min 154.2 max 350.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/e2e_us",
            "value": 322.21,
            "range": "min 299.1 max 733.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 3048,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 287.6,
            "range": "min 268.4 max 365.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/e2e_us",
            "value": 459.54,
            "range": "min 433.8 max 796.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3048,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 176.05,
            "range": "min 163.0 max 254.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/e2e_us",
            "value": 325.72,
            "range": "min 310.3 max 399.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/compile_s",
            "value": 1.59,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9079,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 3208,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 186.64,
            "range": "min 168.6 max 338.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/e2e_us",
            "value": 334.37,
            "range": "min 315.2 max 481.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/compile_s",
            "value": 1.73,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 8951,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3052,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 1366.21,
            "range": "min 599.8 max 1607.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/e2e_us",
            "value": 1835.99,
            "range": "min 815.2 max 2369.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/compile_s",
            "value": 1.74,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 8951,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3052,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 188.46,
            "range": "min 169.4 max 345.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/e2e_us",
            "value": 337.86,
            "range": "min 318.6 max 497.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/compile_s",
            "value": 1.74,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 1076.81,
            "range": "min 596.2 max 1630.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/e2e_us",
            "value": 1616.99,
            "range": "min 918.3 max 2475.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/compile_s",
            "value": 1.74,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 159,
            "range": "min 149.6 max 255.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/e2e_us",
            "value": 302.05,
            "range": "min 285.6 max 468.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 771.99,
            "range": "min 391.0 max 1475.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/e2e_us",
            "value": 1323.76,
            "range": "min 635.1 max 2280.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 175.34,
            "range": "min 158.3 max 251.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/e2e_us",
            "value": 324.22,
            "range": "min 305.3 max 399.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 3092,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 431.84,
            "range": "min 415.3 max 482.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/e2e_us",
            "value": 674.74,
            "range": "min 587.8 max 693.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 3092,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 179.23,
            "range": "min 160.8 max 260.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/e2e_us",
            "value": 326.78,
            "range": "min 308.1 max 401.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/compile_s",
            "value": 1.55,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 3108,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 430.81,
            "range": "min 402.5 max 453.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/e2e_us",
            "value": 659.44,
            "range": "min 570.1 max 708.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/compile_s",
            "value": 1.57,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 3108,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 173.18,
            "range": "min 160.2 max 299.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/e2e_us",
            "value": 323.41,
            "range": "min 303.5 max 459.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/compile_s",
            "value": 1.64,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 3140,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 432.71,
            "range": "min 420.9 max 521.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/e2e_us",
            "value": 677.32,
            "range": "min 647.7 max 757.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 3140,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 165.25,
            "range": "min 154.9 max 254.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/e2e_us",
            "value": 317.73,
            "range": "min 304.5 max 404.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 343.98,
            "range": "min 293.5 max 1807.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/e2e_us",
            "value": 550.23,
            "range": "min 512.4 max 2281.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 257.93,
            "range": "min 241.8 max 362.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/e2e_us",
            "value": 405.2,
            "range": "min 378.1 max 508.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/compile_s",
            "value": 4.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 14649,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 9244,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 2173.39,
            "range": "min 1895.0 max 2760.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/e2e_us",
            "value": 2974.09,
            "range": "min 2400.9 max 3448.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/compile_s",
            "value": 4.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 14649,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 9244,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 225.67,
            "range": "min 205.6 max 318.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/e2e_us",
            "value": 373.82,
            "range": "min 347.3 max 464.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/compile_s",
            "value": 4.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 14441,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 9020,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 1436.06,
            "range": "min 1256.8 max 1935.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/e2e_us",
            "value": 1996.33,
            "range": "min 1512.1 max 2667.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/compile_s",
            "value": 4.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 14441,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 9020,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 217,
            "range": "min 200.5 max 297.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/e2e_us",
            "value": 360.42,
            "range": "min 343.7 max 441.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/compile_s",
            "value": 3.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 14073,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 8632,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 1492.96,
            "range": "min 1201.4 max 1963.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/e2e_us",
            "value": 2001.41,
            "range": "min 1520.7 max 2622.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/compile_s",
            "value": 3.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 14073,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 8632,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 201.76,
            "range": "min 182.5 max 362.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/e2e_us",
            "value": 353.14,
            "range": "min 320.3 max 756.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/compile_s",
            "value": 3.46,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 14249,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 8952,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 1342.32,
            "range": "min 886.0 max 1624.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/e2e_us",
            "value": 1908.4,
            "range": "min 1284.9 max 2420.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/compile_s",
            "value": 3.46,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 14249,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 8952,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 211.1,
            "range": "min 198.2 max 301.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/e2e_us",
            "value": 354.79,
            "range": "min 331.9 max 448.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/compile_s",
            "value": 3.43,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 14377,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 9088,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 1432.38,
            "range": "min 1065.6 max 1842.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/e2e_us",
            "value": 1991.3,
            "range": "min 1482.7 max 2752.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/compile_s",
            "value": 3.43,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 14377,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 9088,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 246.79,
            "range": "min 230.5 max 434.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/e2e_us",
            "value": 399.3,
            "range": "min 370.8 max 636.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/compile_s",
            "value": 3.29,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 16057,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 11628,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 2164.1,
            "range": "min 1797.7 max 2653.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/e2e_us",
            "value": 2906.74,
            "range": "min 2350.0 max 3373.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/compile_s",
            "value": 3.31,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 16057,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 11628,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 163.46,
            "range": "min 150.8 max 255.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/e2e_us",
            "value": 312.71,
            "range": "min 298.6 max 400.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 2972,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 285.07,
            "range": "min 277.3 max 373.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/e2e_us",
            "value": 449.9,
            "range": "min 436.9 max 526.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/compile_s",
            "value": 1.77,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2972,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 172.71,
            "range": "min 158.0 max 184.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/e2e_us",
            "value": 324.16,
            "range": "min 307.9 max 344.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/compile_s",
            "value": 1.76,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 714.96,
            "range": "min 452.0 max 1477.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/e2e_us",
            "value": 1111.01,
            "range": "min 708.5 max 2332.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/compile_s",
            "value": 1.78,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 181.15,
            "range": "min 166.3 max 268.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/e2e_us",
            "value": 337.38,
            "range": "min 314.6 max 756.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 2984,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 532.99,
            "range": "min 438.4 max 1439.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/e2e_us",
            "value": 818.53,
            "range": "min 669.5 max 2122.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/compile_s",
            "value": 1.64,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 2984,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 160,
            "range": "min 148.0 max 263.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/e2e_us",
            "value": 308,
            "range": "min 292.3 max 417.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 9111,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 3216,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 168.83,
            "range": "min 152.8 max 183.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/e2e_us",
            "value": 313.93,
            "range": "min 296.3 max 389.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 9352,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 3492,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 157.99,
            "range": "min 147.8 max 253.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/e2e_us",
            "value": 310.85,
            "range": "min 295.6 max 411.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 9111,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 3140,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 192.88,
            "range": "min 164.8 max 402.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/e2e_us",
            "value": 400.39,
            "range": "min 311.6 max 724.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/compile_s",
            "value": 1.77,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 3524,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 242.25,
            "range": "min 228.9 max 326.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/e2e_us",
            "value": 403.84,
            "range": "min 390.6 max 482.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/compile_s",
            "value": 1.77,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4972,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1814.92,
            "range": "min 1558.5 max 2659.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/e2e_us",
            "value": 2725.42,
            "range": "min 2078.6 max 3217.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/compile_s",
            "value": 1.78,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4972,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 253.09,
            "range": "min 240.9 max 446.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/e2e_us",
            "value": 416.85,
            "range": "min 392.7 max 834.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 9912,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 236.61,
            "range": "min 215.2 max 300.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/e2e_us",
            "value": 445.82,
            "range": "min 370.2 max 508.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/compile_s",
            "value": 1.79,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 10329,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 4716,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 168.77,
            "range": "min 141.9 max 252.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/e2e_us",
            "value": 319.38,
            "range": "min 290.5 max 587.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/compile_s",
            "value": 3.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 10505,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 4688,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 169.76,
            "range": "min 147.5 max 255.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/e2e_us",
            "value": 319.56,
            "range": "min 296.0 max 414.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/compile_s",
            "value": 1.8,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9608,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3768,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 151.56,
            "range": "min 141.0 max 265.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/e2e_us",
            "value": 297.23,
            "range": "min 284.9 max 410.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/compile_s",
            "value": 1.59,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2856,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 150.41,
            "range": "min 139.5 max 173.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/e2e_us",
            "value": 296.43,
            "range": "min 285.9 max 321.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/compile_s",
            "value": 1.6,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2880,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 258.81,
            "range": "min 239.5 max 347.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/e2e_us",
            "value": 408.92,
            "range": "min 386.5 max 500.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/compile_s",
            "value": 3.85,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 14889,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 10196,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 2117.88,
            "range": "min 1859.6 max 2713.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/e2e_us",
            "value": 2915.57,
            "range": "min 2341.9 max 3489.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/compile_s",
            "value": 3.85,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 14889,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 10196,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 194.03,
            "range": "min 174.9 max 266.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/e2e_us",
            "value": 342.69,
            "range": "min 322.2 max 419.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/compile_s",
            "value": 1.57,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 2948,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 190.4,
            "range": "min 176.7 max 270.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/e2e_us",
            "value": 331.38,
            "range": "min 318.7 max 414.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 3136,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 160.96,
            "range": "min 147.6 max 257.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/e2e_us",
            "value": 306.34,
            "range": "min 292.0 max 401.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/compile_s",
            "value": 1.63,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 9816,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 4804,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 171.08,
            "range": "min 163.1 max 288.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/e2e_us",
            "value": 334.46,
            "range": "min 306.7 max 498.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/compile_s",
            "value": 1.57,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3100,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 193.43,
            "range": "min 165.5 max 274.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/e2e_us",
            "value": 400.51,
            "range": "min 312.1 max 791.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/compile_s",
            "value": 1.56,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 8999,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3100,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 180.24,
            "range": "min 168.4 max 358.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/e2e_us",
            "value": 324.92,
            "range": "min 306.6 max 565.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3408,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 196.48,
            "range": "min 180.8 max 315.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/e2e_us",
            "value": 343.26,
            "range": "min 324.1 max 469.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/compile_s",
            "value": 1.64,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 9608,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 4572,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 226.99,
            "range": "min 206.5 max 321.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/e2e_us",
            "value": 374.68,
            "range": "min 354.1 max 471.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/compile_s",
            "value": 3.2,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 11145,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 5748,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 243.21,
            "range": "min 226.4 max 321.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/e2e_us",
            "value": 405.52,
            "range": "min 390.4 max 483.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13785,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 3964,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 239.64,
            "range": "min 226.1 max 328.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/e2e_us",
            "value": 400.37,
            "range": "min 387.8 max 570.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/compile_s",
            "value": 1.76,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13785,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 3968,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 325.58,
            "range": "min 312.4 max 489.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/e2e_us",
            "value": 512.2,
            "range": "min 499.2 max 661.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/compile_s",
            "value": 1.95,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 18489,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 5428,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 290.92,
            "range": "min 284.1 max 394.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/e2e_us",
            "value": 467.51,
            "range": "min 452.4 max 574.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/compile_s",
            "value": 1.88,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 18617,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 8060,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 296.83,
            "range": "min 278.6 max 365.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/e2e_us",
            "value": 474.93,
            "range": "min 453.2 max 569.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/compile_s",
            "value": 1.88,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 18665,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 7196,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 236.4,
            "range": "min 226.2 max 347.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/e2e_us",
            "value": 402.58,
            "range": "min 386.2 max 518.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/compile_s",
            "value": 1.75,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13785,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 3964,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 1408.87,
            "range": "min 1095.5 max 1805.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/e2e_us",
            "value": 2039.49,
            "range": "min 1613.1 max 2505.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/compile_s",
            "value": 1.83,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 48633,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 7704,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 406.33,
            "range": "min 390.5 max 708.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/e2e_us",
            "value": 701.64,
            "range": "min 631.7 max 1705.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/compile_s",
            "value": 1.64,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 47945,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 6944,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 184.63,
            "range": "min 169.8 max 273.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/e2e_us",
            "value": 339.15,
            "range": "min 316.9 max 426.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/compile_s",
            "value": 1.61,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3548,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 189.9,
            "range": "min 169.8 max 326.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/e2e_us",
            "value": 359.89,
            "range": "min 315.3 max 563.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/compile_s",
            "value": 1.62,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3548,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device RyzenAI-npu1 | pmode default"
          }
        ]
      }
    ]
  }
}