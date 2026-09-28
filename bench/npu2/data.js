window.BENCHMARK_DATA = {
  "lastUpdate": 1790577365871,
  "repoUrl": "https://github.com/Xilinx/mlir-aie",
  "entries": {
    "aie_kernels (npu2, default)": [
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
        "date": 1790300321360,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 118.51,
            "range": "min 101.8 max 182.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/e2e_us",
            "value": 327.31,
            "range": "min 220.9 max 593.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 273.41,
            "range": "min 258.8 max 289.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/e2e_us",
            "value": 417.48,
            "range": "min 399.8 max 500.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 33.691,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 100.14,
            "range": "min 90.0 max 106.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/e2e_us",
            "value": 215.89,
            "range": "min 209.6 max 225.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 74,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 18.066,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 109.86,
            "range": "min 92.5 max 122.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/e2e_us",
            "value": 222.04,
            "range": "min 203.9 max 286.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 108.64,
            "range": "min 100.0 max 150.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/e2e_us",
            "value": 326.04,
            "range": "min 307.7 max 586.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 146.24,
            "range": "min 132.2 max 155.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/e2e_us",
            "value": 256.05,
            "range": "min 244.6 max 267.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 109.54,
            "range": "min 98.0 max 178.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/e2e_us",
            "value": 215.06,
            "range": "min 200.6 max 301.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9416,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 4216,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 110.89,
            "range": "min 99.0 max 148.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/e2e_us",
            "value": 206.52,
            "range": "min 193.4 max 253.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 199.47,
            "range": "min 187.5 max 297.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/e2e_us",
            "value": 311.25,
            "range": "min 297.8 max 883.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/compile_s",
            "value": 2.2,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 109.86,
            "range": "min 97.2 max 116.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/e2e_us",
            "value": 207.53,
            "range": "min 194.9 max 215.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 199.67,
            "range": "min 189.6 max 241.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/e2e_us",
            "value": 315.61,
            "range": "min 299.0 max 376.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 110.94,
            "range": "min 100.1 max 134.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/e2e_us",
            "value": 316.29,
            "range": "min 300.0 max 391.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3872,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 143.35,
            "range": "min 130.4 max 148.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/e2e_us",
            "value": 249.08,
            "range": "min 236.6 max 261.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2948,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 102.56,
            "range": "min 90.5 max 109.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/e2e_us",
            "value": 204.25,
            "range": "min 191.6 max 215.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 171.11,
            "range": "min 158.7 max 179.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/e2e_us",
            "value": 278.3,
            "range": "min 270.0 max 321.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 109.55,
            "range": "min 96.0 max 120.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/e2e_us",
            "value": 216.05,
            "range": "min 196.8 max 282.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 183.02,
            "range": "min 158.4 max 200.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/e2e_us",
            "value": 362.97,
            "range": "min 266.4 max 490.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 112.64,
            "range": "min 99.3 max 153.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/e2e_us",
            "value": 314.06,
            "range": "min 224.0 max 409.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 3968,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 172.35,
            "range": "min 158.5 max 187.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/e2e_us",
            "value": 290.03,
            "range": "min 273.1 max 310.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 3060,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 109.07,
            "range": "min 84.5 max 177.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/e2e_us",
            "value": 303.54,
            "range": "min 186.5 max 566.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 3944,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 172.34,
            "range": "min 161.7 max 258.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/e2e_us",
            "value": 295.55,
            "range": "min 265.7 max 753.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 110.34,
            "range": "min 93.4 max 116.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/e2e_us",
            "value": 205.66,
            "range": "min 192.0 max 215.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/compile_s",
            "value": 4.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 9352,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 3932,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 194.72,
            "range": "min 178.2 max 223.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/e2e_us",
            "value": 299.56,
            "range": "min 287.3 max 334.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/compile_s",
            "value": 4.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 2992,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 103.29,
            "range": "min 97.1 max 127.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/e2e_us",
            "value": 199.76,
            "range": "min 194.6 max 223.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/compile_s",
            "value": 4.23,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 3852,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 119.97,
            "range": "min 108.8 max 124.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 219.89,
            "range": "min 211.9 max 230.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.4,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14825,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9928,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 283.69,
            "range": "min 266.6 max 324.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/e2e_us",
            "value": 394.49,
            "range": "min 373.4 max 711.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/compile_s",
            "value": 4.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 8775,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 2912,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 395.8,
            "range": "min 382.4 max 439.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/e2e_us",
            "value": 498.59,
            "range": "min 484.0 max 670.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/compile_s",
            "value": 4.58,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 11033,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 5684,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 5049.09,
            "range": "min 4756.2 max 5473.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/e2e_us",
            "value": 5904.06,
            "range": "min 5174.2 max 6332.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/compile_s",
            "value": 4.53,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 4744,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 100.75,
            "range": "min 85.5 max 121.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/e2e_us",
            "value": 203.93,
            "range": "min 186.9 max 342.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/compile_s",
            "value": 4.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 3884,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 145.38,
            "range": "min 132.0 max 166.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/e2e_us",
            "value": 259.28,
            "range": "min 248.5 max 393.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/compile_s",
            "value": 4.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 2976,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 126.88,
            "range": "min 110.8 max 138.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 317.95,
            "range": "min 213.4 max 383.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.45,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14953,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 10180,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 118.4,
            "range": "min 97.2 max 139.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/e2e_us",
            "value": 254.79,
            "range": "min 194.2 max 366.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/compile_s",
            "value": 4.21,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 3972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 134.19,
            "range": "min 111.4 max 146.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 229.5,
            "range": "min 205.1 max 284.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.36,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14793,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9868,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 182.51,
            "range": "min 161.0 max 220.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/e2e_us",
            "value": 297.85,
            "range": "min 271.2 max 627.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/compile_s",
            "value": 4.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 115.38,
            "range": "min 101.6 max 125.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/e2e_us",
            "value": 210.99,
            "range": "min 198.5 max 222.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/compile_s",
            "value": 4.22,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 9864,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 4656,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 309.7,
            "range": "min 300.3 max 417.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/e2e_us",
            "value": 424.41,
            "range": "min 409.8 max 1049.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/compile_s",
            "value": 4.23,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 3748,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 105.84,
            "range": "min 86.7 max 119.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/e2e_us",
            "value": 202.41,
            "range": "min 188.1 max 364.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3864,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 144.65,
            "range": "min 133.8 max 155.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/e2e_us",
            "value": 248.97,
            "range": "min 238.8 max 260.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/compile_s",
            "value": 2.2,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2956,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/npu_us",
            "value": 387.32,
            "range": "min 370.7 max 408.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/e2e_us",
            "value": 491.7,
            "range": "min 479.5 max 640.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/compile_s",
            "value": 2.37,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/xclbin_bytes",
            "value": 10921,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/core_elf_bytes",
            "value": 5620,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/npu_us",
            "value": 4929.72,
            "range": "min 4747.0 max 5486.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/e2e_us",
            "value": 5767.65,
            "range": "min 5272.1 max 6419.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/compile_s",
            "value": 2.35,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/xclbin_bytes",
            "value": 10457,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/core_elf_bytes",
            "value": 4712,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 109.88,
            "range": "min 97.2 max 115.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/e2e_us",
            "value": 205.44,
            "range": "min 191.8 max 213.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 188.58,
            "range": "min 177.0 max 202.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/e2e_us",
            "value": 304.93,
            "range": "min 290.2 max 317.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/npu_us",
            "value": 109.33,
            "range": "min 96.8 max 153.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/e2e_us",
            "value": 210.68,
            "range": "min 198.7 max 466.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/core_elf_bytes",
            "value": 3924,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/npu_us",
            "value": 205.87,
            "range": "min 188.2 max 267.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/e2e_us",
            "value": 322.35,
            "range": "min 305.4 max 684.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/xclbin_bytes",
            "value": 8823,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/core_elf_bytes",
            "value": 3016,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 110.93,
            "range": "min 98.2 max 113.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/e2e_us",
            "value": 211.39,
            "range": "min 199.4 max 221.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 3840,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 222.5,
            "range": "min 209.1 max 229.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/e2e_us",
            "value": 331.74,
            "range": "min 320.8 max 343.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 2916,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 140.93,
            "range": "min 126.5 max 153.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/e2e_us",
            "value": 335.65,
            "range": "min 320.9 max 350.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 9512,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 4324,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 194.14,
            "range": "min 173.9 max 210.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/e2e_us",
            "value": 300.04,
            "range": "min 276.6 max 312.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 10120,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 5428,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 102.61,
            "range": "min 91.7 max 109.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/e2e_us",
            "value": 204.29,
            "range": "min 192.2 max 216.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 9464,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 4236,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 206.35,
            "range": "min 185.4 max 221.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/e2e_us",
            "value": 347.73,
            "range": "min 291.0 max 394.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 9896,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 5220,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 162.25,
            "range": "min 144.4 max 181.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/e2e_us",
            "value": 274.5,
            "range": "min 252.6 max 355.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/compile_s",
            "value": 2.28,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10088,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4524,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1145.46,
            "range": "min 1094.2 max 1213.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/e2e_us",
            "value": 1711.53,
            "range": "min 1255.1 max 1914.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/compile_s",
            "value": 2.28,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10088,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4524,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 137.2,
            "range": "min 127.2 max 181.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/e2e_us",
            "value": 245.02,
            "range": "min 234.6 max 291.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4076,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 121.66,
            "range": "min 111.1 max 167.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/e2e_us",
            "value": 228.14,
            "range": "min 219.1 max 305.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/compile_s",
            "value": 2.27,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 9688,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 4120,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 111.21,
            "range": "min 96.8 max 114.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/e2e_us",
            "value": 221.4,
            "range": "min 207.8 max 230.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/compile_s",
            "value": 4.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 10233,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 4692,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/npu_us",
            "value": 152.73,
            "range": "min 139.4 max 160.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/e2e_us",
            "value": 252.75,
            "range": "min 239.5 max 267.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/compile_s",
            "value": 2.41,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/xclbin_bytes",
            "value": 9976,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/core_elf_bytes",
            "value": 4420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/npu_us",
            "value": 215.65,
            "range": "min 195.3 max 266.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/e2e_us",
            "value": 314.48,
            "range": "min 294.8 max 724.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/xclbin_bytes",
            "value": 11081,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/core_elf_bytes",
            "value": 6296,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/npu_us",
            "value": 113.97,
            "range": "min 99.7 max 120.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/e2e_us",
            "value": 219.54,
            "range": "min 204.5 max 230.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/compile_s",
            "value": 2.82,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/core_elf_bytes",
            "value": 3208,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/npu_us",
            "value": 120.53,
            "range": "min 104.1 max 126.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/e2e_us",
            "value": 228.18,
            "range": "min 211.2 max 244.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/compile_s",
            "value": 2.36,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/xclbin_bytes",
            "value": 10008,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/core_elf_bytes",
            "value": 4340,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 99.32,
            "range": "min 87.1 max 113.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/e2e_us",
            "value": 205.33,
            "range": "min 191.8 max 219.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/compile_s",
            "value": 2.27,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9624,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3832,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 98.78,
            "range": "min 83.1 max 106.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/e2e_us",
            "value": 197.99,
            "range": "min 181.3 max 206.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8743,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2808,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 96.49,
            "range": "min 82.2 max 100.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/e2e_us",
            "value": 192.82,
            "range": "min 179.8 max 199.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2832,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 132.13,
            "range": "min 123.4 max 149.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/e2e_us",
            "value": 230.05,
            "range": "min 218.6 max 245.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/compile_s",
            "value": 4.19,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 149.08,
            "range": "min 140.3 max 179.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 250.21,
            "range": "min 238.4 max 376.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.33,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14329,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9540,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 650.43,
            "range": "min 642.0 max 666.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/e2e_us",
            "value": 768.3,
            "range": "min 755.8 max 829.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/compile_s",
            "value": 4.2,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 115.24,
            "range": "min 97.8 max 124.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/e2e_us",
            "value": 221.45,
            "range": "min 202.8 max 238.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 4104,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 120.04,
            "range": "min 105.8 max 121.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/e2e_us",
            "value": 226.08,
            "range": "min 210.7 max 234.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 9432,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 108.81,
            "range": "min 96.2 max 115.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/e2e_us",
            "value": 209.41,
            "range": "min 195.1 max 224.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/compile_s",
            "value": 2.19,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 4868,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 110.41,
            "range": "min 93.5 max 115.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/e2e_us",
            "value": 210.33,
            "range": "min 195.0 max 218.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 110.44,
            "range": "min 99.5 max 159.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/e2e_us",
            "value": 212.08,
            "range": "min 198.5 max 580.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3068,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 114.12,
            "range": "min 105.0 max 141.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/e2e_us",
            "value": 313.98,
            "range": "min 300.4 max 539.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3352,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 148.64,
            "range": "min 141.3 max 173.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/e2e_us",
            "value": 255.04,
            "range": "min 246.7 max 277.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 10665,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 6096,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 1714.04,
            "range": "min 1642.0 max 1738.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/e2e_us",
            "value": 2247.61,
            "range": "min 1750.6 max 2384.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/compile_s",
            "value": 4.27,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 9576,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 174.61,
            "range": "min 161.0 max 195.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/e2e_us",
            "value": 284.07,
            "range": "min 271.4 max 313.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 132.35,
            "range": "min 118.4 max 155.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/e2e_us",
            "value": 245.51,
            "range": "min 231.9 max 372.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13673,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 4204,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 204.04,
            "range": "min 187.5 max 226.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/e2e_us",
            "value": 321.8,
            "range": "min 302.1 max 356.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 18025,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 4888,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 199.44,
            "range": "min 189.2 max 209.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/e2e_us",
            "value": 314.73,
            "range": "min 311.4 max 324.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 17785,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 7264,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 200.11,
            "range": "min 186.0 max 245.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/e2e_us",
            "value": 329.12,
            "range": "min 308.3 max 665.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 18601,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 7628,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/npu_us",
            "value": 105.85,
            "range": "min 89.1 max 116.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/e2e_us",
            "value": 230.87,
            "range": "min 213.7 max 249.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/xclbin_bytes",
            "value": 21225,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 174.25,
            "range": "min 160.0 max 180.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/e2e_us",
            "value": 284.38,
            "range": "min 268.6 max 295.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 531.56,
            "range": "min 520.5 max 538.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/e2e_us",
            "value": 706.01,
            "range": "min 692.0 max 722.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 48441,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 271.98,
            "range": "min 260.7 max 282.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/e2e_us",
            "value": 452.44,
            "range": "min 436.5 max 646.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/compile_s",
            "value": 2.19,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 48809,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 7832,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 111.9,
            "range": "min 98.5 max 169.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/e2e_us",
            "value": 302.68,
            "range": "min 204.8 max 553.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 112.28,
            "range": "min 100.6 max 116.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/e2e_us",
            "value": 208.17,
            "range": "min 197.3 max 213.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 110.4,
            "range": "min 97.5 max 158.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 216.56,
            "range": "min 201.0 max 550.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.33,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 12377,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 8564,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 124.52,
            "range": "min 108.5 max 128.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 220.71,
            "range": "min 206.0 max 238.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 12425,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 8640,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/npu_us",
            "value": 210.48,
            "range": "min 196.5 max 214.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/e2e_us",
            "value": 311.05,
            "range": "min 296.1 max 319.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/xclbin_bytes",
            "value": 11977,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/npu_us",
            "value": 223.3,
            "range": "min 208.2 max 241.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/e2e_us",
            "value": 465.9,
            "range": "min 450.3 max 521.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/xclbin_bytes",
            "value": 20233,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/core_elf_bytes",
            "value": 7952,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/npu_us",
            "value": 124.06,
            "range": "min 111.0 max 144.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 260.33,
            "range": "min 208.5 max 302.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.35,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 3132,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/npu_us",
            "value": 110.84,
            "range": "min 98.9 max 118.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/e2e_us",
            "value": 213.18,
            "range": "min 201.3 max 234.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/compile_s",
            "value": 2.28,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/npu_us",
            "value": 180.07,
            "range": "min 170.3 max 184.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/e2e_us",
            "value": 282.15,
            "range": "min 276.1 max 291.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/compile_s",
            "value": 2.3,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/npu_us",
            "value": 151.48,
            "range": "min 138.6 max 155.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/e2e_us",
            "value": 252.57,
            "range": "min 243.2 max 257.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/compile_s",
            "value": 2.27,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/npu_us",
            "value": 108.08,
            "range": "min 98.0 max 115.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/e2e_us",
            "value": 209.59,
            "range": "min 199.4 max 231.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/compile_s",
            "value": 2.29,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 121.82,
            "range": "min 111.9 max 159.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/e2e_us",
            "value": 225.55,
            "range": "min 214.9 max 258.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3744,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 129.09,
            "range": "min 104.0 max 172.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/e2e_us",
            "value": 296.75,
            "range": "min 218.2 max 571.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/compile_s",
            "value": 2.19,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3744,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/npu_us",
            "value": 109.68,
            "range": "min 98.2 max 132.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/e2e_us",
            "value": 206.91,
            "range": "min 197.1 max 382.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/xclbin_bytes",
            "value": 10633,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/core_elf_bytes",
            "value": 7380,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/npu_us",
            "value": 108.2,
            "range": "min 99.3 max 120.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/e2e_us",
            "value": 254.49,
            "range": "min 209.8 max 326.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/compile_s",
            "value": 2.79,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/core_elf_bytes",
            "value": 4052,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/npu_us",
            "value": 116.65,
            "range": "min 102.9 max 121.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/e2e_us",
            "value": 226.19,
            "range": "min 212.8 max 234.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/compile_s",
            "value": 2.91,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/xclbin_bytes",
            "value": 11513,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/core_elf_bytes",
            "value": 5972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
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
        "date": 1790318187695,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 108.98,
            "range": "min 95.0 max 131.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/e2e_us",
            "value": 214.6,
            "range": "min 202.4 max 241.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 271.64,
            "range": "min 254.1 max 302.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/e2e_us",
            "value": 413.07,
            "range": "min 396.8 max 498.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 33.691,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 111.63,
            "range": "min 98.3 max 123.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/e2e_us",
            "value": 216.93,
            "range": "min 205.5 max 230.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 74,
            "unit": "cycles",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 18.066,
            "unit": "cycles/1k-ops",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 102.47,
            "range": "min 87.7 max 130.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/e2e_us",
            "value": 208.16,
            "range": "min 193.1 max 239.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 96.55,
            "range": "min 81.7 max 100.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/e2e_us",
            "value": 199.47,
            "range": "min 184.9 max 213.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 146.38,
            "range": "min 127.6 max 161.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/e2e_us",
            "value": 258.4,
            "range": "min 239.6 max 284.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 109.48,
            "range": "min 97.0 max 114.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/e2e_us",
            "value": 216.51,
            "range": "min 203.1 max 228.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9416,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 4216,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 109.59,
            "range": "min 96.4 max 115.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/e2e_us",
            "value": 205.36,
            "range": "min 189.9 max 211.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 191.88,
            "range": "min 178.4 max 198.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/e2e_us",
            "value": 298.65,
            "range": "min 287.1 max 312.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/compile_s",
            "value": 2.2,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 109.53,
            "range": "min 93.1 max 131.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/e2e_us",
            "value": 211.24,
            "range": "min 193.4 max 236.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 201.57,
            "range": "min 186.9 max 255.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/e2e_us",
            "value": 316.04,
            "range": "min 300.4 max 372.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/compile_s",
            "value": 2.22,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 97.79,
            "range": "min 85.6 max 771.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/e2e_us",
            "value": 202.17,
            "range": "min 185.7 max 888.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3872,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 144.23,
            "range": "min 133.0 max 196.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/e2e_us",
            "value": 249.89,
            "range": "min 241.4 max 607.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2948,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 100.63,
            "range": "min 85.0 max 111.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/e2e_us",
            "value": 199.58,
            "range": "min 181.2 max 215.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 176.83,
            "range": "min 169.7 max 186.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/e2e_us",
            "value": 323.7,
            "range": "min 282.7 max 367.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 107.61,
            "range": "min 95.6 max 113.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/e2e_us",
            "value": 208.51,
            "range": "min 194.3 max 216.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 188.75,
            "range": "min 171.7 max 232.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/e2e_us",
            "value": 404.81,
            "range": "min 291.5 max 636.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 109.53,
            "range": "min 96.7 max 115.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/e2e_us",
            "value": 215.22,
            "range": "min 204.0 max 267.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 3968,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 177.25,
            "range": "min 166.5 max 187.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/e2e_us",
            "value": 329.35,
            "range": "min 309.9 max 361.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 3060,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 109.34,
            "range": "min 84.2 max 167.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/e2e_us",
            "value": 213.34,
            "range": "min 182.7 max 380.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 3944,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 165.66,
            "range": "min 153.8 max 186.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/e2e_us",
            "value": 271.95,
            "range": "min 257.7 max 397.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 109.31,
            "range": "min 96.5 max 139.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/e2e_us",
            "value": 211.32,
            "range": "min 197.0 max 373.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/compile_s",
            "value": 4.2,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 9352,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 3932,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 191.97,
            "range": "min 177.4 max 200.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/e2e_us",
            "value": 306.04,
            "range": "min 289.5 max 312.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/compile_s",
            "value": 4.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 2992,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 111.71,
            "range": "min 93.0 max 125.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/e2e_us",
            "value": 270.65,
            "range": "min 235.3 max 364.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/compile_s",
            "value": 4.2,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 3852,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 129.52,
            "range": "min 115.5 max 136.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 231.9,
            "range": "min 218.1 max 240.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.37,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14825,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9928,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 287.94,
            "range": "min 265.7 max 330.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/e2e_us",
            "value": 457.72,
            "range": "min 374.7 max 758.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/compile_s",
            "value": 4.22,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 8775,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 2912,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 396.53,
            "range": "min 381.6 max 431.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/e2e_us",
            "value": 589.7,
            "range": "min 482.5 max 827.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/compile_s",
            "value": 4.56,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 11033,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 5684,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 4829.85,
            "range": "min 4741.6 max 5406.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/e2e_us",
            "value": 5293.73,
            "range": "min 5084.7 max 6208.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/compile_s",
            "value": 4.54,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 4744,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 100.41,
            "range": "min 92.0 max 130.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/e2e_us",
            "value": 197.45,
            "range": "min 187.1 max 335.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/compile_s",
            "value": 4.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 3884,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 144.21,
            "range": "min 129.9 max 147.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/e2e_us",
            "value": 247.42,
            "range": "min 233.5 max 255.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/compile_s",
            "value": 4.09,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 2976,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 115.82,
            "range": "min 101.9 max 121.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 213.78,
            "range": "min 201.2 max 222.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.47,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14953,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 10180,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 102.75,
            "range": "min 89.7 max 141.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/e2e_us",
            "value": 204.75,
            "range": "min 188.7 max 531.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/compile_s",
            "value": 4.22,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 3972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 126.89,
            "range": "min 114.1 max 170.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 240.41,
            "range": "min 214.0 max 569.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.36,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14793,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9868,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 180.42,
            "range": "min 164.8 max 192.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/e2e_us",
            "value": 287.25,
            "range": "min 281.1 max 357.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/compile_s",
            "value": 4.19,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 110.5,
            "range": "min 96.7 max 114.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/e2e_us",
            "value": 213.02,
            "range": "min 205.1 max 222.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/compile_s",
            "value": 4.21,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 9864,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 4656,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 316.87,
            "range": "min 290.6 max 347.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/e2e_us",
            "value": 532.58,
            "range": "min 413.1 max 738.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/compile_s",
            "value": 4.21,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 3748,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 97.95,
            "range": "min 83.8 max 104.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/e2e_us",
            "value": 202.23,
            "range": "min 192.2 max 216.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/compile_s",
            "value": 2.22,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3864,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 144.29,
            "range": "min 136.6 max 155.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/e2e_us",
            "value": 260.8,
            "range": "min 248.0 max 269.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/compile_s",
            "value": 2.2,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2956,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/npu_us",
            "value": 379.09,
            "range": "min 363.7 max 403.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/e2e_us",
            "value": 486.87,
            "range": "min 466.7 max 679.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/compile_s",
            "value": 2.37,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/xclbin_bytes",
            "value": 10921,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/core_elf_bytes",
            "value": 5620,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/npu_us",
            "value": 4703.66,
            "range": "min 4612.0 max 5334.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/e2e_us",
            "value": 5249.02,
            "range": "min 5078.9 max 6254.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/compile_s",
            "value": 2.34,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/xclbin_bytes",
            "value": 10457,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/core_elf_bytes",
            "value": 4712,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 110.68,
            "range": "min 93.3 max 140.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/e2e_us",
            "value": 210.93,
            "range": "min 193.1 max 378.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/compile_s",
            "value": 2.22,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 188.06,
            "range": "min 179.0 max 200.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/e2e_us",
            "value": 304.29,
            "range": "min 293.5 max 326.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/npu_us",
            "value": 110.51,
            "range": "min 87.0 max 192.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/e2e_us",
            "value": 216.16,
            "range": "min 194.3 max 607.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/core_elf_bytes",
            "value": 3924,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/npu_us",
            "value": 203.24,
            "range": "min 185.7 max 217.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/e2e_us",
            "value": 319.2,
            "range": "min 302.0 max 377.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/xclbin_bytes",
            "value": 8823,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/core_elf_bytes",
            "value": 3016,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 109.86,
            "range": "min 97.0 max 130.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/e2e_us",
            "value": 210.82,
            "range": "min 201.4 max 235.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 3840,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 233.3,
            "range": "min 208.5 max 275.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/e2e_us",
            "value": 445.41,
            "range": "min 311.4 max 684.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 2916,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 139.63,
            "range": "min 118.8 max 159.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/e2e_us",
            "value": 238.18,
            "range": "min 215.9 max 442.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 9512,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 4324,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 186.69,
            "range": "min 172.9 max 208.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/e2e_us",
            "value": 287.18,
            "range": "min 271.2 max 405.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 10120,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 5428,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 109.78,
            "range": "min 102.9 max 125.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/e2e_us",
            "value": 220.54,
            "range": "min 210.0 max 231.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 9464,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 4236,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 207.87,
            "range": "min 194.2 max 257.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/e2e_us",
            "value": 316.65,
            "range": "min 304.8 max 666.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 9896,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 5220,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 170.45,
            "range": "min 158.1 max 176.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/e2e_us",
            "value": 276.68,
            "range": "min 266.2 max 284.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/compile_s",
            "value": 2.29,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10088,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4524,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1097.38,
            "range": "min 1085.0 max 1176.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/e2e_us",
            "value": 1406.65,
            "range": "min 1240.3 max 1890.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10088,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4524,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 130.11,
            "range": "min 116.2 max 132.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/e2e_us",
            "value": 236.05,
            "range": "min 223.4 max 241.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4076,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 111.29,
            "range": "min 99.0 max 156.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/e2e_us",
            "value": 220.03,
            "range": "min 208.8 max 568.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 9688,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 4120,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 108.57,
            "range": "min 97.3 max 128.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/e2e_us",
            "value": 211.54,
            "range": "min 198.2 max 546.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/compile_s",
            "value": 4.15,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 10233,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 4692,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/npu_us",
            "value": 155.39,
            "range": "min 140.7 max 196.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/e2e_us",
            "value": 314.18,
            "range": "min 281.7 max 619.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/compile_s",
            "value": 2.44,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/xclbin_bytes",
            "value": 9976,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/core_elf_bytes",
            "value": 4420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/npu_us",
            "value": 216.26,
            "range": "min 206.5 max 222.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/e2e_us",
            "value": 354.67,
            "range": "min 340.5 max 388.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/xclbin_bytes",
            "value": 11081,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/core_elf_bytes",
            "value": 6296,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/npu_us",
            "value": 110.33,
            "range": "min 96.1 max 122.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/e2e_us",
            "value": 213.37,
            "range": "min 195.5 max 281.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/compile_s",
            "value": 2.84,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/core_elf_bytes",
            "value": 3208,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/npu_us",
            "value": 127.08,
            "range": "min 101.5 max 169.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/e2e_us",
            "value": 298.09,
            "range": "min 252.6 max 489.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/compile_s",
            "value": 2.41,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/xclbin_bytes",
            "value": 10008,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/core_elf_bytes",
            "value": 4340,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 115.44,
            "range": "min 104.0 max 124.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/e2e_us",
            "value": 274.17,
            "range": "min 245.3 max 298.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/compile_s",
            "value": 2.28,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9624,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3832,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 105.78,
            "range": "min 90.4 max 156.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/e2e_us",
            "value": 207.87,
            "range": "min 193.9 max 476.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8743,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2808,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 104.61,
            "range": "min 87.9 max 111.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/e2e_us",
            "value": 199.13,
            "range": "min 182.0 max 207.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2832,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 142.65,
            "range": "min 127.7 max 152.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/e2e_us",
            "value": 246.44,
            "range": "min 235.1 max 259.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/compile_s",
            "value": 4.22,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 155.16,
            "range": "min 135.7 max 167.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 285.35,
            "range": "min 239.9 max 366.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.31,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14329,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9540,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 650.7,
            "range": "min 632.2 max 718.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/e2e_us",
            "value": 775.64,
            "range": "min 760.6 max 1203.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/compile_s",
            "value": 4.19,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 117.06,
            "range": "min 104.6 max 128.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/e2e_us",
            "value": 218.76,
            "range": "min 207.9 max 234.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 4104,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 110.59,
            "range": "min 102.2 max 135.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/e2e_us",
            "value": 216.02,
            "range": "min 208.0 max 242.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 9432,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 110.17,
            "range": "min 98.0 max 160.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/e2e_us",
            "value": 211.6,
            "range": "min 197.9 max 557.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 4868,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 104.18,
            "range": "min 89.2 max 116.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/e2e_us",
            "value": 205.56,
            "range": "min 187.6 max 217.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 108.84,
            "range": "min 85.6 max 121.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/e2e_us",
            "value": 210.1,
            "range": "min 187.9 max 230.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3068,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 114.5,
            "range": "min 99.0 max 119.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/e2e_us",
            "value": 217.09,
            "range": "min 208.2 max 240.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3352,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 143.62,
            "range": "min 127.6 max 150.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/e2e_us",
            "value": 253.3,
            "range": "min 242.0 max 265.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 10665,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 6096,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 1680.79,
            "range": "min 1663.8 max 1758.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/e2e_us",
            "value": 2000.98,
            "range": "min 1988.1 max 2308.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/compile_s",
            "value": 4.25,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 9576,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 164.97,
            "range": "min 151.3 max 181.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/e2e_us",
            "value": 275.34,
            "range": "min 260.1 max 289.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 137.13,
            "range": "min 118.7 max 198.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/e2e_us",
            "value": 250.22,
            "range": "min 227.5 max 632.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13673,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 4204,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 215.65,
            "range": "min 202.2 max 222.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/e2e_us",
            "value": 333.72,
            "range": "min 316.8 max 408.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 18025,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 4888,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 191.86,
            "range": "min 178.1 max 197.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/e2e_us",
            "value": 310.4,
            "range": "min 294.3 max 324.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 17785,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 7264,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 208.64,
            "range": "min 183.0 max 297.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/e2e_us",
            "value": 451.83,
            "range": "min 309.9 max 904.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 18601,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 7628,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/npu_us",
            "value": 97.52,
            "range": "min 78.0 max 149.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/e2e_us",
            "value": 229.73,
            "range": "min 204.3 max 368.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/xclbin_bytes",
            "value": 21225,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 176.13,
            "range": "min 162.2 max 226.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/e2e_us",
            "value": 293.36,
            "range": "min 279.8 max 657.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 534.01,
            "range": "min 527.1 max 544.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/e2e_us",
            "value": 711.25,
            "range": "min 699.9 max 721.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 48441,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 276.8,
            "range": "min 259.9 max 280.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/e2e_us",
            "value": 452.22,
            "range": "min 439.2 max 461.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/compile_s",
            "value": 2.19,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 48809,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 7832,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 107.79,
            "range": "min 86.0 max 114.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/e2e_us",
            "value": 212.67,
            "range": "min 191.1 max 224.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 107.25,
            "range": "min 95.4 max 157.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/e2e_us",
            "value": 305.77,
            "range": "min 200.6 max 583.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 124.43,
            "range": "min 92.5 max 166.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 271.32,
            "range": "min 188.5 max 556.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.32,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 12377,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 8564,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 123.8,
            "range": "min 106.1 max 144.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 225.98,
            "range": "min 209.8 max 256.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 12425,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 8640,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/npu_us",
            "value": 210.28,
            "range": "min 196.3 max 236.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/e2e_us",
            "value": 316.29,
            "range": "min 296.4 max 400.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/xclbin_bytes",
            "value": 11977,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/npu_us",
            "value": 213.15,
            "range": "min 206.3 max 239.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/e2e_us",
            "value": 337.17,
            "range": "min 324.1 max 424.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/xclbin_bytes",
            "value": 20233,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/core_elf_bytes",
            "value": 7952,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/npu_us",
            "value": 112.54,
            "range": "min 100.1 max 120.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 211.41,
            "range": "min 198.3 max 222.2 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.32,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 3132,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/npu_us",
            "value": 111.54,
            "range": "min 98.1 max 118.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/e2e_us",
            "value": 218.7,
            "range": "min 206.5 max 226.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/compile_s",
            "value": 2.28,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/npu_us",
            "value": 177.52,
            "range": "min 164.6 max 192.4 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/e2e_us",
            "value": 342.33,
            "range": "min 290.1 max 390.0 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/compile_s",
            "value": 2.27,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/npu_us",
            "value": 143.23,
            "range": "min 133.7 max 156.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/e2e_us",
            "value": 252.41,
            "range": "min 242.7 max 263.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/compile_s",
            "value": 2.28,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/npu_us",
            "value": 105.53,
            "range": "min 93.0 max 123.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/e2e_us",
            "value": 215.36,
            "range": "min 199.7 max 227.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 129.6,
            "range": "min 107.3 max 175.9 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/e2e_us",
            "value": 256.28,
            "range": "min 226.8 max 578.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3744,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 122.22,
            "range": "min 111.2 max 150.5 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/e2e_us",
            "value": 231.62,
            "range": "min 216.0 max 265.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/compile_s",
            "value": 2.19,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3744,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/npu_us",
            "value": 103.53,
            "range": "min 93.5 max 110.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/e2e_us",
            "value": 203.23,
            "range": "min 192.7 max 212.6 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/xclbin_bytes",
            "value": 10633,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/core_elf_bytes",
            "value": 7380,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/npu_us",
            "value": 108.41,
            "range": "min 93.3 max 162.3 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/e2e_us",
            "value": 213.29,
            "range": "min 200.3 max 589.7 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/compile_s",
            "value": 2.79,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/core_elf_bytes",
            "value": 4052,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/npu_us",
            "value": 111.87,
            "range": "min 102.5 max 117.8 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/e2e_us",
            "value": 220.31,
            "range": "min 214.9 max 233.1 n=50",
            "unit": "us",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/compile_s",
            "value": 2.86,
            "unit": "s",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/xclbin_bytes",
            "value": 11513,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/core_elf_bytes",
            "value": 5972,
            "unit": "bytes",
            "extra": "commit e8d062f9dc | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
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
        "date": 1790404100579,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 111.61,
            "range": "min 99.3 max 124.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/e2e_us",
            "value": 217.23,
            "range": "min 204.3 max 240.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 264.47,
            "range": "min 254.1 max 293.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/e2e_us",
            "value": 404.88,
            "range": "min 389.0 max 517.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/compile_s",
            "value": 2.05,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 33.691,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 102.82,
            "range": "min 91.4 max 171.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/e2e_us",
            "value": 208.76,
            "range": "min 196.2 max 529.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/compile_s",
            "value": 2.06,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 74,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 18.066,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 111.18,
            "range": "min 89.0 max 159.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/e2e_us",
            "value": 317.78,
            "range": "min 201.7 max 579.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 100.77,
            "range": "min 87.1 max 114.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/e2e_us",
            "value": 204,
            "range": "min 189.9 max 219.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 147.34,
            "range": "min 128.8 max 203.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/e2e_us",
            "value": 310.43,
            "range": "min 257.3 max 377.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 111.22,
            "range": "min 100.2 max 122.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/e2e_us",
            "value": 255.78,
            "range": "min 204.3 max 285.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9416,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 4216,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 110.31,
            "range": "min 99.8 max 115.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/e2e_us",
            "value": 207.81,
            "range": "min 195.7 max 220.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/compile_s",
            "value": 2.2,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 192.46,
            "range": "min 177.5 max 216.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/e2e_us",
            "value": 303.95,
            "range": "min 288.0 max 325.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 111.87,
            "range": "min 96.0 max 166.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/e2e_us",
            "value": 306.78,
            "range": "min 196.0 max 575.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 220.59,
            "range": "min 195.4 max 232.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/e2e_us",
            "value": 437.19,
            "range": "min 305.3 max 482.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/compile_s",
            "value": 2.19,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 97.32,
            "range": "min 81.8 max 114.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/e2e_us",
            "value": 200.74,
            "range": "min 186.3 max 340.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3872,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 148.47,
            "range": "min 143.9 max 159.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/e2e_us",
            "value": 358.71,
            "range": "min 301.4 max 373.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2948,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 103.73,
            "range": "min 85.9 max 109.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/e2e_us",
            "value": 207.67,
            "range": "min 187.5 max 222.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 172.69,
            "range": "min 167.8 max 187.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/e2e_us",
            "value": 285.24,
            "range": "min 277.0 max 297.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 99.9,
            "range": "min 97.1 max 119.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/e2e_us",
            "value": 208.75,
            "range": "min 197.5 max 353.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 170.64,
            "range": "min 159.5 max 181.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/e2e_us",
            "value": 280.72,
            "range": "min 272.0 max 290.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 109.19,
            "range": "min 95.9 max 146.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/e2e_us",
            "value": 217.75,
            "range": "min 204.4 max 366.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 3968,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 176.37,
            "range": "min 158.3 max 235.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/e2e_us",
            "value": 293.7,
            "range": "min 277.5 max 665.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 3060,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 105.53,
            "range": "min 94.1 max 113.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/e2e_us",
            "value": 205.97,
            "range": "min 193.8 max 221.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 3944,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 166.05,
            "range": "min 151.6 max 172.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/e2e_us",
            "value": 307.51,
            "range": "min 252.4 max 337.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 112.19,
            "range": "min 85.8 max 118.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/e2e_us",
            "value": 306.42,
            "range": "min 186.9 max 381.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/compile_s",
            "value": 4.18,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 9352,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 3932,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 195.05,
            "range": "min 181.8 max 200.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/e2e_us",
            "value": 300.48,
            "range": "min 289.4 max 308.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/compile_s",
            "value": 4.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 2992,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 124.57,
            "range": "min 111.0 max 143.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/e2e_us",
            "value": 322.46,
            "range": "min 207.4 max 425.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/compile_s",
            "value": 4.2,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 3852,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 126.97,
            "range": "min 115.9 max 133.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 222.95,
            "range": "min 213.0 max 230.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.38,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14825,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9928,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 295.74,
            "range": "min 281.7 max 357.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/e2e_us",
            "value": 461.6,
            "range": "min 425.6 max 763.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/compile_s",
            "value": 4.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 8775,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 2912,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 395.38,
            "range": "min 382.2 max 400.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/e2e_us",
            "value": 493.35,
            "range": "min 484.3 max 504.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/compile_s",
            "value": 4.58,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 11033,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 5684,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 5236.77,
            "range": "min 4881.4 max 5559.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/e2e_us",
            "value": 5986.7,
            "range": "min 5332.7 max 6356.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/compile_s",
            "value": 4.51,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 4744,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 98.36,
            "range": "min 83.1 max 107.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/e2e_us",
            "value": 200.97,
            "range": "min 184.0 max 218.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/compile_s",
            "value": 4.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 3884,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 136.19,
            "range": "min 127.0 max 140.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/e2e_us",
            "value": 248.32,
            "range": "min 240.9 max 262.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/compile_s",
            "value": 4.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 2976,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 128.57,
            "range": "min 104.9 max 147.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 327.73,
            "range": "min 210.3 max 346.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.4,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14953,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 10180,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 110.04,
            "range": "min 101.5 max 132.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/e2e_us",
            "value": 208.2,
            "range": "min 200.7 max 229.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/compile_s",
            "value": 4.19,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 3972,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 121.67,
            "range": "min 107.0 max 133.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 218.89,
            "range": "min 204.9 max 232.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.38,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14793,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9868,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 182.48,
            "range": "min 169.0 max 200.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/e2e_us",
            "value": 294.74,
            "range": "min 282.0 max 393.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/compile_s",
            "value": 4.19,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 116.2,
            "range": "min 102.6 max 123.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/e2e_us",
            "value": 218.83,
            "range": "min 208.0 max 229.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/compile_s",
            "value": 4.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 9864,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 4656,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 314.59,
            "range": "min 294.0 max 357.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/e2e_us",
            "value": 421.17,
            "range": "min 404.0 max 776.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/compile_s",
            "value": 4.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 3748,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 105.89,
            "range": "min 92.8 max 153.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/e2e_us",
            "value": 207.32,
            "range": "min 193.6 max 561.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/compile_s",
            "value": 2.2,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3864,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 145.05,
            "range": "min 131.0 max 171.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/e2e_us",
            "value": 252.76,
            "range": "min 240.7 max 280.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2956,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/npu_us",
            "value": 377.66,
            "range": "min 364.6 max 446.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/e2e_us",
            "value": 482.39,
            "range": "min 467.2 max 551.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/compile_s",
            "value": 2.35,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/xclbin_bytes",
            "value": 10921,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/core_elf_bytes",
            "value": 5620,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/npu_us",
            "value": 4588.9,
            "range": "min 4579.3 max 4673.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/e2e_us",
            "value": 4732.37,
            "range": "min 4714.7 max 5450.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/compile_s",
            "value": 2.33,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/xclbin_bytes",
            "value": 10457,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/core_elf_bytes",
            "value": 4712,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 108.19,
            "range": "min 89.1 max 113.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/e2e_us",
            "value": 207.4,
            "range": "min 187.5 max 216.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 191.34,
            "range": "min 175.2 max 201.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/e2e_us",
            "value": 300.08,
            "range": "min 284.1 max 313.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/npu_us",
            "value": 101.45,
            "range": "min 86.5 max 134.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/e2e_us",
            "value": 203.75,
            "range": "min 185.5 max 247.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/core_elf_bytes",
            "value": 3924,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/npu_us",
            "value": 203,
            "range": "min 193.2 max 211.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/e2e_us",
            "value": 354.71,
            "range": "min 343.1 max 384.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/xclbin_bytes",
            "value": 8823,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/core_elf_bytes",
            "value": 3016,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 106.28,
            "range": "min 91.2 max 130.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/e2e_us",
            "value": 223.15,
            "range": "min 200.2 max 315.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 3840,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 228.6,
            "range": "min 214.6 max 231.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/e2e_us",
            "value": 333.36,
            "range": "min 320.0 max 338.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 2916,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 129.6,
            "range": "min 111.2 max 136.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/e2e_us",
            "value": 231.51,
            "range": "min 217.0 max 277.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 9512,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 4324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 198.61,
            "range": "min 173.6 max 221.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/e2e_us",
            "value": 326.93,
            "range": "min 273.8 max 390.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 10120,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 5428,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 127.18,
            "range": "min 107.8 max 181.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/e2e_us",
            "value": 333.02,
            "range": "min 249.0 max 580.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 9464,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 4236,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 215.1,
            "range": "min 196.3 max 228.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/e2e_us",
            "value": 361.03,
            "range": "min 306.3 max 461.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 9896,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 5220,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 175.97,
            "range": "min 163.7 max 184.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/e2e_us",
            "value": 328.59,
            "range": "min 305.7 max 351.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10088,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4524,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1092.88,
            "range": "min 1081.6 max 1142.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/e2e_us",
            "value": 1293.07,
            "range": "min 1244.4 max 1620.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10088,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4524,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 140.02,
            "range": "min 125.7 max 144.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/e2e_us",
            "value": 248.99,
            "range": "min 233.9 max 262.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4076,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 121.11,
            "range": "min 105.3 max 123.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/e2e_us",
            "value": 232.02,
            "range": "min 219.5 max 247.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 9688,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 4120,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 112.04,
            "range": "min 98.4 max 116.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/e2e_us",
            "value": 217.49,
            "range": "min 206.4 max 222.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/compile_s",
            "value": 4.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 10233,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 4692,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/npu_us",
            "value": 146.39,
            "range": "min 132.3 max 185.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/e2e_us",
            "value": 249.25,
            "range": "min 235.4 max 373.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/compile_s",
            "value": 2.42,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/xclbin_bytes",
            "value": 9976,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/core_elf_bytes",
            "value": 4420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/npu_us",
            "value": 215.68,
            "range": "min 196.5 max 267.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/e2e_us",
            "value": 317.47,
            "range": "min 300.3 max 659.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/xclbin_bytes",
            "value": 11081,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/core_elf_bytes",
            "value": 6296,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/npu_us",
            "value": 114.01,
            "range": "min 105.3 max 175.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/e2e_us",
            "value": 317.93,
            "range": "min 305.2 max 579.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/compile_s",
            "value": 2.84,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/core_elf_bytes",
            "value": 3208,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/npu_us",
            "value": 110.83,
            "range": "min 97.6 max 123.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/e2e_us",
            "value": 220.41,
            "range": "min 200.2 max 228.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/compile_s",
            "value": 2.37,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/xclbin_bytes",
            "value": 10008,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/core_elf_bytes",
            "value": 4340,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 96.28,
            "range": "min 86.2 max 148.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/e2e_us",
            "value": 206.71,
            "range": "min 196.4 max 570.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9624,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 97.12,
            "range": "min 79.0 max 103.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/e2e_us",
            "value": 200.8,
            "range": "min 182.1 max 211.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8743,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2808,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 97.14,
            "range": "min 79.9 max 114.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/e2e_us",
            "value": 199.63,
            "range": "min 182.1 max 335.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 134.13,
            "range": "min 121.0 max 139.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/e2e_us",
            "value": 235.12,
            "range": "min 223.4 max 251.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/compile_s",
            "value": 4.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 153.11,
            "range": "min 141.4 max 164.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 259.02,
            "range": "min 245.3 max 279.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.29,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14329,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9540,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 658.68,
            "range": "min 650.9 max 680.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/e2e_us",
            "value": 781.26,
            "range": "min 770.0 max 802.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/compile_s",
            "value": 4.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 106.9,
            "range": "min 95.0 max 118.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/e2e_us",
            "value": 207.73,
            "range": "min 198.3 max 223.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 4104,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 111.72,
            "range": "min 101.0 max 165.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/e2e_us",
            "value": 220.64,
            "range": "min 207.0 max 582.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 9432,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 101.66,
            "range": "min 82.0 max 111.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/e2e_us",
            "value": 204.75,
            "range": "min 186.3 max 218.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 4868,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 121.58,
            "range": "min 107.8 max 127.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/e2e_us",
            "value": 224.9,
            "range": "min 208.1 max 234.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 103.86,
            "range": "min 89.8 max 115.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/e2e_us",
            "value": 208.38,
            "range": "min 193.6 max 222.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3068,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 103.78,
            "range": "min 90.6 max 107.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/e2e_us",
            "value": 202.53,
            "range": "min 191.4 max 219.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3352,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 156.49,
            "range": "min 149.0 max 217.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/e2e_us",
            "value": 365.16,
            "range": "min 286.9 max 618.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 10665,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 6096,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 1695.29,
            "range": "min 1626.9 max 1706.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/e2e_us",
            "value": 2157.17,
            "range": "min 1733.8 max 2183.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/compile_s",
            "value": 4.27,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 9576,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 164.98,
            "range": "min 148.7 max 169.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/e2e_us",
            "value": 280.11,
            "range": "min 264.8 max 299.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 142.66,
            "range": "min 128.1 max 148.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/e2e_us",
            "value": 359.44,
            "range": "min 343.5 max 371.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13673,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 4204,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 216.59,
            "range": "min 193.6 max 260.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/e2e_us",
            "value": 451.15,
            "range": "min 310.9 max 704.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 18025,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 4888,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 192.46,
            "range": "min 179.1 max 202.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/e2e_us",
            "value": 311.25,
            "range": "min 298.2 max 326.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 17785,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 7264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 189.04,
            "range": "min 175.9 max 194.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/e2e_us",
            "value": 310.16,
            "range": "min 297.6 max 442.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 18601,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 7628,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/npu_us",
            "value": 106.3,
            "range": "min 94.8 max 116.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/e2e_us",
            "value": 231.17,
            "range": "min 219.5 max 241.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/xclbin_bytes",
            "value": 21225,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 165.12,
            "range": "min 152.2 max 175.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/e2e_us",
            "value": 276.26,
            "range": "min 265.0 max 288.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 531.05,
            "range": "min 524.6 max 535.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/e2e_us",
            "value": 705.27,
            "range": "min 698.3 max 722.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 48441,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 271.83,
            "range": "min 259.6 max 280.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/e2e_us",
            "value": 443.38,
            "range": "min 433.3 max 464.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/compile_s",
            "value": 2.19,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 48809,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 7832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 109.33,
            "range": "min 97.1 max 119.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/e2e_us",
            "value": 205.52,
            "range": "min 193.2 max 225.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 108.6,
            "range": "min 99.2 max 115.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/e2e_us",
            "value": 207.38,
            "range": "min 199.6 max 218.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 112.53,
            "range": "min 96.6 max 191.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 209.86,
            "range": "min 198.7 max 754.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.3,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 12377,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 8564,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 125.41,
            "range": "min 113.0 max 140.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 318.99,
            "range": "min 290.8 max 390.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 12425,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 8640,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/npu_us",
            "value": 217.98,
            "range": "min 196.5 max 308.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/e2e_us",
            "value": 355.36,
            "range": "min 307.1 max 823.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/xclbin_bytes",
            "value": 11977,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/npu_us",
            "value": 201.36,
            "range": "min 189.9 max 222.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/e2e_us",
            "value": 330.96,
            "range": "min 307.8 max 341.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/xclbin_bytes",
            "value": 20233,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/core_elf_bytes",
            "value": 7952,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/npu_us",
            "value": 117.77,
            "range": "min 103.7 max 169.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 219.32,
            "range": "min 202.1 max 565.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.34,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 3132,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/npu_us",
            "value": 101.49,
            "range": "min 87.0 max 142.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/e2e_us",
            "value": 201.21,
            "range": "min 190.4 max 370.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/npu_us",
            "value": 181.19,
            "range": "min 159.4 max 192.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/e2e_us",
            "value": 382.25,
            "range": "min 270.9 max 422.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/npu_us",
            "value": 161.03,
            "range": "min 144.1 max 214.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/e2e_us",
            "value": 367.07,
            "range": "min 281.5 max 633.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/npu_us",
            "value": 115.08,
            "range": "min 98.4 max 133.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/e2e_us",
            "value": 223.06,
            "range": "min 201.3 max 344.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 135.31,
            "range": "min 120.0 max 235.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/e2e_us",
            "value": 355.61,
            "range": "min 262.6 max 568.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3744,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 129.14,
            "range": "min 116.6 max 139.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/e2e_us",
            "value": 230.72,
            "range": "min 216.4 max 239.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3744,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/npu_us",
            "value": 108.43,
            "range": "min 94.9 max 128.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/e2e_us",
            "value": 224.43,
            "range": "min 191.3 max 332.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/xclbin_bytes",
            "value": 10633,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/core_elf_bytes",
            "value": 7380,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/npu_us",
            "value": 112.2,
            "range": "min 97.1 max 117.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/e2e_us",
            "value": 215.9,
            "range": "min 201.7 max 230.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/compile_s",
            "value": 2.74,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/core_elf_bytes",
            "value": 4052,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/npu_us",
            "value": 126.92,
            "range": "min 109.3 max 177.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/e2e_us",
            "value": 275.75,
            "range": "min 222.6 max 590.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/compile_s",
            "value": 2.84,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/xclbin_bytes",
            "value": 11513,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/core_elf_bytes",
            "value": 5972,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
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
        "date": 1790490567837,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 103.8,
            "range": "min 89.2 max 131.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/e2e_us",
            "value": 209.57,
            "range": "min 198.8 max 351.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 265.28,
            "range": "min 257.5 max 269.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/e2e_us",
            "value": 401.32,
            "range": "min 390.3 max 410.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 33.691,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 101.59,
            "range": "min 91.9 max 110.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/e2e_us",
            "value": 207.06,
            "range": "min 198.3 max 215.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 74,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 18.066,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 104.33,
            "range": "min 91.7 max 129.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/e2e_us",
            "value": 213.32,
            "range": "min 197.7 max 361.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 96.22,
            "range": "min 79.6 max 102.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/e2e_us",
            "value": 203.03,
            "range": "min 184.3 max 218.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 145.82,
            "range": "min 134.7 max 170.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/e2e_us",
            "value": 317,
            "range": "min 280.6 max 399.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 103.2,
            "range": "min 84.1 max 108.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/e2e_us",
            "value": 210.51,
            "range": "min 190.1 max 223.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9416,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 4216,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 112.12,
            "range": "min 88.9 max 119.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/e2e_us",
            "value": 312.15,
            "range": "min 188.1 max 385.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/compile_s",
            "value": 2.19,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 185.5,
            "range": "min 173.6 max 219.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/e2e_us",
            "value": 296.41,
            "range": "min 283.7 max 409.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/compile_s",
            "value": 2.22,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 102.95,
            "range": "min 90.1 max 123.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/e2e_us",
            "value": 204.4,
            "range": "min 190.6 max 300.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/compile_s",
            "value": 2.2,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 203.84,
            "range": "min 190.4 max 263.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/e2e_us",
            "value": 323.59,
            "range": "min 306.8 max 678.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 105.9,
            "range": "min 91.6 max 111.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/e2e_us",
            "value": 209.44,
            "range": "min 193.6 max 217.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3872,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 144.91,
            "range": "min 130.1 max 148.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/e2e_us",
            "value": 250.88,
            "range": "min 233.5 max 258.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2948,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 102.35,
            "range": "min 89.6 max 118.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/e2e_us",
            "value": 209.8,
            "range": "min 197.4 max 229.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 178.25,
            "range": "min 157.2 max 189.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/e2e_us",
            "value": 401.68,
            "range": "min 267.2 max 454.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/compile_s",
            "value": 2.06,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 102.75,
            "range": "min 89.4 max 107.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/e2e_us",
            "value": 206.21,
            "range": "min 193.7 max 214.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 164.38,
            "range": "min 150.4 max 221.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/e2e_us",
            "value": 276.95,
            "range": "min 262.4 max 589.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 110.05,
            "range": "min 91.1 max 112.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/e2e_us",
            "value": 210.91,
            "range": "min 191.1 max 217.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 3968,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 179.57,
            "range": "min 165.0 max 186.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/e2e_us",
            "value": 289.3,
            "range": "min 272.6 max 299.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 3060,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 96.41,
            "range": "min 81.2 max 110.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/e2e_us",
            "value": 195.24,
            "range": "min 179.3 max 210.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 3944,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 174.49,
            "range": "min 163.1 max 181.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/e2e_us",
            "value": 324.64,
            "range": "min 302.4 max 350.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 103.54,
            "range": "min 84.7 max 109.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/e2e_us",
            "value": 200.55,
            "range": "min 180.5 max 208.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/compile_s",
            "value": 4.2,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 9352,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 3932,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 190.93,
            "range": "min 172.2 max 207.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/e2e_us",
            "value": 406.72,
            "range": "min 277.7 max 495.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/compile_s",
            "value": 4.19,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 2992,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 109.76,
            "range": "min 97.2 max 128.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/e2e_us",
            "value": 208.1,
            "range": "min 199.2 max 233.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/compile_s",
            "value": 4.2,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 3852,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 119.74,
            "range": "min 105.9 max 124.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 219.06,
            "range": "min 207.3 max 230.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.4,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14825,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9928,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 279.5,
            "range": "min 267.0 max 325.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/e2e_us",
            "value": 386.36,
            "range": "min 375.2 max 429.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/compile_s",
            "value": 4.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 8775,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 2912,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 395.02,
            "range": "min 381.7 max 411.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/e2e_us",
            "value": 491.43,
            "range": "min 476.9 max 622.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/compile_s",
            "value": 4.58,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 11033,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 5684,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 4787.8,
            "range": "min 4741.6 max 4855.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/e2e_us",
            "value": 5290.06,
            "range": "min 5114.8 max 5409.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/compile_s",
            "value": 4.58,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 4744,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 103.7,
            "range": "min 88.6 max 107.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/e2e_us",
            "value": 202.62,
            "range": "min 188.2 max 209.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/compile_s",
            "value": 4.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 3884,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 145.06,
            "range": "min 129.6 max 163.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/e2e_us",
            "value": 255.55,
            "range": "min 235.9 max 438.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/compile_s",
            "value": 4.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 2976,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 116.11,
            "range": "min 99.1 max 123.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 211.7,
            "range": "min 196.5 max 221.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.39,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14953,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 10180,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 108.18,
            "range": "min 81.5 max 112.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/e2e_us",
            "value": 205.33,
            "range": "min 176.1 max 210.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/compile_s",
            "value": 4.2,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 3972,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 125.34,
            "range": "min 113.0 max 131.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 223.97,
            "range": "min 210.7 max 231.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.35,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14793,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9868,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 172.58,
            "range": "min 161.3 max 205.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/e2e_us",
            "value": 279.66,
            "range": "min 268.9 max 312.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/compile_s",
            "value": 4.2,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 122.8,
            "range": "min 103.4 max 180.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/e2e_us",
            "value": 230.58,
            "range": "min 207.3 max 583.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/compile_s",
            "value": 4.22,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 9864,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 4656,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 302.08,
            "range": "min 291.6 max 309.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/e2e_us",
            "value": 412.64,
            "range": "min 397.0 max 422.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/compile_s",
            "value": 4.23,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 3748,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 102.65,
            "range": "min 81.6 max 108.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/e2e_us",
            "value": 200.05,
            "range": "min 181.6 max 211.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3864,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 146.48,
            "range": "min 136.7 max 156.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/e2e_us",
            "value": 258.82,
            "range": "min 247.4 max 279.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2956,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/npu_us",
            "value": 379.53,
            "range": "min 373.5 max 398.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/e2e_us",
            "value": 487.81,
            "range": "min 481.1 max 502.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/compile_s",
            "value": 2.37,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/xclbin_bytes",
            "value": 10921,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/core_elf_bytes",
            "value": 5620,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/npu_us",
            "value": 4885.99,
            "range": "min 4638.4 max 5394.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/e2e_us",
            "value": 5686.7,
            "range": "min 5093.7 max 6235.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/compile_s",
            "value": 2.38,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/xclbin_bytes",
            "value": 10457,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/core_elf_bytes",
            "value": 4712,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 110.46,
            "range": "min 88.3 max 122.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/e2e_us",
            "value": 209.92,
            "range": "min 187.5 max 220.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 198.91,
            "range": "min 175.3 max 264.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/e2e_us",
            "value": 415.26,
            "range": "min 291.4 max 748.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/npu_us",
            "value": 110.25,
            "range": "min 97.1 max 118.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/e2e_us",
            "value": 210.43,
            "range": "min 194.6 max 227.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/core_elf_bytes",
            "value": 3924,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/npu_us",
            "value": 203.25,
            "range": "min 187.1 max 253.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/e2e_us",
            "value": 316.7,
            "range": "min 303.2 max 667.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/xclbin_bytes",
            "value": 8823,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/core_elf_bytes",
            "value": 3016,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 110.41,
            "range": "min 99.4 max 114.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/e2e_us",
            "value": 209.9,
            "range": "min 198.3 max 218.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 3840,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 229.18,
            "range": "min 215.7 max 234.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/e2e_us",
            "value": 335.24,
            "range": "min 321.6 max 344.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 2916,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 132.17,
            "range": "min 127.7 max 173.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/e2e_us",
            "value": 333.42,
            "range": "min 231.5 max 388.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 9512,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 4324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 193.16,
            "range": "min 185.2 max 202.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/e2e_us",
            "value": 331.44,
            "range": "min 313.2 max 362.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 10120,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 5428,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 117.14,
            "range": "min 106.4 max 136.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/e2e_us",
            "value": 326.26,
            "range": "min 210.0 max 419.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 9464,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 4236,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 210.06,
            "range": "min 184.5 max 240.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/e2e_us",
            "value": 419.02,
            "range": "min 293.6 max 580.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 9896,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 5220,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 161.19,
            "range": "min 147.3 max 175.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/e2e_us",
            "value": 268.46,
            "range": "min 255.8 max 285.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/compile_s",
            "value": 2.29,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10088,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4524,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1090.58,
            "range": "min 1079.0 max 1096.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/e2e_us",
            "value": 1250.36,
            "range": "min 1240.9 max 1295.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/compile_s",
            "value": 2.27,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10088,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4524,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 132.44,
            "range": "min 122.9 max 140.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/e2e_us",
            "value": 243.87,
            "range": "min 235.4 max 264.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4076,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 102.12,
            "range": "min 95.0 max 120.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/e2e_us",
            "value": 218.92,
            "range": "min 210.1 max 237.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/compile_s",
            "value": 2.29,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 9688,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 4120,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 111.69,
            "range": "min 97.4 max 116.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/e2e_us",
            "value": 215.62,
            "range": "min 202.0 max 223.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/compile_s",
            "value": 4.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 10233,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 4692,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/npu_us",
            "value": 163.75,
            "range": "min 140.7 max 205.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/e2e_us",
            "value": 268.45,
            "range": "min 249.6 max 621.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/compile_s",
            "value": 2.43,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/xclbin_bytes",
            "value": 9976,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/core_elf_bytes",
            "value": 4420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/npu_us",
            "value": 208.57,
            "range": "min 195.7 max 213.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/e2e_us",
            "value": 308.76,
            "range": "min 298.6 max 315.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/xclbin_bytes",
            "value": 11081,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/core_elf_bytes",
            "value": 6296,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/npu_us",
            "value": 108.92,
            "range": "min 95.5 max 115.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/e2e_us",
            "value": 213.55,
            "range": "min 200.5 max 226.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/compile_s",
            "value": 2.84,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/core_elf_bytes",
            "value": 3208,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/npu_us",
            "value": 120.03,
            "range": "min 107.0 max 129.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/e2e_us",
            "value": 226.95,
            "range": "min 216.7 max 235.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/compile_s",
            "value": 2.4,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/xclbin_bytes",
            "value": 10008,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/core_elf_bytes",
            "value": 4340,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 100.62,
            "range": "min 82.9 max 125.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/e2e_us",
            "value": 207.95,
            "range": "min 190.8 max 368.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/compile_s",
            "value": 2.28,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9624,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 104.11,
            "range": "min 94.7 max 118.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/e2e_us",
            "value": 205.39,
            "range": "min 194.8 max 220.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8743,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2808,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 104.36,
            "range": "min 92.2 max 816.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/e2e_us",
            "value": 200.44,
            "range": "min 188.5 max 917.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 140.27,
            "range": "min 129.7 max 149.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/e2e_us",
            "value": 237.21,
            "range": "min 225.2 max 264.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/compile_s",
            "value": 4.19,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 157.59,
            "range": "min 136.6 max 162.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 253.94,
            "range": "min 236.6 max 262.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.32,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14329,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9540,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 650.31,
            "range": "min 634.9 max 713.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/e2e_us",
            "value": 771.64,
            "range": "min 748.3 max 1149.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/compile_s",
            "value": 4.22,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 106.9,
            "range": "min 94.2 max 126.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/e2e_us",
            "value": 208.84,
            "range": "min 198.2 max 299.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 4104,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 118.99,
            "range": "min 103.9 max 122.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/e2e_us",
            "value": 220.72,
            "range": "min 206.0 max 238.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 9432,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 103.48,
            "range": "min 95.3 max 117.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/e2e_us",
            "value": 203.44,
            "range": "min 192.6 max 249.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 4868,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 119.58,
            "range": "min 107.5 max 125.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/e2e_us",
            "value": 322.37,
            "range": "min 307.2 max 332.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 108.73,
            "range": "min 95.3 max 111.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/e2e_us",
            "value": 208.67,
            "range": "min 196.9 max 220.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3068,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 126.02,
            "range": "min 108.7 max 130.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/e2e_us",
            "value": 329.51,
            "range": "min 311.0 max 395.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3352,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 145.47,
            "range": "min 132.6 max 148.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/e2e_us",
            "value": 251.3,
            "range": "min 236.8 max 262.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 10665,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 6096,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 1656.29,
            "range": "min 1642.1 max 1665.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/e2e_us",
            "value": 1769.95,
            "range": "min 1751.8 max 1784.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/compile_s",
            "value": 4.28,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 9576,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 164.15,
            "range": "min 154.0 max 180.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/e2e_us",
            "value": 281.56,
            "range": "min 269.3 max 318.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 140.31,
            "range": "min 121.4 max 197.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/e2e_us",
            "value": 258.24,
            "range": "min 232.5 max 624.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13673,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 4204,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 212.43,
            "range": "min 193.0 max 233.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/e2e_us",
            "value": 342.01,
            "range": "min 318.6 max 383.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 18025,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 4888,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 197.08,
            "range": "min 189.2 max 201.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/e2e_us",
            "value": 311.79,
            "range": "min 301.7 max 319.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 17785,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 7264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 189.18,
            "range": "min 176.6 max 202.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/e2e_us",
            "value": 312.13,
            "range": "min 296.5 max 329.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 18601,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 7628,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/npu_us",
            "value": 98.74,
            "range": "min 84.8 max 105.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/e2e_us",
            "value": 227.96,
            "range": "min 217.4 max 243.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/xclbin_bytes",
            "value": 21225,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 164.11,
            "range": "min 151.9 max 170.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/e2e_us",
            "value": 280.85,
            "range": "min 267.1 max 296.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 533.58,
            "range": "min 522.5 max 545.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/e2e_us",
            "value": 712.38,
            "range": "min 692.9 max 753.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 48441,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 268.14,
            "range": "min 255.1 max 315.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/e2e_us",
            "value": 442.93,
            "range": "min 428.0 max 858.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 48809,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 7832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 100.24,
            "range": "min 96.2 max 106.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/e2e_us",
            "value": 203.66,
            "range": "min 199.2 max 211.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 108.52,
            "range": "min 96.7 max 120.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/e2e_us",
            "value": 214.61,
            "range": "min 202.1 max 226.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 117.72,
            "range": "min 104.1 max 227.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 308.83,
            "range": "min 210.4 max 803.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.3,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 12377,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 8564,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 124.3,
            "range": "min 111.1 max 147.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 224.21,
            "range": "min 213.5 max 411.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.27,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 12425,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 8640,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/npu_us",
            "value": 212.52,
            "range": "min 199.1 max 218.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/e2e_us",
            "value": 319.89,
            "range": "min 309.8 max 337.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/compile_s",
            "value": 2.27,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/xclbin_bytes",
            "value": 11977,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/npu_us",
            "value": 211.24,
            "range": "min 201.4 max 258.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/e2e_us",
            "value": 329.88,
            "range": "min 322.2 max 693.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/xclbin_bytes",
            "value": 20233,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/core_elf_bytes",
            "value": 7952,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/npu_us",
            "value": 128.48,
            "range": "min 115.2 max 175.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 330.44,
            "range": "min 218.7 max 581.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.34,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 3132,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/npu_us",
            "value": 105.65,
            "range": "min 88.7 max 116.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/e2e_us",
            "value": 210.45,
            "range": "min 196.2 max 260.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/compile_s",
            "value": 2.29,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/npu_us",
            "value": 177.07,
            "range": "min 163.4 max 184.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/e2e_us",
            "value": 279.24,
            "range": "min 266.1 max 289.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/compile_s",
            "value": 2.3,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/npu_us",
            "value": 146.81,
            "range": "min 131.2 max 156.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/e2e_us",
            "value": 255.02,
            "range": "min 236.9 max 271.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/compile_s",
            "value": 2.28,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/npu_us",
            "value": 113.16,
            "range": "min 96.3 max 119.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/e2e_us",
            "value": 215.71,
            "range": "min 200.1 max 222.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/compile_s",
            "value": 2.27,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 137.18,
            "range": "min 115.3 max 160.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/e2e_us",
            "value": 340.21,
            "range": "min 224.0 max 436.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/compile_s",
            "value": 2.19,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3744,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 135.87,
            "range": "min 122.9 max 147.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/e2e_us",
            "value": 290.27,
            "range": "min 258.5 max 311.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3744,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/npu_us",
            "value": 114.98,
            "range": "min 96.7 max 169.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/e2e_us",
            "value": 257.4,
            "range": "min 198.0 max 558.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/xclbin_bytes",
            "value": 10633,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/core_elf_bytes",
            "value": 7380,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/npu_us",
            "value": 112.83,
            "range": "min 93.6 max 132.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/e2e_us",
            "value": 257.75,
            "range": "min 202.9 max 341.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/compile_s",
            "value": 2.77,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/core_elf_bytes",
            "value": 4052,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/npu_us",
            "value": 126.17,
            "range": "min 111.3 max 168.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/e2e_us",
            "value": 293.86,
            "range": "min 245.9 max 579.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/compile_s",
            "value": 2.84,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/xclbin_bytes",
            "value": 11513,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/core_elf_bytes",
            "value": 5972,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
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
        "date": 1790577363145,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 105.5,
            "range": "min 90.3 max 124.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/e2e_us",
            "value": 216.49,
            "range": "min 200.3 max 239.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 269.01,
            "range": "min 262.3 max 323.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/e2e_us",
            "value": 411.34,
            "range": "min 402.4 max 505.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/compile_s",
            "value": 2.05,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 138,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 33.691,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 103.2,
            "range": "min 94.5 max 115.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/e2e_us",
            "value": 206.75,
            "range": "min 197.5 max 222.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/compile_s",
            "value": 2.05,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 74,
            "unit": "cycles",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 18.066,
            "unit": "cycles/1k-ops",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 101.77,
            "range": "min 84.6 max 116.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/e2e_us",
            "value": 204.02,
            "range": "min 183.1 max 216.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 98.98,
            "range": "min 82.9 max 104.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/e2e_us",
            "value": 203.61,
            "range": "min 189.2 max 215.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 135.1,
            "range": "min 118.6 max 141.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/e2e_us",
            "value": 245.85,
            "range": "min 229.2 max 253.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 104.56,
            "range": "min 87.0 max 125.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/e2e_us",
            "value": 214.06,
            "range": "min 195.5 max 284.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9416,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 4216,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 103.64,
            "range": "min 95.8 max 109.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/e2e_us",
            "value": 203.48,
            "range": "min 194.1 max 215.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/compile_s",
            "value": 2.19,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 194.09,
            "range": "min 183.3 max 200.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/e2e_us",
            "value": 309.74,
            "range": "min 298.9 max 319.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 100.83,
            "range": "min 86.5 max 123.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/e2e_us",
            "value": 202.35,
            "range": "min 190.6 max 326.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 208.88,
            "range": "min 198.3 max 253.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/e2e_us",
            "value": 320.88,
            "range": "min 311.0 max 364.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 111.7,
            "range": "min 91.6 max 192.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/e2e_us",
            "value": 310.41,
            "range": "min 257.8 max 656.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3872,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 138.3,
            "range": "min 125.6 max 143.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/e2e_us",
            "value": 248.6,
            "range": "min 233.6 max 257.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2948,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 111.76,
            "range": "min 86.2 max 165.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/e2e_us",
            "value": 314.2,
            "range": "min 204.0 max 576.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 173.01,
            "range": "min 153.4 max 177.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/e2e_us",
            "value": 285.01,
            "range": "min 265.2 max 292.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 107.57,
            "range": "min 93.4 max 127.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/e2e_us",
            "value": 210.51,
            "range": "min 195.8 max 271.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 168.63,
            "range": "min 159.4 max 184.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/e2e_us",
            "value": 282.36,
            "range": "min 273.2 max 297.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 111.56,
            "range": "min 98.0 max 123.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/e2e_us",
            "value": 310.51,
            "range": "min 297.2 max 338.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 3968,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 177.63,
            "range": "min 153.3 max 232.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/e2e_us",
            "value": 396.77,
            "range": "min 268.2 max 781.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 3060,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 99.61,
            "range": "min 81.6 max 108.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/e2e_us",
            "value": 198.5,
            "range": "min 181.8 max 211.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 3944,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 172.44,
            "range": "min 151.9 max 197.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/e2e_us",
            "value": 384.13,
            "range": "min 265.6 max 515.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 108.72,
            "range": "min 95.2 max 159.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/e2e_us",
            "value": 208.59,
            "range": "min 192.8 max 554.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/compile_s",
            "value": 4.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 9352,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 3932,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 187.91,
            "range": "min 179.5 max 192.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/e2e_us",
            "value": 299.39,
            "range": "min 289.2 max 315.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/compile_s",
            "value": 4.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 2992,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 109.78,
            "range": "min 96.7 max 206.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/e2e_us",
            "value": 209.93,
            "range": "min 198.8 max 800.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/compile_s",
            "value": 4.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 3852,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 128.22,
            "range": "min 115.3 max 132.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 225.51,
            "range": "min 213.5 max 238.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.36,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14825,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9928,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 279.05,
            "range": "min 267.7 max 283.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/e2e_us",
            "value": 390.72,
            "range": "min 380.4 max 396.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/compile_s",
            "value": 4.19,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 8775,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 2912,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 387.91,
            "range": "min 379.9 max 394.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/e2e_us",
            "value": 491.99,
            "range": "min 481.1 max 502.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/compile_s",
            "value": 4.53,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 11033,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 5684,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 5025.67,
            "range": "min 4752.2 max 5939.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/e2e_us",
            "value": 5886.22,
            "range": "min 4927.9 max 6891.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/compile_s",
            "value": 4.51,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 10537,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 4744,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 99.36,
            "range": "min 92.0 max 150.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/e2e_us",
            "value": 200.61,
            "range": "min 191.6 max 252.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/compile_s",
            "value": 4.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 3884,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 147.5,
            "range": "min 137.7 max 158.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/e2e_us",
            "value": 294.96,
            "range": "min 277.3 max 323.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/compile_s",
            "value": 4.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 2976,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 113.22,
            "range": "min 111.2 max 120.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 211.82,
            "range": "min 207.0 max 228.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.37,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14953,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 10180,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 109.73,
            "range": "min 84.9 max 115.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/e2e_us",
            "value": 208.68,
            "range": "min 184.3 max 220.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/compile_s",
            "value": 4.18,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 3972,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 119.61,
            "range": "min 107.0 max 147.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 253.47,
            "range": "min 215.3 max 356.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.33,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14793,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9868,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 175.64,
            "range": "min 164.3 max 193.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/e2e_us",
            "value": 286.09,
            "range": "min 275.2 max 300.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/compile_s",
            "value": 4.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 115.28,
            "range": "min 101.7 max 148.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/e2e_us",
            "value": 211.38,
            "range": "min 200.2 max 351.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/compile_s",
            "value": 4.18,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 9864,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 4656,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 310.94,
            "range": "min 293.6 max 317.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/e2e_us",
            "value": 418.47,
            "range": "min 399.6 max 428.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/compile_s",
            "value": 4.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 3748,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 98.69,
            "range": "min 87.3 max 108.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/e2e_us",
            "value": 198.95,
            "range": "min 189.0 max 212.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3864,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 145,
            "range": "min 133.6 max 151.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/e2e_us",
            "value": 250.76,
            "range": "min 240.4 max 257.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/compile_s",
            "value": 2.16,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2956,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/npu_us",
            "value": 392.98,
            "range": "min 382.0 max 432.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/e2e_us",
            "value": 589.25,
            "range": "min 527.1 max 861.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/compile_s",
            "value": 2.35,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/xclbin_bytes",
            "value": 10921,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/core_elf_bytes",
            "value": 5620,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/npu_us",
            "value": 4644.17,
            "range": "min 4606.4 max 5229.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/e2e_us",
            "value": 5167.21,
            "range": "min 4805.5 max 5805.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/compile_s",
            "value": 2.35,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/xclbin_bytes",
            "value": 10457,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/core_elf_bytes",
            "value": 4712,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 109.06,
            "range": "min 96.7 max 114.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/e2e_us",
            "value": 209.19,
            "range": "min 200.9 max 222.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 185.16,
            "range": "min 170.6 max 198.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/e2e_us",
            "value": 298.25,
            "range": "min 286.2 max 316.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/compile_s",
            "value": 2.22,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/npu_us",
            "value": 102.96,
            "range": "min 90.2 max 154.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/e2e_us",
            "value": 205.52,
            "range": "min 193.8 max 258.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/core_elf_bytes",
            "value": 3924,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/npu_us",
            "value": 195.34,
            "range": "min 181.3 max 203.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/e2e_us",
            "value": 309.06,
            "range": "min 295.3 max 322.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/xclbin_bytes",
            "value": 8823,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/core_elf_bytes",
            "value": 3016,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 109.63,
            "range": "min 95.2 max 112.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/e2e_us",
            "value": 211.3,
            "range": "min 196.7 max 223.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 3840,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 229.39,
            "range": "min 205.1 max 271.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/e2e_us",
            "value": 376.85,
            "range": "min 316.3 max 696.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 2916,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 127.67,
            "range": "min 105.7 max 139.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/e2e_us",
            "value": 227.98,
            "range": "min 207.5 max 303.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 9512,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 4324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 190,
            "range": "min 176.4 max 196.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/e2e_us",
            "value": 293.23,
            "range": "min 280.4 max 303.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 10120,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 5428,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 115.76,
            "range": "min 98.8 max 157.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/e2e_us",
            "value": 319.99,
            "range": "min 207.7 max 595.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 9464,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 4236,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 207.1,
            "range": "min 195.4 max 256.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/e2e_us",
            "value": 311.42,
            "range": "min 299.5 max 684.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 9896,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 5220,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 158.51,
            "range": "min 144.7 max 175.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/e2e_us",
            "value": 267.03,
            "range": "min 250.7 max 286.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10088,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4524,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1097.92,
            "range": "min 1085.1 max 1216.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/e2e_us",
            "value": 1295.52,
            "range": "min 1250.3 max 1384.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10088,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4524,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 133.41,
            "range": "min 126.7 max 159.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/e2e_us",
            "value": 246.47,
            "range": "min 237.6 max 265.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/compile_s",
            "value": 2.21,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4076,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 117.58,
            "range": "min 101.7 max 122.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/e2e_us",
            "value": 226.18,
            "range": "min 214.1 max 238.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 9688,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 4120,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 113.59,
            "range": "min 93.6 max 127.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/e2e_us",
            "value": 319.27,
            "range": "min 256.8 max 376.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/compile_s",
            "value": 4.1,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 10233,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 4692,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/npu_us",
            "value": 152.73,
            "range": "min 140.2 max 205.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/e2e_us",
            "value": 253.01,
            "range": "min 244.8 max 510.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/compile_s",
            "value": 2.38,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/xclbin_bytes",
            "value": 9976,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/core_elf_bytes",
            "value": 4420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/npu_us",
            "value": 224.56,
            "range": "min 199.3 max 231.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/e2e_us",
            "value": 434.61,
            "range": "min 309.3 max 446.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/xclbin_bytes",
            "value": 11081,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/core_elf_bytes",
            "value": 6296,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/npu_us",
            "value": 109.54,
            "range": "min 100.0 max 116.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/e2e_us",
            "value": 212.04,
            "range": "min 201.4 max 222.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/compile_s",
            "value": 2.87,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/core_elf_bytes",
            "value": 3208,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/npu_us",
            "value": 109.44,
            "range": "min 99.8 max 123.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/e2e_us",
            "value": 218.59,
            "range": "min 204.6 max 228.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/compile_s",
            "value": 2.34,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/xclbin_bytes",
            "value": 10008,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/core_elf_bytes",
            "value": 4340,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 103.3,
            "range": "min 90.0 max 115.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/e2e_us",
            "value": 212.09,
            "range": "min 196.7 max 224.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9624,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 104.91,
            "range": "min 85.5 max 108.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/e2e_us",
            "value": 205.99,
            "range": "min 187.9 max 216.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/compile_s",
            "value": 2.1,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8743,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2808,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 103.79,
            "range": "min 92.3 max 108.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/e2e_us",
            "value": 201.79,
            "range": "min 188.9 max 211.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 130.93,
            "range": "min 114.6 max 228.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/e2e_us",
            "value": 229.08,
            "range": "min 211.1 max 768.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/compile_s",
            "value": 4.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 157.17,
            "range": "min 145.6 max 177.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/e2e_us",
            "value": 254.72,
            "range": "min 249.3 max 401.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/compile_s",
            "value": 4.27,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14329,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9540,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 648.2,
            "range": "min 635.6 max 657.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/e2e_us",
            "value": 770.83,
            "range": "min 750.8 max 780.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/compile_s",
            "value": 4.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 110.11,
            "range": "min 98.4 max 116.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/e2e_us",
            "value": 210.99,
            "range": "min 201.3 max 220.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 4104,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 128.38,
            "range": "min 112.5 max 174.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/e2e_us",
            "value": 332.7,
            "range": "min 290.4 max 568.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 9432,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 106.05,
            "range": "min 81.0 max 167.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/e2e_us",
            "value": 277.8,
            "range": "min 186.2 max 585.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 4868,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 103.58,
            "range": "min 89.9 max 107.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/e2e_us",
            "value": 208.9,
            "range": "min 199.9 max 222.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/compile_s",
            "value": 2.08,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 101.95,
            "range": "min 87.4 max 107.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/e2e_us",
            "value": 204.8,
            "range": "min 189.4 max 212.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/compile_s",
            "value": 2.07,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3068,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 122.57,
            "range": "min 100.9 max 129.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/e2e_us",
            "value": 226.27,
            "range": "min 209.5 max 340.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3352,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 146.41,
            "range": "min 133.9 max 153.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/e2e_us",
            "value": 250.85,
            "range": "min 241.7 max 257.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/compile_s",
            "value": 2.22,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 10665,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 6096,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 1706.43,
            "range": "min 1652.2 max 1782.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/e2e_us",
            "value": 2170.24,
            "range": "min 1766.9 max 2323.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/compile_s",
            "value": 4.27,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 9576,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 169.3,
            "range": "min 157.3 max 178.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/e2e_us",
            "value": 289.47,
            "range": "min 271.7 max 307.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 142.13,
            "range": "min 129.6 max 148.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/e2e_us",
            "value": 256.75,
            "range": "min 247.1 max 268.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13673,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 4204,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 209.91,
            "range": "min 195.9 max 217.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/e2e_us",
            "value": 331.61,
            "range": "min 317.2 max 347.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/compile_s",
            "value": 2.14,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 18025,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 4888,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 189.68,
            "range": "min 172.5 max 195.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/e2e_us",
            "value": 307.47,
            "range": "min 289.5 max 321.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 17785,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 7264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 194.02,
            "range": "min 177.8 max 212.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/e2e_us",
            "value": 334.52,
            "range": "min 301.9 max 405.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/compile_s",
            "value": 2.2,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 18601,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 7628,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/npu_us",
            "value": 103.89,
            "range": "min 83.2 max 116.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/e2e_us",
            "value": 227.68,
            "range": "min 206.6 max 248.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/compile_s",
            "value": 2.09,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/xclbin_bytes",
            "value": 21225,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 174.25,
            "range": "min 150.8 max 178.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/e2e_us",
            "value": 284.52,
            "range": "min 257.6 max 295.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/compile_s",
            "value": 2.11,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 537.82,
            "range": "min 528.9 max 547.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/e2e_us",
            "value": 715.24,
            "range": "min 706.8 max 732.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/compile_s",
            "value": 2.15,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 48441,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 276.85,
            "range": "min 264.3 max 288.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/e2e_us",
            "value": 453.61,
            "range": "min 443.3 max 467.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 48809,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 7832,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 112.24,
            "range": "min 104.6 max 122.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/e2e_us",
            "value": 307.62,
            "range": "min 295.9 max 344.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/compile_s",
            "value": 2.13,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 111.71,
            "range": "min 98.6 max 116.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/e2e_us",
            "value": 209.36,
            "range": "min 196.8 max 215.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/compile_s",
            "value": 2.12,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 114.1,
            "range": "min 103.3 max 127.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 213.57,
            "range": "min 200.2 max 226.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.3,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 12377,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 8564,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 128.62,
            "range": "min 112.6 max 185.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 261.93,
            "range": "min 219.2 max 556.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 12425,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 8640,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/npu_us",
            "value": 212.8,
            "range": "min 200.0 max 244.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/e2e_us",
            "value": 317.81,
            "range": "min 300.2 max 558.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/xclbin_bytes",
            "value": 11977,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/npu_us",
            "value": 221.41,
            "range": "min 208.5 max 226.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/e2e_us",
            "value": 458.56,
            "range": "min 443.0 max 482.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/xclbin_bytes",
            "value": 20233,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/core_elf_bytes",
            "value": 7952,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/npu_us",
            "value": 120.29,
            "range": "min 109.4 max 127.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/e2e_us",
            "value": 219.31,
            "range": "min 207.4 max 239.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/compile_s",
            "value": 2.33,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 8967,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 3132,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/npu_us",
            "value": 101.87,
            "range": "min 78.6 max 106.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/e2e_us",
            "value": 206.27,
            "range": "min 181.9 max 223.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/compile_s",
            "value": 2.25,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/npu_us",
            "value": 187.64,
            "range": "min 178.6 max 199.6 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/e2e_us",
            "value": 401.55,
            "range": "min 382.6 max 423.2 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/compile_s",
            "value": 2.27,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/npu_us",
            "value": 141.41,
            "range": "min 125.8 max 154.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/e2e_us",
            "value": 241.88,
            "range": "min 227.9 max 257.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/compile_s",
            "value": 2.24,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/npu_us",
            "value": 110.88,
            "range": "min 97.2 max 128.5 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/e2e_us",
            "value": 213.21,
            "range": "min 203.6 max 230.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/compile_s",
            "value": 2.26,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/xclbin_bytes",
            "value": 10345,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/core_elf_bytes",
            "value": 5324,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 121.46,
            "range": "min 104.5 max 130.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/e2e_us",
            "value": 224.31,
            "range": "min 205.8 max 232.4 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/compile_s",
            "value": 2.17,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3744,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 122.5,
            "range": "min 109.9 max 151.7 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/e2e_us",
            "value": 224.35,
            "range": "min 214.3 max 369.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/compile_s",
            "value": 2.18,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3744,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/npu_us",
            "value": 107.9,
            "range": "min 94.0 max 129.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/e2e_us",
            "value": 206.94,
            "range": "min 194.3 max 283.3 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/compile_s",
            "value": 2.23,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/xclbin_bytes",
            "value": 10633,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/core_elf_bytes",
            "value": 7380,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/npu_us",
            "value": 103.84,
            "range": "min 90.9 max 113.9 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/e2e_us",
            "value": 213.97,
            "range": "min 197.3 max 226.0 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/compile_s",
            "value": 2.79,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/core_elf_bytes",
            "value": 4052,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/npu_us",
            "value": 119.72,
            "range": "min 96.3 max 122.8 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/e2e_us",
            "value": 222.56,
            "range": "min 199.0 max 230.1 n=50",
            "unit": "us",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/compile_s",
            "value": 2.81,
            "unit": "s",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/xclbin_bytes",
            "value": 11513,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/core_elf_bytes",
            "value": 5972,
            "unit": "bytes",
            "extra": "commit a82bb55c19 | peano 22.0.0.2026092101+0006955e | device NPU Krackan 1 | pmode default"
          }
        ]
      }
    ]
  }
}