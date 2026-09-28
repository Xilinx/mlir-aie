window.BENCHMARK_DATA = {
  "lastUpdate": 1790632664378,
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
          "id": "d53582d3e0f9f8a2b77695bbf4abbd7766d5584a",
          "message": "Single-core Kernel Optimizations and Tooling (#3801)\n\nCo-authored-by: Claude Opus 5 <noreply@anthropic.com>\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
          "timestamp": "2026-09-28T20:24:43Z",
          "url": "https://github.com/Xilinx/mlir-aie/commit/d53582d3e0f9f8a2b77695bbf4abbd7766d5584a"
        },
        "date": 1790632661917,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 138,
            "range": "median 138 max 138 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 121.26,
            "range": "± 6.3; min 100.8 max 181.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 138,
            "range": "median 138 max 138 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 67.383,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 276.34,
            "range": "± 6.3; min 255.3 max 317.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 138,
            "range": "median 138 max 138 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 33.691,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 113.32,
            "range": "± 2.6; min 97.2 max 124.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 74,
            "range": "median 74 max 74 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 18.066,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 109.63,
            "range": "± 1.0; min 85.8 max 112.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/cycles",
            "value": 42,
            "range": "median 42 max 42 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/cycles_per_kop",
            "value": 41.016,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/npu_us",
            "value": 105.38,
            "range": "± 1.5; min 83.3 max 112.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/core_elf_bytes",
            "value": 4024,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/cycles",
            "value": 46,
            "range": "median 46 max 46 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/cycles_per_kop",
            "value": 44.922,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 104.66,
            "range": "± 1.8; min 95.4 max 108.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/cycles",
            "value": 46,
            "range": "median 46 max 47 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/cycles_per_kop",
            "value": 44.922,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 139.32,
            "range": "± 1.1; min 125.4 max 149.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3116,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/cycles",
            "value": 338,
            "range": "median 338 max 338 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/cycles_per_kop",
            "value": 330.078,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 110.2,
            "range": "± 1.0; min 92.2 max 115.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9416,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 4216,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/cycles",
            "value": 150,
            "range": "median 150 max 151 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/cycles_per_kop",
            "value": 146.484,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 103.85,
            "range": "± 4.5; min 87.2 max 110.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/cycles",
            "value": 150,
            "range": "median 150 max 151 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/cycles_per_kop",
            "value": 146.484,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 173.55,
            "range": "± 3.5; min 169.0 max 187.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/cycles",
            "value": 142,
            "range": "median 142 max 151 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/cycles_per_kop",
            "value": 138.672,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 114.47,
            "range": "± 5.4; min 96.7 max 183.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/cycles",
            "value": 142,
            "range": "median 142 max 151 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/cycles_per_kop",
            "value": 138.672,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 168.65,
            "range": "± 3.6; min 158.9 max 200.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/cycles",
            "value": 42,
            "range": "median 42 max 42 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/cycles_per_kop",
            "value": 41.016,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 95.78,
            "range": "± 1.0; min 83.0 max 105.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3872,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/cycles",
            "value": 42,
            "range": "median 42 max 42 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/cycles_per_kop",
            "value": 41.016,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 144.61,
            "range": "± 0.9; min 131.7 max 155.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2948,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/cycles",
            "value": 215,
            "range": "median 215 max 215 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/cycles_per_kop",
            "value": 209.961,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 107.88,
            "range": "± 0.9; min 94.4 max 113.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/cycles",
            "value": 215,
            "range": "median 215 max 215 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/cycles_per_kop",
            "value": 209.961,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 186.23,
            "range": "± 4.6; min 165.0 max 247.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/cycles",
            "value": 216,
            "range": "median 216 max 216 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/cycles_per_kop",
            "value": 210.938,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 103.78,
            "range": "± 0.8; min 91.1 max 113.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 3936,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/cycles",
            "value": 216,
            "range": "median 216 max 216 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/cycles_per_kop",
            "value": 210.938,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 177.48,
            "range": "± 1.6; min 157.4 max 182.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 3028,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/cycles",
            "value": 224,
            "range": "median 224 max 224 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/cycles_per_kop",
            "value": 218.75,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 112.11,
            "range": "± 4.2; min 86.9 max 160.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 9368,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 3968,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/cycles",
            "value": 224,
            "range": "median 224 max 224 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/cycles_per_kop",
            "value": 218.75,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 175.53,
            "range": "± 4.6; min 162.9 max 185.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 3060,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/cycles",
            "value": 412,
            "range": "median 412 max 412 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/cycles_per_kop",
            "value": 402.344,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 100.07,
            "range": "± 2.2; min 86.0 max 129.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 3944,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/cycles",
            "value": 412,
            "range": "median 412 max 412 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/cycles_per_kop",
            "value": 402.344,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 180.67,
            "range": "± 1.2; min 157.6 max 186.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 3036,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/cycles",
            "value": 318,
            "range": "median 318 max 318 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/cycles_per_kop",
            "value": 310.547,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 104.45,
            "range": "± 2.9; min 91.7 max 109.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 9576,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 4156,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/cycles",
            "value": 318,
            "range": "median 318 max 318 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/cycles_per_kop",
            "value": 310.547,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 149.29,
            "range": "± 3.9; min 139.7 max 177.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 9079,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 3216,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/cycles",
            "value": 211,
            "range": "median 211 max 211 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/cycles_per_kop",
            "value": 206.055,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 95.84,
            "range": "± 0.9; min 76.4 max 103.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 9480,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 4060,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/cycles",
            "value": 2658,
            "range": "median 2664 max 2675 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/cycles_per_kop",
            "value": 2595.703,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 118.17,
            "range": "± 0.7; min 114.3 max 126.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14793,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9988,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/cycles",
            "value": 211,
            "range": "median 211 max 211 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/cycles_per_kop",
            "value": 206.055,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 145.6,
            "range": "± 1.5; min 124.9 max 148.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 3120,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/cycles",
            "value": 13458,
            "range": "median 13463 max 13468 n=2; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/cycles_per_kop",
            "value": 13142.578,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 218.7,
            "range": "± 3.6; min 203.2 max 247.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 11273,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 5988,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/cycles",
            "value": 13458,
            "range": "median 13464 max 13468 n=22; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/cycles_per_kop",
            "value": 13142.578,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 2015.28,
            "range": "± 14.5; min 1995.4 max 2101.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 10777,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 5048,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/cycles",
            "value": 138,
            "range": "median 138 max 138 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/cycles_per_kop",
            "value": 134.766,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 98.71,
            "range": "± 1.7; min 88.3 max 105.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 9304,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 3884,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/cycles",
            "value": 138,
            "range": "median 138 max 138 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/cycles_per_kop",
            "value": 134.766,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 155.41,
            "range": "± 2.6; min 130.3 max 160.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 2976,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/cycles",
            "value": 1643,
            "range": "median 1672 max 1685 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/cycles_per_kop",
            "value": 1604.492,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 113.37,
            "range": "± 6.0; min 99.5 max 131.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14953,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 10180,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/cycles",
            "value": 118,
            "range": "median 118 max 118 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/cycles_per_kop",
            "value": 115.234,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 96.66,
            "range": "± 2.0; min 77.7 max 104.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 9416,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 4004,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/cycles",
            "value": 1787,
            "range": "median 1795 max 1810 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/cycles_per_kop",
            "value": 1745.117,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 110.4,
            "range": "± 0.7; min 94.6 max 121.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 15049,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 10248,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/cycles",
            "value": 118,
            "range": "median 118 max 118 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/cycles_per_kop",
            "value": 115.234,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 136.92,
            "range": "± 3.8; min 122.3 max 146.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 8951,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 3096,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/cycles",
            "value": 993,
            "range": "median 993 max 993 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/cycles_per_kop",
            "value": 969.727,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 111.98,
            "range": "± 1.2; min 103.4 max 154.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 9736,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 4544,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/cycles",
            "value": 993,
            "range": "median 993 max 993 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/cycles_per_kop",
            "value": 969.727,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 241.14,
            "range": "± 3.5; min 230.3 max 279.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 3636,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/cycles",
            "value": 86,
            "range": "median 86 max 86 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/cycles_per_kop",
            "value": 83.984,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 105.34,
            "range": "± 3.3; min 94.6 max 145.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3992,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/cycles",
            "value": 86,
            "range": "median 86 max 86 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/cycles_per_kop",
            "value": 83.984,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 135.68,
            "range": "± 1.2; min 131.7 max 144.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 3084,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/cycles",
            "value": 14125,
            "range": "median 14125 max 14126 n=2; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/cycles_per_kop",
            "value": 13793.945,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/npu_us",
            "value": 232.97,
            "range": "± 4.3; min 213.8 max 263.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/xclbin_bytes",
            "value": 10937,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/core_elf_bytes",
            "value": 5668,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/cycles",
            "value": 14125,
            "range": "median 14125 max 14125 n=22; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/cycles_per_kop",
            "value": 13793.945,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/npu_us",
            "value": 2106.64,
            "range": "± 2.0; min 2099.9 max 2156.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/xclbin_bytes",
            "value": 10473,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/core_elf_bytes",
            "value": 4760,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/cycles",
            "value": 178,
            "range": "median 178 max 193 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/cycles_per_kop",
            "value": 86.914,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 109.54,
            "range": "± 1.2; min 95.6 max 125.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3148,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/cycles",
            "value": 178,
            "range": "median 178 max 193 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/cycles_per_kop",
            "value": 86.914,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 182.94,
            "range": "± 1.4; min 171.4 max 187.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3148,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/cycles",
            "value": 88,
            "range": "median 88 max 88 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/cycles_per_kop",
            "value": 85.938,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/npu_us",
            "value": 103.77,
            "range": "± 0.8; min 93.5 max 128.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/xclbin_bytes",
            "value": 9448,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/core_elf_bytes",
            "value": 4192,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/cycles",
            "value": 88,
            "range": "median 88 max 88 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/cycles_per_kop",
            "value": 85.938,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/npu_us",
            "value": 177.67,
            "range": "± 6.0; min 165.4 max 187.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/core_elf_bytes",
            "value": 3284,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/cycles",
            "value": 162,
            "range": "median 162 max 162 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/cycles_per_kop",
            "value": 158.203,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 103.9,
            "range": "± 1.5; min 88.9 max 109.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 3984,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/cycles",
            "value": 162,
            "range": "median 162 max 162 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/cycles_per_kop",
            "value": 158.203,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 143.08,
            "range": "± 2.4; min 129.4 max 161.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 3060,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/cycles",
            "value": 251,
            "range": "median 251 max 251 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 98.81,
            "range": "± 0.9; min 87.3 max 106.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 9512,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 4104,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/cycles",
            "value": 426,
            "range": "median 426 max 426 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 102.82,
            "range": "± 1.2; min 84.7 max 106.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 4424,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/cycles",
            "value": 185,
            "range": "median 185 max 185 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 96.95,
            "range": "± 2.3; min 82.8 max 107.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 9239,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 3832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/cycles",
            "value": 818,
            "range": "median 818 max 818 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 108.18,
            "range": "± 4.4; min 92.9 max 177.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 9784,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 4376,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/cycles",
            "value": 5529,
            "range": "median 5529 max 5530 n=11; init[2] min 262; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/cycles_per_kop",
            "value": 21.091,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 155.86,
            "range": "± 1.6; min 141.7 max 219.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10008,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4444,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/cycles",
            "value": 5529,
            "range": "median 5529 max 5534 n=188; init[2] min 262; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/cycles_per_kop",
            "value": 21.091,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1000.09,
            "range": "± 2.1; min 988.1 max 1042.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10008,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4444,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/cycles",
            "value": 11425,
            "range": "median 11425 max 11425 n=4; init[2] min 262; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/cycles_per_kop",
            "value": 21.791,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/npu_us",
            "value": 220.81,
            "range": "± 3.8; min 201.0 max 252.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/xclbin_bytes",
            "value": 11049,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/core_elf_bytes",
            "value": 6116,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/cycles",
            "value": 1977,
            "range": "median 1984 max 2489 n=16; init[2] min 262",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/cycles_per_kop",
            "value": 7.542,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 120.17,
            "range": "± 1.5; min 108.2 max 131.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4028,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/cycles",
            "value": 733,
            "range": "median 733 max 733 n=16; init[2] min 262",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/cycles_per_kop",
            "value": 2.796,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 109.9,
            "range": "± 3.7; min 96.9 max 120.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 9592,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 3988,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/cycles",
            "value": 1006,
            "range": "median 1008 max 1027 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/cycles_per_kop",
            "value": 30.701,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 99.77,
            "range": "± 1.0; min 84.0 max 106.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 10793,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 5252,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x128x64x4/bfloat16/band_m=32/bfp16_b=True/chunk_k=128/out_chunk=512/cycles",
            "value": 2999,
            "range": "median 2999 max 2999 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x128x64x4/bfloat16/band_m=32/bfp16_b=True/chunk_k=128/out_chunk=512/cycles_per_kop",
            "value": 2.86,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x128x64x4/bfloat16/band_m=32/bfp16_b=True/chunk_k=128/out_chunk=512/npu_us",
            "value": 116.05,
            "range": "± 0.9; min 107.0 max 122.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x128x64x4/bfloat16/band_m=32/bfp16_b=True/chunk_k=128/out_chunk=512/xclbin_bytes",
            "value": 10457,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x128x64x4/bfloat16/band_m=32/bfp16_b=True/chunk_k=128/out_chunk=512/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x128x64x4/bfloat16/band_m=32/bfp16_b=True/chunk_k=128/out_chunk=512/core_elf_bytes",
            "value": 5056,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x32x128x4/bfloat16/band_m=64/bfp16_b=True/chunk_k=32/out_chunk=512/cycles",
            "value": 3889,
            "range": "median 3889 max 3889 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x32x128x4/bfloat16/band_m=64/bfp16_b=True/chunk_k=32/out_chunk=512/cycles_per_kop",
            "value": 7.418,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x32x128x4/bfloat16/band_m=64/bfp16_b=True/chunk_k=32/out_chunk=512/npu_us",
            "value": 129.64,
            "range": "± 3.2; min 104.2 max 211.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x32x128x4/bfloat16/band_m=64/bfp16_b=True/chunk_k=32/out_chunk=512/xclbin_bytes",
            "value": 10521,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x32x128x4/bfloat16/band_m=64/bfp16_b=True/chunk_k=32/out_chunk=512/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/64x32x128x4/bfloat16/band_m=64/bfp16_b=True/chunk_k=32/out_chunk=512/core_elf_bytes",
            "value": 5144,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/16x64x16x4/bfloat16/band_m=16/bfp16_b=True/chunk_k=32/out_chunk=64/cycles",
            "value": 185,
            "range": "median 185 max 193 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/16x64x16x4/bfloat16/band_m=16/bfp16_b=True/chunk_k=32/out_chunk=64/cycles_per_kop",
            "value": 5.646,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/16x64x16x4/bfloat16/band_m=16/bfp16_b=True/chunk_k=32/out_chunk=64/npu_us",
            "value": 105.82,
            "range": "± 2.2; min 85.5 max 110.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/16x64x16x4/bfloat16/band_m=16/bfp16_b=True/chunk_k=32/out_chunk=64/xclbin_bytes",
            "value": 9528,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/16x64x16x4/bfloat16/band_m=16/bfp16_b=True/chunk_k=32/out_chunk=64/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "fused_mm/16x64x16x4/bfloat16/band_m=16/bfp16_b=True/chunk_k=32/out_chunk=64/core_elf_bytes",
            "value": 3736,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/cycles",
            "value": 951,
            "range": "median 951 max 978 n=16; init[2] min 78",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/cycles_per_kop",
            "value": 1.814,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/npu_us",
            "value": 115.46,
            "range": "± 7.8; min 92.7 max 171.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfp16ebs8/core_elf_bytes",
            "value": 4364,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/cycles",
            "value": 4845,
            "range": "median 4845 max 4845 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/npu_us",
            "value": 107.77,
            "range": "± 4.3; min 94.5 max 119.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/xclbin_bytes",
            "value": 11017,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/core_elf_bytes",
            "value": 6156,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/unshuffle=True/cycles",
            "value": 6015,
            "range": "median 6015 max 6015 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/unshuffle=True/npu_us",
            "value": 105.87,
            "range": "± 1.8; min 98.2 max 137.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/unshuffle=True/xclbin_bytes",
            "value": 11017,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/unshuffle=True/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/512x4/bfp16ebs8/unshuffle=True/core_elf_bytes",
            "value": 6156,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/256x4/bfp16ebs8/unshuffle=False/cycles",
            "value": 2461,
            "range": "median 2461 max 2461 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/256x4/bfp16ebs8/unshuffle=False/npu_us",
            "value": 106.95,
            "range": "± 4.1; min 89.1 max 117.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/256x4/bfp16ebs8/unshuffle=False/xclbin_bytes",
            "value": 11017,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/256x4/bfp16ebs8/unshuffle=False/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/256x4/bfp16ebs8/unshuffle=False/core_elf_bytes",
            "value": 6156,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/256x4/bfp16ebs8/unshuffle=True/cycles",
            "value": 3055,
            "range": "median 3055 max 3055 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/256x4/bfp16ebs8/unshuffle=True/npu_us",
            "value": 102.34,
            "range": "± 2.7; min 91.3 max 159.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/256x4/bfp16ebs8/unshuffle=True/xclbin_bytes",
            "value": 11017,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/256x4/bfp16ebs8/unshuffle=True/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp_shuffle/256x4/bfp16ebs8/unshuffle=True/core_elf_bytes",
            "value": 6156,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/cycles",
            "value": 2115,
            "range": "median 2115 max 2116 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/cycles_per_kop",
            "value": 129.089,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/npu_us",
            "value": 103.81,
            "range": "± 1.1; min 89.5 max 131.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "q4nx_dequant/5120x4/uint8/core_elf_bytes",
            "value": 3580,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/cycles",
            "value": 1249,
            "range": "median 1249 max 1281 n=13; init[2] min 134; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/cycles_per_kop",
            "value": 2.382,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/npu_us",
            "value": 114.31,
            "range": "± 6.0; min 95.1 max 163.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/xclbin_bytes",
            "value": 9944,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_bfp/64x64x64x16/bfloat16/mixed=True/core_elf_bytes",
            "value": 4240,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/cycles",
            "value": 127,
            "range": "median 127 max 127 n=15; init[2] min 2; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/cycles_per_kop",
            "value": 62.012,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 98.71,
            "range": "± 0.9; min 83.8 max 124.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9672,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3992,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/cycles",
            "value": 788,
            "range": "median 788 max 788 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/cycles_per_kop",
            "value": 48.096,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/npu_us",
            "value": 112.15,
            "range": "± 1.1; min 100.3 max 119.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/xclbin_bytes",
            "value": 11033,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/core_elf_bytes",
            "value": 5416,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/cycles",
            "value": 110,
            "range": "median 110 max 111 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/cycles_per_kop",
            "value": 214.844,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/npu_us",
            "value": 106.44,
            "range": "± 1.2; min 94.5 max 110.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/xclbin_bytes",
            "value": 10297,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/core_elf_bytes",
            "value": 4680,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/cycles",
            "value": 408,
            "range": "median 408 max 408 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/cycles_per_kop",
            "value": 24.902,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/npu_us",
            "value": 115.49,
            "range": "± 0.7; min 102.2 max 167.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/xclbin_bytes",
            "value": 10905,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/core_elf_bytes",
            "value": 5400,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/cycles",
            "value": 476,
            "range": "median 476 max 476 n=224; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/cycles_per_kop",
            "value": 29.053,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/npu_us",
            "value": 394.23,
            "range": "± 2.1; min 384.7 max 404.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/xclbin_bytes",
            "value": 26745,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/core_elf_bytes",
            "value": 5048,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/cycles",
            "value": 11,
            "range": "median 11 max 11 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/cycles_per_kop",
            "value": 11000,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 96.33,
            "range": "± 1.7; min 81.9 max 104.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8743,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2808,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/cycles",
            "value": 14,
            "range": "median 14 max 14 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/cycles_per_kop",
            "value": 14000,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 99.72,
            "range": "± 4.1; min 83.8 max 110.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8759,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/cycles",
            "value": 856,
            "range": "median 856 max 856 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/cycles_per_kop",
            "value": 139.323,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 114.53,
            "range": "± 1.1; min 101.7 max 122.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 3628,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/cycles",
            "value": 3202,
            "range": "median 3207 max 3234 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/cycles_per_kop",
            "value": 521.159,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/npu_us",
            "value": 142.24,
            "range": "± 3.2; min 118.3 max 195.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/xclbin_bytes",
            "value": 14457,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/use_lut=True/lut/core_elf_bytes",
            "value": 9800,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/cycles",
            "value": 856,
            "range": "median 856 max 856 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/cycles_per_kop",
            "value": 139.323,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 286.34,
            "range": "± 3.2; min 278.7 max 301.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 8935,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 3628,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/cycles",
            "value": 1476,
            "range": "median 1476 max 1476 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/cycles_per_kop",
            "value": 192.188,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 107.62,
            "range": "± 0.6; min 94.4 max 113.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 4104,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/cycles",
            "value": 1771,
            "range": "median 1771 max 1774 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/cycles_per_kop",
            "value": 922.396,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 118.97,
            "range": "± 1.3; min 107.8 max 125.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 9432,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 4168,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/cycles",
            "value": 400,
            "range": "median 400 max 401 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/cycles_per_kop",
            "value": 208.333,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 95.42,
            "range": "± 5.0; min 78.3 max 109.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 4868,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/cycles",
            "value": 306,
            "range": "median 306 max 307 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/cycles_per_kop",
            "value": 159.375,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 107.44,
            "range": "± 3.7; min 95.3 max 113.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3064,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/cycles",
            "value": 306,
            "range": "median 306 max 307 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/cycles_per_kop",
            "value": 159.375,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 99.69,
            "range": "± 1.1; min 87.3 max 103.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3068,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/cycles",
            "value": 1134,
            "range": "median 1134 max 1134 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/cycles_per_kop",
            "value": 196.875,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 107.39,
            "range": "± 1.8; min 98.7 max 173.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9127,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3352,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/cycles",
            "value": 4702,
            "range": "median 4704 max 4763 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/cycles_per_kop",
            "value": 136.053,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 138.49,
            "range": "± 6.1; min 131.2 max 150.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 10665,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 6096,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/cycles",
            "value": 4198,
            "range": "median 4202 max 4204 n=9; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/cycles_per_kop",
            "value": 2186.458,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 147.51,
            "range": "± 1.8; min 134.2 max 163.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 11705,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 6788,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/cycles",
            "value": 15007,
            "range": "median 15041 max 15055 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/cycles_per_kop",
            "value": 57.247,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 169.96,
            "range": "± 5.4; min 158.5 max 183.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/cycles",
            "value": 504,
            "range": "median 504 max 504 n=7; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/cycles_per_kop",
            "value": 1.923,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 97.52,
            "range": "± 6.2; min 78.9 max 110.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13081,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 3428,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/cycles",
            "value": 23793,
            "range": "median 24059 max 24305 n=6; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/cycles_per_kop",
            "value": 45.205,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 216.97,
            "range": "± 5.9; min 198.9 max 229.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 18025,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 4888,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/cycles",
            "value": 23793,
            "range": "median 23793 max 23793 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/cycles_per_kop",
            "value": 45.205,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/npu_us",
            "value": 215.29,
            "range": "± 0.7; min 201.7 max 220.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/xclbin_bytes",
            "value": 18745,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/core_elf_bytes",
            "value": 5156,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/cycles",
            "value": 20147,
            "range": "median 20402 max 20402 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/cycles_per_kop",
            "value": 51.236,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 201.17,
            "range": "± 1.4; min 187.0 max 234.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 17785,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 7264,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/cycles",
            "value": 20146,
            "range": "median 20402 max 20402 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/cycles_per_kop",
            "value": 51.234,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 187.08,
            "range": "± 4.6; min 171.5 max 193.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 18601,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 7628,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/cycles",
            "value": 423,
            "range": "median 423 max 423 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/cycles_per_kop",
            "value": 1.054,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/npu_us",
            "value": 108.08,
            "range": "± 3.9; min 96.0 max 170.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/xclbin_bytes",
            "value": 21225,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/cycles",
            "value": 15007,
            "range": "median 15041 max 15055 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/cycles_per_kop",
            "value": 57.247,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 162.15,
            "range": "± 4.7; min 153.4 max 180.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 4264,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/cycles",
            "value": 97184,
            "range": "median 97184 max 97184 n=2; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/cycles_per_kop",
            "value": 41.192,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 537.24,
            "range": "± 3.0; min 522.9 max 548.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 48441,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 7612,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/cycles",
            "value": 37467,
            "range": "median 37467 max 37467 n=1; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/cycles_per_kop",
            "value": 15.881,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 280.65,
            "range": "± 4.8; min 260.8 max 330.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 48809,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 7832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/cycles",
            "value": 392699,
            "range": "median 392699 max 392702 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/cycles_per_kop",
            "value": 1460.934,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/npu_us",
            "value": 3059.61,
            "range": "± 2.2; min 3032.5 max 3148.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/xclbin_bytes",
            "value": 14281,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/core_elf_bytes",
            "value": 4120,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/cycles",
            "value": 467466,
            "range": "median 467473 max 467483 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/cycles_per_kop",
            "value": 2037.99,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/npu_us",
            "value": 3332.65,
            "range": "± 12.8; min 3313.0 max 3417.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/xclbin_bytes",
            "value": 10505,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/core_elf_bytes",
            "value": 4120,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/cycles",
            "value": 377400,
            "range": "median 377401 max 377408 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/cycles_per_kop",
            "value": 915.664,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/npu_us",
            "value": 4167.03,
            "range": "± 17.1; min 4123.2 max 4299.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/xclbin_bytes",
            "value": 24201,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/core_elf_bytes",
            "value": 4120,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/cycles",
            "value": 294272,
            "range": "median 294272 max 294273 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/cycles_per_kop",
            "value": 2189.524,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/npu_us",
            "value": 1400.52,
            "range": "± 4.8; min 1389.5 max 1428.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/xclbin_bytes",
            "value": 19081,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/core_elf_bytes",
            "value": 4120,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles",
            "value": 298868,
            "range": "median 298868 max 298871 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles_per_kop",
            "value": 1111.86,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/npu_us",
            "value": 2588.15,
            "range": "± 17.0; min 2565.9 max 2650.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/xclbin_bytes",
            "value": 14281,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/core_elf_bytes",
            "value": 4232,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/cycles",
            "value": 381469,
            "range": "median 381471 max 381474 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/cycles_per_kop",
            "value": 2217.43,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/npu_us",
            "value": 1784.87,
            "range": "± 1.6; min 1775.5 max 1834.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/xclbin_bytes",
            "value": 11017,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/core_elf_bytes",
            "value": 4232,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/cycles",
            "value": 295537,
            "range": "median 295537 max 295538 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/cycles_per_kop",
            "value": 549.734,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/npu_us",
            "value": 5003.96,
            "range": "± 14.5; min 4925.1 max 5088.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/xclbin_bytes",
            "value": 28681,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/core_elf_bytes",
            "value": 4232,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/cycles",
            "value": 487323,
            "range": "median 487323 max 487323 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/cycles_per_kop",
            "value": 1294.97,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/npu_us",
            "value": 3452.42,
            "range": "± 1.9; min 3445.1 max 3457.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/xclbin_bytes",
            "value": 36361,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/core_elf_bytes",
            "value": 4232,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles",
            "value": 317081,
            "range": "median 317081 max 317083 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles_per_kop",
            "value": 1174.722,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/npu_us",
            "value": 2672.16,
            "range": "± 10.8; min 2647.9 max 2725.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/xclbin_bytes",
            "value": 14729,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/core_elf_bytes",
            "value": 4640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/cycles",
            "value": 317080,
            "range": "median 317080 max 317082 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/cycles_per_kop",
            "value": 1174.718,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/npu_us",
            "value": 2733.96,
            "range": "± 13.1; min 2689.6 max 2792.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/xclbin_bytes",
            "value": 14713,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/core_elf_bytes",
            "value": 4492,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/cycles",
            "value": 444310,
            "range": "median 444312 max 444315 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/cycles_per_kop",
            "value": 2279.916,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/npu_us",
            "value": 2065.72,
            "range": "± 1.6; min 2049.6 max 2130.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/xclbin_bytes",
            "value": 11657,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/core_elf_bytes",
            "value": 4640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/cycles",
            "value": 275984,
            "range": "median 275984 max 275985 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/cycles_per_kop",
            "value": 2049.183,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/npu_us",
            "value": 1318.14,
            "range": "± 5.2; min 1309.6 max 1363.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/xclbin_bytes",
            "value": 19529,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/core_elf_bytes",
            "value": 4640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/cycles",
            "value": 333982,
            "range": "median 333982 max 333983 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/cycles_per_kop",
            "value": 808.125,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/npu_us",
            "value": 3976.68,
            "range": "± 14.6; min 3930.8 max 4043.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/xclbin_bytes",
            "value": 24633,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/core_elf_bytes",
            "value": 4492,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/cycles",
            "value": 293058,
            "range": "median 293058 max 293170 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/cycles_per_kop",
            "value": 1135.672,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/npu_us",
            "value": 9635.49,
            "range": "± 14.9; min 9583.3 max 10285.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/xclbin_bytes",
            "value": 11273,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/core_elf_bytes",
            "value": 5860,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/cycles",
            "value": 490079,
            "range": "median 490079 max 490080 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/cycles_per_kop",
            "value": 8103.158,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/npu_us",
            "value": 2270.21,
            "range": "± 1.8; min 2262.1 max 2322.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/xclbin_bytes",
            "value": 11017,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/core_elf_bytes",
            "value": 5428,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/cycles",
            "value": 299236,
            "range": "median 299236 max 299237 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/cycles_per_kop",
            "value": 8246.142,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/npu_us",
            "value": 1432.76,
            "range": "± 5.2; min 1417.0 max 1464.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/xclbin_bytes",
            "value": 10313,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/core_elf_bytes",
            "value": 4960,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/cycles",
            "value": 345010,
            "range": "median 345010 max 345013 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/cycles_per_kop",
            "value": 8149.329,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/npu_us",
            "value": 1627.99,
            "range": "± 1.0; min 1617.4 max 1637.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/xclbin_bytes",
            "value": 12681,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/core_elf_bytes",
            "value": 4960,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/cycles",
            "value": 370199,
            "range": "median 370199 max 370200 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/cycles_per_kop",
            "value": 7983.933,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/npu_us",
            "value": 1743.16,
            "range": "± 10.7; min 1728.6 max 1819.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/xclbin_bytes",
            "value": 11593,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/core_elf_bytes",
            "value": 5428,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/cycles",
            "value": 470715,
            "range": "median 470715 max 470720 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/cycles_per_kop",
            "value": 7782.986,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/npu_us",
            "value": 2249.88,
            "range": "± 9.2; min 2181.3 max 2299.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/xclbin_bytes",
            "value": 14713,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/insts_bytes",
            "value": 464,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/core_elf_bytes",
            "value": 5884,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/cycles",
            "value": 363790,
            "range": "median 364522 max 364766 n=8; init[2] min 22",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/cycles_per_kop",
            "value": 2706.771,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/npu_us",
            "value": 1704.84,
            "range": "± 1.4; min 1696.8 max 1715.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/xclbin_bytes",
            "value": 23065,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/core_elf_bytes",
            "value": 10332,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/cycles",
            "value": 41338,
            "range": "median 41338 max 41338 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/cycles_per_kop",
            "value": 2018.457,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/npu_us",
            "value": 279.16,
            "range": "± 3.5; min 262.9 max 289.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/xclbin_bytes",
            "value": 19737,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/core_elf_bytes",
            "value": 4152,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/cycles",
            "value": 31098,
            "range": "median 31098 max 31098 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/cycles_per_kop",
            "value": 1518.457,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/npu_us",
            "value": 236.66,
            "range": "± 2.2; min 226.1 max 245.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/xclbin_bytes",
            "value": 19737,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/core_elf_bytes",
            "value": 4152,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/cycles",
            "value": 911,
            "range": "median 911 max 912 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/cycles_per_kop",
            "value": 889.648,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 112.12,
            "range": "± 8.0; min 97.7 max 124.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/cycles",
            "value": 777,
            "range": "median 777 max 777 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/cycles_per_kop",
            "value": 758.789,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 104.53,
            "range": "± 3.7; min 97.1 max 138.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9063,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3372,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/cycles",
            "value": 438,
            "range": "median 438 max 438 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/cycles_per_kop",
            "value": 106.934,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 99.65,
            "range": "± 3.8; min 81.2 max 112.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 10249,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 5440,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/cycles",
            "value": 581,
            "range": "median 581 max 581 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/cycles_per_kop",
            "value": 94.564,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 109.47,
            "range": "± 1.8; min 99.6 max 131.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 10505,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 5548,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/cycles",
            "value": 5915,
            "range": "median 5915 max 5918 n=5; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/cycles_per_kop",
            "value": 962.728,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/npu_us",
            "value": 151.7,
            "range": "± 1.7; min 143.0 max 246.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/xclbin_bytes",
            "value": 11033,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/core_elf_bytes",
            "value": 5900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/cycles",
            "value": 10519,
            "range": "median 10519 max 10519 n=3; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/cycles_per_kop",
            "value": 1284.058,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/npu_us",
            "value": 194.16,
            "range": "± 2.7; min 174.5 max 207.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/xclbin_bytes",
            "value": 19465,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/core_elf_bytes",
            "value": 6372,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/cycles",
            "value": 298,
            "range": "median 298 max 302 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/cycles_per_kop",
            "value": 97.005,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/npu_us",
            "value": 102.82,
            "range": "± 1.4; min 90.9 max 110.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 9255,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 3528,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/cycles",
            "value": 518,
            "range": "median 518 max 518 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/cycles_per_kop",
            "value": 63.232,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/npu_us",
            "value": 99.58,
            "range": "± 2.8; min 84.9 max 104.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/core_elf_bytes",
            "value": 4776,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/cycles",
            "value": 3213,
            "range": "median 3213 max 3213 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/cycles_per_kop",
            "value": 392.212,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/npu_us",
            "value": 131.98,
            "range": "± 1.9; min 118.0 max 138.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/core_elf_bytes",
            "value": 4776,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/cycles",
            "value": 3090,
            "range": "median 3090 max 3090 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/cycles_per_kop",
            "value": 377.197,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/npu_us",
            "value": 122.65,
            "range": "± 5.7; min 106.9 max 137.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/core_elf_bytes",
            "value": 4776,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/cycles",
            "value": 1106,
            "range": "median 1106 max 1106 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/cycles_per_kop",
            "value": 135.01,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/npu_us",
            "value": 102.27,
            "range": "± 1.0; min 92.3 max 154.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/core_elf_bytes",
            "value": 4776,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles",
            "value": 1505,
            "range": "median 1505 max 1505 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles_per_kop",
            "value": 81.651,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 115.75,
            "range": "± 2.7; min 99.4 max 128.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9784,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3948,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles",
            "value": 1505,
            "range": "median 1505 max 1505 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles_per_kop",
            "value": 81.651,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 112.64,
            "range": "± 5.6; min 101.8 max 126.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 9784,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 3948,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/cycles",
            "value": 98,
            "range": "median 102 max 105 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/cycles_per_kop",
            "value": 38.281,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/npu_us",
            "value": 110.99,
            "range": "± 4.2; min 88.6 max 152.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/xclbin_bytes",
            "value": 10649,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/core_elf_bytes",
            "value": 7396,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/emulate_bf16_mmul_with_bfp16=True/llama-prefill/cycles",
            "value": 1469,
            "range": "median 1469 max 1503 n=16; init[2] min 134",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/emulate_bf16_mmul_with_bfp16=True/llama-prefill/cycles_per_kop",
            "value": 2.802,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/emulate_bf16_mmul_with_bfp16=True/llama-prefill/npu_us",
            "value": 126.25,
            "range": "± 5.8; min 112.8 max 174.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/emulate_bf16_mmul_with_bfp16=True/llama-prefill/xclbin_bytes",
            "value": 9848,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/emulate_bf16_mmul_with_bfp16=True/llama-prefill/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/emulate_bf16_mmul_with_bfp16=True/llama-prefill/core_elf_bytes",
            "value": 4284,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/b_col_maj=True/emulate_bf16_mmul_with_bfp16=True/llama-prefill-lm-head/cycles",
            "value": 2893,
            "range": "median 2893 max 2910 n=10; init[2] min 134; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/b_col_maj=True/emulate_bf16_mmul_with_bfp16=True/llama-prefill-lm-head/cycles_per_kop",
            "value": 5.518,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/b_col_maj=True/emulate_bf16_mmul_with_bfp16=True/llama-prefill-lm-head/npu_us",
            "value": 147.25,
            "range": "± 4.4; min 121.4 max 189.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/b_col_maj=True/emulate_bf16_mmul_with_bfp16=True/llama-prefill-lm-head/xclbin_bytes",
            "value": 9672,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/b_col_maj=True/emulate_bf16_mmul_with_bfp16=True/llama-prefill-lm-head/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16/b_col_maj=True/emulate_bf16_mmul_with_bfp16=True/llama-prefill-lm-head/core_elf_bytes",
            "value": 4108,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/cycles",
            "value": 787,
            "range": "median 787 max 787 n=14; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/cycles_per_kop",
            "value": 192.139,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/npu_us",
            "value": 100.26,
            "range": "± 5.0; min 91.0 max 136.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/xclbin_bytes",
            "value": 9480,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/core_elf_bytes",
            "value": 4064,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/cycles",
            "value": 211,
            "range": "median 211 max 211 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/cycles_per_kop",
            "value": 206.055,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/npu_us",
            "value": 104.05,
            "range": "± 2.2; min 83.2 max 106.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/xclbin_bytes",
            "value": 9480,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/core_elf_bytes",
            "value": 4064,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/cycles",
            "value": 550,
            "range": "median 550 max 583 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/cycles_per_kop",
            "value": 134.277,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/npu_us",
            "value": 120.8,
            "range": "± 1.9; min 106.5 max 125.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/core_elf_bytes",
            "value": 3056,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/cycles",
            "value": 278,
            "range": "median 278 max 294 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/cycles_per_kop",
            "value": 135.742,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/npu_us",
            "value": 113.92,
            "range": "± 1.5; min 101.3 max 132.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/core_elf_bytes",
            "value": 3056,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/cycles",
            "value": 142,
            "range": "median 142 max 150 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/cycles_per_kop",
            "value": 138.672,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/npu_us",
            "value": 103.8,
            "range": "± 6.5; min 90.2 max 151.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/core_elf_bytes",
            "value": 3056,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/cycles",
            "value": 40,
            "range": "median 40 max 42 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/cycles_per_kop",
            "value": 156.25,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/npu_us",
            "value": 101.77,
            "range": "± 4.1; min 89.8 max 108.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/core_elf_bytes",
            "value": 3008,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/cycles",
            "value": 294,
            "range": "median 294 max 295 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/cycles_per_kop",
            "value": 143.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/npu_us",
            "value": 112.19,
            "range": "± 1.3; min 98.8 max 142.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/xclbin_bytes",
            "value": 8903,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/core_elf_bytes",
            "value": 3056,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/cycles",
            "value": 42,
            "range": "median 42 max 43 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/cycles_per_kop",
            "value": 164.062,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/npu_us",
            "value": 101.18,
            "range": "± 5.4; min 82.2 max 145.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/core_elf_bytes",
            "value": 3008,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/cycles",
            "value": 1905,
            "range": "median 1905 max 1905 n=15; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/cycles_per_kop",
            "value": 930.176,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/npu_us",
            "value": 115.11,
            "range": "± 0.7; min 111.3 max 120.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/xclbin_bytes",
            "value": 9736,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/core_elf_bytes",
            "value": 4544,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/cycles",
            "value": 64,
            "range": "median 64 max 64 n=15; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/cycles_per_kop",
            "value": 333.333,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/npu_us",
            "value": 97.33,
            "range": "± 2.8; min 80.2 max 105.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/xclbin_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/core_elf_bytes",
            "value": 3544,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/cycles",
            "value": 3344,
            "range": "median 3344 max 3344 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/npu_us",
            "value": 148.22,
            "range": "± 1.4; min 134.7 max 159.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/xclbin_bytes",
            "value": 9768,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/core_elf_bytes",
            "value": 4396,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/cycles",
            "value": 1633,
            "range": "median 1634 max 1636 n=4; init[2] min 8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/cycles_per_kop",
            "value": 375.23,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/npu_us",
            "value": 111.72,
            "range": "± 1.3; min 98.5 max 120.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/xclbin_bytes",
            "value": 15849,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/insts_bytes",
            "value": 464,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/core_elf_bytes",
            "value": 12280,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/cycles",
            "value": 2106,
            "range": "median 2107 max 2109 n=4; init[2] min 8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/cycles_per_kop",
            "value": 483.915,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/npu_us",
            "value": 110.7,
            "range": "± 2.8; min 97.8 max 120.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/xclbin_bytes",
            "value": 15849,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/insts_bytes",
            "value": 464,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/core_elf_bytes",
            "value": 12280,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/cycles",
            "value": 1265,
            "range": "median 1368 max 1393 n=4; init[0] min 262",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/cycles_per_kop",
            "value": 19.302,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/npu_us",
            "value": 103.16,
            "range": "± 4.3; min 91.4 max 114.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/xclbin_bytes",
            "value": 9960,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/core_elf_bytes",
            "value": 4416,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/cycles",
            "value": 2707,
            "range": "median 2885 max 2963 n=4; init[0] min 262",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/cycles_per_kop",
            "value": 20.653,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/npu_us",
            "value": 110.55,
            "range": "± 2.8; min 98.2 max 151.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/xclbin_bytes",
            "value": 10072,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/core_elf_bytes",
            "value": 4568,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device NPU Krackan 1 | pmode default"
          }
        ]
      }
    ]
  }
}