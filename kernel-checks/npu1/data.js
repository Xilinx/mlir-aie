window.BENCHMARK_DATA = {
  "lastUpdate": 1790663759370,
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
          "id": "d53582d3e0f9f8a2b77695bbf4abbd7766d5584a",
          "message": "Single-core Kernel Optimizations and Tooling (#3801)\n\nCo-authored-by: Claude Opus 5 <noreply@anthropic.com>\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
          "timestamp": "2026-09-28T20:24:43Z",
          "url": "https://github.com/Xilinx/mlir-aie/commit/d53582d3e0f9f8a2b77695bbf4abbd7766d5584a"
        },
        "date": 1790632656742,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 264,
            "range": "median 264 max 264 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 128.906,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 194.73,
            "range": "± 9.0; min 177.1 max 389.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 264,
            "range": "median 264 max 264 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 128.906,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 1353.79,
            "range": "± 135.7; min 950.6 max 1591.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 264,
            "range": "median 264 max 264 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 64.453,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 194.19,
            "range": "± 7.0; min 177.0 max 345.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 136,
            "range": "median 136 max 136 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 33.203,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 174.82,
            "range": "± 10.3; min 158.0 max 322.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/cycles",
            "value": 72,
            "range": "median 72 max 72 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/cycles_per_kop",
            "value": 70.312,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/npu_us",
            "value": 166.39,
            "range": "± 8.2; min 149.5 max 259.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/cycles",
            "value": 78,
            "range": "median 78 max 78 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 161.76,
            "range": "± 8.1; min 151.4 max 257.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 3048,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/cycles",
            "value": 78,
            "range": "median 78 max 78 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 292.45,
            "range": "± 4.2; min 278.4 max 355.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3048,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/cycles",
            "value": 275,
            "range": "median 275 max 275 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/cycles_per_kop",
            "value": 268.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 181.96,
            "range": "± 13.8; min 163.8 max 395.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9079,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 3208,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/cycles",
            "value": 78,
            "range": "median 87 max 131 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 174.46,
            "range": "± 5.7; min 157.0 max 182.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3196,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/cycles",
            "value": 78,
            "range": "median 87 max 131 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 554.14,
            "range": "± 148.0; min 397.3 max 1425.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3196,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/cycles",
            "value": 78,
            "range": "median 87 max 129 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 180.72,
            "range": "± 11.1; min 160.7 max 291.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3244,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/cycles",
            "value": 78,
            "range": "median 87 max 129 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 460.48,
            "range": "± 14.8; min 419.1 max 515.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3244,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/cycles",
            "value": 75,
            "range": "median 75 max 75 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/cycles_per_kop",
            "value": 73.242,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 165.1,
            "range": "± 11.0; min 149.0 max 189.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/cycles",
            "value": 75,
            "range": "median 75 max 75 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/cycles_per_kop",
            "value": 73.242,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 307.02,
            "range": "± 8.5; min 295.2 max 411.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/cycles",
            "value": 147,
            "range": "median 147 max 147 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/cycles_per_kop",
            "value": 143.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 179.03,
            "range": "± 7.8; min 163.8 max 325.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/cycles",
            "value": 147,
            "range": "median 147 max 147 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/cycles_per_kop",
            "value": 143.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 434.24,
            "range": "± 2.6; min 417.4 max 1271.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/cycles",
            "value": 147,
            "range": "median 147 max 147 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/cycles_per_kop",
            "value": 143.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 188.99,
            "range": "± 10.4; min 168.2 max 342.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/cycles",
            "value": 147,
            "range": "median 147 max 147 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/cycles_per_kop",
            "value": 143.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 445.93,
            "range": "± 16.4; min 422.7 max 540.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/cycles",
            "value": 155,
            "range": "median 155 max 155 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/cycles_per_kop",
            "value": 151.367,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 181.07,
            "range": "± 12.0; min 158.9 max 332.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 8823,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/cycles",
            "value": 155,
            "range": "median 155 max 155 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/cycles_per_kop",
            "value": 151.367,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 431.51,
            "range": "± 3.1; min 418.6 max 518.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 8823,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/cycles",
            "value": 90,
            "range": "median 90 max 90 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/cycles_per_kop",
            "value": 87.891,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 179.82,
            "range": "± 6.8; min 151.4 max 261.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 2972,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/cycles",
            "value": 90,
            "range": "median 90 max 90 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/cycles_per_kop",
            "value": 87.891,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 309.84,
            "range": "± 9.7; min 293.9 max 479.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 2972,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/cycles",
            "value": 1785,
            "range": "median 1795 max 1802 n=12; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/cycles_per_kop",
            "value": 1743.164,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 184.86,
            "range": "± 8.8; min 170.3 max 374.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 14617,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/cycles",
            "value": 1785,
            "range": "median 1795 max 1802 n=97; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/cycles_per_kop",
            "value": 1743.164,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 1448.81,
            "range": "± 78.2; min 782.4 max 1662.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 14617,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/cycles",
            "value": 1637,
            "range": "median 1659 max 1663 n=13; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/cycles_per_kop",
            "value": 1598.633,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 184.75,
            "range": "± 7.5; min 170.9 max 344.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 14521,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 9240,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/cycles",
            "value": 1637,
            "range": "median 1659 max 1663 n=103; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/cycles_per_kop",
            "value": 1598.633,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 610.74,
            "range": "± 6.8; min 577.8 max 1457.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 14521,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 9240,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/cycles",
            "value": 1057,
            "range": "median 1088 max 1106 n=12; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/cycles_per_kop",
            "value": 1032.227,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 176.11,
            "range": "± 6.7; min 158.4 max 459.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 14281,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 8888,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/cycles",
            "value": 1056,
            "range": "median 1093 max 1106 n=95; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/cycles_per_kop",
            "value": 1031.25,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 1123.58,
            "range": "± 161.4; min 626.6 max 1470.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 14281,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 8888,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/cycles",
            "value": 774,
            "range": "median 801 max 829 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/cycles_per_kop",
            "value": 755.859,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 166.42,
            "range": "± 1.2; min 159.9 max 293.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 14473,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 9052,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/cycles",
            "value": 800,
            "range": "median 801 max 829 n=201; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/cycles_per_kop",
            "value": 781.25,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 1254.85,
            "range": "± 74.1; min 463.4 max 1413.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 14473,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 9052,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/cycles",
            "value": 1339,
            "range": "median 1376 max 1381 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/cycles_per_kop",
            "value": 1307.617,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 185.01,
            "range": "± 13.4; min 166.9 max 236.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 14361,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 9088,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/cycles",
            "value": 1368,
            "range": "median 1380 max 1381 n=141; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/cycles_per_kop",
            "value": 1335.938,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 1109.09,
            "range": "± 123.3; min 719.9 max 1329.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 14361,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 9088,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/cycles",
            "value": 1845,
            "range": "median 1870 max 1875 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/cycles_per_kop",
            "value": 1801.758,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 180.25,
            "range": "± 2.0; min 172.4 max 346.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 17529,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 13700,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/cycles",
            "value": 1865,
            "range": "median 1870 max 1875 n=137; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/cycles_per_kop",
            "value": 1821.289,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 1318.29,
            "range": "± 143.3; min 879.0 max 1597.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 17529,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 13700,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/cycles",
            "value": 188,
            "range": "median 188 max 188 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/cycles_per_kop",
            "value": 183.594,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 167.86,
            "range": "± 16.0; min 140.8 max 197.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3132,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/cycles",
            "value": 188,
            "range": "median 188 max 188 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/cycles_per_kop",
            "value": 183.594,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 295.59,
            "range": "± 14.4; min 273.6 max 1349.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 3132,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/cycles",
            "value": 11981,
            "range": "median 12009 max 12038 n=2; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/cycles_per_kop",
            "value": 11700.195,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/npu_us",
            "value": 376.22,
            "range": "± 16.5; min 340.4 max 1286.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/xclbin_bytes",
            "value": 11241,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/core_elf_bytes",
            "value": 5480,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/cycles",
            "value": 11981,
            "range": "median 12010 max 12060 n=18; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/cycles_per_kop",
            "value": 11700.195,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/npu_us",
            "value": 3501.37,
            "range": "± 111.3; min 3286.0 max 4172.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/xclbin_bytes",
            "value": 11241,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/core_elf_bytes",
            "value": 5480,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/cycles",
            "value": 87,
            "range": "median 97 max 140 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/cycles_per_kop",
            "value": 42.48,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 170.96,
            "range": "± 7.1; min 154.7 max 265.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 9448,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3736,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/cycles",
            "value": 87,
            "range": "median 94 max 140 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/cycles_per_kop",
            "value": 42.48,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 425.47,
            "range": "± 3.4; min 417.3 max 580.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 9448,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3736,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/cycles",
            "value": 144,
            "range": "median 144 max 144 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/cycles_per_kop",
            "value": 140.625,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/npu_us",
            "value": 177.72,
            "range": "± 8.6; min 161.3 max 290.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/core_elf_bytes",
            "value": 3252,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/cycles",
            "value": 144,
            "range": "median 144 max 144 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/cycles_per_kop",
            "value": 140.625,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/npu_us",
            "value": 1296.03,
            "range": "± 128.0; min 496.8 max 1523.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/core_elf_bytes",
            "value": 3252,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/cycles",
            "value": 161,
            "range": "median 161 max 161 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/cycles_per_kop",
            "value": 157.227,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 170.06,
            "range": "± 12.0; min 153.8 max 264.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/cycles",
            "value": 161,
            "range": "median 161 max 161 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/cycles_per_kop",
            "value": 157.227,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 359.37,
            "range": "± 80.0; min 272.7 max 1352.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/cycles",
            "value": 141,
            "range": "median 141 max 141 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 168.5,
            "range": "± 11.4; min 153.9 max 194.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 2944,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/cycles",
            "value": 144,
            "range": "median 144 max 144 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 157.87,
            "range": "± 4.1; min 147.1 max 255.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 3136,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/cycles",
            "value": 63,
            "range": "median 63 max 63 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 172.06,
            "range": "± 10.1; min 149.6 max 297.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 2944,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/cycles",
            "value": 239,
            "range": "median 239 max 239 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 174.3,
            "range": "± 4.8; min 165.1 max 266.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 2976,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/cycles",
            "value": 1881,
            "range": "median 1961 max 1970 n=13; init[2] min 518; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/cycles_per_kop",
            "value": 7.175,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 222.39,
            "range": "± 2.6; min 211.1 max 308.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10297,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4700,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/cycles",
            "value": 1881,
            "range": "median 1961 max 1970 n=207; init[2] min 518; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/cycles_per_kop",
            "value": 7.175,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1276.79,
            "range": "± 35.1; min 1197.8 max 1579.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10297,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4700,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/cycles",
            "value": 2969,
            "range": "median 3081 max 3113 n=16; init[2] min 518",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/cycles_per_kop",
            "value": 5.663,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/npu_us",
            "value": 285.51,
            "range": "± 5.0; min 267.0 max 1228.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/xclbin_bytes",
            "value": 10633,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/core_elf_bytes",
            "value": 5044,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/cycles",
            "value": 3219,
            "range": "median 3281 max 3478 n=7; init[2] min 518; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/cycles_per_kop",
            "value": 12.28,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 242.67,
            "range": "± 11.9; min 220.8 max 323.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 10201,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4552,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/cycles",
            "value": 1449,
            "range": "median 1481 max 1513 n=16; init[2] min 518",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/cycles_per_kop",
            "value": 5.527,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 229.71,
            "range": "± 2.4; min 211.4 max 349.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 10056,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 4408,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/cycles",
            "value": 691,
            "range": "median 699 max 730 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/cycles_per_kop",
            "value": 21.088,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 158.16,
            "range": "± 4.6; min 152.3 max 275.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 9960,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 4220,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/cycles",
            "value": 95,
            "range": "median 95 max 95 n=16; init[2] min 4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/cycles_per_kop",
            "value": 46.387,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 181.77,
            "range": "± 5.0; min 158.0 max 247.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9768,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3820,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/cycles",
            "value": 1264,
            "range": "median 1264 max 1264 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/cycles_per_kop",
            "value": 77.148,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/npu_us",
            "value": 236.03,
            "range": "± 17.9; min 216.6 max 623.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/xclbin_bytes",
            "value": 12185,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/core_elf_bytes",
            "value": 6608,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/cycles",
            "value": 117,
            "range": "median 117 max 117 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/cycles_per_kop",
            "value": 228.516,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/npu_us",
            "value": 158.92,
            "range": "± 6.0; min 147.1 max 249.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/xclbin_bytes",
            "value": 10441,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/core_elf_bytes",
            "value": 4824,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/cycles",
            "value": 694,
            "range": "median 694 max 694 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/cycles_per_kop",
            "value": 42.358,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/npu_us",
            "value": 225.56,
            "range": "± 5.3; min 214.6 max 389.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/xclbin_bytes",
            "value": 11449,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/core_elf_bytes",
            "value": 5948,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/cycles",
            "value": 1755,
            "range": "median 1755 max 1755 n=72; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/cycles_per_kop",
            "value": 107.117,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/npu_us",
            "value": 1586.42,
            "range": "± 131.3; min 1224.9 max 2204.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/xclbin_bytes",
            "value": 27273,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/core_elf_bytes",
            "value": 5540,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/cycles",
            "value": 11,
            "range": "median 11 max 11 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/cycles_per_kop",
            "value": 11000,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 169.68,
            "range": "± 3.7; min 140.4 max 264.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2856,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/cycles",
            "value": 14,
            "range": "median 14 max 14 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/cycles_per_kop",
            "value": 14000,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 153.4,
            "range": "± 7.9; min 141.3 max 251.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2880,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/cycles",
            "value": 2877,
            "range": "median 2939 max 2990 n=10; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/cycles_per_kop",
            "value": 468.262,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 231.5,
            "range": "± 8.0; min 215.8 max 347.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 14777,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 10200,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/cycles",
            "value": 2875,
            "range": "median 2939 max 2990 n=84; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/cycles_per_kop",
            "value": 467.936,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 1503.5,
            "range": "± 86.8; min 1355.4 max 2206.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 14777,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 10200,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/cycles",
            "value": 274,
            "range": "median 274 max 274 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/cycles_per_kop",
            "value": 35.677,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 197.72,
            "range": "± 18.3; min 174.8 max 364.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 3640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/cycles",
            "value": 337,
            "range": "median 337 max 337 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/cycles_per_kop",
            "value": 175.521,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 203.91,
            "range": "± 8.6; min 175.6 max 350.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 3656,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/cycles",
            "value": 132,
            "range": "median 132 max 132 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/cycles_per_kop",
            "value": 68.75,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 167.37,
            "range": "± 13.3; min 150.7 max 268.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 10201,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 5188,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/cycles",
            "value": 101,
            "range": "median 103 max 128 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/cycles_per_kop",
            "value": 52.604,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 189.09,
            "range": "± 6.7; min 158.7 max 228.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3196,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/cycles",
            "value": 101,
            "range": "median 103 max 128 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/cycles_per_kop",
            "value": 52.604,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 184.37,
            "range": "± 10.1; min 160.9 max 325.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3196,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/cycles",
            "value": 137,
            "range": "median 137 max 163 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/cycles_per_kop",
            "value": 23.785,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 188.34,
            "range": "± 19.1; min 164.5 max 332.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3488,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/cycles",
            "value": 444,
            "range": "median 472 max 500 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/cycles_per_kop",
            "value": 12.847,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 187.63,
            "range": "± 9.5; min 170.9 max 284.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 9976,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 5064,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/cycles",
            "value": 2954,
            "range": "median 2959 max 2987 n=7; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/cycles_per_kop",
            "value": 1538.542,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 217.58,
            "range": "± 3.1; min 196.4 max 316.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 12153,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 7004,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/cycles",
            "value": 801,
            "range": "median 815 max 851 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/cycles_per_kop",
            "value": 3.056,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 165.88,
            "range": "± 10.3; min 150.0 max 320.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13161,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 3232,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/cycles",
            "value": 801,
            "range": "median 815 max 851 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/cycles_per_kop",
            "value": 3.056,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 174.19,
            "range": "± 11.2; min 147.6 max 314.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13161,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 3236,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/cycles",
            "value": 1331,
            "range": "median 1370 max 1583 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/cycles_per_kop",
            "value": 2.529,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 179.09,
            "range": "± 7.7; min 161.5 max 344.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 17897,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 4688,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/cycles",
            "value": 1331,
            "range": "median 1331 max 1345 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/cycles_per_kop",
            "value": 2.529,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/npu_us",
            "value": 172.5,
            "range": "± 2.4; min 166.4 max 199.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/xclbin_bytes",
            "value": 17913,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/core_elf_bytes",
            "value": 3792,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/cycles",
            "value": 1349,
            "range": "median 1379 max 1407 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/cycles_per_kop",
            "value": 3.431,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 183.07,
            "range": "± 8.4; min 159.9 max 337.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 17273,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 6628,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/cycles",
            "value": 1348,
            "range": "median 1378 max 1416 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/cycles_per_kop",
            "value": 3.428,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 177.83,
            "range": "± 4.9; min 162.0 max 295.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 17321,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 5764,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/cycles",
            "value": 990,
            "range": "median 990 max 990 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/cycles_per_kop",
            "value": 2.466,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/npu_us",
            "value": 181.45,
            "range": "± 16.7; min 160.5 max 358.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/xclbin_bytes",
            "value": 21289,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/core_elf_bytes",
            "value": 2828,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/cycles",
            "value": 801,
            "range": "median 815 max 851 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/cycles_per_kop",
            "value": 3.056,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 169.6,
            "range": "± 7.5; min 146.9 max 276.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13161,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 3232,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/cycles",
            "value": 6976,
            "range": "median 6986 max 6990 n=7; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/cycles_per_kop",
            "value": 2.957,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 223.61,
            "range": "± 2.6; min 210.3 max 336.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 46617,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 4664,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/cycles",
            "value": 6969,
            "range": "median 6979 max 6985 n=7; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/cycles_per_kop",
            "value": 2.954,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 262.86,
            "range": "± 35.5; min 214.8 max 515.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 46617,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 4668,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/cycles",
            "value": 2387,
            "range": "median 2423 max 2460 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/cycles_per_kop",
            "value": 8.88,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/npu_us",
            "value": 186.66,
            "range": "± 9.6; min 158.9 max 302.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/xclbin_bytes",
            "value": 17849,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/core_elf_bytes",
            "value": 9268,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/cycles",
            "value": 1861,
            "range": "median 1864 max 1871 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/cycles_per_kop",
            "value": 8.113,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/npu_us",
            "value": 190.93,
            "range": "± 13.8; min 155.7 max 432.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/xclbin_bytes",
            "value": 14073,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/core_elf_bytes",
            "value": 9268,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/cycles",
            "value": 5596,
            "range": "median 5609 max 5650 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/cycles_per_kop",
            "value": 13.577,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/npu_us",
            "value": 202.31,
            "range": "± 5.8; min 190.8 max 315.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/xclbin_bytes",
            "value": 27769,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/core_elf_bytes",
            "value": 9268,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/cycles",
            "value": 2721,
            "range": "median 2872 max 3022 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/cycles_per_kop",
            "value": 20.246,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/npu_us",
            "value": 187.57,
            "range": "± 6.2; min 165.7 max 316.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/xclbin_bytes",
            "value": 22649,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/core_elf_bytes",
            "value": 9268,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles",
            "value": 1904,
            "range": "median 1904 max 1907 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles_per_kop",
            "value": 7.083,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/npu_us",
            "value": 176.78,
            "range": "± 8.7; min 160.8 max 279.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/xclbin_bytes",
            "value": 18089,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/core_elf_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/cycles",
            "value": 1254,
            "range": "median 1259 max 1266 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/cycles_per_kop",
            "value": 7.289,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/npu_us",
            "value": 174.35,
            "range": "± 11.4; min 157.1 max 304.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/xclbin_bytes",
            "value": 14825,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/core_elf_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/cycles",
            "value": 5107,
            "range": "median 5165 max 5237 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/cycles_per_kop",
            "value": 9.5,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/npu_us",
            "value": 207.32,
            "range": "± 4.1; min 188.0 max 363.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/xclbin_bytes",
            "value": 32489,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/core_elf_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/cycles",
            "value": 5730,
            "range": "median 5910 max 6092 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/cycles_per_kop",
            "value": 15.226,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/npu_us",
            "value": 248.74,
            "range": "± 44.3; min 191.8 max 404.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/xclbin_bytes",
            "value": 40169,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/core_elf_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles",
            "value": 2069,
            "range": "median 2069 max 2073 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles_per_kop",
            "value": 7.665,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/npu_us",
            "value": 188.46,
            "range": "± 12.7; min 165.8 max 280.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/xclbin_bytes",
            "value": 19529,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/core_elf_bytes",
            "value": 10996,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/cycles",
            "value": 2072,
            "range": "median 2072 max 2075 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/cycles_per_kop",
            "value": 7.676,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/npu_us",
            "value": 174.85,
            "range": "± 3.4; min 168.2 max 290.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/xclbin_bytes",
            "value": 19321,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/core_elf_bytes",
            "value": 10668,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/cycles",
            "value": 1532,
            "range": "median 1542 max 1553 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/cycles_per_kop",
            "value": 7.861,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/npu_us",
            "value": 172.93,
            "range": "± 6.9; min 158.7 max 321.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/xclbin_bytes",
            "value": 16457,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/core_elf_bytes",
            "value": 10996,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/cycles",
            "value": 2316,
            "range": "median 2316 max 2316 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/cycles_per_kop",
            "value": 17.196,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/npu_us",
            "value": 208.74,
            "range": "± 32.1; min 164.3 max 370.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/xclbin_bytes",
            "value": 24329,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/core_elf_bytes",
            "value": 10996,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/cycles",
            "value": 4651,
            "range": "median 4667 max 4675 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/cycles_per_kop",
            "value": 11.254,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/npu_us",
            "value": 205.06,
            "range": "± 5.8; min 187.0 max 328.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/xclbin_bytes",
            "value": 29241,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/core_elf_bytes",
            "value": 10668,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/cycles",
            "value": 1618,
            "range": "median 1662 max 1690 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/cycles_per_kop",
            "value": 6.27,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/npu_us",
            "value": 180.05,
            "range": "± 2.3; min 167.1 max 328.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/xclbin_bytes",
            "value": 16137,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/core_elf_bytes",
            "value": 12636,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/cycles",
            "value": 4646,
            "range": "median 4716 max 4751 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/cycles_per_kop",
            "value": 76.819,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/npu_us",
            "value": 227.83,
            "range": "± 3.5; min 199.0 max 365.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/core_elf_bytes",
            "value": 8780,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/cycles",
            "value": 3656,
            "range": "median 3665 max 3682 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/cycles_per_kop",
            "value": 100.75,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/npu_us",
            "value": 221.45,
            "range": "± 6.4; min 188.9 max 347.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/xclbin_bytes",
            "value": 12889,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/core_elf_bytes",
            "value": 8084,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/cycles",
            "value": 9080,
            "range": "median 9082 max 9099 n=7; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/cycles_per_kop",
            "value": 214.475,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/npu_us",
            "value": 268.04,
            "range": "± 16.5; min 245.6 max 442.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/xclbin_bytes",
            "value": 15257,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/core_elf_bytes",
            "value": 8084,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/cycles",
            "value": 8407,
            "range": "median 8407 max 8414 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/cycles_per_kop",
            "value": 181.31,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/npu_us",
            "value": 229.9,
            "range": "± 2.4; min 222.6 max 319.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/xclbin_bytes",
            "value": 14313,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/core_elf_bytes",
            "value": 8780,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/cycles",
            "value": 12053,
            "range": "median 12053 max 12065 n=5; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/cycles_per_kop",
            "value": 199.289,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/npu_us",
            "value": 282.2,
            "range": "± 17.9; min 256.5 max 360.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/xclbin_bytes",
            "value": 17609,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/insts_bytes",
            "value": 464,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/core_elf_bytes",
            "value": 9440,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/cycles",
            "value": 4494,
            "range": "median 4494 max 4494 n=8; init[2] min 22",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/cycles_per_kop",
            "value": 33.438,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/npu_us",
            "value": 196.31,
            "range": "± 9.4; min 176.5 max 364.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/xclbin_bytes",
            "value": 27065,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/core_elf_bytes",
            "value": 15704,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/cycles",
            "value": 459,
            "range": "median 459 max 459 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/cycles_per_kop",
            "value": 22.412,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/npu_us",
            "value": 162.58,
            "range": "± 4.8; min 147.7 max 169.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/xclbin_bytes",
            "value": 20473,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/core_elf_bytes",
            "value": 4872,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/cycles",
            "value": 359,
            "range": "median 359 max 359 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/cycles_per_kop",
            "value": 17.529,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/npu_us",
            "value": 167.21,
            "range": "± 7.0; min 148.3 max 287.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/xclbin_bytes",
            "value": 20473,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/core_elf_bytes",
            "value": 4872,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/cycles",
            "value": 85,
            "range": "median 95 max 136 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/cycles_per_kop",
            "value": 83.008,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 185.6,
            "range": "± 9.8; min 160.9 max 232.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3708,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/cycles",
            "value": 85,
            "range": "median 95 max 138 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/cycles_per_kop",
            "value": 83.008,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 182.18,
            "range": "± 10.1; min 160.2 max 322.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3708,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/cycles",
            "value": 429,
            "range": "median 429 max 429 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/cycles_per_kop",
            "value": 104.736,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 164.92,
            "range": "± 6.8; min 145.0 max 216.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 11849,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 7244,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/cycles",
            "value": 509,
            "range": "median 509 max 509 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/cycles_per_kop",
            "value": 82.845,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 179.29,
            "range": "± 9.4; min 153.8 max 291.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 11145,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 5916,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/cycles",
            "value": 1621,
            "range": "median 1621 max 1622 n=11; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/cycles_per_kop",
            "value": 263.835,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/npu_us",
            "value": 179.64,
            "range": "± 4.4; min 171.3 max 289.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/xclbin_bytes",
            "value": 12265,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/core_elf_bytes",
            "value": 7160,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/cycles",
            "value": 2757,
            "range": "median 2757 max 2757 n=8; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/cycles_per_kop",
            "value": 336.548,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/npu_us",
            "value": 195.13,
            "range": "± 1.6; min 186.1 max 335.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/xclbin_bytes",
            "value": 20585,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/core_elf_bytes",
            "value": 7268,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/cycles",
            "value": 261,
            "range": "median 261 max 266 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/cycles_per_kop",
            "value": 84.961,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/npu_us",
            "value": 167.5,
            "range": "± 3.8; min 158.2 max 283.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 9576,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 3848,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/cycles",
            "value": 337,
            "range": "median 337 max 337 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/cycles_per_kop",
            "value": 41.138,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/npu_us",
            "value": 167.39,
            "range": "± 6.8; min 159.0 max 349.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/xclbin_bytes",
            "value": 16537,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/core_elf_bytes",
            "value": 11936,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/cycles",
            "value": 2441,
            "range": "median 2447 max 2454 n=10; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/cycles_per_kop",
            "value": 297.974,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/npu_us",
            "value": 204.54,
            "range": "± 7.8; min 186.0 max 318.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/xclbin_bytes",
            "value": 16537,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/core_elf_bytes",
            "value": 11936,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/cycles",
            "value": 3764,
            "range": "median 3767 max 3776 n=9; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/cycles_per_kop",
            "value": 459.473,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/npu_us",
            "value": 238.52,
            "range": "± 7.9; min 217.2 max 343.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/xclbin_bytes",
            "value": 16537,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/core_elf_bytes",
            "value": 11936,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/cycles",
            "value": 337,
            "range": "median 337 max 337 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/cycles_per_kop",
            "value": 41.138,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/npu_us",
            "value": 171.95,
            "range": "± 7.2; min 154.3 max 343.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/xclbin_bytes",
            "value": 16537,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/core_elf_bytes",
            "value": 11936,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles",
            "value": 666,
            "range": "median 666 max 666 n=15; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles_per_kop",
            "value": 36.133,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 174.31,
            "range": "± 12.9; min 151.5 max 321.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 10313,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 4788,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles",
            "value": 666,
            "range": "median 666 max 666 n=15; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles_per_kop",
            "value": 36.133,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 187.71,
            "range": "± 4.0; min 161.2 max 226.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 10313,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 4788,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/cycles",
            "value": 127,
            "range": "median 132 max 136 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/cycles_per_kop",
            "value": 49.609,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/npu_us",
            "value": 180.86,
            "range": "± 6.9; min 166.4 max 347.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/xclbin_bytes",
            "value": 11049,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/core_elf_bytes",
            "value": 7748,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/cycles",
            "value": 6546,
            "range": "median 6588 max 6590 n=3; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/cycles_per_kop",
            "value": 1598.145,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/npu_us",
            "value": 274.47,
            "range": "± 7.4; min 258.9 max 354.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/xclbin_bytes",
            "value": 14521,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/core_elf_bytes",
            "value": 9244,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/cycles",
            "value": 1639,
            "range": "median 1658 max 1663 n=13; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/cycles_per_kop",
            "value": 1600.586,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/npu_us",
            "value": 198.81,
            "range": "± 7.7; min 177.4 max 217.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/xclbin_bytes",
            "value": 14521,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/core_elf_bytes",
            "value": 9244,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/cycles",
            "value": 270,
            "range": "median 298 max 513 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/cycles_per_kop",
            "value": 65.918,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/npu_us",
            "value": 221.51,
            "range": "± 1.9; min 207.3 max 267.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/xclbin_bytes",
            "value": 9143,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/4096x16/bfloat16/llama-prefill-ffn/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/cycles",
            "value": 142,
            "range": "median 156 max 257 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/cycles_per_kop",
            "value": 69.336,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/npu_us",
            "value": 189.93,
            "range": "± 10.2; min 177.6 max 349.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/xclbin_bytes",
            "value": 9143,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/cycles",
            "value": 78,
            "range": "median 85 max 129 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/npu_us",
            "value": 169.35,
            "range": "± 6.9; min 156.1 max 288.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/xclbin_bytes",
            "value": 9143,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/cycles",
            "value": 30,
            "range": "median 31 max 34 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/cycles_per_kop",
            "value": 117.188,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/npu_us",
            "value": 167.98,
            "range": "± 3.9; min 147.8 max 280.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/xclbin_bytes",
            "value": 9111,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/core_elf_bytes",
            "value": 3264,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/cycles",
            "value": 142,
            "range": "median 156 max 259 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/cycles_per_kop",
            "value": 69.336,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/npu_us",
            "value": 191.93,
            "range": "± 2.1; min 175.2 max 198.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/xclbin_bytes",
            "value": 9143,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/core_elf_bytes",
            "value": 3248,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/cycles",
            "value": 30,
            "range": "median 31 max 35 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/cycles_per_kop",
            "value": 117.188,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/npu_us",
            "value": 158.97,
            "range": "± 4.9; min 149.1 max 292.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/xclbin_bytes",
            "value": 9111,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/core_elf_bytes",
            "value": 3216,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/cycles",
            "value": 3306,
            "range": "median 3308 max 3320 n=8; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/cycles_per_kop",
            "value": 1614.258,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/npu_us",
            "value": 220.54,
            "range": "± 16.4; min 201.3 max 347.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/xclbin_bytes",
            "value": 17529,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/core_elf_bytes",
            "value": 13700,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/cycles",
            "value": 68,
            "range": "median 68 max 68 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/cycles_per_kop",
            "value": 354.167,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/npu_us",
            "value": 161.19,
            "range": "± 7.0; min 148.1 max 275.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/xclbin_bytes",
            "value": 9352,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/core_elf_bytes",
            "value": 3600,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/cycles",
            "value": 1041,
            "range": "median 1041 max 1041 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/npu_us",
            "value": 244.23,
            "range": "± 10.2; min 230.2 max 385.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/xclbin_bytes",
            "value": 9432,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/core_elf_bytes",
            "value": 3656,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/cycles",
            "value": 4896,
            "range": "median 4968 max 5029 n=4; init[2] min 16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/cycles_per_kop",
            "value": 1125,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/npu_us",
            "value": 181.27,
            "range": "± 6.2; min 169.3 max 347.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/xclbin_bytes",
            "value": 17321,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/insts_bytes",
            "value": 464,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/core_elf_bytes",
            "value": 13292,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/cycles",
            "value": 5456,
            "range": "median 5580 max 5580 n=3; init[2] min 16; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/cycles_per_kop",
            "value": 1253.676,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/npu_us",
            "value": 181.11,
            "range": "± 2.6; min 167.4 max 194.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/xclbin_bytes",
            "value": 17321,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/insts_bytes",
            "value": 464,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/core_elf_bytes",
            "value": 13292,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/cycles",
            "value": 1462,
            "range": "median 1464 max 1466 n=4; init[0] min 518",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/cycles_per_kop",
            "value": 22.308,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/npu_us",
            "value": 179.14,
            "range": "± 10.7; min 163.2 max 222.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/xclbin_bytes",
            "value": 9688,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/core_elf_bytes",
            "value": 3848,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/cycles",
            "value": 3077,
            "range": "median 3110 max 3141 n=4; init[0] min 518",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/cycles_per_kop",
            "value": 23.476,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/npu_us",
            "value": 179.52,
            "range": "± 5.6; min 164.8 max 346.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/core_elf_bytes",
            "value": 4028,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
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
          "id": "d53582d3e0f9f8a2b77695bbf4abbd7766d5584a",
          "message": "Single-core Kernel Optimizations and Tooling (#3801)\n\nCo-authored-by: Claude Opus 5 <noreply@anthropic.com>\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
          "timestamp": "2026-09-28T20:24:43Z",
          "url": "https://github.com/Xilinx/mlir-aie/commit/d53582d3e0f9f8a2b77695bbf4abbd7766d5584a"
        },
        "date": 1790663757417,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "passthrough/2048x16/int32/cycles",
            "value": 264,
            "range": "median 264 max 264 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/cycles_per_kop",
            "value": 128.906,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/npu_us",
            "value": 193.56,
            "range": "± 1.6; min 172.8 max 202.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x16/int32/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles",
            "value": 264,
            "range": "median 264 max 264 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/cycles_per_kop",
            "value": 128.906,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/npu_us",
            "value": 1539.55,
            "range": "± 27.7; min 895.5 max 1643.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/2048x256/int32/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles",
            "value": 264,
            "range": "median 264 max 264 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/cycles_per_kop",
            "value": 64.453,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/npu_us",
            "value": 197.06,
            "range": "± 6.8; min 179.9 max 214.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/int16/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles",
            "value": 136,
            "range": "median 136 max 136 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/cycles_per_kop",
            "value": 33.203,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/npu_us",
            "value": 190.5,
            "range": "± 6.5; min 162.7 max 326.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/4096x16/uint8/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/cycles",
            "value": 72,
            "range": "median 72 max 72 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/cycles_per_kop",
            "value": 70.312,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/npu_us",
            "value": 184.43,
            "range": "± 20.4; min 152.7 max 256.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "passthrough/1024x16/int16/core_elf_bytes",
            "value": 3020,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/cycles",
            "value": 78,
            "range": "median 78 max 78 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/npu_us",
            "value": 168.83,
            "range": "± 9.3; min 150.5 max 226.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int16/core_elf_bytes",
            "value": 3048,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/cycles",
            "value": 78,
            "range": "median 78 max 78 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/npu_us",
            "value": 292.2,
            "range": "± 2.0; min 277.6 max 377.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/xclbin_bytes",
            "value": 8919,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x256/int16/core_elf_bytes",
            "value": 3048,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/cycles",
            "value": 275,
            "range": "median 275 max 275 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/cycles_per_kop",
            "value": 268.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/npu_us",
            "value": 178.17,
            "range": "± 11.0; min 162.5 max 306.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/xclbin_bytes",
            "value": 9079,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "scale/1024x16/int32/core_elf_bytes",
            "value": 3208,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/cycles",
            "value": 78,
            "range": "median 87 max 131 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/npu_us",
            "value": 171.55,
            "range": "± 2.3; min 162.6 max 334.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x16/bfloat16/core_elf_bytes",
            "value": 3196,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/cycles",
            "value": 78,
            "range": "median 87 max 131 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/npu_us",
            "value": 431.3,
            "range": "± 11.7; min 413.1 max 1455.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add/1024x256/bfloat16/core_elf_bytes",
            "value": 3196,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/cycles",
            "value": 78,
            "range": "median 87 max 129 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/npu_us",
            "value": 172.22,
            "range": "± 3.8; min 157.8 max 270.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x16/bfloat16/core_elf_bytes",
            "value": 3244,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/cycles",
            "value": 78,
            "range": "median 87 max 129 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/npu_us",
            "value": 1346.28,
            "range": "± 78.1; min 600.7 max 1509.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul/1024x256/bfloat16/core_elf_bytes",
            "value": 3244,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/cycles",
            "value": 75,
            "range": "median 75 max 75 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/cycles_per_kop",
            "value": 73.242,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/npu_us",
            "value": 171.53,
            "range": "± 15.8; min 148.9 max 281.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x16/bfloat16/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/cycles",
            "value": 75,
            "range": "median 75 max 75 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/cycles_per_kop",
            "value": 73.242,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/npu_us",
            "value": 306,
            "range": "± 2.2; min 289.1 max 401.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/xclbin_bytes",
            "value": 8807,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "relu/1024x256/bfloat16/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/cycles",
            "value": 147,
            "range": "median 147 max 147 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/cycles_per_kop",
            "value": 143.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/npu_us",
            "value": 191.06,
            "range": "± 11.8; min 161.1 max 332.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x16/int32/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/cycles",
            "value": 147,
            "range": "median 147 max 147 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/cycles_per_kop",
            "value": 143.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/npu_us",
            "value": 1159.48,
            "range": "± 176.8; min 546.1 max 1694.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_add/1024x256/int32/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/cycles",
            "value": 147,
            "range": "median 147 max 147 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/cycles_per_kop",
            "value": 143.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/npu_us",
            "value": 175.82,
            "range": "± 5.6; min 164.8 max 304.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x16/int32/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/cycles",
            "value": 147,
            "range": "median 147 max 147 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/cycles_per_kop",
            "value": 143.555,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/npu_us",
            "value": 431.47,
            "range": "± 23.4; min 398.5 max 1520.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/xclbin_bytes",
            "value": 8791,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_min/1024x256/int32/core_elf_bytes",
            "value": 2900,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/cycles",
            "value": 155,
            "range": "median 155 max 155 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/cycles_per_kop",
            "value": 151.367,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/npu_us",
            "value": 171.69,
            "range": "± 2.5; min 162.7 max 200.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/xclbin_bytes",
            "value": 8823,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/int32/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/cycles",
            "value": 155,
            "range": "median 155 max 155 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/cycles_per_kop",
            "value": 151.367,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/npu_us",
            "value": 514.48,
            "range": "± 29.9; min 473.1 max 1296.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/xclbin_bytes",
            "value": 8823,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/int32/core_elf_bytes",
            "value": 2932,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/cycles",
            "value": 90,
            "range": "median 90 max 90 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/cycles_per_kop",
            "value": 87.891,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/npu_us",
            "value": 168.35,
            "range": "± 12.2; min 145.2 max 264.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x16/bfloat16/core_elf_bytes",
            "value": 2972,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/cycles",
            "value": 90,
            "range": "median 90 max 90 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/cycles_per_kop",
            "value": 87.891,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/npu_us",
            "value": 329.45,
            "range": "± 41.4; min 277.6 max 1224.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "reduce_max/1024x256/bfloat16/core_elf_bytes",
            "value": 2972,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/cycles",
            "value": 1785,
            "range": "median 1795 max 1802 n=12; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/cycles_per_kop",
            "value": 1743.164,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/npu_us",
            "value": 194.21,
            "range": "± 15.5; min 177.0 max 392.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/xclbin_bytes",
            "value": 14617,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x16/bfloat16/core_elf_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/cycles",
            "value": 1785,
            "range": "median 1795 max 1802 n=97; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/cycles_per_kop",
            "value": 1743.164,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/npu_us",
            "value": 684.47,
            "range": "± 4.5; min 662.7 max 835.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/xclbin_bytes",
            "value": 14617,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gelu/1024x256/bfloat16/core_elf_bytes",
            "value": 9336,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/cycles",
            "value": 1637,
            "range": "median 1659 max 1662 n=13; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/cycles_per_kop",
            "value": 1598.633,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/npu_us",
            "value": 182.75,
            "range": "± 2.8; min 174.0 max 346.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/xclbin_bytes",
            "value": 14521,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x16/bfloat16/core_elf_bytes",
            "value": 9240,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/cycles",
            "value": 1637,
            "range": "median 1659 max 1663 n=103; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/cycles_per_kop",
            "value": 1598.633,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/npu_us",
            "value": 1196.92,
            "range": "± 95.3; min 683.2 max 1576.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/xclbin_bytes",
            "value": 14521,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu/1024x256/bfloat16/core_elf_bytes",
            "value": 9240,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/cycles",
            "value": 1057,
            "range": "median 1088 max 1106 n=12; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/cycles_per_kop",
            "value": 1032.227,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/npu_us",
            "value": 172.02,
            "range": "± 8.0; min 161.2 max 324.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/xclbin_bytes",
            "value": 14281,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x16/bfloat16/core_elf_bytes",
            "value": 8888,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/cycles",
            "value": 1056,
            "range": "median 1100 max 1106 n=95; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/cycles_per_kop",
            "value": 1031.25,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/npu_us",
            "value": 1084.41,
            "range": "± 157.2; min 573.2 max 1581.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/xclbin_bytes",
            "value": 14281,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bf16_exp/1024x256/bfloat16/core_elf_bytes",
            "value": 8888,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/cycles",
            "value": 774,
            "range": "median 801 max 829 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/cycles_per_kop",
            "value": 755.859,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/npu_us",
            "value": 167.55,
            "range": "± 7.5; min 158.5 max 288.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/xclbin_bytes",
            "value": 14473,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x16/bfloat16/core_elf_bytes",
            "value": 9052,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/cycles",
            "value": 800,
            "range": "median 801 max 829 n=201; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/cycles_per_kop",
            "value": 781.25,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/npu_us",
            "value": 404.85,
            "range": "± 22.6; min 374.3 max 1414.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/xclbin_bytes",
            "value": 14473,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "tanh/1024x256/bfloat16/core_elf_bytes",
            "value": 9052,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/cycles",
            "value": 1339,
            "range": "median 1376 max 1381 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/cycles_per_kop",
            "value": 1307.617,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/npu_us",
            "value": 186.16,
            "range": "± 10.2; min 167.6 max 234.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/xclbin_bytes",
            "value": 14361,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x16/bfloat16/core_elf_bytes",
            "value": 9088,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/cycles",
            "value": 1368,
            "range": "median 1380 max 1381 n=141; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/cycles_per_kop",
            "value": 1335.938,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/npu_us",
            "value": 536.72,
            "range": "± 15.5; min 517.7 max 684.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/xclbin_bytes",
            "value": 14361,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "sigmoid/1024x256/bfloat16/core_elf_bytes",
            "value": 9088,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/cycles",
            "value": 1845,
            "range": "median 1870 max 1875 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/cycles_per_kop",
            "value": 1801.758,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/npu_us",
            "value": 185.75,
            "range": "± 8.0; min 165.6 max 353.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/xclbin_bytes",
            "value": 17529,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x16/bfloat16/core_elf_bytes",
            "value": 13700,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/cycles",
            "value": 1865,
            "range": "median 1870 max 1875 n=137; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/cycles_per_kop",
            "value": 1821.289,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/npu_us",
            "value": 1316.58,
            "range": "± 102.8; min 845.4 max 1689.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/xclbin_bytes",
            "value": 17529,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/1024x256/bfloat16/core_elf_bytes",
            "value": 13700,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/cycles",
            "value": 188,
            "range": "median 188 max 188 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/cycles_per_kop",
            "value": 183.594,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/npu_us",
            "value": 156.18,
            "range": "± 4.3; min 143.0 max 279.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x16/bfloat16/core_elf_bytes",
            "value": 3132,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/cycles",
            "value": 188,
            "range": "median 188 max 188 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/cycles_per_kop",
            "value": 183.594,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/npu_us",
            "value": 278.91,
            "range": "± 1.7; min 273.3 max 429.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "leaky_relu/1024x256/bfloat16/core_elf_bytes",
            "value": 3132,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/cycles",
            "value": 11981,
            "range": "median 12012 max 12043 n=2; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/cycles_per_kop",
            "value": 11700.195,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/npu_us",
            "value": 1197.62,
            "range": "± 100.4; min 424.7 max 1374.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/xclbin_bytes",
            "value": 11241,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x16/float32/core_elf_bytes",
            "value": 5480,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/cycles",
            "value": 11981,
            "range": "median 12010 max 12060 n=18; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/cycles_per_kop",
            "value": 11700.195,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/npu_us",
            "value": 3418.82,
            "range": "± 64.2; min 3341.6 max 4137.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/xclbin_bytes",
            "value": 11241,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "exp2f_vec/1024x256/float32/core_elf_bytes",
            "value": 5480,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/cycles",
            "value": 87,
            "range": "median 97 max 140 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/cycles_per_kop",
            "value": 42.48,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/npu_us",
            "value": 175.06,
            "range": "± 6.7; min 159.8 max 251.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/xclbin_bytes",
            "value": 9448,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x16/bfloat16/core_elf_bytes",
            "value": 3736,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/cycles",
            "value": 87,
            "range": "median 94 max 140 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/cycles_per_kop",
            "value": 42.48,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/npu_us",
            "value": 1245.9,
            "range": "± 171.1; min 530.9 max 1457.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/xclbin_bytes",
            "value": 9448,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "axpy/1024x256/bfloat16/core_elf_bytes",
            "value": 3736,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/cycles",
            "value": 144,
            "range": "median 144 max 144 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/cycles_per_kop",
            "value": 140.625,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/npu_us",
            "value": 185.5,
            "range": "± 16.9; min 161.3 max 330.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x16/float32_bfloat16/core_elf_bytes",
            "value": 3252,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/cycles",
            "value": 144,
            "range": "median 144 max 144 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/cycles_per_kop",
            "value": 140.625,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/npu_us",
            "value": 1252.06,
            "range": "± 140.7; min 488.0 max 1619.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/xclbin_bytes",
            "value": 8983,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "convert_copy/1024x256/float32_bfloat16/core_elf_bytes",
            "value": 3252,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/cycles",
            "value": 161,
            "range": "median 161 max 161 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/cycles_per_kop",
            "value": 157.227,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/npu_us",
            "value": 156.57,
            "range": "± 4.8; min 147.3 max 252.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x16/uint8_bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/cycles",
            "value": 161,
            "range": "median 161 max 161 n=256",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/cycles_per_kop",
            "value": 157.227,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/npu_us",
            "value": 294.89,
            "range": "± 3.6; min 269.4 max 309.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/xclbin_bytes",
            "value": 8887,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "expand/576x256/uint8_bfloat16/core_elf_bytes",
            "value": 3000,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/cycles",
            "value": 141,
            "range": "median 141 max 141 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/npu_us",
            "value": 162.63,
            "range": "± 5.0; min 150.5 max 252.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=4/core_elf_bytes",
            "value": 2944,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/cycles",
            "value": 144,
            "range": "median 144 max 144 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/npu_us",
            "value": 168.16,
            "range": "± 9.5; min 148.8 max 218.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/xclbin_bytes",
            "value": 9031,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/bfloat16/subtile=8/core_elf_bytes",
            "value": 3136,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/cycles",
            "value": 63,
            "range": "median 63 max 63 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/npu_us",
            "value": 157.23,
            "range": "± 6.2; min 142.9 max 262.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint8/subtile=4/core_elf_bytes",
            "value": 2944,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/cycles",
            "value": 239,
            "range": "median 239 max 239 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/npu_us",
            "value": 183.78,
            "range": "± 9.1; min 159.2 max 277.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/xclbin_bytes",
            "value": 8871,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/1024x16/uint32/subtile=8/core_elf_bytes",
            "value": 2976,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/cycles",
            "value": 1881,
            "range": "median 1961 max 1973 n=13; init[2] min 518; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/cycles_per_kop",
            "value": 7.175,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/npu_us",
            "value": 232.22,
            "range": "± 8.9; min 213.3 max 1261.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/xclbin_bytes",
            "value": 10297,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/bfloat16_float32/core_elf_bytes",
            "value": 4700,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/cycles",
            "value": 1881,
            "range": "median 1961 max 1970 n=207; init[2] min 518; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/cycles_per_kop",
            "value": 7.175,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/npu_us",
            "value": 1496.33,
            "range": "± 85.9; min 1326.3 max 1959.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/xclbin_bytes",
            "value": 10297,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x256/bfloat16_float32/core_elf_bytes",
            "value": 4700,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/cycles",
            "value": 2969,
            "range": "median 3081 max 3113 n=16; init[2] min 518",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/cycles_per_kop",
            "value": 5.663,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/npu_us",
            "value": 1222.29,
            "range": "± 82.8; min 358.9 max 1374.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/xclbin_bytes",
            "value": 10633,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x64x64x16/bfloat16_float32/b_col_maj=True/core_elf_bytes",
            "value": 5044,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/cycles",
            "value": 3219,
            "range": "median 3281 max 3482 n=7; init[2] min 518; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/cycles_per_kop",
            "value": 12.28,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/npu_us",
            "value": 242.05,
            "range": "± 3.2; min 229.0 max 1290.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/xclbin_bytes",
            "value": 10201,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int16_int32/core_elf_bytes",
            "value": 4552,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/cycles",
            "value": 1449,
            "range": "median 1481 max 1513 n=16; init[2] min 518",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/cycles_per_kop",
            "value": 5.527,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/npu_us",
            "value": 231.54,
            "range": "± 12.6; min 212.4 max 389.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/xclbin_bytes",
            "value": 10056,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm/64x32x64x16/int8_int32/core_elf_bytes",
            "value": 4408,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/cycles",
            "value": 691,
            "range": "median 699 max 730 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/cycles_per_kop",
            "value": 21.088,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/npu_us",
            "value": 161.48,
            "range": "± 5.9; min 153.2 max 259.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/xclbin_bytes",
            "value": 9960,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "fused_mm/32x32x16x4/bfloat16/core_elf_bytes",
            "value": 4220,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/cycles",
            "value": 95,
            "range": "median 95 max 95 n=16; init[2] min 4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/cycles_per_kop",
            "value": 46.387,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/npu_us",
            "value": 172.44,
            "range": "± 10.8; min 154.6 max 381.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/xclbin_bytes",
            "value": 9768,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x32x16/int16_int32/core_elf_bytes",
            "value": 3820,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/cycles",
            "value": 1264,
            "range": "median 1264 max 1264 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/cycles_per_kop",
            "value": 77.148,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/npu_us",
            "value": 279.47,
            "range": "± 59.6; min 211.4 max 650.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/xclbin_bytes",
            "value": 12185,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/32x256x16/bfloat16/core_elf_bytes",
            "value": 6608,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/cycles",
            "value": 117,
            "range": "median 117 max 117 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/cycles_per_kop",
            "value": 228.516,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/npu_us",
            "value": 164.91,
            "range": "± 9.9; min 154.2 max 254.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/xclbin_bytes",
            "value": 10441,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x64x16/bfloat16/vec_size=32/core_elf_bytes",
            "value": 4824,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/cycles",
            "value": 694,
            "range": "median 694 max 694 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/cycles_per_kop",
            "value": 42.358,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/npu_us",
            "value": 237.19,
            "range": "± 7.8; min 214.2 max 294.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/xclbin_bytes",
            "value": 11449,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/4x2048x16/bfloat16/vec_size=64/core_elf_bytes",
            "value": 5948,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/cycles",
            "value": 1755,
            "range": "median 1755 max 1755 n=72; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/cycles_per_kop",
            "value": 107.117,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/npu_us",
            "value": 1530.64,
            "range": "± 108.0; min 1328.9 max 1827.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/xclbin_bytes",
            "value": 27273,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mv/8192x256/bfloat16/output_rows=256/vec_size=64/llama-decode-ffn-down/core_elf_bytes",
            "value": 5540,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/cycles",
            "value": 11,
            "range": "median 11 max 11 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/cycles_per_kop",
            "value": 11000,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/npu_us",
            "value": 161.8,
            "range": "± 9.1; min 140.3 max 187.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/xclbin_bytes",
            "value": 8839,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/1x16/int32/core_elf_bytes",
            "value": 2856,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/cycles",
            "value": 14,
            "range": "median 14 max 14 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/cycles_per_kop",
            "value": 14000,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/npu_us",
            "value": 156.61,
            "range": "± 12.2; min 138.3 max 283.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/xclbin_bytes",
            "value": 8855,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "compute_max/2x16/bfloat16/core_elf_bytes",
            "value": 2880,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/cycles",
            "value": 2877,
            "range": "median 2939 max 2990 n=10; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/cycles_per_kop",
            "value": 468.262,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/npu_us",
            "value": 224.92,
            "range": "± 5.5; min 209.4 max 365.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/xclbin_bytes",
            "value": 14777,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x16/bfloat16/core_elf_bytes",
            "value": 10200,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/cycles",
            "value": 2876,
            "range": "median 2939 max 2990 n=84; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/cycles_per_kop",
            "value": 468.099,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/npu_us",
            "value": 1579.2,
            "range": "± 17.6; min 1317.1 max 1827.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/xclbin_bytes",
            "value": 14777,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "swiglu/1024x256/bfloat16/core_elf_bytes",
            "value": 10200,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/cycles",
            "value": 274,
            "range": "median 274 max 274 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/cycles_per_kop",
            "value": 35.677,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/npu_us",
            "value": 191.21,
            "range": "± 9.9; min 176.1 max 349.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/xclbin_bytes",
            "value": 9271,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "gray2rgba/1920x16/uint8/core_elf_bytes",
            "value": 3640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/cycles",
            "value": 337,
            "range": "median 337 max 337 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/cycles_per_kop",
            "value": 175.521,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/npu_us",
            "value": 201.69,
            "range": "± 9.0; min 174.9 max 351.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/xclbin_bytes",
            "value": 9287,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2gray/7680x16/uint8/core_elf_bytes",
            "value": 3656,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/cycles",
            "value": 132,
            "range": "median 132 max 133 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/cycles_per_kop",
            "value": 68.75,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/npu_us",
            "value": 160.38,
            "range": "± 3.3; min 152.6 max 243.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/xclbin_bytes",
            "value": 10201,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "threshold/1920x16/uint8/core_elf_bytes",
            "value": 5188,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/cycles",
            "value": 101,
            "range": "median 103 max 128 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/cycles_per_kop",
            "value": 52.604,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/npu_us",
            "value": 169.03,
            "range": "± 2.0; min 158.1 max 216.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_or/1920x16/uint8/core_elf_bytes",
            "value": 3196,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/cycles",
            "value": 101,
            "range": "median 103 max 128 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/cycles_per_kop",
            "value": 52.604,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/npu_us",
            "value": 170.44,
            "range": "± 7.5; min 156.0 max 337.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/xclbin_bytes",
            "value": 9095,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bitwise_and/1920x16/uint8/core_elf_bytes",
            "value": 3196,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/cycles",
            "value": 137,
            "range": "median 137 max 163 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/cycles_per_kop",
            "value": 23.785,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/npu_us",
            "value": 189.22,
            "range": "± 4.2; min 163.4 max 311.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/xclbin_bytes",
            "value": 9384,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_weighted/1920x16/uint8/core_elf_bytes",
            "value": 3488,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/cycles",
            "value": 444,
            "range": "median 472 max 500 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/cycles_per_kop",
            "value": 12.847,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/npu_us",
            "value": 188.57,
            "range": "± 10.4; min 166.9 max 333.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/xclbin_bytes",
            "value": 9976,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "filter2d/1920x16/uint8/core_elf_bytes",
            "value": 5064,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/cycles",
            "value": 2954,
            "range": "median 2959 max 2987 n=7; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/cycles_per_kop",
            "value": 1538.542,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/npu_us",
            "value": 222.65,
            "range": "± 5.6; min 195.3 max 288.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/xclbin_bytes",
            "value": 12153,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rgba2hue/7680x16/uint8/core_elf_bytes",
            "value": 7004,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/cycles",
            "value": 801,
            "range": "median 815 max 851 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/cycles_per_kop",
            "value": 3.056,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/npu_us",
            "value": 171.4,
            "range": "± 4.4; min 148.4 max 248.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/xclbin_bytes",
            "value": 13161,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/int8_uint8/core_elf_bytes",
            "value": 3232,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/cycles",
            "value": 801,
            "range": "median 815 max 851 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/cycles_per_kop",
            "value": 3.056,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/npu_us",
            "value": 161.61,
            "range": "± 8.0; min 149.6 max 263.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/xclbin_bytes",
            "value": 13161,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_i8/2048x8/int8/core_elf_bytes",
            "value": 3236,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/cycles",
            "value": 1331,
            "range": "median 1370 max 1583 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/cycles_per_kop",
            "value": 2.529,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/npu_us",
            "value": 177.1,
            "range": "± 3.0; min 170.1 max 300.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/xclbin_bytes",
            "value": 17897,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/core_elf_bytes",
            "value": 4688,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/cycles",
            "value": 1331,
            "range": "median 1331 max 1345 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/cycles_per_kop",
            "value": 2.529,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/npu_us",
            "value": 170.16,
            "range": "± 3.6; min 159.3 max 304.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/xclbin_bytes",
            "value": 17913,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip/2048x8/uint8/input_channels=128/output_channels=64/int8_skip/core_elf_bytes",
            "value": 3792,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/cycles",
            "value": 1349,
            "range": "median 1379 max 1407 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/cycles_per_kop",
            "value": 3.431,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/npu_us",
            "value": 172.11,
            "range": "± 3.7; min 159.6 max 343.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/xclbin_bytes",
            "value": 17273,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/core_elf_bytes",
            "value": 6628,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/cycles",
            "value": 1348,
            "range": "median 1377 max 1416 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/cycles_per_kop",
            "value": 3.428,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/npu_us",
            "value": 181.96,
            "range": "± 12.8; min 156.2 max 339.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/xclbin_bytes",
            "value": 17321,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1_skip_init/1024x8/uint8/input_channels=64/skip_input_channels=32/int8_skip/core_elf_bytes",
            "value": 5764,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/cycles",
            "value": 990,
            "range": "median 990 max 990 n=4",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/cycles_per_kop",
            "value": 2.466,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/npu_us",
            "value": 168.32,
            "range": "± 2.4; min 149.6 max 270.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/xclbin_bytes",
            "value": 21289,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk14/12544x4/uint8_int8/core_elf_bytes",
            "value": 2828,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/cycles",
            "value": 801,
            "range": "median 815 max 851 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/cycles_per_kop",
            "value": 3.056,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/npu_us",
            "value": 163.05,
            "range": "± 4.5; min 151.8 max 222.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/xclbin_bytes",
            "value": 13161,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk1/2048x8/uint8/core_elf_bytes",
            "value": 3232,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/cycles",
            "value": 6977,
            "range": "median 6986 max 6992 n=7; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/cycles_per_kop",
            "value": 2.957,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/npu_us",
            "value": 214.81,
            "range": "± 1.6; min 205.9 max 358.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/xclbin_bytes",
            "value": 46617,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/int8_uint8/core_elf_bytes",
            "value": 4664,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/cycles",
            "value": 6969,
            "range": "median 6979 max 6985 n=7; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/cycles_per_kop",
            "value": 2.954,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/npu_us",
            "value": 227.58,
            "range": "± 3.1; min 210.2 max 378.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/xclbin_bytes",
            "value": 46617,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "conv2dk3/2048x8/uint8/core_elf_bytes",
            "value": 4668,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/cycles",
            "value": 2386,
            "range": "median 2423 max 2460 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/cycles_per_kop",
            "value": 8.876,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/npu_us",
            "value": 186.62,
            "range": "± 12.3; min 157.2 max 334.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/xclbin_bytes",
            "value": 17849,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120/core_elf_bytes",
            "value": 9268,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/cycles",
            "value": 1861,
            "range": "median 1864 max 1871 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/cycles_per_kop",
            "value": 8.113,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/npu_us",
            "value": 186.88,
            "range": "± 11.8; min 159.5 max 275.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/xclbin_bytes",
            "value": 14073,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64/core_elf_bytes",
            "value": 9268,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/cycles",
            "value": 5596,
            "range": "median 5609 max 5650 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/cycles_per_kop",
            "value": 13.577,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/npu_us",
            "value": 197.29,
            "range": "± 3.7; min 188.3 max 331.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/xclbin_bytes",
            "value": 27769,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184/core_elf_bytes",
            "value": 9268,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/cycles",
            "value": 2721,
            "range": "median 2872 max 3021 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/cycles_per_kop",
            "value": 20.246,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/npu_us",
            "value": 190.93,
            "range": "± 16.4; min 168.2 max 348.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/xclbin_bytes",
            "value": 22649,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu/560x8/int8_uint8/input_channels=80/input_width=7/output_channels=120/core_elf_bytes",
            "value": 9268,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles",
            "value": 1904,
            "range": "median 1904 max 1907 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles_per_kop",
            "value": 7.083,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/npu_us",
            "value": 173.95,
            "range": "± 4.3; min 158.4 max 329.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/xclbin_bytes",
            "value": 18089,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/core_elf_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/cycles",
            "value": 1254,
            "range": "median 1259 max 1266 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/cycles_per_kop",
            "value": 7.289,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/npu_us",
            "value": 163.39,
            "range": "± 1.8; min 159.3 max 280.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/xclbin_bytes",
            "value": 14825,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3584x8/uint8_int8/input_channels=64/input_width=56/output_channels=24/core_elf_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/cycles",
            "value": 5107,
            "range": "median 5165 max 5238 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/cycles_per_kop",
            "value": 9.5,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/npu_us",
            "value": 201.96,
            "range": "± 3.7; min 188.7 max 346.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/xclbin_bytes",
            "value": 32489,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80/core_elf_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/cycles",
            "value": 5730,
            "range": "median 5910 max 6092 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/cycles_per_kop",
            "value": 15.226,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/npu_us",
            "value": 206.89,
            "range": "± 5.0; min 191.8 max 367.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/xclbin_bytes",
            "value": 40169,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80/core_elf_bytes",
            "value": 9640,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles",
            "value": 2069,
            "range": "median 2069 max 2073 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/cycles_per_kop",
            "value": 7.665,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/npu_us",
            "value": 170.69,
            "range": "± 3.0; min 164.2 max 303.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/xclbin_bytes",
            "value": 19529,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/core_elf_bytes",
            "value": 10996,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/cycles",
            "value": 2072,
            "range": "median 2072 max 2075 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/cycles_per_kop",
            "value": 7.676,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/npu_us",
            "value": 177.78,
            "range": "± 4.5; min 164.9 max 325.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/xclbin_bytes",
            "value": 19321,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8/core_elf_bytes",
            "value": 10668,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/cycles",
            "value": 1532,
            "range": "median 1542 max 1550 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/cycles_per_kop",
            "value": 7.861,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/npu_us",
            "value": 180.49,
            "range": "± 12.9; min 159.5 max 326.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/xclbin_bytes",
            "value": 16457,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/4032x8/uint8_int8/input_channels=72/input_width=56/output_channels=24/core_elf_bytes",
            "value": 10996,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/cycles",
            "value": 2316,
            "range": "median 2316 max 2316 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/cycles_per_kop",
            "value": 17.196,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/npu_us",
            "value": 180.21,
            "range": "± 2.5; min 164.7 max 316.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/xclbin_bytes",
            "value": 24329,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/1680x8/uint8_int8/input_channels=240/input_width=7/output_channels=40/core_elf_bytes",
            "value": 10996,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/cycles",
            "value": 4658,
            "range": "median 4667 max 4675 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/cycles_per_kop",
            "value": 11.271,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/npu_us",
            "value": 201.91,
            "range": "± 10.0; min 185.2 max 356.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/xclbin_bytes",
            "value": 29241,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8/core_elf_bytes",
            "value": 10668,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/cycles",
            "value": 1618,
            "range": "median 1662 max 1690 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/cycles_per_kop",
            "value": 6.27,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/npu_us",
            "value": 181.87,
            "range": "± 7.3; min 163.3 max 202.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/xclbin_bytes",
            "value": 16137,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16/core_elf_bytes",
            "value": 12636,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/cycles",
            "value": 4646,
            "range": "median 4715 max 4751 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/cycles_per_kop",
            "value": 76.819,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/npu_us",
            "value": 220.97,
            "range": "± 12.6; min 201.3 max 367.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/xclbin_bytes",
            "value": 13737,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120/core_elf_bytes",
            "value": 8780,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/cycles",
            "value": 3656,
            "range": "median 3665 max 3682 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/cycles_per_kop",
            "value": 100.75,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/npu_us",
            "value": 212.69,
            "range": "± 9.2; min 194.0 max 357.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/xclbin_bytes",
            "value": 12889,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4032x8/uint8/input_channels=72/input_width=56/output_channels=72/stride=2/core_elf_bytes",
            "value": 8084,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/cycles",
            "value": 9080,
            "range": "median 9082 max 9099 n=7; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/cycles_per_kop",
            "value": 214.475,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/npu_us",
            "value": 294,
            "range": "± 39.8; min 241.0 max 451.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/xclbin_bytes",
            "value": 15257,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/4704x8/uint8/input_channels=336/input_width=14/output_channels=336/stride=2/core_elf_bytes",
            "value": 8084,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/cycles",
            "value": 8407,
            "range": "median 8407 max 8414 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/cycles_per_kop",
            "value": 181.31,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/npu_us",
            "value": 235.49,
            "range": "± 7.6; min 225.7 max 324.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/xclbin_bytes",
            "value": 14313,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw/2576x8/uint8/input_channels=184/input_width=14/output_channels=184/core_elf_bytes",
            "value": 8780,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/cycles",
            "value": 12053,
            "range": "median 12053 max 12065 n=5; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/cycles_per_kop",
            "value": 199.289,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/npu_us",
            "value": 272.25,
            "range": "± 10.4; min 249.8 max 645.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/xclbin_bytes",
            "value": 17609,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/insts_bytes",
            "value": 464,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk3_dw_out_split/3360x8/uint8_uint8+uint8/input_channels=480/input_width=7/output_split_channels=240/core_elf_bytes",
            "value": 9440,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/cycles",
            "value": 4494,
            "range": "median 4494 max 4494 n=8; init[2] min 22",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/cycles_per_kop",
            "value": 33.438,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/npu_us",
            "value": 192.56,
            "range": "± 5.4; min 183.1 max 324.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/xclbin_bytes",
            "value": 27065,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_conv2dk1_relu_xy_pool_padded/560x8/int8_uint16/input_channels=80/input_width=7/output_channels=120/core_elf_bytes",
            "value": 15704,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/cycles",
            "value": 459,
            "range": "median 459 max 459 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/cycles_per_kop",
            "value": 22.412,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/npu_us",
            "value": 166.8,
            "range": "± 10.4; min 148.5 max 234.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/xclbin_bytes",
            "value": 20473,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/core_elf_bytes",
            "value": 4872,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/cycles",
            "value": 359,
            "range": "median 359 max 359 n=8",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/cycles_per_kop",
            "value": 17.529,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/npu_us",
            "value": 160.03,
            "range": "± 2.5; min 152.4 max 280.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/xclbin_bytes",
            "value": 20473,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "bn_fc_relu_ui16_pad/1280x8/uint16/input_channels=1280/output_channels=8/fc1/core_elf_bytes",
            "value": 4872,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/cycles",
            "value": 85,
            "range": "median 95 max 136 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/cycles_per_kop",
            "value": 83.008,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/npu_us",
            "value": 178.72,
            "range": "± 11.6; min 157.9 max 197.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/core_elf_bytes",
            "value": 3708,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/cycles",
            "value": 85,
            "range": "median 95 max 138 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/cycles_per_kop",
            "value": 83.008,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/npu_us",
            "value": 172.74,
            "range": "± 3.5; min 161.3 max 336.3 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/xclbin_bytes",
            "value": 9400,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_add/1024x16/bfloat16/add/core_elf_bytes",
            "value": 3708,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/cycles",
            "value": 429,
            "range": "median 429 max 429 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/cycles_per_kop",
            "value": 104.736,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 168.87,
            "range": "± 5.2; min 147.8 max 181.9 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 11849,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rms_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 7244,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/cycles",
            "value": 509,
            "range": "median 509 max 509 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/cycles_per_kop",
            "value": 82.845,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/npu_us",
            "value": 164.53,
            "range": "± 7.1; min 149.3 max 261.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 11145,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 5916,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/cycles",
            "value": 1621,
            "range": "median 1621 max 1622 n=11; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/cycles_per_kop",
            "value": 263.835,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/npu_us",
            "value": 183.13,
            "range": "± 5.8; min 170.7 max 350.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/xclbin_bytes",
            "value": 12265,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_f32/1024x16/float32/cols=1024/core_elf_bytes",
            "value": 7160,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/cycles",
            "value": 2757,
            "range": "median 2757 max 2757 n=8; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/cycles_per_kop",
            "value": 336.548,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/npu_us",
            "value": 219.3,
            "range": "± 4.1; min 191.4 max 347.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/xclbin_bytes",
            "value": 20585,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "layer_norm_affine_cast/1024x16/float32_bfloat16/cols=1024/core_elf_bytes",
            "value": 7268,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/cycles",
            "value": 261,
            "range": "median 261 max 267 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/cycles_per_kop",
            "value": 84.961,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/npu_us",
            "value": 169.47,
            "range": "± 5.7; min 156.1 max 255.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/xclbin_bytes",
            "value": 9576,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/1024x16/bfloat16/cols=1024/core_elf_bytes",
            "value": 3848,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/cycles",
            "value": 337,
            "range": "median 337 max 337 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/cycles_per_kop",
            "value": 41.138,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/npu_us",
            "value": 173.11,
            "range": "± 4.9; min 160.9 max 331.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/xclbin_bytes",
            "value": 16537,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/identity/core_elf_bytes",
            "value": 11936,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/cycles",
            "value": 2441,
            "range": "median 2444 max 2454 n=10; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/cycles_per_kop",
            "value": 297.974,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/npu_us",
            "value": 215.44,
            "range": "± 2.5; min 192.2 max 228.5 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/xclbin_bytes",
            "value": 16537,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/silu/core_elf_bytes",
            "value": 11936,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/cycles",
            "value": 3764,
            "range": "median 3769 max 3776 n=9; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/cycles_per_kop",
            "value": 459.473,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/npu_us",
            "value": 234.41,
            "range": "± 11.2; min 212.1 max 385.7 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/xclbin_bytes",
            "value": 16537,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/gelu/core_elf_bytes",
            "value": 11936,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/cycles",
            "value": 337,
            "range": "median 337 max 337 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/cycles_per_kop",
            "value": 41.138,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/npu_us",
            "value": 179.03,
            "range": "± 10.6; min 157.8 max 348.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/xclbin_bytes",
            "value": 16537,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mm_activation_epilogue/1024x16/float32/relu/core_elf_bytes",
            "value": 11936,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles",
            "value": 666,
            "range": "median 666 max 666 n=15; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles_per_kop",
            "value": 36.133,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 171.6,
            "range": "± 12.1; min 150.0 max 380.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 10313,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 4788,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles",
            "value": 666,
            "range": "median 666 max 666 n=15; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/cycles_per_kop",
            "value": 36.133,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/npu_us",
            "value": 172.75,
            "range": "± 10.9; min 158.7 max 301.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/xclbin_bytes",
            "value": 10313,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_first/1040x16/bfloat16/kernel_size=9/seq_len=1024/core_elf_bytes",
            "value": 4788,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/cycles",
            "value": 127,
            "range": "median 132 max 136 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/cycles_per_kop",
            "value": 49.609,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/npu_us",
            "value": 179.17,
            "range": "± 4.4; min 164.2 max 313.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/xclbin_bytes",
            "value": 11049,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "dwconv1d_channels_last/256x16/bfloat16/channels=256/core_elf_bytes",
            "value": 7748,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/cycles",
            "value": 6546,
            "range": "median 6588 max 6588 n=3; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/cycles_per_kop",
            "value": 1598.145,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/npu_us",
            "value": 256.23,
            "range": "± 1.8; min 249.4 max 335.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/xclbin_bytes",
            "value": 14521,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/4096x16/bfloat16/llama-prefill/core_elf_bytes",
            "value": 9244,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/cycles",
            "value": 1639,
            "range": "median 1658 max 1663 n=13; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/cycles_per_kop",
            "value": 1600.586,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/npu_us",
            "value": 182.66,
            "range": "± 3.0; min 174.3 max 335.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/xclbin_bytes",
            "value": 14521,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "silu_sized/1024x16/bfloat16/llama-decode/core_elf_bytes",
            "value": 9244,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/cycles",
            "value": 142,
            "range": "median 156 max 257 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/cycles_per_kop",
            "value": 69.336,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/npu_us",
            "value": 190.9,
            "range": "± 6.7; min 177.8 max 353.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/xclbin_bytes",
            "value": 9143,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/2048x16/bfloat16/llama-prefill-attn-scale/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/cycles",
            "value": 78,
            "range": "median 85 max 129 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/cycles_per_kop",
            "value": 76.172,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/npu_us",
            "value": 193.05,
            "range": "± 23.6; min 165.3 max 342.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/xclbin_bytes",
            "value": 9143,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/1024x16/bfloat16/llama-decode-ffn/core_elf_bytes",
            "value": 3296,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/cycles",
            "value": 30,
            "range": "median 31 max 34 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/cycles_per_kop",
            "value": 117.188,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/npu_us",
            "value": 157.86,
            "range": "± 5.6; min 148.8 max 371.6 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/xclbin_bytes",
            "value": 9111,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mul_sized/256x16/bfloat16/llama-decode-attn-scale/core_elf_bytes",
            "value": 3264,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/cycles",
            "value": 142,
            "range": "median 156 max 259 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/cycles_per_kop",
            "value": 69.336,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/npu_us",
            "value": 191.58,
            "range": "± 7.2; min 175.3 max 350.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/xclbin_bytes",
            "value": 9143,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/2048x16/bfloat16/llama-prefill/core_elf_bytes",
            "value": 3248,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/cycles",
            "value": 30,
            "range": "median 31 max 35 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/cycles_per_kop",
            "value": 117.188,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/npu_us",
            "value": 165.4,
            "range": "± 10.4; min 144.1 max 288.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/xclbin_bytes",
            "value": 9111,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "add_sized/256x16/bfloat16/llama-decode/core_elf_bytes",
            "value": 3216,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/cycles",
            "value": 3306,
            "range": "median 3308 max 3320 n=8; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/cycles_per_kop",
            "value": 1614.258,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/npu_us",
            "value": 218.61,
            "range": "± 4.6; min 201.6 max 346.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/xclbin_bytes",
            "value": 17529,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "softmax/2048x16/bfloat16/llama-prefill/core_elf_bytes",
            "value": 13700,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/cycles",
            "value": 68,
            "range": "median 68 max 68 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/cycles_per_kop",
            "value": 354.167,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/npu_us",
            "value": 151.45,
            "range": "± 3.4; min 142.3 max 261.8 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/xclbin_bytes",
            "value": 9352,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "rope/64x16/bfloat16/cols=64/two_halves=True/llama/core_elf_bytes",
            "value": 3600,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/cycles",
            "value": 1041,
            "range": "median 1041 max 1041 n=16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/npu_us",
            "value": 250.33,
            "range": "± 6.5; min 229.9 max 348.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/xclbin_bytes",
            "value": 9432,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/insts_bytes",
            "value": 300,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "transpose/8192x16/bfloat16/subtile=8/llama-decode/core_elf_bytes",
            "value": 3656,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/cycles",
            "value": 4896,
            "range": "median 4968 max 5029 n=4; init[2] min 16",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/cycles_per_kop",
            "value": 1125,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/npu_us",
            "value": 196.93,
            "range": "± 9.9; min 170.7 max 342.1 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/xclbin_bytes",
            "value": 17321,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/insts_bytes",
            "value": 464,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/core_elf_bytes",
            "value": 13292,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/cycles",
            "value": 5454,
            "range": "median 5580 max 5580 n=3; init[2] min 16; truncated",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/cycles_per_kop",
            "value": 1253.217,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/npu_us",
            "value": 192.4,
            "range": "± 9.6; min 167.9 max 342.0 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/xclbin_bytes",
            "value": 17321,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/insts_bytes",
            "value": 464,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "mha_softmax/4096x4/bfloat16_bfloat16+bfloat16/diagonal/core_elf_bytes",
            "value": 13292,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/cycles",
            "value": 1462,
            "range": "median 1464 max 1466 n=4; init[0] min 518",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/cycles_per_kop",
            "value": 22.308,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/npu_us",
            "value": 181.23,
            "range": "± 2.2; min 166.9 max 199.4 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/xclbin_bytes",
            "value": 9688,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/8x8x512x4/bfloat16_float32/head_dim=512/core_elf_bytes",
            "value": 3848,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/cycles",
            "value": 3077,
            "range": "median 3110 max 3141 n=4; init[0] min 518",
            "unit": "cycles",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/cycles_per_kop",
            "value": 23.476,
            "unit": "cycles/1k-ops",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/npu_us",
            "value": 187.24,
            "range": "± 10.3; min 168.3 max 343.2 n=50",
            "unit": "us",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/xclbin_bytes",
            "value": 9832,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/insts_bytes",
            "value": 420,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          },
          {
            "name": "prefill_fv/16x16x256x4/bfloat16_float32/head_dim=256/core_elf_bytes",
            "value": 4028,
            "unit": "bytes",
            "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"
          }
        ]
      }
    ]
  }
}