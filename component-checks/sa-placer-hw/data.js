window.BENCHMARK_DATA = {
  "lastUpdate": 1790924977315,
  "repoUrl": "https://github.com/Xilinx/mlir-aie",
  "entries": {
    "SA placer hardware check (mobilenet, npu2)": [
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
          "id": "18ca6c1cbf4cbb1b5d6748e09909bce30a5b5c15",
          "message": "AIE2P Kernel Coverage and MobileNet on Tuned Kernels (#3807)\n\nCo-authored-by: Claude <noreply@anthropic.com>\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>",
          "timestamp": "2026-10-02T02:08:45Z",
          "url": "https://github.com/Xilinx/mlir-aie/commit/18ca6c1cbf4cbb1b5d6748e09909bce30a5b5c15"
        },
        "date": 1790924974800,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "sa_placer/hw_fail_count",
            "value": 0,
            "unit": "seeds"
          },
          {
            "name": "sa_placer/hw_seed3_latency_us",
            "value": 295.7,
            "unit": "us"
          },
          {
            "name": "sa_placer/hw_seed2_latency_us",
            "value": 314.9,
            "unit": "us"
          },
          {
            "name": "sa_placer/hw_seed7_latency_us",
            "value": 303.7,
            "unit": "us"
          }
        ]
      }
    ]
  }
}