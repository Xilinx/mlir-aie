<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
# Used in / Cited in

IRON and the [MLIR-AIE](https://github.com/Xilinx/mlir-aie) toolchain are used across a growing body of work on AI Engine and Ryzen AI NPU programming — from research papers that build on or benchmark against them, to higher-level compilers and DSLs that target IRON as a backend, to runtimes and products that ship it as their kernel layer. This page collects a representative selection of that work, alongside the project's relationship to the wider LLVM/MLIR ecosystem. It is curated rather than exhaustive, and favors entries that others can verify and follow.

If you have built on IRON or MLIR-AIE and would like your work considered for this page, open a pull request. Submissions are reviewed for relevance and are not guaranteed to be listed.

## How to cite

Work that uses IRON should cite:

> E. Hunhoff, J. Melber, K. Denolf, A. Bisca, S. Bayliss, S. Neuendorffer, J. Fifield, J. Lo, P. Vasireddy, P. James-Roxby, and E. Keller. "Efficiency, Expressivity, and Extensibility in a Close-to-Metal NPU Programming Interface." 33rd IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM), 2025. [DOI: 10.1109/FCCM62733.2025.00043](https://doi.org/10.1109/FCCM62733.2025.00043)

## Repositories

- [amd/IRON](https://github.com/amd/IRON) — IRON operators, AIE kernels, and example applications, built on the MLIR-AIE Python bindings.
- [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) — the MLIR-based toolchain, dialects, lowering passes, and the IRON Python API bindings for AI Engine devices.
- [Xilinx/llvm-aie (Peano)](https://github.com/Xilinx/llvm-aie) — the LLVM fork adding the AI Engine as a target architecture.

## Research using IRON / MLIR-AIE

- Shouyu Du, Miaoxiang Yu, Zhenyu Xu, Zhiheng Ni, Jillian Cai, Qing Yang, and Tao Wei. "[Mapping Gemma3 onto an Edge Dataflow Architecture](https://arxiv.org/abs/2602.06063)." arXiv:2602.06063, 2026. First end-to-end deployment of the Gemma3 family (language and vision, 1B and 4B) on the AMD Ryzen AI NPU, implemented with MLIR-AIE and the IRON interface. Introduces FlowQKV, FusedDQP, FlowKV, and a Q4NX 4-bit quantization format.
- Endri Taka, Andre Roesti, Joseph Melber, Pranathi Vasireddy, Kristof Denolf, and Diana Marculescu. "[Striking the Balance: GEMM Performance Optimization Across Generations of Ryzen AI NPUs](https://arxiv.org/abs/2512.13282)." arXiv:2512.13282, 2025. Uses IRON for fine-grained control of explicit data movement and DMA access patterns in a multi-level GEMM tiling methodology.
- André Rösti and Michael Franz. "[Unlocking the AMD Neural Processing Unit for ML Training on the Client Using Bare-Metal Programming Tools](https://arxiv.org/abs/2504.03083)." FCCM 2025 (arXiv:2504.03083). Uses the IRON tool-flow to accelerate client-side inference and fine-tuning of GPT-2.

## Frameworks and DSLs that target IRON

- [MLIR-AIR](https://github.com/Xilinx/mlir-air) — a spatial-compute compiler stack whose backend maps AIR constructs onto MLIR-AIE (per-tile code, DMA descriptors, hardware locks), installing MLIR-AIE as a pinned dependency. Described in Erwei Wang et al., "[From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR](https://arxiv.org/abs/2510.14871)" (arXiv:2510.14871, 2025), with matrix-multiplication and LLaMA-2 multi-head-attention case studies.
- Jinming Zhuang, Shaojie Xiang, Hongzheng Chen, Niansong Zhang, Zhuoping Yang, Tony Mao, Zhiru Zhang, and Peipei Zhou. "[ARIES: An Agile MLIR-Based Compilation Flow for Reconfigurable Devices with AI Engines](https://doi.org/10.1145/3706628.3708870)." FPGA '25. An independent MLIR compilation flow for AIE devices, with and without FPGA fabric.
- Shihan Fang, Hongzheng Chen, Niansong Zhang, Jiajie Li, Han Meng, Adrian Liu, and Zhiru Zhang. "[Dato: A Task-Based Programming Model for Dataflow Accelerators](https://arxiv.org/abs/2509.06794)." arXiv:2509.06794, 2025. A Python-embedded task-based model for FPGA and NPU dataflow accelerators; emits MLIR-AIE as its NPU backend and benchmarks against IRON, demonstrating GEMM and fused attention on XDNA.
- [Stream](https://github.com/KULeuven-MICAS/stream) (package `stream-dse`) — a design-space-exploration and constraint-optimization framework for heterogeneous dataflow accelerators. Treats AIE as a native dataflow core type and provides AIE MLIR code generation for the Ryzen AI NPU (GEMM and SwiGLU on AMD Strix), feeding the MLIR-AIE / IRON toolchain. Described in A. Symons, L. Mei, S. Colleman, P. Houshmand, S. Karl, and M. Verhelst, "[Stream: Design Space Exploration of Layer-Fused DNNs on Heterogeneous Dataflow Accelerators](https://kuleuven-micas.github.io/stream/)," IEEE Transactions on Computers, 2025.

## Runtimes and products built on IRON

- [FastFlowLM (FLM)](https://github.com/FastFlowLM/FastFlowLM) — an NPU-first LLM and vision runtime for Ryzen AI XDNA2 with an Ollama-style interface. Its low-level compute kernels are optimized with IRON and MLIR-AIE. The orchestration and CLI are open source (MIT); the NPU-accelerated kernels are distributed as proprietary binaries.
- [Lemonade](https://github.com/lemonade-sdk/lemonade) — an open-source, OpenAI-compatible local LLM server. Provides Ryzen AI NPU acceleration through its FastFlowLM (`flm`) backend, and therefore relies on the IRON / MLIR-AIE kernel path for NPU execution. It is multi-backend, also orchestrating llama.cpp, OnnxRuntime GenAI, and whisper.cpp, and provides a separate Windows Ryzen AI NPU path distinct from FLM.

## Relationship to the LLVM / MLIR ecosystem

MLIR-AIE is built on LLVM/MLIR infrastructure and tracks it closely; releases pin the toolchain to specific LLVM commits. The host LLVM/MLIR distribution is sourced from AMD's [ROCm/llvm-project](https://github.com/ROCm/llvm-project) fork, a downstream fork that tracks upstream LLVM. AI Engine code generation is provided separately through [Peano (llvm-aie)](https://github.com/Xilinx/llvm-aie), an LLVM fork that adds the AI Engine as a target architecture and enables clang-based frontends. This is a maintained fork that tracks upstream rather than a merge of the AIE dialects into the LLVM monorepo.

## Coverage

- Phoronix, March 17, 2026. "[AMD MLIR-AIE Releases New AIECC C++ Compiler To Help Bring New Workloads To Ryzen AI NPUs](https://www.phoronix.com/news/MLIR-AIE-1.3)." Covers the v1.3 release and the C++ `aiecc` compiler driver.
- Phoronix, January 24, 2026. "[AMD Releases MLIR-AIE 1.2 Compiler Toolchain For Targeting Ryzen AI NPUs](https://www.phoronix.com/news/AMD-MLIR-AIE-1.2)." Notes the Python 3.14 wheel, a new IRON host runtime abstraction layer, WSL compatibility work, and Strix BF16 matmul optimizations.

## Presentations

Tutorials and workshop presentations on IRON and MLIR-AIE, delivered at academic conferences. Each links to the slide deck (PDF) and, where available, a description of the session.

| Venue | Year | Title | Links |
|---|---|:--|:--:|
| ASPLOS | 2026 | IRON AI Engine API for Ryzen AI NPU | [PDF](https://www.amd.com/content/dam/amd/en/documents/solutions/ai/iron-aie-api-for-ryzen-ai-npu-tutorial-asplos-2026.pdf) · [Details](conferenceDescriptions/asplos26TutorialDescription.md) |
| ISCA | 2025 | Leveraging the IRON AI Engine API to Program the Ryzen AI NPU | [PDF](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-isca-2025.pdf) · [Details](conferenceDescriptions/isca25TutorialDescription.md) |
| IPDPS | 2025 | Leveraging the IRON AI Engine API to Program the Ryzen AI NPU | [PDF](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-ipdps-2025.pdf) · [Details](conferenceDescriptions/ipdps25TutorialDescription.md) |
| MICRO | 2024 | Leveraging the IRON AI Engine API to Program the Ryzen AI NPU | [PDF](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-micro-2024.pdf) · [Details](conferenceDescriptions/micro24TutorialDescription.md) |
| ASPLOS | 2024 | Spatial Computing with AIR for Ryzen™ AI | [PDF](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/air-for-ryzen-ai-tutorial-asplos-2024.pdf) · [Details](conferenceDescriptions/asplos24TutorialDescription.md) |
| FCCM | 2023 | Leveraging MLIR to Design for AI Engines | [PDF](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/leveraging-mlir-to-design-for-aie-fccm-2023.pdf) |
| ISFPGA | 2023 | Leveraging MLIR to Design for AI Engines | [PDF](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/leveraging-mlir-to-design-for-aie-fpga-2023.pdf) |

<p align="center">Copyright&copy; 2019-2021 Xilinx, Inc.<br>Copyright&copy; 2022-2026 Advanced Micro Devices, Inc.</p>
