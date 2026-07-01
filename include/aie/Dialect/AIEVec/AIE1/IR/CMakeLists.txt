#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_mlir_dialect(AIEVecAIE1Ops aievec_aie1)
add_mlir_doc(AIEVecAIE1Ops AIEVecAIE1Dialect ./ -gen-dialect-doc -dialect=aievec_aie1)

# AIEVecAIE1Ops.td pulls in AIEAttrs.td; ensure generated event enums are ready
add_dependencies(MLIRAIEVecAIE1OpsIncGen GenerateAIEEventsTD)
