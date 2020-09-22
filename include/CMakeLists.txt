# set(LLVM_TARGET_DEFINITIONS AIE.td)
# mlir_tablegen(AIE.h.inc -gen-op-decls)
# mlir_tablegen(AIE.cpp.inc -gen-op-defs)
# mlir_tablegen(AIEEnums.h.inc -gen-enum-decls)
# mlir_tablegen(AIEEnums.cpp.inc -gen-enum-defs)
# add_public_tablegen_target(MLIRAIEIncGen)
add_mlir_dialect(AIE AIE)
add_mlir_doc(AIE -gen-dialect-doc AIEDialect Dialects/)

set(LLVM_TARGET_DEFINITIONS AIE.td)
mlir_tablegen(AIEEnums.h.inc -gen-enum-decls)
mlir_tablegen(AIEEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIEEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS AIEPasses.td)
mlir_tablegen(AIEPasses.h.inc -gen-pass-decls -name AIE)
add_public_tablegen_target(MLIRAIEPassIncGen)

add_mlir_doc(AIEPasses -gen-pass-doc AIEPasses ./)

#set(LLVM_TARGET_DEFINITIONS AIEFindFlows.td)
#mlir_tablegen(AIEFindFlows.inc -gen-rewriters)
#add_public_tablegen_target(MLIRAIEFindFlowsIncGen)

# set(LLVM_TARGET_DEFINITIONS AIE.td)
# mlir_tablegen(LLVMConversions.inc -gen-llvmir-conversions)
# mlir_tablegen(LLVMConversionEnumsToLLVM.inc -gen-enum-to-llvmir-conversions)
# mlir_tablegen(LLVMConversionEnumsFromLLVM.inc -gen-enum-from-llvmir-conversions)
# add_public_tablegen_target(MLIRLLVMConversionsIncGen)
