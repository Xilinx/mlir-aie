set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header -prefix Conversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl -prefix Conversion)
add_public_tablegen_target(MLIRAIEConversionPassIncGen)

add_mlir_doc(Passes MLIRAIEConversionPasses ./ -gen-pass-doc)
