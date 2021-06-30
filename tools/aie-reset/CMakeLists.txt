add_llvm_executable(aie-reset aie-reset.cpp)

install(TARGETS aie-reset
EXPORT AIETargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT aie-reset)
