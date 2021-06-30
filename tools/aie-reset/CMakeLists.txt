add_executable(aie-reset aie-reset.cpp)

include_directories(${XILINX_XAIE_INCLUDE_DIR})
link_libraries(${XILINX_XAIE_LIBS})

install(TARGETS aie-reset
EXPORT AIETargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT aie-reset)
