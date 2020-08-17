add_mlir_dialect_library(AIE
  AIEDialect.cpp
  AIETokenAnalysis.cpp
  AIENetlistAnalysis.cpp
  AIEFindFlows.cpp
  AIECreateFlows.cpp
  AIECreateCores.cpp
  AIECreateLocks.cpp
  AIEBufferMerge.cpp
  AIECoreToLLVM.cpp
  AIEHerdRouting.cpp
  AIECreatePacketFlows.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEIncGen
  MLIRAIEEnumsIncGen
  )

target_link_libraries(AIE
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )

add_subdirectory(Targets)
