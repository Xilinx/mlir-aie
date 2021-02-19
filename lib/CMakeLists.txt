add_mlir_dialect_library(AIE
  AIEAssignBuffers.cpp
  AIEDialect.cpp
  AIETokenAnalysis.cpp
  AIENetlistAnalysis.cpp
  AIEFindFlows.cpp
  AIECreateFlows.cpp
  AIECreateCores.cpp
  AIECreateLocks.cpp
  AIECoreToLLVM.cpp
  AIECoreToStandard.cpp
  AIEHerdRouting.cpp
  AIECreatePacketFlows.cpp
  AIELowerMemcpy.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEPassIncGen
  )

target_link_libraries(AIE
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )

add_subdirectory(Targets)
