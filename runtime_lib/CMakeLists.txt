# Stuff into the build area:
add_custom_target(copy-runtime-libs ALL DEPENDS ${PROJECT_BINARY_DIR}/runtime_lib/me_basic.o)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/runtime_lib/me_basic.o
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.o
                   ${PROJECT_BINARY_DIR}/runtime_lib/me_basic.o
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/me_basic.o)

# Install too
set(INSTALLS me_basic.o test_library.cpp test_library.h chess_intrinsic_wrapper.cpp)
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)
