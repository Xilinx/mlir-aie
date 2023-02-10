
find_package(Doxygen)

if (DOXYGEN_FOUND)
  set(abs_top_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
  set(abs_top_builddir ${CMAKE_CURRENT_BINARY_DIR})

  if (HAVE_DOT)
    set(DOT ${LLVM_PATH_DOT})
  endif()

  set(enable_searchengine "NO")
  set(searchengine_url "")
  set(enable_server_based_search "NO")
  set(enable_external_search "NO")
  set(extra_search_mappings "")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)

  set(abs_top_srcdir)
  set(abs_top_builddir)
  set(DOT)
  set(enable_searchengine)
  set(searchengine_url)
  set(enable_server_based_search)
  set(enable_external_search)
  set(extra_search_mappings)

  add_custom_target(doxygen-aie
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating aie doxygen documentation." VERBATIM)

  add_dependencies(docs doxygen-aie)

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
    DESTINATION docs/html)

endif()

