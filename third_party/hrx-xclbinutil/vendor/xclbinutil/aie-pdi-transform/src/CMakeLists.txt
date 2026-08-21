# CMakeLists.txt in src/

# The PDI header structs use C11 _Alignas/_Static_assert instead of compiler-specific
# attributes. gcc/clang default to gnu17 so this is a no-op for them, but MSVC's C mode
# defaults to C89-with-extensions and needs /std:c11 for those keywords.

# -fvisibility is a GCC/Clang-driver flag with no MSVC-driver equivalent (symbol
# visibility is not a concept for this static lib on Windows). CMake sets MSVC=1 for
# both cl and clang-cl, so this correctly skips it for either Windows compiler; without
# the guard cl warns "D9002: ignoring unknown option '-fvisibility=default'".
if(NOT MSVC)
  add_compile_options(-fvisibility=default)
endif()

# This builds a static library libtransformcdo, which 
# includes all the *.c files in lib/
add_library( transformcdo STATIC
    pdi-parsing.c
    cdo-parsing.c
    pdi-transform.c
    ../lib/cdo_cmd.c
    ../lib/load_pdi.c
    ../lib/pdi_transform.c
)

set_target_properties(transformcdo PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

if(MSVC)
  # These sources use fopen/strcpy; the "unsafe, use fopen_s" deprecation is noise here.
  target_compile_definitions(transformcdo PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  # Two classes of bug that gcc/clang accept silently and Windows does not. Make them
  # errors here so the Linux jobs catch them before the cl/clang-cl jobs ever run.
  #
  # pointer-arith: arithmetic on void* is a GNU extension; cl rejects it outright
  #   (C2036 "unknown size"). -pedantic-errors does NOT diagnose it, and would fail
  #   these sources for unrelated reasons -- this is the flag that actually works.
  # frame-larger-than: Windows' default thread stack is 1 MiB against Linux's 8 MiB.
  #   pdi_transform() had 1000 KiB of arrays as locals and only ever crashed on Windows.
  target_compile_options(transformcdo PRIVATE
    -Werror=pointer-arith
    -Werror=frame-larger-than=65536)
endif()

# Set C flags
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra -fvisibility=default")

target_include_directories(transformcdo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../libinclude
    ${CMAKE_CURRENT_SOURCE_DIR}
)

