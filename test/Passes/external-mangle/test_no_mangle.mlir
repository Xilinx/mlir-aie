// RUN: clang -c %S/test_mangle.cc -o %t_mangle.o
// RUN: sed 's|PATH_TO_OBJ|%t_mangle.o|g' %s | aie-opt --aie-external-mangle | FileCheck %s

module {
  // CHECK: func.func private @dummy1()
  func.func private @dummy1() attributes { link_with = "PATH_TO_OBJ" }
}
