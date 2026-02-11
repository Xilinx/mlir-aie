// RUN: clang -c %S/test_mangle.cc -o %t_mangle.o
// RUN: sed 's|PATH_TO_OBJ|%t_mangle.o|g' %s | aie-opt --aie-external-mangle | FileCheck %s

module {
  // Define the mangled name already
  func.func private @_Z9my_kerneli(%arg0: i32) { return }

  // CHECK: func.func private @_Z9my_kerneli_1(i32) attributes {link_name = "_Z9my_kerneli", link_with = "{{.*}}"}
  func.func private @"my_kernel(int)"(i32) attributes { link_with = "PATH_TO_OBJ" }
}
