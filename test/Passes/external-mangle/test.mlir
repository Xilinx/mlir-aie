// RUN: clang -c %S/test_mangle.cc -o %t_mangle.o
// RUN: clang -c %S/test_mangle2.cc -o %t_mangle2.o
// RUN: sed 's|PATH_TO_OBJ1|%t_mangle.o|g' %s | sed 's|PATH_TO_OBJ2|%t_mangle2.o|g' | aie-opt --aie-external-mangle | FileCheck %s

module {
  // CHECK: func.func private @_Z9my_kerneli(i32)
  func.func private @"my_kernel(int)"(i32) attributes { link_with = "PATH_TO_OBJ1" }

  // CHECK: func.func private @_Z9my_kernelf(f32)
  func.func private @"my_kernel(float)"(f32) attributes { link_with = "PATH_TO_OBJ2" }
}
