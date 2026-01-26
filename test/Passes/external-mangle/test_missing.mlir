// RUN: aie-opt --aie-external-mangle %s 2>&1 | FileCheck %s

module {
  // CHECK: warning: Could not open object file: missing.o
  func.func private @foo() attributes { link_with = "missing.o" }
}
