// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Honor the section directory; default to the current working directory.
// RUN: rm -rf section_directory && mkdir -p section_directory/into section_directory/cwd
// RUN: cd section_directory/cwd && aie-translate --aie-cert-to-asm --cert-section-dir=../into %s -o ../into/main.asm
// RUN: FileCheck %s --check-prefix=MAIN < ../into/main.asm
// RUN: FileCheck %s --check-prefix=BODY < ../into/config_device.asm
// RUN: not test -e config_device.asm
// RUN: aie-translate --aie-cert-to-asm %s -o ../into/default.asm
// RUN: FileCheck %s --check-prefix=BODY < config_device.asm
// RUN: not aie-translate --aie-cert-to-asm --cert-section-dir=../missing %s 2>&1 | FileCheck %s --check-prefix=ERROR

// MAIN: config_device:
// MAIN-NEXT: .include config_device.asm
// MAIN-NEXT: .endl config_device
// BODY: START_JOB 1
// BODY: WRITE_32               0x00001000, 0x00000001
// BODY: EOF
// ERROR: failed to open section file

module {
  aie.device(xcve3858) {
    aiex.cert.section @config_device {
      aiex.cert.page {
        aiex.cert.job(1) {
          aiex.cert.write32(0x1000, 1)
        }
      }
    }
  }
}
