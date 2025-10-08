
<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Programming Examples Utilities</ins>

These utilities are helpful in the current programming examples context and include helpful C/C++ libraries, and python and shell scripts.

- [Open CV Utilities](#open-cv-utilities-opencvutilsh) ([OpenCVUtils.h](./OpenCVUtils.h))
- [Clean microcode shell script](#clean-microcode-shell-script-clean_microcodesh) ([clean_microcode.sh](./clean_microcode.sh))

## <u>Open CV Utilities ([OpenCVUtils.h](./OpenCVUtils.h))</u>
OpenCV utilities used in vision processing pipelines to help read and/or initialize images and video. Currently supported functions include the following. Please view header for more specific function information. 
* imageCompare
* readImage
* initializeSingleGrayImageTest
* initializeSingleImageTest
* initializeVideoCapture
* initializeVideoFile
* addSaltPepperNoise
* medianBlur1D

## <u>Clean microcode shell script ([clean_microcode.sh](./clean_microcode.sh))</u>
Shell script to do in-place cleanup of microcode files (e.g. core_*.lst). When viewing microcode, it's helpful for some of the extra information like hardware and software breakpoints to be removed so it's easier to see back-to-back lines of microcode.
