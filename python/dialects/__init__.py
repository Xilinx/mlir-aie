# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# Marks aie.dialects as a regular (non-namespace) package. MLIR ships its
# dialect directory as a namespace package, but a static type checker can only
# resolve names re-exported through aie.dialects.aie (e.g. the tablegen'd
# WireBundle / use_lock / npu_* wildcards) when the parent is a regular
# package. Empty on purpose; adds no runtime behavior.
