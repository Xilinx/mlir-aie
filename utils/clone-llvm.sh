#!/usr/bin/env bash
##===- utils/clone-llvm.sh - Build LLVM for github workflow --*- Script -*-===##
#
# Copyright (C) 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# This script checks out LLVM.  We use this instead of a git submodule to avoid
# excessive copies of the LLVM tree.
#
# As of the ROCm/llvm-project migration, LLVM is sourced from the ROCm fork
# (github.com/ROCm/llvm-project) rather than upstream llvm/llvm-project.
#
##===----------------------------------------------------------------------===##

# The LLVM commit to use.
LLVM_PROJECT_COMMIT=46fcb339fb61119b337f973c7ca9e710a319fdd0
DATETIME=2026071405
WHEEL_VERSION=23.0.0.$DATETIME+${LLVM_PROJECT_COMMIT:0:8}

############################################################################################
# The way to bump `LLVM_PROJECT_COMMIT`
#   1. (Optional) If you want a particular hash of LLVM, get it (`git rev-parse --short=8 HEAD` or just copy paste the short hash from github);
#   2. Go to mlir-aie github actions and launch an MLIR Distro workflow to build LLVM wheels (see docs/Dev.md);
#   3. Look under the Get latest LLVM commit job -> Get llvm-project commit step -> DATETIME;
#   4. Record it here and push up a PR; the PR will fail until MLIR Distro workflow.
############################################################################################

here=$PWD

if [ x"$1" == x--get-wheel-version ]; then
  echo $WHEEL_VERSION
  exit 0
fi

# Use --worktree <directory-of-local-LLVM-repo> to reuse some existing
# local LLVM git repository
if [ x"$1" == x--llvm-worktree ]; then
  git_central_llvm_repo_dir="$2"
  (
    cd $git_central_llvm_repo_dir
    # Use force just in case there are various experimental iterations
    # after you have removed the llvm directory
    git worktree add --force "$here"/llvm $LLVM_PROJECT_COMMIT
  )
else
  # Fetch main first just to clone
  git clone --depth 1 https://github.com/ROCm/llvm-project.git llvm
  (
    cd llvm
    # Then fetch the interesting part
    git fetch --depth=1 origin $LLVM_PROJECT_COMMIT
    git checkout $LLVM_PROJECT_COMMIT
  )
fi
