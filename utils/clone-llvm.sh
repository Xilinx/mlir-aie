#!/usr/bin/env bash
##===- utils/clone-llvm.sh - Build LLVM for github workflow --*- Script -*-===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# This script checks out LLVM.  We use this instead of a git submodule to avoid
# excessive copies of the LLVM tree.
#
##===----------------------------------------------------------------------===##

# The LLVM commit to use.
LLVM_PROJECT_COMMIT=5cfc7b3342ce4de0bbe182b38baa8a71fc83f8f8
DATETIME=2023122517
WHEEL_VERSION=18.0.0.$DATETIME+${LLVM_PROJECT_COMMIT:0:8}
############################################################################################
# The way to bump `LLVM_PROJECT_COMMIT`
#   1. Find the hash you want (`git rev-parse --short=8 HEAD` or just copy paste from github);
#   2. Go to mlir-aie github actions and launch an MLIR Distro workflow to build against that hash (see docs/Dev.md);
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
  git clone --depth 1 https://github.com/llvm/llvm-project.git llvm
  (
    cd llvm
    # Then fetch the interesting part
    git fetch --depth=1 origin $LLVM_PROJECT_COMMIT
    git checkout $LLVM_PROJECT_COMMIT
  )
fi
