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
# TODO: create a branch or a tag instead, to avoid fetching main and
# this commit later.
commithash=d36b483
# this is for CI (safely ignored if you're actually running this script)
wheel_version=18.0.0.2023121201+$commithash

here=$PWD

if [ x"$1" == x--get-wheel-version ]; then
  echo $wheel_version
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
    git worktree add --force "$here"/llvm $commithash
  )
else
  # Fetch main first just to clone
  git clone --depth 1 https://github.com/llvm/llvm-project.git llvm
  (
    cd llvm
    # Then fetch the interesting part
    git fetch --depth=1 origin $commithash
    git checkout $commithash
  )
fi
