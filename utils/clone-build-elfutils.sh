#!/usr/bin/env bash

##===- utils/clone-build-elfutils.sh -------------------------*- Script -*-===##
#
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script checks out and builds libelf.
# It only build the necessary libraries to minimize build time.
#
# INSTALL_DIR environment variable can be set externally to the install path
# if it is not set, a default path is chosen automatically
#
# Depends on: autoconf, flex, bison, gawk
##===----------------------------------------------------------------------===##

install_dir=${INSTALL_DIR:=elfutils}
HASH="airbin"

# download the source
if [[ ! -d $install_dir ]]; then
  git clone --branch $HASH --depth 1 https://github.com/jnider/elfutils.git $install_dir
fi

cd $install_dir

# Generate the makefile
if [[ ! -e Makefile ]]; then
	# Generate the config file
	if [[ ! -e config.in ]]; then
		echo "config.in not found - generating"
		autoreconf -v -f -i
		if [[ $? != 0 ]]; then
			echo "autoreconf failed"
			exit 1
		fi
	fi
	echo "Makefile not found - generating"
	./configure --program-prefix="air-" --disable-debuginfod --disable-libdebuginfod --enable-maintainer-mode
	if [[ $? != 0 ]]; then
		echo "configure failed"
		exit 1
	fi
fi


# build libeu.a, required for libelf.so
if [[ ! -e lib/libeu.a ]]; then
	echo "Building libeu.a"
	make -C lib
	if [[ $? != 0 ]]; then
		echo "build libeu.a failed"
		exit 1
	fi
fi

# build libelf.a, libelf_pic.a and libelf.so
if [[ ! -e libelf/libelf.a || ! -e libelf/libelf_pic.a || ! -e libelf/libelf.so ]]; then
	echo "Building libelf.a libelf_pic.a and libelf.so"
	make -C libelf
	if [[ $? != 0 ]]; then
		echo "build libelf failed"
		exit 1
	fi
fi

echo "Installed in $install_dir"
