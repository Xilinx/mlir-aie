# Copyright (C) 2021 Xilinx, Inc.
# Copyright (C) 2022 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# This is the GPIO-DEMO apllication recipe
#
#

SUMMARY = "gpio-demo application"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"
SRC_URI = "file://gpio-demo.c \
           file://Makefile \
        "
S = "${WORKDIR}"
CFLAGS_prepend = "-I ${S}/include"
do_compile() {
        oe_runmake
}
do_install() {
        install -d ${D}${bindir}
        install -m 0755 ${S}/gpio-demo ${D}${bindir}

}

