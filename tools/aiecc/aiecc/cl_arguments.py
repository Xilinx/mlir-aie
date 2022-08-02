#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

import argparse
import os
import shlex
import sys
import shutil

from aiecc.configure import *

def parse_args():
    parser = argparse.ArgumentParser(prog='aiecc')
    parser.add_argument('filename',
            metavar="file",
            help='MLIR file to compile')
    parser.add_argument('--sysroot',
            metavar="sysroot",
            default="",
            help='sysroot for cross-compilation')
    parser.add_argument('--tmpdir',
            metavar="tmpdir",
            default="acdc_project",
            help='directory used for temporary file storage')
    parser.add_argument('-v',
            dest="verbose",
            default=False,
            action='store_true',
            help='Trace commands as they are executed')
    parser.add_argument('--vectorize',
            dest="vectorize",
            default=False,
            action='store_true',
            help='Enable MLIR vectorization')
    parser.add_argument('--xbridge',
            dest="xbridge",
            default=aie_link_with_xchesscc,
            action='store_true',
            help='Link using xbridge')
    parser.add_argument('--no-xbridge',
            dest="xbridge",
            default=not aie_link_with_xchesscc,
            action='store_false',
            help='Link using peano')
    parser.add_argument('--xchesscc',
            dest="xchesscc",
            default=aie_compile_with_xchesscc,
            action='store_true',
            help='Compile using xchesscc')
    parser.add_argument('--no-xchesscc',
            dest="xchesscc",
            default=not aie_compile_with_xchesscc,
            action='store_false',
            help='Compile using peano')
    parser.add_argument('--compile',
            dest="compile",
            default=not aie_disable_compile,
            action='store_true',
            help='Enable compiling of AIE code')
    parser.add_argument('--no-compile',
            dest="compile",
            default=aie_disable_compile,
            action='store_false',
            help='Disable compiling of AIE code')
    parser.add_argument('--host-target',
            dest="host_target",
            default="",
            help='Target architecture of the host program')
    parser.add_argument('--compile-host',
            dest="compile_host",
            default=not host_disable_compile,
            action='store_true',
            help='Enable compiling of the host program')
    parser.add_argument('--no-compile-host',
            dest="compile_host",
            default=host_disable_compile,
            action='store_false',
            help='Disable compiling of the host program')
    parser.add_argument('--link',
            dest="link",
            default=not aie_disable_link,
            action='store_true',
            help='Enable linking of AIE code')
    parser.add_argument('--no-link',
            dest="link",
            default=aie_disable_link,
            action='store_false',
            help='Disable linking of AIE code')
    parser.add_argument('--aie-generate-xaiev2',
            dest="xaie",
            default=2,
            action='store_const', const=2,
            help='Generate libxaie v2 drivers (default is v2)')
    parser.add_argument('--aie-generate-xaie',
            dest="xaie",
            default=2,
            action='store_const', const=1,
            help='Generate libxaie v1 drivers (default is v2)')
    parser.add_argument('--aie-generate-airbin',
            dest="airbin",
            default=False,
            action='store_const', const=True,
            help='Generate airbin configuration (default is off)')
    parser.add_argument("arm_args",
            action='store',
            help='arguments for ARM compiler',
            nargs=argparse.REMAINDER)
    parser.add_argument('-j',
            dest="nthreads",
            default=1,
            action='store',
            help='Compile with max n-threads in the machine (default is 1).  An argument of zero corresponds to the maximum number of threads on the machine.')
    parser.add_argument('--profile',
            dest="profiling",
            default=False,
            action='store_true',
            help='Profile commands to find the most expensive executions.')
    parser.add_argument('-n',
            dest="execute",
            default=True,
            action='store_false',
            help='Disable actually executing any commands.')
    parser.add_argument('--progress',
            dest="progress",
            default=False,
            action='store_true',
            help='Show progress visualization')


    opts = parser.parse_args(sys.argv[1:])

    return opts


def _positive_int(arg):
    return _int(arg, 'positive', lambda i: i > 0)


def _non_negative_int(arg):
    return _int(arg, 'non-negative', lambda i: i >= 0)


def _int(arg, kind, pred):
    desc = "requires {} integer, but found '{}'"
    try:
        i = int(arg)
    except ValueError:
        raise _error(desc, kind, arg)
    if not pred(i):
        raise _error(desc, kind, arg)
    return i


def _case_insensitive_regex(arg):
    import re
    try:
        return re.compile(arg, re.IGNORECASE)
    except re.error as reason:
        raise _error("invalid regular expression: '{}', {}", arg, reason)


def _error(desc, *args):
    msg = desc.format(*args)
    return argparse.ArgumentTypeError(msg)
