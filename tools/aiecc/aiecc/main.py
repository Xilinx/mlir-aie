"""
aiecc - AIE compiler driver for MLIR tools
"""

import itertools
import os
import platform
import sys
import time
from subprocess import PIPE, run, call
import tempfile

import aiecc.cl_arguments

def do_call(command):
    if(opts.verbose):
        print(" ".join(command))
    ret = call(command)
    if(ret != 0):
        print("Error encountered while running: " + " ".join(command))
        sys.exit(1)

def main(builtin_params={}):
    global opts
    opts = aiecc.cl_arguments.parse_args()
    is_windows = platform.system() == 'Windows'

    if(opts.verbose):
        sys.stderr.write('\ncompiling %s\n' % opts.filename)

    thispath = os.path.dirname(os.path.realpath(__file__))
    me_basic_o = os.path.join(thispath, '..','..','runtime_lib', 'me_basic.o')

    # with tempfile.TemporaryDirectory() as tmpdirname:
    tmpdirname = "acdc_project"
    try:
        os.mkdir(tmpdirname)
    except FileExistsError:
        pass

    if(opts.verbose):
        print('created temporary directory', tmpdirname)

    file_with_addresses = os.path.join(tmpdirname, 'input_with_addresses.mlir')
    do_call(['aie-opt', '--aie-assign-buffer-addresses', '-convert-scf-to-std', opts.filename, '-o', file_with_addresses])
    t = run(['aie-translate', '--aie-generate-corelist', file_with_addresses], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    cores = eval(t.stdout)

    def corefile(dirname, core, ext):
        (corecol, corerow) = core
        return os.path.join(dirname, 'core_%d_%d.%s' % (corecol, corerow, ext))

    def tmpcorefile(core, ext):
        return corefile(tmpdirname, core, ext)
        
    for core in cores:
        (corecol, corerow) = core
        file_core = tmpcorefile(core, "mlir")
        do_call(['aie-opt', '--aie-standard-lowering=tilecol=%d tilerow=%d' % core,
                            '--convert-std-to-llvm=use-bare-ptr-memref-call-conv', file_with_addresses, '-o', file_core])
        #do_call(['aie-opt', '--aie-llvm-lowering=tilecol=%d tilerow=%d' % core, file_with_addresses, '-o', file_core])
        file_core_bcf = tmpcorefile(core, "bcf")
        do_call(['aie-translate', file_with_addresses, '--aie-generate-bcf', '--tilecol=%d' % corecol, '--tilerow=%d' % corerow, '-o', file_core_bcf])
        file_core_ldscript = tmpcorefile(core, "ld.script")
        do_call(['aie-translate', file_with_addresses, '--aie-generate-ldscript', '--tilecol=%d' % corecol, '--tilerow=%d' % corerow, '-o', file_core_ldscript])
        file_core_llvmir = tmpcorefile(core, "ll")
        do_call(['aie-translate', '--aie-generate-llvmir', file_core, '-o', file_core_llvmir])
        file_core_llvmir_stripped = tmpcorefile(core, "stripped.ll")
        do_call(['opt', '-strip', '-S', file_core_llvmir, '-o', file_core_llvmir_stripped])
        file_core_elf = corefile(".", core, "elf")
        file_core_obj = tmpcorefile(core, "o")
        do_call(['llc', file_core_llvmir_stripped, '-O2', '--march=aie', '--filetype=obj', '-o', file_core_obj])
        if(opts.xbridge == True):
          do_call(['xbridge', file_core_obj, '-c', file_core_bcf, '-o', file_core_elf])
        else:
          do_call(['clang', '-O2', '--target=aie', file_core_obj, me_basic_o, '-Wl,-T,'+file_core_ldscript, '-o', file_core_elf])

    # Generate the included host interface
    file_physical = os.path.join(tmpdirname, 'input_physical.mlir')
    do_call(['aie-opt', '--aie-create-flows', file_with_addresses, '-o', file_physical]);
    file_inc_cpp = os.path.join(tmpdirname, 'aie_inc.cpp')
    do_call(['aie-translate', '--aie-generate-xaie', file_physical, '-o', file_inc_cpp])

    # Lastly, compile the generated host interface with any ARM code.
    cmd = ['clang','--target=aarch64-linux-gnu', '-std=c++11']
    if(opts.sysroot):
      cmd += ['--sysroot=%s' % opts.sysroot]
    cmd += ['-I%s/opt/xaiengine/include' % opts.sysroot]
    cmd += ['-L%s/opt/xaiengine/lib' % opts.sysroot]
    cmd += ['-Iacdc_project']
    cmd += ['-fuse-ld=lld','-rdynamic','-lxaiengine','-lmetal','-lopen_amp','-ldl']

    if(len(opts.arm_args) > 0):
      do_call(cmd + opts.arm_args)
