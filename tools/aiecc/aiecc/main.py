#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

"""
aiecc - AIE compiler driver for MLIR tools
"""

import itertools
import os
import platform
import sys
import time
from subprocess import PIPE, run, call
from multiprocessing.pool import ThreadPool
import tempfile
import shutil
import timeit

import aiecc.cl_arguments

from rich.progress import *

runtimes = dict()
progress_bar = None
maxtasks = 5
stopall = False
def do_call(task, command):
    global opts
    global progress_bar
    global runtimes
    global maxtasks
    global stopall
    if(stopall):
      return

    if(opts.verbose):
        print(" ".join(command))
    commandstr = " ".join(command)
    if(task):
      progress_bar.update(task, advance=0, command=commandstr[0:30])
    start = time.time()
    ret = call(command)
    end = time.time()
    runtimes[commandstr] = end-start
    if(task):
      progress_bar.update(task, advance=1, command="")
      maxtasks = max(progress_bar._tasks[task].completed, maxtasks)
      progress_bar._tasks[task].total = maxtasks

    if(ret != 0):
        progress_bar._tasks[task].description = "[red] Error"
        print("Error encountered while running: " + commandstr)
        sys.exit(1)

def do_run(command):
    global opts
    if(opts.verbose):
        print(" ".join(command))
    ret = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    return ret

def run_flow(opts, tmpdirname):
    thispath = os.path.dirname(os.path.realpath(__file__))
    me_basic_o = os.path.join(thispath, '..','..','runtime_lib', 'me_basic.o')
    libc = os.path.join(thispath, '..','..','runtime_lib', 'libc.a')
    libm = os.path.join(thispath, '..','..','runtime_lib', 'libm.a')
    chess_intrinsic_wrapper_cpp = os.path.join(thispath, '..','..','runtime_lib', 'chess_intrinsic_wrapper.cpp')

    file_with_addresses = os.path.join(tmpdirname, 'input_with_addresses.mlir')
    do_call(None, ['aie-opt', '--lower-affine', '--aie-register-objectFifos', '--aie-objectFifo-stateful-transform', '--aie-lower-broadcast-packet', '--aie-create-packet-flows', '--aie-assign-buffer-addresses', '-convert-scf-to-cf', opts.filename, '-o', file_with_addresses])
    t = do_run(['aie-translate', '--aie-generate-corelist', file_with_addresses])
    cores = eval(t.stdout)

    if(opts.xchesscc == True):
      chess_intrinsic_wrapper = os.path.join(tmpdirname, 'chess_intrinsic_wrapper.ll')
      do_call(None, ['xchesscc_wrapper', '-c', '-d', '-f', '+f', '+P', '4', chess_intrinsic_wrapper_cpp, '-o', chess_intrinsic_wrapper])
      do_call(None, ['sed', '-i', 's/^target.*//', chess_intrinsic_wrapper])

      do_call(None, ['sed', '-i', 's/noalias_sidechannel[^,]*,//', chess_intrinsic_wrapper])
      do_call(None, ['sed', '-i', 's/nocallback[^,]*,//', chess_intrinsic_wrapper])

    def corefile(dirname, core, ext):
        (corecol, corerow, _) = core
        return os.path.join(dirname, 'core_%d_%d.%s' % (corecol, corerow, ext))

    def tmpcorefile(core, ext):
        return corefile(tmpdirname, core, ext)

    # Extract included files from the given Chess linker script.
    # We rely on gnu linker scripts to stuff object files into a compile.  However, the Chess compiler doesn't 
    # do this, so we have to explicitly specify included files on the link line.
    def extract_input_files(file_core_bcf):
        t = do_run(['awk', '/_include _file/ {print($3)}', file_core_bcf])
        return ' '.join(t.stdout.split())

    def process_core(core):
        global progress_bar
        global maxtasks
        global stopall
        if(stopall):
          return
        task = progress_bar.add_task("[yellow] Core (%d, %d)" % core[0:2], total=maxtasks, command="starting")

        (corecol, corerow, elf_file) = core
        file_core = tmpcorefile(core, "mlir")
        do_call(task, ['aie-opt', '--aie-localize-locks',
                            '--aie-standard-lowering=tilecol=%d tilerow=%d' % core[0:2],
                            file_with_addresses, '-o', file_core])

        file_opt_core = tmpcorefile(core, "opt.mlir")
        do_call(task, ['aie-opt', '--aie-normalize-address-spaces',
                            '--canonicalize',
                            '--cse',
                            '--convert-vector-to-llvm',
                            '--convert-memref-to-llvm',
                            '--convert-func-to-llvm=use-bare-ptr-memref-call-conv',
                            '--convert-cf-to-llvm',
                            '--canonicalize', '--cse', file_core, '-o', file_opt_core])
        file_core_bcf = tmpcorefile(core, "bcf")
        do_call(task, ['aie-translate', file_with_addresses, '--aie-generate-bcf', '--tilecol=%d' % corecol, '--tilerow=%d' % corerow, '-o', file_core_bcf])
        file_core_ldscript = tmpcorefile(core, "ld.script")
        do_call(task, ['aie-translate', file_with_addresses, '--aie-generate-ldscript', '--tilecol=%d' % corecol, '--tilerow=%d' % corerow, '-o', file_core_ldscript])
        file_core_llvmir = tmpcorefile(core, "ll")
        do_call(task, ['aie-translate', '--mlir-to-llvmir', '--opaque-pointers=0', file_opt_core, '-o', file_core_llvmir])
        file_core_elf = elf_file if elf_file else corefile(".", core, "elf")
        file_core_obj = tmpcorefile(core, "o")

        if(opts.compile and opts.xchesscc):
          file_core_llvmir_chesshack = tmpcorefile(core, "chesshack.ll")
          do_call(task, ['cp', file_core_llvmir, file_core_llvmir_chesshack])
          do_call(task, ['sed', '-i', 's/noundef//', file_core_llvmir_chesshack])
          do_call(task, ['sed', '-i', 's/noalias_sidechannel[^,],//', file_core_llvmir_chesshack])
          file_core_llvmir_chesslinked = tmpcorefile(core, "chesslinked.ll")
          do_call(task, ['llvm-link', file_core_llvmir_chesshack, chess_intrinsic_wrapper, '-S', '-o', file_core_llvmir_chesslinked])
          do_call(task, ['sed', '-i', 's/noundef//', file_core_llvmir_chesslinked])
          # Formal function argument names not used in older LLVM
          do_call(task, ['sed', '-i', '-E', '/define .*@/ s/%[0-9]*//g', file_core_llvmir_chesslinked])
          do_call(task, ['sed', '-i', '-E', 's/mustprogress//g', file_core_llvmir_chesslinked])
          do_call(task, ['sed', '-i', '-E', 's/poison/undef/g', file_core_llvmir_chesslinked])
          do_call(task, ['sed', '-i', '-E', 's/nocallback//g', file_core_llvmir_chesslinked])
          if(opts.link and opts.xbridge):
            link_with_obj = extract_input_files(file_core_bcf)
            do_call(task, ['xchesscc_wrapper', '-d', '-f', '+P', '4', file_core_llvmir_chesslinked, link_with_obj, '+l', file_core_bcf, '-o', file_core_elf])
          elif(opts.link):
            do_call(task, ['xchesscc_wrapper', '-c', '-d', '-f', '+P', '4', file_core_llvmir_chesslinked, '-o', file_core_obj])
            do_call(task, ['clang', '-O2', '--target=aie', file_core_obj, me_basic_o, libm,
            '-Wl,-T,'+file_core_ldscript, '-o', file_core_elf])
        elif(opts.compile):
          file_core_llvmir_stripped = tmpcorefile(core, "stripped.ll")
          do_call(task, ['opt', '--passes=default<O2>,strip', '-S', file_core_llvmir, '-o', file_core_llvmir_stripped])
          do_call(task, ['llc', file_core_llvmir_stripped, '-O2', '--march=aie', '--filetype=obj', '-o', file_core_obj])
          if(opts.link and opts.xbridge):
            link_with_obj = extract_input_files(file_core_bcf)
            do_call(task, ['xchesscc_wrapper', '-d', '-f', file_core_obj, link_with_obj, '+l', file_core_bcf, '-o', file_core_elf])
          elif(opts.link):
            do_call(task, ['clang', '-O2', '--target=aie', file_core_obj, me_basic_o, libm,
            '-Wl,-T,'+file_core_ldscript, '-o', file_core_elf])

        progress_bar.update(progress_bar.task_completed,advance=1)
        progress_bar.update(task,advance=0,visible=False)

    def process_arm_cgen():
      global progress_bar
      global stopall
      if(stopall):
        return
      task = progress_bar.add_task("[yellow] ARM Core ", total=10, command="starting")
      # Generate the included host interface
      file_physical = os.path.join(tmpdirname, 'input_physical.mlir')
      if(opts.pathfinder):
        do_call(task, ['aie-opt', '--aie-create-pathfinder-flows', '--aie-lower-broadcast-packet', '--aie-create-packet-flows', file_with_addresses, '-o', file_physical]);
      else:
        do_call(task, ['aie-opt', '--aie-create-flows', '--aie-lower-broadcast-packet', '--aie-create-packet-flows', file_with_addresses, '-o', file_physical]);
      file_inc_cpp = os.path.join(tmpdirname, 'aie_inc.cpp')
      if(opts.xaie == 1):
          do_call(task, ['aie-translate', '--aie-generate-xaie', '--xaie-target=v1', file_physical, '-o', file_inc_cpp])
      else:
          do_call(task, ['aie-translate', '--aie-generate-xaie', '--xaie-target=v2', file_physical, '-o', file_inc_cpp])

      if(opts.host_target):
        cmd = ['clang','--target=%s' % opts.host_target, '-std=c++11']
      else:
        cmd = ['clang','--target=aarch64-linux-gnu', '-std=c++11']
      if(opts.sysroot):
        cmd += ['--sysroot=%s' % opts.sysroot]
        # In order to find the toolchain in the sysroot, we need to have
        # a 'target' that includes 'linux' and for the 'lib/gcc/$target/$version'
        # directory to have a corresponding 'include/gcc/$target/$version'.
        # In some of our sysroots, it seems that we find a lib/gcc, but it
        # doesn't have a corresponding include/gcc directory.  Instead
        # force using '/usr/lib,include/gcc'
        if(opts.host_target == 'aarch64-linux-gnu'):
          cmd += ['--gcc-toolchain=%s/usr' % opts.sysroot]
      if(opts.xaie == 2):
        cmd += ['-DLIBXAIENGINEV2']
        cmd += ['-I%s/opt/xaienginev2/include' % opts.sysroot]
        cmd += ['-L%s/opt/xaienginev2/lib' % opts.sysroot]
      else:
        cmd += ['-I%s/opt/xaiengine/include' % opts.sysroot]
        cmd += ['-L%s/opt/xaiengine/lib' % opts.sysroot]
      cmd += ['-I%s' % tmpdirname]
      if(opts.xaie == 1):
        cmd += ['-fuse-ld=lld','-lm','-rdynamic','-lxaiengine','-lmetal','-lopen_amp','-ldl']
      else:
        cmd += ['-fuse-ld=lld','-lm','-rdynamic','-lxaiengine','-ldl']
    

      if(len(opts.arm_args) > 0):
        do_call(task, cmd + opts.arm_args)

      progress_bar.update(progress_bar.task_completed,advance=1)
      progress_bar.update(task,advance=0,visible=False)

    nthreads = int(opts.nthreads)
    nworkers = nthreads
    if(nthreads == 0):
      nworkers = os.cpu_count()

    with Progress(
      *Progress.get_default_columns(),
      TimeElapsedColumn(),
      MofNCompleteColumn(),
      TextColumn("{task.fields[command]}")) as progress:
      progress.task_completed = progress.add_task("[green] AIE Compilation:", total=len(cores)+1, command="")

      global progress_bar
      progress_bar = progress
      if (nthreads == 0 or nthreads > 1):
        if(nthreads == 0):
          nthreads = None
        with ThreadPool(nthreads) as thdpool:
            # prefer to dispatch and process_arm_cgen() first, it typically takes longer than process_core()
            thdpool.apply_async(process_arm_cgen)
            thdpool.map(process_core, (cores))
            thdpool.close()
            thdpool.join()
      else:
        for core in cores:
          process_core(core)
        process_arm_cgen()


def main(builtin_params={}):
    thispath = os.path.dirname(os.path.realpath(__file__))

    # Assume that aie-opt, etc. binaries are relative to this script.
    aie_path = os.path.join(thispath, '..')
    peano_path = os.path.join(thispath, '..', '..', 'peano', 'bin')

    if('VITIS' not in os.environ):
      # Try to find vitis in the path
      vpp_path = shutil.which("v++")
      if(vpp_path):
        vitis_bin_path = os.path.dirname(os.path.realpath(vpp_path))
        vitis_path = os.path.dirname(vitis_bin_path)
        os.environ['VITIS'] = vitis_path
        print("Found Vitis at " + vitis_path)
        os.environ['PATH'] = os.pathsep.join([os.environ['PATH'], vitis_bin_path])
 
    if('VITIS' in os.environ):
      vitis_path = os.environ['VITIS']
      vitis_bin_path = os.path.join(vitis_path, "bin")
      # Find the aietools directory, needed by xchesscc_wrapper
      
      aietools_path = os.path.join(vitis_path, "aietools")
      if(not os.path.exists(aietools_path)):
        aietools_path = os.path.join(vitis_path, "cardano")
      os.environ['AIETOOLS'] = aietools_path

      aietools_bin_path = os.path.join(aietools_path, "bin")
      os.environ['PATH'] = os.pathsep.join([
        os.environ['PATH'],
        aietools_bin_path,
        vitis_bin_path])

    # This path should be generated from cmake
    os.environ['PATH'] = os.pathsep.join([aie_path, os.environ['PATH']])
    os.environ['PATH'] = os.pathsep.join([peano_path, os.environ['PATH']])
    
    global opts
    opts = aiecc.cl_arguments.parse_args()
    is_windows = platform.system() == 'Windows'

    if(opts.verbose):
        sys.stderr.write('\ncompiling %s\n' % opts.filename)

    if(opts.tmpdir):
      tmpdirname = opts.tmpdir
      try:
        os.mkdir(tmpdirname)
      except FileExistsError:
        pass
      if(opts.verbose):
        print('created temporary directory', tmpdirname)

      run_flow(opts, tmpdirname)
    else:
      with tempfile.TemporaryDirectory() as tmpdirname:
        run_flow(opts, tmpdirname)

    if(opts.profiling):
      global runtimes
      sortedruntimes = sorted(runtimes.items(), key=lambda item: item[1], reverse=True)
      for i in range(50):
        if(i < len(sortedruntimes)):
          print("%.4f sec: %s" % (sortedruntimes[i][1], sortedruntimes[i][0]))
