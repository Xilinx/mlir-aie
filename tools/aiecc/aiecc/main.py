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
import tempfile
import shutil
import timeit
import asyncio

import aiecc.cl_arguments

from rich.progress import *


class flow_runner:
  def __init__(self, opts, tmpdirname):
      self.opts = opts
      self.tmpdirname = tmpdirname
      self.runtimes = dict()
      self.progress_bar = None
      self.maxtasks = 5
      self.stopall = False

  async def do_call(self, task, command):
      if(self.stopall):
        return

      commandstr = " ".join(command)
      if(task):
        self.progress_bar.update(task, advance=0, command=commandstr[0:30])
      start = time.time()
      if(self.opts.verbose):
          print(commandstr)
      if(self.opts.execute):
        proc = await asyncio.create_subprocess_exec(*command)
        await proc.wait()
        ret = proc.returncode
      else:
        ret = 0
      end = time.time()
      if(self.opts.verbose):
          print("Done %.3f-%.3f=%.3f %s" % (start, end, end-start, commandstr))
      self.runtimes[commandstr] = end-start
      if(task):
        self.progress_bar.update(task, advance=1, command="")
        self.maxtasks = max(self.progress_bar._tasks[task].completed, self.maxtasks)
        self.progress_bar._tasks[task].total = self.maxtasks

      if(ret != 0):
          self.progress_bar._tasks[task].description = "[red] Error"
          print("Error encountered while running: " + commandstr)
          sys.exit(1)

  def do_run(self, command):
      if(self.opts.verbose):
          print(" ".join(command))
      ret = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
      return ret

  def corefile(self, dirname, core, ext):
      (corecol, corerow, _) = core
      return os.path.join(dirname, 'core_%d_%d.%s' % (corecol, corerow, ext))

  def tmpcorefile(self, core, ext):
      return self.corefile(self.tmpdirname, core, ext)

  # Extract included files from the given Chess linker script.
  # We rely on gnu linker scripts to stuff object files into a compile.  However, the Chess compiler doesn't 
  # do this, so we have to explicitly specify included files on the link line.
  def extract_input_files(self, file_core_bcf):
      t = self.do_run(['awk', '/_include _file/ {print($3)}', file_core_bcf])
      return ' '.join(t.stdout.split())

  async def process_core(self, core):
    async with self.limit:
      if(self.stopall):
        return

      thispath = os.path.dirname(os.path.realpath(__file__))
      me_basic_o = os.path.join(thispath, '..','..','runtime_lib', 'me_basic.o')
      libc = os.path.join(thispath, '..','..','runtime_lib', 'libc.a')
      libm = os.path.join(thispath, '..','..','runtime_lib', 'libm.a')
      chess_intrinsic_wrapper_cpp = os.path.join(thispath, '..','..','runtime_lib', 'chess_intrinsic_wrapper.cpp')

      if(opts.progress):
        task = self.progress_bar.add_task("[yellow] Core (%d, %d)" % core[0:2], total=self.maxtasks, command="starting")
      else:
        task = None

      (corecol, corerow, elf_file) = core
      file_core = self.tmpcorefile(core, "mlir")
      await self.do_call(task, ['aie-opt', '--aie-localize-locks',
                          '--aie-standard-lowering=tilecol=%d tilerow=%d' % core[0:2],
                          self.file_with_addresses, '-o', file_core])

      file_opt_core = self.tmpcorefile(core, "opt.mlir")
      await self.do_call(task, ['aie-opt', '--aie-normalize-address-spaces',
                          '--canonicalize',
                          '--cse',
                          '--convert-vector-to-llvm',
                          '--convert-memref-to-llvm',
                          '--convert-func-to-llvm=use-bare-ptr-memref-call-conv',
                          '--convert-cf-to-llvm',
                          '--canonicalize', '--cse', file_core, '-o', file_opt_core])
      if(self.opts.xbridge):
        file_core_bcf = self.tmpcorefile(core, "bcf")
        await self.do_call(task, ['aie-translate', self.file_with_addresses, '--aie-generate-bcf', '--tilecol=%d' % corecol, '--tilerow=%d' % corerow, '-o', file_core_bcf])
      else:
        file_core_ldscript = self.tmpcorefile(core, "ld.script")
        await self.do_call(task, ['aie-translate', self.file_with_addresses, '--aie-generate-ldscript', '--tilecol=%d' % corecol, '--tilerow=%d' % corerow, '-o', file_core_ldscript])
      file_core_llvmir = self.tmpcorefile(core, "ll")
      await self.do_call(task, ['aie-translate', '--mlir-to-llvmir', '--opaque-pointers=0', file_opt_core, '-o', file_core_llvmir])
      file_core_elf = elf_file if elf_file else self.corefile(".", core, "elf")
      file_core_obj = self.tmpcorefile(core, "o")

      if(opts.compile and opts.xchesscc):
        file_core_llvmir_chesshack = self.tmpcorefile(core, "chesshack.ll")
        await self.do_call(task, ['cp', file_core_llvmir, file_core_llvmir_chesshack])
        await self.do_call(task, ['sed', '-i', 's/noundef//', file_core_llvmir_chesshack])
        await self.do_call(task, ['sed', '-i', 's/noalias_sidechannel[^,],//', file_core_llvmir_chesshack])
        file_core_llvmir_chesslinked = self.tmpcorefile(core, "chesslinked.ll")
        await self.do_call(task, ['llvm-link', file_core_llvmir_chesshack, self.chess_intrinsic_wrapper, '-S', '-o', file_core_llvmir_chesslinked])
        await self.do_call(task, ['sed', '-i', 's/noundef//', file_core_llvmir_chesslinked])
        # Formal function argument names not used in older LLVM
        await self.do_call(task, ['sed', '-i', '-E', '/define .*@/ s/%[0-9]*//g', file_core_llvmir_chesslinked])
        await self.do_call(task, ['sed', '-i', '-E', 's/mustprogress//g', file_core_llvmir_chesslinked])
        await self.do_call(task, ['sed', '-i', '-E', 's/poison/undef/g', file_core_llvmir_chesslinked])
        await self.do_call(task, ['sed', '-i', '-E', 's/nocallback//g', file_core_llvmir_chesslinked])
        if(self.opts.link and self.opts.xbridge):
          link_with_obj = self.extract_input_files(file_core_bcf)
          await self.do_call(task, ['xchesscc_wrapper', '-d', '-f', '+P', '4', file_core_llvmir_chesslinked, link_with_obj, '+l', file_core_bcf, '-o', file_core_elf])
        elif(self.opts.link):
          await self.do_call(task, ['xchesscc_wrapper', '-c', '-d', '-f', '+P', '4', file_core_llvmir_chesslinked, '-o', file_core_obj])
          await self.do_call(task, ['clang', '-O2', '--target=aie', file_core_obj, me_basic_o, libm,
                       '-Wl,-T,'+file_core_ldscript, '-o', file_core_elf])
      elif(opts.compile):
        file_core_llvmir_stripped = self.tmpcorefile(core, "stripped.ll")
        await self.do_call(task, ['opt', '--passes=default<O2>,strip', '-S', file_core_llvmir, '-o', file_core_llvmir_stripped])
        await self.do_call(task, ['llc', file_core_llvmir_stripped, '-O2', '--march=aie', '--function-sections', '--filetype=obj', '-o', file_core_obj])
        if(opts.link and opts.xbridge):
          link_with_obj = self.extract_input_files(file_core_bcf)
          await self.do_call(task, ['xchesscc_wrapper', '-d', '-f', file_core_obj, link_with_obj, '+l', file_core_bcf, '-o', file_core_elf])
        elif(opts.link):
          await self.do_call(task, ['clang', '-O2', '--target=aie', file_core_obj, me_basic_o, libm,
                            '-Wl,-T,'+file_core_ldscript, '-Wl,--gc-sections', '-o', file_core_elf])
        do_call(['aie-opt', '--aie-create-flows', '--aie-lower-broadcast-packet', '--aie-create-packet-flows', file_with_addresses, '-o', file_physical]);
      if(opts.airbin):
        file_airbin = os.path.join(tmpdirname, 'air.bin')
        do_call(['aie-translate', '--aie-generate-airbin', file_physical, '-o', file_airbin])
      else:
        file_inc_cpp = os.path.join(tmpdirname, 'aie_inc.cpp')
        if(opts.xaie == 2):
          do_call(['aie-translate', '--aie-generate-xaie', '--xaie-target=v2', file_physical, '-o', file_inc_cpp])
        else:
          do_call(['aie-translate', '--aie-generate-xaie', '--xaie-target=v1', file_physical, '-o', file_inc_cpp])

      self.progress_bar.update(self.progress_bar.task_completed,advance=1)
      if(task):
        self.progress_bar.update(task,advance=0,visible=False)

  async def process_arm_cgen(self):
    async with self.limit:
      if(self.stopall):
        return

      if(opts.progress):
        task = self.progress_bar.add_task("[yellow] ARM Core ", total=10, command="starting")
      else:
        task = None

      # Generate the included host interface
      file_physical = os.path.join(self.tmpdirname, 'input_physical.mlir')
      await self.do_call(task, ['aie-opt', '--aie-create-pathfinder-flows', '--aie-lower-broadcast-packet', '--aie-create-packet-flows', '--aie-lower-multicast', self.file_with_addresses, '-o', file_physical]);
      file_inc_cpp = os.path.join(self.tmpdirname, 'aie_inc.cpp')
      if(opts.xaie == 1):
          await self.do_call(task, ['aie-translate', '--aie-generate-xaie', '--xaie-target=v1', file_physical, '-o', file_inc_cpp])
      else:
          await self.do_call(task, ['aie-translate', '--aie-generate-xaie', '--xaie-target=v2', file_physical, '-o', file_inc_cpp])

      cmd = ['clang','-std=c++11']
      if(opts.host_target):
        cmd += ['--target=%s' % opts.host_target]
      if(self.opts.sysroot):
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
      cmd += ['-I%s' % self.tmpdirname]
      if(opts.xaie == 1):
        cmd += ['-fuse-ld=lld','-lm','-rdynamic','-lxaiengine','-lmetal','-lopen_amp','-ldl']
      else:
        cmd += ['-fuse-ld=lld','-lm','-rdynamic','-lxaiengine','-ldl']


      if(len(opts.arm_args) > 0):
        await self.do_call(task, cmd + opts.arm_args)

      self.progress_bar.update(self.progress_bar.task_completed,advance=1)
      if(task):
        self.progress_bar.update(task,advance=0,visible=False)

  async def run_flow(self):
      nworkers = int(opts.nthreads)
      if(nworkers == 0):
        nworkers = os.cpu_count()

      self.limit = asyncio.Semaphore(nworkers)

      self.file_with_addresses = os.path.join(self.tmpdirname, 'input_with_addresses.mlir')
      await self.do_call(None, ['aie-opt', '--lower-affine', '--aie-register-objectFifos', '--aie-objectFifo-stateful-transform', '--aie-lower-broadcast-packet', '--aie-create-packet-flows', '--aie-lower-multicast', '--aie-assign-buffer-addresses', '-convert-scf-to-cf', opts.filename, '-o', self.file_with_addresses])
      t = self.do_run(['aie-translate', '--aie-generate-corelist', self.file_with_addresses])
      cores = eval(t.stdout)

      if(opts.xchesscc == True):
        thispath = os.path.dirname(os.path.realpath(__file__))
        chess_intrinsic_wrapper_cpp = os.path.join(thispath, '..','..','runtime_lib', 'chess_intrinsic_wrapper.cpp')

        self.chess_intrinsic_wrapper = os.path.join(self.tmpdirname, 'chess_intrinsic_wrapper.ll')
        await self.do_call(None, ['xchesscc_wrapper', '-c', '-d', '-f', '+f', '+P', '4', chess_intrinsic_wrapper_cpp, '-o', self.chess_intrinsic_wrapper])
        await self.do_call(None, ['sed', '-i', 's/^target.*//', self.chess_intrinsic_wrapper])

        await self.do_call(None, ['sed', '-i', 's/noalias_sidechannel[^,]*,//', self.chess_intrinsic_wrapper])
        await self.do_call(None, ['sed', '-i', 's/nocallback[^,]*,//', self.chess_intrinsic_wrapper])

      with Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[command]}")) as progress:
        progress.task_completed = progress.add_task("[green] AIE Compilation:", total=len(cores)+1, command="%d Workers" % nworkers)

        self.progress_bar = progress
        processes = [self.process_arm_cgen()]
        for core in cores:
          processes.append(self.process_core(core))
        await asyncio.gather(*processes)

  def dumpprofile(self):
      sortedruntimes = sorted(self.runtimes.items(), key=lambda item: item[1], reverse=True)
      for i in range(50):
        if(i < len(sortedruntimes)):
          print("%.4f sec: %s" % (sortedruntimes[i][1], sortedruntimes[i][0]))


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

      runner = flow_runner(opts, tmpdirname)
      asyncio.run(runner.run_flow())
    else:
      with tempfile.TemporaryDirectory() as tmpdirname:
        runner = flow_runner(opts, tmpdirname)
        asyncio.run(runner.run_flow())

    if(opts.profiling):
      runner.dumpprofile()
