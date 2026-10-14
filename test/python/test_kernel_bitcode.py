# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %python %s
# REQUIRES: peano
"""Kernel IR retention, read from what Peano and llvm-objcopy leave on disk.

The host-only cases (flag parsing, IR renaming, cache identity and the make
recipe) are in test_kernel_bitcode_host.py; no NPU needed.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

import aie.utils.compile.utils as compile_utils
import aie.utils.config as config
from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.utils import set_current_device

TILE = 64
tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]


def fill_design(kernel: ExternalFunction):
    """The module of a design whose one core runs `kernel` on its output."""
    of_out = ObjectFifo(tile_ty, name="out")

    def core(of_out, fn):
        for _ in range_(1):
            elem = of_out.acquire(1)
            fn(elem)
            of_out.release(1)

    worker = Worker(core, [of_out.prod(), kernel])

    def sequence(out, out_h):
        out_h.drain(out, wait=True)

    rt = Runtime(sequence, [tile_ty, of_out.cons()])
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program()


class KernelBitcodeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.source = self.work / "kernel.cc"
        self.source.write_text(
            'int counter;\nextern "C" void kernel() { counter++; }\n'
        )
        self.output = self.work / "kernel.o"
        ExternalFunction._instances.clear()
        self.addCleanup(ExternalFunction._instances.clear)
        self.addCleanup(self.restore_objcopy, os.environ.get("AIE_OBJCOPY_PATH"))

    def restore_objcopy(self, value):
        if value is None:
            os.environ.pop("AIE_OBJCOPY_PATH", None)
        else:
            os.environ["AIE_OBJCOPY_PATH"] = value

    def compile(self, **kwargs):
        compile_utils.compile_cxx_core_function(
            str(self.source), "aie2p", str(self.output), **kwargs
        )

    def section_headers(self, path):
        return subprocess.run(
            [config.readobj_path(), "--section-headers", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def bitcode(self, path):
        dump = self.work / "dumped.bc"
        subprocess.run(
            [
                config.objcopy_path(),
                f"--dump-section=.llvmbc={dump}",
                str(path),
                os.devnull,
            ],
            check=True,
            capture_output=True,
        )
        return dump.read_bytes()

    def identity(self, path):
        stat = Path(path).stat()
        return stat.st_ino, stat.st_mtime_ns

    def test_default_compile_does_not_emit_bitcode(self):
        self.compile()
        sections = self.section_headers(self.output)
        for name in (".stack_sizes", ".text.kernel", ".bss.counter"):
            self.assertIn(name, sections)
        self.assertNotIn(".llvmbc", sections)
        self.assertFalse(compile_utils._object_has_bitcode(self.output))
        self.assertFalse(Path(f"{self.output}.bc").exists())

    def test_direct_module_compile_retains_bitcode_when_requested(self):
        set_current_device(NPU2Col1())
        self.source.write_text(
            'extern "C" void fill_seven(int *out) {\n'
            f"  for (int i = 0; i < {TILE}; i++) out[i] = 7;\n"
            "}\n"
        )
        func = ExternalFunction(
            "fill_seven", source_file=str(self.source), arg_types=[tile_ty]
        )
        text = str(fill_design(func))
        unused = ExternalFunction(
            "inline_kernel", source_string='extern "C" void k() {}'
        )
        for index, (options, enabled) in enumerate(
            (
                (None, False),
                ([], False),
                (["--check-lut-banks"], True),
                (["-check-lut-banks=true"], True),
                (["--check-lut-banks=false"], False),
                (["--check-lut-banks", "--check-lut-banks=0"], False),
            )
        ):
            with self.subTest(options=options):
                work = self.work / f"build{index}"
                work.mkdir()
                compile_utils.compile_mlir_module(
                    text,
                    options=options,
                    device=NPU2Col1(),
                    work_dir=work,
                    full_elf_path=work / "design.elf",
                )
                self.assertEqual(
                    compile_utils._object_has_bitcode(work / func.object_file_name),
                    enabled,
                )
                self.assertFalse((work / unused.object_file_name).exists())

    def test_bitcode_compile_preserves_defines_and_include_paths(self):
        include = self.work / "custom" / "include"
        include.mkdir(parents=True)
        (include / "group.h").write_text("#define GROUP_SYMBOL kernel_group_a\n")
        self.source.write_text(
            '#include "group.h"\n'
            "#ifdef GROUPA\n"
            'extern "C" void GROUP_SYMBOL() {}\n'
            "#endif\n"
        )
        self.compile(
            embed_bitcode=True,
            compile_args=["-DGROUPA", "-O1"],
            include_dirs=["custom/include"],
            cwd=str(self.work),
        )
        self.assertIn(".stack_sizes", self.section_headers(self.output))
        self.assertIn(b"kernel_group_a", self.bitcode(self.output))

    def test_inline_merge_kernel_check_flag_does_not_attach_bitcode(self):
        ir = self.work / "kernel.ll"
        compile_utils.compile_cxx_core_function(
            str(self.source),
            "aie2p",
            str(ir),
            inline=True,
            symbol_name="kernel",
            embed_bitcode=compile_utils._check_lut_banks_enabled(
                ["--check-lut-banks=true"]
            ),
        )
        self.assertIn("define linkonce_odr", ir.read_text())
        self.assertFalse(Path(f"{ir}.bc").exists())

    def test_failed_ir_retention_does_not_leave_cached_object(self):
        # The object's own name fits a file name; its IR sibling's does not.
        long_output = self.work / ("k" * 251 + ".o")
        with self.subTest(failure="emit bitcode"):
            long_output.write_bytes(b"object without bitcode")
            with self.assertRaisesRegex(RuntimeError, "emit bitcode"):
                compile_utils.compile_cxx_core_function(
                    str(self.source), "aie2p", str(long_output), embed_bitcode=True
                )
            self.assertFalse(long_output.exists())

        with self.subTest(failure="attach bitcode"):
            self.output.write_bytes(b"object without bitcode")
            bitcode = Path(f"{self.output}.bc")
            bitcode.write_bytes(b"unattached IR")
            os.environ["AIE_OBJCOPY_PATH"] = shutil.which("false")
            with self.assertRaisesRegex(RuntimeError, "attach bitcode"):
                self.compile(embed_bitcode=True)
            self.assertFalse(self.output.exists())
            self.assertFalse(bitcode.exists())

    def test_failed_attachment_cleans_relative_outputs_in_compiler_cwd(self):
        self.output.write_bytes(b"object without bitcode")
        bitcode = Path(f"{self.output}.bc")
        bitcode.write_bytes(b"unattached IR")
        os.environ["AIE_OBJCOPY_PATH"] = str(self.work / "no-objcopy")
        with self.assertRaisesRegex(RuntimeError, "no such file exists"):
            compile_utils.compile_cxx_core_function(
                str(self.source),
                "aie2p",
                self.output.name,
                cwd=str(self.work),
                embed_bitcode=True,
            )
        self.assertFalse(self.output.exists())
        self.assertFalse(bitcode.exists())

    def test_checked_compile_does_not_reuse_unchecked_object(self):
        func = ExternalFunction("kernel", source_file=str(self.source))
        output = self.work / func.object_file_name

        compile_utils.compile_external_kernels([func], self.work, "aie2p")
        self.assertFalse(compile_utils._object_has_bitcode(output))
        compile_utils.compile_external_kernels(
            [func], self.work, "aie2p", embed_bitcode=True
        )
        self.assertTrue(compile_utils._object_has_bitcode(output))
        checked = self.identity(output)
        compile_utils.compile_external_kernels(
            [func], self.work, "aie2p", embed_bitcode=True
        )
        self.assertEqual(self.identity(output), checked)
        output.unlink()
        compile_utils.compile_external_kernels(
            [func], self.work, "aie2p", embed_bitcode=True
        )
        self.assertTrue(compile_utils._object_has_bitcode(output))

    def test_checked_compile_does_not_trust_disk_cache(self):
        func = ExternalFunction("kernel", source_file=str(self.source))
        output = self.work / func.object_file_name
        output.write_bytes(b"old object")
        compile_utils.compile_external_kernel(
            func, self.work, "aie2p", embed_bitcode=True
        )
        self.assertTrue(compile_utils._object_has_bitcode(output))

    def _shared_object_functions(self):
        self.source.write_text(
            "#ifdef GROUPA\n"
            'extern "C" void kernel() {}\n'
            'extern "C" void helper() {}\n'
            "#endif\n"
        )
        funcs = [
            ExternalFunction(
                name,
                source_file=str(self.source),
                compile_flags=["-DGROUPA"],
                object_file_name="kernel.o",
            )
            for name in ("kernel", "helper")
        ]
        self.assertIs(funcs[0].object_file, funcs[1].object_file)
        return funcs

    def test_shared_owner_bitcode_upgrade_is_per_object_and_directory(self):
        funcs = self._shared_object_functions()
        other_work = self.work / "other"
        other_work.mkdir()
        other_output = other_work / "kernel.o"

        compile_utils.compile_external_kernels(funcs, self.work, "aie2p")
        self.assertEqual(
            set(compile_utils._defined_symbols(self.output)), {"kernel", "helper"}
        )
        self.assertFalse(compile_utils._object_has_bitcode(self.output))
        compile_utils.compile_external_kernels(
            funcs, self.work, "aie2p", embed_bitcode=True
        )
        self.assertTrue(compile_utils._object_has_bitcode(self.output))
        upgraded = self.identity(self.output)
        compile_utils.compile_external_kernels(
            funcs[::-1], self.work, "aie2p", embed_bitcode=True
        )
        self.assertEqual(self.identity(self.output), upgraded)

        compile_utils.compile_external_kernels(funcs, other_work, "aie2p")
        self.assertFalse(compile_utils._object_has_bitcode(other_output))
        compile_utils.compile_external_kernels(
            funcs, self.work, "aie2p", embed_bitcode=True
        )
        self.assertEqual(self.identity(self.output), upgraded)
        compile_utils.compile_external_kernels(
            funcs, other_work, "aie2p", embed_bitcode=True
        )
        self.assertTrue(compile_utils._object_has_bitcode(other_output))
        self.assertEqual(
            funcs[0].object_file._compiled_dirs,
            {os.path.realpath(self.work), os.path.realpath(other_work)},
        )

        # Neither an old .bc sidecar nor ownership proves the object has IR.
        Path(f"{self.output}.bc").write_bytes(b"stale IR")
        self.output.write_bytes(b"replacement object without IR")
        compile_utils.compile_external_kernel(
            funcs[1], self.work, "aie2p", embed_bitcode=True
        )
        self.assertTrue(compile_utils._object_has_bitcode(self.output))
        self.output.unlink()
        compile_utils.compile_external_kernel(
            funcs[0], self.work, "aie2p", embed_bitcode=True
        )
        self.assertTrue(compile_utils._object_has_bitcode(self.output))

    def test_failed_shared_owner_upgrade_invalidates_cached_object(self):
        funcs = self._shared_object_functions()
        owner = funcs[0].object_file
        compile_utils.compile_external_kernel(funcs[0], self.work, "aie2p")
        self.assertIn(os.path.realpath(self.work), owner._compiled_dirs)

        good_source = self.source.read_text()
        self.source.write_text('#error "the upgrade does not compile"\n')
        Path(f"{self.output}.d").write_bytes(b"stale dependencies")
        Path(f"{self.output}.bc").write_bytes(b"stale IR")
        with self.assertRaisesRegex(RuntimeError, "the upgrade does not compile"):
            compile_utils.compile_external_kernel(
                funcs[0], self.work, "aie2p", embed_bitcode=True
            )
        self.assertNotIn(os.path.realpath(self.work), owner._compiled_dirs)
        for path in (self.output, Path(f"{self.output}.d"), Path(f"{self.output}.bc")):
            self.assertFalse(path.exists())
        self.assertFalse(compile_utils._compiled_into(funcs[1], self.work))

        self.source.write_text(good_source)
        compile_utils.compile_external_kernel(funcs[1], self.work, "aie2p")
        self.assertTrue(self.output.exists())
        self.assertTrue(compile_utils._compiled_into(funcs[0], self.work))


class ObjectBitcodeTest(unittest.TestCase):
    def test_bitcode_inspection_does_not_rewrite_cached_object(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "kernel.cc"
            source.write_text('extern "C" void kernel() {}\n')
            for embed_bitcode in (False, True):
                with self.subTest(embed_bitcode=embed_bitcode):
                    output = work / f"kernel_{embed_bitcode}.o"
                    compile_utils.compile_cxx_core_function(
                        str(source),
                        "aie2p",
                        str(output),
                        embed_bitcode=embed_bitcode,
                    )
                    before = output.read_bytes(), output.stat().st_mtime_ns
                    listing = sorted(work.iterdir())
                    self.assertEqual(
                        compile_utils._object_has_bitcode(output), embed_bitcode
                    )
                    self.assertEqual(
                        (output.read_bytes(), output.stat().st_mtime_ns), before
                    )
                    self.assertEqual(sorted(work.iterdir()), listing)


if __name__ == "__main__":
    unittest.main()
