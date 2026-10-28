# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %python %s
"""Kernel IR retention checks that need no Peano: flag parsing, IR symbol
renaming, the Chess rejection, the design cache identity and the make recipe.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import aie.utils.compile.jit._hash as compile_hash
import aie.utils.compile.utils as compile_utils


class KernelBitcodeFlagsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.source = self.work / "kernel.cc"
        self.source.write_text('extern "C" void kernel() {}\n')
        self.output = self.work / "kernel.o"

    def compile(self, **kwargs):
        compile_utils.compile_cxx_core_function(
            str(self.source), "aie2p", str(self.output), **kwargs
        )

    def test_boolean_flag_spellings(self):
        for flag in ("--check-lut-banks", "-check-lut-banks"):
            for suffix in ("", "=", "=true", "=True", "=TRUE", "=1"):
                with self.subTest(option=flag + suffix):
                    self.assertTrue(
                        compile_utils._check_lut_banks_enabled([flag + suffix])
                    )
            for suffix in ("=false", "=False", "=FALSE", "=0"):
                with self.subTest(option=flag + suffix):
                    self.assertFalse(
                        compile_utils._check_lut_banks_enabled([flag + suffix])
                    )
        for flags in (
            [],
            ["--unrelated"],
            ["--check-lut-banks-other"],
            ["--", "--check-lut-banks"],
            ["--check-lut-banks", "--check-lut-banks=false"],
        ):
            self.assertFalse(compile_utils._check_lut_banks_enabled(flags))

    def test_ir_symbol_rename_respects_llvm_tokens(self):
        ir = r"""
$helper = comdat any
@table = global [1 x i32] zeroinitializer
@text = private constant [8 x i8] c"@helper\00"
@alias = alias void (), ptr @helper
define void @helper() comdat($helper) {
$helper:
  %local$helper = add i32 0, 1
  %$helper = add i32 %local$helper, 1
  call void @"quoted\2Dhelper"()
  call void @"\01asm_helper"()
  call void @external()
  ret void
}
; @helper and $helper are comments, not references.
!foo$helper = !{!0}
!0 = !{!"@helper", !"linkageName", !"helper"}
"""
        renamed = compile_utils._rename_ir_symbols(
            ir, ["helper", "table", "alias", "quoted-helper", "asm_helper"], "op0_"
        )
        self.assertIn('$"op0_helper" = comdat any', renamed)
        self.assertIn('@"op0_table" = global', renamed)
        self.assertIn('@"op0_alias" = alias void (), ptr @"op0_helper"', renamed)
        self.assertIn('define void @"op0_helper"() comdat($"op0_helper")', renamed)
        self.assertIn('call void @"op0_quoted-helper"()', renamed)
        self.assertIn(r'call void @"\01op0_asm_helper"()', renamed)
        for unchanged in (
            r'c"@helper\00"',
            "call void @external()",
            "$helper:",
            "%local$helper = add i32 0, 1",
            "%$helper = add i32 %local$helper, 1",
            "; @helper and $helper are comments, not references.",
            "!foo$helper = !{!0}",
            '!0 = !{!"@helper", !"linkageName", !"helper"}',
        ):
            self.assertIn(unchanged, renamed)

    def test_chess_rejects_bitcode_before_invoking_compiler(self):
        with self.assertRaisesRegex(ValueError, "requires the Peano toolchain"):
            self.compile(embed_bitcode=True, use_chess=True)
        self.assertFalse(self.output.exists())


class BitcodeCacheIdentityTest(unittest.TestCase):
    def test_check_flag_changes_design_cache_identity(self):
        def generator():
            pass

        hashes = {
            compile_hash._compute_hash(generator, {}, [], [], flags, [])
            for flags in (
                [],
                ["--check-lut-banks"],
                ["--check-lut-banks=true"],
                ["--check-lut-banks=false"],
            )
        }
        self.assertEqual(len(hashes), 4)


@unittest.skipUnless(shutil.which("make") and os.name != "nt", "requires GNU make")
class MagikaBitcodeTest(unittest.TestCase):
    def make(self, *options):
        repo = Path(__file__).resolve().parents[2]
        return subprocess.run(
            [
                "make",
                "--no-print-directory",
                "-n",
                "-B",
                "-C",
                str(repo / "programming_examples/ml/magika"),
                f"MLIR_AIE_DIR={repo}",
                f"AIETOOLS_DIR={repo}",
                f"PEANO_INSTALL_DIR={repo}",
                "AIECC_FLAGS=--verbose",
                "AIE_OBJCOPY=llvm-objcopy",
                *options,
                "build/final.xclbin",
                "build/final_trace.xclbin",
            ],
            capture_output=True,
            text=True,
        )

    def test_group_variants_retain_matching_ir_on_both_devices(self):
        for device in ("npu", "npu2"):
            with self.subTest(device=device):
                result = self.make(f"devicename={device}", "AIE_CHECK_LUT_BANKS=1")
                self.assertEqual(result.returncode, 0, result.stderr)
                for group in ("A", "B"):
                    commands = next(
                        line
                        for line in result.stdout.splitlines()
                        if f"-DGROUP{group}" in line
                    )
                    obj, ir, attach = commands.split(" && ")[1:]
                    self.assertIn(f"-DGROUP{group}", obj)
                    self.assertIn(f"-DGROUP{group}", ir)
                    self.assertIn("-emit-llvm", ir)
                    self.assertIn("--add-section=.llvmbc=", attach)
                links = [
                    line for line in result.stdout.splitlines() if "&& aiecc " in line
                ]
                self.assertEqual(len(links), 2)
                for line in links:
                    self.assertIn("--verbose --check-lut-banks", line)

    def test_default_make_does_not_emit_bitcode(self):
        result = self.make("AIE_CHECK_LUT_BANKS=0", "AIE_OBJCOPY=")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("-emit-llvm", result.stdout)
        self.assertNotIn("--check-lut-banks", result.stdout)

    def test_missing_objcopy_has_actionable_error(self):
        result = self.make("AIE_CHECK_LUT_BANKS=1", "AIE_OBJCOPY=")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires llvm-objcopy; set AIE_OBJCOPY", result.stderr)

    def test_explicit_check_flag_is_not_duplicated(self):
        for flag in ("--check-lut-banks", "--check-lut-banks=true"):
            with self.subTest(flag=flag):
                result = self.make(
                    "AIE_CHECK_LUT_BANKS=1", f"AIECC_FLAGS=--verbose {flag}"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                links = [
                    line for line in result.stdout.splitlines() if "&& aiecc " in line
                ]
                for line in links:
                    self.assertEqual(line.count("--check-lut-banks"), 1)

    def test_failed_ir_retention_removes_make_target(self):
        result = self.make("AIE_CHECK_LUT_BANKS=1", "devicename=npu")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = next(
            line for line in result.stdout.splitlines() if "-DGROUPA" in line
        )
        retain_ir = command.split(" && ", 2)[2]
        compiler = str(Path(__file__).resolve().parents[2] / "bin/clang")
        for failure in ("compile", "attach"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as work:
                obj = Path(work) / "group0a.o"
                obj.write_bytes(b"object without bitcode")
                bitcode = Path(f"{obj}.bc")
                bitcode.write_bytes(b"unattached IR")
                script = retain_ir.replace(
                    compiler, "false" if failure == "compile" else "true"
                ).replace("llvm-objcopy", "false")
                failed = subprocess.run(
                    ["sh", "-c", script], cwd=work, capture_output=True
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse(obj.exists())
                self.assertFalse(bitcode.exists())


if __name__ == "__main__":
    unittest.main()
