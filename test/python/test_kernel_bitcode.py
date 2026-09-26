# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %python %s
"""Kernel IR retention and make recipe regressions; no AIE tools or NPU needed."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import aie.utils.compile.jit._hash as compile_hash
import aie.utils.compile.utils as compile_utils


class KernelBitcodeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.source = self.work / "kernel.cc"
        self.source.write_text('extern "C" void kernel() {}\n')
        self.output = self.work / "kernel.o"
        for name, value in (
            ("peano_cxx_path", "clang++"),
            ("cxx_header_path", "include"),
            ("objcopy_path", "llvm-objcopy"),
        ):
            mock = patch.object(compile_utils.config, name, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch.object(compile_utils, "_object_has_bitcode", return_value=False)
        self.has_bitcode = mock.start()
        self.addCleanup(mock.stop)

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

    def test_default_compile_does_not_emit_bitcode(self):
        with patch.object(
            compile_utils.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as run:
            self.compile()
        self.assertEqual(run.call_count, 1)
        cmd = run.call_args.args[0]
        self.assertIn("-fstack-size-section", cmd)
        self.assertIn("-ffunction-sections", cmd)
        self.assertIn("-fdata-sections", cmd)
        self.assertNotIn("-emit-llvm", cmd)

    def test_direct_module_compile_retains_bitcode_when_requested(self):
        from aie.iron.kernel import ExternalFunction

        func = SimpleNamespace(_source_file=str(self.source), name="kernel")
        for options, enabled in (
            (None, False),
            ([], False),
            (["--check-lut-banks"], True),
            (["-check-lut-banks=true"], True),
            (["--check-lut-banks=false"], False),
            (["--check-lut-banks", "--check-lut-banks=0"], False),
        ):
            with self.subTest(options=options), patch.object(
                ExternalFunction, "_instances", [func, SimpleNamespace()]
            ), patch.object(
                compile_utils.config, "peano_install_dir", return_value="peano"
            ), patch.object(
                compile_utils, "resolve_target_arch", return_value="aie2p"
            ), patch.object(
                compile_utils, "compile_external_kernels"
            ) as compile_kernels, patch.object(
                compile_utils, "_run_aiecc"
            ) as run:
                compile_utils.compile_mlir_module(
                    "module { func.func private @kernel() }",
                    options=options,
                    device="npu2",
                    work_dir=self.work,
                )
                compile_kernels.assert_called_once_with(
                    [func], str(self.work), "aie2p", embed_bitcode=enabled
                )
                run.assert_called_once()

    def test_bitcode_compile_preserves_defines_and_include_paths(self):
        with patch.object(
            compile_utils.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as run:
            self.compile(
                embed_bitcode=True,
                compile_args=["-DGROUPA", "-O1"],
                include_dirs=["custom/include"],
                cwd=str(self.work),
            )
        self.assertEqual(run.call_count, 3)
        obj, ir, attach = [call.args[0] for call in run.call_args_list]
        self.assertIn("-fstack-size-section", obj)
        self.assertNotIn("-fstack-size-section", ir)
        self.assertIn("-emit-llvm", ir)
        self.assertIn("-DGROUPA", ir)
        self.assertIn("-O1", ir)
        self.assertIn("custom/include", ir)
        self.assertEqual(ir[ir.index("-o") + 1], f"{self.output}.bc")
        self.assertEqual(
            attach,
            [
                "llvm-objcopy",
                f"--add-section=.llvmbc={self.output}.bc",
                str(self.output),
            ],
        )
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["cwd"], str(self.work))

    def test_chess_rejects_bitcode_before_invoking_compiler(self):
        with patch.object(compile_utils.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "requires the Peano toolchain"):
                self.compile(embed_bitcode=True, use_chess=True)
        run.assert_not_called()

    def test_inline_merge_kernel_check_flag_does_not_attach_bitcode(self):
        with patch.object(
            compile_utils.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ) as run, patch.object(compile_utils, "_make_ir_inlinable"):
            compile_utils.compile_cxx_core_function(
                str(self.source),
                "aie2p",
                str(self.work / "kernel.ll"),
                inline=True,
                symbol_name="kernel",
                embed_bitcode=compile_utils._check_lut_banks_enabled(
                    ["--check-lut-banks=true"]
                ),
            )
        self.assertEqual(run.call_count, 1)
        self.assertIn("-emit-llvm", run.call_args.args[0])

    def test_failed_ir_retention_does_not_leave_cached_object(self):
        for failure, diagnostic in ((1, "emit bitcode"), (2, "attach bitcode")):
            with self.subTest(failure=failure):
                self.output.write_bytes(b"object without bitcode")
                bitcode = Path(f"{self.output}.bc")
                bitcode.write_bytes(b"unattached IR")
                results = [
                    subprocess.CompletedProcess([], 0, b"", b"") for _ in range(failure)
                ]
                results.append(subprocess.CompletedProcess([], 1, b"", b"failed"))
                with patch.object(compile_utils.subprocess, "run", side_effect=results):
                    with self.assertRaisesRegex(RuntimeError, diagnostic):
                        self.compile(embed_bitcode=True)
                self.assertFalse(self.output.exists())
                self.assertFalse(bitcode.exists())

    def test_failed_attachment_cleans_relative_outputs_in_compiler_cwd(self):
        self.output.write_bytes(b"object without bitcode")
        bitcode = Path(f"{self.output}.bc")
        bitcode.write_bytes(b"unattached IR")
        with patch.object(
            compile_utils.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ), patch.object(
            compile_utils.config,
            "objcopy_path",
            side_effect=RuntimeError("objcopy unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "objcopy unavailable"):
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
        func = SimpleNamespace(
            _name="kernel",
            _original_name="kernel",
            _source_file=str(self.source),
            _source_string=None,
            _include_dirs=[],
            _compile_flags=[],
            _compiled=False,
            object_file_name="kernel.o",
        )

        def fake_compile(**kwargs):
            self.output.write_bytes(b"object")

        with patch.object(
            compile_utils, "compile_cxx_core_function", side_effect=fake_compile
        ) as compile_kernel:
            compile_utils.compile_external_kernels([func], self.work, "aie2p")
            compile_utils.compile_external_kernels(
                [func], self.work, "aie2p", embed_bitcode=True
            )
            self.assertEqual(compile_kernel.call_count, 2)
            self.assertTrue(compile_kernel.call_args.kwargs["embed_bitcode"])
            compile_utils.compile_external_kernels(
                [func], self.work, "aie2p", embed_bitcode=True
            )
            self.assertEqual(compile_kernel.call_count, 2)
            self.output.unlink()
            compile_utils.compile_external_kernels(
                [func], self.work, "aie2p", embed_bitcode=True
            )
            self.assertEqual(compile_kernel.call_count, 3)

    def test_checked_compile_does_not_trust_disk_cache(self):
        func = SimpleNamespace(
            _name="kernel",
            _original_name="kernel",
            _source_file=str(self.source),
            _source_string=None,
            _include_dirs=[],
            _compile_flags=[],
            object_file_name="kernel.o",
        )
        self.output.write_bytes(b"old object")
        with patch.object(compile_utils, "compile_cxx_core_function") as compile_kernel:
            compile_utils.compile_external_kernel(
                func, self.work, "aie2p", embed_bitcode=True
            )
        compile_kernel.assert_called_once()
        self.assertTrue(compile_kernel.call_args.kwargs["embed_bitcode"])

    def _shared_object_functions(self):
        from aie.iron.kernel import ExternalFunction

        registry = patch.object(ExternalFunction, "_instances", set())
        registry.start()
        self.addCleanup(registry.stop)
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
        self.has_bitcode.side_effect = (
            lambda path: Path(path).read_bytes() == b"object with IR"
        )

        def fake_compile(output_path, embed_bitcode, **kwargs):
            self.assertEqual(kwargs["compile_args"], ["-DGROUPA"])
            Path(output_path).write_bytes(
                b"object with IR" if embed_bitcode else b"object"
            )

        with patch.object(
            compile_utils, "compile_cxx_core_function", side_effect=fake_compile
        ) as compile_kernel:
            compile_utils.compile_external_kernels(funcs, self.work, "aie2p")
            self.assertEqual(compile_kernel.call_count, 1)
            compile_utils.compile_external_kernels(
                funcs, self.work, "aie2p", embed_bitcode=True
            )
            self.assertEqual(compile_kernel.call_count, 2)
            compile_utils.compile_external_kernels(
                funcs[::-1], self.work, "aie2p", embed_bitcode=True
            )
            self.assertEqual(compile_kernel.call_count, 2)

            compile_utils.compile_external_kernels(funcs, other_work, "aie2p")
            self.assertEqual(compile_kernel.call_count, 3)
            compile_utils.compile_external_kernels(
                funcs, self.work, "aie2p", embed_bitcode=True
            )
            self.assertEqual(compile_kernel.call_count, 3)
            compile_utils.compile_external_kernels(
                funcs, other_work, "aie2p", embed_bitcode=True
            )
            self.assertEqual(compile_kernel.call_count, 4)
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
            self.assertEqual(compile_kernel.call_count, 5)
            self.output.unlink()
            compile_utils.compile_external_kernel(
                funcs[0], self.work, "aie2p", embed_bitcode=True
            )
            self.assertEqual(compile_kernel.call_count, 6)

    def test_failed_shared_owner_upgrade_invalidates_cached_object(self):
        funcs = self._shared_object_functions()
        owner = funcs[0].object_file
        owner._compiled_dirs.add(os.path.realpath(self.work))
        self.output.write_bytes(b"cached object without IR")

        def fail_compile(output_path, **kwargs):
            Path(output_path).write_bytes(b"partial object")
            Path(f"{output_path}.d").write_bytes(b"partial dependencies")
            Path(f"{output_path}.bc").write_bytes(b"partial IR")
            raise RuntimeError("failed upgrade")

        with patch.object(
            compile_utils, "compile_cxx_core_function", side_effect=fail_compile
        ):
            with self.assertRaisesRegex(RuntimeError, "failed upgrade"):
                compile_utils.compile_external_kernel(
                    funcs[0], self.work, "aie2p", embed_bitcode=True
                )
        self.assertNotIn(os.path.realpath(self.work), owner._compiled_dirs)
        for path in (self.output, Path(f"{self.output}.d"), Path(f"{self.output}.bc")):
            self.assertFalse(path.exists())
        self.assertFalse(compile_utils._compiled_into(funcs[1], self.work))

        with patch.object(
            compile_utils,
            "compile_cxx_core_function",
            side_effect=lambda output_path, **kwargs: Path(output_path).write_bytes(
                b"complete object"
            ),
        ) as compile_kernel:
            compile_utils.compile_external_kernel(funcs[1], self.work, "aie2p")
        compile_kernel.assert_called_once()
        self.assertTrue(compile_utils._compiled_into(funcs[0], self.work))


class ObjectBitcodeTest(unittest.TestCase):
    def test_bitcode_inspection_does_not_rewrite_cached_object(self):
        for returncode in (0, 1):
            with self.subTest(returncode=returncode), patch.object(
                compile_utils.config, "objcopy_path", return_value="llvm-objcopy"
            ), patch.object(
                compile_utils.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], returncode, b"", b""),
            ) as run:
                self.assertEqual(
                    compile_utils._object_has_bitcode("kernel.o"), returncode == 0
                )
                command = run.call_args.args[0]
                self.assertEqual(command[0], "llvm-objcopy")
                self.assertTrue(command[1].startswith("--dump-section=.llvmbc="))
                dump_path = Path(command[1].split("=", 2)[2])
                self.assertEqual(dump_path.name, "kernel.bc")
                self.assertNotEqual(str(dump_path), os.devnull)
                self.assertFalse(dump_path.parent.exists())
                self.assertEqual(command[2:], ["kernel.o", os.devnull])


class BitcodeCacheIdentityTest(unittest.TestCase):
    def test_check_flag_changes_design_cache_identity(self):
        def generator():
            pass

        with patch.object(
            compile_hash, "_compute_artifact_hash", return_value="same-artifacts"
        ):
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
