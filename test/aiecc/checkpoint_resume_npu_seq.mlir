//===- checkpoint_resume_npu_seq.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// Regression for the shared-module fix (NodeDeserializer<OpInModule>,
// Graph.h): a --resume that lands at `npu_seq_{0}.mlir` -- an OpInModule<
// RuntimeSequenceOp> fan-out, unlike checkpoint_resume_ir.mlir's `perCore_{0}.
// mlir` (a *different* OpInModule<KeyOp> instantiation) -- must rehydrate the
// checkpoint by parsing the shared module.mlir ONCE and rebinding each item to
// it, not by re-cloning per item. A per-item-cloning deserializer regression
// reproduces the exact O(N) blowup the forward SplitIRAction fix already
// closed, just on the resume path instead. This reconfig-union input folds
// two designs into ONE host device with two config dispatches plus each
// design's own device, so the cut lands on FIVE items
// (main_init/main_configs_1/main_configs_2/cfg_a_cfg_a_run/cfg_b_cfg_b_run)
// sharing one module -- multi-item, not the single-item case.
//
// The full-elf embeds a PDI, and bootgen stamps a wall-clock timestamp (+
// checksum) into every PDI it emits, so `full_elf.elf` itself is NOT
// byte-reproducible run-to-run independent of this change (see Task 3 golden
// diff). The pre-PDI per-sequence instruction binaries
// (`npu_insts_full_elf_{0}.bin`) are deterministic and are what this test
// diffs; the full ELF is still built on both sides (same reconfig-method
// pipeline) and only its size is checked, to confirm resume drove the build
// all the way through assembly.

// RUN: rm -rf %t && mkdir -p %t

// Straight-through reference build.
// RUN: aiecc --get-full-elf --full-elf-name=%t/ref.elf --get='npu_insts_full_elf_{0}.bin' --reconfig-method=ctrlpkt --tmpdir=%t/ref.prj --output-dir=%t/ref_out %S/Inputs/reconfig_twodevice_a.mlir %S/Inputs/reconfig_twodevice_b.mlir

// Cut at the npu_seq_{0}.mlir OpInModule<RuntimeSequenceOp> frontier (5 items,
// one shared module) and checkpoint it.
// RUN: aiecc --get-full-elf --full-elf-name=%t/cut.elf --reconfig-method=ctrlpkt --tmpdir=%t/cut.prj --output-dir=%t/cut_out --cut='npu_seq_{0}.mlir' --checkpoint=%t/cut.ckpt %S/Inputs/reconfig_twodevice_a.mlir %S/Inputs/reconfig_twodevice_b.mlir

// The checkpoint stores the module once (module.mlir) plus each item's
// focus-op key -- not five per-item clones.
// RUN: ls %t/cut.ckpt/*/ | FileCheck --check-prefix=CKPT %s
// CKPT: module.mlir

// Resume the SAME checkpoint twice (both rehydrate it independently, proving
// the deserializer is stateless / repeatable -- parse module.mlir once per
// invocation, rebind 5 items by walk index):
//   1. No --get: rebuilds everything the manifest originally requested
//      (--get-full-elf's folded ELF, i.e. cut.elf).
//   2. With --get: a *surgical* resume -- passing --get alongside --resume
//      narrows this invocation to exactly the named edge(s) instead of
//      adding to the manifest's build (see CommandLineOptions.h
//      resolveCommandLine / aiecc.cpp's `resume.active` output-filtering),
//      so it must be a separate invocation to also pull out the
//      deterministic per-sequence instruction binaries.
// RUN: rm -rf %t/cut_out && mkdir -p %t/cut_out
// RUN: aiecc --resume=%t/cut.ckpt/manifest.json
// RUN: aiecc --resume=%t/cut.ckpt/manifest.json --get='npu_insts_full_elf_{0}.bin'

// Every per-sequence instruction binary must be byte-identical to the
// straight-through reference.
// RUN: cmp %t/ref_out/npu_insts_full_elf_main_init.bin %t/cut_out/npu_insts_full_elf_main_init.bin
// RUN: cmp %t/ref_out/npu_insts_full_elf_main_configs_1.bin %t/cut_out/npu_insts_full_elf_main_configs_1.bin
// RUN: cmp %t/ref_out/npu_insts_full_elf_main_configs_2.bin %t/cut_out/npu_insts_full_elf_main_configs_2.bin
// RUN: cmp %t/ref_out/npu_insts_full_elf_cfg_a_cfg_a_run.bin %t/cut_out/npu_insts_full_elf_cfg_a_cfg_a_run.bin
// RUN: cmp %t/ref_out/npu_insts_full_elf_cfg_b_cfg_b_run.bin %t/cut_out/npu_insts_full_elf_cfg_b_cfg_b_run.bin

// The full ELF is rebuilt too, at the same size as the reference (its bytes
// may differ from bootgen's PDI timestamp, which is unrelated to this
// change).
// RUN: wc -c < %t/ref.elf > %t/ref.size
// RUN: wc -c < %t/cut.elf > %t/cut.size
// RUN: cmp %t/ref.size %t/cut.size
