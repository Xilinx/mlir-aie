# test_kernel_tree_digest.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""A benchmark row names the kernel sources it ran, not only the commit."""

from aie.utils.benchmark import kernel_tree_digest, provenance


def _tree(root, body):
    (root / "aie_kernels" / "aie2p").mkdir(parents=True)
    (root / "aie_runtime_lib").mkdir()
    (root / "aie_kernels" / "aie2p" / "k.cc").write_text(body)
    return root


def test_the_digest_follows_the_content_of_the_selected_tree(tmp_path, monkeypatch):
    a = _tree(tmp_path / "a", "void k() {}\n")
    b = _tree(tmp_path / "b", "void k() {}\n")
    c = _tree(tmp_path / "c", "void k() { event0(); }\n")
    digests = {}
    for tree in (a, b, c):
        monkeypatch.setenv("MLIR_AIE_KERNEL_SOURCES", str(tree))
        digests[tree.name] = kernel_tree_digest()
    assert digests["a"] == digests["b"] != digests["c"]


def test_provenance_names_the_override_and_its_digest(tmp_path, monkeypatch):
    tree = _tree(tmp_path, "void k() {}\n")
    monkeypatch.setenv("MLIR_AIE_KERNEL_SOURCES", str(tree))
    line = provenance(device="NPU Strix")
    assert f"kernel_sources {tree}" in line
    assert f"kernels {kernel_tree_digest()}" in line
    assert line.endswith("device NPU Strix")
