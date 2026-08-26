# utils.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Low-level helpers for compiling MLIR modules and external C++ kernels to NPU artifacts."""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import aie.utils.config as config
import numpy as np
from ml_dtypes import bfloat16

if TYPE_CHECKING:
    from aie.ir import (  # pyright: ignore[reportMissingImports]
        Module,  # pyright: ignore[reportAttributeAccessIssue]
    )

logger = logging.getLogger(__name__)


def resolve_target_arch(device=None) -> str:
    """Return ``'aie2'`` or ``'aie2p'`` for the given device, or ``'aie2'`` if device is None."""
    if device is None:
        return "aie2"
    from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
        AIEArch,
    )
    from aie.dialects.aie import (
        get_target_model,  # pyright: ignore[reportAttributeAccessIssue]
    )
    from aie.iron.device import Device

    if isinstance(device, Device):
        arch = device.arch
    else:
        arch = AIEArch(get_target_model(device).get_target_arch())

    if arch == AIEArch.AIE2p:
        return "aie2p"
    if arch == AIEArch.AIE2:
        return "aie2"
    raise RuntimeError(
        f"Unsupported device arch: {arch} (device type: {type(device)})."
    )


# Linkage keywords that may appear immediately after ``define``.  Used to tell
# "this function already declares a linkage" from "the next token is a
# preemption specifier / visibility / cconv / return type"; emitting a second
# linkage keyword is a parse error.
_LLVM_LINKAGE_KEYWORDS = frozenset(
    {
        "private",
        "internal",
        "available_externally",
        "linkonce",
        "weak",
        "common",
        "appending",
        "extern_weak",
        "linkonce_odr",
        "weak_odr",
        "external",
    }
)

# The optional tokens the grammar allows between a function's parameter list
# and its attribute list.
_UNNAMED_ADDR_RE = re.compile(r"\s*(?:local_)?unnamed_addr\b")
_ADDRSPACE_RE = re.compile(r"\s*addrspace\(\d+\)")

# Attributes that cannot coexist with ``alwaysinline`` (the LLVM verifier
# rejects the combination).  clang emits both at ``-O0``, which a caller can
# reach by passing ``-O0`` in ``compile_flags``.
_NO_INLINE_ATTRS_RE = re.compile(r"\s*\b(?:noinline|optnone)\b")

_ATTR_GROUP_DEF_RE = re.compile(r"^attributes\s+#(\d+)\s*=\s*\{(.*)\}\s*$")

_ATTR_GROUP_REF_RE = re.compile(r"#(\d+)\b")


def _end_of_parameter_list(line: str, open_paren: int) -> int:
    """Return the index just past the ``)`` closing a function's parameter list.

    Parameter attributes nest parentheses (``byval(%struct.S)``,
    ``sret({ i32 })``, ``align(4)``) and quoted attribute strings may contain
    unbalanced ones, so the closing paren has to be scanned for rather than
    found with ``str.find``.  Returns -1 if the list does not close on `line`.
    """
    depth = 0
    in_string = False
    i = open_paren
    while i < len(line):
        c = line[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _make_ir_inlinable(ir_path: str, symbol_name: str) -> None:
    """Rewrite an emitted LLVM IR kernel so aiecc inlines it into the core.

    Gives the kernel ``define`` ``alwaysinline`` (so aiecc's conservative
    ``-inline-threshold`` still inlines it after the llvm-link merge) and
    ``linkonce_odr`` linkage (so the now-dead definition is DCE'd post-inline
    instead of being codegen'd).

    Both edits have to respect the ``define`` grammar::

        define [linkage] [preemption] [visibility] [dll] [cconv] [ret attrs]
               <ty> @<name>(<params>) [unnamed_addr] [addrspace(N)] [fn attrs]
               [section] [partition] [comdat] [align] [gc] [prefix] [prologue]
               [personality] (!name !N)* { ...

    so ``alwaysinline`` is inserted right after the parameter list (and any
    ``unnamed_addr`` / ``addrspace``), not next to the opening brace: placing a
    function attribute after a ``personality`` clause or a ``!dbg`` attachment
    -- the latter appears as soon as the caller passes ``-g`` -- is a parse
    error.  ``linkonce_odr`` is inserted only when the define does not already
    carry a linkage keyword.
    """
    lines = Path(ir_path).read_text().splitlines()

    symbol_re = re.compile(r"@" + re.escape(symbol_name) + r"\b\s*\(")
    define_idx = -1
    params_at = -1
    for i, text in enumerate(lines):
        if not text.startswith("define"):
            continue
        match = symbol_re.search(text)
        if match:
            define_idx = i
            params_at = match.end() - 1
            break
    if define_idx < 0:
        raise RuntimeError(
            f"inline=True: no `define` for symbol '{symbol_name}' in "
            f"{ir_path}. The kernel must be defined in this translation unit "
            f'and exported under that exact name (declare it `extern "C"` so '
            f"C++ name mangling does not rename it)."
        )

    line = lines[define_idx]
    params_end = _end_of_parameter_list(line, params_at)
    brace = line.rfind("{")
    if params_end < 0 or brace < params_end:
        raise RuntimeError(
            f"inline=True: could not parse the `define` for '{symbol_name}' in "
            f"{ir_path}: {line!r}"
        )

    # Start of the [fn attrs] slot: after the parameter list and the optional
    # unnamed_addr / addrspace tokens that precede it.
    attrs_at = params_end
    for pattern in (_UNNAMED_ADDR_RE, _ADDRSPACE_RE):
        token = pattern.match(line, attrs_at)
        if token:
            attrs_at = token.end()

    region = line[attrs_at:brace]

    # `alwaysinline` is incompatible with `noinline` / `optnone`.  Those
    # normally arrive through a shared `attributes #N = { ... }` group, so
    # repoint this define at a private copy with them dropped rather than
    # editing a group that other functions in the module also reference.
    groups = {
        int(m.group(1)): m.group(2)
        for m in (_ATTR_GROUP_DEF_RE.match(t) for t in lines)
        if m
    }
    referenced = {int(m.group(1)) for m in _ATTR_GROUP_REF_RE.finditer(region)}
    conflicting = sorted(
        gid
        for gid in referenced
        if gid in groups and _NO_INLINE_ATTRS_RE.search(groups[gid])
    )
    if conflicting:
        next_id = max(groups) + 1
        remap = {}
        for gid in conflicting:
            cleaned = _NO_INLINE_ATTRS_RE.sub("", groups[gid]).strip()
            remap[gid] = next_id
            lines.append(f"attributes #{next_id} = {{ {cleaned} }}")
            next_id += 1
        region = _ATTR_GROUP_REF_RE.sub(
            lambda m: f"#{remap.get(int(m.group(1)), int(m.group(1)))}", region
        )
    # ... and drop them if they were spelled out on the define itself.
    region = _NO_INLINE_ATTRS_RE.sub("", region)

    if not re.search(r"\balwaysinline\b", region):
        region = " alwaysinline" + region
    line = line[:attrs_at] + region + line[brace:]

    head = re.match(r"define\s+([A-Za-z_][A-Za-z0-9_]*)", line)
    linkage = head.group(1) if head else None
    if linkage == "external":
        # Strong definition: swap it for a discardable one.
        line = re.sub(r"^define\s+external\b", "define linkonce_odr", line, count=1)
    elif linkage not in _LLVM_LINKAGE_KEYWORDS:
        # No linkage keyword at all (the common case: `define dso_local ...`).
        line = re.sub(r"^define\s+", "define linkonce_odr ", line, count=1)
    # Any other linkage (internal, weak_odr, ...) already allows the definition
    # to be discarded once every call site has been inlined; leave it alone.

    lines[define_idx] = line
    Path(ir_path).write_text("\n".join(lines) + "\n")


def compile_cxx_core_function(
    source_path: str,
    target_arch: str,
    output_path: str,
    include_dirs: list[str] | None = None,
    compile_args: list[str] | None = None,
    cwd: str | None = None,
    use_chess: bool = False,
    inline: bool = False,
    symbol_name: str | None = None,
):
    """Compile a C++ core function via either Peano or the Chess compiler.

    Peano is the default; pass ``use_chess=True`` for Chess.

    Parameters:
        source_path (str): Path to C++ source.
        target_arch (str): Target architecture, e.g., aie2.
        output_path (str): Output object file path (``.o``), or LLVM IR file
            (textual ``.ll`` or binary ``.bc``) when ``inline`` is True.
        include_dirs (list[str], optional): List of include directories to add with -I.
        compile_args (list[str], optional): Additional compile arguments
            forwarded verbatim to the chosen compiler.
        cwd (str, optional): Overrides the current working directory.
        use_chess (bool): When True, invoke ``xchesscc_wrapper`` instead of
            ``clang++`` (Peano).  Equivalent to the makefile-common
            ``KERNEL_CC=xchesscc_wrapper`` path used by the matmul examples'
            ``use_chess=1`` configurations.  ``xchesscc_wrapper`` reads
            ``AIETOOLS_DIR`` (or auto-detects from the path of ``xchesscc``)
            for the AIE-tools include directory; the standard mlir-aie
            include path is added explicitly here so it doesn't depend on
            the Chess wrapper's include search.
        inline (bool): When True, emit inlinable LLVM IR instead of an object.
        symbol_name (str, optional): Required when ``inline`` is True; names the
            LLVM ``define`` for the kernel that ``_make_ir_inlinable`` rewrites
            to ``alwaysinline`` / ``linkonce_odr``. Must match the symbol as it
            appears in the freshly emitted IR.
    """
    if inline and use_chess:
        raise ValueError(
            "inline=True requires the Peano toolchain and cannot be combined "
            "with use_chess=True"
        )
    if inline and not symbol_name:
        raise ValueError("symbol_name is required when inline=True")

    ir_suffix = Path(output_path).suffix.lower()
    if inline and ir_suffix not in (".ll", ".bc"):
        raise ValueError(
            "inline=True output_path must use .ll for textual LLVM IR or .bc "
            f"for binary LLVM IR; got {output_path!r}"
        )

    # Inline IR is first emitted as text so its kernel definition can be marked
    # alwaysinline/linkonce_odr. A requested .bc is assembled afterward.

    # ``-c`` (object) by default; ``-S -emit-llvm`` (textual IR) for inline.
    emit_flags = ["-S", "-emit-llvm"] if inline else ["-c"]
    if use_chess:
        wrapper = shutil.which("xchesscc_wrapper")
        if not wrapper:
            raise RuntimeError(
                "Could not find 'xchesscc_wrapper' on PATH.  Ensure the "
                "AIE tools and mlir-aie's bin/ directory are sourced "
                "(env_setup.sh) before requesting use_chess=True."
            )
        cmd = [
            wrapper,
            target_arch,  # "aie2" or "aie2p"
            *emit_flags,
            source_path,
            "-o",
            f"{output_path}",
            f"-I{config.cxx_header_path()}",
        ]
    else:
        cmd = [
            config.peano_cxx_path(),
            source_path,
            *emit_flags,
            "-o",
            f"{output_path}",
            f"-I{config.cxx_header_path()}",
            "-std=c++20",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            "-O2",
            "-DNDEBUG",
            # Have the compiler report what it actually read, the way ninja and
            # ccache learn a translation unit's real inputs.
            "-MD",
            "-MF",
            f"{output_path}.d",
            # Pre-trip aie_api's aie_adf.hpp include guard so stock upstream
            # aie_api never pulls in <adf.h> (Vitis-only, absent from Peano).
            # No mlir-aie kernel uses adf:: symbols, so this only elides dead
            # code.  (The chess path gets the same define centrally in
            # tools/chess-clang/xchesscc_wrapper.)
            "-D__AIE_API_AIE_ADF_HPP__",
            f"--target={target_arch}-none-unknown-elf",
        ]

    # Add include directories
    if include_dirs:
        for include_dir in include_dirs:
            cmd.extend(["-I", include_dir])

    # Add additional compile arguments
    if compile_args:
        cmd.extend(compile_args)

    logger.debug("Compiling with: %s", " ".join(cmd))
    ret = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    if ret.stdout:
        logger.debug("%s", ret.stdout.decode())
    if ret.returncode != 0:
        tool = "Chess" if use_chess else "Peano"
        if ret.stderr:
            raise RuntimeError(f"[{tool}] compilation failed:\n{ret.stderr.decode()}")
        raise RuntimeError(f"[{tool}] compilation failed")

    if inline:
        assert symbol_name is not None
        _make_ir_inlinable(output_path, symbol_name)
        if ir_suffix == ".bc":
            # Assemble the rewritten text with clang++, not llvm-as: the
            # llvm-aie wheel ships clang++/llvm-link/opt/llc but no llvm-as, so
            # depending on it breaks every wheel-based install.  clang++ is
            # already a hard requirement of this function.
            #
            # -disable-llvm-passes keeps this a pure text->bitcode conversion.
            # clang normalizes `alwaysinline` into the function's attribute
            # group rather than leaving it on the `define`; llvm-link and the
            # always-inliner honor both spellings identically.
            #
            # The IR is piped in because the input and output are the same path.
            textual_ir = Path(output_path).read_bytes()
            assemble = subprocess.run(
                [
                    config.peano_cxx_path(),
                    "-x",
                    "ir",
                    "-",
                    "-c",
                    "-emit-llvm",
                    "-O0",
                    "-Xclang",
                    "-disable-llvm-passes",
                    f"--target={target_arch}-none-unknown-elf",
                    "-o",
                    output_path,
                ],
                input=textual_ir,
                cwd=cwd,
                check=False,
                capture_output=True,
            )
            if assemble.returncode != 0:
                detail = f":\n{assemble.stderr.decode()}" if assemble.stderr else ""
                raise RuntimeError(f"[Peano] LLVM bitcode assembly failed{detail}")


def _run_aiecc(mlir_file: str, args: list[str]):
    aiecc_bin = config.aiecc_path()
    cmd = [aiecc_bin, mlir_file] + args
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.debug("%s", result.stdout)
    if result.stderr:
        logger.debug("%s", result.stderr)
    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else result.stdout
        raise RuntimeError(
            f"[aiecc] Compilation failed with exit code {result.returncode}:\n"
            f"{error_msg}"
        )


def compile_mlir_module(
    mlir_module: "str | Module",
    insts_path: str | Path | None = None,
    pdi_path: str | Path | None = None,
    xclbin_path: str | Path | None = None,
    elf_path: str | Path | None = None,
    full_elf_path: str | Path | None = None,
    verbose=False,
    work_dir: str | Path | None = None,
    options=None,
    use_chess: bool = False,
    device=None,
    fold_ddr_addr_offset: bool = True,
):
    """Compile an MLIR module to instruction, PDI, ELF, and/or xclbin files using the aiecc module.

    Parameters:
        mlir_module (str): MLIR module to compile.
        insts_path (str): Path to the instructions binary file.
        pdi_path (str): Path to the PDI file.
        xclbin_path (str): Path to the xclbin file.
        elf_path (str): Path to an ELF-wrapped version of the NPU instructions
            (produced via ``aiebu-asm``).  Required by C++ testbenches that
            load instructions through ``xrt::elf`` + ``xrt::module``;
            independent of ``insts_path`` (the Python runtime consumes the
            raw ``.bin``).
        full_elf_path (str): Path to a single self-contained "full" ELF that
            bundles the PDIs and TXN control code (via ``--get-full-elf``).
            Unlike ``elf_path`` (which only wraps the NPU instructions and still
            needs an xclbin), a full ELF is loaded standalone through
            ``pyxrt.hw_context(dev, pyxrt.elf(path))``.  When set, xclbin and
            raw-insts generation are skipped -- the full ELF is self-contained.
        verbose (bool): If True, enable verbose output.
        work_dir (str): Compilation working directory.
        options (list[str]): List of additional options.
        use_chess (bool): When True, drive aiecc with the Chess front-end
            (``--unified``) instead of the Peano front-end.  Must agree
            with the per-ExternalFunction ``_use_chess`` settings — the
            JIT compile orchestration in ``compilabledesign.py`` enforces
            agreement and raises on a mixed peano/chess design.
        device: Optional IRON device (or ``AIEDevice`` enum) used to pick
            the target architecture (aie2 vs aie2p) for any
            :class:`aie.iron.kernel.ExternalFunction` instances that have
            a ``source_file=`` and haven't been compiled yet.  When set
            and ``work_dir`` is provided, those externals are auto-built
            into ``work_dir`` before aiecc runs (matching the @iron.jit
            behavior).  Without this, low-level designs going through
            ``compile_mlir_module`` directly (e.g. ``basic/packet_switch``)
            still need a Makefile-side ``.o`` rule.
    """
    if use_chess:
        # Chess-driven aiecc.  --unified runs all cores' xchesscc invocations
        # in a single Chess process to amortise startup cost; matches the
        # makefile-common ``aiecc_chess_flags=--unified`` recipe.  Chess must
        # be named explicitly: aiecc no longer defaults to it.
        args = [
            "--unified",
            "--xchesscc",
            "--xbridge",
        ]
    else:
        args = [
            f"--peano={config.peano_install_dir()}",
        ]
    if full_elf_path:
        # A full ELF is self-contained (bundles PDIs + TXN control code), so the
        # xclbin and raw-insts artifacts are neither needed nor emitted here.
        args.extend(["--get-full-elf", f"--full-elf-name={full_elf_path}"])
    else:
        if insts_path:
            args.extend(["--get-npu-insts", f"--npu-insts-name={insts_path}"])
        if xclbin_path:
            args.extend(["--get-xclbin", f"--xclbin-name={xclbin_path}"])
    # DDR-patch ABI: XRT (and CPU) consume the folded firmware ABI; HRX consumes
    # the producer-independent (unfolded) insts.bin and adds the AIE DDR aperture
    # offset for every arg itself. cl::opt defaults to true, so only pass the
    # flag when unfolding is requested.
    if not fold_ddr_addr_offset:
        args.append("--fold-ddr-addr-offset=false")
    if pdi_path:
        args.extend(["--get-pdi", f"--pdi-name={pdi_path}"])
    if elf_path:
        args.extend(["--get-elf", f"--elf-name={elf_path}"])
    if work_dir:
        args.append(f"--tmpdir={work_dir}")
        # Emit input_with_addresses.mlir into work_dir; the JIT DMA-size
        # validator (parse_dma_sizes) and the trace parser read it from there.
        # It is a requested output, so it lands in --output-dir; point that at
        # work_dir (the insts/xclbin/pdi paths are absolute and unaffected).
        args.append(f"--output-dir={work_dir}")
        args.append("--get-input-with-addresses")
    if verbose:
        args.append("--verbose")
    if options:
        args.extend(options)
    # Auto-build any source-bearing ExternalFunction kernels into work_dir
    # so aiecc's linker can find the .o referenced by link_with.  Mirrors
    # the loop in compilabledesign.py but for callers (e.g. low-level
    # designs using rt.inline_ops) that didn't go through @iron.jit.
    if work_dir and device is not None:
        # Deferred: aie.iron's __init__ imports back into aie.utils.compile.jit,
        # so a module-level import here deadlocks on a cold aie.utils.compile entry.
        from aie.iron.kernel import ExternalFunction

        target_arch = resolve_target_arch(device)
        for func in list(ExternalFunction._instances):
            if not func._compiled and getattr(func, "_source_file", None):
                compile_external_kernel(func, str(work_dir), target_arch)

    # When work_dir is provided, invoke the aiecc binary as a subprocess so
    # that it resolves relative link_with paths (e.g. "add_one.o") against the
    # same directory where compile_external_kernel placed the compiled objects.
    # The MLIR file is written to work_dir/aie.mlir; callers (e.g. jit.py)
    # may have already written it there, in which case this is a no-op write.
    # If no work_dir is provided, fall back to a temporary file instead.
    if work_dir:
        mlir_file = os.path.join(work_dir, "aie.mlir")
        with open(mlir_file, "w") as f:
            f.write(str(mlir_module))
        _run_aiecc(mlir_file, args)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(str(mlir_module))
            mlir_file = f.name
        try:
            _run_aiecc(mlir_file, args)
        finally:
            os.unlink(mlir_file)


def _rename_symbol_in_object(object_path: str, old_name: str, new_name: str) -> None:
    """Rename a symbol in a compiled object file using llvm-objcopy."""
    objcopy = config.objcopy_path()
    result = subprocess.run(
        [objcopy, f"--redefine-sym={old_name}={new_name}", str(object_path)],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Symbol rename failed: {result.stderr.decode()}")


def _demangled_base_name(demangled: str) -> str:
    """Return the qualified function name from a demangled symbol.

    Everything from the first ``(`` -- the parameter list -- is dropped, so
    ``const``, ``__restrict`` and overload parameter types play no part in
    matching.  Namespace qualification is part of the name and is kept.
    """
    paren = demangled.find("(")
    return (demangled if paren == -1 else demangled[:paren]).strip()


def _select_cxx_symbol(symbols: dict[str, str], expected_name: str) -> str | None:
    """Pick the defined symbol that provides ``expected_name``.

    Matching is on the demangled base name rather than on a mangled name
    reconstructed from the IRON signature.  That asymmetry is the whole point:
    ``const``, ``__restrict`` and namespaces all change the mangling but not
    the base name, so they need no representation in ``arg_types``.

    Args:
        symbols: Mapping of mangled symbol name to its demangled form, as
            llvm-nm piped through llvm-cxxfilt produces it.  A C-linkage
            symbol maps to itself.
        expected_name: Symbol name IRON emitted into the MLIR.  May be
            namespace-qualified to disambiguate.

    Returns:
        The mangled symbol to rename, or None when ``expected_name`` is already
        defined -- the kernel has C linkage and nothing needs to happen.

    Raises:
        ValueError: When nothing matches, or when several symbols do.  IRON
            cannot choose between overloads: ``arg_types`` holds numpy types,
            which cannot distinguish ``int*`` from ``int const*``.
    """
    if expected_name in symbols:
        return None

    qualified_suffix = f"::{expected_name}"
    matches = []
    for mangled, demangled in symbols.items():
        base = _demangled_base_name(demangled)
        if base == expected_name or base.endswith(qualified_suffix):
            matches.append(mangled)

    if len(matches) == 1:
        return matches[0]

    if not matches:
        present = "\n  ".join(sorted(symbols.values())) or "(none)"
        raise ValueError(
            f"No symbol named '{expected_name}' is defined. Defined symbols "
            f"are:\n  {present}"
        )

    candidates = "\n  ".join(sorted(symbols[m] for m in matches))
    raise ValueError(
        f"Symbol '{expected_name}' is ambiguous -- several overloads match:"
        f"\n  {candidates}\n"
        "IRON cannot choose between them. Give the kernel a distinct name, or "
        "keep a single overload."
    )


# Demangled spellings llvm-cxxfilt produces for the element types IRON can
# express, verified against a Peano-compiled aie2 object.  Plain ``char`` is
# accepted for both signed and unsigned 8-bit buffers because its signedness is
# implementation-defined, and ``long``/``long long`` both for 64-bit because the
# choice is target-dependent.  Anything absent here is deliberately not checked.
_CXX_SPELLINGS = {
    np.dtype(np.int8): {"signed char", "char"},
    np.dtype(np.uint8): {"unsigned char", "char"},
    np.dtype(np.int16): {"short"},
    np.dtype(np.uint16): {"unsigned short"},
    np.dtype(np.int32): {"int"},
    np.dtype(np.uint32): {"unsigned int"},
    np.dtype(np.int64): {"long", "long long"},
    np.dtype(np.uint64): {"unsigned long", "unsigned long long"},
    np.dtype(np.float32): {"float"},
    np.dtype(np.float64): {"double"},
    np.dtype(bfloat16): {"bfloat16"},
    np.dtype(np.bool_): {"bool"},
}

_CXX_QUALIFIERS = ("const", "volatile", "__restrict", "restrict", "__restrict__")


def _split_top_level_params(param_list: str) -> list[str]:
    """Split a demangled parameter list on its top-level commas.

    Template argument lists and function-pointer parameters contain commas of
    their own -- ``aie::vector<int, 16>*, int`` is two parameters, not three --
    so track nesting depth rather than splitting naively.
    """
    params: list[str] = []
    depth = 0
    current = ""
    for char in param_list:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        if char == "," and depth == 0:
            params.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        params.append(current.strip())
    return params


def _normalize_cxx_param(param: str) -> tuple[str, bool]:
    """Reduce one demangled parameter to ``(base type, is_pointer)``.

    Qualifiers are dropped: they change the mangling but never the meaning of
    the argument as far as IRON can express it.  A reference counts as a
    pointer, since both arrive as an address.
    """
    is_pointer = "*" in param or "&" in param
    base = param.replace("*", " ").replace("&", " ")
    tokens = [t for t in base.split() if t not in _CXX_QUALIFIERS]
    return " ".join(tokens), is_pointer


def _iron_arg_spec(arg) -> tuple[np.dtype | None, bool]:
    """Reduce one ``arg_types`` entry to ``(dtype, is_pointer)``.

    Mirrors ``BaseKernel.arg_dtype``: an ``np.ndarray[shape, np.dtype[T]]``
    carries its element type in ``__args__``; anything else is a scalar.
    Returns a None dtype for entries whose element type cannot be read, which
    the caller treats as "do not check".
    """
    type_args = getattr(arg, "__args__", None)
    if type_args is not None and len(type_args) >= 2:
        dt = type_args[1]
        dt_args = getattr(dt, "__args__", None)
        try:
            return np.dtype(dt_args[0] if dt_args is not None else dt), True
        except TypeError:
            return None, True
    try:
        return np.dtype(arg), False
    except TypeError:
        return None, False


def _check_cxx_signature(demangled: str, arg_types, symbol: str) -> None:
    """Check a kernel's declared ``arg_types`` against its real C++ signature.

    Only meaningful for C++-linkage kernels: an ``extern "C"`` symbol demangles
    to a bare name carrying no parameter list, so there is nothing to compare.

    Strictness is tiered by how costly a false positive would be.  Parameter
    count and pointer-vs-scalar are unambiguous in a demangled signature and
    are hard errors.  Element types are compared only when both the C++
    spelling and the numpy dtype are understood, so a kernel taking an aie_api
    vector or a struct is never rejected by a checker that cannot read it.

    Args:
        demangled: Demangled symbol, e.g. ``f(int*, int*, int)``.
        arg_types: The kernel's declared argument types, or None to skip.
        symbol: Name used in error messages.

    Raises:
        ValueError: On an arity or pointer-vs-scalar mismatch, or on an element
            type mismatch between two understood types.
    """
    if arg_types is None:
        return
    open_paren = demangled.find("(")
    if open_paren == -1 or not demangled.endswith(")"):
        # A bare name: C linkage, nothing to check.
        return

    params = _split_top_level_params(demangled[open_paren + 1 : -1])
    # A no-argument function is spelled `f()` by llvm-cxxfilt, so an empty
    # parameter list is already the empty list.
    if len(params) != len(arg_types):
        raise ValueError(
            f"Kernel '{symbol}' is declared with {len(arg_types)} argument(s) "
            f"but its C++ signature takes {len(params)}: {demangled}"
        )

    for index, (param, arg) in enumerate(zip(params, arg_types), start=1):
        cxx_type, cxx_is_pointer = _normalize_cxx_param(param)
        dtype, iron_is_pointer = _iron_arg_spec(arg)

        if cxx_is_pointer != iron_is_pointer:
            expected = "a buffer" if iron_is_pointer else "a scalar"
            actual = "a pointer" if cxx_is_pointer else "a value"
            raise ValueError(
                f"Kernel '{symbol}' argument {index}: declared as {expected}, "
                f"but the C++ signature takes {actual} ('{param}'). "
                f"Full signature: {demangled}"
            )

        if dtype is None or dtype not in _CXX_SPELLINGS:
            continue
        if cxx_type not in _CXX_SPELLINGS[dtype]:
            # Only complain when the C++ spelling is one we model; an unknown
            # type is not evidence of a mismatch.
            if any(cxx_type in spellings for spellings in _CXX_SPELLINGS.values()):
                raise ValueError(
                    f"Kernel '{symbol}' argument {index}: declared as "
                    f"{dtype.name}, but the C++ signature takes '{cxx_type}'. "
                    f"Full signature: {demangled}"
                )


def _defined_global_symbols(object_path: str) -> dict[str, str]:
    """Return ``{mangled: demangled}`` for an object's defined global symbols.

    ``--extern-only`` drops the local ``.LBB*`` labels that would otherwise
    appear alongside the functions.
    """
    nm = config.peano_nm_path()
    listing = subprocess.run(
        [nm, "--defined-only", "--extern-only", "--format=just-symbols", object_path],
        capture_output=True,
        check=False,
    )
    if listing.returncode != 0:
        raise RuntimeError(
            f"Could not list symbols of {object_path}: {listing.stderr.decode()}"
        )
    mangled = listing.stdout.decode().split()
    if not mangled:
        return {}

    cxxfilt = config.peano_cxxfilt_path()
    demangling = subprocess.run(
        [cxxfilt],
        input="\n".join(mangled),
        capture_output=True,
        text=True,
        check=False,
    )
    if demangling.returncode != 0:
        raise RuntimeError(f"Could not demangle symbols: {demangling.stderr}")
    demangled = demangling.stdout.splitlines()
    if len(demangled) != len(mangled):
        raise RuntimeError(
            f"llvm-cxxfilt returned {len(demangled)} names for {len(mangled)} "
            f"symbols of {object_path}; cannot pair them up."
        )
    return dict(zip(mangled, demangled))


def _resolve_cxx_linkage_symbol(object_path: str, expected_name: str) -> str | None:
    """Make ``expected_name`` resolvable in a C++-compiled kernel object.

    A kernel source without ``extern "C"`` defines a mangled symbol, but IRON
    emitted ``func.func private @<expected_name>`` and the linker needs that
    exact name.  Rename the mangled symbol to it.

    A no-op when the kernel already has C linkage, which also makes this
    idempotent -- required because the JIT re-runs it on a cache hit, against
    an object that was already renamed.

    Returns:
        The demangled signature of the symbol that was renamed, so the caller
        can check it against the kernel's declared ``arg_types``; None when the
        kernel already had C linkage and no signature is recoverable.
    """
    symbols = _defined_global_symbols(object_path)
    mangled = _select_cxx_symbol(symbols, expected_name)
    if mangled is None:
        return None
    _rename_symbol_in_object(object_path, mangled, expected_name)
    return symbols[mangled]


def _apply_symbol_renames(func, output_file: str) -> None:
    """Bring an ExternalFunction's object in line with the symbol IRON emitted.

    Order matters.  A C++-linkage source defines a mangled symbol, not
    ``_original_name``, so the prefix rename below would match nothing and
    silently no-op.  Resolving C++ linkage first restores the invariant that
    rename already assumes::

        _Z10min_kernelPiS_i  ->  min_kernel  ->  pfx_min_kernel

    C++ linkage resolution is skipped on two paths, both of which keep exactly
    the behaviour they had before it existed:

    * ``inline=True`` emits LLVM IR, not an object.  llvm-nm cannot read it,
      and ``_make_ir_inlinable`` already rejects a mangled inline kernel with a
      message naming the ``extern "C"`` fix.
    * ``use_chess=True`` produces an xchesscc object whose compatibility with
      llvm-nm is unverified.  Inspecting it could break chess kernels that
      link fine today, so C++-linkage kernels stay unsupported there until
      someone with the toolchain installed can confirm.

    Both steps are idempotent, as the cache-hit path re-runs them against an
    object that was already renamed.
    """
    original_name = getattr(func, "_original_name", func._name)

    inspectable_object = not (
        getattr(func, "_inline", False) or getattr(func, "_use_chess", False)
    )
    if inspectable_object:
        demangled = _resolve_cxx_linkage_symbol(output_file, original_name)
        if demangled is not None:
            _check_cxx_signature(
                demangled, getattr(func, "_arg_types", None), original_name
            )

    if getattr(func, "_symbol_prefix", None):
        _rename_symbol_in_object(output_file, original_name, func._name)


def compile_external_kernel(func, kernel_dir, target_arch):
    """Compile an ExternalFunction to an object file in the kernel directory.

    The output file is named ``func.object_file_name`` and placed in ``kernel_dir``.
    If the object file already exists in ``kernel_dir``, compilation is skipped.

    Args:
        func: ExternalFunction instance to compile.
        kernel_dir: Directory where the compiled object file will be placed.
            Must be the same directory passed as ``work_dir`` to
            ``compile_mlir_module`` so that relative link_with paths resolve
            correctly.
        target_arch: Peano target architecture string (e.g., "aie2", "aie2p").
    """
    # Skip if already compiled in this session.
    if func._compiled:
        return

    # inline + symbol_prefix is unsupported: the MLIR func.call uses the
    # prefixed func._name, but an inline kernel is emitted as a textual .ll whose
    # ``define`` carries the un-prefixed _original_name. Object mode reconciles
    # the two via an llvm-objcopy --redefine-sym rename, which cannot rewrite a
    # .ll. Fail loudly here rather than downstream in objcopy or as a silent
    # call/define name mismatch at llvm-link time.
    if getattr(func, "_inline", False) and getattr(func, "_symbol_prefix", None):
        raise NotImplementedError(
            f"ExternalFunction '{func._name}': inline=True combined with "
            "symbol_prefix is not supported (an inline kernel is emitted as "
            "LLVM IR and cannot be symbol-renamed). Use inline without a "
            "symbol_prefix, or drop inline for this kernel."
        )

    # Skip if the object file already exists (cache hit).
    output_file = os.path.join(kernel_dir, func.object_file_name)
    if os.path.exists(output_file):
        _apply_symbol_renames(func, output_file)
        return

    original_name = getattr(func, "_original_name", func._name)

    if func._source_string is not None:
        source_file = os.path.join(kernel_dir, f"{original_name}.cc")
        with open(source_file, "w") as f:
            f.write(func._source_string)
        compile_cxx_core_function(
            source_path=source_file,
            target_arch=target_arch,
            output_path=output_file,
            # The source is compiled under _original_name, so that is the symbol
            # in the emitted .ll ``define`` that _make_ir_inlinable must rewrite.
            # (inline + symbol_prefix is rejected above, so no rename applies.)
            symbol_name=func._original_name,
            include_dirs=func._include_dirs,
            compile_args=func._compile_flags,
            cwd=str(kernel_dir),
            inline=getattr(func, "_inline", False),
            use_chess=getattr(func, "_use_chess", False),
        )

    elif func._source_file is not None:
        source_file = os.path.join(kernel_dir, f"{original_name}.cc")
        # Check if source file exists before copying
        if not os.path.exists(func._source_file):
            raise FileNotFoundError(
                f"ExternalFunction '{func._name}': source file not found: {func._source_file}"
            )
        shutil.copy2(func._source_file, source_file)
        # Include the original source file's directory so relative includes
        # (e.g. "../aie_kernel_utils.h") still resolve after the file is
        # copied into kernel_dir.
        src_dir = os.path.dirname(os.path.abspath(func._source_file))
        include_dirs = list(func._include_dirs)
        if src_dir not in include_dirs:
            include_dirs.append(src_dir)
        compile_cxx_core_function(
            source_path=source_file,
            target_arch=target_arch,
            output_path=output_file,
            # _original_name is the symbol in the emitted .ll ``define`` (see
            # the source_string branch above).
            symbol_name=func._original_name,
            include_dirs=include_dirs,
            compile_args=func._compile_flags,
            cwd=kernel_dir,
            inline=getattr(func, "_inline", False),
            use_chess=getattr(func, "_use_chess", False),
        )
    else:
        raise ValueError("Neither source_string nor source_file is provided")

    _apply_symbol_renames(func, output_file)

    func._compiled = True


def _cleanup_failed_compilation(cache_dir):
    """Clean up cache directory after failed compilation.

    Preserves the lock file and, when present, the ``repeater`` reproducer dir
    that aiecc's ``--enable-repeater-scripts`` writes.
    """
    if not os.path.exists(cache_dir):
        return

    for item in os.listdir(cache_dir):
        if item in (".lock", "repeater"):
            continue
        item_path = os.path.join(cache_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
