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
        if not inline:
            # Record each function's frame size in a `.stack_sizes` section,
            # matching the flag aiecc's own core-object `llc` invocation
            # passes (tools/aiecc/aiecc.cpp). Without this, kernel objects
            # carry no stack accounting at all -- and kernels are where large
            # frames actually live. Only meaningful for object codegen; the
            # inline path emits textual LLVM IR with no frame layout yet.
            cmd.append("-fstack-size-section")

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
        if getattr(func, "_symbol_prefix", None):
            # Ensure rename is applied even on cache hit — idempotent with llvm-objcopy
            _rename_symbol_in_object(output_file, func._original_name, func._name)
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

    # Rename symbol if a prefix is set.
    if getattr(func, "_symbol_prefix", None):
        original = func._original_name
        prefixed = func._name  # already prefixed
        _rename_symbol_in_object(output_file, original, prefixed)

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
