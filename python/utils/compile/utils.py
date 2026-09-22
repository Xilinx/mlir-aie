# utils.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Low-level helpers for compiling MLIR modules and external C++ kernels to NPU artifacts."""

import concurrent.futures
import contextlib
import filecmp
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import aie.utils.config as config

if TYPE_CHECKING:
    from aie.ir import (  # pyright: ignore[reportMissingImports]
        Module,  # pyright: ignore[reportAttributeAccessIssue]
    )

logger = logging.getLogger(__name__)

# What open(path, "w") would have produced.  Probed once at import because
# reading the umask means setting it, which is process-wide.
_UMASK = os.umask(0o022)
os.umask(_UMASK)
_DEFAULT_FILE_MODE = 0o666 & ~_UMASK
# Versions 1/2 could leave embedded bitcode unprefixed (v2 on Windows).
_SYMBOL_PREFIX_STAMP_VERSION = 3


SHARED_LIB_SUFFIX = ".dll" if os.name == "nt" else ".so"
SHARED_LIB_FLAGS = ["-shared"] if os.name == "nt" else ["-shared", "-fPIC"]


def host_shared_lib_cmd(src: Path, out: Path, *, opt: str, includes=()) -> list[str]:
    """Build a host shared library with the project's C++17 ABI."""
    compiler = config.host_cxx_path()
    link_flags = []
    if SHARED_LIB_SUFFIX == ".dll":
        target = subprocess.run(
            [compiler, "-dumpmachine"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # PE timestamps must not give identical dispatch builds different hashes.
        link_flags.append(
            "-Wl,/Brepro" if "msvc" in target else "-Wl,--no-insert-timestamp"
        )
    return [
        compiler,
        *SHARED_LIB_FLAGS,
        *link_flags,
        opt,
        "-std=c++17",
        *(f"-I{inc}" for inc in includes),
        str(src),
        "-o",
        str(out),
    ]


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


def _check_lut_banks_enabled(options) -> bool:
    """Recognize LLVM boolean option spellings, not just the bare flag."""
    enabled = False
    for option in options:
        if option == "--":
            break
        name, _, value = option.partition("=")
        if name in ("--check-lut-banks", "-check-lut-banks"):
            enabled = value in ("", "1", "true", "True", "TRUE")
    return enabled


def _object_has_bitcode(output_path) -> bool:
    """Inspect the actual cached object without rewriting it or a sidecar."""
    # objcopy cannot open NUL for both outputs concurrently on Windows.
    with tempfile.TemporaryDirectory(prefix="aie-bitcode-") as tmpdir:
        ret = subprocess.run(
            [
                config.objcopy_path(),
                f"--dump-section=.llvmbc={os.path.join(tmpdir, 'kernel.bc')}",
                str(output_path),
                os.devnull,
            ],
            check=False,
            capture_output=True,
        )
    return ret.returncode == 0


def _attach_bitcode(compile_cmd: list[str], output_path: str, cwd) -> None:
    """Put the kernel's LLVM IR in a `.llvmbc` section of its object.

    aiecc's --check-lut-banks recovers which two tables an aie::lut reads from
    the IR, which nothing else preserves: address spaces are gone by codegen and
    the pairing never reaches the symbol table.

    Emitted separately and attached rather than compiled with -fembed-bitcode,
    which clang rejects alongside the -ffunction-sections/-fdata-sections the
    object needs for stack-size attribution.
    """
    bitcode_path = f"{output_path}.bc"
    drop = {"-c", "-ffunction-sections", "-fdata-sections", "-fstack-size-section"}
    cmd = [a for a in compile_cmd if a not in drop]
    cmd[cmd.index("-o") + 1] = bitcode_path
    cmd += ["-emit-llvm", "-c"]

    ret = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True)
    if ret.returncode != 0:
        raise RuntimeError(
            "Could not emit bitcode for --check-lut-banks:\n" + ret.stderr.decode()
        )

    ret = subprocess.run(
        [config.objcopy_path(), f"--add-section=.llvmbc={bitcode_path}", output_path],
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    if ret.returncode != 0:
        raise RuntimeError(
            "Could not attach bitcode for --check-lut-banks:\n" + ret.stderr.decode()
        )


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
    embed_bitcode: bool = False,
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
        embed_bitcode (bool): Preserve Peano LLVM IR in the object's ``.llvmbc``
            section for ``--check-lut-banks``. Inline kernels already retain IR.
            Not supported with Chess.
    """
    if inline and use_chess:
        raise ValueError(
            "inline=True requires the Peano toolchain and cannot be combined "
            "with use_chess=True"
        )
    if embed_bitcode and use_chess:
        raise ValueError(
            "embed_bitcode=True requires the Peano toolchain and cannot be "
            "combined with use_chess=True (--check-lut-banks needs Peano LLVM IR)"
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
            # aie_api tests capability macros Peano never defines, so -Wundef
            # is only usable once the vendored headers are treated as system.
            "--system-header-prefix=aie_api/",
            "-Werror=undef",
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
            # -fstack-size-section matches -stack-size-section on the llc
            # invocation of aiecc for a core object, so a kernel object carries
            # the same stack accounting. -ffunction-sections and
            # -fdata-sections give each symbol its own section, which the
            # attribution in StackSizeAnalysis.h needs.
            cmd.extend(
                ["-ffunction-sections", "-fdata-sections", "-fstack-size-section"]
            )

    # Add include directories
    if include_dirs:
        for include_dir in include_dirs:
            cmd.extend(["-I", str(include_dir)])

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

    if embed_bitcode and not inline:
        try:
            _attach_bitcode(cmd, output_path, cwd)
        except BaseException:
            # An object without its requested IR must never become a cache hit.
            failed_output = Path(output_path)
            if cwd is not None and not failed_output.is_absolute():
                failed_output = Path(cwd) / failed_output
            failed_output.unlink(missing_ok=True)
            Path(f"{failed_output}.bc").unlink(missing_ok=True)
            raise

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


def _run_aiecc(mlir_file: str, args: list[str], cwd: str | Path | None = None):
    aiecc_bin = os.path.abspath(config.aiecc_path())
    cmd = [aiecc_bin, os.path.abspath(mlir_file)] + args
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        logger.debug("%s", result.stdout)
    if result.stderr:
        logger.debug("%s", result.stderr)
        # Diagnostics from a build that succeeded would otherwise be dropped:
        # debug logging is off by default and the failure path below only runs
        # on a non-zero exit.
        if result.returncode == 0:
            for line in result.stderr.splitlines():
                # Notes are the explanation, not decoration. A warning that
                # queue-depth enforcement could not be applied says why in an
                # attached note, so dropping notes leaves the generic overflow
                # text with no hint that the target is the reason. Diagnostics
                # without a location print without the "file:line:" prefix.
                if any(
                    f": {severity}:" in line or line.startswith(f"{severity}:")
                    for severity in ("warning", "error", "note")
                ):
                    print(f"[aiecc] {line}", file=sys.stderr)
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
    npu_cpp_path: str | Path | None = None,
    npu_cpp_emit_dispatch_shim: bool = False,
):
    """Compile MLIR to instruction, PDI, ELF, xclbin, or C++ files using aiecc.

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
        work_dir (str): Compilation working directory, also used as aiecc's
            subprocess working directory to resolve relative kernel paths.
        options (list[str]): List of additional options. Relative paths in these
            options are interpreted by aiecc from work_dir when provided.
        use_chess (bool): When True, drive aiecc with the Chess front-end
            (``--unified``) instead of the Peano front-end.  Must agree
            with the per-ExternalFunction ``_use_chess`` settings — the
            JIT compile orchestration in ``compilabledesign.py`` enforces
            agreement and raises on a mixed peano/chess design.
        device: Optional IRON device (or ``AIEDevice`` enum) used to pick
            the target architecture (aie2 vs aie2p) for any
            `aie.iron.kernel.ExternalFunction` instances that have
            a ``source_file=`` and haven't been compiled yet.  When set
            and ``work_dir`` is provided, those externals are auto-built
            into ``work_dir`` before aiecc runs (matching the @iron.jit
            behavior).  Without this, low-level designs going through
            ``compile_mlir_module`` directly (e.g. ``basic/packet_switch``)
            still need a Makefile-side ``.o`` rule.
        npu_cpp_path: Output parameterized C++ transaction builder, produced by
            aiecc's same runtime-sequence pipeline as static instructions.
        npu_cpp_emit_dispatch_shim: Include the C ABI used by the Python dispatch
            bridge. Native callers can leave this false and call the generated
            C++ function directly.
    """
    if work_dir:
        work_dir = os.path.abspath(work_dir)
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
            f"--peano={os.path.abspath(config.peano_install_dir())}",
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
    if npu_cpp_path is not None:
        args.extend(["--get-npu-cpp", f"--npu-cpp-name={npu_cpp_path}"])
        if npu_cpp_emit_dispatch_shim:
            args.append("--npu-cpp-emit-dispatch-shim")
    elif npu_cpp_emit_dispatch_shim:
        raise ValueError("npu_cpp_emit_dispatch_shim requires npu_cpp_path.")
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
        compile_external_kernels(
            [
                f
                for f in ExternalFunction._instances
                if getattr(f, "_source_file", None)
            ],
            str(work_dir),
            target_arch,
            embed_bitcode=_check_lut_banks_enabled(options or []),
        )

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
        _run_aiecc(mlir_file, args, cwd=work_dir)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(str(mlir_module))
            mlir_file = f.name
        try:
            _run_aiecc(mlir_file, args)
        finally:
            os.unlink(mlir_file)


def _rename_ir_symbols(ir: str, symbols: list[str], prefix: str) -> str:
    """Apply the native rename map to IR globals and their COMDAT groups."""
    names = {symbol.encode() for symbol in symbols}
    # Consume comments and ordinary strings too: @name in a string constant,
    # inline assembly, or debug metadata is not an LLVM global reference.
    tokens = re.compile(
        r"(?<![-a-zA-Z$._0-9%!])[@$]"
        r'(?:"(?:[^"\\]|\\[0-9a-fA-F]{2})*"|[-a-zA-Z$._0-9]+)'
        r"(?![-a-zA-Z$._0-9:])"
        r'|"(?:[^"\\]|\\.)*"|;[^\n]*'
    )

    def rename(match):
        token = match.group()
        if token[0] not in "@$":
            return token
        name = token[1:]
        if name.startswith('"'):
            name = re.sub(
                rb"\\([0-9a-fA-F]{2})",
                lambda m: bytes([int(m[1], 16)]),
                name[1:-1].encode(),
            )
        else:
            name = name.encode()
        # LLVM's \01 suppresses target mangling and is absent from llvm-nm.
        unmangled = name.startswith(b"\x01")
        native_name = name[1:] if unmangled else name
        if native_name not in names:
            return token
        renamed = (b"\x01" if unmangled else b"") + prefix.encode() + native_name
        escaped = "".join(
            chr(c) if 32 <= c < 127 and c not in (34, 92) else f"\\{c:02X}"
            for c in renamed
        )
        return f'{token[0]}"{escaped}"'

    return tokens.sub(rename, ir)


def _prefix_embedded_bitcode(object_path, bitcode_path, symbols, prefix):
    """Stage renamed bitcode without changing the object on any failure."""
    opt = os.path.join(
        os.path.dirname(config.peano_cxx_path()),
        "opt.exe" if os.name == "nt" else "opt",
    )
    commands = [
        [
            config.objcopy_path(),
            f"--dump-section=.llvmbc={bitcode_path}",
            str(object_path),
            os.devnull,
        ],
        [opt, "-S", str(bitcode_path), "-o", "-"],
        [opt, "-o", str(bitcode_path)],
    ]
    ir = None
    for command in commands:
        result = subprocess.run(command, input=ir, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Embedded bitcode symbol prefixing failed: {result.stderr.decode()}"
            )
        if command == commands[1]:
            ir = _rename_ir_symbols(result.stdout.decode(), symbols, prefix).encode()


def prefix_symbols_in_object(object_path: str, prefix: str) -> None:
    """Prefix every defined, external symbol in a compiled object file.

    Used when linking multiple independently-compiled objects into one module
    (e.g. IRON's operator fusion, or mlir-aie's own kernel memoization: see
    ``compile_external_kernel``'s ``symbol_prefix`` handling), to avoid symbol
    collisions between them: every symbol an object defines is renamed to
    ``{prefix}{symbol}`` before it is linked alongside sibling objects.

    Internally this lists symbols with llvm-nm and bulk-renames them with a
    single llvm-objcopy --redefine-syms= pass, done directly in Python
    (rather than shelling out to sh/awk) so it behaves identically on POSIX
    and Windows. nm's exit status is checked explicitly before objcopy ever
    runs: silently ignoring an nm failure would produce an empty rename map,
    turning this into a silent no-op that only surfaces later as a
    confusing "undefined symbol: <prefix><sym>" at final link time.
    Embedded ``.llvmbc`` IR receives the same rename map before that pass, so
    native definitions and the IR used by ``--check-lut-banks`` stay in sync.

    This operation is intentionally literal: every defined external symbol is
    renamed to ``{prefix}{symbol}``, even if the original spelling already
    starts with ``prefix``. Callers that need one-time application across
    cache hits must track that state explicitly rather than inferring it from
    the symbol names themselves.
    """
    nm = config.nm_path()
    nm_result = subprocess.run(
        [nm, "--defined-only", "--extern-only", str(object_path)],
        capture_output=True,
        check=False,
    )
    if nm_result.returncode != 0:
        raise RuntimeError(f"Symbol listing failed: {nm_result.stderr.decode()}")

    symbols = [
        line.split()[-1]
        for line in nm_result.stdout.decode().splitlines()
        if len(line.split()) >= 3
    ]

    objcopy = config.objcopy_path()
    with tempfile.TemporaryDirectory(
        prefix="aie-symbol-map-", dir=os.path.dirname(object_path) or "."
    ) as tmpdir:
        map_file = os.path.join(tmpdir, "symbols.map")
        with open(map_file, "w") as f:
            for symbol in symbols:
                f.write(f"{symbol} {prefix}{symbol}\n")

        command = [objcopy, f"--redefine-syms={map_file}"]
        if _object_has_bitcode(object_path):
            bitcode_path = os.path.join(tmpdir, "renamed.bc")
            _prefix_embedded_bitcode(object_path, bitcode_path, symbols, prefix)
            command.append(f"--update-section=.llvmbc={bitcode_path}")
        result = subprocess.run(
            [*command, str(object_path)],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Symbol prefixing failed: {result.stderr.decode()}")


def _symbol_prefix_stamp_path(object_path: str, prefix: str) -> str:
    """Return the sidecar path that records one successful symbol-prefix pass."""
    prefix_digest = hashlib.sha256(prefix.encode()).hexdigest()[:16]
    return f"{object_path}.prefix_state.{prefix_digest}.json"


def _sha256_file(path: str) -> str:
    """Return the SHA-256 digest of ``path``'s current contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _has_current_symbol_prefix_stamp(object_path: str, prefix: str) -> bool:
    """Report whether ``object_path`` already carries a current prefix stamp."""
    stamp_path = _symbol_prefix_stamp_path(object_path, prefix)
    try:
        with open(stamp_path) as f:
            state = json.load(f)
        object_sha256 = _sha256_file(object_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return state == {
        "version": _SYMBOL_PREFIX_STAMP_VERSION,
        "prefix": prefix,
        "object_sha256": object_sha256,
    }


def _write_symbol_prefix_stamp(object_path: str, prefix: str) -> None:
    """Record that ``prefix`` has been applied to the current object bytes."""
    stamp_path = _symbol_prefix_stamp_path(object_path, prefix)
    with _staged(stamp_path) as tmp:
        with open(tmp, "w") as f:
            json.dump(
                {
                    "version": _SYMBOL_PREFIX_STAMP_VERSION,
                    "prefix": prefix,
                    "object_sha256": _sha256_file(object_path),
                },
                f,
            )
        os.chmod(tmp, _DEFAULT_FILE_MODE)


@contextlib.contextmanager
def _staged(dest: str):
    """Yield a sibling temp path that replaces ``dest`` atomically on success.

    Kernels are grouped by ``_original_name``, but a source file is named after
    its own basename, so the several ExternalFunctions that share one .cc (only
    their -D flags differ) land in different groups and materialize the same
    path concurrently.  Writing in place truncates that file under a sibling
    compile: the reader either takes SIGBUS when the mapping shrinks beneath it,
    or sees a short prefix, compiles it clean because the missing part was
    behind an #ifdef, and emits an object with no symbol in it.

    Safe while every writer to one ``dest`` stages identical bytes: renaming
    makes the swap atomic, and a compile already holding the old inode keeps
    reading it until it unmaps.  Writers whose bytes differ have to be ordered
    instead; ``compile_external_kernels`` says which of those its grouping
    covers.
    """
    directory = os.path.dirname(dest) or "."
    fd, tmp = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(dest) + ".", suffix=".tmp"
    )
    os.close(fd)
    try:
        yield tmp
        _replace_staged_source(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _replace_staged_source(
    tmp: str, dest: str, *, replace=os.replace, is_windows: bool | None = None
):
    if is_windows is None:
        is_windows = os.name == "nt"
    try:
        replace(tmp, dest)
    except PermissionError:
        # Windows cannot replace a file another compile already has open.
        # When both writers staged identical bytes, the open destination is
        # already the source the later compile needs, so discard the temp
        # and let that compile proceed instead of failing the whole batch.
        if not is_windows:
            raise
        if not os.path.exists(dest) or not filecmp.cmp(tmp, dest, shallow=False):
            raise
        os.unlink(tmp)


def _write_source(dest: str, text: str) -> None:
    """Write ``text`` to ``dest`` without ever truncating it in place."""
    with _staged(dest) as tmp:
        with open(tmp, "w") as f:
            f.write(text)
        # mkstemp created the temp file at 0600; the in-place open() this
        # replaces left the source at the process umask.  copy2 carries the mode
        # over in _copy_source, so only this path has to restore it.
        os.chmod(tmp, _DEFAULT_FILE_MODE)


def _copy_source(dest: str, src: str) -> None:
    """Copy ``src`` onto ``dest`` without ever truncating ``dest`` in place."""
    with _staged(dest) as tmp:
        shutil.copy2(src, tmp)


def _copy_object_files(object_files, work_dir):
    """Stage explicit object files for relative link_with paths in aiecc's cwd."""
    for object_file in object_files:
        source = Path(object_file)
        dest = Path(work_dir) / source.name
        if dest.exists() and source.samefile(dest):
            continue
        _copy_source(str(dest), str(source))


def _compiled_into(func, kernel_dir, embed_bitcode=False) -> bool:
    """Report whether ``func``'s object was already built into this ``kernel_dir``.

    One ExternalFunction can be compiled by several designs, each into its own
    directory, so ``_compiled`` on its own would deny every design after the
    first an object.
    """
    compiled_dir = getattr(func, "_compiled_dir", None)
    if not getattr(func, "_compiled", False) or compiled_dir is None:
        return False
    if embed_bitcode and (
        not getattr(func, "_compiled_embed_bitcode", False)
        or not os.path.exists(os.path.join(kernel_dir, func.object_file_name))
    ):
        return False
    return os.path.abspath(compiled_dir) == os.path.abspath(kernel_dir)


def compile_external_kernels(
    funcs, kernel_dir, target_arch, include_dirs=None, embed_bitcode=False
):
    """Compile every ExternalFunction in ``funcs`` into ``kernel_dir``.

    Independent kernels compile concurrently. `_kernel_compile_groups` orders
    kernels sharing source names or object files to avoid staging and symbol
    renaming races. Source files with the same basename but different entry
    names must currently be supplied in separate batches.
    Set AIE_KERNEL_COMPILE_JOBS to override the default CPU-count job limit.
    """
    pending = [f for f in funcs if not _compiled_into(f, kernel_dir, embed_bitcode)]
    if not pending:
        return

    # Every compile in a batch shares one cwd (kernel_dir), and xchesscc keeps
    # per-invocation state there, so the Chess path runs serially.
    if any(getattr(f, "_use_chess", False) for f in pending):
        for f in pending:
            compile_external_kernel(
                f, kernel_dir, target_arch, include_dirs, embed_bitcode
            )
        return

    groups = _kernel_compile_groups(pending)

    try:
        jobs = int(os.environ.get("AIE_KERNEL_COMPILE_JOBS", "0"))
    except ValueError:
        jobs = 0
    if jobs <= 0:
        jobs = os.cpu_count() or 1
    jobs = min(jobs, len(groups))

    if jobs == 1:
        for group in groups:
            for f in group:
                compile_external_kernel(
                    f, kernel_dir, target_arch, include_dirs, embed_bitcode
                )
        return

    def _run(group):
        for f in group:
            compile_external_kernel(
                f, kernel_dir, target_arch, include_dirs, embed_bitcode
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        # list() re-raises the first failure, after the others have finished --
        # a compile error must not be swallowed by a sibling that succeeded.
        list(pool.map(_run, groups))


def _kernel_compile_groups(funcs):
    """Partition ``funcs`` into lists that must compile one after the other.

    Kernels sharing an ``_original_name`` or ``object_file_name`` are grouped
    transitively, preserving input order within each group.
    """
    parent = list(range(len(funcs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple, int] = {}
    for i, f in enumerate(funcs):
        for key in (
            ("name", getattr(f, "_original_name", f._name)),
            ("object", f.object_file_name),
        ):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups: dict[int, list] = {}
    for i, f in enumerate(funcs):
        groups.setdefault(find(i), []).append(f)
    return list(groups.values())


def compile_external_kernel(
    func, kernel_dir, target_arch, include_dirs=None, embed_bitcode=False
):
    """Compile an ExternalFunction to an object file in the kernel directory.

    The output file is named ``func.object_file_name`` and placed in ``kernel_dir``.
    Existing objects are reused, except that prefixed objects require a matching
    content stamp. Unstamped or modified prefixed objects are rebuilt from source:
    their symbol names cannot establish whether prefixing has already happened.
    A cached object is also rejected when ``embed_bitcode`` requests IR retention
    and the object has no ``.llvmbc`` section.

    Args:
        func: ExternalFunction instance to compile.
        kernel_dir: Directory where the compiled object file will be placed.
            Must be the same directory passed as ``work_dir`` to
            ``compile_mlir_module`` so that relative link_with paths resolve
            correctly.
        target_arch: Peano target architecture string (e.g., "aie2", "aie2p").
        include_dirs: Design-wide include directories appended after the
            ExternalFunction's own include directories.
        embed_bitcode: Preserve Peano kernel LLVM IR for ``--check-lut-banks``.
    """
    if embed_bitcode and getattr(func, "_use_chess", False):
        raise ValueError("--check-lut-banks requires Peano kernels, not Chess")
    if _compiled_into(func, kernel_dir, embed_bitcode):
        return

    # inline + symbol_prefix is unsupported: the MLIR func.call uses the
    # prefixed func._name, but an inline kernel is emitted as a textual .ll whose
    # ``define`` carries the un-prefixed _original_name. Object mode reconciles
    # the two via an llvm-objcopy --redefine-syms rename, which cannot rewrite a
    # .ll. Fail loudly here rather than downstream in objcopy or as a silent
    # call/define name mismatch at llvm-link time.
    if getattr(func, "_inline", False) and getattr(func, "_symbol_prefix", None):
        raise NotImplementedError(
            f"ExternalFunction '{func._name}': inline=True combined with "
            "symbol_prefix is not supported (an inline kernel is emitted as "
            "LLVM IR and cannot be symbol-renamed). Use inline without a "
            "symbol_prefix, or drop inline for this kernel."
        )

    # A missing/stale stamp can mean either a legacy cache entry or an interrupted
    # prefix pass. Never rename those bytes again; rebuild from source instead.
    output_file = os.path.join(kernel_dir, func.object_file_name)
    prefix = (
        f"{func._symbol_prefix}_" if getattr(func, "_symbol_prefix", None) else None
    )
    # The bitcode test inspects the object itself, not its .bc sidecar, which can
    # survive a failed attach. An object built before bitcode was requested
    # cannot serve a request that needs it.
    if (
        os.path.exists(output_file)
        and (prefix is None or _has_current_symbol_prefix_stamp(output_file, prefix))
        and (
            not embed_bitcode
            or getattr(func, "_inline", False)
            or _object_has_bitcode(output_file)
        )
    ):
        # Same three fields the post-compile tail sets, so a cache hit and a
        # fresh build leave the function in the same state.
        func._compiled = True
        func._compiled_dir = os.path.abspath(kernel_dir)
        func._compiled_embed_bitcode = embed_bitcode
        return

    # Invalidate before any writes, so a failed compile, rename, or stamp write
    # cannot leave a cache entry that a later invocation trusts.
    if prefix is not None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(_symbol_prefix_stamp_path(output_file, prefix))

    if func._source_string is not None:
        original_name = getattr(func, "_original_name", func._name)
        source_file = os.path.join(kernel_dir, f"{original_name}.cc")
        _write_source(source_file, func._source_string)
        compile_cxx_core_function(
            source_path=source_file,
            target_arch=target_arch,
            output_path=output_file,
            # The source is compiled under _original_name, so that is the symbol
            # in the emitted .ll ``define`` that _make_ir_inlinable must rewrite.
            # (inline + symbol_prefix is rejected above, so no rename applies.)
            symbol_name=func._original_name,
            include_dirs=[*func._include_dirs, *(include_dirs or ())],
            compile_args=func._compile_flags,
            cwd=str(kernel_dir),
            inline=getattr(func, "_inline", False),
            use_chess=getattr(func, "_use_chess", False),
            embed_bitcode=embed_bitcode,
        )

    elif func._source_file is not None:
        # Named after the real source basename, not the entry point: entry
        # points sharing one object_file_name compile only on the first
        # visit, and `_instances` iteration order (a content-hashed set)
        # shifts whenever any registered kernel's content changes.
        source_file = os.path.join(kernel_dir, os.path.basename(func._source_file))
        # Check if source file exists before copying
        if not os.path.exists(func._source_file):
            raise FileNotFoundError(
                f"ExternalFunction '{func._name}': source file not found: {func._source_file}"
            )
        # realpath, not abspath: the rename in _staged would happily overwrite
        # the kernel's own source with a copy of itself if kernel_dir reaches it
        # through a symlink, where copy2 used to raise SameFileError.
        if os.path.realpath(source_file) != os.path.realpath(func._source_file):
            _copy_source(source_file, func._source_file)
        # Include the original source file's directory so relative includes
        # (e.g. "../aie_kernel_utils.h") still resolve after the file is
        # copied into kernel_dir.
        src_dir = os.path.dirname(os.path.abspath(func._source_file))
        kernel_include_dirs = list(func._include_dirs)
        if src_dir not in kernel_include_dirs:
            kernel_include_dirs.append(src_dir)
        kernel_include_dirs.extend(include_dirs or ())
        compile_cxx_core_function(
            source_path=source_file,
            target_arch=target_arch,
            output_path=output_file,
            # _original_name is the symbol in the emitted .ll ``define`` (see
            # the source_string branch above).
            symbol_name=func._original_name,
            include_dirs=kernel_include_dirs,
            compile_args=func._compile_flags,
            cwd=kernel_dir,
            inline=getattr(func, "_inline", False),
            use_chess=getattr(func, "_use_chess", False),
            embed_bitcode=embed_bitcode,
        )
    else:
        raise ValueError("Neither source_string nor source_file is provided")

    # Prefix every defined symbol in the object if a prefix is set. This covers
    # not just the entry point (func._name is already "{symbol_prefix}_{original}")
    # but any other extern "C" helper symbols the kernel source happens to define,
    # so multiple memoized instantiations of the same source can be linked
    # together without their helpers colliding too.
    if prefix is not None:
        prefix_symbols_in_object(output_file, prefix)
        _write_symbol_prefix_stamp(output_file, prefix)

    func._compiled = True
    func._compiled_dir = os.path.abspath(kernel_dir)
    func._compiled_embed_bitcode = embed_bitcode


def _is_dispatch_library_name(name: str) -> bool:
    return re.fullmatch(r"dispatch-[0-9a-f]{64}\.(?:so|dll)", name) is not None


def _cleanup_failed_compilation(cache_dir):
    """Clean up cache directory after failed compilation.

    Preserves the lock file and, when present, the ``repeater`` reproducer dir
    that aiecc's ``--enable-repeater-scripts`` writes. Published dispatch
    generations are retained cache artifacts, not temporary staging files:
    a caller can still hold their path without having loaded it yet, so they
    stay until cache eviction.
    """
    if not os.path.exists(cache_dir):
        return

    for item in os.listdir(cache_dir):
        if item in (".lock", "repeater") or _is_dispatch_library_name(item):
            continue
        item_path = os.path.join(cache_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
