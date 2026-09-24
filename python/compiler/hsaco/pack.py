# pack.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

"""Inject an AIE section into an hsaco.

The section is a versioned header, a kernel table, a string table and a blob
pool; see :mod:`hsaco.format` for the layout. Kernels come from one of three
input forms -- a PDI plus an instruction stream, an xclbin plus an instruction
stream, or a self-contained full ELF.
"""

import argparse
import glob
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import Counter

from .elf import kernel_names_from_full_elf, make_empty_elf64
from .format import (
    ARCHES,
    ENTRY,
    ENTRY_SIZE,
    HDR,
    HDR_SIZE,
    KIND_COUNT,
    KIND_FULL_ELF,
    KIND_PDI_INSTS,
    MAGIC,
    VERSION_MAJOR,
    VERSION_MINOR,
)


def _executable_name(name):
    return f"{name}.exe" if os.name == "nt" else name


def _bundled_tool(name):
    """Return the MLIR-AIE bin/ copy of ``name``, or ``None``.

    Resolved lazily and tolerantly: ``aie.utils.config`` pulls in the compiled
    bindings, which this packer does not otherwise need. A source checkout with
    no build should still be able to pack bytes using tools from PATH.
    """
    try:
        import aie.utils.config as config

        # root_path() raises when there is no install tree, which is exactly
        # the case this falls back from.
        candidate = os.path.join(config.root_path(), "bin", _executable_name(name))
    except (ImportError, RuntimeError):
        return None
    return candidate if os.path.isfile(candidate) else None


def objcopy_path():
    """Return the ``llvm-objcopy`` used to add the arch section.

    Prefers ``aie.utils.config.objcopy_path`` (which honours ``AIE_OBJCOPY_PATH``
    and the wheel-bundled copy), then falls back to PATH.
    """
    try:
        import aie.utils.config as config
    except ImportError:
        # Only an unimportable config falls through to PATH. A resolution
        # failure from config itself must propagate: its message names every
        # directory it searched, which is far more use than ours.
        pass
    else:
        return config.objcopy_path()
    found = shutil.which("llvm-objcopy")
    if found:
        return found
    raise RuntimeError(
        "Could not find llvm-objcopy. Set AIE_OBJCOPY_PATH, or put it on PATH."
    )


def xclbinutil_path():
    """Return the ``xclbinutil`` used to read a PDI out of an xclbin.

    Resolution order, matching how ``aie.utils.config`` resolves its own tools:
    ``AIE_XCLBINUTIL_PATH``, then the copy mlir-aie installs next to aiecc
    (built from ``tools/hrx-xclbinutil``, which needs no system XRT), then PATH.
    """
    override = os.environ.get("AIE_XCLBINUTIL_PATH")
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(
                f"AIE_XCLBINUTIL_PATH is set to {override}, but no such file exists."
            )
        return override
    bundled = _bundled_tool("xclbinutil")
    if bundled:
        return bundled
    # shutil.which applies PATHEXT itself, so no .exe suffix is needed here.
    found = shutil.which("xclbinutil")
    if found:
        return found
    raise RuntimeError(
        "xclbinutil not found. It is needed to read a PDI out of an xclbin; build "
        "mlir-aie with -DAIE_BUILD_HRXXCLBINUTIL=ON, install XRT, set "
        "AIE_XCLBINUTIL_PATH, or pass the PDI directly with the PDI+insts "
        "--kernel form."
    )


def _uint32(value, field, kernel_name):
    """Return ``value`` as a uint32, or raise ValueError naming the field."""
    number = int(value)
    if not 0 <= number <= 0xFFFFFFFF:
        raise ValueError(
            f"kernel {kernel_name!r}: {field} {number} out of range for a uint32"
        )
    return number


def build_section(arch, kernels):
    """Return the packed arch section for ``kernels``.

    Args:
        arch (str): One of :data:`hsaco.format.ARCHES`.
        kernels (list[dict]): Kernel descriptors with ``name`` and ``insts``,
            optionally ``pdi``, ``kernarg_size``, ``num_cols`` and ``kind``.

    Returns:
        bytes: The section, laid out as header, kernel table, string table, blob pool.

    Raises:
        ValueError: On an unknown arch or kind, empty insts, or a FullElf entry
            that also carries a PDI.
    """
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHES}")

    # ROCR resolves kernels by name, so a duplicate would make one of the two
    # entries permanently unreachable. Reachable without user error: two
    # `elf:` specs can share a kernel:instance pair.
    counts = Counter(k["name"] for k in kernels)
    repeated = sorted(name for name, n in counts.items() if n > 1)
    if repeated:
        raise ValueError(f"duplicate kernel name(s): {', '.join(repeated)}")

    string_table = bytearray()
    name_offsets = []
    for k in kernels:
        name_offsets.append(len(string_table))
        string_table += k["name"].encode() + b"\x00"

    # Blob pool with dedup keyed on raw bytes. Full-ELF entries all embed the
    # same image, so this is what keeps an N-kernel ELF from being stored N times.
    # Blobs are held as a list and joined once at the end rather than
    # accumulated into a buffer: a full ELF is the bulk of the section, and
    # concatenating as we go would copy it an extra time.
    blobs = []
    pool_size = 0
    blob_off = {}

    def place(blob):
        nonlocal pool_size
        if not blob:
            return (0, 0)
        key = bytes(blob)
        if key not in blob_off:
            blob_off[key] = pool_size
            blobs.append(key)
            pool_size += len(key)
        return (blob_off[key], len(key))

    # Layout: [header][kernel table][string table][blob pool]
    header_size = HDR_SIZE
    table_size = ENTRY_SIZE * len(kernels)
    string_table_offset = header_size + table_size
    blob_pool_offset = string_table_offset + len(string_table)

    entries = []
    for k, name_off in zip(kernels, name_offsets):
        insts_off, insts_size = place(k["insts"])
        if insts_size == 0:
            raise ValueError(f"kernel {k['name']!r}: insts must be non-empty")
        pdi_off, pdi_size = place(k.get("pdi"))
        kind = int(k.get("kind", KIND_PDI_INSTS))
        # Bounded by KIND_COUNT, the same rule dump.parse_section applies, so
        # adding a kind to format.py stays a one-file edit.
        if not 0 <= kind < KIND_COUNT:
            raise ValueError(f"kernel {k['name']!r}: unknown kind {kind}")
        if kind == KIND_FULL_ELF and pdi_size:
            raise ValueError(
                f"kernel {k['name']!r}: FullElf entries carry no separate PDI"
            )
        # These land in uint32 fields. Checked here so that a negative or
        # oversized value -- both reachable from the command line -- is a
        # ValueError like every other bad-kernel condition, not a struct.error.
        kernarg_size = _uint32(k.get("kernarg_size", 0), "kernarg_size", k["name"])
        num_cols = _uint32(k.get("num_cols", 1), "num_cols", k["name"])
        entries.append(
            (
                name_off,
                blob_pool_offset + insts_off,
                insts_size,
                (blob_pool_offset + pdi_off) if pdi_size else 0,
                pdi_size,
                kernarg_size,
                num_cols,
                kind,
                0,
                0,
                0,
            )
        )

    header = struct.pack(
        HDR,
        MAGIC,
        VERSION_MAJOR,
        VERSION_MINOR,
        header_size,
        len(kernels),
        ENTRY_SIZE,
        string_table_offset,
        len(string_table),
        blob_pool_offset,
        0,
        0,
        0,
        0,
    )
    return b"".join(
        [
            header,
            *(struct.pack(ENTRY, *e) for e in entries),
            bytes(string_table),
            *blobs,
        ]
    )


def inject(hsaco_path, arch, section_bytes):
    """Add ``section_bytes`` to the hsaco as a section named ``arch``.

    An existing same-named section is removed first, so repacking replaces
    rather than duplicates.

    objcopy reads the original and writes a scratch file, which replaces the
    original only on success -- so the hsaco is never opened for writing and a
    failure leaves it exactly as it was. objcopy applies --remove-section
    before --add-section within one invocation, so replacing a section takes a
    single pass over the file rather than three.

    Raises:
        RuntimeError: If objcopy fails, carrying its stderr.
    """
    objcopy = objcopy_path()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(section_bytes)
        sec_file = f.name
    # Unique per call, not per process: two threads packing different arches
    # into one hsaco would otherwise share a scratch name and clobber each
    # other. Same directory, which os.replace requires.
    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(os.path.abspath(hsaco_path)),
        prefix=os.path.basename(hsaco_path) + ".",
        suffix=".tmp",
        delete=False,
    ) as f:
        scratch = f.name
    try:
        subprocess.run(
            [
                objcopy,
                # A no-op, exiting 0, when the section is not already present.
                f"--remove-section={arch}",
                f"--add-section={arch}={sec_file}",
                f"--set-section-flags={arch}=noload,readonly",
                hsaco_path,
                scratch,
            ],
            check=True,
            capture_output=True,
        )
        os.replace(scratch, hsaco_path)
    except subprocess.CalledProcessError as e:
        # capture_output keeps objcopy's diagnosis off the user's terminal;
        # without this it is captured and then thrown away with the exception.
        detail = (e.stderr or b"").decode(errors="replace").strip()
        raise RuntimeError(
            f"llvm-objcopy failed to write the {arch} section"
            + (f": {detail}" if detail else "")
        ) from e
    finally:
        os.unlink(sec_file)
        if os.path.exists(scratch):
            os.unlink(scratch)


def ensure_hsaco(path):
    """Create a minimal hsaco at ``path`` if it does not already exist."""
    if os.path.exists(path):
        return
    with open(path, "wb") as f:
        f.write(make_empty_elf64())


def kernels_from_full_elf(path, kernarg_size=0, num_cols=1):
    """Return one kernel descriptor per COMDAT group in a full ELF.

    Every descriptor embeds the whole ELF as its ``insts``; the blob pool in
    :func:`build_section` stores it once.

    Args:
        path (str): Path to the full ELF.
        kernarg_size (int): Kernarg buffer size to record for each kernel.
        num_cols (int): Column count to record for each kernel.

    Returns:
        list[dict]: Kernel descriptors of kind ``FullElf``.
    """
    with open(path, "rb") as f:
        blob = f.read()
    try:
        names = kernel_names_from_full_elf(blob)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    return [
        {
            "name": n,
            "insts": blob,
            "pdi": None,
            "kernarg_size": kernarg_size,
            "num_cols": num_cols,
            "kind": KIND_FULL_ELF,
        }
        for n in names
    ]


def pdi_from_xclbin(path):
    """Extract the PDI from an xclbin's AIE_PARTITION section via xclbinutil.

    Args:
        path (str): Path to the xclbin.

    Returns:
        bytes: The PDI image.

    Raises:
        ValueError: If the xclbin does not contain exactly one PDI.
    """
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(
            [
                xclbinutil_path(),
                "--input",
                path,
                "--dump-section",
                f"AIE_PARTITION:JSON:{os.path.join(d, 'aie.json')}",
            ],
            check=True,
            capture_output=True,
        )
        pdis = glob.glob(f"{d}/**/*.pdi", recursive=True)
        if len(pdis) != 1:
            raise ValueError(f"{path}: expected exactly one PDI, found {len(pdis)}")
        with open(pdis[0], "rb") as f:
            return f.read()


def _read(path):
    with open(path, "rb") as f:
        return f.read()


# --- long-form kernel options -------------------------------------------------
#
# The colon-separated --kernel grammar is inherited from ROCR's packer and has
# no escaping, so any path containing a colon -- every absolute Windows path --
# misparses into the wrong fields. These options say the same things without a
# delimiter. A kernel starts at --kernel-name or --kernel-elf and absorbs the
# --kernel-* options that follow it, so they can be repeated for several
# kernels, and mixed with --kernel.

# Options that begin a new kernel, one per input form.
_KERNEL_STARTERS = ("kernel_name", "kernel_elf")
# A full ELF is self-contained, so none of these can accompany --kernel-elf.
_ELF_INCOMPATIBLE = ("kernel_insts", "kernel_pdi", "kernel_xclbin")
# Options whose value is a count rather than a path.
_INT_OPTIONS = ("kernel_kernarg", "kernel_cols")


class _KernelOption(argparse.Action):
    """Record a ``--kernel-*`` option, preserving the order it was given in.

    Grouping is positional -- an attribute belongs to the most recent starter --
    so argparse's usual per-dest accumulation would lose the association.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        # Created lazily rather than via set_defaults: a list default would be
        # shared across parse_args() calls in one process and accumulate.
        if getattr(namespace, "kernel_ops", None) is None:
            namespace.kernel_ops = []
        namespace.kernel_ops.append((self.dest, values))


def _flag(dest):
    return "--" + dest.replace("_", "-")


def _group_kernel_options(ops):
    """Return one item per kernel spec, in the order the options were given.

    Each item is either ``("spec", [descriptors])`` for a ready ``--kernel``
    spec, or ``("group", {dest: value})`` for a long-form group. Both grammars
    travel in one stream so the kernel table ends up in argv order.
    """
    items = []
    for dest, value in ops:
        if dest == "kernel":
            items.append(("spec", value))
            continue
        if dest in _KERNEL_STARTERS:
            items.append(("group", {dest: value}))
            continue
        if not items or items[-1][0] != "group":
            raise argparse.ArgumentTypeError(
                f"{_flag(dest)} must follow {_flag(_KERNEL_STARTERS[0])} or "
                f"{_flag(_KERNEL_STARTERS[1])}"
            )
        group = items[-1][1]
        if dest in group:
            raise argparse.ArgumentTypeError(
                f"{_flag(dest)} given twice for the same kernel"
            )
        group[dest] = value
    return items


def _kernel_from_options(group):
    """Return the kernel descriptors for one group of ``--kernel-*`` options."""
    kernarg_size = group.get("kernel_kernarg", 0)
    num_cols = group.get("kernel_cols", 1)

    if "kernel_elf" in group:
        clashes = [_flag(d) for d in _ELF_INCOMPATIBLE if d in group]
        if clashes:
            raise argparse.ArgumentTypeError(
                f"{_flag('kernel_elf')} is self-contained and cannot be combined "
                f"with {', '.join(clashes)}"
            )
        return kernels_from_full_elf(group["kernel_elf"], kernarg_size, num_cols)

    name = group["kernel_name"]
    if "kernel_insts" not in group:
        raise argparse.ArgumentTypeError(
            f"{_flag('kernel_name')} {name!r} requires {_flag('kernel_insts')}"
        )
    if "kernel_pdi" in group and "kernel_xclbin" in group:
        raise argparse.ArgumentTypeError(
            f"kernel {name!r}: give {_flag('kernel_pdi')} or "
            f"{_flag('kernel_xclbin')}, not both"
        )
    if "kernel_pdi" in group:
        pdi = _read(group["kernel_pdi"])
    elif "kernel_xclbin" in group:
        pdi = pdi_from_xclbin(group["kernel_xclbin"])
    else:
        pdi = None
    return [
        {
            "name": name,
            "insts": _read(group["kernel_insts"]),
            "pdi": pdi,
            "kernarg_size": kernarg_size,
            "num_cols": num_cols,
            "kind": KIND_PDI_INSTS,
        }
    ]


def kernels_from_options(ops):
    """Return every kernel described by the recorded ``--kernel-*`` options.

    Args:
        ops (list[tuple[str, object]]): ``(dest, value)`` pairs in argv order.

    Returns:
        list[dict]: Kernel descriptors.

    Raises:
        argparse.ArgumentTypeError: On a malformed combination, a file that
            cannot be read, or an input that is not what it claims to be.
    """
    kernels = []
    for kind, value in _group_kernel_options(ops):
        if kind == "spec":
            kernels.extend(value)
            continue
        try:
            kernels.extend(_kernel_from_options(value))
        except (OSError, ValueError, subprocess.CalledProcessError) as e:
            raise argparse.ArgumentTypeError(str(e)) from e
    return kernels


def parse_kernel_arg(s):
    """Return the kernel descriptors named by one ``--kernel`` argument.

    Three forms are accepted::

        elf:PATH[:KERNARG_SIZE[:NUM_COLS]]
        xclbin:NAME:XCLBIN:INSTS:KERNARG_SIZE:NUM_COLS
        NAME:INSTS[:PDI]:KERNARG_SIZE:NUM_COLS

    Args:
        s (str): The raw argument.

    Returns:
        list[dict]: One or more kernel descriptors.

    Raises:
        argparse.ArgumentTypeError: If the spec has the wrong number of fields,
            names a file that cannot be read, or names one that is not what it
            claims to be. Everything is funnelled into this one type because
            argparse handles the alternatives badly: an OSError from a
            ``type=`` callable escapes as a raw traceback, and a ValueError is
            caught but its message discarded in favour of a generic "invalid
            value" -- losing the only text that says what was actually wrong.
    """
    try:
        return _kernel_from_options(_kernel_options_from_spec(s))
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        raise argparse.ArgumentTypeError(f"--kernel {s!r}: {e}") from e


def _kernel_options_from_spec(s):
    """Return the ``--kernel-*`` option group a colon-separated spec denotes.

    Splitting fields is all this does. Defaults, file reads and the rules about
    which fields may appear together live in :func:`_kernel_from_options`, so
    the two spellings cannot drift apart.
    """
    parts = s.split(":")
    if parts[0] == "elf":
        if not 2 <= len(parts) <= 4:
            raise argparse.ArgumentTypeError(f"bad --kernel spec {s!r}")
        group = {"kernel_elf": parts[1]}
        fields = zip(("kernel_kernarg", "kernel_cols"), parts[2:])
    elif parts[0] == "xclbin":
        if len(parts) != 6:
            raise argparse.ArgumentTypeError(f"bad --kernel spec {s!r}")
        group = {"kernel_name": parts[1], "kernel_xclbin": parts[2]}
        fields = zip(("kernel_insts", "kernel_kernarg", "kernel_cols"), parts[3:])
    elif len(parts) == 4:
        group = {"kernel_name": parts[0], "kernel_insts": parts[1]}
        fields = zip(("kernel_kernarg", "kernel_cols"), parts[2:])
    elif len(parts) == 5:
        group = {"kernel_name": parts[0], "kernel_insts": parts[1]}
        fields = zip(
            ("kernel_pdi", "kernel_kernarg", "kernel_cols"),
            parts[2:],
        )
    else:
        raise argparse.ArgumentTypeError(f"bad --kernel spec {s!r}")

    for dest, value in fields:
        group[dest] = int(value) if dest in _INT_OPTIONS else value
    return group


def main(argv=None):
    """Pack the kernels named on the command line into an hsaco."""
    ap = argparse.ArgumentParser(
        prog="aie-hsaco",
        description="Inject an aie2/aie2p kernel section into an hsaco.",
    )
    ap.add_argument("--hsaco", required=True, help="hsaco to write; created if absent")
    ap.add_argument("--arch", required=True, choices=ARCHES)
    ap.add_argument(
        "--kernel",
        action=_KernelOption,
        type=parse_kernel_arg,
        help="colon-separated kernel spec; repeatable. One of "
        "NAME:INSTS[:PDI]:KERNARG_SIZE:NUM_COLS, "
        "xclbin:NAME:XCLBIN:INSTS:KERNARG_SIZE:NUM_COLS, or "
        "elf:PATH[:KERNARG_SIZE[:NUM_COLS]]. Cannot express a path containing "
        "a colon, or a kernel named 'elf' or 'xclbin' -- use the --kernel-* "
        "options for those.",
    )
    group = ap.add_argument_group(
        "long-form kernel options",
        "Delimiter-free equivalents of --kernel, safe for paths containing a "
        "colon. A kernel starts at --kernel-name or --kernel-elf and takes the "
        "--kernel-* options that follow it; repeat to describe several.",
    )
    group.add_argument(
        "--kernel-name", action=_KernelOption, help="start a PDI+insts kernel"
    )
    group.add_argument(
        "--kernel-elf", action=_KernelOption, help="start a self-contained full ELF"
    )
    group.add_argument(
        "--kernel-insts", action=_KernelOption, help="instruction stream (insts.bin)"
    )
    group.add_argument("--kernel-pdi", action=_KernelOption, help="PDI (main.pdi)")
    group.add_argument(
        "--kernel-xclbin", action=_KernelOption, help="xclbin to extract the PDI from"
    )
    group.add_argument(
        "--kernel-kernarg",
        type=int,
        action=_KernelOption,
        help="kernarg buffer size (default 0)",
    )
    group.add_argument(
        "--kernel-cols", type=int, action=_KernelOption, help="column count (default 1)"
    )

    args = ap.parse_args(argv)
    try:
        # Both grammars come out of one ordered stream, so the kernel table
        # lands in the order they were written on the command line.
        kernels = kernels_from_options(getattr(args, "kernel_ops", None) or [])
    except argparse.ArgumentTypeError as e:
        ap.error(str(e))
    if not kernels:
        ap.error("no kernels given; use --kernel, --kernel-name or --kernel-elf")
    # Build before creating anything on disk, so a rejected set of kernels does
    # not leave an empty container behind for the next run to adopt. A rejected
    # set is bad input like any other, so it exits through ap.error rather than
    # as a traceback -- the file-not-found path already does.
    try:
        section = build_section(args.arch, kernels)
    except ValueError as e:
        ap.error(str(e))

    created = not os.path.exists(args.hsaco)
    ensure_hsaco(args.hsaco)
    try:
        inject(args.hsaco, args.arch, section)
    except RuntimeError as e:
        # Same invariant as the build failure above: leave nothing behind that
        # a later run would mistake for a real container.
        if created:
            os.unlink(args.hsaco)
        ap.error(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
