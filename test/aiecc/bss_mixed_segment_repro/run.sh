#!/bin/bash
# Minimal reproducer for the aie-rt ELF loader out-of-bounds read.
#
# Builds a one-core design twice and inspects the generated CDO. No hardware
# required -- the defect is visible in the configuration image itself.
#
#   BUG variant     kernel has .data AND .bss -> one PT_LOAD with
#                   0 < p_filesz < p_memsz. aie-rt writes p_memsz bytes from a
#                   p_filesz-sized buffer, so ELF .comment bytes land in .bss.
#   CONTROL variant kernel has .bss only -> separate PT_LOAD with p_filesz == 0,
#                   which aie-rt handles correctly (its calloc path).
#
# Usage: ./run.sh [workdir]     (needs PEANO_INSTALL_DIR and aiecc on PATH)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:-/tmp/aie-rt-bss-repro}"
: "${PEANO_INSTALL_DIR:?set PEANO_INSTALL_DIR}"
AIECC="${AIECC:-$(command -v aiecc)}"
CXX="$PEANO_INSTALL_DIR/bin/clang++"
RE="$PEANO_INSTALL_DIR/bin/llvm-readelf"
mkdir -p "$W"; cd "$W"; cp "$HERE/bssmin.mlir" .

run_variant () { # $1=source $2=tag
  "$CXX" --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -c "$HERE/$1" -o bss_k.o
  rm -rf "work_$2"; mkdir -p "work_$2"
  "$AIECC" bssmin.mlir --peano="$PEANO_INSTALL_DIR" \
      --get-xclbin --xclbin-name="$W/$2.xclbin" \
      --get-npu-insts --npu-insts-name="$W/$2.insts.bin" \
      --tmpdir="$W/work_$2" --output-dir="$W/work_$2" >/dev/null 2>&1
  local elf="work_$2/elfs_main_core_0_2/elfs_main_core_0_2.elf"
  local cdo="work_$2/cdo_main/main_aie_cdo_elfs.bin"
  echo "--- $2 ---"
  "$RE" -lW "$elf" | awk '/LOAD/ && $7=="RW" {printf "    segment: FileSiz %s  MemSiz %s\n", $5, $6}'
  local n; n=$(grep -c -a "Linker: LLD" "$cdo" 2>/dev/null || true)
  echo "    ELF .comment strings baked into the CDO: ${n:-0}"
  if [ "${n:-0}" != "0" ]; then
    echo "    >>> BUG: the configuration image carries ELF metadata that will be"
    echo "        DMA'd into the tile where zero-initialised statics belong."
  else
    echo "    OK: no metadata leaked."
  fi
}

run_variant bss_k.cc         bug
run_variant bss_k_control.cc control
