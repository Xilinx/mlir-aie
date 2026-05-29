#!/usr/bin/env bash
# PyPI publish rehearsal: strip a copy of each wheel, run the same metadata
# and size checks the real upload would, but never push to PyPI. Lets every
# build report what a stripped PyPI-bound wheel would look like without
# committing to publication.
#
# Usage:  pypi_rehearsal.sh <input_wheel_dir> <output_stripped_dir> [<label>]
#
# Writes a markdown summary table to $GITHUB_STEP_SUMMARY. Exits 0 on
# success; exits non-zero only on twine-check failure (size-cap excess is
# flagged in the summary but does not fail the build, so we accumulate
# data before deciding whether to ask PyPI for a per-project cap raise).

set -euo pipefail

WHEELS_IN=${1:?input wheel dir}
WHEELS_OUT=${2:?output stripped wheel dir}
LABEL=${3:-rehearsal}

PYPI_DEFAULT_CAP=$((100 * 1024 * 1024))
PYPI_CLOSE_THRESH=$((80 * 1024 * 1024))

mkdir -p "$WHEELS_OUT"

twine_failed=0

{
  echo "## PyPI rehearsal — $LABEL"
  echo ""
  echo "| wheel | unstripped | stripped | reduction | twine | vs 100 MB cap |"
  echo "|---|---:|---:|---:|---|---|"
} >> "$GITHUB_STEP_SUMMARY"

for whl in "$WHEELS_IN"/*.whl; do
  [ -e "$whl" ] || { echo "no wheels in $WHEELS_IN"; exit 0; }
  name=$(basename "$whl")
  orig_bytes=$(stat -c%s "$whl")

  workdir=$(mktemp -d)
  python -m wheel unpack -d "$workdir" "$whl" >/dev/null
  unpacked=$(ls "$workdir")

  # Shared libs (ELF .so / PE .pyd / PE .dll): --strip-unneeded preserves
  # the dynamic symbol table that dlopen / runtime linker walks; without it,
  # importing the .so would fail symbol lookup.
  while IFS= read -r -d '' f; do
    strip --strip-unneeded "$f" 2>/dev/null || true
  done < <(find "$workdir/$unpacked" -type f \
            \( -name '*.so' -o -name '*.so.*' \
               -o -name '*.pyd' -o -name '*.dll' \) -print0)

  # Tool executables (bin/ + *.exe): --strip-all is safe because nothing
  # dlopens these at runtime; aiecc shells out to them as separate processes.
  # Saves another ~30% over --strip-unneeded.
  while IFS= read -r -d '' f; do
    strip --strip-all "$f" 2>/dev/null || true
  done < <(find "$workdir/$unpacked" -type f \
            \( -path '*/bin/*' -o -name '*.exe' \) -print0)

  python -m wheel pack --dest "$WHEELS_OUT" "$workdir/$unpacked" >/dev/null
  rm -rf "$workdir"

  # `wheel pack` may rename if build tag changes; pick up whatever landed
  out_whl=$(ls -t "$WHEELS_OUT"/*.whl | head -1)
  strip_bytes=$(stat -c%s "$out_whl")
  reduction=$(( orig_bytes > 0 ? 100 - (strip_bytes * 100 / orig_bytes) : 0 ))

  orig_mb=$(awk "BEGIN { printf \"%.1f\", $orig_bytes / 1024 / 1024 }")
  strip_mb=$(awk "BEGIN { printf \"%.1f\", $strip_bytes / 1024 / 1024 }")

  if [ "$strip_bytes" -gt "$PYPI_DEFAULT_CAP" ]; then
    cap_status='over (size-increase request needed)'
  elif [ "$strip_bytes" -gt "$PYPI_CLOSE_THRESH" ]; then
    cap_status='under, within 20 MB'
  else
    cap_status='under'
  fi

  if python -m twine check --strict "$out_whl" > /tmp/twine.out 2>&1; then
    twine_status='ok'
  else
    twine_status='FAIL'
    twine_failed=1
    cat /tmp/twine.out
  fi

  echo "| \`$name\` | ${orig_mb} MB | ${strip_mb} MB | -${reduction}% | $twine_status | $cap_status |" \
    >> "$GITHUB_STEP_SUMMARY"
done

echo "" >> "$GITHUB_STEP_SUMMARY"
echo "_Stripped variants are produced only to measure size and validate \`twine check\`; the GitHub release continues to receive the unstripped wheels._" \
  >> "$GITHUB_STEP_SUMMARY"

exit "$twine_failed"
