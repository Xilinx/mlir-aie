/* Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises pdi_transform() on a real PDI and checks the output byte-for-byte.
 *
 * This is the only test that reaches aie-pdi-transform. The tool's own CLI path
 * to pdi_transform (--transform-pdi) is `#ifndef _WIN32` in
 * XclBinUtilities.cxx, but the library itself is built on every platform -- so
 * linking it directly is what gives the PDI header structs (XilPdi_ImgHdrTbl /
 * XilPdi_ImgHdr / XilPdi_PrtnHdr, whose layout is asserted in load_pdi.h)
 * runtime coverage under MSVC cl as well as gcc/clang.
 *
 * usage: pdi_transform_test <input.pdi> <expected.pdi> <scratch-out.pdi>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Defined in libtransformcdo; takes non-const char* for the two paths. */
extern int pdi_transform(char *pdi_file, char *pdi_file_out,
                         const char *out_file);

static unsigned char *slurp(const char *path, long *size) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long n = ftell(f);
  if (n < 0 || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return NULL;
  }
  unsigned char *buf = (unsigned char *)malloc((size_t)n);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    free(buf);
    return NULL;
  }
  fclose(f);
  *size = n;
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <input.pdi> <expected.pdi> <scratch-out.pdi>\n",
            argv[0]);
    return 2;
  }

  /* pdi_transform's log goes to stdout when the third argument is empty. */
  char in[4096], out[4096];
  snprintf(in, sizeof(in), "%s", argv[1]);
  snprintf(out, sizeof(out), "%s", argv[3]);

  int rc = pdi_transform(in, out, "");
  if (rc != 0) {
    fprintf(stderr, "FAIL: pdi_transform returned %d\n", rc);
    return 1;
  }

  long got_n = 0, want_n = 0;
  unsigned char *got = slurp(argv[3], &got_n);
  unsigned char *want = slurp(argv[2], &want_n);
  if (!got) {
    fprintf(stderr, "FAIL: no output at %s\n", argv[3]);
    return 1;
  }
  if (!want) {
    fprintf(stderr, "FAIL: cannot read expected %s\n", argv[2]);
    free(got);
    return 1;
  }

  int ok = 1;
  if (got_n != want_n) {
    fprintf(stderr, "FAIL: size %ld, expected %ld\n", got_n, want_n);
    ok = 0;
  } else if (memcmp(got, want, (size_t)got_n) != 0) {
    long i = 0;
    while (i < got_n && got[i] == want[i])
      ++i;
    fprintf(stderr,
            "FAIL: first differing byte at offset %ld (0x%02x != 0x%02x)\n", i,
            got[i], want[i]);
    ok = 0;
  }

  if (ok)
    printf("PASS: transformed PDI matches byte-for-byte (%ld bytes)\n", got_n);
  free(got);
  free(want);
  return ok ? 0 : 1;
}
