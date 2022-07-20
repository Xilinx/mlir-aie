
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for MLIR-AIE host kernel.
//
//===----------------------------------------------------------------------===//

#include "test_library.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <xaiengine.h>

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)
#define MLIR_STACK_OFFSET 4096

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  int32_t arg0[32][32];
  int32_t arg1[32][32];
  int32_t arg2[32][32];
  unsigned iter_num = 1;

  printf("Configure AIE array...\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);

  printf("Initialize buffers...\n");

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  assert(fd != -1 && "memory is not available");

  mlir_aie_clear_tile_memory(_xaie, 25, 2);

  unsigned bufIdx;

  int32_t *buf0_ptr = (int32_t *)mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, fd, 0x4000);
  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 32; ++idx0)
    for (int64_t idx1 = 0; idx1 < 32; ++idx1)
      buf0_ptr[bufIdx++] = arg0[idx0][idx1];

  int32_t *buf1_ptr = (int32_t *)mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, fd, 0x5000);
  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 32; ++idx0)
    for (int64_t idx1 = 0; idx1 < 32; ++idx1)
      buf1_ptr[bufIdx++] = arg2[idx0][idx1];

  int32_t *buf2_ptr = (int32_t *)mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, fd, 0x6000);
  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 32; ++idx0)
    for (int64_t idx1 = 0; idx1 < 32; ++idx1)
      buf2_ptr[bufIdx++] = arg1[idx0][idx1];

  bool results[1];

  for (auto &result : results)
    result = false;

  auto kernel_complete = [&]() {
    bool flag = true;
    for (auto result : results) {
      flag &= result;
      // printf("%d ", result);
    }
    // printf("\n");
    return flag;
  };

  printf("Start cores...\n");
  mlir_aie_start_cores(_xaie);

  printf("Release locks...\n\n");
  mlir_aie_release_lock(_xaie, 27, 0, 0, 1, 0);
  mlir_aie_release_lock(_xaie, 26, 0, 2, 1, 0);
  mlir_aie_release_lock(_xaie, 26, 0, 1, 1, 0);

  while (!kernel_complete()) {
    if (mlir_aie_acquire_lock(_xaie, 26, 0, 0, 1, 0))
      results[0] = true;
  }

  int32_t *buf3_ptr = (int32_t *)mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, fd, 0x7000);
  bufIdx = 0;
  for (int64_t idx0 = 0; idx0 < 32; ++idx0)
    for (int64_t idx1 = 0; idx1 < 32; ++idx1)
      arg0[idx0][idx1] = buf3_ptr[bufIdx++];

  mlir_aie_release_lock(_xaie, 26, 0, 0, 0, 0);

  mlir_aie_deinit_libxaie(_xaie);

  printf("Complete compute.\n");
}
