
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

#include "aie_inc.cpp"
#include "memory_allocator.h"

int main(int argc, char *argv[]) {
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

  ext_mem_model_t buf0, buf1, buf2, buf3;
  // Inputs
  int32_t *buf0_ptr = mlir_aie_mem_alloc(_xaie, buf0, 32 * 32);
  int32_t *buf1_ptr = mlir_aie_mem_alloc(_xaie, buf1, 32 * 32);
  int32_t *buf2_ptr = mlir_aie_mem_alloc(_xaie, buf2, 32 * 32);
  // Output
  int32_t *buf3_ptr = mlir_aie_mem_alloc(_xaie, buf3, 32 * 32);

  for (int i = 0; i < 32 * 32; i++) {
    *(buf0_ptr + i) = i;
    *(buf1_ptr + i) = 256 + i;
    *(buf2_ptr + i) = 512 + i;
    *(buf3_ptr + i) = 0;
  }

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
  mlir_aie_sync_mem_dev(buf0);
  mlir_aie_sync_mem_dev(buf1);
  mlir_aie_sync_mem_dev(buf2);
  mlir_aie_sync_mem_dev(buf3);
  mlir_aie_external_set_addr_buf0(_xaie, (u64)buf0_ptr);
  mlir_aie_external_set_addr_buf1(_xaie, (u64)buf1_ptr);
  mlir_aie_external_set_addr_buf2(_xaie, (u64)buf2_ptr);
  mlir_aie_external_set_addr_buf3(_xaie, (u64)buf3_ptr);
  mlir_aie_configure_shimdma_260(_xaie);
  mlir_aie_configure_shimdma_270(_xaie);
  printf("Start cores...\n");
  mlir_aie_start_cores(_xaie);

  printf("Release locks...\n\n");
  mlir_aie_release_input_lock_0(_xaie, 1, 0);
  mlir_aie_release_input_lock_1(_xaie, 1, 0);
  mlir_aie_release_input_lock_2(_xaie, 1, 0);

  int errors = 0;
  if (mlir_aie_acquire_output_lock(_xaie, 1, 10000)) {
    errors++;
    printf("ERROR: timeout hit!\n");
  }

  mlir_aie_print_shimdma_status(_xaie, 26, 0);
  mlir_aie_print_shimdma_status(_xaie, 27, 0);
  mlir_aie_print_dma_status(_xaie, 25, 2);

  mlir_aie_sync_mem_cpu(buf3);

  for (int i = 0; i < 32 * 32; i++) {
    mlir_aie_check("C", *(buf0_ptr + i), i, errors);
    mlir_aie_check("A", *(buf1_ptr + i), 256 + i, errors);
    mlir_aie_check("B", *(buf2_ptr + i), 512 + i, errors);
  }

  mlir_aie_release_output_lock(_xaie, 0, 0);

  mlir_aie_deinit_libxaie(_xaie);

  printf("Complete compute.\n");
}
