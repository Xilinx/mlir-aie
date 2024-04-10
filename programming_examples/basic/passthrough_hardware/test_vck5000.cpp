//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <xaiengine.h>

#include "memory_allocator.h"
#include "test_library.h"

#include "aie_data_movement.cpp"
#include "aie_inc.cpp"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

constexpr int DMA_COUNT = 4096;

void hsa_check_status(const std::string func_name, hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char *status_string(new char[1024]);
    hsa_status_string(status, &status_string);
    std::cout << func_name << " failed: " << status_string << std::endl;
    delete[] status_string;
  } else {
    std::cout << func_name << " success" << std::endl;
  }
}

int main(int argc, char *argv[]) {
  uint64_t row = 0;
  uint64_t col = 6;

  std::vector<hsa_queue_t *> queues;
  uint32_t aie_max_queue_size(0);

  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();

  // This is going to initialize HSA, create a queue
  // and get an agent
  int ret = mlir_aie_init_device(xaie);

  if (ret) {
    std::cout << "[ERROR] Error when calling mlir_aie_init_device)"
              << std::endl;
    return -1;
  }

  // Getting access to all of the HSA agents
  std::vector<hsa_agent_t> agents = xaie->agents;

  if (agents.empty()) {
    std::cout << "No agents found. Exiting." << std::endl;
    return -1;
  }

  std::cout << "Found " << agents.size() << " agents" << std::endl;

  hsa_queue_t *q = xaie->cmd_queue;

  // Adding to our vector of queues
  queues.push_back(q);
  assert(queues.size() > 0 && "No queues were sucesfully created!");

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  // Allocating some device memory
  ext_mem_model_t buf0, buf1, buf2;
  uint32_t *in_a = (uint32_t *)mlir_aie_mem_alloc(xaie, buf0, DMA_COUNT);
  uint32_t *in_b = (uint32_t *)mlir_aie_mem_alloc(xaie, buf1, DMA_COUNT);
  uint32_t *out = (uint32_t *)mlir_aie_mem_alloc(xaie, buf2, DMA_COUNT);
  mlir_aie_sync_mem_dev(buf0);
  mlir_aie_sync_mem_dev(buf1);
  mlir_aie_sync_mem_dev(buf2);

  if (in_a == nullptr || in_b == nullptr || out == nullptr) {
    std::cout << "Could not allocate in device memory" << std::endl;
    return -1;
  }

  for (int i = 0; i < DMA_COUNT; i++) {
    in_a[i] = i + 1;
    in_b[i] = i + 1;
    out[i] = 0xdeface;
  }

  // Pass arguments in the order of dma_memcpys in the mlir
  invoke_data_movement(queues[0], &agents[0], out, in_a);

  int errors = 0;

  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t s = in_a[i];
    uint32_t d = out[i];
    if (d != s) {
      errors++;
      printf("mismatch %x != %x\n", d, s);
    }
  }

  // destroying the queue
  hsa_queue_destroy(queues[0]);

  // Shutdown AIR and HSA
  mlir_aie_deinit_libxaie(xaie);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail %d/%d.\n", errors, DMA_COUNT);
    return -1;
  }
}
