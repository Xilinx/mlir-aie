//===- test_vck5000.cpp -----------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
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

constexpr int IMAGE_WIDTH = 128;
constexpr int IMAGE_HEIGHT = 16;
constexpr int IMAGE_SIZE = (IMAGE_WIDTH * IMAGE_HEIGHT);
constexpr int TILE_WIDTH = 16;
constexpr int TILE_HEIGHT = 8;
constexpr int TILE_SIZE = (TILE_WIDTH * TILE_HEIGHT);

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
  uint32_t *in_a = (uint32_t *)mlir_aie_mem_alloc(xaie, buf0, IMAGE_SIZE);
  uint32_t *in_b = (uint32_t *)mlir_aie_mem_alloc(xaie, buf1, IMAGE_SIZE);
  uint32_t *out = (uint32_t *)mlir_aie_mem_alloc(xaie, buf2, IMAGE_SIZE);
  mlir_aie_sync_mem_dev(buf0);
  mlir_aie_sync_mem_dev(buf1);
  mlir_aie_sync_mem_dev(buf2);

  if (in_a == NULL || in_b == NULL || out == NULL) {
    std::cout << "Could not allocate in device memory" << std::endl;
    return -1;
  }

  for (int i = 0; i < IMAGE_SIZE; i++) {
    in_a[i] = i + 1;
    in_b[i] = 1;
    out[i] = 0xdeface;
  }

  // Pass arguments in the order of dma_memcpys in the mlir
  invoke_data_movement(queues[0], &agents[0], out, in_a);

  int errors = 0;

  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t row = i / IMAGE_WIDTH;
    uint32_t col = i % IMAGE_WIDTH;
    uint32_t s = in_a[i];
    uint32_t d = out[i];

    if (row < TILE_HEIGHT && col < TILE_WIDTH) {
      if (d != s + 1) {
        errors++;
        printf("[ERROR] row %d and col %d, %d != %d\n", row, col, s, d);
      }
    } else {
      if (d == s + 1) {
        errors++;
        printf("[ERROR] row %d and col %d, %d == %d -- this was not supposed "
               "to be changed\n",
               row, col, s, d);
      }
    }

    printf("s[%d, %d] = 0x%x\n", row, col, s);
    printf("d[%d, %d] = 0x%x\n", row, col, d);
  }

  // destroying the queue
  hsa_queue_destroy(queues[0]);

  // Shutdown AIR and HSA
  mlir_aie_deinit_libxaie(xaie);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail %d/%d.\n", errors, IMAGE_SIZE);
    return -1;
  }
}
