//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "test_library.h"
#include "memory_allocator.h"

#include "aie_inc.cpp"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define XAIE_NUM_COLS 10

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

  if(ret) {
    std::cout << "[ERROR] Error when calling mlir_aie_init_device)" << std::endl;
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

#define DMA_COUNT 16

  // Allocating some device memory
  ext_mem_model_t buf0, buf1;
  uint32_t *src = (uint32_t *)mlir_aie_mem_alloc(xaie, buf0, DMA_COUNT);
  uint32_t *dst = (uint32_t *)mlir_aie_mem_alloc(xaie, buf1, DMA_COUNT);
  mlir_aie_sync_mem_dev(buf0);
  mlir_aie_sync_mem_dev(buf1);

  if (src == NULL || dst == NULL) {
    std::cout << "Could not allocate src and dst in device memory" << std::endl;
    return -1;
  }

  for (int i = 0; i < DMA_COUNT; i++) {
    src[i] = i + 1;
    dst[i] = 0xdeface;
  }

  for (int i = 0; i < 8; i++) {
    mlir_aie_write_buffer_ping_in(xaie, i, 0xabbaba00 + i);
    mlir_aie_write_buffer_pong_in(xaie, i, 0xdeeded00 + i);
    mlir_aie_write_buffer_ping_out(xaie, i, 0x12345670 + i);
    mlir_aie_write_buffer_pong_out(xaie, i, 0x76543210 + i);
  }

  //
  // send the data
  //

  uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t write_pkt;
  air_packet_nd_memcpy(&write_pkt, 0, col, 1, 0, 4, 2,
                       reinterpret_cast<uint64_t>(src),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &write_pkt);

  //
  // read the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t read_pkt;
  air_packet_nd_memcpy(&read_pkt, 0, col, 0, 0, 4, 2,
                       reinterpret_cast<uint64_t>(dst),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &read_pkt);

  int errors = 0;

  for (int i = 0; i < 8; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_in(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_in(xaie, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_out(xaie, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_out(xaie, i);
    if (d0 + 1 != d2) {
      printf("mismatch ping %x != %x\n", d0, d2);
      errors++;
    }
    if (d1 + 1 != d3) {
      printf("mismatch pong %x != %x\n", d1, d3);
      errors++;
    }
  }

  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t s = src[i];
    uint32_t d = dst[i];
    //printf("src[%d] = 0x%lx\n", i, src[i]);
    //printf("dst[%d] = 0x%lx\n", i, dst[i]);
    if (d != (s + 1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, s);
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
