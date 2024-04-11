//===- memory_allocator_hsa.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "memory_allocator.h"
#include "test_library.h"
#include <iostream>

//
// This memory allocator links against the HSA allocator
//
int *mlir_aie_mem_alloc(aie_libxaie_ctx_t *_xaie, ext_mem_model_t &handle,
                        int size) {
  int size_bytes = size * sizeof(int);
  hsa_amd_memory_pool_allocate(_xaie->global_mem_pool, size_bytes, 0,
                               (void **)&(handle.virtualAddr));

  if (handle.virtualAddr) {
    handle.size = size_bytes;
  } else {
    printf("ExtMemModel: Failed to allocate %d memory.\n", size_bytes);
  }

  std::cout << "ExtMemModel constructor: virtual address " << std::hex
            << handle.virtualAddr << ", size " << std::dec << handle.size
            << std::endl;

  return (int *)handle.virtualAddr;
}

/*
  The device memory allocator directly maps device memory over
  PCIe MMIO. These accesses are uncached and thus don't require
  explicit synchronization between the host and device
*/
void mlir_aie_sync_mem_cpu(ext_mem_model_t &handle) {}

void mlir_aie_sync_mem_dev(ext_mem_model_t &handle) {}

/*
  The only component that knows the proper translation from
  VA->PA is the command processor. Sending a request to the
  command processor to perform the translation.
*/
u64 mlir_aie_get_device_address(struct aie_libxaie_ctx_t *_xaie, void *VA) {

  // Checking to make sure that the agent is setup properly
  if (_xaie == NULL) {
    printf("[ERROR] %s passed NULL context ptr\n", __func__);
    return NULL;
  }

  if (_xaie->agents.size() == 0) {
    printf("[ERROR] %s passed context has no agents\n", __func__);
    return NULL;
  }

  if (_xaie->cmd_queue == NULL) {
    printf("[ERROR] %s passed context has no queue\n", __func__);
    return NULL;
  }

  // Getting pointers to the queue and the agent
  hsa_queue_t *queue = _xaie->cmd_queue;
  hsa_agent_t agent = _xaie->agents[0];

  uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);
  uint64_t packet_id = wr_idx % queue->size;
  hsa_agent_dispatch_packet_t pkt;
  mlir_aie_packet_req_translation(&pkt, (uint64_t)VA);
  hsa_amd_signal_create_on_agent(1, 0, nullptr, &agent, 0,
                                 &(pkt.completion_signal));
  reinterpret_cast<hsa_agent_dispatch_packet_t *>(
      queue->base_address)[packet_id] = pkt;

  // Ringing the doorbell to notify the command processor of the packet
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  // wait for packet completion
  while (hsa_signal_wait_scacquire(pkt.completion_signal,
                                   HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
                                   HSA_WAIT_STATE_ACTIVE) != 0)
    ;

  // We encode the response in the packet, so need to peek in to get the data
  hsa_agent_dispatch_packet_t *pkt_peek =
      &reinterpret_cast<hsa_agent_dispatch_packet_t *>(
          queue->base_address)[packet_id];

  // Copying the translated address
  uint64_t PA;
  PA = (uint64_t)pkt_peek->return_address;

  // Destroying the signal
  hsa_signal_destroy(pkt.completion_signal);

  return (u64)PA; // The platform will convert the address for us
}
