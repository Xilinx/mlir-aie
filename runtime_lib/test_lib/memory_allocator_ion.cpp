//===- memory_allocator.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This code is heavily based on aienginev2_v3_0/src/io_backend/ext/xaie_linux.c
// Current version of this library no longer implements memory allocation, so we
// have to do it ourselves.

/***************************** Include Files *********************************/
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/dma-buf.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "ion.h"
#include "memory_allocator.h"

/***************************** Macro Definitions *****************************/
#define XAIE_128BIT_ALIGN_MASK 0xFF

/**
 * This is the memory function to allocate a memory
 *
 * @param	handle: Device Instance
 * @param	size: Size of the memory
 *
 * @return	Pointer to the allocated memory instance.
 *******************************************************************************/
int *mlir_aie_mem_alloc(struct aie_libxaie_ctx_t *ctx, ext_mem_model_t &handle,
                        int size) {
  int RC;
  int Fd, Ret;
  uint32_t HeapNum;
  void *VAddr;
  struct ion_allocation_data AllocArgs;
  struct ion_heap_query Query;
  struct ion_heap_data *Heaps;
  u64 DevAddr = 0;

  Fd = open("/dev/ion", O_RDONLY);
  if (Fd < 0) {
    XAIE_ERROR("Failed to open ion.\n");
    return NULL;
  }

  memset(&Query, 0, sizeof(Query));
  Ret = ioctl(Fd, ION_IOC_HEAP_QUERY, &Query);
  if (Ret != 0) {
    XAIE_ERROR("Failed to enquire ion heaps.\n");
    goto error_ion;
  }

  Heaps = (struct ion_heap_data *)calloc(Query.cnt, sizeof(*Heaps));
  if (Heaps == NULL) {
    XAIE_ERROR("Failed to allocate memory for heap details\n");
    goto error_ion;
  }

  Query.heaps = (uint64_t)Heaps;
  Ret = ioctl(Fd, ION_IOC_HEAP_QUERY, &Query);
  if (Ret != 0) {
    XAIE_ERROR("Failed to enquire ion heap details.\n");
    free(Heaps);
    goto error_ion;
  }

  HeapNum = UINT_MAX;
  for (uint32_t i = 0; i < Query.cnt; i++) {
    XAIE_DBG("Heap id: %u, Heap name: %s, Heap type: %u\n", Heaps[i].heap_id,
             Heaps[i].name, Heaps[i].type);
    if (Heaps[i].type == ION_HEAP_TYPE_SYSTEM_CONTIG) {
      HeapNum = i;
      break;
    }
  }

  if (HeapNum == UINT_MAX) {
    XAIE_ERROR("Failed to find contiguous heap\n");
    free(Heaps);
    goto error_ion;
  }

  memset(&AllocArgs, 0, sizeof(AllocArgs));
  AllocArgs.len = size;
  AllocArgs.heap_id_mask = 1 << Heaps[HeapNum].heap_id;
  free(Heaps);
  // if(Cache == XAIE_MEM_CACHEABLE) {
  // 	AllocArgs.flags = ION_FLAG_CACHED;
  // }

  Ret = ioctl(Fd, ION_IOC_ALLOC, &AllocArgs);
  if (Ret != 0) {
    XAIE_ERROR("Failed to allocate memory of %lu bytes\n");
    goto error_ion;
  }

  VAddr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, AllocArgs.fd, 0);
  if (VAddr == NULL) {
    XAIE_ERROR("Failed to mmap\n");
    goto error_alloc_fd;
  }

  handle.fd = AllocArgs.fd;
  handle.virtualAddr = VAddr;
  handle.size = size;

  // Map the memory
  if (XAie_MemAttach(ctx->XAieDevInst, &(handle.MemInst), DevAddr, (u64)VAddr,
                     size, XAIE_MEM_NONCACHEABLE, handle.fd) != XAIE_OK) {
    XAIE_ERROR("dmabuf map failed\n");
    goto error_map;
  }

  close(Fd);
  return (int *)VAddr;

error_map:
  munmap(VAddr, size);
error_alloc_fd:
  close(handle.fd);
error_ion:
  close(Fd);
  return NULL;
}

/*****************************************************************************/
/**
 *
 * This is function to attach the allocated memory descriptor to kernel driver
 *
 * @param	IOInst: Linux IO instance pointer
 * @param	MemInst: Linux Memory instance pointer.
 *
 * @return	XAIE_OK on success, Error code on failure.
 *
 * @note		Internal only.
 *
 *******************************************************************************/
// static int _XAie_LinuxMemDetach(XAie_LinuxIO *IOInst, XAie_LinuxMem *MemInst)
// {
// 	int Ret;

// 	Ret = ioctl(IOInst->PartitionFd, AIE_DETACH_DMABUF_IOCTL,
// 			MemInst->BufferFd);
// 	if(Ret != 0) {
// 		XAIE_ERROR("Failed to detach dmabuf\n");
// 		return XAIE_ERR;
// 	}

// 	return XAIE_OK;
// }

/*****************************************************************************/
/**
 *
 * This is the memory function to free the memory
 *
 * @param	MemInst: Memory instance pointer.
 *
 * @return	XAIE_OK on success, Error code on failure.
 *
 * @note		Internal only.
 *
 *******************************************************************************/
// static int XAie_LinuxMemFree(XAie_MemInst *MemInst)
// {
// 	int RC;
// 	XAie_LinuxMem *LinuxMemInst =
// 		(XAie_LinuxMem *)MemInst->BackendHandle;

// 	RC = _XAie_LinuxMemDetach((XAie_LinuxIO *)MemInst->DevInst->IOInst,
// 			LinuxMemInst);
// 	if(RC != XAIE_OK) {
// 		return RC;
// 	}

// 	munmap(MemInst->VAddr, MemInst->Size);
// 	close(LinuxMemInst->BufferFd);
// 	free(LinuxMemInst);
// 	free(MemInst);

// 	return XAIE_OK;
// }

/*****************************************************************************/
/**
 *
 * This is the memory function to sync the memory for CPU.
 *
 * @param	MemInst: Memory instance pointer.
 *
 * @return	XAIE_OK on success, Error code on failure.
 *
 * @note		Internal only.
 *
 *******************************************************************************/
void mlir_aie_sync_mem_cpu(ext_mem_model_t &handle) {
  struct dma_buf_sync Sync;
  int Ret;

  memset(&Sync, 0, sizeof(Sync));
  Sync.flags = DMA_BUF_SYNC_RW | DMA_BUF_SYNC_START;
  Ret = ioctl(handle.fd, DMA_BUF_IOCTL_SYNC, &Sync);
  if (Ret != 0) {
    XAIE_ERROR("Failed to sync, %s.\n", strerror(errno));
    // return XAIE_ERR;
  }

  // return XAIE_OK;
}

/*****************************************************************************/
/**
 *
 * This is the memory function to sync the memory for Device.
 *
 * @param	MemInst: Memory instance pointer.
 *
 * @return	XAIE_OK on success, Error code on failure.
 *
 * @note		Internal only.
 *
 *******************************************************************************/
void mlir_aie_sync_mem_dev(ext_mem_model_t &handle) {
  struct dma_buf_sync Sync;
  int Ret;

  memset(&Sync, 0, sizeof(Sync));
  Sync.flags = DMA_BUF_SYNC_RW | DMA_BUF_SYNC_END;
  Ret = ioctl(handle.fd, DMA_BUF_IOCTL_SYNC, &Sync);
  if (Ret != 0) {
    XAIE_ERROR("Failed to sync, %s.\n", strerror(errno));
    //		return;
    //		return XAIE_ERR;
  }

  //	return XAIE_OK;
}

u64 mlir_aie_get_device_address(struct aie_libxaie_ctx_t *_xaie, void *VA) {
  return (u64)VA; // LibXAIE will take care of converting this for us.
}

/** @} */
