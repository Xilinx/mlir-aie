/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * drivers/staging/android/uapi/ion.h
 *
 * Copyright (C) 2011 Google, Inc.
 */

#ifndef _UAPI_LINUX_ION_H
#define _UAPI_LINUX_ION_H

#include <linux/ioctl.h>
#include <linux/types.h>

/// enum ion_heap_types - list of all possible types of heaps
enum ion_heap_type {
  ION_HEAP_TYPE_SYSTEM,         ///< memory allocated via vmalloc
  ION_HEAP_TYPE_SYSTEM_CONTIG,  ///< memory allocated via kmalloc
  ION_HEAP_TYPE_CARVEOUT,       ///< memory allocated from a prereserved carveout heap, allocations are physically contiguous
  ION_HEAP_TYPE_CHUNK,
  ION_HEAP_TYPE_DMA,            ///< memory allocated via DMA API
  ION_HEAP_TYPE_CUSTOM,         ///< must be last so device specific heaps always are at the end of this enum
};

#define ION_NUM_HEAP_IDS (sizeof(unsigned int) * 8)

/**
 * allocation flags - the lower 16 bits are used by core ion, the upper 16
 * bits are reserved for use by the heaps themselves.
 */

/*
 * mappings of this buffer should be cached, ion will do cache maintenance
 * when the buffer is mapped for dma
 */
#define ION_FLAG_CACHED 1

/**
 * DOC: Ion Userspace API
 *
 * create a client by opening /dev/ion
 * most operations handled via following ioctls
 *
 */

/**
 * struct ion_allocation_data - metadata passed from userspace for allocations
 *
 * Provided by userspace as an argument to the ioctl
 */
struct ion_allocation_data {
  __u64 len;            ///< size of the allocation
  __u32 heap_id_mask;   ///< mask of heap ids to allocate from
  __u32 flags;          ///< flags passed to heap
  __u32 fd;             ///< file descriptor for this allocation
  __u32 unused;         ///< unused field
};

#define MAX_HEAP_NAME 32

/**
 * struct ion_heap_data - data about a heap
 */
struct ion_heap_data {
  char name[MAX_HEAP_NAME]; ///< first 32 characters of the heap name
  __u32 type;               ///< heap type
  __u32 heap_id;            ///< heap id for the heap
  __u32 reserved0;          ///< reserved field
  __u32 reserved1;          ///< reserved field
  __u32 reserved2;          ///< reserved field
};

/**
 * struct ion_heap_query - collection of data about all heaps
 */
struct ion_heap_query {
  __u32 cnt;       ///< Total number of heaps to be copied
  __u32 reserved0; ///< align to 64bits
  __u64 heaps;     ///< buffer to be populated
  __u32 reserved1; ///< reserved field
  __u32 reserved2; ///< reserved field
};

#define ION_IOC_MAGIC 'I'

/**
 * DOC: ION_IOC_ALLOC - allocate memory
 *
 * Takes an ion_allocation_data struct and returns it with the handle field
 * populated with the opaque handle for the allocation.
 */
#define ION_IOC_ALLOC _IOWR(ION_IOC_MAGIC, 0, struct ion_allocation_data)

/**
 * DOC: ION_IOC_HEAP_QUERY - information about available heaps
 *
 * Takes an ion_heap_query structure and populates information about
 * available Ion heaps.
 */
#define ION_IOC_HEAP_QUERY _IOWR(ION_IOC_MAGIC, 8, struct ion_heap_query)

#endif /* _UAPI_LINUX_ION_H */
