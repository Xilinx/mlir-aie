//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

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

#include "memory_allocator.h"
#include "test_library.h"
#include "aie_inc.cpp"


#define BUF_SIZE (4*10)  // # ints

void write_stream(aie_libxaie_ctx_t *_xaie, ext_mem_model_t &ping_buf, ext_mem_model_t &pong_buf, int *ping, int *pong) {
    for(int i = 0; i < BUF_SIZE; i++) {
        if(i % 2 == 0) {
            assert(XAIE_OK == mlir_aie_acquire_fifo0_prod_lock(_xaie, -1, 10000));
            ping[0] = i;
            mlir_aie_sync_mem_dev(ping_buf);
            mlir_aie_release_fifo0_cons_lock(_xaie, 1, 0);
        } else {
            assert(XAIE_OK == mlir_aie_acquire_fifo0_prod_lock(_xaie, -1, 10000));
            pong[0] = i;
            mlir_aie_sync_mem_dev(pong_buf);
            mlir_aie_release_fifo0_cons_lock(_xaie, 1, 0);
        }
    }
}

void read_all_into(aie_libxaie_ctx_t *_xaie, int *buf, int (*read_buf_func)(aie_libxaie_ctx_t *, int)) {
    for(int i = 0; i < BUF_SIZE; i++) {
        buf[i] = read_buf_func(_xaie, i);
    }
}

void populate_expected(int *buf) {
    int sum = 0;
    for(int i = 0; i < BUF_SIZE; i++) {
        buf[i] = i;
    }
}

int main(int argc, char *argv[]) {
    int errors = 0;
    int seen33[BUF_SIZE], seen34[BUF_SIZE], seen35[BUF_SIZE];
    int expected[BUF_SIZE];
    ext_mem_model_t buf0, buf1;
    int *ping, *pong;

    aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
    assert(NULL != _xaie);
    mlir_aie_init_device(_xaie);
    mlir_aie_configure_cores(_xaie);
    mlir_aie_configure_switchboxes(_xaie);
    mlir_aie_configure_dmas(_xaie);
    mlir_aie_initialize_locks(_xaie);

    // External buffer setup
    ping = mlir_aie_mem_alloc(buf0, 1);
    pong = mlir_aie_mem_alloc(buf1, 1);
#ifdef __AIESIM__
    mlir_aie_external_set_addr_extbuf0((uint64_t)buf0.physicalAddr);
    mlir_aie_external_set_addr_extbuf1((uint64_t)buf1.physicalAddr);
#else
    mlir_aie_external_set_addr_extbuf0((uint64_t)ping);
    mlir_aie_external_set_addr_extbuf1((uint64_t)pong);
#endif

    mlir_aie_configure_shimdma_30(_xaie);

    mlir_aie_sync_mem_dev(buf0);
    mlir_aie_sync_mem_dev(buf1);

    // Start cores
    mlir_aie_start_cores(_xaie);

    // Write to stream
    write_stream(_xaie, buf0, buf1, ping, pong);

    // Wait until all cores have completed
    assert(XAIE_OK == mlir_aie_acquire_lock33(_xaie, -1, 5000));
    assert(XAIE_OK == mlir_aie_acquire_lock34(_xaie, -1, 5000));
    assert(XAIE_OK == mlir_aie_acquire_lock35(_xaie, -1, 5000));

    read_all_into(_xaie, seen33, mlir_aie_read_buffer_buf33);
    read_all_into(_xaie, seen34, mlir_aie_read_buffer_buf34);
    read_all_into(_xaie, seen35, mlir_aie_read_buffer_buf35);

    mlir_aie_deinit_libxaie(_xaie);

    populate_expected(expected);

    for(int i = 0; i < BUF_SIZE; i++) {
        printf("%04d=?=%04d=?=%04d=?=%04d ", seen33[i], seen34[i], seen35[i], expected[i]);
        if((i+1) % 3 == 0) {
            printf("\n");
        }
        if(seen33[i] != expected[i] || seen34[i] != expected[i] || seen35[i] != expected[i]) {
            printf("\nFAIL!\n");
            return 1;
        }
    }

    printf("\nPASS!\n");   
    return 0;
}
