/******************************************************************************
* Copyright (C) 2019 - 2020 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef __COM_HELPER_H__
#define __COM_HELPER_H__

/* SemaphoreHandle_t / TaskHandle_t are provided by cdo_io_debug.h, which cdo_io.h
 * includes before this header. */

enum ipu_resources {
    ERT_QUEUE_TAIL_H2C_DOORBELL = 0,
    ERT_QUEUE_TAIL_C2H_DOORBELL = 0,
    DPU_EVENT,
    AIE_EVENT,
    ADMA,
    IPU_RES_NUM
};

struct event_resources {
    enum ipu_resources res;
    SemaphoreHandle_t sem;
};

struct com_helper_arg {
    TaskHandle_t tHdl;
};

void wait_event(enum ipu_resources res);
int com_usleep(unsigned int usec);
int com_helper_init(struct com_helper_arg *arg);
void com_enable_interrupts(void);

#endif
