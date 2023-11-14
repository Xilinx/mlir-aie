//===- ipulog.h --------------------------------------------000---*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdlib.h>

class IPULogger {
private:
  uint8_t *_buffer;
  uint32_t _maxlen;
  uint32_t _count;
  uint32_t _dropped_msgs;

public:
  IPULogger(uint8_t *buffer, uint32_t len) : _buffer(buffer), _maxlen(len) {
    _count = 0;
    _dropped_msgs = 0;
  }

  ~IPULogger() {}

  //----------------------------------
  // peel parameters of write
  //----------------------------------
  // zeroth case
  template <typename T = void>
  void log_peel_params(uint32_t *hmsg, uint32_t *cnt) {
    { return; }
  }

  // general case: peels the parameters off the argument list and appends them
  // to the message
  template <typename P1, typename... Param>
  void log_peel_params(uint32_t *hmsg, uint32_t *cnt, const P1 &p1,
                       Param &...param) {
    hmsg[*cnt + 1] = *(
        (uint32_t *)&p1); // prune type and place raw bytes in the host message
    *cnt = *cnt + 1;
    log_peel_params(hmsg, cnt, param...); // keep on peeling
    return;
  }
  //----------------------------------

  template <typename... Param>
  void write(const char *msg, const Param &...param) {
    if (almost_full(16)) {
      _write("Log buffer is full -- we have dropped %u messages",
             _dropped_msgs++);
      _buffer -= 8;
      _count -= 8;
      return;
    }
    _write(msg, param...);
  }

  template <typename... Param>
  void _write(const char *msg, const Param &...param) {
    // create a message
    uint32_t hmsg[40];

    // assign the constant string addr in memory
    hmsg[0] = (uint32_t)((uint32_t *)((void *)(msg)));

    // recursively peel off the parameters and assign
    uint32_t param_cnt = 0;
    log_peel_params(&hmsg[0], &param_cnt, param...);

    memcpy(_buffer, &hmsg, (param_cnt + 1) * 4);
    _buffer += (param_cnt + 1) * 4;
    _count += (param_cnt + 1) * 4;

    return;
  }

  bool almost_full(uint32_t amount) { return _count >= (_maxlen - amount); }
};
