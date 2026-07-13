// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Single-threaded fallback for lms-hash-sigs' hss_thread.h abstraction, for
// platforms without POSIX pthreads (e.g. MSVC). This bootgen snapshot only
// vendors the pthread-based implementation (hss_thread_pthread.c); per
// hss_thread.h's own documented contract, a thread collection of 0/NULL is a
// valid, non-threaded return, and callers must tolerate work being completed
// synchronously within hss_thread_issue_work.

#include "hss_thread.h"

struct thread_collection *hss_thread_init(int num_threads) {
  (void)num_threads;
  return 0;
}

void hss_thread_issue_work(struct thread_collection *col,
                            void (*function)(const void *detail,
                                              struct thread_collection *col),
                            const void *detail, size_t size_detail_structure) {
  (void)size_detail_structure;
  function(detail, col);
}

void hss_thread_done(struct thread_collection *col) { (void)col; }

void hss_thread_before_write(struct thread_collection *collect) {
  (void)collect;
}

void hss_thread_after_write(struct thread_collection *collect) {
  (void)collect;
}

unsigned hss_thread_num_tracks(int num_threads) {
  (void)num_threads;
  return 1;
}
