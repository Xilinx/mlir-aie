#if defined(__chess__)
#define AIE_PREPARE_FOR_PIPELINE [[using chess: prepare_for_pipelining]]
#define AIE_LOOP_MIN_ITERATION_COUNT(x) [[using chess: min_loop_count(x)]]
#define AIE_LOOP_MAX_ITERATION_COUNT(x) [[using chess: max_loop_count(x)]]
#define AIE_LOOP_RANGE(a, ...)                                                 \
  [[chess::min_loop_count(a)]] __VA_OPT__(                                     \
      [[chess::max_loop_count(__VA_ARGS__)]])
#define AIE_LOOP_UNROLL(x) [[using chess: unroll_loop(x)]]
#define AIE_LOOP_UNROLL_FULL [[using chess: unroll_loop]]
#define AIE_LOOP_FLATTEN [[using chess: flatten_loop]]
// TODO Chess no unroll
#define AIE_LOOP_NO_UNROLL

#elif defined(__AIECC__)
#ifndef __STRINGIFY
#define __STRINGIFY(a) #a
#endif

#define AIE_PREPARE_FOR_PIPELINE
#define AIE_LOOP_MIN_ITERATION_COUNT(x)                                        \
  _Pragma(__STRINGIFY(clang loop min_iteration_count(x)))
#define AIE_LOOP_MAX_ITERATION_COUNT(x)                                        \
  _Pragma(__STRINGIFY(clang loop max_iteration_count(x)))
#define AIE_LOOP_RANGE(a, ...)                                                 \
  AIE_LOOP_MIN_ITERATION_COUNT(a)                                              \
  __VA_OPT__(AIE_LOOP_MAX_ITERATION_COUNT(__VA_ARGS__))
#define AIE_LOOP_UNROLL(x)                                                     \
  _Pragma(__STRINGIFY(clang loop unroll_count(x)))
#define AIE_LOOP_UNROLL_FULL                                                   \
  _Pragma("clang loop unroll(full)")
#define AIE_LOOP_NO_UNROLL                                                     \
  _Pragma("clang loop unroll(disable)")
#define AIE_LOOP_FLATTEN [[using chess: flatten_loop]]

#endif
