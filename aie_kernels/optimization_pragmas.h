#if defined(__chess__)
#define AIE_LOOP_MIN_ITERATION_COUNT(x) [[chess::min_loop_count(x)]]
#define AIE_LOOP_MAX_ITERATION_COUNT(x) [[chess::max_loop_count(x)]]
#define AIE_LOOP_RANGE(a,...) [[chess::min_loop_count(a)]]__VA_OPT__([[chess::max_loop_count(__VA_ARGS__)]])

#elif defined(__AIECC__)
#ifndef __STRINGIFY
  #define __STRINGIFY(a) #a
#endif

#define AIE_LOOP_MIN_ITERATION_COUNT(x) _Pragma(__STRINGIFY(clang loop min_iteration_count(x)))
#define AIE_LOOP_MAX_ITERATION_COUNT(x) _Pragma(__STRINGIFY(clang loop max_iteration_count(x)))
#define AIE_LOOP_RANGE(a,...) AIE_LOOP_MIN_ITERATION_COUNT(a)__VA_OPT__(AIE_LOOP_MAX_ITERATION_COUNT(__VA_ARGS__))
#endif