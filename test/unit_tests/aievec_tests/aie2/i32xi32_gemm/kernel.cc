#include "testbench.h"

#if __AIEARCH__ == 10
#include "gen_aie.cc"
#else
#include "gen_aie-ml.cc"
#endif
