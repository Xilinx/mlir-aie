#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

extern "C" {
void saxpy(bfloat16 *restrict x, bfloat16 *restrict y, bfloat16 *restrict z) {
  event0();
  ::aie::vector<bfloat16, 64> a_v = ::aie::broadcast<bfloat16, 64>(3.f);
#pragma clang loop min_iteration_count(4)
  for (int i = 0; i < 4096; i += 64) {
    ::aie::vector<bfloat16, 64> x_v = ::aie::load_v<64>(x);
    x += 64;
    ::aie::vector<bfloat16, 64> y_v = ::aie::load_v<64>(y);
    y += 64;
    ::aie::accum<accfloat, 64> ax_v = ::aie::mul(x_v, a_v);
    ::aie::vector<bfloat16, 64> ax_v_converted = ax_v.to_vector<bfloat16>();
    ::aie::vector<bfloat16, 64> z_v = ::aie::add(ax_v_converted, y_v);
    ::aie::store_v(z, z_v);
    z += 64;
  }
  event1();
}
}
