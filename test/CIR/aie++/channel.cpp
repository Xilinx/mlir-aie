/*
  Iron MICRO24 slide 34.
  https://github.com/Xilinx/mlir-aie/blob/main/mlir_tutorials
*/

#include "aie++.hpp"
#include <stdint.h>

void consume(std::int32_t in[256]) {
  // ...
}

void produce(std::int32_t out[][256]) {
  // ...
}

int main() {
  aie::device<aie::npu1> d;
  auto a = d.tile<1, 3>();
  auto b = d.tile<2, 3>();
  auto of = a.channel_to<std::int32_t, 256>(b, 3);
  a.program([&] {
    for (int i = 0; i < 3; ++i) {
      auto acc = of.out_acquire_release(1);
      produce(acc);
    }
  });
  b.program([&] {
    auto acc = of.in_acquire_release(3);
    consume(acc[0]);
    consume(acc[1]);
    consume(acc[2]);
  });
  d.run();
}
