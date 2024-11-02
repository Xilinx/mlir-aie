/* A tile with an empty program
*/

#include "aie++.hpp"

int main() {
  aie::device<aie::npu1> d;
  d.tile<2, 3>().program([]{});
  d.run();
  // Check we can get another type for another accelerator of the same kind
  aie::device<aie::npu1> other_device_unused;
  auto unused_tile = other_device_unused.tile<2, 3>();
}
