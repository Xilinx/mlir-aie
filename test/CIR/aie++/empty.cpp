/* A tile with an empty program

   Show that declaration of multiple devices of the same kind are actually
   different types, allowing 2 devices in the same platform distinguished in a
   type-safe way
*/

#include "aie++.hpp"

int main() {
  aie::device<aie::npu1> d;
  d.tile<2, 3>().program([] {});
  d.run();
  // Check we can get another type for another accelerator of the same kind
  aie::device<aie::npu1> other_device_unused;
  auto unused_tile = other_device_unused.tile<2, 3>();

  // Check the type safety
  static_assert(!std::is_same_v<decltype(d), decltype(other_device_unused)>);
  static_assert(
      !std::is_same_v<decltype(d.tile<2, 3>()), decltype(unused_tile)>);
}
