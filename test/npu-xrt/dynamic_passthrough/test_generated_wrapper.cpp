// Wrapper to test generated C++ transaction code
#include <cassert>
#include <iostream>
#include <vector>
#include <cstdint>

// Forward declare the generated function
// (will be defined in the generated .cpp file)
namespace aie_runtime {
  std::vector<uint32_t> generate_txn_parameterized_sequence(size_t num_writes, size_t base_addr);
}

int main() {
  // Test with different parameters
  auto txn_small = aie_runtime::generate_txn_parameterized_sequence(4, 0x1000);
  std::cout << "Small transaction (4 writes): " << txn_small.size() << " words\n";

  auto txn_medium = aie_runtime::generate_txn_parameterized_sequence(16, 0x2000);
  std::cout << "Medium transaction (16 writes): " << txn_medium.size() << " words\n";

  auto txn_large = aie_runtime::generate_txn_parameterized_sequence(64, 0x3000);
  std::cout << "Large transaction (64 writes): " << txn_large.size() << " words\n";

  // Verify scaling
  assert(txn_medium.size() > txn_small.size());
  assert(txn_large.size() > txn_medium.size());

  // Verify transaction contains header (first 4 words)
  assert(txn_small.size() >= 4);
  assert(txn_medium.size() >= 4);
  assert(txn_large.size() >= 4);

  std::cout << "\nRuntime transaction generation test: PASSED\n";
  std::cout << "Successfully generated transactions for different problem sizes!\n";

  return 0;
}
