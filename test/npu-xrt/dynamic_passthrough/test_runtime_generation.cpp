// Test for runtime transaction generation using standalone encoding library
// This demonstrates generating NPU transactions at runtime for different sizes

#include "npu_instructions/npu_instructions.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

// Simulated transaction generation for passthrough kernel
// This function can generate transactions for any multiple of line_size
std::vector<uint32_t> generate_passthrough_txn(size_t num_lines) {
  constexpr size_t line_size = 16; // 16 x i32 = 64 bytes

  std::vector<uint32_t> txn;
  txn.reserve(num_lines * 20); // Rough estimate

  // For each line, generate DMA operations
  for (size_t line = 0; line < num_lines; ++line) {
    uint32_t offset = line * line_size * sizeof(uint32_t);

    // In a real implementation, these would be BD programming operations
    // For now, we'll use write32 operations as placeholders

    // Example: Write BD base address
    uint32_t bd_addr_base = 0x1D000000; // Example shim DMA BD address
    aie::npu::appendWrite32(txn, bd_addr_base + 0, offset); // BD address low

    // Example: Write BD length
    aie::npu::appendWrite32(txn, bd_addr_base + 4, line_size * sizeof(uint32_t));

    // Example: Trigger BD
    aie::npu::appendWrite32(txn, bd_addr_base + 8, 0x80000000); // Start bit

    // Sync after each line transfer
    aie::npu::appendSync(txn,
                         0,  // column
                         0,  // row
                         0,  // direction (S2MM)
                         0,  // channel
                         1,  // column_num
                         1); // row_num
  }

  // Prepend transaction header
  aie::npu::prependHeader(txn);

  return txn;
}

int main() {
  // Test 1: Generate transaction for 1 line (minimum)
  auto txn1 = generate_passthrough_txn(1);
  std::cout << "Transaction for 1 line: " << txn1.size() << " words\n";
  assert(txn1.size() > 4); // At least header + some instructions

  // Test 2: Generate transaction for 8 lines
  auto txn8 = generate_passthrough_txn(8);
  std::cout << "Transaction for 8 lines: " << txn8.size() << " words\n";
  assert(txn8.size() > txn1.size()); // Should be larger

  // Test 3: Generate transaction for 32 lines
  auto txn32 = generate_passthrough_txn(32);
  std::cout << "Transaction for 32 lines: " << txn32.size() << " words\n";
  assert(txn32.size() > txn8.size());

  // Verify scaling relationship (should scale linearly with num_lines)
  size_t ops_per_line = (txn8.size() - 4) / 8; // Subtract header, divide by lines
  size_t expected_txn32_size = 4 + (ops_per_line * 32);
  std::cout << "Expected size for 32 lines: " << expected_txn32_size << "\n";
  std::cout << "Actual size for 32 lines: " << txn32.size() << "\n";

  // Should be close (might not be exact due to header overhead)
  assert(txn32.size() >= expected_txn32_size - 10);
  assert(txn32.size() <= expected_txn32_size + 10);

  std::cout << "All tests passed!\n";
  std::cout << "\nTransaction structure:\n";
  std::cout << "  Header: 4 words (16 bytes)\n";
  std::cout << "  Per-line operations: ~" << ops_per_line << " words\n";
  std::cout << "  Scaling: Linear with num_lines\n";

  return 0;
}
