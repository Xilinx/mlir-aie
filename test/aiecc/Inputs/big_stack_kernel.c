// Deliberately oversized frame: a 2048-byte local buffer whose address
// escapes to a global, so the compiler cannot shrink it to just its touched
// bytes. Stands in for a real kernel (conv accumulator, im2col scratch, ...)
// whose frame exceeds a core's declared stack_size -- see
// cpp_stack_size_overflow.mlir.
volatile char *escape;
void big_stack_kernel(void) {
  char big[2048];
  big[0] = 1;
  big[2047] = 2;
  escape = big;
}
