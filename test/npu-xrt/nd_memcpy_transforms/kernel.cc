typedef int16_t my_t;

extern "C" {
void concat(my_t *a, my_t *b, my_t *c, int a_sz, int b_sz, int c_sz) {
  // Concatenates a and b and writes the result to c.
  int i = 0;
  for (; i < c_sz && i < a_sz; i++) {
    c[i] = a[i];
  }
  for (; i < c_sz && i - a_sz < b_sz; i++) {
    c[i] = b[i - a_sz];
  }
}
}
