#include <cstdio>

int entry(void);

int main(void) {
  int r = entry();
  if (r)
    printf("ERROR: %d", r);
  printf("SUCCESS");
  return r;
}
