#include <stdio.h>
#include <stdlib.h>

#include "stubsplit.h"

int marks;
int depth;
int stubsx;
long long stubs;

int callback(void *userdata, struct Stub *stub)
{
  if (stub->length < depth) {
    stub_split(stub, callback, 0);
  } else {
    stubsx++;
    if (stubsx == 1000000) {
      stubs += stubsx;
      stubsx = 0;
      printf("\033[K%d %d %Ld...", marks, depth, stubs);
      for (int i = 0; i < stub->length; i++) {
        printf("%d ", stub->diffs[i]);
      }
      printf("\r");
      fflush(stdout);
    }
  }
  return 1;
}

int main(int argc, char *argv[])
{
  if (argc < 3) {
    printf("Usage: test-stubsplit marks depth\n");
    return 1;
  }

  marks = atoi(argv[1]);
  depth = atoi(argv[2]);
  
  struct Stub stub;
  stub.marks = marks;
  stub.length = 0;
  stubs = 0;
  stub_split(&stub, callback, 0);
  stubs += stubsx;
  printf("\033[K%d %d %Ld\n", marks, depth, stubs);

  return 0;
}
