#include <stdio.h>

#include "stubsplit.h"

int callback(void *userdata, struct Stub *stub)
{
  int i;
  printf("%d ", stub->marks);
  for (i = 0; i < stub->length; i++) {
    printf("%d ", stub->stub[i]);
  }
  printf("\n");
  return 1;
}

int main(int argc, char *argv[])
{
  struct Stub stub;
  int i;

  if (argc < 2) {
    printf("Usage: test-stubsplit marks dist1 dist2 ...\n");
    return 1;
  }
  if (argc > 1+STUB_MAX) {
    printf("Stub too long.\n");
    return 1;
  }
  
  stub.marks = atoi(argv[1]);
  stub.length = argc - 2;
  for (i = 0; i < stub.length; i++) {
    stub.stub[i] = atoi(argv[2+i]);
  }
  stub_split(&stub, callback, 0);

  return 0;
}
