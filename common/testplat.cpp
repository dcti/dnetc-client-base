#include <stdio.h>
#include <string.h>
#define IGNOREUNKNOWNCPUOS
#include "cputypes.h"


int main(int argc, char *argv[])
{
  if (argc != 2)
    {
      printf("Specify 'cpu', 'os', or 'intsizes' as argument.\n");
      return -1;
    }
  if (strcmp(argv[1], "cpu") == 0)
    printf("%i\n", (int) CLIENT_CPU);
  else if (strcmp(argv[1], "os") == 0)
    printf("%i\n", (int) CLIENT_OS);
  else if (strcmp(argv[1], "intsizes") == 0)
    printf("%i%i%i\n", (int) sizeof(long), (int) sizeof(int), (int) sizeof(short));
  return 0;
}
