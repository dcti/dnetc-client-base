#define WIN32_LEAN_AND_MEAN
#include <string.h>
#include <windows.h>

int strcmpi(const char *s1, const char *s2)
{
  return lstrcmpi(s1, s2);
}