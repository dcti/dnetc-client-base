/* $Id: next_sup.cpp,v 1.1.2.1 2003/01/19 14:33:56 andreasb Exp $ */

#include "next_sup.h"
#include <stdlib.h>
#include <string.h>

char *strdup(const char *src)
{
  char *dst = (char *)malloc(strlen(src + 1));

  if (dst != NULL)
    strcpy(dst, src);

  return dst;
}
