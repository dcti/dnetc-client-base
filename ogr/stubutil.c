#include <stdio.h>
#include <string.h>
#include "stubutil.h"

const char *stubstr(const Stub *stub)
{
  static char buf[80];
  if (stub->length > 5) {
    sprintf(buf, "(error:%d/%d)", stub->marks, stub->length);
    return buf;
  }
  sprintf(buf, "%d/", stub->marks);
  if (stub->length == 0) {
    strcat(buf, "-");
    return buf;
  }
  int len = stub->length;
  for (int i = 0; i < len; i++) {
    sprintf(&buf[strlen(buf)], "%d", stub->stub[i]);
    if (i+1 < len) {
      strcat(buf, "-");
    }
  }
  return buf;
}
