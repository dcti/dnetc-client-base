/*
 * OGR support routines and data.
 *
 * "@(#)$Id: ogr_sup.cpp,v 1.2 2000/07/11 02:33:42 mfeiri Exp $"
*/

#include <stdio.h>
#include <string.h>

#include "ogr.h"

unsigned long ogr_nodecount(const struct Stub *stub)
{
  stub = stub; /* shaddup compiler */
  return 1;
}

const char *ogr_stubstr_r(const struct Stub *stub,
                          char *buffer, unsigned int bufflen)
{
  if (buffer && stub && bufflen)
  {
    char buf[80];
    int i, len = (int)stub->length;

    if (len > STUB_MAX) {
      sprintf(buf, "(error:%d/%d)", (int)stub->marks, len);
    }
    else {
      sprintf(buf, "%d/", (int)stub->marks);
      if (len == 0) {
        strcat(buf, "-");
      }
      else {
        for (i = 0; i < len; i++) {
          sprintf(&buf[strlen(buf)], "%d", (int)stub->diffs[i]);
          if (i+1 < len) {
            strcat(buf, "-");
          }
        }
      }  
    }  
    buffer[0] = '\0';
    if (bufflen > 1) {
      strncpy(buffer,buf,bufflen);
      buffer[bufflen-1] = '\0';
    }
    return buffer;
  }  
  return "";
}

const char *ogr_stubstr(const struct Stub *stub)
{
  static char buf[80];
  return ogr_stubstr_r(stub, buf, sizeof(buf));
}
