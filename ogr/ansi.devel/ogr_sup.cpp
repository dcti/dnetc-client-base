/*
 * OGR support routines and data.
 *
*/
const char *ogr_sup_cpp(void) {
return "@(#)$Id: ogr_sup.cpp,v 1.1.2.3 2001/01/19 02:29:46 andreasb Exp $"; }

#include <stdio.h>
#include <string.h>

#include "ogr.h"

unsigned long ogr_nodecount(const struct Stub *stub)
{
  stub = stub; /* shaddup compiler */
  return 1;
}

const char *ogr_stubstr_r(const struct Stub *stub,
                          char *buffer, unsigned int bufflen,
                          int worklength /* 0 or workstub.worklength */)
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
        if (worklength > len) {
          strcat(buf, "+");
          for (i = len; i < worklength && i < STUB_MAX; i++) {
            if (i > len) {
              strcat(buf, "-");
            }
            sprintf(&buf[strlen(buf)], "%d", (int)stub->diffs[i]);
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
  return ogr_stubstr_r(stub, buf, sizeof(buf), 0);
}
