/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * OGR support routines and data.
 *
 * $Id: ogr_sup.cpp,v 1.9 2008/12/31 00:26:17 kakace Exp $
*/
#include <stdio.h>
#include <string.h>

#include "unused.h"     /* DNETC_UNUSED_* */
#include "ogr.h"

const char* ogr_errormsg(int errorcode)
{
   switch(errorcode) {
      case CORE_E_MEMORY:    return "CORE_E_MEMORY: Insufficient memory";
      case CORE_E_STUB:      return "CORE_E_STUB: Invalid initial ruler";
      case CORE_E_FORMAT:    return "CORE_E_FORMAT: Format or range error";
      case CORE_E_INTERNAL:  return "CORE_E_INTERNAL: Bogus core";
      case CORE_E_CORRUPTED: return "CORE_E_CORRUPTED: Core damaged";
      default:
        break;
   }
   return "unknown error";
}

/*==============================================================================
** Legacy stub to string converstion routine.
*/

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


/*==============================================================================
** OGR-NG stub to string converstion routine.
** Display formats are based on the recommandations stated in bug #4082.
*/

const char *ogrng_stubstr_r(const struct OgrWorkStub *stub,
                            char *buffer, unsigned int bufflen,
                            int maxdiff /* 0 = stub only, >0 = how many diffs */)
{
  if (buffer && stub && bufflen)
  {
    char buf[80];
    int i, len = (int)stub->stub.length;
    
    if (len > OGR_STUB_MAX) {
      sprintf(buf, "(error:%d/%d)", (int)stub->stub.marks, len);
    }
    else {
      sprintf(buf, "%d/", (int)stub->stub.marks);
      if (len == 0) {
        strcat(buf, "-");
      }
      else {
        for (i = 0; i < len; i++) {
          int diff = (int) stub->stub.diffs[i];
          if (i+1 == len && stub->collapsed != 0) {
            diff = (int) stub->collapsed;   // Restore initial stub diff.
          }
          sprintf(&buf[strlen(buf)], "%d", diff);
          if (i+1 < len) {
            strcat(buf, "-");
          }
          else if (maxdiff == 0 && stub->collapsed != 0) {
            strcat(buf, "*");               // Indicate a combined stub.
          }
        }
        if (maxdiff > 0 && stub->worklength > len) {
          if (stub->collapsed != 0) {
            // Combined stub : Display the current value of the last stub diff.
            sprintf(&buf[strlen(buf)], "@%d", stub->stub.diffs[len-1]);
          }
          strcat(buf, "+");
          for (i = len; i < stub->worklength && i < maxdiff; i++) {
            if (i > len) {
              strcat(buf, "-");
            }
            sprintf(&buf[strlen(buf)], "%d", (int)stub->stub.diffs[i]);
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

const char *ogrng_stubstr(const struct OgrWorkStub *stub)
{
  static char buf[80];
  return ogrng_stubstr_r(stub, buf, sizeof(buf), 0);
}
