/*
 * OGR support routines and data.
 *
 * $Id: ogr_sup.cpp,v 1.1.2.5.2.1 2001/07/08 18:25:32 andreasb Exp $
*/
#include <baseincs.h>

#include "ogr.h"

#if 0
unsigned long ogr_nodecount(const struct Stub *stub)
{
  stub = stub; /* shaddup compiler */
  return 1;
}
#endif

#ifdef OGR_OLD_STUB
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
#else

int ogr_init_stub2_from_testcasedata(struct Stub2 *stub, u32 marks, u32 ndiffs, const u32 *diffs)
{
  if (!stub)
    return -1;
  
  memset(stub, 0, sizeof(struct Stub2));
  stub->marks = (u8)marks;
  stub->depth = 0;
  for (unsigned int i = 0; i < ndiffs; ++i)
  {
    if (i >= STUB_MAX)
      return -1;
      stub->diffs[i] = (u8)(diffs[i]);
      if (stub->diffs[i] == 0)
        break;
      stub->depth++;  
  }
  return 0;
}

/* hton(struct Stub2) / ntoh(struct Stub2) */
int ogr_switch_byte_order(struct Stub2 *stub)
{
  if (!stub)
    return -1;

  stub->nodes.lo = htonl(stub->nodes.lo);
  stub->nodes.hi = htonl(stub->nodes.hi);
  /* everything else is stored as u8 */
  
  return 0;
}

int ogr_netstub2_to_stub2(const struct NetStub2 *netstub, struct Stub2 *stub, int switch_byte_order)
{
  if (!netstub || !stub)
    return -1;

  memset(stub, 0, sizeof(struct Stub2));
  stub->nodes.lo = netstub->nodes.lo;
  stub->nodes.hi = netstub->nodes.hi;
  stub->marks = netstub->marks;
  stub->depth = netstub->depth;
  stub->workdepth = netstub->workdepth;
  for (int i = 0; i < STUB_MAX && i < NET_STUB_MAX; i++)
    stub->diffs[i] = netstub->diffs[i];
  stub->depththdiff = netstub->depththdiff;
  stub->core = netstub->core;
  stub->cycle = netstub->cycle;

  if (switch_byte_order) {
    stub->nodes.lo = ntohl(stub->nodes.lo);
    stub->nodes.hi = ntohl(stub->nodes.hi);
  }
  
  return 0;
}


int ogr_stub2_to_netstub2(const struct Stub2 *stub, struct NetStub2 *netstub, int switch_byte_order)
{
  if (!netstub || !stub)
    return -1;

  memset(netstub, 0, sizeof(struct NetStub2));
  netstub->nodes.lo = stub->nodes.lo;
  netstub->nodes.hi = stub->nodes.hi;
  netstub->marks = stub->marks;
  netstub->depth = stub->depth;
  netstub->workdepth = stub->workdepth;
  if (netstub->workdepth > NET_STUB_MAX) {
    /* WARNING: nodes.{hi|lo} will become incorrect if the stub gets processed after this step ... !!! */
    netstub->workdepth = NET_STUB_MAX;
  }
  for (int i = 0; i < STUB_MAX && i < NET_STUB_MAX; i++)
    netstub->diffs[i] = stub->diffs[i];
  netstub->depththdiff = stub->depththdiff;
  netstub->core = stub->core;
  netstub->cycle = stub->cycle;
  
  if (switch_byte_order) {
    netstub->nodes.lo = ntohl(netstub->nodes.lo);
    netstub->nodes.hi = ntohl(netstub->nodes.hi);
  }
  
  return 0;
}

int ogr_reset_stub(struct Stub2 *stub)
{
  if (!stub)
    return -1;

  stub->nodes.hi = stub->nodes.lo = 0;
  stub->workdepth = 0;
  if (stub->depththdiff)
    stub->diffs[stub->depth-1] = stub->depththdiff;
  for (int i = stub->depth; i < STUB_MAX; ++i)
    stub->diffs[i] = 0;
  stub->core = 0;
  
  return 0;
}

int ogr_benchmark_stub(struct Stub2* stub)
{
  if (!stub)
    return -1;

  memset(stub, 0, sizeof(struct Stub2));
  //24/2-22-32-21-5-1-12
  //25/6-9-30-14-10-11
  stub->nodes.lo = stub->nodes.hi = 0;
  stub->marks = 25;    //24;
  stub->depth = 6;     //7;
  stub->workdepth = 6; //7;
  stub->diffs[0] = 6;  //2;
  stub->diffs[1] = 9;  //22;
  stub->diffs[2] = 30; //32;
  stub->diffs[3] = 14; //21;
  stub->diffs[4] = 10; //5;
  stub->diffs[5] = 11; //1;
  stub->diffs[6] = 0;  //12;
  
  return 0;
}

const char *ogr_stubstr_r(const struct Stub2 *stub,
                          char *buffer, unsigned int bufflen,
                          int format /* format = 0 without workpos / format = 1 with workpos */ )
/*
 * 25/1-2-3-4-5-6, 25/1-2-3-4-5*6
 * 25/1-2-3-4-5-6+7-8-9-10, 25/1-2-3-4-5*6|6+7-8-9-10, 25/1-2-3-4-5*6|99+7+8+9+10
 */
{
  if (buffer && stub && bufflen)
  {
    char buf[128]; /* (1+29)*(max(1,2,3)+1)+4 */
    int i;
    int depth      = (int)stub->depth;
    int wdepth     = (int)stub->workdepth;
    int printdepth = (format == 0) ? depth : (wdepth<depth)?depth:wdepth;
    int asterisk   = (stub->depththdiff == 0) ? -1 : depth-1;
    int plus       = (format == 0) ? -1 : depth;
    int diff;

    if (depth > STUB_MAX || wdepth > STUB_MAX) {
      sprintf(buf, "(error:%d/%d|%d)", (int)stub->marks, depth, wdepth);
    }
    else {
      sprintf(buf, "%d/", (int)stub->marks);
      if (printdepth == 0) {
        strcat(buf, "-");
      }
      else {
        for (i = 0; i < printdepth; i++) {
          if (i == asterisk)
            strcat(buf, "*");
          else if (i == plus)
            strcat(buf, "+");
          else if (i > 0)
            strcat(buf, "-");
          diff = stub->diffs[i];          
          if (stub->depththdiff != 0 && i == depth-1)
          {
            if (format == 0)
              diff = stub->depththdiff;
            else
              sprintf(&buf[strlen(buf)], "%d|", (int)stub->depththdiff);
          }
          sprintf(&buf[strlen(buf)], "%d", diff);
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

const char *ogr_stubstr(const struct Stub2 *stub)
{
  static char buf[80];
  return ogr_stubstr_r(stub, buf, sizeof(buf), 0);
}

#endif
