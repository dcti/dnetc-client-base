// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
#ifndef __AUTOBUFF_H__
#define __AUTOBUFF_H__

#include "cputypes.h"


#ifndef AUTOBUFFER_INCREMENT
#define AUTOBUFFER_INCREMENT 100
#endif

#if (CLIENT_OS == OS_AMIGA)
extern "C" {
#endif

#include <string.h>

#if (CLIENT_OS == OS_AMIGA)
}
#endif

class AutoBuffer
{
  char *buffer;
  u32 bufferfilled;
  u32 buffersize;
public:
  AutoBuffer(void);
  AutoBuffer(const AutoBuffer &that);
  AutoBuffer(const char *szText);
  AutoBuffer(const char *chData, u32 amount);
  ~AutoBuffer(void);
  operator const char* (void) const {return buffer;}
  char *GetHead(void) const {return buffer;}
  char *GetTail(void) const {return buffer + (int)bufferfilled;}
  char *Reserve(u32 amount);
  void MarkUsed(u32 amount);
  void RemoveHead(u32 amount);
  void RemoveTail(u32 amount);
  u32 GetLength(void) const {return bufferfilled;}
  u32 GetSlack(void) const {return buffersize - bufferfilled;}
  void Clear(void) {bufferfilled = 0;}
  void operator+= (const AutoBuffer &that);
  void operator= (const AutoBuffer &that);
  AutoBuffer operator+ (const AutoBuffer &that);
  bool RemoveLine(AutoBuffer &line);
};



#endif

