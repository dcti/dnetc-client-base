/* 
 * Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * Dynamicly growing buffering class oriented for containing
 * arbitrary binary data for network communications.
 * Created by Jeff Lawson.
 * ----------------------------------------------------------------------
*/ 
#ifndef __AUTOBUFF_H__
#define __AUTOBUFF_H__ "@(#)$Id: autobuff.h,v 1.11 1999/04/05 13:28:38 cyp Exp $"

#ifndef AUTOBUFFER_INCREMENT
#define AUTOBUFFER_INCREMENT 100
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
  AutoBuffer operator+ (const AutoBuffer &that) const;
  bool RemoveLine(AutoBuffer &line);
  bool StepLine(AutoBuffer &line, u32 &offset) const;
};

#endif /* __AUTOBUFF_H__ */

