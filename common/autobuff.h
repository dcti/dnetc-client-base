// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// ----------------------------------------------------------------------
// Dynamicly growing buffering class oriented for containing
// arbitrary binary data for network communications.
// Created by Jeff Lawson.
// ----------------------------------------------------------------------
// 

#ifndef __AUTOBUFF_H__
#define __AUTOBUFF_H__ "@(#)$Id: autobuff.h,v 1.15 1999/11/23 22:36:36 cyp Exp $"

#ifndef AUTOBUFFER_INCREMENT
#define AUTOBUFFER_INCREMENT 100
#endif

class AutoBuffer
{
  char *buffer;
  unsigned int bufferfilled;
  unsigned int buffersize;
public:
  AutoBuffer(void);
  AutoBuffer(const AutoBuffer &that);
  AutoBuffer(const char *szText);
  AutoBuffer(const char *chData, unsigned int amount);
  ~AutoBuffer(void);
  operator const char* (void) const {return buffer;}
  char *GetHead(void) const {return buffer;}
  char *GetTail(void) const {return buffer + (int)bufferfilled;}
  char *Reserve(unsigned int amount);
  void MarkUsed(unsigned int amount);
  void RemoveHead(unsigned int amount);
  void RemoveTail(unsigned int amount);
  unsigned int GetLength(void) const {return bufferfilled;}
  unsigned int GetSlack(void) const {return buffersize - bufferfilled;}
  void Clear(void) {bufferfilled = 0;}
  void operator+= (const AutoBuffer &that);
  void operator= (const AutoBuffer &that);
  #ifdef PROXYTYPE /* not for client: aggregate returns are not portable */
  AutoBuffer operator+ (const AutoBuffer &that) const;
  #endif
  bool RemoveLine(AutoBuffer *line);
  bool StepLine(AutoBuffer *line, unsigned int *offset) const;
};

#endif /* __AUTOBUFF_H__ */

