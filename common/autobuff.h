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
#define __AUTOBUFF_H__ "@(#)$Id: autobuff.h,v 1.14.2.2 2000/03/09 01:47:19 jlawson Exp $"

#ifndef AUTOBUFFER_INCREMENT
#define AUTOBUFFER_INCREMENT 100
#endif

class AutoBuffer
{
  char *buffer;
  unsigned int bufferfilled;    // number of actually used bytes
  unsigned int buffersize;      // current maximum size of buffer
  unsigned int bufferskip;      // unused slack at start of buffer
public:
  AutoBuffer(void);
  AutoBuffer(const AutoBuffer &that);
  AutoBuffer(const char *szText);
  AutoBuffer(const char *chData, unsigned int amount);
  ~AutoBuffer(void);
  operator const char* (void) const {return GetHead();}
  char *GetHead(void) const {return buffer + (int)bufferskip;}
  char *GetTail(void) const {return buffer + (int)bufferskip + (int)bufferfilled;}
  char *Reserve(unsigned int amount);
  void MarkUsed(unsigned int amount);
  void RemoveHead(unsigned int amount);
  void RemoveTail(unsigned int amount);
  unsigned int GetLength(void) const {return bufferfilled;}
  unsigned int GetTailSlack(void) const {return buffersize - bufferskip - bufferfilled;}
  void Clear(void) {bufferfilled = bufferskip = 0;}
  void operator+= (const AutoBuffer &that);
  void operator= (const AutoBuffer &that);
  bool RemoveLine(AutoBuffer *line);
  bool StepLine(AutoBuffer *line, unsigned int *offset) const;
};

#endif /* __AUTOBUFF_H__ */

