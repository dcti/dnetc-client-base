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

const char *autobuff_cpp(void) {
return "@(#)$Id: autobuff.cpp,v 1.17 2000/06/02 06:24:52 jlawson Exp $"; }

#include <string.h> /* memmove() */
//#include <assert.h>
#include "autobuff.h"

AutoBuffer::AutoBuffer(void)
{
  buffersize = AUTOBUFFER_INCREMENT;
  bufferfilled = bufferskip = 0;
  buffer = new char[(int)buffersize];
  //assert(buffer != NULL); /* what good is this supposed to do? */
}

AutoBuffer::AutoBuffer(const AutoBuffer &that)
{
  buffersize = that.buffersize;
  bufferskip = 0;
  bufferfilled = that.GetLength();
  buffer = new char[(int)buffersize];
  //assert(buffer != NULL); /* what good is this supposed to do? */
  if (buffer) memmove(buffer, that.GetHead(), (int)bufferfilled);
}

AutoBuffer::AutoBuffer(const char *szText)
{
  bufferfilled = strlen(szText);
  bufferskip = 0;
  buffersize = bufferfilled + AUTOBUFFER_INCREMENT;
  buffer = new char[(int)buffersize];
  //assert(buffer != NULL); /* what good is this supposed to do? */
  if (buffer) memmove(buffer, szText, (int)bufferfilled);
}

AutoBuffer::AutoBuffer(const char *chData, unsigned int amount)
{
  bufferfilled = amount;
  buffersize = bufferfilled + AUTOBUFFER_INCREMENT;
  bufferskip = 0;
  buffer = new char[(int)buffersize];
  //assert(buffer != NULL); /* what good is this supposed to do? */
  if (buffer) memmove(buffer, chData, (int)bufferfilled);
}

AutoBuffer::~AutoBuffer(void)
{
  if (buffer != NULL) delete [] buffer;
}

// Ensures that the buffer is large enough to contain at least
// the indicated number of characters, enlarging it if necessary.
char *AutoBuffer::Reserve(unsigned int amount)
{
  if (!buffer) {
    // This case shouldn't happen, since buffer should always have
    // been already allocated in the constructor.
    buffersize = amount + AUTOBUFFER_INCREMENT;
    bufferfilled = bufferskip = 0;
    buffer = new char[(int)buffersize];
  } else if (amount > buffersize - bufferfilled) {
    // Buffer is not large enough to accomodate, so resize.
    char *oldbuffer = buffer;
    buffersize = bufferfilled + amount + AUTOBUFFER_INCREMENT;
    buffer = new char[(int)buffersize];
    memmove(buffer, oldbuffer + (int)bufferskip, (int)bufferfilled);
    bufferskip = 0;
    delete [] oldbuffer;
  } else if (amount > buffersize - bufferfilled - bufferskip) {
    // Buffer is large enough, but only when you count the skipped
    // slack at the beginning of the buffer, so move things forward.
    //assert(amount <= buffersize - bufferfilled); /* what good is this supposed to do? */
    memmove(buffer, buffer + (int)bufferskip, (int)bufferfilled);
    bufferskip = 0;
  }
  return buffer;
}

// Indicates that the specified number of characters beyond the
// currently used tail have now been occupied.
void AutoBuffer::MarkUsed(unsigned int amount)
{
  if (buffersize - bufferskip >= amount + bufferfilled) {
    bufferfilled += amount;
  }
  //else assert(0);     // should not happen.
}

// Deallocates the indicated number of characters starting from the
// front of the buffer.
void AutoBuffer::RemoveHead(unsigned int amount)
{
  if (bufferfilled >= amount) {
    bufferskip += amount;
    bufferfilled -= amount;
  }
  //else assert(0);     // should not happen.
}

// Deallocates the indicated number of characters starting from
// the end of the currently allocated buffer.
void AutoBuffer::RemoveTail(unsigned int amount)
{
  if (bufferfilled >= amount) {
    bufferfilled -= amount;
  }
  //else assert(0);     // should not happen.
}

// appending operator.  redefines the buffer to contain the combined
// result of the argument buffer.
void AutoBuffer::operator+= (const AutoBuffer &that)
{
  Reserve(that.GetLength());
  memmove(GetTail(), that.GetHead(), (int)that.GetLength());
  MarkUsed(that.GetLength());
}

// assignment operator.  redefines the buffer to contain a copy
// of the argument buffer.
void AutoBuffer::operator= (const AutoBuffer &that)
{
  Clear();
  Reserve(that.GetLength());
  memmove(GetHead(), that.GetHead(), (int)that.GetLength());
  MarkUsed(that.GetLength());
}

// destructively returns a copy of the first whole text line,
// and removes it from the head of the buffer.
// returns true if a complete line was found.
int AutoBuffer::RemoveLine(AutoBuffer *line)
{
  if (!line) return 0;
  //assert(line != NULL); /* what good is this supposed to do? */
  unsigned int offset = 0;
  int result = StepLine(line, &offset);
  if (result) RemoveHead(offset);
  return result;
}

// non-destructively returns a copy of the first whole text line.
// returns true if a complete line was found.
int AutoBuffer::StepLine(AutoBuffer *line, unsigned int *offset) const
{
  if (!line || !offset)
    return 0;
  //assert(line != NULL && offset != NULL); /* what good is this supposed to do? */
  line->Clear();

  // find the position of the next line break.
  int eol = -1;
  for (int pos = (int)(*offset) ; pos < (int)GetLength(); pos++)
  {
    char ch = GetHead()[pos];
    if (ch == 0 || ch == '\r' || ch == '\n')
      { eol = (pos - (int)(*offset) ); break; }
  }
  if (eol < 0) return 0; /* false */

  // copy the separated line into the target buffer.
  line->Reserve(eol + 1);
  memmove(line->GetHead(), GetHead() + (*offset), eol);
  line->GetHead()[eol] = 0;   // end with a null, but don't include...
  line->MarkUsed(eol);        // ...as part of allocated buffer

  // store the position of the start of the next line.
  (*offset) += eol;
  if (GetHead()[(int)(*offset)] == '\r' &&
      (int)GetLength() > (int)(*offset) + 1 &&
      GetHead()[(int)(*offset) + 1] == '\n') (*offset) += 2;
  else (*offset)++;
  return 1; /* true */
}


