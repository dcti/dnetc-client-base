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
return "@(#)$Id: autobuff.cpp,v 1.15.2.2 2000/02/06 00:19:00 sampo Exp $"; }

#include <string.h> /* memmove() */
#include "autobuff.h"

AutoBuffer::AutoBuffer(void)
{
  buffersize = AUTOBUFFER_INCREMENT;
  bufferfilled = 0;
  buffer = new char[(int)buffersize];
}

AutoBuffer::AutoBuffer(const AutoBuffer &that)
{
  buffersize = that.buffersize;
  bufferfilled = that.GetLength();
  buffer = new char[(int)buffersize];
  memmove(buffer, that.GetHead(), (int)bufferfilled);
}

AutoBuffer::AutoBuffer(const char *szText)
{
  bufferfilled = strlen(szText);
  buffersize = bufferfilled + AUTOBUFFER_INCREMENT;
  buffer = new char[(int)buffersize];
  memmove(buffer, szText, (int)bufferfilled);
}

AutoBuffer::AutoBuffer(const char *chData, unsigned int amount)
{
  bufferfilled = amount;
  buffersize = bufferfilled + AUTOBUFFER_INCREMENT;
  buffer = new char[(int)buffersize];
  memmove(buffer, chData, (int)bufferfilled);
}

AutoBuffer::~AutoBuffer(void)
{
  delete [] buffer;
}

// Ensures that the buffer is large enough to contain at least
// the indicated number of characters, enlarging it if necessary.
char *AutoBuffer::Reserve(unsigned int amount)
{
  if (!buffer) {
    buffersize = amount + AUTOBUFFER_INCREMENT;
    bufferfilled = 0;
    buffer = new char[(int)buffersize];
  } else if (amount > buffersize - bufferfilled) {
    char *oldbuffer = buffer;
    buffersize = bufferfilled + amount + AUTOBUFFER_INCREMENT;
    buffer = new char[(int)buffersize];
    memmove(buffer, oldbuffer, (int)bufferfilled);
    delete [] oldbuffer;
  }
  return buffer;
}

// Indicates that the specified number of characters beyond the
// currently used tail have now been occupied.
void AutoBuffer::MarkUsed(unsigned int amount)
{
  if (buffersize >= amount + bufferfilled) bufferfilled += amount;
}

// Deallocates the indicated number of characters starting from the
// front of the buffer.
void AutoBuffer::RemoveHead(unsigned int amount)
{
  if (bufferfilled >= amount) {
    memmove(buffer, buffer + (int)amount, (int)(bufferfilled - amount));
    bufferfilled -= amount;
  }
}

// Deallocates the indicated number of characters starting from
// the end of the currently allocated buffer.
void AutoBuffer::RemoveTail(unsigned int amount)
{
  if (bufferfilled >= amount) bufferfilled -= amount;
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

#ifdef PROXYTYPE /* Not for client. Aggregate returns are not portable */
// Returns a dynamic copy of the two combined buffers
AutoBuffer AutoBuffer::operator+ (const AutoBuffer &that) const
{
  AutoBuffer output;
  output.Reserve(GetLength() + that.GetLength());
  memmove(output.GetHead(), GetHead(), (int)GetLength());
  memmove(output.GetHead() + (int)GetLength(), that.GetHead(), (int)that.GetLength());
  output.MarkUsed(GetLength() + that.GetLength());
  return output;
}
#endif

// destructively returns a copy of the first whole text line,
// and removes it from the head of the buffer.
// returns true if a complete line was found.
bool AutoBuffer::RemoveLine(AutoBuffer *line)
{
  unsigned int offset = 0;
  bool result = StepLine(line, &offset);
  if (result) RemoveHead(offset);
  return result;
}

// non-destructively returns a copy of the first whole text line.
// returns true if a complete line was found.
bool AutoBuffer::StepLine(AutoBuffer *line, unsigned int *offset) const
{
  line->Clear();
  int eol = -1;
  for (int pos = (int)(*offset) ; pos < (int)GetLength(); pos++)
  {
    char ch = GetHead()[pos];
    if (ch == 0 || ch == '\r' || ch == '\n')
      { eol = (pos - (int)(*offset) ); break; }
  }
  if (eol < 0) return false;

  line->Reserve(eol + 1);
  memmove(line->GetHead(), GetHead() + (*offset), eol);
  line->GetHead()[eol] = 0;   // end with a null, but don't include...
  line->MarkUsed(eol);        // ...as part of allocated buffer

  (*offset) += eol;
  if (GetHead()[(int)(*offset)] == '\r' &&
      (int)GetLength() > (int)(*offset) + 1 &&
      GetHead()[(int)(*offset) + 1] == '\n') (*offset) += 2;
  else (*offset)++;
  return true;
}


