// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: autobuff.cpp,v $
// Revision 1.9  1998/07/08 23:30:33  remi
// Cleared a GCC warning.
// Tweaked $Id$.
//
// Revision 1.8  1998/07/08 09:23:17  jlawson
// eliminated integer type warnings on win16
//
// Revision 1.7  1998/07/07 21:54:57  cyruspatel
// Serious house cleaning - client.h has been split into client.h (Client
// class, FileEntry struct etc - but nothing that depends on anything) and
// baseincs.h (inclusion of generic, also platform-specific, header files).
// The catchall '#include "client.h"' has been removed where appropriate and
// replaced with correct dependancies. cvs Ids have been encapsulated in
// functions which are later called from cliident.cpp. Corrected other
// compile-time warnings where I caught them. Removed obsolete timer and
// display code previously def'd out with #if NEW_STATS_AND_LOGMSG_STUFF.
// Made MailMessage in the client class a static object (in client.cpp) in
// anticipation of global log functions.
//
// Revision 1.6  1998/07/06 09:21:19  jlawson
// added lint tags around cvs id's to suppress unused variable warnings.
//
// Revision 1.5  1998/06/15 12:03:42  kbracey
// Lots of consts.
//
// Revision 1.4  1998/06/14 08:26:34  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.3  1998/06/14 08:12:23  friedbait
// 'Log' keywords added to maintain automatic change history
//
//

#if (!defined(lint) && defined(__showids__))
const char *autobuff_cpp(void) {
return "@(#)$Id: autobuff.cpp,v 1.9 1998/07/08 23:30:33 remi Exp $"; }
#endif

#include <string.h>
#include "cputypes.h" //u32 
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

AutoBuffer::AutoBuffer(const char *chData, u32 amount)
{
  bufferfilled = amount;
  buffersize = bufferfilled + AUTOBUFFER_INCREMENT;
  buffer = new char[(int)buffersize];
  memmove(buffer, chData, (int)bufferfilled);
}

AutoBuffer::~AutoBuffer(void)
{
  delete buffer;
}

char *AutoBuffer::Reserve(u32 amount)
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
    delete oldbuffer;
  }
  return buffer;
}

void AutoBuffer::MarkUsed(u32 amount)
{
  if (buffersize >= amount + bufferfilled) bufferfilled += amount;
}

void AutoBuffer::RemoveHead(u32 amount)
{
  if (bufferfilled >= amount) {
    memmove(buffer, buffer + (int)amount, (int)(bufferfilled - amount));
    bufferfilled -= amount;
  }
}

void AutoBuffer::RemoveTail(u32 amount)
{
  if (bufferfilled >= amount) bufferfilled -= amount;
}

void AutoBuffer::operator+= (const AutoBuffer &that)
{
  Reserve(that.GetLength());
  memmove(GetTail(), that.GetHead(), (int)that.GetLength());
  MarkUsed(that.GetLength());
}

void AutoBuffer::operator= (const AutoBuffer &that)
{
  Clear();
  Reserve(that.GetLength());
  memmove(GetHead(), that.GetHead(), (int)that.GetLength());
  MarkUsed(that.GetLength());
}

AutoBuffer AutoBuffer::operator+ (const AutoBuffer &that)
{
  AutoBuffer output;
  output.Reserve(GetLength() + that.GetLength());
  memmove(output.GetHead(), GetHead(), (int)GetLength());
  memmove(output.GetHead() + (int)GetLength(), that.GetHead(), (int)that.GetLength());
  output.MarkUsed(GetLength() + that.GetLength());
  return output;
}

// returns true if a complete line was found
bool AutoBuffer::RemoveLine(AutoBuffer &line)
{
  line.Clear();
  int eol = -1;
  for (int pos = 0; pos < (int)GetLength(); pos++)
  {
    char ch = GetHead()[pos];
    if (ch == 0 || ch == '\r' || ch == '\n') {eol = pos; break;}
  }
  if (eol < 0) return false;

  line.Reserve(eol + 1);
  memmove(line.GetHead(), GetHead(), eol);
  line.GetHead()[eol] = 0;      // end with a null, but don't include...
  line.MarkUsed(eol);           // ...as part of allocated buffer
  if (GetHead()[eol] == '\r' &&
      (int)GetLength() > eol + 1 &&
      GetHead()[eol + 1] == '\n') RemoveHead(eol + 2);
  else RemoveHead(eol + 1);
  return true;
}
