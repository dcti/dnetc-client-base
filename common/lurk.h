// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: lurk.h,v $
// Revision 1.7  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.6  1998/11/12 13:09:03  silby
// Added a stop function, made start and stop public.
//
// Revision 1.5  1998/10/03 03:24:36  cyp
// Fixed a broken #endif (had trailing comment without //). Added
// #ifndef __LURK_H__ /#define __LURK_H__ /#endif nesting.
//
//
//
#ifndef __LURK_H__
#define __LURK_H__

#include "cputypes.h"
#if (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32)

#ifndef LURK
#define LURK
#endif

class Lurk
  {
public:

// These are config variables, everything needs access to them

s32 lurkmode;
  // Mode of operation
  // 0 = disabled
  // 1 = fill buffers while online, try to connect when emptied
  // 2 = fill buffers while online, never connect
s32 dialwhenneeded;
  // 0 = Don't dial, let autodial handle it or fail
  // 1 = Have the client manually dial/hangup when a flush happens.
char connectionname[100];
  // For win32, name of connection to use, perhaps useful for other lurkers.

Lurk();

char *GetEntryList(long *finalcount);
  // Gets the list of possible dial-up networking connections for the
  // user to select. - called in cliconfig

s32 CheckIfConnectRequested(void);
  // Returns the possible values of connectrequested

s32 CheckForStatusChange(void);
  // Checks to see if we've suddenly disconnected
  // Return values:
  // 0 = connection has not changed or has just connected
  // -1 = we just lost the connection

s32 DialIfNeeded(s32 force);
  // Dials the connection if current parameters allow it.
  // Force values:
  // 0 = act normal
  // 1 = override lurk-only mode and connect anyway.
  // return values:
  // 0=Already connected or connect succeeded.
  // -1=There was an error, we're not connected.

s32 HangupIfNeeded(void);
  // Hangs up the connection if current parameters allow it.
  // return values - 0 is the only return as of now.

s32 Start(void);
  // Initializes Lurk Mode -> 0=success, -1 = failure

s32 Stop(void);
  // Terminates Lurk Mode -> 0=success, -1 = failure

protected:

s32 InitiateConnection(void);
  // Initiates a dialup connection
  // 0 = already connected, 1 = connection started,
  // -1 = connection failed

s32 TerminateConnection(void);
  // -1 = connection did not terminate properly, 0 = connection
  // terminated

s32 Status(void);
  // Checks status of connection -> !0 = connected

s32 islurkstarted;
  // 0 = lurk functionality has not been initialized
  // 1 = lurk functionality has been initialized

s32 oldlurkstatus;
  // Status of LurkStatus as of the last check.

s32 dialstatus;
  // Tells us if we're in the process of using a dialup connection
  // 0 = We did not initiate any dialing, don't touch the connect.
  // 1 = We dialed, you're welcome to hangup.
};

extern Lurk dialup;

#endif //CLIENT_OS == ...
#endif //__LURK_H__

