// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: lurk.h,v $
// Revision 1.11  1999/02/06 10:42:55  remi
// - the default for dialup.ifacestowatch is now 'ppp0:sl0'.
// - #ifdef'ed dialup.ifacestowatch (only Linux at the moment)
// - modified a bit the help text in confopt.cpp
//
// Revision 1.10  1999/02/06 09:08:08  remi
// Enhanced the lurk fonctionnality on Linux. Now it use a list of interfaces
// to watch for online/offline status. If this list is empty (the default), any
// interface up and running (besides the lookback one) will trigger the online
// status.
// Fixed formating in lurk.cpp.
//
// Revision 1.9  1999/02/04 07:47:06  cyp
// Re-import from proxy base. Cleaned up. Added linux and win16 support.
//
// Revision 1.3  1999/01/28 19:53:11  trevorh
// Added lurkonly support for OS/2
//
// Revision 1.2  1999/01/24 23:14:58  dbaker
// copyright 1999 changes
//
// Revision 1.1  1998/12/30 01:45:11  jlawson
// added lurk code from client.
//
// Revision 1.6  1998/11/12 13:09:03  silby
// Added a stop function, made start and stop public.
//
// Revision 1.5  1998/10/03 03:24:36  cyp
// Fixed a broken #endif (had trailing comment without //). Added
// #ifndef __LURK_H__ /#define __LURK_H__ /#endif nesting.
//
#ifndef __LURK_H__
#define __LURK_H__

#define CONNECT_LURKONLY 2
#define CONNECT_LURK     1

class Lurk
{
public:

int lurkmode;
  // Mode of operation
  // 0 = disabled
  // 1 = CONNECT_LURK = fill buffers while online, try to connect when emptied
  // 2 = CONNECT_LURKONLY = fill buffers while online, never connect

int dialwhenneeded;
  // 0 = Don't dial, let autodial handle it or fail
  // !0 = Have the client manually dial/hangup when a flush happens.

char connectionname[100];
  // For win32, name of connection to use, perhaps useful for other lurkers.
  // used by linux and OS/2 as name of script to call to initiate connection
char stopconnection[100];  
  // used by linux and OS/2 as name of script to call to stop connection

char *GetEntryList(long *finalcount);
  // Gets the list of possible dial-up networking connections for the
  // user to select. - called in cliconfig

#if (CLIENT_OS == OS_LINUX) 
char ifacestowatch[100];
  // Used by Linux as a list of interfaces to watch for detecting online status
  // accept a ':' separated list of interface names, for example :
  // "ppp0:eth0:eth1"
  // "\0" means any interface (besides the loopback one)
#endif
 
int CheckIfConnectRequested(void); // -> 0=no, !0=yes
int CheckForStatusChange(void);    // -> 0 = nochange, !0 connection dropped
int DialIfNeeded(int ignore_lurkonly_flag); // -> 0=success, !0 = failure
int HangupIfNeeded(void);          // -> 0=success, !0 = failure
int Start(void);                   // Start lurking -> 0=success, !0 = failure
int Stop(void);                    // Stop lurking -> 0=success, !0 = failure

Lurk() { islurkstarted = lastcheckshowedconnect = 0; };  // Init lurk
~Lurk() {;}; // guess!

protected:

int IsConnected(void);      // Checks status of connection -> !0 = connected
int islurkstarted;          //was lurk.Start() successful?
int lastcheckshowedconnect; //the connect state at the last CheckIfConnectRequested()
int dohangupcontrol;        //if we dialed, we're welcome to hangup

};

extern Lurk dialup;

#endif //__LURK_H__
