// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// This file contains functions for obtaining/formatting/manipulating 
// the time. 'time' is always stored/passed/returned in timeval format.
// 
// $Log: clitime.h,v $
// Revision 1.9  1998/07/28 11:44:54  blast
// Amiga specific changes
//
// Revision 1.8  1998/07/08 05:19:28  jlawson
// updates to get Borland C++ to compile under Win32.
//
// Revision 1.7  1998/07/07 21:55:32  cyruspatel
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
// Revision 1.6  1998/06/29 06:57:58  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.5  1998/06/14 08:12:46  friedbait
// 'Log' keywords added to maintain automatic change history
//
// 

#ifndef _CLITIME_H_
#define _CLITIME_H_
#if (CLIENT_OS == OS_AMIGAOS)
#include <sys/time.h> // To make it compile, deine from this file needed..
#endif

struct timeval;     // prototype


// Get the current time in timeval format (pass NULL if storage not req'd)
struct timeval *CliTimer( struct timeval *tv );

// Get time as string. Curr time if tv is NULL. Separate buffers for each
// type: 0=blank type 1, 1="MMM dd hh:mm:ss GMT", 2="hhhh:mm:ss.pp"
const char *CliGetTimeString( struct timeval *tv, int strtype );

// Get the time since program start (pass NULL if storage not required)
struct timeval *CliClock( struct timeval *tv );

// Add 'tv1' to 'tv2' and store in 'result'. Uses curr time if a 'tv' is NULL
// tv1/tv2 are not modified (unless 'result' is the same as one of them).
int CliTimerAdd( struct timeval *result, struct timeval *tv1, struct timeval *tv2 );

// Store non-negative diff of tv1 and tv2 in 'result'. Uses current time if a 'tv' is NULL
// tv1/tv2 are not modified (unless 'result' is the same as one of them).
int CliTimerDiff( struct timeval *result, struct timeval *tv1, struct timeval *tv2 );

#endif //ifndef _CLITIME_H_
