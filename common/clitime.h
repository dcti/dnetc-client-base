// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// ----------------------------------------------------------------------
// This file contains functions for obtaining/formatting/manipulating 
// the time. 'time' is always stored/passed/returned in timeval format.
// ----------------------------------------------------------------------
// 
// $Log: clitime.h,v $
// Revision 1.17  1999/03/31 11:41:39  cyp
// a) lots of const. b) added #error where OS support was missing.
//
// Revision 1.16  1999/03/18 03:11:25  cyp
// New function CliTimeGetBuildDate() returns build time_t. Used to check
// that time obtained from proxy is (somewhat) sane.
//
// Revision 1.15  1999/03/03 04:29:36  cyp
// created CliTimeGetMinutesWest() and CliTimerSetDelta(). See .h for descrip.
//
// Revision 1.14  1999/01/29 19:02:22  jlawson
// fixed formatting.
//
// Revision 1.13  1999/01/19 09:36:58  patrick
// OS2-EMX needs sys/time.h
//
// Revision 1.12  1999/01/14 23:02:12  pct
// Updates for Digital Unix alpha client and ev5 related code.  This also
// includes inital code for autodetection of CPU type and SMP.
//
// Revision 1.11  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.10  1998/11/25 06:02:46  dicamillo
// Update because BeOS also needs "sys/time.h".
//
// Revision 1.9  1998/07/28 11:44:54  blast
// Amiga specific changes
//
// Revision 1.8  1998/07/08 05:19:28  jlawson
// updates to get Borland C++ to compile under Win32.
//
// Revision 1.7  1998/07/07 21:55:32  cyruspatel
// client.h has been split into client.h and baseincs.h 
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

#include "baseincs.h" /* struct timeval */


// Get year-round (ie after compensating for DST) TZ offset in minutes
int CliTimeGetMinutesWest(void);

// Get the current time in timeval format (pass NULL if storage not req'd)
struct timeval *CliTimer( struct timeval *tv );

// Set the 'time delta', a value added to the tv_sec member by CliTimer()
// before it the time is returned. CliTimerSetDelta() returns the old delta.
int CliTimerSetDelta( int delta );

// Get Date/Time this module was built. Used, for instance, to 'ensure' 
// that time from the .ini or recvd from a proxy is sane.
time_t CliTimeGetBuildDate(void);

// Get time as string. Curr time if tv is NULL. Separate buffers for each
// type: 0=blank type 1, 1="MMM dd hh:mm:ss GMT", 2="hhhh:mm:ss.pp"
const char *CliGetTimeString( const struct timeval *tv, int strtype );

// Get the time since program start (pass NULL if storage not required)
struct timeval *CliClock( struct timeval *tv );

// Add 'tv1' to 'tv2' and store in 'result'. Uses curr time if a 'tv' is NULL
// tv1/tv2 are not modified (unless 'result' is the same as one of them).
int CliTimerAdd( struct timeval *result, const struct timeval *tv1, const struct timeval *tv2 );

// Store non-negative diff of tv1 and tv2 in 'result'. Uses current time if a 'tv' is NULL
// tv1/tv2 are not modified (unless 'result' is the same as one of them).
int CliTimerDiff( struct timeval *result, const struct timeval *tv1, const struct timeval *tv2 );

#endif //ifndef _CLITIME_H_
