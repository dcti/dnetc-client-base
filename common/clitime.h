/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This module contains functions for obtaining/formatting/manipulating 
 * the time. 
 * ----------------------------------------------------------------------
 *
*/ 
#ifndef __CLITIME_H__
#define __CLITIME_H__ "@(#)$Id: clitime.h,v 1.22 1999/11/08 02:02:38 cyp Exp $"

#include "baseincs.h" /* struct timeval */

// Get year-round (ie after compensating for DST) TZ offset in minutes
int CliTimeGetMinutesWest(void);

// Get the current time in timeval format (pass NULL if storage not req'd)
struct timeval *CliTimer( struct timeval *tv );

// Get monotonic millisecs. Used when timing with CliTimer() aint so hot 
unsigned long CliMillisecClock(void);

// Set the 'time delta', a value added to the tv_sec member by CliTimer()
// before it the time is returned. CliTimerSetDelta() returns the old delta.
int CliTimerSetDelta( int delta );

// Get time as string. Curr time if tv is NULL. Separate buffers for 
// each type: See source for valid types.
const char *CliGetTimeString( const struct timeval *tv, int strtype );

// Get the time since program start (pass NULL if storage not required)
struct timeval *CliClock( struct timeval *tv );

// Add 'tv1' to 'tv2' and store in 'result'. Uses curr time if a 'tv' is NULL
// tv1/tv2 are not modified (unless 'result' is the same as one of them).
int CliTimerAdd( struct timeval *result, const struct timeval *tv1, const struct timeval *tv2 );

// Store non-negative diff of tv1 and tv2 in 'result'. Uses current time if a 'tv' is NULL
// tv1/tv2 are not modified (unless 'result' is the same as one of them).
int CliTimerDiff( struct timeval *result, const struct timeval *tv1, const struct timeval *tv2 );

// do we have a valid timezone to work with? 
// (currently supported by DOS,WIN[16],OS/2 only)
int CliIsTimeZoneInvalid(void);

#endif /* __CLITIME_H__ */
