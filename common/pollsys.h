/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __POLLSYS_H__
#define __POLLSYS_H__ "@(#)$Id: pollsys.h,v 1.4 1999/04/06 10:20:48 cyp Exp $"

#include "clitime.h"  /* needed for timeval struct */

int DeinitializePolling(void);
int InitializePolling(void);

/*
 * RegPolledProcedure() adds a procedure to be called from the polling loop.
 * Procedures may *not* use 'sleep()' or 'usleep()' directly! (Its a stack 
 * issue, not a reentrancy problem). Procedures are automatically unregistered 
 * when called (they can re-register themselves). The 'interval' argument 
 * specifies how much time must elapse before the proc is scheduled to run - 
 * the default is {0,0}, ie schedule as soon as possible. Returns a non-zero 
 * handle on success or -1 if error. Care should be taken to ensure that
 * procedures registered with a high priority have an interval long enough
 * to allow procedures with a low(er) priority to run.
*/ 

int RegPolledProcedure( auto void (*proc)(void *), void *arg, 
                        struct timeval *interval, unsigned int priority );

/* 
 * UnregPolledProcedure() unregisters a procedure previously registered with
 * RegPolledProcedure(). Procedures are auto unregistered when executed.
*/   
int UnregPolledProcedure( int handle );

/* 
 * PolledSleep() and PolledUSleep() are automatic/default replacements for 
 * sleep() and usleep() (see sleepdef.h) and yield control to the polling 
 * process.
*/   
void PolledSleep( unsigned int seconds );
void PolledUSleep( unsigned int usecs );

/* 
 * NonPolledSleep() and NonPolledUSleep() are "real" sleepers. This are 
 * required for real threads (a la Go_mt()) that need to yield control to 
 * other threads.
*/
void NonPolledSleep( unsigned int seconds );
void NonPolledUSleep( unsigned int usecs );

#endif /* __POLLSYS_H__ */
