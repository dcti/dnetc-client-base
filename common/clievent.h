// Hey, Emacs, this a -*-C++-*- file !
//
// Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
//
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
/* 
 * Refer to clievent.cpp for documentation.
 */ 
//
// $Log: clievent.h,v $
// Revision 1.4  1998/12/29 20:36:23  silby
// Added new event to cause GUIs to update their percent bars.
// (Assumes percent bar routines have their own logic
// similar to logscreen_percent.)
//
// Revision 1.3  1998/12/29 19:18:24  cyp
// Added ..._PROBLEM_TFILL[STARTED|FINISHED] for client.LoadSaveProblems()
//
// Revision 1.2  1998/12/28 21:06:54  cyp
// Added event types for benchmark and selftest.
//
// Revision 1.1  1998/12/28 18:16:19  cyp
// Created.
//

#ifndef __CLIEVENT_H__
#define __CLIEVENT_H__

/* Event id's are composed of a subsystem id and a running number.
   If you add a new event id, _document_ it. */

                                              /* parm is ... */
#define CLIEVENT_CLIENT_STARTED        0x0001 /* ...client ptr */
#define CLIEVENT_CLIENT_FINISHED       0x0002 /* ...restarting flag */
#define CLIEVENT_CLIENT_RUNSTARTED     0x0003 /* ...0 */
#define CLIEVENT_CLIENT_RUNFINISHED    0x0004 /* ...0 */
#define CLIEVENT_PROBLEM_STARTED       0x0101 /* ...problem id */
#define CLIEVENT_PROBLEM_FINISHED      0x0102 /* ...problem id */
#define CLIEVENT_PROBLEM_TFILLSTARTED  0x0103 /* ...# of problems to check */
#define CLIEVENT_PROBLEM_TFILLFINISHED 0x0104 /* ...# of problems changed */
#define CLIEVENT_PROBLEM_CONTINUED     0x0105 /* ...0 (update % bar)*/
#define CLIEVENT_BUFFER_FETCHBEGIN     0x0201 /* ...(long)(&proxymsg) */
#define CLIEVENT_BUFFER_FETCHFETCHED   0x0202 /* ...sequence # */
#define CLIEVENT_BUFFER_FETCHEND       0x0203 /* ...total fetched */
#define CLIEVENT_BUFFER_FLUSHBEGIN     0x0204 /* ...(long)(&proxymsg) */
#define CLIEVENT_BUFFER_FLUSHFLUSHED   0x0205 /* ...sequence # */
#define CLIEVENT_BUFFER_FLUSHEND       0x0206 /* ...total fetched */
#define CLIEVENT_SELFTEST_STARTED      0x0301 /* ...contest id */
#define CLIEVENT_SELFTEST_TESTBEGIN    0x0302 /* ...problem ptr */
#define CLIEVENT_SELFTEST_TESTEND      0x0303 /* ...passed/!0 or failed/0 */
#define CLIEVENT_SELFTEST_FINISHED     0x0304 /* ...[+|-]successes */
#define CLIEVENT_BENCHMARK_STARTED     0x0401 /* ...problem ptr */
#define CLIEVENT_BENCHMARK_BENCHING    0x0402 /* ...problem ptr */
#define CLIEVENT_BENCHMARK_FINISHED    0x0403 /* ...rate (double ptr) or NULL*/


/*
  #if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
        NewProxyMessage(proxymessage);
  #endif     
  #if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
      FinishThreadProgress(prob_i, rc5result.iterations);
  #endif
  #if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
        StartActiveFetch();
  #endif
  #if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
        EndActiveFetch();
  #endif
  #if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
        StartActiveFlush();
  #endif
  #if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
        EndActiveFlush();
  #endif
*/


/* add a procedure that will be called when a particular event occurs */
int ClientEventAddListener(int event_id, void (*proc)(int event_id, long parm));

/* remove a procedure from the listen queue */
int ClientEventRemoveListener(int event_id, void (*proc)(int event_id, long parm));

/* post an event. returns number of listeners that the message was posted to */
int ClientEventSyncPost( int event_id, long parm );

#endif
