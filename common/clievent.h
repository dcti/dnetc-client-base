/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * 
 * Event id's are composed of a subsystem id and a running number.
 * If you add a new event id, _document_ it.                 -cyp
*/

#ifndef __CLIEVENT_H__
#define __CLIEVENT_H__ "@(#)$Id: clievent.h,v 1.9 1999/04/06 10:20:47 cyp Exp $"

                                              /* parm is ... */
#define CLIEVENT_CLIENT_STARTED        0x0001 /* ...client ptr */
#define CLIEVENT_CLIENT_FINISHED       0x0002 /* ...restarting flag */
#define CLIEVENT_CLIENT_RUNSTARTED     0x0003 /* ...0 */
#define CLIEVENT_CLIENT_RUNFINISHED    0x0004 /* ...0 */
#define CLIEVENT_PROBLEM_STARTED       0x0101 /* ...problem id */
#define CLIEVENT_PROBLEM_FINISHED      0x0102 /* ...problem id */
#define CLIEVENT_PROBLEM_TFILLSTARTED  0x0103 /* ...# of problems to check */
#define CLIEVENT_PROBLEM_TFILLFINISHED 0x0104 /* ...# of problems changed */
#define CLIEVENT_BUFFER_FETCHBEGIN     0x0201 /* ...(long)(&proxymsg) */
#define CLIEVENT_BUFFER_FETCHFETCHED   0x0202 /* ...(long)(&Fetch_Flush_Info) */
#define CLIEVENT_BUFFER_FETCHEND       0x0203 /* ...total fetched */
#define CLIEVENT_BUFFER_FLUSHBEGIN     0x0204 /* ...(long)(&proxymsg) */
#define CLIEVENT_BUFFER_FLUSHFLUSHED   0x0205 /* ...(long)(&Fetch_Flush_Info) */
#define CLIEVENT_BUFFER_FLUSHEND       0x0206 /* ...total fetched */
#define CLIEVENT_SELFTEST_STARTED      0x0301 /* ...contest id */
#define CLIEVENT_SELFTEST_TESTBEGIN    0x0302 /* ...problem ptr */
#define CLIEVENT_SELFTEST_TESTEND      0x0303 /* ...passed/!0 or failed/0 */
#define CLIEVENT_SELFTEST_FINISHED     0x0304 /* ...[+|-]successes */
#define CLIEVENT_BENCHMARK_STARTED     0x0401 /* ...problem ptr */
#define CLIEVENT_BENCHMARK_BENCHING    0x0402 /* ...problem ptr */
#define CLIEVENT_BENCHMARK_FINISHED    0x0403 /* ...rate (double ptr) or NULL*/

/* 
  Structures defined for used as parameters
*/

struct Fetch_Flush_Info {
  unsigned long contest;
  unsigned long contesttrans;	// blocks transferred for specified contest
  unsigned long combinedtrans;	// blocks transferred for all contests
};
  
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
