// Hey, Emacs, this a -*-C++-*- file !

// Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
//
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
/* 
 * This is a trivial event handling mechanism for clients that do
 * something when a special condition arises.
 *
 * Refer to clievent.cpp for more documentation.
 *
 */ 
// $Log: clievent.h,v $
// Revision 1.1  1998/12/28 18:16:19  cyp
// Created.
//
//
//

#ifndef __CLIEVENT_H__
#define __CLIEVENT_H__

/* Event id's are composed of a subsystem id and a running number.
   If you add a new event id, _document_ it. */

                                             /* parm is ... */
#define CLIEVENT_PROBLEM_STARTED     0x0101  /* ... problem id */
#define CLIEVENT_PROBLEM_FINISHED    0x0102  /* ... problem id */
#define CLIEVENT_BUFFER_FETCHBEGIN   0x0201  /* ... (long)(&proxymsg) */
#define CLIEVENT_BUFFER_FETCHFETCHED 0x0202  /* ... sequence # */
#define CLIEVENT_BUFFER_FETCHEND     0x0203  /* ... total fetched */
#define CLIEVENT_BUFFER_FLUSHBEGIN   0x0204  /* ... (long)(&proxymsg) */
#define CLIEVENT_BUFFER_FLUSHFLUSHED 0x0205  /* ... sequence # */
#define CLIEVENT_BUFFER_FLUSHEND     0x0206  /* ... total fetched */

/*
      #if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
      FinishThreadProgress(prob_i, rc5result.iterations);
      #endif
*/


/* add a procedure that will be called when a particular event occurs */
int ClientEventAddListener(int event_id, void (*proc)(int event_id, long parm));

/* remove a procedure from the listen queue */
int ClientEventRemoveListener(int event_id, void (*proc)(int event_id, long parm));

/* post an event. returns number of listeners that the message was posted to */
int ClientEventSyncPost( int event_id, long parm );

#endif
