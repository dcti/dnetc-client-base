/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __TRIGGERS_H__
#define __TRIGGERS_H__ "@(#)$Id: triggers.h,v 1.6.2.2 2000/01/22 00:56:34 ctate Exp $"

#include "baseincs.h"

/* These constants define symbolically the signal names used by the
 * dnetc -pause / -unpause mechanism.  The idea is that the "usual"
 * alternatives, SIGTSTP and SIGCONT, aren't as portable as one
 * might like, so here we provide a mechanism to allow per-platform
 * selection of the appropriate signal pair.
 *
 * Currently these defines are only used for __unix__ platforms, as
 * well as BeOS because it acts like Unix for most CLI purposes.
 */

#if (CLIENT_OS == OS_BEOS)
  /* Even though the BeOS build defines __unix__ for most purposes,
   * SIGCONT is reserved by the native thread suspend/resume mechanism
   * so we have to use something else */
  #define TRIGGER_PAUSE_SIGNAL SIGUSR1
  #define TRIGGER_UNPAUSE_SIGNAL SIGUSR2
#elif defined(__unix__)
  /* the usual Unix way to handle things like this is SIGTSTP/SIGCONT */
  #define TRIGGER_PAUSE_SIGNAL SIGTSTP
  #define TRIGGER_UNPAUSE_SIGNAL SIGCONT
#endif

//initialize... first call initializes the signal handler. args can be NULL
extern int InitializeTriggers(int doingmodes, 
                              const char *exitfile, const char *pausefile);

//deinitialize...
extern int DeinitializeTriggers(void);

//set the exit trigger ONLY (don't worry: it doesn't use raise())
extern int RaiseExitRequestTrigger(void); 

//set the restart AND exit triggers
extern int RaiseRestartRequestTrigger(void); 

//set the pause request trigger
extern int RaisePauseRequestTrigger(void);

//refresh/get the exit trigger state
//preferred method for main thread 
extern int CheckExitRequestTrigger(void); 

//refresh/get the pause trigger state
//preferred method for main thread 
extern int CheckPauseRequestTrigger(void); 

//clear the pause request trigger
extern int ClearPauseRequestTrigger(void);

//refresh/get the restart trigger state
//implemented as do { main() } while CheckRestartRequestTrigger
extern int CheckRestartRequestTrigger(void); 

//just return the exit trigger state (no poll cycle) 
//preferred method for child threads 
extern int CheckExitRequestTriggerNoIO(void); 

//just return the pause trigger state (no poll cycle)
//preferred method for child threads 
extern int CheckPauseRequestTriggerNoIO(void); 

#endif //__TRIGGERS_H__
