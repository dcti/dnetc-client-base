/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __TRIGGERS_H__
#define __TRIGGERS_H__ "@(#)$Id: triggers.h,v 1.6 1999/04/06 10:20:48 cyp Exp $"

#define PAUSEFILE_CHECKTIME_WHENON  (3)  //seconds
#define PAUSEFILE_CHECKTIME_WHENOFF (3*PAUSEFILE_CHECKTIME_WHENON)
#define EXITFILE_CHECKTIME          (PAUSEFILE_CHECKTIME_WHENOFF)

//initialize... first call initializes the signal handler. args can be NULL
extern int InitializeTriggers(const char *exitfile, const char *pausefile);

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

//Hooks whatever signals are needed to be hooked.
//Normally should NEVER be called, since initializetriggers
//handles this.  Win32 needs to reinit this when consoles
//are created, however.
extern void CliSetupSignals( void );


#endif //__TRIGGERS_H__
