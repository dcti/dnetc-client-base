/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __TRIGGERS_H__
#define __TRIGGERS_H__ "@(#)$Id: triggers.h,v 1.8 2000/06/02 06:25:01 jlawson Exp $"

#if defined(__unix__) && !defined(__EMX__)
  /* These constants define symbolically the signal names used by the
   * dnetc -pause / -unpause mechanism.  The idea is that the "usual"
   * alternatives, SIGTSTP and SIGCONT, aren't as portable as one
   * might like, so here we provide a mechanism to allow per-platform
   * selection of the appropriate signal pair.
   */
  #if (CLIENT_OS == OS_BEOS)
    /* Even though the BeOS build defines __unix__ for most purposes,
     * SIGCONT is reserved by the native thread suspend/resume mechanism
     * so we have to use something else */
    #define TRIGGER_PAUSE_SIGNAL SIGUSR1
    #define TRIGGER_UNPAUSE_SIGNAL SIGUSR2
  #else
    #define TRIGGER_PAUSE_SIGNAL SIGTSTP
    #define TRIGGER_UNPAUSE_SIGNAL SIGCONT
  #endif
#endif

//initialize... first call initializes the signal handler. args can be NULL
extern int InitializeTriggers(int doingmodes, const char *exitfile, 
                              const char *pausefile, const char *pauseplist,
                              int restartoninichange, const char *inifile,
                              int watchcputempthresh, const char *cputempthresh,
                              int pauseifnomainspower );
//deinitialize...
extern int DeinitializeTriggers(void);

//set the various triggers (don't worry: it doesn't use raise()). Note 
//that setting restart implies setting exit, and checking exit implies 
//checking restart.
extern int RaiseExitRequestTrigger(void);
extern int RaiseRestartRequestTrigger(void);
extern int RaisePauseRequestTrigger(void);

//the inverse of RaisePauseRequestTrigger(), which doesn't necessarily 
//clear all all pause flags. Also note that it is the only clearable
//trigger. (the restart trigger can only be cleared by restarting).
extern int ClearPauseRequestTrigger(void);

/* Check...() functions return a combination of one or more of these */
#define TRIGSETBY_SIGNAL       0x01 /* signal or explicit call to raise */ 
#define TRIGSETBY_FLAGFILE     0x02 /* flag file */
#define TRIGSETBY_CUSTOM       0x04 /* something other than the above */

//check/refresh the various triggers, poll external semaphore files or 
//platform-specific callouts as appropriate. Print messages as appropriate.
//These calls can be relatively slow and must be assumed to be thread unsafe.
extern int CheckExitRequestTrigger(void);
extern int CheckPauseRequestTrigger(void);
extern int CheckRestartRequestTrigger(void); //should only be used from 
                                             //do {main()} while (restart);

//same as above but without external I/O cycles, fast, and thread safe.
extern int CheckExitRequestTriggerNoIO(void);
extern int CheckPauseRequestTriggerNoIO(void);

//don't fire the next restart cycle caused by config file change
//won't prevent a restart if one has already been signalled
int OverrideNextConffileChangeTrigger(void);

#endif //__TRIGGERS_H__
