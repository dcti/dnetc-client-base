/* Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * This module contains functions for raising/checking flags normally set
 * (asynchronously) by user request. Encapsulating the flags in 
 * functions has two benefits: (1) Transparency: the caller doesn't 
 * (_shouldn't_) need to care whether the triggers are async from signals
 * or polled. (2) Portability: we don't need a bunch of #if (CLIENT_OS...) 
 * sections preceding every signal variable check. As such, someone writing 
 * new code doesn't need to ensure that someone else's signal handling isn't 
 * affected, and inversely, that coder doesn't need to check if his platform 
 * is affected by every itty-bitty change. (3) Modularity: gawd knows we need
 * some of this. (4) Extensibility: hup, two, three, four...  - cyp
*/   

const char *triggers_cpp(void) {
return "@(#)$Id: triggers.cpp,v 1.23 1999/12/02 05:25:24 cyp Exp $"; }

/* ------------------------------------------------------------------------ */

#include "cputypes.h"
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "pathwork.h"  // GetFullPathForFilename()
#include "logstuff.h"  // LogScreen()
#include "triggers.h"  // for xxx_CHECKTIME defines

/* ----------------------------------------------------------------------- */

#define TRIGSETBY_INTERNAL  1  /* signal or explicit call to raise */ 
#define TRIGSETBY_EXTERNAL  2  /* flag file */

struct trigstruct 
{
  const char *flagfile; 
  struct { unsigned int whenon, whenoff; } pollinterval;
  unsigned int incheck; //recursion check
  void (*pollproc)(void);
  volatile int trigger; 
  time_t nextcheck;
};

static struct 
{
  int exitmsgwasseen;
  struct trigstruct exittrig;
  struct trigstruct pausetrig;
  struct trigstruct huptrig;
  char pausefilebuf[64];
  char exitfilebuf[64];
} trigstatics;

static void __assert_statics(void)
{
  static int initialized = -1;
  if (initialized == -1)
  {
    memset( &trigstatics, 0, sizeof(trigstatics) );
    initialized = +1;
  }
}

// -----------------------------------------------------------------------

int RaiseExitRequestTrigger(void) 
{ 
  int oldstate;
  __assert_statics();
  oldstate = trigstatics.exittrig.trigger;
  trigstatics.exittrig.trigger = TRIGSETBY_INTERNAL;
  return (oldstate);
}  

int RaiseRestartRequestTrigger(void) 
{ 
  int oldstate;
  __assert_statics();
  oldstate = trigstatics.huptrig.trigger;
  trigstatics.exittrig.trigger = TRIGSETBY_INTERNAL;
  trigstatics.huptrig.trigger = TRIGSETBY_INTERNAL;
  return (oldstate);
}  

static int ClearRestartRequestTrigger(void) /* used internally */
{
  int oldstate;
  __assert_statics();
  oldstate = trigstatics.huptrig.trigger;
  trigstatics.huptrig.trigger = 0;
  return oldstate;
}  

int RaisePauseRequestTrigger(void) 
{ 
  int oldstate;
  __assert_statics();
  oldstate = trigstatics.pausetrig.trigger;
  trigstatics.pausetrig.trigger = TRIGSETBY_INTERNAL;
  return (oldstate);
}  

int ClearPauseRequestTrigger(void)
{
  int oldstate;
  __assert_statics();
  if ( trigstatics.pausetrig.flagfile && 
    access( GetFullPathForFilename( trigstatics.pausetrig.flagfile ),0)==0)
    unlink( GetFullPathForFilename( trigstatics.pausetrig.flagfile ) );
  oldstate = trigstatics.pausetrig.trigger;
  trigstatics.pausetrig.trigger = 0;
  return oldstate;
}  

int CheckExitRequestTriggerNoIO(void) 
{ __assert_statics(); return (trigstatics.exittrig.trigger); } 
int CheckPauseRequestTriggerNoIO(void) 
{ __assert_statics(); return (trigstatics.pausetrig.trigger); }
int CheckRestartRequestTriggerNoIO(void) 
{ __assert_statics(); return (trigstatics.huptrig.trigger); }
int CheckRestartRequestTrigger(void) 
{ __assert_statics(); return (trigstatics.huptrig.trigger); }

// -----------------------------------------------------------------------

void *RegisterPollDrivenBreakCheck( register void (*proc)(void) )
{
  register void (*oldproc)(void);
  __assert_statics(); 
  oldproc = trigstatics.exittrig.pollproc;
  trigstatics.exittrig.pollproc = proc;
  return (void *)oldproc;
}

// -----------------------------------------------------------------------

static void __PollExternalTrigger(struct trigstruct *trig, int undoable)
{
  __assert_statics(); 
  #if (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN32S)
  // we treat a running defrag as another flagfile (we need to ensure
  // that two 'external' type checks don't cancel each other out.)
  if (trig == &trigstatics.pausetrig)
  {
    static int defrag_set_it = 0;
    if (FindWindow("MSDefragWClass1",NULL))
    {
      if (!defrag_set_it)
      {
        defrag_set_it = 1;
        if (trig->trigger) /*was already paused */
          defrag_set_it++;
        trig->trigger = TRIGSETBY_EXTERNAL; 
        Log("Found defrag to be running.%s..\n",
           ((defrag_set_it>1)?(" Pause level raised."):("")) );
      }
      return;
    } 
    else if (defrag_set_it) /* we set it, we clear it */
    {
      Log("Defrag is no longer running.%s..\n",
          ((defrag_set_it>1)?(" Pause level lowered."):("")) );
      if (defrag_set_it < 2) /* it wasn't paused before */ 
        trig->trigger = 0;
      defrag_set_it = 0;
    }
  }
  #endif
  if ((undoable || !trig->trigger) && trig->flagfile)
  {
    time_t now;
    if ((now = time(NULL)) >= trig->nextcheck) 
    {
      if ( access( GetFullPathForFilename( trig->flagfile ), 0 ) == 0 )
      {
        trig->nextcheck = now + (time_t)trig->pollinterval.whenon;
        trig->trigger = TRIGSETBY_EXTERNAL;
      }
      else
      {
        trig->nextcheck = now + (time_t)trig->pollinterval.whenoff;
        trig->trigger = 0;
      }
    }
  }
  return;
}

// -----------------------------------------------------------------------

int CheckExitRequestTrigger(void) 
{
  __assert_statics(); 
  if (!trigstatics.exitmsgwasseen && !trigstatics.exittrig.incheck)
  {
    ++trigstatics.exittrig.incheck;
    if ( !trigstatics.exittrig.trigger )
    {
      if (trigstatics.exittrig.pollproc)
        (*trigstatics.exittrig.pollproc)();
    }
    if ( !trigstatics.exittrig.trigger )
      __PollExternalTrigger( &trigstatics.exittrig, 0 );
    if ( trigstatics.exittrig.trigger )
    {
      LogScreen("*Break*%s\n", 
       (trigstatics.exittrig.trigger == TRIGSETBY_EXTERNAL)?
         (" (found exit flag file)"): 
         ((trigstatics.huptrig.trigger)?(" Restarting..."):
         (" Shutting down...")) );
      trigstatics.exitmsgwasseen = 1;             
    }
    --trigstatics.exittrig.incheck;
  }
  return( trigstatics.exittrig.trigger );
}  

// -----------------------------------------------------------------------

int CheckPauseRequestTrigger(void) 
{
  __assert_statics(); 
  if ( CheckExitRequestTrigger() )   //only check if not exiting
    return 0;
  if ( !trigstatics.pausetrig.incheck && 
       trigstatics.pausetrig.trigger != TRIGSETBY_INTERNAL )
  {
    ++trigstatics.pausetrig.incheck;
    __PollExternalTrigger( &trigstatics.pausetrig, 1 );
    --trigstatics.pausetrig.incheck;
  }
  return( trigstatics.pausetrig.trigger );
}   

// -----------------------------------------------------------------------

int DeinitializeTriggers(void)
{
  int huptrig;
  __assert_statics(); 
  huptrig = trigstatics.huptrig.trigger;
  /* clear everything to ensure we don't use IO after DeInit */
  memset( (void *)(&trigstatics), 0, sizeof(trigstatics) );
  return huptrig;
}  

// -----------------------------------------------------------------------

int InitializeTriggers( const char *exitfile, const char *pausefile )
{
  __assert_statics(); 
  memset( (void *)(&trigstatics), 0, sizeof(trigstatics) );
  trigstatics.exittrig.pollinterval.whenon = 0;
  trigstatics.exittrig.pollinterval.whenoff = EXITFILE_CHECKTIME;
  trigstatics.pausetrig.pollinterval.whenon = PAUSEFILE_CHECKTIME_WHENON;
  trigstatics.pausetrig.pollinterval.whenoff = PAUSEFILE_CHECKTIME_WHENOFF;
  CliSetupSignals();

  if (exitfile)
  {
    trigstatics.exittrig.flagfile = NULL;
    while (*exitfile && isspace(*exitfile))
      exitfile++;
    strncpy(trigstatics.exitfilebuf,exitfile,sizeof(trigstatics.exitfilebuf)-1);
    trigstatics.exitfilebuf[sizeof(trigstatics.exitfilebuf)-1]=0;
    unsigned int len=strlen(trigstatics.exitfilebuf);
    while (len > 0 && isspace(trigstatics.exitfilebuf[len-1]))
      trigstatics.exitfilebuf[--len]=0;
    if (len > 0 && strcmp(trigstatics.exitfilebuf,"none")!=0)
      trigstatics.exittrig.flagfile = trigstatics.exitfilebuf;
  }

  if (pausefile)
  {
    trigstatics.pausetrig.flagfile = NULL;
    while (*pausefile && isspace(*pausefile))
      pausefile++;
    strncpy(trigstatics.pausefilebuf,pausefile,sizeof(trigstatics.pausefilebuf)-1);
    trigstatics.pausefilebuf[sizeof(trigstatics.pausefilebuf)-1]=0;
    unsigned int len=strlen(trigstatics.pausefilebuf);
    while (len > 0 && isspace(trigstatics.pausefilebuf[len-1]))
      trigstatics.pausefilebuf[--len]=0;
    if (len > 0 && strcmp(trigstatics.pausefilebuf,"none")!=0)
      trigstatics.pausetrig.flagfile = trigstatics.pausefilebuf;
  }
  return (CheckExitRequestTrigger());
}  

// =======================================================================

void __PollDrivenBreakCheck( void ) /* not static to avoid compiler warnings */
{
  #if (CLIENT_OS == OS_RISCOS)
  if (_kernel_escape_seen())
      RaiseExitRequestTrigger();
  #elif (CLIENT_OS == OS_AMIGAOS)
  if ( SetSignal(0L,0L) & SIGBREAKF_CTRL_C )
    RaiseExitRequestTrigger();
  #elif (CLIENT_OS == OS_NETWARE)
    nwCliCheckForUserBreak(); //in nwccons.cpp
  #elif (CLIENT_OS == OS_WIN32)
    w32ConOut("");    /* benign call to keep ^C handling alive */
  #elif (CLIENT_OS == OS_DOS)
    _asm mov ah,0x0b  /* benign dos call (kbhit()) */
    _asm int 0x21     /* to keep int23h (^C) handling alive */
  #endif
  return;  
}      

// =======================================================================

#if (CLIENT_OS == OS_AMIGAOS)
extern "C" void __regargs __chkabort(void) 
{ 
  /* Disable SAS/C CTRL-C handing */
  return;
}
#define CLISIGHANDLER_IS_SPECIAL
void CliSetupSignals( void )
{
  __assert_statics(); 
  SetSignal(0L, SIGBREAKF_CTRL_C); // Clear the signal triggers
  RegisterPollDrivenBreakCheck( __PollDrivenBreakCheck );
}    
#endif

// -----------------------------------------------------------------------

#if (CLIENT_OS == OS_WIN32)
BOOL WINAPI CliSignalHandler(DWORD dwCtrlType)
{
  if ( dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_CLOSE_EVENT || 
       dwCtrlType == CTRL_SHUTDOWN_EVENT)
  {
    RaiseExitRequestTrigger();
    return TRUE;
  }
  else if (dwCtrlType == CTRL_BREAK_EVENT)
  {
    RaiseRestartRequestTrigger();
    return TRUE;
  }
  return FALSE;
}
#define CLISIGHANDLER_IS_SPECIAL
void CliSetupSignals( void ) 
{
  __assert_statics(); 
  SetConsoleCtrlHandler( /*(PHANDLER_ROUTINE)*/CliSignalHandler, FALSE );
  SetConsoleCtrlHandler( /*(PHANDLER_ROUTINE)*/CliSignalHandler, TRUE );
  RegisterPollDrivenBreakCheck( __PollDrivenBreakCheck );
}
#endif

// -----------------------------------------------------------------------

#ifndef CLISIGHANDLER_IS_SPECIAL
extern "C" void CliSignalHandler( int sig )
{
  #if defined(SIGTSTP) && defined(SIGCONT) //&& defined(__unix__)
  if (sig == SIGTSTP)
  {
    signal(sig,CliSignalHandler);
    RaisePauseRequestTrigger();
    return;
  }
  if (sig == SIGCONT)
  {
    signal(sig,CliSignalHandler);
    ClearPauseRequestTrigger();
    return;
  }
  #endif  
  #if defined(SIGHUP)
  if (sig == SIGHUP)
  {
    signal(sig,CliSignalHandler);
    RaiseRestartRequestTrigger();
    return;
  }  
  #endif
  ClearRestartRequestTrigger();
  RaiseExitRequestTrigger();

  signal(sig,SIG_IGN);
}  
#endif //ifndef CLISIGHANDLER_IS_SPECIAL

// -----------------------------------------------------------------------

#ifndef CLISIGHANDLER_IS_SPECIAL
void CliSetupSignals( void )
{
  __assert_statics(); 
  #if (CLIENT_OS == OS_SOLARIS)
  signal( SIGPIPE, SIG_IGN );
  #endif
  #if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_RISCOS)
  RegisterPollDrivenBreakCheck( __PollDrivenBreakCheck );
  #endif
  #if (CLIENT_OS == OS_DOS)
  break_on(); //break on any dos call, not just term i/o
  RegisterPollDrivenBreakCheck( __PollDrivenBreakCheck );
  #endif
  #if defined(SIGHUP)
  signal( SIGHUP, CliSignalHandler );   //restart
  #endif
  #if defined(SIGCONT) && defined(SIGTSTP)
  #if defined(__unix__)
  // stop the shell from seeing SIGTSTP and putting the client
  // into the background when we '-pause' it.
  // porters : those calls are POSIX.1, 
  // - on BSD you might need to change setpgid(0,0) to setpgrp()
  // - on SYSV you might need to change getpgrp() to getpgid(0)
  if( getpgrp() != getpid() )
    setpgid( 0, 0 );
  #endif
  signal( SIGTSTP, CliSignalHandler );  //pause
  signal( SIGCONT, CliSignalHandler );  //continue
  #endif
  #if defined(SIGQUIT)
  signal( SIGQUIT, CliSignalHandler );  //shutdown
  #endif
  #if defined(SIGSTOP)
  signal( SIGSTOP, CliSignalHandler );  //shutdown, maskable some places
  #endif
  #if defined(SIGABRT)
  signal( SIGABRT, CliSignalHandler );  //shutdown
  #endif
  #if defined(SIGBREAK)
  signal( SIGBREAK, CliSignalHandler ); //shutdown
  #endif
  signal( SIGTERM, CliSignalHandler );  //shutdown
  signal( SIGINT, CliSignalHandler );   //shutdown
}
#endif

// -----------------------------------------------------------------------

#if (CLIENT_OS == OS_FREEBSD)
#include <sys/mman.h>
int TBF_MakeTriggersVMInheritable(void)
{
  int mflag = 0; /*VM_INHERIT_SHARE*/ /*MAP_SHARED|MAP_INHERIT*/;
  if (minherit((void*)&trigstatics,sizeof(trigstatics),mflag)!=0)
    return -1;
  return 0;
}  
#endif
