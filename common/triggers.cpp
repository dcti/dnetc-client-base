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
return "@(#)$Id: triggers.cpp,v 1.16.2.23 2000/02/10 17:33:02 cyp Exp $"; }

/* ------------------------------------------------------------------------ */

#include "cputypes.h"
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "pathwork.h"  // GetFullPathForFilename()
#include "util.h"      // TRACE and utilxxx()
#include "logstuff.h"  // LogScreen()
#include "triggers.h"  // keep prototypes in sync

/* ----------------------------------------------------------------------- */

#define PAUSEFILE_CHECKTIME_WHENON  (3)  //seconds
#define PAUSEFILE_CHECKTIME_WHENOFF (3*PAUSEFILE_CHECKTIME_WHENON)
#define EXITFILE_CHECKTIME          (PAUSEFILE_CHECKTIME_WHENOFF)


#define TRIGSETBY_SIGNAL   0x1  /*signal or explicit call to raise */ 
#define TRIGSETBY_FLAGFILE 0x2  /*flag file */
#define TRIGSETBY_APPACTIV 0x4  /*pause due to a particular app being active*/
#define TRIGSETBY_CUSTOM   0x8  /* (platform specific) */

struct trigstruct 
{
  const char *flagfile; 
  struct { unsigned int whenon, whenoff; } pollinterval;
  unsigned int incheck; //recursion check
  void (*pollproc)(void);
  volatile int trigger; 
  int laststate;
  time_t nextcheck;
};

static struct 
{
  struct trigstruct exittrig;
  struct trigstruct pausetrig;
  struct trigstruct huptrig;
  char pausefilebuf[128]; /* includes path */
  char exitfilebuf[128];
  time_t nextinifilecheck;
  time_t currinifiletime;
  char inifile[128];
  char pauseplistbuffer[128];
  char *pauseplist[16];
  int lastactivep;
  int doingmodes;
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

static int __trig_raise(struct trigstruct *trig )
{
  int oldstate;
  __assert_statics();
  oldstate = trig->trigger;
  trig->trigger |= TRIGSETBY_SIGNAL;
  return oldstate;
}  

static int __trig_clear(struct trigstruct *trig )
{
  int oldstate;
  __assert_statics();
  oldstate = trig->trigger;
  trig->trigger &= ~TRIGSETBY_SIGNAL;
  return oldstate;
}

int RaiseExitRequestTrigger(void) 
{ return __trig_raise( &trigstatics.exittrig ); }
int RaiseRestartRequestTrigger(void) 
{ RaiseExitRequestTrigger(); return __trig_raise( &trigstatics.huptrig ); }
static int ClearRestartRequestTrigger(void) /* used internally */
{ return __trig_clear( &trigstatics.huptrig ); }
int RaisePauseRequestTrigger(void) 
{ return __trig_raise( &trigstatics.pausetrig ); }
int ClearPauseRequestTrigger(void)
{ 
  int oldstate = __trig_clear( &trigstatics.pausetrig );
  #if 0
  if ((trigstatics.pausetrig.trigger & TRIGSETBY_FLAGFILE)!=0 &&
      trigstatics.pausetrig.flagfile)
  {
    if (access( trigstatics.pausetrig.flagfile, 0 ) == 0)
    {
      unlink( trigstatics.pausetrig.flagfile );
      trigstatics.pausetrig.trigger &= ~TRIGSETBY_FLAGFILE;
    }
  }
  #endif  
  return oldstate;
}  
int CheckExitRequestTriggerNoIO(void) 
{ __assert_statics(); return (trigstatics.exittrig.trigger); } 
int CheckPauseRequestTriggerNoIO(void) 
{ __assert_statics(); return (trigstatics.pausetrig.trigger); }
int CheckRestartRequestTriggerNoIO(void) 
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
  if ((undoable || (trig->trigger & TRIGSETBY_FLAGFILE)==0) && trig->flagfile)
  {
    time_t now;
    if ((now = time(NULL)) >= trig->nextcheck) 
    {
      if ( access( trig->flagfile, 0 ) == 0 )
      {
        trig->nextcheck = now + (time_t)trig->pollinterval.whenon;
        trig->trigger |= TRIGSETBY_FLAGFILE;
      }
      else
      {
        trig->nextcheck = now + (time_t)trig->pollinterval.whenoff;
        trig->trigger &= ~TRIGSETBY_FLAGFILE;
      }
    }
  }
  return;
}

// -----------------------------------------------------------------------

static void __CheckIniFileChangeStuff(void)
{
  __assert_statics(); 
  if (trigstatics.inifile[0]) /* time_t */
  {
    time_t now = time(NULL);
    if (trigstatics.nextinifilecheck == 0 ||
        trigstatics.nextinifilecheck >= now)
    {
      time_t filetime = 0;
      struct stat statblk;
      if (stat( trigstatics.inifile, &statblk ) == 0)
        filetime = statblk.st_mtime;
      if (filetime)
      {
        if (!trigstatics.currinifiletime)                /* first time */
        {
          trigstatics.currinifiletime = filetime;
        }  
        else if (trigstatics.currinifiletime == ((time_t)1)) 
        {                                   /* got change some time ago */
          RaiseRestartRequestTrigger();
          trigstatics.currinifiletime = filetime;
        }
        else if (filetime != trigstatics.currinifiletime)
        {                                                /* mark change */
          trigstatics.currinifiletime = ((time_t)1);
        }
      }  
      trigstatics.nextinifilecheck = now + ((time_t)60); /* one minute */
    }
  }
  return;
}  

// -----------------------------------------------------------------------

int CheckExitRequestTrigger(void) 
{
  __assert_statics(); 
  if (!trigstatics.exittrig.laststate && !trigstatics.exittrig.incheck)
  {
    ++trigstatics.exittrig.incheck;
    if ( !trigstatics.exittrig.trigger )
    {
      if (trigstatics.exittrig.pollproc)
        (*trigstatics.exittrig.pollproc)();
    }
    if ( !trigstatics.exittrig.trigger )
      __PollExternalTrigger( &trigstatics.exittrig, 0 );
    if ( !trigstatics.exittrig.trigger )
      __CheckIniFileChangeStuff();
    if ( trigstatics.exittrig.trigger )
    {
      trigstatics.exittrig.laststate = 1;             
      if (!trigstatics.doingmodes)
      {
        LogScreen("*Break* %s\n",
          ( ((trigstatics.exittrig.trigger & TRIGSETBY_FLAGFILE)!=0)?
               ("(found exit flag file)"): 
           ((trigstatics.huptrig.trigger)?("Restarting..."):("Shutting down..."))
            ) );
      }
    }
    --trigstatics.exittrig.incheck;
  }
  return( trigstatics.exittrig.trigger );
}  

// -----------------------------------------------------------------------

int CheckRestartRequestTrigger(void) 
{ 
  CheckExitRequestTrigger(); /* does both */
  return (trigstatics.huptrig.trigger); 
}

// -----------------------------------------------------------------------

int CheckPauseRequestTrigger(void) 
{
  __assert_statics(); 
  if ( CheckExitRequestTrigger() )   //only check if not exiting
    return 0;
  if (trigstatics.doingmodes)
    return 0;
  if ( (++trigstatics.pausetrig.incheck) == 1 )
  {
    const char *custom_now_active = "";
    const char *custom_now_inactive = "";
    const char *app_now_active = "";
  
    #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16)
    {
      custom_now_active = "(defrag running)";
      custom_now_inactive = "(defrag no longer running)";
      if (FindWindow("MSDefragWClass1",NULL))
        trigstatics.pausetrig.trigger |= TRIGSETBY_CUSTOM;
      else
        trigstatics.pausetrig.trigger &= ~TRIGSETBY_CUSTOM;
    }
    #endif

    if (trigstatics.pauseplist[0])
    {
      int index, nowcleared = 0;
      char **pp = &trigstatics.pauseplist[0];
      long pidlist[1];
      if ((trigstatics.pausetrig.trigger & TRIGSETBY_APPACTIV) != 0)
      {
        /* this is a HACK for the sake of user convenience so 
           that there is no transition between pause->nopause->pause
        */
        index = trigstatics.lastactivep;
        if (!(utilGetPIDList( pp[index], &pidlist[0], 1 ) > 0))
        {
          Log("%s... ('%s' inactive)\n",
              (((trigstatics.pausetrig.laststate 
                   & ~TRIGSETBY_APPACTIV)!=0)?("Pause level lowered"):
              ("Running again after pause")), pp[index] );
          trigstatics.pausetrig.laststate &= ~TRIGSETBY_APPACTIV;
          trigstatics.pausetrig.trigger &= ~TRIGSETBY_APPACTIV;
          trigstatics.lastactivep = 0;
          nowcleared = (-(index+1));
        }
      }
      if ((trigstatics.pausetrig.trigger & TRIGSETBY_APPACTIV) == 0)
      {
        index = 0;
        while (pp[index])
        {
          if (nowcleared == (-(index+1)))
            ; /* already done above */
          else if (utilGetPIDList( pp[index], &pidlist[0], 1 ) > 0)
          {
            trigstatics.pausetrig.trigger |= TRIGSETBY_APPACTIV;
            trigstatics.lastactivep = index;
            app_now_active = pp[index];
            break;
          }
          index++;
        }
      }
    }
    
    __PollExternalTrigger( &trigstatics.pausetrig, 1 );
    if (trigstatics.pausetrig.laststate != trigstatics.pausetrig.trigger)
    {
      if ((trigstatics.pausetrig.trigger & TRIGSETBY_SIGNAL)!=0 &&
          (trigstatics.pausetrig.laststate & TRIGSETBY_SIGNAL)==0)
      {
        Log("Pause%sd... (user generated)\n",
             ((trigstatics.pausetrig.laststate)?(" level raise"):("")) );
        trigstatics.pausetrig.laststate |= TRIGSETBY_SIGNAL;
      }
      else if ((trigstatics.pausetrig.trigger & TRIGSETBY_SIGNAL)==0 &&
          (trigstatics.pausetrig.laststate & TRIGSETBY_SIGNAL)!=0)
      {
        trigstatics.pausetrig.laststate &= ~TRIGSETBY_SIGNAL;
        Log("%s... (user cleared)\n",
            ((trigstatics.pausetrig.laststate)?("Pause level lowered"):
            ("Running again after pause")) );
      }
      if ((trigstatics.pausetrig.trigger & TRIGSETBY_FLAGFILE)!=0 &&
          (trigstatics.pausetrig.laststate & TRIGSETBY_FLAGFILE)==0)
      {
        Log("Pause%sd... (found flagfile)\n",
              ((trigstatics.pausetrig.laststate)?(" level raise"):("")) );
        trigstatics.pausetrig.laststate |= TRIGSETBY_FLAGFILE;
      }
      else if ((trigstatics.pausetrig.trigger & TRIGSETBY_FLAGFILE)==0 &&
          (trigstatics.pausetrig.laststate & TRIGSETBY_FLAGFILE)!=0)
      {
        trigstatics.pausetrig.laststate &= ~TRIGSETBY_FLAGFILE;
        Log("%s... (flagfile cleared)\n",
          ((trigstatics.pausetrig.laststate)?("Pause level lowered"):
          ("Running again after pause")) );
      }
      if ((trigstatics.pausetrig.trigger & TRIGSETBY_CUSTOM)!=0 &&
          (trigstatics.pausetrig.laststate & TRIGSETBY_CUSTOM)==0)
      {
        Log("Pause%sd... %s\n",
             ((trigstatics.pausetrig.laststate)?(" level raise"):("")),
             custom_now_active );
        trigstatics.pausetrig.laststate |= TRIGSETBY_CUSTOM;
      }
      else if ((trigstatics.pausetrig.trigger & TRIGSETBY_CUSTOM)==0 &&
          (trigstatics.pausetrig.laststate & TRIGSETBY_CUSTOM)!=0)
      {
        trigstatics.pausetrig.laststate &= ~TRIGSETBY_CUSTOM;
        Log("%s... %s\n",
          ((trigstatics.pausetrig.laststate)?("Pause level lowered"):
          ("Running again after pause")), custom_now_inactive );
      }
      if ((trigstatics.pausetrig.trigger & TRIGSETBY_APPACTIV)!=0 &&
          (trigstatics.pausetrig.laststate & TRIGSETBY_APPACTIV)==0)
      {
        Log("Pause%sd... ('%s' active)\n",
             ((trigstatics.pausetrig.laststate)?(" level raise"):("")),
             ((*app_now_active)?(app_now_active):("process")) );
        trigstatics.pausetrig.laststate |= TRIGSETBY_APPACTIV;
      }
      else if ((trigstatics.pausetrig.trigger & TRIGSETBY_APPACTIV)==0 &&
          (trigstatics.pausetrig.laststate & TRIGSETBY_APPACTIV)!=0)
      {
        trigstatics.pausetrig.laststate &= ~TRIGSETBY_APPACTIV;
        //Log("%s... %s\n",
        //  ((trigstatics.pausetrig.laststate)?("Pause level lowered"):
        //  ("Running again after pause")), "(process no longer active)" );
      }
    }
  }
  --trigstatics.pausetrig.incheck;
  return( trigstatics.pausetrig.trigger );
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
static void __init_signal_handlers( int doingmodes )
{
  __assert_statics(); 
  doingmodes = doingmodes; /* unused */
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
static void __init_signal_handlers( int doingmodes ) 
{
  __assert_statics(); 
  doingmodes = doingmodes; /* unused */
  SetConsoleCtrlHandler( /*(PHANDLER_ROUTINE)*/CliSignalHandler, FALSE );
  SetConsoleCtrlHandler( /*(PHANDLER_ROUTINE)*/CliSignalHandler, TRUE );
  RegisterPollDrivenBreakCheck( __PollDrivenBreakCheck );
}
#endif

// -----------------------------------------------------------------------

#ifndef CLISIGHANDLER_IS_SPECIAL
#if (CLIENT_OS == OS_IRIX)
static void (*SETSIGNAL(int signo, void (*proc)(int)))(int)
{
  struct sigaction sa, osa;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sa.sa_handler = (void (*)())proc;
  //if (!sigismember(&_sigintr, signo))
    sa.sa_flags |= SA_RESTART;
  if (sigaction(signo, &sa, &osa) < 0)
    return (void (*)(int))(SIG_ERR);
  return (void (*)(int))(osa.sa_handler);
}
#else
#define SETSIGNAL signal
#endif

extern "C" void CliSignalHandler( int sig )
{
  #if defined(TRIGGER_PAUSE_SIGNAL)
  if (sig == TRIGGER_PAUSE_SIGNAL)
  {
    SETSIGNAL(sig,CliSignalHandler);
    RaisePauseRequestTrigger();
    return;
  }
  if (sig == TRIGGER_UNPAUSE_SIGNAL)
  {
    SETSIGNAL(sig,CliSignalHandler);
    ClearPauseRequestTrigger();
    return;
  }
  #endif  
  #if defined(SIGHUP)
  if (sig == SIGHUP)
  {
    SETSIGNAL(sig,CliSignalHandler);
    RaiseRestartRequestTrigger();
    return;
  }  
  #endif
  ClearRestartRequestTrigger();
  RaiseExitRequestTrigger();

  SETSIGNAL(sig,SIG_IGN);
}  
#endif //ifndef CLISIGHANDLER_IS_SPECIAL

// -----------------------------------------------------------------------

#ifndef CLISIGHANDLER_IS_SPECIAL
static void __init_signal_handlers( int doingmodes )
{
  __assert_statics(); 
  doingmodes = doingmodes; /* possibly unused */
  #if (CLIENT_OS == OS_SOLARIS)
  SETSIGNAL( SIGPIPE, SIG_IGN );
  #endif
  #if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_RISCOS)
  RegisterPollDrivenBreakCheck( __PollDrivenBreakCheck );
  #endif
  #if (CLIENT_OS == OS_DOS)
  break_on(); //break on any dos call, not just term i/o
  RegisterPollDrivenBreakCheck( __PollDrivenBreakCheck );
  #endif
  #if defined(SIGHUP)
  SETSIGNAL( SIGHUP, CliSignalHandler );   //restart
  #endif
  #if defined(TRIGGER_PAUSE_SIGNAL)  // signal-based pause/unpause mechanism?
  if (!doingmodes)
  {
    #if defined(__unix__) && (CLIENT_OS != OS_BEOS) && (CLIENT_OS != OS_NEXTSTEP)
    // stop the shell from seeing SIGTSTP and putting the client
    // into the background when we '-pause' it.
    // porters : those calls are POSIX.1, 
    // - on BSD you might need to change setpgid(0,0) to setpgrp()
    // - on SYSV you might need to change getpgrp() to getpgid(0)
    if( getpgrp() != getpid() )
      setpgid( 0, 0 );
    #endif
    SETSIGNAL( TRIGGER_PAUSE_SIGNAL, CliSignalHandler );  //pause
    SETSIGNAL( TRIGGER_UNPAUSE_SIGNAL, CliSignalHandler );  //continue
  }
  #endif
  #if defined(SIGQUIT)
  SETSIGNAL( SIGQUIT, CliSignalHandler );  //shutdown
  #endif
  #if defined(SIGSTOP)
  SETSIGNAL( SIGSTOP, CliSignalHandler );  //shutdown, maskable some places
  #endif
  #if defined(SIGABRT)
  SETSIGNAL( SIGABRT, CliSignalHandler );  //shutdown
  #endif
  #if defined(SIGBREAK)
  SETSIGNAL( SIGBREAK, CliSignalHandler ); //shutdown
  #endif
  SETSIGNAL( SIGTERM, CliSignalHandler );  //shutdown
  SETSIGNAL( SIGINT, CliSignalHandler );   //shutdown
}
#endif

// =======================================================================

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

static const char *_init_trigfile(const char *fn, char *buffer, unsigned int bufsize )
{
  if (buffer && bufsize)
  {
    if (fn)
    {
      while (*fn && isspace(*fn))
        fn++;
      if (*fn)  
      {
        unsigned int len = strlen(fn);
        while (len > 0 && isspace(fn[len-1]))
          len--;
        if (len && len < (bufsize-1))
        {
          strncpy( buffer, fn, len );
          buffer[len] = '\0';
          if (strcmp( buffer, "none" ) != 0)
          {
            fn = GetFullPathForFilename( buffer );
            if (fn)
            {
              if (strlen(fn) < (bufsize-1))
              {
                strcpy( buffer, fn );
                return buffer;
              }
            }
          }
        }
      }
    }
    buffer[0] = '\0';
  }
  return (const char *)0;
}

static void _init_pauseplist( const char *plist )
{
  const char *p = plist;
  unsigned int wpos = 0, index = 0;
  while (*p)
  {
    while (*p && (isspace(*p) || *p == '|'))
      p++;
    if (*p)
    {
      unsigned int len = 0;
      plist = p;
      while (*p && *p != '|')
      {
        p++;
        len++;
      }
      while (len > 0 && isspace(p[len-1]))
        len--;
      if (len)
      {
        if ((wpos + len) >= (sizeof(trigstatics.pauseplistbuffer) - 2))
        {
          break;
        }
        trigstatics.pauseplist[index++]=&(trigstatics.pauseplistbuffer[wpos]);
        memcpy( &(trigstatics.pauseplistbuffer[wpos]), plist, len );
        wpos += len;
        trigstatics.pauseplistbuffer[wpos++] = '\0';
        if (index == ((sizeof(trigstatics.pauseplist)/
                       sizeof(trigstatics.pauseplist[0]))-1) )
        {
          break;
        }
      }
    }
  }
  trigstatics.pauseplist[index] = (char *)0;
}


int InitializeTriggers(int doingmodes,
                       const char *exitfile, const char *pausefile,
                       const char *pauseplist,
                       int restartoninichange, const char *inifile )
{
  __assert_statics(); 
  memset( (void *)(&trigstatics), 0, sizeof(trigstatics) );
  trigstatics.exittrig.pollinterval.whenon = 0;
  trigstatics.exittrig.pollinterval.whenoff = EXITFILE_CHECKTIME;
  trigstatics.pausetrig.pollinterval.whenon = PAUSEFILE_CHECKTIME_WHENON;
  trigstatics.pausetrig.pollinterval.whenoff = PAUSEFILE_CHECKTIME_WHENOFF;
  __init_signal_handlers( doingmodes );

  trigstatics.doingmodes = doingmodes;
  if (!doingmodes)
  {
    trigstatics.exittrig.flagfile = _init_trigfile(exitfile, 
                 trigstatics.exitfilebuf, sizeof(trigstatics.exitfilebuf) );
    trigstatics.pausetrig.flagfile = _init_trigfile(pausefile, 
                 trigstatics.pausefilebuf, sizeof(trigstatics.pausefilebuf) );
    if (restartoninichange)
      _init_trigfile(inifile,trigstatics.inifile,sizeof(trigstatics.inifile));
    _init_pauseplist( pauseplist );
  }
  return (CheckExitRequestTrigger());
}  


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
