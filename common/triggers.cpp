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
return "@(#)$Id: triggers.cpp,v 1.16.2.51 2000/09/17 11:46:34 cyp Exp $"; }

/* ------------------------------------------------------------------------ */

//#define TRACE
#include "cputypes.h"
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "pathwork.h"  // GetFullPathForFilename()
#include "clitime.h"   // CliClock()
#include "util.h"      // TRACE and utilxxx()
#include "logstuff.h"  // LogScreen()
#include "triggers.h"  // keep prototypes in sync

/* ----------------------------------------------------------------------- */

#define PAUSEFILE_CHECKTIME_WHENON  (3)  //seconds
#define PAUSEFILE_CHECKTIME_WHENOFF (3*PAUSEFILE_CHECKTIME_WHENON)
#define EXITFILE_CHECKTIME          (PAUSEFILE_CHECKTIME_WHENOFF)

/* note that all flags are single bits */
#if !defined(TRIGSETBY_SIGNAL) /* defined like this in triggers.h */
#define TRIGSETBY_SIGNAL       0x01 /* signal or explicit call to raise */ 
#define TRIGSETBY_FLAGFILE     0x02 /* flag file */
#define TRIGSETBY_CUSTOM       0x04 /* something other than the above */
#endif
/* the following are internal and are exported as TRIGSETBY_CUSTOM */
#define TRIGPAUSEBY_APPACTIVE  0x10 /* pause due to app being active*/
#define TRIGPAUSEBY_SRCBATTERY 0x20 /* pause due to running on battery */
#define TRIGPAUSEBY_CPUTEMP    0x40 /* cpu temperature guard */

struct trigstruct 
{
  const char *flagfile; 
  struct { unsigned int whenon, whenoff; } pollinterval;
  unsigned int incheck; //recursion check
  void (*pollproc)(int io_cycle_allowed);
  volatile int trigger; 
  int laststate;
  time_t nextcheck;
};

static struct 
{
  int doingmodes;
  struct trigstruct exittrig;
  struct trigstruct pausetrig;
  struct trigstruct huptrig;
  char pausefilebuf[128]; /* includes path */
  char exitfilebuf[128];
  int overrideinifiletime;
  time_t nextinifilecheck;
  unsigned long currinifiletime;
  char inifile[128];
  char pauseplistbuffer[128];
  const char *pauseplist[16];
  int lastactivep;
  int pause_if_no_mains_power;
  struct
  {
    unsigned int lothresh, hithresh; /* in Kelvin */
    int marking_high; /* we were >= high, waiting for < lowthresh */
  } cputemp;  
} trigstatics;

// -----------------------------------------------------------------------

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
  if ((trigstatics.pausetrig.trigger & TRIGSETBY_FLAGFILE)!=TRIGSETBY_FLAGFILE 
     && trigstatics.pausetrig.flagfile)
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
{ 
  __assert_statics(); 
  if (trigstatics.exittrig.pollproc)
    (*trigstatics.exittrig.pollproc)(0 /* io_cycle_NOT_allowed */);
  return (trigstatics.exittrig.trigger); 
} 
int CheckPauseRequestTriggerNoIO(void) 
{ __assert_statics(); 
  return ((trigstatics.pausetrig.trigger&(TRIGSETBY_SIGNAL|TRIGSETBY_FLAGFILE))
         |((trigstatics.pausetrig.trigger&
           (~(TRIGSETBY_SIGNAL|TRIGSETBY_FLAGFILE)))?(TRIGSETBY_CUSTOM):(0)));
}           
int CheckRestartRequestTriggerNoIO(void) 
{ __assert_statics(); return (trigstatics.huptrig.trigger); }

// -----------------------------------------------------------------------

void *RegisterPollDrivenBreakCheck( register void (*proc)(int) )
{
  register void (*oldproc)(int);
  __assert_statics(); 
  oldproc = trigstatics.exittrig.pollproc;
  trigstatics.exittrig.pollproc = proc;
  return (void *)oldproc;
}

// -----------------------------------------------------------------------

static void __PollExternalTrigger(struct trigstruct *trig, int undoable)
{
  __assert_statics(); 
  if ((undoable || (trig->trigger & TRIGSETBY_FLAGFILE) == 0) && trig->flagfile)
  {
    struct timeval tv;
    if (CliClock(&tv) == 0)
    {
      time_t now = tv.tv_sec;
      if (now >= trig->nextcheck) 
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
  }
  return;
}

// -----------------------------------------------------------------------

static unsigned long __get_file_time(const char *filename)
{
  unsigned long filetime = 0; /* returns zero on error */
#if (CLIENT_OS == OS_RISCOS)
  riscos_get_file_modified(filename,(unsigned long *)(&filetime));
#else
  struct stat statblk;
  if (stat( filename, &statblk ) == 0)
    filetime = (unsigned long)statblk.st_mtime;
#endif
  return filetime;
}    

static void __CheckIniFileChangeStuff(void)
{
  __assert_statics(); 
  if (trigstatics.inifile[0]) /* have an ini filename? */
  {
    struct timeval tv;
    if (CliClock(&tv) == 0)
    {
      time_t now = tv.tv_sec;
      if (now > trigstatics.nextinifilecheck)
      {
        unsigned long filetime = __get_file_time(trigstatics.inifile);
        trigstatics.nextinifilecheck = now + ((time_t)5);
        if (filetime)
        {
          if (trigstatics.overrideinifiletime > 0)
          {
            trigstatics.currinifiletime = 0;
            trigstatics.overrideinifiletime--;
          }
          else if (!trigstatics.currinifiletime)     /* first time */
          {
            trigstatics.currinifiletime = filetime;
          }  
          else if (trigstatics.currinifiletime == 1) 
          {                                   /* got change some time ago */
            RaiseRestartRequestTrigger();
            trigstatics.currinifiletime = 0;
            trigstatics.nextinifilecheck = now + ((time_t)60);
          }
          else if (filetime != trigstatics.currinifiletime)
          {                                                /* mark change */
            trigstatics.currinifiletime = 1;
          }
        } 
      }  
    }
  }
  return;
}  

// -----------------------------------------------------------------------

static const char *__mangle_pauseapp_name(const char *name, int unmangle_it )
{
  #if ((CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16))
  /* these two are frequently used 16bit apps that aren't visible (?) 
     to utilGetPIDList so they are searched for by window class name, 
     which is a unique identifier so we can find/tack them on to the 
     end of the pauseplist in __init. (They used to be hardcoded in
     the days before utilGetPIDList)
  */   
  if (winGetVersion() >= 400 && winGetVersion() <2000) /* win9x only */
  {
    static const char *app2wclass[] = { "scandisk",  "#ScanDskWDlgClass",
                                        "scandiskw", "#ScanDskWDlgClass",
                                        "defrag",    "#MSDefragWClass1" };
    unsigned int app;
    if (unmangle_it)
    {
      TRACE_OUT((+1,"x1: demangle: '%s'\n",name));
      for (app = 0; app < (sizeof(app2wclass)/sizeof(app2wclass[0])); app+=2)
      {
        if ( strcmp( name, app2wclass[app+1]) == 0)
        {
          name = app2wclass[app+0];
          break;
        }
      }
      TRACE_OUT((-1,"x2: demangle: '%s'\n",name));
    }
    else /* this only happens once per InitializeTriggers() */
    {
      unsigned int bpos, blen;
      blen = bpos = strlen( name );
      while (bpos>0 && name[bpos-1]!='\\' && name[bpos-1]!='/' && name[bpos-1]!=':')
        bpos--;
      blen -= bpos;
      if (blen > 3 && strcmpi(&name[bpos+(blen-4)],".exe") == 0)
        blen-=4;        /* only need to look for '.exe' since all are .exe */
      TRACE_OUT((+1,"x1: mangle: '%s', pos=%u,len=%u\n",name,bpos,blen));
      for (app = 0; app < (sizeof(app2wclass)/sizeof(app2wclass[0])); app += 2)
      {
        if ( memicmp( name+bpos, app2wclass[app+0], blen) == 0)
        {
          name = app2wclass[app+1];
          break;
        }
      }    
      TRACE_OUT((-1,"x2: mangle: '%s'\n",name));
    }                                
  } /* win9x? */  
  #endif /* ((CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)) */

  unmangle_it = unmangle_it; /* shaddup compiler */
  return name;
}

// -----------------------------------------------------------------------

#if (CLIENT_OS == OS_NETBSD) && (CLIENT_CPU == CPU_X86)
// for apm support in __IsRunningOnBattery
#include <machine/apmvar.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#elif (CLIENT_OS == OS_FREEBSD) && (CLIENT_CPU == CPU_X86)
#include <fcntl.h>
#include <machine/apm_bios.h>
#endif

static int __IsRunningOnBattery(void) /*returns 0=no, >0=yes, <0=err/unknown*/
{
  if (trigstatics.pause_if_no_mains_power)
  {
    #if (CLIENT_OS == OS_WIN32)
    static FARPROC getsps = ((FARPROC)-1);
    if (getsps == ((FARPROC)-1))
    {
      HMODULE hKernel = GetModuleHandle("kernel32.dll");
      if (hKernel)
        getsps = GetProcAddress( hKernel, "GetSystemPowerStatus");
      else
        getsps = (FARPROC)0;
    }
    if (!getsps)
      trigstatics.pause_if_no_mains_power = 0;
    else  
    {    
      SYSTEM_POWER_STATUS sps;
      sps.ACLineStatus = 255;
      if ((*((BOOL (WINAPI *)(LPSYSTEM_POWER_STATUS))getsps))(&sps))
      {
        TRACE_OUT((0,"sps: ACLineStatus = 0x%02x, BatteryFlag = 0x%02x\n",sps.ACLineStatus,sps.BatteryFlag));
        if (sps.ACLineStatus == 1) /* AC power is online */
          return 0; /* no, we are not on battery */
        if (sps.ACLineStatus == 0) /* AC power is offline */
          return 1; /* yes, we are on battery */
        /* third condition is 0xff ("unknown"), so fall through */
      }
    }
    #elif (CLIENT_OS == OS_LINUX) && (CLIENT_CPU == CPU_X86)
    {
      int disableme= 1; // if this is still set when we get to the end
                        // then disable further apm checking
      char buffer[256]; // must be big enough for complete read of /proc/apm
                        // nn.nn nn.nn 0xnn 0xnn 0xnn 0xnn [-]nnn% [-]nnnn *s\n
      int readsz = -1;
      int fd = open( "/proc/apm", O_RDONLY );

      if (fd == -1)
      {
        if (errno == ENOMEM || errno == EAGAIN) /*ENOENT,ENXIO,EIO,EPERM */
          disableme = 0;  /* ENOMEM because apm kmallocs a struct per open */
          
        TRACE_OUT((0,"sps: open(\"/proc/apm\",O_RDONLY) => %s, disableme=%d\n", strerror(errno), disableme));
        #if defined(TRACE) /* real-life example (1.2 is similar) */
        readsz = strlen(strcpy(buffer, "1.13 1.2 0x07 0xff 0xff 0xff -1% -1 ?"));
        #endif
      }
      else 
      {
        readsz = read( fd, buffer, sizeof(buffer));
        close(fd);
      }

      /* read should never fail for /proc/apm, and the size must be less 
         than sizeof(buffer) otherwise its some /proc/apm that we
         don't know how to parse.
      */   
      if (readsz > 0 && ((unsigned int)readsz) < sizeof(buffer))
      {
        unsigned int drv_maj, drv_min; /* "1.2","1.9","1.10","1.12,"1.13" etc*/
        int bios_maj, bios_min; /* %d.%d */
        int bios_flags, ac_line_status, batt_status, batt_flag; /* 0x%02x */
        /* remaining fields are percentage, time_units, units. (%d %d %s) */

        buffer[readsz-1] = '\0';
        if (sscanf( buffer, "%u.%u %d.%d 0x%02x 0x%02x 0x%02x 0x%02x",
                            &drv_maj, &drv_min, &bios_maj, &bios_min,
                            &bios_flags, &ac_line_status, 
                            &batt_status, &batt_flag ) == 8 )
        {			      
          TRACE_OUT((0,"sps: drvver:%u.%u biosver:%d.%d biosflags:0x%02x "
                       "ac_line_status=0x%02x, batt_status=0x%02x\n",
                       drv_maj, drv_min, bios_maj, bios_min, bios_flags,
                       ac_line_status, batt_status ));      
          if (drv_maj == 1)
          {
            #define _APM_16_BIT_SUPPORT   (1<<0)
            #define _APM_32_BIT_SUPPORT   (1<<1)
            //      _APM_IDLE_SLOWS_CLOCK (1<<2)
            #define _APM_BIOS_DISABLED    (1<<3)
            //      _APM_BIOS_DISENGAGED  (1<<4)	      
            if ((bios_flags & (_APM_16_BIT_SUPPORT | _APM_32_BIT_SUPPORT))!=0
              && (bios_flags & _APM_BIOS_DISABLED) == 0)
            { 
              disableme = 0;
              ac_line_status &= 0xff; /* its a char */
              /* From /usr/src/[]/arch/i386/apm.c for (1.2)1996-(1.13)2/2000
                 3) AC Line Status:
                    0x00: Off-line
                    0x01: On-line
                    0x02: On backup-power (APM BIOS 1.1+ only)
                    0xff: Unknown
              */      
              if (ac_line_status == 1)
                return 0; /* we are not on battery */
              if (ac_line_status != 0xff) /* 0x00, 0x02 */
                return 1; /* yes we are on battery */
              /* fallthrough, return -1 */    
            }
          } /* drv_maj == 1 */
        } /* sscanf() == 8 */ 
      } /* readsz */
    
      if (disableme) /* disable further checks */
      {
        TRACE_OUT((0,"sps: further pause_if_no_mains_power checks now disabled\n"));
        trigstatics.pause_if_no_mains_power = 0;
      }
    } /* #if (linux) */
    #elif (CLIENT_OS == OS_FREEBSD) && (CLIENT_CPU == CPU_X86)
    {
      /* This is sick and sooo un-BSD like! Tatsumi Hokosawa must have
         been dealing too much with linux and forgot all about sysctl. :)
      */
      int disableme = 1; /* assume further apm checking should be disabled */
      int fd = open("/dev/apm", O_RDONLY);
      if (fd != -1)
      {
        #if defined(APMIO_GETINFO_OLD) /* (__FreeBSD__ >= 3) */
        struct apm_info_old info;      /* want compatibility for 2.2 */
        int whichioctl = APMIO_GETINFO_OLD;
        #else
        struct apm_info info;
        int whichioctl = APMIO_GETINFO;
        #endif
        disableme = 0;
        
        memset( &info, 0, sizeof(info));
        if (ioctl(fd, whichioctl, (caddr_t)&info, 0 )!=0)
        {
          disableme = 1;
          info.ai_acline = 255; /* what apm returns for "unknown" */
        }  
        else
        {
          TRACE_OUT((+1,"APM check\n"));
          TRACE_OUT((0,"aiop->ai_major = %d\n", info.ai_major));
	    		TRACE_OUT((0,"aiop->ai_minor = %d\n", info.ai_minor));
		    	TRACE_OUT((0,"aiop->ai_acline = %d\n",  info.ai_acline));
			    TRACE_OUT((0,"aiop->ai_batt_stat = %d\n", info.ai_batt_stat));
    			TRACE_OUT((0,"aiop->ai_batt_life = %d\n", info.ai_batt_life));
	    		TRACE_OUT((0,"aiop->ai_status = %d\n", info.ai_status));
          TRACE_OUT((-1,"conclusion: AC line state: %s\n", ((info.ai_acline==0)?
                 ("offline"):((info.ai_acline==1)?("online"):("unknown"))) ));
        }
        close(fd);
        
        if (info.ai_acline == 1)
          return 0; /* We have AC power */
        if (info.ai_acline == 0)
          return 1; /* no AC power */  
      }  
      if (disableme)
      {
        /* possible causes for a disable are
	         EPERM: no permission to open /dev/apm) or 
  	       ENXIO: apm device not configured, or disabled [kern default],
                  or (for ioctl()) real<->pmode transition or bios error.
       	*/   
        TRACE_OUT((0,"pause_if_no_mains_power check error: %s\n", strerror(errno)));
        TRACE_OUT((0,"disabling further pause_if_no_mains_power checks\n"));
        trigstatics.pause_if_no_mains_power = 0;
      }
    } /* freebsd */     
    #elif (CLIENT_OS == OS_NETBSD) && (CLIENT_CPU == CPU_X86)
    {
      struct apm_power_info buff;
      int fd;
      #define _PATH_APM_DEV "/dev/apm"

      fd = open(_PATH_APM_DEV, O_RDONLY);

      if (fd != -1) {
        if (ioctl(fd, APM_IOC_GETPOWER, &buff) == 0) {
          close(fd);
          TRACE_OUT((0,"sps: ACLineStatus = 0x%08x, BatteryFlag = 0x%08x\n",
            buff.ac_state, buff.battery_state));

          if (buff.ac_state == APM_AC_ON)
            return 0;       /* we have AC power */
          if (buff.ac_state == APM_AC_OFF)
            return 1;       /* we don't have AC */
        }
      }
      close(fd);
      // We seem to have no apm driver in the kernel, so disable it.
      trigstatics.pause_if_no_mains_power = 0;
    } /* #if (NetBSD && i386) */
    #endif
  }  
  return -1; /* unknown */
}

// -----------------------------------------------------------------------

static int __CPUTemperaturePoll(void)
{
  int lowthresh = (int)trigstatics.cputemp.lothresh;
  int highthresh = (int)trigstatics.cputemp.hithresh;
  if (highthresh > lowthresh) /* otherwise values are invalid */
  {
    /* read the cpu temp in Kelvin. For multiple cpus, gets one 
       with highest temp. On error, returns < 0.
       Note that cputemp is in Kelvin, if your OS returns a value in
       Farenheit or Celsius, see _init_cputemp for conversion functions.
    */
    int cputemp = -1;
    #if (CLIENT_OS == OS_MACOS)
      cputemp = macosCPUTemp();
    #elif 0 /* other client_os */
    cputemp = fooyaddablahblahbar();
    #endif
    if (cputemp < 0) /* error */
      ; 
    else if (cputemp >= highthresh)
      trigstatics.cputemp.marking_high = 1;
    else if (cputemp < lowthresh)
      trigstatics.cputemp.marking_high = 0;
  }
  return trigstatics.cputemp.marking_high;
}

// -----------------------------------------------------------------------

static void __PollDrivenBreakCheck(int io_cycle_allowed)
{
  /* io_cycle_allowed is non-zero when called through CheckExitRequestTrigger
     and is zero when called through CheckExitRequestTriggerNoIO()
  */
  io_cycle_allowed = io_cycle_allowed; /* shaddup compiler */
  #if (CLIENT_OS == OS_RISCOS)
  if (_kernel_escape_seen())
      RaiseExitRequestTrigger();
  #elif (CLIENT_OS == OS_AMIGAOS)
  ULONG trigs;
  if ( (trigs = amigaGetTriggerSigs()) )  // checks for ^C and other sigs
  {
    if ( trigs & DNETC_MSG_SHUTDOWN )
      RaiseExitRequestTrigger();
    if ( trigs & DNETC_MSG_RESTART )
      RaiseRestartRequestTrigger();
    if ( trigs & DNETC_MSG_PAUSE )
      RaisePauseRequestTrigger();
    if ( trigs & DNETC_MSG_UNPAUSE )
      ClearPauseRequestTrigger();
  }
  #elif (CLIENT_OS == OS_NETWARE)
  if (io_cycle_allowed)
    nwCliCheckForUserBreak(); //in nwccons.cpp
  #elif (CLIENT_OS == OS_WIN16)
    w32ConOut("");    /* benign call to keep ^C handling alive */
  #elif (CLIENT_OS == OS_WIN16)
  if (io_cycle_allowed)
    w32ConOut("");    /* benign call to keep ^C handling alive */
  #elif (CLIENT_OS == OS_DOS)
    _asm mov ah,0x0b  /* benign dos call (kbhit()) */
    _asm int 0x21     /* to keep int23h (^C) handling alive */
  #endif
  return;  
}      

// =======================================================================

int OverrideNextConffileChangeTrigger(void)
{
  __assert_statics(); 
  return (++trigstatics.overrideinifiletime);
}

int CheckExitRequestTrigger(void) 
{
  __assert_statics(); 
  if (!trigstatics.exittrig.laststate && !trigstatics.exittrig.incheck)
  {
    ++trigstatics.exittrig.incheck;
    if ( !trigstatics.exittrig.trigger )
    {
      if (trigstatics.exittrig.pollproc)
        (*trigstatics.exittrig.pollproc)(1 /* io_cycle_allowed */);
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
        Log("*Break* %s\n",
          ( ((trigstatics.exittrig.trigger & TRIGSETBY_FLAGFILE)!=0)?
               ("(found exit flag file)"): 
           ((trigstatics.huptrig.trigger)?("Restarting..."):("Shutting down..."))
            ) );
      }
    }
    --trigstatics.exittrig.incheck;
  }
  TRACE_OUT((0, "CheckExitRequestTrigger() = %d\n", trigstatics.exittrig.trigger));
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

    if (trigstatics.pauseplist[0] != NULL)
    {
      int idx, nowcleared = -1;
      const char **pp = &trigstatics.pauseplist[0];

      /* the use of "nowcleared" is a hack for the sake of optimization
         so that we can sequence the code for the paused->unpause and
         the unpause->pause transitions and still avoid an extra call
         to utilGetPIDList() if we know the app is definitely not running.
      */
      if ((trigstatics.pausetrig.trigger & TRIGPAUSEBY_APPACTIVE) != 0)
      {
        idx = trigstatics.lastactivep;
        TRACE_OUT((+1,"y1: idx=%d,'%s'\n", idx, pp[idx]));
        if (utilGetPIDList( pp[idx], NULL, 0 ) <= 0)
        {
          /* program is no longer running. */
          Log("%s... ('%s' inactive)\n",
              (((trigstatics.pausetrig.laststate 
                   & ~TRIGPAUSEBY_APPACTIVE)!=0)?("Pause level lowered"):
              ("Running again after pause")), 
              __mangle_pauseapp_name(pp[idx],1 /* unmangle */) );
          trigstatics.pausetrig.laststate &= ~TRIGPAUSEBY_APPACTIVE;
          trigstatics.pausetrig.trigger &= ~TRIGPAUSEBY_APPACTIVE;
          trigstatics.lastactivep = 0;
          nowcleared = idx;
        }
        TRACE_OUT((-1,"y2: nowcleared=%d\n", nowcleared));
      }
      if ((trigstatics.pausetrig.trigger & TRIGPAUSEBY_APPACTIVE) == 0)
      {
        idx = 0;
        while (pp[idx] != NULL)
        {
          TRACE_OUT((+1,"z1: %d, '%s'\n",idx, pp[idx]));
          if (idx == nowcleared)
          {
            /* if this matched, then we came from the transition
               above and know that this is definitely not running.
            */   
          }
          else if (utilGetPIDList( pp[idx], NULL, 0 ) > 0)
          {
            /* Triggered program is now running. */
            trigstatics.pausetrig.trigger |= TRIGPAUSEBY_APPACTIVE;
            trigstatics.lastactivep = idx;
            app_now_active = __mangle_pauseapp_name(pp[idx],1 /* unmangle */);
            break;
          }
          TRACE_OUT((-1,"z2\n"));
          idx++;
        }
      }
    }

    {
      int isbat = __IsRunningOnBattery(); /* <0=unknown/err, 0=no, >0=yes */
      if (isbat > 0)
        trigstatics.pausetrig.trigger |= TRIGPAUSEBY_SRCBATTERY;
      else if (isbat == 0)
        trigstatics.pausetrig.trigger &= ~TRIGPAUSEBY_SRCBATTERY;  
      /* otherwise error/unknown, so don't change anything */  
    }    
    if (__CPUTemperaturePoll())
      trigstatics.pausetrig.trigger |= TRIGPAUSEBY_CPUTEMP;
    else
      trigstatics.pausetrig.trigger &= ~TRIGPAUSEBY_CPUTEMP;  

    __PollExternalTrigger( &trigstatics.pausetrig, 1 );
    
    /* +++ */
    
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
      if ((trigstatics.pausetrig.trigger & TRIGPAUSEBY_APPACTIVE)!=0 &&
          (trigstatics.pausetrig.laststate & TRIGPAUSEBY_APPACTIVE)==0)
      {
        Log("Pause%sd... ('%s' active)\n",
             ((trigstatics.pausetrig.laststate)?(" level raise"):("")),
             ((*app_now_active)?(app_now_active):("process")) );
        trigstatics.pausetrig.laststate |= TRIGPAUSEBY_APPACTIVE;
      }
      else if ((trigstatics.pausetrig.trigger & TRIGPAUSEBY_APPACTIVE)==0 &&
          (trigstatics.pausetrig.laststate & TRIGPAUSEBY_APPACTIVE)!=0)
      {
        trigstatics.pausetrig.laststate &= ~TRIGPAUSEBY_APPACTIVE;
        /* message was already printed above
        //Log("%s... %s\n",
        //  ((trigstatics.pausetrig.laststate)?("Pause level lowered"):
        //  ("Running again after pause")), "(process no longer active)" );
        */
      }
      if ((trigstatics.pausetrig.trigger & TRIGPAUSEBY_SRCBATTERY)!=0 &&
          (trigstatics.pausetrig.laststate & TRIGPAUSEBY_SRCBATTERY)==0)
      {
        Log("Pause%sd... (No mains power)\n",
             ((trigstatics.pausetrig.laststate)?(" level raise"):("")) );
        trigstatics.pausetrig.laststate |= TRIGPAUSEBY_SRCBATTERY;
      }
      else if ((trigstatics.pausetrig.trigger & TRIGPAUSEBY_SRCBATTERY)==0 &&
          (trigstatics.pausetrig.laststate & TRIGPAUSEBY_SRCBATTERY)!=0)
      {          
        trigstatics.pausetrig.laststate &= ~TRIGPAUSEBY_SRCBATTERY;
        Log("%s... (Mains power restored)\n",
          ((trigstatics.pausetrig.laststate)?("Pause level lowered"):
          ("Running again after pause")) );
      }
      if ((trigstatics.pausetrig.trigger & TRIGPAUSEBY_CPUTEMP)!=0 &&
          (trigstatics.pausetrig.laststate & TRIGPAUSEBY_CPUTEMP)==0)
      {
        Log("Pause%sd... (CPU temperature exceeds %uK)\n",
             ((trigstatics.pausetrig.laststate)?(" level raise"):("")),
               trigstatics.cputemp.hithresh );
        trigstatics.pausetrig.laststate |= TRIGPAUSEBY_CPUTEMP;
      }
      else if ((trigstatics.pausetrig.trigger & TRIGPAUSEBY_CPUTEMP)==0 &&
          (trigstatics.pausetrig.laststate & TRIGPAUSEBY_CPUTEMP)!=0)
      {          
        trigstatics.pausetrig.laststate &= ~TRIGPAUSEBY_CPUTEMP;
        Log("%s... (CPU temperature below %uK)\n",
          ((trigstatics.pausetrig.laststate)?("Pause level lowered"):
          ("Running again after pause")), trigstatics.cputemp.lothresh );
      }
    }
  }
  --trigstatics.pausetrig.incheck;
  return CheckPauseRequestTriggerNoIO(); /* return public mask */
}   

// =======================================================================

#if (CLIENT_OS == OS_AMIGAOS)
extern "C" void __chkabort(void) 
{ 
  /* Disable SAS/C / GCC CTRL-C handing */
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
#if (CLIENT_OS == OS_IRIX) //or #if defined(SA_NOCLDSTOP) for posix compat
static void (*SETSIGNAL(int signo, void (*proc)(int)))(int)
{
  struct sigaction sa, osa;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  /* sa.sa_handler = proc; cast to get around c to c++ non-portability */
  *((void **)(&(sa.sa_handler))) = *((void **)(&proc));
  #if defined(SA_RESTART) /* SA_RESTART is not _POSIX_SOURCE */
  /* if (!sigismember(&_sigintr, signo)) */
    sa.sa_flags |= SA_RESTART;
  #endif
  if (sigaction(signo, &sa, &osa) < 0)
    return (void (*)(int))(SIG_ERR);
  return (void (*)(int))(osa.sa_handler);
}
#else
#define SETSIGNAL signal
#endif

// -----------------------------------------------------------------------

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
  #if defined(__unix__) && defined(TRIGGER_PAUSE_SIGNAL)
  if (!doingmodes)                  // signal-based pause/unpause mechanism?
  {
    #if (CLIENT_OS != OS_NTO2) && (CLIENT_OS != OS_BEOS)
    // stop the shell from seeing SIGTSTP and putting the client into 
    // the background when we '-pause' it.
    if( getpgrp() != getpid() ) 
      setpgid( 0, 0 );  // 
    // porters : those calls are POSIX.1, 
    // - on BSD you might need to change setpgid(0,0) to setpgrp()
    // - on SYSV you might need to change getpgrp() to getpgid(0)
    #endif
    SETSIGNAL( TRIGGER_PAUSE_SIGNAL, CliSignalHandler );  //pause
    SETSIGNAL( TRIGGER_UNPAUSE_SIGNAL, CliSignalHandler );  //continue
  }
  #endif /* defined(__unix__) && defined(TRIGGER_PAUSE_SIGNAL) */
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

/* ---------------------------------------------------------------- */

static void _init_cputemp( const char *p ) /* cpu temperature string */
{
  int K[2];
  int which;

  trigstatics.cputemp.hithresh = trigstatics.cputemp.lothresh = 0;
  trigstatics.cputemp.marking_high = 0;

  K[0] = K[1] = -1;
  for (which=0;which<2;which++)
  {
    int val = 0, len = 0, neg = 0;
    while (*p && isspace(*p))
      p++;
    if (!*p)  
      break;
    if (*p == '-' || *p == '+')
      neg = (*p++ == '-');
    while (isdigit(*p))
    {
      val = val*10;
      val += (*p - '0'); /* this is safe in ebcidic as well */
      len++;
      p++;
    }
    if (len == 0) /* bad number */
      return;
    if (*p == '.') /* hmm, decimal pt */
    {
      p++;
      if (!isdigit(*p))
        return;
      if (*p >= '5')
        val++;  
      while (isdigit(*p))  
        p++;
    }  
    if (neg)
      val = -val;
    while (*p && isspace(*p))
      p++;  
    if (*p=='F' || *p=='f' || *p=='R' || *p=='r') /* farenheit or rankine */
    {
      if (*p == 'R' || *p == 'r') /* convert rankine to farenheit first */
        val -= 459; /* 459.67 */
      val = (((val - 32) * 5)/9) + 273/*.15*/;  /* F -> K */
      p++;
    }
    else if (*p == 'C' || *p == 'c') /* celcius/centigrade */    
    {   
      val += 273/*.15*/; /* C -> K */
      p++;
    }
    else if (*p == 'K' || *p == 'k') /* Kelvin */
    {
      p++; 
    }
    if (val < 0) /* below absolute zero, uh, huh */
      return;
    K[which] = val;
    while (*p && isspace(*p))
      p++;
    if (*p != ':')
      break;
    p++;
  }  
  if (K[0] > 1) /* be gracious :) */
  {
    if (K[1] < 0) /* only single temp provided */
    {
      K[1] = K[0]; /* then make that the high water mark */
      K[0] -= (K[0]/10); /* low water mark is 90% of high water mark */
    }
    TRACE_OUT((0,"cputemp: %dK:%dK\n", K[0], K[1]));

    trigstatics.cputemp.lothresh = K[0];
    trigstatics.cputemp.hithresh = K[1];
  }
  return;
}

/* ---------------------------------------------------------------- */

static void _init_pauseplist( const char *plist )
{
  const char *p = plist;
  unsigned int wpos = 0, idx = 0, len;
  while (*p)
  {
    while (*p && (isspace(*p) || *p == '|'))
      p++;
    if (*p)
    {
      len = 0;
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
        const char *appname = &(trigstatics.pauseplistbuffer[wpos]);
        if ((wpos + len) >= (sizeof(trigstatics.pauseplistbuffer) - 2))
        {
          break;
        }
        memcpy( &(trigstatics.pauseplistbuffer[wpos]), plist, len );
        wpos += len;
        trigstatics.pauseplistbuffer[wpos++] = '\0';
        trigstatics.pauseplist[idx++] = __mangle_pauseapp_name(appname,0);
        if (idx == ((sizeof(trigstatics.pauseplist)/
                       sizeof(trigstatics.pauseplist[0]))-1) )
        {
          break;
        }
      }
    }
  }
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
  /* 
  sorry about this piece of OS-specific uglyness. These were
  pause triggers before the days of utilGetPidlist(), and are now 
  tacked on at the end of the guard list for compatibility's sake. 
  Yes, they really are needed - I've seen people curse these two
  till they were blue in the face, not realizing what the heck it 
  was that so crippled their machines. Yeah, yeah. win9x stinks. :)
  The mangling done above will already have given us unique identifiers,
  so we at least don't have to parse the whole lot again.
  */
  if (winGetVersion() >= 400 && winGetVersion() < 2000) /* win9x only */
  {
    for (len = 0; len < 2; len++)
    {
      const char *clist;
      if (idx == ((sizeof(trigstatics.pauseplist)/
                   sizeof(trigstatics.pauseplist[0]))-1) )
        break;
      clist = __mangle_pauseapp_name( ((len==0)?("defrag"):("scandisk")),0);
      for (wpos = 0; wpos < idx; wpos++)
      {
        if ( strcmp( trigstatics.pauseplist[wpos], clist ) == 0 )
        {
          clist = (const char *)0;
          break;
        }
      }
      if (clist)
        trigstatics.pauseplist[idx++] = clist;
    }
  }
  #endif
  trigstatics.pauseplist[idx] = (const char *)0;
}

/* ---------------------------------------------------------------- */

int InitializeTriggers(int doingmodes,
                       const char *exitfile, const char *pausefile,
                       const char *pauseplist,
                       int restartoninichange, const char *inifile,
                       int watchcputempthresh, const char *cputempthresh,
                       int pauseifnomainspower )
{
  TRACE_OUT( (+1, "InitializeTriggers\n") );
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
    TRACE_OUT((0, "exitfile: %s\n", trigstatics.exittrig.flagfile));
    trigstatics.pausetrig.flagfile = _init_trigfile(pausefile, 
                 trigstatics.pausefilebuf, sizeof(trigstatics.pausefilebuf) );
    TRACE_OUT((0, "pausefile: %s\n", trigstatics.pausetrig.flagfile));
    if (restartoninichange && inifile)
      _init_trigfile(inifile,trigstatics.inifile,sizeof(trigstatics.inifile));
    TRACE_OUT((0, "restartfile: %s\n", trigstatics.inifile));
    _init_pauseplist( pauseplist );
    if (watchcputempthresh && cputempthresh)
      _init_cputemp( cputempthresh ); /* cpu temp string */
    trigstatics.pause_if_no_mains_power = pauseifnomainspower;
    if (doingmodes) /* dummy if, always false */
      __PollDrivenBreakCheck(1); /* shaddup compiler */
  }
  TRACE_OUT( (-1, "InitializeTriggers\n") );
  return 0;
}  

/* ---------------------------------------------------------------- */

int DeinitializeTriggers(void)
{
  int huptrig;
  __assert_statics(); 
  huptrig = trigstatics.huptrig.trigger;
  /* clear everything to ensure we don't use IO after DeInit */
  memset( (void *)(&trigstatics), 0, sizeof(trigstatics) );
  return huptrig;
}  

/* ---------------------------------------------------------------- */

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
