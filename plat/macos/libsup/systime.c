/* 
 * Created for use in distributed.net projects, but incorporation
 * in other libraries is encouraged.
 *
 * $Id: systime.c,v 1.1.2.3 2001/07/27 08:51:15 mfeiri Exp $
 *
 * POSIX/ANSI time and date functions for MacOS.
 * 
 * The epoch for all functions that return/receive an absolute time is
 * the same as that returned from time() [time() is assumed to be 
 * available in the vendor's C library]
 *
 * Although functions here all return/accept the same structures/values 
 * published in the ANSI/POSIX standards, no ANSI/POSIX headers are 
 * #included from here. This is in order to ensure that the functions 
 * don't pollute the global namespace.
 *
 * Thread-safety note: Some of the functions here initialize static
 * data (such as the process start time) the first time they are
 * called. It is assumed that either gettimeofday() or clock_gettime() 
 * are called at least once before any (preemptable) threads are created. 
 * Either function will complete this initialization phase.
*/

/* note that we do not #include anything beyond what is is required for */
/*  the __kern_* functions. We do this to avoid poisoning the namespace. */

#ifdef __cplusplus
extern "C" {
#endif
  /* ------- <time.h> --------- */
  typedef unsigned long time_t;
  extern time_t time(time_t *); /* get # of secs since epoch (whenever) */
                                /* time() is assumed to be in the vendor lib */
                                
  extern void tzset(void);
  extern long timezone;         /* seconds east of Greenwich (uncompensated) */
  extern int  daylight;         /* "is daylight savings in effect?" flag */
  extern char *tzname[2];
  /* ----- <sys/time.h> ------- */
  #pragma pack(1)
  typedef int clockid_t;
  struct  timeval  { time_t tv_sec; long tv_usec; };
  struct  __timezone { int tz_minuteswest, tz_dsttype; };
  #define timezone __timezone
  struct  timespec { time_t tv_sec; long tv_nsec; };
  #pragma pack()
  extern int clock_gettime(clockid_t clktype, struct timespec *tsp);
  extern int clock_settime(clockid_t clktype, struct timespec *tsp);
  extern int clock_getres(clockid_t clktype, struct timespec *tsp);
  #define CLOCK_REALTIME  0
  #define CLOCK_VIRTUAL   1
  #define CLOCK_PROF      2
  #define CLOCK_MONOTONIC 3
  #define CLOCK_TIMEOFDAY 4
  extern int settimeofday(struct timeval *, struct timezone *);
  extern int gettimeofday(struct timeval *, struct timezone *);
  #define DST_NONE  0
  #define DST_USA   1
  #define DST_AUST  2
  #define DST_WET   3
  #define DST_MET   4
  #define DST_EET   5
  #define DST_CAN   6
  #define DST_GB    7
  #define DST_RUM   8
  #define DST_TUR   9
  #define DST_AUSTALT 10
#ifdef __cplusplus
}
#endif

/* ====================================================================== */
/*                    BEGIN HARDWARE/OS PRIMITIVES                        */
/* ====================================================================== */

#include <DateTimeUtils.h>  // Set/GetDateTime
#include <DriverServices.h> // UpTime

/* returns resolution: 1,000,000,000 if using Uptime(), else 1,000,000 */
static unsigned long __kern_get_nano_uptime(unsigned long *secs, 
                                            unsigned long *nsecs)
{
  /* For discussion of several timing-related routines available on the Mac:
     http://www.idevgames.com/downloads/tutorials/Ollmann/TimeTechniques.sit
     and http://developer.apple.com/dev/techsupport/develop/issue29/minow.html
  */     
  #if defined(__POWERPC__)  
  static int use_nano = -1; /* undetermined */
  if (use_nano == -1)
  {
    if ( (UInt32)UpTime == (UInt32)kUnresolvedCFragSymbolAddress )
      use_nano = 0; /* UpTime is not available in early (NuBus) PowerMacs */
    else 
      use_nano = 1;
  }
  if (use_nano) /* uses the PCI Bus clock */
  {
    /* Absolute Time -> Nanoseconds (unsigned wide) */
    Nanoseconds now = AbsoluteToNanoseconds(UpTime());
    *nsecs = *((unsigned long long *)&now) % (unsigned long long)1000000000;
    *secs  = *((unsigned long long *)&now) / (unsigned long long)1000000000;
    // return (float)UnsignedWideToUInt64(now)/1000.0;
    return 1000000000ul;
  }
  #endif  /* __POWERPC__ */
  /* __MC68K__ or NuBus PowerMacs */
  {    
    UnsignedWide now;
    unsigned long long usecs;
    Microseconds(&now); /* Microseconds (toolbox call) */
    usecs = ((((unsigned long long)now.hi)<<32) 
             + ((unsigned long long)now.lo));
    *nsecs = ((unsigned long)(usecs % 1000000ul)) * 1000ul;
    *secs = ((unsigned long)(usecs / 1000000ul));
  }
  return 1000000ul;
}  

static unsigned long __kern_get_secs_since_Jan1_1904(void)
{
  unsigned long since_jan1_1904;
  GetDateTime(&since_jan1_1904);
  return since_jan1_1904;
}

static unsigned long __kern_set_secs_since_Jan1_1904(unsigned long newtime)
{
  if (SetDateTime(newtime) != 0)
    return 0;
  return newtime;
}

static void __kern_get_timezone_info(long *seconds_east, 
                                     int *dst_in_effect_flag)
{
  long gmtDelta = 0;
  int in_dst = 0;
  MachineLocation theLocation;
  ReadLocation(&theLocation);

  if ((theLocation.u.dlsDelta & 0x80) != 0)
    in_dst = 1;  /* daylight savings is in effect */
  gmtDelta = theLocation.u.gmtDelta & 0x00ffffff;  /* mask off dlsDelta */
  if ( gmtDelta & 0x00800000 )  /* need to sign extend gmtDelta */
    gmtDelta |= 0xff000000;
  *seconds_east = gmtDelta;
  *dst_in_effect_flag = in_dst;
  return ;
}  

/* ====================================================================== */
/*                    END OF HARDWARE/OS PRIMITIVES                       */
/* ====================================================================== */

static int __internal_dsttype_guess = 0;   /* guess for DST_xxx */
#undef timezone               /* may collide with 'struct timezone' */
long timezone = 0;            /* seconds east of Greenwich (uncompensated) */
int  daylight = 0;            /* "is daylight savings in effect?" flag */
char *tzname[2] = {"GMT",""};

static void __internal_tzset(int only_if_needed /* eg from global init */ )
{
  static int need_initialization = 1;
  long minutes_west, my_seconds_east; 
  int my_daylight_flag;

  if (only_if_needed && !need_initialization)
    return;
  need_initialization = 0;

  __kern_get_timezone_info(&my_seconds_east, &my_daylight_flag);
  minutes_west = -((my_seconds_east-((my_daylight_flag)?(3600):(0))) / 60);

  if (minutes_west < -(12*60) || minutes_west > +(12*60))
  {
    __internal_dsttype_guess = 0;
    timezone = 0;
    daylight = 0;
    tzname[0] = "";
    tzname[1] = "";
  }
  else
  {  
    static char my_tzname_buf[2*5] = {0,0,0,0,0, 0,0,0,0,0};
    static struct { long min_west; int dsttype; const char *name; }
    zonemap[] = { 
                { -((12*60)+00), 1      ,  "DL" }, /* dateline */
                { -((11*60)+00), 1      ,  "WP" }, /* west-pacific */
                { -((10*60)+00), 2      ,  "AA" }, /* aust/arctic */
                { -(( 9*60)+30), 2      ,  "AS" }, /* adelaide/darwin */
                { -(( 9*60)+00), 2      ,  "AE" }, /* far-east asia */
                { -(( 8*60)+00), 1      ,  "ST" }, /* straights time */
                { -(( 7*60)+00), 1      ,  "NP" }, /* n-m-peninsular */
                { -(( 6*60)+00), 1      ,  "EI" }, /* e-indian ocean */
                { -(( 6*60)+30), 0      ,  "IS" }, /* indian standard,no-dst*/
                { -(( 5*60)+30), 1      ,  "MI" }, /* m-indian ocean */
                { -(( 5*60)+00), 1      ,  "CI" }, /* c-indian ocean */
                { -(( 4*60)+00), 1      ,  "AC" }, /* central-asia */
                { -(( 3*60)+30), 1      ,  "WI" }, /* w-indian-oc (caucacus) */
                { -(( 3*60)+00), 1      ,  "AN" }, /* near-east-asia (a-p) */
                { -(( 2*60)+00), 4      ,  "EE" }, /* eastern-europe */
                { -(( 1*60)+00), 4      ,  "ME" }, /* mid-europe */
                {  (( 0*60)+00), 0      ,  "GM" }, /* time meridian */
                { +(( 1*60)+00), 4      ,  "EA" }, /* east-atl (azores) */
                { +(( 2*60)+00), 1      ,  "MA" }, /* mid-atlantic */
                { +(( 3*60)+00), 1      ,  "CA" }, /* greenland */
                { +(( 4*60)+00), 1      ,  "WA" }, /* west-atlantic */
                { +(( 4*60)+30), 1      ,  "N"  }, /* new-foundland */
                { +(( 5*60)+00), 1      ,  "E"  }, /* eastern seaboard */
                { +(( 6*60)+00), 1      ,  "C"  }, /* central */
                { +(( 7*60)+00), 1      ,  "M"  }, /* mountain */
                { +(( 8*60)+00), 1      ,  "P"  }, /* pacific */
                { +(( 9*60)+00), 1      ,  "EP" }, /* east-pacific (alaska)*/
                { +((10*60)+00), 1      ,  "CP" }, /* (hawaii) */
                { +((11*60)+00), 1      ,  "MP" }, /* mid-pacific */
                { +((12*60)+00), 1      ,  "DL" }  /* dateline */
                };
    unsigned int i;            
    int dsttype = 0;
    for (i=1;i<(sizeof(zonemap)/sizeof(zonemap[0]));i++)
    {
      if (minutes_west  > zonemap[i-1].min_west && 
          minutes_west <= zonemap[i+0].min_west)
      {
        /* single letter names become 'xST,xDT'. double letter names */
        /* become xxT and xxS for west of GMT/xxD for east of GMT */
        char c[4] = {0,'S','D','T'};
        c[0] = (char)zonemap[i].name[0];
        if (zonemap[i].name[1])
        {
          c[1] = c[2] = (char)zonemap[i].name[1];
          c[3] = ((minutes_west < 0)?('S'):('D'));
        }  
        my_tzname_buf[0] = c[0]; my_tzname_buf[4] = c[0];
        my_tzname_buf[1] = c[1]; my_tzname_buf[5] = c[2];
        my_tzname_buf[2] = 'T';  my_tzname_buf[6] = c[3];
        my_tzname_buf[3] = '\0'; my_tzname_buf[7] = '\0';
        dsttype = zonemap[i].dsttype;
        break;
      }    
    }    
    __internal_dsttype_guess = dsttype;
    tzname[0] = &my_tzname_buf[0];
    tzname[1] = &my_tzname_buf[4];
    daylight  = my_daylight_flag;
    timezone  = my_seconds_east;
  }  
  return;
}

void tzset(void)
{
  __internal_tzset(0);
  return;
}  

/* -------------------------------------------------------------------- */

/* get the secs difference between what the libc uses as epoch and */
/* standard Mac epoch (1 Jan 1904, 0:00). The difference is then applied */
/* to all return values that return an absolute time so that they can be */
/* used in functions that convert time to something else, eg gmtime et al. */
/* If the libc epoch is greater than mac epoch (eg POSIX1), then the diff */
/* is positive, if the libc epoch is less than mac epoch (eg CWPro5), then */
/* the diff is negative. Thus, to convert a Mac epoch time_t to a libc epoch */
/* time_t, just add the return value of __internal_get_libc_epoch_diff(). */
/* If time() is implemented via gettimeofday(), then the diff will be zero. */

static long __internal_get_libc_epoch_diff(void)
{
  static long diffsecs = -1L;
  if (diffsecs == -1L)
  {
    /* standard mac epoch is Jan 1, 1904 0:00; 
       msl epoch (as of CWPro 5) is Jan 1, 1900, 0:00, (sub ((4 yrs*365)*86400)
       POSIX epoch is Jan 1, 1970; (add ((66 yrs*365)+17leap days)*86400 )
    */
    static int recursive = -1; /* gettimeofday() depends on this function, */
    if ((++recursive) != 0)    /* and this function depends on time(), and */
      diffsecs = 0;            /* time() *MIGHT* depend on gettimeofday(). */
    else                       /* Guard against those conditions. */
    {
      unsigned long kernsecs = 0, libcsecs = 0;
      long diffdays;
      do {
        libcsecs = ((unsigned long)time(0));
        kernsecs = __kern_get_secs_since_Jan1_1904();
      } while (((unsigned long)time(0)) != libcsecs);
      if (diffsecs == -1L) /* if no recursion happened */
      {
        diffdays = (((long)libcsecs) - ((long)kernsecs))/86400L; /* truncate */
        diffsecs = diffdays * 86400L;
      }  
      /* for POSIX1 epoch, diffsecs is +2082844800 (((66UL*365UL)+17UL)*24UL*60UL*60UL) */
      /* for CWPro5 epoch, diffsecs is -126144000  (((4UL*365UL)+0UL)*24UL*60UL*60UL) */
    }
    recursive--;
  }
  return diffsecs;
}

/* ---------------------------------------------------------------- */

static int __internal_settime( struct timespec *newtime, 
                               struct timespec *currtime,
                               long *gettime_nsdelta )
{
  long secs_diff = 0, new_delta = 0;
  struct timespec diff;

  if (((long)newtime->tv_nsec) >= 1000000000l || ((long)newtime->tv_nsec) < 0)
  {
    /* errno = EINVAL; */
    return -1;
  }
  else if (currtime->tv_sec < newtime->tv_sec || 
          (currtime->tv_sec == newtime->tv_sec && 
          currtime->tv_nsec < newtime->tv_nsec))
  {
    diff.tv_sec = currtime->tv_sec;
    diff.tv_nsec = currtime->tv_nsec;
    if (diff.tv_nsec < newtime->tv_nsec)
    {
      diff.tv_nsec += 1000000000ul;
      diff.tv_sec --;
    }  
    diff.tv_sec -= newtime->tv_sec;
    diff.tv_nsec -= newtime->tv_nsec;
    secs_diff = -diff.tv_sec;
    new_delta = -diff.tv_nsec;
  }
  else if (currtime->tv_sec > newtime->tv_sec ||  
          (currtime->tv_sec == newtime->tv_sec && 
           currtime->tv_nsec > newtime->tv_nsec))
  {
    diff.tv_sec = newtime->tv_sec;
    diff.tv_nsec = newtime->tv_nsec;
    if (diff.tv_nsec < currtime->tv_nsec)
    {
      diff.tv_nsec += 1000000000ul;
      diff.tv_sec --;
    }  
    diff.tv_sec -= currtime->tv_sec;
    diff.tv_nsec -= currtime->tv_nsec;
    secs_diff = +diff.tv_sec;
    new_delta = +diff.tv_nsec;
  }        
  if (secs_diff != 0)
  {
    unsigned long newsecs, oldsecs = __kern_get_secs_since_Jan1_1904();
    if (secs_diff < 0 && ((-secs_diff) > oldsecs))
    {
      /* errno = EINVAL */
      return -1;
    }
    newsecs = oldsecs + secs_diff;
    if (__kern_set_secs_since_Jan1_1904(newsecs) != newsecs)
    {
      __kern_set_secs_since_Jan1_1904(oldsecs);
      /* errno = EINVAL */
      return -1;
    }               
  }  
  *gettime_nsdelta = new_delta;
  return 0;
}  

/* ---------------------------------------------------------------- */

static int __internal_clock_getsettime(int doset, clockid_t clktype, struct timespec *tsp)
{
  static unsigned long proc_starttime_secs = 0, proc_starttime_nsecs = 0;
  static unsigned long realtime_base = 0;
  unsigned long secs, nsecs;
  /* int new_errno = EINVAL; */

  /* +++++++++++++ BEGIN STATIC INITIALIZATION +++++++++++++++ */
  if (proc_starttime_nsecs == 0) /* this should be done from a constructor */
  {             
    __internal_tzset(1); /* initialize tz vars if needed */
    realtime_base = __kern_get_secs_since_Jan1_1904();
    __kern_get_nano_uptime(&secs, &nsecs);
    if (nsecs == 0)
      nsecs = 1;
    proc_starttime_secs  = secs;
    proc_starttime_nsecs = nsecs;
    __internal_get_libc_epoch_diff(); /* initialize diff if needed */
  }
  /* +++++++++++++ END STATIC INITIALIZATION +++++++++++++++++ */

  if (doset && clktype != CLOCK_TIMEOFDAY)
  {
    ; /* fallthrough with errno = EINVAL */
  }  
  else if (clktype == CLOCK_REALTIME  ||
           clktype == CLOCK_VIRTUAL   ||
           clktype == CLOCK_PROF      ||   
           clktype == CLOCK_MONOTONIC ||
           clktype == CLOCK_TIMEOFDAY)
  {
    if (tsp)
    {
      __kern_get_nano_uptime(&secs, &nsecs);
      if (clktype == CLOCK_TIMEOFDAY)
      {   
        static struct timespec gettime_delta = {0,0};
        unsigned long time_now, delta_sec; long delta_nsec;
        delta_sec = gettime_delta.tv_sec;
        delta_nsec = gettime_delta.tv_nsec;

        secs += delta_sec;
        time_now = __kern_get_secs_since_Jan1_1904();
        if (secs < (time_now-1) || secs > (time_now+1)) 
        {                                   /* user changed time-of-day */
          secs -= delta_sec;
          delta_sec = time_now - secs;
          secs += delta_sec;
          gettime_delta.tv_sec = delta_sec;
        }  
        secs += __internal_get_libc_epoch_diff(); /*make it time() compatible*/
        if (doset)
        {
          struct timespec currtime;
          currtime.tv_sec = (time_t)secs;
          currtime.tv_nsec = (long)nsecs;
          return __internal_settime( tsp, &currtime, &gettime_delta.tv_nsec );
        }
        else if (delta_nsec != 0)
        {
          if (delta_nsec < 0 && nsecs < ((unsigned long)-delta_nsec))
          {
            nsecs += 1000000000ul;
            secs --;
          }
          nsecs += delta_nsec;
          if (nsecs > 1000000000ul)
          {
            nsecs -= 1000000000ul;
            secs ++;
          }
        }
      }
      else if (clktype == CLOCK_REALTIME)
      {
        if (nsecs < proc_starttime_nsecs)
        {
          nsecs += 1000000000ul;
          secs--;
        }
        nsecs-= proc_starttime_nsecs;
        secs -= proc_starttime_secs;
        secs += realtime_base;
        secs += __internal_get_libc_epoch_diff(); /*make it time() compatible*/
      }      
      else if (clktype == CLOCK_PROF ||  /* we don't have profiling counters */
               clktype == CLOCK_VIRTUAL) /* so these two are identical */
      {
        unsigned long thread_starttime_secs, thread_starttime_nsecs;
        unsigned long thread_kernltime_secs, thread_kernltime_nsecs;
        /* kernel time is time that thread sleeps/yields/does a toolbox call */

        #if 1 /* no thread statistics, so just get it for the whole process */
        thread_starttime_secs = proc_starttime_secs;
        thread_starttime_nsecs = proc_starttime_nsecs;
        thread_kernltime_secs = thread_kernltime_nsecs = 0;
        #else
        __kern_get_currthread_times(&thread_starttime_secs, &thread_starttime_nsecs,
                              &thread_kernltime_secs, &thread_kernltime_nsecs);
        #endif
        if (nsecs < thread_starttime_nsecs)
        {
          nsecs += 1000000000ul;
          secs--;
        }
        nsecs -= thread_starttime_nsecs;
        secs  -= thread_starttime_secs;
        if (nsecs < thread_kernltime_nsecs)
        {
          nsecs += 1000000000ul;
          secs--;
        }
        nsecs -= thread_kernltime_nsecs;
        secs  -= thread_kernltime_secs;
      }
      tsp->tv_sec = secs;
      tsp->tv_nsec = nsecs;
      return 0;
    }
    /* new_errno = EFAULT; */
  }  
  /* errno = new_errno; */
  return -1;
}  

/* ================================================================ */

int clock_gettime(clockid_t clktype, struct timespec *tsp)
{
  return __internal_clock_getsettime(0, clktype, tsp);
}

/* ---------------------------------------------------------------- */

int clock_settime(clockid_t clktype, struct timespec *tsp)
{
  return __internal_clock_getsettime(1, clktype, tsp);
}

/* ---------------------------------------------------------------- */

int clock_getres(clockid_t clktype, struct timespec *tsp)
{
  if (clktype == CLOCK_REALTIME  ||
      clktype == CLOCK_VIRTUAL   ||
      clktype == CLOCK_PROF      ||   
      clktype == CLOCK_MONOTONIC ||
      clktype == CLOCK_TIMEOFDAY)
  {
    if (tsp)
    {
      unsigned long secs, nsecs;
      tsp->tv_nsec = __kern_get_nano_uptime(&secs, &nsecs);
      tsp->tv_sec = 0;
    }
    return 0;
  }  
  /* errno = EINVAL; */
  return -1;
}  

/* ---------------------------------------------------------------- */

int settimeofday(struct timeval *tv, struct __timezone *tz)
{
  /* int new_errno = EINVAL; */
  if (!tz) /* timezone is never settable by settimeofday */
  {
    /* new_errno = EFAULT; */
    if (tv)
    {
      /* new_errno = EINVAL; */
      if (((long)tv->tv_usec) >= 0 && ((long)tv->tv_usec) < 1000000l)
      {
        struct timespec ts;
        ts.tv_sec = tv->tv_sec;
        ts.tv_nsec = tv->tv_usec * 1000;
        return clock_settime(CLOCK_TIMEOFDAY, &ts);
      }  
    }  
  }
  /* errno = new_errno; */
  return -1;
}

/* ---------------------------------------------------------------- */

int gettimeofday(struct timeval *tv, struct __timezone *tz)
{
  if (tv)
  {
    struct timespec ts;
    if (clock_gettime(CLOCK_TIMEOFDAY, &ts) != 0)
      return -1;
    if (tz)
    {
      long minutes_west = -((timezone-((daylight)?(3600):(0)))/60);
      int  dsttype = ((daylight)?(__internal_dsttype_guess):(0 /*DST_NONE*/));
      tz->tz_minuteswest = (int)minutes_west; /*minutes west of Greenwich*/
      tz->tz_dsttype = dsttype; /* type of daylight switch */
    }
    tv->tv_sec = ts.tv_sec;
    tv->tv_usec = ts.tv_nsec/1000;
    return 0;
  }
  /* errno = EFAULT; */
  return -1;
}  

/* =================================================================== */

