/* Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This file contains functions for obtaining/formatting/manipulating
 * the time. 'time' is usually stored/passed/returned in timeval format.
 *
 * Please use native OS functions where possible.
 *                                                                 - cyp
 * ----------------------------------------------------------------------
*/
const char *clitime_cpp(void) {
return "@(#)$Id: clitime.cpp,v 1.37.2.53 2001/05/06 11:01:06 teichp Exp $"; }

#include "cputypes.h"
#include "baseincs.h" // for timeval, time, clock, sprintf, gettimeofday etc
#include "clitime.h"  // keep the prototypes in sync

#if (CLIENT_OS == OS_WIN32) && (CLIENT_CPU == CPU_ALPHA)
extern "C" int _AcquireSpinLockCount(long *, int);
extern "C" void _ReleaseSpinLock(long *);
#pragma intrinsic(_AcquireSpinLockCount, _ReleaseSpinLock)
#endif


#if defined(__unix__)
  #define HAVE_GETRUSAGE
  #include <sys/resource.h>
  #undef THREADS_HAVE_OWN_ACCOUNTING
  #if !defined(CLIENT_SUPPORTS_SMP) || \
      defined(HAVE_MULTICRUNCH_VIA_FORK) || \
      (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD)
    // if threads have their own pid, then we can use getrusage() to
    // obtain thread-time, otherwise resort to something else
    #define THREADS_HAVE_OWN_ACCOUNTING
  #endif
#endif

/* --------------------------------------------------------------------- */

int InitializeTimers(void)
{
  #if (CLIENT_OS != OS_AMIGAOS)
  CliIsTimeZoneInvalid(); /* go assume TZ=GMT if invalid timezone */
  tzset();                /* set correct timezone for everyone else */
  #endif
  if (CliClock(NULL)!=0) /* do the one-shot clock init */
    return -1;
  CliGetThreadUserTime(NULL); /* do init if needed */
  /* currently don't have anything else to do */
  return 0;
}

int DeinitializeTimers(void)
{
  return 0;
}

/* --------------------------------------------------------------------- */

/* Get current system time (UTC). */
static int __GetTimeOfDay( struct timeval *tv )
{
  if (tv)
  {
    #if (CLIENT_OS == OS_SCO) || (CLIENT_OS == OS_OS2) || \
        (CLIENT_OS == OS_VMS) || (CLIENT_OS == OS_WIN16)
    {
      struct timeb tb;
      ftime(&tb);
      tv->tv_sec = tb.time;
      tv->tv_usec = tb.millitm*1000;
    }
    #elif (CLIENT_OS == OS_WIN32)
    {
      unsigned __int64 now, epoch;
      unsigned long ell;
      FILETIME ft;
      SYSTEMTIME st;
      GetSystemTime(&st);
      SystemTimeToFileTime(&st, &ft);
      //epoch.dwHighDate = 27111902UL;
      //epoch.dwLowDate = 3577643008UL;
      epoch = 116444736000000000ui64;
      now = ft.dwHighDateTime;
      now <<= 32;
      now += ft.dwLowDateTime;
      now -= epoch;
      now /= 10UL;
      ell = (unsigned long)(now % 1000000ul);
      tv->tv_usec = ell;
      ell = (unsigned long)(now / 1000000ul);
      tv->tv_sec = ell;
    }
    #elif (CLIENT_OS == OS_NETWARE)
    {
      unsigned long cas[3];
      unsigned long secs, fsec;

      /* emulated for nw3 in nwlemu.c */
      GetClockStatus(cas);

      secs = cas[0]; /* full secs (UTC) */
      fsec = cas[1]; /* frac secs 0-0xfffffffful */
      /* cas[3] has sync state flags */

      #if (CLIENT_CPU == CPU_X86)
      /* avoid yanking in watcom's crappy static clib just for int64 mul */
      _asm mov eax, fsec
      _asm xor edx, edx
      _asm mov ecx, 1000000
      _asm mul ecx
      _asm mov fsec, edx /* edx:eax divided by 1<<32 */
      #else
      fsec = (unsigned long)(((unsigned __int64)
             (((unsigned __int64)fsec) * 1000000ul)) >> 32);
      #endif

      tv->tv_sec = (time_t)secs;
      tv->tv_usec = (long)fsec;
    }
    #elif (CLIENT_OS == OS_RISCOS)
    {
      time_t t;

      time(&t);
      tv->tv_sec=t;
      tv->tv_usec=0;
    }
    #else
    {
      //struct timezone tz;
      return gettimeofday(tv, 0);
    }
    #endif
  }
  return 0;
}

/* --------------------------------------------------------------------- */

// timezone offset such that the number returned is constant for any
// time of year (ie, dst unadjusted, just as if the date was always Jan 1)
// "minuteswest" means west of utc > 0, east < 0, ie utctime-localtime
// CliGetMinutesWest() caches the value returned from __GetMinutesWest()

static int __GetMinutesWest(void)
{
  int minwest;
#if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_WIN16) || \
  ((CLIENT_OS == OS_OS2) && !defined(__EMX__))
  /* ANSI rules :) - 'timezone' doesn't reflect 'daylight' state. */
  /* ie utctime-localtime == timezone-(daylight*3600) */
  minwest = (int)(((long)timezone)/60L);
#elif (CLIENT_OS == OS_WIN32)
  TIME_ZONE_INFORMATION TZInfo;
  if (GetTimeZoneInformation(&TZInfo) == 0xFFFFFFFFL)
    return 0;
  minwest = TZInfo.Bias; /* sdk doc is wrong. .Bias is always !dst */
#elif (CLIENT_OS == OS_SCO) || (CLIENT_OS == OS_VMS) || \
      (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_NETBSD) || \
      (CLIENT_OS == OS_OPENBSD) || (CLIENT_OS == OS_BSDOS) || \
      (CLIENT_OS == OS_MACOSX) /* *BSDs don't set timezone in gettimeofday() */
  time_t timenow;
  struct tm * tmP;
  struct tm loctime, utctime;
  int haveutctime, haveloctime, tzdiff;

  tzset();
  timenow = time(NULL);
  tmP = localtime( (const time_t *) &timenow);
  if (!tmP) return 0;
  haveloctime = (tmP != NULL);
  if (haveloctime != 0)
    memcpy( &loctime, tmP, sizeof( struct tm ));
  tmP = gmtime( (const time_t *) &timenow);
  if (!tmP) return 0;
  haveutctime = (tmP != NULL);
  if (haveutctime != 0)
    memcpy( &utctime, tmP, sizeof( struct tm ));
  if (!haveutctime && !haveloctime)
    return 0;
  if (haveloctime && !haveutctime)
    memcpy( &utctime, &loctime, sizeof( struct tm ));
  else if (haveutctime && !haveloctime)
    memcpy( &loctime, &utctime, sizeof( struct tm ));

  tzdiff =  ((loctime.tm_min  - utctime.tm_min) )
          +((loctime.tm_hour - utctime.tm_hour)*60 );
  /* last two are when the time is on a year boundary */
  if      (loctime.tm_yday == utctime.tm_yday)     { ;/* no change */}
  else if (loctime.tm_yday == utctime.tm_yday + 1) { tzdiff += 1440; }
  else if (loctime.tm_yday == utctime.tm_yday - 1) { tzdiff -= 1440; }
  else if (loctime.tm_yday <  utctime.tm_yday)     { tzdiff += 1440; }
  else                                             { tzdiff -= 1440; }

  if (loctime.tm_isdst > 0)
    tzdiff -= 60;
  if (tzdiff < -(12*60))
    tzdiff = -(12*60);
  else if (tzdiff > +(12*60))
    tzdiff = +(12*60);

  minwest = -tzdiff;
#else
  /* POSIX rules :) */
  struct timezone tz; struct timeval tv;
  if ( gettimeofday(&tv, &tz) )
    return 0;
  minwest = tz.tz_minuteswest;
  if (tz.tz_dsttime) /* is this correct? does minuteswest really change when */
    minwest += 60;   /* dst comes into effect/is no longer effective? */
#endif
  return minwest;
}

/* --------------------------------------------------------------------- */

static int precalced_minuteswest = -1234;
static int adj_time_delta = 0;

/* offset in seconds to add to value returned by CliTimer() */

int CliTimerSetDelta( int delta )
{
  int old = adj_time_delta;
  adj_time_delta = delta;
  if ( ((old<delta)?(delta-old):(old-delta)) >= 20 )
    precalced_minuteswest = -1234;
  return old;
}

/* --------------------------------------------------------------------- */

// timezone offset after compensating for dst (west of utc > 0, east < 0)
// such that the number returned is constant for any time of year
// CliGetMinutesWest() caches the value returned from __GetMinutesWest()

int CliTimeGetMinutesWest(void)
{
  if (precalced_minuteswest == -1234)
    precalced_minuteswest = __GetMinutesWest();
  return precalced_minuteswest;
}

// ---------------------------------------------------------------------

/* Get the current time in timeval format (pass NULL if storage not req'd) */
struct timeval *CliTimer( struct timeval *tv )
{
  static struct timeval stv = {0,0};
  struct timeval ttv;
  if (__GetTimeOfDay( &ttv ) == 0)
  {
    stv.tv_sec = ttv.tv_sec;
    stv.tv_usec = ttv.tv_usec;
    stv.tv_sec += adj_time_delta;
  }
  if (tv)
  {
    tv->tv_sec = stv.tv_sec;
    tv->tv_usec = stv.tv_usec;
    return tv;
  }
  return (&stv);
}

/* --------------------------------------------------------------------- */

// CliClock() is a wrapper around CliGetMonotonicClock() and returns the
// time since the first call (assumed to be the time the client started).
// Please don't be tempted to merge this functionality into GetMonotonicClock

int CliClock(struct timeval *tv)
{
  static struct timeval base = {0,0};
  static int need_base_time = 1;
  if (need_base_time)
  {
    if (CliGetMonotonicClock(&base) != 0)
    {
      return -1;
    }
    need_base_time = 0;
    if (tv)
    {
      tv->tv_sec = 0;
      tv->tv_usec = 1;
    }
    return 0;
  }

  if (tv)
  {
    struct timeval now;
    if (CliGetMonotonicClock(&now) != 0)
    {
      return -1;
    }
    if (now.tv_sec < base.tv_sec ||
       (now.tv_sec == base.tv_sec && now.tv_usec < base.tv_usec))
    {
      return -1;
    }
    if (now.tv_usec < base.tv_usec)
    {
      now.tv_usec += 1000000UL;
      now.tv_sec--;
    }
    tv->tv_usec = now.tv_usec - base.tv_usec;
    tv->tv_sec  = now.tv_sec  - base.tv_sec;
  }
  return 0;
}

/* --------------------------------------------------------------------- */
// only used if clock can't be used
#if !defined(CLOCK_REALTIME)
// __clks2tv() is called/inline'd from CliGetMonotonicClock() and converts
// 'ticks' (ticks/clks/whatever: whatever it is that your time-since-whenever
// function returns) to secs/usecs. 'hz' is that function's equivalent of
// CLOCKS_PER_SECOND. 'wrap_ctr' is the number of times 'ticks' wrapped.
// Caveat emptor: 'hz' cannot be greater than 1000000ul

inline void __clks2tv( unsigned long hz, register unsigned long ticks,
                       unsigned long wrap_ctr, struct timeval *tv )
{
  register unsigned long sadj = 0, wadj = 0;
  if (wrap_ctr) /* number of times 'ticks' wrapped */
  {
    sadj = (hz/10); /* intermediate temp to suppress optimization */
    wadj = wrap_ctr * (6UL+(10*((ULONG_MAX/10UL)%(hz/10)))); /* ((1<<ws)%hz) */
    sadj = wrap_ctr * ((ULONG_MAX/10UL)/sadj);               /* ((1<<ws)/hz) */
    sadj += wadj / hz; wadj %= hz;
  }
  tv->tv_sec = (time_t) ( (ticks / hz) + sadj);
  tv->tv_usec = (long)( ( (ticks % hz) + wadj) * ((1000000ul+(hz>>1))/hz) );
  return;
}
#endif /* !defined(CLOCK_REALTIME) */

/* --------------------------------------------------------------------- */

// CliGetMonotonicClock() should return a ...
// real (not virtual per process, but secs that increment as a wall clock
// would), monotonic (won't speed up/slow down), linear time (won't go
// backward, won't wrap) and is not subject to resetting or the user changing
// day/date. It need not be correlated to the time-of-day. The epoch
// (base time) can be anything just as long as it remains constant over
// the course of the client's lifetime.
//
// This function is used to determine total elapsed runtime for completed
// work (both single and "Summary:" stats - see CliClock() above).
//
// If CliGetThreadUserTime() is not supported, then this function is also
// used for the other (fine-res) crunch timing eg core selection/timeslice
// optimization etc.
//
// On non-preemptive systems this function is particularly critical since
// clients on non-preemptive systems measure their own run/yield quantums.

int CliGetMonotonicClock( struct timeval *tv )
{
  if (tv)
  {
    #if (CLIENT_OS == OS_BEOS)
    {
      bigtime_t now = system_time();
      tv->tv_sec = (time_t)(now / 1000000LL); /* microseconds -> seconds */
      tv->tv_usec = (time_t)(now % 1000000LL); /* microseconds < 1 second */
    }
    #elif (CLIENT_OS == OS_NETWARE)
    {
      /* atomic_xchg()/MPKYieldThread() are stubbed/emulated in nwmpk.c */
      /* [NW]GetHighResolutionTimer() is stubbed/emulated in nwlemu.c */
      static char splbuf[4] = {0,0,0,0}; /* 64bit buffer in data,not bss */
      static unsigned long wrap_count = 0, last_ctr = 0;
      char *splcp = (char *)&splbuf[0]; unsigned long *spllp;
      unsigned long ctr, l_wrap_count = 0;
      int locktries = 0, lacquired = 0;

      while ((((unsigned long)splcp) & (sizeof(unsigned long)-1)) != 0)
        splcp++;
      spllp = (unsigned long *)splcp;
      while (!lacquired)
      {
        if (atomic_xchg(spllp,1)==0)
          lacquired = 1;
        if (!lacquired && ((++locktries)&0x0f)==0)
          MPKYieldThread();
      }
      ctr = GetHighResolutionTimer(); /* 100us count since boot */
      l_wrap_count = wrap_count;
      if (ctr < last_ctr)
        wrap_count = ++l_wrap_count;
      last_ctr = ctr;
      *spllp = 0;
      __clks2tv( 10000, ctr, l_wrap_count, tv );
    }
    #elif (CLIENT_OS == OS_RISCOS)
    {
      static unsigned long last_ctr = 0, wrap_ctr = 0;
      unsigned long ctr = read_monotonic_time(); /* hsecs since boot */
      if (ctr < last_ctr) wrap_ctr++;
      last_ctr = ctr;
      __clks2tv( 100, ctr, wrap_ctr, tv );
      //adj = (wrap_ctr * 96UL);
      //tv->tv_usec = 100000UL * ((ticks%100) + (adj % 100));
      //tv->tv_sec = (time_t)((ticks/100)+(adj/100)+(wrap_ctr*42949672UL));
    }
    #elif (CLIENT_OS == OS_WIN16)
    {
      static DWORD last_ctr = 0, wrap_ctr = 0;
      DWORD ctr = GetTickCount(); /*millisec since boot*/
      if (ctr < last_ctr) wrap_ctr++;
      last_ctr = ctr;
      __clks2tv( 1000, ctr, wrap_ctr, tv );
      //adj = (wrap_ctr * 296UL);
      //tv->tv_usec = ((ticks % 1000) + (adj % 1000)) * 1000UL;
      //tv->tv_sec = (time_t)((ticks/1000)+(adj/1000)+(wrap_ctr*4294967UL));
    }
    #elif (CLIENT_OS == OS_WIN32)
    {
      #if 0 /* too many failures to be useful */
      static int using_qff = -1;
      if (using_qff != 0)
      {
        static unsigned __int64 freq = 0;
        unsigned __int64 now; int gotit = 0;
        LARGE_INTEGER qperf;
        if (winGetVersion() >= 400 && /* not efficient on win32s */
           QueryPerformanceCounter(&qperf))
        {
          if (qperf.LowPart || qperf.HighPart)
          {
            gotit = +1;
            if (using_qff < 0)
            {
              /* guard against Japanese Win95 (PC9800 version) which always
              ** returns 1193180 (.eq. QueryPerfFrequency()) as counter.
              ** See KB article Q152145
              */
              LARGE_INTEGER qcheck;
              Sleep(5); /* not really necessary, but doesn't hurt */
              gotit = 0;
              if (QueryPerformanceCounter(&qcheck))
              {
                if ((qcheck.LowPart || qcheck.HighPart) &&
                    ((qcheck.LowPart != qperf.LowPart) ||
                     (qcheck.HighPart != qperf.HighPart)))
                {
                  qperf.LowPart = qcheck.LowPart;
                  qperf.HighPart = qcheck.HighPart;
                  if (QueryPerformanceFrequency(&qcheck))
                  {
                    now = qcheck.HighPart;
                    now <<= 32;
                    now += qcheck.LowPart;
                    freq = now;
                    gotit = +1;
                  }
                }
              }
            }
          }
        }
        if (using_qff < 0)
          using_qff = gotit;
        else if (!gotit)
          return -1;
        if (gotit)
        {
          now = qperf.HighPart;
          now <<= 32;
          now += qperf.LowPart;
          tv->tv_sec = (time_t)(now / freq);
          now = now % freq;
          now = now * 1000000ui64;
          tv->tv_usec = (time_t)(now / freq);
          return 0;
        }
        /* fallthrough: using_qff == 0 */
      }
      #endif
      /* if (using_qff == 0) */
      {
        /* '-benchmark rc5' in keys/sec with disabled ThreadUserTime(): */
        /* using spinlocks: 1,386,849.10 | using critical section: 994,861.48 */
        static long splbuf[4] = {0,0,0,0}; /* 64bit*2 */
        static DWORD lastticks = 0, wrap_count = 0;
        DWORD ticks, l_wrap_count; long *spllp;
        char *splcp = (char *)&splbuf[0];

        int lacquired = 0, locktries = 0;
        splcp += ((64/8) - (((unsigned long)splcp) & ((64/8)-1)));
        spllp = (long *)splcp; /* long * to 64bit-aligned spinlock space */
        while (!lacquired)
        {
          #if (CLIENT_CPU == CPU_ALPHA)
          lacquired = _AcquireSpinLockCount(spllp, 0x0f); /* VC6 intrinsic */
          locktries += 0x0f;
          #else
          if (InterlockedExchange(spllp,1)==0) /* spl must be 32bit-aligned */
            lacquired = 1;
          #endif
          if (!lacquired && ((++locktries)&0x0f)==0)
            Sleep(0);
        }
        ticks = GetTickCount(); /* millisecs elapsed since OS start */
        l_wrap_count = wrap_count;
        if (ticks < lastticks)
          wrap_count = ++l_wrap_count;
        lastticks = ticks;
        #if (CLIENT_CPU == CPU_ALPHA)
        _ReleaseSpinLock(spllp);  /* VC6 intrinsic */
        #else
        *spllp = 0;
        #endif
        __clks2tv( 1000, ticks, l_wrap_count, tv );
      }
    }
    #elif (CLIENT_OS == OS_OS2)
    {
      static long splbuf[2] = {0,0}; /* space for 32bit alignment */
      char *splptr = (char *)&splbuf[0];
      ULONG ticks, l_wrap_count = 0;
      int gotit = 0, lacquired = 0;

      splptr += (sizeof(long)-(((unsigned long)splptr) & (sizeof(long)-1)));
      while (!lacquired)
      {
        #if defined(__GNUC__)
        /* gcc is sometimes too clever */
        struct __fool_gcc_volatile { unsigned long a[100]; };
        /* note: no 'lock' prefix even on SMP since xchg is always atomic */
        __asm__ __volatile__(
                   "movl $1,%0\n\t"
                   "xchgl %0,%1\n\t"
                   "xorl $1,%0\n\t"
                   : "=r"(lacquired)
                   : "m"(*((struct __fool_gcc_volatile *)(splptr)))
                   : "memory");
        #elif defined(__WATCOMC__)
        _asm mov edx, splptr
        _asm mov eax, 1
        _asm xchg eax,[edx]
        _asm xor eax, 1
        _asm mov lacquired,eax
        #else
        #error whats up doc?
        #endif
        if (!lacquired)
          DosSleep(0);
      }
      if (!DosQuerySysInfo(QSV_MS_COUNT, QSV_MS_COUNT, &ticks, sizeof(ticks)))
      {
        static ULONG wrap_count = 0, lastticks = 0;
        l_wrap_count = wrap_count;
        if (ticks < lastticks)
          wrap_count = ++l_wrap_count;
        lastticks = ticks;
        gotit = 1;
      }
      *((long *)splptr) = 0;
      if (!gotit)
        return -1;
      __clks2tv( 1000, ticks, l_wrap_count, tv );
    }
    #elif defined(CTL_KERN) && defined(KERN_BOOTTIME) /* *BSD */
    {
      struct timeval boot, now;
      int mib[2]; size_t argsize = sizeof(boot);
      mib[0] = CTL_KERN; mib[1] = KERN_BOOTTIME;
      if (gettimeofday(&now, 0))
        return -1;
      if (sysctl(&mib[0], 2, &boot, &argsize, NULL, 0) == -1)
        return -1;
      if (now.tv_sec < boot.tv_sec || /* should never happen */
          (now.tv_sec == boot.tv_sec && now.tv_usec < boot.tv_sec))
        return -1;
      if (now.tv_usec < boot.tv_usec) {
        now.tv_usec += 1000000;
        now.tv_sec--;
      }
      tv->tv_sec = now.tv_sec - boot.tv_sec;
      tv->tv_usec = now.tv_usec - boot.tv_usec;
    }
    #elif (CLIENT_OS == OS_DOS)
    {
      /* in platforms/dos/dostime.cpp */
      if (getmicrotime(tv)!=0)
        return -1;
    }
    #elif (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_SOLARIS)
    {
      hrtime_t hirestime = gethrtime(); /* nanosecs since boot */
      hirestime /= 1000; /* nanosecs to microsecs */
      tv->tv_sec = (time_t)(hirestime / 1000000);
      tv->tv_usec = (unsigned long)(hirestime % 1000000);
    }
    #elif (CLIENT_OS == OS_LINUX) /*only RTlinux has clock_gettime/gethrtime*/
    {
      /* this is computationally expensive, but we don't have a choice.
         /proc/uptime is buggy even in the newest kernel (2.4-test2):
         it wraps at jiffies/HZ, ie ~497 days on a 32bit cpu (and the
         fact that that hasn't been noticed in 5 years is a pretty good
         indication that no linux box ever runs more than 497 days :)
      */
      #ifdef HAVE_KTHREADS
      int fd = -1;
      #else
      static int fd = -1;
      #endif
      int rc = -1;
      if (fd == -1)
        fd = open("/proc/uptime",O_RDONLY);
      if (fd != -1)
      {
        if (lseek( fd, 0, SEEK_SET)==0)
        {
          char buffer[128];
          int len = read( fd, buffer, sizeof(buffer));
          if (len >= 1 && len < ((int)(sizeof(buffer)-1)) )
          {
            unsigned long tt = 0, t2 = 0, t1 = 0;
            register char *p = buffer;
            buffer[len-1] = '\0';
            while (t1>=tt && *p >= '0' && *p <='9')
            {
              tt = t1;
              t1 = (t1*10)+((*p++)-'0');
	    }
            if (*p++ == '.')
            {
              tt=0;
              while (t2>=tt && *p >= '0' && *p <='9')
              {
                tt = t2;
                t2 = (t2*10)+((*p++)-'0');
	      }
              if (*p++ == ' ')
              {
                tv->tv_usec = (long)(10000UL * t2);
                tv->tv_sec = (time_t)t1;
                //printf("\rt=%d.%06d\n",tv->tv_sec,tv->tv_usec);
                rc = 0;
              }
            }
          }
        } /* lseek */
        #ifdef HAVE_KTHREADS
        close(fd);
        #endif
      } /* open */
      return rc;
    }
    #elif (CLIENT_OS == OS_AMIGAOS)
    {
      if (amigaGetMonoClock(tv) != 0)
        return -1;
    }
    #elif defined(CLOCK_MONOTONIC) /* POSIX 1003.1c */
    {                           /* defined doesn't always mean supported :( */
      struct timespec ts;
      if (clock_gettime(CLOCK_MONOTONIC, &ts))
        return -1;
      tv->tv_sec = ts.tv_sec;
      tv->tv_usec = ts.tv_nsec / 1000;
    }
    #elif defined(CLOCK_REALTIME) /* POSIX 1003.1b-1993 but not 1003.1-1990 */
    {
      struct timespec ts;
      if (clock_gettime(CLOCK_REALTIME, &ts))
        return -1;
      tv->tv_sec = ts.tv_sec;
      tv->tv_usec = ts.tv_nsec / 1000;
    }
    #elif 0
    {
      /* ***** this code is not thread-safe! ******
         AND DO NOT USE THIS WITHOUT ENSURING ...
         a) that clock() is not dependant on system time (all watcom clibs
            have this bug). Otherwise you may as well use __GetTimeOfDay()
         b) that clock() does not return virtual time. Under unix clock()
            is often implemented via times() and is thus virtual. Look at
            uptime source or see if something in top source (get_system_info
            in machine/m_[yourplat].c) is usable.
         c) clock_t is at least an unsigned long
         d) that the value from clock() does indeed count up to ULONG_MAX
            before wrapping. At least one implementation (EMX <=0.9) is
            known to wrap at (0xfffffffful/10).
         e) CLOCKS_PER_SECOND is not > 1000000
         ***** this code is not thread-safe! ******
      */
      static unsigned long lastcheck = 0;
      static unsigned long wrap_count = 0;
      unsigned long l_wrap_count, counter;
      clock_t raw_counter;

      raw_counter = clock();
      if (raw_counter == ((clock_t)-1))
        return -1;
      counter = (unsigned long)raw_counter;
      /* this code is not thread-safe! */
      l_wrap_count = wrap_count;
      if (counter < lastcheck)
        wrap_count = ++l_wrap_count;
      lastcheck = counter;

      __tick2tv( CLOCKS_PER_SEC, counter, l_wrap_count, tv );
    }
    #elif (CLIENT_OS == OS_DYNIX)
    // This is bad, but I at a loss to find something better.
    return __GetTimeOfDay( tv );
    #else
    // this is a bad thing because time-of-day is user modifiable.
    //if (__GetTimeOfDay( tv ))
      return -1;
    #endif
  }
  return 0;
}

/* --------------------------------------------------------------------- */

/* Get thread (user) cpu time, used for fine-slice benchmark etc. */
/* if tv is NULL, the function must still return 0 if supported */
int CliGetThreadUserTime( struct timeval *tv )
{
#if (CLIENT_OS == OS_BEOS)
  if (tv)
  {
    thread_info tInfo;
    get_thread_info(find_thread(NULL), &tInfo);
    tv->tv_sec = tInfo.user_time / 1000000; // convert from microseconds
    tv->tv_usec = tInfo.user_time % 1000000;
  }
  return 0;
#elif defined(HAVE_GETRUSAGE) && defined(THREADS_HAVE_OWN_ACCOUNTING)
  if (tv)
  {
    struct rusage rus;
    if (getrusage(RUSAGE_SELF,&rus) != 0)
      return -1;
    tv->tv_sec = rus.ru_utime.tv_sec;
    tv->tv_usec = rus.ru_utime.tv_usec;
    //printf("\rgetrusage(%d) => %d.%02d\n", getpid(), tv->tv_sec, tv->tv_usec/10000 );
  }
  return 0;
#elif (CLIENT_OS == OS_WIN32)
  static int is_supp = -1;
  FILETIME ct,et,kt,ut;
  if (is_supp == 0)
    return -1;
  if ( !tv && is_supp > 0 )
    return 0;
  if (!GetThreadTimes(GetCurrentThread(),&ct,&et,&kt,&ut))
  {
    if (is_supp < 0) /* first time? */
      is_supp = 0; /* don't try again */
    return -1;
  }
  if (tv)
  {
    unsigned __int64 now, epoch;
    unsigned long ell;
    //epoch.dwHighDate = 27111902UL;
    //epoch.dwLowDate = 3577643008UL;
    epoch = 116444736000000000ui64;
    now = ut.dwHighDateTime;
    now <<= 32;
    now += ut.dwLowDateTime;
    now -= epoch;
    now /= 10UL;
    ell = (unsigned long)(now % 1000000ul);
    tv->tv_usec = ell;
    ell = (unsigned long)(now / 1000000ul);
    tv->tv_sec = ell;
  }
  is_supp = 1;
  return 0;
#else
  tv = tv; /* shaddup compiler */
  return -1;
#endif
}

// ---------------------------------------------------------------------

// Add 'tv1' to 'tv2' and store in 'result'. Uses curr time if a 'tv' is NULL
// tv1/tv2 are not modified (unless 'result' is the same as one of them).
int CliTimerAdd( struct timeval *result, const struct timeval *tv1, const struct timeval *tv2 )
{
  if (result)
  {
    if (!tv1 || !tv2)
    {
      CliTimer( result );
      if (!tv1 && !tv2)
        return 0;
      if (!tv1)
        tv1 = (const struct timeval *)result;
      if (!tv2)
        tv2 = (const struct timeval *)result;
    }
    result->tv_sec = tv1->tv_sec + tv2->tv_sec;
    result->tv_usec = tv1->tv_usec + tv2->tv_usec;
    if (result->tv_usec >= 1000000L)
    {
      result->tv_sec += result->tv_usec / 1000000L;
      result->tv_usec %= 1000000L;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------

// Store non-negative diff of tv1 and tv2 in 'result'. Uses current time if a 'tv' is NULL
// tv1/tv2 are not modified (unless 'result' is the same as one of them).
int CliTimerDiff( struct timeval *result, const struct timeval *tv1, const struct timeval *tv2 )
{
  if (result)
  {
    if (!tv1 && !tv2)
      result->tv_sec = result->tv_usec = 0;
    else
    {
      struct timeval tvdiff, tvtemp;
      const struct timeval *tv0;
      if (!tv1 || !tv2)
      {
        CliTimer( &tvtemp );
        if (!tv1) tv1 = (const struct timeval *)&tvtemp;
        else tv2 = (const struct timeval *)&tvtemp;
      }
      if ((((unsigned long)(tv2->tv_sec)) < ((unsigned long)(tv1->tv_sec))) ||
         ((tv2->tv_sec == tv1->tv_sec) &&
           ((unsigned long)(tv2->tv_usec)) < ((unsigned long)(tv1->tv_usec))))
      {
        tv0 = tv1; tv1 = tv2; tv2 = tv0;
      }
      tvdiff.tv_sec = tv2->tv_sec;
      tvdiff.tv_usec = tv2->tv_usec;
      if (((unsigned long)(tvdiff.tv_usec)) < ((unsigned long)(tv1->tv_usec)))
      {
        tvdiff.tv_usec += 1000000L;
        tvdiff.tv_sec--;
      }
      result->tv_sec  = tvdiff.tv_sec - tv1->tv_sec;
      result->tv_usec = tvdiff.tv_usec - tv1->tv_usec;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------

// Determines if the there is sufficient information to determine
// the configured system timezone.  Returns 0 if okay, non-zero if
// the timezone is not known (thus default to GMT).

int CliIsTimeZoneInvalid(void)
{
  #if ((CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16) || \
       (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32))
  static int needfixup = -1;
  if (needfixup == -1)
  {
    needfixup = 0;
    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
    if (winGetVersion() < 400)
    #endif
    if (!getenv("TZ"))
    {
      // No timezone was yet configured, so just assume GMT.
      needfixup = 1;
      putenv("TZ=GMT+0");
      tzset();
    }
  }
  return needfixup;
  #elif (CLIENT_OS == OS_MACOS)
  #warning I should make use of this sometime in the future!
  return 0;
  #else
  return 0;
  #endif
}

// ---------------------------------------------------------------------

// Get time as string. Curr time if tv is NULL. Separate buffers for each
// type: 0=blank type 1,
//       1="MMM dd hh:mm:ss UTC",
//       2="hhhh:mm:ss.pp"
//       3="yyyy/mm/dd hh:mm:ss" (iso/cvs format, implied utc)
//       4="yymmddhh"            (bugzilla format)
//       5="MMM dd hh:mm:ss ZTZ" (like 1, but localtime)
const char *CliGetTimeString( const struct timeval *tv, int strtype )
{
  static const char *monnames[]={ "Jan","Feb","Mar","Apr","May","Jun",
                                  "Jul","Aug","Sep","Oct","Nov","Dec"};
  static unsigned long timelast = 0;
  static const char *timestr = "";
  static int lasttype = 0;
  unsigned long longtime;

  if (strtype < -1 || strtype > 5)
    return "";

  if (strtype == 0)
  {
    static char spacestring[30] = {0};
    if (!spacestring[0])
    {
      register char *ss = spacestring;
      strcpy( spacestring, CliGetTimeString( NULL, 1 ) );
      while (*ss) *ss++=' ';
    }
    return spacestring;
  }

  if (!tv) tv = CliTimer(NULL);/* show where CliTimer() is returning gunk */
  longtime = (unsigned long)tv->tv_sec;

  if (strtype == 2)
  {
    static char hourstring[8+1 +2+1 +2+1+2 +1 +2];
    int days = (longtime / 86400UL);
    if (days < 0 || days > 365)
      return "-.--:--:--.--";
    sprintf( hourstring,  "%d.%02d:%02d:%02d.%02d", (int) (longtime / 86400UL),
      (int) ((longtime % 86400L) / 3600UL), (int) ((longtime % 3600UL)/60),
      (int) (longtime % 60), (int) ((tv->tv_usec/10000L)%100) );
      //if ((longtime / 86400UL)==0 ) //don't show days if not needed
      //  return &hourstring[2]; // skip the "0."
    return hourstring;
  }

  if (longtime && ((longtime != timelast) || (lasttype != strtype)))
  {
    time_t timenow = tv->tv_sec;
    struct tm *gmt = (struct tm *)0;
    struct tm tmbuf;

    if (CliIsTimeZoneInvalid()) /* initializes it if not initialized */
    {
      if (strtype == 1)
        strtype = 5; /* like 1 but local time */
    }

    lasttype = strtype;
    timelast = longtime;

    if (strtype == 5) /* "MMM dd hh:mm:ss ZTZ" (like 1, but localtime) */
    {
      gmt = localtime( (const time_t *) &timenow);
      strtype = 1; /* just like 1 */
    }
    else
    {
      gmt = gmtime( (const time_t *) &timenow );
    }
    if (!gmt)
    {
      memset((void *)&tmbuf, 0, sizeof(tmbuf));
      gmt = &tmbuf;
    }

    if (strtype == 3) // "yyyy/mm/dd hh:mm:ss" (cvs/iso format, implied utc)
    {
      static char timestring3[4+   4 +1 +2+1 +2+1 +2+1 +2+1+2  ];
      sprintf( timestring3,      "%04d/%02d/%02d %02d:%02d:%02d",
               gmt->tm_year+1900, gmt->tm_mon + 1, gmt->tm_mday,
               gmt->tm_hour,  gmt->tm_min, gmt->tm_sec );
      timestr = (const char *)&timestring3[0];
    }
    else if (strtype == 4) // yymmddhh (bugzilla version date format)
    {
      static char timestring4[4+  2   +2  +2  +2 ];
      sprintf( timestring4,     "%02d%02d%02d%02d",
               gmt->tm_year%100, gmt->tm_mon + 1, gmt->tm_mday,
               gmt->tm_hour );
      timestr = (const char *)&timestring4[0];
    }
    else if (strtype == -1) // old "un-PC" type of length 21 OR 23 chars
    {
      // old: "04/03/98 11:22:33 GMT"
      static char timestringX[8+  2+1 +2+1+2 +1 +2+1+2 +1+2 +1+3  ]; //21 or 23
      sprintf( timestringX,    "%02d/%02d/%02d %02d:%02d:%02d GMT",
               gmt->tm_mon + 1, gmt->tm_mday,
               gmt->tm_year%100, gmt->tm_hour,
               gmt->tm_min, gmt->tm_sec );
      timestr = (const char *)&timestringX[0];
    }
    else // strtype == 1 == new type of fixed length and neutral locale
    {                      // ie `date -u` without year
      // new: "Apr 03 11:22:33 UTC" year = gmt->tm_year%100,
      static char timestring1[4+ 3+1 +2+1 +2+1 +2+1 +2+1+3]; // = 19
      sprintf( timestring1,     "%s %02d %02d:%02d:%02d %s",
              monnames[gmt->tm_mon%12], gmt->tm_mday,
              gmt->tm_hour, gmt->tm_min, gmt->tm_sec,
              ((lasttype == 5)?("ZTZ"):("UTC")) );
      timestr = (const char *)&timestring1[0];
    }
  }
  return timestr;
}

// ---------------------------------------------------------------------

