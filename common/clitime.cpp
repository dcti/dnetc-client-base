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
return "@(#)$Id: clitime.cpp,v 1.37.2.24 2000/05/27 11:07:39 trevorh Exp $"; }

#include "cputypes.h"
#include "baseincs.h" // for timeval, time, clock, sprintf, gettimeofday etc
#include "clitime.h"  // keep the prototypes in sync

#if defined(__unix__) && !defined(__EMX__)
  #define HAVE_GETRUSAGE
  #include <sys/resource.h>
  #undef THREADS_HAVE_OWN_ACCOUNTING
  #if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD)
    // if threads have their own pid, then we can use getrusage() to
    // obtain thread-time, otherwise resort to something else (eg gettimeofday)
    #define THREADS_HAVE_OWN_ACCOUNTING
  #endif
#endif

/* --------------------------------------------------------------------- */

int InitializeTimers(void)
{
  if (CliClock(NULL)!=0) /* this is a one-shot */
    return -1;
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
        (CLIENT_OS == OS_VMS)
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
    #elif (CLIENT_OS==OS_WIN16)
    {
      static DWORD lastcheck = 0;
      static time_t basetime = 0;
      DWORD ticks = GetTickCount(); /* millisecs elapsed since OS start */
      if (lastcheck == 0 || (ticks < lastcheck))
      {
        lastcheck = ticks;
        basetime = time(NULL) - (time_t)(ticks/1000);
      }
      tv->tv_usec = (ticks%1000)*1000;
      tv->tv_sec = basetime + (time_t)(ticks/1000);
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
      /* avoid yanking in watcom's crappy static clib for int64 mul/div */
      _asm mov eax, fsec
      _asm xor edx, edx
      _asm mov ecx, 1000000
      _asm mul ecx
      _asm xor ecx, ecx
      _asm dec ecx
      _asm div ecx
      _asm mov fsec, eax
      #else
      fsec = (unsigned long)(((unsigned __int64)
             (((unsigned __int64)fsec) * 1000000ul)) / 0xfffffffful);
      #endif

      tv->tv_sec = (time_t)secs;
      tv->tv_usec = (long)fsec;
    }
    #elif (CLIENT_OS == OS_AMIGAOS)
    {
      return timer((unsigned int *)tv );
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

// timezone offset after compensating for dst (west of utc > 0, east < 0)
// such that the number returned is constant for any time of year
// CliGetMinutesWest() caches the value returned from __GetMinutesWest()

static int __GetMinutesWest(void)
{
  int minwest;
#if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_WIN16) || \
  ((CLIENT_OS == OS_OS2) && !defined(__EMX__))
  /* ANSI rules :) */
  minwest = ((int)timezone)/60;
  if (daylight)
    minwest += 60;
  minwest = -minwest;      /* UTC = localtime + (timezone/60) */
#elif (CLIENT_OS == OS_WIN32)
  TIME_ZONE_INFORMATION TZInfo;
  if (GetTimeZoneInformation(&TZInfo) == 0xFFFFFFFFL)
    return 0;
  minwest = TZInfo.Bias; /* sdk doc is wrong. .Bias is always !dst */
#elif (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_SCO) || \
      (CLIENT_OS == OS_AMIGA) || (CLIENT_OS == OS_VMS)
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
  if      (loctime.tm_yday == utctime.tm_yday) { ;/* no change */ }
  else if (loctime.tm_yday == utctime.tm_yday + 1) { tzdiff += 1440; }
  else if (loctime.tm_yday == utctime.tm_yday - 1) { tzdiff -= 1440; }
  else if (loctime.tm_yday <  utctime.tm_yday) { tzdiff += 1440; }
  else { tzdiff -= 1440; }

  if (utctime.tm_isdst > 0)
    tzdiff -= 60;
  if (tzdiff < -(12*60))
    tzdiff = -(12*60);
  else if (tzdiff > +(12*60))
    tzdiff = +(12*60);

  minwest = -tzdiff;
#else
  /* POSIX rules :) */
  /* FreeBSD does not provide timezone information with gettimeofday */
  struct timezone tz; struct timeval tv;
  if ( gettimeofday(&tv, &tz) )
    return 0;
  minwest = tz.tz_minuteswest;
  if (tz.tz_dsttime)
    minwest += 60;
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
//
// This function is used only for generation of "Summary" stats.

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
    #if defined(CLOCK_REALTIME) /* POSIX 1003.1b-1993 but not 1003.1-1990 */
    {
      struct timespec ts;
      if (clock_gettime(CLOCK_REALTIME, &ts))
        return -1;
      tv->tv_sec = ts.tv_sec;
      tv->tv_usec = ts.tv_nsec / 1000;
    }
	#elif (CLIENT_OS == OS_BEOS)
    {
      bigtime_t now = system_time();
      tv->tv_sec = (time_t)(now / 1000000LL);    /* microseconds -> seconds */
      tv->tv_usec = (time_t)(now % 1000000LL);    /* microseconds < 1 second */
    }
    #elif (CLIENT_OS == OS_NETWARE)
    {
      /* wrapper (for scaling/emu) around GetHighResolutionTimer() */
      if (nwCliGetHardwareClock(tv)!=0) /* microsecs since boot */
        return -1;
    }
    #elif (CLIENT_OS == OS_RISCOS)
    {
      static unsigned long last_ctr = 0, wrap_hi = 0, wrap_lo = 0;
      unsigned long usecs, ctr = read_monotonic_time(); /* hsecs since boot */
      if (ctr < last_ctr)
      {
        wrap_hi += 42949672UL;
        wrap_lo += 960000UL;
      }
      last_ctr = ctr;
      usecs = ((ctr%100UL)*10000UL) + wrap_lo;
      tv->tv_sec = (time_t)((ctr/100UL) + wrap_hi + (usecs / 1000000UL);
      tv->tv_usec = usecs % 1000000UL;
    }
    #elif (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS == OS_OS2)
    {
      #if (CLIENT_OS == OS_OS2)
        #define myULONG ULONG
      #else
        #define myULONG DWORD
      #endif
      static int sguard = -1;
      static myULONG lastcheck = 0, wrap_count = 0;
      unsigned long usecs;
      myULONG ticks, l_wrap_count;

      while ((++sguard)!=0)
        sguard--;
      l_wrap_count = wrap_count;
      #if (CLIENT_OS == OS_OS2)
      if (DosQuerySysInfo(QSV_MS_COUNT, QSV_MS_COUNT, &ticks, sizeof(ticks)))
        {sguard--; return -1; }
      #else
      ticks = GetTickCount(); /* millisecs elapsed since OS start */
      #endif
      if (ticks < lastcheck)
        wrap_count = ++l_wrap_count;
      lastcheck = ticks;
      sguard--;

      usecs = ((ticks%1000UL)*1000UL) + (l_wrap_count * 296000UL);
      tv->tv_usec = usecs % 1000000ul;
      tv->tv_sec = (time_t)((ticks/1000UL) + (l_wrap_count * 4294967UL)) +
                               (usecs / 1000000ul);
    }
    #elif (CLIENT_OS == OS_DOS)
    {
      /* in platforms/dos/dostime.cpp */
      /* resolution (granularity) is 54925us */
      if (getmicrotime(tv)!=0)
        return -1;
    }
    #elif (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_SOLARIS)
    {
      hrtime_t hirestime = gethrtime();
      hirestime /= 1000; /* nanosecs to microsecs */
      tv.tv_sec = (time_t)(hirestime / 1000000);
      tv.tv_usec = (unsigned long)(hirestime % 1000000);
    }
    #elif (CLIENT_OS == OS_LINUX) /*only RTlinux has clock_gettime/gethrtime*/
    {
      /* we have no choice but to use getrusage() [we can */
      /* do that because each thread has its own pid] */
      struct rusage rus;
      if (getrusage(RUSAGE_SELF,&rus) != 0)
        return -1;
      tv->tv_sec  = rus.ru_utime.tv_sec  + rus.ru_stime.tv_sec;
      tv->tv_usec = rus.ru_utime.tv_usec + rus.ru_stime.tv_usec;
      if (tv->tv_usec >= 1000000)
      {
        tv->tv_usec -= 1000000;
        tv->tv_sec++;
      }
    }
    #elif 0
    {
      /* DO NOT USE THIS WITHOUT ENSURING ...
         a) that clock() does *not* return virtual time.
            (under unix clock() is often implemented via
            times() is thus virtual. posix 1b compatible OSs
            should have clock_gettime())
         b) that clock() is not dependant on system time
            (all watcom clibs have this bug)
         c) that the value from clock() does indeed count up to
            Uxxx_MAX (whatever size clock_t is) before wrapping.
            At least one implementation (EMX 0.9) is known to
            wrap at (0xfffffffful/10).
      */
      static int sguard = -1;
      static clock_t lastcheck = 0;
      static unsigned long wrap_count = 0;
      clock_t cps, counter, wrap_hi, wrap_lo;
      unsigned long l_wrap_count;

      while ((++sguard)!=0)
        --sguard;
      l_wrap_count = wrap_count;
      counter = clock();
      if (counter == ((clock_t)-1))
        {--sguard; return -1;}
      if (counter < lastcheck)
        wrap_count = ++l_wrap_count;
      lastcheck = counter;
      sguard--;

      cps = CLOCKS_PER_SEC;
      tv->tv_usec = ((counter%cps)*(1000000ul/cps));
      tv->tv_sec = (time_t)(basetime + (counter / cps);

      if (cps > 1000000)
      {
        #if defined(HAVE_I64)
        ui64 usecs = (ui64)(counter%cps))
        usecs *= 1000000ul;
        usecs /= cps;
        tv->tv_usec = (unsigned long)usecs;
        #else
        tv->tv_usec = (unsigned long)((((double)(counter%cps))
                                     * 1000000.0)/((double)cps));
        #endif
      }
      if (l_wrap_count)
      {
        double x = (((double)l_wrap_count)*(((double)((clock_t)-1))+1.0) /
                    ((double)cps));
        unsigned long secs = (unsigned long)x;
        unsigned long usecs = (unsigned long)(1000000.0 * (x-((double)secs)));
        tv->tv_secs += secs;
        tv->tv_usecs += usecs;
      }
      if (tv->tv_usec > 1000000)
      {
        tv->tv_sec += tv->tv_usec / 1000000ul;
        tv->tv_usec %= 1000000ul;
      }
    }
    #else
    // this is a bad thing because time-of-day is user modifyable.
    //if (__GetTimeOfDay( tv ))
      return -1;
    #endif
  }
  return 0;
}

/* --------------------------------------------------------------------- */

/* Get thread (user) cpu time, used for fine-slice benchmark etc. */

int CliGetThreadUserTime( struct timeval *tv )
{
  if (tv)
  {
    #if (CLIENT_OS == OS_BEOS)
    {
      thread_info tInfo;
      get_thread_info(find_thread(NULL), &tInfo);
      tv->tv_sec = tInfo.user_time / 1000000; // convert from microseconds
      tv->tv_usec = tInfo.user_time % 1000000;
      return 0;
    }
    #elif defined(HAVE_GETRUSAGE) && defined(THREADS_HAVE_OWN_ACCOUNTING)
    struct rusage rus;
    if (getrusage(RUSAGE_SELF,&rus) == 0)
    {
      tv->tv_sec = rus.ru_utime.tv_sec;
      tv->tv_usec = rus.ru_utime.tv_usec;
      return 0;
    }
    #elif (CLIENT_OS == OS_WIN32)
    {
      static int isnt = -1;
      #if 0
      if (isnt == -1)
      {
        if ( winGetVersion() < 2000)
          isnt = 0;
      }
      #endif
      if ( isnt != 0 )
      {
        FILETIME ct,et,kt,ut;
        if (GetThreadTimes(GetCurrentThread(),&ct,&et,&kt,&ut))
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
          isnt = 1;
          return 0;
        }
        if (isnt < 0) /* first try? */
          isnt = 0; /* don't try again */
      }
    }
    #endif
  }
  return -1;
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

