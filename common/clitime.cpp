/* Created by Cyrus Patel <cyp@fb14.uni-mainz.de> 
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This file contains functions for obtaining/formatting/manipulating
 * the time. 'time' is usually stored/passed/returned in timeval format.
 *
 * CliTimer() requires porting so that it returns the time as gettimeofday()
 * would, ie seconds since 1.1.70 GMT in tv_sec, and remaining fraction in
 * microseconds in tv_usec;
 *
 * CliTimer() is assumed to return a valid (possibly adjusted) time_t value
 * in tv_sec by much of the client code. If you see wierd time strings,
 * your implementation is borked. 
 *
 * Please use native OS functions where possible.
 *                                                                 - cyp
 * ----------------------------------------------------------------------
*/ 
const char *clitime_cpp(void) {
return "@(#)$Id: clitime.cpp,v 1.40 1999/08/22 12:44:54 cyp Exp $"; }

#include "cputypes.h"
#include "baseincs.h" // for timeval, time, clock, sprintf, gettimeofday etc
#include "clitime.h"  // keep the prototypes in sync

// ---------------------------------------------------------------------

static int __GetTimeOfDay( struct timeval *tv )
{
  if (tv)
  {
    #if (CLIENT_OS==OS_SCO) || (CLIENT_OS==OS_OS2) || (CLIENT_OS==OS_SOLARIS) || \
       (CLIENT_OS==OS_VMS)
    {     
      struct timeb tb;
      ftime(&tb);
      tv->tv_sec = tb.time;
      tv->tv_usec = tb.millitm*1000;
    }
    #elif (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
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
      nwCliGetTimeOfDay( tv );
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

static int __GetMinutesWest(void) /* see CliTimeGetMinutesWest() for descr */
{
  int minwest;
#if (CLIENT_OS == OS_NETWARE) || ((CLIENT_OS == OS_OS2) && !defined(EMX)) || \
    (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  /* ANSI rules :) */
  minwest = timezone/60;
  if (daylight)
    minwest += 60;
  minwest = -minwest;      /* UTC = localtime + (timezone/60) */
#elif (CLIENT_OS == OS_WIN32)
  TIME_ZONE_INFORMATION TZInfo;
  if (GetTimeZoneInformation(&TZInfo) == 0xFFFFFFFFL)
    return 0;
  minwest = TZInfo.Bias; /* sdk doc is wrong. .Bias is always !dst */
#elif (CLIENT_OS == OS_MACOS)
  MachineLocation thisLocation;
  ReadLocation(&thisLocation);
  minwest = GetGmtDelta(thisLocation);
  minwest = minwest/60;
  minwest = -minwest;
  if (IsDaylightSavingsOn())
    minwest += 60;
#elif (CLIENT_OS==OS_SCO) || (CLIENT_OS==OS_AMIGA) || \
  ((CLIENT_OS == OS_VMS) && !defined(MULTINET))
  #error How does this OS natively deal with timezone? ANSI or Posix rule? (use native calls where possible)
#else
  /* POSIX rules :) */
  struct timezone tz; struct timeval tv;
  if ( gettimeofday(&tv, &tz) )
    return 0;
  minwest = tz.tz_minuteswest;
  if (tz.tz_dsttime)
    minwest += 60;
#endif
  return minwest;
}

// ---------------------------------------------------------------------
  
static int precalced_minuteswest = -1234;
static struct timeval cliclock = {0,0};  //base time for CliClock()
static int adj_time_delta = 0;
static const char *monnames[]={ "Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"};

/*
 * Set the 'time delta', a value added to the tv_sec member by CliTimer()
 * before the time is returned. CliTimerSetDelta() returns the old delta.
 */
int CliTimerSetDelta( int delta )
{
  int old = adj_time_delta;
  adj_time_delta = delta;
  if ( abs( old - delta ) >= 20 )
    precalced_minuteswest = -1234;
  return old;
}

/*
 * timezone offset after compensating for dst (west of utc > 0, east < 0)
 * such that the number returned is constant for any time of year
 */
int CliTimeGetMinutesWest(void)
{
  if (precalced_minuteswest == -1234)
    precalced_minuteswest = __GetMinutesWest();
  return precalced_minuteswest;
}

/*
 * Get the time since first call to CliTimer (pass NULL if storage not reqd)
*/
struct timeval *CliClock( struct timeval *tv )
{
  static struct timeval stv = {0,0};
  if (cliclock.tv_sec == 0)
  {
    CliTimer( NULL ); //set cliclock to current time
    stv.tv_usec = 21; //just something (the meaning of life)
    stv.tv_sec = 0;
  }
  else
  {
    __GetTimeOfDay( &stv );
    if (stv.tv_usec < cliclock.tv_usec )
    {
      stv.tv_usec += 1000000L;
      stv.tv_sec--;
    }
    stv.tv_usec -= cliclock.tv_usec;
    stv.tv_sec -= cliclock.tv_sec;
  }
  if (tv)
  {
    tv->tv_sec = stv.tv_sec;
    tv->tv_usec = stv.tv_usec;
    return tv;
  }
  return (&stv);
}

// ---------------------------------------------------------------------

// Get the current time in timeval format (pass NULL if storage not req'd)
struct timeval *CliTimer( struct timeval *tv )
{
  static struct timeval stv = {0,0};
  struct timeval ttv;
  if (__GetTimeOfDay( &ttv ) == 0)
  {
    stv.tv_sec = ttv.tv_sec;
    stv.tv_usec = ttv.tv_usec;
    if (cliclock.tv_sec == 0) //CliClock() not initialized
    {
      cliclock.tv_sec = ttv.tv_sec;
      cliclock.tv_usec = ttv.tv_usec;
    }
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
    if (result->tv_usec > 1000000L)
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

int CliIsTimeZoneInvalid(void)
{
  #if ((CLIENT_OS==OS_DOS) || (CLIENT_OS==OS_WIN16) || \
       (CLIENT_OS==OS_WIN32S) || (CLIENT_OS==OS_OS2) || \
       (CLIENT_OS==OS_WIN32))
  static int needfixup = -1;       
  if (needfixup == -1)
  {
    needfixup = 0;
    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
    if (winGetVersion() < 400)
    #endif
    if (!getenv("TZ"))
    {
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

