/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This file contains functions for obtaining/formatting/manipulating
 * the time. 'time' is always stored/passed/returned in timeval format.
 *
 * CliTimer() requires porting so that it returns the time as gettimeofday()
 * would, ie seconds since 1.1.70 GMT in tv_sec, and remaining fraction in
 * microseconds in tv_usec;
 *
 * CliTimer() is assumed to return a valid (possibly adjusted) time_t value
 * in tv_sec by much of the client code. If you see wierd time strings,
 * your implementation is borked. tv_usec is assumed to provide the highest
 * resolution time your OS supports (scaled of course).
 *
 * Please use native OS functions where possible.
 *                                                                 - cyp
 * ----------------------------------------------------------------------
*/ 
const char *clitime_cpp(void) {
return "@(#)$Id: clitime.cpp,v 1.36 1999/04/09 12:15:50 cyp Exp $"; }

#include "cputypes.h"
#include "baseincs.h" // for timeval, time, clock, sprintf, gettimeofday etc
#include "cliident.h" // CliGetNewestModuleTime()
#include "clitime.h"  // keep the prototypes in sync

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
  int minwest;

  if (precalced_minuteswest != -1234)
    return precalced_minuteswest;

#if (CLIENT_OS == OS_NETWARE) || ((CLIENT_OS == OS_OS2) && !defined(EMX)) || \
    (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  /* ANSI rules :) */
  minwest = timezone/60;
  if (daylight)
    minwest += 60;
  minwest = -minwest;      /* UTC = localtime + (timezone/60) */
  precalced_minuteswest = minwest;
#elif (CLIENT_OS == OS_WIN32)
  TIME_ZONE_INFORMATION TZInfo;
  if (GetTimeZoneInformation(&TZInfo) == 0xFFFFFFFFL)
    return 0;
  minwest = TZInfo.Bias; /* sdk doc is wrong. .Bias is always !dst */
  precalced_minuteswest = minwest;
#elif (CLIENT_OS == OS_MACOS)
  MachineLocation thisLocation;
  ReadLocation(&thisLocation);
  minwest = GetGmtDelta(thisLocation);
  minwest = minwest/60;
  minwest = -minwest;
  if (IsDaylightSavingsOn())
    minwest += 60;
  precalced_minuteswest = minwest;
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
  precalced_minuteswest = minwest;
#endif
  return minwest;
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
    CliTimer( &stv );
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
  int dofallback = 0;

#if (CLIENT_OS == OS_SCO) || (CLIENT_OS == OS_OS2) || \
    ((CLIENT_OS == OS_VMS) && !defined(MULTINET))
  struct timeb tb;
  ftime(&tb);
  stv.tv_sec = tb.time;
  stv.tv_usec = tb.millitm*1000;
#elif (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
  static DWORD lastcheck = 0;
  static time_t basetime = 0;
  DWORD ticks = GetTickCount(); /* millisecs elapsed since OS start */
  if (lastcheck == 0 || (ticks < lastcheck))
  {
    lastcheck = ticks;
    basetime = time(NULL) - (time_t)(ticks/1000);
  }
  stv.tv_usec = (ticks%1000)*1000;
  stv.tv_sec = basetime + (time_t)(ticks/1000);
#elif (CLIENT_OS == OS_NETWARE)
  nwCliGetTimeOfDay( &stv );
#elif (CLIENT_OS == OS_AMIGAOS)
  dofallback = timer((unsigned int *)&stv );
#else
  struct timezone tz;
  dofallback = gettimeofday(&stv, &tz);
#endif
  if (dofallback)
  {
    static unsigned long lastcheck = 0;
    static time_t timebase = (time_t)0;
    unsigned long now = (unsigned long)clock();
    unsigned long rate = CLOCKS_PER_SEC;
    unsigned long secs = (now / rate);

    if ((lastcheck == 0) || (lastcheck < now))
    {
      timebase = ((unsigned long)(time(NULL))) - secs;
      lastcheck = now;
    }
    stv.tv_sec = (time_t)(timebase + secs);
    now -= (secs * rate);
    if ( now == 0 )  
      stv.tv_usec = 0;
    else if ( rate <= 1000000UL )
      stv.tv_usec = now * (1000000UL/rate);
    else if (rate <= 1000000000UL )
      stv.tv_usec = ((now * 1000) / (rate/1000));
    else
      stv.tv_usec = now / (rate/1000000UL);
    if (stv.tv_usec > 1000000L)
    {
      stv.tv_sec += stv.tv_usec/1000000L;
      stv.tv_usec %= 1000000L;
    }
  }
  stv.tv_sec += adj_time_delta;
  if (cliclock.tv_sec == 0) //CliClock() not initialized
  {
    cliclock.tv_sec = stv.tv_sec;
    cliclock.tv_usec = stv.tv_usec;
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

/*
 *  Get Date/Time this module was built. Used, for instance, to 'ensure' 
 *  that time from the .ini or recvd from a proxy is sane. (could also use
 *  CVS ID)
 */
time_t CliTimeGetBuildDate(void)
{
  static time_t bdtime = 0;
  if (bdtime == 0)
    bdtime = CliGetNewestModuleTime(); /* cliident.cpp */
  if (bdtime == 0)
  {
    struct tm bd;
    bd.tm_mon = 99;

    bdtime = (time_t)0x362a72f0L; /* my 32nd birthday :) */

    #ifdef __DATE__
    {
      int i;
      const char *p = __DATE__;
      memset((void *)&bd,0,sizeof(bd));
      bd.tm_mon = 99;
      if (p[3]==' ')
      {
        for (i=0;i<12;i++)
        {
          if (memcmp(monnames[i],p,3)==0)
          {
            bd.tm_mon = i;
            bd.tm_mday = atoi(p+4);
            bd.tm_year = (atoi(p+7) - 1900);
            #ifdef __TIME__
            p = __TIME__;
            bd.tm_hour = atoi(p);
            bd.tm_min = atoi(p+3);
            bd.tm_sec = atoi(p+6);
            if (bd.tm_hour<0 || bd.tm_hour>23 || bd.tm_min<0 || bd.tm_min>59)
              bd.tm_hour = bd.tm_min = bd.tm_sec = 0;
            else if (bd.tm_sec < 0 || bd.tm_sec > 61)
              bd.tm_sec = 0;
            #endif
            break;
          }
        }
      }
    }
    #endif
    if (bd.tm_mon >= 0 && bd.tm_mon <= 11 && 
        bd.tm_mday >= 1 && bd.tm_mday <= 31 &&
        bd.tm_year >= 99 && bd.tm_year <= 132 )
    {
      bdtime = mktime( &bd );
    }
  }
  return bdtime;
}
   

// ---------------------------------------------------------------------

// Get time as string. Curr time if tv is NULL. Separate buffers for each
// type: 0=blank type 1, 1="MMM dd hh:mm:ss UTC", 2="hhhh:mm:ss.pp"
//       3="yyyy/mm/dd hh:mm:ss" (iso/cvs format, implied utc)
const char *CliGetTimeString( const struct timeval *tv, int strtype )
{
  static time_t timelast = (time_t)NULL;
  static int lasttype;
  static char timestring[30], spacestring[30], hourstring[30];

  if (!timelast)
  {
    timestring[0] = spacestring[0] = hourstring[0] = 0;
    timelast = 1;
    lasttype = 0;
  }

  if (strtype == 0)
  {
    if (!spacestring[0])
    {
      CliGetTimeString( NULL, 1 );
      register char *ts = timestring, *ss = spacestring;
      while (*ts++) *ss++=' '; *ss=0;
    }
    return spacestring;
  }
  else if (strtype == 1 || strtype == -1 || strtype == 3) //new fmt = 1, old fmt = -1
  {
    if (!tv) tv = CliTimer(NULL);/* show where CliTimer() is returning gunk */
    time_t timenow = tv->tv_sec;

    if (timenow && (timenow != timelast) && (lasttype != strtype))
    {
      struct tm *gmt;
      if (( gmt = gmtime( (const time_t *) &timenow) ) == NULL)
      {
        time_t lt = timenow - (time_t)(CliTimeGetMinutesWest() * 60);
        gmt = localtime( (const time_t *) &lt);
      }

      if (gmt)
      {
        timelast = timenow;

	if (strtype == 3) // "yyyy/mm/dd hh:mm:ss" (cvs/iso format, implied utc)
	{
          sprintf( timestring, "%04d/%02d/%02d %02d:%02d:%02d",
	       gmt->tm_year+1900, gmt->tm_mon + 1, gmt->tm_mday,
               gmt->tm_hour,  gmt->tm_min, gmt->tm_sec );
	  timelast = 0;
	}
        else if (strtype == -1) // old "un-PC" type of length 21 OR 23 chars
        {
          // old: "04/03/98 11:22:33 GMT"
          //                      2 1  2 1 2  1  2 1 2  1 2  1 3/5 = 21 or 23
          sprintf( timestring, "%02d/%02d/%02d %02d:%02d:%02d GMT",
               gmt->tm_mon + 1, gmt->tm_mday,
               gmt->tm_year%100, gmt->tm_hour,
               gmt->tm_min, gmt->tm_sec );
        }
        else // strtype == 1 == new type of fixed length and neutral locale
        {                      // ie clock(8) without year
          // new: "Apr 03 11:22:33 UTC" year = gmt->tm_year%100,
          //                    3 1  2 1  2 1  2 1  2 1 3   = 19
          sprintf( timestring, "%s %02d %02d:%02d:%02d UTC",
             monnames[gmt->tm_mon%12], gmt->tm_mday,
             gmt->tm_hour, gmt->tm_min, gmt->tm_sec );
        }
      }
    }
    return timestring;
  }
  else if (strtype == 2)
  {
    if (!tv) tv = CliTimer( NULL );
    sprintf( hourstring, "%u.%02u:%02u:%02u.%02u", (unsigned) (tv->tv_sec / 86400L),
      (unsigned) ((tv->tv_sec % 86400L) / 3600L), (unsigned) ((tv->tv_sec % 3600L)/60),
      (unsigned) (tv->tv_sec % 60), (unsigned) ((tv->tv_usec/10000L)%100) );
    //if ((tv->tv_sec / 86400L)==0 ) //don't show days if not needed
    //  return &hourstring[2]; // skip the "0."
    return hourstring;
  }
  return "";
}

// ---------------------------------------------------------------------

