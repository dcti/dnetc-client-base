// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// ----------------------------------------------------------------------
// This file contains functions for obtaining/formatting/manipulating
// the time. 'time' is always stored/passed/returned in timeval format.
//
// CliTimer() requires porting so that it returns the time as gettimeofday()
// would, ie seconds since 1.1.70 GMT in tv_sec, and remaining fraction in
// microseconds in tv_usec;
//
// CliTimer() is assumed to return a valid (possibly adjusted) time_t value
// in tv_sec by much of the client code. If you see wierd time strings,
// your implementation is borked. tv_usec is assumed to provide the highest
// resolution time your OS supports (scaled of course).
// ----------------------------------------------------------------------
//
// $Log: clitime.cpp,v $
// Revision 1.29  1999/03/10 11:54:48  cyp
// Fixed win32 returning minutes east.
//
// Revision 1.28  1999/03/04 02:22:04  silby
// Fixed typo.
//
// Revision 1.27  1999/03/04 00:55:17  cyp
// adj_time_delta was being added to the wrong variable.
//
// Revision 1.26  1999/03/04 00:17:44  trevorh
// Correct c0000005 error in CliTimer() after last update
//
// Revision 1.25  1999/03/03 04:29:36  cyp
// created CliTimeGetMinutesWest() and CliTimerSetDelta(). See .h for descrip.
//
// Revision 1.24  1999/01/29 18:58:04  jlawson
// fixed formatting.
//
// Revision 1.23  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.22  1998/12/08 05:37:27  dicamillo
// MacOS update: use standard time routines instead of Microseconds.
//
// Revision 1.21  1998/11/19 10:49:04  cyp
// Timestrings are now "UTC" instead of "GMT"
//
// Revision 1.20  1998/10/29 23:16:16  sampo
// MacOS doesn't use tzset()
//
// Revision 1.19  1998/08/10 20:04:55  cyruspatel
// NetWare specific change: moved gettime code to a function in netware.cpp
//
// Revision 1.18  1998/07/13 03:29:57  cyruspatel
// Added 'const's or 'register's where the compiler was complaining about
// ambiguities. ("declaration/type or an expression")
//
// Revision 1.17  1998/07/07 21:55:29  cyruspatel
// client.h has been split into client.h and baseincs.h
//
// Revision 1.16  1998/07/06 09:21:22  jlawson
// added lint tags around cvs id's to suppress unused variable warnings.
//
// Revision 1.15  1998/06/29 08:44:06  jlawson
// More OS_WIN32S/OS_WIN16 differences and long constants added.
//
// Revision 1.14  1998/06/29 06:57:55  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.13  1998/06/15 12:03:55  kbracey
// Lots of consts.
//
// Revision 1.12  1998/06/14 08:26:44  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.11  1998/06/14 08:12:45  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 1.00  1998/05/01 05:01:08  cyruspatel
// Created
//

#if (!defined(lint) && defined(__showids__))
const char *clitime_cpp(void) {
return "@(#)$Id: clitime.cpp,v 1.29 1999/03/10 11:54:48 cyp Exp $"; }
#endif

#include "cputypes.h"
#include "baseincs.h" // for timeval, time, clock, sprintf, gettimeofday etc
#include "sleepdef.h" // usleep for amiga and where gettimeofday is supported
#include "clitime.h"  // keep the prototypes in sync

// ---------------------------------------------------------------------

static int precalced_minuteswest = -1234;
static struct timeval cliclock = {0,0};  //base time for CliClock()
static int adj_time_delta = 0;

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

#if (CLIENT_OS == OS_NETWARE) || ((CLIENT_OS == OS_OS2) && !defined(EMX))
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
#elif (CLIENT_OS==OS_SCO) || (CLIENT_OS==OS_AMIGA) || \
  (CLIENT_OS==OS_MACOS) || ((CLIENT_OS == OS_VMS) && !defined(MULTINET)) || \
  (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  time_t timenow; int haveutctime, haveloctime;
  struct tm * tmP; struct tm loctime, utctime;
  #if (CLIENT_OS != OS_MACOS)
  tzset();       /* this should be nop #defined if macos can't support it */
  #endif
  timenow = CliTimer(NULL)->tv_sec;
  tmP = localtime( (const time_t *) &timenow);
  if ((haveloctime = (tmP != NULL))!=0)
    memcpy( &loctime, tmP, sizeof( struct tm ));
  tmP = gmtime( (const time_t *) &timenow);
  if ((haveutctime = (tmP != NULL))!=0)
    memcpy( &utctime, tmP, sizeof( struct tm ));
  if (!haveutctime || !haveloctime)
    return 0;
  minwest =  ((loctime.tm_min  - utctime.tm_min) )
            +((loctime.tm_hour - utctime.tm_hour)*60 );
  /* last two are when the time is on a year boundary */
  if      (loctime.tm_yday == utctime.tm_yday)     { ;/* no change */ }
  else if (loctime.tm_yday == utctime.tm_yday + 1) { minwest += 1440; }
  else if (loctime.tm_yday == utctime.tm_yday - 1) { minwest -= 1440; }
  else if (loctime.tm_yday <  utctime.tm_yday)     { minwest += 1440; }
  else                                             { minwest -= 1440; }
  if (utctime.tm_isdst>0)
    minwest-=60;
  if (minwest < -(12*60))
    minwest = -(12*60);
  else if (minwest > +(12*60))
    minwest = +(12*60);
  minwest = -minwest;
  precalced_minuteswest = minwest;
#else
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
  int dofallback = timer((unsigned int *)&stv );
  #define REQUIRES_TIMER_FALLBACK
#else
  struct timezone tz;
  int dofallback =( gettimeofday(&stv, &tz) );
  #define REQUIRES_TIMER_FALLBACK
#endif
#ifdef REQUIRES_TIMER_FALLBACK
#undef REQUIRES_TIMER_FALLBACK
  if (dofallback)
  {
    static unsigned int timebase = 0;
    unsigned int secs, rate, xclock = (unsigned int)(clock());

    if (!xclock)
    {
      if (!stv.tv_sec && !stv.tv_usec)
      {
        stv.tv_sec = ((unsigned int)(time(NULL)));
        stv.tv_usec = 0;
      }
      else if ((long) stv.tv_sec == (long) (secs = ((unsigned int)(time(NULL)))))
      {
        usleep(100000L); //1/10 sec
        stv.tv_usec += 100000L;
      }
      else
      {
        stv.tv_sec = secs;
        stv.tv_usec = 0;
      }
    }
    else
    {
      rate = CLOCKS_PER_SEC;
      secs = (xclock/rate);
      if (!timebase) timebase = ((unsigned int)(time(NULL))) - secs;
      stv.tv_sec = (time_t)(timebase + secs);
      xclock -= (secs * rate);
      if ( rate <= 1000000L )
        stv.tv_usec = xclock * (1000000L/rate);
      else
        stv.tv_usec = xclock / (rate/1000000L);
    }

    if (stv.tv_usec > 1000000L)
    {
      stv.tv_sec += stv.tv_usec/1000000L;
      stv.tv_usec %= 1000000L;
    }
  }
#endif
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
int CliTimerAdd( struct timeval *dest, struct timeval *tv1, struct timeval *tv2 )
{
  if (dest)
  {
    if (!tv1 || !tv2)
    {
      CliTimer( NULL );
      if (!tv1) tv1 = dest;
      if (!tv2) tv2 = dest;
    }
    dest->tv_sec = tv1->tv_sec + tv2->tv_sec;
    dest->tv_usec = tv1->tv_usec + tv2->tv_usec;
    if (dest->tv_usec > 1000000L)
    {
      dest->tv_sec += dest->tv_usec / 1000000L;
      dest->tv_usec %= 1000000L;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------

// Store non-negative diff of tv1 and tv2 in 'result'. Uses current time if a 'tv' is NULL
// tv1/tv2 are not modified (unless 'result' is the same as one of them).
int CliTimerDiff( struct timeval *dest, struct timeval *tv1, struct timeval *tv2 )
{
  struct timeval tvdiff, tvtemp;
  struct timeval *tv0;

  if (dest)
  {
    if (!tv1 && !tv2)
      dest->tv_sec = dest->tv_usec = 0;
    else
    {
      if (!tv1 || !tv2)
      {
        CliTimer( &tvtemp );
        if (!tv1) tv1 = &tvtemp;
        else tv2 = &tvtemp;
      }
      if ((((unsigned int)(tv2->tv_sec)) < ((unsigned int)(tv1->tv_sec))) ||
         ((tv2->tv_sec == tv1->tv_sec) &&
           ((unsigned int)(tv2->tv_usec)) < ((unsigned int)(tv1->tv_usec))))
      {
        tv0 = tv1; tv1 = tv2; tv2 = tv0;
      }
      tvdiff.tv_sec = tv2->tv_sec;
      tvdiff.tv_usec = tv2->tv_usec;
      if (((unsigned int)(tvdiff.tv_usec)) < ((unsigned int)(tv1->tv_usec)))
      {
        tvdiff.tv_usec += 1000000L;
        tvdiff.tv_sec--;
      }
      dest->tv_sec  = tvdiff.tv_sec - tv1->tv_sec;
      dest->tv_usec = tvdiff.tv_usec - tv1->tv_usec;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------

// Get time as string. Curr time if tv is NULL. Separate buffers for each
// type: 0=blank type 1, 1="MMM dd hh:mm:ss GMT", 2="hhhh:mm:ss.pp"
const char *CliGetTimeString( struct timeval *tv, int strtype )
{
  static time_t timelast = (time_t)NULL;
  static int lasttype;
  static char timestring[30], spacestring[30], hourstring[30];

  if (!timelast)
  {
    timestring[0]=spacestring[0]=hourstring[0]=0;
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
  else if (strtype == 1 || strtype == -1) //new fmt = 1, old fmt = -1
  {
    #if ((CLIENT_OS != OS_RISCOS) && (CLIENT_OS != OS_MACOS))
    tzset();
    #endif
    if (!tv) tv = CliTimer(NULL);/* show where CliTimer() is returning gunk */
    time_t timenow = tv->tv_sec;

    if (timenow && (timenow != timelast) && (lasttype != strtype))
    {
      struct tm *gmt;
      int utc = 1;
      #if 0 /*((CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16) || \
              (CLIENT_OS == OS_WIN32S) || (CLIENT_OS == OS_OS2)) */
      utc = ( getenv("TZ") != NULL );
      #endif
      if ( utc) utc = (( gmt = gmtime( (const time_t *) &timenow) ) != NULL);
      if (!utc) gmt = localtime( (const time_t *) &timenow);

      if (gmt)
      {
        timelast = timenow;

        if (strtype == -1) // old "un-PC" type of length 21 OR 23 chars
        {
          // old: "04/03/98 11:22:33 GMT"
          //                      2 1  2 1 2  1  2 1 2  1 2  1 3/5 = 21 or 23
          sprintf( timestring, "%02d/%02d/%02d %02d:%02d:%02d %s",
               gmt->tm_mon + 1, gmt->tm_mday,
               gmt->tm_year%100, gmt->tm_hour,
               gmt->tm_min, gmt->tm_sec, ((utc)?("GMT"):("local")) );
        }
        else // strtype == 1 == new type of fixed length and neutral locale
        {
          static const char *monnames[]={ "Jan","Feb","Mar","Apr","May","Jun",
                                          "Jul","Aug","Sep","Oct","Nov","Dec"};

          // new: "Apr 03 11:22:33 GMT" year = gmt->tm_year%100,
          //                    3 1  2 1  2 1  2 1  2 1 3   = 19
          sprintf( timestring, "%s %02d %02d:%02d:%02d %s",
             monnames[gmt->tm_mon%12], gmt->tm_mday,
             gmt->tm_hour, gmt->tm_min, gmt->tm_sec, ((utc)?("UTC"):("---")) );
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

