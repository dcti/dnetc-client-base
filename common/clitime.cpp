// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
/*
   This file contains functions for obtaining/formatting/manipulating
   the time. 'time' is always stored/passed/returned in timeval format.

   CliTimer() requires porting so that it returns the time as gettimeofday() 
   would, ie seconds since 1.1.70 GMT in tv_sec, and remaining fraction in 
   microseconds in tv_usec;
*/
// $Log: clitime.cpp,v $
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
return "@(#)$Id: clitime.cpp,v 1.20 1998/10/29 23:16:16 sampo Exp $"; }
#endif

#include "cputypes.h"
#include "baseincs.h" // for timeval, time, clock, sprintf, gettimeofday etc
#include "sleepdef.h" // usleep for amiga and where gettimeofday is supported
#include "clitime.h"  // just to keep the prototypes in sync

// ---------------------------------------------------------------------

static struct timeval cliclock = {0,0};  //base time for CliClock()

// Get the time since first call to CliTimer (pass NULL if storage not reqd)
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

#if (CLIENT_OS == OS_MACOS)
  unsigned long long t;
  Microseconds( (UnsignedWide *)&t );
  stv.tv_sec = t / 1000000U;
  stv.tv_usec = t % 1000000U;
#elif (CLIENT_OS == OS_SCO) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_DOS) || ((CLIENT_OS == OS_VMS) && !defined(MULTINET))
  struct timeb tb;
  ftime(&tb);
  stv.tv_sec = tb.time;
  stv.tv_usec = tb.millitm*1000;
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
    time_t timenow = ((tv)?(tv->tv_sec):(time(NULL)));

    if (timenow && (timenow != timelast) && (lasttype != strtype))
    {
      struct tm *gmt;
      int utc = (( gmt = gmtime( (const time_t *) &timenow) ) != NULL);
      if (!utc) gmt = localtime( (const time_t *) &timenow);

      if (gmt)
      {
        timelast = timenow;

        if (strtype == -1) // old "unfriendly" type of length 21 OR 23 chars
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
             gmt->tm_hour, gmt->tm_min, gmt->tm_sec, ((utc)?("GMT"):("---")) );
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
    //  return hourstring+sizeof("0.");
    return hourstring;
  }
  return "";
}

// ---------------------------------------------------------------------

