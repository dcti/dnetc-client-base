/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
const char *netres_cpp(void) {
return "@(#)$Id: netres.cpp,v 1.24.2.2 1999/04/13 19:45:26 jlawson Exp $"; }

//---------------------------------------------------------------------

// define to show what network::resolve() is resolving.
//#define RESDEBUG

// define this to timezone we want to appear to be in (debugging only).
//#define RESDEBUGZONE +12

//---------------------------------------------------------------------

#include "cputypes.h"
#include "network2.h"
#include "clitime.h"    // CliTimeGetMinutesWest()
#include <ctype.h>      // tolower()

//------------------------------------------------------------------------

struct proxylist
{
  unsigned int numproxies;
  const char **proxies;
};

// this structure defines which proxies are 'responsible' for which
// time zone. The timezones overlap, and users in an overlapped
// area will have 2 (or more) proxies at their disposal.  
struct proxyzoitype
{  
  const char *name;  
  int minzone;
  int maxzone;
  int midzone;
};

// actual definitions of the timezones and their names.
static const struct proxyzoitype proxyzoi[] =
{ 
  { "us",    -10, -1 ,  -5 },   
  { "euro",   -2, +6 ,  +2 }, //euro crosses 0 degrees longitude
  { "asia",   +5, +10,  +9 },
  { "aussie", +9, -9 , +12 }, //jp and aussie cross the dateline
  { "jp",    +10, -10, -11 }   
};

// what gets appended to produce a qualified name.
// (NOT char*, since we do a sizeof to get its length).
static const char DNET_PROXY_DOMAINNAME[] = "v27.distributed.net";

//------------------------------------------------------------------------

// computes the difference between the locally detected timezone and
// standard GMT time.  returned difference is in seconds.

static int __calc_tzmins(void)
{
  #ifndef TEST
  return -CliTimeGetMinutesWest();  /* clitime.cpp */
  #else
  static int saved_tz = -12345; 
  time_t timenow;
  struct tm * tmP;
  struct tm loctime, utctime;
  int haveutctime, haveloctime, tzdiff;

  if (saved_tz != -12345)
    return saved_tz;

  tzset();
  timenow = time(NULL);
  tmP = localtime( (const time_t *) &timenow);
  if ((haveloctime = (tmP != NULL))!=0)
    memcpy( &loctime, tmP, sizeof( struct tm ));
  tmP = gmtime( (const time_t *) &timenow);
  if ((haveutctime = (tmP != NULL))!=0)
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
  if      (loctime.tm_yday == utctime.tm_yday)     { ;/* no change */ }
  else if (loctime.tm_yday == utctime.tm_yday + 1) { tzdiff += 1440; }
  else if (loctime.tm_yday == utctime.tm_yday - 1) { tzdiff -= 1440; }
  else if (loctime.tm_yday <  utctime.tm_yday)     { tzdiff += 1440; }
  else                                             { tzdiff -= 1440; }

  if (utctime.tm_isdst > 0)
    tzdiff -= 60;
  if (tzdiff < -(12*60)) 
    tzdiff = -(12*60); 
  else if (tzdiff > +(12*60)) 
    tzdiff = +(12*60);
  if (haveutctime && haveloctime)
    saved_tz = tzdiff;

  return tzdiff;
  #endif
}

//-----------------------------------------------------------------------

// returns a pointer to a list containing all of the round-robins
// that seem to overlap into specified timezone and port combination.

static struct proxylist *__GetApplicableProxyList(int port, int tzdiff)
{
  static char *proxies[sizeof(proxyzoi) / sizeof(proxyzoi[1])];
  static char proxynames[sizeof(proxyzoi) / sizeof(proxyzoi[1])][30];
  static struct proxylist retlist = { 0xFFFF, (const char **)&proxies[0] };
  
  int inrange, tz_min, tz_max;
  unsigned int pos;
  char cport[6];
  
  retlist.numproxies = 0;

  if ( port != 0 )
  {
    cport[0] = 0;
    if ( port != 2064 ) 
      sprintf( cport, "%d", port );
    tzdiff /= 60; 
      
    for (pos = 0; pos < (sizeof(proxyzoi)/sizeof(proxyzoi[1])); pos++ )
    {
      tz_min = proxyzoi[pos].minzone;
      tz_max = proxyzoi[pos].maxzone;
      if ( (tz_min > 0) && (tz_max < 0) ) //straddles dateline
        inrange = ( tzdiff >= tz_min || tzdiff <= tz_max);
      else             
        inrange = ( tzdiff >= tz_min && tzdiff <= tz_max);
      if ( inrange )
      {
        sprintf( proxynames[retlist.numproxies], 
           "%s%s.%s", proxyzoi[pos].name, cport, DNET_PROXY_DOMAINNAME );
        proxies[retlist.numproxies] = proxynames[retlist.numproxies];
        retlist.numproxies++;
      }
    }
    if (retlist.numproxies == 0) /* should never happen! */
    {
      sprintf( proxynames[0], 
           "us%s.%s", cport, DNET_PROXY_DOMAINNAME );
      proxies[0] = proxynames[0];
      retlist.numproxies = 1;
    }
  }
  return (struct proxylist *)&retlist;
}

//-----------------------------------------------------------------------

int Network::AutoFindServer( char *host, int hostlen )
{
  struct proxylist *plist;

  // get the list of applicable proxies in our area.
  #ifdef RESDEBUGZONE
  plist = __GetApplicableProxyList( server_port, RESDEBUGZONE*60 ); 
  #else
  plist = __GetApplicableProxyList( server_port, __calc_tzmins() ); 
  #endif

  // copy the first choice.
  if ( plist->numproxies >= 1 )
    strncpy( host, plist->proxies[0], hostlen );
  else
    host[0] = 0;  

  return 0;
}

//-----------------------------------------------------------------------

