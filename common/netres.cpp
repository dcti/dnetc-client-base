// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: netres.cpp,v $
// Revision 1.7  1998/10/26 02:55:08  cyp
// win16 changes
//
// Revision 1.6  1998/10/04 11:35:46  remi
// Id tags fun.
//
// Revision 1.5  1998/08/15 21:34:00  jlawson
// corrected loss of precision warning
//
// Revision 1.4  1998/08/10 21:53:56  cyruspatel
// Two major changes to work around a lack of a method to detect if the network
// availability state had changed (or existed to begin with) and also protect
// against any re-definition of client.offlinemode. (a) The NO!NETWORK define is
// now obsolete. Whether a platform has networking capabilities or not is now
// a purely network.cpp thing. (b) NetworkInitialize()/NetworkDeinitialize()
// are no longer one-shot-and-be-done-with-it affairs. ** Documentation ** is
// in netinit.cpp.
//
// Revision 1.3  1998/08/03 19:37:51  jlawson
// changed order of "static" to eliminate gcc warning
//
// Revision 1.2  1998/07/26 13:20:51  cyruspatel
// Fixed a signed vs unsigned comparison.
//
// Revision 1.1  1998/07/26 12:34:46  cyruspatel
// Created.
//

#if (!defined(lint) && defined(__showids__))
const char *netres_cpp(void) {
return "@(#)$Id: netres.cpp,v 1.7 1998/10/26 02:55:08 cyp Exp $"; }
#endif

//---------------------------------------------------------------------
//#define TEST  //standalone test

//#define RESDEBUG //to show what network::resolve() is resolving
#ifdef RESDEBUG
  //#define RESDEBUGZONE +12  //the timezone we want to appear to be in 
#endif

#if defined(TEST)
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#else
#include "cputypes.h"
#include "network.h"
#include <ctype.h> //tolower()
#endif

//------------------------------------------------------------------------

#ifdef STUBIFY_ME    //ooookay. do what it asks
#define OLDRESOLVE   //and don't drag in the statics
#endif

//------------------------------------------------------------------------

#ifndef OLDRESOLVE
static const struct        // this structure defines which proxies are 
{                          // 'responsible' for which time zone. The 
  const char *name;        // timezones overlap, and users in an overlapped
  int minzone;             // area will have 2 (or more) proxies at their
  int maxzone;             // disposal.  
  int midzone;
} proxyzoi[] = { 
                { "us",    -10, -1 ,  -5 },   
                { "euro",   -2, +6 ,  +2 }, //euro crosses 0 degrees longitude
                { "asia",   +5, +10,  +9 },
                { "aussie", +9, -9 , +12 }, //jp and aussie cross the dateline
                { "jp",    +10, -8 , -11 }   
               };
static const char DNET_PROXY_DOMAINNAME[]="v27.distributed.net"; // NOT char* 
#endif

//-------------------------------------------------------------------------

#ifndef OLDRESOLVE
static int IsHostnameDNetKeyserver( const char *hostname, int *tzdiff ) 
{
  char *p; 
  char buffer[ sizeof( DNET_PROXY_DOMAINNAME )+15 ];
  unsigned int pos, i = strlen( hostname );

  if ( i < sizeof( buffer ) )
    {
    strcpy( buffer, hostname );
    for ( pos = 0; pos < i; pos++ )
      buffer[pos] = (char)tolower( hostname[pos] );
    if ( strcmp( buffer, "rc5proxy.distributed.net" )==0 ) //old name
      {
      if ( tzdiff ) *tzdiff = 0;
      return 1;
      }
    p = strchr( buffer, '.' );
    if (p != NULL && strcmp( p+1, DNET_PROXY_DOMAINNAME ) == 0)
      {
      for (pos=0;pos<(sizeof(proxyzoi)/sizeof(proxyzoi[0]));pos++)
        {
        for (i=0;;i++)
          {
          if ((proxyzoi[pos].name[i] == 0) && ((buffer[i]=='.') || 
                 atoi(buffer+i)==80 || atoi(buffer+i)==23))
            {
            if (tzdiff) *tzdiff = proxyzoi[pos].midzone * 60;
            return 1;
            }
          if ( buffer[i] != proxyzoi[pos].name[i])
            break;
          }
        }
      }      
    }
  return 0;
}  
#endif

//------------------------------------------------------------------------

#ifndef OLDRESOLVE
static int calc_tzmins(void)
{
  static int saved_tz = -123; 
  time_t timenow;
  struct tm * tmP;
  struct tm loctime, utctime;
  int haveutctime, haveloctime, tzdiff;

  if (saved_tz != -123)
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

  if (utctime.tm_isdst>0)  
    tzdiff-=60;
  if (tzdiff < -(12*60)) 
    tzdiff = -(12*60); 
  else if (tzdiff > +(12*60)) 
    tzdiff = +(12*60);
  if (haveutctime && haveloctime)
    saved_tz = tzdiff;

  return tzdiff;
}
#endif

//-----------------------------------------------------------------------

struct proxylist
{
  unsigned int numproxies;
  const char **proxies;
};

//-----------------------------------------------------------------------

#ifndef OLDRESOLVE
struct proxylist *GetApplicableProxyList(int port, int tzdiff) /*host order*/
{
  static char *proxies[sizeof(proxyzoi)/sizeof(proxyzoi[1])];
  static char proxynames[sizeof(proxyzoi)/sizeof(proxyzoi[1])][30];
  static struct proxylist retlist = { 0xFFFF, (const char **)&proxies[0] };
  
  int inrange, tz_min, tz_max;
  unsigned int pos;
  char cport[6];
  
  retlist.numproxies = 0;
  
  cport[0] = 0;
  if ( port == 23 || port == 80 ) 
    sprintf( cport, "%d", port );
  tzdiff /= 60; 
      
  for (pos = 0; pos < (sizeof(proxyzoi)/sizeof(proxyzoi[1])); pos++ )
    {
    tz_min = proxyzoi[pos].minzone;
    tz_max = proxyzoi[pos].maxzone;
    if ( (tz_min > 0) && (tz_max < 0) ) //straddles dateline
      inrange = (( tzdiff >= tz_min || tzdiff <= tz_max) ? 1 : 0);
    else             
      inrange = (( tzdiff >= tz_min && tzdiff <= tz_max) ? 1 : 0);
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
  return (struct proxylist *)&retlist;
}
#endif

//-----------------------------------------------------------------------

#ifdef TEST

   //nothing

#elif defined(STUBIFY_ME) //ooookay...

s32 Network::Resolve(const char * , u32 & )
{ return -1; }

#elif defined(OLDRESOLVE)  //***** OLD ****

// returns -1 on error, 0 on success
s32 Network::Resolve(const char *host, u32 &hostaddress )
{
  if ((hostaddress = inet_addr((char*)host)) == 0xFFFFFFFFL)
    {
    struct hostent *hp;
    if ((hp = gethostbyname((char*)host)) == NULL) return -1;

    int addrcount;

    // randomly select one
    for (addrcount = 0; hp->h_addr_list[addrcount]; addrcount++);
    int index = rand() % addrcount;
    memcpy((void*) &hostaddress, (void*) hp->h_addr_list[index], sizeof(u32));
    }
  return 0;
}

#else

s32 Network::Resolve(const char *host, u32 &hostaddress )
{
  struct proxylist *plist;
  struct proxylist dummylist;
  u32 addrlist[64]; /* should be more than enough */
  unsigned int proxypos, maxaddr, addrpos, addrcount;
  struct hostent *hp;

  int resport = lastport;            // as found in the network class
  int resauto = autofindkeyserver;

  if ( !host || !*host )
    {
    host = "";
    resauto = 1;
    resport = 0; // ie default port
    }
  else if ( *host && ((hostaddress = inet_addr((char*)host)) != 0xFFFFFFFFL))
    return 0;

  #ifdef RESDEBUG
    printf("host:=%s:%d autofindkeyserver=%d\n", host, (int)lastport, (int)resauto );
  #endif

  if ( resauto && (!*host || IsHostnameDNetKeyserver( host, NULL )))
    #ifdef RESDEBUGZONE
    plist = GetApplicableProxyList( resport, RESDEBUGZONE*60 ); 
    #else
    plist = GetApplicableProxyList( resport, calc_tzmins() ); 
    #endif
  else
    {
    dummylist.numproxies = 1;
    dummylist.proxies = &host;
    plist = &dummylist;
    }

  addrcount = 0;
  for (proxypos = 0; (proxypos < (plist->numproxies)) &&
         (addrcount < (sizeof(addrlist)/sizeof(addrlist[0]))); proxypos++ )
    {
    #ifdef RESDEBUG
      printf(" => %d:\"%s\"\n", proxypos+1, plist->proxies[proxypos] );
    #endif

    if ((hp = gethostbyname((char*)(plist->proxies[proxypos]))) != NULL) 
      {
      maxaddr = addrcount + 1;
      #ifdef NAMESERVERS_DONT_ROTATE
      maxaddr = (sizeof(addrlist)/sizeof(addrlist[0]));
      #endif
      for ( addrpos = 0; hp->h_addr_list[addrpos] && 
                                  (addrcount < maxaddr ); addrpos++ )
        {      
        memcpy((void*) &addrlist[addrcount], 
               (void*) hp->h_addr_list[addrpos], sizeof(u32));
        addrcount++;
        }
      }
    }

  if (addrcount < 1)
    return -1;
  if (addrcount == 1)  
    hostaddress = addrlist[0];
  else
    hostaddress = addrlist[rand() % addrcount];

  #ifdef RESDEBUG                                           
  printf(" total adds==%d  Selected add=%d.%d.%d.%d\n", //screw inet_ntoa()
               addrcount, (int)(hostaddress & 0xff), (int)((hostaddress >> 8) & 0xff),
               (int)((hostaddress >> 16) & 0xff), (int)((hostaddress >> 24) & 0xff) );
  #endif
  return 0;
}

#endif //OLDRESOLVE | stub | TEST

//-----------------------------------------------------------------------

#ifdef TEST
int main(void)
{
  struct proxylist *plist;
  char getbuffer[20];
  char hostname[60];
  int port, pos, pos2, proceed = 1, abstzdiff, tzdiff;
  char *p = "(after DST compensation)";

  tzdiff = calc_tzmins();
  while (proceed)
    {
    abstzdiff = ((tzdiff<0)?(-tzdiff):(tzdiff));
    printf("  Result for timezone %c%02d%02d %s:\n", 
               ((tzdiff<0)?('-'):('+')), abstzdiff/60, abstzdiff%60, p );
    for (pos = 0;pos<3;pos++)
      {
      port = ((pos==0)?(23):((pos==1)?(80):(0)));
      plist = GetApplicableProxyList( port, tzdiff );
   
      printf("  Port %-10.10s:",((pos==0)?("23"):((pos==1)?("80"):("(other)"))));
   
      for (pos2=0;pos2<plist->numproxies;pos2++)
       {
       if ((p=strchr(plist->proxies[pos2],'.'))!=NULL)
         *p=0;
       printf(" %-10.10s  ",plist->proxies[pos2]);
       }
      printf("\n");
      }
    proceed = -1;
    while (proceed == -1)
      {
      getbuffer[0]=0;
      printf( "\nEnter TZ (-1200 to +1200) or "
              "name (\"euro\", \"jp\" etc) to test : " );
      scanf( "%10s", getbuffer ); 
      
      if (getbuffer[0] == 0)
        proceed = 0;
      else if (getbuffer[0] == '-' || getbuffer[0]=='+' 
                 || (getbuffer[0]>='0' && getbuffer[0]<='9'))
        {
        tzdiff = atoi( getbuffer );
        if (tzdiff > -60 && tzdiff < +60)
          tzdiff = tzdiff*100;
        if ( tzdiff < -1200 || tzdiff > +1200 )
          printf( "That timezone doesn't appear to be valid\n" );
        else
          {
          tzdiff = ((tzdiff/100)*60)+(tzdiff%100);
          p = "(assumes DST was pre-compensated)";
          proceed = 1;
          }
        }
      else 
        {
        if ( strchr( getbuffer, '.' ) == NULL )
          sprintf( hostname, "%s.%s", getbuffer, DNET_PROXY_DOMAINNAME );
        else
          strcpy( hostname, getbuffer );
        if ( IsHostnameDNetKeyserver( hostname, &tzdiff ) )
          {
          proceed = 1;
          if ( strchr( getbuffer, '.' ) == NULL )
            sprintf( hostname, "(proxy %s.%s)", getbuffer, DNET_PROXY_DOMAINNAME );
          else
            sprintf( hostname, "(%s)", getbuffer );
          p = hostname;
          }
        else 
          printf( "That proxyname doesn't appear to be valid\n" );
        }
      }
    }
  return 0;
}  
#endif
