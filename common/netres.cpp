/* Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/
const char *netres_cpp(void) {
return "@(#)$Id: netres.cpp,v 1.25.2.13 2000/05/25 14:40:56 cyp Exp $"; }

//#define TEST  //standalone test
//#define RESDEBUG //to show what network::resolve() is resolving
#ifdef RESDEBUG
  //#define RESDEBUGZONE +12  //the timezone we want to appear to be in
#endif

#if defined(TEST) || defined(PROXYTYPE)
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#endif

#if !defined(TEST)
  #include "cputypes.h"
  #if defined(PROXYTYPE)
    #include "netio.h"
  #else
    #include "baseincs.h"
    #include "network.h"
    #include "clitime.h" /* CliTimeGetMinutesWest() */
    #include <ctype.h> //tolower()
  #endif
#endif

#undef NETRES_STUBS_ONLY
#if !defined(AF_INET) && !defined(SOCK_STREAM)
  #define NETRES_STUBS_ONLY
#endif

//------------------------------------------------------------------------

#ifndef NETRES_STUBS_ONLY
static const struct        // this structure defines which proxies are
{                          // 'responsible' for which time zone. The
  const char *name;        // timezones overlap, and users in an overlapped
  int minzone;             // area will have 2 (or more) proxies at their
  int maxzone;             // disposal.
  int midzone;
} proxyzoi[] = {
                { "us",    -10, -1 ,  -5 },
                { "euro",   -2, +4 ,  +2 }, //euro crosses 0 degrees longitude
                { "asia",   +3, +10,  +9 },
                { "aussie", +9, -9 , +12 }, //jp and aussie cross the dateline
                { "jp",    +10, -10, -11 }
               };
//static int dnet_portlist[] = {80,23,2064,3064,110};
static const char DNET_PROXY_DOMAINNAME[]="v27.distributed.net"; // NOT char*
#endif

//-------------------------------------------------------------------------

#ifndef NETRES_STUBS_ONLY
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
          int xport = ((isdigit(buffer[i]))?(atoi(buffer+i)):(-1));
          if ((proxyzoi[pos].name[i] == 0) && ((buffer[i]=='.') ||
                 xport==80 || xport==23 || xport==2064 ||
                 xport==3064 || xport==110 ))
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

#if (!defined(NETRES_STUBS_ONLY) || defined(TEST))
static int calc_tzmins(void)
{
  #if !defined(TEST) && !defined(PROXYTPE)
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

  if (utctime.tm_isdst>0)
    tzdiff-=60;
  if (tzdiff < -(12*60))
    tzdiff = -(12*60);
  else if (tzdiff > +(12*60))
    tzdiff = +(12*60);
  if (haveutctime && haveloctime)
    saved_tz = tzdiff;

  return tzdiff;
  #endif
}
#endif

//-----------------------------------------------------------------------

struct proxylist
{
  unsigned int numproxies;
  const char **proxies;
};

//-----------------------------------------------------------------------

#ifndef NETRES_STUBS_ONLY
static struct proxylist *GetApplicableProxyList(int port, int tzdiff) /*host order*/
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

// Linux especially has problems with proper name resolution handling
// because of the existence of C Libraries based on libc5, glibc2.0,
// and glibc2.1; all of which present binary compatibility issues
// typically resulting in only name resolutions failing.  This small
// shim attempts to explicitly dynamically load the library containing
// the resolver and directly call it, bypassing the statically linked
// version of gethostbyname() bound to our executable.

#if (CLIENT_OS == OS_LINUX) && defined(UNIVERSALRESOLVER)
#include <dlfcn.h>
static struct hostent *__linux_gethostbyname_shim( const char *hostname )
{
  static struct { const char *lib, *sym; } libsyms[] = {
         { "/usr/lib/libresolv.so", "res_gethostbyname" },
         { "/usr/lib/libc.so",      "gethostbyname"     }    };
  struct hostent *hp = gethostbyname(hostname);
  int libsym;

  for (libsym=0; !hp && libsym<int(sizeof(libsyms)/sizeof(libsyms[0])); libsym++)
  {
    void *plib = dlopen( libsyms[libsym].lib, RTLD_LAZY);
    if (!plib) 
    {
      //printf("failed to load '%s'\n", libsyms[libsym].lib);
    }
    else 
    {
      void *funcptr = dlsym(plib, libsyms[libsym].sym );
      if (!funcptr) 
      {
        //printf("failed to get pointer to '%s'\n", libsyms[libsym].sym);
      }
      else
      {
        struct hostent *(*pgethostbyname)(const char *name) =
              (struct hostent*(*)(const char *)) funcptr;
        hp = pgethostbyname(hostname);
        if (!hp)
        {
          //printf("no results\n");
        }
        else
        {
          static char hostbuf[1024], aliasbuf[1024];
          static char addrbuf[32*sizeof(ulong)];
          static char *aliaslist[2]; 
          static char *addrlist[32];
          static struct hostent hent;
  
          hostbuf[0] = 0;
          aliaslist[0]= (char *)0;
          addrlist[0] = (char *)0;
  
          hent.h_addrtype = hp->h_addrtype;
          hent.h_length = hp->h_length;
          hent.h_name = &(hostbuf[0]);
          hent.h_aliases = &(aliaslist[0]);
          hent.h_addr_list = &(addrlist[0]);
  
          if (hp->h_addr_list)
          {
            unsigned int addrpos = 0;
            while ( hp->h_addr_list[addrpos] 
              && ( ((addrpos+1)*hp->h_length) < sizeof(addrbuf))
              && ( (addrpos+1) < (sizeof(addrlist)/sizeof(addrlist[0])) ))
            {
              char *p = hp->h_addr_list[addrpos];
              char *q = &addrbuf[addrpos*hp->h_length];
              memcpy(q, p, hp->h_length );
              addrlist[addrpos++] = q;
            }  
            addrlist[addrpos] = (char *)0;
          }
          if (hp->h_name)
          {
            strncpy( hostbuf, hp->h_name, sizeof(hostbuf) );
            hostbuf[sizeof(hostbuf)-1] = '\0';
          }
          if (hp->h_aliases)
          {
            if (hp->h_aliases[0])
            {
              strncpy( aliasbuf, hp->h_aliases[0], sizeof(aliasbuf));
              aliasbuf[sizeof(aliasbuf)-1] = '\0';
              aliaslist[0] = &aliasbuf[0];
              aliaslist[1] = (char *)0;
            }  
          }
          hp = &hent;
        }
        //free symbol?
      }
      dlclose(plib);
    }
  }
  return hp;
}
// remap calls to our replacement shim.
#undef gethostbyname
#define gethostbyname(xx) __linux_gethostbyname_shim(xx)
#endif

//-----------------------------------------------------------------------

#ifndef NETRES_STUBS_ONLY
// Returns -1 if the resolve fails, or 0 on success.
static int __LowLevelGethostbyname(const char *hostname,
    u32 *addrlist, unsigned int addrlistcount,
    unsigned int *foundaddrcount)
{
  struct hostent *hp;
  char *lookup;

  // copy pointer to "lookup" to work around gethostbyname()
  // not prototyped to take const arg on some platforms.
  *((const char **)&lookup) = hostname;

  if ((hp = gethostbyname(lookup) ) != NULL)
  {
    unsigned int addrpos;

    // Iterate through the matching IP Address list and add only
    // addresses that we haven't already added, ignoring the
    // duplicates that already exist.
    for ( addrpos = 0; (hp->h_addr_list[addrpos] &&
         (*foundaddrcount < addrlistcount)); addrpos++ )
    {
      unsigned int dupcheck = 0;
      addrlist[*foundaddrcount] = *((u32 *)(hp->h_addr_list[addrpos]));
      while (dupcheck < *foundaddrcount)
      {
        if (addrlist[*foundaddrcount] == addrlist[dupcheck])
          break;
        dupcheck++;
      }
      if (!(dupcheck < *foundaddrcount)) /* no dupes */
        (*foundaddrcount)++;
    }
    return 0;
  }
  return -1;
}
#endif

//-----------------------------------------------------------------------

int NetResolve( const char *host, int resport, int resauto,
                u32 *addrlist, unsigned int addrlistcount,
                char *resolve_hostname, unsigned int resolve_hostname_sz )
{
  unsigned int foundaddrcount = 0;

  if (!resolve_hostname)
    resolve_hostname_sz = 0;
  else if (resolve_hostname_sz)
    resolve_hostname[0] = '\0';

  if (!host || !addrlist || !addrlistcount)
    return -1;

  resport = resport; //shaddup compiler
  resauto = resauto;

  #if (!defined(TEST) && !defined(NETRES_STUBS_ONLY))
  {
    int whichpass, maxpass;
    unsigned int pos;
    char hostname[64];

    while (*host && isspace(*host))
      host++;
    pos = 0;
    while (*host && !isspace(*host) &&
                    isprint(*host) && *host!='\r' && *host!='\n')
    {
      if (pos == (sizeof(hostname)-1))
        return -1;
      hostname[pos++]=(char)tolower(*host++);
    }
    hostname[pos]=0;
    host = hostname;

    if (pos == 0) /* len == 0 */
      resauto = 1;
    else 
    {
      addrlist[0] = 0xFFFFFFFFL;
      if (pos > 13 && strcmp(&hostname[pos-13],".in-addr.arpa")==0)
      {
        hostname[pos-=13]='\0';
        #if (CLIENT_OS == OS_MACOS)
        addrlist[0] = (u32)(inet_addr(hostname).s_addr);
        #else
        addrlist[0] = (u32)(inet_addr(hostname));
        #endif
        if (addrlist[0] != 0xFFFFFFFFL)
          addrlist[0] = ((addrlist[0]>>24)&0xff)|((addrlist[0]>>8)&0xff00)|
                        ((addrlist[0]&0xff)<<24)|((addrlist[0]&0xff00)<<8);
      }
      else
      {
        #if (CLIENT_OS == OS_MACOS)
        addrlist[0] = (u32)(inet_addr(hostname).s_addr);
        #else
        addrlist[0] = (u32)(inet_addr(hostname));
        #endif
      }  
      if (addrlist[0] != 0xFFFFFFFFL)
      {
        if (addrlist[0] == 0)
          return -1;
        if (resolve_hostname_sz)
        {
          host = (const char *)(addrlist);
          sprintf(hostname, "%d.%d.%d.%d.in-addr.arpa",
                    (host[3]&255),(host[2]&255),(host[1]&255),(host[0]&255) );
          strncpy(resolve_hostname,hostname,resolve_hostname_sz);
          resolve_hostname[resolve_hostname_sz-1] = '\0';
        }
        return 1; /* only one address */
      }
    }

    if ( resauto && !resport )
      return -1;

    #ifdef RESDEBUG
      printf("host:=%s:%d autofindkeyserver=%d\n", hostname, (int)resport, (int)resauto );
    #endif

    if (resauto && (!hostname[0] || IsHostnameDNetKeyserver(hostname,NULL)))
    {
      resauto = 1;
      maxpass = 2;
    }
    else
    {
      resauto = 0;
      maxpass = 1;
    }

    foundaddrcount = 0;
    for (whichpass = 0; (foundaddrcount==0 && whichpass<maxpass); whichpass++)
    {
      struct proxylist *plist;
      struct proxylist dummylist;

      // If automatic geographic zone selection based on time zone should
      // be done, then first identify which zones apply to us.
      if (resauto)
      {
        int tzmin = -6*60; /* middle of us.d.net */
        if (whichpass == 0) /* first pass */
        {
          #ifdef RESDEBUGZONE
          tzmin = RESDEBUGZONE*60;
          #else
          tzmin = calc_tzmins();
          #endif
        }
        plist = GetApplicableProxyList( resport, tzmin );

        #ifdef RESDEBUG
        for (pos=0;pos<plist->numproxies;pos++)
          printf("%s resolved to %s\n", hostname, plist->proxies[pos]);
        #endif
      }
      else
      {
        host = (const char *)&hostname[0];
        dummylist.proxies = &host;
        dummylist.numproxies = 1;
        plist = &dummylist;
      }

      // Iterate through all of the hostnames lists and place all
      // of the resolved IP Addresses into the "addrlist".
      for (pos = 0; ((pos < (plist->numproxies)) &&
                         (foundaddrcount < addrlistcount)); pos++ )
      {
        #ifdef RESDEBUG
        printf(" => %d:\"%s\"\n", pos+1, plist->proxies[pos] );
        #endif

        if (__LowLevelGethostbyname(plist->proxies[pos], addrlist,
            addrlistcount, &foundaddrcount) >= 0)
        {
          // when successful, copy the first good hostname into
          // the "resolve_hostname" buffer for display purposes.
          if (resolve_hostname_sz)
          {
            if (resolve_hostname[0] == '\0')
            {
              strncpy(resolve_hostname,plist->proxies[pos],
                      resolve_hostname_sz);
              resolve_hostname[resolve_hostname_sz-1]='\0';
            }
          }
        }
      }

      // If a successful zone name was found, then do something with it.
      if (resolve_hostname_sz)
      {
        if (resolve_hostname[0] == '\0')
        {
          if ( plist->numproxies >=1 )
            strncpy( resolve_hostname, plist->proxies[0], resolve_hostname_sz );
          else
            strncpy( resolve_hostname, hostname, resolve_hostname_sz );
          resolve_hostname[resolve_hostname_sz-1]='\0';
        }  
      }
    }
  }
  #endif /* (!defined(TEST) && !defined(NETRES_STUBS_ONLY)) */

  if (foundaddrcount < 1)
    return -1;
  return (int)foundaddrcount;
}

//-----------------------------------------------------------------------

#if 0
int Network::Resolve(const char *host, u32 *hostaddress, int resport )
{
  u32 addrlist[64]; /* should be more than enough */
  int acount;

  acount = NetResolve( host, resport, autofindkeyserver,
                       &addrlist[0], (sizeof(addrlist)/sizeof(addrlist[0])),
                       resolve_hostname, sizeof(resolve_hostname) );
  if (acount < 1) /* failed */
    return -1;

  *hostaddress = addrlist[0]; //let the nameserver handle rotation
  //*hostaddress = addrlist[rand() % acount];

  return 0;
}
#endif

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

//-----------------------------------------------------------------------

