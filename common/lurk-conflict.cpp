//#define LURKDEBUG
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this sce violates copyright.
//
/*
   This module contains functions for both lurking and dial initiation/hangup.
   The workhorse protected function, IsConnected(), is needed to support both.

   Lurk detection is trivial and contain no OS specific routines.
     CheckIfConnectRequested() and CheckForStatusChange()
   Public function used for dial initiation/hangup:
     DialIfNeeded() and HangupIfNeeded()

*/
// $Log: lurk-conflict.cpp,v $
// Revision 1.34  1999/03/05 21:39:33  patrick
//
// added define for SIOSTATAT, defined solcose to be close for OS2-EMX target
//
// Revision 1.33  1999/02/14 02:48:58  silby
// Switch back to old win32 naming convention for RAS procs.
//
// Revision 1.32  1999/02/14 00:32:32  silby
// Changed order DialIfNeeded checks status so RAS is not inited if
// it doesn't need to be.
//
// Revision 1.31  1999/02/12 23:21:19  silby
// Changed win32 RAS function names used so that they
// do not conflict with system headers.
//
// Revision 1.30  1999/02/10 21:39:21  remi
// Cleared a warning on FreeBSD.
//
// Revision 1.29  1999/02/10 03:49:30  cyp
// fixed a couple of LogScreen()s that weren't terminated with '\n'
//
// Revision 1.28  1999/02/10 00:30:11  trevorh
// Removed #include for platforms\os2cli\os2defs.h
//
// Revision 1.27  1999/02/10 00:01:56  trevorh
// Change delay() to sleep() for ifdefed OS/2 code
//
// Revision 1.26  1999/02/09 23:46:22  cyp
// a) Default Ifacemask is internal to lurk; b) added wildcard support for
// iface masking. b) Changed mask default to '\0' which now means 'all
// dialup' ('*' is 'all')
//
// Revision 1.25  1999/02/09 10:45:00  remi
// Added GetDefaultIFaceMask().
// Lots of Win16/Win32 fixes from Cyp.
//
// Revision 1.24  1999/02/08 23:21:52  remi
// Rewrote connifacemask[] parsing code and moved it out of CLIENT_OS defines.
// Added FreeBSD support, same code may work on other BSD systems.
//
// Revision 1.23  1999/02/08 17:11:39  cyp
// re-added errno.h for linux
//
// Revision 1.22  1999/02/07 16:02:38  cyp
// gericificied variable/function names; fixed blatant bugs.
//
// Revision 1.21  1999/02/06 09:08:08  remi
// Enhanced the lurk fonctionnality on Linux. Now it use a list of interfaces
// to watch for online/offline status. If this list is empty (the default), any
// interface up and running (besides the loopback one) will trigger the online
// status.
//
// Revision 1.20  1999/02/05 23:24:38  silby
// Changed long -> unsigned long so type matched.
//
// Revision 1.19  1999/02/04 22:53:15  trevorh
// Corrected compilation errors for OS/2 after cyp change
//
// Revision 1.18  1999/02/04 07:47:06  cyp
// Re-import from proxy base. Cleaned up. Added linux and win16 support.
//
// Revision 1.17  1999/02/03 20:18:57  patrick
// changed the define for the SIOSTATIF call
//
// Revision 1.16  1999/01/29 18:49:05  jlawson
// fixed formatting.
//
// Revision 1.4  1999/01/29 01:27:28  trevorh
// Corrected detection of lurkonly mode. The move from client to proxy
// changed the numbers used for lurkonly. Now uses CONNECT_LURKONLY define.
//
// Revision 1.3  1999/01/28 19:52:46  trevorh
// Detect existing dialup connection under OS/2
//
// Revision 1.2  1999/01/24 23:14:58  dbaker
// copyright 1999 changes
//
// Revision 1.1  1998/12/30 01:45:11  jlawson
// added lurk code from client.
//
// Revision 1.11  1998/12/01 19:49:14  cyp
// Cleaned up MULT1THREAD #define. See cputypes.h for details.
//
// Revision 1.10  1998/11/17 05:49:02  silby
// Fixed an uninit variable that was causing updates that should not have been.
//
// Revision 1.9  1998/11/12 13:09:01  silby
// Added a stop function, made start and stop public.
//
// Revision 1.8  1998/11/01 20:35:11  cyp
// Adjusted so that attempts to LoadLibrary("RASAPI.DLL") don't put the client
// into an infinite loop on NT/Win32s. Note to Silby: lurk never de-initializes
// itself - a balancing FreeLibrary() is missing.
//
// Revision 1.7  1998/10/04 11:37:48  remi
// Added Log and Id tags.
//
#if (!defined(lint) && defined(__showids__))
const char *lurk_cpp(void) {
return "@(#)$Id: lurk-conflict.cpp,v 1.34 1999/03/05 21:39:33 patrick Exp $"; }
#endif

#include "cputypes.h"
#include "lurk.h"
#ifdef PROXYTYPE
#include "globals.h"
#define  LogScreen blah
#else
#include "logstuff.h"
#endif

Lurk dialup;

/* ---------------------------------------------------------- */

int Lurk::Stop(void) // also called from constructor/destructor
{
  islurkstarted = lastcheckshowedconnect = dohangupcontrol = 0;
  lurkmode = dialwhenneeded = 0;
  conndevice[0] = connprofile[0] = connifacemask[0] = 0;
  connstartcmd[0] = connstopcmd[0] = ifacemaskcopy[0]=0;
  ifacestowatch[0] = (const char *)0;
  return 0;
}

/* ---------------------------------------------------------- */

int Lurk::CheckIfConnectRequested(void) //yes/no
{
  if ((lurkmode & (CONNECT_LURKONLY|CONNECT_LURK)) == 0)
    return 0; // We're not supposed to lurk!

  if (IsConnected()) //we are connected!
    {
    if ( lastcheckshowedconnect == 0 ) // We previously weren't connected
      {
      if (conndevice[0]==0)
        LogScreen("Dialup Connection Detected...\n"); // so this is the first time
      else
        LogScreen("Connection detected on '%s'...\n", conndevice );
      }
    lastcheckshowedconnect = 1;
    return 1;
    }
  else if ( lastcheckshowedconnect ) // we were previously connected...connection lost
    {
    char *msg = "";
    lastcheckshowedconnect = 0;        // So we know next time through this loop
    if (lurkmode == CONNECT_LURKONLY) // Lurk-only mode
      msg = "\nConnections will not be initiated by the client.";
    LogScreen("Dialup Connection Disconnected.%s\n",msg);
    }
  return 0;
}

/* ---------------------------------------------------------- */

int Lurk::CheckForStatusChange(void) //returns -1 if connection dropped
{
  if ( (lurkmode == 0) && (!dialwhenneeded) )
    return 0; // We're not lurking.
  if ( lastcheckshowedconnect && !IsConnected() ) //if (Status() < oldlurkstatus)
    return -1;  // we got disconnected!
  return 0;
}


/* ================================================================== */
/* ************** OS SPECIFIC STUFF BEGINS HERE ********************* */
/* ================================================================== */


#if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD)
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h> // linux/if.h
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>

#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)

#include <windows.h>
#include <string.h>
static HINSTANCE hWinsockInst = NULL;

#elif (CLIENT_OS == OS_WIN32)

#include <windows.h>
#include <ras.h>
#include <raserror.h>
#include <string.h>
#include "w32ras.cpp"

static HRASCONN hRasDialConnHandle = NULL; /* conn we opened with RasDial */

#elif (CLIENT_OS == OS_OS2)

#define INCL_DOSPROCESS
#include <os2.h>

#define TCPIPV4               //should also work with V3 though
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#if defined(__EMX__)
#include <sys/process.h>
#include <sys/types.h>
#define MAXSOCKETS 2048
#define soclose(s) close(s)     //handled by EMX
#else //IBM distributed OS/2 developers toolkit
#include <process.h>
#include <types.h>  
#endif
extern "C" {
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
}
#include <net/if.h>          // ifmib

#ifndef SIOSTATIF
#define SIOSTATIF  _IOR('n', 48, struct ifmib)
#define SIOSTATAT  _IOR('n', 49, struct ifmib)
#endif
#ifndef ULONG
  typedef unsigned long ULONG;
#endif
#pragma pack(2)
struct ifact
{
  short ifNumber;
  struct iftable
    {
    ULONG ifa_addr;
    short  ifIndex;
    ULONG ifa_netm;
    ULONG ifa_brdcast;
    } iftable[IFMIB_ENTRIES];
};
#pragma pack()
#endif

/* ========================================================== */

int Lurk::GetCapabilityFlags(void)
{
  int what = 0;
#if (CLIENT_OS == OS_WIN32)
  OFSTRUCT ofstruct;
  ofstruct.cBytes = sizeof(ofstruct);
  #ifndef OF_SEARCH
  #define OF_SEARCH 0x0400
  #endif
  if ( OpenFile( "RASAPI32.DLL", &ofstruct, OF_EXIST|OF_SEARCH) != HFILE_ERROR)
    what = (CONNECT_LURK|CONNECT_LURKONLY|CONNECT_DODBYPROFILE);
#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  OFSTRUCT ofstruct;
  ofstruct.cBytes = sizeof(ofstruct);
  if ( OpenFile( "WINSOCK.DLL", &ofstruct, OF_EXIST|OF_SEARCH) != HFILE_ERROR)
    {
    char *p = strrchr( ofstruct.szPathName, '\\' );
    const char *q = "TRUMPWSK.INI";
    what = ( CONNECT_LURK | CONNECT_LURKONLY);
    if (p != NULL)
      {
      strcpy( p+1, q );
      q = (const char *)(&ofstruct.szPathName[0]);
      }
    if ( OpenFile( q, &ofstruct, OF_EXIST|OF_SEARCH) != HFILE_ERROR)
      {
      int i=GetPrivateProfileInt( "Trumpet Winsock", "dial-option", 0, ofstruct.szPathName );
      if (i != 0) /* 1==login on demand, 2=login/logout on demand */
        what |= CONNECT_DODBYPROFILE;
      }
    }
#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD)
  what = (CONNECT_LURK | CONNECT_LURKONLY | CONNECT_DODBYSCRIPT | CONNECT_IFACEMASK);
#elif (CLIENT_OS == OS_OS2)
  what = (CONNECT_LURK | CONNECT_LURKONLY | CONNECT_DODBYSCRIPT | CONNECT_IFACEMASK);
#endif
  return what;
}

/* ---------------------------------------------------------- */

const char **Lurk::GetConnectionProfileList(void)
{
#if (CLIENT_OS == OS_WIN32)
  static const char *firstentry = ""; //the first entry is blank, ie use default
  static RASENTRYNAME rasentries[4];
  static const char *configptrs[(sizeof(rasentries)/sizeof(rasentries[0]))+2];
  DWORD buffersize = sizeof(rasentries);
  DWORD maxentries = 0;
  int rasok = 0;

  rasentries[0].dwSize = sizeof(RASENTRYNAME);
  if (RasEnumEntries(NULL,NULL,&rasentries[0],&buffersize,&maxentries) == 0)
    rasok = 1;
  else if (buffersize > (DWORD)(sizeof(rasentries)))
    {
    RASENTRYNAME *rasentryp = (RASENTRYNAME *)malloc((int)buffersize);
    if (rasentryp)
      {
      maxentries = 0;
      rasentryp->dwSize=sizeof(RASENTRYNAME);
      if ( RasEnumEntries(NULL,NULL,rasentryp,&buffersize,&maxentries) == 0)
        {
        rasok = 1;
        buffersize = (DWORD)(maxentries*sizeof(rasentries[0]));
        if (buffersize > sizeof(rasentries))
          buffersize = sizeof(rasentries);
        maxentries = buffersize/sizeof(rasentries[0]);
        memcpy((void *)(&rasentries[0]),(void *)rasentryp,buffersize);
        }
      free((void *)rasentryp);
      }
    }
  if (rasok)
    {
    DWORD entry = 0;
    unsigned int index = 0;
    configptrs[index++] = firstentry; //the first entry is "", ie use default
    for (entry = 0; entry < maxentries;entry++)
      configptrs[index++] = rasentries[entry].szEntryName;
    configptrs[index] = NULL;
    return (const char **)(&configptrs[0]);
    }
#endif
  return NULL;
}

/* ---------------------------------------------------------- */

int Lurk::Start(void)// Initializes Lurk Mode. returns 0 on success.
{
  int flags = GetCapabilityFlags();

  if (lurkmode != CONNECT_LURKONLY && lurkmode != CONNECT_LURK)
    lurkmode = 0;           /* can only be one or the other */

  if (lurkmode || dialwhenneeded)
    {
    if (lurkmode && (flags & (CONNECT_LURK|CONNECT_LURKONLY))==0)
      {              //only happens if user used -lurk on the command line
      lurkmode = 0;
      #if (CLIENT_OS == OS_WIN32)
      LogScreen( "Dial-up must be installed for lurk/lurkonly/dialing\n" );
      dialwhenneeded = 0; //if we can't support lurk, we can't support dod either
      #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
      LogScreen("Winsock must be available for -lurk/-lurkonly.\n");
      dialwhenneeded = 0; //if we can't support lurk, we can't support dod either
      #else
      LogScreen("-lurk/-lurkonly is not supported. Option ignored.\n");
      #endif
      }
    if (dialwhenneeded && (flags & (CONNECT_DOD))==0)
      {               //should never happen since dod is not a cmdline option
      dialwhenneeded = 0;
      #if (CLIENT_OS == OS_WIN32)
      LogScreen( "Dial-up must be installed for demand dialing\n" );
      #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
      LogScreen("Demand dialing is only supported with Trumpet Winsock.\n");
      #else
      LogScreen("Demand dialing is currently unsupported.\n");
      #endif
      }
    }

  if (connprofile[0]!=0)
    {
    int n=0, pos=0;
    while (connprofile[pos] && isspace(connprofile[pos]))
      pos++;
    while (connprofile[pos])
      connprofile[n++] = connprofile[pos++];
    while (n>0 && isspace(connprofile[n-1]))
      --n;
    connprofile[n]=0;
    }

  mask_include_all = mask_default_only = 0;
  ifacestowatch[0]=NULL;
  ifacemaskcopy[0]=0;

  if ((flags & CONNECT_IFACEMASK)==0)
    mask_include_all = 1;
  else if (connifacemask[0]==0)
    mask_default_only = 1;
  else if (strcmp(connifacemask,"*")==0)
    mask_include_all = 1;
  else
    {
    // Parse connifacemask[] and store each iface name in *ifacestowatch[]
    int ptrindex = 0, stindex = 0;
    char *c = &(connifacemask[0]);
    do{
      while (*c && (isspace(*c) || *c==':'))
        c++;
      if (*c)
        {
        char *p = &ifacemaskcopy[stindex];
        while (*c && !isspace(*c) && *c!=':')
          ifacemaskcopy[stindex++] = *c++;
        ifacemaskcopy[stindex++]='\0';
        if (strcmp( p, "*" )==0)
          {
          ptrindex = 0;
          mask_include_all = 1;
          break;
          }
        #if (CLIENT_OS == OS_OS2) //convert 'eth*' names to 'lan*'
        if (*p=='e' && p[1]=='t' && p[2]=='h' && (isdigit(p[3]) || p[3]=='*'))
          {*p='l'; p[1]='a'; p[2]='n'; }
        #endif
        ifacestowatch[ptrindex++] = (const char *)p;
        }
      } while (*c);
    if (ptrindex == 0 && !mask_include_all) //nothing in list
      mask_default_only = 1;
    ifacestowatch[ptrindex] = NULL;
    }
  #ifdef LURKDEBUG
  printf("mask flags: include_all=%d, defaults_only=%d\niface list:\n",
                 mask_include_all, mask_default_only );
  for (int ptrindex=0;ifacestowatch[ptrindex];ptrindex++)
    printf("  %d) '%s'\n",ptrindex+1,ifacestowatch[ptrindex]);
  printf("lurkmode=%d dialwhenneeded=%d\n",lurkmode,dialwhenneeded);
  #endif

  islurkstarted=1;
  return 0;
}

/* ---------------------------------------------------------- */

#if (CLIENT_OS==OS_LINUX) || (CLIENT_OS==OS_FREEBSD)
static int __MatchMask( const char *ifrname, int mask_include_all,
                       int mask_default_only, const char *ifacestowatch[] )
{
  int ismatched = 0;
  char wildmask[32+4]; //should be sizeof((struct ifreq.ifr_name)+4
  const char *matchedname = "*";

  if (mask_include_all)
    ismatched = 1;
  else
    {
    int maskpos=0;
    strncpy(wildmask,ifrname,sizeof(wildmask));
    wildmask[sizeof(wildmask)-1]=0;
    while (maskpos < ((int)(sizeof(wildmask)-2)) &&
       wildmask[maskpos] && !isdigit(wildmask[maskpos]))
      maskpos++;
    wildmask[maskpos++]='*';
    wildmask[maskpos]='\0';
    if (mask_default_only)
      {
      ismatched = (strcmp(wildmask,"ppp*")==0 || strcmp(wildmask,"sl*")==0);
      #if (CLIENT_OS == OS_FREEBSD)
      if (!ismatched && strcmp(wildmask,"dun*")==0)
        ismatched = 1;
      #endif
      matchedname = ((!ismatched)?(NULL):((const char *)(&wildmask[0])));
      }
    else
      {
      for (maskpos=0;!ismatched && (matchedname=ifacestowatch[maskpos])!=NULL;maskpos++)
        ismatched = (strcmp(ifacestowatch[maskpos],ifrname)==0 ||
                     strcmp(ifacestowatch[maskpos],wildmask)==0);
      }
    }
  #ifdef LURKDEBUG
  LogScreen("matched?=%s ifrname=='%s' matchname=='%s'\n",
    (ismatched?"yes":"no"), ifrname, matchedname?matchedname:"(not found)" );
  #endif
  return ismatched;
}
#endif

int Lurk::IsConnected(void)               // Checks status of connection
{
  conndevice[0]=0;

  if (!islurkstarted)
    return 0;

#if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN16S)
  if ( GetModuleHandle("WINSOCK") )
    return 1;

#elif (CLIENT_OS == OS_WIN32)
  RASCONN rasconn;
  RASCONN *rasconnp = NULL;
  DWORD cb, whichconn, cConnections;
  int foundconn = 0;

  cb = sizeof(rasconn);
  rasconn.dwSize = sizeof(RASCONN);
  rasconnp = &rasconn;
  if (RasEnumConnections( rasconnp, &cb, &cConnections) != 0)
    {
    cConnections = 0;
    if (cb > (DWORD)(sizeof(RASCONN)))
      {
      rasconnp = (RASCONN *) malloc( (int)cb );
      if (rasconnp)
        {
        rasconnp->dwSize = sizeof(RASCONN);
        if (RasEnumConnections( rasconnp, &cb, &cConnections) != 0)
          cConnections = 0;
        }
      }
    }

  for (whichconn = 0; whichconn < cConnections; whichconn++ )
    {
    HRASCONN hrasconn = rasconnp[whichconn].hrasconn;
    char *connname = rasconnp[whichconn].szEntryName;
    RASCONNSTATUS rasconnstatus;
    rasconnstatus.dwSize = sizeof(RASCONNSTATUS);
    if (RasGetConnectStatus(hrasconn,&rasconnstatus) == 0)
      {
      if (rasconnstatus.rasconnstate == RASCS_Connected)
        {
        foundconn = 1;
        strncpy( conndevice, connname, sizeof(conndevice) );
        conndevice[sizeof(conndevice)-1]=0;
        break;
        }
      }
    }

  if (rasconnp != NULL && rasconnp != &rasconn)
    free((void *)rasconnp );
  if (foundconn)
    return 1;

#elif (CLIENT_OS == OS_OS2)
   int s, i, rc, j, foundif = 0;
   struct ifmib MyIFMib = {0};
   struct ifact MyIFNet = {0};
   int ismatched = 0;

   MyIFNet.ifNumber = 0;
   s = socket(PF_INET, SOCK_STREAM, 0);
   if (s >= 0)
     {
     /* get active interfaces list */
     if ( ioctl(s, SIOSTATAT, (char *)&MyIFNet, sizeof(MyIFNet)) >= 0 )
       {
       if ( ioctl(s, SIOSTATIF, (char *)&MyIFMib, sizeof(MyIFMib)) < 0)
         MyIFNet.ifNumber = 0;
       }
     for (i = 0; i < MyIFNet.ifNumber; i++)
       {
       j = MyIFNet.iftable[i].ifIndex;      /* j is now the index into the stats table for this i/f */
       if (mask_default_only == 0 || MyIFMib.iftable[j].ifType != HT_ETHER)   /* i/f is not ethernet */
         {
         if (MyIFMib.iftable[j].ifType != HT_PPP)  /* i/f is not loopback (yes I know it says PPP) */
           {
           if (MyIFNet.iftable[i].ifa_addr != 0x0100007f)  /* same thing for TCPIP < 4.1 */
             {
             struct ifreq MyIFReq = {0};
             const char *wildmask = "";
             if (j < 9)
               {
               sprintf(MyIFReq.ifr_name, "lan%d", j);
               wildmask = "lan*";
               }
             else if (j > 9)
               {
               sprintf(MyIFReq.ifr_name, "ppp%d", j-10);
               wildmask = "ppp*";
               }
             else
               {
               strcpy(MyIFReq.ifr_name, "lo");
               }
             strncpy( conndevice, MyIFReq.ifr_name, sizeof(conndevice) );
             conndevice[sizeof(conndevice)-1]=0;
             if (ioctl(s, SIOCGIFFLAGS, (char*)&MyIFReq, sizeof(MyIFReq))==0)
               {
               if ((MyIFReq.ifr_flags & IFF_UP) != 0)
                 {
                 int ismatched = 0;
                 if (mask_include_all)
                   ismatched = 1;
                 else if (mask_default_only)
                   ismatched = (MyIFMib.iftable[j].ifType != HT_ETHER);
                 else
                   {
                   int maskpos;
                   for (maskpos=0;!ismatched && ifacestowatch[maskpos];maskpos++)
                     ismatched= (strcmp(ifacestowatch[maskpos],MyIFReq.ifr_name)==0
                     || (*wildmask && strcmp(ifacestowatch[maskpos],wildmask)==0));
                   }
                 if (ismatched)
                   {
                   foundif = i+1; // Report online if SLIP or PPP detected
                   break;
                   }
                 }
               } // ioctl(s, SIOCGIFFLAGS ) == 0
             } // != 0x0100007f
           } // != HT_PPP (loopback actually)
         } // != HT_ETHER
       } //for ...
     soclose(s);
     }
   if (foundif)
     return 1;
   conndevice[0]=0;

#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD)  // maybe other *BSD systems
   struct ifconf ifc;
   struct ifreq *ifr;
   int n, foundif = 0;
   char *p;

   int fd = socket(PF_INET,SOCK_STREAM,0);
   if (fd >= 0)
     {
     // retrive names of all interfaces
     // note : SIOCGIFCOUNT doesn't work so we have to hack around
     //        to retrieve the number of interfaces
     unsigned numreqs = 10;
     ifc.ifc_len = 0;
     ifc.ifc_buf = (caddr_t) malloc( sizeof(struct ifreq) * numreqs );
     while (ifc.ifc_buf)
       {
       ifc.ifc_len = sizeof(struct ifreq) * numreqs;
       if (ioctl (fd, SIOCGIFCONF, &ifc) < 0)
         {
         ifc.ifc_len = 0;
         break;
         }
       if ((ifc.ifc_len / sizeof(struct ifreq)) < numreqs )
         break;
       // assume overflow, enlarge buffer
       numreqs += 10;
       p = (char *) realloc( (void *)ifc.ifc_buf, sizeof(struct ifreq)*numreqs );
       if (!p)
         break;
       ifc.ifc_buf = (caddr_t)p;
       }

     if (ifc.ifc_len)
       {
       #if (CLIENT_OS == OS_LINUX)
       for (n = 0, ifr = ifc.ifc_req; n < ifc.ifc_len; n += sizeof(struct ifreq), ifr++)
         {
         if (__MatchMask(ifr->ifr_name,mask_include_all,
                         mask_default_only, &ifacestowatch[0] ))
           {
           strncpy( conndevice, ifr->ifr_name, sizeof(conndevice) );
           conndevice[sizeof(conndevice)-1] = 0;
           ioctl (fd, SIOCGIFFLAGS, ifr); // get iface flags
           if ((ifr->ifr_flags & (IFF_UP | IFF_RUNNING | IFF_LOOPBACK))
               == (IFF_UP | IFF_RUNNING))
             {
             foundif = (n / sizeof(struct ifreq)) + 1;
             break;
             }
           }
         }
       #elif (CLIENT_OS == OS_FREEBSD)  // maybe other *BSD systems
       for (n = ifc.ifc_len, ifr = ifc.ifc_req; n >= (int)sizeof(struct ifreq); )
         {
         /*
          * In BSD4.4, SIOCGIFCONF returns an entry for every address
          * associated with the if. including physical.. They even included
          * a sockaddr of VARIABLE LENGTH!
          *
          * On n0 (FreeBSD 2.2.8) the last entry seems bogus, as it has only
          * 17 bytes while sizeof(struct ifreq) is 32.
          */
         struct sockaddr *sa = &(ifr->ifr_addr);
         if (sa->sa_family == AF_INET)  // filter-out anything other than AF_INET
           {                            // (in fact this filter-out AF_LINK)
           if (__MatchMask(ifr->ifr_name,mask_include_all,
                           mask_default_only, &ifacestowatch[0] ))
             {
             strncpy( conndevice, ifr->ifr_name, sizeof(conndevice) );
             conndevice[sizeof(conndevice)-1] = 0;
             ioctl (fd, SIOCGIFFLAGS, ifr); // get iface flags
             if ((ifr->ifr_flags & (IFF_UP | IFF_RUNNING | IFF_LOOPBACK))
                 == (IFF_UP | IFF_RUNNING))
               {
               foundif = (n / sizeof(struct ifreq)) + 1;
               break;
               }
             }
           }
         // calculate the length of this entry and jump to the next
         int ifrsize = IFNAMSIZ + sa->sa_len;
         ifr = (struct ifreq *)((caddr_t)ifr + ifrsize);
         n -= ifrsize;
         }
       #else
       #error "What's up Doc ?"
       #endif
       }
     if (ifc.ifc_buf)
       free (ifc.ifc_buf);
     close (fd);
     }

   if (foundif)
     return 1;
   conndevice[0]=0;

#endif // OS_LINUX || FREEBSD
  return 0;// Not connected
}

/* ---------------------------------------------------------- */

int Lurk::DialIfNeeded(int force /* !0== override lurk-only */ )
{                                /* returns 0 if connected, -1 if error */
  if (!islurkstarted)
    return -1; // Lurk can't be started, evidently

  if (!dialwhenneeded)           // We don't handle dialing
    return 0;

  if (IsConnected()) // We're already connected
    return 0;

  if (lurkmode == CONNECT_LURKONLY && !force)
    return -1; // lurk-only, we're not allowed to connect unless forced

#if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)

  if (hWinsockInst != NULL) //programmer error - should never happen
    {
    LogScreen("SyncError: repeated calls to Lurk::DialIfNeeded()\n"
              "without intervening Lurk::HangupIfNeeded()\n" );
    return -1;
    }
  dohangupcontrol = 0;           // whether we do HangupIfNeeded() or not
  if ((hWinsockInst = LoadLibrary("WINSOCK.DLL")) < ((HINSTANCE)(32)))
    {
    hWinsockInst = NULL;
    return -1;
    }
  dohangupcontrol = 1;  // we should also control hangup
  return 0;

#elif (CLIENT_OS == OS_WIN32)

  RASDIALPARAMS dialparameters;
  BOOL passwordretrieved;
  DWORD returnvalue;
  char buffer[260]; /* maximum registry key length */
  const char *connname = (const char *)(&connprofile[0]);

  dohangupcontrol = 0;           // whether we do HangupIfNeeded() or not

  if (*connname == 0)
    {
    HKEY hkey;
    if (RegOpenKey(HKEY_CURRENT_USER,"RemoteAccess",&hkey) == ERROR_SUCCESS)
      {
      DWORD valuetype = REG_SZ;
      DWORD valuesize = sizeof(buffer);
      if ( RegQueryValueEx(hkey, "InternetProfile", NULL, &valuetype,
               (unsigned char *)(&buffer[0]), &valuesize) == ERROR_SUCCESS )
        connname = &buffer[0];
      RegCloseKey(hkey);
      }
    if (*connname == 0)
      {
      const char **connlist = GetConnectionProfileList();
      if (connlist)
        {
        int j;
        for (j=0;*connname==0 && connlist[j];j++)
          connname = connlist[j];
        }
      if (*connname == 0)
        return -1;
      }
    }
  dialparameters.dwSize=sizeof(RASDIALPARAMS);
  strcpy(dialparameters.szEntryName,connname);
  strcpy(dialparameters.szPhoneNumber,"");
  strcpy(dialparameters.szCallbackNumber,"*");
  strcpy(dialparameters.szUserName,"");
  strcpy(dialparameters.szPassword,"");
  strcpy(dialparameters.szDomain,"*");

  returnvalue =
    RasGetEntryDialParams(NULL,&dialparameters,&passwordretrieved);

  if ( returnvalue==0 )
    {
    HRASCONN connhandle = NULL;

    //if (passwordretrieved != TRUE)
    //  LogScreen("Password could not be found, connection may fail.\n");

    LogScreen("Dialing '%s'...\n",dialparameters.szEntryName);
    returnvalue = RasDial(NULL,NULL,&dialparameters,NULL,NULL,&connhandle);

    if (returnvalue == 0)
      {
      hRasDialConnHandle = connhandle; //we only hangup this connection
      dohangupcontrol = 1;  // we also control hangup
      return 0;
      }
    else if (connhandle != NULL)
      {
      RasHangUp(connhandle);
      Sleep(3000);
      }
    }

  if (returnvalue == ERROR_CANNOT_FIND_PHONEBOOK_ENTRY)
    {
    LogScreen("Dial cancelled: Unable to find phonebook entry\n%s\n",
               dialparameters.szEntryName);
    }
  else if (returnvalue == ERROR_CANNOT_OPEN_PHONEBOOK)
    {
    LogScreen("Dial cancelled: Unable to open phonebook.\n");
    }
  else
    {
    if (RasGetErrorString(returnvalue,buffer,sizeof(buffer)) == 0)
      LogScreen("Dial cancelled: %s\n", buffer);
    else
      LogScreen("Dial cancelled: Unknown RAS error %ld\n", (long) returnvalue);
    }
  return -1;

#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_FREEBSD)

  dohangupcontrol = 0;
  if (connstartcmd[0] == 0)  /* we don't do dialup */
    {
    LogScreen("Dial Error. No dial-start-command specified.\n");
    dialwhenneeded = 0; //disable it!
    return -1;
    }
  if (system( connstartcmd ) == 127 /*exec error */)
    {                                        //pppstart of whatever
    LogScreen("Unable to exec '%s'\n%s\n", connstartcmd, strerror(errno));
    return -1;
    }
  int retry;
  for (retry = 0; retry < 30; retry++)  // 30s max to connect with a modem
    {
    sleep(1);
    if (IsConnected())
      {
      dohangupcontrol = 1;  // we should also control hangup
      return 0;
      }
    }
  if (connstopcmd[0] != 0)
    system( connstopcmd );
  return -1;

#else
  return -1; //failed

#endif
}

/* ---------------------------------------------------------- */

int Lurk::HangupIfNeeded(void) //returns 0 on success, -1 on fail
{
  int isconnected;

  if (!islurkstarted)      // Lurk can't be started, evidently
    return -1;
  if (!dialwhenneeded)     // We don't handle dialing
    return 0;

  isconnected = IsConnected();

  if (!dohangupcontrol) //if we didn't initiate, we shouldn't terminate
    return ((isconnected)?(-1):(0));

#if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)

  if (hWinsockInst)
    {
    FreeLibrary(hWinsockInst);
    hWinsockInst = NULL;
    }
  dohangupcontrol = 0;
  return 0;

#elif (CLIENT_OS == OS_WIN32)

  if (isconnected)
    {
    RASCONN rasconn;
    RASCONN *rasconnp = NULL;
    DWORD cb, whichconn, cConnections;

    cb = sizeof(rasconn);
    rasconn.dwSize = sizeof(RASCONN);
    rasconnp = &rasconn;
    if (RasEnumConnections( rasconnp, &cb, &cConnections) != 0)
      {
      cConnections = 0;
      if (cb > (DWORD)(sizeof(RASCONN)))
        {
        rasconnp = (RASCONN *) malloc( (int)cb );
        if (rasconnp)
          {
          rasconnp->dwSize = sizeof(RASCONN);
          if (RasEnumConnections( rasconnp, &cb, &cConnections) != 0)
            cConnections = 0;
          }
        }
      }

    for (whichconn = 0; whichconn < cConnections; whichconn++ )
      {
      HRASCONN hrasconn = rasconnp[whichconn].hrasconn;
      if (hrasconn == hRasDialConnHandle) // same conn as opened with rasdial?
        {
        RASCONNSTATUS rasconnstatus;
        rasconnstatus.dwSize = sizeof(RASCONNSTATUS);
        do{
          if (RasGetConnectStatus(hrasconn,&rasconnstatus) != 0)
            break;
          if (rasconnstatus.rasconnstate == RASCS_Connected)
            {
            if (RasHangUp( hrasconn ) != 0)
              break;
            Sleep(1000);
            }
          } while (rasconnstatus.rasconnstate == RASCS_Connected);
        }
      }

    if (rasconnp != NULL && rasconnp != &rasconn)
      free((void *)rasconnp );
    }

  hRasDialConnHandle = NULL;
  dohangupcontrol = 0;
  return 0;

#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_FREEBSD)

  if (isconnected)
    {
    int droppedconn = 0;
    if (connstopcmd[0] == 0) //what can we do?
      droppedconn = 1;
    else if (system( connstopcmd ) == 127 /* exec error */)
      LogScreen("Unable to exec '%s'\n%s\n", connstopcmd, strerror(errno));
    else
      {
      int retry = 0;
      while (((droppedconn = (!IsConnected())) == 0) && ((++retry)<10))
        {
        sleep(1);
        }
      }
    if (!droppedconn)
      return -1;
    }
  dohangupcontrol = 0;
  return 0;

#else
  return 0;
#endif
}

/* ---------------------------------------------------------- */
