/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * This module contains functions for both lurking and dial initiation/hangup.
 *
 * The workhorse function, InternalIsConnected(), is needed to support both.
 * - InternalIsConnected() may be called at any time and always returns yes/no.
 *   If the state could not be determined due to error or because lurk has
 *   not been initialized, it returns zero ('no').
 *
 * State functions are trivial and contain no OS specific routines.
 * - IsWatching() 
 *   returns zero if lurk has not been initialized.
 *   Otherwise, it returns the bitmask of enabled modes: 
 *   CONNECT_LURKONLY | CONNECT_LURK | CONNECT_DOD
 * - IsWatcherPassive()
 *   returns (IsWatching() & CONNECT_LURKONLY)
 * - IsConnected()
 *   IsConnected() is a verbose version of InternalIsConnected().
 *   return values are the same as InternalIsConnected()
 *
 * old state functions superceded by IsConnected()/IsWatch*():
 * - CheckIfConnectRequested() is identical to IsConnected() with the
 *   following exception: It always returns zero if neither CONNECT_LURKONLY 
 *   nor CONNECT_LURK are set. This implies that CheckIfConnectRequested()
 *   cannot be used if _only_ dial-on-demand is enabled.
 * - CheckForStatusChange() returns non-zero if the connect state at
 *   the time of the last call to CheckIfConnectRequested() is not the
 *   same as the current connect state.
 * In other words, CheckIfConnectRequested() begins a connection bracket
 * and CheckForStatusChange() closes it. Both functions are trivial and
 * require no OS support.
 *  
 * Public function used for dial initiation/hangup:
 * - DialIfNeeded(int override_lurkonly)
 *   does nothing if lurk is not initialized (returns -1) 
 *   does nothing if dial-on-demand is not enabled (returns 0)
 *   does nothing if already connected (returns 0)
 *   does nothing if CONNECT_LURKONLY and override_lurkonly is zero (return -1)
 *   otherwise it dials and returns zero on success, or -1 if a connection
 *             could not be established.
 * - HangupIfNeeded()
 *   does nothing if lurk is not initialized (returns -1)
 *   does nothing if dial-on-demand is not enabled (returns 0)
 *   does nothing if the connection wasn't previously 
 *        initiated with DialIfNeeded() (returns -1 if IsConnected(), else 0)
 *   otherwise it hangs up and returns zero. (no longer connected)
*/ 
const char *lurk_cpp(void) {
return "@(#)$Id: lurk.cpp,v 1.43.2.27 2000/11/02 18:29:46 cyp Exp $"; }

//#define TRACE

#include <stdio.h>
#include <string.h>
#include "cputypes.h"
#include "lurk.h"
#ifdef PROXYTYPE
#include "globals.h"
#define TRACE_OUT(x) /* nothing */
#else
#include "logstuff.h"
#include "util.h" //trace
#endif

Lurk dialup;        // publicly exported class instance.

/* ---------------------------------------------------------- */

int Lurk::Stop(void)
{
  TRACE_OUT((+1,"Lurk:Stop()\n"));
  islurkstarted = showedconnectcount = dohangupcontrol = 0;
  conf.lurkmode = conf.dialwhenneeded = 0;
  conf.connprofile[0] = conf.connifacemask[0] = 0;
  conf.connstartcmd[0] = conf.connstopcmd[0] =
  ifacemaskcopy[0] =  0;
  ifacestowatch[0] = (const char *)0;
  #ifdef LURK_MULTIDEV_TRACK
  memset(&conndevices[0],0,sizeof(conndevices));
  #else
  conndevice[0] = previous_conndevice[0] = 0;
  #endif
  TRACE_OUT((-1,"Lurk:Stop()\n"));
  return 0;
}

Lurk::Lurk()  { TRACE_OUT((+1,"Lurk:Lurk()\n")); Stop(); TRACE_OUT((-1,"Lurk:Lurk()\n")); }
Lurk::~Lurk() { TRACE_OUT((+1,"Lurk:~Lurk()\n")); Stop(); TRACE_OUT((-1,"Lurk:~Lurk()\n")); }

int Lurk::IsWatching(void)
{
  int rc;
  TRACE_OUT((+1,"Lurk::IsWatching() (islurkstarted?=%d)\n",islurkstarted));
  if (!islurkstarted)
  {
    TRACE_OUT((-1,"!islurkstarted. returning 0\n"));
    return 0;
  }
  rc = (conf.lurkmode & (CONNECT_LURKONLY|CONNECT_LURK));
  if (conf.dialwhenneeded)
    rc |= CONNECT_DOD;
  TRACE_OUT((-1,"IsWatching=>(CONNECT_LURKONLY|CONNECT_LURK|CONNECT_DOD)=>0x%x\n",rc));
  return rc;
}
int Lurk::IsWatcherPassive(void) { return (IsWatching() & CONNECT_LURKONLY); }

/* ---------------------------------------------------------- */

int Lurk::IsConnected(void)
{
  int rc = 0; /* assume not connected */
  TRACE_OUT((+1,"Lurk::IsConnected() (islurkstarted?=%d)\n",islurkstarted));
  if (islurkstarted)
  {
    TRACE_OUT((0,"beginning InternalIsConnected()\n"));
    rc = InternalIsConnected();
    TRACE_OUT((0,"end InternalIsConnected() => %d\n", rc ));
    #ifndef LURK_MULTIDEV_TRACK
    if (rc) //we are connected!
    {
      if (showedconnectcount == 0) /* there was no previous device */
        previous_conndevice[0] = '\0';
      if ((conndevice[0]==0) && (showedconnectcount == 0)) /* win16 and macos have no name */
      {
        LogScreen("Dialup link detected...\n"); // so this is the first time
        showedconnectcount = 1; // and there is only one device
      }
      else if (strcmp(conndevice,previous_conndevice)!=0) /*different device?*/
      {
	/* its AF_INET, not IP, of course */
        LogScreen("Tracking %sIP-link on '%s'...\n", 
                  ((showedconnectcount == 0)?(""):("new")), conndevice );
        strcpy(previous_conndevice,conndevice);
        showedconnectcount++;
      }
    }
    else if ( showedconnectcount > 0 ) /* no longer connected */
    {
      if (conndevice[0]==0) /* win16 and macos have no name */
      {
        LogScreen("(Dialup-)link was dropped%s.\n",
                 ((conf.lurkmode == CONNECT_LURKONLY)?
                 (" and will not be re-initiated"):("")));
      }
      else
      {
        LogScreen("Tracked termination of %s.\n",
          ((showedconnectcount > 1)?("all IP links"):("IP link")) );
      }
      showedconnectcount = 0;
      previous_conndevice[0] = '\0';
    }
    #endif /* LURK_MULTIDEV_TRACK */
  }  
  TRACE_OUT((-1,"Lurk::IsConnected()=>%d conncount=>%d\n",rc,showedconnectcount));
  return rc;
}

/* ---------------------------------------------------------- */

#if 0
int Lurk::CheckForStatusChange(void) //returns -1 if connection dropped
{
  TRACE_OUT((+1,"Lurk::CheckForStatusChange() (islurkstarted?=%d)\n",islurkstarted));
  if (!islurkstarted)
  {
    TRACE_OUT((-1,"!islurkstarted. returning 0\n"));
    return 0;
  }
  if ( (conf.lurkmode == 0) && (!conf.dialwhenneeded) )
  {
    TRACE_OUT((-1,"((conf.lurkmode == 0) && (!conf.dialwhenneeded)). returning 0\n"));
    return 0; // We're not lurking.
  }
  TRACE_OUT((0,"showedconnectcount? => %d\n", showedconnectcount));
  if (showedconnectcount > 0) /* we had shown a connected message */
  {
    TRACE_OUT((0,"beginning InternalIsConnected()\n"));
    if ( !InternalIsConnected() ) //if (Status() < oldlurkstatus)
    {
      TRACE_OUT((-1,"InternalIsConnected() => no. we got disconnected. returning -1\n"));
      return -1;  // we got disconnected!
    }
    TRACE_OUT((0,"InternalIsConnected() => yes. still connected.\n"));
  }
  TRACE_OUT((-1,"returning 0.\n"));
  return 0;
}
#endif

#if 0
int Lurk::CheckIfConnectRequested(void) //yes/no
{
  TRACE_OUT((+1,"Lurk::CheckIfConnectRequested() (islurkstarted?=%d)\n",islurkstarted));
  if (!islurkstarted)
  {
    TRACE_OUT((-1,"!islurkstarted. returning 0\n"));
    return 0; /* if this is changed, don't forget to change InternalIsConnected */
  }
  if ((conf.lurkmode & (CONNECT_LURKONLY|CONNECT_LURK)) == 0)
  {
    TRACE_OUT((-1,"(conf.lurkmode & (CONNECT_LURKONLY|CONNECT_LURK)=>0\n"));
    return 0; // We're not supposed to lurk!
  }
  return IsConnected();
}  
#endif


/* ================================================================== */
/* ************** OS SPECIFIC STUFF BEGINS HERE ********************* */
/* ================================================================== */


#if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || \
    (CLIENT_OS == OS_OPENBSD) || (CLIENT_OS == OS_NETBSD) || \
    (CLIENT_OS == OS_BSDOS) || \
    ((CLIENT_OS == OS_MACOSX) && !defined(__RHAPSODY__))
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>

#elif (CLIENT_OS == OS_MACOS)

#include <ctype.h>
#include <Gestalt.h>
#include <OpenTransportProviders.h>

EndpointRef fEndPoint = kOTInvalidEndpointRef;

#ifdef LURK_LISTENER
static pascal void __OTListener(void *context, OTEventCode code, OTResult result, void *cookie);
static int isonline = -1; /* 0=no, 1=yes, -1=don't know yet */
#endif

#elif (CLIENT_OS == OS_WIN16)

#include <windows.h>
#include <string.h>
static HINSTANCE hWinsockInst = NULL;

#elif (CLIENT_OS == OS_WIN32)

#include <windows.h>
#include <string.h>
#include <ras.h>
#include <raserror.h>

static HRASCONN hRasDialConnHandle = NULL; /* conn we opened with RasDial */

#elif (CLIENT_OS == OS_OS2)

#define INCL_DOSPROCESS
#include <os2.h>

#define TCPIPV4               //should also work with V3 though
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
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
#define _EMX_TCPIP
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

#elif (CLIENT_OS == OS_AMIGAOS)
#include "baseincs.h"
#include "sleepdef.h"
#include "triggers.h"
#include <net/if.h>
#include <fcntl.h>
#include <proto/miami.h>
#include "platforms/amiga/amiga.h"
#define _KERNEL
#include <sys/socket.h>
#undef _KERNEL
#include <proto/socket.h>
#include <sys/ioctl.h>
#define inet_ntoa(addr) Inet_NtoA(addr.s_addr)
#define ioctl(a,b,c) IoctlSocket(a,b,(char *)c)
#define close(a) CloseSocket(a)
#endif

/* ========================================================== */

int Lurk::GetCapabilityFlags(void)
{
  int what = 0;

  TRACE_OUT((+1,"Lurk::GetCapabilityFlags() (islurkstarted?=%d)\n",islurkstarted));

#if (CLIENT_OS == OS_WIN32)
  {
    static int caps = -1;
    if (caps == -1)
    {
      OSVERSIONINFO osver;
      osver.dwOSVersionInfoSize = sizeof(osver);
      if ( GetVersionEx( &osver ) != 0 )
      {
        int isok = 1; /* its always ok for win9x */
        OFSTRUCT ofstruct;
        ofstruct.cBytes = sizeof(ofstruct);
        #ifndef OF_SEARCH
        #define OF_SEARCH 0x0400
        #endif
        TRACE_OUT((+1,"Lurk::GetCapabilityFlags() ioctl check.\n"));
        if (osver.dwPlatformId == VER_PLATFORM_WIN32_NT)
        {
          isok = ((osver.dwMajorVersion > 4) ||
                  (osver.dwMajorVersion == 4 &&
                   strncmp(osver.szCSDVersion,"Service Pack ",13)==0 &&
                   atoi(&(osver.szCSDVersion[13])) >= 4));
          //http://support.microsoft.com/support/kb/articles/q181/5/20.asp
          //http://support.microsoft.com/support/kb/articles/q170/6/42.asp
        }
        if (isok && OpenFile( "WS2_32.DLL", &ofstruct, OF_EXIST|OF_SEARCH)!=HFILE_ERROR)
          what |= (CONNECT_LURK|CONNECT_LURKONLY|CONNECT_IFACEMASK);
        TRACE_OUT((-1,"ioctl check end. caps=0x%08x\n",what));
      }
      caps = what;
    }
    what |= caps;
  }
  TRACE_OUT((+1,"Lurk::GetCapabilityFlags() ras check.\n"));
  //if ( RasHangUp( (HRASCONN)-1 ) == ERROR_INVALID_HANDLE )
  //  what |= (CONNECT_LURK|CONNECT_LURKONLY|CONNECT_DODBYPROFILE);
  if (GetConnectionProfileList() != NULL)
    what |= (CONNECT_LURK|CONNECT_LURKONLY|CONNECT_DODBYPROFILE);
  TRACE_OUT((-1,"ras checked. caps=0x%08x\n",what));
#elif (CLIENT_OS == OS_WIN16)
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
#elif (CLIENT_OS == OS_MACOS)
  {
    static int caps = -1;
    if (caps == -1)
    {
      long response;
      //OpenTransport/Remote Access PPP must be present
      Gestalt(gestaltOpenTptRemoteAccess, &response);
      if (response & (1 << gestaltOpenTptPPPPresent))
      {
        InitOpenTransport();
        fEndPoint = OTOpenEndpoint(OTCreateConfiguration(kPPPControlName),0, nil, &response);  
        if ((response == kOTNoError) && (fEndPoint != kOTInvalidEndpointRef))
        {
          int canlurk = 1; 
          #ifdef LURK_LISTENER
          canlurk = 0;
          if( fEndPoint->InstallNotifier((OTNotifyProcPtr)__OTListener,this) == kOTNoError) 
          {
            OTSetAsynchronous(fEndPoint);
            if(OTIoctl(fEndPoint, I_OTGetMiscellaneousEvents, (void*)1) == kOTNoError )
            {
              canlurk = 1; /* OT Listener is now installed */
            }
          }
          #endif
          if (canlurk)
             what = (CONNECT_LURK | CONNECT_LURKONLY );  
        }
      }
      caps = what;
    }
    what |= caps;
  }
#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || \
      (CLIENT_OS == OS_OPENBSD) || (CLIENT_OS == OS_NETBSD) || \
      (CLIENT_OS == OS_BSDOS) || (CLIENT_OS == OS_OS2) || \
      ((CLIENT_OS == OS_MACOSX) && !defined(__RHAPSODY__))
  what = (CONNECT_LURK | CONNECT_LURKONLY | CONNECT_DODBYSCRIPT | CONNECT_IFACEMASK);
#elif (CLIENT_OS == OS_AMIGAOS)
  what = (CONNECT_LURK | CONNECT_LURKONLY | CONNECT_DODBYPROFILE | CONNECT_IFACEMASK);
#endif

  TRACE_OUT((-1,"Lurk::GetCapabilityFlags() [1]. what=0x%08x\n",(u32)what));

  return what;
}

/* ---------------------------------------------------------- */

const char **Lurk::GetConnectionProfileList(void)
{
  #if (CLIENT_OS == OS_WIN32)
    static const char *firstentry = ""; //the first entry is blank, ie use default
    static RASENTRYNAME rasentries[10];
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
  #elif (CLIENT_OS == OS_AMIGAOS)
    static const char *firstentry = ""; //the first entry is blank, ie use default
    static char namestorage[8][IFNAMSIZ];
    static const char *configptrs[10];
    struct Library *MiamiBase;
    configptrs[0] = NULL;
    if ((MiamiBase = OpenLibrary((unsigned char *)"miami.library",11)))
    {
      struct if_nameindex *name,*nameindex;
      if ((nameindex = if_nameindex())) {
        int cnt = 0;
        name = nameindex;
        configptrs[cnt++] = firstentry;
        while (!(name->if_index == 0 && name->if_name == NULL) && cnt < 8) {
          if (strncmp(name->if_name,"lo",2) != 0 && strncmp(name->if_name,"mi",2) != 0) {
            strcpy(namestorage[cnt-1],name->if_name);
            configptrs[cnt] = (const char *)&namestorage[cnt-1];
            cnt++;
          }
          name++;
        }
        configptrs[cnt] = NULL;
        if_freenameindex(nameindex);
      }
      CloseLibrary(MiamiBase);
    }
    return configptrs;
  #endif
  return ((const char **)0);
}

/* ---------------------------------------------------------- */

int Lurk::Start(int nonetworking,struct dialup_conf *params)
{                             // Initializes Lurk Mode. returns 0 on success.
  Stop(); //zap variables/state

  TRACE_OUT((+1,"Lurk::Start()\n"));
  if (!nonetworking) //no networking equals 'everything as default'.
  {
    int flags = GetCapabilityFlags();

    conf.lurkmode = conf.dialwhenneeded = 0;
    if (params->lurkmode || params->dialwhenneeded)
    {
      int lurkmode = params->lurkmode;
      int dialwhenneeded = params->dialwhenneeded;
      if (lurkmode != CONNECT_LURKONLY && lurkmode != CONNECT_LURK)
        lurkmode = 0;    /* can only be one or the other */
      if (lurkmode && (flags & (CONNECT_LURK|CONNECT_LURKONLY))==0)
      {              //only happens if user used -lurk on the command line
        lurkmode = 0;
        #if (CLIENT_OS == OS_WIN32)
        //LogScreen( "Dial-up must be installed for lurk/lurkonly/dialing\n" );
        dialwhenneeded = 0; //if we can't support lurk, we can't support dod either
        #elif (CLIENT_OS == OS_WIN16)
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
        LogScreen( "Dial-up-Networking must be installed for demand dialing\n" );
        #elif (CLIENT_OS == OS_WIN16)
        LogScreen("Demand dialing is only supported with Trumpet Winsock.\n");
        #else
        LogScreen("Demand dialing is currently unsupported.\n");
        #endif
      }
      conf.lurkmode = lurkmode;
      conf.dialwhenneeded = dialwhenneeded;

      TRACE_OUT((0,"lurkmode=%d dialwhenneeded=%d\n",conf.lurkmode,conf.dialwhenneeded));
    }

    conf.connprofile[0] = 0;
    if (conf.dialwhenneeded && params->connprofile[0]!=0)
    {
      int n=0, pos=0;
      while (params->connprofile[pos] && isspace(params->connprofile[pos]))
        pos++;
      while (params->connprofile[pos])
        conf.connprofile[n++] = params->connprofile[pos++];
      while (n>0 && isspace(conf.connprofile[n-1]))
        n--;
      conf.connprofile[n]=0;
    }

    mask_include_all = mask_default_only = 0;
    conf.connifacemask[0] = ifacemaskcopy[0]=0; 
    ifacestowatch[0] = (const char *)0;
    if ((conf.lurkmode || conf.dialwhenneeded) && params->connifacemask[0])
    {
      int n=0, pos=0;
      while (params->connifacemask[pos] && isspace(params->connifacemask[pos]))
        pos++;
      while (params->connifacemask[pos])
        conf.connifacemask[n++] = params->connifacemask[pos++];
      while (n>0 && isspace(conf.connifacemask[n-1]))
        n--;
      conf.connifacemask[n]=0;
    }
    if ((flags & CONNECT_IFACEMASK)==0)
      mask_include_all = 1;
    else if (conf.connifacemask[0] == '\0')
      mask_default_only = 1;
    else if (conf.connifacemask[0]=='*' && conf.connifacemask[1]=='\0')
      mask_include_all = 1;
    else
    {
      // Parse connifacemask[] and store each iface name in *ifacestowatch[]
      unsigned int ptrindex = 0, stindex = 0;
      char *c = &(conf.connifacemask[0]);
      do
      {
        while (*c && (isspace(*c) || *c==':'))
          c++;
        if (*c)
        {
          char *p = &ifacemaskcopy[stindex];
          while (*c && !isspace(*c) && *c!=':')
            ifacemaskcopy[stindex++] = *c++;
          ifacemaskcopy[stindex++]='\0';
          if (p[0] == '*' && p[1]=='\0')
          {
            ptrindex = 0;
            mask_include_all = 1;
            break;
          }
          #if (CLIENT_OS == OS_OS2)  //convert 'eth*' names to 'lan*'
          if (*p=='e' && p[1]=='t' && p[2]=='h' && (isdigit(p[3]) || p[3]=='*'))
          {*p='l'; p[1]='a'; p[2]='n'; }
          #elif (CLIENT_OS == OS_WIN32)
          if (*p=='s' && p[1]=='l' && (isdigit(p[2]) || p[2]=='*'))
          {                          //convert 'sl*' names to 'ppp*'
            char buf[sizeof(ifacemaskcopy)];
            strcpy(buf,p+2);strcat(strcpy(p,"ppp"),buf);
            stindex++;
          }
          #endif
          ifacestowatch[ptrindex++] = (const char *)p;
          if (ptrindex == ((sizeof(ifacestowatch)/sizeof(ifacestowatch[0]))-1))
            break;
        }
      } while (*c);
      if (ptrindex == 0 && !mask_include_all) //nothing in list
        mask_default_only = 1;
      ifacestowatch[ptrindex] = (const char *)0;

      #ifdef TRACE
      TRACE_OUT((0,"mask flags: include_all=%d, defaults_only=%d\niface list:\n",
                   mask_include_all, mask_default_only ));
      for (ptrindex=0;ifacestowatch[ptrindex];ptrindex++)
        TRACE_OUT((0,"  %d) '%s'\n",ptrindex+1,ifacestowatch[ptrindex]));
      #endif
    }

    conf.connstartcmd[0] = 0;
    if (conf.dialwhenneeded && params->connstartcmd[0])
    {
      int n=0, pos=0;
      while (params->connstartcmd[pos] && isspace(params->connstartcmd[pos]))
        pos++;
      while (params->connstartcmd[pos])
        conf.connstartcmd[n++] = params->connstartcmd[pos++];
      while (n>0 && isspace(conf.connstartcmd[n-1]))
        n--;
      conf.connstartcmd[n]=0;
    }

    conf.connstopcmd[0] = 0;
    if (conf.dialwhenneeded && params->connstopcmd[0])
    {
      int n=0, pos=0;
      while (params->connstopcmd[pos] && isspace(params->connstopcmd[pos]))
        pos++;
      while (params->connstopcmd[pos])
        conf.connstopcmd[n++] = params->connstopcmd[pos++];
      while (n>0 && isspace(conf.connstopcmd[n-1]))
        n--;
      conf.connstopcmd[n]=0;
    }
  }
  islurkstarted=1;

  TRACE_OUT((-1,"Lurk::Start() (islurkstarted?=%d)\n",islurkstarted));
  return 0;
}

/* ---------------------------------------------------------- */

#if (!((CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_OS2) && !defined(__EMX__) || \
      (CLIENT_OS == OS_MACOS)))
/* needed by all except win16 and non-emx-os/2 and macos*/
static int __MatchMask( const char *ifrname, int mask_include_all,
                       int mask_default_only, const char *ifacestowatch[] )
{
  int ismatched = 0;
  char wildmask[32+4]; //should be sizeof((struct ifreq.ifr_name)+4
  const char *matchedname = "*";

  TRACE_OUT((+1,"__MatchMask(ifrname='%s',mask_include_all=%d,"
                "mask_default_only=%d,ifacestowatch=%p)\n",
                ifrname,mask_include_all,mask_default_only,ifacestowatch));

  if (mask_include_all) //mask was "*"
    ismatched = 1;
  else
  {
    int maskpos=0;
    //create a wildcard version of ifrname (eg, "eth1" becomes "eth*")
    strncpy(wildmask,ifrname,sizeof(wildmask));
    wildmask[sizeof(wildmask)-1]=0;
    while (maskpos < ((int)(sizeof(wildmask)-2)) &&
       wildmask[maskpos] && !isdigit(wildmask[maskpos]))
      maskpos++;
    wildmask[maskpos++]='*';
    wildmask[maskpos]='\0';
    if (mask_default_only) //user didn't specify a mask, so default is any
    {                      //dialup device
      ismatched = (strcmp(wildmask,"ppp*")==0
      #if (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_OPENBSD) || \
        (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_BSDOS) || \
        ((CLIENT_OS == OS_MACOSX) && !defined(__RHAPSODY__))
      || strcmp(wildmask,"dun*")==0 
      || strcmp(wildmask,"tun*")==0
      #elif (CLIENT_OS == OS_AMIGAOS)
      || strcmp(wildmask,"mi*")==0  // required for Miami (not MiamiDx or Genesis)
      #endif
      || strcmp(wildmask,"sl*")==0);
      matchedname = ((const char *)(&wildmask[0]));
      if (!ismatched)
        matchedname = ((const char *)0);  
    }
    else //user specified a mask. must match ifrname, or (if the mask is
    {    //a wildcard version) the wildcard version of ifrname
      maskpos = 0;
      do { 
        matchedname = ifacestowatch[maskpos++];
        if (matchedname)
          ismatched = (strcmp( matchedname, ifrname )==0 ||
                       strcmp( matchedname, wildmask )==0);
      } while (!ismatched && matchedname);
    }
  }
  TRACE_OUT((-1,"__Maskmatch: matched?=%s ifrname=='%s' matchname=='%s'\n",
    (ismatched?"yes":"no"), ifrname, matchedname?matchedname:"(not found)" ));
  return ismatched;
}
#endif


/* ---------------------------------------------------------- */

#ifdef LURK_MULTIDEV_TRACK
static int __insdel_devname(const char *devname, int isup,
                            char *conndevices, unsigned int conndevices_sz)
{
  unsigned int len;
  char *p = conndevices;
  if (!devname)
  {            
    while (*p)
    {
      if (*p == '|' && isup)
      {
        *p++ = '?';
      }  
      else if (*p == '?' && !isup)
      {
        char *q = p+1;
        while (*q && *q != '|' && *q != '?')
          q++;
        *p = *q;
        *q = '\0';
        LogScreen("Detected IP-link drop for '%s'.\n", p+1 );
	/* its AF_INET, not IP, of course */
        if (*p)
          strcpy(p+1,q+1);
      }  
      else 
        p++;
    }
    return 0;
  }  
  len = strlen(devname)+1;
  while ((*p == '?' || *p == '|') && 
        ((p+len) < &conndevices[conndevices_sz]))
  {
    char *q = p+1;
    while (*q && *q != '?' && *q != '|')
      q++;
    if ((&p[len] == q) && memcmp(p+1,devname,len-1)==0)
    {
      if (isup)
        *p = '|';
      else
      {
        strcpy( p, q ); 
        LogScreen("Detected IP-link drop for '%s'.\n", devname );
	/* its AF_INET, not IP, of course */
      }
      return 0;
    }
    p = q;
  }
  if (!isup)
    return 0;
  if ((strlen(conndevices)+len) >= (conndevices_sz-2))
    return -1; /* ENOSPC */
  LogScreen("Detected IP-link on '%s'...\n", devname );
  /* its AF_INET, not IP, of course */
  strcat(strcat(conndevices,"|"),devname);
  return 0;
}  
#endif


/* ---------------------------------------------------------- */

#ifdef LURK_LISTENER /* asynchronous callback */
static pascal void __OTListener(void *context, OTEventCode code, OTResult result, void *cookie)
{
  //printf("default: %X\n",code);
  switch (code)
  {
    case kPPPIPCPDownEvent:
         isonline=0;
         break;
    case kPPPIPCPUpEvent:
         if (result == kOTNoError)
           isonline=1;
         break;
    default:
         break;
  }
  return;
}
#endif

/* ---------------------------------------------------------- */

int Lurk::InternalIsConnected(void) //must always returns a valid yes/no
{
  #ifndef LURK_MULTIDEV_TRACK
  conndevice[0]=0;
  #endif

  TRACE_OUT((+1,"Lurk::InternalIsConnected() (islurkstarted=%d)\n",islurkstarted));

  if (!islurkstarted)
  {
    TRACE_OUT((-1,"!islurkstarted. returning 0\n"));
    return 0;/* if this is changed, don't forget to change IsConnected() */
  }
  if (!conf.lurkmode && !conf.dialwhenneeded)
  {
    TRACE_OUT((-1,"(!conf.lurkmode && !conf.dialwhenneeded) returning 1\n"));
    return 1;
  }

#if (CLIENT_OS == OS_WIN16)
  if ( GetModuleHandle("WINSOCK") )
  {
    TRACE_OUT((-1,"winsock.dll is loaded. returning 1\n"));
    return 1;
  }
#elif (CLIENT_OS == OS_WIN32)
  if ((GetCapabilityFlags() & CONNECT_IFACEMASK) != 0 /* have working WS2_32 */
   && (!mask_default_only || (GetCapabilityFlags() & CONNECT_DODBYPROFILE)==0))
  {
    TRACE_OUT((+1,"ioctl InternalIsConnected()\n"));
    HINSTANCE ws2lib = LoadLibrary( "WS2_32.DLL" );
    int upcount = 0;

    #ifdef LURK_MULTIDEV_TRACK
    __insdel_devname(NULL,1,conndevices,sizeof(conndevices)); /* init tracking */
    #endif
    if (ws2lib)
    {
      FARPROC __WSAStartup  = GetProcAddress( ws2lib, "WSAStartup" );
      FARPROC __WSASocket   = GetProcAddress( ws2lib, "WSASocketA" );
      FARPROC __WSAIoctl    = GetProcAddress( ws2lib, "WSAIoctl" );
      FARPROC __WSACleanup  = GetProcAddress( ws2lib, "WSACleanup" );
      FARPROC __closesocket = GetProcAddress( ws2lib, "closesocket" );
      if (__WSAStartup && __WSASocket && __WSAIoctl &&
          __WSACleanup && __closesocket)
      {
        WSADATA winsockData;
        TRACE_OUT((0,"ioctl InternalIsConnected() [2]\n"));

        if (( (*((int (PASCAL FAR *)(WORD, LPWSADATA))(__WSAStartup)))
                                         (MAKEWORD(2,2), &winsockData)) == 0)
        {
          #define LPWSAPROTOCOL_INFO void *
          #define GROUP unsigned int
          SOCKET s =
          (*((int (PASCAL FAR *)(int,int,int,LPWSAPROTOCOL_INFO,GROUP,DWORD))
            (__WSASocket)))(AF_INET, SOCK_DGRAM, IPPROTO_UDP, NULL, 0, 0);

          TRACE_OUT((0,"ioctl InternalIsConnected() [3]\n"));

          if (s != INVALID_SOCKET)
          {
            #define SIO_GET_INTERFACE_LIST _IOR('t', 127, u_long) // <TBD>
            #define LPWSAOVERLAPPED void *
            #define LPWSAOVERLAPPED_COMPLETION_ROUTINE void *
            #define IFF_UP           0x00000001 /* Interface is up */
            #define IFF_BROADCAST    0x00000002 /* Broadcast is  supported */
            #define IFF_LOOPBACK     0x00000004 /* this is loopback interface */
            #define IFF_POINTTOPOINT 0x00000008 /* this is p2p interface*/
            #define IFF_MULTICAST    0x00000010 /* multicast is supported */
            DWORD bytesReturned;

            TRACE_OUT((0,"ioctl InternalIsConnected() [4]\n"));

            #pragma pack(1)
            struct if_info_v4 /* 4+(3*16) */
            {
              u_long   iiFlags;
              struct { short   sin_family; /* AF_INET */
                       u_short sin_port;
                       u_long  sin_addr;
                       char    sin_zero[8];
                     } iiAddress, iiBroadcastaddress, iiNetmask;
            };
            struct if_info_v6  /* 4+(3*24) */
            {
              u_long   iiFlags;
              struct { short   sin6_family; /* AF_INET6 */
                       u_short sin6_port;
                       u_long  sin6_flowinfo;
                       u_char  sin6_addr[16];
                     } iiAddress, iiBroadcastaddress, iiNetmask;
            };
            #pragma pack()
            char if_info[sizeof(struct if_info_v6)*10]; // Assume no more than 10 IP interfaces
            memset((void *)(&if_info[0]),0,sizeof(if_info));

            //don't use INTERFACE_INFO format due to NT4SP<4 not grokking IPV6
            //IMO, Thats a serious bug in the API that returns AF_INET6 data
            //for an AF_INET[4] socket.

            int wsError = (*((int (PASCAL FAR *)(
                SOCKET,DWORD,LPVOID,DWORD,LPVOID,DWORD,LPDWORD,
                LPWSAOVERLAPPED,LPWSAOVERLAPPED_COMPLETION_ROUTINE))
                (__WSAIoctl)))(s, SIO_GET_INTERFACE_LIST, NULL, 0,
                (&(if_info[0])), sizeof(if_info), &bytesReturned, NULL, NULL);
            if (wsError == 0)
            {
              unsigned int i, pppdev = 0, ethdev = 0, slipdev = 0;
              unsigned int stepsize = sizeof(struct if_info_v6);
              const char *ifp = if_info;
              if ((bytesReturned%sizeof(struct if_info_v6))!=0)
                stepsize = sizeof(struct if_info_v4);
              TRACE_OUT((0,"stage5 %u: v4:%u v6:%u struct family=IPv%d\n",bytesReturned,sizeof(struct if_info_v4),sizeof(struct if_info_v6),(stepsize==sizeof(struct if_info_v6)?(6):(4))));

              for (i=0; i<bytesReturned; i+=stepsize)
              {
                int isup = 0;
                u_long if_flags = ((struct if_info_v4 *)(ifp))->iiFlags;
                u_long if_addr  = ((struct if_info_v4 *)(ifp))->iiAddress.sin_addr;
                /* if_addr is always (?) an AF_INET[4] addr because thats what our socket is */
                ifp+=stepsize;

                TRACE_OUT((0,"stage6: adapter: %u flags: 0x%08x %s\n", i/stepsize, if_flags,inet_ntoa(*((struct in_addr *)&if_addr)) ));

                if ((if_flags & IFF_LOOPBACK)==0 && if_addr != 0x0100007ful)
                {
                  char seqname[20], devname[20];
                  wsprintf(seqname,"lan%u",pppdev+ethdev+slipdev);
                  if (if_addr==0 && (if_flags & IFF_POINTTOPOINT)==0)
                  {
                    //Dial-Up adapters are never down. They appear as normal
                    //ether adapters, but have an zero address when not up.
                    if_flags|=IFF_POINTTOPOINT;
                    if_flags&=~IFF_UP;
                  }
                  if ((if_flags & IFF_POINTTOPOINT)!=IFF_POINTTOPOINT)
                    wsprintf(devname,"eth%u",ethdev++);
                  //else if ((if_flags & (IFF_BROADCAST|IFF_MULTICAST))==0)
                  //  wsprintf(devname,"sl%u",slipdev++);
                  else
                    wsprintf(devname,"ppp%u",pppdev++);
                  TRACE_OUT((0,"stage7: not lo. up?=%s, seqname=%s devname=%s\n", ((if_flags & IFF_UP)?"yes":"no"), seqname, devname ));
                  if ((if_flags & IFF_UP)==IFF_UP &&
                     (__MatchMask(devname,mask_include_all,
                         mask_default_only, &ifacestowatch[0] ) ||
                      __MatchMask(seqname,mask_include_all,
                         mask_default_only, &ifacestowatch[0] )))
                  {
                    upcount++;
                    isup = 1;
                    TRACE_OUT((0,"stage8: mask matched. name=%s\n", devname ));
                  }
                  strcat( devname, "/" );
                  strcat( devname, seqname );
                  if (isup)
                  {
                    #ifdef LURK_MULTIDEV_TRACK
                    if (__insdel_devname(devname,1,conndevices,sizeof(conndevices))!=0)
                      break; /* table is full */
                    #else
                    strncpy( conndevice, devname, sizeof(conndevice) );
                    conndevice[sizeof(conndevice)-1] = 0;
                    break;
                    #endif
                  }
                  #ifdef LURK_MULTIDEV_TRACK
                  else
                    __insdel_devname(devname,0,conndevices,sizeof(conndevices));
                  #endif
                }
              }
            }
            (*((int (PASCAL FAR *)(SOCKET))(__closesocket)))(s);
          }
          (*((int (PASCAL FAR *)(void))(__WSACleanup)))();
        }
      }
      FreeLibrary(ws2lib);
    }
    #ifdef LURK_MULTIDEV_TRACK
    __insdel_devname(NULL,0,conndevices,sizeof(conndevices)); /* stop tracking */
    #endif
    TRACE_OUT((-1,"ioctl InternalIsConnected() =>%d\n",upcount));

    if (upcount)
    {
      TRACE_OUT((-1,"Lurk::InternalIsConnected() => 1\n"));
      return 1;
    }
  }
  if ((GetCapabilityFlags() & CONNECT_DODBYPROFILE)!=0) /* have ras */
  {
    RASCONN rasconn;
    RASCONN *rasconnp = NULL;
    DWORD cb, whichconn, cConnections;
    int foundconn = 0;

    TRACE_OUT((+1,"Lurk::InternalIsConnected() [BYPROFILE]\n"));
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
    TRACE_OUT((0,"number of profiles: %d\n",cConnections));

    #ifdef LURK_MULTIDEV_TRACK
    __insdel_devname(NULL,1,conndevices,sizeof(conndevices)); /* begin tracking */
    #endif
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
          #ifdef LURK_MULTIDEV_TRACK
          if (__insdel_devname(connname,1,conndevices,sizeof(conndevices)))
            break; /* table is full */
          #else
          strncpy( conn, connname, sizeof(conndevice) );
          conndevice[sizeof(conndevice)-1]=0;
          break;
          #endif
        }
        #ifdef LURK_MULTIDEV_TRACK
        else
          __insdel_devname(connname,0,conndevices,sizeof(conndevices));
        #endif
      }
    }
    #ifdef LURK_MULTIDEV_TRACK
    __insdel_devname(NULL,0,conndevices,sizeof(conndevices)); /* end tracking */
    #endif

    if (rasconnp != NULL && rasconnp != &rasconn)
      free((void *)rasconnp );

    TRACE_OUT((-1,"found conn?: %d\n",foundconn));
    if (foundconn)
    {
      TRACE_OUT((-1,"Lurk::InternalIsConnected() => 1\n"));
      return 1;
    }
  }
#elif (CLIENT_OS == OS_OS2) && !defined(__EMX__)
   int s, i, rc, j, foundif = 0;
   struct ifmib MyIFMib = {0};
   struct ifact MyIFNet = {0};

   MyIFNet.ifNumber = 0;
   s = socket(PF_INET, SOCK_STREAM, 0);
   if (s >= 0)
   {
     /* get active interfaces list */
     i =  ioctl(s, SIOSTATAT, (char *)&MyIFNet, sizeof(MyIFNet));
     if ( i >= 0 )
     {
       i = ioctl(s, SIOSTATIF, (char *)&MyIFMib, sizeof(MyIFMib));
       if ( i < 0)
         MyIFNet.ifNumber = 0;
     }
     #ifdef LURK_MULTIDEV_TRACK
     __insdel_devname(NULL,1,conndevices,sizeof(conndevices)); /* begin tracking */
     #endif
     for (i = 0; i < MyIFNet.ifNumber; i++)
     {
       j = MyIFNet.iftable[i].ifIndex;      /* j is now the index into the stats table for this i/f */
       if (mask_default_only == 0 || MyIFMib.iftable[j].ifType != HT_ETHER)   /* i/f is not ethernet */
       {
         if (MyIFMib.iftable[j].ifType != HT_PPP)  /* i/f is not loopback (yes I know it says PPP) */
         {
           if (MyIFNet.iftable[i].ifa_addr != 0x0100007f)  /* same thing for TCPIP < 4.1 */
           {
             char devname[64];
             int ismatched = 0;
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
             strncpy( devname, MyIFReq.ifr_name, sizeof(devname) );
             devname[sizeof(devname)-1]=0;
             if (ioctl(s, SIOCGIFFLAGS, (char*)&MyIFReq, sizeof(MyIFReq))==0)
             {
               if ((MyIFReq.ifr_flags & IFF_UP) != 0)
               {
                 if (mask_include_all)
                   ismatched = 1;
                 else if (mask_default_only)
                   ismatched = (MyIFMib.iftable[j].ifType != HT_ETHER);
                 else
                 {
                   int maskpos;
                   for (maskpos=0;!ismatched && ifacestowatch[maskpos];maskpos++)
                     ismatched= (stricmp(ifacestowatch[maskpos],MyIFReq.ifr_name)==0
                     || (*wildmask && stricmp(ifacestowatch[maskpos],wildmask)==0));
                 }
               } // if ((MyIFReq.ifr_flags & IFF_UP) != 0)
             } // ioctl(s, SIOCGIFFLAGS ) == 0  
             if (ismatched)
             {
               foundif = i+1; // Report online if SLIP or PPP detected
               #ifdef LURK_MULTIDEV_TRACK
               if (__insdel_devname(devname,1,conndevices,sizeof(conndevices))!=0)
                 break; /* table is full */
               #else
               strncpy( conndevice, devname, sizeof(conndevice) );
               conndevice[sizeof(conndevice)-1]=0;
               #endif
             }
             #ifdef LURK_MULTIDEV_TRACK
             else
               __insdel_devname(devname,0,conndevices,sizeof(conndevices));
             #endif
           } // != 0x0100007f
         } // != HT_PPP (loopback actually)
       } // != HT_ETHER
     } //for ...
     #ifdef LURK_MULTIDEV_TRACK
     __insdel_devname(NULL,0,conndevices,sizeof(conndevices)); /* end tracking */
     #endif
     soclose(s);
   }
   if (foundif)
   {
     TRACE_OUT((-1,"Lurk::InternalIsConnected() => 1\n"));
     return 1;
   }

#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || \
      (CLIENT_OS == OS_OPENBSD) || (CLIENT_OS == OS_NETBSD) || \
      (CLIENT_OS == OS_BSDOS) || (CLIENT_OS == OS_AMIGAOS) || \
      ((CLIENT_OS == OS_MACOSX) && !defined(__RHAPSODY__)) || \
      ((CLIENT_OS == OS_OS2) && defined(__EMX__))
   struct ifconf ifc;
   struct ifreq *ifr;
   int n, foundif = 0;
   char *p;

   #if (CLIENT_OS == OS_AMIGAOS)
   // not being able to access the socket lib means tcp/ip is unavailable,
   // implying that we must actually be offline
   if (!amigaOpenSocketLib()) return 0;
   #endif

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
       #ifdef LURK_MULTIDEV_TRACK
       __insdel_devname(NULL,1,conndevices,sizeof(conndevices)); /* begin tracking */
       #endif
       #if (CLIENT_OS == OS_LINUX) || ((CLIENT_OS == OS_OS2) && defined(__EMX__))
       for (n = 0, ifr = ifc.ifc_req; n < ifc.ifc_len; n += sizeof(struct ifreq), ifr++)
       {
         if (__MatchMask(ifr->ifr_name,mask_include_all,
                         mask_default_only, &ifacestowatch[0] ))
         {
           char devname[64];
           strncpy(devname,ifr->ifr_name,sizeof(devname));
           devname[sizeof(devname)-1] = '\0';

           ioctl (fd, SIOCGIFFLAGS, ifr); // get iface flags
           #ifndef __EMX__
           if ((ifr->ifr_flags & (IFF_UP | IFF_RUNNING | IFF_LOOPBACK))
               == (IFF_UP | IFF_RUNNING))
           #else
           if (((ifr->ifr_flags & (IFF_UP | IFF_POINTOPOINT))            == (IFF_UP | IFF_POINTOPOINT)) || \
               ((ifr->ifr_flags & (IFF_UP | IFF_RUNNING | IFF_LOOPBACK)) ==     (IFF_UP | IFF_RUNNING)))
           #endif
           {
             foundif = (n / sizeof(struct ifreq)) + 1;
             #ifdef LURK_MULTIDEV_TRACK
             if (__insdel_devname(devname,1,conndevices,sizeof(conndevices))!=0)
               break; /* table is full */
             #else
             strncpy( conndevice, devname, sizeof(conndevice) );
             conndevice[sizeof(conndevice)-1]=0;
             break;
             #endif
           }  
           #ifdef LURK_MULTIDEV_TRACK
           else
             __insdel_devname(devname,0,conndevices,sizeof(conndevices));
           #endif  
         } /* if __MatchMask */
       } /* for (n = 0, ... ) */
       #elif (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_OPENBSD) || \
             (CLIENT_OS == OS_BSDOS) || (CLIENT_OS == OS_NETBSD) || \
             ((CLIENT_OS == OS_MACOSX) && !defined(__RHAPSODY__)) || \
             (CLIENT_OS == OS_AMIGAOS)
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
         int sa_len = sa->sa_len;
         if (sa->sa_family == AF_INET)  // filter-out anything other than AF_INET
         {                            // (in fact this filter-out AF_LINK)
           if (__MatchMask(ifr->ifr_name,mask_include_all,
                           mask_default_only, &ifacestowatch[0] ))
           {
             int isup = 0;
             char devname[64];
             strncpy(devname,ifr->ifr_name,sizeof(devname));
             devname[sizeof(devname)-1] = '\0';
             ioctl (fd, SIOCGIFFLAGS, ifr); // get iface flags
             #if (CLIENT_OS == OS_AMIGAOS)
             if (strcmp(ifr->ifr_name,"mi0") == 0) // IFF_UP is always set for mi0
             {
               struct Library *MiamiBase;
               if ((MiamiBase = OpenLibrary((unsigned char *)"miami.library",11)))
               {
                 isup = MiamiIsOnline("mi0");
                 CloseLibrary(MiamiBase);
               }
             }
             else if ((ifr->ifr_flags & (IFF_UP | IFF_LOOPBACK)) == IFF_UP)
             #else
             if ((ifr->ifr_flags & (IFF_UP | IFF_RUNNING | IFF_LOOPBACK))
                 == (IFF_UP | IFF_RUNNING))
             #endif
             {
               isup = 1;
             }
             if (isup)
             {
               foundif = (n / sizeof(struct ifreq)) + 1;
               #ifdef LURK_MULTIDEV_TRACK
               if (__insdel_devname(devname,1,conndevices,sizeof(conndevices))!=0)
                 break; /* table is full */
               #else
               strncpy( conndevice, devname, sizeof(conndevice) );
               conndevice[sizeof(conndevice)-1]=0;
               break;
               #endif
             }  
             #ifdef LURK_MULTIDEV_TRACK
             else
               __insdel_devname(devname,0,conndevices,sizeof(conndevices));
             #endif  
           } /* if matchmask */
         } /* if (sa->sa_family == AF_INET) */
         // calculate the length of this entry and jump to the next
         int ifrsize = IFNAMSIZ + sa_len;
         ifr = (struct ifreq *)((caddr_t)ifr + ifrsize);
         n -= ifrsize;
       } /* for (n = 0, ... ) */
       #else
       #error "What's up Doc ?"
       #endif
       #ifdef LURK_MULTIDEV_TRACK
       __insdel_devname(NULL,0,conndevices,sizeof(conndevices)); /* end tracking */
       #endif
     }
     if (ifc.ifc_buf)
       free (ifc.ifc_buf);
     close (fd);
   }

   #if (CLIENT_OS == OS_AMIGAOS)
   amigaCloseSocketLib();
   #endif

   if (foundif)
   {
     TRACE_OUT((-1,"Lurk::InternalIsConnected() => 1\n"));
     return 1;
   }

#elif (CLIENT_OS == OS_MACOS)

   #ifdef LURK_LISTENER /* using an asychronous listen callback */
   if (isonline != -1)  /* already determined yes or no? */
     return isonline;   /* return the state if so */
   isonline = 0;        /* the callback hasn't been called yet */
   /* fallthrough */    /* so get initial state manually below */
   #endif

   TOptMgmt cmd;
   TOption* option;
   UInt8 buf[128];
   cmd.opt.buf = buf;
   cmd.opt.len = sizeof(TOptionHeader);
   cmd.opt.maxlen = sizeof buf;
   cmd.flags = T_CURRENT;
   option = (TOption *) buf;
   option->level = COM_PPP;
   option->name = CC_OPT_GETMISCINFO;
   option->status = 0;
   option->len = sizeof(TOptionHeader);

   OTOptionManagement(fEndPoint, &cmd, &cmd);
   option = (TOption *) cmd.opt.buf;

   if ((option->status == T_SUCCESS) || (option->status == T_READONLY))
   {
     CCMiscInfo *info = (CCMiscInfo *) &option->value[0];
     if (info->connectionStatus == kPPPConnectionStatusConnected)
     {
       #ifdef LURK_LISTENER
       isonline=1;
       #endif
       return 1;
     }
   }

#else
  #error "InternalIsConnected() must always return a valid yes/no."
  #error "There is no default return value."
#endif

  TRACE_OUT((-1,"Lurk::InternalIsConnected() => 0\n"));
  return 0;// Not connected
}

/* ---------------------------------------------------------- */

int Lurk::DialIfNeeded(int force /* !0== override lurk-only */ )
{                                /* returns 0 if connected, -1 if error */
  TRACE_OUT((+1,"Lurk::DialIfNeeded() (islurkstarted?=%d)\n",islurkstarted));

  if (!islurkstarted)
  {
    TRACE_OUT((-1,"!islurkstarted. returning -1\n"));
    return -1; // Lurk can't be started, evidently
  }

  if (!conf.dialwhenneeded)           // We don't handle dialing
  {
    TRACE_OUT((-1,"!conf.dialwhenneeded. returning 0\n"));
    return 0;
  }

  if (IsConnected()) // We're already connected
  {
    TRACE_OUT((-1,"already connected. returning 0\n"));
    return 0;
  }

  if (conf.lurkmode == CONNECT_LURKONLY && !force)
  {
    TRACE_OUT((-1,"(conf.lurkmode == CONNECT_LURKONLY && !force). returning -1\n"));
    return -1; // lurk-only, we're not allowed to connect unless forced
  }

#if (CLIENT_OS == OS_WIN16)

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

  if ((GetCapabilityFlags() & CONNECT_DODBYPROFILE) != 0) /* have ras */
  {
    RASDIALPARAMS dialparameters;
    BOOL passwordretrieved;
    DWORD returnvalue;
    char buffer[260]; /* maximum registry key length */
    const char *connname = (const char *)(&conf.connprofile[0]);

    TRACE_OUT((0,"((GetCapabilityFlags() & CONNECT_DODBYPROFILE) != 0)\n"));

    dohangupcontrol = 0;           // whether we do HangupIfNeeded() or not

    if (*connname == 0)
    {
      HKEY hkey;
      TRACE_OUT((0,"No connname provided. Trying to get default.\n"));
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
        TRACE_OUT((0,"No default, getting first on list\n"));
        const char **connlist = GetConnectionProfileList();
        TRACE_OUT((0,"GetConnectionProfileList() => %p\n", connlist));
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
    TRACE_OUT((0,"using profile='%s'\n", connname));
    dialparameters.dwSize=sizeof(RASDIALPARAMS);
    strcpy(dialparameters.szEntryName,connname);
    strcpy(dialparameters.szPhoneNumber,"");
    strcpy(dialparameters.szCallbackNumber,"*");
    strcpy(dialparameters.szUserName,"");
    strcpy(dialparameters.szPassword,"");
    strcpy(dialparameters.szDomain,"*");

    TRACE_OUT((0,"RasGetEntryDialParams()\n"));
    returnvalue =
      RasGetEntryDialParams(NULL,&dialparameters,&passwordretrieved);
    TRACE_OUT((0,"RasGetEntryDialParams() => %d\n", returnvalue));

    if ( returnvalue==0 )
    {
      HRASCONN connhandle = NULL;

      LogScreen("Dialing '%s'...\n",dialparameters.szEntryName);
      TRACE_OUT((0,"RasDial()\n"));
      returnvalue = RasDial(NULL,NULL,&dialparameters,NULL,NULL,&connhandle);
      TRACE_OUT((0,"RasDial() => %d\n", returnvalue));

      if (returnvalue == 0)
      {
        hRasDialConnHandle = connhandle; //we only hangup this connection
        dohangupcontrol = 1;  // we also control hangup
        TRACE_OUT((-1,"DialIfNeeded() => 0\n"));
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
      if (RasGetErrorString(returnvalue,buffer,sizeof(buffer)) != 0)
        sprintf( buffer, "Unknown RAS error %ld", (long) returnvalue );
      LogScreen("Dial cancelled:\n%s\n", buffer);
    }
  }
  TRACE_OUT((-1,"DialIfNeeded() => -1\n"));
  return -1;

#elif (CLIENT_OS == OS_AMIGAOS)

  if (strlen(conf.connprofile) > 0)
    LogScreen("Attempting to put '%s' online...\n",conf.connprofile);
  else
    LogScreen("Attempting to put default interface online...\n");

  int async, connected = 0;
  if ((async = amigaOnOffline(TRUE,conf.connprofile)))
  {
    int retry = 0, maxretry = (async == 2) ? 40 : 1; // 40s max to connect with a modem
    while (((connected = IsConnected()) == 0) && ((++retry)<maxretry))
    {
      sleep(1);
      if (CheckExitRequestTriggerNoIO()) break;
    }
  }
  if (!connected)
  {
    TRACE_OUT((-1,"DialIfNeeded() => -1\n"));
    return -1;
  }
  dohangupcontrol = 1;  // we should also control hangup
  TRACE_OUT((-1,"DialIfNeeded() => 0\n"));
  return 0;

#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_OS2) || \
     (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_OPENBSD) || \
     (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_BSDOS) || \
     ((CLIENT_OS == OS_MACOSX) && !defined(__RHAPSODY__))
  dohangupcontrol = 0;
  if (conf.connstartcmd[0] == 0)  /* we don't do dialup */
  {
    LogScreen("Dial Error. No dial-start-command specified.\n");
    conf.dialwhenneeded = 0; //disable it!
    return -1;
  }
  if (system( conf.connstartcmd ) == 127 /*exec error */)
  {                                        //pppstart of whatever
    LogScreen("Unable to exec '%s'\n%s\n", conf.connstartcmd, strerror(errno));
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
  if (conf.connstopcmd[0] != 0)
    system( conf.connstopcmd );
  return -1;

#else
  return -1; //failed

#endif
}

/* ---------------------------------------------------------- */

int Lurk::HangupIfNeeded(void) //returns 0 on success, -1 on fail
{
  int isconnected;

  TRACE_OUT((+1,"Lurk::HangupIfNeeded()\n"));

  if (!islurkstarted)      // Lurk can't be started, evidently
  {
    TRACE_OUT((-1,"!islurkstarted. returning -1\n"));
    return -1;
  }
  if (!conf.dialwhenneeded)     // We don't handle dialing
  {
    TRACE_OUT((-1,"!conf.dialwhenneeded. returning 0\n"));
    return 0;
  }

  TRACE_OUT((0,"IsConnected() check\n"));
  isconnected = IsConnected();
  TRACE_OUT((0,"IsConnected() returned %d\n",isconnected));

  if (!dohangupcontrol) //if we didn't initiate, we shouldn't terminate
  {
    TRACE_OUT((-1,"we didn't initiate, so returning\n"));
    return ((isconnected)?(-1):(0));
  }

#if (CLIENT_OS == OS_WIN16)
  if (hWinsockInst)
  {
    FreeLibrary(hWinsockInst);
    hWinsockInst = NULL;
  }
  dohangupcontrol = 0;
  return 0;

#elif (CLIENT_OS == OS_WIN32)

  TRACE_OUT((0,"HUP-stage-01\n"));
  if (isconnected)
  {
    RASCONN rasconn;
    RASCONN *rasconnp = NULL;
    DWORD cb, whichconn, cConnections;

    cb = sizeof(rasconn);
    rasconn.dwSize = sizeof(RASCONN);
    rasconnp = &rasconn;
    TRACE_OUT((0,"HUP-stage-02\n"));
    if (RasEnumConnections( rasconnp, &cb, &cConnections) != 0)
    {
      cConnections = 0;
      TRACE_OUT((0,"HUP-stage-03\n"));
      if (cb > (DWORD)(sizeof(RASCONN)))
      {
        rasconnp = (RASCONN *) malloc( (int)cb );
        TRACE_OUT((0,"HUP-stage-04\n"));
        if (rasconnp)
        {
          TRACE_OUT((0,"HUP-stage-05\n"));
          rasconnp->dwSize = sizeof(RASCONN);
          if (RasEnumConnections( rasconnp, &cb, &cConnections) != 0)
            cConnections = 0;
          TRACE_OUT((0,"HUP-stage-06\n"));
        }
        TRACE_OUT((0,"HUP-stage-07\n"));
      }
      TRACE_OUT((0,"HUP-stage-08\n"));
    }
    TRACE_OUT((0,"HUP-stage-09\n"));


    TRACE_OUT((0,"HUP-stage-10\n"));
    for (whichconn = 0; whichconn < cConnections; whichconn++ )
    {
      HRASCONN hrasconn = rasconnp[whichconn].hrasconn;
      if (hrasconn == hRasDialConnHandle) // same conn as opened with rasdial?
      {
        TRACE_OUT((0,"HUP-stage-11\n"));
        RASCONNSTATUS rasconnstatus;
        rasconnstatus.dwSize = sizeof(RASCONNSTATUS);
        do{
          TRACE_OUT((0,"HUP-stage-12\n"));
          if (RasGetConnectStatus(hrasconn,&rasconnstatus) != 0)
            break;
          TRACE_OUT((0,"HUP-stage-13\n"));
          if (rasconnstatus.rasconnstate == RASCS_Connected)
          {
            TRACE_OUT((0,"HUP-stage-14\n"));
            if (RasHangUp( hrasconn ) != 0)
              break;
            TRACE_OUT((0,"HUP-stage-15\n"));
            Sleep(1000);
          }
          TRACE_OUT((0,"HUP-stage-16\n"));
        } while (rasconnstatus.rasconnstate == RASCS_Connected);
        TRACE_OUT((0,"HUP-stage-17\n"));
      }
      TRACE_OUT((0,"HUP-stage-18\n"));
    }
    TRACE_OUT((0,"HUP-stage-19\n"));

    if (rasconnp != NULL && rasconnp != &rasconn)
    {
      TRACE_OUT((0,"HUP-stage-20\n"));
      free((void *)rasconnp );
    }
  }

  TRACE_OUT((-1,"returning 0\n"));
  hRasDialConnHandle = NULL;
  dohangupcontrol = 0;
  return 0;

#elif (CLIENT_OS == OS_AMIGAOS)

  if (isconnected)
  {
    int droppedconn = 0, async;
    if ((async = amigaOnOffline(FALSE,conf.connprofile)))
    {
      int retry = 0, maxretry = (async == 2) ? 10 : 1;
      while (((droppedconn = (!IsConnected())) == 0) && ((++retry)<maxretry))
      {
        sleep(1);
        if (CheckExitRequestTriggerNoIO()) break;
      }
    }
    if (!droppedconn)
    {
      TRACE_OUT((-1,"DialIfNeeded() => -1\n"));
      return -1;
    }
  }

  TRACE_OUT((-1,"HangupIfNeeded() => 0\n"));
  dohangupcontrol = 0;
  return 0;

#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_OS2) || \
      (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_OPENBSD) || \
      (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_BSDOS) || \
      ((CLIENT_OS == OS_MACOSX) && !defined(__RHAPSODY__))

  if (isconnected)
  {
    int droppedconn = 0;
    if (conf.connstopcmd[0] == 0) //what can we do?
      droppedconn = 1;
    else if (system( conf.connstopcmd ) == 127 /* exec error */)
      LogScreen("Unable to exec '%s'\n%s\n", conf.connstopcmd, strerror(errno));
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
