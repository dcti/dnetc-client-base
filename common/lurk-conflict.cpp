// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: lurk-conflict.cpp,v $
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
return "@(#)$Id: lurk-conflict.cpp,v 1.23 1999/02/08 17:11:39 cyp Exp $"; }
#endif

/* --------------------------------- */

#include "cputypes.h"
#include "logstuff.h"
#include "lurk.h"

/* -------------------------------- */

Lurk dialup;

int Lurk::CheckIfConnectRequested(void) //yes/no
{
  if (lurkmode != CONNECT_LURKONLY && lurkmode != CONNECT_LURK)
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
    LogScreen("Dialup Connection Disconnected.%s",msg);
    }
  return 0;
}

/* ---------------------------------------------------------- */

int Lurk::CheckForStatusChange(void) //returns -1 if connection dropped
{
  //if ( (lurkmode == 0) && (!dialwhenneeded) )
  //  return 0; // We're not lurking.
  if ( lastcheckshowedconnect && !IsConnected() ) //if (Status() < oldlurkstatus)
    return -1;  // we got disconnected!
  return 0;
}


/* ================================================================== */
/* ************** OS SPECIFIC STUFF BEGINS HERE ********************* */
/* ================================================================== */


#if (CLIENT_OS == OS_LINUX)
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
//#include <linux/if_slip.h>
//#include <linux/if.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>

#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)

#include <windows.h>
#include <string.h>
static int have_winsock = 0;
static int demand_loadable = 0;
static HINSTANCE hwsockinst = NULL;

#elif (CLIENT_OS == OS_WIN32)

#include <windows.h>
#include <ras.h>
#include <raserror.h>
#include <string.h>

typedef DWORD (WINAPI *rasenumconnectionsT)(LPRASCONN, LPDWORD, LPDWORD);
typedef DWORD (WINAPI *rasgetconnectstatusT)(HRASCONN, LPRASCONNSTATUS);
typedef DWORD (WINAPI *rashangupT)(HRASCONN);
typedef DWORD (WINAPI *rasdialT)(LPRASDIALEXTENSIONS, LPSTR,
                 LPRASDIALPARAMS, DWORD, LPVOID, LPHRASCONN);
typedef DWORD (WINAPI *rasgeterrorstringT)(UINT, LPTSTR, DWORD);
typedef DWORD (WINAPI *rasgetentrydialparamsT)(LPSTR,
                 LPRASDIALPARAMS, LPBOOL);
typedef DWORD (WINAPI *rasenumentriesT)(LPTSTR, LPTSTR,
                 LPRASENTRYNAME, LPDWORD, LPDWORD);

static rasenumconnectionsT rasenumconnections = NULL;
static rasgetconnectstatusT rasgetconnectstatus = NULL;
static rashangupT rashangup = NULL;
static rasdialT rasdial = NULL;
static rasgeterrorstringT rasgeterrorstring = NULL;
static rasgetentrydialparamsT rasgetentrydialparams = NULL;
static rasenumentriesT rasenumentries = NULL;

static HINSTANCE hrasapiInstance = NULL;
static int rasapiinitialized = 0;
char rasapiiniterrmsg[64];

static int DeinitializeRASAPI(void)
{
  if ((--rasapiinitialized) == 0) 
    {
    if (hrasapiInstance)
      FreeLibrary(hrasapiInstance);
    hrasapiInstance = NULL;
    }
  return 0;
}

static int InitializeRASAPI(void)
{
  if ((++rasapiinitialized) == 1)
    {
    OFSTRUCT ofstruct;
    ofstruct.cBytes = sizeof(ofstruct);

    rasapiinitialized = 0;

    #ifndef OF_SEARCH
    #define OF_SEARCH 0x0400
    #endif
    if ( OpenFile( "RASAPI32.dll", &ofstruct, OF_EXIST|OF_SEARCH) >= 0)
      {
      hrasapiInstance = LoadLibrary( ofstruct.szPathName );
      if ((UINT)hrasapiInstance <= 32)
        hrasapiInstance = NULL;
      else
        rasapiinitialized = 1;
      }
    }
  return ((rasapiinitialized > 0)?(0):(-1));
}

static FARPROC LoadRASAPIProc( const char *procname )
{
  if (rasapiinitialized > 0 && hrasapiInstance)
    return GetProcAddress( hrasapiInstance, procname );
  return NULL;
}

#elif (CLIENT_OS == OS_OS2)

#define TCPIPV4               //should also work with V3 though
#if defined(__WATCOMC__)
#include <stdio.h>
#include <string.h>
#include <types.h>
#include "..\platforms\os2cli\os2defs.h"
#endif
#include <net/if.h>          // ifmib
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
#if defined(__EMX__)
#include  <sys/process.h>     // P_NOWAIT, spawnl()
#else
#include <process.h>
#endif
#ifndef SIOSTATIF
#define SIOSTATIF         _IOR('n', 48, struct ifmib)
#endif

#endif


/* ========================================================== */

Lurk::Lurk()
{
  islurkstarted = lastcheckshowedconnect = dohangupcontrol = 0;
  lurkmode = dialwhenneeded = 0;
  conndevice[0] = connprofile[0] = connifacemask[0] = 0;
  connstartcmd[0] = connstopcmd[0] = 0;

#if (CLIENT_OS == OS_WIN32)
  if (InitializeRASAPI() != 0)
    strcpy(rasapiiniterrmsg,"RASAPI32.DLL is not available.");
  else  
    {
    rasapiiniterrmsg[0]=0;
    rasenumconnections = (rasenumconnectionsT) LoadRASAPIProc("RasEnumConnectionsA");
    rasgetconnectstatus = (rasgetconnectstatusT) LoadRASAPIProc("RasGetConnectStatusA");
    rashangup = (rashangupT) LoadRASAPIProc("RasHangUpA");
    rasdial = (rasdialT) LoadRASAPIProc("RasDialA");
    rasgeterrorstring = (rasgeterrorstringT) LoadRASAPIProc("RasGetErrorStringA");
    rasgetentrydialparams = (rasgetentrydialparamsT)LoadRASAPIProc("RasGetEntryDialParamsA");
    rasenumentries = (rasenumentriesT)LoadRASAPIProc("RasEnumEntriesA");

    if (!rasenumconnections || !rasgetconnectstatus || !rashangup ||
        !rasdial || !rasgeterrorstring || !rasgetentrydialparams ||
        !rasenumentries )
      {
      LPVOID lpMsgBuf;
      FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,GetLastError(),MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
        (LPTSTR) &lpMsgBuf,0,NULL);
      strncpy( rasapiiniterrmsg, (char *)lpMsgBuf, sizeof(rasapiiniterrmsg));
      rasapiiniterrmsg[sizeof(rasapiiniterrmsg)-1]=0;
      LocalFree( lpMsgBuf );
      DeinitializeRASAPI();
      }
    }
#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  OFSTRUCT ofstruct;
  ofstruct.cBytes = sizeof(ofstruct);

  demand_loadable = 0;
  have_winsock = 0;
  if ( OpenFile( "WINSOCK.DLL", &ofstruct, OF_EXIST|OF_SEARCH) >= 0)
    {
    char *p = strrchr( ofstruct.szPathName, '\\' );
    const char *q = "TRUMPWSK.INI";
    have_winsock = 1;
    if (p != NULL)
      {
      strcpy( p+1, q );
      q = (const char *)(&ofstruct.szPathName[0]);
      }
    if ( OpenFile( q, &ofstruct, OF_EXIST|OF_SEARCH) >= 0)
      {      
      int i=GetPrivateProfileInt( "Trumpet Winsock", "dial-option", 0, ofstruct.szPathName );
      if (i != 0) /* 1==login on demand, 2=login/logout on demand */
        demand_loadable = 1;
      }
    }
#endif
  return;
}

/* ---------------------------------------------------------- */

Lurk::~Lurk()
{
  islurkstarted = lastcheckshowedconnect = 0;
  #if (CLIENT_OS==OS_WIN32)
  DeinitializeRASAPI();
  #endif
  return;
}

/* ---------------------------------------------------------- */

const char **Lurk::GetConnectionProfileList(void)
{
#if (CLIENT_OS==OS_WIN32)
  static RASENTRYNAME rasentries[10];
  static char *configptrs[(sizeof(rasentries)/sizeof(rasentries[0]))+1];
  if (rasapiinitialized > 0)
    {
    DWORD buffersize = sizeof(rasentries);
    DWORD maxentries = 0;
    rasentries[0].dwSize=sizeof(RASENTRYNAME);
    if ( rasenumentries(NULL,NULL,&rasentries[0],&buffersize,&maxentries) != 0)
      maxentries = 0;
    if (maxentries >= 1)
      {
      DWORD entry = 0;
      unsigned int index = 0;
      for (;((entry < maxentries) && 
            (index < (sizeof(configptrs)/sizeof(configptrs[0])))); entry++)
        configptrs[index++]=rasentries[entry].szEntryName;
      configptrs[index] = NULL;
      return (const char **)(&configptrs[0]);
      }
    }    
#endif
  return NULL;
}

/* ---------------------------------------------------------- */

int Lurk::Start(void)// Initializes Lurk Mode. returns 0 on success.
{
  if (lurkmode != CONNECT_LURKONLY && lurkmode != CONNECT_LURK)
    {
    lurkmode = 0;
    return -1; // We're not supposed to lurk!
    }
  if (islurkstarted)
    return 0;

#if (CLIENT_OS == OS_WIN32)
  if (rasapiinitialized <= 0)
    {
    char *msg = "Dial-up must be installed for -lurk/-lurkonly.";
    if (rasapiiniterrmsg)
      LogScreen("%s\n%s", rasapiiniterrmsg, msg );
    else
      LogScreen(msg);
    lurkmode = 0;
    return -1;
    }
#elif (CLIENT_OS == OS_WIN16)
  if (!have_winsock)
    {
    LogScreen("Winsock must be available for -lurk/-lurkonly.");
    dialwhenneeded = lurkmode = 0;
    return -1;
    }
  if (dialwhenneeded && !demand_loadable)
    {
    LogScreen("Demand dialing is only supported for Trumpet Winsock.\n");
    dialwhenneeded = 0;
    }
#elif (CLIENT_OS == OS_OS2)
  dialwhenneeded = 0; /* disabled until we can get this fixed */  
#endif

  islurkstarted=1;
  return 0;
}

/* ---------------------------------------------------------- */

int Lurk::Stop(void)// DeInitializes Lurk Mode. returns 0 on success.
{
  islurkstarted=0;
  return 0;
}

/* ---------------------------------------------------------- */

int Lurk::GetCapabilityFlags(void)
{
  int what = 0;
  #if (CLIENT_OS == OS_WIN32) /* does not support iface masking */
  if (rasapiinitialized > 0)
    what = (CONNECT_LURK|CONNECT_LURKONLY|CONNECT_DOD|CONNECT_DODBYPROFILE);
  #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  if (have_winsock) what = ( CONNECT_LURK | CONNECT_LURKONLY);
  if (demand_loadable) what |= CONNECT_DOD;
  #elif (CLIENT_OS == OS_LINUX)
  what = (CONNECT_LURK | CONNECT_LURKONLY | CONNECT_DOD | CONNECT_IFACEMASK);
  #elif (CLIENT_OS == OS_OS2)
  what = ( CONNECT_LURK | CONNECT_LURKONLY);
  #endif
  return what;
}

/* ---------------------------------------------------------- */

int Lurk::IsConnected(void)               // Checks status of connection
{                                              
  conndevice[0]=0;

  if (!islurkstarted)
    return 0;

#if (CLIENT_OS == OS_WIN32)
  if (rasenumconnections && rasgetconnectstatus)
    {
    RASCONN rasconn[8];
    DWORD cb;
    DWORD cConnections;
    RASCONNSTATUS rasconnstatus;
    (rasconn[0]).dwSize = sizeof(RASCONN);
    cb = sizeof(rasconn);
    if (rasenumconnections(&rasconn[0], &cb, &cConnections) == 0)
      {
      if (cConnections > 0)
        {
        rasconnstatus.dwSize = sizeof(RASCONNSTATUS);
        for (DWORD whichconn = 1; whichconn <= cConnections; whichconn++)
          {
          if (rasgetconnectstatus((rasconn[whichconn-1]).hrasconn,&rasconnstatus) == 0 && 
              rasconnstatus.rasconnstate == RASCS_Connected)
            {
            strncpy( conndevice, (rasconn[whichconn-1]).szEntryName, sizeof(conndevice) );
            conndevice[sizeof(conndevice)-1]=0;
            return 1;// We're connected
            }
          }
        }
      }
    }
#elif (CLIENT_OS == OS_OS2)
   int s, i, rc, j, foundif = 0;
   struct ifmib MyIFMib = {0};
   struct ifact MyIFNet = {0};

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
       if (MyIFMib.iftable[j].ifType != HT_ETHER)   /* i/f is not ethernet */
         {
         if (MyIFMib.iftable[j].ifType != HT_PPP)  /* i/f is not loopback (yes I know it says PPP) */
           {
           if (MyIFNet.iftable[i].ifa_addr != 0x0100007f)  /* same thing for TCPIP < 4.1 */
             {
             struct ifreq MyIFReq = {0};
             if (j < 9)
               {
               sprintf(MyIFReq.ifr_name, "lan%d", j);
               }
             else if (j > 9)
               {
               sprintf(MyIFReq.ifr_name, "ppp%d", j-10);
               }
             else
               {
               strcpy(MyIFReq.ifr_name, "lo");
               }
             strncpy( conndevice, MyIFReq.ifr_name, sizeof(conndevice) );
             conndevice[sizeof(conndevice)-1]=0;
             rc = ioctl(s, SIOCGIFFLAGS, (char*)&MyIFReq, sizeof(MyIFReq));
             if (!rc)
               {
               if ((MyIFReq.ifr_flags & IFF_UP) != 0)
                 {
                 foundif = i+1; // Report online if SLIP or PPP detected
                 break;
                 }
               }
             } // != 0x0100007f
           } // != HT_PPP (loopback actually)
         } // != HT_ETHER
       } //for ...
     soclose(s);
     }
   if (foundif)
     return 1;
   conndevice[0]=0;
   
#elif ((CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN16S))
   if ( GetModuleHandle("WINSOCK") )
     return 1;
#elif (CLIENT_OS == OS_LINUX)
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
       p = (char *)realloc((void *)ifc.ifc_buf, sizeof(struct ifreq)*numreqs);
       if (!p)
         break;
       ifc.ifc_buf = (caddr_t)p;
       } 

     // now browse through the interface watch list
     // and check if there is an interface up and running
     if (ifc.ifc_len)
       {
       char ifacestowatch[sizeof(connifacemask)+4]; //64+4
       int i = 1, anydev = 0;
       strcpy( ifacestowatch, ":" );
       for (n=0;connifacemask[n];n++)
         {
         if (!isspace(connifacemask[n]))
           ifacestowatch[i++]=connifacemask[n];
         }
       ifacestowatch[i]=0;
       anydev = (i==1);
       strcat( ifacestowatch, ":" );
       
       //LogScreenf ("ifacestowatch = '%s'\n", ifacestowatch);
       for (n = 0, ifr = ifc.ifc_req; n < ifc.ifc_len; n += sizeof(struct ifreq), ifr++)
         {
         int founddev = 0;
         //LogScreenf ("looking at '%s'\n", ifr->ifr_name);
         if (anydev)
           founddev = 1;
         else
           {
           p = strstr( ifacestowatch, ifr->ifr_name );
           if (p) 
             founddev = (*(p-1)==':' && p[strlen(ifr->ifr_name)]==':');
           }
         if (founddev)
           {
           strncpy( conndevice, ifr->ifr_name, sizeof(conndevice) );
           conndevice[sizeof(conndevice)-1]=0;
           // now check if this iface is up and running
           if ( ioctl(fd, SIOCGIFFLAGS, ifr) >= 0) // get iface flags
             {
             //LogScreenf ("   flags = 0x%x\n", ifr->ifr_flags);
             if ((ifr->ifr_flags & (IFF_UP | IFF_RUNNING | IFF_LOOPBACK)) 
                                   == (IFF_UP | IFF_RUNNING))
               {
               foundif = (n / sizeof(struct ifreq)) + 1;
               break;
               }
             }
           }
         }
       }

     if (ifc.ifc_buf) 
       free (ifc.ifc_buf);
     close (fd);
     }

   if (foundif)
     return 1;
   conndevice[0]=0;
     
#endif // OS_LINUX
  return 0;// Not connected
}

/* ---------------------------------------------------------- */

int Lurk::DialIfNeeded(int force /* !0== override lurk-only */ )
{                                /* returns 0 if connected, -1 if error */
  if (!islurkstarted)
    return -1; // Lurk can't be started, evidently

  if (IsConnected()) // We're already connected
    return 0;

  if (lurkmode == CONNECT_LURKONLY && !force)
    return -1; // lurk-only, we're not allowed to connect unless forced

  if (!dialwhenneeded)           // We don't handle dialing
    return 0;

  dohangupcontrol = 0;           // whether we do HangupIfNeeded() or not

#if (CLIENT_OS == OS_WIN32)

  RASDIALPARAMS dialparameters;
  BOOL passwordretrieved;
  HRASCONN connectionhandle;
  DWORD returnvalue;
  char errorstring[128];

  if (connprofile[0]==0)
    {
    int n=0;
    const char **connlist = GetConnectionProfileList();
    if (connlist == NULL)
      return -1;
    if (connlist[0]==NULL)
      return -1;
    while (connlist[n])
      {
      if (strlen(connlist[n]) < sizeof(connprofile))
        {
        strcpy( connprofile, connlist[n] );
        break;
        }
      n++;
      }
    if (connprofile[0]==0)
      return -1;
    }

  dialparameters.dwSize=sizeof(RASDIALPARAMS);
  strcpy(dialparameters.szEntryName,connprofile);
  strcpy(dialparameters.szPhoneNumber,"");
  strcpy(dialparameters.szCallbackNumber,"*");
  strcpy(dialparameters.szUserName,"");
  strcpy(dialparameters.szPassword,"");
  strcpy(dialparameters.szDomain,"*");

  returnvalue =
    rasgetentrydialparams(NULL,&dialparameters,&passwordretrieved);

  if ( returnvalue==0 )
    {
    //if (passwordretrieved != TRUE)
    //  LogScreen("Password could not be found, connection may fail.");
    }
  else
    {
    switch(returnvalue)
      {
      case ERROR_CANNOT_FIND_PHONEBOOK_ENTRY:
        LogScreen("Phonebook entry %s could not be found, aborting dial.",
                  dialparameters.szEntryName);
        return -1;
      case ERROR_CANNOT_OPEN_PHONEBOOK:
        LogScreen("The phonebook cound not be opened, aborting dial.");
        return -1;
      case ERROR_BUFFER_INVALID:
        LogScreen( "Invalid buffer passed, aborting dial.");
        return -1;
      }
    }

  LogScreen("Dialing %s...",dialparameters.szEntryName);
  returnvalue=rasdial(NULL,NULL,&dialparameters,NULL,NULL,&connectionhandle);

  if (returnvalue == 0)
    {
    dohangupcontrol = 1;  // should we also control hangup?
    return 0;             // If we got here, connection successful.
    }

  rasgeterrorstring(returnvalue,errorstring,sizeof(errorstring));
  LogScreen("Connection initiation error:\n%s",errorstring);

#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_OS2)

  if (connstartcmd[0] == 0)  /* we don't do dialup */
    return 0;                /* bad! but same result as !dialwhenneeded */

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
      //LogScreen("Opened connection on %s...\n", conndevice );
      dohangupcontrol = 1;  // we should also control hangup
      return 0;
      }
    }
  if (connstopcmd[0] != 0)
    system( connstopcmd );

#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  if (demand_loadable)
    {
    HINSTANCE tmp;
    if ((tmp = LoadLibrary("WINSOCK.DLL")) < ((HINSTANCE)(32)))
      return -1;
    hwsockinst = tmp;
    dohangupcontrol = 1;  // we should also control hangup
    return 0;
    }    
#endif

  return -1; //failed
}


/* ---------------------------------------------------------- */

int Lurk::HangupIfNeeded(void) //returns 0 on success, -1 on fail
{
  if (!islurkstarted)
    return -1;  // Lurk can't be started, evidently

  if (!dialwhenneeded)     // We don't handle dialing
    return 0;

  int isconnected = IsConnected();

  if (!dohangupcontrol) //if we didn't initiate, we shouldn't terminate
    return ((isconnected)?(-1):(0));

  if (!isconnected)     // We're already disconnected
    {
    dohangupcontrol = 0;
    return 0; 
    }

#if (CLIENT_OS == OS_WIN32)
  if (rasenumconnections && rasgetconnectstatus)
    {
    RASCONN rasconn[8];
    DWORD cb;
    DWORD cConnections;
    RASCONNSTATUS rasconnstatus;
    (rasconn[0]).dwSize = sizeof(RASCONN);
    cb = sizeof(rasconn);
    if (rasenumconnections(&rasconn[0], &cb, &cConnections) == 0)
      {
      if (cConnections > 0)
        {
        rasconnstatus.dwSize = sizeof(RASCONNSTATUS);
        for (DWORD whichconn = 1; whichconn <= cConnections; whichconn++)
          {
          if (rasgetconnectstatus((rasconn[whichconn-1]).hrasconn,&rasconnstatus) == 0 && 
              rasconnstatus.rasconnstate == RASCS_Connected)
            {
            // We're connected
            if (rashangup(rasconn[whichconn-1].hrasconn) == 0) // So kill it!
              {
              dohangupcontrol = 0;
              return 0; // Successful hangup
              }
            return -1; // RasHangUp reported an error.
            }
          }
        }
      }
    }
#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_OS2)

  if (connstopcmd[0] == 0) //what can we do?
    {
    dohangupcontrol = 0;
    return 0;
    }
  if (system( connstopcmd ) == 127 /* exec error */)
    {                                               //pppstop of whatever
    LogScreen("Unable to exec '%s'\n%s\n", connstopcmd, strerror(errno));
    return -1;
    } 
  int droppedconn = 0, retry = 0;
  for (;retry < 10;retry++)
    {
    if ((droppedconn = (!IsConnected()))!=0) //whee! connection closed
      break;
    sleep(1);
    }
  if (!droppedconn)
    return -1;
  dohangupcontrol = 0;

#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  if (demand_loadable)
    {
    if (hwsockinst)
      {
      FreeLibrary(hwsockinst);
      hwsockinst = NULL;
      }
    dohangupcontrol = 0;
    return 0;
    }
#endif
  return 0;
}
