/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This module is a self-contained abstraction layer between the client/proxy
 * and rasapi32.dll allowing the use of RAS API calls without having to
 * worry whether ras/dun is installed or not. 
 *
 * note: this pseudo-static-library only contains a subset of 
 * the full RAS API
 *
 * Because RAS and/or dependant dll's are not permanently in memory, loading
 * and unloading them on the fly (particularly if its done several times
 * a minute, or disks spindown) is not a good idea. 
 * #define DYNAMIC_RASAPI if you want it anyway.        -cyp 10 Apr 1999
 *
 * ----------------------------------------------------------------------
*/

const char *w32ras_cpp(void) {
return "@(#)$Id: w32ras.cpp,v 1.1.2.1 2001/01/21 15:10:25 cyp Exp $"; }

#include <windows.h>
#include <ras.h>
#include <raserror.h>

#ifdef RDEOPT_Router /* only defined in ras.h >= 1997 */
#define my_LPCSTR LPCSTR /* someone finally learned all about const */
#else
#define my_LPCSTR LPSTR
#endif

/* ------------------------------------------------------------------- */

static int __is_ras_installed(void)
{
  static int ras_detect_state = 0;

  if (ras_detect_state == 0) /* not yet checked */
  {
    /* KB article Q181518 is wrong. Simply loading rasapi32.dll and 
       testing for load success is innapropriate on NT since rasapi32's
       WEP will throw up an error box if RAS is _no_longer_ installed.
       We *have* to use the registry.                 -cyp 18 Apr 1999
    */
    HKEY key;
    OFSTRUCT ofstruct;
    ofstruct.cBytes = sizeof(ofstruct);
  
    #ifndef OF_SEARCH
    #define OF_SEARCH 0x0400
    #endif
    if ( OpenFile( "RASAPI32.DLL", &ofstruct, OF_EXIST|OF_SEARCH) == HFILE_ERROR)
      ras_detect_state = -1; /* definitive not installed */
    else
    {
      ras_detect_state = -1;   /* assume not installed */
      if (ERROR_SUCCESS == RegOpenKeyEx(HKEY_LOCAL_MACHINE,
          "SYSTEM\\CurrentControlSet\\Services\\RemoteAccess",
           0, KEY_READ, &key))
      {
        ras_detect_state = 0;  /* *may* be installed */
        RegCloseKey(key);
      }
    }

    if (ras_detect_state == 0)
    {
      OSVERSIONINFO osver;
      osver.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
      GetVersionEx(&osver);

      if (osver.dwPlatformId == VER_PLATFORM_WIN32_NT)
      {
        /* 
        NT deletes all RAS specific registry keys when RAS is 
        deinstalled, so that will have been caught in the regcheck above.
        HKLM = HKEY_LOCAL_MACHINE
        CCS = Current Control Set
        HKLM\CCS\Services: AsyncMac, NdisTapi, NdisWan, RemoteAccess, 
                      Ras[Acd|Arp|Auto|Man] NetBT\Adapters\NdisWan(x) 
        HKLM\SOFTWARE\Microsoft: AsyncMac, NdisWan, RAS, RasAuto, RasMan,
                      RemoteAccess, TAPI DEVICES, RAS Autodial, RAS Monitor, 
                      RAS Phonebook
        See also: 
        http://msdn.microsoft.com/library/winresource/dnwinnt/S7BF7.HTM
        
        However, we still need to check if RAS is installed *but* has been 
        disabled as a service.
        */

        SC_HANDLE hscm;
        ras_detect_state = -1;   /* assume not installed */


        hscm = OpenSCManager(NULL, NULL, GENERIC_READ);
        //TRACE_OUT((+1,"begin: NT ras check. OpenSCManager()=>%08x\n",hscm));
        if (hscm)
        {
          ENUM_SERVICE_STATUS  essServices[16];
          DWORD dwResume = 0, cbNeeded = 1;

          while (ras_detect_state < 0 && cbNeeded > 0)
          { 
            DWORD index, csReturned = 0;
            if (!EnumServicesStatus(hscm, SERVICE_WIN32, SERVICE_ACTIVE,
               essServices, sizeof(essServices), &cbNeeded, &csReturned, 
               &dwResume))
            {
              if (GetLastError() != ERROR_MORE_DATA)
              {
                //TRACE_OUT((0,"NT ras check. EnumServiceStatus() err=%08x\n",GetLastError()));
                break;
              }
            }
            for (index = 0; index < csReturned; index++)
            {
              //TRACE_OUT((0,"NT ras check. found EnumServiceStatus() =>%s\n",essServices[index].lpServiceName));
              if (0==lstrcmp(essServices[index].lpServiceName,TEXT("RasMan")))
              {
                // service exists. RAS is installed.
                ras_detect_state = +1;
                break;
              }
            }
          }
          CloseServiceHandle (hscm);
        }
        //TRACE_OUT((-1,"end: NT ras check.\n"));
      }
      else /* win9x (may also be work for win2k) */
      {
        ras_detect_state = -1;   /* assume not installed */
        if (RegOpenKey(HKEY_LOCAL_MACHINE,
          "Software\\Microsoft\\Windows\\CurrentVersion\\"
          "Setup\\OptionalComponents\\RNA", &key) == ERROR_SUCCESS)
        {
          unsigned int i;
          DWORD valuetype = REG_SZ;
          char buffer[260]; /* maximum registry key length */
          DWORD valuesize = sizeof(buffer);
          if ( RegQueryValueEx(key, "Installed", NULL, &valuetype, 
             (unsigned char *)(&buffer[0]), &valuesize) != ERROR_SUCCESS )
            buffer[0] = 0;
          for (i=0;buffer[i];i++)
          {
            if (buffer[i]=='y' || buffer[i]=='Y' ||
               (buffer[i]>='1' && buffer[i]<='9'))
              
            {
              ras_detect_state = +1; /* definitive is installed */
              break;
            }
            else if (buffer[i]!=' ' && buffer[i]!='\t')
            {
              break;
            }
          }
          RegCloseKey(key);
        }
      }
    }
  }

  #if 0
  if (ras_detect_state > 0)
  MessageBox(NULL,"rasisinstalled: found=yes","Blah",MB_OK);
  else
  MessageBox(NULL,"rasisinstalled: found=no","Blah",MB_OK);
  #endif
  return (ras_detect_state > 0);
}  

/* ------------------------------------------------------------------- */

#ifdef UNICODE
#error Cannot use this with unicode
#endif

typedef DWORD (WINAPI *rasenumconnectionsT)(LPRASCONNA, LPDWORD, LPDWORD);
typedef DWORD (WINAPI *rasgetconnectstatusT)(HRASCONN, LPRASCONNSTATUSA );
typedef DWORD (WINAPI *rashangupT)(HRASCONN);
typedef DWORD (WINAPI *rasgeterrorstringT)(UINT, LPSTR, DWORD);
typedef DWORD (WINAPI *rasdialT)(LPRASDIALEXTENSIONS, my_LPCSTR, 
                                 LPRASDIALPARAMSA, DWORD, LPVOID, LPHRASCONN );
typedef DWORD (WINAPI *rasgetentrydialparamsT)(my_LPCSTR,
                                               LPRASDIALPARAMSA, LPBOOL);
typedef DWORD (WINAPI *rasenumentriesT)(my_LPCSTR, my_LPCSTR,
                                        LPRASENTRYNAMEA, LPDWORD, LPDWORD);

static rasenumconnectionsT rasenumconnections = NULL;
static rasgetconnectstatusT rasgetconnectstatus = NULL;
static rashangupT rashangup = NULL;
static rasdialT rasdial = NULL;
static rasgeterrorstringT rasgeterrorstring = NULL;
static rasgetentrydialparamsT rasgetentrydialparams = NULL;
static rasenumentriesT rasenumentries = NULL;


static int __rasapi_init( int doalloc = 1 )
{
  static HINSTANCE hrasapiInstance = NULL;
  static int rasapiinitialized = 0;
  int rc = 0;

  if (doalloc)
  {
    if (__is_ras_installed() == 0)
      return -1;

    if ((++rasapiinitialized) == 1)
    {
      UINT olderrmode = SetErrorMode(SEM_NOOPENFILEERRORBOX);
      hrasapiInstance = LoadLibrary("RASAPI32.DLL");
      SetErrorMode(olderrmode);
      if (hrasapiInstance <= ((HINSTANCE)(32)))
      {
        hrasapiInstance = NULL;
        rc = -1;
        doalloc = 0;
      }
      else
      {
        rasenumconnections = (rasenumconnectionsT) GetProcAddress( hrasapiInstance, "RasEnumConnectionsA");
        rasgetconnectstatus = (rasgetconnectstatusT) GetProcAddress( hrasapiInstance, "RasGetConnectStatusA");
        rashangup = (rashangupT) GetProcAddress( hrasapiInstance, "RasHangUpA");
        rasdial = (rasdialT) GetProcAddress( hrasapiInstance, "RasDialA");
        rasgeterrorstring = (rasgeterrorstringT) GetProcAddress( hrasapiInstance, "RasGetErrorStringA");
        rasgetentrydialparams = (rasgetentrydialparamsT)GetProcAddress( hrasapiInstance, "RasGetEntryDialParamsA");
        rasenumentries = (rasenumentriesT)GetProcAddress( hrasapiInstance, "RasEnumEntriesA");
        if (!rasenumconnections || !rasgetconnectstatus || !rashangup || !rasdial || 
            !rasgeterrorstring || !rasgetentrydialparams || !rasenumentries )
        {
          rc = -1;
          doalloc = 0;
        }
      }
    }
  }

  if (!doalloc && rasapiinitialized > 0 /* sync error if <= 0 */)
  {
    if ((--rasapiinitialized) == 0)
    {
      if ( hrasapiInstance )
      {
        if (hrasapiInstance)
          FreeLibrary(hrasapiInstance);
        hrasapiInstance = NULL;
      }
    }
  }
  return rc;
}

static int __rasapi_deinit(void)
{
   return __rasapi_init(0);
}   

/* =============================================================== */

#if defined(DYNAMIC_RASAPI)
  static int InitRasAPIProcs(void)   { return __rasapi_init();   }
  static int DeinitRasAPIProcs(void) { return __rasapi_deinit(); }
#else
  static class __undyn_rasapi_class {
  public:
  int state;
  __undyn_rasapi_class() { state = 0; }
  ~__undyn_rasapi_class() { if (state>0) __rasapi_deinit(); }
  int getstate(void) { if (!state) state=((__rasapi_init()==0)?(+1):(-1)); 
                       return (state > 0 ? 0 : -1 ); }
  } __undyn_rasapi;
  static int InitRasAPIProcs(void)   { return __undyn_rasapi.getstate(); }
  static int DeinitRasAPIProcs(void) { return __undyn_rasapi.getstate(); }
#endif

/* =============================================================== */

DWORD WINAPI RasGetErrorStringA (UINT uErrorValue, LPSTR lpszErrorString, DWORD cBufSize)
{ 
  if (InitRasAPIProcs() == 0) /* *must be balanced if success* */
  {
    DWORD rc = rasgeterrorstring(uErrorValue,lpszErrorString,cBufSize);
    DeinitRasAPIProcs(); /* init succeeded, so balance */
    return rc;
  }
  if (uErrorValue == ERROR_RASMAN_CANNOT_INITIALIZE)
  {
    static char msg[] = "RAS is disabled or not installed";
    if (lpszErrorString == NULL)
      return ERROR_INVALID_PARAMETER;
    if (cBufSize < (sizeof(msg) + 1))
      return ERROR_INSUFFICIENT_BUFFER;
    lstrcpy(lpszErrorString, msg );
    return 0;
  }
  return ERROR_DLL_INIT_FAILED;
}

/* ------------------------------------------------------------------- */

DWORD WINAPI RasEnumConnectionsA (LPRASCONNA lprasconn, LPDWORD lpcb, LPDWORD lpcConnections )
{
  DWORD rc = ERROR_RASMAN_CANNOT_INITIALIZE;
  if (InitRasAPIProcs() == 0) /* *must be balanced if success* */
  {
    rc = rasenumconnections(lprasconn, lpcb, lpcConnections);
    DeinitRasAPIProcs(); /* init succeeded, so balance */
  }
  return rc;
}  
  
/* ------------------------------------------------------------------- */

DWORD WINAPI RasGetConnectStatusA (HRASCONN hrasconn,LPRASCONNSTATUSA lprasconnstatus)
{
  DWORD rc = ERROR_RASMAN_CANNOT_INITIALIZE;
  if (InitRasAPIProcs() == 0) /* *must be balanced if success* */
  {
    rc = rasgetconnectstatus(hrasconn, lprasconnstatus);
    DeinitRasAPIProcs(); /* init succeeded, so balance */
  }
  return rc;
}  

/* ------------------------------------------------------------------- */

DWORD WINAPI RasHangUpA ( HRASCONN hrasconn )
{
  DWORD rc = ERROR_RASMAN_CANNOT_INITIALIZE;
  if (InitRasAPIProcs() == 0) /* *must be balanced if success* */
  {
    rc = rashangup( hrasconn );
    DeinitRasAPIProcs(); /* init succeeded, so balance */
  }
  return rc;
}  

/* ------------------------------------------------------------------- */

DWORD WINAPI RasDialA (LPRASDIALEXTENSIONS lpRasDialExtensions,
        my_LPCSTR lpszPhonebook, LPRASDIALPARAMSA lpRasDialParams,
        DWORD dwNotifierType, LPVOID lpvNotifier, LPHRASCONN lphRasConn )
{
  DWORD rc = ERROR_RASMAN_CANNOT_INITIALIZE;
  if (InitRasAPIProcs() == 0) /* *must be balanced if success* */
  {
    rc = rasdial(lpRasDialExtensions,lpszPhonebook,lpRasDialParams,
                 dwNotifierType,lpvNotifier,lphRasConn);
    DeinitRasAPIProcs(); /* init succeeded, so balance */
  }
  return rc;
}  

/* ------------------------------------------------------------------- */

DWORD WINAPI RasGetEntryDialParamsA (my_LPCSTR lpszPhonebook, 
        LPRASDIALPARAMSA lprasdialparams,LPBOOL lpfPassword)
{
  DWORD rc = ERROR_RASMAN_CANNOT_INITIALIZE;
  if (InitRasAPIProcs() == 0) /* *must be balanced if success* */
  {
    rc = rasgetentrydialparams(lpszPhonebook, lprasdialparams, lpfPassword);
    DeinitRasAPIProcs(); /* init succeeded, so balance */
  }
  return rc;
}  

/* ------------------------------------------------------------------- */

DWORD WINAPI RasEnumEntriesA ( my_LPCSTR reserved, my_LPCSTR lpszPhonebook, 
     LPRASENTRYNAMEA lprasentryname, LPDWORD lpcb, LPDWORD lpcEntries )
{
  DWORD rc = ERROR_RASMAN_CANNOT_INITIALIZE;
  if (InitRasAPIProcs() == 0) /* *must be balanced if success* */
  {
    rc = rasenumentries( reserved, lpszPhonebook, lprasentryname, lpcb, lpcEntries );
    DeinitRasAPIProcs(); /* init succeeded, so balance */
  }
  return rc;
}  

/* ------------------------------------------------------------------- */
