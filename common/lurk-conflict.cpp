// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: lurk-conflict.cpp,v $
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
return "@(#)$Id: lurk-conflict.cpp,v 1.8 1998/11/01 20:35:11 cyp Exp $"; }
#endif

/* --------------------------------- */

#include "baseincs.h"
#include "lurk.h"
#include "logstuff.h"

/* -------------------------------- */

Lurk dialup;

#if (CLIENT_OS == OS_WIN32)
#include <windows.h>
#include <ras.h>
#include <raserror.h>

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

/* ---------------------------------------------------------- */

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

/* ---------------------------------------------------------- */

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

/* ---------------------------------------------------------- */
    
static FARPROC LoadRASAPIProc( const char *procname )
{
  if (rasapiinitialized > 0 && hrasapiInstance)
    return GetProcAddress( hrasapiInstance, procname );
  return NULL;
}  

/* ---------------------------------------------------------- */




#endif

/* ========================================================== */


Lurk::Lurk()  // Init lurk internal variables
{
  islurkstarted=0;
  oldlurkstatus=0;
}

/* ---------------------------------------------------------- */

char *Lurk::GetEntryList(long *finalcount)
{
#if (CLIENT_OS==OS_WIN32)
  {
  RASENTRYNAME rasentries[10];
  static char configentries[10][60];
  unsigned long buffersize;
  u32 entrycount;
  char *EntryList;

  if (islurkstarted != 1) 
    Start();
  if (islurkstarted != 1) 
    return NULL; // Lurk can't be started, evidently

  rasentries[0].dwSize=sizeof(RASENTRYNAME);

  buffersize=sizeof(rasentries);
  entrycount=0;

  rasenumentries(NULL,NULL,&rasentries[0],&buffersize,&entrycount);

  if (entrycount >= 1)
    {
    for (unsigned int temp=0;temp < entrycount;temp++)
        strncpy(&configentries[temp][0],&rasentries[temp].szEntryName[0], 60);
    }

  EntryList=&configentries[0][0];
  *finalcount=(int)entrycount;
  return EntryList;
  }
#else
  {
  *finalcount=0;
  return NULL;
  }
#endif
}

/* ---------------------------------------------------------- */

s32 Lurk::CheckIfConnectRequested(void) //Get possible values of connectrequested
{
  s32 connectrequested;

  if (lurkmode < 1) 
    return 0; // We're not supposed to lurk!

  if (Status() == 0) // We're not connected
    {
    if( oldlurkstatus != 0)    // Lost the connection
      {
      Log("\nDialup Connection Disconnected");
      oldlurkstatus = 0;// So we know next time through this loop
      if(lurkmode == 2) // Lurk-only mode
        {
        Log(" - Connections will not be initiated by the client.");
         // lurkonly needs a live connect - also, don't
         // interfere if offlinemode already ==1 or ==2
        connectrequested = 0; // cancel any connection requests
        };
      Log("\n");
      };
    }
  else // We're connected!
    {
    connectrequested=2;// Trigger an update
    if(oldlurkstatus != 1) // We previously weren't connected
      {
      // Only put out message the first time.
      Log("\nDialup Connection Detected\n");
      oldlurkstatus = 1;
      };
    };
  return connectrequested;
}

/* ---------------------------------------------------------- */

s32 Lurk::CheckForStatusChange(void)
{
  // Checks to see if we've suddenly disconnected
  // Return values:
  // 0 = connection has not changed or has just connected
  // -1 = we just lost the connection

  if ( (lurkmode < 1) && (dialwhenneeded < 1) )
    return 0; // We're not lurking.

  if (Status() < oldlurkstatus) 
    return -1;  // we got disconnected!
  return 0;
}

/* ---------------------------------------------------------- */

s32 Lurk::DialIfNeeded(s32 force)
{
  // Dials the connection if current parameters allow it.
  // Force values:
  // 0 = act normal
  // 1 = override lurk-only mode and connect anyway.
  // return values:
  // 0=Already connected or connect succeeded.
  // -1=There was an error, we're not connected.

  s32 returnvalue;

  if (lurkmode < 1) 
    return 0; // We're not supposed to lurk!

  if (Status() == 1) // We're already connected
    {
    dialstatus=0; // Make sure we won't hangup
    return 0;
    };

  // Ok, we're not connected, check if we should dial.

  // First, check if we're in lurk-only, and if we're allowed to dial.

  if (force == 0)        // No override
    {
    if (lurkmode == 2)   // lurk-only mode, we're not allowed to
      return -1;         // connect
    }

  if (dialwhenneeded==0) // We'll have to let auto-dial handle this.
    {
    dialstatus=0;
    return 0;
    }
  else if (dialwhenneeded==1) // Let's dial!
    {
    returnvalue=InitiateConnection(); // Go dial! Yeah!
    switch (returnvalue)
      {
      case -1: // Connection failed, error.
           dialstatus=0;
           return -1;
      case 0: // We were already connected.
           dialstatus=0;
           return 0;
       case 1: // We just made a connection.
           dialstatus=1; // We have to hangup later.
           return 0;
       };
    };
  return -1;  // I don't know how you'd get here, so it's an error.
}

/* ---------------------------------------------------------- */

s32 Lurk::HangupIfNeeded(void)
{
  if (dialwhenneeded != 1) 
    return 0; // We don't handle dialing
  if (Status() == 0) 
    return 0; // We're already disconnected
  if (dialstatus == 1) // We dialed, so we should hangup
    {
    dialstatus=0; // Make sure we don't hangup twice.
    TerminateConnection(); // Hangup.
    };
  return 0;
}

/* ---------------------------------------------------------- */

s32 Lurk::Start(void)// Initializes Lurk Mode. returns 0 on success.
{
  if (lurkmode < 1) 
    return -1; // We're not supposed to lurk!

  #if (CLIENT_OS == OS_WIN32)

  if (InitializeRASAPI() != 0)
    {
    LogScreen("Couldn't load rasapi32.dll\n"
              "Dial-up must be installed for -lurk/-lurkonly\n");
    lurkmode = 0;
    return -1;
    }

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
    LogScreen("%s\nDial-up must be installed for -lurk/-lurkonly\n",lpMsgBuf);
    LocalFree( lpMsgBuf );
    DeinitializeRASAPI();
    lurkmode = 0;
    return -1;
    }
  #endif
  
  islurkstarted=1;
  return 0;
}

/* ---------------------------------------------------------- */

s32 Lurk::Status(void)// Checks status of connection
{
  // 0 == not currently connected
  // 1 == currently connected

  if (lurkmode < 1) 
    return 1; // We're not supposed to lurk!

  if (islurkstarted != 1) 
    Start();
  if (islurkstarted != 1) 
    return 0; // Lurk can't be started, evidently

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
          if (rasgetconnectstatus((rasconn[whichconn-1]).hrasconn,
              &rasconnstatus) == 0 && rasconnstatus.rasconnstate == RASCS_Connected)
              return 1;// We're connected
          }
        }
      }
    }
#elif (CLIENT_OS == OS_OS2)
  DialOnDemand dod;       // just used to check online status for lurk
  return dod.rweonline();
#endif
return 0;// Not connected
}

/* ---------------------------------------------------------- */

// Initiates a dialup connection
// 0 = already connected, 1 = connection started,
// -1 = connection failed

s32 Lurk::InitiateConnection(void)
{
  if (lurkmode < 1) 
    return 0; // We're not supposed to lurk!

  if (islurkstarted != 1) 
    Start();
  if (islurkstarted != 1) 
    return -1; // Lurk can't be started, evidently

  if (Status() == 1) 
    return 0; // We're already connected!

  #if (CLIENT_OS == OS_WIN32)

  RASDIALPARAMS dialparameters;
  BOOL passwordretrieved;
  HRASCONN connectionhandle;
  DWORD returnvalue;
  char errorstring[128];

  dialparameters.dwSize=sizeof(RASDIALPARAMS);
  strcpy(dialparameters.szEntryName,connectionname);
  strcpy(dialparameters.szPhoneNumber,"");
  strcpy(dialparameters.szCallbackNumber,"*");
  strcpy(dialparameters.szUserName,"");
  strcpy(dialparameters.szPassword,"");
  strcpy(dialparameters.szDomain,"*");

  returnvalue = 
    rasgetentrydialparams(NULL,&dialparameters,&passwordretrieved);

  if (returnvalue==0)
    {
    if (passwordretrieved != TRUE) 
      LogScreen("Password could not be found, connection may fail.\n");
    }
  else
    {
    switch(returnvalue)
      {
      case ERROR_CANNOT_FIND_PHONEBOOK_ENTRY:
        LogScreen("Phonebook entry %s could not be found, aborting dial.\n",
                   connectionname);
        return -1;
      case ERROR_CANNOT_OPEN_PHONEBOOK:
        LogScreen("The phonebook cound not be opened, aborting dial.\n");
        return -1;
      case ERROR_BUFFER_INVALID:
        LogScreen("Invalid buffer passed, aborting dial.\n");
        return -1;
      };  
    }

  LogScreen("Dialing phonebook entry %s...\n",connectionname);
  returnvalue=rasdial(NULL,NULL,&dialparameters,NULL,NULL,&connectionhandle);

  if (returnvalue == 0)
    return 1; //If we got here, connection successful.
  
  rasgeterrorstring(returnvalue,errorstring,sizeof(errorstring));
  LogScreen("There was an error initiating a connection: %s\n",errorstring);
  #endif

  return -1; //failed
}

s32 Lurk::TerminateConnection(void)
  // -1 = connection did not terminate properly, 0 = connection
  // terminated
{
  if (lurkmode < 1) 
    return 0; // We're not supposed to lurk!

  if (islurkstarted != 1) 
    Start();
  if (islurkstarted != 1) 
    return -1; // Lurk can't be started, evidently

  if (Status() == 0) 
    return 0; // We're already disconnected

  #if (CLIENT_OS == OS_WIN32) && defined(MULTITHREAD)
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
          if (rasgetconnectstatus((rasconn[whichconn-1]).hrasconn,
              &rasconnstatus) == 0 && rasconnstatus.rasconnstate == RASCS_Connected)
            {
            // We're connected
            if (rashangup(rasconn[whichconn-1].hrasconn) == 0) // So kill it!
              return 0; // Successful hangup
            return -1; // RasHangUp reported an error.
            }
          }   
        }
      }
    }
  #endif  
  return 0;
}
  
