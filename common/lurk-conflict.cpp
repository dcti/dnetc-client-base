// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#include "baseincs.h"
#include "lurk.h"

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

static rasenumconnectionsT rasenumconnections = NULL;
static rasgetconnectstatusT rasgetconnectstatus = NULL;
static rashangupT rashangup = NULL;
static rasdialT rasdial = NULL;
static rasgeterrorstringT rasgeterrorstring = NULL;
static rasgetentrydialparamsT rasgetentrydialparams = NULL;
#endif

s32 Lurk::CheckIfConnectRequested(void)
  // Returns the possible values of connectrequested
{
s32 connectrequested;
if (Status() == 0) // We're not connected
  {
  if(oldlurkstatus != 0)    // Lost the connection
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

s32 Lurk::Start(void)// Initializes Lurk Mode
  // 0 == Successfully started lurk mode
  // -1 == Start of lurk mode failed
{
#if (CLIENT_OS == OS_WIN32) && defined(MULTITHREAD)
  LPVOID lpMsgBuf;

  if (!rasenumconnections || !rasgetconnectstatus)
  {
    HINSTANCE hinstance;
    hinstance=LoadLibrary("RASAPI32.dll");
    if (hinstance == NULL)
    {
      LogScreen("Couldn't load rasapi32.dll\n");
      LogScreen("Dial-up must be installed for -lurk/-lurkonly\n");
      return -1;
    }
    rasenumconnections = (rasenumconnectionsT) GetProcAddress(hinstance,"RasEnumConnectionsA");
    if (rasenumconnections==NULL)
    {
      FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,GetLastError(),MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
        (LPTSTR) &lpMsgBuf,0,NULL);
      LogScreenf("%s\n",lpMsgBuf);
      LogScreen("Dial-up must be installed for -lurk/-lurkonly\n");
      LocalFree( lpMsgBuf );
      return -1;
    }
    rasgetconnectstatus = (rasgetconnectstatusT) GetProcAddress(hinstance,"RasGetConnectStatusA");
    if (rasgetconnectstatus==NULL)
    {
      FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,GetLastError(),MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
        (LPTSTR) &lpMsgBuf,0,NULL);
      LogScreenf("%s\n",lpMsgBuf);
      LogScreen("Dial-up must be installed for -lurk/-lurkonly\n");
      LocalFree( lpMsgBuf );
      return -1;
    }
    rashangup = (rashangupT) GetProcAddress(hinstance,"RasHangUpA");
    if (rashangup==NULL)
    {
      FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,GetLastError(),MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
        (LPTSTR) &lpMsgBuf,0,NULL);
      LogScreenf("%s\n",lpMsgBuf);
      LogScreen("Dial-up must be installed for -lurk/-lurkonly\n");
      LocalFree( lpMsgBuf );
      return -1;
    }
    rasdial = (rasdialT) GetProcAddress(hinstance,"RasDialA");
    if (rasdial==NULL)
    {
      FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,GetLastError(),MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
        (LPTSTR) &lpMsgBuf,0,NULL);
      LogScreenf("%s\n",lpMsgBuf);
      LogScreen("Dial-up must be installed for -lurk/-lurkonly\n");
      LocalFree( lpMsgBuf );
      return -1;
    }
    rasgeterrorstring = (rasgeterrorstringT) GetProcAddress(hinstance,"RasGetErrorStringA");
    if (rasgeterrorstring==NULL)
    {
      FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,GetLastError(),MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
        (LPTSTR) &lpMsgBuf,0,NULL);
      LogScreenf("%s\n",lpMsgBuf);
      LogScreen("Dial-up must be installed for -lurk/-lurkonly\n");
      LocalFree( lpMsgBuf );
      return -1;
    }
    rasgetentrydialparams = (rasgetentrydialparamsT)
      GetProcAddress(hinstance,"RasGetEntryDialParamsA");
    if (rasgetentrydialparams==NULL)
    {
      FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,GetLastError(),MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
        (LPTSTR) &lpMsgBuf,0,NULL);
      LogScreenf("%s\n",lpMsgBuf);
      LogScreen("Dial-up must be installed for -lurk/-lurkonly\n");
      LocalFree( lpMsgBuf );
      return -1;
    }

  }
#endif
islurkstarted=1;
return 0;
}

// ---------------------------------------------------------------------------

s32 Lurk::Status(void)// Checks status of connection
  // 0 == not currently connected
  // 1 == currently connected
{
if (islurkstarted != 1) Start();
if (islurkstarted != 1) return 0; // Lurk can't be started, evidently

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
            return 1;// We're connected
        }
      }
    }
  }
#elif (CLIENT_OS == OS_OS2) && defined(MULTITHREAD)
  DialOnDemand dod;       // just used to check online status for lurk
  return dod.rweonline();
#endif
return 0;// Not connected
}

s32 Lurk::InitiateConnection(void)
  // Initiates a dialup connection
  // 0 = already connected, 1 = connection started,
  // -1 = connection failed
{
if (islurkstarted != 1) Start();
if (islurkstarted != 1) return -1; // Lurk can't be started, evidently

if (Status() == 1) return 0; // We're already connected!

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

returnvalue=rasgetentrydialparams(NULL,&dialparameters,&passwordretrieved);

if (returnvalue==0)
  {
  if (passwordretrieved != TRUE) LogScreen("Password could not be found, connection may fail.\n");
  }
else
  switch(returnvalue)
    {
    case ERROR_CANNOT_FIND_PHONEBOOK_ENTRY:
      LogScreenf("Phonebook entry %s could not be found, aborting dial.\n",
                 connectionname);
      return -1;
    case ERROR_CANNOT_OPEN_PHONEBOOK:
      LogScreen("The phonebook cound not be opened, aborting dial.\n");
      return -1;
    case ERROR_BUFFER_INVALID:
      LogScreen("Invalid buffer passed, aborting dial.\n");
      return -1;
    };

LogScreenf("Phonebook entry %s found, dialing.\n",connectionname);
returnvalue=rasdial(NULL,NULL,&dialparameters,NULL,NULL,&connectionhandle);

if (returnvalue != 0)
  {  
  rasgeterrorstring(returnvalue,errorstring,sizeof(errorstring));
  LogScreenf("There was an error initiating a connection: %s\n",errorstring);
  return -1;
  };
return 1; // If we got here, connection successful.

#endif

return -1; // Can't dial on a platform that's not implemented
}

s32 Lurk::TerminateConnection(void)
  // -1 = connection did not terminate properly, 0 = connection
  // terminated
{
if (islurkstarted != 1) Start();
if (islurkstarted != 1) return -1; // Lurk can't be started, evidently

if (Status() == 0) return 0; // We're already disconnected

#if (CLIENT_OS == OS_WIN32) && defined(MULTITHREAD)
if (lurkmode && rasenumconnections && rasgetconnectstatus)
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
            // We're connected
            if (rashangup(rasconn[whichconn-1].hrasconn) == 0) // So kill it!
              return 0; // Successful hangup
            else return -1; // RasHangUp reported an error.
 
        }   
      }
    }
  }
return 0;
#else
return 0;
#endif


}


