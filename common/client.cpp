// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: client.cpp,v $
// Revision 1.152.2.2  1998/11/08 11:50:22  remi
// Lots of $Log tags.
//
// Sync with :
//   Revision 1.152  1998/11/08 01:01:41  silby
//   Buncha hacks to get win32gui to compile, lots of cleanup to do.
//
// Synchronized with official 1.151

// --------------------------------------------------------------------------

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "client.h"    // Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "triggers.h"  // RestartRequestTrigger()
#include "clitime.h"   // CliTimer(), Time()/(CliGetTimeString(NULL,1))
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "console.h"   // [De]InitializeConsole(), ConOutErr()
#include "modereq.h"   // ModeReqIsSet()/ModeReqRun()

// --------------------------------------------------------------------------

#if (CLIENT_OS == OS_AMIGAOS)
#if (CLIENT_CPU == CPU_68K)
long __near __stack  = 65536L;  // AmigaOS has no automatic stack extension
      // seems standard stack isn't enough
#endif // (CLIENT_CPU == CPU_68K)
#endif // (CLIENT_OS == OS_AMIGAOS)

#if (CLIENT_OS == OS_RISCOS)
s32 guiriscos, guirestart;
#endif

// --------------------------------------------------------------------------

Client::Client()
{
  totalBlocksDone[0] = totalBlocksDone[1] = 0;
  cputype=-1;
  srand( (unsigned) CliTimer( NULL )->tv_usec );
}

// --------------------------------------------------------------------------

static void PrintBanner(void)
{
  static unsigned int level = 0; //incrementing so messages are not repeated
            //0 = show copyright/version,  1 = show startup message
  
  if (level == 0)
    {
    level++; //will never print this message again

    LogScreenRaw( "\nRC5DES " CLIENT_VERSIONSTRING 
               " client - a project of distributed.net\n"
               "Copyright distributed.net 1997-1998\n" );
    #if (CLIENT_CPU == CPU_68K)
    LogScreenRaw( "RC5 68K assembly by John Girvin\n");
    #endif
    #if (CLIENT_CPU == CPU_POWERPC)
    LogScreenRaw( "PowerPC assembly by Dan Oetting at USGS\n");
    #endif
    #if defined(KWAN) && defined(MEGGS)
    LogScreenRaw( "DES bitslice driver Copyright Andrew Meggs\n" 
                  "DES sboxes routines Copyright Matthew Kwan\n" );
    #elif defined(KWAN) 
    LogScreenRaw( "DES search routines Copyright Matthew Kwan\n" );
    #endif
    #if (CLIENT_CPU == CPU_X86)
    LogScreenRaw( "DES search routines Copyright Svend Olaf Mikkelsen\n");
    #endif
    #if (CLIENT_OS == OS_DOS)  //PMODE (c) string if not win16 
    LogScreenRaw( "%s", dosCliGetPmodeCopyrightMsg() );
    #endif
    LogScreenRaw( "Please visit http://www.distributed.net/ for up-to-date contest information.\n" );
    #if (CLIENT_OS == OS_DOS)
      dosCliCheckPlatform(); //show warning if pure DOS client is in win/os2 VM
    #endif
    }
  
  if ( level == 1 )
    {  
    level++; //will never print this message again

    LogRaw("\nRC5DES Client %s started.\n\n",
         CLIENT_VERSIONSTRING);
    }
  return;
}

//------------------------------------------------------------------------

int Client::Main( int argc, const char *argv[], int restarted )
{
  int retcode = 0;

  //set up break handlers
  if (InitializeTriggers(NULL, NULL)==0) //CliSetupSignals();
    {
    //get -ini options/defaults, then ReadConfig(), then get -quiet/-hidden
    if (ParseCommandline( 0, argc, argv, &retcode, 0 ) == 0) //!0 if "mode"
      {                                                
      if (InitializeConsole(0) == 0) //initialize conio
        {
        int autoclosecon = 1; // let console close without user intervention
        InitializeLogging(0); //enable only screen logging for now

        PrintBanner(); //tracks restart state itself

        //get remaining option overrides and set "mode" bits if applicable
        ParseCommandline( 2, argc, argv, &retcode, 1 );
        if (ModeReqIsSet( -1 ) == 0)
	  ModeReqSet( MODEREQ_HELP );
	ModeReqRun( this );     
	autoclosecon = 0; //wait for a keypress before closing the console

        DeinitializeLogging();
        DeinitializeConsole(autoclosecon);
        }
      }
    DeinitializeTriggers();
    }
  return retcode;
}  

// --------------------------------------------------------------------------

int realmain( int argc, char *argv[] )
{
  // This is the main client object.  we 'new'/malloc it, rather than make 
  // it static in the hope that people will think twice about using exit()
  // or otherwise breaking flow. (wanna bet it'll happen anyway?)
  // The if (success) thing is for nesting without {} nesting.
  Client *clientP = NULL;
  int retcode = -1, init_success = 1;
  int restarted = 0;
  
  //------------------------------
  
  #if (CLIENT_OS == OS_RISCOS)
  if (init_success) //protect ourselves
    {
    riscos_in_taskwindow = riscos_check_taskwindow();
    if (riscos_find_local_directory(argv[0])) 
      init_success = 0;
    }
  #endif

  //-----------------------------

  #if (CLIENT_OS == OS_WIN32)
  HANDLE hmutex = NULL;
  if (init_success) 
    {
    SetLastError(0); // only allow one running instance
    hmutex = CreateMutex(NULL, TRUE, "Bovine RC5/DES Win32 Client");
    init_success = (GetLastError() == 0);
    }
  #endif

  //-----------------------------

  if ( init_success )
    {
    init_success = (( clientP = new Client() ) != NULL);
    if (!init_success) 
      ConOutErr( "Unable to create client object. Out of memory." );
    }

  //----------------------------

  #if (CLIENT_OS == OS_NETWARE) 
  //create stdout/screen, set cwd etc. save ptr to client for fnames/niceness
  if ( init_success )
    init_success = ( nwCliInitClient( argc, argv, clientP ) == 0);
  #endif

  //----------------------------

  if ( init_success )
    {
    do {
       retcode = clientP->Main( argc, (const char **)argv, restarted );
       restarted = 1; //for the next round
       } while ( CheckRestartRequestTrigger() );
    }

  //------------------------------

  #if (CLIENT_OS == OS_AMIGAOS)
  if (retcode) retcode = 5; // 5 = Warning
  #endif // (CLIENT_OS == OS_AMIGAOS)

  //------------------------------
  
  #if (CLIENT_OS == OS_NETWARE)
  if (init_success)
    nwCliExitClient(); // destroys AES process, screen, polling procedure
  #endif

  //------------------------------
  
  if (clientP)
    delete clientP;

  #if (CLIENT_OS == OS_WIN32)
  if (hmutex)
    ReleaseMutex( hmutex );
  #endif  

  return (retcode);
}


/* ----------------------------------------------------------------- */

#if !((CLIENT_OS==OS_WIN32) && defined(NEEDVIRTUALMETHODS))

#if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
int PASCAL WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpszCmdLine, 
    int nCmdShow) 
{ /* abstraction layer between WinMain() and realmain() */
  return winClientPrelude( hInst, hPrevInst, lpszCmdLine, nCmdShow, realmain);
}
#else
int main( int argc, char *argv[] )
{ 
  return realmain( argc, argv ); 
}
#endif
#endif
