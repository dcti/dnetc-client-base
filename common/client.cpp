// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: client.cpp,v $
// Revision 1.152.2.6  1998/11/18 15:41:51  remi
// Cleaned-up the mess I put in the last night... sorry folks !
//
// Revision 1.152.2.5  1998/11/17 09:23:03  remi
// Synced with :
//  Revision 1.164  1998/11/17 04:39:33  silby
//  Gave GetBuildOrEnvDescription the fixing it was pining for.
//
// Revision 1.152.2.4  1998/11/17 00:01:42  remi
// Synced with :
//  Revision 1.163  1998/11/16 22:31:09  cyp
//  Cleaned up banner(s) and made use of CLIENT_OS_NAME.
//
//  Revision 1.156  1998/11/11 05:26:57  cramer
//  Various minor fixes...
//
// Revision 1.152.2.3  1998/11/11 03:07:42  remi
// Synced with :
//  Revision 1.154  1998/11/10 21:51:32  cyp
//  Completely reorganized Client::Main() initialization order so that (a) the
//  client object is completely initialized (both ini file and cmdline) before
//  anything else, (b) whether the client is running "modes" is known from the
//  beginning (c) Single instance protection can occur conditionally (ie only
//  if the client will not be running "modes").
//
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

static const char *GetBuildOrEnvDescription(void)
{
  /*
  <cyp> hmm. would it make sense to have the client print build data 
  when starting? running OS, static/dynamic, libver etc? For idiots who
  send us log extracts but don't mention the OS they are running on.
  Only useful for win-weenies I suppose. 
  */

  #if ((CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S))
  static char buffer[64];
  int major, minor;
  w32ConGetWinVersion(&major,&minor);
  sprintf(buffer,"Running under Windows%s %u.%u", (major>20)?(" NT"):(""), major%20, minor );
  return buffer;
  #else
  return "";
  #endif
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
               "Copyright 1997-1998 distributed.net\n" );
    
    #if (CLIENT_CPU == CPU_68K)
    LogScreenRaw( "RC5 68K assembly by John Girvin\n");
    #endif
    #if (CLIENT_CPU == CPU_POWERPC)
    LogScreenRaw( "PowerPC assembly by Dan Oetting at USGS\n");
    #endif
    #if defined(KWAN) && defined(MEGGS)
    LogScreenRaw( "DES bitslice driver Copyright 1997-1998, Andrew Meggs\n" 
                  "DES sboxes routines Copyright 1997-1998, Matthew Kwan\n" );
    #elif defined(KWAN) 
    LogScreenRaw( "DES search routines Copyright 1997-1998, Matthew Kwan\n" );
    #endif
    #if (CLIENT_CPU == CPU_X86)
    LogScreenRaw( "DES search routines Copyright 1997-1998, Svend Olaf Mikkelsen\n");
    #endif
    #if (CLIENT_OS == OS_DOS)  
    LogScreenRaw( "PMODE DOS extender Copyright 1994-1998, Charles Scheffold and Thomas Pytel\n");
    #endif
    LogScreenRaw( "Please visit http://www.distributed.net/ for up-to-date contest information.\n" );
    #if (CLIENT_OS == OS_DOS)
      dosCliCheckPlatform(); //show warning if pure DOS client is in win/os2 VM
    #endif

    #if ((CLIENT_OS==OS_DOS) || (CLIENT_OS==OS_WIN16) || \
         (CLIENT_OS==OS_WIN32S) || (CLIENT_OS==OS_OS2) || \
         ((CLIENT_OS==OS_WIN32) && !defined(NEEDVIRTUALMETHODS)))
    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
    int major=0;
    w32ConGetWinVersion(&major,NULL);
    if ((major%20) <= 3) /* >=20 == NT */
    #endif   
    if (getenv("TZ") == NULL)
      {
      LogScreenRaw("Warning: The TZ= variable is not set in the environment. "
       "The client will\nprobably display the wrong time and/or select the "
       "wrong keyserver.\n");
      putenv("TZ=GMT+0"); //use local time.
      }
    #endif
    }
  
  if ( level == 1 )
    {  
    level++; //will never print this message again
    LogRaw("\nRC5DES Client %s for %s started.\n", CLIENT_VERSIONSTRING,
                                                   CLIENT_OS_NAME );
    const char *msg = GetBuildOrEnvDescription();
    if (msg && *msg) LogRaw( "%s\n", msg );

    }
  return;
}

//------------------------------------------------------------------------

int Client::Main( int argc, const char *argv[], int /* restarted */ )
{
  int retcode = 0;
  int domodes = 0;

  ModeReqSet( MODEREQ_CMDLINE_HELP );

  //ReadConfig() and parse command line - returns !0 if shouldn't continue
  if (ParseCommandline( 0, argc, argv, &retcode, 0 ) == 0)
    {
    domodes = (ModeReqIsSet(-1) != 0);
    if (InitializeTriggers(NULL,NULL)==0)
      {
      if (InitializeConsole(0,domodes) == 0)
        {
        InitializeLogging(0); //enable only screen logging for now
        PrintBanner(); //tracks restart state itself
        ParseCommandline( 1, argc, argv, NULL, 1 ); //show cmdline overrides
      
        if (domodes)
          {
          ModeReqRun( this );     
          }
        DeinitializeLogging();
        DeinitializeConsole();
      }
      DeinitializeTriggers();
      }
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

  return (retcode);
}


/* ----------------------------------------------------------------- */

int main( int argc, char *argv[] )
{ 
  return realmain( argc, argv ); 
}
