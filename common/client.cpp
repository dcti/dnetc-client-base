// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: client.cpp,v $
// Revision 1.139  1998/09/28 21:42:07  remi
// Cleared a warning in InitConsole. Wrapped $Log comments.
//
// Revision 1.138  1998/09/28 04:13:08  cyp
// Split: clirun.cpp, bench.cpp, setprio.cpp, probfill.cpp; client.cpp is now
// startup code only; bugs fixed here (client.cpp): win32 client now does
// run-once check properly; win32 console client no longer opens a console
// until it knows -hide is not an option; mail is no longer written if a bad
// arg is passed; PrintBanner() will no longer print banners when restarting.
//
// Revision 1.137  1998/09/23 22:26:42  silby
// Changed checkifbetaexpired from s32 to int
//
// Revision 1.136  1998/09/19 08:50:15  silby
// Added in beta test client timeouts.  Enabled/controlled from
// version.h by defining BETA, and setting the expiration time.
//
// Revision 1.135  1998/08/28 22:28:12  cyp
// Restructured main() so that it is now restartable. Command line is
// reusable (is no longer overwritten).
//
// Revision 1.134  1998/08/24 23:41:20  cyp
// Saves and restores the state of 'offlinemode' around -fetch/-flush to
// suppress undesirable attempts to send mail when the client exits.
//
// Revision 1.133  1998/08/24 04:56:26  cyruspatel
// enforced rc5 fileentry cpu/os/build checks for all platforms, not just x86.
//
// Revision 1.132  1998/08/21 18:18:22  cyruspatel
// Failure to start a thread will no longer force a client to exit. ::Run
// will continue with a reduced number of threads or switch to non-threaded
// mode if no threads could be started. Loaded but unneeded blocks are
// written back out to disk. A multithread-capable client can still be forced
// to run in non-threaded mode by setting numcpu=0.
//
// Revision 1.131  1998/08/21 16:05:51  cyruspatel
// Extended the DES mmx define wrapper from #if MMX_BITSLICER to
// #if (defined(MMX_BITSLICER) && defined(KWAN) && defined(MEGGS)) to
// differentiate between DES and RC5 MMX cores. Partially completed
// blocks are now also tagged with the core type and CLIENT_BUILD_FRAC
//
// Revision 1.130  1998/08/21 09:05:42  cberry
// Fixed block size suggestion for CPUs so slow that they can't do a 2^28 block in an hour.
//
// Revision 1.129  1998/08/20 19:34:34  cyruspatel
// Removed that terrible PIPELINE_COUNT hack: Timeslice and pipeline count
// are now computed in Problem::LoadState(). Client::SelectCore() now saves
// core type to Client::cputype.
//
// Revision 1.128  1998/08/20 03:48:59  silby
// Quite hack to get winnt service compiling.
//
// Revision 1.127  1998/08/20 02:40:34  silby
// Kicked version to 2.7100.418-BETA1, ensured that clients report the
// string ver (which has beta1 in it) in the startup.
//
// Revision 1.126  1998/08/16 06:00:28  silby
// Changed ::Update back so that it checks contest/buffer status
// before connecting (lurk connecting every few seconds wasn't
// pretty.)
// Also, changed command line option handling so that update() would
// be called with force so that it would connect over all.
//
// Revision 1.125  1998/08/15 21:32:49  jlawson
// added parens around an abiguous shift operation.
//
// Revision 1.124  1998/08/14 00:04:53  silby
// Changes for rc5 mmx core integration.
//
// Revision 1.123  1998/08/13 00:24:17  silby
// Change to a NOMAIN definition so that the win32gui will compile.
//
// Revision 1.122  1998/08/10 23:02:12  cyruspatel
// xxxTrigger and pausefilefound flags are now wrapped in functions in 
// trigger.cpp. NetworkInitialize()/NetworkDeinitialize() related changes.
//
// Revision 1.121  1998/08/08 00:55:25  silby
// Changes to get win32gui working again
//
// Revision 1.120  1998/08/07 20:35:31  cyruspatel
// NetWare specific change: Fixed broken IsNetworkAvailable() test
//
// Revision 1.119  1998/08/07 18:01:38  cyruspatel
// Modified Fetch()/Flush() and Benchmark() to display normalized blocksizes
// (ie 4*2^28 versus 1*2^30). Also added some functionality to Benchmark()
// to assist users in selecting a 'preferredblocksize' and hint at what
// sensible max/min buffer thresholds might be.
//
// Revision 1.118  1998/08/07 10:59:11  cberry
// Changed handling of -benchmarkXXX so it performs the benchmark rather 
// than giving the menu.
//
// Revision 1.117  1998/08/05 18:28:40  cyruspatel
// Converted more printf()s to LogScreen()s, changed some Log()/LogScreen()s
// to LogRaw()/LogScreenRaw()s, ensured that DeinitializeLogging() is called,
// and InitializeLogging() is called only once (*before* the banner is shown)
//
// Revision 1.116  1998/08/02 16:17:37  cyruspatel
// Completed support for logging.
//
// Revision 1.115  1998/08/02 05:36:19  silby
// Lurk functionality is now fully encapsulated inside the Lurk Class,
// much less code floating inside client.cpp now.
//
// Revision 1.114  1998/08/02 03:16:31  silby
// Major reorganization:  Log,LogScreen, and LogScreenf 
// are now in logging.cpp, and are global functions - 
// client.h #includes logging.h, which is all you need to use those
// functions.  Lurk handling has been added into the Lurk class, which 
// resides in lurk .cpp, and is auto-included by client.h if lurk is 
// defined as well. baseincs.h has had lurk-specific win32 includes moved
// to lurk.cpp, cliconfig.cpp has been modified to reflect the changes to 
// log/logscreen/logscreenf, and mail.cpp uses logscreen now, instead of 
// printf. client.cpp has had variable names changed as well, etc.
//
// Revision 1.113  1998/07/30 05:08:59  silby
// Fixed DONT_USE_PATHWORK handling, ini_etc strings were still being
// included, now they are not. Also, added the logic for
// dialwhenneeded, which is a new lurk feature.
//
// Revision 1.112  1998/07/30 02:18:18  blast
// AmigaOS update
//
// Revision 1.111  1998/07/29 05:14:40  silby
// Changes to win32 so that LurkInitiateConnection now works -
// required the addition of a new .ini key connectionname=.  Username
// and password are automatically retrieved based on the
// connectionname.
//
// Revision 1.110  1998/07/26 12:45:52  cyruspatel
// new inifile option: 'autofindkeyserver', ie if keyproxy= points to a
// xx.v27.distributed.net then that will be interpreted by Network::Resolve()
// to mean 'find a keyserver that covers the timezone I am in'. Network
// constructor extended to take this as an argument.
//
// Revision 1.109  1998/07/25 06:31:39  silby
// Added lurk functions to initiate a connection and hangup a
// connection.  win32 hangup is functional.
//
// Revision 1.108  1998/07/25 05:29:49  silby
// Changed all lurk options to use a LURK define (automatically set in
// client.h) so that lurk integration of mac/amiga clients needs only
// touch client.h and two functions in client.cpp
//
// Revision 1.107  1998/07/20 00:32:19  silby
// Changes to facilitate 95 CLI/NT service integration
//
// Revision 1.106  1998/07/19 14:42:12  cyruspatel
// NetWare SMP adjustments
//
// Revision 1.105  1998/07/16 19:19:36  remi
// Added -cpuinfo option (you forget this one cyp! :-)
//
// Revision 1.104  1998/07/16 16:58:58  silby
// x86 clients in MMX mode will now permit des on > 2 processors.
// Bryddes is still set at two, however.
//
// Revision 1.103  1998/07/16 08:25:07  cyruspatel
// Added more NO!NETWORK wrappers around calls to Update/Fetch/Flush. Balanced
// the '{' and '}' in Fetch and Flush. Also, Flush/Fetch will now end with
// 100% unless there was a real send/retrieve fault.
//
// Revision 1.101  1998/07/15 06:58:03  silby
// Changes to Flush, Fetch, and Update so that when the win32 gui sets
// connectoften to initiate one of the above more verbose feedback
// will be given.  Also, when force=1, a connect will be made
// regardless of offlinemode and lurk.
//
// Revision 1.100  1998/07/15 06:10:54  silby
// Fixed an improper #ifdef
//
#if (!defined(lint) && defined(__showids__))
const char *client_cpp(void) {
return "@(#)$Id: client.cpp,v 1.139 1998/09/28 21:42:07 remi Exp $"; }
#endif

// --------------------------------------------------------------------------

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "client.h"    // Packet, FileHeader, Client class, etc
#include "scram.h"     // InitRandom() 
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "pathwork.h"  // EXTN_SEP
#include "triggers.h"  // RestartRequestTrigger()
#include "clitime.h"   //CliTimer(), Time()/(CliGetTimeString(NULL,1))
#define Time() (CliGetTimeString(NULL,1))
#include "logstuff.h"  //Log()/LogScreen()/LogScreenPercent()/LogFlush()

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
  id[0] = 0;
  inthreshold[0] = 10;
  outthreshold[0] = 10;
  inthreshold[1] = 10;
  outthreshold[1] = 10;
  blockcount = 0;
  minutes = 0;
  strcpy(hours,"0.0");
  keyproxy[0] = 0;
  keyport = 2064;
  httpproxy[0] = 0;
  httpport = 80;
  uuehttpmode = 1;
  strcpy(httpid,"");
  totalBlocksDone[0] = totalBlocksDone[1] = 0;
  timeStarted = 0;
  cputype=-1;
  offlinemode = 0;
  autofindkeyserver = 1;  //implies 'only if keyproxy==dnetkeyserver'

#ifdef DONT_USE_PATHWORK
  strcpy(ini_logname, "none");
  strcpy(ini_in_buffer_file[0], "buff-in" EXTN_SEP "rc5");
  strcpy(ini_out_buffer_file[0], "buff-out" EXTN_SEP "rc5");
  strcpy(ini_in_buffer_file[1], "buff-in" EXTN_SEP "des");
  strcpy(ini_out_buffer_file[1], "buff-out" EXTN_SEP "des");
  strcpy(ini_exit_flag_file, "exitrc5" EXTN_SEP "now");
  strcpy(ini_checkpoint_file[0],"none");
  strcpy(ini_checkpoint_file[1],"none");
  strcpy(ini_pausefile,"none");

  strcpy(logname, "none");
  strcpy(inifilename, InternalGetLocalFilename("rc5des" EXTN_SEP "ini"));
  strcpy(in_buffer_file[0], InternalGetLocalFilename("buff-in" EXTN_SEP "rc5"));
  strcpy(out_buffer_file[0], InternalGetLocalFilename("buff-out" EXTN_SEP "rc5"));
  strcpy(in_buffer_file[1], InternalGetLocalFilename("buff-in" EXTN_SEP "des"));
  strcpy(out_buffer_file[0], InternalGetLocalFilename("buff-out" EXTN_SEP "des"));
  strcpy(exit_flag_file, InternalGetLocalFilename("exitrc5" EXTN_SEP "now"));
  strcpy(checkpoint_file[1],"none");
  strcpy(pausefile,"none");
#else
  strcpy(logname, "none");
  strcpy(inifilename, "rc5des" EXTN_SEP "ini");
  strcpy(in_buffer_file[0], "buff-in" EXTN_SEP "rc5");
  strcpy(out_buffer_file[0], "buff-out" EXTN_SEP "rc5");
  strcpy(in_buffer_file[1], "buff-in" EXTN_SEP "des");
  strcpy(out_buffer_file[1], "buff-out" EXTN_SEP "des");
  strcpy(exit_flag_file, "exitrc5" EXTN_SEP "now");
  strcpy(checkpoint_file[1],"none");
  strcpy(pausefile,"none");
#endif
  messagelen = 0;
  smtpport = 25;
  strcpy(smtpsrvr,"your.smtp.server");
  strcpy(smtpfrom,"RC5Notify");
  strcpy(smtpdest,"you@your.site");
  numcpu = -1;
  numcputemp=1;
  strcpy(checkpoint_file[0],"none");
  checkpoint_min=5;
  percentprintingoff=0;
  connectoften=0;
  nodiskbuffers=0;
  membuffcount[0][0]=0;
  membuffcount[1][0]=0;
  membuffcount[0][1]=0;
  membuffcount[1][1]=0;
  for (int i1=0;i1<2;i1++) {
    for (int i2=0;i2<500;i2++) {
      for (int i3=0;i3<2;i3++) {
        membuff[i1][i2][i3]=NULL;
      }
    }
  }
  nofallback=0;
  randomprefix=100;
  preferred_contest_id = 1;
  preferred_blocksize=30;
  randomchanged=0;
  consecutivesolutions[0]=0;
  consecutivesolutions[1]=0;
  quietmode=0;
  nonewblocks=0;
  nettimeout=60;
  noexitfilecheck=0;
  exitfilechecktime=30;
  runhidden=0;
#if defined(LURK)
  dialup.lurkmode=0;
  dialup.dialwhenneeded=0;
#endif
  contestdone[0]=contestdone[1]=0;
  srand( (unsigned) CliTimer( NULL )->tv_usec );
  InitRandom();
#ifdef MMX_BITSLICER
  usemmx = 1;
#endif
}


// --------------------------------------------------------------------------

static void PrintBanner(const char *dnet_id)
{
  static unsigned int level = 0; //incrementing so messages are not repeated
            //0 = show copyright/version,  1 = show startup message
  
  if (level == 0)
    {
    level++; //will never print this message again

    LogScreenRaw( "\nRC5DES " CLIENT_VERSIONSTRING 
               " client - a project of distributed.net\n"
               "Copyright distributed.net 1997-1998\n" );
    #if defined(KWAN)
    #if defined(MEGGS)
    LogScreenRaw( "DES bitslice driver Copyright Andrew Meggs\n" 
               "DES sboxes routines Copyright Matthew Kwan\n" );
    #else
    LogScreenRaw( "DES search routines Copyright Matthew Kwan\n" );
    #endif
    #endif
    #if (CLIENT_CPU == CPU_X86)
    LogScreenRaw( "DES search routines Copyright Svend Olaf Mikkelsen\n");
    #endif
    #if (CLIENT_OS == OS_DOS)  //PMODE (c) string if not win16 
    LogScreenRaw( "%s", dosCliGetPmodeCopyrightMsg() );
    #endif
    LogScreenRaw( "Please visit http://www.distributed.net/ for up-to-date contest information.\n"
               "%s\n",
            #if (CLIENT_OS == OS_RISCOS)
            guiriscos ?
            "Interactive help is available, or select 'Help contents' from the menu for\n"
            "detailed client information.\n" :
            #endif
            "Execute with option '-help' for online help, or read rc5des" EXTN_SEP "txt\n"
            "for a list of command line options.\n"
            );
    #if (CLIENT_OS == OS_DOS)
      dosCliCheckPlatform(); //show warning if pure DOS client is in win/os2 VM
    #endif
    }
  
  if ( level == 1 )
    {  
    level++; //will never print this message again

    LogRaw("\nRC5DES Client v2.%d.%d started.\n"
             "Using distributed.net ID %s\n\n",
         CLIENT_CONTEST*100+CLIENT_BUILD,CLIENT_BUILD_FRAC,dnet_id);
    }

  return;
}

// --------------------------------------------------------------------------

#if (CLIENT_OS == OS_WIN32) && defined(WINNTSERVICE)
#define NTSERVICEID "rc5desnt"
#include <winsvc.h>

static SERVICE_STATUS_HANDLE serviceStatusHandle;

void __stdcall ServiceCtrlHandler(DWORD controlCode)
{
  // update our status to stopped
  SERVICE_STATUS serviceStatus;
  serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
  serviceStatus.dwWin32ExitCode = NO_ERROR;
  serviceStatus.dwServiceSpecificExitCode = 0;
  serviceStatus.dwCheckPoint = 0;
  if (controlCode == SERVICE_CONTROL_SHUTDOWN ||
      controlCode == SERVICE_CONTROL_STOP)
  {
    serviceStatus.dwCurrentState = SERVICE_STOP_PENDING;
    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwWaitHint = 10000;
    RaiseExitRequestTrigger();
  } else {
    // SERVICE_CONTROL_INTERROGATE
    serviceStatus.dwCurrentState = SERVICE_RUNNING;
    serviceStatus.dwWaitHint = 0;
  }
  SetServiceStatus(serviceStatusHandle, &serviceStatus);
}
#endif

// ---------------------------------------------------------------------------

#if (CLIENT_OS == OS_WIN32) && defined(WINNTSERVICE)
static Client *mainclient; 

void ServiceMain(DWORD Argc, LPTSTR *Argv)
{
  SERVICE_STATUS serviceStatus;
  
  serviceStatusHandle = RegisterServiceCtrlHandler(NTSERVICEID,
                                              ServiceCtrlHandler);

  // update our status to running
  serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
  serviceStatus.dwCurrentState = SERVICE_RUNNING;
  serviceStatus.dwControlsAccepted = (SERVICE_ACCEPT_SHUTDOWN | SERVICE_ACCEPT_STOP);
  serviceStatus.dwWin32ExitCode = NO_ERROR;
  serviceStatus.dwServiceSpecificExitCode = 0;
  serviceStatus.dwCheckPoint = 0;
  serviceStatus.dwWaitHint = 0;
  SetServiceStatus(serviceStatusHandle, &serviceStatus);

  mainclient = new Client();
  if (mainclient == NULL)
    {
    MessageBox( NULL, "Unable to initialize client.\n",
        "RC5DES", MB_OK | MB_TASKMODAL);
    }
  else
    {
    mainclient->Main( (int)(Argc), (const char **)Argv, -1 ); //restarted == -1
    delete mainclient;
    }

  // update our status to stopped
  serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
  serviceStatus.dwCurrentState = SERVICE_STOPPED;
  serviceStatus.dwControlsAccepted = 0;
  serviceStatus.dwWin32ExitCode = NO_ERROR;
  serviceStatus.dwServiceSpecificExitCode = 0;
  serviceStatus.dwCheckPoint = 0;
  serviceStatus.dwWaitHint = 0;
  SetServiceStatus(serviceStatusHandle, &serviceStatus);
}
#endif

// ---------------------------------------------------------------------

static int RunShutdown(void) { return 0; }


// do stuff before calling Run() for the first time
// returns: non-zero on failure

static int RunStartup(int restarted)
{
  int retcode = 0;

  if (restarted == 0)
    {
    #if (CLIENT_OS == OS_WIN32) && defined(WINNTSERVICE)
      {
      LogScreen("Attempting to start up NT service.\n");
      //mainclient = this;  //ignored - service must create a new one
      SERVICE_TABLE_ENTRY serviceTable[] = {
        {NTSERVICEID, (LPSERVICE_MAIN_FUNCTION) ServiceMain},
        {NULL, NULL}};
      if (!StartServiceCtrlDispatcher(serviceTable))
        {
        LogScreen("Error starting up NT service.  Please remember that this\n"
           "client cannot be invoked directly.  If you wish to install it\n"
           "as a service, use the -install option\n");
        }
      retcode = -1; //always -1
      }
    #elif ((CLIENT_OS == OS_WIN32) && (!defined(WINNTSERVICE)))
      {
      OSVERSIONINFO osver;
      osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
      GetVersionEx(&osver);

      //check if we are registered as a w9x "service" (survive logouts)
      if (VER_PLATFORM_WIN32_NT != osver.dwPlatformId)
        {
        HKEY srvkey = NULL;
        int run_as_w9x_service = 0;
        #ifdef W9x_ALWAYS_RUN_AS_SERVICE
        run_as_w9x_service = 1;
        #endif
        
        if (!run_as_w9x_service)
          {
          if (RegOpenKey(HKEY_LOCAL_MACHINE, 
            "Software\\Microsoft\\Windows\\CurrentVersion\\RunServices",
            &srvkey) == ERROR_SUCCESS)
            {
            DWORD valuetype = REG_SZ;
            char buffer[260]; // maximum registry key length
            DWORD valuesize = sizeof(buffer);

            if ( RegQueryValueEx(srvkey, "bovwin32", NULL,
                &valuetype, (unsigned char *)(&buffer[0]), 
                &valuesize) == ERROR_SUCCESS )
              {
              run_as_w9x_service = 1;
              }
            RegCloseKey(srvkey);
            }
          }

        if (run_as_w9x_service)
          {              
          // register ourself as a Win95 service
          HMODULE kernl = GetModuleHandle("KERNEL32");
          if (kernl)
            {
            typedef DWORD (CALLBACK *ULPRET)(DWORD,DWORD);
            ULPRET func = (ULPRET) GetProcAddress(kernl, "RegisterServiceProcess");
            if (func) (*func)(0, 1);
            }
          }
        
        }
      retcode = 0;
      }
    #endif
    }

  return retcode;
}

// -----------------------------------------------------------------------

static int DeinitializeConsole(void)
{
  #if ((CLIENT_OS == OS_WIN32) && defined(CONSOLE))
    {
    FreeConsole();
    }
  #endif
  return 0;
}  

// ---------------------------------------------------------------------------

#if 0 //((CLIENT_OS == OS_WIN32) && defined(CONSOLE))

static WNDPROC (*oldwndproc)(HWND,unsigned,UINT,LONG);

/* CALLBACK */
extern "C" WNDPROC FAR __export PASCAL WindowProc( HWND hwnd, unsigned msg,
             UINT wparam, LONG lparam );

WNDPROC FAR __export PASCAL WindowProc( HWND hwnd, unsigned msg,
             UINT wparam, LONG lparam )
{
  if (msg == WM_DESTROY )
    {
    RaiseExitRequestTrigger();
    PostQuitMessage( 0 );
    return(0L);
    }
  return( (*oldwndproc)( hwnd, msg, wparam, lparam ) );
}    
#endif

#if (CLIENT_OS != OS_WIN32)
static int InitializeConsole(int) { return 0; }
#else
static int InitializeConsole(int runhidden)
{
  int retcode = 0;

  //the win32 console client is really a GUI client without a GUI - cyrus
  #if ((CLIENT_OS == OS_WIN32) && defined(CONSOLE))
    {
    const char *contitle = "Distributed.Net RC5/DES Client " 
                             "" CLIENT_VERSIONSTRING "";
    OSVERSIONINFO osver;
    osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
    GetVersionEx(&osver);
      
    // only allow one running instance
    CreateMutex(NULL, TRUE, "Bovine RC5/DES Win32 Client");
    if (GetLastError()) 
      {
      retcode = -1;
      }
    else if (runhidden)
      {
      //nothing - console won't be created
      retcode = 0;
      }
    else if (!AllocConsole())
      {
      retcode = -1;
      MessageBox( NULL, "Unable to create console window.",
                           contitle, MB_OK | MB_TASKMODAL);
      }
    else
      {
      retcode = 0;
      SetConsoleTitle(contitle);
      
      #if 0
      HWND hwnd = FindWindow( NULL, contitle );
      if ( hwnd )
        {
        oldwndproc = (WNDPROC (*)(HWND,unsigned,UINT,LONG))
                     GetWindowLong(hwnd, GWL_WNDPROC);
                         
        if (oldwndproc)
          {             //thing doesn't set. probably a mem access issue.
          SetWindowLong(hwnd, GWL_WNDPROC, (LPARAM)(WNDPROC)(WindowProc));
          //SubclassWindow(hwnd, WindowProc);
          }
        }
      #endif

      // Now re-map the C Runtime STDIO handles
      if (VER_PLATFORM_WIN32_NT == osver.dwPlatformId)
        {
        //microsoft method - fails on win98 (_fdopen fails)
        //http://support.microsoft.com/support/kb/articles/q105/3/05.asp

        int hCrt = _open_osfhandle((long)GetStdHandle(STD_OUTPUT_HANDLE), _O_TEXT);
        FILE *hf = _fdopen(hCrt, "w");
        *stdout = *hf;
        setvbuf(stdout, NULL, _IONBF, 0);         
        hCrt = _open_osfhandle((long)GetStdHandle(STD_ERROR_HANDLE), _O_TEXT);
        hf = _fdopen(hCrt, "w");
        *stderr = *hf;
        setvbuf(stderr, NULL, _IONBF, 0);         
        hCrt = _open_osfhandle((long)GetStdHandle(STD_INPUT_HANDLE), _O_TEXT);
        hf = _fdopen(hCrt, "r");
        *stdin = *hf;
        setvbuf(stdin, NULL, _IONBF, 0);
        }
      else
        {
        FILE *hf;
        hf = fopen("CONOUT$", "w+t");
        if (!hf)
          {
          MessageBox( NULL, "Unable to open console for write.",
                            contitle, MB_OK | MB_TASKMODAL);
          retcode = -1;
          }
        else
          {
          *stdout = *hf;
          setvbuf(stdout, NULL, _IONBF, 0);
          *stderr = *hf;
          setvbuf(stderr, NULL, _IONBF, 0);
            
          hf = fopen("CONIN$", "rt");
          if (!hf)
            {
            MessageBox( NULL, "Unable to open console for read.",
                              contitle, MB_OK | MB_TASKMODAL);
            retcode = -1;
            }
          else
            {
            *stdin = *hf;
            setvbuf(stdin, NULL, _IONBF, 0);
            }
          }
        #if 0
        else
          {
          SECURITY_ATTRIBUTES sa;
          sa.nLength      = sizeof(SECURITY_ATTRIBUTES);
          sa.lpSecurityDescriptor = NULL;
          sa.bInheritHandle   = TRUE;

          HANDLE hIFile = CreateFile( "CONIN$", GENERIC_READ /*dwDesiredAccess*/, 
          FILE_SHARE_READ /*dwShareMode*/, &sa /*lpSecurityAttributes*/,
          OPEN_EXISTING /*dwCreationDistribution*/, 0 /*dwFlagsAndAttributes*/,
          0 /*hTemplateFile*/ );
          HANDLE hOFile = CreateFile( "CONOUT$", GENERIC_WRITE /*dwDesiredAccess*/, 
          FILE_SHARE_WRITE /*dwShareMode*/, &sa /*lpSecurityAttributes*/,
          OPEN_EXISTING /*dwCreationDistribution*/, 0 /*dwFlagsAndAttributes*/,
          0 /*hTemplateFile*/ );

          SetStdHandle( STD_OUTPUT_HANDLE, hOFile );
          SetStdHandle( STD_ERROR_HANDLE, hOFile );
          SetStdHandle( STD_INPUT_HANDLE, hIFile );
          }
        #endif
        }
      }
    }
  #elif ((CLIENT_OS == OS_WIN32) && (!defined(WINNTSERVICE)))
    {
    const char *contitle = "Distributed.Net RC5/DES Client " 
                           "" CLIENT_VERSIONSTRING "";
    OSVERSIONINFO osver;
    osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
    GetVersionEx(&osver);

    SetConsoleTitle(contitle);
        
    // only allow one running instance
    CreateMutex(NULL, TRUE, "Bovine RC5/DES Win32 Client");
    if (GetLastError()) 
      {
      retcode = -1;
      }
    else if (!runhidden)
      {
      //nothing - screen is already visible
      }
    else if (VER_PLATFORM_WIN32_NT == osver.dwPlatformId)
      {
      MessageBox( NULL, "Running -hidden is not recommended under NT.\n"
      "(There have been cases of conflicts with system process csrss.exe)\n"
        "Please use the NT Service client.", contitle, MB_OK | MB_TASKMODAL);
      retcode = -1;
      }
    else
      {
      FreeConsole();
      }
    }
  #endif

  return retcode;
}  
#endif // (CLIENT_OS != OS_WIN32)


#if !defined(NOMAIN)
int main( int argc, char *argv[] )
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

  if ( init_success )
    {
    init_success = (( clientP = new Client() ) != NULL);
    if (!init_success) fprintf( stderr, "\nRC5DES: Out of memory.\n" );
    }

  #if (CLIENT_OS == OS_NETWARE) 
  //create stdout/screen, set cwd etc. save ptr to client for fnames/niceness
  if ( init_success )
    init_success = ( nwCliInitClient( argc, argv, clientP ) == 0);
  #endif

  if ( init_success )
    {
    do {
       retcode = clientP->Main( argc, (const char **)argv, restarted );
       restarted = 1; //for the next round
       } while (CheckRestartRequestTrigger());

    #if (CLIENT_OS == OS_AMIGAOS)
    if (retcode) retcode = 5; // 5 = Warning
    #endif // (CLIENT_OS == OS_AMIGAOS)
    }
  
  #if (CLIENT_OS == OS_NETWARE)
  if (init_success)
    nwCliExitClient(); // destroys AES process, screen, polling procedure
  #endif
  
  if (clientP)
    delete clientP;

  return (retcode);
}

//------------------------------------------------------------------------

int Client::Main( int argc, const char *argv[], int restarted )
{
  int retcode = 0;

  // set up break handlers
  if (InitializeTriggers(NULL, NULL)==0) //CliSetupSignals();
    {
    //get inifilename and get -quiet/-hidden overrides
    if (ParseCommandline( 0, argc, argv, NULL, &retcode, 0 ) == 0) //change defaults
      {
      int inimissing = (ReadConfig() != 0); //reads using defaults
      InitializeTriggers( ((noexitfilecheck)?(NULL):("exitrc5" EXTN_SEP "now")),pausefile);

      InitializeLogging(); //let -quiet take affect
      if (InitializeConsole(runhidden) == 0)  //create console (if required)
        {
        PrintBanner(id); //tracks restart state itself

        if ( ParseCommandline( 2, argc, argv, &inimissing, &retcode, 1 )==0 )
          {
          if (inimissing)
            {
            if (Configure() ==1 ) 
              WriteFullConfig(); //full new build
            }
          else if ( RunStartup(restarted) == 0 ) 
            {
            ValidateConfig();
            PrintBanner(id);  //tracks restart state itself
            retcode = Run();
            RunShutdown();
            }
          }
        DeinitializeConsole();
        }
      DeinitializeLogging();
      }
    DeinitializeTriggers();
    }
  return retcode;
}  
#endif


// ---------------------------------------------------------------------------

int Client::Install()
{
#if (CLIENT_OS == OS_WIN32) && defined(WINNTSERVICE)
  char mypath[200];
  GetModuleFileName(NULL, mypath, sizeof(mypath));
  SC_HANDLE myService, scm;
  scm = OpenSCManager(0, 0, SC_MANAGER_CREATE_SERVICE);
  if (scm)
  {
    myService = CreateService(scm, NTSERVICEID,
        "Distributed.Net RC5/DES Service Client",
        SERVICE_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS,
        SERVICE_AUTO_START, SERVICE_ERROR_NORMAL,
        mypath, 0, 0, 0, 0, 0);
    if (myService)
    {
      LogScreen("Windows NT Service installation complete.\n"
          "Click on the 'Services' icon in 'Control Panel' and ensure that the\n"
          "Distributed.Net RC5/DES Service Client is set to startup automatically.\n");
      CloseServiceHandle(myService);
    } else {
      LogScreen("Error creating service entry.\n");
    }
    CloseServiceHandle(scm);
  } else {
    LogScreen("Error opening service control manager.\n");
  }
#elif ((CLIENT_OS == OS_WIN32) && !defined(WINNTSERVICE))
  HKEY srvkey=NULL;
  DWORD dwDisp=NULL;
  char mypath[260];

  OSVERSIONINFO osver;
  osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
  GetVersionEx(&osver);

  if (VER_PLATFORM_WIN32_NT == osver.dwPlatformId)
    {
    LogScreen("-install failed. This version of the client was built "
              "without NT service support.\n" );
    }
  else
    {
    GetModuleFileName(NULL, mypath, sizeof(mypath));
 
    strcat( mypath, " -hide" );

    // register a Win95 "RunService" item
    if (RegCreateKeyEx(HKEY_LOCAL_MACHINE,
        "Software\\Microsoft\\Windows\\CurrentVersion\\RunServices",0,"",
              REG_OPTION_NON_VOLATILE,KEY_ALL_ACCESS,NULL,
              &srvkey,&dwDisp) == ERROR_SUCCESS)
      {
      RegSetValueEx(srvkey, "bovwin32", 0, REG_SZ, (unsigned const char *)mypath, strlen(mypath) + 1);
      RegCloseKey(srvkey);
      }

    // unregister a Win95 "Run" item
    if (RegOpenKey(HKEY_LOCAL_MACHINE,
        "Software\\Microsoft\\Windows\\CurrentVersion\\Run",
      &srvkey) == ERROR_SUCCESS)
      {
      RegDeleteValue(srvkey, "bovwin32");
      RegCloseKey(srvkey);
      }
    LogScreen("Win95 Service installation complete.\n");
    }

#elif (CLIENT_OS == OS_OS2)
  int rc;
  const int len = 4068;

  char   pszClassName[] = "WPProgram";
  char   pszTitle[] = "RC5-DES Cracking Client";
  char   pszLocation[] = "<WP_START>";    // Startup Folder
  ULONG ulFlags = 0;

  char   pszSetupString[len] =
            "OBJECTID=<RC5DES-CLI>;"
            "MINIMIZED=YES;"
            "PROGTYPE=WINDOWABLEVIO;";

  // Add full path of the program
  strncat(pszSetupString, "EXENAME=",len);

  if(runhidden == 1)   // Run detached
  {
    strncat(pszSetupString, "CMD.EXE;", len);     // command processor
    strncat(pszSetupString, "PARAMETERS=/c detach ", len);   // detach
  }

  // Add exepath and exename
  strncat(pszSetupString, exepath, len);
  strncat(pszSetupString, exename, len);
  strncat(pszSetupString, ";", len);

  // Add on Working Directory
  strncat(pszSetupString, "STARTUPDIR=", len);
  strncat(pszSetupString, exepath, len);
  strncat(pszSetupString, ";", len);

  rc = WinCreateObject(pszClassName, pszTitle, pszSetupString,
              pszLocation, ulFlags);
  if(rc == NULLHANDLE)
    LogScreen("ERROR: RC5-DES Program object could not be added "
            "into your Startup Folder\n"
            "RC5-DES is probably already installed\n");
  else
    LogScreen("RC5-DES Program object has been added into your Startup Folder\n");
#endif
  return 0;
}

// ---------------------------------------------------------------------------

int Client::Uninstall(void)
{
#if (CLIENT_OS == OS_OS2)
  HOBJECT hObject = WinQueryObject("<RC5DES-CLI>");

  if(hObject == NULLHANDLE)
    LogScreen("ERROR: RC5-DES Client object was not found\n"
          "No RC5-DES client installed in the Startup folder\n");
  else
    {
    LogScreen("RC5-DES Client object %s removed from the Startup Folder.\n",
      ((WinDestroyObject(hObject) == TRUE)?("was"):("could not be"))  );
    }
#endif

#if (CLIENT_OS == OS_WIN32) && defined(WINNTSERVICE)

  SC_HANDLE myService, scm;
  SERVICE_STATUS status;
  scm = OpenSCManager(0, 0, SC_MANAGER_CREATE_SERVICE);
  if (scm)
  {
    myService = OpenService(scm, NTSERVICEID,
        SERVICE_ALL_ACCESS | DELETE);
    if (myService)
    {
      if (QueryServiceStatus(myService, &status) &&
        status.dwCurrentState != SERVICE_STOPPED)
      {
        LogScreen("Service currently active.  Stopping service...\n");
        if (!ControlService(myService, SERVICE_CONTROL_STOP, &status))
          LogScreen("Failed to stop service!\n");
      }
      if (DeleteService(myService))
      {
        LogScreen("Windows NT Service uninstallation complete.\n");
      } else {
        LogScreen("Error deleting service entry.\n");
      }
      CloseServiceHandle(myService);
    }
    CloseServiceHandle(scm);
  } else {
    LogScreen("Error opening service control manager.\n");
  }
#endif

#if (CLIENT_OS == OS_WIN32) && (!defined(WINNTSERVICE))
  HKEY srvkey;

  OSVERSIONINFO osver;
  osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
  GetVersionEx(&osver);

  if (VER_PLATFORM_WIN32_NT == osver.dwPlatformId)
    {
    LogScreen("-uninstall failed. This version of the client was built "
              "without NT service support.\n" );
    }
  else
    {
    // unregister a Win95 "RunService" item
    if (RegOpenKey(HKEY_LOCAL_MACHINE,
        "Software\\Microsoft\\Windows\\CurrentVersion\\RunServices",
        &srvkey) == ERROR_SUCCESS)
      {
      RegDeleteValue(srvkey, "bovwin32");
      RegCloseKey(srvkey);
      }

    // unregister a Win95 "Run" item
    if (RegOpenKey(HKEY_LOCAL_MACHINE,
      "Software\\Microsoft\\Windows\\CurrentVersion\\Run",
      &srvkey) == ERROR_SUCCESS)
      {
      RegDeleteValue(srvkey, "bovwin32");
      RegCloseKey(srvkey);
      }
    LogScreen("Win95 Service uninstallation complete.\n");
    }
#endif

  return 0;
}

