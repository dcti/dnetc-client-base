// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: client.cpp,v $
// Revision 1.175  1998/12/08 05:35:47  dicamillo
// MacOS updates: allow PrintBanner to be called by Mac client;
// added help text; Mac client provides its own "main".
//
// Revision 1.174  1998/12/01 23:31:43  cyp
// Removed ::totalBlocksDone. count was inaccurate if client was restarted.
// ::Main() (as opposed to realmain()) now controls restart.
//
// Revision 1.173  1998/11/28 19:44:34  cyp
// InitializeLogging() and DeinitializeLogging() are no longer Client class
// methods.
//
// Revision 1.172  1998/11/26 06:54:25  cyp
// Updated client contructor.
//
// Revision 1.171  1998/11/25 09:23:27  chrisb
// various changes to support x86 coprocessor under RISC OS
//
// Revision 1.170  1998/11/24 22:41:25  silby
// Commented out winmain for win32gui - it will call realmain from its own code.
//
// Revision 1.169  1998/11/20 03:43:14  silby
// Can't call static func (realmain) from a seperate source file.
//
// Revision 1.168  1998/11/19 23:36:26  cyp
// PrintBanner() now gets its level and restart flag from ::Main()
//
// Revision 1.167  1998/11/19 23:11:52  cyp
// Logging is now initialized with time stamping and logfile/mail logs
// enabled. This will probably break stuff, but logging -update etc to file
// seems to be a priority for some people.
//
// Revision 1.166  1998/11/19 20:48:49  cyp
// Rewrote -until/-h handling. Did away with useless client.hours (time-to-die
// is handled by client.minutes anyway). -until/-h/hours all accept "hh:mm"
// format now (although they do continue to support the asinine "hh.mm").
//
// Revision 1.165  1998/11/19 08:34:43  silby
// Removed win32gui specific winmain.
//
// Revision 1.164  1998/11/17 04:39:33  silby
// Gave GetBuildOrEnvDescription the fixing it was pining for.
//
// Revision 1.163  1998/11/16 22:31:09  cyp
// Cleaned up banner(s) and made use of CLIENT_OS_NAME.
//
// Revision 1.162  1998/11/14 14:01:38  cyp
// Removed DoInstanceInitialization(). Moved win32 mutex alloc back to its
// window init code; unix'ish client fork-on-hidden code is in ParseCmdLine.
//
// Revision 1.161  1998/11/13 15:31:45  silby
// Changed default blocksize to 31 now.
//
// Revision 1.160  1998/11/13 12:35:49  remi
// uuehttpmode set to 0 (no special encoding) instead of 1 (uue encoding)
// in Client::Client().
//
// Revision 1.159  1998/11/12 23:50:15  cyp
// Added -hide/-quiet support for unix'ish clients
//
// Revision 1.158  1998/11/12 13:13:17  silby
// Lurk mode is initialized in the main loop now.
//
// Revision 1.157  1998/11/11 06:04:31  cyp
// Added a win version check <=3.x before asserting TZ= validity
//
// Revision 1.156  1998/11/11 05:26:57  cramer
// Various minor fixes...
//
// Revision 1.155  1998/11/11 00:32:07  cyp
// Fixed DoSingleInstanceProtection() for win32. (many thanks davehart)
//
// Revision 1.154  1998/11/10 21:51:32  cyp
// Completely reorganized Client::Main() initialization order so that the
// client object is initialized (read config/command line parse) first.
//
// Revision 1.153  1998/11/09 20:05:11  cyp
// Did away with client.cktime altogether. Time-to-Checkpoint is calculated
// dynamically based on problem completion state and is now the greater of 1
// minute and time_to_complete_1_percent (an average change of 1% that is).
//
// Revision 1.152  1998/11/08 01:01:41  silby
// Buncha hacks to get win32gui to compile, lots of cleanup to do.
//
// Revision 1.151  1998/11/04 21:28:01  cyp
// Removed redundant ::hidden option. ::quiet was always equal to ::hidden.
//
// Revision 1.150  1998/11/02 04:40:13  cyp
// Removed redundant ::numcputemp. ::numcpu does it all.
//
// Revision 1.149  1998/10/26 02:51:41  cyp
// out_buffer_file[0] was being initialized with the wrong suffix ('des'
// instead of 'rc5') in the client constructor.
//
// Revision 1.148  1998/10/19 12:42:09  cyp
// win16 changes
//
// Revision 1.147  1998/10/11 00:41:23  cyp
// Implemented ModeReq
//
// Revision 1.146  1998/10/07 20:43:32  silby
// Various quick hacks to make the win32gui operational again (will be cleaned up).
//
// Revision 1.145  1998/10/06 22:28:53  cyp
// Changed initialization order so that initialization that requires filenames
// follows the second ParseCommandLine().
//
// Revision 1.144  1998/10/05 05:21:30  cyp
// Added PPC core attribute to PrintBanner()
//
// Revision 1.143  1998/10/04 16:54:05  remi
// Moved a misplaced #endif. Wrapped $Log comments.
//
// Revision 1.142  1998/10/04 03:22:14  silby
// Changed startup logging code so that CLIENT_VERSIONSTRING was used
// so that it's obvious if a BETA is being used in logfiles (could not
// be determined otherwise)
//
// Revision 1.141  1998/10/03 23:27:51  remi
// Use 'usemmx' .ini setting if any MMX core is compiled in.
//
// Revision 1.140  1998/10/03 03:33:13  cyp
// Removed RunStartup() altogether, changed fprintf(stderr,) to ConOutErr(),
// moved 68k kudos from selcore.cpp to PrintBanner(), removed ::Install and
// ::Uninstall, added winNT svc startup code to realmain(), added WinMain().
//
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
#if (!defined(lint) && defined(__showids__))
const char *client_cpp(void) {
return "@(#)$Id: client.cpp,v 1.175 1998/12/08 05:35:47 dicamillo Exp $"; }
#endif

// --------------------------------------------------------------------------

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "client.h"    // Packet, FileHeader, Client class, etc
#include "scram.h"     // InitRandom() 
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "pathwork.h"  // EXTN_SEP
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
  id[0] = 0;
  inthreshold[0] = 10;
  outthreshold[0] = 10;
  inthreshold[1] = 10;
  outthreshold[1] = 10;
  blockcount = 0;
  minutes = 0;
  stopiniio = 0;
  keyproxy[0] = 0;
  keyport = 2064;
  httpproxy[0] = 0;
  httpport = 80;
  uuehttpmode = 0;
  httpid[0] = 0;
  cputype=-1;
  offlinemode = 0;
  autofindkeyserver = 1;  //implies 'only if keyproxy==dnetkeyserver'

  pausefile[0]=logname[0]=0;
  strcpy(inifilename, "rc5des" EXTN_SEP "ini");
  strcpy(in_buffer_file[0], "buff-in" EXTN_SEP "rc5");
  strcpy(out_buffer_file[0], "buff-out" EXTN_SEP "rc5");
  strcpy(in_buffer_file[1], "buff-in" EXTN_SEP "des");
  strcpy(out_buffer_file[1], "buff-out" EXTN_SEP "des");
  strcpy(exit_flag_file,     "exitrc5" EXTN_SEP "now" );
  checkpoint_file[0][0]=checkpoint_file[1][0]=0;

  messagelen = 0;
  smtpport = 25;
  strcpy(smtpsrvr,"your.smtp.server");
  strcpy(smtpfrom,"RC5Notify");
  strcpy(smtpdest,"you@your.site");
  numcpu = -1;
  percentprintingoff=0;
  connectoften=0;
  nodiskbuffers=0;
  membufftable[0].in.count=
  membufftable[0].out.count=
  membufftable[1].in.count=
  membufftable[1].out.count=0;
  nofallback=0;
  randomprefix=100;
  preferred_contest_id = 1;
  preferred_blocksize=31;
  randomchanged=0;
  consecutivesolutions[0]=0;
  consecutivesolutions[1]=0;
  quietmode=0;
  nonewblocks=0;
  nettimeout=60;
  noexitfilecheck=0;
#if defined(LURK)
  dialup.lurkmode=0;
  dialup.dialwhenneeded=0;
#endif
  contestdone[0]=contestdone[1]=0;
  srand( (unsigned) CliTimer( NULL )->tv_usec );
  InitRandom();
#if defined(MMX_BITSLICER) || defined(MMX_RC5)
  usemmx = 1;
#endif
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

  #if (CLIENT_OS == OS_DOS)
  return dosCliGetEmulationDescription(); //if in win/os2 VM 
  #elif ((CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S))
  static char buffer[32]; long ver = winGetVersion(); /* w32pre.cpp */
  sprintf(buffer,"under Windows%s %u.%u", (ver>=2000)?(" NT"):(""), (ver/100)%20, ver%100 );
  return buffer;
  #else
  return "";
  #endif
}  

// --------------------------------------------------------------------------

void PrintBanner(const char *dnet_id,int level,int restarted)
{
  //level = 0 = show copyright/version,  1 = show startup message
  
  if (!restarted)
    {
    if (level == 0)
      {
      LogScreenRaw( "\nRC5DES " CLIENT_VERSIONSTRING 
                 " client - a project of distributed.net\n"
                 "Copyright 1997-1998 distributed.net\n" );
      
      #if (CLIENT_CPU == CPU_68K)
      LogScreenRaw( "RC5 68K assembly by John Girvin\n");
      #endif
      #if (CLIENT_CPU == CPU_POWERPC)
      LogScreenRaw( "PowerPC assembly by Dan Oetting at USGS\n");
      #endif

      #if (CLIENT_CPU == CPU_ARM)
      LogScreenRaw( "ARM assembly by Steve Lee\n");
      #if (CLIENT_OS == OS_RISCOS)
      LogScreenRaw( "PC Card support by Dominic Plunkett\n");
      #endif
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
      LogScreenRaw( "Please visit http://www.distributed.net/ for up-to-date contest information.\n"
                 "%s\n",
              #if (CLIENT_OS == OS_RISCOS)
              guiriscos ?
              "Interactive help is available, or select 'Help contents' from the menu for\n"
              "detailed client information.\n" :
              #endif
			  #if (CLIENT_OS == OS_MACOS)
                "Select ""Help..."" from the Apple menu for detailed client information.\n"
			  #else
              "Start the client with '-help' for a list of valid command line options.\n"
			  #endif
              );
      }
    else if ( level == 1 )
      {  
      #if ((CLIENT_OS==OS_DOS) || (CLIENT_OS==OS_WIN16) || \
           (CLIENT_OS==OS_WIN32S) || (CLIENT_OS==OS_OS2) || \
           ((CLIENT_OS==OS_WIN32) && !defined(NEEDVIRTUALMETHODS)))
      #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
      if ((winGetVersion()%2000) < 400) /* w32pre.cpp. +2000==NT */
      #endif   
      if (getenv("TZ") == NULL)
        {
        LogScreenRaw("Warning: The TZ= variable is not set in the environment. "
         "The client will\nprobably display the wrong time and/or select the "
         "wrong keyserver.\n");
        putenv("TZ=GMT+0"); //use local time.
        }
      #endif

      const char *msg = GetBuildOrEnvDescription();
      if (msg == NULL) msg="";

      LogRaw("\nRC5DES Client %s for %s%s%s%s started.\n",CLIENT_VERSIONSTRING,
             CLIENT_OS_NAME, ((*msg)?(" ("):("")), msg, ((*msg)?(")"):("")) );
  
      LogRaw( "Using distributed.net ID %s\n\n", dnet_id );
      
      #if defined(BETA) && defined(BETA_EXPIRATION_TIME) && (BETA_EXPIRATION_TIME != 0)
      timeval currenttime;
      timeval expirationtime;
  
      CliTimer(&currenttime);
      expirationtime.tv_usec= 0;
      expirationtime.tv_sec = BETA_EXPIRATION_TIME;
  
      if (currenttime.tv_sec > expirationtime.tv_sec ||
        currenttime.tv_sec < (BETA_EXPIRATION_TIME - 1814400))
        {
        ; //nothing - start run, recover checkpoints and _then_ exit.
        } 
      else
        {
        LogScreenRaw("Notice: This is a beta release and expires on %s\n\n",
         CliGetTimeString(&expirationtime,1) );
        }
      #endif // BETA
      }
    }
  return;
}

//------------------------------------------------------------------------

int InitializeConnectivity(void)
{
  #ifdef LURK
  dialup.Start();
  #endif
  return 0;
}

//------------------------------------------------------------------------

int DeinitializeConnectivity(void)
{
  #ifdef LURK
  dialup.Stop();
  #endif
  return 0;
}

//------------------------------------------------------------------------


int Client::Main( int argc, const char *argv[] )
{
  int retcode = 0;
  int domodes = 0;
  int restart = 0;
  int restarted;

  do{
    restarted = restart;
    restart = 0;

    //ReadConfig() and parse command line - returns !0 if shouldn't continue
    if (ParseCommandline( 0, argc, argv, &retcode, 0 ) == 0)
      {
      domodes = (ModeReqIsSet(-1) != 0);
      if (InitializeTriggers(((noexitfilecheck)?(NULL):(exit_flag_file)),pausefile)==0)
        {
        if (InitializeConnectivity() == 0)
          {
          if (InitializeConsole(quietmode,domodes) == 0)
            {
            InitializeLogging( (quietmode!=0), (percentprintingoff!=0), 
                               logname, LOGFILETYPE_NOLIMIT, 0, messagelen, 
                               smtpsrvr, smtpport, smtpfrom, smtpdest, id );
            PrintBanner(id,0,restarted);
            ParseCommandline( 1, argc, argv, NULL, (quietmode==0)); //show overrides
          
            if (domodes)
              {
              ModeReqRun( this );     
              }
            else
              {
              PrintBanner(id,1,restarted);
              SelectCore( 0 );
              retcode = Run();
              restart = CheckRestartRequestTrigger();
              }
            DeinitializeLogging();
            DeinitializeConsole();
            }
          DeinitializeConnectivity();
          }
        DeinitializeTriggers();
        }
      }
    } while (restart);
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
    retcode = clientP->Main( argc, (const char **)argv );
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


#if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
#if !defined(WIN32GUI)
int PASCAL WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpszCmdLine, 
    int nCmdShow) 
{ /* abstraction layer between WinMain() and realmain() */
  return winClientPrelude( hInst, hPrevInst, lpszCmdLine, nCmdShow, realmain);
}
#endif
#elif (CLIENT_OS == OS_MACOS)
// nothing- Mac framework provides main
#else
int main( int argc, char *argv[] )
{ 
  return realmain( argc, argv ); 
}
#endif

