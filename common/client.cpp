// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: client.cpp,v $
// Revision 1.190  1999/02/03 17:49:38  cyp
// Cleaned up CLIENT_VERSIONSTRING #define
//
// Revision 1.189  1999/02/03 04:31:16  cyp
// cosmetic change: fixed a misplaced space in the startup banner.
//
// Revision 1.188  1999/02/03 03:41:56  cyp
// InitializeConnectivity()/DeinitializeConnectivity() are now in netinit.cpp
//
// Revision 1.187  1999/01/27 16:34:23  cyp
// if one variable wasn't being initialized, there could have been others.
// Consequently, added priority that had been missing as well.
//
// Revision 1.186  1999/01/27 02:47:30  silby
// timeslice is now initialized during client creation.
//
// Revision 1.185  1999/01/26 17:27:58  michmarc
// Updated banner messages for new DES slicing routines
//
// Revision 1.184  1999/01/17 15:59:25  cyp
// Do an InitRandom2() before any work starts.
//
// Revision 1.183  1999/01/15 00:32:44  cyp
// changed phrasing of 'distributed.net ID' at Nugget's request.
//
// Revision 1.182  1999/01/06 22:16:10  dicamillo
// Changed credit for Dan Oetting at his request.
//
// Revision 1.181  1999/01/04 02:49:10  cyp
// Enforced single checkpoint file for all contests.
//
// Revision 1.180  1999/01/03 04:58:59  dicamillo
// Restore missing initialization for checkpoint_file[1].
//
// Revision 1.179  1999/01/03 02:31:10  cyp
// Client::Main() reinitializes the client object when restarting. ::pausefile
// and ::exitflagfile are now ignored when starting with "modes".
//
// Revision 1.178  1999/01/01 02:45:14  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.177  1998/12/22 15:58:24  jcmichot
// QNX needs to initialize srand() with time(NULL)
//
// Revision 1.176  1998/12/12 12:28:49  cyp
// win16/win32 change: win16/win32 console code needs access to the client
// object so that it can force a shutdown when it gets an endsession message.
//
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
return "@(#)$Id: client.cpp,v 1.190 1999/02/03 17:49:38 cyp Exp $"; }
#endif

// --------------------------------------------------------------------------

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "client.h"    // Client class
#include "scram.h"     // InitRandom() 
#include "pathwork.h"  // EXTN_SEP
#include "clitime.h"   // CliTimer()
#include "modereq.h"   // ModeReqIsSet()/ModeReqRun()
#include "triggers.h"  // [De]InitializeTriggers(),RestartRequestTrigger()
#include "logstuff.h"  // [De]InitializeLogging(),Log()/LogScreen()
#include "console.h"   // [De]InitializeConsole(), ConOutErr()
#include "network.h"   // [De]InitializeConnectivity()

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

static void __initialize_client_object(Client *client)
{
  strcpy(client->id, "rc5@distributed.net" );
  client->inthreshold[0] = 10;
  client->outthreshold[0] = 10;
  client->inthreshold[1] = 10;
  client->outthreshold[1] = 10;
  client->minutes = 0;
  client->priority = 0;

  client->blockcount = 0;
  client->timeslice = 0x10000;
  client->stopiniio = 0;
  client->keyproxy[0] = 0;
  client->keyport = 0;
  client->httpproxy[0] = 0;
  client->httpport = 0;
  client->uuehttpmode = 0;
  client->httpid[0] = 0;
  client->cputype=-1;
  client->offlinemode = 0;
  client->autofindkeyserver = 1;  //implies 'only if keyproxy==dnetkeyserver'

  client->pausefile[0]=
  client->logname[0]=
  client->checkpoint_file[0]=0;
  strcpy(client->inifilename, "rc5des" EXTN_SEP "ini");
  strcpy(client->in_buffer_file[0], "buff-in" EXTN_SEP "rc5");
  strcpy(client->out_buffer_file[0], "buff-out" EXTN_SEP "rc5");
  strcpy(client->in_buffer_file[1], "buff-in" EXTN_SEP "des");
  strcpy(client->out_buffer_file[1], "buff-out" EXTN_SEP "des");
  strcpy(client->exit_flag_file,     "exitrc5" EXTN_SEP "now" );

  client->messagelen = 0;
  client->smtpport = 25;
  client->smtpsrvr[0]=
  client->smtpfrom[0]=
  client->smtpdest[0]=0;
  client->contestdone[0]=
  client->contestdone[1]=0;
  client->numcpu = -1;
  client->percentprintingoff=0;
  client->connectoften=0;
  client->nodiskbuffers=0;
  client->membufftable[0].in.count=
  client->membufftable[0].out.count=
  client->membufftable[1].in.count=
  client->membufftable[1].out.count=0;
  client->nofallback=0;
  client->randomprefix=100;
  client->preferred_contest_id = 1;
  client->preferred_blocksize=31;
  client->randomchanged=0;
  client->consecutivesolutions[0]=0;
  client->consecutivesolutions[1]=0;
  client->quietmode=0;
  client->nonewblocks=0;
  client->nettimeout=60;
  client->noexitfilecheck=0;
#if defined(MMX_BITSLICER) || defined(MMX_RC5)
  client->usemmx = 1;
#endif
#if defined(LURK)
  dialup.lurkmode=0;
  dialup.dialwhenneeded=0;
#endif
  srand( (unsigned) time(NULL) );
  InitRandom();
}

Client::Client()
{ 
  __initialize_client_object(this); 
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

int ClientIsGUI(void)
{
  #if defined(WIN32GUI) || defined(MAC_GUI) || defined(OS2GUI)
  return 1;
  #elif (CLIENT_OS == OS_RISCOS)
  return (guiriscos!=0);
  #else
  return 0;
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
      LogScreenRaw( "\nRC5DES v" CLIENT_VERSIONSTRING 
                 " client - a project of distributed.net\n"
                 "Copyright 1997-1999 distributed.net\n" );
      
      #if (CLIENT_CPU == CPU_68K)
      LogScreenRaw( "RC5 68K assembly by John Girvin\n");
      #endif
      #if (CLIENT_CPU == CPU_POWERPC)
      LogScreenRaw( "PowerPC assembly by Dan Oetting\n");
      #endif
      #if (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32)
      LogScreenRaw( "RC5 Alpha assembly by Mike Marcelais\n");
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
      #elif defined(KWAN) && defined(DWORZ)
      LogScreenRaw( "DES bitslice driver Copyright 1999, Christoph Dworzak\n"
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
      LogScreenRaw( "Please visit http://www.distributed.net/ for up-to-date contest information.\n");
      LogScreenRaw( 
        #if (CLIENT_OS == OS_RISCOS)
        guiriscos ?
        "Interactive help is available, or select 'Help contents' from the menu for\n"
        "detailed client information.\n\n" :
        #elif (CLIENT_OS == OS_MACOS)
        "Select ""Help..."" from the Apple menu for detailed client information.\n\n"
        #else
        "Start the client with '-help' for a list of valid command line options.\n\n"
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

      LogRaw("\nRC5DES v%s %sClient for %s%s%s%s started.\n",
            CLIENT_VERSIONSTRING, ((ClientIsGUI())?("GUI "):("")), 
            CLIENT_OS_NAME, ((*msg)?(" ("):("")), msg, ((*msg)?(")"):("")) );
  
      LogRaw( "Using email address (distributed.net ID) \'%s\'\n\n", dnet_id );
      
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


int Client::Main( int argc, const char *argv[] )
{
  int retcode = 0;
  int domodes = 0;
  int restart = 0;
  int restarted;

  do{
    restarted = restart;
    restart = 0;
    __initialize_client_object(this); /* reset everything in the object */

    //ReadConfig() and parse command line - returns !0 if shouldn't continue
    if (ParseCommandline( 0, argc, argv, &retcode, 0 ) == 0)
      {
      domodes = (ModeReqIsSet(-1) != 0);
      if (InitializeTriggers(((noexitfilecheck || 
                              domodes)?(NULL):(exit_flag_file)),
                              ((domodes)?(NULL):(pausefile)) )==0)
        {
        if (InitializeConnectivity() == 0) //do global initialization
          {
          if (InitializeConsole(quietmode,domodes) == 0)
            {
            InitializeLogging( (quietmode!=0), (percentprintingoff!=0), 
                               logname, LOGFILETYPE_NOLIMIT, 0, messagelen, 
                               smtpsrvr, smtpport, smtpfrom, smtpdest, id );
            PrintBanner(id,0,restarted);
            ParseCommandline( 1, argc, argv, NULL, (quietmode==0)); //show overrides
            InitRandom2( id );
          
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
          DeinitializeConnectivity(); //netinit.cpp
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

  #if (CLIENT_OS==OS_WIN16 || CLIENT_OS==OS_WIN32S || CLIENT_OS==OS_WIN32)
  if ( init_success )
    w32ConSetClientPointer( clientP ); // save the client * so we can bail out 
  #endif                               // when we get a WM_ENDSESSION message

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

  #if (CLIENT_OS==OS_WIN16 || CLIENT_OS==OS_WIN32S || CLIENT_OS==OS_WIN32)
  w32ConSetClientPointer( NULL ); // clear the client * 
  #endif

  //------------------------------
  
  if (clientP)
    delete clientP;

  return (retcode);
}


/* ----------------------------------------------------------------- */

#if (CLIENT_OS == OS_MACOS)
//
// nothing - Mac framework provides main
//
#elif ((CLIENT_OS == OS_WIN32) && defined(WIN32GUI))
//
// nothing - realmain() is called from elsewhere
//
#elif (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
int PASCAL WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpszCmdLine, int nCmdShow) 
{ /* abstraction layer between WinMain() and realmain() */
  return winClientPrelude( hInst, hPrevInst, lpszCmdLine, nCmdShow, realmain);
}
#else
int main( int argc, char *argv[] )
{ 
  return realmain( argc, argv ); 
}
#endif
