/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *client_cpp(void) {
return "@(#)$Id: client.cpp,v 1.204 1999/04/20 02:41:08 cyp Exp $"; }

/* ------------------------------------------------------------------------ */

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "client.h"    // Client class
#include "random.h"    // InitRandom()
#include "pathwork.h"  // EXTN_SEP
#include "clitime.h"   // CliTimer()
#include "util.h"      // projectmap_build()
#include "modereq.h"   // ModeReqIsSet()/ModeReqRun()
#include "triggers.h"  // [De]InitializeTriggers(),RestartRequestTrigger()
#include "logstuff.h"  // [De]InitializeLogging(),Log()/LogScreen()
#include "console.h"   // [De]InitializeConsole(), ConOutErr()
#include "network.h"   // [De]InitializeConnectivity()

/* ------------------------------------------------------------------------ */

#if (CLIENT_OS == OS_AMIGAOS)
#if (CLIENT_CPU == CPU_68K)
#error please put this in your ./platforms/amigaos/
long __near __stack  = 65536L;  // AmigaOS has no automatic stack extension
      // seems standard stack isn't enough
#endif // (CLIENT_CPU == CPU_68K)
#endif // (CLIENT_OS == OS_AMIGAOS)

#if (CLIENT_OS == OS_RISCOS)
#error please put this in your ./platforms/riscos/ and make it
#error riscosClientIsGui(). guirestart ought now to be obsolete.
s32 guiriscos, guirestart;
#endif

/* ------------------------------------------------------------------------ */

static void __initialize_client_object(Client *client)
{
  /* everything here should also be validated in the appropriate subsystem, 
     so if zeroing here causes something to break elsewhere, the subsystem 
     init needs fixing. 
     Variables are initialized in the same order that they are declared in
     the client object. Keep it that way to ensure we don't miss something.
   */

  int contest;

  /* non-user-configurable variables */
  client->nonewblocks=0;
  client->randomchanged=0;
  client->randomprefix=100;
  client->rc564closed=0;
  client->stopiniio=0;
  client->scheduledupdatetime = 0;
  client->inifilename[0]=0;
  for (contest=0; contest<CONTEST_COUNT; contest++)
    memset((void *)&(client->membufftable[contest]),0,sizeof(client->membufftable[0]));

  /* -- general -- */
  strcpy(client->id, "rc5@distributed.net" );
  client->quietmode=0;
  client->blockcount = 0;
  client->minutes = 0;
  client->percentprintingoff=0;
  client->noexitfilecheck=0;
  client->pausefile[0]=0;
  projectmap_build(client->loadorder_map,"");

  /* -- buffers -- */
  client->nodiskbuffers=0;
  client->in_buffer_basename[0] = '\0';
  client->out_buffer_basename[0] = '\0';
  client->checkpoint_file[0]=0;
  client->offlinemode = 0;
    /* -- net -- */
    client->nettimeout=60;
    client->nofallback=0;
    client->autofindkeyserver=1;
    client->keyproxy[0] = 0;
    client->keyport = 0;
    client->httpproxy[0] = 0;
    client->httpport = 0;
    client->uuehttpmode = 0;
    client->httpid[0] = 0;
  client->noupdatefromfile = 0;
    client->remote_update_dir[0] = '\0';
  client->connectoften=0;
  client->preferred_blocksize=31;
  for (contest=0; contest<CONTEST_COUNT; contest++)
    client->inthreshold[contest] = client->outthreshold[contest] = 10;

  /* -- perf -- */
  client->numcpu = -1;
  client->cputype = -1;
  client->priority = 0;

  /* -- log -- */
  client->logname[0]= 0;
  client->logfiletype[0] = 0;
  client->logfilelimit[0] = 0;
  client->messagelen = 0;
  client->smtpport = 25;
  client->smtpsrvr[0]=0;
  client->smtpfrom[0]=0;
  client->smtpdest[0]=0;
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
  #elif defined(BDESCRIP)
  return BDESCRIP;
  #else
  return "";
  #endif
}

int ClientIsGUI(void)
{
  #if defined(WIN32GUI) || defined(MAC_GUI) || defined(OS2_PM)
  return 1;
  #elif (CLIENT_OS == OS_RISCOS)
  return (guiriscos!=0);
  #else
  return 0;
  #endif
}

/* ---------------------------------------------------------------------- */

void PrintBanner(const char *dnet_id,int level,int restarted)
{
  /* level = 0 = show copyright/version,  1 = show startup message */
  if (!restarted)
  {
    if (level == 0)
    {
      LogScreenRaw( "\nRC5DES client - a project of distributed.net\n"
                    "Copyright 1997-1999 distributed.net\n");

      #if (CLIENT_CPU == CPU_68K)
      LogScreenRaw( "RC5 68K assembly by John Girvin\n");
      #endif
      #if (CLIENT_CPU == CPU_POWERPC)
      LogScreenRaw( "RC5 PowerPC assembly by Dan Oetting\n");
      #endif
      #if (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32)
      LogScreenRaw( "RC5 Alpha assembly by Mike Marcelais\n");
      #endif
      #if (CLIENT_CPU == CPU_ARM)
      LogScreenRaw( "ARM assembly by Steve Lee\n");
      #if (CLIENT_OS == OS_RISCOS)
      LogScreenRaw( "RISCOS/PC Card support by Dominic Plunkett\n");
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
           (CLIENT_OS==OS_WIN32))
      #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
      if ((winGetVersion()%2000) < 400) /* w32pre.cpp. +2000==NT */
      #endif
      if (getenv("TZ") == NULL)
      {
        LogScreenRaw("Warning: The TZ= variable is not set in the environment. "
         "The client will\nprobably display the wrong time and/or select the "
         "wrong keyserver.\n");
        putenv("TZ=GMT+0"); //use local, not redmond or armonk or saltlakecity time
      }
      #endif

      const char *msg = GetBuildOrEnvDescription();
      if (msg == NULL) msg="";

      struct timeval tv; tv.tv_usec = 0; tv.tv_sec = CliTimeGetBuildDate();
      LogRaw("\nRC5DES v" CLIENT_VERSIONSTRING "-"
                       "%c" /* GUI == "G", CLI == "C" */
                       #ifdef CLIENT_SUPPORTS_SMP
                       "T" /* threads */
                       #else
                       "P" /* polling */
                       #endif
                       #if (defined(BETA) || defined(BETA_PERIOD))
                       "L" /* limited release */
                       #else
                       "R" /* public release */
                       #endif
                       "-%s " /* date is in bugzilla format yymmddhh */ 
                       "client for %s%s%s%s started.\n",
            ((ClientIsGUI())?('G'):('C')),  CliGetTimeString(&tv,4),
            CLIENT_OS_NAME, ((*msg)?(" ("):("")), msg, ((*msg)?(")"):("")) );
      LogScreenRaw( "Please provide the *entire* version descriptor "
                    "when submitting bug reports.\n");
      LogScreenRaw( "The distributed.net bug report pages are at "
                    "http://www.distributed.net/bugs/\n");
      LogRaw( "Using email address (distributed.net ID) \'%s\'\n\n", dnet_id );
    }
  }
  return;
}

/* ---------------------------------------------------------------------- */

int Client::Main( int argc, const char *argv[] )
{
  int retcode = 0;
  int restart = 0;

  do
  {
    int restarted = restart;
    restart = 0;
    __initialize_client_object(this); /* reset everything in the object */

    //ReadConfig() and parse command line - returns !0 if shouldn't continue
    if (ParseCommandline( 0, argc, argv, &retcode, 0 ) == 0)
    {
      int domodes = (ModeReqIsSet(-1) != 0);
      if (InitializeTriggers(((noexitfilecheck ||
                              domodes)?(NULL):("exitrc5" EXTN_SEP "now")),
                              ((domodes)?(NULL):(pausefile)) )==0)
      {
        if (InitializeConnectivity() == 0) //do global initialization
        {
          if (InitializeConsole(quietmode,domodes) == 0)
          {
            InitializeLogging( (quietmode!=0), (percentprintingoff!=0),
                               logname, logfiletype, logfilelimit, 
                               messagelen, smtpsrvr, smtpport, smtpfrom, 
                               smtpdest, id );
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

/* ---------------------------------------------------------------------- */

int realmain( int argc, char *argv[] )
{
  // This is the main client object.  we 'new'/malloc it, rather than make
  // it static in the hope that people will think twice about using exit()
  // or otherwise breaking flow. (wanna bet it'll happen anyway?)
  // The if (success) thing is for nesting without {} nesting. - cyp
  Client *clientP = NULL;
  int retcode = -1, init_success = 1;
  srand( (unsigned) time(NULL) );
  InitRandom();

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
#elif ((CLIENT_OS == OS_OS2) && defined(OS2_PM))
// nothing
#else
int main( int argc, char *argv[] )
{
  return realmain( argc, argv );
}
#endif
