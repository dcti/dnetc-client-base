/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *client_cpp(void) {
return "@(#)$Id: client.cpp,v 1.213 1999/07/26 04:24:12 sampo Exp $"; }

/* ------------------------------------------------------------------------ */

//#define TRACE

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "client.h"    // Client class
#include "random.h"    // InitRandom()
#include "pathwork.h"  // EXTN_SEP
#include "clitime.h"   // CliTimer()
#include "util.h"      // projectmap_build() and trace
#include "modereq.h"   // ModeReqIsSet()/ModeReqRun()
#include "triggers.h"  // [De]InitializeTriggers(),RestartRequestTrigger()
#include "logstuff.h"  // [De]InitializeLogging(),Log()/LogScreen()
#include "console.h"   // [De]InitializeConsole(), ConOutErr()
#include "network.h"   // [De]InitializeConnectivity()

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
  sprintf(buffer,"Windows%s %u.%u", (ver>=2000)?("NT"):(""), (ver/100)%20, ver%100 );
  return buffer;
#elif defined(__unix__) /* uname -sr */
  struct utsname ut;
  if (uname(&ut)==0) {
    static char buffer[sizeof(ut.sysname)+1+sizeof(ut.release)+1];
    return strcat(strcat(strcpy(buffer,ut.sysname)," "),ut.release);
  }
  return "";
#else
  return "";
#endif
}

/* ---------------------------------------------------------------------- */

static void PrintBanner(const char *dnet_id,int level,int restarted)
{
  /* level = 0 = show copyright/version,  1 = show startup message */
  restarted = 0; /* yes, always show everything */
  
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
                       "client for %s%s%s%s.\n",
            ((ConIsGUI())?('G'):('C')),  CliGetTimeString(&tv,4),
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

  TRACE_OUT((+1,"Client.Main()\n"));
  do
  {
    int restarted = restart;
    restart = 0;

    __initialize_client_object(this); /* reset everything in the object */
    //ReadConfig() and parse command line - returns !0 if shouldn't continue

    TRACE_OUT((0,"Client.parsecmdline restarted?: %d\n", restarted));
    if (ParseCommandline( 0, argc, argv, &retcode, 0 ) == 0)
    {
      int domodes = (ModeReqIsSet(-1) != 0);
      TRACE_OUT((0,"initializetriggers\n"));
      if (InitializeTriggers(((noexitfilecheck ||
                              domodes)?(NULL):("exitrc5" EXTN_SEP "now")),
                              ((domodes)?(NULL):(pausefile)) )==0)
      {
        TRACE_OUT((0,"initializeconnectivity\n"));
        if (InitializeConnectivity() == 0) //do global initialization
        {
          TRACE_OUT((0,"initializeconsole\n"));
          if (InitializeConsole(quietmode,domodes) == 0)
          {
            TRACE_OUT((+1,"initializelogging\n"));
            InitializeLogging( (quietmode!=0), (percentprintingoff!=0),
                               logname, logfiletype, logfilelimit, 
                               messagelen, smtpsrvr, smtpport, smtpfrom, 
                               smtpdest, id );
            TRACE_OUT((-1,"initializelogging\n"));
            PrintBanner(id,0,restarted);
            TRACE_OUT((+1,"parsecmdline(1)\n"));
            ParseCommandline( 1, argc, argv, NULL, (quietmode==0)); //show overrides
            TRACE_OUT((-1,"parsecmdline(1)\n"));
            InitRandom2( id );

            if (domodes)
            {
              TRACE_OUT((+1,"modereqrun\n"));
              ModeReqRun( this );
              TRACE_OUT((-1,"modereqrun\n"));
            }
            else
            {
              PrintBanner(id,1,restarted);
              TRACE_OUT((+1,"selectcore\n"));
              SelectCore( 0 );
              TRACE_OUT((-1,"selectcore\n"));
              TRACE_OUT((+1,"client.run\n"));
              retcode = Run();
              TRACE_OUT((-1,"client.run\n"));
              restart = CheckRestartRequestTrigger();
            }
            DeinitializeLogging();
            DeinitializeConsole();
          }
          TRACE_OUT((0,"deinitializeconsole\n"));
          DeinitializeConnectivity(); //netinit.cpp
        }
        TRACE_OUT((0,"deinitializeconnectivity\n"));
        DeinitializeTriggers();
      }
    }
    TRACE_OUT((0,"client.parsecmdline restarting?: %d\n", restart));
  } while (restart);

  TRACE_OUT((-1,"Client.Main()\n"));
  return retcode;
}

/* ---------------------------------------------------------------------- */

static int realmain( int argc, char *argv[] ) /* YES, *STATIC* */
{
  // This is the main client object.  we 'new'/malloc it, rather than make
  // it static in the hope that people will think twice about using exit()
  // or otherwise breaking flow. (wanna bet it'll happen anyway?)
  // The if (success) thing is for nesting without {} nesting. - cyp
  Client *clientP = NULL;
  int retcode = -1, init_success = 1;
  srand( (unsigned) time(NULL) );
  InitRandom();

  TRACE_OUT((+1,"realmain()\n"));

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
    void *mbuf = malloc(sizeof(Client)*2); //use this roundabout way
    if (mbuf) //to ensure we have enough heap space to create the object
    {
      free(mbuf);
      init_success = (( clientP = new Client() ) != NULL);
    }
    if (!init_success)
      ConOutErr( "Unable to create client object. Out of memory." );
  }

  //----------------------------

  #if (CLIENT_OS == OS_NETWARE)
  //set cwd etc. save ptr to client for fnames/niceness
  if ( init_success )
    init_success = ( nwCliInitClient( argc, argv, clientP ) == 0);
  #endif

  //----------------------------

  #if (CLIENT_OS==OS_WIN16 || CLIENT_OS==OS_WIN32S || CLIENT_OS==OS_WIN32)
  if ( init_success )
    w32ConSetClientPointer( clientP ); // save the client * so we can bail out
  #endif                               // when we get a WM_ENDSESSION message

  //----------------------------

  #if (CLIENT_OS == OS_MACOS)
  if ( init_success )                    // save the client * so we can 
    macCliSetClientPointer( clientP ); // load it when appropriate
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

  #if (CLIENT_OS==OS_WIN16 || CLIENT_OS==OS_WIN32S || CLIENT_OS==OS_WIN32)
  w32ConSetClientPointer( NULL ); // clear the client *
  #endif

  //------------------------------

  #if (CLIENT_OS == OS_MACOS)
  if ( init_success )                    
    macCliSetClientPointer( NULL ); // clear the client *
  #endif

  //------------------------------

  if (clientP)
    delete clientP;

  TRACE_OUT((-1,"realmain()\n"));

  return (retcode);
}


/* ----------------------------------------------------------------- */

#if (CLIENT_OS == OS_MACOS)
#ifdef CLIENT_17
void main(void)
{
  macosCliMain(realmain); /* sythesise a command line for realmain */
  return;                 /* UI will be initialized later via console.cpp */
}
#endif  
#elif (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
int PASCAL WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpszCmdLine, int nCmdShow)
{ /* parse the command line and call the bootstrap */
  TRACE_OUT((+1,"WinMain()\n"));
  int rc=winClientPrelude( hInst, hPrevInst, lpszCmdLine, nCmdShow, realmain);
  TRACE_OUT((-1,"WinMain()\n"));
  return rc;
}
#elif defined(__unix__)
int main( int argc, char *argv[] )
{
  /* the SPT_* constants refer to sendmail source (conf.[c|h]) */
  char defname[]={('r'),('c'),('5'),('d'),('e'),('s'),0};
  int needchange = 0;
  if (argv && argv[0])
  {
    char *p = strrchr( argv[0], '/' );
    needchange = (strcmp( ((p)?(p+1):(argv[0])), defname )!=0);
  }
  #if (CLIENT_OS == OS_HPUX)                         //SPT_TYPE == SPT_PSTAT
  if (needchange)
  {
    union pstun pst;
    pst.pst_command = &defname[0];
    pstat(PSTAT_SETCMD,pst,strlen(defname),0,0);
  }
  #elif (CLIENT_OS == OS_SCO)                        //SPT_TYPE == SPT_SCO
  if (needchange)
  {
    #ifndef _PATH_KMEM /* should have been defined in <paths.h> */
    # define _PATH_KMEM "/dev/kmem"
    #endif
    int kmem = open(_PATH_KMEM, O_RDWR, 0);
    if (kmem >= 0)
    {  
      //pid_t kmempid = getpid();
      struct user u;
      off_t seek_off;
      char buf[PSARGSZ];
      (void) fcntl(kmem, F_SETFD, 1);
      memset( buf, 0, PSARGSZ );
      strncpy( buf, defname, PSARGSZ );
      buf[PSARGSZ - 1] = '\0';
      seek_off = UVUBLK + (off_t) u.u_psargs - (off_t) &u;
      if (lseek(kmem, (off_t) seek_off, SEEK_SET) == seek_off)
        (void) write(kmem, buf, PSARGSZ);
      /* yes, it stays open */
    }
  }
  #elif 0 /* eg, Rhapsody */                  /* SPT_TYPE==SPT_PSSTRINGS */
  if (needchange)
  {
    PS_STRINGS->ps_nargvstr = 1;
    PS_STRINGS->ps_argvstr = defname;
  }
  #elif 0                                     /*  SPT_TYPE == SPT_SYSMIPS */
  if (needchange)
  {
    sysmips(SONY_SYSNEWS, NEWS_SETPSARGS, buf);
  }
  #else //SPT_TYPE == SPT_CHANGEARGV (mach) || SPT_NONE (OS_DGUX|OS_DYNIX)
        //|| SPT_REUSEARGV (the rest)
  /* [Net|Free|Open|BSD[i]] are of type SPT_BUILTIN, ie use setproctitle() 
     which is a stupid smart-wrapper for SPT_PSSTRINGS (see above). The result
     is exactly the same as when we reexec(): ps will show "rc5des (filename)"
     so we gain nothing other than avoiding the exec() call, but the way /we/ 
     would use it is non-standard, so we'd better leave it be:
     __progname = defname; setproctitle(NULL); //uses default, ie __progname
  */
  #if (CLIENT_OS != OS_FREEBSD) && (CLIENT_OS != OS_NETBSD) && \
      (CLIENT_OS != OS_BSDOS) && (CLIENT_OS != OS_OPENBSD) && \
      (CLIENT_OS != OS_DGUX) && (CLIENT_OS != OS_DYNIX)
  /* ... all the SPT_REUSEARGV types */
  if (needchange && strlen(argv[0]) >= strlen(defname))
  {
    char *q = "RC5PROG";
    int didset = 0;
    #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_IRIX)
    char *m = (char *)malloc( strlen(q)+1+strlen(argv[0])+1 );
    if (m) {
      didset=(0==putenv(strcat(strcat(strcpy(m,q),"="),argv[0]))); //BSD4.3
      free((void *)m); 
    }
    #else
    didset = (setenv( q, argv[0], 1 ) == 0); //SYSV7 and posix
    #endif
    if (didset)
    {
      if ((q = (char *)getenv(q)) != ((char *)0))
      {
        int padchar = 0; /* non-linux/qnx/aix may need ' ' here */
        memset(argv[0],padchar,strlen(argv[0]));
        memcpy(argv[0],defname,strlen(defname));
        argv[0] = q;
        needchange = 0;
      }
    }
  }
  #endif
  if (needchange)
  {
    char buffer[512];
    unsigned int len;  char *p;
    if (getenv("RC5INI") == NULL)
    {
      int have_ini = 0;
      for (len=1;len<((unsigned int)argc);len++)
      {
        p = argv[len];
        if (*p == '-')
        {
          if (p[1]=='-') 
            p++;
          if (strcmp(p,"-ini")==0)
          {
            have_ini++;
            break;
          }
        }
      }
      if (!have_ini)
      {
        strcpy( buffer, "RC5INI=" );
        strncpy( &buffer[7], argv[0], sizeof(buffer)-7 );
        buffer[sizeof(buffer)-5]='\0';
        strcat( buffer, ".ini" );
        #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_IRIX)
        putenv( buffer );                 //BSD4.3
        #else
        setenv("RC5INI", &buffer[7], 1 ); //SYSV7 and posix
        #endif
      }
    }
      
    p = argv[0];
    argv[0] = defname;
    if ((strlen(p) + 5) < sizeof(buffer))
    {
      char *s;
      strcpy( buffer, p );
      s = strrchr( buffer, '/' );
      if (s != NULL)
      {
        s[1]='\0';
        strcat( buffer, defname );
        argv[0] = buffer;
      }
    }

    if ( execvp( p, &argv[0] ) == -1)
    {
      ConOutErr("Unable to restart self");
      return -1;
    }
    return 0;
  }
  #endif
  return realmain( argc, argv );
}      
#else
int main( int argc, char *argv[] )
{
  return realmain( argc, argv );
}
#endif

