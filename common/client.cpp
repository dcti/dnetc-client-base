/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *client_cpp(void) {
return "@(#)$Id: client.cpp,v 1.206.2.95 2000/11/22 19:04:53 cyp Exp $"; }

/* ------------------------------------------------------------------------ */

//#define TRACE

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "client.h"    // Client class
#include "cliident.h"  // CliGetFullVersionDescriptor()
#include "clievent.h"  // ClientEventSyncPost(),_CLIENT_STARTED|FINISHED
#include "random.h"    // InitRandom()
#include "pathwork.h"  // EXTN_SEP
#include "util.h"      // projectmap_build(), trace, utilCheckIfBetaExpired
#include "modereq.h"   // ModeReqIsSet()/ModeReqRun()
#include "cmdline.h"   // ParseCommandLine() and load config
#include "clitime.h"   // [De]InitializeTimers(),CliTimer()
#include "netbase.h"   // net_[de]initialize()
#include "triggers.h"  // [De]InitializeTriggers(),RestartRequestTrigger()
#include "logstuff.h"  // [De]InitializeLogging(),Log()/LogScreen()
#include "console.h"   // [De]InitializeConsole(), ConOutErr()
#include "selcore.h"   // [De]InitializeCoreTable()

/* ------------------------------------------------------------------------ */

void ResetClientData(Client *client)
{
  /* everything here should also be validated in the appropriate subsystem,
     so if zeroing here causes something to break elsewhere, the subsystem
     init needs fixing.
     When creating new variables, name them such that the default is 0/"",
     eg 'nopauseonbatterypower'.
  */

  int contest;
  memset((void *)client,0,sizeof(Client));

  client->nettimeout=60;
  client->autofindkeyserver=1;
  client->crunchmeter=-1;
  projectmap_build(client->loadorder_map,"");
  client->numcpu = -1;
  for (contest=0; contest<CONTEST_COUNT; contest++)
    client->coretypes[contest] = -1;
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
#elif ((CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16))
  static char buffer[32]; long ver = winGetVersion(); /* w32pre.cpp */
  sprintf(buffer,"Windows%s %u.%u", (ver>=2000)?("NT"):(""), (ver/100)%20, ver%100 );
  return buffer;
#elif (CLIENT_OS == OS_NETWARE)
  static char buffer[40];
  const char *speshul = "";
  int major, minor, revision, servType, loaderType;
  major = GetFileServerMajorVersionNumber();
  minor = GetFileServerMinorVersionNumber();
  revision = GetFileServerRevisionNumber();
  #if 0 /* hmm, this is wrong. NetWare 4.02 != 4.2 */
  if (minor < 10) 
    minor *= 10; /* .02 => .20 */
  #endif  
  revision = ((revision == 0 || revision > 26)?(0):(revision + ('a'-1)));
  GetServerConfigurationInfo(&servType, &loaderType);
  if (servType == 0 && loaderType > 1) /* OS/2/UnixWare/etc loader */
    speshul = " (nondedicated)";
  else if (servType == 1)
    speshul = " (SFT III IOE)";
  else if (servType == 2)
    speshul = " (SFT III MSE)";
  sprintf(buffer, "NetWare%s %u.%u%c", speshul, major, minor, revision );
  return buffer;
#elif (CLIENT_OS == OS_MACOS)
  const char *osname = "Mac OS";
  long osversion;
  if ( Gestalt( gestaltAUXVersion, &osversion ) != noErr)
    osversion = 0;
  if (osversion != 0)
    osname = "A/UX";
  else if ( Gestalt( gestaltSystemVersion, &osversion ) != noErr)
    osversion = 0;
  if ((osversion & 0xffff) != 0)
  {
    static char buffer[40];
    sprintf( buffer, "%s %d.%d%c%d", osname,
      (int)((osversion&0xff00)>>8), (int)((osversion&0x00f0)>>4),
      (((osversion & 0x000f) == 0)?(0):('.')), (int)((osversion&0x000f)) );
    return buffer;
  }
  return "";
#elif (CLIENT_OS == OS_AMIGAOS)
  static char buffer[40];
  #ifdef __PPC__
    #ifdef __POWERUP__
    #define PPCINFOTAG_EMULATION (TAG_USER + 0x1f0ff)
    sprintf(buffer,"OS %s, PowerUp%s %ld.%ld",amigaGetOSVersion(),((PPCGetAttr(PPCINFOTAG_EMULATION) == 'WARP') ? " Emu" : ""),PPCVersion(),PPCRevision());
    #else
    #define LIBVER(lib) *((UWORD *)(((UBYTE *)lib)+20))
    #define LIBREV(lib) *((UWORD *)(((UBYTE *)lib)+22))
    sprintf(buffer,"OS %s, WarpOS %d.%d",amigaGetOSVersion(),LIBVER(PowerPCBase),LIBREV(PowerPCBase));
    #endif
  #else
  sprintf(buffer,"OS %s, 68K",amigaGetOSVersion());
  #endif
  return buffer;
#elif defined(__unix__) /* uname -sr */
  struct utsname ut;
  if (uname(&ut)==0) 
  {
    #if (CLIENT_OS == OS_AIX)
    // on AIX version is the major and release the minor
    static char buffer[sizeof(ut.sysname)+1+sizeof(ut.release)+1+sizeof(ut.version)+1];
    return strcat(strcat(strcat(strcat(strcpy(buffer,ut.sysname)," "),ut.version),"."),ut.release);
    #else
    static char buffer[sizeof(ut.sysname)+1+sizeof(ut.release)+1];
    return strcat(strcat(strcpy(buffer,ut.sysname)," "),ut.release);
    #endif
  }
  return "";
#else
  return "";
#endif
}

/* ---------------------------------------------------------------------- */

static void PrintBanner(const char *dnet_id,int level,int restarted,int logscreenonly)
{
  restarted = 0; /* yes, always show everything */
  int logto = LOGTO_RAWMODE|LOGTO_SCREEN;
  if (!logscreenonly)
    logto |= LOGTO_FILE|LOGTO_MAIL;

  if (!restarted)
  {
    if (level == 0)
    {
      LogScreenRaw( "\ndistributed.net client for " CLIENT_OS_NAME " "
                    "Copyright 1997-2000, distributed.net\n");
      #if (CLIENT_CPU == CPU_68K)
      LogScreenRaw( "RC5 68K assembly by John Girvin\n");
      #endif
      #if (CLIENT_CPU == CPU_POWERPC)
      LogScreenRaw( "RC5 PowerPC and AltiVec assembly by Dan Oetting\n"
                    "Enhancements for 604e CPUs by Roberto Ragusa\n");
      #endif
      #if (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32)
      LogScreenRaw( "RC5 Alpha assembly by Mike Marcelais\n");
      #endif
      #if (CLIENT_CPU == CPU_ARM)
      LogScreenRaw( "RC5 ARM assembly by Steve Lee\n");
      #if (CLIENT_OS == OS_RISCOS)
      LogScreenRaw( "RISCOS/PC Card support by Dominic Plunkett\n");
      #endif
      #endif
      #if defined(HAVE_DES_CORES)
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
      #endif
      #if (CLIENT_OS == OS_DOS)
      LogScreenRaw( "PMODE DOS extender Copyright 1994-1998, Charles Scheffold and Thomas Pytel\n");
      #endif

      LogScreenRaw( "Please visit http://www.distributed.net/ for up-to-date contest information.\n");
      #if (!CLIENT_OS == OS_MACOS)
      LogScreenRaw( "Start the client with '-help' for a list of valid command line options.\n" );
      #endif
      LogScreenRaw( "\n" );
    }
    else if ( level == 1 || level == 2 )
    {
      const char *msg = GetBuildOrEnvDescription();
      if (msg == NULL) msg="";

      LogTo( logto, "\n%s%s%s%s.\n",
                    CliGetFullVersionDescriptor(), /* cliident.cpp */
                    ((*msg)?(" ("):("")), msg, ((*msg)?(")"):("")) );
      LogScreenRaw( "Please provide the *entire* version descriptor "
                    "when submitting bug reports.\n");
      LogScreenRaw( "The distributed.net bug report pages are at "
                    "http://www.distributed.net/bugs/\n");
      if (dnet_id[0] != '\0' && strcmp(dnet_id,"rc5@distributed.net") !=0 )
        LogTo( logto, "Using email address (distributed.net ID) \'%s\'\n",dnet_id);
      else if (level == 2)
        LogTo( logto, "\n* =========================================================================="
                      "\n* The client is not configured with your email address (distributed.net ID) "
                      "\n* Work done cannot be credited until it is set. Please run '%s -config'"
                      "\n* =========================================================================="
                      "\n", utilGetAppName());
      LogTo( logto, "\n");
      #if CLIENT_OS == OS_IRIX
      /*
       * Irix 6.5 has a kernel bug that will cause the system
       * to freeze if we're running as root on a multi-processor
       * machine and we try to use more than one CPU.
       */
      if (geteuid() == 0) {
	       LogScreenRaw("* Cannot run as the superuser on Irix.\n"
		                  "* Please run the client under a non-zero uid.\n\n");
         exit(1);
      }
      #endif /* CLIENT_OS == OS_IRIX */

      if (CliIsTimeZoneInvalid()) /*clitime.cpp (currently DOS,OS/2,WIN[16] only)*/
      {
        LogScreenRaw("Warning: The TZ= variable is not set in the environment. "
         "The client will\nprobably display the wrong time and/or select the "
         "wrong keyserver.\n");
      }
    }
  }
  return;
}
/* ---------------------------------------------------------------------- */

static int ClientMain( int argc, char *argv[] )
{
  Client *client = (Client *)0;
  int retcode = 0;
  int restart = 0;

  TRACE_OUT((+1,"Client.Main()\n"));

  if (InitializeTimers()!=0)
  {
    ConOutErr( "Unable to initialize timers." );
    return -1;
  }
  client = (Client *)malloc(sizeof(Client));
  if (!client)
  {
    DeinitializeTimers();
    ConOutErr( "Unable to initialize client. Out of memory." );
    return -1;
  }
  srand( (unsigned) time(NULL) );
  InitRandom();
  
  do    
  {
    int restarted = restart;
    restart = 0;

    ResetClientData(client); /* reset everything in the object */
    ClientEventSyncPost( CLIEVENT_CLIENT_STARTED, *((long*)(&client)) );
    //ReadConfig() and parse command line - returns !0 if shouldn't continue

    TRACE_OUT((0,"Client.parsecmdline restarted?: %d\n", restarted));
    if (!ParseCommandline(client,0,argc,(const char **)argv,&retcode,restarted))
    {                               
      int domodes = ModeReqIsSet(-1); /* get current flags */
      TRACE_OUT((0,"initializetriggers\n"));
      if (InitializeTriggers(domodes, 
                             ((domodes)?(""):(client->exitflagfile)),
                             client->pausefile,
                             client->pauseplist,
                             ((domodes)?(0):(client->restartoninichange)),
                             client->inifilename,
                             client->watchcputempthresh, client->cputempthresh,
                             (!client->nopauseifnomainspower) ) == 0)
      {
        TRACE_OUT((0,"CheckExitRequestTrigger()=%d\n",CheckExitRequestTrigger()));
        // no messages from the following CheckExitRequestTrigger() will
        // ever be seen. Tough cookies. We're not about to implement pre-mount
        // type kernel msg logging for the few people with exit flagfile.
        if (!CheckExitRequestTrigger())
        {
          TRACE_OUT((0,"initializeconnectivity\n"));
          if (net_initialize() == 0) //do global initialization
          {
            #ifdef LURK /* start must come just after initializeconnectivity */
            TRACE_OUT((0,"LurkStart()\n")); /*and always before initlogging*/
            LurkStart(client->offlinemode, &(client->lurk_conf));
            #endif
            TRACE_OUT((0,"initializeconsole\n"));
            if (InitializeConsole(&(client->quietmode),domodes) == 0)
            {
              //some plats need to wait for user input before closing the screen
              int con_waitforuser = 0; //only used if doing modes (and !-config)
  
              TRACE_OUT((+1,"initializelogging\n"));
              InitializeLogging( (client->quietmode!=0),
                                 client->crunchmeter,
                                 0, /* nobaton */
                                 client->logname,
                                 client->logfiletype,
                                 client->logfilelimit,
                                 ((domodes)?(0):(client->messagelen)),
                                 client->smtpsrvr, 0,
                                 client->smtpfrom,
                                 client->smtpdest,
                                 client->id );
              TRACE_OUT((-1,"initializelogging\n"));
              if ((domodes & MODEREQ_CONFIG)==0)
              {
                PrintBanner(client->id,0,restarted,0);

                TRACE_OUT((+1,"parsecmdline(1)\n"));
                if (!client->quietmode) 
                {                 //show overrides
                  ParseCommandline( client, 1, argc, (const char **)argv,
                                    &retcode, restarted ); 
                }                      
                TRACE_OUT((-1,"parsecmdline(1)\n"));
              }
              InitRandom2( client->id );
              TRACE_OUT((+1,"initcoretable\n"));
              InitializeCoreTable( &(client->coretypes[0]) );
              TRACE_OUT((-1,"initcoretable\n"));

              if (domodes)
              {
                con_waitforuser = ((domodes & ~MODEREQ_CONFIG)!=0);
                if ((domodes & MODEREQ_CONFIG)==0) 
                { /* avoid printing/logging banners for nothing */
                  PrintBanner(client->id,1,restarted,((domodes & MODEREQ_CMDLINE_HELP)!=0));
                }
                TRACE_OUT((+1,"modereqrun\n"));
                ModeReqRun( client );
                TRACE_OUT((-1,"modereqrun\n"));
                restart = 0; /* *never* restart when doing modes */
              }
              else if (!utilCheckIfBetaExpired(1)) /* prints message */
              {
                con_waitforuser = 0;
                PrintBanner(client->id,2,restarted,0);
                TRACE_OUT((+1,"client.run\n"));
                retcode = ClientRun(client);
                TRACE_OUT((-1,"client.run\n"));
                restart = CheckRestartRequestTrigger();
              }

              TRACE_OUT((0,"deinit coretable\n"));
              DeinitializeCoreTable();

              TRACE_OUT((0,"deinitialize logging\n"));
              DeinitializeLogging();
              TRACE_OUT((0,"deinitialize console (waitforuser? %d)\n",con_waitforuser));
              DeinitializeConsole(con_waitforuser);
            } /* if (InitializeConsole() == 0) */
            #ifdef LURK
            TRACE_OUT((0,"LurkStop()\n"));
            LurkStop(); /* just before DeinitializeConnectivity() */
            #endif
            TRACE_OUT((0,"deinitialize connectivity\n"));
            net_deinitialize(!restart /* final call */);
          } /* if (InitializeConnectivity() == 0) */
        } /* if (!CheckExitRequestTrigger()) */
        TRACE_OUT((0,"deinitialize triggers\n"));
        DeinitializeTriggers();
      }
    }
    ClientEventSyncPost( CLIEVENT_CLIENT_FINISHED, (long)restart );
    TRACE_OUT((0,"client.parsecmdline restarting?: %d\n", restart));
  } while (restart);

  free((void *)client);
  DeinitializeTimers();

  TRACE_OUT((-1,"Client.Main()\n"));
  return retcode;
}

/* ---------------------------------------------------------------------- */

#if (CLIENT_OS == OS_MACOS)
int main( void )
{
  char *argv[2];
  ((const char **)argv)[0] = utilGetAppName();
  argv[1] = (char *)0;
  macosInitialize();
  ClientMain(1,argv);
  return 0;
}
#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)
//if you get compile or link errors it is probably because you compiled with
//STRICT, which is a no-no when using cpp (think 'overloaded')
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "w32pre.h"       // prelude
int PASCAL WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpszCmdLine, int nCmdShow)
{ /* parse the command line and call the bootstrap */
  TRACE_OUT((+1,"WinMain()\n"));
  int rc=winClientPrelude( hInst, hPrevInst, lpszCmdLine, nCmdShow, ClientMain);
  TRACE_OUT((-1,"WinMain()\n"));
  return rc;
}
#elif defined(__unix__)
int main( int argc, char *argv[] )
{
  /* the SPT_* constants refer to sendmail source (conf.[c|h]) */
  const char *defname = utilGetAppName(); /* 'rc5/des' or 'dnetc' etc */
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
    pst.pst_command = (char *)defname;
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
    PS_STRINGS->ps_argvstr = (char *)defname;
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
     is exactly the same as when we reexec(): ps will show "appname (filename)"
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
    #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_IRIX) || \
        (CLIENT_OS == OS_AIX) || (CLIENT_OS == OS_BEOS)
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
        #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_IRIX) || \
            (CLIENT_OS == OS_AIX) || (CLIENT_OS == OS_BEOS)
        putenv( buffer );                 //BSD4.3
        #else
        setenv("RC5INI", &buffer[7], 1 ); //SYSV7 and posix
        #endif
      }
    }

    p = argv[0];
    ((const char **)argv)[0] = defname;
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
  return ClientMain( argc, argv );
}
#elif (CLIENT_OS == OS_RISCOS)
int main( int argc, char *argv[] )
{
  riscos_in_taskwindow = riscos_check_taskwindow();
  if (riscos_find_local_directory(argv[0]))
    return -1;
  return ClientMain( argc, argv );
}
#elif (CLIENT_OS == OS_NETWARE)
int main( int argc, char *argv[] )
{
  int rc = 0;
  if ( nwCliInitClient( argc, argv ) != 0) /* set cwd etc */
    return -1;
  rc = ClientMain( argc, argv );
  nwCliExitClient(); // destroys AES process, screen, polling procedure
  return rc;
}
#elif (CLIENT_OS == OS_AMIGAOS)
int main( int argc, char *argv[] )
{
  int rc = 20;
  if (amigaInit())
  {  
    rc = ClientMain( argc, argv );
    if (rc) rc = 5; //Warning
    amigaExit();
  }
  return rc;
}
#else
int main( int argc, char *argv[] )
{
  int rc;
  TRACE_OUT((+1,"main()\n"));
  rc = ClientMain( argc, argv );
  TRACE_OUT((-1,"main()\n"));
  return rc;
}
#endif

