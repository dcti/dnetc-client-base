/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *client_cpp(void) {
return "@(#)$Id: client.cpp,v 1.206.2.77 2000/06/04 09:48:09 oliver Exp $"; }

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
#include "clicdata.h"  // CliGetContestWorkUnitSpeed()
#include "cpucheck.h"  // GetNumberOfDetectedProcessors()
#include "util.h"      // projectmap_build(), trace, utilCheckIfBetaExpired
#include "modereq.h"   // ModeReqIsSet()/ModeReqRun()
#include "cmdline.h"   // ParseCommandLine() and load config
#include "clitime.h"   // [De]InitializeTimers(),CliTimer()
#include "triggers.h"  // [De]InitializeTriggers(),RestartRequestTrigger()
#include "logstuff.h"  // [De]InitializeLogging(),Log()/LogScreen()
#include "console.h"   // [De]InitializeConsole(), ConOutErr()
#include "selcore.h"   // [De]InitializeCoreTable()
#include "network.h"   // [De]InitializeConnectivity()

/* ------------------------------------------------------------------------ */

void ResetClientData(Client *client)
{
  /* everything here should also be validated in the appropriate subsystem,
     so if zeroing here causes something to break elsewhere, the subsystem
     init needs fixing.
     Variables are initialized in the same order that they are declared in
     the client object. Keep it that way to ensure we don't miss something.
     When creating new variables, name them such that the default is 0/"",
     eg 'nopauseonbatterypower'.
   */

  int contest;
  memset((void *)client,0,sizeof(Client));

  /* non-user-configurable variables */
  client->nonewblocks=0;
  client->randomchanged=0;
  client->randomprefix=100;
  client->rc564closed=0;
  client->stopiniio=0;
  client->scheduledupdatetime = 0;
  client->inifilename[0]=0;

  /* -- general -- */
  client->id[0]='\0';
  client->quietmode=0;
  client->blockcount = 0;
  client->minutes = 0;
  client->percentprintingoff=0;

  /* -- buffers -- */
  client->nodiskbuffers=0;
  memset((void *)&(client->membufftable[0]),0,sizeof(client->membufftable));
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
  #ifdef LURK 
  memset(&(client->lurk_conf),0,sizeof(client->lurk_conf));
  #endif
  client->connectoften=0;
  memset(&(client->inthreshold[0]),0,sizeof(client->inthreshold));
  memset(&(client->timethreshold[0]),0,sizeof(client->timethreshold));
  #if (!defined(NO_OUTBUFFER_THRESHOLDS))
  minupdateinterval = 0;
  memset(&(client->outhreshold),0,sizeof(client->outhreshold));
  #endif
  memset(&(client->preferred_blocksize[0]),0,sizeof(client->preferred_blocksize));
  projectmap_build(client->loadorder_map,"");

  /* -- perf -- */
  client->numcpu = -1;
  client->priority = 0;
  for (contest=0; contest<CONTEST_COUNT; contest++)
    client->coretypes[contest] = -1;

  /* triggers */
  client->restartoninichange=0;
  client->pauseplist[0]=0;
  client->pausefile[0]=0;
  client->exitflagfile[0]=0;
  client->nopauseifnomainspower=0;
  client->watchcputempthresh=0;
  client->cputempthresh[0]=0;

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

// --------------------------------------------------------------------------

static int numcpusinuse = -1; /* *real* count */
void ClientSetNumberOfProcessorsInUse(int num) /* from probfill.cpp */
{
  numcpusinuse = num;
}

unsigned int ClientGetInThreshold(Client *client, 
                                  int contestid, int force /*=0*/)
{
  // If inthreshold is <=0, then use time exclusively.
  // If time threshold is 0, then use inthreshold exclusively.
  // If inthreshold is <=0, AND time is 0, then use BUFTHRESHOLD_DEFAULT
  // If inthreshold > 0 AND time > 0, then use MAX(inthreshold, effective_workunits(time))
  int thresh = BUFTHRESHOLD_DEFAULT;

  // OGR time threshold NYI
  client->timethreshold[OGR] = 0;

  if (contestid < CONTEST_COUNT)
  {
    thresh = client->inthreshold[contestid];
    if (thresh <= 0)
      thresh = BUFTHRESHOLD_DEFAULT; /* default (if time is also zero) */

    if (client->timethreshold[contestid] > 0) /* use time */
    {
      int proc;
      unsigned int sec;
      int timethresh = 0;
      proc = GetNumberOfDetectedProcessors();
      if (proc < 1)
        proc = 1;

      // get the speed
      sec = CliGetContestWorkUnitSpeed(contestid, force);
      if (sec != 0) /* we have a rate */
        timethresh = 1 + (client->timethreshold[contestid] * 3600 * proc/sec);
        
      if (timethresh > client->inthreshold[contestid])
        thresh = timethresh;
    }
  }
  return thresh;
}

/* 
   How thresholds affect contest rotation and contest fallover:

   For contest rotation, outthreshold checks (and connectoften) must be 
   disabled, otherwise the client will update before it hits the end of 
   the load_order, resulting in more work becoming available for all 
   projects.
   Inversely, for contest fallover, outthreshold checks (or connectoften) 
   must be enabled.

   Example scenarios:
   1) User wants to run both OGR and RC5, allocating 3 times as much cpu
      time to OGR than to RC5. These would be the required settings:
      load_order=OGR,RC5      (the order in which the client LOOKs for work)
      inthresholds=OGR=6,RC5=2    (OGR thresh is 3 times RC5 thresh)
      outthresholds=OGR=0,RC5=0 (disable outthresh checking)  
      what happens: 
         client looks for work. OGR is available. does OGR.
         (repeat OGR inthresh times)
         client looks for work, no OGR is available, RC5 is. does RC5.
         (repeat RC5 inthresh times)
         client looks for work, no OGR, no RC5 available. 
                fetches&flushes. 
         client looks for work. OGR is available. does OGR.
   2) User wants to run OGR as long as OGR is available, the do RC5 (until 
      OGR is available again).
      load_order=OGR,RC5
      inthresholds=OGR=<something>,RC5=<something>
      outthresholds=OGR=<not zero and less than inthresh>,RC5=<whatever>
      what happens: 
         client looks for work. OGR is available. does OGR.
         (repeat OGR outhresh times) 
         out threshold now crossed. flushes&fetches.
         client looks for work. OGR is available. does OGR.
   3) User wants to run ONLY contest XXX.
      load_order=XXX,<all others>=0
      if contest XXX is NOT RC5, and no work is available, the client
      will exit. if contest XXX is RC5, and no work is available, it will
      do randoms.
*/

unsigned int ClientGetOutThreshold(Client *client, 
                                   int contestid, int /* force */)
{
  int outthresh = 0;  /* returns zero if outthresholds are not to be checked */
  client = client; /* shaddup compiler. */

  if (contestid < CONTEST_COUNT)
  {
    #if (!defined(NO_OUTBUFFER_THRESHOLDS))
    outthresh = client->outthreshold[contestid]; /* never time driven */
    if (outthresh != 0) /* outthresh=0 => outthresh=inthresh => return 0 */
    {
      unsigned int inthres = ClientGetInThreshold(client, contestid, 0);
      if (inthresh > 0) /* no error */
      {
        if (outthresh <= 0) /* relative to inthresh */
        {
          /*
          a) if the outthreshold (as per .ini) is <=0, then outthreshold is 
          to be interpreted as a value relative to the (computed) inthreshold.
          ie, computed_outthreshold = computed_intthreshold + ini_outthreshold.
          [thus an ini_threshold equal to zero implies rule c)]
          */
          outthresh = inthresh + outthresh;
        }
        if (outthresh >= inthresh)
        {
          /*
          b) if the outthreshold (according to the .ini) was > inthresh
          then inthreshold rules are effective because outthresh can never
          be greater than inthresh (inthresh will have been checked first).
          (The exception is when using shared buffers, and another client 
          fetches but does not flush).
          Consequence: Only inthreshold is effective and outthresh 
          doesn't need to be checked. ClientGetOutThreshold() returns 0.
          c) if the outthreshold (according to the .ini) was equal to inthresh
          there there is usually no point checking outthresh because 
          the result of both checks would be the same. (The exception is 
          when using shared buffers, and another client fetches but does
          not flush).
          Consequence: inthreshold is effective and outthresh 
          doesn't need to be checked. ClientGetOutThreshold() returns 0.
          */
          outthresh = 0;
        }
      }
    }
    #endif
  }
  return ((unsigned int)outthresh);      
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
#elif (CLIENT_OS == OS_NEXTSTEP)
  return "";
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

static void PrintBanner(const char *dnet_id,int level,int restarted)
{
  restarted = 0; /* yes, always show everything */

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
      LogScreenRaw( "RC5 PowerPC and AltiVec assembly by Dan Oetting\n");
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

      LogRaw("\n%s%s%s%s.\n",
             CliGetFullVersionDescriptor(), /* cliident.cpp */
             ((*msg)?(" ("):("")), msg, ((*msg)?(")"):("")) );
      LogScreenRaw( "Please provide the *entire* version descriptor "
                    "when submitting bug reports.\n");
      LogScreenRaw( "The distributed.net bug report pages are at "
                    "http://www.distributed.net/bugs/\n");
      if (dnet_id[0] != '\0' && strcmp(dnet_id,"rc5@distributed.net") !=0 )
        LogRaw( "Using email address (distributed.net ID) \'%s\'\n",dnet_id);
      else if (level == 2)
        LogRaw( "\n* =========================================================================="
                "\n* The client is not configured with your email address (distributed.net ID) "
                "\n* Work done cannot be credited until it is set. Please run '%s -config'"
                "\n* =========================================================================="
                "\n", utilGetAppName());
      LogRaw("\n");

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
  Client *client;
  int retcode = 0;
  int restart = 0;

  TRACE_OUT((+1,"Client.Main()\n"));

  if (InitializeTimers()!=0)
  {
    ConOutErr( "Unable to initialize timers." );
    return -1;
  }

  srand( (unsigned) time(NULL) );
  InitRandom();

  client = (Client *)malloc(sizeof(Client));
  if (!client)
  {
    DeinitializeTimers();
    ConOutErr( "Unable to initialize client. Out of memory." );
    return -1;
  }
  
  do
  {
    int restarted = restart;
    restart = 0;

    ResetClientData(client); /* reset everything in the object */
    ClientEventSyncPost( CLIEVENT_CLIENT_STARTED, *((long*)(&client)) );
    //ReadConfig() and parse command line - returns !0 if shouldn't continue

    TRACE_OUT((0,"Client.parsecmdline restarted?: %d\n", restarted));
    if (ParseCommandline(client,0,argc,(const char **)argv,&retcode,0)==0)
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
          if (InitializeConnectivity() == 0) //do global initialization
          {
            #ifdef LURK /* start must come just after initializeconnectivity */
            TRACE_OUT((0,"dialup.Start()\n")); /*and always before initlogging*/
            dialup.Start(client->offlinemode, &(client->lurk_conf));
            #endif
            TRACE_OUT((0,"initializeconsole\n"));
            if (InitializeConsole(&(client->quietmode),domodes) == 0)
            {
              //some plats need to wait for user input before closing the screen
              int con_waitforuser = 0; //only used if doing modes (and !-config)
  
              TRACE_OUT((+1,"initializelogging\n"));
              InitializeLogging( (client->quietmode!=0),
                                 (client->percentprintingoff!=0),
                                 client->logname,
                                 client->logfiletype,
                                 client->logfilelimit,
                                 ((domodes)?(0):(client->messagelen)),
                                 client->smtpsrvr,
                                 client->smtpport,
                                 client->smtpfrom,
                                 client->smtpdest,
                                 client->id );
              TRACE_OUT((-1,"initializelogging\n"));
              if ((domodes & MODEREQ_CONFIG)==0)
              {
                PrintBanner(client->id,0,restarted);

                TRACE_OUT((+1,"parsecmdline(1)\n"));
                ParseCommandline( client, 1, argc, (const char **)argv, NULL,
                                      (client->quietmode==0)); //show overrides
                TRACE_OUT((-1,"parsecmdline(1)\n"));
              }
              InitRandom2( client->id );
              TRACE_OUT((+1,"initcoretable\n"));
              InitializeCoreTable( &(client->coretypes[0]) );
              TRACE_OUT((-1,"initcoretable\n"));
              ClientSetNumberOfProcessorsInUse(-1); /* reset */

              if (domodes)
              {
                con_waitforuser = ((domodes & ~MODEREQ_CONFIG)!=0);
                if ((domodes & MODEREQ_CONFIG)==0) 
                { /* avoid printing/logging banners for nothing */
                  PrintBanner(client->id,1,restarted);
                }
                TRACE_OUT((+1,"modereqrun\n"));
                ModeReqRun( client );
                TRACE_OUT((-1,"modereqrun\n"));
                restart = 0; /* *never* restart when doing modes */
              }
              else if (!utilCheckIfBetaExpired(1)) /* prints message */
              {
                con_waitforuser = 0;
                PrintBanner(client->id,2,restarted);
                TRACE_OUT((+1,"client.run\n"));
                retcode = ClientRun(client);
                TRACE_OUT((-1,"client.run\n"));
                restart = CheckRestartRequestTrigger();
              }

              ClientSetNumberOfProcessorsInUse(-1); /* reset */
              TRACE_OUT((0,"deinit coretable\n"));
              DeinitializeCoreTable();

              TRACE_OUT((0,"deinitialize logging\n"));
              DeinitializeLogging();
              TRACE_OUT((0,"deinitialize console (waitforuser? %d)\n",con_waitforuser));
              DeinitializeConsole(con_waitforuser);
            } /* if (InitializeConsole() == 0) */
            #ifdef LURK
            TRACE_OUT((0,"dialup.Stop()\n"));
            dialup.Stop(); /* just before DeinitializeConnectivity() */
            #endif
            TRACE_OUT((0,"deinitialize connectivity\n"));
            DeinitializeConnectivity(); //netinit.cpp
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
  macosDeInitialize();
  return 0;
}
#elif (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32)
int PASCAL WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpszCmdLine, int nCmdShow)
{ /* parse the command line and call the bootstrap */
  TRACE_OUT((+1,"WinMain()\n"));
  int rc=winClientPrelude( hInst, hPrevInst, lpszCmdLine, nCmdShow, ClientMain);
  TRACE_OUT((-1,"WinMain()\n"));
  return rc;
}
#elif defined(__unix__) && !defined(__EMX__)
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
    #elif (CLIENT_OS != OS_NEXTSTEP)
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
        #elif (CLIENT_OS != OS_NEXTSTEP)
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

