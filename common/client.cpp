// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: client.cpp,v $
// Revision 1.137  1998/09/23 22:26:42  silby
// Changed checkifbetaexpired from s32 to int
//
// Revision 1.136  1998/09/19 08:50:15  silby
// Added in beta test client timeouts.  Enabled/controlled from version.h by defining BETA, and setting the expiration time.
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
// Kicked version to 2.7100.418-BETA1, ensured that clients report the string ver (which has beta1 in it) in the startup.
//
// Revision 1.126  1998/08/16 06:00:28  silby
// Changed ::Update back so that it checks contest/buffer status before connecting (lurk connecting every few seconds wasn't pretty.)
// Also, changed command line option handling so that update() would be called with force so that it would connect over all.
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
// Lurk functionality is now fully encapsulated inside the Lurk Class, much less code floating inside client.cpp now.
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
// Fixed DONT_USE_PATHWORK handling, ini_etc strings were still being included, now they are not. Also, added the logic for dialwhenneeded, which is a new lurk feature.
//
// Revision 1.112  1998/07/30 02:18:18  blast
// AmigaOS update
//
// Revision 1.111  1998/07/29 05:14:40  silby
// Changes to win32 so that LurkInitiateConnection now works - required the addition of a new .ini key connectionname=.  Username and password are automatically retrieved based on the connectionname.
//
// Revision 1.110  1998/07/26 12:45:52  cyruspatel
// new inifile option: 'autofindkeyserver', ie if keyproxy= points to a
// xx.v27.distributed.net then that will be interpreted by Network::Resolve()
// to mean 'find a keyserver that covers the timezone I am in'. Network
// constructor extended to take this as an argument.
//
// Revision 1.109  1998/07/25 06:31:39  silby
// Added lurk functions to initiate a connection and hangup a connection.  win32 hangup is functional.
//
// Revision 1.108  1998/07/25 05:29:49  silby
// Changed all lurk options to use a LURK define (automatically set in client.h) so that lurk integration of mac/amiga clients needs only touch client.h and two functions in client.cpp
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
// x86 clients in MMX mode will now permit des on > 2 processors.  Bryddes is still set at two, however.
//
// Revision 1.103  1998/07/16 08:25:07  cyruspatel
// Added more NO!NETWORK wrappers around calls to Update/Fetch/Flush. Balanced
// the '{' and '}' in Fetch and Flush. Also, Flush/Fetch will now end with
// 100% unless there was a real send/retrieve fault.
//
// Revision 1.101  1998/07/15 06:58:03  silby
// Changes to Flush, Fetch, and Update so that when the win32 gui sets connectoften to initiate one of the above more verbose feedback will be given.  Also, when force=1, a connect will be made regardless of offlinemode and lurk.
//
// Revision 1.100  1998/07/15 06:10:54  silby
// Fixed an improper #ifdef
//

#if (!defined(lint) && defined(__showids__))
const char *client_cpp(void) {
return "@(#)$Id: client.cpp,v 1.137 1998/09/23 22:26:42 silby Exp $"; }
#endif

// --------------------------------------------------------------------------

#include "cputypes.h"
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "version.h"
#include "problem.h"
#include "network.h"
#include "mail.h"
#include "scram.h"
#include "convdes.h"  // convert_key_from_des_to_inc 
#include "triggers.h" //[Check|Raise][Pause|Exit]RequestTrigger()
#include "sleepdef.h" //sleep(), usleep()
#include "threadcd.h"
#include "buffwork.h"
#include "clitime.h"
#include "clirate.h"
#include "clisrate.h"
#include "clicdata.h"
#include "pathwork.h"
#include "cpucheck.h"  //GetTimesliceBaseline()
#include "logstuff.h"  //Log()/LogScreen()/LogScreenPercent()/LogFlush()

#if ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32))
#include "lurk.h"      //lurk stuff
#endif

#define Time() (CliGetTimeString(NULL,1))

// --------------------------------------------------------------------------
#if ((CLIENT_CPU > 0x01F /* 0-31 */) || ((CLIENT_CONTEST-64) > 0x0F /* 64-79 */) || \
     (CLIENT_BUILD > 0x07 /* 0-7 */) || (CLIENT_BUILD_FRAC > 0x03FF /* 0-1023 */) || \
     (CLIENT_OS  > 0x3F  /* 0-63 */)) // + cputype 0-15
#error CLIENT_CPU/_OS/_CONTEST/_BUILD are out of range for FileEntry check tags
#endif    

#define FILEENTRY_CPU    ((u8)(((cputype & 0x0F)<<4) | (CLIENT_CPU & 0x0F)))
#define FILEENTRY_OS      ((CLIENT_OS & 0x3F) | ((CLIENT_CPU & 0x10) << 3) | \
                           (((CLIENT_BUILD_FRAC>>8)&2)<<5))
#define FILEENTRY_BUILDHI ((((CLIENT_CONTEST-64)&0x0F)<<4) | \
                            ((CLIENT_BUILD & 0x07)<<1) | \
                            ((CLIENT_BUILD_FRAC>>8)&1)) 
#define FILEENTRY_BUILDLO ((CLIENT_BUILD_FRAC) & 0xff)  

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

Problem problem[2*MAXCPUS];

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
  srand( (unsigned) time( NULL ) );
  InitRandom();
#ifdef MMX_BITSLICER
  usemmx = 1;
#endif
}


// --------------------------------------------------------------------------

Client::~Client()
{
  cputype=-1; //dummy to suppress compiler 'Warning:'
}

// --------------------------------------------------------------------------

void Client::RandomWork( FileEntry * data )
{
  u32 randompref2;

  randompref2 = ( ( (u32) randomprefix) + 1 ) & 0xFF;

  data->key.lo = htonl( Random( NULL, 0 ) & 0xF0000000L );
  data->key.hi = htonl( (Random( NULL, 0 ) & 0x00FFFFFFL) + ( randompref2 << 24) ); // 64 bits significant

  data->iv.lo = htonl( 0xD5D5CE79L );
  data->iv.hi = htonl( 0xFCEA7550L );
  data->cypher.lo = htonl( 0x550155BFL );
  data->cypher.hi = htonl( 0x4BF226DCL );
  data->plain.lo = htonl( 0x20656854L );
  data->plain.hi = htonl( 0x6E6B6E75L );
  data->keysdone.lo = htonl( 0 );
  data->keysdone.hi = htonl( 0 );
  data->iterations.lo = htonl( 0x10000000L );
  data->iterations.hi = htonl( 0 );
  data->id[0] = 0;
//82E51B9F:9CC718F9 -- sample problem from RSA pseudo-contest...
//  data->key.lo = htonl(0x9CC718F9L & 0xFF000000L );
//  data->key.hi = htonl(0x82E51B9FL & 0xFFFFFFFFL );
//  data->iv.lo = htonl( 0xF839A5D9L );
//  data->iv.hi = htonl( 0xC41F78C1L );
//  data->cypher.lo = htonl( 0xB74BE041L );
//  data->cypher.hi = htonl( 0x496DEF29L );
//  data->plain.lo = htonl( 0x20656854L );
//  data->plain.hi = htonl( 0x6E6B6E75L );
//  data->iterations.lo = htonl( 0x01000000L );
//END SAMPLE PROBLEM
  data->op = htonl( OP_DATA );
  data->os = 0;
  data->cpu = 0;
  data->buildhi = 0;
  data->buildlo = 0;

  data->contest = 0; // Random blocks are always RC5, not DES.

  data->checksum =
    htonl( Checksum( (u32 *) data, ( sizeof(FileEntry) / 4 ) - 2 ) );
  data->scramble = htonl( Random( NULL, 0 ) );
  Scramble( ntohl(data->scramble), (u32 *) data, ( sizeof(FileEntry) / 4 ) - 1 );

}

// ---------------------------------------------------------------------------

u32 Client::Benchmark( u8 contest, u32 numk )
{
  ContestWork contestwork;

  unsigned int itersize;
  unsigned int keycountshift;
  const char *contestname;
  unsigned int contestid;
  u32 tslice;

  if (numk == 0)
    itersize = 23;         //8388608 instead of 10000000L;
  else if ( numk < (1<<20))   //max(numk,1000000L);
    itersize = 20;         //1048576 instead of 1000000L
  else 
    {  
    itersize = 31;
    while (( numk & (1<<itersize) ) == 0)
      itersize--;
    }

  if (contest == 2 && itersize < 31) //Assumes that DES is (at least)
    itersize++;                      //twice as fast as RC5.

  if (contest == 2)
    {
    keycountshift = 1;
    contestname = "DES";
    contestid = 1;
    }
  else
    {
    keycountshift = 0;
    contestname = "RC5";
    contestid = 0;
    }

  if (SelectCore() || CheckExitRequestTrigger()) 
    return 0;

  tslice = 100000L;

  #if (CLIENT_OS == OS_NETWARE)
    tslice = GetTimesliceBaseline(); //in cpucheck.cpp
  #endif

  LogScreenRaw( "\nBenchmarking %s with 1*2^%d tests (%u keys):\n", 
                 contestname, itersize+keycountshift,
                          (int)(1<<(itersize+keycountshift)) );

  contestwork.key.lo = htonl( 0 );
  contestwork.key.hi = htonl( 0 );
  contestwork.iv.lo = htonl( 0 );
  contestwork.iv.hi = htonl( 0 );
  contestwork.plain.lo = htonl( 0 );
  contestwork.plain.hi = htonl( 0 );
  contestwork.cypher.lo = htonl( 0 );
  contestwork.cypher.hi = htonl( 0 );
  contestwork.keysdone.lo = htonl( 0 );
  contestwork.keysdone.hi = htonl( 0 );
  contestwork.iterations.lo = htonl( (1<<itersize) );
  contestwork.iterations.hi = htonl( 0 );

  (problem[0]).LoadState( &contestwork , (u32) (contestid), tslice, cputype );

  (problem[0]).percent = 0;

  while ( (problem[0]).Run( 0 ) == 0 ) //threadnum
    {
    if (!percentprintingoff)
      LogScreenPercent( 1 ); //logstuff.cpp - number of loaded problems

    #if (CLIENT_OS == OS_NETWARE)   //yield
      nwCliThreadSwitchLowPriority();
    #endif

    if ( CheckExitRequestTrigger() )
      return 0;
    }
  LogScreenPercent( 1 ); //finish the percent bar

  struct timeval tv;
  char ratestr[32];
  double rate = CliGetKeyrateForProblemNoSave( &(problem[0]) );
  tv.tv_sec = (problem[0]).timehi;  //read the time the problem:run started
  tv.tv_usec = (problem[0]).timelo;
  CliTimerDiff( &tv, &tv, NULL );    //get the elapsed time
  LogScreenRaw("\nCompleted in %s [%skeys/sec]\n", CliGetTimeString( &tv, 2 ),
                             CliGetKeyrateAsString( ratestr, rate ) );

  itersize+=keycountshift;
  while ((tv.tv_sec<(60*60) && itersize<31) || (itersize < 28))
    {
    tv.tv_sec<<=1;
    tv.tv_usec<<=1;
    tv.tv_sec+=(tv.tv_usec/1000000L);
    tv.tv_usec%=1000000L;
    itersize++;
    }


  LogScreenRaw(
  "The preferred %s blocksize for this machine should be set to %d (%d*2^28 keys).\n"
  "At the benchmarked keyrate (ie, under ideal conditions) each processor\n"
  "would finish a block of that size in approximately %s.\n", contestname, 
   (unsigned int)itersize, (unsigned int)((((u32)(1<<itersize))/((u32)(1<<28)))),
   CliGetTimeString( &tv, 2 ));  

  #if 0 //for proof-of-concept testing plehzure...
  //what follows is probably true for all processors, but oh well...
  u32 krate = ((contest==2)?(451485):(127254)); //real numbers for a 90Mhz P5
  u32 prate = 90;

  LogScreenRaw( 
  "If this client is running on a cooperative multitasking system, then a good\n"
  "%s timeslice setting may be determined by dividing the benchmarked rate by\n"
  "the processor clock rate in MHz. For example, if the %s keyrate is %d\n"
  "and this is %dMHz machine, then an ideal %s timeslice would be about %u.\n", 
  contestname, contestname, (int)(krate), (int)(prate), contestname, 
                                         (int)(((krate)+(prate>>1))/prate) );
  #endif  
  
  return (u32)(rate);
}

// ---------------------------------------------------------------------------

static int IsFilenameValid( const char *filename )
{ return ( filename && *filename != 0 && strcmp( filename, "none" ) != 0 ); }

static int DoesFileExist( const char *filename )
{
  if ( !IsFilenameValid( filename ) )
    return 0;
  return ( access( GetFullPathForFilename( filename ), 0 ) == 0 );
}

// ---------------------------------------------------------------------------

#if defined(MULTITHREAD)
void Go_mt( void * parm )
{
// Serve both problem[cpunum] and problem[cpunum+numcputemp] until interrupted.
// 2 are used to avoid stalls when network traffic becomes required.
// The main thread of execution will remove finished blocks &
// insert new ones.
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
  char * FAR *argv = (char * FAR *) parm;
  #elif (CLIENT_OS == OS_NETWARE)
  char * *argv = (char * *) parm;
  #else
  char * *argv = (char * *) parm;
  sigset_t signals_to_block;
  #endif
  s32 tempi2;
  s32 numcputemp;
  s32 timeslice;
  u32 run;
  s32 niceness;

  #if (CLIENT_OS == OS_WIN32)
  DWORD LAffinity, LProcessAffinity, LSystemAffinity;
  OSVERSIONINFO osver;
  #endif

  tempi2 = atol(argv[0]);
  numcputemp = atol(argv[1]);
  timeslice = atol(argv[2]);
  niceness = atol(argv[3]);
//LogScreen("tempi2: %d\n",tempi2);
//LogScreen("numcpu: %d\n",numcputemp);
//LogScreen("timeslice: %d\n",timeslice);
//LogScreen("niceness: %d\n",niceness);


#if (CLIENT_OS == OS_WIN32)
  if (niceness == 0)
    SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_IDLE);

  osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
  GetVersionEx(&osver);
  if ((VER_PLATFORM_WIN32_NT == osver.dwPlatformId) && (numcputemp > 1))
  {
    if (GetProcessAffinityMask(GetCurrentProcess(), &LProcessAffinity, &LSystemAffinity))
    {
      LAffinity = 1L << tempi2;
      if (LProcessAffinity & LAffinity)
        SetThreadAffinityMask(GetCurrentThread(), LAffinity);
    }
  }
#elif (CLIENT_OS == OS_NETWARE)
  {
  nwCliInitializeThread( tempi2+1 ); //in netware.cpp
  }
#elif (CLIENT_OS == OS_OS2)
#elif (CLIENT_OS == OS_BEOS)
#else
  sigemptyset(&signals_to_block);
  sigaddset(&signals_to_block, SIGINT);
  sigaddset(&signals_to_block, SIGTERM);
  sigaddset(&signals_to_block, SIGKILL);
  sigaddset(&signals_to_block, SIGHUP);
  pthread_sigmask(SIG_BLOCK, &signals_to_block, NULL);
#endif

  while (!CheckExitRequestTriggerNoIO())
    {
    for (s32 tempi = tempi2; tempi < 2*numcputemp ; tempi += numcputemp)
      {
      run = 0;
      while (!CheckExitRequestTriggerNoIO() && (run == 0))
        {
        if (CheckPauseRequestTriggerNoIO()) 
          {
          run = 0;
          sleep( 1 ); // don't race in this loop
          }
        else
          {
          #if (CLIENT_OS == OS_NETWARE)
              //sets up and uses a polling procedure that runs as
              //an OS callback when the system enters an idle loop.
          run = nwCliRunProblemAsCallback( &(problem[tempi]), tempi2, niceness );
          #else
          // This will return without doing anything if uninitialized...
          run = (problem[tempi]).Run( tempi2 ); //threadnum
          #endif
          } 
        }
      }
    sleep( 1 ); 
    }
  #if (CLIENT_OS == OS_BEOS)
  exit(0);
  #endif
}
#endif

// ---------------------------------------------------------------------------

// returns:
//    -2 = exit by error (all contests closed)
//    -1 = exit by error (critical)
//     0 = exit for unknown reason
//     1 = exit by user request
//     2 = exit by exit file check
//     3 = exit by time limit expiration
//     4 = exit by block count expiration
s32 Client::Run( void )
{
  FileEntry fileentry;
  RC5Result rc5result;

#if defined(MULTITHREAD)
  char buffer[MAXCPUS][4][40];
  #if (CLIENT_OS == OS_BEOS)
    static char * thstart[MAXCPUS][4];
  #else
    char * thstart[MAXCPUS][4];
  #endif
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_NETWARE)
    unsigned long threadid[MAXCPUS];
  #elif (CLIENT_OS == OS_BEOS)
    thread_id threadid[MAXCPUS];
    char thread_name[32];
    char thread_error;
    long be_priority;
    static status_t be_exit_value;
  #elif defined(_POSIX_THREAD_PRIORITY_SCHEDULING)
    pthread_attr_t thread_sched[MAXCPUS];
    pthread_t threadid[MAXCPUS];
  #else
    pthread_t threadid[MAXCPUS];
  #endif
#endif

  s32 count = 0, nextcheckpointtime = 0;
  s32 TimeToQuit = 0, getbuff_errs = 0;

  #if (CLIENT_OS == OS_WIN32) && defined(NEEDVIRTUALMETHODS)
    connectrequested = 0;         // uses public class member
  #else
    u32 connectrequested = 0;
  #endif
  u32 connectloops = 0;

  s32 cpu_i;
  s32 exitchecktime;
  s32 tmpcontest;
  s32 exitcode = 0;

  if ( contestdone[0] && contestdone[1])
    {
    Log( "Both contests are marked as over.  Correct the ini file and restart\n" );
    Log( "This may mean the contests are over.  Check at http://www.distributed.net/rc5/\n" );
    return (-2);
    }

  // --------------------------------------
  // Recover blocks from checkpoint files
  // --------------------------------------

  if ( DoesFileExist( checkpoint_file[0] ) )
    {
    s32 recovered = CkpointToBufferInput(0); // Recover any checkpointed information in case we abnormally quit.
    if (recovered != 0) Log("Recovered %d block%s from RC5 checkpoint file\n",recovered,recovered==1?"":"s");
    }
  if ( DoesFileExist( checkpoint_file[1] ) )
    {
    s32 recovered = CkpointToBufferInput(1); // Recover any checkpointed information in case we abnormally quit.
    if (recovered != 0) Log("Recovered %d block%s from DES checkpoint file\n",recovered,recovered==1?"":"s");
    }

  // --------------------------------------
  // Select an appropriate core, niceness and timeslice setting
  // --------------------------------------

  if (SelectCore())
    return -1;

  #if (CLIENT_CPU == CPU_POWERPC) //this should be in SelectCore()
  switch (whichcrunch)
    {
    case 0:
      Log("Using the 601 core.\n\n");
      break;
    case 1:
      Log("Using the 603/604/750 core.\n\n");
      break;
  }
#endif

  SetNiceness();

  // --------------------------------------
  // Initialize the timers
  // --------------------------------------

  timeStarted = time( NULL );
  exitchecktime = timeStarted + 5;

  // --------------------------------------
  // Determine the number of problems to work with. Number is used everywhere.
  // --------------------------------------

  int load_problem_count = 1;
  #ifdef MULTITHREAD
    if (numcputemp == 0) //multithread compile but user requests non-mt
      numcputemp = 1;
    #if (CLIENT_OS == OS_NETWARE)
    else if (numcputemp == 1) //NetWare client prefers non-MT if only one 
      load_problem_count = 1; //thread/processor is to used
    #endif
    else
      load_problem_count = 2*numcputemp;
  #endif

  // --------------------------------------
  // Set up initial state of each problem[]...
  // uses 2 active buffers per CPU to avoid stalls
  // --------------------------------------

  for (cpu_i = 0; cpu_i < load_problem_count; cpu_i++ )
  {
#if ((CLIENT_CPU == CPU_X86) || (CLIENT_OS == OS_BEOS))
    if (((cpu_i%numcputemp)>=2)
#if ((CLIENT_CPU == CPU_X86) && defined(MMX_BITSLICER) && defined(KWAN) && defined(MEGGS))
      && (des_unit_func!=des_unit_func_mmx) // we're not using the mmx cores
#endif
      )
    {
      // Not the 1st or 2nd cracking thread...
      // Must do RC5.  DES x86 cores aren't multithread safe.
      // Note that if rc5 contest is over, this will return -2...
      count = GetBufferInput( &fileentry , 0);
      if (contestdone[0])
        count = -2; //means that this thread won't actually start
    }
    else
#endif
    {
      if (getbuff_errs == 0)
      {
        if (!contestdone[ (int) preferred_contest_id ])
        {
          // Neither contest is done...
          count = GetBufferInput( &fileentry , (u8) preferred_contest_id);
          if (contestdone[ (int) preferred_contest_id ]) // This contest just finished.
          {
            goto PreferredIsDone1;
          }
          else
          {
            if (count == -3)
            {
              // No DES blocks available while in offline mode.  Do rc5...
              count = GetBufferInput( &fileentry , (u8) ( ! preferred_contest_id));
            }
          }
        }
        else
        {
          // Preferred contest is done...
PreferredIsDone1:
          count = GetBufferInput( &fileentry , (u8) ( ! preferred_contest_id));
          if (contestdone[ ! preferred_contest_id ])
          {
            // This contest just finished.
            count = -2; // Both contests finished!
          }
        }
      }
    }

    if (count == -1)
    {
      getbuff_errs++;
    }
    else if ((!nonewblocks) && (count != -2))
    {
      // LoadWork expects things descrambled.
      Descramble( ntohl( fileentry.scramble ),
                     (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );
      // If a block was finished with an 'odd' number of keys done, then make it redo the last
      // key -- this will prevent a 2-pipelined core from looping forever.
      if ((ntohl(fileentry.iterations.lo) & 0x00000001L) == 1)
      {
        fileentry.iterations.lo = htonl((ntohl(fileentry.iterations.lo) & 0xFFFFFFFEL) + 1);
        fileentry.key.lo = htonl(ntohl(fileentry.key.lo) & 0xFEFFFFFFL);
      }
      if (fileentry.contest != 1)
        fileentry.contest=0;

      // If this is a partial block, and completed by a different cpu/os/build, then
      // reset the keysdone to 0...
      {
        if ( (ntohl(fileentry.keysdone.lo)!=0) || (ntohl(fileentry.keysdone.hi)!=0) )
        {
         if ((fileentry.cpu     != FILEENTRY_CPU) ||
             (fileentry.os      != FILEENTRY_OS) ||
             (fileentry.buildhi != FILEENTRY_BUILDHI) || 
             (fileentry.buildlo != FILEENTRY_BUILDLO))
          {
            fileentry.keysdone.lo = fileentry.keysdone.hi = htonl(0);
            LogScreen("[%s] Read partial block from another cpu/os/build.\n",Time());
            LogScreen("[%s] Marking entire block as unchecked.\n",Time());
          }
        }
      }

      {
        if (cpu_i==0 && load_problem_count>1)
          Log( "[%s] %s\n", CliGetTimeString(NULL,1),
                                  "Loading two blocks per thread...");

        Log( "[%s] %s\n", CliGetTimeString(NULL,1),
                       CliGetMessageForFileentryLoaded( &fileentry ) );

        //only display the "remaining blocks in file" once
        static char have_loaded_buffers[2]={0,0};
        have_loaded_buffers[fileentry.contest]=1;

        if (cpu_i == (load_problem_count-1)) //last loop?
        {
          if (load_problem_count == 2)
            Log("[%s] 1 Child thread has been started.\n", Time());
          else if (load_problem_count > 2)
            Log("[%s] %d Child threads ('A'%s'%c') have been started.\n",
              Time(), load_problem_count>>1,
              ((load_problem_count>4)?("-"):(" and ")),
              'A'+((load_problem_count>>1)-1));

          for (s32 tmpc = 0; tmpc < 2; tmpc++) //once for each contest
          {
            if (have_loaded_buffers[(int) tmpc]) //load any of this type?
            {
              int in = (int) CountBufferInput((u8) tmpc);
              int out = (int) CountBufferOutput((u8) tmpc);
              Log( "[%s] %d %s block%s remain%s in file %s\n", CliGetTimeString(NULL,1),
                in,
                CliGetContestNameFromID((int) tmpc),
                in == 1 ? "" : "s",
                in == 1 ? "s" : "",
                (nodiskbuffers ? "(memory-in)" :
#ifdef DONT_USE_PATHWORK
                ini_in_buffer_file[(int) tmpc]));
#else
                in_buffer_file[(int) tmpc]));
#endif
              Log( "[%s] %d %s block%s %s in file %s\n", CliGetTimeString(NULL,1),
                out,
                CliGetContestNameFromID((int) tmpc),
                out == 1 ? "" : "s",
                out == 1 ? "is" : "are",
                (nodiskbuffers ? "(memory-out)" :
#ifdef DONT_USE_PATHWORK
                ini_out_buffer_file[(int) tmpc]));
#else
                out_buffer_file[(int) tmpc]));
#endif
            }
          }
        }
      }

      (problem[(int) cpu_i]).LoadState( (ContestWork *) &fileentry , 
               (u32) (fileentry.contest), timeslice, cputype );

      //----------------------------
      //spin off a thread for this problem
      //----------------------------

#if defined(MULTITHREAD)  //this is the last time we use the MULTITHREAD define. 
  #undef MULTITHREAD //protect against abuse lower down. A client can be mt capable
  //but be running single threaded, so we need to check (load_problem_count > 1)
  //and not whether its mt capable or not.
      {
        //Only launch a thread if we have really loaded 2*threadcount buffers
        if ((load_problem_count > 1) && (cpu_i < numcputemp))
        {
          // Start the thread for this cpu
          sprintf(buffer[cpu_i][0],"%d",(int)cpu_i);
          sprintf(buffer[cpu_i][1],"%d",(int)numcputemp);
          sprintf(buffer[cpu_i][2],"%d",(int)timeslice);
          sprintf(buffer[cpu_i][3],"%d",(int)niceness);
          thstart[cpu_i][0] = &buffer[cpu_i][0][0];
          thstart[cpu_i][1] = &buffer[cpu_i][1][0];
          thstart[cpu_i][2] = &buffer[cpu_i][2][0];
          thstart[cpu_i][3] = &buffer[cpu_i][3][0];
#if (CLIENT_OS == OS_WIN32)
          threadid[cpu_i] = _beginthread( Go_mt, 8192, thstart[cpu_i]);
          //if ( threadid[cpu_i] == 0)
          //  threadid[cpu_i] = NULL; //0
#elif (CLIENT_OS == OS_OS2)
          threadid[cpu_i] = _beginthread( Go_mt, NULL, 8192, thstart[cpu_i]);
          if ( threadid[cpu_i] == -1)
            threadid[cpu_i] = NULL; //0
#elif (CLIENT_OS == OS_NETWARE)
          threadid[cpu_i] = BeginThread( Go_mt, NULL, 8192, thstart[cpu_i]);
          if ( threadid[cpu_i] == -1)
            threadid[cpu_i] = NULL; //0
#elif (CLIENT_OS == OS_BEOS)
          switch(niceness)
          {
            case 0: be_priority = B_LOW_PRIORITY; break;
            case 1: be_priority = (B_LOW_PRIORITY + B_NORMAL_PRIORITY) / 2; break;
            case 2: be_priority = B_NORMAL_PRIORITY; break;
            default: be_priority = B_LOW_PRIORITY; break;
          }
          sprintf(thread_name, "RC5DES crunch#%d", cpu_i + 1);
          threadid[cpu_i] = spawn_thread((long (*)(void *)) Go_mt, thread_name,
                be_priority, (void *)thstart[cpu_i]);
          thread_error = (threadid[cpu_i] < B_NO_ERROR);
          if (!thread_error)
            thread_error =  (resume_thread(threadid[cpu_i]) != B_NO_ERROR);
          if (thread_error)
            threadid[cpu_i] = NULL; //0
#elif defined(_POSIX_THREAD_PRIORITY_SCHEDULING)
          pthread_attr_init(&thread_sched[cpu_i]);
          pthread_attr_setscope(&thread_sched[cpu_i],PTHREAD_SCOPE_SYSTEM);
          pthread_attr_setinheritsched(&thread_sched[cpu_i],PTHREAD_INHERIT_SCHED);
          if (pthread_create( &threadid[cpu_i], &thread_sched[cpu_i], (void *(*)(void*)) Go_mt, thstart[cpu_i]) )
            threadid[cpu_i] = (pthread_t) NULL; //0
#else
          #define USING_POSIX_THREADS //so we can stop later without using MULTITHREAD
          if (pthread_create( &threadid[cpu_i], NULL, (void *(*)(void*)) Go_mt, thstart[cpu_i]) )
            threadid[cpu_i] = (pthread_t) NULL; //0
#endif

          if ( !threadid[cpu_i] )
          {
            Log("[%s] Could not start child thread '%c'.\n",Time(),cpu_i+'A');

            numcputemp = cpu_i+1;            //# of threads already loaded

            if ( cpu_i == 0 ) //was it the first thread that failed?
            {
              load_problem_count = 1; //then switch to non-threaded mode
              Log("[%s] Switching to single-threaded mode.\n", Time());
              break;
            }
            else
            {
              load_problem_count = numcputemp * 2; //resize ourselves
              
              fileentry.contest = (u8) (problem[(int)cpu_i]).RetrieveState( (ContestWork *) &fileentry , 1 );
              fileentry.op = htonl( OP_DATA );

              fileentry.cpu     = FILEENTRY_CPU;
              fileentry.os      = FILEENTRY_OS;
              fileentry.buildhi = FILEENTRY_BUILDHI; 
              fileentry.buildlo = FILEENTRY_BUILDLO;

              fileentry.checksum =
                  htonl( Checksum( (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 2 ) );
              Scramble( ntohl( fileentry.scramble ),
                         (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );
              PutBufferInput( &fileentry );  // send it back...
            }
          }
        }
      }
#endif
    } //if ((!nonewblocks) && (count != -2))
  } //for (cpu_i = 0; cpu_i < load_problem_count; cpu_i ++)


  //------------------------------------
  // display the percent bar so the user sees some action
  //------------------------------------

  if (!percentprintingoff)
    LogScreenPercent( load_problem_count ); //logstuff.cpp

  //============================= MAIN LOOP =====================
  //now begin looping until we have a reason to quit
  //------------------------------------

  // -- cramer - until we have a better way of telling how many blocks
  //             are loaded and if we can get more, this is gonna be a
  //             a little complicated.  getbuff_errs and nonewblocks
  //             control the exit process.  getbuff_errs indicates the
  //             number of attempts to load new blocks that failed.
  //             nonewblocks indcates that we aren't get anymore blocks.
  //             Together, they can signal when the buffers have been
  //             truely exhausted.  The magic below is there to let
  //             the client finish processing those blocks before exiting.

  // Start of MAIN LOOP
  while (TimeToQuit == 0)
  {
    //------------------------------------
    //Do keyboard stuff for clients that allow user interaction during the run
    //------------------------------------

    #if ((CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)) && !defined(NEEDVIRTUALMETHODS)
    {
      while ( kbhit() )
      {
        int hitchar = getch();
          if (hitchar == 0) //extended keystroke
            getch();
          else
          {
            if (hitchar == 3 || hitchar == 'X' || hitchar == 'x' || hitchar == '!')
            {
              // exit after current blocks
              if (blockcount > 0)
              {
                blockcount = min(blockcount, (s32) (totalBlocksDone[0] + totalBlocksDone[1] + numcputemp));
              } else {
                blockcount = (s32) (totalBlocksDone[0] + totalBlocksDone[1] + numcputemp);
              }
              Log("Exiting after current block\n");
              exitcode = 1;
            }
            if ((load_problem_count > 1) && (hitchar == 'u' || hitchar == 'U'))
            {
              Log("Keyblock Update forced\n");
              connectrequested = 1;
            }
          }
        }
    }
    #endif

    //------------------------------------
    //special update request (by keyboard or by lurking) handling
    //------------------------------------

    if (load_problem_count > 1)  //ie multi-threaded
      {
      if ((connectoften && ((connectloops++)==19)) || (connectrequested > 0) )
        {
        // Connect every 20*3=60 seconds
        // Non-MT 60 + (time for a client.run())
        connectloops=0;
        if (connectrequested == 1) // forced update by a user
          {
          Update(0 ,1,1,1);  // RC5 We care about the errors, force update.
          Update(1 ,1,1,1);  // DES We care about the errors, force update.
          LogScreen("Keyblock Update completed.\n");
          connectrequested=0;
          }
        else if (connectrequested == 2) // automatic update
          {
          Update(0 ,0,0);  // RC5 We don't care about any of the errors.
          Update(1 ,0,0);  // DES 
          connectrequested=0;
          }
        else if (connectrequested == 3) // forced flush
          {
          Flush(0,NULL,1,1); // Show errors, force flush
          Flush(1,NULL,1,1);
          LogScreen("Flush request completed.\n");
          connectrequested=0;
          }
        else if (connectrequested == 4) // forced fetch
          {
          Fetch(0,NULL,1,1); // Show errors, force fetch
          Fetch(1,NULL,1,1);
          LogScreen("Fetch request completed.\n");
          connectrequested=0;
          };
        }
      }

    //------------------------------------
    // Lurking
    //------------------------------------

#if defined(LURK)
if(dialup.lurkmode) // check to make sure lurk mode is enabled
  connectrequested=dialup.CheckIfConnectRequested();
#endif

    //------------------------------------
    //sleep, run or pause...
    //------------------------------------

    if (load_problem_count > 1) //ie multi-threaded
      {
      // prevent the main thread from racing & bogging everything down.
      sleep(3);
      }
    else if (CheckPauseRequestTrigger()) //threads have their own sleep section
      {
      #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
        SurrenderCPU();
      #elif (CLIENT_OS != OS_DOS)
        sleep(1);
      #endif
      }
    else //only one problem and we are not paused
      {
      //Actually run a problem
      #if (CLIENT_OS == OS_NETWARE)
        {
        //sets up and uses a polling procedure that runs as
        //an OS callback when the system enters an idle loop.
        nwCliRunProblemAsCallback( &(problem[0]), 0 , niceness );
        }
      #else
        {
        (problem[0]).Run( 0 ); //threadnum
        #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
        SurrenderCPU();
        #endif
        }
      #endif //if non-mt, netware or not
      }
   

    //------------------------------------
    //update the status bar
    //------------------------------------

    if (!percentprintingoff)
      LogScreenPercent( load_problem_count ); //logstuff.cpp


    //------------------------------------
    //now check all problems for change, do checkpointing, reloading etc
    //------------------------------------

    for (cpu_i = 0; ((!CheckPauseRequestTrigger()) && (!CheckExitRequestTrigger()) 
                    && (cpu_i < load_problem_count)); cpu_i++)
    {

      // -------------
      // check for finished blocks that need reloading
      // -------------

      // Did any threads finish a block???
      if ((problem[(int) cpu_i]).finished == 1)
      {
      #if defined(BETA)
      if (checkifbetaexpired() > 0) RaiseExitRequestTrigger();
      #endif
        (problem[(int) cpu_i]).GetResult( &rc5result );

        //-----------------
        //only do something if RESULT_FOUND or RESULT_NOTHING
        //Q: when can it be finished AND result_working?
        //-----------------

        if ((rc5result.result == RESULT_FOUND) || (rc5result.result == RESULT_NOTHING))
        {
          //---------------------
          //print the keyrate and update the totals for this contest
          //---------------------

          {
            Log( "\n[%s] %s", CliGetTimeString(NULL,1), /* == Time() */
                   CliGetMessageForProblemCompleted( &(problem[(int) cpu_i]) ) );
          }

          //----------------------------------------
          // Figure out which contest block was from, and increment totals
          //----------------------------------------

          tmpcontest = (u8) (problem[(int) cpu_i]).RetrieveState( (ContestWork *) &fileentry , 1 );

          totalBlocksDone[(int) tmpcontest]++;

          //----------------------------------------
          // Print contest totals
          //----------------------------------------

          // Detect/report any changes to the total completed blocks...


          {
          //display summaries only of contests w/ more than one block done
          int i = 1;
          for (s32 tmpc = 0; tmpc < 2; tmpc++)
            {
              if (totalBlocksDone[(int) tmpc] > 0)
                {
                  Log( "%c%s%c Summary: %s\n",
                       ((i == 1) ? ('[') : (' ')), CliGetTimeString(NULL, i),
                       ((i == 1) ? (']') : (' ')), CliGetSummaryStringForContest((int) tmpc) );
                  if ((--i) < 0) i = 0;
                }
            }
          }

          //---------------------
          //put the completed problem away
          //---------------------

          tmpcontest = fileentry.contest = (u8) (problem[(int) cpu_i]).RetrieveState( (ContestWork *) &fileentry , 1 );

          // make it into a reply
          if (rc5result.result == RESULT_FOUND)
          {
            consecutivesolutions[fileentry.contest]++;
            if (keyport == 3064)
                LogScreen("Success\n");
            fileentry.op = htonl( OP_SUCCESS_MULTI );
            fileentry.key.lo = htonl( ntohl( fileentry.key.lo ) +
                                ntohl( fileentry.keysdone.lo ) );
          }
          else
          {
            if (keyport == 3064)
              LogScreen("Success was not detected!\n");
            fileentry.op = htonl( OP_DONE_MULTI );
          }

          fileentry.os = CLIENT_OS;
          fileentry.cpu = CLIENT_CPU;
          fileentry.buildhi = CLIENT_CONTEST;
          fileentry.buildlo = CLIENT_BUILD;
          strncpy( fileentry.id, id , sizeof(fileentry.id)-1); // set id for this block
          fileentry.id[sizeof(fileentry.id)-1]=0;  // in case id>58 bytes, truncate.

          fileentry.checksum =
              htonl( Checksum( (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 2 ) );
          Scramble( ntohl( fileentry.scramble ),
                     (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );

          // send it back...
          if ( PutBufferOutput( &fileentry ) == -1 )
            {
            Log( "PutBuffer Error\n" );

            // Block didn't get put into a buffer, subtract it from the count.
            totalBlocksDone[(int)tmpcontest]--;
            };

          //---------------------
          //delete the checkpoint file, info is outdated
          //---------------------

          // Checkpoint info just became outdated...

          if ( DoesFileExist( checkpoint_file[0] ) )
            EraseCheckpointFile(checkpoint_file[0]); //buffwork.cpp
          if ( DoesFileExist( checkpoint_file[1] ) )
            EraseCheckpointFile(checkpoint_file[1]); //buffwork.cpp

          //---------------------
          // See if the request to quit after the completed block
          //---------------------
          if(exitcode == 1) TimeToQuit=1; // Time to quit

          //---------------------
          //now load another block for this contest
          //---------------------

          // Get another block...

#if ((CLIENT_CPU == CPU_X86) || (CLIENT_OS == OS_BEOS))
          if (((cpu_i%numcputemp)>=2)
#if ((CLIENT_CPU == CPU_X86) && defined(MMX_BITSLICER) && defined(KWAN) && defined(MEGGS))
       && (des_unit_func!=des_unit_func_mmx) // we're not using the mmx cores
#endif
          )
          {
              // Not the 1st or 2nd cracking thread...
              // Must do RC5.  DES x86 cores aren't multithread safe.
              // Note that if rc5 contest is over, this will return -2...
              count = GetBufferInput( &fileentry , 0);
            if (contestdone[0])
              count = -2;
          }
          else
#endif
          {
            if (getbuff_errs == 0)
            {
              if (!contestdone[ (int) preferred_contest_id ])
              {
                // Neither contest is done...
                count = GetBufferInput( &fileentry , (u8) preferred_contest_id);
                if (contestdone[ (int) preferred_contest_id ]) // This contest just finished.
                {
                  goto PreferredIsDone2;
                }
                else
                {
                  if (count == -3)
                  {
                    // No DES blocks available while in offline mode.  Do rc5...
                    count = GetBufferInput( &fileentry , (u8) ( ! preferred_contest_id));
                  }
                }
              }
              else
              {
                // Preferred contest is done...
      PreferredIsDone2:
                count = GetBufferInput( &fileentry , (u8) ( ! preferred_contest_id));
                if (contestdone[ ! preferred_contest_id ])
                {
                  // This contest just finished.
                  count = -2; // Both contests finished!
                }
              }
            }
            else if (nonewblocks) getbuff_errs++; // cramer magic #1 (mt)
          }

          if (count < 0)
          {
            getbuff_errs++; // cramer magic #2 (non-mt)
            if (!nonewblocks)
            {
              TimeToQuit=1; // Force blocks to be saved
              exitcode = -2;
              continue;  //break out of the next cpu_i loop
            }
          }

          //---------------------
          // correct any potential problems in the freshly loaded fileentry
          //---------------------

          if (!nonewblocks)
          {
            // LoadWork expects things descrambled.
            Descramble( ntohl( fileentry.scramble ),
                       (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );
            // If a block was finished with an 'odd' number of keys done, then make it redo the last
            // key -- this will prevent a 2-pipelined core from looping forever.
            if ((ntohl(fileentry.iterations.lo) & 0x00000001L) == 1)
            {
              fileentry.iterations.lo = htonl((ntohl(fileentry.iterations.lo) & 0xFFFFFFFEL) + 1);
              fileentry.key.lo = htonl(ntohl(fileentry.key.lo) & 0xFEFFFFFFL);
            }
            if (fileentry.contest != 1)
              fileentry.contest=0;

            // If this is a partial block, and completed by a different
            // cpu/os/build, then reset the keysdone to 0...
            {
              if ( (ntohl(fileentry.keysdone.lo)!=0) || (ntohl(fileentry.keysdone.hi)!=0) )
              {
                if ((fileentry.cpu     != FILEENTRY_CPU) ||
                    (fileentry.os      != FILEENTRY_OS) ||
                    (fileentry.buildhi != FILEENTRY_BUILDHI) || 
                    (fileentry.buildlo != FILEENTRY_BUILDLO))
                {
                  fileentry.keysdone.lo = fileentry.keysdone.hi = htonl(0);
                  LogScreen("[%s] Read partial block from another cpu/os/build.\n",Time());
                  LogScreen("[%s] Marking entire block as unchecked.\n",Time());
                }
              }
            }
          }

          //---------------------
          // display the status of the file buffers
          //---------------------

          if (!nonewblocks)
          {
            int outcount = (int) CountBufferOutput((u8) fileentry.contest);
            Log( "[%s] %s\n", CliGetTimeString(NULL,1), /* == Time() */
                              CliGetMessageForFileentryLoaded( &fileentry ) );
            Log( "[%s] %d %s block%s remain%s in file %s\n"
                 "[%s] %d %s block%s %s in file %s\n",
                 CliGetTimeString(NULL,1), count, CliGetContestNameFromID(fileentry.contest),
                 count == 1 ? "" : "s", count == 1 ? "s" : "",
                 (nodiskbuffers ? "(memory-in)" :
#ifdef DONT_USE_PATHWORK
                 ini_in_buffer_file[(int)fileentry.contest]),
#else
                 in_buffer_file[(int)fileentry.contest]),
#endif
                 CliGetTimeString(NULL,1), outcount, CliGetContestNameFromID(fileentry.contest),
                 outcount == 1 ? "" : "s", outcount == 1 ? "is" : "are",
                 (nodiskbuffers ? "(memory-out)" :
#ifdef DONT_USE_PATHWORK
                 ini_out_buffer_file[(int)fileentry.contest]) );
#else
                 out_buffer_file[(int)fileentry.contest]) );
#endif
          }

          //---------------------
          // now load the problem with the fileentry
          //---------------------
          if (!nonewblocks)
            (problem[(int)cpu_i]).LoadState( (ContestWork *) &fileentry , 
               (u32) (fileentry.contest), timeslice, cputype );

        } // end (if 'found' or 'nothing')

        DoCheckpoint( load_problem_count );
      } // end(if finished)
    } // endfor(cpu_i)

    //----------------------------------------
    // Check for time limit...
    //----------------------------------------

    if ( ( minutes > 0 ) &&
           (s32) ( time( NULL ) > (s32) ( timeStarted + ( 60 * minutes ) ) ) )
    {
      Log( "\n[%s] Shutdown - %u.%02u hours expired\n", Time(), minutes/60, (minutes%60) );
      TimeToQuit = 1;
      exitcode = 3;
    }

    //----------------------------------------
    // Check for user break
    //----------------------------------------

    if ( CheckExitRequestTrigger() )
    {
      Log( "\n[%s] Shutdown message received - Block being saved.\n", Time() );
      TimeToQuit = 1;
      exitcode = 1;
    }

    //----------------------------------------
    // Check for 32 consecutive solutions
    //----------------------------------------

    for (int tmpc = 0; tmpc < 2; tmpc++)
    {
      const char *contname = CliGetContestNameFromID( tmpc ); //clicdata.cpp
      if ((consecutivesolutions[tmpc] >= 32) && !contestdone[tmpc])
      {
        Log( "\n[%s] Too many consecutive %s solutions detected.\n", Time(), contname );
        Log( "[%s] Either the contest is over, or this client is pointed at a test port.\n", Time() );
        Log( "[%s] Marking %s contest as over\n", Time(), contname );
        Log( "[%s] Further %s blocks will not be processed.\n", Time(), contname );
        contestdone[tmpc] = 1;
        WriteContestandPrefixConfig( );
      }
    }
    if (contestdone[0] && contestdone[1])
    {
      TimeToQuit = 1;
      Log( "\n[%s] Both RC5 and DES are marked as finished.\n", Time() );
      exitcode = -2;
    }

    //----------------------------------------
    // Has -runbuffers exhausted all buffers?
    //----------------------------------------

    // cramer magic (voodoo)
    if (nonewblocks > 0 && (getbuff_errs >= load_problem_count))
    {
      TimeToQuit = 1;
      exitcode = 4;
    }

    //----------------------------------------
    // Reached the -b limit?
    //----------------------------------------

    // Done enough blocks?
    if ( ( blockcount > 0 ) && ( totalBlocksDone[0]+totalBlocksDone[1] >= (u32) blockcount ) )
      {
      Log( "[%s] Shutdown - %d blocks completed\n", Time(), (u32) totalBlocksDone[0]+totalBlocksDone[1] );
      TimeToQuit = 1;
      exitcode = 4;
      }

    if (!TimeToQuit && CheckExitRequestTrigger())
      {
      TimeToQuit = 1;
      exitcode = 2;
      }

    //----------------------------------------
    // Are we quitting?
    //----------------------------------------

    if ( TimeToQuit )
    {
      // ----------------
      // Shutting down: shut down threads
      // ----------------

      RaiseExitRequestTrigger(); // will make other threads exit

      if (load_problem_count > 1)  //we have threads running
        {
        LogScreen("[%s] Shutting threads down...\n", Time());
        // Wait for all threads to end...
        for (cpu_i = 0; cpu_i < numcputemp; cpu_i++)
          {
#if (CLIENT_OS == OS_OS2)
          DosWaitThread(&threadid[cpu_i], DCWW_WAIT);
#elif (CLIENT_OS == OS_WIN32)
          WaitForSingleObject((HANDLE)threadid[cpu_i], INFINITE);
#elif (CLIENT_OS == OS_BEOS)
          wait_for_thread(threadid[cpu_i], &be_exit_value);
#elif (CLIENT_OS == OS_NETWARE)
          nwCliWaitForThreadExit( threadid[cpu_i] ); //in netware.cpp
#elif defined(_POSIX_THREAD_PRIORITY_SCHEDULING) || defined(USING_POSIX_THREADS)
          pthread_join(threadid[cpu_i], NULL);
#endif
          }
        }

      // ----------------
      // Shutting down: save problem buffers
      // ----------------

      for (cpu_i = (load_problem_count - 1); cpu_i >= 0; cpu_i-- )
      {
        if ((problem[(int)cpu_i]).IsInitialized())
        {
          fileentry.contest = (u8) (problem[(int)cpu_i]).RetrieveState( (ContestWork *) &fileentry , 1 );
          fileentry.op = htonl( OP_DATA );

          fileentry.cpu     = FILEENTRY_CPU;
          fileentry.os      = FILEENTRY_OS;
          fileentry.buildhi = FILEENTRY_BUILDHI; 
          fileentry.buildlo = FILEENTRY_BUILDLO;

          fileentry.checksum =
                 htonl( Checksum( (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 2 ) );
          u32 temphi = ntohl( fileentry.key.hi );
          u32 templo = ntohl( fileentry.key.lo );
          u32 percent2 = (u32) ( (double) 10000.0 *
                           ( (double) ntohl(fileentry.keysdone.lo) /
                              (double) ntohl(fileentry.iterations.lo) ) );
          Scramble( ntohl( fileentry.scramble ),
                       (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );

          // send it back...
          if ( PutBufferInput( &fileentry ) == -1 )
          {
            Log( "Buffer Error\n" );
          }
          else
          {
            Log( "[%s] Saved block %08lX:%08lX (%d.%02d percent complete)\n",
                Time(), (unsigned long) temphi, (unsigned long) templo,
                percent2/100, percent2%100 );
          }
        }
      } //endfor(cpu_i)

      // ----------------
      // Shutting down: delete checkpoint files
      // ----------------

      if ( DoesFileExist( checkpoint_file[0] ) )
        EraseCheckpointFile(checkpoint_file[0]);
      if ( DoesFileExist( checkpoint_file[1] ) )
        EraseCheckpointFile(checkpoint_file[1]);

      // ----------------
      // Shutting down: do a net flush if we don't have diskbuffers
      // ----------------

      // no disk buffers -- we had better flush everything.
      if (nodiskbuffers)
      {
        ForceFlush((u8) preferred_contest_id ) ;
        ForceFlush((u8) ! preferred_contest_id );
      }

    } // TimeToQuit

    //----------------------------------------
    // If not quitting, then write checkpoints
    //----------------------------------------

    if (!TimeToQuit)
    {
      // Time to checkpoint?
      if ((IsFilenameValid( checkpoint_file[0] ) ||
           IsFilenameValid( checkpoint_file[1] ))
           && (!nodiskbuffers) && (!CheckPauseRequestTrigger()))
        {
       if ( (!TimeToQuit ) && ( ( (s32) time( NULL ) ) > ( (s32) nextcheckpointtime ) ) )

        {
          nextcheckpointtime = time(NULL) + checkpoint_min * 60;
          //Checkpoints may be slightly late (a few seconds). However,
          //this eliminates checkpoint catchup due to pausefiles/clock
          //changes/other nasty things that change the clock
          DoCheckpoint(load_problem_count);
        }
      } // Checkpointing
    }

  }  // End of MAIN LOOP

  //======================END OF MAIN LOOP =====================

  if (randomchanged)  
    WriteContestandPrefixConfig();

  #if (CLIENT_OS == OS_VMS)
    nice(0);
  #endif
  return exitcode;
}

// ---------------------------------------------------------------------------

void Client::DoCheckpoint( int load_problem_count )
{
  FileEntry fileentry;

  for (int j = 0; j < 2; j++)
  {
    if ( IsFilenameValid(checkpoint_file[j] ) )
    {
      EraseCheckpointFile(checkpoint_file[j]); // Remove prior checkpoint information (if any).

      for (int cpu_i = 0 ; cpu_i < (int) load_problem_count ; cpu_i++)
      {
        fileentry.contest = (u8) (problem[cpu_i]).RetrieveState( (ContestWork *) &fileentry , 0 );
        if (fileentry.contest == j)
        {
          fileentry.op = htonl( OP_DATA );
          fileentry.cpu     = FILEENTRY_CPU;
          fileentry.os      = FILEENTRY_OS;
          fileentry.buildhi = FILEENTRY_BUILDHI; 
          fileentry.buildlo = FILEENTRY_BUILDLO;
          fileentry.checksum=
              htonl( Checksum( (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 2 ) );
          Scramble( ntohl( fileentry.scramble ),
                      (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );

          // send it back...
          if (InternalPutBuffer( this->checkpoint_file[j], &fileentry ) == -1)
            Log( "Checkpoint Buffer Error\n" );
        }
      } //endfor(cpu_i)
    }
  }
}

// ---------------------------------------------------------------------------

s32 Client::SetContestDoneState( Packet * packet)
{
  u32 detect;

  // Set the contestdone state, if possible...
  // Move contestdone[] from 0->1, or 1->0.
  detect = 0;
  if (packet->descontestdone == ntohl(0xBEEFF00DL)) {
    if (contestdone[1]==0) {detect = 2; contestdone[1] = 1;}
  } else {
    if (contestdone[1]==1) {detect = 2; contestdone[1] = 0;}
  }
  if (detect == 2) {
    Log( "Received notification: %s contest %s.\n",
         (detect == 2 ? "DES" : "RC5"),
         (contestdone[(int)detect-1]?"is not currently active":"has started") );
  }

  if (packet->rc564contestdone == ntohl(0xBEEFF00DL)) {
    if (contestdone[0] == 0) {detect = 1; contestdone[0] = 1;}
  } else {
    if (contestdone[0] == 1) {detect = 1; contestdone[0] = 0;}
  }
  if (detect == 1) {
    Log( "Received notification: %s CONTEST %s\n",
        (detect == 2 ? "DES" : "RC5"),
        (contestdone[(int)detect-1]?"IS OVER":"HAS STARTED") );
  }

  if (detect != 0) {
    WriteContestandPrefixConfig();
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------

#if !defined(NOMAIN)
int main( int argc, char *argv[] )
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
    retcode = clientP->Main( argc, (const char **)argv );
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

int Client::Main( int argc, const char *argv[] )
{
  int retcode = 0;

  // set up break handlers
  if (InitializeTriggers(NULL, NULL)==0) //CliSetupSignals();
    {
    //get inifilename and get -quiet/-hidden overrides
    if (ParseCommandline( 0, argc, argv, NULL, &retcode, 0 ) == 0) //change defaults
      {
      int inimissing = (ReadConfig() != 0); //reads using defaults
      InitializeLogging(); //let -quiet take affect
      PrintBanner(0);
      if ( ParseCommandline( 2, argc, argv, &inimissing, &retcode, 1 )==0 )
        {
        if (inimissing)
          {
          if (Configure() ==1 ) 
            WriteFullConfig(); //full new build
          }
        else if ( RunStartup() == 0 ) //also checks the triggers
          {
          ValidateConfig();
          PrintBanner(1);
          #if defined(BETA)
          if (checkifbetaexpired() == 0)
          #endif
          retcode = (int)Run();
          RunShutdown();
          }
        }
      DeinitializeLogging();
      }
    DeinitializeTriggers();
    }
  return retcode;
}  
#endif

// --------------------------------------------------------------------------

void Client::PrintBanner( int level )
{
  #if (CLIENT_OS == OS_RISCOS)
  if (guiriscos && guirestart)
    return;
  #endif

  if (level == 0)
    {
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
    #if (CLIENT_OS == OS_RISCOS)
    if (guirestart) return;
    #endif
    LogRaw("\nRC5DES Client v2.%d.%d started.\n"
             "Using distributed.net ID %s\n\n",
             CLIENT_CONTEST*100+CLIENT_BUILD,CLIENT_BUILD_FRAC,id);
    }

  return;
}

// --------------------------------------------------------------------------

#if defined(WINNTSERVICE)
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

#if defined(WINNTSERVICE)

static Client *mainclient;

#pragma argsused
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

  // start working
  mainclient->ValidateConfig();

  InitializeTriggers( ((mainclient->noexitfilecheck)?(NULL):
                      ("exitrc5" EXTN_SEP "now")),mainclient->pausefile);
  mainclient->InitializeLogging(); //in logstuff.cpp - copies the smtp ini settings over
  mainclient->Run();
  mainclient->DeinitializeLogging(); //flush and stop logging to file/mail
  DeinitializeTriggers();

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

// ---------------------------------------------------------------------------

s32 Client::Install()
{
#if (!defined(WINNTSERVICE)) && (CLIENT_OS == OS_WIN32) && !defined(NOMAIN)
  HKEY srvkey=NULL;
  DWORD dwDisp=NULL;
  char mypath[200];
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
#elif defined(WINNTSERVICE) && (CLIENT_OS == OS_WIN32)
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

s32 Client::Uninstall(void)
{
#if (!defined(WINNTSERVICE)) && (CLIENT_OS == OS_WIN32) && !defined(NOMAIN)
  HKEY srvkey;

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
#elif defined(WINNTSERVICE) && (CLIENT_OS == OS_WIN32)
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
#elif (CLIENT_OS == OS_OS2)
  int rc;
  const int len = 4068;
  char *cwd;

  char pObjectID[len];
  HOBJECT hObject;

  hObject = WinQueryObject("<RC5DES-CLI>");

  if(hObject == NULLHANDLE)
    LogScreen("ERROR: RC5-DES Client object was not found\n"
          "No RC5-DES client installed in the Startup folder\n");
  else
  {
    LogScreen("RC5-DES Client object found in Startup Folder... ");

    rc = WinDestroyObject(hObject);
    if(rc == TRUE)
      LogScreen("Object removed\n");
    else
      LogScreen("Object NOT removed\n");
  }
#endif
  return 0;
}

// ---------------------------------------------------------------------------

s32 Client::RunStartup(void)
{
  int retcode = 0;

  if ( InitializeTriggers( ((noexitfilecheck)?(NULL):
                        ("exitrc5" EXTN_SEP "now")),pausefile) )
    retcode = -1;   //will have checked the exitstate right away
  else
    {
    #if ((CLIENT_OS == OS_WIN32) && defined(WINNTSERVICE))
      {
      LogScreen("Attempting to start up NT service.\n");
      mainclient = this;
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
      // register ourself as a Win95 service
      SetConsoleTitle("Distributed.Net RC5/DES Client "CLIENT_VERSIONSTRING);
      if (runhidden)
        {
        HMODULE kernl = GetModuleHandle("KERNEL32");
        if (kernl)
          {
          typedef DWORD (CALLBACK *ULPRET)(DWORD,DWORD);
          ULPRET func = (ULPRET) GetProcAddress(kernl, "RegisterServiceProcess");
          if (func) (*func)(0, 1);
          }
        // free the console window
        OSVERSIONINFO osver;
        osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
        GetVersionEx(&osver);
        if (VER_PLATFORM_WIN32_NT == osver.dwPlatformId)
          {
          LogScreen("\n This is not recommended under NT.  Please use the NT Service client"
            "\n (There have been cases of this conflicting with system process csrss.exe)\n"
            "Continuing...\n");
          sleep(2);
          }
        FreeConsole();

        // only allow one running instance
        CreateMutex(NULL, TRUE, "Bovine RC5/DES Win32 Client");
        if (GetLastError()) 
          retcode = -1;
        }
      }
    #endif
    }
  
  return retcode;
}

// ---------------------------------------------------------------------------

void Client::SetNiceness(void)
{
  // renice maximally
  #if (CLIENT_OS == OS_IRIX)
    if ( niceness == 0 )     schedctl( NDPRI, 0, 200 );
    // else                  /* nothing */;
  #elif (CLIENT_OS == OS_OS2)
    if ( niceness == 0 )      DosSetPriority( 2, PRTYC_IDLETIME, 0, 0 );
    else if ( niceness == 1 ) DosSetPriority( 2, PRTYC_IDLETIME, 31, 0 );
    // else                  /* nothing */;
  #elif (CLIENT_OS == OS_WIN32)
    if ( niceness != 2 )      SetPriorityClass( GetCurrentProcess(), IDLE_PRIORITY_CLASS );
    if ( niceness == 0 )      SetThreadPriority( GetCurrentThread() ,THREAD_PRIORITY_IDLE );
    // else                  /* nothing */;
  #elif (CLIENT_OS == OS_MACOS)
     // nothing
  #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
     // nothing - could use the same setting as DOS though
  #elif (CLIENT_OS == OS_NETWARE)
     // nothing - netware sets timeslice dynamically
  #elif (CLIENT_OS == OS_DOS)
     timeslice = dosCliGetTimeslice(); //65536 or GetTimesliceBaseline if win16
  #elif (CLIENT_OS == OS_BEOS)
     // Main control thread runs at normal priority, since it does very little;
     // priority of crunching threads is set when they are created.
  #elif (CLIENT_OS == OS_RISCOS)
     // nothing
  #elif (CLIENT_OS == OS_VMS)
    if ( niceness == 0 )      nice( 4 ); // Assumes base priority of 4, (the
    else if ( niceness == 1 ) nice( 2 ); // default). 0 is highest priority.
    // else                  /* nothing */; // GO-VMS.COM can also be used
  #elif (CLIENT_OS == OS_AMIGAOS)
    if ( niceness == 0 )      SetTaskPri(FindTask(NULL), -20);
    else if ( niceness == 1 ) SetTaskPri(FindTask(NULL), -10);
    // else                  /* nothing */;
  #elif (CLIENT_OS == OS_QNX)
    if ( niceness == 0 )      setprio( 0, getprio(0)-1 );
    else if ( niceness == 1 ) setprio( 0, getprio(0)+1 );
    // else                  /* nothing */;
  #else
    if ( niceness == 0 )      nice( 19 );
    else if ( niceness == 1 ) nice( 10 );
    // else                  /* nothing */;
  #endif
}

#if defined(BETA)
int checkifbetaexpired(void)
{
timeval currenttime;
timeval expirationtime;

expirationtime.tv_sec=EXPIRATIONTIME;

Log("Checking to see if this beta has gone stale:\n");
CliTimer(&currenttime);
Log("Current Date/Time: %s\n",CliGetTimeString(&currenttime,1));
Log("Expiration Date/Time: %s\n",CliGetTimeString(&expirationtime,1));
if (currenttime.tv_sec > expirationtime.tv_sec)
  {
  Log("This beta is old, and may no longer be run.\n"
      "Please download a newer beta, or run a standard release client.\n");
  return 1;
  }
else if (currenttime.tv_sec < expirationtime.tv_sec-1814400)
  {
  Log("Somehow, your date is set BEFORE this beta was released.\n"
      "Please, don't try to fool the date checking system.\n");
  return 1;
  }
else // Finally, it's actually within range.
  {
  Log("Date check passed, please continue testing this client.\n\n");
  };
return 0;
}

#endif

// --------------------------------------------------------------------------

