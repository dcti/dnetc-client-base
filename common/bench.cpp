// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: bench.cpp,v $
// Revision 1.11  1999/01/01 02:45:14  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.10  1998/12/28 21:23:09  cyp
// Added event support.
//
// Revision 1.9  1998/12/15 14:11:56  cyp
// x86 change: Tell the user if his/her client does not support the 
// fastest core that is 'available' for contest X.
//
// Revision 1.8  1998/12/15 13:28:22  cyp
// Prettified: reformatted long log messages, converted LogScreenRaw() to 
// LogScreen() etc
//
// Revision 1.7  1998/12/08 05:29:45  dicamillo
// MacOS updates for timeslice and GUI display
//
// Revision 1.6  1998/10/26 04:15:31  cyp
// Replaces use of the IS_A_TTY() macro with a call to ConIsScreen()
//
// Revision 1.5  1998/10/11 00:46:29  cyp
// Benchmark() is now standalone.
//
// Revision 1.4  1998/10/09 00:42:50  blast
// Benchmark was looking at contest 2=DES, other=RC5 and cmdline.cpp
// was setting 0=RC5, 1=DES, made it run two rc5 benchmarks. FIXED
// Changed Calling convention for Benchmark() from u8 to unsigned int.
//
// Revision 1.3  1998/10/04 21:31:08  blast
// Added #include "baseincs.h"
//
// Revision 1.2  1998/09/29 10:03:33  chrisb
// RISC OS doesn't understand fileno
//
// Revision 1.1  1998/09/28 01:38:04  cyp
// Spun off from client.cpp  Note: the problem object is local so it does not
// need to be assigned from the problem table. Another positive side effect
// is that benchmarks can be run without shutting down the client.
//
//

#if (!defined(lint) && defined(__showids__))
const char *bench_cpp(void) {
return "@(#)$Id: bench.cpp,v 1.11 1999/01/01 02:45:14 cramer Exp $"; }
#endif

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "baseincs.h"  // general includes
#include "problem.h"   // Problem class
#include "network.h"   // ntohl()/htonl()
#include "triggers.h"  // CheckExitRequestTriggerNoIO()
#include "clitime.h"   // CliTimerDiff(), CliGetTimeString()
#include "clirate.h"   // CliGetKeyrateForProblemNoSave()
#include "client.h"    // needed for fileentry which is needed by clisrate.h
#include "clisrate.h"  // CliGetKeyrateAsString()
#include "cpucheck.h"  // GetTimesliceBaseline()
#include "logstuff.h"  // LogScreen()
#include "console.h"   // ConIsScreen()
#include "clievent.h"  // event post etc.
#include "bench.h"     // ourselves

// --------------------------------------------------------------------------

//returns preferred block size or 0 if break
u32 Benchmark( unsigned int contest, u32 numkeys, int cputype )
{
  ContestWork contestwork;
  Problem problem;

  u32 run;
  const char *sm4;
  char cm1, cm2, cm3;

  unsigned int itersize;
  unsigned int keycountshift;
  const char *contestname;
  unsigned int contestid;
  u32 tslice;

  if (numkeys == 0)
    itersize = 23;            //8388608 instead of 10000000L;
  else if ( numkeys < (1<<20))   //max(numkeys,1000000L);
    itersize = 20;            //1048576 instead of 1000000L
  else 
    {  
    itersize = 31;
    while (( numkeys & (1<<itersize) ) == 0)
      itersize--;
    }

  if (contest == 1 && itersize < 31) //Assumes that DES is (at least)
    itersize++;                      //twice as fast as RC5.

  if (contest == 1)
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

  tslice = 100000L;

  #if (CLIENT_OS == OS_NETWARE)
    tslice = GetTimesliceBaseline(); //in cpucheck.cpp
  #endif

  #if (CLIENT_OS == OS_MACOS)
    tslice = GetTimesliceToUse(contestid);
  #endif
  
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

  problem.LoadState( &contestwork, contestid, tslice, cputype );

  problem.percent = 0;
  cm1 = '\n'; 
  cm2 = ((ConIsScreen())?(' '):(0));
  cm3 = ((cm2)?('\r'):(0)); //console.h
  sm4 = ((cm3)?(""):("\n"));
  run = 0;

  #if (CLIENT_CPU == CPU_X86) && \
      (!defined(SMC) || !defined(MMX_RC5) || !defined(MMX_BITSLICER))
  unsigned int detectedtype = GetProcessorType(1);
  const char *not_supported = "Note: this client does not support the %s core.\n";
  if (contestid == 0)
    {
    #if (!defined(SMC))            /* all non-gcc platforms except netware */
    if (cputype == 1) /* 486 */
      LogScreen(not_supported, "RC5/486/SMC");
    #endif
    #if (!defined(MMX_RC5))        /* all non-nasm platforms (bsdi etc) */
    if (cputype == 0 && (detectedtype & 0x100)!=0) /* P5 + mmx */
      LogScreen(not_supported, "RC5/P5/MMX");
    #endif
    }
  else 
    {
    #if (!defined(MMX_BITSLICER))
    if ((detectedtype & 0x100) != 0) /* mmx */
      LogScreen(not_supported, "DES/MMX");
    #endif
    }
  #endif

  ClientEventSyncPost( CLIEVENT_BENCHMARK_STARTED, (long)((Problem *)(&problem)));

  do{
    if ( CheckExitRequestTriggerNoIO() )
      {
      problem.percent = 101;
      run = 1;
      }
    if (cm1)
      {
      if (problem.percent >= 100)
        {
        sm4 = ((problem.percent == 101)?("    \n*Break*\n"):("    \n"));
        cm2 = 0;
        }
      ClientEventSyncPost( CLIEVENT_BENCHMARK_BENCHING, (long)((Problem *)(&problem)));

      LogScreen( "%cBenchmarking %s with 1*2^%d tests (%u keys):%s%c%u%%",
          cm1, contestname, itersize+keycountshift,
          (unsigned int)(1<<(itersize+keycountshift)), sm4, cm2, 
          (unsigned int)(problem.percent) );
      }
    cm1 = cm3;
    problem.percent = 0;

    if ( run == 0 )
      {
      ClientEventSyncPost( CLIEVENT_BENCHMARK_BENCHING, (long)((Problem *)(&problem)));

      run = problem.Run( 0 );  //threadnum
      if ( run )
        {
        if ( problem.finished )
          problem.percent = 100;
        }
      else
        {
        problem.percent = problem.CalcPercent();
        if (problem.percent == 0)
          problem.percent = 1;

        #if (CLIENT_OS == OS_NETWARE)   //yield
          nwCliThreadSwitchLowPriority();
        #endif
        }
      }
    } while (problem.percent != 0);

  if ( CheckExitRequestTriggerNoIO() )
    {
    ClientEventSyncPost( CLIEVENT_BENCHMARK_FINISHED, 0 /* NULL */ );
    return 0;
    }

  struct timeval tv;
  char ratestr[32];
  double rate = CliGetKeyrateForProblemNoSave( &problem );
  tv.tv_sec = problem.timehi;  //read the time the problem:run started
  tv.tv_usec = problem.timelo;
  CliTimerDiff( &tv, &tv, NULL );    //get the elapsed time
  ClientEventSyncPost( CLIEVENT_BENCHMARK_FINISHED, (long)((double *)(&rate)));

  LogScreen("Completed in %s [%skeys/sec]\n",  CliGetTimeString( &tv, 2 ),
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

  LogScreen( "The preferred %s blocksize for this machine should be\n"
             "set to %d (%d*2^28 keys). At the benchmarked keyrate\n"
             "(ie, under ideal conditions) each processor would finish\n"
             "a block of that size in approximately %s.\n", contestname, 
             (unsigned int)itersize, 
             (unsigned int)((((u32)(1<<itersize))/((u32)(1<<28)))),
             CliGetTimeString( &tv, 2 ));  

  return (u32)(itersize);
}

// ---------------------------------------------------------------------------
