// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: bench.cpp,v $
// Revision 1.4  1998/10/09 00:42:50  blast
// Benchmark was looking at contest 2=DES, other=RC5 and cmdline.cpp
// was setting 0=RC5, 1=DES, made it run two rc5 benchmarks. FIXED
//
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
return "@(#)$Id: bench.cpp,v 1.4 1998/10/09 00:42:50 blast Exp $"; }
#endif

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "problem.h"   // Problem class
#include "network.h"   // ntohl()/htonl()
#include "triggers.h"  //[Check|Raise][Pause|Exit]RequestTrigger()
#include "clitime.h"
#include "clirate.h"
#include "clisrate.h"
#include "cpucheck.h"  //GetTimesliceBaseline()
#include "logstuff.h"  //Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "baseincs.h"  //General includes

// --------------------------------------------------------------------------

u32 Client::Benchmark( unsigned int contest, u32 numk )
{
  static int done_selcore = 0;
  
  ContestWork contestwork;
  Problem problem;

  u32 run;
  const char *sm0, *sm4;
  char cm1, cm2, cm3;

  unsigned int itersize;
  unsigned int keycountshift;
  const char *contestname;
  unsigned int contestid;
  u32 tslice;

  if (!done_selcore && SelectCore()) 
    return 0;
  done_selcore = 1;

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

  problem.LoadState( &contestwork , (u32) (contestid), tslice, cputype );

  problem.percent = 0;
  sm0 = "%cBenchmarking %s with 1*2^%d tests (%u keys):%s%c%u%%";
  cm1 = '\n'; 
  #if (defined(NEEDVIRTUALMETHODS)) || (CLIENT_OS == OS_RISCOS)
  cm3 = '\r';
  #else
  cm3 = ((isatty(fileno(stdout)))?('\r'):(0));
  #endif
  sm4 = ((cm3)?(""):("\n"));
  cm2 = ((cm3)?(' '):(0));
  run = 0;

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
        sm4 = ((problem.percent == 101)?(" *Break* \n"):("             \n"));
        cm2 = 0;
        }
      LogScreenRaw( sm0, cm1, contestname, itersize+keycountshift,
          (unsigned int)(1<<(itersize+keycountshift)), sm4, cm2, 
          (unsigned int)(problem.percent) );
      }
    cm1 = cm3;
    problem.percent = 0;

    if ( run == 0 )
      {
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
    return 0;

  struct timeval tv;
  char ratestr[32];
  double rate = CliGetKeyrateForProblemNoSave( &problem );
  tv.tv_sec = problem.timehi;  //read the time the problem:run started
  tv.tv_usec = problem.timelo;
  CliTimerDiff( &tv, &tv, NULL );    //get the elapsed time
  LogScreenRaw("Completed in %s [%skeys/sec]\n",  CliGetTimeString( &tv, 2 ),
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
