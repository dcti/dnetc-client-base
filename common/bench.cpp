/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *bench_cpp(void) {
return "@(#)$Id: bench.cpp,v 1.28 1999/07/09 14:09:35 cyp Exp $"; }

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "baseincs.h"  // general includes
#include "problem.h"   // Problem class
#include "triggers.h"  // CheckExitRequestTriggerNoIO()
#include "clitime.h"   // CliTimerDiff(), CliGetTimeString()
#include "clirate.h"   // CliGetKeyrateForProblemNoSave()
#include "client.h"    // needed for fileentry which is needed by clisrate.h
#include "clisrate.h"  // CliGetKeyrateAsString()
#include "clicdata.h"  // GetContestNameFromID()
#include "cpucheck.h"  // GetTimesliceBaseline()
#include "logstuff.h"  // LogScreen()
#include "console.h"   // ConIsScreen()
#include "clievent.h"  // event post etc.
#include "bench.h"     // ourselves
#include "confrwv.h"   // Read/Validate/WriteConfig()

// --------------------------------------------------------------------------

//Sets buffer thresholds
void AutoSetThreshold( Client *clientp, unsigned int contestid,
                       unsigned int /*inbuffer*/, unsigned int /*outbuffer*/ )
{
  int blockstobuffer;
  Client *configclient;
  int configchanged;

  if (clientp && contestid < CONTEST_COUNT ) 
  {
    if (clientp->stopiniio == 0 && clientp->nodiskbuffers == 0)
    {
      configclient = new Client;
      if (configclient)
      {
        configchanged = 1;
        strcpy(configclient->inifilename,clientp->inifilename);
        ReadConfig(configclient);

        //for (contestid == 0; contestid < CONTEST_COUNT; contestid++)
        {
          blockstobuffer=0;
          if (Benchmark(contestid,1L<<20,clientp->cputype,&blockstobuffer)!=0)
          {
            if (blockstobuffer != 0)
            {
              LogScreen("Setting %s buffer threshold to %i block%s.\n",
                   CliGetContestNameFromID(contestid),blockstobuffer,
                   (blockstobuffer == 1 ? "s" : ""));
  
              if ((clientp->inthreshold[contestid] != blockstobuffer ||
                  configclient->inthreshold[contestid] != blockstobuffer) 
                  /* && inbuffer */ )
              {
                configchanged = 1;
                configclient->inthreshold[contestid]=blockstobuffer;
                clientp->inthreshold[contestid]=blockstobuffer;
              }
              if ((clientp->outthreshold[contestid] != blockstobuffer ||
                  configclient->outthreshold[contestid] != blockstobuffer)
                  /* && outbuffer */ )
              {
                configchanged = 1;
                configclient->outthreshold[contestid] = blockstobuffer;
                clientp->outthreshold[contestid] = blockstobuffer;
              }
            }
          }
        }
        if (configchanged)
          WriteConfig(configclient, 1);
        delete configclient;
      }
    }
  }
  return;
}


/* ----------------------------------------------------------------- */

//returns preferred block size or 0 if break
u32 Benchmark( unsigned int contestid, u32 numkeys, int cputype, int *numblocks)
{
  ContestWork contestwork;
  Problem problem;

  int run;
  const char *sm4;
  char cm1, cm2, cm3;
  unsigned int itersize;
  unsigned int keycountshift;
  unsigned int percent;
  unsigned int recommendedblockcount=0;
  unsigned int hourstobuffer = 0;
  u32 tslice;
  const char *contname = CliGetContestNameFromID(contestid);

  if (numkeys == 0)
    itersize = 23;            //8388608 instead of 10000000L;
  else if ( numkeys < (1 << 20))   //max(numkeys,1000000L);
    itersize = 20;            //1048576 instead of 1000000L
  else 
  {
    itersize = 31;
    while (( numkeys & (1<<itersize) ) == 0)
      itersize--;
  }

  if (contestid == DES)
  {
    keycountshift = 1;
    if (itersize < 31) //Assumes that DES is (at least)
      itersize++;      //twice as fast as RC5.
    hourstobuffer = 3; // 3 Hours for DES
  }
  else if (contestid == RC5)
  {
    keycountshift = 0;
    contestid = RC5;
    hourstobuffer = (3*24); // 3 Days for RC5
  }
  else 
  {
    //LogScreen("Error: Contest %s cannot be benchmarked\n", contname );
    return 0;
  }

  tslice = 0x10000;

  #if (CLIENT_OS == OS_NETWARE)
    tslice = GetTimesliceBaseline(); //in cpucheck.cpp
  #endif

  #if (CLIENT_OS == OS_MACOS)
    tslice = GetTimesliceToUse(contestid);
  #endif
  
  contestwork.crypto.key.lo = ( 0 );
  contestwork.crypto.key.hi = ( 0 );
  contestwork.crypto.iv.lo = ( 0 );
  contestwork.crypto.iv.hi = ( 0 );
  contestwork.crypto.plain.lo = ( 0 );
  contestwork.crypto.plain.hi = ( 0 );
  contestwork.crypto.cypher.lo = ( 0 );
  contestwork.crypto.cypher.hi = ( 0 );
  contestwork.crypto.keysdone.lo = ( 0 );
  contestwork.crypto.keysdone.hi = ( 0 );
  contestwork.crypto.iterations.lo = ( (1<<itersize) );
  contestwork.crypto.iterations.hi = ( 0 );

  problem.LoadState( &contestwork, contestid, tslice, cputype );

  percent = 0;
  cm1 = '\n'; 
  cm2 = ((ConIsScreen())?(' '):(0));
  cm3 = ((cm2)?('\r'):(0)); //console.h
  sm4 = ((cm3)?(""):("\n"));
  run = RESULT_WORKING;

  #if (CLIENT_CPU == CPU_X86) && \
      (!defined(SMC) || !defined(MMX_RC5) || !defined(MMX_BITSLICER))
  unsigned int detectedtype = GetProcessorType(1);
  const char *not_supported = NULL;
  if (contestid == RC5)
  {
    if (cputype == 1) /* 486 */
    {
    #if (!defined(SMC))            /* currently only linux */
      not_supported = "RC5/486/SMC";
    #endif
    }
    else if (cputype == 0 && (detectedtype & 0x100)!=0) /* P5 + mmx */
    {
    #if (!defined(MMX_RC5))        /* all non-nasm platforms (bsdi etc) */
      not_supported = "RC5/P5/MMX";
    #endif
    }
  }
  else 
  {
    if ((detectedtype & 0x100) != 0) /* mmx */
    {
    #if (!defined(MMX_BITSLICER))
      not_supported = "DES/MMX";
    #endif
    }
  }
  if (not_supported)
    LogScreen( "Note: this client does not support the %s core.\n", not_supported );
  #endif

  ClientEventSyncPost( CLIEVENT_BENCHMARK_STARTED, (long)((Problem *)(&problem)));

  do
  {
    if ( CheckExitRequestTriggerNoIO() )
    {
      percent = 101;
      run = -1; /* error */
    }
    if (cm1)
    {
      if (percent >= 100)
      {
        sm4 = ((percent == 101)?("    \n*Break*\n"):("    \n"));
        cm2 = 0;
      }
      ClientEventSyncPost( CLIEVENT_BENCHMARK_BENCHING, (long)((Problem *)(&problem)));

      LogScreen( "%cBenchmarking %s with 1*2^%d tests (%u keys):%s%c%u%%",
          cm1, contname, itersize+keycountshift,
          (unsigned int)(1<<(itersize+keycountshift)), sm4, cm2, 
          (unsigned int)(percent) );
    }
    cm1 = cm3;
    percent = 0;

    if ( run == RESULT_WORKING )
    {
      ClientEventSyncPost( CLIEVENT_BENCHMARK_BENCHING, (long)((Problem *)(&problem)));

      run = problem.Run();
      percent = (problem.CalcPermille() + 5)/10;
      if (percent == 0)
        percent = 1;

      #if (CLIENT_OS == OS_NETWARE)   //yield
        nwCliThreadSwitchLowPriority();
      #endif
    }
  } while (percent != 0);

  if ( CheckExitRequestTriggerNoIO() )
  {
    ClientEventSyncPost( CLIEVENT_BENCHMARK_FINISHED, 0 /* NULL */ );
    return 0;
  }

  struct timeval tv;
  char ratestr[32];
  double rate = CliGetKeyrateForProblemNoSave( &problem );
  tv.tv_sec = problem.runtime_sec;  //read the real core time
  tv.tv_usec = problem.runtime_usec;
  ClientEventSyncPost( CLIEVENT_BENCHMARK_FINISHED, (long)((double *)(&rate)));

  LogScreen("Completed in %s [%skeys/sec]\n",  CliGetTimeString( &tv, 2 ),
                    CliGetKeyrateAsString( ratestr, rate ) );

  itersize += keycountshift;
  while ((tv.tv_sec < (60*60) && itersize < 33) || (itersize < 28))
  {
    tv.tv_sec <<= 1;
    tv.tv_usec <<= 1;
    tv.tv_sec += (tv.tv_usec/1000000L);
    tv.tv_usec %= 1000000L;
    itersize++;
  }

  recommendedblockcount = (hourstobuffer*(60*60))/tv.tv_sec;
  if (numblocks) 
    *numblocks = recommendedblockcount;

  LogScreen( "The preferred %s blocksize for this machine should be\n"
             "set to %d (%d*2^28 keys). At the benchmarked keyrate\n"
             "(ie, under ideal conditions) each processor would finish\n"
             "a block of that size in approximately %s.\n"
             "Your buffer thresholds should be set to %u blocks,\n"
             "enough for approximately %u %s.\n", 
             CliGetContestNameFromID(contestid), 
             (unsigned int)itersize, 
             (1<<(itersize-28)),
             CliGetTimeString( &tv, 2 ),recommendedblockcount,
             ((hourstobuffer > 24)?(hourstobuffer/24):(hourstobuffer)),
             ((hourstobuffer > 24)?("days"):("hours")) );

  return (u32)(itersize);
}

// ---------------------------------------------------------------------------

