/* 
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *bench_cpp(void) {
return "@(#)$Id: bench.cpp,v 1.27.2.45 2000/11/07 12:44:10 cyp Exp $"; }

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "baseincs.h"  // general includes
#include "client.h"    // contest enum
#include "problem.h"   // Problem class
#include "triggers.h"  // CheckExitRequestTriggerNoIO()
#include "clitime.h"   // CliGetTimeString()
#include "clicdata.h"  // GetContestNameFromID()
#include "selcore.h"   // selcoreGet[SelectedCoreForContest|DisplayName]()
#include "logstuff.h"  // LogScreen()
#include "clievent.h"  // event post etc.
#include "bench.h"     // ourselves

#define TBENCHMARK_CALIBRATION 0x80

long TBenchmark( unsigned int contestid, unsigned int numsecs, int flags )
{
  /* non-preemptive os minimum yields per second */
  struct { int yps, did_adjust; } non_preemptive_os;
  long retvalue = -1L; /* assume error */
  Problem *problem; 
  u32 tslice; 

  if (!IsProblemLoadPermitted(-1 /*any thread*/, contestid))
    return 0;

  /* ++++++ determine initial 'timeslice' +++++ */

  tslice = 0; /* zero means 'use calibrated value' */
  non_preemptive_os.yps = 0; /* assume preemptive OS */
  non_preemptive_os.did_adjust = 0;

  #if (CLIENT_OS == OS_NETWARE)
  if ( ( flags & TBENCHMARK_CALIBRATION ) != 0 ) // 2 seconds without yield
    numsecs = ((numsecs > 2) ? (2) : (numsecs)); // ... is acceptable
  else
  {
    non_preemptive_os.yps = 1000/10; /* 10 ms minimum yield rate */ 
    tslice = 0; /* zero means 'use calibrated value' */
  }  
  #elif (CLIENT_OS == OS_MACOS)
  if ( ( flags & TBENCHMARK_CALIBRATION ) != 0 ) // 2 seconds without yield
    numsecs = ((numsecs > 2) ? (2) : (numsecs)); // ... is acceptable
  else
  {    
    non_preemptive_os.yps = 1000/20; /* 20 ms minimum yield rate */
    tslice = 0; /* zero means 'use calibrated value' */
  }
  #elif (CLIENT_OS == OS_WIN16 || CLIENT_OS == OS_WIN32 /* win32s */) 
  if ( ( flags & TBENCHMARK_CALIBRATION ) != 0 ) // 2 seconds without yield
    numsecs = ((numsecs > 2) ? (2) : (numsecs)); // ... is acceptable
  else if (winGetVersion() < 400) /* win16 or win32s */
  {  
    non_preemptive_os.yps = 1000/20; /* 20 ms minimum yield rate */ 
    tslice = 0;  /* zero means 'use calibrated value' */
  } 
  #elif (CLIENT_OS == OS_RISCOS)
  if ( ( flags & TBENCHMARK_CALIBRATION ) != 0 ) // 2 seconds without yield
    numsecs = ((numsecs > 2) ? (2) : (numsecs)); // ... is acceptable
  else if (riscos_check_taskwindow())
  {
    non_preemptive_os.yps = 1000/100; /* 100 ms minimum yield rate */ 
    tslice = 0;  /* zero means 'use calibrated value' */
  }
  #endif

  if (tslice == 0 && ( flags & TBENCHMARK_CALIBRATION ) == 0 )
  {
    long res;
    if ((flags & TBENCHMARK_QUIET) == 0)
    {
      selcoreGetSelectedCoreForContest(contestid); /* let selcore message */
      //LogScreen("Calibrating ... " );
    }  
    res = TBenchmark( contestid, 2, TBENCHMARK_QUIET | TBENCHMARK_IGNBRK | TBENCHMARK_CALIBRATION );
    if ( res == -1 ) /* LoadState failed */
    {
      if ((flags & TBENCHMARK_QUIET) == 0)
        LogScreen("\rCalibration failed!\n");
      return -1;
    }  
    tslice = (((u32)res) + 0xFFF) & 0xFFFFF000;
    //if ((flags & TBENCHMARK_QUIET) == 0)
    //  LogScreen("\rCalibrating ... done. (%lu)\n", (unsigned long)tslice );
    if (non_preemptive_os.yps)
      tslice /= non_preemptive_os.yps;
    else
      tslice <<= 1; /* try for two second steps */
  }
  if (tslice == 0)
  { 
    tslice = 0x10000;
    if (non_preemptive_os.yps)
      tslice = 4096;
  }

  /* ++++++ run the benchmark +++++ */
  
  problem = new Problem();
  if ( problem->LoadState( CONTESTWORK_MAGIC_BENCHMARK, 
                           contestid, tslice, 0, 0, 0, 0) == 0)
  {
    const char *contname = CliGetContestNameFromID(contestid);
    int silent = 1, run = RESULT_WORKING;
    unsigned long last_permille = 1001;

    //ClientEventSyncPost(CLIEVENT_BENCHMARK_STARTED, (long)problem );
    if ((flags & TBENCHMARK_QUIET) == 0)
    {
      silent = 0;
      LogScreen("%s: Benchmarking ... ", contname );
    }
    while ( run == RESULT_WORKING )
    {
      unsigned long permille; u32 ratehi, ratelo;

      if (non_preemptive_os.yps) /* is this a non-preemptive environment? */
      {
        if (non_preemptive_os.did_adjust < 30 /* don't do this too often */
           && problem->runtime_sec >= ((u32)(2+non_preemptive_os.did_adjust)))
        {
          u32 newtslice;
          if (problem->GetProblemInfo(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      &ratelo, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0 ) == -1)
          {
            run = -2;
            break;
          }
          newtslice = (u32)(ratelo/((u32)non_preemptive_os.yps));
          if (newtslice > (tslice + (tslice/10)))
          {
            non_preemptive_os.did_adjust++;
            numsecs++; /* bench for a bit more */
          }
          if (newtslice > tslice)
            problem->tslice = tslice = newtslice;
        }
        #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32) /* win32s */
        w32Yield(); /* pump waiting messages */
        #elif (CLIENT_OS == OS_MACOS)
        macosSmartYield();
        #elif (CLIENT_OS == OS_RISCOS)
        riscos_upcall_6();
        #elif (CLIENT_OS == OS_NETWARE)
        ThreadSwitchLowPriority();
        #endif
      }
      else /* preemptive environment */ if (problem->last_runtime_sec < 2)
      {
        /* this tweak is only meaningful for contests that slice */
        /* precisely and don't have a cap on the maximum tslice, */
        /* and is probably only useful during the calibration phase. */
        /* (a simple counter test showed that it only came here one */
        /* time after tslice had been calibrated) */
        u32 newtslice; 
        u32 elapsedus = (problem->last_runtime_sec * 100000ul)+
                         problem->last_runtime_usec;
        tslice = problem->tslice;                           
        newtslice = tslice + ((tslice/100)*(100-(elapsedus/20000ul)));
        if (newtslice > tslice)
        {
          problem->tslice = tslice = newtslice;
        }                                    
      }
      run = problem->Run();
      if ( run < 0 )
      {
        run = -1;
        break;
      }   
      if ((flags & TBENCHMARK_IGNBRK)==0 && CheckExitRequestTriggerNoIO())
      {
        if (!silent)
          LogScreen("\r%s: Benchmarking ... *Break*", contname );
        run = -3;
        break;
      }
      permille = 1000; /* assume finished */
      if (run == RESULT_WORKING) /* not finished */
      {
        permille = ((problem->runtime_sec  * 1000) + 
                    (problem->runtime_usec / 1000)) / numsecs;
        if (permille > 1000)
          permille = 1000;
      }
      if (permille == 1000 || last_permille != (permille / 10))
      {    
        if (!silent)
          LogScreen("\r%s: Benchmarking ... %u.%02u%% done", 
                     contname, (unsigned int)(permille/10), 
                             (unsigned int)((permille%10)*10) );
        last_permille = (permille / 10);
        //ClientEventSyncPost(CLIEVENT_BENCHMARK_BENCHING, (long)problem );
      }                                 
      if (permille == 1000) /* time is up or ran out of work */
      {
        char ratebuf[32]; u32 secs, usecs;
        if (problem->GetProblemInfo(0, 0, /* cont_id, cont_name */
                                    &secs, &usecs, 
                                    0, 0, 0, /* swucount, pad_strings, unit_name */
                                    0, 0, 0, /* currpermille, startperm, poie */
                                    0, 0,    /* idbuf, idbufsz */
                                    &ratehi, &ratelo, 
                                    ratebuf, sizeof(ratebuf),
                                    0, 0, 0, 0,   0, 0, 0, 0, 
                                    0, 0, 0, 0 ) == -1)
        {
          run = -4;
          break;
        }
        retvalue = (long)ratelo;
        if (!silent)
        {
          struct timeval tv; 
          tv.tv_sec = secs; tv.tv_usec = usecs;
          LogScreen("\r");
          Log("%s: Benchmark for core #%d (%s)\n%s [%s/sec]\n",
             contname, problem->coresel, 
             selcoreGetDisplayName(contestid, problem->coresel),
             CliGetTimeString( &tv, 2 ), ratebuf );
        }
        //ClientEventSyncPost(CLIEVENT_BENCHMARK_FINISHED, (long)problem );
        break;
      } /* permille == 1000 */
    } /* while ( run == RESULT_WORKING ) */

    if (!silent)
    {
      if (run < 0) /* error */
      {
        LogScreen("\r");
        Log("%s: Benchmark failed (error: %d). Unable to determine rate.", run, contname );
      }
    }
    problem->RetrieveState(NULL, NULL, 1, 0); //purge the problem

  } /* if (LoadState() == 0) */

  delete problem;

  return retvalue;
}  


