/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *bench_cpp(void) {
return "@(#)$Id: bench.cpp,v 1.44 1999/12/12 15:45:02 cyp Exp $"; }

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "baseincs.h"  // general includes
#include "client.h"    // contest enum
#include "problem.h"   // Problem class
#include "triggers.h"  // CheckExitRequestTriggerNoIO()
#include "clitime.h"   // CliGetTimeString()
#include "clisrate.h"  // CliGetKeyrateAsString()
#include "clicdata.h"  // GetContestNameFromID()
#include "selcore.h"   // 
#include "cpucheck.h"  // GetProcessorType()
#include "logstuff.h"  // LogScreen()
#include "clievent.h"  // event post etc.
#include "bench.h"     // ourselves

/* ----------------------------------------------------------------- */

static void __show_notbest_msg(unsigned int contestid)
{
  #if (CLIENT_CPU == CPU_X86) && \
      (!defined(SMC) || !defined(MMX_RC5) || !defined(MMX_BITSLICER))
  int corenum = selcoreGetSelectedCoreForContest( contestid );
  unsigned int detectedtype = GetProcessorType(1);
  const char *not_supported = NULL;
  if (contestid == RC5)
  {
    if (corenum == 1) /* 486 */
    {
      #if (!defined(SMC))            /* currently only linux */
        not_supported = "RC5/486/SMC";
      #endif
    }
    else if (corenum == 0 && (detectedtype & 0x100)!=0) /* P5 + mmx */
    {
      #if (!defined(MMX_RC5))        /* all non-nasm platforms (bsdi etc) */
        not_supported = "RC5/P5/MMX";
      #endif
    }
  }
  else if (contestid == CSC)
  {
    if ((detectedtype & 0x100) != 0) /* mmx */
    {
      #if (!defined(MMX_CSC))
        not_supported = "CSC/MMX bitslice";
      #endif
    }
  }
  else if (contestid == DES)
  {
    if ((detectedtype & 0x100) != 0) /* mmx */
    {
      #if (!defined(MMX_BITSLICER))
        not_supported = "DES/MMX bitslice";
      #endif
    }
  }
  if (not_supported)
    LogScreen( "Note: this client does not support\nthe %s core.\n", not_supported );
  #endif

  contestid = contestid;  
  return;
}

/* ----------------------------------------------------------------- */

static void __add_u64( struct fake_u64 *result, 
                       const struct fake_u64 *a, 
                       const struct fake_u64 *b )
{
  u32 lo = a->lo + b->lo;
  result->hi = a->hi + b->hi;
  if( lo < a->lo || lo < b->lo )
    result->hi++;
  result->lo = lo;
}

/* ----------------------------------------------------------------- */

static double __calc_rate( unsigned int contestid, 
                           const ContestWork *contestwork, 
                           int last_run_result, 
                           const struct fake_u64 *keys_done_already,
                           const struct timeval *totalruntime, 
                           int print_it )
{                         
  char ratestr[32];
  double rate;
  double keysdone;
  rate = 0.0;
  struct fake_u64 keys_already_done;
  keys_already_done.hi = keys_done_already->hi;
  keys_already_done.lo = keys_done_already->lo;

  switch (contestid)
  {
    case RC5:
    case DES:
    case CSC:
    {
      unsigned int count;
      if (CliGetContestInfoBaseData( contestid, NULL, &count ) != 0)
        count = 1;
      if( last_run_result == RESULT_WORKING )
        __add_u64( &keys_already_done, &keys_already_done, &(contestwork->crypto.keysdone) );
      keysdone = 
        (double)keys_already_done.lo + 
        (double)keys_already_done.hi * 4294967296.0 /* 2^32 */;
      if (count>1) //iteration-to-keycount-multiplication-factor
        keysdone = (keysdone)*((double)(count));
      rate = keysdone / (((double)(totalruntime->tv_sec))+
                 (((double)(totalruntime->tv_usec))/((double)(1000000L))));
      if (print_it)
      {
        LogScreen("\rCompleted in %s [%skeys/sec]\n",  
                 CliGetTimeString( totalruntime, 2 ),
                 CliGetKeyrateAsString( ratestr, rate ) );
      }
      break;
    }
    case OGR:
    {
      if( last_run_result == RESULT_WORKING )
        __add_u64( &keys_already_done, &keys_already_done, &(contestwork->ogr.nodes) );
      keysdone = 
        (double)keys_already_done.lo + 
        (double)keys_already_done.hi * 4294967296.0 /* 2^32 */;
      rate = keysdone / (((double)(totalruntime->tv_sec))+
              (((double)(totalruntime->tv_usec))/((double)(1000000L))));
      if (print_it)
      {
        LogScreen("\rCompleted in %s [%snodes/sec]\n",
                 CliGetTimeString( totalruntime, 2 ),
                 CliGetKeyrateAsString( ratestr, rate ) );
      }
      break;
    }
  }
  return (rate);
}

/* ----------------------------------------------------------------- */

#define TBENCHMARK_CALIBRATION 0x80

long TBenchmark( unsigned int contestid, unsigned int numsecs, int flags )
{
  long retvalue;
  int run, scropen; u32 tslice; 
  struct { int yps, did_adjust; } non_preemptive_os;
  /* non-preemptive os minimum yields per second */
  Problem *problem;
  unsigned long last_permille;
  ContestWork contestwork;
  const char *contname;
  struct timeval totalruntime;
  struct fake_u64 keysdone;

  contname = CliGetContestNameFromID(contestid);
  if (!contname)
    return 0;
  if (!IsProblemLoadPermitted(-1 /*any thread*/, contestid))
    return 0;

  switch (contestid)
  {
    case RC5:
    case DES:
    case CSC:
    {
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
      contestwork.crypto.iterations.lo = ( (1<<20) );
      contestwork.crypto.iterations.hi = ( 0 );
      break;
    }
    case OGR:
    {
      contestwork.ogr.workstub.stub.marks = 24;
      contestwork.ogr.workstub.worklength = 
      contestwork.ogr.workstub.stub.length = 7;
      contestwork.ogr.workstub.stub.diffs[0] = 2;
      contestwork.ogr.workstub.stub.diffs[1] = 22;
      contestwork.ogr.workstub.stub.diffs[2] = 32;
      contestwork.ogr.workstub.stub.diffs[3] = 21;
      contestwork.ogr.workstub.stub.diffs[4] = 5;
      contestwork.ogr.workstub.stub.diffs[5] = 1;
      contestwork.ogr.workstub.stub.diffs[6] = 12;
      contestwork.ogr.nodes.lo = 0;
      contestwork.ogr.nodes.hi = 0;
      break;
    }
  }

  tslice = 0; /* zero means 'use calibrated value' */
  non_preemptive_os.yps = 0; /* assume preemptive OS */
  non_preemptive_os.did_adjust = 0;

  #if (CLIENT_OS == OS_NETWARE)
  if ( ( flags & TBENCHMARK_CALIBRATION ) != 0 ) // 2 seconds without yield
    numsecs = ((numsecs > 2) ? (2) : (numsecs)); // ... is acceptable
  else if (GetFileServerMajorVersionNumber() < 5)
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
  }
  if (tslice == 0)
  { 
    tslice = 0x10000;
    if (non_preemptive_os.yps)
      tslice = 4096;
  }
  
  totalruntime.tv_sec = 0;
  totalruntime.tv_usec = 0;
  scropen = run = -1;
  keysdone.lo = 0;
  keysdone.hi = 0;
  last_permille = 10000;
  
  /* --------------------------- */

  problem = new Problem();
  run = RESULT_WORKING;

  while (((unsigned int)totalruntime.tv_sec) < numsecs)
  {
    run = RESULT_WORKING;
    if ( problem->LoadState( &contestwork, contestid, tslice, 0 /*unused*/) != 0)
      run = -1;
    else if ((flags & TBENCHMARK_QUIET) == 0 && scropen < 0)
    {
      scropen = 1;
      __show_notbest_msg(contestid);
      LogScreen("Benchmarking %s ... ", contname );
    }
    while ( run == RESULT_WORKING )
    {
      if (non_preemptive_os.yps) /* is this a non-preemptive environment? */
      {
        if (non_preemptive_os.did_adjust < 30 /* don't do this too often */
           && totalruntime.tv_sec >= (2+non_preemptive_os.did_adjust))
        {
          ContestWork tmp_work;
          if (problem->RetrieveState(&tmp_work, NULL, 0) >= 0)
          {
            double rate = __calc_rate(contestid, &tmp_work, run, 
                                      &keysdone, &totalruntime, 0);
            u32 newtslice = (u32)(rate/((double)non_preemptive_os.yps));
            if (newtslice > (tslice + (tslice/10)))
            {
              non_preemptive_os.did_adjust++;
              numsecs++; /* bench for a bit more */
            }
            if (newtslice > tslice)
              tslice = newtslice;
          }
        }
        #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32) /* win32s */
        w32Yield(); /* pump waiting messages */
        #elif (CLIENT_OS == OS_MACOS)
        DoYieldToMain(0);
        #elif (CLIENT_OS == OS_RISCOS)
        riscos_upcall_6();
        #elif (CLIENT_OS == OS_NETWARE)
        nwCliThreadSwitchLowPriority();
        #endif
      }
      run = problem->Run();
      if ( run < 0 )
        break;
      else if ((flags & TBENCHMARK_IGNBRK)==0 && 
                CheckExitRequestTriggerNoIO())
      {
        run = -1; /* error */
        if (scropen > 0)
          LogScreen("\rBenchmarking %s ... *Break*       ", contname );
        break;
      }
      else
      {
        struct timeval runtime;
        runtime.tv_sec = totalruntime.tv_sec  + problem->runtime_sec;
        runtime.tv_usec = totalruntime.tv_usec + problem->runtime_usec;
        if (runtime.tv_usec >= 1000000)
        {
          runtime.tv_usec -= 1000000;
          runtime.tv_sec++;
        }
        if (scropen > 0)
        {
          unsigned long permille = (((runtime.tv_sec * 1000) + 
                                    (runtime.tv_usec / 1000)) ) / numsecs;
          if (permille >= 1000)
            permille = 1000;
          if (last_permille != permille)
          {    
            LogScreen("\rBenchmarking %s ... %u.%02u%% done", 
                       contname, (unsigned int)(permille/10), 
                                 (unsigned int)((permille%10)*10) );
            last_permille = permille;
          }                                 
        }
        if ( run != RESULT_WORKING) /* finished this block */
        {
          ContestWork tmp_work;
          switch( contestid ) 
          {
          case RC5:
          case DES:
          case CSC:
            if( problem->RetrieveState(&tmp_work, NULL, 0) >= 0 )
               __add_u64( &keysdone, &keysdone, &(tmp_work.crypto.keysdone) );
            break;

          case OGR:
            if( problem->RetrieveState(&tmp_work, NULL, 0) >= 0 )
              __add_u64( &keysdone, &keysdone, &(tmp_work.ogr.nodes) );
            break;
          }
          totalruntime.tv_sec = runtime.tv_sec;
          totalruntime.tv_usec = runtime.tv_usec;
        }
        else if ( ((unsigned int)runtime.tv_sec) >= numsecs )
        {
          totalruntime.tv_sec = runtime.tv_sec;
          totalruntime.tv_usec = runtime.tv_usec;
          break;
        }
      }
    }
    if ( run < 0 )
      break;
  }
  if (run < 0) /* errors or ^C */
    run = -1; /* core error */
  else if (problem->RetrieveState(&contestwork, NULL, 0) < 0)
    run = -1; /* core error */
  if (scropen > 0 && run < 0)
    LogScreen("\n");
  delete problem;
  
  /* --------------------------- */
  
  retvalue = -1; /* assume error */
  if (run >= 0) /* no errors, no ^C */
    retvalue = (long)__calc_rate(contestid, &contestwork, run, &keysdone, 
                         &totalruntime, (!(flags & TBENCHMARK_QUIET)) );

  return retvalue;
}  

// ---------------------------------------------------------------------------

//old style
u32 Benchmark( unsigned int contestid, u32 numkeys, int * /*numblocks*/)
{                                                        
  unsigned int numsecs = 8;
  if (numkeys == 0)
    numsecs = 32;  
  else if ( numkeys >= (1 << 23)) /* 1<<23 used to be our "long" bench */
    numsecs = 16;                 /* our "short" bench used to be 1<<20 */
  TBenchmark( contestid, numsecs, 0 );
  return 0;
}

