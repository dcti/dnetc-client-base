/* 
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *bench_cpp(void) {
return "@(#)$Id: bench.cpp,v 1.48 2000/01/08 23:36:03 cyp Exp $"; }

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "baseincs.h"  // general includes
#include "client.h"    // contest enum
#include "problem.h"   // Problem class
#include "triggers.h"  // CheckExitRequestTriggerNoIO()
#include "clitime.h"   // CliGetTimeString()
#include "clisrate.h"  // CliGetKeyrateAsString()
#include "clicdata.h"  // GetContestNameFromID()
#include "selcore.h"   // selcoreGet[SelectedCoreForContest|DisplayName]()
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

static double __calc_rate( unsigned int contestid, 
                           const ContestWork *contestwork, 
                           int last_run_result, 
                           u32 keysdone_already_hi,
                           u32 keysdone_already_lo,
                           const struct timeval *totalruntime, 
                           const char *contname, 
                           int corenum, int corecpu, int print_it )
{                         
  double rate;
  const char *rateunit = "";
  double keysdone = (double)keysdone_already_lo + 
                    (double)keysdone_already_hi * 4294967296.0 /* 2^32 */;
  corecpu = corecpu; /* unused */
  
  switch (contestid)
  {
    case RC5:
    case DES:
    case CSC:
    {
      unsigned int multiplier;
      if ( last_run_result == RESULT_WORKING )
        keysdone = keysdone +
        (double)contestwork->crypto.keysdone.lo + 
        (double)contestwork->crypto.keysdone.hi * 4294967296.0 /* 2^32 */;
      if (CliGetContestInfoBaseData( contestid, NULL, &multiplier ) == 0)
      {
        if (multiplier > 1) //iteration-to-keycount multiplication-factor
          keysdone = (keysdone)*((double)(multiplier));
      }
      rateunit = "keys/sec";
      break;
    }
    case OGR:
    {
      if ( last_run_result == RESULT_WORKING )
        keysdone = keysdone +
        (double)contestwork->ogr.nodes.lo + 
        (double)contestwork->ogr.nodes.hi * 4294967296.0 /* 2^32 */;
      rateunit = "nodes/sec";
      break;
    }      
  }        

  rate = keysdone;  /* guard against divide by zero */
  if (totalruntime->tv_sec != 0 || totalruntime->tv_usec != 0)
    rate = keysdone / (((double)(totalruntime->tv_sec))+
                   (((double)(totalruntime->tv_usec))/((double)(1000000L))));

  if (print_it)
  {
    char ratestr[32];
    LogScreen("\r%s: Benchmark completed in %s"
              "\n%s: core #%d (%s) - [%s%s]\n",
              contname, CliGetTimeString( totalruntime, 2 ),
              contname, corenum, selcoreGetDisplayName(contestid, corenum),
                        CliGetKeyrateAsString( ratestr, rate ), rateunit );
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
  u32 keysdone_hi, keysdone_lo;
  unsigned int workunitsec = 0;

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
  keysdone_lo = 0;
  keysdone_hi = 0;
  last_permille = 1001;
  
  /* --------------------------- */

  problem = new Problem();
  run = RESULT_WORKING;

  while (((unsigned int)totalruntime.tv_sec) < numsecs)
  {
    run = RESULT_WORKING;
    if ( problem->LoadState( &contestwork, contestid, tslice, 0, 0, 0, 0) != 0)
      run = -1;
    else if ((flags & TBENCHMARK_QUIET) == 0 && scropen < 0)
    {
      scropen = 1;
      __show_notbest_msg(contestid);
      LogScreen("%s: Benchmarking ... ", contname );
    }
    while ( run == RESULT_WORKING )
    {
      ContestWork tmp_work;
      if (non_preemptive_os.yps) /* is this a non-preemptive environment? */
      {
        if (non_preemptive_os.did_adjust < 30 /* don't do this too often */
           && totalruntime.tv_sec >= (2+non_preemptive_os.did_adjust))
        {
          if (problem->RetrieveState(&tmp_work, NULL, 0) >= 0)
          {
            double rate = __calc_rate(contestid, &tmp_work, run, 
                                      keysdone_hi, keysdone_lo, 
                                      &totalruntime, contname, 
                                      problem->coresel, problem->client_cpu, 0);
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
        sched_yield(); /* posix <sched.h> */
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
          LogScreen("\r%s: Benchmarking ... *Break*       ", contname );
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
          if (last_permille != (permille / 10))
          {    
            LogScreen("\r%s: Benchmarking ... %u.%02u%% done", 
                       contname, (unsigned int)(permille/10), 
                                 (unsigned int)((permille%10)*10) );
            last_permille = (permille / 10);
          }                                 
        }
        if ( run != RESULT_WORKING) /* finished this block */
        {
          if ( problem->RetrieveState(&tmp_work, NULL, 0) >= 0 )
          {
            u32 old_lo, frag_hi = 0, frag_lo = 0;
            switch( contestid ) 
            {
              case RC5:
              case DES:
              case CSC:
                frag_hi = tmp_work.crypto.keysdone.hi;
                frag_lo = tmp_work.crypto.keysdone.lo;
                break;
              case OGR:
                frag_hi = tmp_work.ogr.nodes.hi;
                frag_lo = tmp_work.ogr.nodes.lo;
                break;
            }        
            old_lo = keysdone_lo; 
            keysdone_lo += frag_lo;
            keysdone_hi += frag_hi;
            if ( keysdone_lo < old_lo || keysdone_lo < frag_lo )
              keysdone_hi++;
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

  /* --------------------------- */
  
  retvalue = -1; /* assume error */
  if (run >= 0) /* no errors, no ^C */
    retvalue = (long)__calc_rate(contestid, &contestwork, run, 
                         keysdone_hi, keysdone_lo, 
                         &totalruntime, contname, 
                         problem->coresel, problem->client_cpu,
                         (!(flags & TBENCHMARK_QUIET)) );

  delete problem;
  
  workunitsec = 0;
  switch (contestid)
  {
    case RC5:
    case DES:
    case CSC:
      workunitsec = 1 + (1<<28)/retvalue;
      break;

    case OGR:
      // NYI
      workunitsec = 0;
      break;
  };

  CliSetContestWorkUnitSpeed(contestid, workunitsec);

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

