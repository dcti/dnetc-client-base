// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: problem.cpp,v $
// Revision 1.69  1999/01/18 12:12:35  cramer
// - Added code for ncpu detection for linux/alpha
// - Corrected the alpha RC5 core handling (support "timeslice")
// - Changed the way selftest runs... it will not stop if a test fails,
//     but will terminate at the end of each contest selftest if any test
//     failed.  Interrupting the test is seen as the remaining tests
//     having failed (to be fixed later)
//
// Revision 1.68  1999/01/17 22:55:10  silby
// Change casts to make msvc happy.
//
// Revision 1.67  1999/01/17 21:38:52  cyp
// memblock for bruce ford's deseval-mmx is now passed from the problem object.
//
// Revision 1.65  1999/01/11 20:59:34  patrick
// updated to not raise an error if RC5ANSICORE is defined
//
// Revision 1.64  1999/01/11 05:45:10  pct
// Ultrix modifications for updated client.
//
// Revision 1.63  1999/01/08 02:59:29  michmarc
// Added support for Alpha/NT architecture.
//
// Revision 1.62  1999/01/06 09:54:29  chrisb
// fixes to the RISC OS timeslice stuff for DES - now runs about 2.5 times as fast
//
// Revision 1.61  1999/01/01 02:45:16  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.60  1998/12/28 21:37:54  cramer
// Misc. cleanups for the disappearing RC5CORECOPY junk and minor stuff to
// get a solaris client to build.
//
// Revision 1.59  1998/12/26 21:19:28  cyp
// Fixed condition where x86/mt/non-mmx would default to slice.
//
// Revision 1.58  1998/12/25 03:08:57  cyp
// x86 Bryd is runnable on upto 4 threads (threads 3 and 4 use the two
// non-optimal cores, ie pro cores on a p5 machine and vice versa).
// Made some non-core related stuff u64 clean.
//
// Revision 1.57  1998/12/22 15:58:24  jcmichot
// QNX changes.
//
// Revision 1.56  1998/12/19 04:30:23  chrisb
// fixed a broken comment which was giving errors
//
// Revision 1.55  1998/12/15 03:08:46  dicamillo
// Changed "whichcrunch" to "cputype" in PowerPC Run code.
//
// Revision 1.54  1998/12/14 12:48:59  cyp
// This is the final revision of problem.cpp/problem.h before the class goes
// 'u64 clean'. Please check/declare all core prototypes.
//
// Revision 1.53  1998/12/14 09:38:59  snake
// Re-integrated non-nasm x86 cores, cause nasm doesn't support all x86 cores.
// Sorry, no bye-bye to .cpp cores. Moved RC5X86_SRCS to NASM_RC5X86_SRCS and 
// corrected other targets.
//
// Revision 1.52  1998/12/14 05:16:10  dicamillo
// Mac OS updates to eliminate use of MULTITHREAD and have a singe client
// for MT and non-MT machines.
//
// Revision 1.51  1998/12/09 07:38:40  dicamillo
// Fixed missing // in log comment (MacCVS Pro bug!).
//
// Revision 1.50  1998/12/09 07:36:34  dicamillo
// Added support for MacOS client scheduling, and info needed
// by MacOS GUI.
//
// Revision 1.49  1998/12/07 15:21:23  chrisb
// more riscos/x86 changes
//
// Revision 1.48  1998/12/06 03:06:11  cyp
// Define STRESS_THREADS_AND_BUFFERS to configure problem.cpp for 'dummy
// crunching', ie problems are finished before they even start. You will be
// warned (at runtime) if you define it.
//
// Revision 1.47  1998/12/01 11:24:11  chrisb
// more riscos x86 changes
//
// Revision 1.46  1998/11/28 19:43:17  cyp
// Def'd out two LogScreen()s
//
// Revision 1.45  1998/11/25 09:23:36  chrisb
// various changes to support x86 coprocessor under RISC OS
//
// Revision 1.44  1998/11/25 06:05:48  dicamillo
// Use parenthesis when calling a function via a pointer.  Gcc in BeOS R4
// for Intel requires this.
//
// Revision 1.43  1998/11/16 16:44:07  remi
// Allow some targets to use deseval.cpp instead of Meggs' bitslicers.
//
// Revision 1.42  1998/11/14 14:12:05  cyp
// Fixed assignment of -1 to an unsigned variable.
//
// Revision 1.41  1998/11/14 13:56:15  cyp
// Fixed pipeline_count for x86 clients (DES cores were running with 4
// pipelines). Fixed unused parameter warning in LoadState(). Problem manager
// saves its probman_index in the Problem object (needed by chrisb's x86
// copro board code.)
//
// Revision 1.40  1998/11/12 22:58:31  remi
// Reworked a bit AIX ppc & power defines, based on Patrick Hildenbrand
// <patrick@de.ibm.com> advice.
//
// Revision 1.39  1998/11/10 09:18:13  silby
// Added alpha-linux target, should use axp-bmeyer core.
//
// Revision 1.38  1998/11/08 22:25:48  silby
// Fixed RC5_MMX pipeline count selection, was incorrect.
//
// Revision 1.37  1998/10/02 16:59:03  chrisb
// lots of fiddling in a vain attempt to get the NON_PREEMPTIVE_OS_PROFILING to be a bit sane under RISC OS
//
// Revision 1.36  1998/09/29 22:03:00  blast
// Fixed a bug I introduced with generic core usage, and removed
// a few old comments that weren't valid anymore (for 68k)
//
// Revision 1.35  1998/09/25 11:31:18  chrisb
// Added stuff to support 3 cores in the ARM clients.
//
// Revision 1.34  1998/09/23 22:05:20  blast
// Multi-core support added for m68k. Autodetection of cores added for 
// AmigaOS. (Manual selection possible of course). Two new 68k cores are 
// now added. rc5-000_030-jg.s and rc5-040_060-jg.s Both made by John Girvin.
//
// Revision 1.33  1998/08/24 04:43:26  cyruspatel
// timeslice is now rounded up to be multiple of PIPELINE_COUNT and even.
//
// Revision 1.32  1998/08/22 08:00:40  silby
// added in pipeline_count=2 "just in case" for x86
//
// Revision 1.31  1998/08/20 19:34:28  cyruspatel
// Removed that terrible PIPELINE_COUNT hack: Timeslice and pipeline count
// are now computed in Problem::LoadState(). Client::SelectCore() now saves
// core type to Client::cputype.
//
// Revision 1.30  1998/08/14 00:05:07  silby
// Changes for rc5 mmx core integration.
//
// Revision 1.29  1998/08/05 16:43:29  cberry
// ARM clients now define PIPELINE_COUNT=2, and RC5 cores return number of 
// keys checked, rather than number of keys left to check
//
// Revision 1.28  1998/08/02 16:18:27  cyruspatel
// Completed support for logging.
//
// Revision 1.27  1998/07/13 12:40:33  kbracey
// RISC OS update. Added -noquiet option.
//
// Revision 1.26  1998/07/13 03:31:52  cyruspatel
// Added 'const's or 'register's where the compiler was complaining about
// "declaration/type or an expression" ambiguities.
//
// Revision 1.25  1998/07/07 21:55:50  cyruspatel
// client.h has been split into client.h and baseincs.h 
//
// Revision 1.24  1998/07/06 09:21:26  jlawson
// added lint tags around cvs id's to suppress unused variable warnings.
//
// Revision 1.23  1998/06/17 02:14:47  blast
// Added code to test a new 68030 core which I got from an outside
// source ... Commented out of course ...
//
// Revision 1.22  1998/06/16 21:53:28  silby
// Added support for dual x86 DES cores (p5/ppro)
//
// Revision 1.21  1998/06/15 12:04:05  kbracey
// Lots of consts.
//
// Revision 1.20  1998/06/15 06:18:37  dicamillo
// Updates for BeOS
//
// Revision 1.19  1998/06/15 00:12:24  skand
// fix id marker so it won't interfere when another .cpp file is 
// #included here
//
// Revision 1.18  1998/06/14 10:13:43  skand
// use #if 0 (or 1) to turn on some debugging info, rather than // on each line
//
// Revision 1.17  1998/06/14 08:26:54  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.16  1998/06/14 08:13:04  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 1.15  1998/06/14 00:06:07  remi
// Added $Log.
//

#if (!defined(lint) && defined(__showids__))
const char *problem_cpp(void) {
return "@(#)$Id: problem.cpp,v 1.69 1999/01/18 12:12:35 cramer Exp $"; }
#endif

#include "cputypes.h"
#include "baseincs.h"
#include "problem.h"
#include "network.h" // for timeval and htonl/ntohl
#include "clitime.h" //for CliTimer() which gets a timeval of the current time
#include "logstuff.h" //LogScreen()
#include "cpucheck.h"
#include "console.h"
#include "triggers.h"
#if (CLIENT_OS == OS_RISCOS)
#include "../platforms/riscos/riscos_x86.h"
extern "C" void riscos_upcall_6(void);
extern void CliSignalHandler(int);
#endif

/* ------------------------------------------------------------- */

//#define STRESS_THREADS_AND_BUFFERS /* !be careful with this! */

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif

/* ------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_X86)
  extern "C" u32 rc5_unit_func_486( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_p5( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_p6( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_6x86( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_k5( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_k6( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_p5_mmx( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_486_smc( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern u32 p1des_unit_func_p5( RC5UnitWork * rc5unitwork, u32 nbbits );
  extern u32 p1des_unit_func_pro( RC5UnitWork * rc5unitwork, u32 nbbits );
  extern u32 p2des_unit_func_p5( RC5UnitWork * rc5unitwork, u32 nbbits );
  extern u32 p2des_unit_func_pro( RC5UnitWork * rc5unitwork, u32 nbbits );
  extern u32 des_unit_func_mmx( RC5UnitWork * rc5unitwork, u32 nbbits, char *coremem );
  extern u32 des_unit_func_slice( RC5UnitWork * rc5unitwork, u32 nbbits );
  #if (PIPELINE_COUNT != 2)
  #error "Expecting PIPELINE_COUNT=2"
  #endif
#elif (CLIENT_CPU == CPU_POWERPC)
  #if (CLIENT_OS == OS_WIN32)   // NT PPC doesn't have good assembly
  extern u32 rc5_unit_func( RC5UnitWork * rc5unitwork ); //rc5ansi2-rg.cpp
  #else
  extern "C" int crunch_allitnil( RC5UnitWork *work, unsigned long iterations);
  extern "C" int crunch_lintilla( RC5UnitWork *work, unsigned long iterations);
  #endif
  extern u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
  #if (PIPELINE_COUNT != 1)
  #error "Expecting PIPELINE_COUNT=1"
  #endif
#elif (CLIENT_CPU == CPU_68K)
  extern "C" __asm u32 rc5_unit_func_000_030( register __a0 RC5UnitWork * work, register __d0 unsigned long timeslice );
  extern "C" __asm u32 rc5_unit_func_040_060( register __a0 RC5UnitWork * work, register __d0 unsigned long timeslice );
  extern u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
  #if (PIPELINE_COUNT != 2)
  #error "Expecting PIPELINE_COUNT=2"
  #endif
#elif (CLIENT_CPU == CPU_ARM) 
  extern "C" u32 rc5_unit_func_arm_1( RC5UnitWork * rc5unitwork , unsigned long t);
  extern "C" u32 rc5_unit_func_arm_2( RC5UnitWork * rc5unitwork , unsigned long t);
  extern "C" u32 rc5_unit_func_arm_3( RC5UnitWork * rc5unitwork , unsigned long t);
  extern "C" u32 des_unit_func_arm( RC5UnitWork * rc5unitwork , unsigned long t);
  extern "C" u32 des_unit_func_strongarm( RC5UnitWork * rc5unitwork , unsigned long t);
  #if (PIPELINE_COUNT != 2)
  #error "Expecting PIPELINE_COUNT=2"
  #endif
#elif (CLIENT_CPU == CPU_PA_RISC)
  extern u32 rc5_unit_func( RC5UnitWork * rc5unitwork );
  extern u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
  #error Please verify these core prototypes
#elif (CLIENT_CPU == CPU_SPARC)
  #if (ULTRA_CRUNCH == 1)
  extern "C++" u32 crunch( register RC5UnitWork * rc5unitwork, u32 timeslice);
  extern "C++" u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
  #else
  extern "C++" u32 rc5_unit_func( RC5UnitWork * rc5unitwork );
  extern "C++" u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
  #endif
  // CRAMER // #error Please verify these core prototypes
#elif (CLIENT_CPU == CPU_MIPS)
  #if (CLIENT_OS != OS_ULTRIX)
    #if (MIPS_CRUNCH == 1)
    extern "C" unsigned long crunch( register RC5UnitWork * rc5unitwork, u32 timeslice);
    extern u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
    #else
    extern u32 rc5_unit_func( RC5UnitWork * rc5unitwork );
    extern u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
    #endif
    #error Please verify these core prototypes
  #else /* OS_ULTRIX */
    extern u32 rc5_unit_func( RC5UnitWork * rc5unitwork );
    extern u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
  #endif
#elif (CLIENT_CPU == CPU_ALPHA)
  #if (CLIENT_OS == OS_WIN32)
     extern "C" u32 rc5_unit_func ( RC5UnitWork *rc5unitwork, u32 timeslice);
     extern u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
     #if (PIPELINE_COUNT != 2)
     #error "Expecting PIPELINE_COUNT=2"
     #endif
  #else
     extern u32 rc5_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
     extern u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
     // CRAMER // #error Please verify these core prototypes
  #endif
#else
  extern u32 rc5_unit_func( RC5UnitWork * rc5unitwork );
  extern u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice );
  #ifndef RC5ANSICORE   
    // Hey, this error makes no sense for ANSI cores, so why not #ifdef (patrick)
    #error Please declare/prototype cores by CLIENT_CPU if you are not using ansi*.cpp cores.
  #endif
#endif

/* ------------------------------------------------------------------- */

Problem::Problem(long _threadindex /* defaults to -1L */)
{
  threadindex_is_valid = (_threadindex!=-1L);
  threadindex = ((threadindex_is_valid)?((unsigned int)_threadindex):(0));

//LogScreen("Problem created. threadindex=%u\n",threadindex);

  initialized = 0;
  finished = 0;
  started = 0;

#ifdef STRESS_THREADS_AND_BUFFERS 
  static int runlevel = 0;
  if (runlevel != -12345)
    {
    if ((++runlevel) != 1)
      {
      --runlevel;
      return;
      }
    RaisePauseRequestTrigger();
    LogScreen("Warning! STRESS_THREADS_AND_BUFFERS is defined.\n"
              "Are you sure that the client is pointing at\n"
              "a test proxy? If so, type 'yes': ");
    char getyes[10];
    ConInStr(getyes,4,0);
    ClearPauseRequestTrigger();
    if (strcmpi(getyes,"yes") != 0)
      {
      runlevel = +12345;
      RaiseExitRequestTrigger();
      return;
      }
    runlevel = -12345;
    }
#endif    
}

/* ------------------------------------------------------------------- */

Problem::~Problem()
{
  started = 0; // nothing to do. - suppress compiler warning
#if (CLIENT_OS == OS_RISCOS)
  if (GetProblemIndexFromPointer(this) == 1)
    {
    _kernel_swi_regs r;
    r.r[0] = 0;
    _kernel_swi(RC5PC_RetriveBlock,&r,&r);
    _kernel_swi(RC5PC_Off,&r,&r);
    
    }
#endif
}

/* ------------------------------------------------------------------- */

int Problem::LoadState( ContestWork * work, unsigned int _contest, 
                              u32 _timeslice, int _cputype )
{
  contest = _contest;
  cputype = _cputype;

  if (contest != 0 && contest != 1)
    return -1;

  pipeline_count = PIPELINE_COUNT;
  
#if (CLIENT_CPU == CPU_ARM)
  switch(cputype)
    {
    case 0: rc5_unit_func = rc5_unit_func_arm_1;
            des_unit_func = des_unit_func_arm;
            pipeline_count = 1;
            break;
    default:
    case 1: rc5_unit_func = rc5_unit_func_arm_3;
            des_unit_func = des_unit_func_strongarm;
            pipeline_count = 3;
            break;
    case 2: rc5_unit_func = rc5_unit_func_arm_2;
            des_unit_func = des_unit_func_strongarm;
            pipeline_count = 2;
            break;
    case 3: rc5_unit_func = rc5_unit_func_arm_3;
            des_unit_func = des_unit_func_arm;
            pipeline_count = 3;
            break;
    }
#endif

#if (CLIENT_CPU == CPU_68K)
  if (cputype < 0 || cputype > 5) /* just to be safe */
    cputype = 0;
  if (cputype == 4 || cputype == 5 ) // there is no 68050, so type5=060
    rc5_unit_func = rc5_unit_func_040_060;
  else //if (cputype == 0 || cputype == 1 || cputype == 2 || cputype == 3)
    rc5_unit_func = rc5_unit_func_000_030;
#endif    

#if (CLIENT_CPU == CPU_X86)
  static int detectedtype = -1;
  if (detectedtype == -1)
    detectedtype = GetProcessorType(1 /* quietly */);

  if (cputype < 0 || cputype > 5)
    cputype = 0;

  if (contest == 1)
    {
    #if defined(MMX_BITSLICER) 
    #if defined(MMX_RC5)  /* someone added this. why? - cyp */
    if ((detectedtype & 0x100) != 0)
      {
      unit_func = (u32 (*)(RC5UnitWork *,u32))des_unit_func_mmx;
      }
    else
    #endif
    #endif
      {
      //
      // p1* and p2* are effectively the same core (as called from
      // des-x86.cpp) if client was not compiled for mt - cyp
      //
      if (cputype == 2 || cputype == 3 || cputype == 5)
        unit_func = p1des_unit_func_pro;
      else
        unit_func = p1des_unit_func_p5;
      #if defined(CLIENT_SUPPORTS_SMP) 
      if (threadindex == 1)
        {
        if (unit_func == p1des_unit_func_p5)
          unit_func = p2des_unit_func_p5;
        else 
          unit_func = p2des_unit_func_pro;
        }
      else if (threadindex == 2)  
        {
        if (unit_func == p1des_unit_func_p5) // use the other unused cores.
          unit_func = p1des_unit_func_pro;   // non-optimal but ...
        else                                 // ... still better than slice
          unit_func = p1des_unit_func_p5;
        }
      else if (threadindex == 3)
        {
        if (unit_func == p1des_unit_func_p5) // use the other unused cores.
          unit_func = p2des_unit_func_pro;   // non-optimal but ...
        else                                 // ... still better than slice
          unit_func = p2des_unit_func_p5;
        }
      else if (threadindex > 4)              // fall back to slice if 
        {                                    // running with > 4 processors
        unit_func = des_unit_func_slice;
        }
      #endif
#if 0      
if (unit_func == p1des_unit_func_p5)
  LogScreen("Using bryd 51. (0/2) threadindex=%u\n",threadindex);
else if (unit_func == p2des_unit_func_p5)
  LogScreen("Using bryd 52. (1/3) threadindex=%u\n",threadindex);
else if (unit_func == p1des_unit_func_pro)
  LogScreen("Using bryd 61. (2/0) threadindex=%u\n",threadindex);
else if (unit_func == p2des_unit_func_pro)
  LogScreen("Using bryd 62. (3/1) threadindex=%u\n",threadindex);
else
  LogScreen("Using slice. threadindex=%u\n",threadindex);
LogScreen("Press any key to continue..." );  
ConInKey(-1);
#endif
      }
    }
  else //if (contest == 0) 
    {
    if (cputype == 1)   // Intel 386/486
      {
      #if defined(SMC) 
      if (threadindex == 0)
        unit_func =  rc5_unit_func_486_smc;
      else
      #endif
        unit_func = rc5_unit_func_486;
      }
    else if (cputype == 2) // Ppro/PII
      unit_func = rc5_unit_func_p6;
    else if (cputype == 3) // 6x86(mx)
      unit_func = rc5_unit_func_6x86;
    else if (cputype == 4) // K5
      unit_func = rc5_unit_func_k5;
    else if (cputype == 5) // K6/K6-2
      unit_func = rc5_unit_func_k6;
    else // Pentium (0/6) + others
      {
      unit_func = rc5_unit_func_p5;
      #if defined(MMX_RC5)
      if (detectedtype == 0x106)  /* Pentium MMX only! */
        {
        unit_func = rc5_unit_func_p5_mmx;
        pipeline_count = 4; // RC5 MMX core is 4 pipelines
        }
      #endif
      cputype = 0;
      }
    }
#endif    


  //----------------------------------------------------------------

  // copy over the state information
  contestwork.key.hi = ntohl( work->key.hi );
  contestwork.key.lo = ntohl( work->key.lo );
  contestwork.iv.hi = ntohl( work->iv.hi );
  contestwork.iv.lo = ntohl( work->iv.lo );
  contestwork.plain.hi = ntohl( work->plain.hi );
  contestwork.plain.lo = ntohl( work->plain.lo );
  contestwork.cypher.hi = ntohl( work->cypher.hi );
  contestwork.cypher.lo = ntohl( work->cypher.lo );
  contestwork.keysdone.hi = ntohl( work->keysdone.hi );
  contestwork.keysdone.lo = ntohl( work->keysdone.lo );
  contestwork.iterations.hi = ntohl( work->iterations.hi );
  contestwork.iterations.lo = ntohl( work->iterations.lo );

  #if 0
  // determine the starting key number
  // (note: doesn't account for carryover to hi or high end of keysdone)
  u64 key;
  key.hi = contestwork.key.hi;
  key.lo = contestwork.key.lo + contestwork.keysdone.lo;
  #endif

  //determine starting key number. accounts for carryover & highend of keysdone
  u64 key;
  key.hi = contestwork.key.hi + contestwork.keysdone.hi + 
     ((((contestwork.key.lo & 0xffff) + (contestwork.keysdone.lo & 0xffff)) + 
       ((contestwork.key.lo >> 16) + (contestwork.keysdone.lo >> 16))) >> 16);
  key.lo = contestwork.key.lo + contestwork.keysdone.lo;

  // set up the unitwork structure
  rc5unitwork.plain.hi = contestwork.plain.hi ^ contestwork.iv.hi;
  rc5unitwork.plain.lo = contestwork.plain.lo ^ contestwork.iv.lo;
  rc5unitwork.cypher.hi = contestwork.cypher.hi;
  rc5unitwork.cypher.lo = contestwork.cypher.lo;

  if (contest == 0)
    {
    rc5unitwork.L0.lo = ((key.hi >> 24) & 0x000000FFL) |
        ((key.hi >>  8) & 0x0000FF00L) |
        ((key.hi <<  8) & 0x00FF0000L) |
        ((key.hi << 24) & 0xFF000000L);
    rc5unitwork.L0.hi = ((key.lo >> 24) & 0x000000FFL) |
        ((key.lo >>  8) & 0x0000FF00L) |
        ((key.lo <<  8) & 0x00FF0000L) |
        ((key.lo << 24) & 0xFF000000L);
    } 
  else 
    {
    rc5unitwork.L0.lo = key.lo;
    rc5unitwork.L0.hi = key.hi;
    }

  // set up the current result state
  rc5result.key.hi = contestwork.key.hi;
  rc5result.key.lo = contestwork.key.lo;
  rc5result.keysdone.hi = contestwork.keysdone.hi;
  rc5result.keysdone.lo = contestwork.keysdone.lo;
  rc5result.iterations.hi = contestwork.iterations.hi;
  rc5result.iterations.lo = contestwork.iterations.lo;
  rc5result.result = RESULT_WORKING;

  //---------------------------------------------------------------
  
  tslice = _timeslice;
  AlignTimeslice(); /* requires a loaded ContestWork */

  //--------------------------------------------------------------- 

  #if 0
  startpercent = (u32) ( (double) 100000.0 *
     ( (double) (contestwork.keysdone.lo) /
       (double) (contestwork.iterations.lo) ) );
  #endif
  
  startpercent = (u32)( ((double)(100000.0)) *
        (((((double)(contestwork.keysdone.hi))*((double)(4294967296.0)))+
                                 ((double)(contestwork.keysdone.lo))) /
        ((((double)(contestwork.iterations.hi))*((double)(4294967296.0)))+
                                 ((double)(contestwork.iterations.lo)))) );
  percent=0;
  restart = ( contestwork.keysdone.lo!=0 || contestwork.keysdone.hi!=0 );

  initialized = 1;
  finished = 0;
  started = 0;


  //-------------------------------------------------------------------

#if (CLIENT_OS == OS_RISCOS)
  if (threadindex == 1 /*x86 thread*/)
    {
    RC5PCstruct rc5pc;
    _kernel_swi_regs r;

    rc5pc.key.hi = contestwork.key.hi;
    rc5pc.key.lo = contestwork.key.lo;
    rc5pc.iv.hi = contestwork.iv.hi;
    rc5pc.iv.lo = contestwork.iv.lo;
    rc5pc.plain.hi = contestwork.plain.hi;
    rc5pc.plain.lo = contestwork.plain.lo;
    rc5pc.cypher.hi = contestwork.cypher.hi;
    rc5pc.cypher.lo = contestwork.cypher.lo;
    rc5pc.keysdone.hi = contestwork.keysdone.hi;
    rc5pc.keysdone.lo = contestwork.keysdone.lo;
    rc5pc.iterations.hi = contestwork.iterations.hi;
    rc5pc.iterations.lo = contestwork.iterations.lo;
    rc5pc.timeslice = tslice;

    _kernel_swi(RC5PC_On,&r,&r);
    r.r[1] = (int)&rc5pc;
    _kernel_swi(RC5PC_AddBlock,&r,&r);
    if (r.r[0] == -1)
      {
      LogScreen("Failed to add block to x86 cruncher\n");
      }
    }
#endif

  return( 0 );
}

/* ------------------------------------------------------------------- */

s32 Problem::GetResult( RC5Result * result )
{
  if ( !initialized )
    return ( -1 );

  // note that all but result go back to network byte order at this point.
  result->key.hi = htonl( rc5result.key.hi );
  result->key.lo = htonl( rc5result.key.lo );
  result->keysdone.hi = htonl( rc5result.keysdone.hi );
  result->keysdone.lo = htonl( rc5result.keysdone.lo );
  result->iterations.hi = htonl( rc5result.iterations.hi );
  result->iterations.lo = htonl( rc5result.iterations.lo );
  result->result = rc5result.result;

  return ( contest );
}

/* ------------------------------------------------------------------- */

s32 Problem::RetrieveState( ContestWork * work , s32 setflags )
{
  // store back the state information
  work->key.hi = htonl( contestwork.key.hi );
  work->key.lo = htonl( contestwork.key.lo );
  work->iv.hi = htonl( contestwork.iv.hi );
  work->iv.lo = htonl( contestwork.iv.lo );
  work->plain.hi = htonl( contestwork.plain.hi );
  work->plain.lo = htonl( contestwork.plain.lo );
  work->cypher.hi = htonl( contestwork.cypher.hi );
  work->cypher.lo = htonl( contestwork.cypher.lo );
  work->keysdone.hi = htonl( contestwork.keysdone.hi );
  work->keysdone.lo = htonl( contestwork.keysdone.lo );
  work->iterations.hi = htonl( contestwork.iterations.hi );
  work->iterations.lo = htonl( contestwork.iterations.lo );

  if (setflags) 
    {
    initialized = 0;
    finished = 0;
    }
  return( contest );
}

/* ------------------------------------------------------------- */

u32 Problem::AlignTimeslice(void) // align the timeslice to an even 
{                        // multiple of pipeline_count and 2 
  u32 alignfact = pipeline_count + (pipeline_count & 1);
  u32 timeslice = ((tslice + (alignfact - 1)) & ~(alignfact - 1));

  // don't allow a too large of a timeslice be used
  // (technically not necessary, but may save some wasted time)
  if (contestwork.keysdone.hi == contestwork.iterations.hi)
    {
    u32 todo = contestwork.iterations.lo-contestwork.keysdone.lo;
    if (todo < timeslice)
      {
      timeslice = todo;
      timeslice = ((timeslice + (alignfact - 1)) & ~(alignfact - 1));
      }
    }

  #if 0 /* ---- old code ------ */
  //if (timeslice <= pipeline_count)
  //  timeslice = pipeline_count;
  //else
  // timeslice=((timeslice+(pipeline_count-1)) & ~(pipeline_count-1));
  //
  // old code - doesn't account for high end or carry over
  //
  //if ( ( contestwork.keysdone.lo + timeslice ) > contestwork.iterations.lo)
  //   timeslice = (( contestwork.iterations.lo - contestwork.keysdone.lo +
  //           pipeline_count - 1 ) / pipeline_count+1);
  //timeslice *= pipeline_count; //was implied
  #endif    

#if 0
LogScreen("AlignTimeslice(): effective timeslice: %lu (0x%lx),\n"
          "suggested timeslice: %lu (0x%lx)\n"
          "pipeline_count = %lu, timeslice%%pipeline_count = %lu\n", 
          (unsigned long)timeslice, (unsigned long)timeslice,
          (unsigned long)tslice, (unsigned long)tslice,
          pipeline_count, timeslice%pipeline_count );
#endif

  return timeslice;
}

/* ------------------------------------------------------------- */

s32 Problem::Run( u32 /*unused*/ )
{
  u32 timeslice;
  if ( !initialized )
    return ( -1 );

  if ( finished )
    return ( 1 );

  if (!started)
    {
    struct timeval stop;
    CliTimer(&stop);
    timehi = stop.tv_sec;
    timelo = stop.tv_usec;
    started=1;

#ifdef STRESS_THREADS_AND_BUFFERS 
    contestwork.keysdone.hi = contestwork.iterations.hi;
    contestwork.keysdone.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_NOTHING;
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;

    finished = 1;
    return 1;
#endif    
    }

  // align the timeslice to an even multiple of pipeline_count and 2
  // after checking for an excessive timeslice (>(iter-keysdone)) 
  timeslice = (AlignTimeslice() / pipeline_count);
  
#if (CLIENT_CPU == CPU_POWERPC)
  {
  unsigned long kiter = 0;
  if (contest == 0) 
    {
#if (CLIENT_OS == OS_WIN32)
    kiter = rc5_unit_func( &rc5unitwork );
#else    
#if ((CLIENT_OS != OS_BEOS) || (CLIENT_OS != OS_AMIGAOS))
    if (cputype == 0)
      kiter = crunch_allitnil( &rc5unitwork, timeslice );
    else
#endif
      {
      do{
        kiter += crunch_lintilla( &rc5unitwork, timeslice - kiter );
        if (kiter < timeslice) 
          {
          if (crunch_allitnil( &rc5unitwork, 1 ) == 0) 
            {
            break;
            }
          kiter++;
          }
        } while (kiter < timeslice);
      }
#endif
    }
  else
    {
    // protect the innocent
    timeslice *= pipeline_count;

    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / pipeline_count;
    kiter = des_unit_func ( &rc5unitwork, nbits );
    }

  // Mac code yields here because it needs to know the contest
  // DES code yields in the core itself because it can run for a
  // long time (with respect to good user response time) before returning
  #if (CLIENT_OS == OS_MACOS)
    if ((contest == 0) && (MP_active == 0)) {
      YieldToMain(1);
    }
  #endif


  contestwork.keysdone.lo += kiter;

  if (kiter < timeslice)
    {
    // found it?
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_FOUND;
    finished = 1;
    return( 1 );
    }
  }
#elif (CLIENT_CPU == CPU_X86)
  {
  unsigned long kiter;
  if (contest == 0)
    {
    kiter = (*unit_func)( &rc5unitwork, timeslice );
    }
  else
    {
    // protect the innocent
    timeslice *= pipeline_count;

    u32 min_bits = 8;  /* bryd and kwan cores only need a min of 256 */
    u32 max_bits = 24; /* these are the defaults if !MEGGS && !DES_ULTRA */

    #if defined(MMX_BITSLICER)
    #if defined(MMX_RC5)
    if (((u32 (*)(RC5UnitWork *,u32, char *))(unit_func) == des_unit_func_mmx))
      {
      #if defined(BITSLICER_WITH_LESS_BITS)
      min_bits = 16;
      #else
      min_bits = 20;
      #endif
      max_bits = min_bits; /* meggs driver has equal MIN and MAX */
      }
    #endif
    #endif

    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;
//LogScreen("x86: nbits %lu, min %lu, max %lu\n", nbits, min_bits, max_bits );

    if (nbits < min_bits) nbits = min_bits;
    else if (nbits > max_bits) nbits = max_bits;
    timeslice = (1ul << nbits) / pipeline_count;

    #if defined(MMX_BITSLICER)
    #if defined(MMX_RC5)
    if (((u32 (*)(RC5UnitWork *,u32, char *))(unit_func) == des_unit_func_mmx))
      kiter = des_unit_func_mmx( &rc5unitwork, nbits, core_membuffer );
    else
    #endif
    #endif
    kiter = (*unit_func)( &rc5unitwork, nbits );
    }

  contestwork.keysdone.lo += kiter;

  if ( kiter < timeslice * pipeline_count )
    {
    // found it?
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_FOUND;
    finished = 1;
    return( 1 );
    }
  else if ( kiter != timeslice * pipeline_count )
    {
    LogScreen("kiter wrong %ld %d\n", kiter, (int)(timeslice*pipeline_count));
    }
  }
#elif (CLIENT_CPU == CPU_SPARC) && (ULTRA_CRUNCH == 1)
  {
  unsigned long kiter;
  if (contest == 0) 
    {
    kiter = crunch( &rc5unitwork, timeslice );
    } 
  else 
    {
    // protect the innocent
    timeslice *= pipeline_count;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / pipeline_count;
    kiter = des_unit_func ( &rc5unitwork, nbits );
    }
  contestwork.keysdone.lo += kiter;
  if (kiter < ( timeslice * pipeline_count ) )
    {
    // found it?
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_FOUND;
    finished = 1;
    return( 1 );
    }
  else if (kiter != ( timeslice * pipeline_count ) )
    {
    LogScreen("kiter wrong %ld %d\n", (long) kiter, (int) (timeslice*pipeline_count));
    }
  }
#elif ((CLIENT_CPU == CPU_MIPS) && (MIPS_CRUNCH == 1))
  {
  unsigned long kiter;
  if (contest == 0) 
    {
    kiter = crunch( &rc5unitwork, timeslice );
    } 
  else 
    {
    // protect the innocent
    timeslice *= pipeline_count;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / pipeline_count;
    kiter = des_unit_func ( &rc5unitwork, nbits );
    }
  contestwork.keysdone.lo += kiter;
  if (kiter < ( timeslice * pipeline_count ) )
    {
    // found it?
    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
    rc5result.iterations.hi = contestwork.iterations.hi;
    rc5result.iterations.lo = contestwork.iterations.lo;
    rc5result.result = RESULT_FOUND;
    finished = 1;
    return( 1 );
    }
  else if (kiter != (timeslice * pipeline_count))
    {
    LogScreen("kiter wrong %ld %d\n", kiter, timeslice*pipeline_count);
    }
  }
#elif (CLIENT_CPU == CPU_ARM)
  {
  unsigned long kiter;
#if (CLIENT_OS == OS_RISCOS)
  if (_kernel_escape_seen())
    {
    CliSignalHandler(SIGINT);
    }
#endif
//  timeslice *= pipeline_count;
//  done in the cores.

  if (contest == 0)
    {
#if (CLIENT_OS == OS_RISCOS)
    if (threadindex == 0)
      {
#endif
#ifdef DEBUG
      static u32 ts;
      if ((ts < timeslice-500) || (ts > timeslice+500))
        {
        ts = timeslice;
        printf("timeslice = %d\n",timeslice);
        }
#endif
//    printf("ARM thread running, ts=%08lx\n",timeslice);
      if ((rc5_unit_func == rc5_unit_func_arm_2)&&( rc5unitwork.L0.hi&(1<<24)))
        {
        rc5unitwork.L0.hi -= 1<<24;
        if (contestwork.keysdone.lo & 1)
          {
          contestwork.keysdone.lo--;
          }
        else
          {
          LogScreen("Something really bad has happened - the number of keys looks wrong.\n");
          for(;;); // probably a bit bogus, but hey.
          }
        }
      /*
      Now returns number of keys processed!
      (Since 5/8/1998, SA core 1.5, ARM core 1.6).
      */
      kiter = rc5_unit_func(&rc5unitwork, timeslice);
      contestwork.keysdone.lo += kiter;
         
//    printf("kiter is %d\n",kiter);
      if (kiter != (timeslice*pipeline_count))
        {
        // found it?
        rc5result.key.hi = contestwork.key.hi;
        rc5result.key.lo = contestwork.key.lo;
        rc5result.keysdone.hi = contestwork.keysdone.hi;
        rc5result.keysdone.lo = contestwork.keysdone.lo;
        rc5result.iterations.hi = contestwork.iterations.hi;
        rc5result.iterations.lo = contestwork.iterations.lo;
        rc5result.result = RESULT_FOUND;
        finished = 1;
        return( 1 );
        }
#if (CLIENT_OS == OS_RISCOS)
      }
    else // threadindex == 1
      {
          /*
            This is the RISC OS specific x86 2nd thread magic.
          */
          _kernel_swi_regs r;
          volatile RC5PCstruct *rc5pcr;
          _kernel_swi(RC5PC_BufferStatus,&r,&r);

          /*
            contestwork.keysdone.lo is 0 for a completed block,
            so take care when setting it.
           */
          rc5pcr = (volatile RC5PCstruct *)r.r[1];

          
          if (r.r[2]==1)
          {
              /*
                block finished
              */
              r.r[0] = 0;
              _kernel_swi(RC5PC_RetriveBlock,&r,&r);
              rc5pcr = (volatile RC5PCstruct *)r.r[1];
              
              if (rc5pcr->result == RESULT_FOUND)
              {
                  contestwork.keysdone.lo = rc5pcr->keysdone.lo;
//printf("x86:FF Keysdone %08lx\n",contestwork.keysdone.lo);

                  rc5result.key.hi = contestwork.key.hi;
                  rc5result.key.lo = contestwork.key.lo;
                  rc5result.keysdone.hi = contestwork.keysdone.hi;
                  rc5result.keysdone.lo = contestwork.keysdone.lo;
                  rc5result.iterations.hi = contestwork.iterations.hi;
                  rc5result.iterations.lo = contestwork.iterations.lo;
                  rc5result.result = RESULT_FOUND;
                  finished = 1;
                  return( 1 );
              }
              else
              {
                  contestwork.keysdone.lo = contestwork.iterations.lo;
//printf("x86:FN Keysdone %08lx\n",contestwork.keysdone.lo);
              }

          }
          else
          {
              contestwork.keysdone.lo = rc5pcr->keysdone.lo;
//printf("x86:NF Keysdone %08lx\n",contestwork.keysdone.lo);
          }
      }
#endif
    }
  else
    {
    // protect the innocent
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;
//    static unsigned long arse;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits);// / pipeline_count;
    kiter = des_unit_func ( &rc5unitwork, nbits );
/*
if (arse != timeslice)
 {
     arse = timeslice;
printf("DES: timeslice is %d, nbits is %d\n",timeslice,nbits);
printf("DES: kiter is %d\n",kiter);
 }
*/
    contestwork.keysdone.lo += kiter;
    if (kiter < timeslice)
      {
      // found it?
      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;
      finished = 1;
      return( 1 );
      }
    }
  }
#elif (CLIENT_CPU == CPU_68K)
  unsigned long kiter = 0;
  if (contest == 0) 
    {
    kiter = rc5_unit_func( &rc5unitwork, timeslice );
    if ( kiter < timeslice*pipeline_count )
      {
      // found it?
      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo + kiter;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;
      finished = 1;
      return( 1 );
      }
    // increment the count of keys done
    // note: doesn't account for carry
    contestwork.keysdone.lo += ((pipeline_count*timeslice) + pipeline_count);
    }
  else
    {
    timeslice *= pipeline_count;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / pipeline_count;
    kiter = des_unit_func ( &rc5unitwork, nbits );
    contestwork.keysdone.lo += kiter;
    if (kiter < ( timeslice * pipeline_count ) )
      {
      // found it?
      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;
      finished = 1;
      return( 1 );
      }
    else if (kiter != (timeslice * pipeline_count))
      {
      LogScreen("kiter wrong %ld %ld\n",
               (long) kiter, (long)(timeslice*pipeline_count));
      }
    }
#elif (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32)
  if (contest == 0) {
    u32 result = rc5_unit_func( &rc5unitwork, timeslice);
    if ( result )
    {
      // found it?
      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo + timeslice*pipeline_count - result;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;
      finished = 1;
      return( 1 );
    }

    contestwork.keysdone.lo += timeslice*pipeline_count;
    if (contestwork.keysdone.lo < timeslice*pipeline_count)
       contestwork.keysdone.hi++;
  }
  else  /* DES portion taken from the ANSI routines below */
  {
    unsigned long kiter = 0;

    timeslice *= 1;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits);
    kiter = des_unit_func ( &rc5unitwork, nbits );
    contestwork.keysdone.lo += kiter;
    if (kiter < timeslice )
    {
      // found it?
      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;
      finished = 1;
      return( 1 );
    }
    else if (kiter != (timeslice))
    {
        LogScreen("kiter wrong %ld %ld\n",
               (long) kiter, (long)(timeslice));
    }
  }
#elif (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_LINUX)
  if (contest == 0) {
//#define DEBUG_ALPHA

#ifdef DEBUG_ALPHA
printf("BEGIN: timeslice[%X] key[%X/%X] keysdone[%X/%X] iters[%X/%X]\n",
	timeslice,
	rc5result.key.hi, rc5result.key.lo,
	rc5result.keysdone.hi, rc5result.keysdone.lo,
	rc5result.iterations.hi, rc5result.iterations.lo);
#endif

    u32 result = rc5_unit_func( &rc5unitwork, timeslice);
    if ( result && (result != (timeslice*PIPELINE_COUNT)))
    {
#ifdef DEBUG_ALPHA
printf("RESULT1: result[%X] key[%X/%X] keysdone[%X/%X] iters[%X/%X]\n",
	result,
	rc5result.key.hi, rc5result.key.lo,
	rc5result.keysdone.hi, rc5result.keysdone.lo,
	rc5result.iterations.hi, rc5result.iterations.lo);
#endif

      // found it?
      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo + result;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;

#ifdef DEBUG_ALPHA
printf("RESULT2: result[%X] key[%X/%X] keysdone[%X/%X] iters[%X/%X]\n",
	result,
	rc5result.key.hi, rc5result.key.lo,
	rc5result.keysdone.hi, rc5result.keysdone.lo,
	rc5result.iterations.hi, rc5result.iterations.lo);
#endif

      finished = 1;
      return( 1 );
    }

    contestwork.keysdone.lo += timeslice*PIPELINE_COUNT;
    if (contestwork.keysdone.lo < timeslice*PIPELINE_COUNT)
      contestwork.keysdone.hi++;

    rc5result.key.hi = contestwork.key.hi;
    rc5result.key.lo = contestwork.key.lo;
    rc5result.keysdone.hi = contestwork.keysdone.hi;
    rc5result.keysdone.lo = contestwork.keysdone.lo;
  }
  else  /* DES portion taken from the ANSI routines below */
  {
    unsigned long kiter = 0;

    timeslice *= 1;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits);
    kiter = des_unit_func ( &rc5unitwork, nbits );
    contestwork.keysdone.lo += kiter;
    if (kiter < timeslice )
    {
      // found it?
      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;
      finished = 1;
      return( 1 );
    }
    else if (kiter != (timeslice))
    {
        LogScreen("kiter wrong %ld %ld\n",
               (long) kiter, (long)(timeslice));
    }
  }
#else
  unsigned long kiter = 0;
  if (contest == 0) 
    {
    while ( timeslice-- ) // timeslice ignores the number of pipelines
      {
      u32 result = rc5_unit_func( &rc5unitwork );
      if ( result )
        {
        // found it?
        rc5result.key.hi = contestwork.key.hi;
        rc5result.key.lo = contestwork.key.lo;
        rc5result.keysdone.hi = contestwork.keysdone.hi;
        rc5result.keysdone.lo = contestwork.keysdone.lo + result - 1;
        rc5result.iterations.hi = contestwork.iterations.hi;
        rc5result.iterations.lo = contestwork.iterations.lo;
        rc5result.result = RESULT_FOUND;
        finished = 1;
        return( 1 );
        }
      else
        {
        // "mangle-increment" the key number by the number of pipelines
        rc5unitwork.L0.hi = (rc5unitwork.L0.hi + (pipeline_count << 24)) & 0xFFFFFFFF;
        if (!(rc5unitwork.L0.hi & 0xFF000000)) 
          {
          rc5unitwork.L0.hi = (rc5unitwork.L0.hi + 0x00010000) & 0x00FFFFFF;
          if (!(rc5unitwork.L0.hi & 0x00FF0000)) 
            {
            rc5unitwork.L0.hi = (rc5unitwork.L0.hi + 0x00000100) & 0x0000FFFF;
            if (!(rc5unitwork.L0.hi & 0x0000FF00)) 
              {
              rc5unitwork.L0.hi = (rc5unitwork.L0.hi + 0x00000001) & 0x000000FF;
              if (!(rc5unitwork.L0.hi & 0x000000FF)) 
                {
                rc5unitwork.L0.hi = 0x00000000;
                rc5unitwork.L0.lo = rc5unitwork.L0.lo + 0x01000000;
                if (!(rc5unitwork.L0.lo & 0xFF000000)) 
                  {
                  rc5unitwork.L0.lo = (rc5unitwork.L0.lo + 0x00010000) & 0x00FFFFFF;
                  if (!(rc5unitwork.L0.lo & 0x00FF0000)) 
                    {
                    rc5unitwork.L0.lo = (rc5unitwork.L0.lo + 0x00000100) & 0x0000FFFF;
                    if (!(rc5unitwork.L0.lo & 0x0000FF00)) 
                      {
                      rc5unitwork.L0.lo = (rc5unitwork.L0.lo + 0x00000001) & 0x000000FF;
                      }
                    }
                  }
                }
              }
            }
          }
        // increment the count of keys done
        // note: doesn't account for carry
        contestwork.keysdone.lo += pipeline_count;
        }
      }
    #if (CLIENT_OS == OS_MACOS)
      if (MP_active == 0) {     
        YieldToMain(1);
    #endif
    }
  else
    {
    timeslice *= pipeline_count;
    u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

    if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
    else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
    timeslice = (1ul << nbits) / pipeline_count;
    kiter = des_unit_func ( &rc5unitwork, nbits );
    contestwork.keysdone.lo += kiter;
    if (kiter < ( timeslice * pipeline_count ) )
      {
      // found it?
      rc5result.key.hi = contestwork.key.hi;
      rc5result.key.lo = contestwork.key.lo;
      rc5result.keysdone.hi = contestwork.keysdone.hi;
      rc5result.keysdone.lo = contestwork.keysdone.lo;
      rc5result.iterations.hi = contestwork.iterations.hi;
      rc5result.iterations.lo = contestwork.iterations.lo;
      rc5result.result = RESULT_FOUND;
      finished = 1;
      return( 1 );
      }
    else if (kiter != (timeslice * pipeline_count))
      {
      LogScreen("kiter wrong %ld %ld\n",
               (long) kiter, (long)(timeslice*pipeline_count));
      }
    }
#endif

  if ( ( contestwork.keysdone.hi > contestwork.iterations.hi ) ||
       ( ( contestwork.keysdone.hi == contestwork.iterations.hi ) &&
       ( contestwork.keysdone.lo >= contestwork.iterations.lo ) ) )
    {
    // done with this block and nothing found
    rc5result.result = RESULT_NOTHING;
    finished = 1;
    }
  else
    {
    // more to do, come back later.
    rc5result.result = RESULT_WORKING;
    finished = 0;
    }

  rc5result.key.hi = contestwork.key.hi;
  rc5result.key.lo = contestwork.key.lo;
  rc5result.keysdone.hi = contestwork.keysdone.hi;
  rc5result.keysdone.lo = contestwork.keysdone.lo;
  rc5result.iterations.hi = contestwork.iterations.hi;
  rc5result.iterations.lo = contestwork.iterations.lo;

#if 0
#if (CLIENT_OS == OS_RISCOS)    
if (!finished)
  LogScreen("Thread %d: Didn't find the key",threadindex);
else if (threadindex == 1)
  LogScreen("working... %08lx",contestwork.keysdone.lo);
#endif
#endif
  return( finished );
}
