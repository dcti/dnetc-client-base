// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: problem.h,v $
// Revision 1.25  1998/09/25 11:31:20  chrisb
// Added stuff to support 3 cores in the ARM clients.
//
// Revision 1.24  1998/09/23 22:08:54  blast
// problem.h updated for rc5 68k multicore support.
//
// Revision 1.22  1998/08/20 19:34:30  cyruspatel
// Removed that terrible PIPELINE_COUNT hack: Timeslice and pipeline count
// are now computed in Problem::LoadState(). Client::SelectCore() now saves
// core type to Client::cputype.
//
// Revision 1.21  1998/08/20 01:54:41  silby
// Chnages to accomodate rc5mmx cores
//
// Revision 1.20  1998/08/15 21:30:27  jlawson
// modified PIPELINE_COUNT definition
//
// Revision 1.19  1998/08/14 00:05:11  silby
// Changes for rc5 mmx core integration.
//
// Revision 1.18  1998/08/05 16:42:15  cberry
// ARM clients now define PIPELINE_COUNT=2
//
// Revision 1.17  1998/07/29 21:31:44  blast
// Changed the default 68K cpu PIPELINE_COUNT for AmigaOS to 256...
// Testing new core that's faster on 68060...
//
// Revision 1.16  1998/07/14 10:43:33  remi
// Added support for a minimum timeslice value of 16 instead of 20 when
// using BIT_64, which is needed by MMX_BITSLICER. Will help some platforms
// like Netware or Win16. I added support in deseval-meggs3.cpp, but it's just
// for completness, Alphas don't need this patch.
//
// Important note : this patch **WON'T** work with deseval-meggs2.cpp, but
// according to the configure script it isn't used anymore. If you compile
// des-slice-meggs.cpp and deseval-meggs2.cpp with BIT_64 and
// BITSLICER_WITH_LESS_BITS, the DES self-test will fail.
//
// Revision 1.15  1998/07/08 09:56:11  remi
// Added support for the MMX bitslicer.
//
// Revision 1.14  1998/06/23 21:58:56  remi
// Use only two x86 DES cores (P5 & PPro) when not multithreaded.
//
// Revision 1.13  1998/06/20 10:04:16  cyruspatel
// Modified so x86 make with /DKWAN will work: Renamed des_unit_func() in
// des_slice to des_unit_func_slice() to resolve conflict with (*des_unit_func)().
// Added prototype in problem.h, cliconfig x86/SelectCore() is /DKWAN aware.
//
// Revision 1.12  1998/06/16 21:53:30  silby
// Added support for dual x86 DES cores (p5/ppro)
//
// Revision 1.11  1998/06/15 06:18:39  dicamillo
// Updates for BeOS
//
// Revision 1.10  1998/06/14 15:17:05  remi
// UltraSparc DES core integration.
//
// Revision 1.9  1998/06/14 08:13:06  friedbait
// 'Log' keywords added to maintain automatic change history
//
// 

#ifndef _PROBLEM_H_
#define _PROBLEM_H_

#include "cputypes.h"

#ifndef PIPELINE_COUNT
 #define PIPELINE_COUNT  2  // normally 1, but 2+ if we do more then one unit in parallel
#endif

#if !defined(MEGGS) && !defined(DES_ULTRA)
  #define MIN_DES_BITS  8
  #define MAX_DES_BITS 24
#else
  #if defined(BIT_32)
    #define MIN_DES_BITS 19
    #define MAX_DES_BITS 19
  #elif defined(BIT_64) && defined(BITSLICER_WITH_LESS_BITS)
    #define MIN_DES_BITS 16
    #define MAX_DES_BITS 16
  #elif defined(BIT_64)
    #define MIN_DES_BITS 20
    #define MAX_DES_BITS 20
  #endif
#endif

typedef struct
{
  u64 plain;            // plaintext (already mixed with iv!)
  u64 cypher;           // cyphertext
  u64 L0;               // key, changes with every unit * PIPELINE_COUNT.
                        // Note: data is now in RC5/platform useful form
} RC5UnitWork;

// this has to stay 'in sync' with FileEntry
typedef struct
{
  u64 key;              // starting key
  u64 iv;               // initialization vector
  u64 plain;            // plaintext we're searching for
  u64 cypher;           // cyphertext
  u64 keysdone;         // iterations done (also current position in block)
  u64 iterations;       // iterations to do
} ContestWork;

typedef struct
{
  u32 result;           // result code
  u64 key;              // starting key
  u64 keysdone;         // iterations done (also current position in block)
                        // this is also the "answer" for a RESULT_FOUND
  u64 iterations;       // iterations to do
} RC5Result;

typedef enum
{
  RESULT_NOTHING,
  RESULT_FOUND,
  RESULT_WORKING
} Resultcode;


#if (CLIENT_CPU == CPU_X86)
  #if (PIPELINE_COUNT != 2)
  #error "Expecting PIPELINE_COUNT=2"
  #endif

  #ifdef __WATCOMC__
    #define rc5_unit_func_486 _rc5_unit_func_486
    #define rc5_unit_func_p5 _rc5_unit_func_p5
    #define rc5_unit_func_p6 _rc5_unit_func_p6
    #define rc5_unit_func_6x86 _rc5_unit_func_6x86
    #define rc5_unit_func_k5 _rc5_unit_func_k5
    #define rc5_unit_func_k6 _rc5_unit_func_k6
  #endif

  extern u32 (*des_unit_func)( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern u32 (*des_unit_func2)( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern u32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_486( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_p5( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_p6( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_6x86( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_k5( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_k6( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern "C" u32 rc5_unit_func_p5_mmx( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern u32 p1des_unit_func_p5( RC5UnitWork * rc5unitwork, u32 timeslice );
  extern u32 p1des_unit_func_pro( RC5UnitWork * rc5unitwork, u32 timeslice );
  #if defined(MULTITHREAD)
    extern u32 p2des_unit_func_p5( RC5UnitWork * rc5unitwork, u32 timeslice );
    extern u32 p2des_unit_func_pro( RC5UnitWork * rc5unitwork, u32 timeslice );
  #endif
  #if defined(MMX_BITSLICER)
    extern u32 des_unit_func_mmx( RC5UnitWork * rc5unitwork, u32 nbbits );
  #elif defined(MEGGS) || defined(KWAN)
    extern u32 des_unit_func_slice( RC5UnitWork * rc5unitwork, u32 nbbits );
  #endif

#elif (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32)
  #if (PIPELINE_COUNT != 1)
  #error "Expecting PIPELINE_COUNT=1"
  #endif
#elif (CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS != OS_WIN32)
  #if (PIPELINE_COUNT != 1)
  #error "Expecting PIPELINE_COUNT=1"
  #endif
  extern int whichcrunch;
#elif (CLIENT_CPU == CPU_68K)
  extern "C" extern __asm u32 (*rc5_unit_func)( register __a0 RC5UnitWork *work, register __d0 u32 timeslice);
  extern "C" __asm u32 rc5_unit_func_000_030( register __a0 RC5UnitWork * work, register __d0 unsigned long timeslice );
  extern "C" __asm u32 rc5_unit_func_040_060( register __a0 RC5UnitWork * work, register __d0 unsigned long timeslice );
  #if (PIPELINE_COUNT != 2)
  #error "Expecting PIPELINE_COUNT=2"
  #endif
#elif (CLIENT_CPU == CPU_ARM)
  #if (PIPELINE_COUNT != 2)
  #error "Expecting PIPELINE_COUNT=2"
  #endif
  extern u32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork, unsigned long t );
  extern u32 (*des_unit_func)( RC5UnitWork * rc5unitwork, unsigned long t );
  extern "C" u32 rc5_unit_func_arm_1( RC5UnitWork * rc5unitwork , unsigned long t);
  extern "C" u32 rc5_unit_func_arm_2( RC5UnitWork * rc5unitwork , unsigned long t);
  extern "C" u32 rc5_unit_func_arm_3( RC5UnitWork * rc5unitwork , unsigned long t);

  extern "C" u32 des_unit_func_arm( RC5UnitWork * rc5unitwork , unsigned long t);
  extern "C" u32 des_unit_func_strongarm( RC5UnitWork * rc5unitwork , unsigned long t);
#endif

class Problem
{
public:
  u32 finished;
  u32 startpercent;
  u32 percent;
  bool restart;
  u32 timehi, timelo;
  u32 started;
  u32 contest;

  u32 pipeline_count;
  u32 tslice; 
  
protected:
  u32 initialized;
  ContestWork contestwork;
  RC5UnitWork rc5unitwork;
  RC5Result rc5result;

public:
  Problem();
  ~Problem();

  s32 IsInitialized(); 

  s32 LoadState( ContestWork * work, u32 contesttype, u32 _timeslice, u32 _cputype );
    // Load state into internal structures.
    // state is invalid (will generate errors) until this is called.
    // returns: -1 on error, 0 is OK
    // Note: data is all in Network Byte order (going in)( Big Endian )

  s32 RetrieveState( ContestWork * work , s32 setflags );
    // Retrieve state from internal structures.
    // state is invalid (will generate errors) immediately after this is called, if setflags==1.
    // returns: -1 on error, 0 is OK
    // Note: data is all in Network Byte order (coming out)( Big Endian )

  s32 Run( u32 threadnum );
    // Runs calling rc5_unit for timeslice times...
    // Returns:
    //   -1 if something goes wrong (state not loaded, already done etc...)
    //   0 if more work to be done
    //   1 if we're done, go get results

  s32 GetResult( RC5Result * result );
    // fetch the results... act based on result code...
    // returns: contest=0 (RC5), contest=1 (DES), or -1 = invalid data (state not loaded).
    // Note: data (except result) is all in Network Byte order ( Big Endian )

  u32 CalcPercent();
    // Return the % completed in the current block, to nearest 1%.

};

#endif

