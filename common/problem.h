// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: problem.h,v $
// Revision 1.44  1999/03/09 07:15:45  gregh
// Various OGR changes.
//
// Revision 1.43  1999/03/08 02:54:51  sampo
// remove 'extern "C"' from PowerPC func call to fix build bustage
//
// Revision 1.42  1999/03/04 01:25:28  cyp
// Added (*[rc5|des]_unit_func) function pointers for ppc and alpha+dec_unix.
//
// Revision 1.41  1999/03/01 08:19:44  gregh
// Changed ContestWork to a union that contains crypto (RC5/DES) and OGR data.
//
// Revision 1.40  1999/02/21 21:44:59  cyp
// tossed all redundant byte order changing. all host<->net order conversion
// as well as scram/descram/checksumming is done at [get|put][net|disk] points
// and nowhere else.
//
// Revision 1.39  1999/02/21 09:58:37  silby
// Removed prototype for IncrementKey
//
// Revision 1.38  1999/02/17 19:09:13  remi
// Fix for non-x86 targets : an RC5 key should always be 'mangle-incremented',
// whatever endianess we have. But h.tonl()/n.tohl() does work for DES, so I
// added a contest parameter to IncrementKey().
//
// Revision 1.37  1999/02/15 06:26:36  silby
// Complete rewrite of Problem::Run to make it 64-bit
// compliant and begin combination of all processor
// types into common code.  Additional checks to
// ensure that cores are performing properly have also been
// added.  Porters should verify that their core is
// working properly with these changes.
//
// Revision 1.36  1999/02/04 22:49:32  remi
// Added #ifdef(DWORZ) into the {MIN,MAX}_DES_BITS selection code.
//
// Revision 1.35  1999/01/17 21:38:52  cyp
// memblock for bruce ford's deseval-mmx is now passed from the problem object.
//
// Revision 1.34  1999/01/01 02:45:16  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.33  1998/12/25 03:08:57  cyp
// x86 Bryd is runnable on upto 4 threads (threads 3 and 4 use the two
// non-optimal cores, ie pro cores on a p5 machine and vice versa).
// Made some non-core related stuff u64 clean.
//
// Revision 1.32  1998/12/14 12:48:59  cyp
// This is the final revision of problem.cpp/problem.h before the class goes
// to 'u64-clean'. Please check/declare all core prototypes.
//
// Revision 1.31  1998/12/08 05:59:40  dicamillo
// Define new method, GetKeysDone, needed for MacOS GUI.
//
// Revision 1.30  1998/12/01 19:49:14  cyp
// Cleaned up MULT1THREAD #define. See cputypes.h for full log entry.
//
// Revision 1.29  1998/11/28 17:45:58  remi
// Integration of the 386/486 self-modifying core.
//
// Revision 1.28  1998/11/25 09:23:37  chrisb
// various changes to support x86 coprocessor under RISC OS
//
// Revision 1.27  1998/11/14 14:07:54  cyp
// Removed trailing ^Z
//
// Revision 1.26  1998/11/14 13:56:13  cyp
// Fixed pipeline_count for x86 clients (DES cores were running with 4
// pipelines). Fixed unused parameter warning in LoadState(). Problem manager
// saves its probman_index in the Problem object (needed by chrisb's x86
// copro board code.)
//
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
// Use only two x86 DES cores (P5 & PPro) when not multi-threaded.
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
#include "client2.h"
#include "stub.h"

#if (CLIENT_CPU == CPU_X86)
  #define MAX_MEM_REQUIRED_BY_CORE (17*1024)
#endif

//#ifndef PIPELINE_COUNT
// #define PIPELINE_COUNT  2  // normally 1, but 2+ if we do more then one unit in parallel
//#endif

#if !defined(MEGGS) && !defined(DES_ULTRA) && !defined(DWORZ)
  #define MIN_DES_BITS  8
  #define MAX_DES_BITS 24
#else
  #if defined(BIT_32)
    #define MIN_DES_BITS 19
    #define MAX_DES_BITS 19
  #elif (defined(BIT_64) && defined(BITSLICER_WITH_LESS_BITS) && !defined(DWORZ))
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
typedef union
{
  struct {
    u64 key;              // starting key
    u64 iv;               // initialization vector
    u64 plain;            // plaintext we're searching for
    u64 cypher;           // cyphertext
    u64 keysdone;         // iterations done (also current position in block)
    u64 iterations;       // iterations to do
  } crypto;
  struct {
    Stub stub;
  } ogr;
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


class Problem
{
public:
  int finished;
  u32 startpercent;
  u32 percent;
  int restart;
  u32 timehi, timelo;
  int started;
  unsigned int contest;
  int cputype;

  unsigned int pipeline_count;
  u32 tslice; 

  #ifdef MAX_MEM_REQUIRED_BY_CORE
  char core_membuffer[MAX_MEM_REQUIRED_BY_CORE];
  #endif

  unsigned int threadindex; /* index of this problem in the problem table */
  int threadindex_is_valid; /* 0 if the problem is not managed by probman*/
  
// protected: ahem.
  u32 initialized;
  ContestWork contestwork;
  RC5UnitWork rc5unitwork;
  RC5Result rc5result;
  u64 refL0;
  CoreDispatchTable *ogr;
  void *ogrstate;

  #if (CLIENT_CPU == CPU_X86)
  u32 (*unit_func)( RC5UnitWork * rc5unitwork, u32 timeslice );
  #elif (CLIENT_CPU == CPU_68K)
  extern "C" __asm u32 (*rc5_unit_func)( register __a0 RC5UnitWork *work, register __d0 u32 timeslice);
  #elif (CLIENT_CPU == CPU_ARM)
  u32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork, unsigned long iterations );
  u32 (*des_unit_func)( RC5UnitWork * rc5unitwork, u32 timeslice );
  #elif (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_DEC_UNIX) //defined(DEC_UNIX_CPU_SELECT)
  u32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork );
  u32 (*des_unit_func)( RC5UnitWork * rc5unitwork, u32 nbits );
  #elif (CLIENT_CPU == CPU_POWERPC)
  int (*rc5_unit_func)( RC5UnitWork * rc5unitwork, unsigned long iterations );
  #endif

public:
  Problem(long _threadindex = -1L);
  ~Problem();

  int IsInitialized() { return (initialized!=0); }

  int LoadState( ContestWork * work, unsigned int _contest, u32 _timeslice, 
                                                            int _cputype );
    // Load state into internal structures.
    // state is invalid (will generate errors) until this is called.
    // returns: -1 on error, 0 is OK

  s32 RetrieveState( ContestWork * work , s32 setflags );
    // Retrieve state from internal structures.
    // state is invalid (will generate errors) immediately after this is called, if setflags==1.
    // returns: -1 on error, 0 is OK

  s32 Run( u32 /* unused */ );
    // Runs calling rc5_unit for timeslice times...
    // Returns:
    //   -1 if something goes wrong (state not loaded, already done etc...)
    //   0 if more work to be done
    //   1 if we're done, go get results

  s32 GetResult( RC5Result * result );
    // fetch the results... act based on result code...
    // returns: contest=0 (RC5), contest=1 (DES), or -1 = invalid data (state not loaded).

  u32 CalcPercent() { return (u32)( ((double)(100.0)) *
    /* Return the % completed in the current block, to nearest 1%. */
        (((((double)(contestwork.crypto.keysdone.hi))*((double)(4294967296.0)))+
                                 ((double)(contestwork.crypto.keysdone.lo))) /
        ((((double)(contestwork.crypto.iterations.hi))*((double)(4294967296.0)))+
                                 ((double)(contestwork.crypto.iterations.lo)))) ); }

  u32 AlignTimeslice(void);
     // return a modified ::tslice value that [has been adjusted if 
     // >(iter-keysdone) and] is an even multiple of pipeline_count and 2 


#if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
  u32 GetKeysDone() { return(rc5result.keysdone.lo); }
    // Returns keys completed for Mac GUI display.
#endif

};

#endif

