/* Hey, Emacs, this is *not* a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Parameter/Result definitions for crypto cores:
 * Caution: ******** this header is used from C source ************
 *
 * Crypto cores take a RC5UnitWork struct as an argument, and
 * a) [obsolete] return the number of keys they checked in that call
 * b) [TODO] return one of the result codes defined in the ResultCode enum
 *
 * The ideal core prototype is:
 * extern s32 (*core)(RC5UnitWork *, u32 *timeslice, void *scratch_area );
 * - which returns a ResultCode member, or 0xffffffff (-1) if error
 * - timeslice is a *pointer*. On input this is the number of timeslices to
 *   do, and on return it contains the number of timeslices done. This may be
 *   greater than the number requested.
 * - scratch_area is a membuffer of core dependant size. This buffer is part
 *   of the problem object (ie created when the object is new'd)
*/
#ifndef __CCOREIO_H__
#define __CCOREIO_H__ "@(#)$Id: ccoreio.h,v 1.23 2009/04/01 15:49:21 andreasb Exp $"

#include "cputypes.h"   /* u32 etc. used here and in the cores */


#if (CLIENT_OS == OS_QNX) && !defined( __QNXNTO__ )
  #define CDECL cdecl
#elif  (CLIENT_OS == OS_AMIGAOS) && (CLIENT_CPU == CPU_68K)
  #define CDECL __regargs
#elif defined(__WATCOMC__)
  #define CDECL __cdecl
#endif
#ifndef CDECL
  #define CDECL /* nothing */
#endif


typedef enum
{
  RESULT_WORKING = 0,   /* do not change RESULT_* code order/init value */
  RESULT_NOTHING = 1,
  RESULT_FOUND   = 2
} Resultcode;

#include "pack1.h"

typedef struct
{
  struct {u32 hi,lo;} plain;  /* plaintext (already mixed with iv!) */
  struct {u32 hi,lo;} cypher; /* cyphertext */
  struct {u32 hi,lo;} L0;     /* key, changes with every unit * PIPELINE_COUNT. */
} DNETC_PACKED RC5UnitWork;

typedef struct
{
  struct {u32 hi,lo;} plain;  /* plaintext (already mixed with iv!) */
  struct {u32 hi,lo;} cypher; /* cyphertext */
  struct {u32 hi,mid,lo;} L0; /* key, changes with every unit * PIPELINE_COUNT. */
  struct {u32 count; u32 hi,mid,lo;} check; /* counter-measure check */
#if (CLIENT_CPU == CPU_CUDA) || (CLIENT_CPU == CPU_ATI_STREAM)
  // HACK! better move these members to someplace else
  int threadnum; /* To track which GPU to use */
  int optimal_timeslice_increment; /* Best timeslice increment for the core */
  long best_time; /* Remember the best time for optimal sleeping in subsequent runs */
#endif
} DNETC_PACKED RC5_72UnitWork;

#include "pack0.h"

#endif /* __CCOREIO_H__ */
