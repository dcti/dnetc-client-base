/* Hey, Emacs, this is *not* a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
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
#define __CCOREIO_H__ "@(#)$Id: ccoreio.h,v 1.2 1999/04/18 14:42:44 patrick Exp $"

typedef enum
{
  RESULT_WORKING = 0,   /* do not change RESULT_* code order/init value */
  RESULT_NOTHING = 1,
  RESULT_FOUND   = 2
} Resultcode;

typedef struct
{
  u64 plain;            /* plaintext (already mixed with iv!) */
  u64 cypher;           /* cyphertext */
  u64 L0;               /* key, changes with every unit * PIPELINE_COUNT. */
                        /* Note: data is now in RC5/platform useful form */
} RC5UnitWork;

#if (CLIENT_OS == OS_AIX)
extern "C" s32 rc5_ansi_2_rg_unit_func( RC5UnitWork *rc5unitwork, u32 timeslice );
// extern "C" static void  __SwitchRC5Format(u64 *_key);
#endif

#endif /* __CCOREIO_H__ */
