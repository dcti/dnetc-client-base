// dual-key, mixed round 3 and encryption, A1/A2 use for last value,
// non-arrayed S1/S2 tables

// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

/*  This file is included from rc5.cpp so we can use __inline__.  */

#include "problem.h"
#include "rotate.h"

extern "C" 
{
  unsigned int alpha_keycracker( RC5UnitWork * rc5unitwork, unsigned long count);
}


#define CORE_INCREMENTS_KEY 1

#if (PIPELINE_COUNT != 2)
#error "Expecting pipeline count of 2"
#endif

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif


#define P     0xB7E15163L
#define Q     0x9E3779B9L
#define PROT3 0xbf0a8b1dL

#define S_not(n)      P+Q*n

#define S8_not(x) ((S_not(x))<<3)

unsigned long alpha_S_not[1024]={ 
  PROT3,
  (S_not(1L)+PROT3)<<3,
  S8_not(2L),
  S8_not(3L),
  S8_not(4L),
  S8_not(5L),
  S8_not(6L),
  S8_not(7L),
  S8_not(8L),
  S8_not(9L),
  S8_not(10L),
  S8_not(11L),
  S8_not(12L),
  S8_not(13L),
  S8_not(14L),
  S8_not(15L),
  S8_not(16L),
  S8_not(17L),
  S8_not(18L),
  S8_not(19L),
  S8_not(20L),
  S8_not(21L),
  S8_not(22L),
  S8_not(23L),
  S8_not(24L),
  S8_not(25L)};

static __inline__
u32 rc5_unit_func( RC5UnitWork * rc5unitwork, u32 timeslice)
{
  return alpha_keycracker(rc5unitwork,timeslice); 
}
  


