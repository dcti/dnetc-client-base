// dual-key, mixed round 3 and encryption, A1/A2 use for last value,
// non-arrayed S1/S2 tables

// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: rc5ansi2-bo2.cpp,v $
// Revision 1.5  1998/06/14 08:27:32  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.4  1998/06/14 08:13:49  friedbait
// 'Log' keywords added to maintain automatic change history
//
//


/*  This file is included from rc5.cpp so we can use __inline__.  */

static char *id="@(#)$Id: rc5ansi2-bo2.cpp,v 1.5 1998/06/14 08:27:32 friedbait Exp $";

#include "problem.h"
#include "rotate.h"


#if (PIPELINE_COUNT != 2)
#error "Expecting pipeline count of 2"
#endif

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif


#define P     0xB7E15163
#define Q     0x9E3779B9

#define S_not(n)      P+Q*n

#define ROUND1EVEN(N, S1N, S2N) \
    A1 = S1N = ROTL3(S_not(N) + A1 + Lhi1);       \
    A2 = S2N = ROTL3(S_not(N) + A2 + Lhi2);       \
    Llo1 = ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1);       \
    Llo2 = ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2);

#define  ROUND1ODD(N, S1N, S2N) \
    A1 = S1N = ROTL3(S_not(N) + A1 + Llo1);       \
    A2 = S2N = ROTL3(S_not(N) + A2 + Llo2);       \
    Lhi1 = ROTL(Lhi1 + A1 + Llo1, A1 + Llo1);       \
    Lhi2 = ROTL(Lhi2 + A2 + Llo2, A2 + Llo2);

#define ROUND2EVEN(S1N, S2N) \
    A1 = S1N = ROTL3(S1N + A1 + Lhi1);        \
    A2 = S2N = ROTL3(S2N + A2 + Lhi2);        \
    Llo1 = ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1);     \
    Llo2 = ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2);

#define  ROUND2ODD(S1N, S2N) \
    A1 = S1N = ROTL3(S1N + A1 + Llo1);        \
    A2 = S2N = ROTL3(S2N + A2 + Llo2);        \
    Lhi1 = ROTL(Lhi1 + A1 + Llo1, A1 + Llo1);     \
    Lhi2 = ROTL(Lhi2 + A2 + Llo2, A2 + Llo2);

#define ROUND3EVEN(S1N, S2N) \
    eA1 = ROTL(eA1 ^ eB1, eB1) + (A1 = ROTL3(S1N + A1 + Lhi1)); \
    eA2 = ROTL(eA2 ^ eB2, eB2) + (A2 = ROTL3(S2N + A2 + Lhi2)); \
    Llo1 = ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1);      \
    Llo2 = ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2);

#define ROUND3ODD(S1N, S2N)  \
    eB1 = ROTL(eB1 ^ eA1, eA1) + (A1 = ROTL3(S1N + A1 + Llo1)); \
    eB2 = ROTL(eB2 ^ eA2, eA2) + (A2 = ROTL3(S2N + A2 + Llo2)); \
    Lhi1 = ROTL(Lhi1 + A1 + Llo1, A1 + Llo1);          \
    Lhi2 = ROTL(Lhi2 + A2 + Llo2, A2 + Llo2);



// rc5_unit will get passed an RC5WorkUnit to complete
// this is where all the actually work occurs, this is where you optimize.
// assembly gurus encouraged.
// Returns: 0 - nothing found, 1 - found on pipeline 1,
//   2 - found pipeline 2, 3 - ... etc ...

static __inline__
u32 rc5_unit_func( RC5UnitWork * rc5unitwork )
{
  u32 S1_00,S1_01,S1_02,S1_03,S1_04,S1_05,S1_06,S1_07,S1_08,S1_09,
      S1_10,S1_11,S1_12,S1_13,S1_14,S1_15,S1_16,S1_17,S1_18,S1_19,
      S1_20,S1_21,S1_22,S1_23,S1_24,S1_25;

  u32 S2_00,S2_01,S2_02,S2_03,S2_04,S2_05,S2_06,S2_07,S2_08,S2_09,
      S2_10,S2_11,S2_12,S2_13,S2_14,S2_15,S2_16,S2_17,S2_18,S2_19,
      S2_20,S2_21,S2_22,S2_23,S2_24,S2_25;

  register u32 A1, Llo1, Lhi1;
  register u32 A2, Llo2, Lhi2;

  Llo2 = Llo1 = rc5unitwork->L0.lo;
  Lhi2 = (Lhi1 = rc5unitwork->L0.hi) + 0x01000000;

  /* Begin round 1 of key expansion */

  /*  Special case while A and B are known to be zero.  */
  S1_00 = A1 = ROTL3(S_not(0));
  Llo1 = ROTL(Llo1 + A1, A1);
  S2_00 = A2 = ROTL3(S_not(0));
  Llo2 = ROTL(Llo2 + A2, A2);

  ROUND1ODD (1, S1_01, S2_01)
  ROUND1EVEN(2, S1_02, S2_02)
  ROUND1ODD (3, S1_03, S2_03)
  ROUND1EVEN(4, S1_04, S2_04)
  ROUND1ODD (5, S1_05, S2_05)
  ROUND1EVEN(6, S1_06, S2_06)
  ROUND1ODD (7, S1_07, S2_07)
  ROUND1EVEN(8, S1_08, S2_08)
  ROUND1ODD (9, S1_09, S2_09)
  ROUND1EVEN(10, S1_10, S2_10)
  ROUND1ODD (11, S1_11, S2_11)
  ROUND1EVEN(12, S1_12, S2_12)
  ROUND1ODD (13, S1_13, S2_13)
  ROUND1EVEN(14, S1_14, S2_14)
  ROUND1ODD (15, S1_15, S2_15)
  ROUND1EVEN(16, S1_16, S2_16)
  ROUND1ODD (17, S1_17, S2_17)
  ROUND1EVEN(18, S1_18, S2_18)
  ROUND1ODD (19, S1_19, S2_19)
  ROUND1EVEN(20, S1_20, S2_20)
  ROUND1ODD (21, S1_21, S2_21)
  ROUND1EVEN(22, S1_22, S2_22)
  ROUND1ODD (23, S1_23, S2_23)
  ROUND1EVEN(24, S1_24, S2_24)
  ROUND1ODD (25, S1_25, S2_25)

  /* Begin round 2 of key expansion */
  ROUND2EVEN(S1_00, S2_00)
  ROUND2ODD (S1_01, S2_01)
  ROUND2EVEN(S1_02, S2_02)
  ROUND2ODD (S1_03, S2_03)
  ROUND2EVEN(S1_04, S2_04)
  ROUND2ODD (S1_05, S2_05)
  ROUND2EVEN(S1_06, S2_06)
  ROUND2ODD (S1_07, S2_07)
  ROUND2EVEN(S1_08, S2_08)
  ROUND2ODD (S1_09, S2_09)
  ROUND2EVEN(S1_10, S2_10)
  ROUND2ODD (S1_11, S2_11)
  ROUND2EVEN(S1_12, S2_12)
  ROUND2ODD (S1_13, S2_13)
  ROUND2EVEN(S1_14, S2_14)
  ROUND2ODD (S1_15, S2_15)
  ROUND2EVEN(S1_16, S2_16)
  ROUND2ODD (S1_17, S2_17)
  ROUND2EVEN(S1_18, S2_18)
  ROUND2ODD (S1_19, S2_19)
  ROUND2EVEN(S1_20, S2_20)
  ROUND2ODD (S1_21, S2_21)
  ROUND2EVEN(S1_22, S2_22)
  ROUND2ODD (S1_23, S2_23)
  ROUND2EVEN(S1_24, S2_24)
  ROUND2ODD (S1_25, S2_25)
  {
    register u32 eA1, eB1, eA2, eB2;
    /* Begin round 3 of key expansion (and encryption round) */

    eA1 = rc5unitwork->plain.lo + (A1 = ROTL3(S1_00 + A1 + Lhi1));
    eA2 = rc5unitwork->plain.lo + (A2 = ROTL3(S2_00 + A2 + Lhi2));
    Llo1 = ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1);
    Llo2 = ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2);
    eB1 = rc5unitwork->plain.hi + (A1 = ROTL3(S1_01 + A1 + Llo1));
    eB2 = rc5unitwork->plain.hi + (A2 = ROTL3(S2_01 + A2 + Llo2));
    Lhi1 = ROTL(Lhi1 + A1 + Llo1, A1 + Llo1);
    Lhi2 = ROTL(Lhi2 + A2 + Llo2, A2 + Llo2);

    ROUND3EVEN(S1_02, S2_02)
    ROUND3ODD (S1_03, S2_03)
    ROUND3EVEN(S1_04, S2_04)
    ROUND3ODD (S1_05, S2_05)
    ROUND3EVEN(S1_06, S2_06)
    ROUND3ODD (S1_07, S2_07)
    ROUND3EVEN(S1_08, S2_08)
    ROUND3ODD (S1_09, S2_09)
    ROUND3EVEN(S1_10, S2_10)
    ROUND3ODD (S1_11, S2_11)
    ROUND3EVEN(S1_12, S2_12)
    ROUND3ODD (S1_13, S2_13)
    ROUND3EVEN(S1_14, S2_14)
    ROUND3ODD (S1_15, S2_15)
    ROUND3EVEN(S1_16, S2_16)
    ROUND3ODD (S1_17, S2_17)
    ROUND3EVEN(S1_18, S2_18)
    ROUND3ODD (S1_19, S2_19)
    ROUND3EVEN(S1_20, S2_20)
    ROUND3ODD (S1_21, S2_21)
    ROUND3EVEN(S1_22, S2_22)
    ROUND3ODD (S1_23, S2_23)

    eA1 = ROTL(eA1 ^ eB1, eB1) + (A1 = ROTL3(S1_24 + A1 + Lhi1));
    eA2 = ROTL(eA2 ^ eB2, eB2) + (A2 = ROTL3(S2_24 + A2 + Lhi2));
	
    if (rc5unitwork->cypher.lo == eA1 &&
	    rc5unitwork->cypher.hi == ROTL(eB1 ^ eA1, eA1) +
	      ROTL3(S1_25 + A1 + ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1))) return 1;
    if (rc5unitwork->cypher.lo == eA2 &&
	    rc5unitwork->cypher.hi == ROTL(eB2 ^ eA2, eA2) +
	      ROTL3(S2_25 + A2 + ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2))) return 2;
	  return 0;	
  }
}


