// dual-key, mixed round 3 and encryption, A1/A2 use for last value,
// arrayed S1/S2 tables

// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: rc5ansi2-bo1.cpp,v $
// Revision 1.6  1998/07/08 22:59:47  remi
// Lots of $Id$ stuff.
//
// Revision 1.5  1998/06/14 08:27:31  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.4  1998/06/14 08:13:48  friedbait
// 'Log' keywords added to maintain automatic change history
//
//


/*  This file is included from rc5.cpp so we can use __inline__.  */

#if (!defined(lint) && defined(__showids__))
const char *rc5ansi2_bo1_cpp (void) {
return "@(#)$Id: rc5ansi2-bo1.cpp,v 1.6 1998/07/08 22:59:47 remi Exp $"; }
#endif

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

#define ROUND1EVEN(N) \
    A1 = S1[N] = ROTL3(S_not(N) + A1 + Lhi1);       \
    A2 = S2[N] = ROTL3(S_not(N) + A2 + Lhi2);       \
    Llo1 = ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1);       \
    Llo2 = ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2);

#define  ROUND1ODD(N) \
    A1 = S1[N] = ROTL3(S_not(N) + A1 + Llo1);       \
    A2 = S2[N] = ROTL3(S_not(N) + A2 + Llo2);       \
    Lhi1 = ROTL(Lhi1 + A1 + Llo1, A1 + Llo1);       \
    Lhi2 = ROTL(Lhi2 + A2 + Llo2, A2 + Llo2);

#define ROUND2EVEN(N) \
    A1 = S1[N] = ROTL3(S1[N] + A1 + Lhi1);        \
    A2 = S2[N] = ROTL3(S2[N] + A2 + Lhi2);        \
    Llo1 = ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1);     \
    Llo2 = ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2);

#define  ROUND2ODD(N) \
    A1 = S1[N] = ROTL3(S1[N] + A1 + Llo1);        \
    A2 = S2[N] = ROTL3(S2[N] + A2 + Llo2);        \
    Lhi1 = ROTL(Lhi1 + A1 + Llo1, A1 + Llo1);     \
    Lhi2 = ROTL(Lhi2 + A2 + Llo2, A2 + Llo2);

#define ROUND3EVEN(N) \
    eA1 = ROTL(eA1 ^ eB1, eB1) + (A1 = ROTL3(S1[N] + A1 + Lhi1)); \
    eA2 = ROTL(eA2 ^ eB2, eB2) + (A2 = ROTL3(S2[N] + A2 + Lhi2)); \
    Llo1 = ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1);      \
    Llo2 = ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2);

#define ROUND3ODD(N)  \
    eB1 = ROTL(eB1 ^ eA1, eA1) + (A1 = ROTL3(S1[N] + A1 + Llo1)); \
    eB2 = ROTL(eB2 ^ eA2, eA2) + (A2 = ROTL3(S2[N] + A2 + Llo2)); \
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
  static u32 S1[26], S2[26];

    register u32 A1, Llo1, Lhi1;
    register u32 A2, Llo2, Lhi2;

    Llo2 = Llo1 = rc5unitwork->L0.lo;
    Lhi2 = (Lhi1 = rc5unitwork->L0.hi) + 0x01000000;

    /* Begin round 1 of key expansion */

    /*  Special case while A and B are known to be zero.  */
    S1[0] = A1 = ROTL3(S_not(0));
    Llo1 = ROTL(Llo1 + A1, A1);
    S2[0] = A2 = ROTL3(S_not(0));
    Llo2 = ROTL(Llo2 + A2, A2);

    ROUND1ODD (1)
    ROUND1EVEN(2)
    ROUND1ODD (3)
    ROUND1EVEN(4)
    ROUND1ODD (5)
    ROUND1EVEN(6)
    ROUND1ODD (7)
    ROUND1EVEN(8)
    ROUND1ODD (9)
    ROUND1EVEN(10)
    ROUND1ODD (11)
    ROUND1EVEN(12)
    ROUND1ODD (13)
    ROUND1EVEN(14)
    ROUND1ODD (15)
    ROUND1EVEN(16)
    ROUND1ODD (17)
    ROUND1EVEN(18)
    ROUND1ODD (19)
    ROUND1EVEN(20)
    ROUND1ODD (21)
    ROUND1EVEN(22)
    ROUND1ODD (23)
    ROUND1EVEN(24)
    ROUND1ODD (25)

    /* Begin round 2 of key expansion */
    ROUND2EVEN(0)
    ROUND2ODD (1)
    ROUND2EVEN(2)
    ROUND2ODD (3)
    ROUND2EVEN(4)
    ROUND2ODD (5)
    ROUND2EVEN(6)
    ROUND2ODD (7)
    ROUND2EVEN(8)
    ROUND2ODD (9)
    ROUND2EVEN(10)
    ROUND2ODD (11)
    ROUND2EVEN(12)
    ROUND2ODD (13)
    ROUND2EVEN(14)
    ROUND2ODD (15)
    ROUND2EVEN(16)
    ROUND2ODD (17)
    ROUND2EVEN(18)
    ROUND2ODD (19)
    ROUND2EVEN(20)
    ROUND2ODD (21)
    ROUND2EVEN(22)
    ROUND2ODD (23)
    ROUND2EVEN(24)
    ROUND2ODD (25)

  {
    register u32 eA1, eB1, eA2, eB2;
    /* Begin round 3 of key expansion (and encryption round) */

    eA1 = rc5unitwork->plain.lo + (A1 = ROTL3(S1[0] + A1 + Lhi1));
    eA2 = rc5unitwork->plain.lo + (A2 = ROTL3(S2[0] + A2 + Lhi2));
    Llo1 = ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1);
    Llo2 = ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2);
    eB1 = rc5unitwork->plain.hi + (A1 = ROTL3(S1[1] + A1 + Llo1));
    eB2 = rc5unitwork->plain.hi + (A2 = ROTL3(S2[1] + A2 + Llo2));
    Lhi1 = ROTL(Lhi1 + A1 + Llo1, A1 + Llo1);
    Lhi2 = ROTL(Lhi2 + A2 + Llo2, A2 + Llo2);

    ROUND3EVEN(2)
    ROUND3ODD (3)
    ROUND3EVEN(4)
    ROUND3ODD (5)
    ROUND3EVEN(6)
    ROUND3ODD (7)
    ROUND3EVEN(8)
    ROUND3ODD (9)
    ROUND3EVEN(10)
    ROUND3ODD (11)
    ROUND3EVEN(12)
    ROUND3ODD (13)
    ROUND3EVEN(14)
    ROUND3ODD (15)
    ROUND3EVEN(16)
    ROUND3ODD (17)
    ROUND3EVEN(18)
    ROUND3ODD (19)
    ROUND3EVEN(20)
    ROUND3ODD (21)
    ROUND3EVEN(22)
    ROUND3ODD (23)

    eA1 = ROTL(eA1 ^ eB1, eB1) + (A1 = ROTL3(S1[24] + A1 + Lhi1));
    eA2 = ROTL(eA2 ^ eB2, eB2) + (A2 = ROTL3(S2[24] + A2 + Lhi2));
	
    if ((rc5unitwork->cypher.lo == eA1) &&
	    rc5unitwork->cypher.hi == ROTL(eB1 ^ eA1, eA1) +
	      ROTL3(S1[25] + A1 + ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1))) return 1;
    if ((rc5unitwork->cypher.lo == eA2) &&
	    rc5unitwork->cypher.hi == ROTL(eB2 ^ eA2, eA2) +
	      ROTL3(S2[25] + A2 + ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2))) return 2;
	  return 0;	
  }
}


