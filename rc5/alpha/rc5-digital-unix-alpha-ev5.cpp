// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net p/ojects only.
// Any other distribution or use of this source violates copyright.
//
// dual-key, mixed round 3 and encryption, A1/A2 use for last value,
// non-arrayed S1/S2 tables

// This file is based on rc5ansi2-bo2.cpp but has been optimised for
// the Digital EV5 series of chips.


#if (!defined(lint) && defined(__showids__))
const char *rc5ansi2_bo2_cpp (void) {
return "@(#)$Id: rc5-digital-unix-alpha-ev5.cpp,v 1.1.2.1 1999/12/08 02:08:07 cyp Exp $"; }
#endif

#include "stdio.h"
#include "problem.h"
#include "rotate-digital-unix-alpha-ev5.h"

#define PIPELINE_COUNT 2 //this is a two pipeline core
extern "C" u32 rc5_alpha_osf_ev5( RC5UnitWork * , u32 );   /* this */

#define P     0xB7E15163
#define Q     0x9E3779B9

static unsigned long long	SNOT0 = P;
static u32	SNOT[26]=
	{
		(P) & 0x00000000ffffffff,
		(P+Q) & 0x00000000ffffffff,
		(P+Q*2) & 0x00000000ffffffff,
		(P+Q*3) & 0x00000000ffffffff,
		(P+Q*4) & 0x00000000ffffffff,
		(P+Q*5) & 0x00000000ffffffff,
		(P+Q*6) & 0x00000000ffffffff,
		(P+Q*7) & 0x00000000ffffffff,
		(P+Q*8) & 0x00000000ffffffff,
		(P+Q*9) & 0x00000000ffffffff,
		(P+Q*10) & 0x00000000ffffffff,
		(P+Q*11) & 0x00000000ffffffff,
		(P+Q*12) & 0x00000000ffffffff,
		(P+Q*13) & 0x00000000ffffffff,
		(P+Q*14) & 0x00000000ffffffff,
		(P+Q*15) & 0x00000000ffffffff,
		(P+Q*16) & 0x00000000ffffffff,
		(P+Q*17) & 0x00000000ffffffff,
		(P+Q*18) & 0x00000000ffffffff,
		(P+Q*19) & 0x00000000ffffffff,
		(P+Q*20) & 0x00000000ffffffff,
		(P+Q*21) & 0x00000000ffffffff,
		(P+Q*22) & 0x00000000ffffffff,
		(P+Q*23) & 0x00000000ffffffff,
		(P+Q*24) & 0x00000000ffffffff,
		(P+Q*25) & 0x00000000ffffffff
	};

#define ROUND1EVEN(N, S1N, S2N) \
    A1 = S1N = ROTL3(SNOT[N] + A1 + Lhi1);       \
    A2 = S2N = ROTL3(SNOT[N] + A2 + Lhi2);       \
    Llo1 = ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1);       \
    Llo2 = ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2);

#define  ROUND1ODD(N, S1N, S2N) \
    A1 = S1N = ROTL3(SNOT[N] + A1 + Llo1);       \
    A2 = S2N = ROTL3(SNOT[N] + A2 + Llo2);       \
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

static u32 rc5_unit_func( RC5UnitWork * rc5unitwork )
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
  S1_00 = A1 = ROTLL3(SNOT0);
  Llo1 = ROTL(Llo1 + A1, A1);
  S2_00 = A2 = ROTLL3(SNOT0);
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
	      ROTL3(S1_25 + A1 + ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1)))
		return 1;
    if (rc5unitwork->cypher.lo == eA2 &&
	    rc5unitwork->cypher.hi == ROTL(eB2 ^ eA2, eA2) +
	      ROTL3(S2_25 + A2 + ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2)))
		return 2;
	  return 0;	
  }
}



/* -----------------------------------------------------------------

u32 rc5_alpha_osf_ev5( RC5UnitWork * rc5unitwork, u32 iterations )
{                                
  u32 kiter = 0;
  int keycount = iterations;
  int pipeline_count = PIPELINE_COUNT;
  
  //LogScreenf ("rc5unitwork = %08X:%08X (%X)\n", rc5unitwork.L0.hi, rc5unitwork.L0.lo, keycount);
  while ( keycount-- ) // iterations ignores the number of pipelines
  {
    u32 result = rc5_unit_func( rc5unitwork );
    if ( result )
    {
      kiter += result-1;
      break;
    }
    else
    {
      /* note: we switch the order */  
      register u32 tempkeylo = rc5unitwork->L0.hi; 
      register u32 tempkeyhi = rc5unitwork->L0.lo;
      rc5unitwork->L0.lo =
        ((tempkeylo >> 24) & 0x000000FFL) |                               
        ((tempkeylo >>  8) & 0x0000FF00L) |                               
        ((tempkeylo <<  8) & 0x00FF0000L) |                               
        ((tempkeylo << 24) & 0xFF000000L);                                
      rc5unitwork->L0.hi = 
        ((tempkeyhi >> 24) & 0x000000FFL) |                               
        ((tempkeyhi >>  8) & 0x0000FF00L) |                               
        ((tempkeyhi <<  8) & 0x00FF0000L) |                               
        ((tempkeyhi << 24) & 0xFF000000L);                                
      rc5unitwork->L0.lo += pipeline_count;
      if (rc5unitwork->L0.lo < ((u32)pipeline_count))
        rc5unitwork->L0.hi++;
      tempkeylo = rc5unitwork->L0.hi; 
      tempkeyhi = rc5unitwork->L0.lo;
      rc5unitwork->L0.lo =
        ((tempkeylo >> 24) & 0x000000FFL) |                               
        ((tempkeylo >>  8) & 0x0000FF00L) |                               
        ((tempkeylo <<  8) & 0x00FF0000L) |                               
        ((tempkeylo << 24) & 0xFF000000L);                                
      rc5unitwork->L0.hi = 
        ((tempkeyhi >> 24) & 0x000000FFL) |                               
        ((tempkeyhi >>  8) & 0x0000FF00L) |                               
        ((tempkeyhi <<  8) & 0x00FF0000L) |                               
        ((tempkeyhi << 24) & 0xFF000000L);                                
      kiter += pipeline_count;
    }
  }
  return kiter;
}  
