/* Copyright distributed.net 1997 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * u32 rc5_ansi_1_b2_rg_unit_func ( RC5UnitWork * rc5unitwork, u32 iterations )
 * a 32 bit processor is assumed
 *
*/
#if (!defined(lint) && defined(__showids__))
const char *rc5ansi1_b2_cpp (void) {
return "@(#)$Id: rc5ansi1-b2.cpp,v 1.7 1999/12/02 01:52:46 cyp Exp $"; }
#endif

#include "problem.h"
#include "cputypes.h"
#include "rotate.h"
#define PIPELINE_COUNT 1 /* this is how many the core works with */

//#ifndef _CPU_32BIT_
//#error "everything assumes a 32bit CPU..."
//#endif

#define P      0xB7E15163
#define Q      0x9E3779B9

#define S_not(n)      P+Q*n


#define ROUND1_ODD(SN, N) \
    A = SN = ROTL3(S_not(N) + A + Llo); \
    Lhi = ROTL(Lhi + A + Llo, A + Llo)
#define ROUND1_EVEN(SN, N) \
    A = SN = ROTL3(S_not(N) + A + Lhi); \
    Llo = ROTL(Llo + A + Lhi, A + Lhi)

#define ROUND2_EVEN(SN) \
  A = SN = ROTL3(SN + A + Lhi); \
  Llo = ROTL(Llo + A + Lhi, A + Lhi)
#define ROUND2_ODD(SN) \
  A = SN = ROTL3(SN + A + Llo); \
  Lhi = ROTL(Lhi + A + Llo, A + Llo)

#define ROUND3_EVEN(SN) \
  eA = ROTL(eA ^ eB, eB) + (A = ROTL3(SN + A + Lhi)); \
  Llo = ROTL(Llo + A + Lhi, A + Lhi)
#define ROUND3_ODD(SN) \
  eB = ROTL(eB ^ eA, eA) + (A = ROTL3(SN + A + Llo)); \
  Lhi = ROTL(Lhi + A + Llo, A + Llo)

static __inline__ u32 rc5_unit_func ( RC5UnitWork * rc5unitwork )
{
  register u32 S00,S01,S02,S03,S04,S05,S06,S07,S08,S09,S10,S11,S12,
    S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25;

  register u32 A, Llo, Lhi;
  Llo = rc5unitwork->L0.lo;
  Lhi = rc5unitwork->L0.hi;

  /* Begin round 1 of key expansion */
  A = S00 = ROTL3(S_not(0)); Llo = ROTL(Llo + A, A);
  ROUND1_ODD (S01, 1);
  ROUND1_EVEN(S02, 2);
  ROUND1_ODD (S03, 3);
  ROUND1_EVEN(S04, 4);
  ROUND1_ODD (S05, 5);
  ROUND1_EVEN(S06, 6);
  ROUND1_ODD (S07, 7);
  ROUND1_EVEN(S08, 8);
  ROUND1_ODD (S09, 9);
  ROUND1_EVEN(S10,10);
  ROUND1_ODD (S11,11);
  ROUND1_EVEN(S12,12);
  ROUND1_ODD (S13,13);
  ROUND1_EVEN(S14,14);
  ROUND1_ODD (S15,15);
  ROUND1_EVEN(S16,16);
  ROUND1_ODD (S17,17);
  ROUND1_EVEN(S18,18);
  ROUND1_ODD (S19,19);
  ROUND1_EVEN(S20,20);
  ROUND1_ODD (S21,21);
  ROUND1_EVEN(S22,22);
  ROUND1_ODD (S23,23);
  ROUND1_EVEN(S24,24);
  ROUND1_ODD (S25,25);

  /* Begin round 2 of key expansion */
  ROUND2_EVEN(S00);
  ROUND2_ODD (S01);
  ROUND2_EVEN(S02);
  ROUND2_ODD (S03);
  ROUND2_EVEN(S04);
  ROUND2_ODD (S05);
  ROUND2_EVEN(S06);
  ROUND2_ODD (S07);
  ROUND2_EVEN(S08);
  ROUND2_ODD (S09);
  ROUND2_EVEN(S10);
  ROUND2_ODD (S11);
  ROUND2_EVEN(S12);
  ROUND2_ODD (S13);
  ROUND2_EVEN(S14);
  ROUND2_ODD (S15);
  ROUND2_EVEN(S16);
  ROUND2_ODD (S17);
  ROUND2_EVEN(S18);
  ROUND2_ODD (S19);
  ROUND2_EVEN(S20);
  ROUND2_ODD (S21);
  ROUND2_EVEN(S22);
  ROUND2_ODD (S23);
  ROUND2_EVEN(S24);
  ROUND2_ODD (S25);

  {
    register u32 eA, eB;
    /* Begin round 3 of key expansion (and encryption round) */

    eA = rc5unitwork->plain.lo + (A = ROTL3(S00 + A + Lhi));
    Llo = ROTL(Llo + A + Lhi, A + Lhi);
    eB = rc5unitwork->plain.hi + (A = ROTL3(S01 + A + Llo));
    Lhi = ROTL(Lhi + A + Llo, A + Llo);

    ROUND3_EVEN(S02);
    ROUND3_ODD (S03);
    ROUND3_EVEN(S04);
    ROUND3_ODD (S05);
    ROUND3_EVEN(S06);
    ROUND3_ODD (S07);
    ROUND3_EVEN(S08);
    ROUND3_ODD (S09);
    ROUND3_EVEN(S10);
    ROUND3_ODD (S11);
    ROUND3_EVEN(S12);
    ROUND3_ODD (S13);
    ROUND3_EVEN(S14);
    ROUND3_ODD (S15);
    ROUND3_EVEN(S16);
    ROUND3_ODD (S17);
    ROUND3_EVEN(S18);
    ROUND3_ODD (S19);
    ROUND3_EVEN(S20);
    ROUND3_ODD (S21);
    ROUND3_EVEN(S22);
    ROUND3_ODD (S23);

    eA = ROTL(eA ^ eB, eB) + (A = ROTL3(S24 + A + Lhi));
	
    return (rc5unitwork->cypher.lo == eA) &&
	    rc5unitwork->cypher.hi == ROTL(eB ^ eA, eA) +
	      ROTL3(S25 + A + ROTL(Llo + A + Lhi, A + Lhi));
  }
}

/* -------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" u32 rc5_ansi_1_b2_rg_unit_func ( RC5UnitWork *, u32 );
#endif

u32 rc5_ansi_1_b2_rg_unit_func ( RC5UnitWork * rc5unitwork, u32 iterations )
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
