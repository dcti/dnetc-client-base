/*
 * Copyright distributed.net 1997 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ---------------------------------------------------------------
 *
 * extern "C" s32 rc5_unit_func_ansi_1_b2( RC5UnitWork *work,
 *			u32 *timeslice, void *scratch_area  );
 *             //returns RESULT_FOUND,RESULT_WORKING or -1,
 * ---------------------------------------------------------------
*/

#if (!defined(lint) && defined(__showids__))
const char *rc5ansi_1_b2_cpp (void) {
return "@(#)$Id: rc5ansi_1-b2.cpp,v 1.1.2.1 2000/01/06 11:43:33 patrick Exp $"; }
#endif

#define PIPELINE_COUNT = 1
#define USE_ANSI_INCREMENT

#include "problem.h"
#include "rotate.h"

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

#if defined(__cplusplus)
extern "C" s32 rc5_unit_func_ansi_1_b2( RC5UnitWork *work, 
				u32 *timeslice, void *scratch_area );
#endif


// rc5_unit will get passed an RC5WorkUnit to complete
// this is where all the actually work occurs, this is where you optimize.
// assembly gurus encouraged.
// Returns: 0 - nothing found, 1 - found on pipeline 1,
//   2 - found pipeline 2, 3 - ... etc ...
// since this core is for a single pipeline only, iterations == keystocheck !

s32 rc5_unit_func_ansi_1_b2( RC5UnitWork *work, u32 *timeslice, 
						void *scratch_area )
{
  register u32 S00,S01,S02,S03,S04,S05,S06,S07,S08,S09,S10,S11,S12,
    S13,S14,S15,S16,S17,S18,S19,S20,S21,S22,S23,S24,S25;

  register u32 A, Llo, Lhi;

  u32 kiter = 0;
  u32 keycount = *timeslice;

  while ( keycount-- ) {// timeslice ignores the number of pipelines
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
   	
       if (rc5unitwork->cypher.lo == eA) &&
   	    rc5unitwork->cypher.hi == ROTL(eB ^ eA, eA) +
   	      ROTL3(S25 + A + ROTL(Llo + A + Lhi, A + Lhi))
	  break;		// found a key
     }
    // "mangle-increment" the key number by the number of pipelines
    mangle_increment(rc5unitwork);
    kiter += PIPELINE_COUNT;
   }
  }
  if ( kiter == *timeslice ) { /* tested all */
	return RESULT_NOTHING;
  } else if ( kiter < *timeslice ) {
	*timeslice = kiter;	/* save how many we actually did */
	return RESULT_FOUND;
  } 
  /* this coude will never be reached and is mostly to satisfy the compiler */
  scratch_area = scratch_area; /* unused arg. shaddup compiler */
  return -1; /* error */
}
