/*
 * Copyright distributed.net 1997 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ---------------------------------------------------------------
 * dual-key, mixed round 3 and encryption, A1/A2 use for last value,
 * non-arrayed S1/S2 tables, run-time generation of S0[]
 *
 * extern "C" s32 rc5_ansi_rg_unified_form( RC5UnitWork *work, 
 *                            u32 *timeslice, void *scratch_area );
 *            //returns RESULT_FOUND,RESULT_WORKING or -1,
 *
 * extern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *, u32 timeslice );
 *            //returns timeslice
 * ---------------------------------------------------------------
*/

#if (!defined(lint) && defined(__showids__))
const char *rc5ansi2_rg_cpp (void) {
return "@(#)$Id: 2-rg.c,v 1.4.2.3 2000/02/08 21:57:36 remi Exp $"; }
#endif

#include "cputypes.h"
#include "ccoreio.h"
#include "rotate.h"

/*
 *
 * Run-time generation of S0[] :
 *
 *	- loading a large constant on RISC need two instructions.
 *	  (ie, on sparc :)
 *		sethi %hi(1444465436),%g2
 *		or %g2,%lo(1444465436),%g2
 *
 *	- generating S0[] at run time need only one instruction
 *	  since S0[n] = S0[n-1] + Q
 *	  (ie, : currentS0 += Q )
 *
 *	- drawback : we need two more registers
 *	  one for 'currentS0' and one for 'Q'
 *
 * some chips can't do two shifts at once
 *	- Sparcs
 *	- Alphas
 *
 * out of order :
 *	- ppc604
 *	- R10000
 *	- PA8000
 * in order :
 *	- all others
*/

#define _P_RC5   0xB7E15163u
#define _Q       0x9E3779B9u
#define S_not(n) _P_RC5+_Q*n

/* Round 1 macros */
/* -------------- */

#define ROUND1EVEN(S1N, S2N)	\
    cS0 += Q;			\
    A1 += cS0;			\
    A2 += cS0;			\
    A1 += Lhi1;			\
    A2 += Lhi2;			\
    A1 = ROTL3(A1);		\
    A2 = ROTL3(A2);		\
    S1N = A1;			\
    tmp1 = A1 + Lhi1;		\
    S2N = A2;			\
    tmp2 = A2 + Lhi2;		\
    Llo1 += tmp1;		\
    Llo2 += tmp2;		\
    Llo1 = ROTL(Llo1, tmp1);	\
    Llo2 = ROTL(Llo2, tmp2);

#define  ROUND1ODD(S1N, S2N)	\
    cS0 += Q;			\
    A1 += cS0;			\
    A2 += cS0;			\
    A1 += Llo1;			\
    A2 += Llo2;			\
    A1 = ROTL3(A1);		\
    A2 = ROTL3(A2);		\
    S1N = A1;			\
    tmp1 = A1 + Llo1;		\
    S2N = A2;			\
    tmp2 = A2 + Llo2;		\
    Lhi1 += tmp1;		\
    Lhi2 += tmp2;		\
    Lhi1 = ROTL(Lhi1, tmp1);	\
    Lhi2 = ROTL(Lhi2, tmp2);


/* Round 2 macros */
/* -------------- */

#define ROUND2EVEN(S1N, S2N)	\
    tmp1 = S1N;			\
    A1 += Lhi1;			\
    tmp2 = S2N;			\
    A2 += Lhi2;			\
    A1 += tmp1;			\
    A2 += tmp2;			\
    A1 = ROTL3(A1);		\
    A2 = ROTL3(A2);		\
    S1N = A1;			\
    tmp1 = A1 + Lhi1;		\
    S2N = A2;			\
    tmp2 = A2 + Lhi2;		\
    Llo1 += tmp1;		\
    Llo2 += tmp2;		\
    Llo1 = ROTL(Llo1,tmp1);	\
    Llo2 = ROTL(Llo2,tmp2)

#define  ROUND2ODD(S1N, S2N)	\
    tmp1 = S1N;			\
    A1 += Llo1;			\
    tmp2 = S2N;			\
    A2 += Llo2;			\
    A1 += tmp1;			\
    A2 += tmp2;			\
    A1 = ROTL3(A1);		\
    A2 = ROTL3(A2);		\
    S1N = A1;			\
    tmp1 = A1 + Llo1;		\
    S2N = A2;			\
    tmp2 = A2 + Llo2;		\
    Lhi1 += tmp1;		\
    Lhi2 += tmp2;		\
    Lhi1 = ROTL(Lhi1,tmp1);	\
    Lhi2 = ROTL(Lhi2,tmp2)

/* Round 3 macros */
/* -------------- */

#define ROUND3EVEN(S1N, S2N)	\
    tmp1 = S1N;			\
    A1 += Lhi1;			\
    tmp2 = S2N;			\
    A2 += Lhi2;			\
    A1 += tmp1;			\
    A2 += tmp2;			\
    A1 = ROTL3(A1);		\
    eA1 ^= eB1;			\
    A2 = ROTL3(A2);		\
    eA2 ^= eB2;			\
    eA1 = ROTL(eA1,eB1);	\
    eA2 = ROTL(eA2,eB2);	\
    eA1 += A1;			\
    eA2 += A2;			\
    tmp1 = A1 + Lhi1;		\
    tmp2 = A2 + Lhi2;		\
    Llo1 += tmp1;		\
    Llo2 += tmp2;		\
    Llo1 = ROTL(Llo1,tmp1);	\
    Llo2 = ROTL(Llo2,tmp2);
	
#define ROUND3ODD(S1N, S2N)	\
    tmp1 = S1N;			\
    A1 += Llo1;			\
    tmp2 = S2N;			\
    A2 += Llo2;			\
    A1 += tmp1;			\
    A2 += tmp2;			\
    A1 = ROTL3(A1);		\
    eB1 ^= eA1;			\
    A2 = ROTL3(A2);		\
    eB2 ^= eA2;			\
    eB1 = ROTL(eB1,eA1);	\
    eB2 = ROTL(eB2,eA2);	\
    eB1 += A1;			\
    eB2 += A2;			\
    tmp1 = A1 + Llo1;		\
    tmp2 = A2 + Llo2;		\
    Lhi1 += tmp1;		\
    Lhi2 += tmp2;		\
    Lhi1 = ROTL(Lhi1,tmp1);	\
    Lhi2 = ROTL(Lhi2,tmp2);

/* -------------------------------------------------------------------- */

#if defined(__cplusplus)
extern "C" u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *rc5unitwork, u32 tslice );
#endif

u32 rc5_ansi_2_rg_unit_func( RC5UnitWork *rc5unitwork, u32 timeslice )
{
  u32 kiter = 0;
  int keycount=(int)timeslice;
  u32 S1_00,S1_01,S1_02,S1_03,S1_04,S1_05,S1_06,S1_07,S1_08,S1_09,
      S1_10,S1_11,S1_12,S1_13,S1_14,S1_15,S1_16,S1_17,S1_18,S1_19,
      S1_20,S1_21,S1_22,S1_23,S1_24,S1_25;

  u32 S2_00,S2_01,S2_02,S2_03,S2_04,S2_05,S2_06,S2_07,S2_08,S2_09,
      S2_10,S2_11,S2_12,S2_13,S2_14,S2_15,S2_16,S2_17,S2_18,S2_19,
      S2_20,S2_21,S2_22,S2_23,S2_24,S2_25;

  register u32 A1, Llo1, Lhi1;
  register u32 A2, Llo2, Lhi2;
  register u32 tmp1, tmp2;

  while ( keycount-- ) /* timeslice ignores the number of pipelines */
  {
    Llo2 = Llo1 = rc5unitwork->L0.lo;
    Lhi2 = (Lhi1 = rc5unitwork->L0.hi) + 0x01000000;
  
    /* Begin round 1 of key expansion */
  
    {  register u32 cS0, Q;
  
      /*  Special case while A and B are known to be zero.  */
      cS0 = _P_RC5;
      Q = _Q;
  
      S1_00 = A1 =
      S2_00 = A2 = ROTL3(cS0);
      Llo1 = ROTL(Llo1 + A1, A1);
      Llo2 = ROTL(Llo2 + A2, A2);
  
      ROUND1ODD  (S1_01, S2_01);
      ROUND1EVEN (S1_02, S2_02);
      ROUND1ODD  (S1_03, S2_03);
      ROUND1EVEN (S1_04, S2_04);
      ROUND1ODD  (S1_05, S2_05);
      ROUND1EVEN (S1_06, S2_06);
      ROUND1ODD  (S1_07, S2_07);
      ROUND1EVEN (S1_08, S2_08);
      ROUND1ODD  (S1_09, S2_09);
      ROUND1EVEN (S1_10, S2_10);
      ROUND1ODD  (S1_11, S2_11);
      ROUND1EVEN (S1_12, S2_12);
      ROUND1ODD  (S1_13, S2_13);
      ROUND1EVEN (S1_14, S2_14);
      ROUND1ODD  (S1_15, S2_15);
      ROUND1EVEN (S1_16, S2_16);
      ROUND1ODD  (S1_17, S2_17);
      ROUND1EVEN (S1_18, S2_18);
      ROUND1ODD  (S1_19, S2_19);
      ROUND1EVEN (S1_20, S2_20);
      ROUND1ODD  (S1_21, S2_21);
      ROUND1EVEN (S1_22, S2_22);
      ROUND1ODD  (S1_23, S2_23);
      ROUND1EVEN (S1_24, S2_24);
      ROUND1ODD  (S1_25, S2_25);
    }
  
  
    /* Begin round 2 of key expansion */
  				
    ROUND2EVEN(S1_00, S2_00);
    ROUND2ODD (S1_01, S2_01);
    ROUND2EVEN(S1_02, S2_02);
    ROUND2ODD (S1_03, S2_03);
    ROUND2EVEN(S1_04, S2_04);
    ROUND2ODD (S1_05, S2_05);
    ROUND2EVEN(S1_06, S2_06);
    ROUND2ODD (S1_07, S2_07);
    ROUND2EVEN(S1_08, S2_08);
    ROUND2ODD (S1_09, S2_09);
    ROUND2EVEN(S1_10, S2_10);
    ROUND2ODD (S1_11, S2_11);
    ROUND2EVEN(S1_12, S2_12);
    ROUND2ODD (S1_13, S2_13);
    ROUND2EVEN(S1_14, S2_14);
    ROUND2ODD (S1_15, S2_15);
    ROUND2EVEN(S1_16, S2_16);
    ROUND2ODD (S1_17, S2_17);
    ROUND2EVEN(S1_18, S2_18);
    ROUND2ODD (S1_19, S2_19);
    ROUND2EVEN(S1_20, S2_20);
    ROUND2ODD (S1_21, S2_21);
    ROUND2EVEN(S1_22, S2_22);
    ROUND2ODD (S1_23, S2_23);
    ROUND2EVEN(S1_24, S2_24);
    ROUND2ODD (S1_25, S2_25);
  
    {
      register u32 eA1, eB1, eA2, eB2;
      /* Begin round 3 of key expansion (and encryption round) */
  
      eA1 = rc5unitwork->plain.lo + (A1 = ROTL3(S1_00 + Lhi1 + A1));
      eA2 = rc5unitwork->plain.lo + (A2 = ROTL3(S2_00 + Lhi2 + A2));
      Llo1 = ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1);
      Llo2 = ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2);
  
      eB1 = rc5unitwork->plain.hi + (A1 = ROTL3(S1_01 + Llo1 + A1));
      eB2 = rc5unitwork->plain.hi + (A2 = ROTL3(S2_01 + Llo2 + A2));
      Lhi1 = ROTL(Lhi1 + A1 + Llo1, A1 + Llo1);
      Lhi2 = ROTL(Lhi2 + A2 + Llo2, A2 + Llo2);
  				
      ROUND3EVEN(S1_02, S2_02);
      ROUND3ODD (S1_03, S2_03);
      ROUND3EVEN(S1_04, S2_04);
      ROUND3ODD (S1_05, S2_05);
      ROUND3EVEN(S1_06, S2_06);
      ROUND3ODD (S1_07, S2_07);
      ROUND3EVEN(S1_08, S2_08);
      ROUND3ODD (S1_09, S2_09);
      ROUND3EVEN(S1_10, S2_10);
      ROUND3ODD (S1_11, S2_11);
      ROUND3EVEN(S1_12, S2_12);
      ROUND3ODD (S1_13, S2_13);
      ROUND3EVEN(S1_14, S2_14);
      ROUND3ODD (S1_15, S2_15);
      ROUND3EVEN(S1_16, S2_16);
      ROUND3ODD (S1_17, S2_17);
      ROUND3EVEN(S1_18, S2_18);
      ROUND3ODD (S1_19, S2_19);
      ROUND3EVEN(S1_20, S2_20);
      ROUND3ODD (S1_21, S2_21);
      ROUND3EVEN(S1_22, S2_22);
      ROUND3ODD (S1_23, S2_23);
  
      eA1 = ROTL(eA1 ^ eB1, eB1) + (A1 = ROTL3(S1_24 + A1 + Lhi1));
      eA2 = ROTL(eA2 ^ eB2, eB2) + (A2 = ROTL3(S2_24 + A2 + Lhi2));
  	
      if (rc5unitwork->cypher.lo == eA1 &&
  	    rc5unitwork->cypher.hi == ROTL(eB1 ^ eA1, eA1) +
  	      ROTL3(S1_25 + A1 + ROTL(Llo1 + A1 + Lhi1, A1 + Lhi1))) return kiter;
      if (rc5unitwork->cypher.lo == eA2 &&
  	    rc5unitwork->cypher.hi == ROTL(eB2 ^ eA2, eA2) +
  	      ROTL3(S2_25 + A2 + ROTL(Llo2 + A2 + Lhi2, A2 + Lhi2))) return ++kiter;
    }
    /* "mangle-increment" the key number by the number of pipelines */
    /* (2 in this case) - didn't like to change the whole thing */
    #define key rc5unitwork->L0
    key.hi = (key.hi + ( 2 << 24)) & 0xFFFFFFFFu;
    if (!(key.hi & 0xFF000000u))
    {
      key.hi = (key.hi + 0x00010000) & 0x00FFFFFF;
      if (!(key.hi & 0x00FF0000))
      {
        key.hi = (key.hi + 0x00000100) & 0x0000FFFF;
        if (!(key.hi & 0x0000FF00))
        {
          key.hi = (key.hi + 0x00000001) & 0x000000FF;
	  /* we do not need to mask here, was done above */
          if (!(key.hi))
          {
            key.lo = key.lo + 0x01000000;
            if (!(key.lo & 0xFF000000u))
            {
              key.lo = (key.lo + 0x00010000) & 0x00FFFFFF;
              if (!(key.lo & 0x00FF0000))
              {
                key.lo = (key.lo + 0x00000100) & 0x0000FFFF;
                if (!(key.lo & 0x0000FF00))
                {
                  key.lo = (key.lo + 0x00000001) & 0x000000FF;
                }
              }
            }
          }
        }
      }
    }

    kiter += 2;
  }
  return kiter;
}

#if defined(__cplusplus)
extern "C" s32 rc5_ansi_2_rg_unified_form( RC5UnitWork * work,
                                u32 *timeslice, void *scratch_area );
#endif

s32 rc5_ansi_rg_unified_form( RC5UnitWork *work, 
                              u32 *timeslice, void *scratch_area )
{
  /*  this is a two pipeline core, so ... iterations_to_do == timeslice / 2
   *                              and ... iterations_done  == retval * 2
  */                                
  u32 kiter, xiter = (*timeslice / 2);
  scratch_area = scratch_area; /* shaddup compiler */

  kiter = rc5_ansi_2_rg_unit_func( work, xiter );
  *timeslice = kiter * 2;
  
  if (xiter == kiter) {
    return RESULT_WORKING;
  } else if (xiter < kiter) {
    return RESULT_FOUND;
  } 

  return -1; /* error */
}

