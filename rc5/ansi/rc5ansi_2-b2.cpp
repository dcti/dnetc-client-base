/*
 * Copyright distributed.net 1997 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ---------------------------------------------------------------
 *
 * extern "C" s32 rc5_unit_func_ansi_2_b2( RC5UnitWork *work,
 *                      u32 *timeslice, void *scratch_area  );
 *             //returns RESULT_FOUND,RESULT_WORKING or -1,
 * ---------------------------------------------------------------
*/

#if (!defined(lint) && defined(__showids__))
const char *rc5ansi_2_b2_cpp (void) {
return "@(#)$Id: rc5ansi_2-b2.cpp,v 1.1.2.1 2000/01/06 12:06:55 patrick Exp $"; }
#endif

#define PIPELINE_COUNT = 2
#define USE_ANSI_INCREMENT

#include "problem.h"
#include "rotate.h"


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

#define ROUND3EVEN(N) ROUND2EVEN(N)

#define ROUND3ODD(N)  ROUND2ODD(N)

#define ENCRYPT(N,A1,B1,A2,B2) \
    A1 = ROTL(A1 ^ B1, B1) + S1[N];    \
    A2 = ROTL(A2 ^ B2, B2) + S2[N];

#define ENCRYPTEVEN(N)  ENCRYPT(N,A1,B1,A2,B2)
#define ENCRYPTODD(N)   ENCRYPT(N,B1,A1,B2,A2)

#if defined(__cplusplus)
extern "C" s32 rc5_unit_func_ansi_2_b2( RC5UnitWork *work,
                                u32 *timeslice, void *scratch_area );
#endif


// rc5_unit will get passed an RC5WorkUnit to complete
// this is where all the actually work occurs, this is where you optimize.
// assembly gurus encouraged.
// Returns: 0 - nothing found, 1 - found on pipeline 1,
//   2 - found pipeline 2, 3 - ... etc ...
// since this core is for a dual pipeline, iterations == keystocheck/2 !

s32 rc5_unit_func_ansi_2_b2( RC5UnitWork *work, u32 *timeslice,
                                                void *scratch_area )
{
/* these tests could be done to assure the correctness of the data:
 *
 * the number of keys to check has to be a multiple of the number of pipelines
  if (*timeslice % PIPELINE_COUNT) 
	return -1; 
 * to my knowledge the KEY itself is not required to fit ! 
 */

  u32 kiter = 0;
  u32 iterations = *timeslice/PIPELINE_COUNT;

  while ( iterations-- ) 
   {
     static u32 S1[26], S2[26];
     {
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
   
       /* Begin round 3 of key expansion */
   
       ROUND3EVEN(0)
       ROUND3ODD (1)
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
       ROUND3EVEN(24)
   
       /*  Special case, don't need to compute B, or double assign to A.  */
       S1[25] = ROTL3(S1[25] + A1 + Llo1);
       S2[25] = ROTL3(S2[25] + A2 + Llo2);
   
     }
     {
       /* Begin the encryption */
       register u32 A1, B1;
       register u32 A2, B2;
   
       A1 = rc5unitwork->plain.lo + S1[0];
       A2 = rc5unitwork->plain.lo + S2[0];
       B1 = rc5unitwork->plain.hi + S1[1];
       B2 = rc5unitwork->plain.hi + S2[1];
   
       ENCRYPTEVEN(2)
       ENCRYPTODD (3)
       ENCRYPTEVEN(4)
       ENCRYPTODD (5)
       ENCRYPTEVEN(6)
       ENCRYPTODD (7)
       ENCRYPTEVEN(8)
       ENCRYPTODD (9)
       ENCRYPTEVEN(10)
       ENCRYPTODD (11)
       ENCRYPTEVEN(12)
       ENCRYPTODD (13)
       ENCRYPTEVEN(14)
       ENCRYPTODD (15)
       ENCRYPTEVEN(16)
       ENCRYPTODD (17)
       ENCRYPTEVEN(18)
       ENCRYPTODD (19)
       ENCRYPTEVEN(20)
       ENCRYPTODD (21)
       ENCRYPTEVEN(22)
       ENCRYPTODD (23)
       ENCRYPTEVEN(24)
   
   
       /* an 'if' is less expensive than a rotation, which we
        * will avoid if the first part fails.
        */
   
       if (rc5unitwork->cypher.lo == A1 &&
             rc5unitwork->cypher.hi == ROTL(B1 ^ A1, A1) + S1[25]) break;
       if (rc5unitwork->cypher.lo == A2 &&
             rc5unitwork->cypher.hi == ROTL(B2 ^ A2, A2) + S2[25]) {
		kiter++;
		break;
	}
     }
     // "mangle-increment" the key number by the number of pipelines
     mangle_increment(rc5unitwork);
     kiter += PIPELINE_COUNT;
   }
  if ( kiter == *timeslice ) { /* tested all */
        return RESULT_NOTHING;
  } else if ( kiter < *timeslice ) {
        *timeslice = kiter;     /* save how many we actually did */
        return RESULT_FOUND;
  }
  /* this code will never be reached and is mostly to satisfy the compiler */
  scratch_area = scratch_area; /* unused arg. shaddup compiler */
  return -1; /* error */
}   
