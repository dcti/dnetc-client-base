/* 
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *r72_altivec_4_cpp(void) {
return "@(#)$Id: r72-altivec-4.cpp,v 1.1.2.1 2002/12/25 03:27:07 mfeiri Exp $"; }

#include "ccoreio.h"
#include "../ansi/rotate.h"

#define P 0xB7E15163
#define Q 0x9E3779B9

#ifdef __cplusplus
extern "C" s32 rc5_72_unit_func_altivec_4 ( RC5_72UnitWork *, u32 *, void * );
#endif

#if !defined(__VEC__)
#error Altivec is not supported
#endif


#define USE_STATIC_EXPANDED_TABLE   1


/*
** Altivec optimized code (MPC7400/MPC7410, aka G4).
** The following function checks 4 keys at once using the Altivec vector units.
** The iteration count and the first key to check are guaranteed to be even multiples
** of 24 (see problem.cpp/MINIMUM_ITERATIONS).
*/

typedef union {
	vector unsigned int v;
	unsigned int e[4];
} vec;


#define VECTOR(x) (vector unsigned int)((P)+(Q)*(x))

static const vector unsigned int exptable[26] = {
	VECTOR(0), VECTOR(1), VECTOR(2), VECTOR(3), VECTOR(4), VECTOR(5), VECTOR(6), VECTOR(7),
	VECTOR(8), VECTOR(9), VECTOR(10), VECTOR(11), VECTOR(12), VECTOR(13), VECTOR(14), VECTOR(15),
	VECTOR(16), VECTOR(17), VECTOR(18), VECTOR(19), VECTOR(20), VECTOR(21), VECTOR(22), VECTOR(23),
	VECTOR(24), VECTOR(25)
};


s32 rc5_72_unit_func_altivec_4 (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
	u32 i;
	u32 kiter = *iterations / 4;

	vector unsigned int  S0,  S1,  S2,  S3,  S4,  S5,  S6,  S7,  S8,  S9;
	vector unsigned int S10, S11, S12, S13, S14, S15, S16, S17, S18, S19;
	vector unsigned int S20, S21, S22, S23, S24, S25;
	vector unsigned int L0, L1, L2, tmp;
	const vector unsigned int rot3 = (vector unsigned int) (3);
	static vec T1, T2;
	static vector unsigned int plain_lo, plain_hi;
	
	T1.e[0] = rc5_72unitwork->plain.lo;
	T1.e[1] = rc5_72unitwork->plain.hi;
	plain_lo = vec_splat(T1.v, 0);
	plain_hi = vec_splat(T1.v, 1);
	
	while (kiter--) {
		L0  = (vector unsigned int) (0, 1, 2, 3);
        S0 = exptable[0];
   
		T1.e[2] = rc5_72unitwork->L0.hi;	
		T1.e[1] = rc5_72unitwork->L0.mid;
		T1.e[0] = rc5_72unitwork->L0.lo;

		tmp = T1.v;
		L2 = vec_add(L0, vec_splat(tmp, 2));	// L2 :=  key0.hi,  key1.hi,  key2.hi,  key3.hi
		L1 = vec_splat(tmp, 1);					// L1 :=  key0.mid, key1.mid, key2.mid, key3.mid
		L0 = vec_splat(tmp, 0);					// L0 :=  key0.lo,  key1.lo,  key2.lo,  key3.lo

		// First step : A = B = 0
		S1 = exptable[1];
		S0 = vec_rl(S0, rot3);					// A = S0 = ROTL3(S0)
		L0 = vec_add(L0, S0);					// x = L0 + A = L0 + S0
		L0 = vec_rl(L0, S0);					// B = L0 = ROTL(L0 + A, A)
		tmp = vec_add(S0, L0);					// tmp = A + B = S0 + L0

		/*
		** Mix the secret keys with the expanded key table.
		** The expanded key table is reloaded from memory. Since
		** this involves the LSU, the Altivec units are free to
		** process some work.
		*/
		
        #define MIX(Sx, Lx1, Lx2) 				 	\
		    Sx = vec_rl(vec_add(Sx, tmp), rot3); 	\
			tmp = vec_add(Lx1, Sx);  			 	\
			Lx2 = vec_rl(vec_add(Lx2, tmp), tmp);	\
			tmp = vec_add(Lx2, Sx);	

		S2  = exptable[2];  MIX( S1, L0, L1);
		S3  = exptable[3];  MIX( S2, L1, L2);
		S4  = exptable[4];  MIX( S3, L2, L0);
		S5  = exptable[5];  MIX( S4, L0, L1);
		S6  = exptable[6];  MIX( S5, L1, L2);
		S7  = exptable[7];  MIX( S6, L2, L0);
		S8  = exptable[8];  MIX( S7, L0, L1);
		S9  = exptable[9];  MIX( S8, L1, L2);
		S10 = exptable[10]; MIX( S9, L2, L0);
		S11 = exptable[11]; MIX(S10, L0, L1);
		S12 = exptable[12]; MIX(S11, L1, L2);
		S13 = exptable[13]; MIX(S12, L2, L0);
		S14 = exptable[14]; MIX(S13, L0, L1);
		S15 = exptable[15]; MIX(S14, L1, L2);
		S16 = exptable[16]; MIX(S15, L2, L0);
		S17 = exptable[17]; MIX(S16, L0, L1);
		S18 = exptable[18]; MIX(S17, L1, L2);
		S19 = exptable[19]; MIX(S18, L2, L0);
		S20 = exptable[20]; MIX(S19, L0, L1);
		S21 = exptable[21]; MIX(S20, L1, L2);
		S22 = exptable[22]; MIX(S21, L2, L0);
		S23 = exptable[23]; MIX(S22, L0, L1);
		S24 = exptable[24]; MIX(S23, L1, L2);
		S25 = exptable[25]; MIX(S24, L2, L0);
	    MIX(S25, L0, L1);

		// Second pass
		MIX(S0, L1, L2);
		MIX(S1, L2, L0);
		MIX(S2, L0, L1);
		MIX(S3, L1, L2);
		MIX(S4, L2, L0);
		MIX(S5, L0, L1);
		MIX(S6, L1, L2);
		MIX(S7, L2, L0);
		MIX(S8, L0, L1);
		MIX(S9, L1, L2);
		MIX(S10, L2, L0);
		MIX(S11, L0, L1);
		MIX(S12, L1, L2);
		MIX(S13, L2, L0);
		MIX(S14, L0, L1);
		MIX(S15, L1, L2);
		MIX(S16, L2, L0);
		MIX(S17, L0, L1);
		MIX(S18, L1, L2);
		MIX(S19, L2, L0);
		MIX(S20, L0, L1);
		MIX(S21, L1, L2);
		MIX(S22, L2, L0);
		MIX(S23, L0, L1);
		MIX(S24, L1, L2);
		MIX(S25, L2, L0);

		// Third pass
		MIX(S0, L0, L1);
		MIX(S1, L1, L2);
		MIX(S2, L2, L0);
		MIX(S3, L0, L1);
		MIX(S4, L1, L2);
		MIX(S5, L2, L0);
		MIX(S6, L0, L1);
		MIX(S7, L1, L2);
		MIX(S8, L2, L0);
		MIX(S9, L0, L1);
		MIX(S10, L1, L2);
		MIX(S11, L2, L0);
		MIX(S12, L0, L1);
		MIX(S13, L1, L2);
		MIX(S14, L2, L0);
		MIX(S15, L0, L1);
		MIX(S16, L1, L2);
		MIX(S17, L2, L0);
		MIX(S18, L0, L1);
		MIX(S19, L1, L2);
		MIX(S20, L2, L0);
		MIX(S21, L0, L1);
		MIX(S22, L1, L2);
		MIX(S23, L2, L0);
		MIX(S24, L0, L1);
		S25 = vec_rl(vec_add(S25, tmp), rot3);

		L2 = plain_lo;
		tmp = plain_hi;
		S0 = vec_add(S0, L2);				// == A
		S1 = vec_add(S1, tmp);				// == B

		/*
		** Cypher pass.
		*/
        #define ROUND(Sx1, Sx2)						\
		    tmp = vec_xor(S0, S1);					\
			S0 = vec_add(Sx1, vec_rl(tmp, S1));		\
			tmp = vec_xor(S1, S0);					\
			S1 = vec_add(Sx2, vec_rl(tmp, S0));			

		ROUND(S2, S3);
		ROUND(S4, S5);
		ROUND(S6, S7);
		ROUND(S8, S9);
		ROUND(S10, S11);
		ROUND(S12, S13);
		ROUND(S14, S15);
		ROUND(S16, S17);
		ROUND(S18, S19);
		ROUND(S20, S21);
		ROUND(S22, S23);
		ROUND(S24, S25);

		T1.v = S0;			// A
		T2.v = S1;			// B

		for (i = 0; i < 4; i++) {
			if (T1.e[i] == rc5_72unitwork->cypher.lo) {
				++rc5_72unitwork->check.count;
				rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + i;
				rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
				rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;

				if (T2.e[i] == rc5_72unitwork->cypher.hi) {
					*iterations -= (kiter + 1) * 4 - i;
					return RESULT_FOUND;
				}
			}
		}

		/*
		** Compute the next key to be checked.
		** The speed gain provided by the assembly code is marginal.
		*/
		
		if ( !(rc5_72unitwork->L0.hi = (rc5_72unitwork->L0.hi + 4) & 0x000000FF)) {
			u32 *mid = &rc5_72unitwork->L0.mid;
			u32 *lo  = &rc5_72unitwork->L0.lo;
			{ asm ("lwbrx  r4,0,%0\n"			// L0.mid (translated to big-endian)
		           "lwbrx  r5,0,%1\n"			// L0.lo  (translated to big-endian)
		           "addic  r4,r4,1\n"
		           "addze  r5,r5\n"
		           "stwbrx r4,0,%0\n"			// write back L0.mid (little-endian)
		           "stwbrx r5,0,%1"				// write back L0.lo  (little-endian)
		           : /* no output */ : "r" (mid), "r" (lo) : "r4", "r5");
		    }
		}
	}
	return RESULT_NOTHING;
}
