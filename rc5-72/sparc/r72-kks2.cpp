/* 
** Copyright distributed.net 1997-2003 - All Rights Reserved
** For use in distributed.net projects only.
** Any other distribution or use of this source violates copyright.
**
** Experimental 2-pipes core for Sparc processors written by
** Didier Levet (kakace@wanadoo.fr)
**
** In order to ease the tests, this core is designed to replace the
** ANSI 2-pipe core.
**
** Inspired in part by the RC5-64 sparc core by Lawrence Butcher
** (lbutcher@eng.sun.com), Remi Guyomarch (rguyom@mail.dotcom.fr), et. al.
**
*/

#include "ccoreio.h"
#include "rotate.h"

#define P 0xB7E15163
#define Q 0x9E3779B9

#ifdef __cplusplus
extern "C" s32 CDECL rc5_72_unit_func_KKS_2 ( RC5_72UnitWork *, u32 *, void * );
#endif


/*
** Inline assembly macros.
** Note that PowerPC CPUs use the lower 6 bits as shift count
** when using slw/srw instructions, so the shift count shall
** be masked to [0; 31] values.
*/

#if defined(__POWERPC__)

	#define ASM_ADD(res, op1, op2)		\
		__asm__ volatile ("add %0,%1,%2" : "=r" (res) : "r" (op1), "r" (op2));
	#define ASM_COMP(res, op)	\
		__asm__ volatile ("subfic %0,%1,32" : "=r" (res) : "r" (op));
	#define ASM_OR(res, op1, op2)		\
		__asm__ volatile ("or %0,%1,%2" : "=r" (res) : "r" (op1), "r" (op2));
	#define ASM_XOR(res, op1, op2)	\
		__asm__ volatile ("xor %0,%1,%2" : "=r" (res) : "r" (op1), "r" (op2));
	#define ASM_SL(res, op, dist)		\
		__asm__ volatile ("clrlwi %3,%2,27\nslw %0,%1,%3" : "=r" (res) : "r" (op), "r" (dist), "r" (LShft));
	#define ASM_SR(res, op, dist)		\
		__asm__ volatile ("clrlwi %3,%2,27\nsrw %0,%1,%3" : "=r" (res) : "r" (op), "r" (dist), "r" (LShft));
	#define ASM_SL3(res, op)		\
		__asm__ volatile ("slwi %0,%1,3" : "=r" (res) : "r" (op));
	#define ASM_SR29(res, op)		\
		__asm__ volatile ("srwi %0,%1,29" : "=r" (res) : "r" (op));

#elsif (0 == 1)
/* ASM Macros, requires gcc to compile */
	#define ASM_ADD(res, op1, op2)		\
		__asm__ volatile ("add %1, %2, %0" : "=r" (res) : "r" (op1), "r" (op2));
	#define ASM_COMP(res, wsize, op)	\
		__asm__ volatile ("sub %1, %2, %0" : "=r" (res) : "rc" (wsize), "rc" (op));
	#define ASM_OR(res, op1, op2)		\
		__asm__ volatile ("or %1, %2, %0" : "=r" (res) : "r" (op1), "r" (op2));
	#define ASM_XOR(res, op1, op2)	\
		__asm__ volatile ("xor %1, %2, %0" : "=r" (res) : "r" (op1), "r" (op2));
	#define ASM_SL(res, op, dist)		\
		__asm__ volatile ("sll %1, %2, %0" : "=r" (res) : "r" (op), "rc" (dist));
	#define ASM_SR(res, op, dist)		\
		__asm__ volatile ("srl %1, %2, %0" : "=r" (res) : "r" (op), "rc" (dist));
	#define ASM_SL3(res, op)		\
		__asm__ volatile ("sll %1, 3, %0" : "=r" (res) : "r" (op));
	#define ASM_SR29(res, op)		\
		__asm__ volatile ("srl %1, 29, %0" : "=r" (res) : "r" (op));

#else
/* C macros, compiles faster with sun compiler than asm macros */
        #define ASM_ADD(res, op1, op2)          \
		res = op1 + op2;
        #define ASM_COMP(res, wsize, op)        \
		res = wsize - op;
        #define ASM_OR(res, op1, op2)           \
		res = op1 | op2;
        #define ASM_XOR(res, op1, op2)  \
		res = op1 ^ op2;
        #define ASM_SL(res, op, dist)           \
		res = op << dist;
        #define ASM_SR(res, op, dist)           \
		res = op >> dist;
        #define ASM_SL3(res, op)                \
		res = op << 3;
        #define ASM_SR29(res, op)               \
		res = op >> 29;

#endif


/*
** MIX_1 macro :
** PIPE A						PIPE B
** Ta = Aa + Ba					Tb = Ab + Bb
** Sk += Q
** La += Ta						Lb += Tb
** Ba = La = ROTL(La, Ta)		Bb = Lb = ROTL(Lb, Tb)
** Sa = Sk + Aa					Sb = Sk + Aa
** Sa += Ba						Sb += Bb
** Aa = Sa = ROTL3(Sa)			Ab = Sb = ROTL3(Sb)
** Sa[index] = Aa				Sb[index] = Ab
*/

#define MIX_1(index, Aa, Ab, La, Lb, Ba, Bb)	\
		ASM_ADD(Tmp, Aa, Ba)		\
		ASM_ADD(Sk, Sk, q)		\
		ASM_ADD(TRes, La, Tmp)		\
		ASM_COMP(RShft, word_size, Tmp)	\
		ASM_SL(La, TRes, Tmp)		\
		ASM_SR(TRes, TRes, RShft)	\
		ASM_ADD(Tmp, Ab, Bb)		\
		ASM_OR(La, La, TRes)		\
		ASM_ADD(TRes, Lb, Tmp)		\
		ASM_COMP(RShft, word_size, Tmp)	\
		ASM_SL(Lb, TRes, Tmp)		\
		ASM_SR(TRes, TRes, RShft)	\
		ASM_ADD(Aa, Aa, Sk)		\
		ASM_ADD(Ab, Ab, Sk)		\
		ASM_OR(Lb, Lb, TRes)		\
		ASM_ADD(Aa, Aa, La)		\
		ASM_ADD(Ab, Ab, Lb)		\
		ASM_SL3(TRes, Aa)		\
		ASM_SL3(Tmp, Ab)		\
		ASM_SR29(Aa, Aa)		\
		ASM_SR29(Ab, Ab)		\
		ASM_OR(Aa, Aa, TRes)		\
		ASM_OR(Ab, Ab, Tmp)		\
		Sa[index] = Aa;			\
		Sb[index] = Ab


/*
** MIX_2 macro :
** PIPE A						PIPE B
** Sk = Sa[index]				q = Sb[index]
** Ta = Aa + Ba					Tb = Ab + Bb
** La += Ta						Lb += Tb
** Ba = La = ROTL(La, Ta)		Bb = Lb = ROTL(Lb, Tb)
** Sa = Sk + Aa					Sb = q + Aa
** Sa += Ba						Sb += Bb
** Aa = Sa = ROTL3(Sa)			Ab = Sb = ROTL3(Sb)
** Sa[index] = Aa				Sb[index] = Ab
*/

#define MIX_2(index, Aa, Ab, La, Lb, Ba, Bb)	\
		Sk = Sa[index];			\
		ASM_ADD(Tmp, Aa, Ba)		\
		q = Sb[index];			\
		ASM_ADD(TRes, La, Tmp)		\
		ASM_COMP(RShft, word_size, Tmp)	\
		ASM_SL(La, TRes, Tmp)		\
		ASM_SR(TRes, TRes, RShft)	\
		ASM_ADD(Tmp, Ab, Bb)		\
		ASM_OR(La, La, TRes)		\
		ASM_ADD(TRes, Lb, Tmp)		\
		ASM_COMP(RShft, word_size, Tmp)	\
		ASM_SL(Lb, TRes, Tmp)		\
		ASM_SR(TRes, TRes, RShft)	\
		ASM_ADD(Aa, Aa, Sk)		\
		ASM_ADD(Ab, Ab, q)		\
		ASM_OR(Lb, Lb, TRes)		\
		ASM_ADD(Aa, Aa, La)		\
		ASM_ADD(Ab, Ab, Lb)		\
		ASM_SL3(TRes, Aa)		\
		ASM_SL3(Tmp, Ab)		\
		ASM_SR29(Aa, Aa)		\
		ASM_SR29(Ab, Ab)		\
		ASM_OR(Aa, Aa, TRes)		\
		ASM_OR(Ab, Ab, Tmp)		\
		Sa[index] = Aa;			\
		Sb[index] = Ab


/*
** ROUND_1 macro :
** PIPE A						PIPE B
** Sk = Sa[index]				q = Sb[index]
** Ta = Aa + Ba					Tb = Ab + Bb
** La += Ta						Lb += Tb
** Ba = La = ROTL(La, Ta)		Bb = Lb = ROTL(Lb, Tb)
** Sa = Sk + Aa					Sb = q + Aa
** Sa += Ba						Sb += Bb
** Aa = Sa = ROTL3(Sa)			Ab = Sb = ROTL3(Sb)
** RaA ^= RaB					RbA ^= RbB
** RaA = ROTL(RaA, RaB)			RbA = ROTL(RbA, RbB)
** RaA += Aa					RbA += Ab
*/

#define ROUND_1(index, Aa, Ab, La, Lb, Ba, Bb)	\
		Sk = Sa[index];			\
		ASM_ADD(Tmp, Aa, Ba)		\
		q = Sb[index];			\
		ASM_ADD(TRes, La, Tmp)		\
		ASM_COMP(RShft, word_size, Tmp)	\
		ASM_SL(La, TRes, Tmp)		\
		ASM_SR(TRes, TRes, RShft)	\
		ASM_ADD(Tmp, Ab, Bb)		\
		ASM_OR(La, La, TRes)		\
		ASM_ADD(TRes, Lb, Tmp)		\
		ASM_COMP(RShft, word_size, Tmp)	\
		ASM_SL(Lb, TRes, Tmp)		\
		ASM_SR(TRes, TRes, RShft)	\
		ASM_ADD(Aa, Aa, Sk)		\
		ASM_ADD(Ab, Ab, q)		\
		ASM_OR(Lb, Lb, TRes)		\
		ASM_ADD(Aa, Aa, La)		\
		ASM_ADD(Ab, Ab, Lb)		\
		ASM_SL3(TRes, Aa)		\
		ASM_SL3(Tmp, Ab)		\
		ASM_SR29(Aa, Aa)		\
		ASM_SR29(Ab, Ab)		\
		ASM_OR(Aa, Aa, TRes)		\
		ASM_OR(Ab, Ab, Tmp)		\
		ASM_XOR(RaA, RaA, RaB)		\
		ASM_XOR(RbA, RbA, RbB)		\
		ASM_COMP(RShft, word_size, RaB)	\
		ASM_SL(TRes, RaA, RaB)		\
		ASM_SR(RaA, RaA, RShft)		\
		ASM_COMP(RShft, word_size, RbB)	\
		ASM_SL(Tmp, RbA, RbB)		\
		ASM_SR(RbA, RbA, RShft)		\
		ASM_OR(RaA, RaA, TRes)		\
		ASM_OR(RbA, RbA, Tmp)		\
		ASM_ADD(RaA, RaA, Aa)		\
		ASM_ADD(RbA, RbA, Ab)


/*
** ROUND_2 macro :
** PIPE A						PIPE B
** Sk = Sa[index]				q = Sb[index]
** Ta = Aa + Ba					Tb = Ab + Bb
** La += Ta						Lb += Tb
** Ba = La = ROTL(La, Ta)		Bb = Lb = ROTL(Lb, Tb)
** Sa = Sk + Aa					Sb = q + Aa
** Sa += Ba						Sb += Bb
** Aa = Sa = ROTL3(Sa)			Ab = Sb = ROTL3(Sb)
** RaB ^= RaA					RbB ^= RbA
** RaB = ROTL(RaB, RaA)			RbB = ROTL(RbB, RbA)
** RaB += Aa					RbB += Ab
*/

#define ROUND_2(index, Aa, Ab, La, Lb, Ba, Bb)	\
		Sk = Sa[index];			\
		ASM_ADD(Tmp, Aa, Ba)		\
		q = Sb[index];			\
		ASM_ADD(TRes, La, Tmp)		\
		ASM_COMP(RShft, word_size, Tmp)	\
		ASM_SL(La, TRes, Tmp)		\
		ASM_SR(TRes, TRes, RShft)	\
		ASM_ADD(Tmp, Ab, Bb)		\
		ASM_OR(La, La, TRes)		\
		ASM_ADD(TRes, Lb, Tmp)		\
		ASM_COMP(RShft, word_size, Tmp)	\
		ASM_SL(Lb, TRes, Tmp)		\
		ASM_SR(TRes, TRes, RShft)	\
		ASM_ADD(Aa, Aa, Sk)		\
		ASM_ADD(Ab, Ab, q)		\
		ASM_OR(Lb, Lb, TRes)		\
		ASM_ADD(Aa, Aa, La)		\
		ASM_ADD(Ab, Ab, Lb)		\
		ASM_SL3(TRes, Aa)		\
		ASM_SL3(Tmp, Ab)		\
		ASM_SR29(Aa, Aa)		\
		ASM_SR29(Ab, Ab)		\
		ASM_OR(Aa, Aa, TRes)		\
		ASM_OR(Ab, Ab, Tmp)		\
		ASM_XOR(RaB, RaB, RaA)		\
		ASM_XOR(RbB, RbB, RbA)		\
		ASM_COMP(RShft, word_size, RaA)	\
		ASM_SL(TRes, RaB, RaA)		\
		ASM_SR(RaB, RaB, RShft)		\
		ASM_COMP(RShft, word_size, RbA)	\
		ASM_SL(Tmp, RbB, RbA)		\
		ASM_SR(RbB, RbB, RShft)		\
		ASM_OR(RaB, RaB, TRes)		\
		ASM_OR(RbB, RbB, Tmp)		\
		ASM_ADD(RaB, RaB, Aa)		\
		ASM_ADD(RbB, RbB, Ab)


s32 CDECL rc5_72_unit_func_KKS_2 (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
	u32 Sa[26], Sb[26];
	register u32 Sk, q, S2;
	u32 kiter;
	const u32 cached_s0 = ROTL3(P);
	u32 cached_s1, cached_s2, cached_l0, cached_l1;
	register u32 La0, La1, La2;
	register u32 Lb0, Lb1, Lb2;
	register u32 SaA, SbA, TRes, Tmp, RShft;
	register u32 RaA, RaB, RbA, RbB;
	u32 word_size = 32;

	#ifdef __POWERPC__
	u32 LShft;				/* PowerPC kludge */
	#endif

	kiter = *iterations / 2;

	/*
	** Compute the part which is independant from L0.hi
	** Also initialize S[0], S[1] and S[2].
	*/

	q = Q;
	Sk = P + Q;
	cached_l0 = ROTL(rc5_72unitwork->L0.lo + cached_s0, cached_s0);
	cached_s1 = ROTL3(Sk + cached_s0 + cached_l0);
	
	Tmp = cached_s1 + cached_l0;
	Sk += q;		/* == P + 2Q */
	S2 = Sk;
	cached_l1 = ROTL(rc5_72unitwork->L0.mid + Tmp, Tmp);
	cached_s2 = ROTL3(Sk + cached_s1 + cached_l1);

	do {
		Sa[0] = Sb[0] = cached_s0;
		Sa[1] = Sb[1] = cached_s1;
		Sa[2] = Sb[2] = cached_s2;
		SaA = SbA = cached_s2;
		La0 = Lb0 = cached_l0;
		La1 = Lb1 = cached_l1;
		La2 = rc5_72unitwork->L0.hi;
		Lb2 = La2 + 1;
		Sk = S2;
		q = Q;

		MIX_1( 3, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_1( 4, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_1( 5, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_1( 6, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_1( 7, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_1( 8, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_1( 9, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_1(10, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_1(11, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_1(12, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_1(13, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_1(14, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_1(15, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_1(16, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_1(17, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_1(18, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_1(19, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_1(20, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_1(21, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_1(22, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_1(23, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_1(24, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_1(25, SaA, SbA, La0, Lb0, La2, Lb2);

		MIX_2( 0, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_2( 1, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_2( 2, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_2( 3, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_2( 4, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_2( 5, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_2( 6, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_2( 7, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_2( 8, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_2( 9, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_2(10, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_2(11, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_2(12, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_2(13, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_2(14, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_2(15, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_2(16, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_2(17, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_2(18, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_2(19, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_2(20, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_2(21, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_2(22, SaA, SbA, La2, Lb2, La1, Lb1);
		MIX_2(23, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_2(24, SaA, SbA, La1, Lb1, La0, Lb0);
		MIX_2(25, SaA, SbA, La2, Lb2, La1, Lb1);

		MIX_2( 0, SaA, SbA, La0, Lb0, La2, Lb2);
		MIX_2( 1, SaA, SbA, La1, Lb1, La0, Lb0);

		RaA = Sa[0] + rc5_72unitwork->plain.lo;
		RbA = Sb[0] + rc5_72unitwork->plain.lo;
		RaB = Sa[1] + rc5_72unitwork->plain.hi;
		RbB = Sb[1] + rc5_72unitwork->plain.hi;

		ROUND_1( 2, SaA, SbA, La2, Lb2, La1, Lb1);
		ROUND_2( 3, SaA, SbA, La0, Lb0, La2, Lb2);
		ROUND_1( 4, SaA, SbA, La1, Lb1, La0, Lb0);
		ROUND_2( 5, SaA, SbA, La2, Lb2, La1, Lb1);
		ROUND_1( 6, SaA, SbA, La0, Lb0, La2, Lb2);
		ROUND_2( 7, SaA, SbA, La1, Lb1, La0, Lb0);
		ROUND_1( 8, SaA, SbA, La2, Lb2, La1, Lb1);
		ROUND_2( 9, SaA, SbA, La0, Lb0, La2, Lb2);
		ROUND_1(10, SaA, SbA, La1, Lb1, La0, Lb0);
		ROUND_2(11, SaA, SbA, La2, Lb2, La1, Lb1);
		ROUND_1(12, SaA, SbA, La0, Lb0, La2, Lb2);
		ROUND_2(13, SaA, SbA, La1, Lb1, La0, Lb0);
		ROUND_1(14, SaA, SbA, La2, Lb2, La1, Lb1);
		ROUND_2(15, SaA, SbA, La0, Lb0, La2, Lb2);
		ROUND_1(16, SaA, SbA, La1, Lb1, La0, Lb0);
		ROUND_2(17, SaA, SbA, La2, Lb2, La1, Lb1);
		ROUND_1(18, SaA, SbA, La0, Lb0, La2, Lb2);
		ROUND_2(19, SaA, SbA, La1, Lb1, La0, Lb0);
		ROUND_1(20, SaA, SbA, La2, Lb2, La1, Lb1);
		ROUND_2(21, SaA, SbA, La0, Lb0, La2, Lb2);
		ROUND_1(22, SaA, SbA, La1, Lb1, La0, Lb0);
		ROUND_2(23, SaA, SbA, La2, Lb2, La1, Lb1);
		ROUND_1(24, SaA, SbA, La0, Lb0, La2, Lb2);

		Tmp = rc5_72unitwork->cypher.lo;
		if (RaA == Tmp || RbA == Tmp) {
			ROUND_2(25, SaA, SbA, La1, Lb1, La0, Lb0);
		}

		if (RaA == rc5_72unitwork->cypher.lo) {
			++rc5_72unitwork->check.count;
			rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi;
			rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
			rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
			
			if (RaB == rc5_72unitwork->cypher.hi) {
				*iterations -= kiter * 2;
				return RESULT_FOUND;
			}
		}
		
		if (RbA == rc5_72unitwork->cypher.lo) {
			++rc5_72unitwork->check.count;
			rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 1;
			rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
			rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;

			if (RbB == rc5_72unitwork->cypher.hi) {
				*iterations -= kiter * 2 - 1;
				return RESULT_FOUND;
			}
		}

		#define key rc5_72unitwork->L0
		key.hi = (key.hi + 0x02) & 0x000000FF;
		if (!key.hi) {
			key.mid = key.mid + 0x01000000;
			if (!(key.mid & 0xFF000000u)) {
				key.mid = (key.mid + 0x00010000) & 0x00FFFFFF;
				if (!(key.mid & 0x00FF0000)) {
					key.mid = (key.mid + 0x00000100) & 0x0000FFFF;
					if (!(key.mid & 0x0000FF00)) {
						key.mid = (key.mid + 0x00000001) & 0x000000FF;
						if (!key.mid) {
							key.lo = key.lo + 0x01000000;
							if (!(key.lo & 0xFF000000u)) {
								key.lo = (key.lo + 0x00010000) & 0x00FFFFFF;
								if (!(key.lo & 0x00FF0000)) {
									key.lo = (key.lo + 0x00000100) & 0x0000FFFF;
									if (!(key.lo & 0x0000FF00)) {
										key.lo = (key.lo + 0x00000001) & 0x000000FF;
									}
								}
							}
							/* key.lo changed */
							Sk = P + Q;
							cached_l0 = ROTL(key.lo + cached_s0, cached_s0);
							cached_s1 = ROTL3(Sk + cached_s0 + cached_l0);
						}
					}
				}
			}
			/* key.mid changed */
			Tmp = cached_s1 + cached_l0;
			cached_l1 = ROTL(key.mid + Tmp, Tmp);
			cached_s2 = ROTL3(S2 + cached_s1 + cached_l1);
		}
	} while (--kiter);

	return RESULT_NOTHING;
}
