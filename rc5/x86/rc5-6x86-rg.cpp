// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// torment.ntr.net K6 233 sean@ntr.net
//
// $Log: rc5-6x86-rg.cpp,v $
// Revision 1.2  1998/06/14 08:13:34  friedbait
// 'Log' keywords added to maintain automatic change history
//
//

// Cyrix 6x86 optimized version
//
// 980226 :
//	- Corrected bug in the key incrementation algorithm that caused the
//	  client to core-dump at the end of some blocks.
//	  As a side-effect, this fix re-enable support for blocks of up to 2^64 keys
//	- Added alignement pseudo-instructions.
//	- Converted :
//		subl $0x01000000, %%reg   to   addl $0xFF000100, %%reg
//		addl $0x00000100, %%reg
//	  and :
//		subl $0x00010000, %%reg   to   addl $0xFFFF0001, %%reg
//		addl $0x00000001, %%reg
//
// 980118 :
//	- Sean McPherson <sean@ntr.net> has allowed me to use his Linux box to
//	  test & debug this new core. Thanks a lot, Sean !
//	  This one is 17% faster than the previous ! On Sean PC
//	  (6x86 PR200) it ran at ~354 kkeys/s where the previous core got
//	  only 304 kkeys/s
//
// 980104 :
//	- precalculate some things for ROUND1 & ROUND2
//
// 971226 :
//	* Edited ROUND1 & ROUND2 to avoid potential stall on such code :
//		roll	$3,    %%eax
//		movl	%%eax, %%ecx
//	* ROUND3 cycle counts was bad, it was really 14 cycles in 971220,
//	  not 13 cycles. Modified a bit and back to 13 real cycles.
//
// 971220 :
//	* It seems that this processor can't decode two instructions
//	  per clock when one of the pair is >= 7 bytes.
//	  Unfortunatly, there's a lot in this code, because "s2" is located
//	  more than 127 bytes from %esp. So each access to S2 will be coded
//	  as a long (4 bytes) displacement.
//	  S1 access will be coded with a short displacement (8 bits signed).
//	* Modified ROUND3 to access s1 & s2 with ebp as the base, and so with
//	  a short displacement.
//	* Modified also ROUND1 with ROUND2 as a template, since ROUND2 seems
//	  to suffer less from this limitation.
//
// 971214 :
//	* modified ROUND1 & ROUND2 to avoid :
//		leal	(%%eax,%%edx), %%ecx
//		movl	%%eax, "S1(N)"
//		addl	%%ecx, %%ebx	# doesn't seems to pair with 2nd clock of "leal"
//	  -> 1 clock less in each macro
//
//	* modified ROUND3 to avoid a stall on the Y pipe
//	  -> 1 clock less
//
// 971209 :
//	* First public release
//
//
// PRating versus clock speed:
//
//    6x86		 6x86MX (aka M2)
//
// PR200 = 150		PR266 = 225 | 233
// PR166 = 133		PR233 = 188 | 200
// PR150 = 120		PR200 = 166
// PR133 = 110		PR166 = 150

#define CORE_INCREMENTS_KEY

// This file is included from rc5.cpp so we can use __inline__.
#include "problem.h"

#if (PIPELINE_COUNT != 2)
#error "Expecting pipeline count of 2"
#endif

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif


// Stringify macro.

#define _(s)    __(s)
#define __(s)   #s

// The S0 values for key expansion round 1 are constants.

#define P         0xB7E15163
#define Q         0x9E3779B9
#define S_not(N)  _((P+Q*(N)))

//  Structure used in rc5_unit_func_*

struct work_struct {
    u32 s1[26];		// +  0
    u32 s2[26];		// +104
    u32 add_iter;	// +208
    u32 P_0;		// +212
    u32 P_1;		// +216
    u32 C_0;		// +220
    u32 C_1;		// +224
    u32 save_ebp;	// +228
    u32 key2_ebp;	// +232
    u32 key2_edi;	// +236
    u32 key2_esi;	// +240
    u32 key_hi;		// +244
    u32 key_lo;		// +248
    u32 iterations;	// +252
    u32 pre1_r1;	// +256
    u32 pre2_r1;	// +260
    u32 pre3_r1;	// +264
};

//  Offsets to access work_struct fields.

#define	work_s1             "%0"
#define	work_s2         "104+%0"
#define	work_add_iter   "208+%0"
#define	work_P_0        "212+%0"
#define	work_P_1        "216+%0"
#define	work_C_0        "220+%0"
#define	work_C_1        "224+%0"
#define	work_save_ebp   "228+%0"
#define	work_key2_ebp   "232+%0"
#define	work_key2_edi   "236+%0"
#define	work_key2_esi   "240+%0"
#define work_key_hi     "244+%0"
#define work_key_lo     "248+%0"
#define work_iterations "252+%0"
#define work_pre1_r1    "256+%0"
#define work_pre2_r1    "260+%0"
#define work_pre3_r1    "264+%0"

//  Macros to access the S arrays.

#define S1(N)    _(((N)*4)+%0)
#define S2(N)    _(((N)*4)+104+%0)

//  Offsets to access struct RC5UnitWork fields.

#define RC5UnitWork_plainhi   "0"
#define RC5UnitWork_plainlo   "4"
#define RC5UnitWork_cypherhi  "8"
#define RC5UnitWork_cypherlo  "12"
#define RC5UnitWork_L0hi      "16"
#define RC5UnitWork_L0lo      "20"

  // A1   = %eax  A2   = %ebp
  // Llo1 = %ebx  Llo2 = %esi
  // Lhi1 = %edx  Lhi2 = %edi

/*
3 cycles
	roll	%%cl,  %%esi		#
	addl	$"S_not(N)", %%eax	#
	addl	$"S_not(N)", %%ebp	#

it's not the length of instruction, it's that it can't decode when instruction is executed
3 cycles :
	roll	%%cl,  %%edx	#
	addl	0(%%ebp),%%eax	#
	xorl	%%edi, %%esi	#


leal is subject of AGI (2 cycles) if an operand is modified 1 cycle before
leal is subject of AGI (1 cycles) if an operand is modified 2 cycles before

leal (%%edx,%%eax), %%eax take one cycle and is pairable
leal 12345678(%%edx,%%eax), %%eax takes two cycles and isn't pairable
*/

// ------------------------------------------------------------------
// S1(N) = A1 = ROTL3 (A1 + Lhi1 + S_not(N));
// S2(N) = A2 = ROTL3 (A2 + Lhi2 + S_not(N));
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
#define ROUND_1_EVEN(N) \
"	addl	$"S_not(N)",%%eax	# . pairs with roll in previous iteration
	addl	%%edi, %%ebp		# 1
	addl	%%edx, %%eax		#
	roll	$3,    %%ebp		# 1
	roll	$3,    %%eax		#
	movl	%%eax, %%ecx		# 1
	addl	%%edx, %%ecx		#   yes, it works
	movl	%%eax, "S1(N)"		# 1
	addl	%%ecx, %%ebx		#
	addl	$"S_not(N+1)",%%eax	# 1
	movl	%%ebp, "S2(N)"		#
	roll	%%cl,  %%ebx		# 2
	leal   (%%ebp, %%edi),%%ecx	#
	addl	%%ecx, %%esi		# 1
	addl	$"S_not(N+1)",%%ebp	#
	roll	%%cl,  %%esi		# 2  \n"

// S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
// S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
// Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
#define ROUND_1_ODD(N) \
"	addl	%%ebx,  %%eax		# . pairs with roll in previous iteration
	addl	%%esi,  %%ebp		# 1
	roll	$3,     %%eax		#
	roll	$3,     %%ebp		# 1
	movl	%%eax,  "S1(N)"		#
	movl	%%eax,  %%ecx		# 1
	addl	%%ebx,  %%ecx		#
	addl	%%ecx,  %%edx		# 1
	movl	%%ebp,  "S2(N)"		#
	roll	%%cl,   %%edx		# 2
	leal   (%%ebp,  %%esi),%%ecx	#
	addl	%%ecx,  %%edi		# 1
	addl	$"S_not(N+1)",%%ebp	#
	roll	%%cl,   %%edi		# 2	sum = 19 (r1 & r2) \n"

#define ROUND_1_LAST(N) \
"	addl	%%ebx,  %%eax		# . pairs with roll in previous iteration
	addl	%%esi,  %%ebp		# 1
	roll	$3,     %%eax		#
	roll	$3,     %%ebp		# 1
	movl	%%eax,  "S1(N)"		#   yes, it works !
	movl	%%eax,  %%ecx		# 1
	addl	%%ebx,  %%ecx		#
	addl	%%ecx,  %%edx		# 1
	movl	%%ebp,  "S2(N)"		#
	roll	%%cl,   %%edx		# 2
	leal   (%%ebp,  %%esi),%%ecx	#
	addl	%%ecx,  %%edi		# 1
	addl	$"S0_ROTL3",%%eax	#
	roll	%%cl,   %%edi		# 2	sum = 19 (r1 & r2) \n"

#define ROUND_1_EVEN_AND_ODD(N) \
	ROUND_1_EVEN(N) \
	ROUND_1_ODD (N+1)

// ------------------------------------------------------------------
// S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
// S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
#define ROUND_2_EVEN(N) \
"	addl	"S1(N)",%%eax		# . pairs with roll in previous iteration
	addl	%%edi, %%ebp		# 1
	addl	%%edx, %%eax		#
	roll	$3,    %%ebp		# 1
	roll	$3,    %%eax		#
	movl	%%eax, %%ecx		# 1
	addl	%%edx, %%ecx		#   yes, it works
	movl	%%eax, "S1(N)"		# 1
	addl	%%ecx, %%ebx		#
	addl	"S1(N+1)",%%eax		# 1
	movl	%%ebp, "S2(N)"		#
	roll	%%cl,  %%ebx		# 2
	leal   (%%ebp, %%edi),%%ecx	#
	addl	%%ecx, %%esi		# 1
	addl	"S2(N+1)",%%ebp		#
	roll	%%cl,  %%esi		# 2  \n"

// S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
// S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
// Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
#define ROUND_2_ODD(N) \
"	addl	%%ebx,  %%eax		# . pairs with roll in previous iteration
	addl	%%esi,  %%ebp		# 1
	roll	$3,     %%eax		#
	roll	$3,     %%ebp		# 1
	movl	%%eax,  "S1(N)"		#   yes, it works !
	movl	%%eax,  %%ecx		# 1
	addl	%%ebx,  %%ecx		#
	addl	%%ecx,  %%edx		# 1
	movl	%%ebp,  "S2(N)"		#
	roll	%%cl,   %%edx		# 2
	leal   (%%ebp,  %%esi),%%ecx	#
	addl	%%ecx,  %%edi		# 1
	addl	"S2(N+1)",%%ebp		#
	roll	%%cl,   %%edi		# 2	sum = 19 (r1 & r2) \n"

#define ROUND_2_LAST(N) \
"	addl	%%ebx,  %%eax		# . pairs with roll in previous iteration
	addl	%%esi,  %%ebp		# 1
	roll	$3,     %%eax		#
	roll	$3,     %%ebp		# 1
	movl	%%eax,  "S1(N)"		#   yes, it works !
	movl	%%eax,  %%ecx		# 1
	addl	%%ebx,  %%ecx		#
	addl	%%ecx,  %%edx		# 1
	movl	%%ebp,  "S2(N)"		#
	roll	%%cl,   %%edx		# 2
	leal   (%%ebp,  %%esi),%%ecx	#
	addl	%%ecx,  %%edi		# 1
	movl	%%ebp, "work_key2_ebp"	#
	roll	%%cl,   %%edi		# 2	sum = 19 (r1 & r2) \n"

#define ROUND_2_EVEN_AND_ODD(N) \
	ROUND_2_EVEN(N) \
	ROUND_2_ODD (N+1)

// ------------------------------------------------------------------
// eA1 = ROTL (eA1 ^ eB1, eB1) + (A1 = ROTL3 (A1 + Lhi1 + S1(N)));
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// eB1 = ROTL (eA1 ^ eB1, eA1) + (A1 = ROTL3 (A1 + Llo1 + S1(N)));
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);

// A  = %eax  eA = %esi
// L0 = %ebx  eB = %edi
// L1 = %edx  .. = %ebp
// %%ebp is either &S1 or &S2
#define S3(N) _((N)*4)"(%%ebp)"

#define ROUND_3_EVEN_AND_ODD(N) \
"	addl	"S3(N)",%%eax	#
	xorl	%%edi, %%esi	#
	movl	%%edi, %%ecx	# 1
	addl	%%edx, %%eax	#
	roll	%%cl,  %%esi	# 2
	roll	$3,    %%eax	#
	movl	%%eax, %%ecx	#
	addl	%%edx, %%ecx	# 1
	addl	%%eax, %%esi	#
	addl	%%ecx, %%ebx	# 1
	addl	"S3(N+1)",%%eax	#
	roll	%%cl,  %%ebx	# 2

	xorl	%%esi, %%edi	#
	movl	%%esi, %%ecx	#
	roll	%%cl,  %%edi	# 2
	addl	%%ebx, %%eax	#
	roll	$3,    %%eax	#
	movl	%%eax, %%ecx	# 1
	addl	%%ebx, %%ecx	#
	addl	%%ecx, %%edx	# 1
	addl	%%eax, %%edi	#
	roll	%%cl,  %%edx	# 2	sum = 13 \n"

// ------------------------------------------------------------------
// rc5_unit will get passed an RC5WorkUnit to complete
//
// Returns number of keys checked before a possible good key is found, or
// timeslice*PIPELINE_COUNT if no keys are 'good' keys.
// (ie:      if (result == timeslice*PIPELINE_COUNT) NOTHING_FOUND
//      else if (result < timeslice*PIPELINE_COUNT) SOMETHING_FOUND at result+1
//      else SOMETHING_WENT_WRONG... )
//
// There is no way to tell gcc to save %ebp.
//	(putting %ebp in the clobbered register list has no effect)
// Even worse, if '-fomit-frame-pointer' isn't used, gcc will compile
// this function with local variables referenced with %ebp (!!).

static
u32 rc5_unit_func_6x86( RC5UnitWork * rc5unitwork, u32 timeslice )
{
    work_struct work;

    work.iterations = timeslice;
    work.add_iter = 0;

    __asm__ __volatile__ (

	/* save %ebp */
"	movl	%%ebp,"work_save_ebp" \n"
	
	/* pointer to rc5unitwork already loaded in %eax (see constraint 'a') */

	/* load parameters */
"	movl	"RC5UnitWork_L0lo"(%%eax), %%ebx	# ebx = l0 = Llo1
	movl	"RC5UnitWork_L0hi"(%%eax), %%edx	# edx = l1 = Lhi1
	movl	%%ebx, %%esi				# esi = l2 = Llo2
	leal	0x01000000(%%edx), %%edi		# edi = l3 = lhi2
	movl	%%ebx, "work_key_lo"
	movl	%%edx, "work_key_hi" \n"

	/* Save other parameters */
	/* (it's faster to do so, since we will only load 1 value */
	/* each time in RC5_ROUND_3xy, instead of two if we save  */
	/* only the pointer to the RC5 struct)                    */
"	movl	"RC5UnitWork_plainlo"(%%eax), %%ebp
	movl	%%ebp, "work_P_0"
	movl	"RC5UnitWork_plainhi"(%%eax), %%ebp
	movl	%%ebp, "work_P_1"
	movl	"RC5UnitWork_cypherlo"(%%eax), %%ebp
	movl	%%ebp, "work_C_0"
	movl	"RC5UnitWork_cypherhi"(%%eax), %%ebp
	movl	%%ebp, "work_C_1" \n"

	/* Pre-calculate things. Assume work.key_lo won't change it this loop */
	/* (it's pretty safe to assume that, because we're working on 28 bits */
	/* blocks) */
	/* It means also that %%ebx == %%esi (Llo1 == Llo2) */

//#define S0_ROTL3  _(((P<<3) | (P>>29)))
#define S0_ROTL3 _(0xbf0a8b1d)
//#define FIRST_ROTL _((S0_ROTL3 & 0x1f))
#define FIRST_ROTL _(0x1d)
//#define S1_S0_ROTL3 _((S_not(1) + S0_ROTL3))
#define S1_S0_ROTL3 _(0x15235639)

"_bigger_loop_6x86:
	addl	$"S0_ROTL3", %%ebx
	roll	$"FIRST_ROTL",  %%ebx
	movl	%%ebx, "work_pre1_r1"

	leal	"S1_S0_ROTL3"(%%ebx),%%eax
	roll	$3,    %%eax
	movl	%%eax, "work_pre2_r1"

	leal	(%%eax,%%ebx), %%ecx
	movl	%%ecx, "work_pre3_r1"

.balign 4
_loaded_6x86:\n"

    /* ------------------------------ */
    /* Begin round 1 of key expansion */
    /* ------------------------------ */

"	movl	"work_pre1_r1", %%ebx	# 1
	movl	"work_pre2_r1", %%eax	#
	movl	%%ebx, %%esi		# 1
	movl	%%eax, %%ebp		#

	movl	"work_pre3_r1", %%ecx	# 1
	addl	%%ecx, %%edx		#
	roll	%%cl,  %%edx		# 2
	addl	%%ecx, %%edi		#
	movl	%%eax, "S1(1)"		# 1
	addl	$"S_not(2)",%%ebp	#
	roll	%%cl,  %%edi		# 2	sum = 8  \n"

	ROUND_1_EVEN_AND_ODD ( 2)
	ROUND_1_EVEN_AND_ODD ( 4)
	ROUND_1_EVEN_AND_ODD ( 6)
	ROUND_1_EVEN_AND_ODD ( 8)
	ROUND_1_EVEN_AND_ODD (10)
	ROUND_1_EVEN_AND_ODD (12)
	ROUND_1_EVEN_AND_ODD (14)
	ROUND_1_EVEN_AND_ODD (16)
	ROUND_1_EVEN_AND_ODD (18)
	ROUND_1_EVEN_AND_ODD (20)
	ROUND_1_EVEN_AND_ODD (22)
	ROUND_1_EVEN         (24)
	ROUND_1_LAST         (25)
"_end_round1_6x86: \n"


    /* ------------------------------ */
    /* Begin round 2 of key expansion */
    /* ------------------------------ */

"#	addl	$"S0_ROTL3",%%eax	# . already done in ROUND_1_LAST
	addl	$"S0_ROTL3",%%ebp	#
	addl	%%edx, %%eax		# 1
	addl	%%edi, %%ebp		#
	roll	$3,    %%eax		# 1
	roll	$3,    %%ebp		#
	movl	%%eax, %%ecx		# 1
	addl	%%edx, %%ecx		#
	movl	%%eax, "S1(0)"		# 1
	addl	%%ecx, %%ebx		#
	addl	"S1(1)",%%eax		# 1
	movl	%%ebp, "S2(0)"		#
	roll	%%cl,  %%ebx		# 2
	leal   (%%ebp, %%edi),%%ecx	#
	addl	%%ecx, %%esi		# 1
	addl	"S1(1)",%%ebp		#
	roll	%%cl,  %%esi		# 2 \n"

	ROUND_2_ODD          ( 1)
        ROUND_2_EVEN_AND_ODD ( 2)
        ROUND_2_EVEN_AND_ODD ( 4)
        ROUND_2_EVEN_AND_ODD ( 6)
        ROUND_2_EVEN_AND_ODD ( 8)
        ROUND_2_EVEN_AND_ODD (10)
        ROUND_2_EVEN_AND_ODD (12)
        ROUND_2_EVEN_AND_ODD (14)
        ROUND_2_EVEN_AND_ODD (16)
        ROUND_2_EVEN_AND_ODD (18)
        ROUND_2_EVEN_AND_ODD (20)
        ROUND_2_EVEN_AND_ODD (22)
	ROUND_2_EVEN         (24)
	ROUND_2_LAST         (25)

    /* Save 2nd key parameters */
"_end_round2_6x86:
#	movl	%%ebp, "work_key2_ebp"	already done in ROUND_2_LAST
	movl	%%esi, "work_key2_esi"
	movl	%%edi, "work_key2_edi" \n"

    /* ---------------------------------------------------- */
    /* Begin round 3 of key expansion mixed with encryption */
    /* ---------------------------------------------------- */
    /* (first key)					    */

	// A  = %eax  eA = %esi
	// L0 = %ebx  eB = %edi
	// L1 = %edx  .. = %ebp

"	leal	"work_s1",%%ebp \n"

	/* A = ROTL3(S00 + A + L1); */
	/* eA = P_0 + A; */
	/* L0 = ROTL(L0 + A + L1, A + L1);*/
"	addl	"S3(0)",%%eax	#	(pairs with leal)
	addl	%%edx,  %%eax	# 1
	movl	"work_P_0",%%esi#
	roll	$3,     %%eax	# 1
	movl	%%edx,  %%ecx	#
	addl	%%eax,  %%esi	# 1
	addl	%%eax,  %%ecx	#
	addl	%%ecx,  %%ebx	# 1
	addl	"S3(1)",%%eax	#
	roll	%%cl,   %%ebx	# 2 \n"
	/* A = ROTL3(S01 + A + L0); */
	/* eB = P_1 + A; */
	/* L1 = ROTL(L1 + A + L0, A + L0);*/
"	addl	%%ebx,  %%eax	#
	movl	%%ebx,  %%ecx	# 1
	roll	$3,     %%eax	#
	movl	"work_P_1",%%edi# 1
	addl	%%eax,  %%edi	#
	addl	%%eax,  %%ecx	# 1
	addl	%%ecx,  %%edx	# 1
	roll	%%cl,   %%edx	# 2 \n"
	ROUND_3_EVEN_AND_ODD ( 2)
	ROUND_3_EVEN_AND_ODD ( 4)
	ROUND_3_EVEN_AND_ODD ( 6)
	ROUND_3_EVEN_AND_ODD ( 8)
	ROUND_3_EVEN_AND_ODD (10)
	ROUND_3_EVEN_AND_ODD (12)
	ROUND_3_EVEN_AND_ODD (14)
	ROUND_3_EVEN_AND_ODD (16)
	ROUND_3_EVEN_AND_ODD (18)
	ROUND_3_EVEN_AND_ODD (20)
	ROUND_3_EVEN_AND_ODD (22)
	/* early exit */
"_end_round3_1_6x86:
	addl	"S3(24)", %%eax	#    	A = ROTL3(S24 + A + L1);
	movl	%%edi, %%ecx	# 1 	eA = ROTL(eA ^ eB, eB) + A;
	addl	%%edx, %%eax	#
	xorl	%%edi, %%esi	# 1
	roll	$3,    %%eax	#
	roll	%%cl,  %%esi	# 2
	addl	%%eax, %%esi	# 1
					
	cmp	"work_C_0", %%esi
	jne	__exit_1_6x86
					
	movl	%%eax, %%ecx	# 1	L0 = ROTL(L0 + A + L1, A + L1);
	addl	%%edx, %%ecx	#	A = ROTL3(S25 + A + L0);
	xorl	%%esi, %%edi	# 1	eB = ROTL(eB ^ eA, eA) + A;
	addl	%%ecx, %%ebx	#
	roll	%%cl,  %%ebx	# 2
	addl	"S3(25)", %%eax	#
	movl	%%esi, %%ecx	# 1
	addl	%%ebx, %%eax	#
	roll	%%cl,  %%edi	# 2
	roll	$3,    %%eax	#
	addl	%%eax, %%edi	# 1

	cmpl	"work_C_1", %%edi
	je	_full_exit_6x86

.balign 4
__exit_1_6x86: \n"

    /* Restore 2nd key parameters */
"	movl	"work_key2_edi",%%edx
	movl	"work_key2_esi",%%ebx
	movl	"work_key2_ebp",%%eax\n"

    /* ---------------------------------------------------- */
    /* Begin round 3 of key expansion mixed with encryption */
    /* ---------------------------------------------------- */
    /* (second key)					    */

	// A  = %eax  eA = %esi
	// L0 = %ebx  eB = %edi
	// L1 = %edx  .. = %ebp

"	leal	"work_s2",%%ebp \n"

	/* A = ROTL3(S00 + A + L1); */
	/* eA = P_0 + A; */
	/* L0 = ROTL(L0 + A + L1, A + L1);*/
"	addl	%%edx,  %%eax	# 1
	movl	%%edx,  %%ecx	#
	addl	"S3(0)",%%eax	# 1
	roll	$3,     %%eax	# 1
	movl	"work_P_0",%%esi#
	addl	%%eax,  %%esi	# 1
	addl	%%eax,  %%ecx	#
	addl	%%ecx,  %%ebx	# 1
	addl	"S3(1)",%%eax	#
	roll	%%cl,   %%ebx	# 2 \n"
	/* A = ROTL3(S01 + A + L0); */
	/* eB = P_1 + A; */
	/* L1 = ROTL(L1 + A + L0, A + L0);*/
"	addl	%%ebx,  %%eax	#
	movl	%%ebx,  %%ecx	# 1
	roll	$3,     %%eax	#
	movl	"work_P_1",%%edi# 1
	addl	%%eax,  %%edi	#
	addl	%%eax,  %%ecx	# 1
	addl	%%ecx,  %%edx	# 1
	roll	%%cl,   %%edx	# 2 \n"
	ROUND_3_EVEN_AND_ODD ( 2)
	ROUND_3_EVEN_AND_ODD ( 4)
	ROUND_3_EVEN_AND_ODD ( 6)
	ROUND_3_EVEN_AND_ODD ( 8)
	ROUND_3_EVEN_AND_ODD (10)
	ROUND_3_EVEN_AND_ODD (12)
	ROUND_3_EVEN_AND_ODD (14)
	ROUND_3_EVEN_AND_ODD (16)
	ROUND_3_EVEN_AND_ODD (18)
	ROUND_3_EVEN_AND_ODD (20)
	ROUND_3_EVEN_AND_ODD (22)
	/* early exit */
"_end_round3_2_6x86:
	addl	"S3(24)", %%eax	#   	A = ROTL3(S24 + A + L1);
	movl	%%edi, %%ecx	# 1  	eA = ROTL(eA ^ eB, eB) + A;
	addl	%%edx, %%eax	#
	xorl	%%edi, %%esi	# 1
	roll	$3,    %%eax	#
	roll	%%cl,  %%esi	# 2
	addl	%%eax, %%esi	# 1
					
	cmp	"work_C_0", %%esi
	jne	__exit_2_6x86
	
	movl	%%eax, %%ecx	# 1	L0 = ROTL(L0 + A + L1, A + L1);
	addl	%%edx, %%ecx	#	A = ROTL3(S25 + A + L0);
	xorl	%%esi, %%edi	# 1	eB = ROTL(eB ^ eA, eA) + A;
	addl	%%ecx, %%ebx	#
	roll	%%cl,  %%ebx	# 2
	addl	"S3(25)", %%eax	#
	movl	%%esi, %%ecx	# 1
	addl	%%ebx, %%eax	#
	roll	%%cl,  %%edi	# 2
	roll	$3,    %%eax	#
	addl	%%eax, %%edi	# 1

	cmpl	"work_C_1", %%edi
	jne	__exit_2_6x86
	movl	$1, "work_add_iter"
	jmp	_full_exit_6x86

.balign 4
__exit_2_6x86:

	movl	"work_key_hi", %%edx

"/* Jumps not taken are faster */"
	addl	$0x02000000,%%edx
	jc	_next_inc_6x86

_next_iter_6x86:
	movl	%%edx, "work_key_hi"
	leal	 0x01000000(%%edx), %%edi
	decl	"work_iterations"
	jg	_loaded_6x86
	movl	%1, %%eax				# pointer to rc5unitwork
	movl	"work_key_lo", %%ebx
	movl	%%ebx, "RC5UnitWork_L0lo"(%%eax)	# Update real data
	movl	%%edx, "RC5UnitWork_L0hi"(%%eax)	# (used by caller)
	jmp	_full_exit_6x86

.balign 4
_next_iter2_6x86:
	movl	%%ebx, "work_key_lo"
	movl	%%edx, "work_key_hi"
	leal	 0x01000000(%%edx), %%edi
	movl	%%ebx, %%esi
	decl	"work_iterations"
	jg	_bigger_loop_6x86
	movl	%1, %%eax				# pointer to rc5unitwork
	movl	%%ebx, "RC5UnitWork_L0lo"(%%eax)	# Update real data
	movl	%%edx, "RC5UnitWork_L0hi"(%%eax)	# (used by caller)
	jmp	_full_exit_6x86

.balign 4
_next_inc_6x86:
	addl	$0x00010000, %%edx
	testl	$0x00FF0000, %%edx
	jnz	_next_iter_6x86

	addl	$0xFF000100, %%edx
	testl	$0x0000FF00, %%edx
	jnz	_next_iter_6x86

	addl	$0xFFFF0001, %%edx
	testl	$0x000000FF, %%edx
	jnz	_next_iter_6x86


	movl	"work_key_lo", %%ebx

	subl	$0x00000100, %%edx
	addl	$0x01000000, %%ebx
	jnc	_next_iter2_6x86

	addl	$0x00010000, %%ebx
	testl	$0x00FF0000, %%ebx
	jnz	_next_iter2_6x86

	addl	$0xFF000100, %%ebx
	testl	$0x0000FF00, %%ebx
	jnz	_next_iter2_6x86

	addl	$0xFFFF0001, %%ebx
	testl	$0x000000FF, %%ebx
	jnz	_next_iter2_6x86

	# Moo !
	# We have just finished checking the last key
	# of the rc5-64 keyspace...
	# Not much to do here, since we have finished the block ...


.balign 4
_full_exit_6x86:
	movl	"work_save_ebp",%%ebp \n"

: "=m"(work),
  "=m"(rc5unitwork)
: "a" (rc5unitwork)
: "%eax","%ebx","%ecx","%edx","%esi","%edi","cc");

    return (timeslice - work.iterations) * 2 + work.add_iter;
}
