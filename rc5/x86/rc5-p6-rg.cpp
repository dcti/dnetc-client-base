// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: rc5-p6-rg.cpp,v $
// Revision 1.3  1998/06/14 08:27:22  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.2  1998/06/14 08:13:40  friedbait
// 'Log' keywords added to maintain automatic change history
//
//
// Pentium Pro optimized version
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
// 980104 :
//	- precalculate some things for ROUND1 & ROUND2

static char *id="@(#)$Id: rc5-p6-rg.cpp,v 1.3 1998/06/14 08:27:22 friedbait Exp $";

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
    u32 add_iter;	// +  0
    u32 s1[26];		// +  4
    u32 s2[26];		// +108
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

#define	work_add_iter   "0+%0"
#define	work_s1         "4+%0"
#define	work_s2         "108+%0"
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

#define S1(N)    _(((N)*4)+4+%0)
#define S2(N)    _(((N)*4)+108+%0)

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

// ------------------------------------------------------------------
// S1(N) = A1 = ROTL3 (A1 + Lhi1 + S_not(N));
// S2(N) = A2 = ROTL3 (A2 + Lhi2 + S_not(N));
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
#define ROUND_1_EVEN(N) \
"	leal	"S_not(N)"(%%ebp,%%edi),%%ebp
	roll	$3,    %%eax
	roll	$3,    %%ebp
	movl	%%eax, "S1(N)"
	movl	%%ebp, "S2(N)"
	leal	(%%eax,%%edx), %%ecx
	addl	%%ecx, %%ebx
	roll	%%cl,  %%ebx
	leal	(%%ebp,%%edi), %%ecx
	addl	%%ecx, %%esi
	leal	"S_not(N+1)"(%%eax,%%ebx),%%eax
	roll	%%cl,  %%esi \n"

// S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
// S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
// Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
#define ROUND_1_ODD(N) \
"	leal	"S_not(N)"(%%ebp,%%esi),%%ebp
	roll	$3,    %%eax
	roll	$3,    %%ebp
	movl	%%eax, "S1(N)"
	movl	%%ebp, "S2(N)"
	leal	(%%eax,%%ebx), %%ecx
	addl	%%ecx, %%edx
	roll	%%cl,  %%edx
	leal	(%%ebp,%%esi), %%ecx
	addl	%%ecx, %%edi
	leal	"S_not(N+1)"(%%eax,%%edx),%%eax
	roll	%%cl,  %%edi \n"

#define ROUND_1_LAST(N) \
"	leal	"S_not(N)"(%%ebp,%%esi),%%ebp
	roll	$3,    %%eax
	roll	$3,    %%ebp
	movl	%%eax, "S1(N)"
	movl	%%ebp, "S2(N)"
	leal	(%%eax,%%ebx), %%ecx
	addl	%%ecx, %%edx
	roll	%%cl,  %%edx
	leal	(%%ebp,%%esi), %%ecx
	addl	%%ecx, %%edi
	leal	"S0_ROTL3"(%%eax,%%edx),%%eax	# wrap with start of ROUND2
	roll	%%cl,  %%edi \n"


#define ROUND_1_ODD_AND_EVEN(N1,N2) \
  ROUND_1_ODD (N1) \
  ROUND_1_EVEN(N2)

// ------------------------------------------------------------------
// S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
// S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
#define ROUND_2_EVEN(N) \
"	addl	%%edx,  %%eax
	addl	%%edi,  %%ebp
	roll	$3,     %%eax
	roll	$3,     %%ebp
	movl	%%eax,  "S1(N)"
	movl	%%ebp,  "S2(N)"
	leal	(%%eax, %%edx), %%ecx
	addl	"S1(N+1)",%%eax
	addl	%%ecx,  %%ebx
	roll	%%cl,   %%ebx
	leal	(%%ebp, %%edi), %%ecx
	addl	"S2(N+1)",%%ebp
	addl	%%ecx,  %%esi
	roll	%%cl,   %%esi \n"

#define ROUND_2_ODD(N) \
"	addl	%%ebx,  %%eax
	addl	%%esi,  %%ebp
	roll	$3,     %%eax
	roll	$3,     %%ebp
	movl	%%eax,  "S1(N)"
	movl	%%ebp,  "S2(N)"
	leal	(%%eax, %%ebx), %%ecx
	addl	"S1(N+1)",%%eax
	addl	%%ecx,  %%edx
	roll	%%cl,   %%edx
	leal	(%%ebp, %%esi), %%ecx
	addl	"S2(N+1)",%%ebp
	addl	%%ecx,  %%edi
	roll	%%cl,   %%edi \n"

#define ROUND_2_LAST(N) \
"	addl	%%ebx,  %%eax
	addl	%%esi,  %%ebp
	roll	$3,     %%eax
	roll	$3,     %%ebp
	movl	%%eax,  "S1(N)"
	movl	%%ebp,  "S2(N)"
	leal	(%%eax, %%ebx), %%ecx
#	addl	"S1(N+1)",%%eax
	addl	%%ecx,  %%edx
	roll	%%cl,   %%edx
	leal	(%%ebp, %%esi), %%ecx
	addl	"S1(0)",%%eax		# wrap with first part of ROUND3
	addl	%%ecx,  %%edi
	roll	%%cl,   %%edi \n"


// S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
// S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
// Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);

#define ROUND_2_ODD_AND_EVEN(N1,N2) \
  ROUND_2_ODD (N1) \
  ROUND_2_EVEN(N2)

// ------------------------------------------------------------------
// A = ROTL3 (A + Lhi + S(N));
// Llo = ROTL (Llo + A + Lhi, A + Lhi);
// eA = ROTL (eA ^ eB, eB) + A;

// A = ROTL3 (A + Llo + S(N));
// Lhi = ROTL (Lhi + A + Llo, A + Llo);
// eB = ROTL (eA ^ eB, eA) + A;

// A  = %eax  eA = %esi
// L0 = %ebx  eB = %edi
// L1 = %edx  .. = %ebp
#define ROUND_3_EVEN_AND_ODD(N,Sx) \
"	addl	"Sx(N)",  %%eax
	addl	%%edx,    %%eax
	roll	$3,       %%eax
	movl	%%edi,    %%ecx
	xorl	%%edi,    %%esi
	roll	%%cl,     %%esi
	leal	(%%eax,   %%edx), %%ecx
	addl	%%eax,    %%esi
	addl	%%ecx,    %%ebx
	roll	%%cl,     %%ebx
	
	addl	"Sx(N+1)",%%eax
	addl	%%ebx,    %%eax
	roll	$3,       %%eax
	movl	%%esi,    %%ecx
	xorl	%%esi,    %%edi
	roll	%%cl,     %%edi
	leal	(%%eax,   %%ebx), %%ecx
	addl	%%eax,    %%edi
	addl	%%ecx,    %%edx
	roll	%%cl,     %%edx \n"

// ------------------------------------------------------------------
// rc5_unit will get passed an RC5WorkUnit to complete
// this is where all the actually work occurs, this is where you optimize.
// assembly gurus encouraged.
// Returns number of keys checked before a possible good key is found, or
// timeslice*PIPELINE_COUNT if no keys are 'good' keys.
// (ie:      if (result == timeslice*PIPELINE_COUNT) NOTHING_FOUND
//      else if (result < timeslice*PIPELINE_COUNT) SOMETHING_FOUND at result+1
//      else SOMETHING_GET_WRONG... )

// There is no way to tell gcc to save %ebp.
//  (putting %ebp in the clobbered register list has no effect)
// Even worse, if '-fomit-frame-pointer' isn't used, gcc will compile
// this function with local variables referenced with %ebp (!!).
//
// This is why I'm saving %ebp on the stack and I'm using only static variables

u32 rc5_unit_func_p6( RC5UnitWork * rc5unitwork, u32 timeslice )
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
	movl	"RC5UnitWork_L0hi"(%%eax), %%edx  # edx = l1 = Lhi1
	movl	%%ebx, %%esi			# esi = l2 = Llo2
	leal	0x01000000(%%edx), %%edi	# edi = l3 = lhi2
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

"_bigger_loop_p6:
	addl	$"S0_ROTL3", %%ebx
	roll	$"FIRST_ROTL",  %%ebx
	movl	%%ebx, "work_pre1_r1"

	leal	"S1_S0_ROTL3"(%%ebx),%%eax
	roll	$3,    %%eax
	movl	%%eax, "work_pre2_r1"

	leal	(%%eax,%%ebx), %%ecx
	movl	%%ecx, "work_pre3_r1"

.balign 4
_loaded_p6:\n"

    /* ------------------------------ */
    /* Begin round 1 of key expansion */
    /* ------------------------------ */

"
	movl	"work_pre1_r1", %%ebx
	movl	"work_pre2_r1", %%eax
	movl	%%ebx, %%esi
	movl	%%eax, %%ebp

	movl	"work_pre3_r1", %%ecx
	addl	%%ecx, %%edx
	roll	%%cl,  %%edx
	addl	%%ecx, %%edi
	leal	"S_not(2)"(%%eax,%%edx),%%eax
	roll	%%cl,  %%edi		 \n"

	ROUND_1_EVEN            ( 2)
	ROUND_1_ODD_AND_EVEN ( 3, 4)
	ROUND_1_ODD_AND_EVEN ( 5, 6)
	ROUND_1_ODD_AND_EVEN ( 7, 8)
	ROUND_1_ODD_AND_EVEN ( 9,10)
	ROUND_1_ODD_AND_EVEN (11,12)
	ROUND_1_ODD_AND_EVEN (13,14)
	ROUND_1_ODD_AND_EVEN (15,16)
	ROUND_1_ODD_AND_EVEN (17,18)
	ROUND_1_ODD_AND_EVEN (19,20)
	ROUND_1_ODD_AND_EVEN (21,22)
	ROUND_1_ODD_AND_EVEN (23,24)
	ROUND_1_LAST         (25)


    /* ------------------------------ */
    /* Begin round 2 of key expansion */
    /* ------------------------------ */

"_end_round1_p6:
	#leal	"S0_ROTL3"(%%edx,%%eax),  %%eax		 # already in ROUND1_LAST
	leal	"S0_ROTL3"(%%edi,%%ebp),  %%ebp
	roll	$3,     %%eax
	roll	$3,     %%ebp
	movl	%%eax,  "S1(0)"
	movl	%%ebp,  "S2(0)"

	leal	(%%eax,%%edx),  %%ecx
	addl	"work_pre2_r1",%%eax
	addl	%%ecx,  %%ebx
	roll	%%cl,   %%ebx

	leal	(%%ebp,%%edi),  %%ecx
	addl	"work_pre2_r1",%%ebp
	addl	%%ecx,  %%esi
	roll	%%cl,   %%esi \n"

	ROUND_2_ODD_AND_EVEN ( 1, 2)
	ROUND_2_ODD_AND_EVEN ( 3, 4)
	ROUND_2_ODD_AND_EVEN ( 5, 6)
	ROUND_2_ODD_AND_EVEN ( 7, 8)
	ROUND_2_ODD_AND_EVEN ( 9,10)
	ROUND_2_ODD_AND_EVEN (11,12)
	ROUND_2_ODD_AND_EVEN (13,14)
	ROUND_2_ODD_AND_EVEN (15,16)
	ROUND_2_ODD_AND_EVEN (17,18)
	ROUND_2_ODD_AND_EVEN (19,20)
	ROUND_2_ODD_AND_EVEN (21,22)
	ROUND_2_ODD_AND_EVEN (23,24)
	ROUND_2_LAST (25)

    /* Save 2nd key parameters and initialize result variable

       I'm using the stack instead of a memory location, because
       gcc don't allow me to put more than 10 constraints in an
       asm() statement.
    */
"_end_round2_p6:
	movl	%%ebp,"work_key2_ebp"
	movl	%%esi,"work_key2_esi"
	movl	%%edi,"work_key2_edi" \n"

    /* ---------------------------------------------------- */
    /* Begin round 3 of key expansion mixed with encryption */
    /* ---------------------------------------------------- */
    /* (first key)              */

	// A  = %eax  eA = %esi
	// L0 = %ebx  eB = %edi
	// L1 = %edx  .. = %ebp

"	#addl	"S1(0)",%%eax	# already in ROUND_2_LAST
	addl	%%edx,  %%eax
	roll	$3,     %%eax
	movl	"work_P_0", %%esi
	leal	(%%eax, %%edx), %%ecx
	addl	%%eax,  %%esi
	addl	%%ecx,  %%ebx
	roll	%%cl,   %%ebx

	addl	"S1(1)",%%eax
	addl	%%ebx,  %%eax
	roll	$3,     %%eax
	movl	"work_P_1", %%edi
	leal	(%%eax, %%ebx), %%ecx
	addl	%%eax,  %%edi
	addl	%%ecx,  %%edx
	roll	%%cl,   %%edx \n"
	
	ROUND_3_EVEN_AND_ODD ( 2,S1)
	ROUND_3_EVEN_AND_ODD ( 4,S1)
	ROUND_3_EVEN_AND_ODD ( 6,S1)
	ROUND_3_EVEN_AND_ODD ( 8,S1)
	ROUND_3_EVEN_AND_ODD (10,S1)
	ROUND_3_EVEN_AND_ODD (12,S1)
	ROUND_3_EVEN_AND_ODD (14,S1)
	ROUND_3_EVEN_AND_ODD (16,S1)
	ROUND_3_EVEN_AND_ODD (18,S1)
	ROUND_3_EVEN_AND_ODD (20,S1)
	ROUND_3_EVEN_AND_ODD (22,S1)

  /* early exit */
"_end_round3_1_p6:
	addl	"S1(24)",%%eax
	addl	%%edx,   %%eax
	roll	$3,      %%eax
	movl	%%edi,   %%ecx
	xorl	%%edi,   %%esi
	roll	%%cl,    %%esi
	addl	%%eax,   %%esi
					
	cmp	"work_C_0", %%esi
	jne	__exit_1_p6
					
	leal	(%%eax,%%edx), %%ecx
	addl	%%ecx,   %%ebx
	roll	%%cl,    %%ebx
	addl	"S1(25)",%%eax
	addl	%%ebx,   %%eax
	roll	$3,      %%eax
	movl	%%esi,   %%ecx
	xorl	%%esi,   %%edi
	roll	%%cl,    %%edi
	addl	%%eax,   %%edi

	cmpl	"work_C_1", %%edi
	je	_full_exit_p6

.balign 4
__exit_1_p6: \n"

    /* Restore 2nd key parameters */
"	movl	"work_key2_edi",%%edx
	movl	"work_key2_esi",%%ebx
	movl	"work_key2_ebp",%%eax\n"

    /* ---------------------------------------------------- */
    /* Begin round 3 of key expansion mixed with encryption */
    /* ---------------------------------------------------- */
    /* (second key)              */

  // A  = %eax  eA = %esi
  // L0 = %ebx  eB = %edi
  // L1 = %edx  .. = %ebp

"	addl	"S2(0)",%%eax
	addl	%%edx,  %%eax
	roll	$3,     %%eax
	movl	"work_P_0", %%esi
	leal	(%%eax, %%edx), %%ecx
	addl	%%eax,  %%esi
	addl	%%ecx,  %%ebx
	roll	%%cl,   %%ebx

	addl	"S2(1)",%%eax
	addl	%%ebx,  %%eax
	roll	$3,     %%eax
	movl	"work_P_1", %%edi
	leal	(%%eax, %%ebx), %%ecx
	addl	%%eax,  %%edi
	addl	%%ecx,  %%edx
	roll	%%cl,   %%edx \n"
	
	ROUND_3_EVEN_AND_ODD ( 2,S2)
	ROUND_3_EVEN_AND_ODD ( 4,S2)
	ROUND_3_EVEN_AND_ODD ( 6,S2)
	ROUND_3_EVEN_AND_ODD ( 8,S2)
	ROUND_3_EVEN_AND_ODD (10,S2)
	ROUND_3_EVEN_AND_ODD (12,S2)
	ROUND_3_EVEN_AND_ODD (14,S2)
	ROUND_3_EVEN_AND_ODD (16,S2)
	ROUND_3_EVEN_AND_ODD (18,S2)
	ROUND_3_EVEN_AND_ODD (20,S2)
	ROUND_3_EVEN_AND_ODD (22,S2)

  /* early exit */
"_end_round3_2_p6:
	addl	"S2(24)",%%eax
	addl	%%edx,   %%eax
	roll	$3,      %%eax
	movl	%%edi,   %%ecx
	xorl	%%edi,   %%esi
	roll	%%cl,    %%esi
	addl	%%eax,   %%esi
					
	cmp	"work_C_0", %%esi
	jne	__exit_2_p6
					
	leal	(%%eax,%%edx), %%ecx
	addl	%%ecx,   %%ebx
	roll	%%cl,    %%ebx
	addl	"S2(25)",%%eax
	addl	%%ebx,   %%eax
	roll	$3,      %%eax
	movl	%%esi,   %%ecx
	xorl	%%esi,   %%edi
	roll	%%cl,    %%edi
	addl	%%eax,   %%edi

	cmpl	"work_C_1", %%edi
	jne	 __exit_2_p6
	movl	$1, "work_add_iter"
	jmp	_full_exit_p6

.balign 4
__exit_2_p6:

	movl	"work_key_hi", %%edx

"/* Jumps not taken are faster */"
	addl	$0x02000000,%%edx
	jc	_next_inc_p6

_next_iter_p6:
	movl	%%edx, "work_key_hi"
	leal	 0x01000000(%%edx), %%edi
	decl	"work_iterations"
	jg	_loaded_p6
	movl	%1, %%eax				# pointer to rc5unitwork
	movl	"work_key_lo", %%ebx
	movl	%%ebx, "RC5UnitWork_L0lo"(%%eax)	# Update real data
	movl	%%edx, "RC5UnitWork_L0hi"(%%eax)	# (used by caller)
	jmp	_full_exit_p6

.balign 4
_next_iter2_p6:
	movl	%%ebx, "work_key_lo"
	movl	%%edx, "work_key_hi"
	leal	 0x01000000(%%edx), %%edi
	movl	%%ebx, %%esi
	decl	"work_iterations"
	jg	_bigger_loop_p6
	movl	%1, %%eax				# pointer to rc5unitwork
	movl	%%ebx, "RC5UnitWork_L0lo"(%%eax)	# Update real data
	movl	%%edx, "RC5UnitWork_L0hi"(%%eax)	# (used by caller)
	jmp	_full_exit_p6

.balign 4
_next_inc_p6:
	addl	$0x00010000, %%edx
	testl	$0x00FF0000, %%edx
	jnz	_next_iter_p6

	addl	$0xFF000100, %%edx
	testl	$0x0000FF00, %%edx
	jnz	_next_iter_p6

	addl	$0xFFFF0001, %%edx
	testl	$0x000000FF, %%edx
	jnz	_next_iter_p6


	movl	"work_key_lo", %%ebx

	subl	$0x00000100, %%edx
	addl	$0x01000000, %%ebx
	jnc	_next_iter2_p6

	addl	$0x00010000, %%ebx
	testl	$0x00FF0000, %%ebx
	jnz	_next_iter2_p6

	addl	$0xFF000100, %%ebx
	testl	$0x0000FF00, %%ebx
	jnz	_next_iter2_p6

	addl	$0xFFFF0001, %%ebx
	testl	$0x000000FF, %%ebx
	jnz	_next_iter2_p6

	# Moo !
	# We have just finished checking the last key
	# of the rc5-64 keyspace...
	# Not much to do here, since we have finished the block ...


.balign 4
_full_exit_p6:
	movl	"work_save_ebp",%%ebp \n"

: "=m"(work),
  "=m"(rc5unitwork)
: "a" (rc5unitwork)
: "%eax","%ebx","%ecx","%edx","%esi","%edi","cc");

    return (timeslice - work.iterations) * 2 + work.add_iter;
}

