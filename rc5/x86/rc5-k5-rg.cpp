// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: rc5-k5-rg.cpp,v $
// Revision 1.9  1998/11/28 17:59:03  remi
// Fixed BALIGN4 macro for *BSD.
//
// Revision 1.8  1998/11/20 23:45:09  remi
// Added FreeBSD support in the BALIGN macro.
//
// Revision 1.7  1998/08/20 00:25:20  silby
// Took out PIPELINE_COUNT checks inside .cpp x86 cores - they were
// causing build problems with new PIPELINE_COUNT architecture on x86.
//
// Revision 1.6  1998/07/08 22:59:36  remi
// Lots of $Id: rc5-k5-rg.cpp,v 1.9 1998/11/28 17:59:03 remi Exp $ stuff.
//
// Revision 1.5  1998/07/08 18:47:46  remi
// $Id fun ...
//
// Revision 1.4  1998/06/14 10:03:56  skand
// define and use a preprocessor macro to hide the .balign directive for
// ancient assemblers
//
// Revision 1.3  1998/06/14 08:27:18  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.2  1998/06/14 08:13:36  friedbait
// 'Log' keywords added to maintain automatic change history
//
//
// AMD K5 optimized version
// Rémi Guyomarch <rguyom@mail.dotcom.fr>
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
//
//		no AGI					
//		roll $3,%reg = 1 cycle			
//		roll %cl,%reg = 1 cycle			
// but roll can only be executed in alu-1 (2nd insn?)	
//		no WAW stall	-|			
//		no WAR stall	-| via reg. renaming	
//		2 alus units				
//		2 load/store units			
//			1 load & 1 store		
//		     or 2 loads				
//		1 branch unit				
//		1 FP unit				
//							
// -register renaming					
// -data forwarding					
// -up to four instructions issued per cycle		
// -out of order issue and completion			
//							
// expected speed :					
//							
// PR200 = 133   / 66		           rg=286 ? / 453-497 ?
// PR166 = 116.7 / 66    v1=186 v2=335-341 rg=250 ? / 398-436 ?
// PR133 = 100   / 66           v2=287-300 rg=220   / 341-374 ?
// PR120 =  90   / 60			   rg=193 ? / 307-336 ?
// PR??? =  75   / ??    v1=120 v2=215-225 rg=165   / 256-280 ?

#if (!defined(lint) && defined(__showids__))
const char *rc5_k5_rg_cpp (void) {
return "@(#)$Id: rc5-k5-rg.cpp,v 1.9 1998/11/28 17:59:03 remi Exp $"; }
#endif

#define CORE_INCREMENTS_KEY

// This file is included from rc5.cpp so we can use __inline__.
#include "problem.h"

// With different pipeline counts for different cores, this check cannot
// be done here
//#if (PIPELINE_COUNT != 2)
//#error "Expecting pipeline count of 2"
//#endif

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif


// Stringify macro.

#define _(s)    __(s)
#define __(s)   #s

#if defined(__NetBSD__) || defined(__bsdi__) || (defined(__FreeBSD__) && !defined(__ELF__))
#define BALIGN4 ".align 2, 0x90"
#else
#define BALIGN4 ".balign 4"
#endif

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
"      #roll	%%cl,  %%edi			  1	alu	previous iteration
	addl	%%edx, %%ebx			#	alu
	leal	"S_not(N)"(%%edx,%%eax),%%eax	#  	ld
	roll	$3,    %%eax			# 1	alu
	leal	"S_not(N)"(%%edi,%%ebp),%%ebp	#	ld
	addl	%%edi, %%esi			#	alu
	roll	$3,    %%ebp			# 1	alu
	movl	%%eax, "S1(N)"			#	st
	leal   (%%eax, %%edx), %%ecx		#	ld
	addl	%%eax, %%ebx			#	alu
	roll	%%cl,  %%ebx			# 1	alu
	movl	%%ebp, "S2(N)"			#	st
	leal   (%%ebp, %%edi), %%ecx		# 	ld
	addl	%%ebp, %%esi			#	alu
	roll	%%cl,  %%esi			# 1	alu	\n"
						
// S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
// S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
// Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
#define ROUND_1_ODD(N) \
"	addl	%%ebx, %%edx			#	alu
	leal	"S_not(N)"(%%ebx,%%eax),%%eax	#  	ld
	roll	$3,    %%eax			# 1	alu
	leal	"S_not(N)"(%%esi,%%ebp),%%ebp	#	ld
	addl	%%esi, %%edi			#	alu
	roll	$3,    %%ebp			# 1	alu
	movl	%%eax, "S1(N)"			#	st
	leal	(%%eax,%%ebx), %%ecx		# 	ld
	addl	%%eax, %%edx			#	alu
	roll	%%cl,  %%edx			# 1	alu
	movl	%%ebp, "S2(N)"			#	st
	leal	(%%ebp,%%esi), %%ecx		#	ld
	addl	%%ebp, %%edi			#	alu
	roll	%%cl,  %%edi			# 1	alu	sum = 8 \n"

#define ROUND_1_ODD_AND_EVEN(N1,N2) \
	ROUND_1_ODD (N1) \
	ROUND_1_EVEN(N2)

// ------------------------------------------------------------------
// S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
// S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
#define ROUND_2_EVEN(N) \
"      #roll	%%cl,     %%edi		  3	alu	previous iteration
	addl	%%edx,    %%eax		#  	alu
	addl	"S1(N)",  %%eax		# 	ld  -	alu
	leal   (%%edx,    %%ebx), %%ebx	#	ld	
	addl	%%edi,    %%ebp		#	    -	alu
	addl	"S2(N)",  %%ebp		#		ld  -	alu
	roll	$3,       %%eax		#			alu
	movl	%%eax,    "S1(N)"	#			st
	leal   (%%eax,    %%edx), %%ecx	#			ld
	addl	%%eax,    %%ebx		# 2	alu
	roll	$3,       %%ebp		#	alu
	addl	"S1(N+1)",%%eax		#	ld  -	alu
	leal   (%%edi,    %%esi), %%esi	#	ld	
	movl	%%ebp,    "S2(N)"	#		st
	roll	%%cl,     %%ebx		#		alu
	leal   (%%ebp,    %%edi), %%ecx	#		ld
	addl	%%ebx,    %%eax		# 2	alu
	addl	%%ebp,    %%esi		#  	alu
	addl	"S2(N+1)",%%ebp		#	ld  -	alu
	roll	%%cl,     %%esi		#		alu \n"

// S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
// S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
// Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
#define ROUND_2_ODD(N) \
"	leal   (%%ebx, %%edx), %%edx	#		ld
	roll	$3,    %%eax		# 1	alu
	addl	%%esi, %%ebp		# 	alu
	leal   (%%esi, %%edi), %%edi	#	ld
	roll	$3,    %%ebp		# 1	alu
	movl	%%eax, "S1(N)"		#	st
	leal   (%%eax, %%ebx), %%ecx	#	ld
	addl	%%eax, %%edx		# 1	alu
	movl	%%ebp, "S2(N)"		#	st
	roll	%%cl,  %%edx		#	alu
	leal   (%%ebp, %%esi), %%ecx	# 1	ld
	addl	%%ebp, %%edi		#	alu
	roll	%%cl,  %%edi		# ..	alu	sum = 11 \n"

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
#define ROUND_3_EVEN_AND_ODD(N,Sx) \
"	#roll	%%cl,  %%edx		  1  alu (previous iteration)
	#addl	%%eax, %%edi		     alu (previous iteration)
	movl	"Sx(N)",  %%ebp		#   ld
	leal   (%%ebp, %%edx),%%ebp	# 1 ld
	addl	%%ebp, %%eax		#   alu (data forwarding)
	xorl	%%edi, %%esi		#   alu
	roll	$3,    %%eax		# 1 alu
	movl	%%edi, %%ecx		#   alu
	leal   (%%edx, %%ebx),%%ebp	#   ld
	roll	%%cl,  %%esi		# 1 alu
	leal   (%%eax, %%ebp),%%ebx	#   ld
	leal   (%%eax, %%edx),%%ecx	#   ld
	roll	%%cl,  %%ebx		# 1 alu
	addl	%%eax, %%esi		#   alu

	movl	"Sx(N+1)",  %%ebp	#   ld
	leal   (%%ebp, %%ebx),%%ebp	# 1 ld
	addl	%%ebp, %%eax		#   alu (data forwarding)
	xorl	%%esi, %%edi		#   alu
	roll	$3,    %%eax		# 1 alu
	movl	%%esi, %%ecx		#   alu
	leal   (%%ebx, %%edx),%%ebp	#   ld
	roll	%%cl,  %%edi		# 1 alu
	leal   (%%eax, %%ebp),%%edx	#   ld
	leal   (%%eax, %%ebx),%%ecx	#   ld
	roll	%%cl,  %%edx		# 1 alu
	addl	%%eax, %%edi		#   alu	sum = 8 \n"


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
//	(putting %ebp in the clobbered register list has no effect)
// Even worse, if '-fomit-frame-pointer' isn't used, gcc will compile
// this function with local variables referenced with %ebp (!!).
//
// This is why I'm saving %ebp on the stack and I'm using only static variables

u32 rc5_unit_func_k5( RC5UnitWork * rc5unitwork, u32 timeslice )
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

"_bigger_loop_k5:
	addl	$"S0_ROTL3", %%ebx		# 1
	roll	$"FIRST_ROTL",  %%ebx		# 3
	movl	%%ebx, "work_pre1_r1"		# 1

	leal	"S1_S0_ROTL3"(%%ebx),%%eax	# 1
	roll	$3,    %%eax			# 2
	movl	%%eax, "work_pre2_r1"		# 1

	leal	(%%eax,%%ebx), %%ecx		# 2
	movl	%%ecx, "work_pre3_r1"		# 1

"BALIGN4"
_loaded_k5:\n"

    /* ------------------------------ */
    /* Begin round 1 of key expansion */
    /* ------------------------------ */

"	movl	"work_pre1_r1", %%ebx		# 1 ld
	movl	"work_pre2_r1", %%eax		#   ld
	movl	%%ebx, %%esi			# 1 alu
	movl	%%eax, %%ebp			#   alu
	movl	"work_pre3_r1", %%ecx		#   ld
	addl	%%ecx, %%edx			# 1 alu
	addl	%%ecx, %%edi			#   alu
	roll	%%cl,  %%edx			# 1 alu
	roll	%%cl,  %%edi			#   alu		sum = 4 \n"

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
	ROUND_1_ODD          (25)


    /* ------------------------------ */
    /* Begin round 2 of key expansion */
    /* ------------------------------ */

      /* xS00 & yS00 has been precomputed (see start of round 1) */
      /* so we can skip two reads from memory */
	
	// xA = xS00 = ROTL3(xS00 + xA + L_1);
	// yA = yS00 = ROTL3(yS00 + yA + L_3);
/* modified in : */
	// xA = xS00 = ROTL3(_init + xA + L_1);
	// yA = yS00 = ROTL3(_init + yA + L_3);

"_end_round1_k5:
       #roll	%%cl,   %%edi		   1	alu	previous iteration (round 1)
	addl	%%edx,  %%eax		#  	alu
	addl	$"S0_ROTL3", %%eax	# 1	alu
	leal   (%%edx,  %%ebx), %%ebx	#	ld
	addl	%%edi,  %%ebp		#	alu
	addl	$"S0_ROTL3", %%ebp	# 1	alu
	roll	$3,     %%eax		#	alu
	movl	%%eax,  "S1(0)"		#	st
	leal   (%%eax,  %%edx), %%ecx	#	ld
	addl	%%eax,  %%ebx		# 2	alu
	roll	$3,     %%ebp		#	alu
	addl	"work_pre2_r1",%%eax	#	ld  -	alu
	leal   (%%edi,  %%esi), %%esi	#	ld	
	movl	%%ebp,  "S2(0)"		#		st
	roll	%%cl,   %%ebx		#		alu
	leal   (%%ebp,  %%edi), %%ecx	#		ld
	addl	%%ebx,  %%eax		# 2	alu
	addl	%%ebp,  %%esi		#  	alu
	addl	"work_pre2_r1",%%ebp	#	ld  -	alu
	roll	%%cl,   %%esi		#		alu \n"

        ROUND_2_ODD (1)
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
        ROUND_2_EVEN_AND_ODD (24)

    /* Save 2nd key parameters and initialize result variable

       I'm using the stack instead of a memory location, because
       gcc don't allow me to put more than 10 constraints in an
       asm() statement.
    */
"_end_round2_k5:
	movl	%%ebp, "work_key2_ebp"
	movl	%%esi, "work_key2_esi"
	movl	%%edi, "work_key2_edi" \n"

    /* ---------------------------------------------------- */
    /* Begin round 3 of key expansion mixed with encryption */
    /* ---------------------------------------------------- */
    /* (first key)					    */

	// A  = %eax  eA = %esi
	// L0 = %ebx  eB = %edi
	// L1 = %edx  .. = %ebp

	/* A = ROTL3(S00 + A + L1); */
	/* eA = P_0 + A; */
	/* L0 = ROTL(L0 + A + L1, A + L1);*/
"	addl	%%edx,  %%eax		# 2	alu
	addl	"S1(0)",%%eax		#	ld  -	alu
	movl	"work_P_0",%%esi		#	ld
	addl	%%edx,  %%ebx		#	alu
	roll	$3,     %%eax		# 1	alu
	leal	(%%eax, %%edx), %%ecx	# 1	ld
	addl	%%eax,  %%ebx		#	alu
	roll	%%cl,   %%ebx		# 2	alu
	addl	%%eax,  %%esi		#	alu \n"
	/* A = ROTL3(S01 + A + L0); */
	/* eB = P_1 + A; */
	/* L1 = ROTL(L1 + A + L0, A + L0);*/
"	leal   (%%eax,  %%ebx),%%eax	#	ld	
	addl	"S1(1)",%%eax		#	ld  -	alu
	movl	"work_P_1", %%edi	#		ld
	roll	$3,     %%eax		# 1	alu
	leal   (%%ebx,  %%edx),%%ebp	#	ld
	leal   (%%eax,  %%ebp),%%edx	# 1	ld
	leal   (%%eax,  %%ebx),%%ecx	#	ld
	roll	%%cl,   %%edx		# 1	alu
	addl	%%eax,  %%edi		#	alu	sum = 9 \n"
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
"_end_round3_1_k5:
	addl	%%edx, %%eax
	movl	"S1(24)",%%ebp
	movl	%%edi, %%ecx
	addl	%%ebp, %%eax
	xorl	%%edi, %%esi
	roll	$3,    %%eax
	roll	%%cl,  %%esi
	addl	%%eax, %%esi
					
	cmpl	"work_C_0", %%esi
	jne	__exit_1_k5
					
	leal	(%%eax,%%edx), %%ecx
	movl	"S1(25)", %%ebp
	addl	%%ecx, %%ebx
	roll	%%cl,  %%ebx
	addl	%%ebx, %%eax
	movl	%%esi, %%ecx
	addl	%%ebp, %%eax
	xorl	%%esi, %%edi
	roll	$3,    %%eax
	roll	%%cl,  %%edi
	addl	%%eax, %%edi

	cmpl	"work_C_1", %%edi
	je	_full_exit_k5

"BALIGN4"
__exit_1_k5: \n"

    /* Restore 2nd key parameters */
"	movl	"work_key2_edi", %%edx
	movl	"work_key2_esi", %%ebx
	movl	"work_key2_ebp", %%eax \n"

    /* ---------------------------------------------------- */
    /* Begin round 3 of key expansion mixed with encryption */
    /* ---------------------------------------------------- */
    /* (second key)					    */

	// A  = %eax  eA = %esi
	// L0 = %ebx  eB = %edi
	// L1 = %edx  .. = %ebp

	/* A = ROTL3(S00 + A + L1); */
	/* eA = P_0 + A; */
	/* L0 = ROTL(L0 + A + L1, A + L1);*/
"	addl	%%edx,  %%eax		# 2	alu
	addl	"S2(0)",%%eax		#	ld  -	alu
	movl	"work_P_0", %%esi	#	ld
	addl	%%edx,  %%ebx		#	alu
	roll	$3,     %%eax		# 1	alu
	leal	(%%eax, %%edx), %%ecx	# 1	ld
	addl	%%eax,  %%ebx		#	alu
	roll	%%cl,   %%ebx		# 2	alu
	addl	%%eax,  %%esi		#	alu \n"
	/* A = ROTL3(S01 + A + L0); */
	/* eB = P_1 + A; */
	/* L1 = ROTL(L1 + A + L0, A + L0);*/
"	leal   (%%eax,  %%ebx),%%eax	#	ld	
	addl	"S2(1)",%%eax		#	ld  -	alu
	movl	"work_P_1", %%edi	#		ld
	roll	$3,     %%eax		# 1	alu
	leal   (%%ebx,  %%edx),%%ebp	#	ld
	leal   (%%eax,  %%ebp),%%edx	# 1	ld
	leal   (%%eax,  %%ebx),%%ecx	#	ld
	roll	%%cl,   %%edx		# 1	alu
	addl	%%eax,  %%edi		#	alu	sum = 9 \n"
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
"_end_round3_2_k5:
	addl	%%edx, %%eax
	movl	"S2(24)",%%ebp
	movl	%%edi, %%ecx
	addl	%%ebp, %%eax
	xorl	%%edi, %%esi
	roll	$3,    %%eax
	roll	%%cl,  %%esi
	addl	%%eax, %%esi
					
	cmpl	"work_C_0", %%esi
	jne	__exit_2_k5
					
	leal	(%%eax,%%edx), %%ecx
	movl	"S2(25)", %%ebp
	addl	%%ecx, %%ebx
	roll	%%cl,  %%ebx
	addl	%%ebx, %%eax
	movl	%%esi, %%ecx
	addl	%%ebp, %%eax
	xorl	%%esi, %%edi
	roll	$3,    %%eax
	roll	%%cl,  %%edi
	addl	%%eax, %%edi

	cmpl	"work_C_1", %%edi
	jne	__exit_2_k5
	movl	$1, "work_add_iter"
	jmp	_full_exit_k5

"BALIGN4"
__exit_2_k5:

	movl	"work_key_hi", %%edx

"/* Jumps not taken are faster */"
	addl	$0x02000000,%%edx
	jc	_next_inc_k5

_next_iter_k5:
	movl	%%edx, "work_key_hi"
	leal	 0x01000000(%%edx), %%edi
	decl	"work_iterations"
	jg	_loaded_k5
	movl	%1, %%eax				# pointer to rc5unitwork
	movl	"work_key_lo", %%ebx
	movl	%%ebx, "RC5UnitWork_L0lo"(%%eax)	# Update real data
	movl	%%edx, "RC5UnitWork_L0hi"(%%eax)	# (used by caller)
	jmp	_full_exit_k5

"BALIGN4"
_next_iter2_k5:
	movl	%%ebx, "work_key_lo"
	movl	%%edx, "work_key_hi"
	leal	 0x01000000(%%edx), %%edi
	movl	%%ebx, %%esi
	decl	"work_iterations"
	jg	_bigger_loop_k5
	movl	%1, %%eax				# pointer to rc5unitwork
	movl	%%ebx, "RC5UnitWork_L0lo"(%%eax)	# Update real data
	movl	%%edx, "RC5UnitWork_L0hi"(%%eax)	# (used by caller)
	jmp	_full_exit_k5

"BALIGN4"
_next_inc_k5:
	addl	$0x00010000, %%edx
	testl	$0x00FF0000, %%edx
	jnz	_next_iter_k5

	addl	$0xFF000100, %%edx
	testl	$0x0000FF00, %%edx
	jnz	_next_iter_k5

	addl	$0xFFFF0001, %%edx
	testl	$0x000000FF, %%edx
	jnz	_next_iter_k5


	movl	"work_key_lo", %%ebx

	subl	$0x00000100, %%edx
	addl	$0x01000000, %%ebx
	jnc	_next_iter2_k5

	addl	$0x00010000, %%ebx
	testl	$0x00FF0000, %%ebx
	jnz	_next_iter2_k5

	addl	$0xFF000100, %%ebx
	testl	$0x0000FF00, %%ebx
	jnz	_next_iter2_k5

	addl	$0xFFFF0001, %%ebx
	testl	$0x000000FF, %%ebx
	jnz	_next_iter2_k5

	# Moo !
	# We have just finished checking the last key
	# of the rc5-64 keyspace...
	# Not much to do here, since we have finished the block ...


"BALIGN4"
_full_exit_k5:
	movl	"work_save_ebp",%%ebp \n"

: "=m"(work),
  "=m"(rc5unitwork)
: "a" (rc5unitwork)
: "%eax","%ebx","%ecx","%edx","%esi","%edi","cc");

    return (timeslice - work.iterations) * 2 + work.add_iter;
}
