// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: rc5-486-smc-rg.cpp,v $
// Revision 1.5  1998/11/28 17:52:18  remi
// Corrected BALIGN4 macro for *BSD.
//
// Revision 1.4  1998/11/28 17:37:52  remi
// This one is a bit faster, as I removed an AGI stall in ROUND3. Does
// 102.3 kkeys/s on my DX4/100.
// Renamed the function to integrate it in the client, without replacing
// the original, multithreaded 386/486 core.
//
// Revision 1.3  1998/11/20 23:45:08  remi
// Added FreeBSD support in the BALIGN macro.
//
// Revision 1.2  1998/11/18 16:04:19  remi
// Nothing changed, just a few numbers :
//
//                          DX4/100         P/200 MMX       PII/400
//
// classic 486 core         97 kkeys/s     224 kkeys/s      984 kkeys/s
// 486 SMC core            102 kkeys/s     145 kkeys/s       92 kkeys/s (!)
// using the best core      //             400 kkeys/s     1120 kkeys/s
// for the given processor
//
// DX4/100 under Linux
// P/200 MMX and PII/400 under NT4
//
// Revision 1.1  1998/11/18 05:53:07  remi
// Ok, this one does 101.8 kkeys/s on my DX4/100 instead of 97 kkeys/s
// for the previous 386/486 code.
//
// In this core I modify the code itself at run-time to store
// intermediate results instead of using the stack. Thus I can avoid
// reading from memory, and it save some cycles.
//
// This core will never be multithread-safe. That's not a big deal,
// because there is very few 486 SMP boards anyway.
//
// This speed-up is absolutely dependent on the correct alignment of
// instructions to be modified. Thus I had to insert some nop's, and to
// change some instructions to inflate by a few bytes the code.
//
// It require some support from the compiler tools too. On most platforms, a
// given piece of code isn't allowed to write in the .text section. On
// linux, passing "-Xlinker -omagic" (!) to GCC at link time does the
// trick. On Win32 there's a command-line utility in MS VC++ 5.0 which
// can modify flags in the EXE header, but I don't recall the exact
// command right now.
// Without this support from the compiler / linker, your binary will
// probably segfault as soon as it enters this code.
//
// Revision 1.7  1998/08/20 00:25:17  silby
// Took out PIPELINE_COUNT checks inside .cpp x86 cores - they were causing
// build problems with new PIPELINE_COUNT architecture on x86.
//
// Revision 1.6  1998/07/08 22:59:33  remi
// Lots of $Id: rc5-486-smc-rg.cpp,v 1.5 1998/11/28 17:52:18 remi Exp $ stuff.
//
// Revision 1.5  1998/07/08 18:47:43  remi
// $Id fun ...
//
// Revision 1.4  1998/06/14 10:03:54  skand
// define and use a preprocessor macro to hide the .balign directive for
// ancient assemblers
//
// Revision 1.3  1998/06/14 08:27:16  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.2  1998/06/14 08:13:33  friedbait
// 'Log' keywords added to maintain automatic change history
//
//
// 386/486 optimized version
// Rémi Guyomarch - rguyom@mail.dotcom.fr
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
// Checking two keys at once is still a (slight) win for a 486
// probably because less load/store operations

#if (!defined(lint) && defined(__showids__))
const char *rc5_486_smc_rg_cpp (void) {
return "@(#)$Id: rc5-486-smc-rg.cpp,v 1.5 1998/11/28 17:52:18 remi Exp $"; }
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
    u32 save_ebp;	// +  4
    u32 key2_ebp;	// +  8
    u32 key2_edi;	// + 12
    u32 key2_esi;	// + 16
    u32 key_hi;		// + 20
    u32 key_lo;		// + 24
    u32 iterations;	// + 28
    u32 pre1_r1;	// + 32
    u32 pre2_r1;	// + 36
    u32 pre3_r1;	// + 40
};

//  Offsets to access work_struct fields.

#define	work_add_iter   "0+%0"
#define	work_save_ebp   "4+%0" 
#define	work_key2_ebp   "8+%0" 
#define	work_key2_edi   "12+%0"
#define	work_key2_esi   "16+%0"
#define work_key_hi     "20+%0"
#define work_key_lo     "24+%0"
#define work_iterations "28+%0"
#define work_pre1_r1    "32+%0"
#define work_pre2_r1    "36+%0"
#define work_pre3_r1    "40+%0"

//  Macros to access the S arrays.

//#define S1(N)    _(((N)*4)+4+%0)
//#define S2(N)    _(((N)*4)+108+%0)

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
#define ROUND_1_EVEN(N)	\
"	leal	"S_not(N)"(%%eax,%%edx),%%eax	# 2
	leal	"S_not(N)"(%%ebp,%%edi),%%ebp	# 2
	roll	$3,    %%eax			# 2
	movl	%%eax, _modif486_r2_S1_"_(N)"+3	# 1
	roll	$3,    %%ebp			# 2
	movl	%%ebp, _modif486_r2_S2_"_(N)"+3	# 1
	leal	(%%eax,%%edx), %%ecx		# 2
	addl	%%ecx, %%ebx			# 1
	roll	%%cl,  %%ebx			# 3
	leal	(%%ebp,%%edi), %%ecx		# 2
	addl	%%ecx, %%esi			# 1
	roll	%%cl,  %%esi			# 3	sum = 22 \n"

// S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
// S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
// Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
#define ROUND_1_ODD(N) \
"	leal	"S_not(N)"(%%eax,%%ebx),%%eax	# 2
	leal	"S_not(N)"(%%ebp,%%esi),%%ebp	# 2
	roll	$3,    %%eax			# 2
	movl	%%eax, _modif486_r2_S1_"_(N)"+3	# 1
	roll	$3,    %%ebp			# 2
	movl	%%ebp, _modif486_r2_S2_"_(N)"+3	# 1
	leal	(%%eax,%%ebx), %%ecx		# 2
	addl	%%ecx, %%edx			# 1
	roll	%%cl,  %%edx			# 3
	leal	(%%ebp,%%esi), %%ecx		# 2
	addl	%%ecx, %%edi			# 1
	roll	%%cl,  %%edi			# 3	sum = 22 \n"

#define ROUND_1_EVEN_AND_ODD(N1,N2) \
	ROUND_1_EVEN(N1) \
	ROUND_1_ODD (N2)

// ------------------------------------------------------------------
// S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
// S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
// Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
// S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
// S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
#define ROUND_2_EVEN(N) \
"	roll	$3,     %%eax			# 2 - 3
	movl	%%eax, _modif486_r3_S1_"_(N)"+1	# 1 - 5
	leal	(%%eax, %%edx), %%ecx		# 2 - 3
_modif486_r2_S2_"_(N)":				#	20
	leal	0x90ABCDEF(%%edi,%%ebp),%%ebp	# 2 - 7
	addl	%%ecx,  %%ebx			# 1 - 2
	roll	%%cl,   %%ebx			# 3 - 2
	roll	$3,     %%ebp			# 2 - 3
	movl	%%ebp, _modif486_r3_S2_"_(N)"+1	# 1 - 6
	leal	(%%ebp, %%edi), %%ecx		# 2 - 4
	addl	%%ecx,  %%esi			# 1 - 2
	roll	%%cl,   %%esi			# 3 - 2	 \n"	
	
#define ROUND_2_ODD(N2,N3) \
"_modif486_r2_S1_"_(N2)":
	leal	0x12345678(%%eax,%%ebx),%%eax	# 2 - 7
	roll	$3,     %%eax			# 2 - 3
	movl	%%eax, _modif486_r3_S1_"_(N2)"+1# 1 - 5
	leal	(%%eax, %%ebx), %%ecx		# 2 - 3
	addl	%%ecx,  %%edx			# 1 - 2
_modif486_r2_S2_"_(N2)":			#	20
	leal	0x90ABCDEF(%%esi,%%ebp),%%ebp	# 2 - 7
	roll	%%cl,   %%edx			# 3 - 2
	roll	$3,     %%ebp			# 2 - 3
	movl	%%ebp, _modif486_r3_S2_"_(N2)"+1# 1 - 6
	leal	(%%ebp, %%esi), %%ecx		# 2 - 4
	addl	%%ecx,  %%edi			# 1 - 2
_modif486_r2_S1_"_(N3)":			#	24
	leal	0x12345678(%%eax,%%edx),%%eax	# 2 - 7
	roll	%%cl,   %%edi			# 3 - 2  \n"

#define ROUND_2_EVEN_AND_ODD(N1,N2,N3) \
	ROUND_2_EVEN(N1) \
	ROUND_2_ODD (N2,N3)

#define ROUND_2_ODD_LAST(N2) \
"_modif486_r2_S1_"_(N2)":
	leal	0x12345678(%%eax,%%ebx),%%eax	# 2 - 7
	roll	$3,     %%eax			# 2 - 3
	movl	%%eax, _modif486_r3_S1_"_(N2)"+1# 1 - 5
	leal	(%%eax, %%ebx), %%ecx		# 2 - 3
	addl	%%ecx,  %%edx			# 1 - 2
_modif486_r2_S2_"_(N2)":			#	20
	leal	0x90ABCDEF(%%esi,%%ebp),%%ebp	# 2 - 7
	roll	%%cl,   %%edx			# 3 - 2
	roll	$3,     %%ebp			# 2 - 3
	movl	%%ebp, _modif486_r3_S2_"_(N2)"+1# 1 - 6
	leal	(%%ebp, %%esi), %%ecx		# 2 - 4
	addl	%%ecx,  %%edi			# 1 - 2
	roll	%%cl,   %%edi			# 3 - 2  \n"

#define ROUND_2_EVEN_AND_ODD_LAST(N1,N2) \
	ROUND_2_EVEN(N1) \
	ROUND_2_ODD_LAST(N2)

// ------------------------------------------------------------------
// It's faster to do 1 key at a time with round3 and encryption mixed
// than to do 2 keys at once but round3 and encryption separated
// Too bad x86 hasn't more registers ...
	
// eA1 = ROTL (eA1 ^ eB1, eB1) + (A1 = ROTL3 (A1 + Lhi1 + S1(N)));
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// eB1 = ROTL (eA1 ^ eB1, eA1) + (A1 = ROTL3 (A1 + Llo1 + S1(N)));
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);

// A  = %eax  eA = %esi
// L0 = %ebx  eB = %edi
// L1 = %edx  .. = %ebp
#define ROUND_3_EVEN_AND_ODD(Sx,N1,N2) \
"_modif486_r3_"_(Sx)"_"_(N1)":
	addl	$0x12345678, %%eax		# 1 - 5
	addl	%%edx,    %%eax			# 1 - 2
	roll	$3,       %%eax			# 2 - 3
	movl	%%edi,    %%ecx			# 1 - 2
	xorl	%%edi,    %%esi			# 1 - 2
	roll	%%cl,     %%esi			# 3 - 2
	addl	%%eax,    %%esi			# 1 - 2
	movl	%%eax,    %%ecx			# 1 - 2
	addl	%%edx,    %%ecx			# 1 - 2
	addl	%%ecx,    %%ebx			# 1 - 2
_modif486_r3_"_(Sx)"_"_(N2)":			#	24
	addl	$0x12345678, %%eax		# 1 - 5
	roll	%%cl,     %%ebx			# 3 - 2
	addl	%%ebx,    %%eax			# 1 - 2
	roll	$3,       %%eax			# 2 - 3
	movl	%%esi,    %%ecx			# 1 - 2
	xorl	%%esi,    %%edi			# 1 - 2
	roll	%%cl,     %%edi			# 3 - 2
	addl	%%eax,    %%edi			# 1 - 2
	movl	%%eax,    %%ecx			# 1 - 2
	addl	%%ebx,    %%ecx			# 1 - 2
	addl	%%ecx,    %%edx			# 1 - 2
	roll	%%cl,     %%edx			# 3 - 2	28	sum = 32 \n"


// ------------------------------------------------------------------
// rc5_unit will get passed an RC5WorkUnit to complete
// this is where all the actually work occurs, this is where you optimize.
// assembly gurus encouraged.
// Returns number of keys checked before a possible good key is found, or
// timeslice*PIPELINE_COUNT if no keys are 'good' keys.
// (ie:      if (result == timeslice*PIPELINE_COUNT) NOTHING_FOUND
//      else if (result < timeslice*PIPELINE_COUNT) SOMETHING_FOUND at result+1
//      else SOMETHING_WENT_WRONG... )

// There is no way to tell gcc to save %ebp.
//	(putting %ebp in the clobbered register list has no effect)
// Even worse, if '-fomit-frame-pointer' isn't used, gcc will compile
// this function with local variables referenced with %ebp (!!).
//
// I use a structure to make this function multi-thread safe.
// (can't use static variables, and can't use push/pop in this
//  function because &work_struct is relative to %esp)

extern "C" u32 rc5_unit_func_486_smc( RC5UnitWork * rc5unitwork, u32 timeslice ) ;

//static
u32 rc5_unit_func_486_smc( RC5UnitWork * rc5unitwork, u32 timeslice ) 
{
    work_struct work;

    work.iterations = timeslice;
    work.add_iter = 0;

    __asm__ __volatile__ (

	/* save %ebp */
"	movl	%%ebp, "work_save_ebp" \n"
	
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
	movl	%%ebp, 	_modif_486_work_P_0_1 + 2
	movl	%%ebp, 	_modif_486_work_P_0_2 + 2
	movl	"RC5UnitWork_plainhi"(%%eax), %%ebp
	movl	%%ebp, 	_modif_486_work_P_1_1 + 2
	movl	%%ebp, 	_modif_486_work_P_1_2 + 2
	movl	"RC5UnitWork_cypherlo"(%%eax), %%ebp
	movl	%%ebp, _modif486_work_C_0_1 + 2
	movl	%%ebp, _modif486_work_C_0_2 + 2
	movl	"RC5UnitWork_cypherhi"(%%eax), %%ebp
	movl	%%ebp, _modif486_work_C_1_1 + 2
	movl	%%ebp, _modif486_work_C_1_2 + 2  \n"

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

"_bigger_loop_486:
	addl	$"S0_ROTL3", %%ebx		# 1
	roll	$"FIRST_ROTL",  %%ebx		# 3
	movl	%%ebx, "work_pre1_r1"		# 1

	leal	"S1_S0_ROTL3"(%%ebx),%%eax	# 1
	roll	$3,    %%eax			# 2
	movl	%%eax, "work_pre2_r1"		# 1

	leal	(%%eax,%%ebx), %%ecx		# 2
	movl	%%ecx, "work_pre3_r1"		# 1

"BALIGN4"
_loaded_486:\n"

    /* ------------------------------ */
    /* Begin round 1 of key expansion */
    /* ------------------------------ */

"	movl	"work_pre1_r1", %%ebx		# 1
	movl	%%ebx, %%esi			# 1
	movl	"work_pre2_r1", %%eax		# 1
	movl	%%eax, %%ebp			# 1
	movl	"work_pre3_r1", %%ecx		# 1
	addl	%%ecx, %%edx			# 1
	roll	%%cl,  %%edx			# 3
	addl	%%ecx, %%edi			# 1
	roll	%%cl,  %%edi			# 3	sum = 12 \n"

	ROUND_1_EVEN_AND_ODD ( 2, 3)
	ROUND_1_EVEN_AND_ODD ( 4, 5)
	ROUND_1_EVEN_AND_ODD ( 6, 7)
	ROUND_1_EVEN_AND_ODD ( 8, 9)
	ROUND_1_EVEN_AND_ODD (10,11)
	ROUND_1_EVEN_AND_ODD (12,13)
	ROUND_1_EVEN_AND_ODD (14,15)
	ROUND_1_EVEN_AND_ODD (16,17)
	ROUND_1_EVEN_AND_ODD (18,19)
	ROUND_1_EVEN_AND_ODD (20,21)
	ROUND_1_EVEN_AND_ODD (22,23)
	ROUND_1_EVEN_AND_ODD (24,25)


    /* ------------------------------ */
    /* Begin round 2 of key expansion */
    /* ------------------------------ */

"_end_round1_486:
	leal	"S0_ROTL3"(%%eax,%%edx),%%eax	# 2
	leal	"S0_ROTL3"(%%ebp,%%edi),%%ebp	# 2
	roll	$3,    %%eax			# 2
	movl	%%eax, _modif486_r3_S1_0 + 1	# 1
	roll	$3,    %%ebp			# 2
	movl	%%ebp, _modif486_r3_S2_0 + 1	# 1
	
	movl	%%eax, %%ecx			# 1
	addl	%%edx, %%ecx			# 1
	addl	%%ecx, %%ebx			# 1
	roll	%%cl,  %%ebx			# 3
	leal	(%%ebp,%%edi), %%ecx		# 2
	addl	%%ecx, %%esi			# 1
	roll	%%cl,  %%esi			# 3 \n"

"	movl	"work_pre2_r1", %%ecx		# 1
	addl	%%ebx, %%eax			# 1
	addl	%%ecx, %%eax			# 1
	addl	%%esi, %%ebp			# 1
	addl	%%ecx, %%ebp			# 1
	roll	$3,    %%eax			# 2
	movl	%%eax, _modif486_r3_S1_1 + 1	# 1
	roll	$3,    %%ebp			# 2
	movl	%%ebp, _modif486_r3_S2_1 + 1	# 1
	leal	(%%eax,%%ebx), %%ecx		# 2
	addl	%%ecx, %%edx			# 1
	roll	%%cl,  %%edx			# 3
	leal	(%%ebp,%%esi), %%ecx		# 2
	addl	%%ecx, %%edi			# 1
_modif486_r2_S1_2:
	leal	0x12345678(%%eax,%%edx),%%eax	# 2 - 7
	roll	%%cl,  %%edi			# 3	sum = 23 \n"

	ROUND_2_EVEN_AND_ODD      ( 2, 3, 4)
	ROUND_2_EVEN_AND_ODD      ( 4, 5, 6)
	ROUND_2_EVEN_AND_ODD      ( 6, 7, 8)
	ROUND_2_EVEN_AND_ODD      ( 8, 9,10)
	ROUND_2_EVEN_AND_ODD      (10,11,12)
	ROUND_2_EVEN_AND_ODD      (12,13,14)
	ROUND_2_EVEN_AND_ODD      (14,15,16)
	ROUND_2_EVEN_AND_ODD      (16,17,18)
	ROUND_2_EVEN_AND_ODD      (18,19,20)
	ROUND_2_EVEN_AND_ODD      (20,21,22)
	ROUND_2_EVEN_AND_ODD      (22,23,24)
	ROUND_2_EVEN_AND_ODD_LAST (24,25)

    /* Save 2nd key parameters and initialize result variable */
"_end_round2_486:
	movl	%%ebp,"work_key2_ebp"
	movl	%%esi,"work_key2_esi"
	movl	%%edi,"work_key2_edi" \n"

    /* ---------------------------------------------------- */
    /* Begin round 3 of key expansion mixed with encryption */
    /* ---------------------------------------------------- */
    /* (first key)					    */

	// A  = %eax  eA = %esi
	// L0 = %ebx  eB = %edi
	// L1 = %edx  .. = %ebp

"_modif486_r3_S1_0:
	addl	$0x12345678, %%eax	# 1 - 5
	addl	%%edx,    %%eax		# 1 - 2	A = ROTL3(S00 + A + L1);
	roll	$3,     %%eax		# 2
_modif_486_work_P_0_1:
	leal	0x12345678(%%eax),%%esi	# 1	eA = P_0 + A;
	movl	%%eax,  %%ecx		# 1
	addl	%%edx,  %%ecx		# 1	L0 = ROTL(L0 + A + L1, A + L1);
	addl	%%ecx,  %%ebx		# 1
	roll	%%cl,   %%ebx		# 3
	       			
_modif486_r3_S1_1:
	addl	$0x90ABCDEF,%%eax	# 2
	addl	%%ebx,  %%eax		# 1	A = ROTL3(S01 + A + L0);
	roll	$3,     %%eax		# 2
_modif_486_work_P_1_1:
	leal	0x12345678(%%eax),%%edi	# 1	eB = P_1 + A;
	movl	%%eax,  %%ecx		# 1
	addl	%%ebx,  %%ecx		# 1	L1 = ROTL(L1 + A + L0, A + L0);
	addl	%%ecx,  %%edx		# 1
	roll	%%cl,   %%edx		# 3	sum = 26 \n"

	ROUND_3_EVEN_AND_ODD (S1, 2, 3)
	ROUND_3_EVEN_AND_ODD (S1, 4, 5)
	ROUND_3_EVEN_AND_ODD (S1, 6, 7)
	ROUND_3_EVEN_AND_ODD (S1, 8, 9)
	ROUND_3_EVEN_AND_ODD (S1,10,11)
	ROUND_3_EVEN_AND_ODD (S1,12,13)
	ROUND_3_EVEN_AND_ODD (S1,14,15)
	ROUND_3_EVEN_AND_ODD (S1,16,17)
	ROUND_3_EVEN_AND_ODD (S1,18,19)
	ROUND_3_EVEN_AND_ODD (S1,20,21)
	ROUND_3_EVEN_AND_ODD (S1,22,23)

	/* early exit */
"_end_round3_1_486:
_modif486_r3_S1_24:
	addl	$0x12345678, %%eax	# 1 - 5
	addl	%%edx,    %%eax		# 1 - 2	A = ROTL3(S24 + A + L1);
	roll	$3,      %%eax		# 2
	movl	%%edi,   %%ecx		# 1	eA = ROTL(eA ^ eB, eB) + A
	xorl	%%edi,   %%esi		# 1
	roll	%%cl,    %%esi		# 3
	addl	%%eax,   %%esi		# 1
					
_modif486_work_C_0_1:			# this label isn't aligned, but we don't care
	cmp	$0x12345678, %%esi	# since we write on it only one time per call
	jne	__exit_1_486
					
	movl	%%eax,   %%ecx		# 1
	addl	%%edx,   %%ecx		# 1	L0 = ROTL(L0 + A + L1, A + L1);
	addl	%%ecx,   %%ebx		# 1
	roll	%%cl,    %%ebx		# 3
	movl	%%esi,   %%ecx		# 1	eB = ROTL(eB ^ eA, eA) + A
_modif486_r3_S1_25:
	addl	$0x90ABCDEF,%%eax	# 2
	addl	%%ebx,   %%eax		# 1	A = ROTL3(S25 + A + L0);
	roll	$3,      %%eax		# 2
	xorl	%%esi,   %%edi		# 1
	roll	%%cl,    %%edi		# 3
	addl	%%eax,   %%edi		# 1

_modif486_work_C_1_1:
	cmpl	$0x12345678, %%edi
	je	_full_exit_486

"BALIGN4"
nop	# spacers, won't be executed anyway
nop
nop
__exit_1_486: \n"

    /* Restore 2nd key parameters */
"	movl	"work_key2_edi", %%edx
	movl	"work_key2_esi", %%ebx
	movl	"work_key2_ebp", %%eax\n"

    /* ---------------------------------------------------- */
    /* Begin round 3 of key expansion mixed with encryption */
    /* ---------------------------------------------------- */
    /* (second key)					    */

	// A  = %eax  eA = %esi
	// L0 = %ebx  eB = %edi
	// L1 = %edx  .. = %ebp

"_modif486_r3_S2_0:
	addl	$0x12345678, %%eax	# 1 - 5
	addl	%%edx,    %%eax		# 1 - 2	A = ROTL3(S00 + A + L1);
	roll	$3,     %%eax		# 2
_modif_486_work_P_0_2:
	leal	0x12345678(%%eax),%%esi	# 1	eA = P_0 + A;
	movl	%%eax,  %%ecx		# 1
	addl	%%edx,  %%ecx		# 1	L0 = ROTL(L0 + A + L1, A + L1);
	addl	%%ecx,  %%ebx		# 1
	roll	%%cl,   %%ebx		# 3
	       			
_modif486_r3_S2_1:
	addl	$0x90ABCDEF,%%eax	# 2
	addl	%%ebx,  %%eax		# 1	A = ROTL3(S01 + A + L0);
	roll	$3,     %%eax		# 2
_modif_486_work_P_1_2:
	leal	0x12345678(%%eax),%%edi	# 1	eB = P_1 + A;
	movl	%%eax,  %%ecx		# 1
	addl	%%ebx,  %%ecx		# 1	L1 = ROTL(L1 + A + L0, A + L0);
	addl	%%ecx,  %%edx		# 1
	roll	%%cl,   %%edx		# 3	sum = 26 \n"

	ROUND_3_EVEN_AND_ODD (S2, 2, 3)
	ROUND_3_EVEN_AND_ODD (S2, 4, 5)
	ROUND_3_EVEN_AND_ODD (S2, 6, 7)
	ROUND_3_EVEN_AND_ODD (S2, 8, 9)
	ROUND_3_EVEN_AND_ODD (S2,10,11)
	ROUND_3_EVEN_AND_ODD (S2,12,13)
	ROUND_3_EVEN_AND_ODD (S2,14,15)
	ROUND_3_EVEN_AND_ODD (S2,16,17)
	ROUND_3_EVEN_AND_ODD (S2,18,19)
	ROUND_3_EVEN_AND_ODD (S2,20,21)
	ROUND_3_EVEN_AND_ODD (S2,22,23)

	/* early exit */
"_end_round3_2_486:
_modif486_r3_S2_24:
	addl	$0x12345678, %%eax	# 1 - 5
	addl	%%edx,    %%eax		# 1 - 2	A = ROTL3(S24 + A + L1);
	roll	$3,      %%eax		# 2
	movl	%%edi,   %%ecx		# 1	eA = ROTL(eA ^ eB, eB) + A
	xorl	%%edi,   %%esi		# 1
	roll	%%cl,    %%esi		# 3
	addl	%%eax,   %%esi		# 1

_modif486_work_C_0_2:
	cmp	$0x12345678, %%esi
	jne	__exit_2_486
					
	movl	%%eax,   %%ecx		# 1
	addl	%%edx,   %%ecx		# 1	L0 = ROTL(L0 + A + L1, A + L1);
	addl	%%ecx,   %%ebx		# 1
	roll	%%cl,    %%ebx		# 3
	addl	%%ebx,   %%eax		# 1	A = ROTL3(S25 + A + L0);
_modif486_r3_S2_25:
	addl	$0x90ABCDEF,%%eax	# 2
	roll	$3,      %%eax		# 2
	movl	%%esi,   %%ecx		# 1	eB = ROTL(eB ^ eA, eA) + A
	xorl	%%esi,   %%edi		# 1
	roll	%%cl,    %%edi		# 3
	addl	%%eax,   %%edi		# 1

_modif486_work_C_1_2:
	cmpl	$0x12345678, %%edi
	jne	__exit_2_486
	movl	$1, "work_add_iter"
	jmp	_full_exit_486

"BALIGN4"
__exit_2_486:

	movl	"work_key_hi", %%edx

"/* Jumps not taken are faster */"
	addl	$0x02000000,%%edx
	jc	_next_inc_486

_next_iter_486:
	movl	%%edx, "work_key_hi"
	leal	 0x01000000(%%edx), %%edi
	decl	"work_iterations"
	jg	_loaded_486
	movl	%1, %%eax				# pointer to rc5unitwork
	movl	"work_key_lo", %%ebx
	movl	%%ebx, "RC5UnitWork_L0lo"(%%eax)	# Update real data
	movl	%%edx, "RC5UnitWork_L0hi"(%%eax)	# (used by caller)
	jmp	_full_exit_486

"BALIGN4"
_next_iter2_486:
	movl	%%ebx, "work_key_lo"
	movl	%%edx, "work_key_hi"
	leal	 0x01000000(%%edx), %%edi
	movl	%%ebx, %%esi
	decl	"work_iterations"
	jg	_bigger_loop_486
	movl	%1, %%eax				# pointer to rc5unitwork
	movl	%%ebx, "RC5UnitWork_L0lo"(%%eax)	# Update real data
	movl	%%edx, "RC5UnitWork_L0hi"(%%eax)	# (used by caller)
	jmp	_full_exit_486

"BALIGN4"
_next_inc_486:
	addl	$0x00010000, %%edx
	testl	$0x00FF0000, %%edx
	jnz	_next_iter_486

	addl	$0xFF000100, %%edx
	testl	$0x0000FF00, %%edx
	jnz	_next_iter_486

	addl	$0xFFFF0001, %%edx
	testl	$0x000000FF, %%edx
	jnz	_next_iter_486


	movl	"work_key_lo", %%ebx

	subl	$0x00000100, %%edx
	addl	$0x01000000, %%ebx
	jnc	_next_iter2_486

	addl	$0x00010000, %%ebx
	testl	$0x00FF0000, %%ebx
	jnz	_next_iter2_486

	addl	$0xFF000100, %%ebx
	testl	$0x0000FF00, %%ebx
	jnz	_next_iter2_486

	addl	$0xFFFF0001, %%ebx
	testl	$0x000000FF, %%ebx
	jnz	_next_iter2_486

	# Moo !
	# We have just finished checking the last key
	# of the rc5-64 keyspace...
	# Not much to do here, since we have finished the block ...


"BALIGN4"
_full_exit_486:
	movl	"work_save_ebp", %%ebp \n"

: "=m"(work),
  "=m"(rc5unitwork)
: "a" (rc5unitwork)
: "%eax","%ebx","%ecx","%edx","%esi","%edi","cc");

    return (timeslice - work.iterations) * 2 + work.add_iter;
}
