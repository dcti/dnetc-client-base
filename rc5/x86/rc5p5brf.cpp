// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: rc5p5brf.cpp,v $
// Revision 1.6  1998/08/20 00:25:25  silby
// Took out PIPELINE_COUNT checks inside .cpp x86 cores - they were causing build problems with new PIPELINE_COUNT architecture on x86.
//
// Revision 1.5  1998/07/08 22:59:53  remi
// Lots of $Id: rc5p5brf.cpp,v 1.6 1998/08/20 00:25:25 silby Exp $ stuff.
//
// Revision 1.4  1998/07/08 18:47:42  remi
// $Id fun ...
//
// Revision 1.3  1998/06/14 08:27:37  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.2  1998/06/14 08:13:54  friedbait
// 'Log' keywords added to maintain automatic change history
//
//


// Pentium optimized version
// Rémi Guyomarch - rguyom@mail.dotcom.fr - 97/07/13
//
// Minor improvements:
// Bruce Ford - b.ford@qut.edu.au - 97/12/21
//
// roll %cl, ... can't pair
// roll $3,  ... can't pair either :-(
// (despite what intel say)
// (their manual is really screwed up :-( )
//
// it seems that only roll $1, ... can pair :-(
//
// read after write, do not pair
// write after write, do not pair
//
// write after read, pair OK
// read after read, pair OK
// read and write after read, pair OK
//
// For a really *good* pentium optimization manual :
//	http://announce.com/agner/assem

#if (!defined(lint) && defined(__showids__))
const char *rc5p5brf_cpp (void) {
return "@(#)$Id: rc5p5brf.cpp,v 1.6 1998/08/20 00:25:25 silby Exp $"; }
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

// The S0 values for key expansion round 1 are constants.

#define P         0xB7E15163
#define Q         0x9E3779B9
#define S_not(N)  _((P+Q*(N)))
//#define S0_ROTL3  _(((P<<3) | (P>>29)))
#define S0_ROTL3 _(0xbf0a8b1d)
//#define S1_S0_ROTL3 _((S_not(1) + S0_ROTL3))
#define S1_S0_ROTL3 _(0x15235639)
//#define FIRST_ROTL _((S0_ROTL3 & 0x1f))
#define FIRST_ROTL _(0x1d)

//  Structure used in rc5_unit_func_*

struct work_struct {
    u32 add_iter;	   // +  0
    u32 key_hi;		// +  4
    u32 key_lo;		// +  8
    u32 L0_ecx;      // + 12   Used to store results from
    u32 L0_ebx;      // + 16   L0 calculations made outside
    u32 L0_esi;      // + 20   the main key loop.  BRF
    u32 P_0;		   // + 24   Order changed to help with cache access.  BRF
    u32 P_1;		   // + 28
    u32 s1[26];		// + 32
    u32 s2[26];		// +136
    u32 C_0;		   // +240
    u32 C_1;		   // +244
                     // key2_ebp removed as it is identical to s2[25]. BRF
    u32 key2_edi;	   // +248
    u32 key2_esi;	   // +252
    u32 iterations;	// +256
    u32 save_ebp;	   // +260
};

//  Offsets to access work_struct fields.

#define	work_add_iter   "0+%0"
#define  work_key_hi     "4+%0"
#define  work_key_hi1    "5+%0"
#define  work_key_hi2    "6+%0"
#define  work_key_hi3    "7+%0"
#define  work_key_lo     "8+%0"
#define  work_key_lo1    "9+%0"
#define  work_key_lo2    "10+%0"
#define  work_key_lo3    "11+%0"
#define	work_L0_ecx     "12+%0"
#define	work_L0_ebx     "16+%0"
#define	work_L0_esi     "20+%0"
#define	work_P_0        "24+%0"
#define	work_P_1        "28+%0"
// s1 and s2 interleaved for p5. BRF
#define	work_s1         "32+%0"
#define	work_s2         "36+%0"
#define	work_C_0        "240+%0"
#define	work_C_1        "244+%0"
#define	work_key2_edi   "248+%0"
#define	work_key2_esi   "252+%0"
#define  work_iterations "256+%0"
#define	work_save_ebp   "260+%0"

//  Macros to access the S arrays.
//  S arrays are now interleaved preventing cache stalls.  BRF

#define S1(N)    _(((N)*8)+32+%0)
#define S2(N)    _(((N)*8)+36+%0)

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
// Merge end of previous iteration with next iteration
// to avoid AGI stall on %edi / %esi
// ROUND_1_LAST will merge with ROUND_2_EVEN

// S1(N) = A1 = ROTL3 (A1 + Lhi1 + S_not(N));
// S2(N) = A2 = ROTL3 (A2 + Lhi2 + S_not(N));
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
#define ROUND_1_EVEN(N)	\
"	roll	%%cl,  %%esi			# 4
	leal	(%%ebx,%%edx), %%ecx	        # 1
	movl	%%ebp, "S2(N)"			#
	addl	%%ecx, %%eax			# 1
 	leal	"S_not(N+1)"(%%ebp,%%esi),%%ebp #
	roll	$3,    %%ebp			# 1
	roll	%%cl,  %%eax			# 4
	leal	(%%ebp,%%esi), %%ecx		# 1
	movl	%%ebx, "S1(N)"			#
	addl	%%ecx, %%edi			# 1
 	leal	"S_not(N+1)"(%%ebx,%%eax),%%ebx #
	roll	$3,    %%ebx			# 1  sum = 14 \n"

// S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
// S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
// Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
#define ROUND_1_ODD(N) \
"	roll	%%cl,  %%edi			# 4
	leal	(%%eax,%%ebx), %%ecx		# 1
	movl	%%ebp, "S2(N)"			#
	addl	%%ecx, %%edx			# 1
 	leal	"S_not(N+1)"(%%ebp,%%edi),%%ebp #
	roll	$3,    %%ebp			# 1
	roll	%%cl,  %%edx			# 4
	leal	(%%ebp,%%edi), %%ecx		# 1
	movl	%%ebx, "S1(N)"			#
	addl	%%ecx, %%esi			# 1
 	leal	"S_not(N+1)"(%%ebx,%%edx),%%ebx #
	roll	$3,    %%ebx			# 1  sum = 14 \n"

// Same as above, but wrap to first part of round 2
#define ROUND_1_LAST(N)	\
"	roll	%%cl,  %%edi			# 4
	leal	(%%eax,%%ebx), %%ecx		# 1
	movl	%%ebp, "S2(25)"		        #
	addl	%%ecx, %%edx			# 1
 	leal	"S0_ROTL3"(%%ebp,%%edi),%%ebp   #
	roll	$3,    %%ebp			# 1
	roll	%%cl,  %%edx			# 4
	leal	(%%ebp,%%edi), %%ecx		# 1
	movl	%%ebx, "S1(25)"		        #
 	leal	"S0_ROTL3"(%%ebx,%%edx),%%ebx   # 1
	addl	%%ecx, %%esi			#
	roll	$3,    %%ebx			# 1
	roll	%%cl,  %%esi			# 4
	movl	%%ebp, "S2(0)"			# 1
	movl	"work_L0_ebx", %%ecx	        #
	addl	%%ecx, %%ebp			# 1
	leal	(%%ebx,%%edx), %%ecx	        #
	addl	%%esi, %%ebp			# 1
	movl	%%ebx, "S1(0)"		        #
	addl	%%ecx, %%eax			# 1   Spare slot
						#   sum = 22 \n"

#define ROUND_1_ODD_AND_EVEN(N1,N2) \
	ROUND_1_ODD (N1) \
	ROUND_1_EVEN(N2)

// ------------------------------------------------------------------
// Merge 'even' with 'odd', it reduce this macros by 1 cycle

// S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
// S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
#define ROUND_2_EVEN(N) \
"	roll	$3,    %%ebp			# 1
	roll	%%cl,  %%edx			# 4
	leal	(%%ebp,%%edi), %%ecx		# 1
	movl	%%ebp, "S2(N)"			#
	addl	%%edx, %%ebx			# 1
	addl	%%ecx, %%esi			#
	addl	"S1(N)", %%ebx			# 2
	addl	"S2(N+1)", %%ebp		#
	roll	$3,    %%ebx			# 1
	roll	%%cl,  %%esi			# 4
	leal	(%%ebx,%%edx), %%ecx	        # 1
	addl	%%esi, %%ebp			#
	movl	%%ebx, "S1(N)"			# 1
	addl	%%ecx, %%eax			#   sum = 16 \n"
	
// S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
// S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
// Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
#define ROUND_2_ODD(N) \
"	roll	$3,    %%ebp			# 1
	roll	%%cl,  %%eax			# 4
	leal	(%%ebp,%%esi), %%ecx		# 1
	movl	%%ebp, "S2(N)"			#
	addl	%%eax, %%ebx			# 1
	addl	%%ecx, %%edi			#
	addl	"S1(N)", %%ebx			# 2
	addl	"S2(N+1)", %%ebp		#
	roll	$3,    %%ebx			# 1
	roll	%%cl,  %%edi			# 4
	leal	(%%ebx,%%eax), %%ecx	        # 1
	addl	%%edi, %%ebp			#
	movl	%%ebx, "S1(N)"			# 1
	addl	%%ecx, %%edx			#   sum = 16 \n"

#define ROUND_2_ODD_AND_EVEN(N1,N2) \
	ROUND_2_ODD (N1) \
	ROUND_2_EVEN(N2)

// ------------------------------------------------------------------
// It's faster to do 1 key at a time with round3 and encryption mixed
// than to do 2 keys at once but round3 and encryption separated
// Too bad x86 hasn't more registers ...
	
// Assume the following code has already been executed :
//	movl	S1(N),  %ebp
// It reduce this macro by 2 cycles.
// note: the last iteration will be test for short exit, the
// last iteration of this macros won't be the last iteration for
// the third round.
// well, if it's not very clear, look at RC5_ROUND_3...

// eA1 = ROTL (eA1 ^ eB1, eB1) + (A1 = ROTL3 (A1 + Lhi1 + S1(N)));
// Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
// eB1 = ROTL (eA1 ^ eB1, eA1) + (A1 = ROTL3 (A1 + Llo1 + S1(N)));
// Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);

// A  = %eax  eA = %esi
// L0 = %ebx  eB = %edi
// L1 = %edx  .. = %ebp
#define ROUND_3_EVEN_AND_ODD(N,Sx) \
"_round3_p5_"_(Sx)"_"_(N)":
	addl	%%edx,    %%ebx		# 1	
	movl	%%edi,    %%ecx		#
	addl	%%ebp,    %%ebx		# 1
	xorl	%%edi,    %%esi		#
	roll	$3,       %%ebx		# 1
	roll	%%cl,     %%esi		# 4
	leal	(%%ebx,   %%edx), %%ecx	# 1
	addl	%%ebx,    %%esi		#
	addl	%%ecx,    %%eax		# 1
	movl	"Sx(N+1)",%%ebp		#
	roll	%%cl,     %%eax		# 4
					
	addl	%%eax,    %%ebx		# 1
	movl	%%esi,    %%ecx		#
	addl	%%ebp,    %%ebx		# 1
	xorl	%%esi,    %%edi		#
	roll	$3,       %%ebx		# 1
	roll	%%cl,     %%edi		# 4
	leal	(%%eax,   %%ebx), %%ecx	# 1
	addl	%%ebx,    %%edi		#
	addl	%%ecx,    %%edx		# 1
	movl	"Sx(N+2)",%%ebp		#
	roll	%%cl,     %%edx		# 4   sum = 26 \n"


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
// I use a structure to make this function multi-thread safe.
// (can't use static variables, and can't use push/pop in this
//  function because &work_struct is relative to %esp)

u32 rc5_unit_func_p5( RC5UnitWork * rc5unitwork, u32 timeslice )
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
	movl	%%ebp, "work_C_1"

_loaded_p5:\n"

    /* ----------------------------------------------------------- */
    /* Pre-calculate first rotate of L0 as it rarely changes.  BRF */
    /* ----------------------------------------------------------- */

"	movl	"work_key_lo", %%esi	# 1
	movl	$"S1_S0_ROTL3", %%ebx	#
	addl	$"S0_ROTL3", %%esi	# 1   Spare slot (not that it matters here)  BRF
	roll	$"FIRST_ROTL",  %%esi	# 1
	addl	%%esi, %%ebx		# 1
	movl	%%esi, %%ecx		#
	roll	$3,    %%ebx		# 1
	addl	%%ebx, %%ecx		# 1
	movl	%%ebx, "work_L0_ebx"	#
	movl	%%esi, "work_L0_esi"	# 1
	movl	%%ecx, "work_L0_ecx"	#  sum = 7 every 2147483648 loops or on subroutine
					#        entry.  The latter happens more often.  BRF

_next_key:\n"

    /* ------------------------------ */
    /* Begin round 1 of key expansion */
    /* ------------------------------ */

"  movl	"work_key_hi", %%edi              # 1
   movl  "work_L0_ebx", %%ebx             #
   addl  %%ecx, %%edi                     #
   movl  %%edi, %%edx                     # 1
   addl  $0x01000000, %%edi               #
   roll  %%cl,  %%edi                     # 4
   roll  %%cl,  %%edx                     # 4
   leal	"S_not(2)"(%%ebx,%%edi),%%ebp     # 1
   movl  %%edi, %%ecx                     #
   roll  $3,    %%ebp                     # 1
   leal	"S_not(2)"(%%ebx,%%edx),%%ebx     # 1
   addl  %%ebp, %%ecx                     #
   roll  $3,    %%ebx                     # 1
   addl  %%ecx, %%esi                     # 1
   movl  "work_L0_esi", %%eax             #  sum = 16 \n"
					
  	ROUND_1_EVEN         (2)
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

	// see end of ROUND_1_LAST

"_end_round1_p5:
	roll	$3,    %%ebp		# 1
	roll	%%cl,  %%eax		# 4
	leal	(%%ebp,%%esi), %%ecx	# 1
	movl	%%ebp, "S2(1)"		#
	addl	%%eax, %%ebx		# 1
	addl	%%ecx, %%edi		#
	addl	"work_L0_ebx", %%ebx	# 2
	addl	"S2(2)", %%ebp 		#
	roll	$3,    %%ebx		# 1
	roll	%%cl,  %%edi		# 4
	leal	(%%ebx,%%eax), %%ecx	# 1
	addl	%%edi, %%ebp		#
	movl	%%ebx, "S1(1)"		# 1
	addl	%%ecx, %%edx		#   sum = 16 \n"

     ROUND_2_EVEN (2)
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

"	roll	$3,    %%ebp		# 1
	roll	%%cl,  %%eax		# 4
	leal	(%%ebp,%%esi), %%ecx	# 1
	movl	%%ebp, "S2(25)"		#
	addl	%%eax, %%ebx		# 1
	movl	"S1(25)", %%ebp   	#
	addl	%%ebp, %%ebx    	# 1
	addl	%%ecx, %%edi		#
	roll	$3,    %%ebx		# 1
	roll	%%cl,  %%edi		# 4
	leal	(%%ebx,%%eax), %%ecx	# 1
	movl	%%edi, "work_key2_edi"  #
	movl	%%ebx, "S1(25)"		# 1
	addl	%%ecx, %%edx		#
	roll	%%cl,  %%edx		# 4
	movl	%%esi, "work_key2_esi"  # 1
	addl	%%edx, %%ebx		#   sum = 20

_end_round2_p5:\n"

    /* Save 2nd key parameters and initialize result variable

       I'm using the stack instead of a memory location, because
       gcc don't allow me to put more than 10 constraints in an
       asm() statement.
    */

    /* ---------------------------------------------------- */
    /* Begin round 3 of key expansion mixed with encryption */
    /* ---------------------------------------------------- */
    /* (first key)					    */

	// A  = %eax  eA = %esi
	// L0 = %ebx  eB = %edi
	// L1 = %edx  .. = %ebp

"	movl	"S1(0)",%%ebp		# 1
	movl	"work_P_0", %%esi	#  	eA = P_0 + A;
	addl	%%ebx,  %%ebp		# 1
	movl	"S1(1)",%%ebx		#
	roll	$3,     %%ebp		# 1
	addl	%%ebp,  %%esi		# 1
	addl	%%ebp,  %%ebx		#
	leal	(%%ebp, %%edx), %%ecx	# 1  	L0 = ROTL(L0 + A + L1, A + L1);
	movl	"work_P_1", %%edi	#  	eB = P_1 + A;
	addl	%%ecx,  %%eax		# 1
                                        #  Spare slot
	roll	%%cl,   %%eax		# 4
					
	addl	%%eax,  %%ebx		# 1	A = ROTL3(S00 + A + L1);
	movl	%%eax,  %%ecx		#  	A = ROTL3(S03 + A + L0);
	roll	$3,     %%ebx		# 1
	addl	%%ebx,  %%edi		# 1
	addl	%%ebx,  %%ecx		#
	addl	%%ecx,  %%edx		# 1
	movl	"S1(2)",%%ebp		#
	roll	%%cl,   %%edx		# 4	sum = 18 \n"

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
"_end_round3_1_p5:
	addl	%%edx, %%ebx		# 1	A = ROTL3(S24 + A + L1);
	movl	%%edi, %%ecx		#	   eA = ROTL(eA ^ eB, eB) + A
	addl	%%ebp, %%ebx		# 1
	xorl	%%edi, %%esi		#
	roll	$3,    %%ebx		# 1
	roll	%%cl,  %%esi		# 4
	addl	%%ebx, %%esi		# 1
	movl	"work_C_0", %%ebp       #     Places je in V pipe for pairing.  BRF
					
	cmpl 	%%ebp, %%esi            # 1
	je 	_testC1_1_p5            #  sum = 9
					
_second_key: \n"

    /* Restore 2nd key parameters */
"	movl	"work_key2_edi",%%edx   # 1
	movl	"S2(25)", %%ebx         #
	movl	"work_key2_esi",%%eax   # 1
	addl	%%edx, %%ebx            #  \n"

    /* ---------------------------------------------------- */
    /* Begin round 3 of key expansion mixed with encryption */
    /* ---------------------------------------------------- */
    /* (second key)					    */

	// A  = %eax  eA = %esi
	// L0 = %ebx  eB = %edi
	// L1 = %edx  .. = %ebp

" 	movl	"S2(0)",%%ebp		# 1
	movl	"work_P_0", %%esi	#  	eA = P_0 + A;
	addl	%%ebx,  %%ebp		# 1
	movl	"S2(1)",%%ebx		#
	roll	$3,     %%ebp		# 1
	addl	%%ebp,  %%esi		# 1
	addl	%%ebp,  %%ebx		#
	leal	(%%ebp, %%edx), %%ecx	# 1  	L0 = ROTL(L0 + A + L1, A + L1);
	movl	"work_P_1", %%edi	#  	eB = P_1 + A;
	addl	%%ecx,  %%eax		# 1
                                        #  Spare slot
	roll	%%cl,   %%eax		# 4

	addl	%%eax,  %%ebx		# 1	A = ROTL3(S00 + A + L1);
	movl	%%eax,  %%ecx		#  	A = ROTL3(S03 + A + L0);
	roll	$3,     %%ebx		# 1
	addl	%%ebx,  %%edi		# 1
	addl	%%ebx,  %%ecx		#
	addl	%%ecx,  %%edx		# 1
	movl	"S2(2)",%%ebp		#
	roll	%%cl,   %%edx		# 4	sum = 20 \n"

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
"_end_round3_2_p5:
	addl	%%edx, %%ebx		# 1	A = ROTL3(S24 + A + L1);
	movl	%%edi, %%ecx		#	eA = ROTL(eA ^ eB, eB) + A
	addl	%%ebp, %%ebx		# 1
	xorl	%%edi, %%esi		#
	roll	$3,    %%ebx		# 1
	roll	%%cl,  %%esi		# 4
	addl	%%ebx, %%esi		# 1
	movl	"work_C_0", %%ebp       #    Places je in V pipe for pairing.  BRF
					
	cmpl 	%%ebp, %%esi            # 1
	je 	_testC1_2_p5            #  sum = 9

_incr_key:
	decl	"work_iterations"       # 3
   jz    _full_exit_p5

   movb  "work_key_hi3", %%dl           # 1  All this is to try and save one clock
					#    at the jnc below
   movl  "work_L0_ecx", %%ecx           #    Costs nothing (in clocks) to try.  BRF
   addb  $2   , %%dl                    # 1
   movl  "work_L0_esi", %%esi           #
   movb  %%dl ,"work_key_hi3"           # 1
   jnc   _next_key                      #

   incb  "work_key_hi2"
   jnz   _next_key
   incb  "work_key_hi1"
   jnz   _next_key
   incb  "work_key_hi"
   jnz   _next_key
   incb  "work_key_lo3"
   jnz   _loaded_p5
   incb  "work_key_lo2"
   jnz   _loaded_p5
   incb  "work_key_lo1"
   jnz   _loaded_p5
   incb  "work_key_lo"
   jmp   _loaded_p5                       # Wrap the keyspace

_testC1_1_p5:
	leal	(%%ebx,%%edx), %%ecx	            # 1	L0 = ROTL(L0 + A + L1, A + L1);
	movl	"S1(25)", %%ebp		            #
	addl	%%ecx, %%eax		               # 1
	xorl	%%esi, %%edi		               #
	roll	%%cl,  %%eax		               # 4
	addl	%%eax, %%ebx		               # 1	A = ROTL3(S25 + A + L0);
	movl	%%esi, %%ecx		               #	   eB = ROTL(eB ^ eA, eA) + A
	addl	%%ebp, %%ebx		               # 1
                                          #  Spare slot (not that it matters)  BRF
	roll	$3,    %%ebx		               # 1
	roll	%%cl,  %%edi		               # 4
	addl	%%ebx, %%edi		               # 1
	movl	"work_C_1", %%ebp                #     Places jne in V pipe for pairing.  BRF

	cmpl	%%ebp, %%edi                     # 1
	jne	_second_key
   jmp   _done

_testC1_2_p5:
	leal	(%%ebx,%%edx), %%ecx	            # 1	L0 = ROTL(L0 + A + L1, A + L1);
	movl	"S2(25)", %%ebp		            #
	addl	%%ecx, %%eax		               # 1
	xorl	%%esi, %%edi		               #
	roll	%%cl,  %%eax		               # 4
	addl	%%eax, %%ebx		               # 1	A = ROTL3(S25 + A + L0);
	movl	%%esi, %%ecx		               #	   eB = ROTL(eB ^ eA, eA) + A
	addl	%%ebp, %%ebx		               # 1
                                          #  Spare slot (not that it matters)  BRF
	roll	$3,    %%ebx		               # 1
	roll	%%cl,  %%edi		               # 4
	addl	%%ebx, %%edi		               # 1
	movl	"work_C_1", %%ebp                #     Places jne in V pipe for pairing.  BRF

	cmpl	%%ebp, %%edi                     # 1
	jne	_incr_key
	movl	$1, "work_add_iter"
   jmp   _done

_full_exit_p5:
   addb  $2   ,"work_key_hi3"
   jnc   _key_updated
   incb  "work_key_hi2"
   jnz   _key_updated
   incb  "work_key_hi1"
   jnz   _key_updated
   incb  "work_key_hi"
   jnz   _key_updated
   incb  "work_key_lo3"
   jnz   _key_updated
   incb  "work_key_lo2"
   jnz   _key_updated
   incb  "work_key_lo1"
   jnz   _key_updated
   incb  "work_key_lo"

_key_updated:
	movl	%1, %%eax				# pointer to rc5unitwork
   movl  "work_key_lo", %%ebx
   movl  "work_key_hi", %%edx
	movl	%%ebx, "RC5UnitWork_L0lo"(%%eax)	# Update real data
	movl	%%edx, "RC5UnitWork_L0hi"(%%eax)	# (used by caller)

_done:
	movl	"work_save_ebp", %%ebp \n"

: "=m"(work),
  "=m"(rc5unitwork)
: "a" (rc5unitwork)
: "%eax","%ebx","%ecx","%edx","%esi","%edi","cc");

    return (timeslice - work.iterations) * 2 + work.add_iter;
}

