/* s_paramaters.h */

/* When compiling DCTI clients, this file is *NOT* used.
 * Defines are set in configure & Makefile.
 */

/* $Log: s_paramaters.h,v $
/* Revision 1.1  1998/06/14 14:23:51  remi
/* Initial revision
/* */


/* toplevel defines that control what executable to build */

/* define the following if using a smart compiler that understands inline */

/* #define INLINE inline		/* if compiler is not brain-dead */
#define INLINE		/* if compiler is slightly smart */


/* These defines control the des_ultra_crunch routine, which implements whack16 */

/* define this to test code.  No define produces whack16 routine for contest */

#define DEBUG_MAIN		/* define to create main() for testing */


/* define only ONE of these next defines at once with DEBUG_MAIN to test contest code */

/* #define TIME_WHACK16		/* define to test speed of contest code */
#define TEST_REMI_KEYS		/* define to test correctness of contest code */
/* #define TEST_TONS_OF_KEYS	/* define to test correctness of contest code */

/* 64 bit results are available if the inner loop is in "c" and uses "long long",
 * or if the inner loop is in assembler and uses full 64-bit registers.
 *
 * 32 bit results must be examined depending on machine type.  Byte order hell.
 *
 * For a Sparc, if the inner loop uses 32-bit instructions while using pointers
 * to 64-bit operands, the HIGH part of each result will be valid.
 *
 * For a Sparc, if the inner loop uses 64-bit instructions WITHOUT checking
 * whether the high half of the register is clobbered, the LOW part of the
 * register will be valid.
 */

/* define only ONE of these next defines at once to examine calculated data */
/* remember, FULL for 64-bit, HIGH for debugging with 32-bit on an Ultra */

/* #define FULL_64_BIT_VALID	/* long long inner loop, or ASM */
#define HIGH_WORD_VALID     /* on SPARC, inner loop uses long, outer long long */
/* #define LOW_WORD_VALID    /* on sparc, inner loop uses long long, no retry */


/* These defines control the inner loop code do_all_fancy, which is either
 * defined in the several do_* routines, or in simple.c and many s*.h files
 */

/* define this to use the whole set of hand-optimized code included in simple.c
 * and the many companion .h files.  Otherwise, use machine-generated functions
 * included in the many do_* routines
 */
/* #define ASM			/* define to use hand-generated code */


/* define these to use hand-generated register allocation.  Otherwise, use gcc */

#define MANUAL_REGISTER_ALLOCATION	/* use hand register allocation */


/* define this to use the floating point registers to speed the inner loop */

#define DO_FLOAT_PIPE		/* use SPARC VIS instructions for extra speed */
#define USE_IDENTICAL_FLOAT_REGISTERS	/* use hand register allocation */


/* define this if the UltraSparc might stomp the upper 32 bits of registers. */

#define USE_64_BIT_SENTINEL
