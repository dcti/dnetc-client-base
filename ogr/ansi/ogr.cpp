/*
 * Copyright distributed.net 1999-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogr.cpp,v 1.2.4.30 2004/07/13 22:22:59 kakace Exp $
 */
#include <stdlib.h> /* malloc (if using non-static choose dat) */
#include <string.h> /* memset */

#define HAVE_STATIC_CHOOSEDAT /* choosedat table is static, pre-generated */
/* #define CRC_CHOOSEDAT_ANYWAY */ /* you'll need to link crc32 if this is defd */

/* #define OGR_DEBUG           */ /* turn on miscellaneous debug information */ 
/* #define OGR_TEST_FIRSTBLANK */ /* test firstblank logic (table or asm) */
/* #define OGR_TEST_BITOFLIST  */ /* test bitoflist table */

#if defined(OGR_DEBUG) || defined(OGR_TEST_FIRSTBLANK) || defined(OGR_TEST_BITOFLIST)
#include <stdio.h>  /* printf for debugging */
#endif

/* --- various optimization option overrides ----------------------------- */

/* baseline/reference == ogr.cpp without optimization == ~old ogr.cpp */
#if defined(NO_OGR_OPTIMIZATION) || defined(GIMME_BASELINE_OGR_CPP)
  #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 0/1 - default is 1 ('yes') */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0/1 - default is hw dependant */
  #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 0-2 - default is 1 */
  #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 0-2 - default is 2 */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - default is 1 ('yes') */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - default is 0 ('no') */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0-2 - default is 0 */
#elif (defined(OVERWRITE_DEFAULT_OPTIMIZATIONS))  
  /* defines reside in an external file */
#elif (defined(ASM_X86) || defined(__386__) || defined(_M_IX86))
  #if defined(OGR_NOFFZ) 
    /* the bsr insn is slooooow on anything less than a PPro */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 0
  #endif
  #define OGROPT_BITOFLIST_DIRECT_BIT 0          /* we want 'no' */
#elif defined(ASM_68K)
  #if defined(mc68040)
    #define OGROPT_BITOFLIST_DIRECT_BIT 0        /* we want 'no' */
  #else // 68000/020/030/060
    #define OGROPT_STRENGTH_REDUCE_CHOOSE 0      /* GCC is better */
    #define OGROPT_COMBINE_COPY_LIST_SET_BIT_COPY_DIST_COMP 1
  #endif
  /* as on NeXT doesn't know .balignw */
  #if !defined(__NeXT__) && \
      (defined(mc68020) || defined(mc68030) || defined(mc68040) || defined(mc68060))
    #define OGROPT_CYCLE_CACHE_ALIGN 1
  #endif
#elif defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__)
  #if defined(HAVE_KOGE_PPC_CORES)
    /* ASM-optimized OGR cores. Only set relevant options for ogr_create() */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_NON_STATIC_FOUND_ONE           1
  #elif defined(__MWERKS__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #if (__MWERKS__ >= 0x2400)
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* MWC is better    */
    #else
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* MWC benefits     */
    #endif
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 2 /* use switch_asm   */
  #elif defined(__MRC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* MrC is better    */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* MrC is better    */
  #elif defined(__APPLE_CC__) && (__APPLE_CC__ <= 1175)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* ACC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* ACC is better    */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* no balignl       */
  #elif defined(__GNUC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* GCC is better    */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 2 /* use switch_asm   */
    #if !defined(_AIX) && !defined(__APPLE_CC__)    /* no balignl       */
      #define OGROPT_CYCLE_CACHE_ALIGN            1
    #endif
    #if (__GNUC__ >= 3)
      #define OGROPT_NO_FUNCTION_INLINE           1
    #endif
  #elif defined(__xlC__)
    #include <builtins.h>                           /* __cntlz4()       */
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* xlC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    /* xlC has __rlwimi but lacks __rlwinm */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* no balignl       */
  #else
    #error play with the defines to find optimal settings for your compiler
  #endif
#elif defined(ASM_POWER)
  #if defined (__GNUC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* GCC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 2 /* use switch_asm   */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* no balignl       */
  #elif defined(__xlC__)
    #include <builtins.h>                           /* __cntlz4()       */
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* xlC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    /* xlC has __rlwimi but lacks __rlwinm */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* no balignl       */
  #else
    #error play with the defines to find optimal settings for your compiler
  #endif
#elif defined(ASM_ARM)
  #define OGROPT_BITOFLIST_DIRECT_BIT           0
  #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 2
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1
  #define OGROPT_ALTERNATE_CYCLE                0
  #define OGROPT_COMBINE_COPY_LIST_SET_BIT_COPY_DIST_COMP 1
  #define OGROPT_NON_STATIC_FOUND_ONE
#elif defined(ASM_SPARC)
  #define OGROPT_BITOFLIST_DIRECT_BIT           1  /* default */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0  /* default */
  #define OGROPT_COPY_LIST_SET_BIT_JUMPS        1  /* default */
  #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 2  /* default */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1  /* default */
  #define OGROPT_ALTERNATE_CYCLE                0  /* default */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0  /* irrelevant */    
  #define OGROPT_COMBINE_COPY_LIST_SET_BIT_COPY_DIST_COMP 1
#endif

/* -- various optimization option defaults ------------------------------- */

/* optimization for machines where mem access is slower than a shift+sub+and.
   Particularly effective with small data cache.
   If not set to 1, BITOFLIST will use a pre-computed memtable lookup,
   otherwise it will compute the value at runtime (0x80000000>>((x-1)&0x1f))
*/
#ifndef OGROPT_BITOFLIST_DIRECT_BIT
#define OGROPT_BITOFLIST_DIRECT_BIT 1 /* the default is "yes" */
#endif

/* optimization for available hardware insn(s) for 'find first zero bit',
   counting from highest bit, ie 0xEFFFFFFF returns 1, and 0xFFFFFFFE => 32
   This is the second (or first) most effective speed optimization.
*/
#if defined(OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM) && \
           (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 0)
  #undef OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
#else
  #if (!defined(OVERWRITE_DEFAULT_OPTIMIZATIONS) &&     \
       defined(OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM)) || \
       !defined(OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM)
    #undef OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
    #if defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__)         \
        || defined(ASM_POWER) || (defined(__WATCOMC__) && defined(__386__))  \
        || (defined(__MWERKS__) && defined(__INTEL__))                       \
        || (defined(__ICC)) /* icc is Intel only */ || defined(ALPHA_CIX)    \
        || (defined(__GNUC__) && (defined(ASM_X86)  || (defined(ASM_68K) &&  \
                                    (defined(mc68020) || defined(mc68030)    \
                                  || defined(mc68040) || defined(mc68060)))))
      #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 1
      /* #define OGR_TEST_FIRSTBLANK */ /* ... to test */
    #endif
  #endif
#endif

/* optimize COPY_LIST_SET_BIT macro for when jumps are expensive.
   0=no reduction (6 'if'), 1=one 'if'+manual copy; 2=one 'if' plus memcpy;
   This is the most (or second most) effective speed optimization.
   If your compiler has an intrinsic memcpy() AND optimizes that for size
   and alignment (ie, doesn't just inline memcpy) AND/OR the target arch
   is register-rich, then 2 is faster than 1.
*/
#ifndef OGROPT_COPY_LIST_SET_BIT_JUMPS
#define OGROPT_COPY_LIST_SET_BIT_JUMPS 1        /* 0 (no opt) or 1 or 2 */
#endif


/* reduction of found_one maps by using single bits intead of whole chars
   0=no reduction (upto 1024 octets); 1=1024 bits in 128 chars; 2=120 chars
   opt 1 or 2 adds two shifts, two bitwise 'and's and one bitwise 'or'.
   NOTE: that found_one() is not a speed critical function, and *should*
   have no effect on -benchmark at all since it is never called for -bench,
   Some compilers/archs show a negative impact on -benchmark because
   they optimize register usage in ogr_cycle while tracking those used in
   found_one and/or are size sensitive, and the increased size of found_one()
   skews -benchmark. [the latter can be compensated for by telling the
   compiler to align functions, the former by making found_one() non static]
*/
#ifndef OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE
#define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 1 /* 0 (no opt) or 1 or 2 */
#endif


/* Note changes:
   top level is now in registers (thus requires a register rich arch)
   bit[] and first[] are not used
   dist is not saved except on exit
   newbit is shifted into list instead of setting the bit for last mark
   cnt1 and lev2 have been eliminated

   To Do:
   ogr_create() should be updated to match
   dist is not needed in lev*, we only need the final value in state

   OGROPT_ALTERNATE_CYCLE == 0 -> default (GARSP) ogr_cycle()
   OGROPT_ALTERNATE_CYCLE == 1 -> tuned (for ppc [only?]) ogr_cycle() by dan and chris
   OGROPT_ALTERNATE_CYCLE == 2 -> vectorized ogr_cycle() contibuted by dan and chris
*/
#ifndef OGROPT_ALTERNATE_CYCLE
#define OGROPT_ALTERNATE_CYCLE 0 /* 0 (GARSP) or 1 or 2 */
#endif


/* If CHOOSE_MARKS == 12 we can strength reduce the multiply -- which is slow
   on MANY processors -- from "12*(x)" to "((x)<<3)+((x)<<2)" in choose(x,y).
   Note that very smart compilers can sometimes do a better job at replacing
   the original statement with intrinsics than we can do by inserting these
   shift operations (e.g.: MrC). Thanks to Chris Cox for this optimization.
   If CHOOSE_MARKS != 12 this setting will have no effect.
*/
#ifndef OGROPT_STRENGTH_REDUCE_CHOOSE
#define OGROPT_STRENGTH_REDUCE_CHOOSE 1 /* the default is "yes" */
#endif


/* These are alternatives for the COMP_LEFT_LIST_RIGHT macro to "shift the
   list to add or extend the first mark".
   Note that this option gets ignored if OGROPT_ALTERNATE_CYCLE != 1.

   OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 0 -> default COMP_LEFT_LIST_RIGHT
   OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1 -> alternate approach by sampo
   OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 2 -> assembly versions
*/
#ifndef OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT
#define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0 (no opt) or 1 or 2 */
#endif


/* OGROPT_IGNORE_TIME_CONSTRAINT_ARG: By default, ogr_cycle() treats the 
   nodes_to_do argument as a hint rather than a precise number. This is a 
   BadThing(TM) in non-preemptive or real-time environments that are running 
   the cruncher in time contraints, so ogr_cycle() also gets passed a 
   'with_time_constraints' argument that tells the it whether such an 
   environment is present or not.
   However, on most platforms this will never be true, and under those 
   circumstances, testing the 'with_time_constraints' flag is an uneccessary
   waste of time. OGROPT_IGNORE_TIME_CONSTRAINT_ARG serves to disable the
   test. 
*/
#define OGROPT_IGNORE_TIME_CONSTRAINT_ARG
#if defined(macintosh) || defined(__riscos) || \
    defined(__NETWARE__) || defined(NETWARE) || \
    defined(__WINDOWS386__) /* 16bit windows */ || \
    (defined(ASM_X86) && defined(GENERATE_ASM))
    /* ASM_X86: the ogr core used by all x86 platforms is a hand optimized */
    /* .S/.asm version of this core - If we're compiling for asm_x86 then */
    /* we're either generating an .S for later optimization, or compiling */
    /* for comparison with an existing .S */
  #undef OGROPT_IGNORE_TIME_CONSTRAINT_ARG
#endif  

/* Some cpus benefit from having the top of the main ogr_cycle() loop being
   aligned on a specific memory boundary, for optimum performance during
   cache line reads.  Your compiler may well already align the code optimumly,
   but notably gcc for 68k, and PPC, does not.  If you turn this option on,
   you'll need to supply suitable code for the OGR_CYCLE_CACHE_ALIGN macro.
*/
#ifndef OGROPT_CYCLE_CACHE_ALIGN
#define OGROPT_CYCLE_CACHE_ALIGN 0 /* the default is "no" */
#endif

/* Some compilers (notably GCC 3.4) may choose to inline functions, which
   may have an adverse effect on performance of the OGR core.  This option
   currently only affects the found_one() function and will declare it so
   that the compiler will never inline it.  If you turn this option on,
   you'll need to supply suitable code for the OGR_NO_FUNCTION_INLINE macro.
*/
#ifndef OGROPT_NO_FUNCTION_INLINE
#define OGROPT_NO_FUNCTION_INLINE 0 /* the default is "no" */
#endif

/* ----------------------------------------------------------------------- */

#if !defined(HAVE_STATIC_CHOOSEDAT) || defined(CRC_CHOOSEDAT_ANYWAY)
#include "crc32.h" /* only need to crc choose_dat if its not static */
#endif

#include "ogr.h"

#ifndef OGROPT_NEW_CHOOSEDAT
  // maximum number of marks supported by ogr_choose_dat
  #define CHOOSE_MARKS       12
  // number of bits from the beginning of dist bitmaps supported by ogr_choose_dat
  #define CHOOSE_DIST_BITS   12
  #define ttmDISTBITS (32-CHOOSE_DIST_BITS)
#else /* OGROPT_NEW_CHOOSEDAT */
  // maximum number of marks supported by ogr_choose_dat
  #define CHOOSE_MAX_MARKS   13
  #define CHOOSE_MARKS       CHOOSE_MAX_MARKS
  // alignment musn't be equal to CHOOSE_MAX_MARKS
  #define CHOOSE_ALIGN_MARKS 16
  // number of bits from the beginning of dist bitmaps supported by ogr_choose_dat
  #define CHOOSE_DIST_BITS   12
  #define ttmDISTBITS (32-CHOOSE_DIST_BITS)
#endif /* OGROPT_NEW_CHOOSEDAT */

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef OGROPT_NEW_CHOOSEDAT
  #ifdef HAVE_STATIC_CHOOSEDAT  /* choosedat table is static, pre-generated */
    extern const unsigned char ogr_choose_dat[]; /* this is in ogr_dat.cpp */
    #if (CHOOSE_MARKS == 12 && OGROPT_STRENGTH_REDUCE_CHOOSE == 1)
      // strength reduce the multiply -- which is slow on MANY processors
      #define choose(x,y) (ogr_choose_dat[((x)<<3)+((x)<<2)+(y+3)]) /*+3 skips header */
    #else
      #define choose(x,y) (ogr_choose_dat[CHOOSE_MARKS*(x)+(y+3)]) /*+3 skips header */
    #endif
  #else
    static const unsigned char *choosedat;/* set in init_load_choose() */
    #if (CHOOSE_MARKS == 12 && OGROPT_STRENGTH_REDUCE_CHOOSE == 1)
      // strength reduce the multiply -- which is slow on MANY processors
      #define choose(x,y) (choosedat[((x)<<3)+((x)<<2)+(y)])
    #else
      #define choose(x,y) (choosedat[CHOOSE_MARKS*(x)+(y)])
    #endif
  #endif
#else /* we have OGROPT_NEW_CHOOSEDAT */
  #ifdef HAVE_STATIC_CHOOSEDAT  /* choosedat table is static, pre-generated */
    //  extern const unsigned char ogr_choose_dat2[]; /* this is in ogr_dat2.cpp */
    #if (CHOOSE_ALIGN_MARKS == 16 && OGROPT_STRENGTH_REDUCE_CHOOSE == 1)
      // strength reduce the multiply -- which is slow on MANY processors
      #define choose(x,y) (ogr_choose_dat2[((x)<<4)+(y)])
    #else
      #define choose(x,y) (ogr_choose_dat2[CHOOSE_ALIGN_MARKS*(x)+(y)])
    #endif
  #else
    #error OGROPT_NEW_CHOOSEDAT and not HAVE_STATIC_CHOOSEDAT ???   
  #endif
#endif /* OGROPT_NEW_CHOOSEDAT */

/* ----------------------------------------------------------------------- */

static const int OGR[] = {
  /*  1 */    0,   1,   3,   6,  11,  17,  25,  34,  44,  55,
  /* 11 */   72,  85, 106, 127, 151, 177, 199, 216, 246, 283,
  /* 21 */  333, 356, 372, 425, 480, 492, 553, 585, 623
};

#if (OGROPT_NO_FUNCTION_INLINE == 1)
   #if defined(__GNUC__) && (__GNUC__ >= 3)
      #define OGR_NO_FUNCTION_INLINE(x) x __attribute__ ((noinline))
   #else
      #error OGROPT_NO_FUNCTION_INLINE is defined, and no code to match
   #endif
#else
   #define OGR_NO_FUNCTION_INLINE(x) x
#endif

#ifndef __MRC__
static int init_load_choose(void);
#if  (OGROPT_ALTERNATE_CYCLE != 2) || !defined(HAVE_KOGE_PPC_CORES)
  // We only need one instance when using both KOGE cores
  #if defined(OGROPT_NON_STATIC_FOUND_ONE)
  OGR_NO_FUNCTION_INLINE(int found_one(const struct State *oState));
  #else
  OGR_NO_FUNCTION_INLINE(static int found_one(const struct State *oState));
  #endif
#endif
static int ogr_init(void);
static int ogr_cycle(void *state, int *pnodes, int with_time_constraints);
static int ogr_getresult(void *state, void *result, int resultlen);
static int ogr_destroy(void *state);
static int ogr_cleanup(void);

#if (defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__)) \
    && defined(HAVE_KOGE_PPC_CORES)
int cycle_ppc_scalar(void *state, int *pnodes, const unsigned char *choose, const int *OGR);
#if defined(__VEC__) || defined(__ALTIVEC__)
int cycle_ppc_hybrid(void *state, int *pnodes, const unsigned char *choose, const int *OGR);
#endif
#endif  // HAVE_KOGE_PPC_CORES

#if defined(HAVE_OGR_CORES)
static int ogr_create(void *input, int inputlen, void *state, int statelen, int minpos);
#endif
#if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
static int ogr_count(void *state);
static int ogr_save(void *state, void *buffer, int buflen);
static int ogr_load(void *buffer, int buflen, void **state);
#endif
#endif  /* __MRC__ */

#ifndef OGR_GET_DISPATCH_TABLE_FXN
  #define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table
#endif

#ifndef OGR_P2_GET_DISPATCH_TABLE_FXN
  #define OGR_P2_GET_DISPATCH_TABLE_FXN ogr_p2_get_dispatch_table
#endif

#if defined(HAVE_OGR_CORES)
extern CoreDispatchTable * OGR_GET_DISPATCH_TABLE_FXN (void);
#endif
extern CoreDispatchTable * OGR_P2_GET_DISPATCH_TABLE_FXN (void);

#if defined(__cplusplus)
}
#endif

/* ------------------------------------------------------------------ */

/***********************************************************************************
 * The following macros define the BITLIST CLASS
 * The variables defined here should only be manipulated within these class macros.
 ***********************************************************************************/

#if (OGROPT_ALTERNATE_CYCLE == 1) /* support macros for the alternate ogr_cycle() routine */

#define SETUP_TOP_STATE(state,lev)                                      \
   U  comp0 = lev->comp[0], comp1 = lev->comp[1], comp2 = lev->comp[2], \
      comp3 = lev->comp[3], comp4 = lev->comp[4];                       \
   U  list0 = lev->list[0], list1 = lev->list[1], list2 = lev->list[2], \
      list3 = lev->list[3], list4 = lev->list[4];                       \
   U  dist0 = state->dist[0], dist1 = state->dist[1], dist2 = state->dist[2], \
      dist3 = state->dist[3], dist4 = state->dist[4];                   \
   int cnt2 = lev->cnt2;                                                \
   int newbit = 1;                                                      \
   int limit;

/* shift the list to add or extend the first mark */
#if (OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 2)

/* gcc lacks builtin functions for rlwinm and rlwimi but has
** inline assembly to call them directly */
#if defined(__GNUC__) && defined(ASM_POWER)
  #define __rlwinm(Rs,SH,MB,ME) ({      \
    int Ra;                             \
    __asm__ ("rlinm %0,%1,%2,%3,%4"     \
      : "=r" (Ra) : "r" (Rs), "n" (SH), "n" (MB), "n" (ME)); \
    Ra;                                 \
  })

  #define __rlwimi(Ra,Rs,SH,MB,ME) ({   \
    __asm__ ("rlimi %0,%2,%3,%4,%5"     \
      : "=r" (Ra) : "0" (Ra), "r" (Rs), "n" (SH), "n" (MB), "n" (ME)); \
    Ra;                                 \
  })

  #define __nop ({ __asm__ volatile ("nop");})
#elif defined(__GNUC__) && (defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__))
  #define __rlwinm(Rs,SH,MB,ME) ({      \
    int Ra;                             \
    __asm__ ("rlwinm %0,%1,%2,%3,%4"    \
      : "=r" (Ra) : "r" (Rs), "n" (SH), "n" (MB), "n" (ME)); \
    Ra;                                 \
  })

  #define __rlwimi(Ra,Rs,SH,MB,ME) ({   \
    __asm__ ("rlwimi %0,%2,%3,%4,%5"    \
      : "=r" (Ra) : "0" (Ra), "r" (Rs), "n" (SH), "n" (MB), "n" (ME)); \
    Ra;                                 \
  })

  #define __nop ({ __asm__ volatile ("nop");})
#endif

#if defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__) || \
    defined(ASM_POWER)
  #define COMP_LEFT_LIST_RIGHT_BASIC(k)       \
    comp0 = __rlwinm(comp0,k,0,31);           \
    comp1 = __rlwinm(comp1,k,0,31);           \
    comp2 = __rlwinm(comp2,k,0,31);           \
    comp3 = __rlwinm(comp3,k,0,31);           \
    list4 = __rlwinm(list4,32-k,0,31);        \
    list3 = __rlwinm(list3,32-k,0,31);        \
    list2 = __rlwinm(list2,32-k,0,31);        \
    list1 = __rlwinm(list1,32-k,0,31);        \
    list0 = __rlwinm(list0,32-k,0,31);        \
    comp0 = __rlwimi(comp0,comp1,0,32-k,31);  \
    comp1 = __rlwimi(comp1,comp2,0,32-k,31);  \
    comp2 = __rlwimi(comp2,comp3,0,32-k,31);  \
    comp3 = __rlwimi(comp3,comp4,k,32-k,31);  \
    comp4 = __rlwinm(comp4,k,0,31-k);         \
    list4 = __rlwimi(list4,list3,0,0,k-1);    \
    list3 = __rlwimi(list3,list2,0,0,k-1);    \
    list2 = __rlwimi(list2,list1,0,0,k-1);    \
    list1 = __rlwimi(list1,list0,0,0,k-1);    \
    list0 = __rlwimi(list0,newbit,32-k,0,k-1);

  #define COMP_LEFT_LIST_RIGHT(lev, s)             \
  {                                                \
    switch (s)                                     \
    {                                              \
      case 0:                                      \
         __nop;                                    \
         break;                                    \
      case 1:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(1)             \
         break;                                    \
      case 2:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(2)             \
         break;                                    \
      case 3:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(3)             \
         break;                                    \
      case 4:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(4)             \
         break;                                    \
      case 5:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(5)             \
         break;                                    \
      case 6:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(6)             \
         break;                                    \
      case 7:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(7)             \
         break;                                    \
      case 8:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(8)             \
         break;                                    \
      case 9:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(9)             \
         break;                                    \
      case 10:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(10)            \
         break;                                    \
      case 11:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(11)            \
         break;                                    \
      case 12:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(12)            \
         break;                                    \
      case 13:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(13)            \
         break;                                    \
      case 14:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(14)            \
         break;                                    \
      case 15:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(15)            \
         break;                                    \
      case 16:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(16)            \
         break;                                    \
      case 17:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(17)            \
         break;                                    \
      case 18:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(18)            \
         break;                                    \
      case 19:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(19)            \
         break;                                    \
      case 20:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(20)            \
         break;                                    \
      case 21:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(21)            \
         break;                                    \
      case 22:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(22)            \
         break;                                    \
      case 23:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(23)            \
         break;                                    \
      case 24:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(24)            \
         break;                                    \
      case 25:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(25)            \
         break;                                    \
      case 26:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(26)            \
         break;                                    \
      case 27:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(27)            \
         break;                                    \
      case 28:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(28)            \
         break;                                    \
      case 29:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(29)            \
         break;                                    \
      case 30:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(30)            \
         break;                                    \
      case 31:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(31)            \
         break;                                    \
      case 32:                                     \
         comp0 = comp1;                            \
         comp1 = comp2;                            \
         comp2 = comp3;                            \
         comp3 = comp4;                            \
         comp4 = 0;                                \
         list4 = list3;                            \
         list3 = list2;                            \
         list2 = list1;                            \
         list1 = list0;                            \
         list0 = newbit;                           \
         break;                                    \
    }                                              \
    newbit = 0;                                    \
  }
#elif defined(__386__) && defined(__WATCOMC__)
   void COMP_LEFT_LIST_RIGHT_xx(U *levcomp, U *levlist, int s);
    #pragma aux COMP_LEFT_LIST_RIGHT_xx =  \
    "mov eax,[edi+4]"                   \
    "mov edx,[esi+12]"                  \
    "shld [edi+0],eax,cl"               \
    "shrd [esi+16],edx,cl"              \
    "mov eax,[edi+8]"                   \
    "mov edx,[esi+8]"                   \
    "shld [edi+4],eax,cl"               \
    "shrd [esi+12],edx,cl"              \
    "mov eax,[edi+12]"                  \
    "mov edx,[esi+4]"                   \
    "shld [edi+8],eax,cl"               \
    "shrd [esi+8],edx,cl"               \
    "mov eax,[edi+16]"                  \
    "mov edx,[esi+0]"                   \
    "shld [edi+12],eax,cl"              \
    "shrd [esi+4],edx,cl"               \
    "shl eax,cl"                        \
    "shr edx,cl"                        \
    "mov [edi+16],eax"                  \
    "mov [esi+0],edx"                   \
    parm [edi] [esi] [ecx] modify exact [edx eax];
  #define COMP_LEFT_LIST_RIGHT(lev,s) \
        COMP_LEFT_LIST_RIGHT_xx(&(lev->comp[0]),&(lev->list[0]),s)
#elif defined(ASM_X86) && defined(__GNUC__)
  #define COMP_LEFT_LIST_RIGHT(lev,s)       \
  {                                         \
    asm(                                    \
      "movl  4(%0),%%eax\n\t"               \
      "movl  12(%1),%%edx\n\t"              \
                                            \
      "shldl %%cl,%%eax,(%0)\n\t"           \
      "movl  8(%0),%%eax\n\t"               \
                                            \
      "shrdl %%cl,%%edx,16(%1)\n\t"         \
      "movl  8(%1),%%edx\n\t"               \
                                            \
      "shldl %%cl,%%eax,4(%0)\n\t"          \
      "movl  12(%0),%%eax\n\t"              \
                                            \
      "shrdl %%cl,%%edx,12(%1)\n\t"         \
      "movl  4(%1),%%edx\n\t"               \
                                            \
      "shldl %%cl,%%eax,8(%0)\n\t"          \
      "movl  16(%0),%%eax\n\t"              \
                                            \
      "shrdl %%cl,%%edx,8(%1)\n\t"          \
      "movl  (%1),%%edx\n\t"                \
                                            \
      "shldl %%cl,%%eax,12(%0)\n\t"         \
      "shrdl %%cl,%%edx,4(%1)\n\t"          \
                                            \
      "shll  %%cl,16(%0)\n\t"               \
      "shrl  %%cl,(%1)\n\t"                 \
                                            \
      : /* no output */                     \
      : "D" (&(lev->comp)), "S" (&(lev->list)), \
        "c" (s) /* get s in ecx*/           \
      : "memory", "cc", "eax", "edx"        \
    );                                      \
  }
#else
  #error you dont have inline assembly for COMP_LEFT_LIST_RIGHT
#endif
#elif (OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1)
  #define COMP_LEFT_LIST_RIGHT_BASIC(k) {          \
    comp0 = (comp0 << (k)) | (comp1 >> (32-(k)));  \
    list4 = (list4 >> (k)) | (list3 << (32-(k));   \
    comp1 = (comp1 << (k)) | (comp2 >> (32-(k));   \
    list3 = (list3 >> (k)) | (list2 << (32-(k));   \
    comp2 = (comp2 << (k)) | (comp3 >> (32-(k));   \
    list2 = (list2 >> (k)) | (list1 << (32-(k));   \
    comp3 = (comp3 << (k)) | (comp4 >> (32-(k));   \
    list1 = (list1 >> (k)) | (list0 << (32-(k));   \
    list0 = (list0 >> (k)) | (newbit << (32-(k));  \
    comp4 = comp4 << (k);
  
  #define COMP_LEFT_LIST_RIGHT(lev, s) {           \
    switch (s)                                     \
      {                                            \
      case 0:                                      \
         break;                                    \
      case 1:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(1)             \
         break;                                    \
      case 2:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(2)             \
         break;                                    \
      case 3:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(3)             \
         break;                                    \
      case 4:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(4)             \
         break;                                    \
      case 5:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(5)             \
         break;                                    \
      case 6:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(6)             \
         break;                                    \
      case 7:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(7)             \
         break;                                    \
      case 8:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(8)             \
         break;                                    \
      case 9:                                      \
         COMP_LEFT_LIST_RIGHT_BASIC(9)             \
         break;                                    \
      case 10:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(10)            \
         break;                                    \
      case 11:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(11)            \
         break;                                    \
      case 12:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(12)            \
         break;                                    \
      case 13:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(13)            \
         break;                                    \
      case 14:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(14)            \
         break;                                    \
      case 15:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(15)            \
         break;                                    \
      case 16:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(16)            \
         break;                                    \
      case 17:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(17)            \
         break;                                    \
      case 18:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(18)            \
         break;                                    \
      case 19:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(19)            \
         break;                                    \
      case 20:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(20)            \
         break;                                    \
      case 21:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(21)            \
         break;                                    \
      case 22:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(22)            \
         break;                                    \
      case 23:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(23)            \
         break;                                    \
      case 24:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(24)            \
         break;                                    \
      case 25:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(25)            \
         break;                                    \
      case 26:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(26)            \
         break;                                    \
      case 27:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(27)            \
         break;                                    \
      case 28:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(28)            \
         break;                                    \
      case 29:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(29)            \
         break;                                    \
      case 30:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(30)            \
         break;                                    \
      case 31:                                     \
         COMP_LEFT_LIST_RIGHT_BASIC(31)            \
         break;                                    \
      case 32:                                     \
         comp0 = comp1;                            \
         comp1 = comp2;                            \
         comp2 = comp3;                            \
         comp3 = comp4;                            \
         comp4 = 0;                                \
         list4 = list3;                            \
         list3 = list2;                            \
         list2 = list1;                            \
         list1 = list0;                            \
         list0 = newbit;                           \
         break;                                    \
      }                                            \
   newbit = 0;                                     \
}
#else // OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 0
  #define COMP_LEFT_LIST_RIGHT(lev, s) {           \
    int ss = 32 - s;                               \
    comp0 = (comp0 << s) | (comp1 >> ss);          \
    comp1 = (comp1 << s) | (comp2 >> ss);          \
    comp2 = (comp2 << s) | (comp3 >> ss);          \
    comp3 = (comp3 << s) | (comp4 >> ss);          \
    comp4 = comp4 << s;                            \
    list4 = (list4 >> s) | (list3 << ss);          \
    list3 = (list3 >> s) | (list2 << ss);          \
    list2 = (list2 >> s) | (list1 << ss);          \
    list1 = (list1 >> s) | (list0 << ss);          \
    list0 = (list0 >> s) | (newbit << ss);         \
    newbit = 0;                                    \
  }
#endif

/* shift by word size */
#define COMP_LEFT_LIST_RIGHT_32(lev) { \
  comp0 = comp1; comp1 = comp2; comp2 = comp3; comp3 = comp4; comp4 = 0;  \
  list4 = list3; list3 = list2; list2 = list1; list1 = list0; list0 = newbit; \
  newbit = 0; \
}

/* set the current mark and push a level to start a new mark */
#define PUSH_LEVEL_UPDATE_STATE(lev) {  \
  lev->list[0] = list0; dist0 |= list0; \
  lev->list[1] = list1; dist1 |= list1; \
  lev->list[2] = list2; dist2 |= list2; \
  lev->list[3] = list3; dist3 |= list3; \
  lev->list[4] = list4; dist4 |= list4; \
  lev->comp[0] = comp0; comp0 |= dist0; \
  lev->comp[1] = comp1; comp1 |= dist1; \
  lev->comp[2] = comp2; comp2 |= dist2; \
  lev->comp[3] = comp3; comp3 |= dist3; \
  lev->comp[4] = comp4; comp4 |= dist4; \
  newbit = 1; \
  lev->cnt2 = cnt2; \
  lev->limit = limit; \
}

/* pop a level to continue work on previous mark */
#define POP_LEVEL(lev) {                                                  \
  limit = lev->limit;                                                     \
  list0 = lev->list[0], list1 = lev->list[1], list2 = lev->list[2],       \
  list3 = lev->list[3], list4 = lev->list[4];                             \
  dist0 = dist0 & ~list0; dist1 = dist1 & ~list1; dist2 = dist2 & ~list2; \
  dist3 = dist3 & ~list3; dist4 = dist4 & ~list4;                         \
  comp0 = lev->comp[0], comp1 = lev->comp[1], comp2 = lev->comp[2],       \
  comp3 = lev->comp[3], comp4 = lev->comp[4];                             \
  newbit = 0;                                                             \
  cnt2 = lev->cnt2;                                                       \
}

/* save the local state variables */
#define SAVE_FINAL_STATE(state,lev) {                                       \
   lev->list[0] = list0; lev->list[1] = list1; lev->list[2] = list2;        \
   lev->list[3] = list3; lev->list[4] = list4;                              \
   state->dist[0] = dist0; state->dist[1] = dist1; state->dist[2] = dist2;  \
   state->dist[3] = dist3; state->dist[4] = dist4;                          \
   lev->comp[0] = comp0; lev->comp[1] = comp1; lev->comp[2] = comp2;        \
   lev->comp[3] = comp3; lev->comp[4] = comp4;                              \
   lev->cnt2 = cnt2;                                                        \
}

#elif (OGROPT_ALTERNATE_CYCLE == 0)  /* support macros for the ogriginal ogr_cycle() routine */

#define COMP_LEFT_LIST_RIGHT(lev,s)                             \
  {                                                             \
    register int ss = 32 - s;                                   \
    lev->comp[0] = (lev->comp[0] << s) | (lev->comp[1] >> ss);  \
    lev->comp[1] = (lev->comp[1] << s) | (lev->comp[2] >> ss);  \
    lev->comp[2] = (lev->comp[2] << s) | (lev->comp[3] >> ss);  \
    lev->comp[3] = (lev->comp[3] << s) | (lev->comp[4] >> ss);  \
    lev->comp[4] <<= s;                                         \
    lev->list[4] = (lev->list[4] >> s) | (lev->list[3] << ss);  \
    lev->list[3] = (lev->list[3] >> s) | (lev->list[2] << ss);  \
    lev->list[2] = (lev->list[2] >> s) | (lev->list[1] << ss);  \
    lev->list[1] = (lev->list[1] >> s) | (lev->list[0] << ss);  \
    lev->list[0] >>= s;                                         \
  }

#if defined(ASM_ARM) && defined(__GNUC__)
  #define COMP_LEFT_LIST_RIGHT_32(lev) \
  { \
    int a1, a2; \
    \
    asm ("ldr %0,[%2,#44]\n \
          ldr %1,[%2,#48]\n \
          str %0,[%2,#40]\n \
          ldr %0,[%2,#52]\n \
          str %1,[%2,#44]\n \
          ldr %1,[%2,#56]\n \
          str %0,[%2,#48]\n \
          ldr %0,[%2,#12]\n \
          str %1,[%2,#52]\n \
          ldr %1,[%2,#8]\n \
          str %0,[%2,#16]\n \
          ldr %0,[%2,#4]\n \
          str %1,[%2,#12]\n \
          ldr %1,[%2,#0]\n \
          str %0,[%2,#8]\n \
          mov %0,#0\n \
          str %1,[%2,#4]\n \
          str %0,[%2,#56]\n \
          str %0,[%2,#0]" : \
         "=r" (a1), "=r" (a2),\
         "=r" (lev) : "2" (lev)); \
  }
#else
  #define COMP_LEFT_LIST_RIGHT_32(lev)              \
    lev->comp[0] = lev->comp[1];                    \
    lev->comp[1] = lev->comp[2];                    \
    lev->comp[2] = lev->comp[3];                    \
    lev->comp[3] = lev->comp[4];                    \
    lev->comp[4] = 0;                               \
    lev->list[4] = lev->list[3];                    \
    lev->list[3] = lev->list[2];                    \
    lev->list[2] = lev->list[1];                    \
    lev->list[1] = lev->list[0];                    \
    lev->list[0] = 0;
#endif

#if (OGROPT_BITOFLIST_DIRECT_BIT == 0) && (OGROPT_ALTERNATE_CYCLE == 0)
  #define BITOFLIST(x) ogr_bit_of_LIST[x] /* which bit of LIST to update */
  /* ogr_bit_of_LIST[n] = 0x80000000 >> ((n-1) % 32); */
  #define BoL(__n) (0x80000000 >> ((__n - 1) % 32)) //(0x80000000>>((__n - 1)&0x1f))
  static const U ogr_bit_of_LIST[200] = {
        0 , BoL(  1), BoL(  2), BoL(  3), BoL(  4), BoL(  5), BoL(  6), BoL(  7),
  BoL(  8), BoL(  9), BoL( 10), BoL( 11), BoL( 12), BoL( 13), BoL( 14), BoL( 15),
  BoL( 16), BoL( 17), BoL( 18), BoL( 19), BoL( 20), BoL( 21), BoL( 22), BoL( 23),
  BoL( 24), BoL( 25), BoL( 26), BoL( 27), BoL( 28), BoL( 29), BoL( 30), BoL( 31),
  BoL( 32), BoL( 33), BoL( 34), BoL( 35), BoL( 36), BoL( 37), BoL( 38), BoL( 39),
  BoL( 40), BoL( 41), BoL( 42), BoL( 43), BoL( 44), BoL( 45), BoL( 46), BoL( 47),
  BoL( 48), BoL( 49), BoL( 50), BoL( 51), BoL( 52), BoL( 53), BoL( 54), BoL( 55),
  BoL( 56), BoL( 57), BoL( 58), BoL( 59), BoL( 60), BoL( 61), BoL( 62), BoL( 63),
  BoL( 64), BoL( 65), BoL( 66), BoL( 67), BoL( 68), BoL( 69), BoL( 70), BoL( 71),
  BoL( 72), BoL( 73), BoL( 74), BoL( 75), BoL( 76), BoL( 77), BoL( 78), BoL( 79),
  BoL( 80), BoL( 81), BoL( 82), BoL( 83), BoL( 84), BoL( 85), BoL( 86), BoL( 87),
  BoL( 88), BoL( 89), BoL( 90), BoL( 91), BoL( 92), BoL( 93), BoL( 94), BoL( 95),
  BoL( 96), BoL( 97), BoL( 98), BoL( 99), BoL(100), BoL(101), BoL(102), BoL(103),
  BoL(104), BoL(105), BoL(106), BoL(107), BoL(108), BoL(109), BoL(110), BoL(111),
  BoL(112), BoL(113), BoL(114), BoL(115), BoL(116), BoL(117), BoL(118), BoL(119),
  BoL(120), BoL(121), BoL(122), BoL(123), BoL(124), BoL(125), BoL(126), BoL(127),
  BoL(128), BoL(129), BoL(130), BoL(131), BoL(132), BoL(133), BoL(134), BoL(135),
  BoL(136), BoL(137), BoL(138), BoL(139), BoL(140), BoL(141), BoL(142), BoL(143),
  BoL(144), BoL(145), BoL(146), BoL(147), BoL(148), BoL(149), BoL(150), BoL(151),
  BoL(152), BoL(153), BoL(154), BoL(155), BoL(156), BoL(157), BoL(158), BoL(159),
  BoL(160), BoL(161), BoL(162), BoL(163), BoL(164), BoL(165), BoL(166), BoL(167),
  BoL(168), BoL(169), BoL(170), BoL(171), BoL(172), BoL(173), BoL(174), BoL(175),
  BoL(176), BoL(177), BoL(178), BoL(179), BoL(180), BoL(181), BoL(182), BoL(183),
  BoL(184), BoL(185), BoL(186), BoL(187), BoL(188), BoL(189), BoL(190), BoL(191),
  BoL(192), BoL(193), BoL(194), BoL(195), BoL(196), BoL(197), BoL(198), BoL(199)
  #undef BoL
};
#else
  #define BITOFLIST(x) (0x80000000>>((x-1)&0x1f)) /*0x80000000 >> ((x-1) % 32)*/
#endif


#if (OGROPT_COPY_LIST_SET_BIT_JUMPS == 1)
#define COPY_LIST_SET_BIT(lev2,lev,bitindex)      \
  {                                               \
    register unsigned int d = bitindex;           \
    lev2->list[0] = lev->list[0];                 \
    lev2->list[1] = lev->list[1];                 \
    lev2->list[2] = lev->list[2];                 \
    lev2->list[3] = lev->list[3];                 \
    lev2->list[4] = lev->list[4];                 \
    if (d <= (32*5))                              \
      lev2->list[(d-1)>>5] |= BITOFLIST( d );     \
  }
#elif (OGROPT_COPY_LIST_SET_BIT_JUMPS == 2)
#define COPY_LIST_SET_BIT(lev2,lev,bitindex)      \
  {                                               \
    register unsigned int d = bitindex;           \
    memcpy( &(lev2->list[0]), &(lev->list[0]), sizeof(lev2->list[0])*5 ); \
    if (d <= (32*5))                              \
      lev2->list[(d-1)>>5] |= BITOFLIST( d );     \
  }
#else
#define COPY_LIST_SET_BIT(lev2,lev,bitindex)      \
  {                                               \
    register unsigned int d = bitindex;           \
    register int bit = BITOFLIST( d );            \
    if (d <= 32) {                                \
       lev2->list[0] = lev->list[0] | bit;        \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 64) {                         \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1] | bit;        \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 96) {                         \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2] | bit;        \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 128) {                        \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3] | bit;        \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 160) {                        \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4] | bit;        \
    } else {                                      \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    }                                             \
  }
#endif

#define COPY_DIST_COMP(lev2,lev)                  \
  lev2->dist[0] = lev->dist[0] | lev2->list[0];   \
  lev2->dist[1] = lev->dist[1] | lev2->list[1];   \
  lev2->dist[2] = lev->dist[2] | lev2->list[2];   \
  lev2->dist[3] = lev->dist[3] | lev2->list[3];   \
  lev2->dist[4] = lev->dist[4] | lev2->list[4];   \
  lev2->comp[0] = lev->comp[0] | lev2->dist[0];   \
  lev2->comp[1] = lev->comp[1] | lev2->dist[1];   \
  lev2->comp[2] = lev->comp[2] | lev2->dist[2];   \
  lev2->comp[3] = lev->comp[3] | lev2->dist[3];   \
  lev2->comp[4] = lev->comp[4] | lev2->dist[4];

#define COPY_LIST_SET_BIT_COPY_DIST_COMP(lev2,lev,bitindex) \
  {                                   \
    int b, d;                         \
    int a0, a1, a2, a3, a4;           \
                                      \
    b = BITOFLIST(bitindex);          \
    d = bitindex;                     \
    if(d<=32)                         \
    {                                 \
      a0 = lev->list[0] | b;          \
      a1 = lev->list[1];              \
      a2 = lev->list[2];              \
      a3 = lev->list[3];              \
      a4 = lev->list[4];              \
    }                                 \
    else if(d<=64)                    \
    {                                 \
      a0 = lev->list[0];              \
      a1 = lev->list[1] | b;          \
      a2 = lev->list[2];              \
      a3 = lev->list[3];              \
      a4 = lev->list[4];              \
    }                                 \
    else if(d<=96)                    \
    {                                 \
      a0 = lev->list[0];              \
      a1 = lev->list[1];              \
      a2 = lev->list[2] | b;          \
      a3 = lev->list[3];              \
      a4 = lev->list[4];              \
    }                                 \
    else if(d<=128)                   \
    {                                 \
      a0 = lev->list[0];              \
      a1 = lev->list[1];              \
      a2 = lev->list[2];              \
      a3 = lev->list[3] | b;          \
      a4 = lev->list[4];              \
    }                                 \
    else if(d<=160)                   \
    {                                 \
      a0 = lev->list[0];              \
      a1 = lev->list[1];              \
      a2 = lev->list[2];              \
      a3 = lev->list[3];              \
      a4 = lev->list[4] | b;          \
    }                                 \
    else                              \
    {                                 \
      a0 = lev->list[0];              \
      a1 = lev->list[1];              \
      a2 = lev->list[2];              \
      a3 = lev->list[3];              \
      a4 = lev->list[4];              \
    }                                 \
    lev2->list[0] = a0;               \
    lev2->list[1] = a1;               \
    lev2->list[2] = a2;               \
    lev2->list[3] = a3;               \
    b = lev->dist[0];                 \
    lev2->list[4] = a4;               \
    a0 = b | a0;                      \
    b = lev->dist[1];                 \
    lev2->dist[0] = a0;               \
    a1 = b | a1;                      \
    b = lev->dist[2];                 \
    lev2->dist[1] = a1;               \
    a2 = b | a2;                      \
    b = lev->dist[3];                 \
    lev2->dist[2] = a2;               \
    a3 = b | a3;                      \
    b = lev->dist[4];                 \
    lev2->dist[3] = a3;               \
    a4 = b | a4;                      \
    b = lev->comp[0];                 \
    lev2->dist[4] = a4;               \
    a0 = b | a0;                      \
    b = lev->comp[1];                 \
    lev2->comp[0] = a0;               \
    a1 = b | a1;                      \
    b = lev->comp[2];                 \
    lev2->comp[1] = a1;               \
    a2 = b | a2;                      \
    b = lev->comp[3];                 \
    lev2->comp[2] = a2;               \
    a3 = b | a3;                      \
    b = lev->comp[4];                 \
    lev2->comp[3] = a3;               \
    a4 = b | a4;                      \
    lev2->comp[4] = a4;               \
  }

#endif

/* ------------------------------------------------------------------ */

#ifndef OGROPT_NEW_CHOOSEDAT
static int init_load_choose(void)
{
#ifndef HAVE_STATIC_CHOOSEDAT
  #error choose_dat needs to be created/loaded here
#endif
  if (CHOOSE_DIST_BITS != ogr_choose_dat[2]) {
    return CORE_E_FORMAT;
  }
#ifndef HAVE_STATIC_CHOOSEDAT
  /* skip over the choose.dat header */
  choosedat = &ogr_choose_dat[3];
#endif

#if !defined(HAVE_STATIC_CHOOSEDAT) || defined(CRC_CHOOSEDAT_ANYWAY)
  /* CRC32 check */
  {
    static const unsigned chooseCRC32[24] = {
    0x00000000,   /* 0 */
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,   /* 5 */
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,   /* 10 */
    0x00000000,
    0x01138a7d,
    0x00000000,
    0x00000000,
    0x00000000,   /* 15 */
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,   /* 20 */
    0x00000000,
    0x00000000,
    0x00000000
    };
    int i, j;
    unsigned crc32 = 0xffffffff;
    crc32 = CRC32(crc32, ogr_choose_dat[0]);
    crc32 = CRC32(crc32, ogr_choose_dat[1]);
    crc32 = CRC32(crc32, ogr_choose_dat[2]); /* This varies a lot */
    for (j = 0; j < (1 << CHOOSE_DIST_BITS); j++) {
      for (i = 0; i < CHOOSE_MARKS; ++i) crc32 = CRC32(crc32, choose(j, i));
    }
    crc32 = ~crc32;
    if (chooseCRC32[CHOOSE_DIST_BITS] != crc32) {
      /* printf("Your choose.dat (CRC=%08x) is corrupted! Oh well, continuing anyway.\n", crc32); */
      return CORE_E_FORMAT;
    }
  }

#endif
  return CORE_S_OK;
}
#else /* OGROPT_NEW_CHOOSEDAT */
static int init_load_choose(void)
{
  #ifndef HAVE_STATIC_CHOOSEDAT
    #error non static choosedat not supported with OGROPT_NEW_CHOOSEDAT
  #endif
  if ( (CHOOSE_ALIGN_MARKS != choose_align_marks) ||
       (CHOOSE_DIST_BITS   != choose_distbits)    ||
       (CHOOSE_MAX_MARKS   >  choose_max_marks) )
  {
    return CORE_E_FORMAT;
  }

  return CORE_S_OK;
}
#endif /* OGROPT_NEW_CHOOSEDAT */


/*-----------------------------------------*/
/*  found_one() - print out golomb rulers  */
/*-----------------------------------------*/
#if (OGROPT_ALTERNATE_CYCLE == 0)
#if defined(OGROPT_NON_STATIC_FOUND_ONE)
int found_one(const struct State *oState)
#else
static int found_one(const struct State *oState)
#endif
{
  /* confirm ruler is golomb */
  {
    register int i, j;
    #if (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 2)
    char diffs[((1024-64)+7)/8];
    #elif (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 1)
    char diffs[((1024)+7)/8];
    #else
    char diffs[1024];
    #endif
    register int max = oState->max;
    register int maxdepth = oState->maxdepth;
    #if 1 /* (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 1) || \
             (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 2) */
    memset( diffs, 0, sizeof(diffs) );
    #else
    for (i = max>>1; i>=1; i--) diffs[i] = 0;
    #endif
    for (i = 1; i < maxdepth; i++) {
      register int marks_i = oState->marks[i];
      for (j = 0; j < i; j++) {
        register int diff = marks_i - oState->marks[j];
        if (diff+diff <= max) {        /* Principle 1 */
          if (diff <= 64) break;      /* 2 bitmaps always tracked */
          #if (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 2) || \
              (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 1)
          {
            register int mask;
            #if (OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE == 2)
            diff -= 64;
            #endif
            mask = 1<<(diff&7);
            diff >>= 3;
            if ((diffs[diff] & mask)!=0) return 0;
            diffs[diff] |= (char)mask;
          }
          #else
          if (diffs[diff]) return 0;
          diffs[diff] = 1;
          #endif
        }
      }
    }
  }
  return 1;
}
#elif (OGROPT_ALTERNATE_CYCLE == 1) || !defined(HAVE_KOGE_PPC_CORES)
#if defined(OGROPT_NON_STATIC_FOUND_ONE)
int found_one(const struct State *oState)
#else
static int found_one(const struct State *oState)
#endif
{
   /* confirm ruler is golomb */
   int i, j;
   const int maximum = oState->max;
   const int maximum2 = maximum >> 1;     // shouldn't this be rounded up?
   const int maxdepth = oState->maxdepth;
   const struct Level *levels = &oState->Levels[0];
   char diffs[1024]; // first BITMAPS*32 entries will never be used!

   // always check for buffer overruns!
   if (maximum2 >= 1024)
      return CORE_E_MEMORY;

   memset( diffs, 0, maximum2 + 1 );

   for (i = 1; i < maxdepth; i++) {
      int levelICount = levels[i].cnt2;

      for (j = 0; j < i; j++) {
           int diff = levelICount - levels[j].cnt2;

         if (2*diff <= maximum) {      /* Principle 1 */
            if (diff <= (BITMAPS * 32))
               break;     /* 'BITMAPS' bitmaps always tracked */

            if (diffs[diff] != 0)
               return CORE_S_CONTINUE;

            diffs[diff] = 1;
         }
      }   /* for (j = 0; j < i; j++) */
   }  /* for (i = 1; i < maxdepth; i++) */

  return CORE_S_SUCCESS;
}
#endif

#if !defined(OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM) /* 0 <= x < 0xfffffffe */
  static const char ogr_first_blank_8bit[256] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9
  };
  #if defined(ASM_ARM) && defined(__GNUC__)
    static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int input)
    {
      register int temp, result;
      __asm__ ("mov     %0,#0\n\t"             \
               "cmp     %1,#0xffff0000\n\t"    \
               "movcs   %1,%1,lsl#16\n\t"      \
               "addcs   %0,%0,#16\n\t"         \
               "cmp     %1,#0xff000000\n\t"    \
               "movcs   %1,%1,lsl#8\n\t"       \
               "ldrb    %1,[%3,%1,lsr#24]\n\t" \
               "addcs   %0,%0,#8\n\t"          \
               "add     %0,%0,%1"              \
               :"=r" (result), "=r" (temp)
               : "1" (input), "r" ((unsigned int)ogr_first_blank_8bit));
      return result;
    }
  #elif defined(ASM_68K) && defined(__GNUC__) && (__NeXT__)
    static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int input)
    {
      register int result;

      /* gcc-2.5.8 on NeXT needs (&ogr_first_blank_8bit[0]) for
       * address register parameters. Otherweise it will give:
       * ogr/ansi/ogr.cpp:2172: inconsistent operand constraints in an `asm'
       */
      __asm__ ("   cmpl    #0xffff0000,%1\n"
               "   bcs     0f\n"
               "   moveq   #16,%0\n"
               "   bra     1f\n"
               "0: swap    %1\n"
               "   moveq   #0,%0\n"
               "1: cmpw    #0xff00,%1\n"
               "   bcs     2f\n"
               "   lslw    #8,%1\n"
               "   addql   #8,%0\n"
               "2: lsrw    #8,%1\n"
               "   addb    %3@(0,%1:w),%0"
               :"=d" (result), "=d" (input)
               : "1" (input), "a" (&ogr_first_blank_8bit[0]));
      return result;
    }
  #elif defined(ASM_68K) && defined(__GNUC__)
    static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int input)
    {
      register int result;
      __asm__ ("   cmp.l   #0xffff0000,%1\n"
               "   bcs.b   0f\n"
               "   moveq   #16,%0\n"
               "   bra.b   1f\n"
               "0: swap    %1\n"
               "   moveq   #0,%0\n"
               "1: cmp.w   #0xff00,%1\n"
               "   bcs.b   2f\n"
               "   lsl.w   #8,%1\n"
               "   addq    #8,%0\n"
               "2: lsr.w   #8,%1\n"
               "   add.b   0(%3,%1.w),%0"
               :"=d" (result), "=d" (input)
               : "1" (input), "a" (ogr_first_blank_8bit));
      return result;
    }
  #elif defined(FP_CLZ_LITTLEEND) /* using the exponent in floating point double format */
  static inline int LOOKUP_FIRSTBLANK(register unsigned int input)
  {
    unsigned int i;
    union {
      double d;
      int i[2];
    } u;
    
    i=~input;
    u.d=i;
    
    return i == 0 ? 33 : 1055 - (u.i[1] >> 20);
  }
  #else /* C code, no asm */
  static inline int LOOKUP_FIRSTBLANK(register unsigned int input)
  {
    register int result = 0;
    if (input >= 0xffff0000) {
      input <<= 16;
      result += 16;
    }
    if (input >= 0xff000000) {
      input <<= 8;
      result += 8;
    }
    result += ogr_first_blank_8bit[input>>24];
    return result;
  }
  #endif
#elif defined(ASM_PPC) || defined(__PPC__) || defined (__POWERPC__) /* CouNT Leading Zeros Word */
  #if defined(__GNUC__)
    static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i)
    { i = ~i; __asm__ ("cntlzw %0,%0" : "=r" (i) : "0" (i)); return ++i; }
  #elif defined(__MWERKS__) || defined(__MRC__)
    #define LOOKUP_FIRSTBLANK(x)  (__cntlzw(~((unsigned int)(x)))+1)
  #elif defined(__xlC__)
    #define LOOKUP_FIRSTBLANK(x)  (__cntlz4(~((unsigned int)(x)))+1)
  #else
    #error "Please check this (define OGR_TEST_FIRSTBLANK to test)"
  #endif
#elif defined(ASM_POWER) /* CouNT Leading Zeros */
  #if defined(__GNUC__)
    static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i)
    { i = ~i; __asm__ ("cntlz %0,%0" : "=r" (i) : "0" (i)); return ++i; }
  #elif defined(__xlC__)
    #define LOOKUP_FIRSTBLANK(x)  (__cntlz4(~((unsigned int)(x)))+1)
  #else
    #error "Please check this (define OGR_TEST_FIRSTBLANK to test)"
  #endif
#elif defined(ALPHA_CIX)
  #if defined(__GNUC__)
    static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i)
    { 
      register unsigned long j = ~((unsigned long)i << 32);
      __asm__ ("ctlz %0,%0" : "=r"(j) : "0" (j));
      return (int)(j)+1;
    }
  #else
    static inline int LOOKUP_FIRSTBLANK(register unsigned int i)
    {
      __int64 r = asm("ctlz %a0, %v0;", ~((unsigned long)i << 32));
      return (int)(r)+1;
    } 
  #endif
#elif defined(ASM_X86) && defined(__GNUC__) || \
      defined(__386__) && defined(__WATCOMC__) || \
      defined(__INTEL__) && defined(__MWERKS__) || \
      defined(__ICC)
  /* If we were to cover the whole range of 0x00000000 ... 0xffffffff
     we would need ...
     static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int input)
     {
        register unsigned int result;
        __asm__("notl %1\n\t"     \
                "movl $33,%0\n\t" \
                "bsrl %1,%1\n\t"  \
                "jz   0f\n\t"     \
                "subl %1,%0\n\t"  \
                "decl %0\n\t"     \
                "0:"              \
                :"=r"(result), "=r"(input) : "1"(input) : "cc" );
        return result;
     }
     but since the function is only executed for (comp0 < 0xfffffffe),
     we can optimize it to...
  */
  #if defined(__GNUC__)
    static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int input)
    {
       register unsigned int result;
       __asm__("notl %1\n\t"     \
               "movl $32,%0\n\t" \
               "bsrl %1,%1\n\t"  \
               "subl %1,%0\n\t"  \
               :"=r"(result), "=r"(input) : "1"(input) : "cc" );
       return result;
    }
  #elif defined(__WATCOMC__)
    int LOOKUP_FIRSTBLANK(unsigned int);
    #pragma aux LOOKUP_FIRSTBLANK =  \
                      "not  eax"     \
                      "mov  edx,20h" \
                      "bsr  eax,eax" \
                      "sub  edx,eax" \
            value [edx] parm [eax] modify exact [eax edx] nomemory;
  #else /* if defined(__ICC) */
    static inline int LOOKUP_FIRSTBLANK(register unsigned int i)
    {
      _asm mov eax,i
      _asm not eax
      _asm mov edx,20h
      _asm bsr eax,eax
      _asm sub edx,eax
      _asm mov i,edx
      return i;
    }
  #endif
#elif defined(ASM_68K) && defined(__GNUC__) /* Bit field find first one set (020+) */
  static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i)
#if defined(__NeXT__)
  { i = ~i; __asm__ ("bfffo %0{#0:#0},%0" : "=d" (i) : "0" (i)); return ++i; }
#else
  { i = ~i; __asm__ ("bfffo %0,0,0,%0" : "=d" (i) : "0" (i)); return ++i; }
#endif
#else
  #error OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM is defined, and no code to match
#endif

#if (OGROPT_CYCLE_CACHE_ALIGN == 1)
  #if defined(ASM_68K) && defined(__GNUC__)
    #if defined(mc68040)
      // align to 8-byte boundary - pad with nops
      #define OGR_CYCLE_CACHE_ALIGN __asm__ __volatile__ (".balignw 8,0x4e71; nop" : : )
    #elif defined(mc68060) || defined(mc68030) || defined(mc68020)
      // align to 4-byte boundary - pad with nops
      #define OGR_CYCLE_CACHE_ALIGN __asm__ __volatile__ (".balignw 4,0x4e71; nop" : : )
    #endif
  #elif defined(ASM_PPC) && defined(__GNUC__)
    // align to 32-byte boundary - pad with nops
    #define OGR_CYCLE_CACHE_ALIGN __asm__ __volatile__ (".balignl 32,0x60000000; nop; nop" : : )
  #else
    #error OGROPT_CYCLE_CACHE_ALIGN is defined, and no code to match
  #endif
#else
  #define OGR_CYCLE_CACHE_ALIGN { }
#endif

/* ------------------------------------------------------------------ */

static int ogr_init(void)
{
  int r = init_load_choose();
  if (r != CORE_S_OK) {
    return r;
  }

  #if (OGROPT_BITOFLIST_DIRECT_BIT == 0) && (OGROPT_ALTERNATE_CYCLE == 0) && \
      defined(OGR_TEST_BITOFLIST)
  {
    unsigned int n, err_count = 0;
    printf("begin bit of list test\n");
    for( n = 0; n < (sizeof(ogr_bit_of_LIST)/sizeof(ogr_bit_of_LIST[0])); n++) 
    {
      U exp = 0x80000000 >> ((n-1) % 32); if (n == 0) exp = 0;
      if (exp != ogr_bit_of_LIST[n])
      {
        printf("ogr_bit_of_LIST[%d]=%u (but expected %u)\n", n, ogr_bit_of_LIST[n], exp);
        err_count++;
      }     
    }
    printf("end bit of list test. %d errors\n", err_count);
    if (err_count)
      return -1;
  }
  #endif    

  #if defined(OGR_TEST_FIRSTBLANK)
  {
    static int done_test = -1;
    static char ogr_first_blank[65537]; /* first blank in 16 bit COMP bitmap, range: 1..16 */
    /* first zero bit in 16 bits */
    int i, j, k = 0, m = 0x8000;

    for (i = 1; i <= 16; i++) {
      for (j = k; j < k+m; j++) ogr_first_blank[j] = (char)i;
      k += m;
      m >>= 1;
    }
    ogr_first_blank[0xffff] = 17;     /* just in case we use it */
    if ((++done_test) == 0)
    {
      unsigned int q, err_count = 0, first_fail = 0xffffffff;
      int last_s1 = 0, last_s2 = 0;
      printf("begin firstblank test\n"
             "(this may take a looooong time and requires a -KILL to stop)\n");
      for (q = 0; q <= 0xfffffffe; q++)
      {
        int s1 = ((q < 0xffff0000) ? \
          (ogr_first_blank[q>>16]) : (16 + ogr_first_blank[q - 0xffff0000]));
        int s2 = LOOKUP_FIRSTBLANK(q);
        int show_it = (q == 0xfffffffe || (q & 0xfffff) == 0xfffff);
        if (s1 != s2)
        {
          if (first_fail == 0xffffffff || (last_s1 != s1 || last_s2 != s2)) {
            printf("\n");
            first_fail = q;
            show_it = 1;
            last_s1 = s1;
            last_s2 = s2;
          }  
          if (show_it) {
            printf("\rfirstblank FAIL 0x%08x-0x%08x (should be %d, got %d) ", first_fail, q, s1, s2);
            fflush(stdout);
          }   
          err_count++;
        }
        else 
        {
          if (first_fail != 0xffffffff) {
            printf("\n");
            first_fail = 0xffffffff;
            show_it = 1;
          }  
          if (show_it) {
            printf("\rfirstblank [ok] 0x%08x-0x%08x ", q & 0xfff00000, q);
            fflush(stdout);
          }  
        }
      }
      printf("\nend firstblank test (%u errors)\n", err_count);
    }
    done_test = 0;
  }
  #endif

  return CORE_S_OK;
}

#ifdef OGR_DEBUG
static void dump(int depth, struct Level *lev, int limit)
{
  printf("--- depth %d\n", depth);
  printf("list=%08x%08x%08x%08x%08x\n", lev->list[0], lev->list[1], lev->list[2], lev->list[3], lev->list[4]);
  printf("dist=%08x%08x%08x%08x%08x\n", lev->dist[0], lev->dist[1], lev->dist[2], lev->dist[3], lev->dist[4]);
  printf("comp=%08x%08x%08x%08x%08x\n", lev->comp[0], lev->comp[1], lev->comp[2], lev->comp[3], lev->comp[4]);
  printf("cnt1=%d cnt2=%d limit=%d\n", lev->cnt1, lev->cnt2, limit);
  //sleep(1);
}
#endif

#if defined(HAVE_OGR_CORES)
static int ogr_create(void *input, int inputlen, void *state, int statelen, int dummy)
{
  struct State *oState;
  struct WorkStub *workstub = (struct WorkStub *)input;
  dummy = dummy;

  if (!input || inputlen != sizeof(struct WorkStub)) {
    return CORE_E_FORMAT;
  }

  if (((unsigned int)statelen) < sizeof(struct State)) {
    return CORE_E_FORMAT;
  }
  oState = (struct State *)state;
  if (!oState) {
    return CORE_E_MEMORY;
  }

  memset(oState, 0, sizeof(struct State));

  oState->maxdepth = workstub->stub.marks;
  oState->maxdepthm1 = oState->maxdepth-1;

  if (((unsigned int)oState->maxdepth) > (sizeof(OGR)/sizeof(OGR[0]))) {
    return CORE_E_FORMAT;
  }

  oState->max = OGR[oState->maxdepthm1];

  /* Note, marks are labled 0, 1...  so mark @ depth=1 is 2nd mark */
  oState->half_depth2 = oState->half_depth = ((oState->maxdepth+1) >> 1) - 1;
  if (!(oState->maxdepth % 2)) oState->half_depth2++;  /* if even, use 2 marks */

  /* Simulate GVANT's "KTEST=1" */
  oState->half_depth--;
  oState->half_depth2++;
  /*------------------
  Since:  half_depth2 = half_depth+2 (or 3 if maxdepth even) ...
  We get: half_length2 >= half_length + 3 (or 6 if maxdepth even)
  But:    half_length2 + half_length <= max-1    (our midpoint reduction)
  So:     half_length + 3 (6 if maxdepth even) + half_length <= max-1
  ------------------*/
                               oState->half_length = (oState->max-4) >> 1;
  if ( !(oState->maxdepth%2) ) oState->half_length = (oState->max-7) >> 1;

  oState->depth = 1;
  
#ifdef OGROPT_NEW_CHOOSEDAT
  /* would we choose values somewhere behind the precalculated values from 
     ogr_choose_dat2 ? */
  if (oState->maxdepthm1 - (oState->half_depth+1) > (CHOOSE_MAX_MARKS-1) ) {
    return CORE_E_CHOOSE;
  }
#endif

#if (OGROPT_ALTERNATE_CYCLE == 0)

  {
    int i, n;
    struct Level *lev, *lev2;

    n = workstub->worklength;
    if (n < workstub->stub.length) {
      n = workstub->stub.length;
    }
    if (n > STUB_MAX) {
      return CORE_E_FORMAT;
    }

    /* // level 0 - already done by memset
    lev = &oState->Levels[0];
    lev->cnt1 = lev->cnt2 = oState->marks[0] = 0;
    lev->limit = lev->maxlimit = 0;
    */
    
    lev = &oState->Levels[1];
    for (i = 0; i < n; i++) {
      int limit;
      if (oState->depth <= oState->half_depth2) {
        if (oState->depth <= oState->half_depth) {
          limit = oState->max - OGR[oState->maxdepthm1 - oState->depth];
          limit = limit < oState->half_length ? limit : oState->half_length;
        } else {
          limit = oState->max - choose(lev->dist[0] >> ttmDISTBITS, oState->maxdepthm1 - oState->depth);
          limit = limit < oState->max - oState->marks[oState->half_depth]-1 ? limit : oState->max - oState->marks[oState->half_depth]-1;
        }
      } else {
        limit = oState->max - choose(lev->dist[0] >> ttmDISTBITS, oState->maxdepthm1 - oState->depth);
      }
      lev->limit = limit;
      register int s = workstub->stub.diffs[i];
      
      if (s <= (32*5))
        if (lev->comp[(s-1)>>5] & BITOFLIST(s))
          return CORE_E_STUB;

      //dump(oState->depth, lev, 0);
      oState->marks[i+1] = oState->marks[i] + s;
      if ((lev->cnt2 += s) > limit)
        return CORE_E_STUB;

      register int t = s;
      while (t >= 32) {
        COMP_LEFT_LIST_RIGHT_32(lev);
        t -= 32;
      }
      if (t > 0) {
        COMP_LEFT_LIST_RIGHT(lev, t);
      }
      lev2 = lev + 1;
      COPY_LIST_SET_BIT(lev2, lev, s);
      COPY_DIST_COMP(lev2, lev);
      lev2->cnt1 = lev->cnt2;
      lev2->cnt2 = lev->cnt2;
      lev++;
      oState->depth++;
    }
    oState->depth--; // externally visible depth is one less than internal
  }

#else /* OGROPT_ALTERNATE_CYCLE > 0 */

  {
    int i, n;
    struct Level *lev = &oState->Levels[0];
    SETUP_TOP_STATE(oState,lev);
    lev++;
    n = workstub->worklength;

    if (n < workstub->stub.length) {
      n = workstub->stub.length;
    }

    if (n > STUB_MAX) {
      return CORE_E_FORMAT;
    }

    const int oStateMax = oState->max;
    const int oStateMaxDepthM1 = oState->maxdepthm1;
    const int oStateHalfDepth2 = oState->half_depth2;
    const int oStateHalfDepth = oState->half_depth;
    const int oStateHalfLength = oState->half_length;
    int oStateDepth = oState->depth;

    for (i = 0; i < n; i++) {
     int maxMinusDepth = oStateMaxDepthM1 - oStateDepth;

      if (oStateDepth <= oStateHalfDepth2) {
        if (oStateDepth <= oStateHalfDepth) {
          limit = oStateMax - OGR[maxMinusDepth];
          limit = (limit < oStateHalfLength) ? limit : oStateHalfLength;
        } else {
          limit = oStateMax - choose(dist0 >> ttmDISTBITS, maxMinusDepth);
        int tempLimit = oStateMax - oState->Levels[oStateHalfDepth].cnt2-1;
          limit = (limit < tempLimit) ? limit : tempLimit;
        }
      } else {
        limit = oStateMax - choose(dist0 >> ttmDISTBITS, maxMinusDepth);
      }

      int s = workstub->stub.diffs[i];
      
      //dump(oStateDepth, lev, 0);

// The following line is the same as:  oState->Levels[i+1].cnt2 = oState->Levels[i].cnt2 + s;
//   lev->cnt2 = lev[-1].cnt2 + s;
// because:  lev == oState->Levels[i+1]
// AND because we replace the count below, this assignment isn't needed at all!

      if ((cnt2 += s) > limit)
        return CORE_E_STUB;

      while (s>=32) {
        COMP_LEFT_LIST_RIGHT_32(lev);
        s -= 32;
      }

      COMP_LEFT_LIST_RIGHT(lev, s);
      PUSH_LEVEL_UPDATE_STATE(lev);
      lev++;
      oStateDepth++;

    }

    SAVE_FINAL_STATE(oState,lev);
    oState->depth = oStateDepth - 1; // externally visible depth is one less than internal
  }
#endif  /* OGROPT_ALTERNATE_CYCLE */

  oState->startdepth = workstub->stub.length;

#ifdef OGR_WINDOW
   oState->wind = oState->depth;
   oState->turn = 0;
#endif
#ifdef OGR_PROFILE
   oState->prof.hd = 0;
   oState->prof.hd2 = 0;
   oState->prof.ghd = 0;
   oState->prof.lt16 = 0;
   oState->prof.lt32 = 0;
   oState->prof.ge32 = 0;
   oState->prof.fo = 0;
   oState->prof.push = 0;
#endif
/*
  printf("sizeof      = %d\n", sizeof(struct State));
  printf("max         = %d\n", oState->max);
  printf("maxdepth    = %d\n", oState->maxdepth);
  printf("maxdepthm1  = %d\n", oState->maxdepthm1);
  printf("half_length = %d\n", oState->half_length);
  printf("half_depth  = %d\n", oState->half_depth);
  printf("half_depth2 = %d\n", oState->half_depth2);
  {
    int i;
    printf("marks       = ");
    for (i = 1; i < oState->depth; i++) {
      printf("%d ", oState->marks[i]-oState->marks[i-1]);
    }
    printf("\n");
  }
*/

  return CORE_S_OK;
}
#endif  /* HAVE_OGR_CORES */

#ifdef OGR_DEBUG
static void dump_ruler(struct State *oState, int depth)
{
  int i;
  printf("max %d ruler ", oState->max);
  for (i = 1; i < depth; i++) {
    #if (OGROPT_ALTERNATE_CYCLE == 0)
    printf("%d ", oState->marks[i] - oState->marks[i-1]);
    #else
    printf("%d ", oState->Levels[i].cnt2 - oState->Levels[i-1].cnt2);
    #endif
  }
  printf("\n");
}
#endif

/* ------------------------------------------------------------------ */

#if (defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__)) \
    && defined(HAVE_KOGE_PPC_CORES)
#if !define(OGROPT_IGNORE_TIME_CONSTRAINT_ARG)
  #error KOGE cores are NOT time constrained
#endif
#if (OGROPT_ALTERNATE_CYCLE == 2) && (defined(__VEC__) || defined(__ALTIVEC__))
static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
{
  with_time_constraints = with_time_constraints;
  return cycle_ppc_hybrid(state, pnodes, &choose(0,0), OGR);
}
#elif (OGROPT_ALTERNATE_CYCLE == 1)
static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
{
  with_time_constraints = with_time_constraints;
  return cycle_ppc_scalar(state, pnodes, &choose(0,0), OGR);
}
#else /* (OGROPT_ALTERNATE_CYCLE == 0) */
  #error unsupported setting
#endif

#else /* !HAVE_KOGE_PPC_CORES */

#if (OGROPT_ALTERNATE_CYCLE == 1) || (OGROPT_ALTERNATE_CYCLE == 2)
static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
{
  struct State *oState = (struct State *)state;
  int depth = oState->depth+1;      /* the depth of recursion */
  struct Level *lev = &oState->Levels[depth];
  int nodes = 0;
  const int oStateMax = oState->max;
  int remainingDepth = oState->maxdepthm1 - depth;
  const int oStateHalfDepth2 = oState->half_depth2;
  const int oStateHalfDepth  = oState->half_depth;
  struct Level *levHalfDepth = &oState->Levels[oStateHalfDepth];
  int retval = CORE_S_CONTINUE;

  SETUP_TOP_STATE(oState,lev);
  OGR_CYCLE_CACHE_ALIGN;

  for (;;) {
    int firstbit;
    limit = choose(dist0 >> ttmDISTBITS, remainingDepth);

    if (with_time_constraints) { /* if (...) is optimized away if unused */
      #if !defined(OGROPT_IGNORE_TIME_CONSTRAINT_ARG)
        if (nodes >= *pnodes) {
           break;
        }  
      #endif  
    }

    limit = oStateMax - limit;

    if (depth <= oStateHalfDepth2) {
      if (depth <= oStateHalfDepth) {
        limit = oStateMax - OGR[remainingDepth];
        if (nodes >= *pnodes) {
          break;
        }
        if (limit > oState->half_length)
          limit = oState->half_length;
      }
      else {
        int temp  = oStateMax - levHalfDepth->cnt2 - 1;
        if (limit > temp)
          limit = temp;
      }
    }

    nodes++;

    /* Find the next available mark location for this level */

  stay:
    if (comp0 < 0xfffffffe) {
      firstbit = LOOKUP_FIRSTBLANK(comp0);
      if ((cnt2 += firstbit) > limit)   goto up; /* no spaces left */
      COMP_LEFT_LIST_RIGHT(lev, firstbit);
    }
    else { /* firstbit >= 32 */
      if ((cnt2 += 32) > limit)  goto up; /* no spaces left */
      if (comp0 == ~0u) {
        COMP_LEFT_LIST_RIGHT_32(lev);
        goto stay;
      }
      else {
        COMP_LEFT_LIST_RIGHT_32(lev);
      }
    }
    /* New ruler? */
    if (remainingDepth == 0) {
      oState->Levels[oState->maxdepthm1].cnt2 = cnt2;
      retval = found_one(oState);
      if (retval != CORE_S_CONTINUE) {
        break;
      }
      goto stay;
    }

    /* Go Deeper */
    PUSH_LEVEL_UPDATE_STATE(lev);
    remainingDepth--;
    lev++;
    depth++;
    continue;

  up:
    lev--;
    depth--;
    remainingDepth++;
    POP_LEVEL(lev);

    if (depth <= oState->startdepth) {
      retval = CORE_S_OK;
      break;
    }
    goto stay; /* repeat this level till done */
  }

  SAVE_FINAL_STATE(oState,lev);
  oState->depth = depth-1;
  *pnodes = nodes;
  return retval;
}

#else   /* OGROPT_ALTERNATE_CYCLE == 0 */

static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
{
  struct State *oState = (struct State *)state;
  /* oState->depth is the level of the last placed mark */
  int depth = oState->depth+1;      /* the depth of recursion */
  /* our depth is the level where the next mark will be placed */
  struct Level *lev = &oState->Levels[depth];
  struct Level *lev2;
  int nodes = 0;
  int nodeslimit = *pnodes;
  int retval = CORE_S_CONTINUE;
  int limit;
  U comp0;

  #ifdef OGR_DEBUG
  oState->LOGGING = 1;
  #endif

  OGR_CYCLE_CACHE_ALIGN;

  for (;;) {

    if (with_time_constraints) { /* if (...) is optimized away if unused */
       #if !defined(OGROPT_IGNORE_TIME_CONSTRAINT_ARG)
       if (nodes >= nodeslimit) {
         break;
       }  
       #endif  
    }

    #ifdef OGR_DEBUG
    if (oState->LOGGING) dump_ruler(oState, depth);
    #endif

    if (depth <= oState->half_depth2) {
      if (depth <= oState->half_depth) {

        //dump_ruler(oState, depth);
        if (nodes >= nodeslimit) {
          break;
        }

        limit = oState->max - OGR[oState->maxdepthm1 - depth];
        limit = limit < oState->half_length ? limit : oState->half_length;
      } else {
        limit = oState->max - choose(lev->dist[0] >> ttmDISTBITS, oState->maxdepthm1 - depth);
        limit = limit < oState->max - oState->marks[oState->half_depth]-1 ? limit : oState->max - oState->marks[oState->half_depth]-1;
      }
    } else {
      limit = oState->max - choose(lev->dist[0] >> ttmDISTBITS, oState->maxdepthm1 - depth);
    }

    #ifdef OGR_DEBUG
    if (oState->LOGGING) dump(depth, lev, limit);
    #endif

    nodes++;

    /* Find the next available mark location for this level */
stay:
    comp0 = lev->comp[0];
    #ifdef OGR_DEBUG
    if (oState->LOGGING) printf("comp0=%08x\n", comp0);
    #endif
    if (comp0 < 0xfffffffe) {
      int s = LOOKUP_FIRSTBLANK( comp0 );
    #ifdef OGR_DEBUG
    if (oState->LOGGING) printf("depth=%d s=%d len=%d limit=%d\n", depth, s+(lev->cnt2-lev->cnt1), lev->cnt2+s, limit);
    #endif
      if ((lev->cnt2 += s) > limit) goto up; /* no spaces left */
      COMP_LEFT_LIST_RIGHT(lev, s);
    } else {
      /* s>32 */
      if ((lev->cnt2 += 32) > limit) goto up; /* no spaces left */
      COMP_LEFT_LIST_RIGHT_32(lev);
      if (comp0 == 0xffffffff) goto stay;
    }


    /* New ruler? */
    if (depth == oState->maxdepthm1) {
      oState->marks[oState->maxdepthm1] = lev->cnt2;       /* not placed yet into list arrays! */
      if (found_one(oState)) {
        retval = CORE_S_SUCCESS;
        break;
      }
      goto stay;
    }

    /* Go Deeper */
    lev2 = lev + 1;
  #if (OGROPT_COMBINE_COPY_LIST_SET_BIT_COPY_DIST_COMP == 1)
    COPY_LIST_SET_BIT_COPY_DIST_COMP(lev2, lev, lev->cnt2-lev->cnt1);
  #else
    COPY_LIST_SET_BIT(lev2, lev, lev->cnt2-lev->cnt1);
    COPY_DIST_COMP(lev2, lev);
  #endif
    oState->marks[depth] = lev->cnt2;
    lev2->cnt1 = lev->cnt2;
    lev2->cnt2 = lev->cnt2;
    lev->limit = limit;
    oState->depth = depth;
    lev++;
    depth++;

    continue;

up:
    lev--;
    depth--;
    oState->depth = depth-1;
    if (depth <= oState->startdepth) {
      retval = CORE_S_OK;
      break;
    }
    limit = lev->limit;

    goto stay; /* repeat this level till done */
  }

  oState->depth = depth-1;

  *pnodes = nodes;

  return retval;
}
#endif  /* OGROPT_ALTERNATE_CYCLE */
#endif  /* !HAVE_KOGE_PPC_CORES */

static int ogr_getresult(void *state, void *result, int resultlen)
{
  struct State *oState = (struct State *)state;
  struct WorkStub *workstub = (struct WorkStub *)result;
  int i;

  if (resultlen != sizeof(struct WorkStub)) {
    return CORE_E_FORMAT;
  }
  workstub->stub.marks = (u16)oState->maxdepth;
  workstub->stub.length = (u16)oState->startdepth;
  for (i = 0; i < STUB_MAX; i++) {
    #if (OGROPT_ALTERNATE_CYCLE == 0)
    workstub->stub.diffs[i] = (u16)(oState->marks[i+1] - oState->marks[i]);
    #else
    workstub->stub.diffs[i] = (u16)(oState->Levels[i+1].cnt2 - oState->Levels[i].cnt2);
    #endif
  }
  workstub->worklength = oState->depth;
  if (workstub->worklength > STUB_MAX) {
    workstub->worklength = STUB_MAX;
  }
  return CORE_S_OK;
}

static int ogr_destroy(void *state)
{
  #if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
  if (state) free(state);
  #else
  state = state;
  #endif
  return CORE_S_OK;
}

#if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
static int ogr_count(void *state)
{
  return sizeof(struct State);
}

static int ogr_save(void *state, void *buffer, int buflen)
{
  if (buflen < sizeof(struct State)) {
    return CORE_E_MEMORY;
  }
  memcpy(buffer, state, sizeof(struct State));
  return CORE_S_OK;
}

static int ogr_load(void *buffer, int buflen, void **state)
{
  if (buflen < sizeof(struct State)) {
    return CORE_E_FORMAT;
  }
  *state = malloc(sizeof(struct State));
  if (!*state) {
    return CORE_E_MEMORY;
  }
  memcpy(*state, buffer, sizeof(struct State));
  return CORE_S_OK;
}
#endif  /* HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS */

static int ogr_cleanup(void)
{
  return CORE_S_OK;
}

#if defined(HAVE_OGR_CORES)
CoreDispatchTable * OGR_GET_DISPATCH_TABLE_FXN (void)
{
  static CoreDispatchTable dispatch_table;
  dispatch_table.init      = ogr_init;
  dispatch_table.create    = ogr_create;
  dispatch_table.cycle     = ogr_cycle;
  dispatch_table.getresult = ogr_getresult;
  dispatch_table.destroy   = ogr_destroy;
  #if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
  dispatch_table.count     = ogr_count;
  dispatch_table.save      = ogr_save;
  dispatch_table.load      = ogr_load;
  #endif
  dispatch_table.cleanup   = ogr_cleanup;
  return &dispatch_table;
}
#endif


/*==========================================================================*/
/*========================== OGR-24/25 FINAL STEPS =========================*/
/*==========================================================================*/
/*
** OGR-24 :
** The method is based upon 3-diffs (4 marks) and 4-diffs (5 marks) stubs :
** + 3-diffs stubs : Check rulers that have the 5th mark at position >= 80
**   (by construction, the 6th mark cannot be placed at a position <= 70).
** + 4-diffs stubs : Check rulers that have the 6th mark at position >= 71.
**
** Stated otherwise, let 24/a-b-c-d be a valid 4-diffs stub
** - If a+b+c+d < 80, the client expects this 4-diffs stub.
** - If a+b+c+d >= 80, the client expects the 3-diffs stub 24/a-b-c
**
**
** OGR-25 :
** The method is based upon 3-diffs (4 marks), 4-diffs (5 marks), 5-diffs
** (6 marks) stubs, and regular 6-diffs stubs :
** + 3-diffs stubs : Check rulers that have the 5th mark at position >= 125.
** + 4-diffs stubs : Check rulers that have the 6th mark at position >= 115.
** + 5-diffs stubs : Check rulers that have the 7th mark at position > 100.
** + 6-diffs stubs : Similar to classic OGR, but the threshold is 100 (vs 70).
**
** Stated otherwise, let 25/a-b-c-d-e be a valid 5-diffs stub
** - If a+b+c+d >= 125, the client expects the 3-diffs stub 25/a-b-c
** - If a+b+c+d+e >= 115, the client expects the 4-diffs stub 25/a-b-c-d
** - If a+b+c+d+e < 115, the client expects the 5-diffs stub 25/a-b-c-d-e
**
** The real limits are specified on the keyserver. The client just knows how
** to handle sub-normal stubs (which have one diff less than regular stubs).
** Regular stubs are processed as usual (100% OGR compatible).
*/

static int ogr_create_pass2(void *input, int inputlen, void *state, 
        int statelen, int minpos)
{
  int finalization_stub = 0;
  struct State *oState;
  struct WorkStub *workstub = (struct WorkStub *)input;

  if (!input || inputlen != sizeof(struct WorkStub)) {
    return CORE_E_FORMAT;
  }

  if (((unsigned int)statelen) < sizeof(struct State)) {
    return CORE_E_FORMAT;
  }
  oState = (struct State *)state;
  if (!oState) {
    return CORE_E_MEMORY;
  }

  memset(oState, 0, sizeof(struct State));

  oState->maxdepth = workstub->stub.marks;
  oState->maxdepthm1 = oState->maxdepth-1;

  if (((unsigned int)oState->maxdepth) > (sizeof(OGR)/sizeof(OGR[0]))) {
    return CORE_E_FORMAT;
  }

  /*
  ** Sanity check
  ** Make sure we only get OGR-24/OGR-25 finalization stubs, and make
  ** sure supposed finalization stubs (that have less diffs than regular
  ** stubs) come with the corresponding starting point.
  ** OGR already handled all stubs upto (and including) length 70, so the
  ** starting point must be higher.
  */
  if (workstub->stub.marks == 24 && workstub->stub.length < 5) {
    if (minpos <= 70 || minpos > OGR[24-1] - OGR[(24-2) - workstub->stub.length]) {
      return CORE_E_FORMAT;         // Too low.
    }
    finalization_stub = 1;
  }
  else if (workstub->stub.marks == 25 && workstub->stub.length < 6) {
    if (minpos <= 70 || minpos > OGR[25-1] - OGR[(25-2) - workstub->stub.length]) {
      return CORE_E_FORMAT;         // Too low.
    }
    finalization_stub = 1;
  }
  else if (minpos != 0) {
    // Unsuspected starting point
    return CORE_E_FORMAT;
  }

  if (workstub->stub.length < workstub->worklength) {
    /* BUGFIX : Reset the flag if the stub has already been started to prevent
    ** inaccurate node counts when the user stop then restart the client.
    ** (the init loop performs one 'cycle' iteration for finalization stubs,
    ** and that iteration is not taken into account in the final node count) */
    finalization_stub = 0;
  }

  oState->max = OGR[oState->maxdepthm1];

  /* Note, marks are labled 0, 1...  so mark @ depth=1 is 2nd mark */
  oState->half_depth2 = oState->half_depth = ((oState->maxdepth+1) >> 1) - 1;
  if (!(oState->maxdepth % 2)) oState->half_depth2++;  /* if even, use 2 marks */

  /* Simulate GVANT's "KTEST=1" */
  oState->half_depth--;
  oState->half_depth2++;
  /*------------------
  Since:  half_depth2 = half_depth+2 (or 3 if maxdepth even) ...
  We get: half_length2 >= half_length + 3 (or 6 if maxdepth even)
  But:    half_length2 + half_length <= max-1    (our midpoint reduction)
  So:     half_length + 3 (6 if maxdepth even) + half_length <= max-1
  ------------------*/
                               oState->half_length = (oState->max-4) >> 1;
  if ( !(oState->maxdepth%2) ) oState->half_length = (oState->max-7) >> 1;

  oState->depth = 1;
  
  #ifdef OGROPT_NEW_CHOOSEDAT
  /* would we choose values somewhere behind the precalculated values from 
     ogr_choose_dat2 ? */
  if (oState->maxdepthm1 - (oState->half_depth+1) > (CHOOSE_MAX_MARKS-1) ) {
    return CORE_E_CHOOSE;
  }
  #endif

#if (OGROPT_ALTERNATE_CYCLE == 0)

  {
    int i, n;
    struct Level *lev, *lev2;
    U comp0;

    n = workstub->worklength;
    if (n < workstub->stub.length) {
      n = workstub->stub.length;
    }
    if (n > STUB_MAX) {
      return CORE_E_FORMAT;
    }

    /* // level 0 - already done by memset
    lev = &oState->Levels[0];
    lev->cnt1 = lev->cnt2 = oState->marks[0] = 0;
    lev->limit = lev->maxlimit = 0;
    */
    
    lev = &oState->Levels[1];
    for (i = 0; i < n + finalization_stub; i++) {
      int limit;
      if (oState->depth <= oState->half_depth2) {
        if (oState->depth <= oState->half_depth) {
          limit = oState->max - OGR[oState->maxdepthm1 - oState->depth];
          limit = limit < oState->half_length ? limit : oState->half_length;
        } else {
          limit = oState->max - choose(lev->dist[0] >> ttmDISTBITS, oState->maxdepthm1 - oState->depth);
          limit = limit < oState->max - oState->marks[oState->half_depth]-1 ? limit : oState->max - oState->marks[oState->half_depth]-1;
        }
      } else {
        limit = oState->max - choose(lev->dist[0] >> ttmDISTBITS, oState->maxdepthm1 - oState->depth);
      }
      lev->limit = limit;
      register int s = workstub->stub.diffs[i];
      
      if (i == n) {
        /*
        ** Finalization stub
        ** Skip to the minimum position specified by the keyserver, then
        ** place the next mark at the first position available from here
        ** (including the specified minimum position).
        ** As a result, ogr_cycle() will start one level deeper, then
        ** proceed as usual until it backtracks to depth oState->startdepth.
        */
        s = minpos - 1;
        if (lev->cnt2 < s) {
          int k = s - lev->cnt2;
          while (k >= 32) {
            COMP_LEFT_LIST_RIGHT_32(lev);
            k -= 32;
          }
          if (k > 0) {
            COMP_LEFT_LIST_RIGHT(lev, k);
          }
          lev->cnt2 = s;
        }

        stay:
        comp0 = lev->comp[0];
        if (comp0 < 0xfffffffe) {
          s = LOOKUP_FIRSTBLANK( comp0 );
          if ((lev->cnt2 += s) > limit)
            return CORE_E_STUB;
          COMP_LEFT_LIST_RIGHT(lev, s);
        } 
        else { /* s>32 */
          if ((lev->cnt2 += 32) > limit)
            return CORE_E_STUB;
          COMP_LEFT_LIST_RIGHT_32(lev);
          if (comp0 == 0xffffffff) goto stay;
        }
        oState->marks[i+1] = lev->cnt2;
        s = lev->cnt2 - lev->cnt1;
      }
      else {
        /*
        ** Regular stubs
        */
        if (s <= (32*5))
          if (lev->comp[(s-1)>>5] & BITOFLIST(s))
            return CORE_E_STUB;     // Not golomb

        oState->marks[i+1] = oState->marks[i] + s;
        if ((lev->cnt2 += s) > limit)
          return CORE_E_STUB;

        register int t = s;
        while (t >= 32) {
          COMP_LEFT_LIST_RIGHT_32(lev);
          t -= 32;
        }
        if (t > 0) {
          COMP_LEFT_LIST_RIGHT(lev, t);
        }
      }
      lev2 = lev + 1;
      COPY_LIST_SET_BIT(lev2, lev, s);
      COPY_DIST_COMP(lev2, lev);
      lev2->cnt1 = lev->cnt2;
      lev2->cnt2 = lev->cnt2;
      lev++;
      oState->depth++;
    }
    oState->depth--; // externally visible depth is one less than internal
  }

#else   /* OGROPT_ALTERNATE_CYCLE > 0 */

  {
    int i, n;
    struct Level *lev = &oState->Levels[0];
    SETUP_TOP_STATE(oState,lev);
    lev++;
    n = workstub->worklength;

    if (n < workstub->stub.length) {
      n = workstub->stub.length;
    }

    if (n > STUB_MAX) {
      return CORE_E_FORMAT;
    }

    const int oStateMax = oState->max;
    const int oStateMaxDepthM1 = oState->maxdepthm1;
    const int oStateHalfDepth2 = oState->half_depth2;
    const int oStateHalfDepth = oState->half_depth;
    const int oStateHalfLength = oState->half_length;
    int oStateDepth = oState->depth;

    for (i = 0; i < n + finalization_stub; i++) {
     int maxMinusDepth = oStateMaxDepthM1 - oStateDepth;

      if (oStateDepth <= oStateHalfDepth2) {
        if (oStateDepth <= oStateHalfDepth) {
          limit = oStateMax - OGR[maxMinusDepth];
          limit = (limit < oStateHalfLength) ? limit : oStateHalfLength;
        } else {
          limit = oStateMax - choose(dist0 >> ttmDISTBITS, maxMinusDepth);
        int tempLimit = oStateMax - oState->Levels[oStateHalfDepth].cnt2-1;
          limit = (limit < tempLimit) ? limit : tempLimit;
        }
      } else {
        limit = oStateMax - choose(dist0 >> ttmDISTBITS, maxMinusDepth);
      }

      int s = workstub->stub.diffs[i];

      if (i == n) {
        /*
        ** Finalization stub
        ** Skip to the minimum position specified by the keyserver, then
        ** place the next mark at the first position available from here
        ** (including the specified minimum position)
        ** As a result, ogr_cycle() will start one level deeper, then
        ** proceed as usual until it backtracks to depth oState->startdepth.
        */
        s = minpos - 1;
        if (cnt2 < s) {
          int k = s - cnt2;
          while (k >= 32) {
            COMP_LEFT_LIST_RIGHT_32(lev);
            k -= 32;
          }
          if (k > 0) {
            COMP_LEFT_LIST_RIGHT(lev, k);
          }
          cnt2 = s;
        }

        stay:
        if (comp0 < 0xfffffffe) {
          s = LOOKUP_FIRSTBLANK( comp0 );
          if ((cnt2 += s) > limit)
            return CORE_E_STUB;
          COMP_LEFT_LIST_RIGHT(lev, s);
        } 
        else { /* s>32 */
          U comp = comp0;
          if ((cnt2 += 32) > limit)
            return CORE_E_STUB;
          COMP_LEFT_LIST_RIGHT_32(lev);
          if (comp == 0xffffffff)    goto stay;
        }
      }
      else {
        /*
        ** Regular stubs
        */
        if ((cnt2 += s) > limit)
          return CORE_E_STUB;

        while (s>=32) {
          COMP_LEFT_LIST_RIGHT_32(lev);
          s -= 32;
        }
        if (s > 0) {
          COMP_LEFT_LIST_RIGHT(lev, s);
        }
      }
      PUSH_LEVEL_UPDATE_STATE(lev);
      lev++;
      oStateDepth++;
    }

    SAVE_FINAL_STATE(oState,lev);
    oState->depth = oStateDepth - 1; // externally visible depth is one less than internal
  }
#endif    /* OGROPT_ALTERNATE_CYCLE */

  oState->startdepth = workstub->stub.length;

  return CORE_S_OK;
}

CoreDispatchTable * OGR_P2_GET_DISPATCH_TABLE_FXN (void)
{
  static CoreDispatchTable dispatch_table_pass2;
  dispatch_table_pass2.init      = ogr_init;
  dispatch_table_pass2.create    = ogr_create_pass2;
  dispatch_table_pass2.cycle     = ogr_cycle;
  dispatch_table_pass2.getresult = ogr_getresult;
  dispatch_table_pass2.destroy   = ogr_destroy;
  #if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
  dispatch_table_pass2.count     = ogr_count;
  dispatch_table_pass2.save      = ogr_save;
  dispatch_table_pass2.load      = ogr_load;
  #endif
  dispatch_table_pass2.cleanup   = ogr_cleanup;
  return &dispatch_table_pass2;
}
