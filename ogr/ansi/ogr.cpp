/*
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogr.cpp,v 1.1.2.42 2001/02/19 08:10:39 mfeiri Exp $
 */
#include <stdio.h>  /* printf for debugging */
#include <stdlib.h> /* malloc (if using non-static choose dat) */
#include <string.h> /* memset */

#define HAVE_STATIC_CHOOSEDAT /* choosedat table is static, pre-generated */
/* #define CRC_CHOOSEDAT_ANYWAY */ /* you'll need to link crc32 if this is defd */

/* #define OGR_TEST_FIRSTBLANK */ /* test firstblank logic (table or asm) */
/* #define OGR_TEST_BITOFLIST  */ /* test bitoflist table */

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
#elif (defined(ASM_X86) || defined(__386__))
  #if defined(OGR_NOFFZ) 
    /* the bsr insn is slooooow on anything less than a PPro */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 0
  #endif
  #define OGROPT_BITOFLIST_DIRECT_BIT 0          /* we want 'no' */
#elif defined(ASM_68K)
  #define OGROPT_BITOFLIST_DIRECT_BIT 0          /* we want 'no' */
#elif defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__)
  #if (__MWERKS__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* MWC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 2 /* use switch_asm   */
  #elif (__MRC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* MrC is better    */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* MrC is better    */
  #elif (__APPLE_CC__)//GCC with exclusive ppc, mach-o and ObjC extensions
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* ACC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* ACC is better    */
  #elif (__GNUC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* GCC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 2 /* use switch_asm   */
  #else
    #error play with the defines to find optimal settings for your compiler
  #endif
#elif defined(ASM_POWER)
  #if (__GNUC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* GCC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 2 /* use switch_asm   */
  #else
    #error play with the defines to find optimal settings for your compiler
  #endif
#elif defined(ASM_ARM)
  #if (__GNUC__)
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1
    #define OGROPT_BITOFLIST_DIRECT_BIT           0
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 2
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1
    #define OGROPT_ALTERNATE_CYCLE                0
    #define OGROPT_COMBINE_COPY_LIST_SET_BIT_COPY_DIST_COMP 1
  #endif
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
  #undef OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
  #if (defined(__PPC__) || defined(ASM_PPC)) || defined(__POWERPC__) ||\
      (defined(__WATCOMC__) && defined(__386__)) || \
      (defined(__ICC)) /* icc is Intel only (duh!) */ || \
      (defined(__GNUC__) && (defined(ASM_ALPHA) \
                             || defined(ASM_X86) \
                             || defined(ASM_ARM) \
                             || (defined(ASM_68K) && (defined(mc68020) \
                             || defined(mc68030) || defined(mc68040) \
                             || defined(mc68060)))))
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 1
    /* #define OGR_TEST_FIRSTBLANK */ /* ... to test */
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
#define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 2 /* 0 (no opt) or 1 or 2 */
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
   OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1 -> alternate approach by sampoo
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
    (defined(ASM_X86) && defined(GENERATE_ASM))
    /* ASM_X86: the ogr core used by all x86 platforms is a hand optimized */
    /* .S/.asm version of this core - If we're compiling for asm_x86 then */
    /* we're either generating an .S for later optimization, or compiling */
    /* for comparison with an existing .S */
  #undef OGROPT_IGNORE_TIME_CONSTRAINT_ARG
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

static const int OGR[] = {
  /*  1 */    0,   1,   3,   6,  11,  17,  25,  34,  44,  55,
  /* 11 */   72,  85, 106, 127, 151, 177, 199, 216, 246, 283,
  /* 21 */  333, 356, 372, 425, 480, 492, 553, 585, 623
};
#ifndef __MRC__
static int init_load_choose(void);
static int found_one(const struct State *oState);
static int ogr_init(void);
static int ogr_create(void *input, int inputlen, void *state, int statelen);
static int ogr_cycle(void *state, int *pnodes, int with_time_constraints);
static int ogr_getresult(void *state, void *result, int resultlen);
static int ogr_destroy(void *state);
#if defined(HAVE_OGR_COUNT_SAVE_LOAD_FUNCTIONS)
static int ogr_count(void *state);
static int ogr_save(void *state, void *buffer, int buflen);
static int ogr_load(void *buffer, int buflen, void **state);
#endif
static int ogr_cleanup(void);
#endif

#if defined(_AIXALL) && defined(ASM_POWER)
  #define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table_power
#else
  #ifndef OGR_GET_DISPATCH_TABLE_FXN
    #define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table
  #endif
#endif  

extern CoreDispatchTable * OGR_GET_DISPATCH_TABLE_FXN (void);

#if defined(__cplusplus)
}
#endif

/* ------------------------------------------------------------------ */

/***********************************************************************************
 * The following macros define the BITLIST CLASS
 * The variables defined here should only be manipulated within these class macros.
 ***********************************************************************************/

#if (OGROPT_ALTERNATE_CYCLE == 2) /* support macros for the vectorized ogr_cycle() routine */

 /* define the local variables used for the top recursion state */
#define SETUP_TOP_STATE(state,lev)                                  \
   vector unsigned int  compV0;                                     \
   vector unsigned int  compV1;                                     \
   vector unsigned int  listV0;                                     \
   vector unsigned int  listV1;                                     \
   vector unsigned int  distV0;                                     \
   vector unsigned int  distV1;                                     \
   int cnt2 = lev->cnt2;                                            \
   vector unsigned int ZEROBIT = (vector unsigned int)(0, 0, 1, 0); \
   vector unsigned int ZEROS = (vector unsigned int)(0);            \
   vector unsigned int ONES = vec_nor(ZEROS,ZEROS);                 \
   int limit;                                                       \
   union {                                                          \
      vector unsigned int V;                                        \
      unsigned int U[4];                                            \
   } VU;                                                            \
   compV0 = lev->compV0;                                            \
   compV1 = lev->compV1;                                            \
   listV0 = vec_or(lev->listV0, ZEROBIT);                           \
   listV1 = lev->listV1;                                            \
   distV0 = state->distV0;                                          \
   distV1 = state->distV1;

#define VEC_TO_INT(v,n) (VU.V = (v), VU.U[n])

/* shift the list to add or extend the first mark */
#define COMP_LEFT_LIST_RIGHT(lev, s)                                \
   VU.U[3] = s;                                                     \
   vector unsigned int Vs = vec_splat(VU.V,3);                      \
   vector unsigned int Vm = vec_sl(ONES,Vs);                        \
   vector unsigned int Vss = vec_sub(ZEROS,Vs);                     \
   compV0 = vec_rl(compV0,Vs);                                      \
   compV1 = vec_rl(compV1,Vs);                                      \
   compV0 = vec_sel(vec_sld(compV0,compV1,4),compV0,Vm);            \
   compV1 = vec_sel(vec_sld(compV1,ZEROS,4),compV1,Vm);             \
   listV1 = vec_sel(vec_sld(listV0,listV1,12),listV1,Vm);           \
   listV0 = vec_sel(vec_sld(ZEROS,listV0,12),listV0,Vm);            \
   listV1 = vec_rl(listV1,Vss);                                     \
   listV0 = vec_rl(listV0,Vss);

/* shift by word size */
#define COMP_LEFT_LIST_RIGHT_32(lev)                                \
   compV0 = vec_sld(compV0, compV1, 4);                             \
   compV1 = vec_sld(compV1, ZEROS, 4);                              \
   listV1 = vec_sld(listV0, listV1, 12);                            \
   listV0 = vec_sld(ZEROS, listV0, 12);

/* set the current mark and push a level to start a new mark */
#define PUSH_LEVEL_UPDATE_STATE(lev)                                \
   lev->listV0 = listV0;                                            \
   lev->listV1 = listV1;                                            \
   listV0 = vec_or(listV0, ZEROBIT);                                \
   distV0 = vec_or(distV0, listV0);                                 \
   distV1 = vec_or(distV1, listV1);                                 \
   lev->compV0 = compV0;                                            \
   lev->compV1 = compV1;                                            \
   compV0 = vec_or(compV0, distV0);                                 \
   compV1 = vec_or(compV1, distV1);                                 \
   lev->cnt2 = cnt2;                                                \
   lev->limit = limit;

/* pop a level to continue work on previous mark */
#define POP_LEVEL(lev)                                              \
   listV0 = lev->listV0;                                            \
   listV1 = lev->listV1;                                            \
   distV0 = vec_andc(distV0, listV0);                               \
   distV1 = vec_andc(distV1, listV1);                               \
   compV0 = lev->compV0;                                            \
   compV1 = lev->compV1;                                            \
   limit = lev->limit;                                              \
   cnt2 = lev->cnt2;

/* save the local state variables */
#define SAVE_FINAL_STATE(state,lev)                                 \
   lev->listV0 = listV0;                                            \
   lev->listV1 = listV1;                                            \
   state->distV0 = distV0;                                          \
   state->distV1 = distV1;                                          \
   lev->compV0 = compV0;                                            \
   lev->compV1 = compV1;                                            \
   lev->cnt2 = cnt2;

#elif (OGROPT_ALTERNATE_CYCLE == 1) /* support macros for the alternate ogr_cycle() routine */

#define SETUP_TOP_STATE(state,lev)                 \
   U  comp0 = lev->comp[0], comp1 = lev->comp[1], comp2 = lev->comp[2], comp3 = lev->comp[3], comp4 = lev->comp[4]; \
   U  list0 = lev->list[0], list1 = lev->list[1], list2 = lev->list[2], list3 = lev->list[3], list4 = lev->list[4]; \
   U  dist0 = state->dist[0], dist1 = state->dist[1], dist2 = state->dist[2], dist3 = state->dist[3], dist4 = state->dist[4]; \
   int cnt2 = lev->cnt2;                           \
   int newbit = 1;                                 \
   int limit;

/* shift the list to add or extend the first mark */
#if (OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 2)

  #if defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__) || \
      defined(ASM_POWER)
  #if defined(__GNUC__)
     #define __rlwinm(Rs,SH,MB,ME) \
     ({ int Ra; __asm__ ("rlwinm %0,%1,%2,%3,%4" : "=r" (Ra) : "r" (Rs), "n" (SH), "n" (MB), "n" (ME)); Ra; })
     #define __rlwimi(Ra,Rs,SH,MB,ME) \
     ({ __asm__ ("rlwimi %0,%2,%3,%4,%5" : "=r" (Ra) : "0" (Ra), "r" (Rs), "n" (SH), "n" (MB), "n" (ME)); Ra; })
  #endif /* __GNUC__ */
  #define COMP_LEFT_LIST_RIGHT(lev, s)             \
  {                                                \
    switch (s)                                     \
    {                                              \
      case 0:                                      \
         break;                                    \
      case 1:                                      \
         comp0 = __rlwinm(comp0,1,0,30);           \
         list4 = __rlwinm(list4,31,1,31);          \
         comp0 = __rlwimi(comp0,comp1,1,31,31);    \
         list4 = __rlwimi(list4,list3,31,0,0);     \
         comp1 = __rlwinm(comp1,1,0,30);           \
         list3 = __rlwinm(list3,31,1,31);          \
         comp1 = __rlwimi(comp1,comp2,1,31,31);    \
         list3 = __rlwimi(list3,list2,31,0,0);     \
         comp2 = __rlwinm(comp2,1,0,30);           \
         list2 = __rlwinm(list2,31,1,31);          \
         comp2 = __rlwimi(comp2,comp3,1,31,31);    \
         list2 = __rlwimi(list2,list1,31,0,0);     \
         comp3 = __rlwinm(comp3,1,0,30);           \
         list1 = __rlwinm(list1,31,1,31);          \
         comp3 = __rlwimi(comp3,comp4,1,31,31);    \
         list1 = __rlwimi(list1,list0,31,0,0);     \
         comp4 = __rlwinm(comp4,1,0,30);           \
         list0 = __rlwinm(list0,31,1,31);          \
         list0 = __rlwimi(list0,newbit,31,0,0);    \
         break;                                    \
      case 2:                                      \
         comp0 = __rlwinm(comp0,2,0,29);           \
         list4 = __rlwinm(list4,30,2,31);          \
         comp0 = __rlwimi(comp0,comp1,2,30,31);    \
         list4 = __rlwimi(list4,list3,30,0,1);     \
         comp1 = __rlwinm(comp1,2,0,29);           \
         list3 = __rlwinm(list3,30,2,31);          \
         comp1 = __rlwimi(comp1,comp2,2,30,31);    \
         list3 = __rlwimi(list3,list2,30,0,1);     \
         comp2 = __rlwinm(comp2,2,0,29);           \
         list2 = __rlwinm(list2,30,2,31);          \
         comp2 = __rlwimi(comp2,comp3,2,30,31);    \
         list2 = __rlwimi(list2,list1,30,0,1);     \
         comp3 = __rlwinm(comp3,2,0,29);           \
         list1 = __rlwinm(list1,30,2,31);          \
         comp3 = __rlwimi(comp3,comp4,2,30,31);    \
         list1 = __rlwimi(list1,list0,30,0,1);     \
         comp4 = __rlwinm(comp4,2,0,29);           \
         list0 = __rlwinm(list0,30,2,31);          \
         list0 = __rlwimi(list0,newbit,30,0,1);    \
         break;                                    \
      case 3:                                      \
         comp0 = __rlwinm(comp0,3,0,28);           \
         list4 = __rlwinm(list4,29,3,31);          \
         comp0 = __rlwimi(comp0,comp1,3,29,31);    \
         list4 = __rlwimi(list4,list3,29,0,2);     \
         comp1 = __rlwinm(comp1,3,0,28);           \
         list3 = __rlwinm(list3,29,3,31);          \
         comp1 = __rlwimi(comp1,comp2,3,29,31);    \
         list3 = __rlwimi(list3,list2,29,0,2);     \
         comp2 = __rlwinm(comp2,3,0,28);           \
         list2 = __rlwinm(list2,29,3,31);          \
         comp2 = __rlwimi(comp2,comp3,3,29,31);    \
         list2 = __rlwimi(list2,list1,29,0,2);     \
         comp3 = __rlwinm(comp3,3,0,28);           \
         list1 = __rlwinm(list1,29,3,31);          \
         comp3 = __rlwimi(comp3,comp4,3,29,31);    \
         list1 = __rlwimi(list1,list0,29,0,2);     \
         comp4 = __rlwinm(comp4,3,0,28);           \
         list0 = __rlwinm(list0,29,3,31);          \
         list0 = __rlwimi(list0,newbit,29,0,2);    \
         break;                                    \
      case 4:                                      \
         comp0 = __rlwinm(comp0,4,0,27);           \
         list4 = __rlwinm(list4,28,4,31);          \
         comp0 = __rlwimi(comp0,comp1,4,28,31);    \
         list4 = __rlwimi(list4,list3,28,0,3);     \
         comp1 = __rlwinm(comp1,4,0,27);           \
         list3 = __rlwinm(list3,28,4,31);          \
         comp1 = __rlwimi(comp1,comp2,4,28,31);    \
         list3 = __rlwimi(list3,list2,28,0,3);     \
         comp2 = __rlwinm(comp2,4,0,27);           \
         list2 = __rlwinm(list2,28,4,31);          \
         comp2 = __rlwimi(comp2,comp3,4,28,31);    \
         list2 = __rlwimi(list2,list1,28,0,3);     \
         comp3 = __rlwinm(comp3,4,0,27);           \
         list1 = __rlwinm(list1,28,4,31);          \
         comp3 = __rlwimi(comp3,comp4,4,28,31);    \
         list1 = __rlwimi(list1,list0,28,0,3);     \
         comp4 = __rlwinm(comp4,4,0,27);           \
         list0 = __rlwinm(list0,28,4,31);          \
         list0 = __rlwimi(list0,newbit,28,0,3);    \
         break;                                    \
      case 5:                                      \
         comp0 = __rlwinm(comp0,5,0,26);           \
         list4 = __rlwinm(list4,27,5,31);          \
         comp0 = __rlwimi(comp0,comp1,5,27,31);    \
         list4 = __rlwimi(list4,list3,27,0,4);     \
         comp1 = __rlwinm(comp1,5,0,26);           \
         list3 = __rlwinm(list3,27,5,31);          \
         comp1 = __rlwimi(comp1,comp2,5,27,31);    \
         list3 = __rlwimi(list3,list2,27,0,4);     \
         comp2 = __rlwinm(comp2,5,0,26);           \
         list2 = __rlwinm(list2,27,5,31);          \
         comp2 = __rlwimi(comp2,comp3,5,27,31);    \
         list2 = __rlwimi(list2,list1,27,0,4);     \
         comp3 = __rlwinm(comp3,5,0,26);           \
         list1 = __rlwinm(list1,27,5,31);          \
         comp3 = __rlwimi(comp3,comp4,5,27,31);    \
         list1 = __rlwimi(list1,list0,27,0,4);     \
         comp4 = __rlwinm(comp4,5,0,26);           \
         list0 = __rlwinm(list0,27,5,31);          \
         list0 = __rlwimi(list0,newbit,27,0,4);    \
         break;                                    \
      case 6:                                      \
         comp0 = __rlwinm(comp0,6,0,25);           \
         list4 = __rlwinm(list4,26,6,31);          \
         comp0 = __rlwimi(comp0,comp1,6,26,31);    \
         list4 = __rlwimi(list4,list3,26,0,5);     \
         comp1 = __rlwinm(comp1,6,0,25);           \
         list3 = __rlwinm(list3,26,6,31);          \
         comp1 = __rlwimi(comp1,comp2,6,26,31);    \
         list3 = __rlwimi(list3,list2,26,0,5);     \
         comp2 = __rlwinm(comp2,6,0,25);           \
         list2 = __rlwinm(list2,26,6,31);          \
         comp2 = __rlwimi(comp2,comp3,6,26,31);    \
         list2 = __rlwimi(list2,list1,26,0,5);     \
         comp3 = __rlwinm(comp3,6,0,25);           \
         list1 = __rlwinm(list1,26,6,31);          \
         comp3 = __rlwimi(comp3,comp4,6,26,31);    \
         list1 = __rlwimi(list1,list0,26,0,5);     \
         comp4 = __rlwinm(comp4,6,0,25);           \
         list0 = __rlwinm(list0,26,6,31);          \
         list0 = __rlwimi(list0,newbit,26,0,5);    \
         break;                                    \
      case 7:                                      \
         comp0 = __rlwinm(comp0,7,0,24);           \
         list4 = __rlwinm(list4,25,7,31);          \
         comp0 = __rlwimi(comp0,comp1,7,25,31);    \
         list4 = __rlwimi(list4,list3,25,0,6);     \
         comp1 = __rlwinm(comp1,7,0,24);           \
         list3 = __rlwinm(list3,25,7,31);          \
         comp1 = __rlwimi(comp1,comp2,7,25,31);    \
         list3 = __rlwimi(list3,list2,25,0,6);     \
         comp2 = __rlwinm(comp2,7,0,24);           \
         list2 = __rlwinm(list2,25,7,31);          \
         comp2 = __rlwimi(comp2,comp3,7,25,31);    \
         list2 = __rlwimi(list2,list1,25,0,6);     \
         comp3 = __rlwinm(comp3,7,0,24);           \
         list1 = __rlwinm(list1,25,7,31);          \
         comp3 = __rlwimi(comp3,comp4,7,25,31);    \
         list1 = __rlwimi(list1,list0,25,0,6);     \
         comp4 = __rlwinm(comp4,7,0,24);           \
         list0 = __rlwinm(list0,25,7,31);          \
         list0 = __rlwimi(list0,newbit,25,0,6);    \
         break;                                    \
      case 8:                                      \
         comp0 = __rlwinm(comp0,8,0,23);           \
         list4 = __rlwinm(list4,24,8,31);          \
         comp0 = __rlwimi(comp0,comp1,8,24,31);    \
         list4 = __rlwimi(list4,list3,24,0,7);     \
         comp1 = __rlwinm(comp1,8,0,23);           \
         list3 = __rlwinm(list3,24,8,31);          \
         comp1 = __rlwimi(comp1,comp2,8,24,31);    \
         list3 = __rlwimi(list3,list2,24,0,7);     \
         comp2 = __rlwinm(comp2,8,0,23);           \
         list2 = __rlwinm(list2,24,8,31);          \
         comp2 = __rlwimi(comp2,comp3,8,24,31);    \
         list2 = __rlwimi(list2,list1,24,0,7);     \
         comp3 = __rlwinm(comp3,8,0,23);           \
         list1 = __rlwinm(list1,24,8,31);          \
         comp3 = __rlwimi(comp3,comp4,8,24,31);    \
         list1 = __rlwimi(list1,list0,24,0,7);     \
         comp4 = __rlwinm(comp4,8,0,23);           \
         list0 = __rlwinm(list0,24,8,31);          \
         list0 = __rlwimi(list0,newbit,24,0,7);    \
         break;                                    \
      case 9:                                      \
         comp0 = __rlwinm(comp0,9,0,22);           \
         list4 = __rlwinm(list4,23,9,31);          \
         comp0 = __rlwimi(comp0,comp1,9,23,31);    \
         list4 = __rlwimi(list4,list3,23,0,8);     \
         comp1 = __rlwinm(comp1,9,0,22);           \
         list3 = __rlwinm(list3,23,9,31);          \
         comp1 = __rlwimi(comp1,comp2,9,23,31);    \
         list3 = __rlwimi(list3,list2,23,0,8);     \
         comp2 = __rlwinm(comp2,9,0,22);           \
         list2 = __rlwinm(list2,23,9,31);          \
         comp2 = __rlwimi(comp2,comp3,9,23,31);    \
         list2 = __rlwimi(list2,list1,23,0,8);     \
         comp3 = __rlwinm(comp3,9,0,22);           \
         list1 = __rlwinm(list1,23,9,31);          \
         comp3 = __rlwimi(comp3,comp4,9,23,31);    \
         list1 = __rlwimi(list1,list0,23,0,8);     \
         comp4 = __rlwinm(comp4,9,0,22);           \
         list0 = __rlwinm(list0,23,9,31);          \
         list0 = __rlwimi(list0,newbit,23,0,8);    \
         break;                                    \
      case 10:                                     \
         comp0 = __rlwinm(comp0,10,0,21);          \
         list4 = __rlwinm(list4,22,10,31);         \
         comp0 = __rlwimi(comp0,comp1,10,22,31);   \
         list4 = __rlwimi(list4,list3,22,0,9);     \
         comp1 = __rlwinm(comp1,10,0,21);          \
         list3 = __rlwinm(list3,22,10,31);         \
         comp1 = __rlwimi(comp1,comp2,10,22,31);   \
         list3 = __rlwimi(list3,list2,22,0,9);     \
         comp2 = __rlwinm(comp2,10,0,21);          \
         list2 = __rlwinm(list2,22,10,31);         \
         comp2 = __rlwimi(comp2,comp3,10,22,31);   \
         list2 = __rlwimi(list2,list1,22,0,9);     \
         comp3 = __rlwinm(comp3,10,0,21);          \
         list1 = __rlwinm(list1,22,10,31);         \
         comp3 = __rlwimi(comp3,comp4,10,22,31);   \
         list1 = __rlwimi(list1,list0,22,0,9);     \
         comp4 = __rlwinm(comp4,10,0,21);          \
         list0 = __rlwinm(list0,22,10,31);         \
         list0 = __rlwimi(list0,newbit,22,0,9);    \
         break;                                    \
      case 11:                                     \
         comp0 = __rlwinm(comp0,11,0,20);          \
         list4 = __rlwinm(list4,21,11,31);         \
         comp0 = __rlwimi(comp0,comp1,11,21,31);   \
         list4 = __rlwimi(list4,list3,21,0,10);    \
         comp1 = __rlwinm(comp1,11,0,20);          \
         list3 = __rlwinm(list3,21,11,31);         \
         comp1 = __rlwimi(comp1,comp2,11,21,31);   \
         list3 = __rlwimi(list3,list2,21,0,10);    \
         comp2 = __rlwinm(comp2,11,0,20);          \
         list2 = __rlwinm(list2,21,11,31);         \
         comp2 = __rlwimi(comp2,comp3,11,21,31);   \
         list2 = __rlwimi(list2,list1,21,0,10);    \
         comp3 = __rlwinm(comp3,11,0,20);          \
         list1 = __rlwinm(list1,21,11,31);         \
         comp3 = __rlwimi(comp3,comp4,11,21,31);   \
         list1 = __rlwimi(list1,list0,21,0,10);    \
         comp4 = __rlwinm(comp4,11,0,20);          \
         list0 = __rlwinm(list0,21,11,31);         \
         list0 = __rlwimi(list0,newbit,21,0,10);   \
         break;                                    \
      case 12:                                     \
         comp0 = __rlwinm(comp0,12,0,19);          \
         list4 = __rlwinm(list4,20,12,31);         \
         comp0 = __rlwimi(comp0,comp1,12,20,31);   \
         list4 = __rlwimi(list4,list3,20,0,11);    \
         comp1 = __rlwinm(comp1,12,0,19);          \
         list3 = __rlwinm(list3,20,12,31);         \
         comp1 = __rlwimi(comp1,comp2,12,20,31);   \
         list3 = __rlwimi(list3,list2,20,0,11);    \
         comp2 = __rlwinm(comp2,12,0,19);          \
         list2 = __rlwinm(list2,20,12,31);         \
         comp2 = __rlwimi(comp2,comp3,12,20,31);   \
         list2 = __rlwimi(list2,list1,20,0,11);    \
         comp3 = __rlwinm(comp3,12,0,19);          \
         list1 = __rlwinm(list1,20,12,31);         \
         comp3 = __rlwimi(comp3,comp4,12,20,31);   \
         list1 = __rlwimi(list1,list0,20,0,11);    \
         comp4 = __rlwinm(comp4,12,0,19);          \
         list0 = __rlwinm(list0,20,12,31);         \
         list0 = __rlwimi(list0,newbit,20,0,11);   \
         break;                                    \
      case 13:                                     \
         comp0 = __rlwinm(comp0,13,0,18);          \
         list4 = __rlwinm(list4,19,13,31);         \
         comp0 = __rlwimi(comp0,comp1,13,19,31);   \
         list4 = __rlwimi(list4,list3,19,0,12);    \
         comp1 = __rlwinm(comp1,13,0,18);          \
         list3 = __rlwinm(list3,19,13,31);         \
         comp1 = __rlwimi(comp1,comp2,13,19,31);   \
         list3 = __rlwimi(list3,list2,19,0,12);    \
         comp2 = __rlwinm(comp2,13,0,18);          \
         list2 = __rlwinm(list2,19,13,31);         \
         comp2 = __rlwimi(comp2,comp3,13,19,31);   \
         list2 = __rlwimi(list2,list1,19,0,12);    \
         comp3 = __rlwinm(comp3,13,0,18);          \
         list1 = __rlwinm(list1,19,13,31);         \
         comp3 = __rlwimi(comp3,comp4,13,19,31);   \
         list1 = __rlwimi(list1,list0,19,0,12);    \
         comp4 = __rlwinm(comp4,13,0,18);          \
         list0 = __rlwinm(list0,19,13,31);         \
         list0 = __rlwimi(list0,newbit,19,0,12);   \
         break;                                    \
      case 14:                                     \
         comp0 = __rlwinm(comp0,14,0,17);          \
         list4 = __rlwinm(list4,18,14,31);         \
         comp0 = __rlwimi(comp0,comp1,14,18,31);   \
         list4 = __rlwimi(list4,list3,18,0,13);    \
         comp1 = __rlwinm(comp1,14,0,17);          \
         list3 = __rlwinm(list3,18,14,31);         \
         comp1 = __rlwimi(comp1,comp2,14,18,31);   \
         list3 = __rlwimi(list3,list2,18,0,13);    \
         comp2 = __rlwinm(comp2,14,0,17);          \
         list2 = __rlwinm(list2,18,14,31);         \
         comp2 = __rlwimi(comp2,comp3,14,18,31);   \
         list2 = __rlwimi(list2,list1,18,0,13);    \
         comp3 = __rlwinm(comp3,14,0,17);          \
         list1 = __rlwinm(list1,18,14,31);         \
         comp3 = __rlwimi(comp3,comp4,14,18,31);   \
         list1 = __rlwimi(list1,list0,18,0,13);    \
         comp4 = __rlwinm(comp4,14,0,17);          \
         list0 = __rlwinm(list0,18,14,31);         \
         list0 = __rlwimi(list0,newbit,18,0,13);   \
         break;                                    \
      case 15:                                     \
         comp0 = __rlwinm(comp0,15,0,16);          \
         list4 = __rlwinm(list4,17,15,31);         \
         comp0 = __rlwimi(comp0,comp1,15,17,31);   \
         list4 = __rlwimi(list4,list3,17,0,14);    \
         comp1 = __rlwinm(comp1,15,0,16);          \
         list3 = __rlwinm(list3,17,15,31);         \
         comp1 = __rlwimi(comp1,comp2,15,17,31);   \
         list3 = __rlwimi(list3,list2,17,0,14);    \
         comp2 = __rlwinm(comp2,15,0,16);          \
         list2 = __rlwinm(list2,17,15,31);         \
         comp2 = __rlwimi(comp2,comp3,15,17,31);   \
         list2 = __rlwimi(list2,list1,17,0,14);    \
         comp3 = __rlwinm(comp3,15,0,16);          \
         list1 = __rlwinm(list1,17,15,31);         \
         comp3 = __rlwimi(comp3,comp4,15,17,31);   \
         list1 = __rlwimi(list1,list0,17,0,14);    \
         comp4 = __rlwinm(comp4,15,0,16);          \
         list0 = __rlwinm(list0,17,15,31);         \
         list0 = __rlwimi(list0,newbit,17,0,14);   \
         break;                                    \
      case 16:                                     \
         comp0 = __rlwinm(comp0,16,0,15);          \
         list4 = __rlwinm(list4,16,16,31);         \
         comp0 = __rlwimi(comp0,comp1,16,16,31);   \
         list4 = __rlwimi(list4,list3,16,0,15);    \
         comp1 = __rlwinm(comp1,16,0,15);          \
         list3 = __rlwinm(list3,16,16,31);         \
         comp1 = __rlwimi(comp1,comp2,16,16,31);   \
         list3 = __rlwimi(list3,list2,16,0,15);    \
         comp2 = __rlwinm(comp2,16,0,15);          \
         list2 = __rlwinm(list2,16,16,31);         \
         comp2 = __rlwimi(comp2,comp3,16,16,31);   \
         list2 = __rlwimi(list2,list1,16,0,15);    \
         comp3 = __rlwinm(comp3,16,0,15);          \
         list1 = __rlwinm(list1,16,16,31);         \
         comp3 = __rlwimi(comp3,comp4,16,16,31);   \
         list1 = __rlwimi(list1,list0,16,0,15);    \
         comp4 = __rlwinm(comp4,16,0,15);          \
         list0 = __rlwinm(list0,16,16,31);         \
         list0 = __rlwimi(list0,newbit,16,0,15);   \
         break;                                    \
      case 17:                                     \
         comp0 = __rlwinm(comp0,17,0,14);          \
         list4 = __rlwinm(list4,15,17,31);         \
         comp0 = __rlwimi(comp0,comp1,17,15,31);   \
         list4 = __rlwimi(list4,list3,15,0,16);    \
         comp1 = __rlwinm(comp1,17,0,14);          \
         list3 = __rlwinm(list3,15,17,31);         \
         comp1 = __rlwimi(comp1,comp2,17,15,31);   \
         list3 = __rlwimi(list3,list2,15,0,16);    \
         comp2 = __rlwinm(comp2,17,0,14);          \
         list2 = __rlwinm(list2,15,17,31);         \
         comp2 = __rlwimi(comp2,comp3,17,15,31);   \
         list2 = __rlwimi(list2,list1,15,0,16);    \
         comp3 = __rlwinm(comp3,17,0,14);          \
         list1 = __rlwinm(list1,15,17,31);         \
         comp3 = __rlwimi(comp3,comp4,17,15,31);   \
         list1 = __rlwimi(list1,list0,15,0,16);    \
         comp4 = __rlwinm(comp4,17,0,14);          \
         list0 = __rlwinm(list0,15,17,31);         \
         list0 = __rlwimi(list0,newbit,15,0,16);   \
         break;                                    \
      case 18:                                     \
         comp0 = __rlwinm(comp0,18,0,13);          \
         list4 = __rlwinm(list4,14,18,31);         \
         comp0 = __rlwimi(comp0,comp1,18,14,31);   \
         list4 = __rlwimi(list4,list3,14,0,17);    \
         comp1 = __rlwinm(comp1,18,0,13);          \
         list3 = __rlwinm(list3,14,18,31);         \
         comp1 = __rlwimi(comp1,comp2,18,14,31);   \
         list3 = __rlwimi(list3,list2,14,0,17);    \
         comp2 = __rlwinm(comp2,18,0,13);          \
         list2 = __rlwinm(list2,14,18,31);         \
         comp2 = __rlwimi(comp2,comp3,18,14,31);   \
         list2 = __rlwimi(list2,list1,14,0,17);    \
         comp3 = __rlwinm(comp3,18,0,13);          \
         list1 = __rlwinm(list1,14,18,31);         \
         comp3 = __rlwimi(comp3,comp4,18,14,31);   \
         list1 = __rlwimi(list1,list0,14,0,17);    \
         comp4 = __rlwinm(comp4,18,0,13);          \
         list0 = __rlwinm(list0,14,18,31);         \
         list0 = __rlwimi(list0,newbit,14,0,17);   \
         break;                                    \
      case 19:                                     \
         comp0 = __rlwinm(comp0,19,0,12);          \
         list4 = __rlwinm(list4,13,19,31);         \
         comp0 = __rlwimi(comp0,comp1,19,13,31);   \
         list4 = __rlwimi(list4,list3,13,0,18);    \
         comp1 = __rlwinm(comp1,19,0,12);          \
         list3 = __rlwinm(list3,13,19,31);         \
         comp1 = __rlwimi(comp1,comp2,19,13,31);   \
         list3 = __rlwimi(list3,list2,13,0,18);    \
         comp2 = __rlwinm(comp2,19,0,12);          \
         list2 = __rlwinm(list2,13,19,31);         \
         comp2 = __rlwimi(comp2,comp3,19,13,31);   \
         list2 = __rlwimi(list2,list1,13,0,18);    \
         comp3 = __rlwinm(comp3,19,0,12);          \
         list1 = __rlwinm(list1,13,19,31);         \
         comp3 = __rlwimi(comp3,comp4,19,13,31);   \
         list1 = __rlwimi(list1,list0,13,0,18);    \
         comp4 = __rlwinm(comp4,19,0,12);          \
         list0 = __rlwinm(list0,13,19,31);         \
         list0 = __rlwimi(list0,newbit,13,0,18);   \
         break;                                    \
      case 20:                                     \
         comp0 = __rlwinm(comp0,20,0,11);          \
         list4 = __rlwinm(list4,12,20,31);         \
         comp0 = __rlwimi(comp0,comp1,20,12,31);   \
         list4 = __rlwimi(list4,list3,12,0,19);    \
         comp1 = __rlwinm(comp1,20,0,11);          \
         list3 = __rlwinm(list3,12,20,31);         \
         comp1 = __rlwimi(comp1,comp2,20,12,31);   \
         list3 = __rlwimi(list3,list2,12,0,19);    \
         comp2 = __rlwinm(comp2,20,0,11);          \
         list2 = __rlwinm(list2,12,20,31);         \
         comp2 = __rlwimi(comp2,comp3,20,12,31);   \
         list2 = __rlwimi(list2,list1,12,0,19);    \
         comp3 = __rlwinm(comp3,20,0,11);          \
         list1 = __rlwinm(list1,12,20,31);         \
         comp3 = __rlwimi(comp3,comp4,20,12,31);   \
         list1 = __rlwimi(list1,list0,12,0,19);    \
         comp4 = __rlwinm(comp4,20,0,11);          \
         list0 = __rlwinm(list0,12,20,31);         \
         list0 = __rlwimi(list0,newbit,12,0,19);   \
         break;                                    \
      case 21:                                     \
         comp0 = __rlwinm(comp0,21,0,10);          \
         list4 = __rlwinm(list4,11,21,31);         \
         comp0 = __rlwimi(comp0,comp1,21,11,31);   \
         list4 = __rlwimi(list4,list3,11,0,20);    \
         comp1 = __rlwinm(comp1,21,0,10);          \
         list3 = __rlwinm(list3,11,21,31);         \
         comp1 = __rlwimi(comp1,comp2,21,11,31);   \
         list3 = __rlwimi(list3,list2,11,0,20);    \
         comp2 = __rlwinm(comp2,21,0,10);          \
         list2 = __rlwinm(list2,11,21,31);         \
         comp2 = __rlwimi(comp2,comp3,21,11,31);   \
         list2 = __rlwimi(list2,list1,11,0,20);    \
         comp3 = __rlwinm(comp3,21,0,10);          \
         list1 = __rlwinm(list1,11,21,31);         \
         comp3 = __rlwimi(comp3,comp4,21,11,31);   \
         list1 = __rlwimi(list1,list0,11,0,20);    \
         comp4 = __rlwinm(comp4,21,0,10);          \
         list0 = __rlwinm(list0,11,21,31);         \
         list0 = __rlwimi(list0,newbit,11,0,20);   \
         break;                                    \
      case 22:                                     \
         comp0 = __rlwinm(comp0,22,0,9);           \
         list4 = __rlwinm(list4,10,22,31);         \
         comp0 = __rlwimi(comp0,comp1,22,10,31);   \
         list4 = __rlwimi(list4,list3,10,0,21);    \
         comp1 = __rlwinm(comp1,22,0,9);           \
         list3 = __rlwinm(list3,10,22,31);         \
         comp1 = __rlwimi(comp1,comp2,22,10,31);   \
         list3 = __rlwimi(list3,list2,10,0,21);    \
         comp2 = __rlwinm(comp2,22,0,9);           \
         list2 = __rlwinm(list2,10,22,31);         \
         comp2 = __rlwimi(comp2,comp3,22,10,31);   \
         list2 = __rlwimi(list2,list1,10,0,21);    \
         comp3 = __rlwinm(comp3,22,0,9);           \
         list1 = __rlwinm(list1,10,22,31);         \
         comp3 = __rlwimi(comp3,comp4,22,10,31);   \
         list1 = __rlwimi(list1,list0,10,0,21);    \
         comp4 = __rlwinm(comp4,22,0,9);           \
         list0 = __rlwinm(list0,10,22,31);         \
         list0 = __rlwimi(list0,newbit,10,0,21);   \
         break;                                    \
      case 23:                                     \
         comp0 = __rlwinm(comp0,23,0,8);           \
         list4 = __rlwinm(list4,9,23,31);          \
         comp0 = __rlwimi(comp0,comp1,23,9,31);    \
         list4 = __rlwimi(list4,list3,9,0,22);     \
         comp1 = __rlwinm(comp1,23,0,8);           \
         list3 = __rlwinm(list3,9,23,31);          \
         comp1 = __rlwimi(comp1,comp2,23,9,31);    \
         list3 = __rlwimi(list3,list2,9,0,22);     \
         comp2 = __rlwinm(comp2,23,0,8);           \
         list2 = __rlwinm(list2,9,23,31);          \
         comp2 = __rlwimi(comp2,comp3,23,9,31);    \
         list2 = __rlwimi(list2,list1,9,0,22);     \
         comp3 = __rlwinm(comp3,23,0,8);           \
         list1 = __rlwinm(list1,9,23,31);          \
         comp3 = __rlwimi(comp3,comp4,23,9,31);    \
         list1 = __rlwimi(list1,list0,9,0,22);     \
         comp4 = __rlwinm(comp4,23,0,8);           \
         list0 = __rlwinm(list0,9,23,31);          \
         list0 = __rlwimi(list0,newbit,9,0,22);    \
         break;                                    \
      case 24:                                     \
         comp0 = __rlwinm(comp0,24,0,7);           \
         list4 = __rlwinm(list4,8,24,31);          \
         comp0 = __rlwimi(comp0,comp1,24,8,31);    \
         list4 = __rlwimi(list4,list3,8,0,23);     \
         comp1 = __rlwinm(comp1,24,0,7);           \
         list3 = __rlwinm(list3,8,24,31);          \
         comp1 = __rlwimi(comp1,comp2,24,8,31);    \
         list3 = __rlwimi(list3,list2,8,0,23);     \
         comp2 = __rlwinm(comp2,24,0,7);           \
         list2 = __rlwinm(list2,8,24,31);          \
         comp2 = __rlwimi(comp2,comp3,24,8,31);    \
         list2 = __rlwimi(list2,list1,8,0,23);     \
         comp3 = __rlwinm(comp3,24,0,7);           \
         list1 = __rlwinm(list1,8,24,31);          \
         comp3 = __rlwimi(comp3,comp4,24,8,31);    \
         list1 = __rlwimi(list1,list0,8,0,23);     \
         comp4 = __rlwinm(comp4,24,0,7);           \
         list0 = __rlwinm(list0,8,24,31);          \
         list0 = __rlwimi(list0,newbit,8,0,23);    \
         break;                                    \
      case 25:                                     \
         comp0 = __rlwinm(comp0,25,0,6);           \
         list4 = __rlwinm(list4,7,25,31);          \
         comp0 = __rlwimi(comp0,comp1,25,7,31);    \
         list4 = __rlwimi(list4,list3,7,0,24);     \
         comp1 = __rlwinm(comp1,25,0,6);           \
         list3 = __rlwinm(list3,7,25,31);          \
         comp1 = __rlwimi(comp1,comp2,25,7,31);    \
         list3 = __rlwimi(list3,list2,7,0,24);     \
         comp2 = __rlwinm(comp2,25,0,6);           \
         list2 = __rlwinm(list2,7,25,31);          \
         comp2 = __rlwimi(comp2,comp3,25,7,31);    \
         list2 = __rlwimi(list2,list1,7,0,24);     \
         comp3 = __rlwinm(comp3,25,0,6);           \
         list1 = __rlwinm(list1,7,25,31);          \
         comp3 = __rlwimi(comp3,comp4,25,7,31);    \
         list1 = __rlwimi(list1,list0,7,0,24);     \
         comp4 = __rlwinm(comp4,25,0,6);           \
         list0 = __rlwinm(list0,7,25,31);          \
         list0 = __rlwimi(list0,newbit,7,0,24);    \
         break;                                    \
      case 26:                                     \
         comp0 = __rlwinm(comp0,26,0,5);           \
         list4 = __rlwinm(list4,6,26,31);          \
         comp0 = __rlwimi(comp0,comp1,26,6,31);    \
         list4 = __rlwimi(list4,list3,6,0,25);     \
         comp1 = __rlwinm(comp1,26,0,5);           \
         list3 = __rlwinm(list3,6,26,31);          \
         comp1 = __rlwimi(comp1,comp2,26,6,31);    \
         list3 = __rlwimi(list3,list2,6,0,25);     \
         comp2 = __rlwinm(comp2,26,0,5);           \
         list2 = __rlwinm(list2,6,26,31);          \
         comp2 = __rlwimi(comp2,comp3,26,6,31);    \
         list2 = __rlwimi(list2,list1,6,0,25);     \
         comp3 = __rlwinm(comp3,26,0,5);           \
         list1 = __rlwinm(list1,6,26,31);          \
         comp3 = __rlwimi(comp3,comp4,26,6,31);    \
         list1 = __rlwimi(list1,list0,6,0,25);     \
         comp4 = __rlwinm(comp4,26,0,5);           \
         list0 = __rlwinm(list0,6,26,31);          \
         list0 = __rlwimi(list0,newbit,6,0,25);    \
         break;                                    \
      case 27:                                     \
         comp0 = __rlwinm(comp0,27,0,4);           \
         list4 = __rlwinm(list4,5,27,31);          \
         comp0 = __rlwimi(comp0,comp1,27,5,31);    \
         list4 = __rlwimi(list4,list3,5,0,26);     \
         comp1 = __rlwinm(comp1,27,0,4);           \
         list3 = __rlwinm(list3,5,27,31);          \
         comp1 = __rlwimi(comp1,comp2,27,5,31);    \
         list3 = __rlwimi(list3,list2,5,0,26);     \
         comp2 = __rlwinm(comp2,27,0,4);           \
         list2 = __rlwinm(list2,5,27,31);          \
         comp2 = __rlwimi(comp2,comp3,27,5,31);    \
         list2 = __rlwimi(list2,list1,5,0,26);     \
         comp3 = __rlwinm(comp3,27,0,4);           \
         list1 = __rlwinm(list1,5,27,31);          \
         comp3 = __rlwimi(comp3,comp4,27,5,31);    \
         list1 = __rlwimi(list1,list0,5,0,26);     \
         comp4 = __rlwinm(comp4,27,0,4);           \
         list0 = __rlwinm(list0,5,27,31);          \
         list0 = __rlwimi(list0,newbit,5,0,26);    \
         break;                                    \
      case 28:                                     \
         comp0 = __rlwinm(comp0,28,0,3);           \
         list4 = __rlwinm(list4,4,28,31);          \
         comp0 = __rlwimi(comp0,comp1,28,4,31);    \
         list4 = __rlwimi(list4,list3,4,0,27);     \
         comp1 = __rlwinm(comp1,28,0,3);           \
         list3 = __rlwinm(list3,4,28,31);          \
         comp1 = __rlwimi(comp1,comp2,28,4,31);    \
         list3 = __rlwimi(list3,list2,4,0,27);     \
         comp2 = __rlwinm(comp2,28,0,3);           \
         list2 = __rlwinm(list2,4,28,31);          \
         comp2 = __rlwimi(comp2,comp3,28,4,31);    \
         list2 = __rlwimi(list2,list1,4,0,27);     \
         comp3 = __rlwinm(comp3,28,0,3);           \
         list1 = __rlwinm(list1,4,28,31);          \
         comp3 = __rlwimi(comp3,comp4,28,4,31);    \
         list1 = __rlwimi(list1,list0,4,0,27);     \
         comp4 = __rlwinm(comp4,28,0,3);           \
         list0 = __rlwinm(list0,4,28,31);          \
         list0 = __rlwimi(list0,newbit,4,0,27);    \
         break;                                    \
      case 29:                                     \
         comp0 = __rlwinm(comp0,29,0,2);           \
         list4 = __rlwinm(list4,3,29,31);          \
         comp0 = __rlwimi(comp0,comp1,29,3,31);    \
         list4 = __rlwimi(list4,list3,3,0,28);     \
         comp1 = __rlwinm(comp1,29,0,2);           \
         list3 = __rlwinm(list3,3,29,31);          \
         comp1 = __rlwimi(comp1,comp2,29,3,31);    \
         list3 = __rlwimi(list3,list2,3,0,28);     \
         comp2 = __rlwinm(comp2,29,0,2);           \
         list2 = __rlwinm(list2,3,29,31);          \
         comp2 = __rlwimi(comp2,comp3,29,3,31);    \
         list2 = __rlwimi(list2,list1,3,0,28);     \
         comp3 = __rlwinm(comp3,29,0,2);           \
         list1 = __rlwinm(list1,3,29,31);          \
         comp3 = __rlwimi(comp3,comp4,29,3,31);    \
         list1 = __rlwimi(list1,list0,3,0,28);     \
         comp4 = __rlwinm(comp4,29,0,2);           \
         list0 = __rlwinm(list0,3,29,31);          \
         list0 = __rlwimi(list0,newbit,3,0,28);    \
         break;                                    \
      case 30:                                     \
         comp0 = __rlwinm(comp0,30,0,1);           \
         list4 = __rlwinm(list4,2,30,31);          \
         comp0 = __rlwimi(comp0,comp1,30,2,31);    \
         list4 = __rlwimi(list4,list3,2,0,29);     \
         comp1 = __rlwinm(comp1,30,0,1);           \
         list3 = __rlwinm(list3,2,30,31);          \
         comp1 = __rlwimi(comp1,comp2,30,2,31);    \
         list3 = __rlwimi(list3,list2,2,0,29);     \
         comp2 = __rlwinm(comp2,30,0,1);           \
         list2 = __rlwinm(list2,2,30,31);          \
         comp2 = __rlwimi(comp2,comp3,30,2,31);    \
         list2 = __rlwimi(list2,list1,2,0,29);     \
         comp3 = __rlwinm(comp3,30,0,1);           \
         list1 = __rlwinm(list1,2,30,31);          \
         comp3 = __rlwimi(comp3,comp4,30,2,31);    \
         list1 = __rlwimi(list1,list0,2,0,29);     \
         comp4 = __rlwinm(comp4,30,0,1);           \
         list0 = __rlwinm(list0,2,30,31);          \
         list0 = __rlwimi(list0,newbit,2,0,29);    \
         break;                                    \
      case 31:                                     \
         comp0 = __rlwinm(comp0,31,0,0);           \
         list4 = __rlwinm(list4,1,31,31);          \
         comp0 = __rlwimi(comp0,comp1,31,1,31);    \
         list4 = __rlwimi(list4,list3,1,0,30);     \
         comp1 = __rlwinm(comp1,31,0,0);           \
         list3 = __rlwinm(list3,1,31,31);          \
         comp1 = __rlwimi(comp1,comp2,31,1,31);    \
         list3 = __rlwimi(list3,list2,1,0,30);     \
         comp2 = __rlwinm(comp2,31,0,0);           \
         list2 = __rlwinm(list2,1,31,31);          \
         comp2 = __rlwimi(comp2,comp3,31,1,31);    \
         list2 = __rlwimi(list2,list1,1,0,30);     \
         comp3 = __rlwinm(comp3,31,0,0);           \
         list1 = __rlwinm(list1,1,31,31);          \
         comp3 = __rlwimi(comp3,comp4,31,1,31);    \
         list1 = __rlwimi(list1,list0,1,0,30);     \
         comp4 = __rlwinm(comp4,31,0,0);           \
         list0 = __rlwinm(list0,1,31,31);          \
         list0 = __rlwimi(list0,newbit,1,0,30);    \
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
#define COMP_LEFT_LIST_RIGHT(lev, s) {             \
   switch (s)                                      \
      {                                            \
      case 0:                                      \
         comp0 = (comp0 << 0) | (comp1 >> 32);     \
         list4 = (list4 >> 0) | (list3 << 32);     \
         comp1 = (comp1 << 0) | (comp2 >> 32);     \
         list3 = (list3 >> 0) | (list2 << 32);     \
         comp2 = (comp2 << 0) | (comp3 >> 32);     \
         list2 = (list2 >> 0) | (list1 << 32);     \
         comp3 = (comp3 << 0) | (comp4 >> 32);     \
         list1 = (list1 >> 0) | (list0 << 32);     \
         list0 = (list0 >> 0) | (newbit << 32);    \
         comp4 = comp4 << 0;                       \
         break;                                    \
      case 1:                                      \
         comp0 = (comp0 << 1) | (comp1 >> 31);     \
         list4 = (list4 >> 1) | (list3 << 31);     \
         comp1 = (comp1 << 1) | (comp2 >> 31);     \
         list3 = (list3 >> 1) | (list2 << 31);     \
         comp2 = (comp2 << 1) | (comp3 >> 31);     \
         list2 = (list2 >> 1) | (list1 << 31);     \
         comp3 = (comp3 << 1) | (comp4 >> 31);     \
         list1 = (list1 >> 1) | (list0 << 31);     \
         list0 = (list0 >> 1) | (newbit << 31);    \
         comp4 = comp4 << 1;                       \
         break;                                    \
      case 2:                                      \
         comp0 = (comp0 << 2) | (comp1 >> 30);     \
         list4 = (list4 >> 2) | (list3 << 30);     \
         comp1 = (comp1 << 2) | (comp2 >> 30);     \
         list3 = (list3 >> 2) | (list2 << 30);     \
         comp2 = (comp2 << 2) | (comp3 >> 30);     \
         list2 = (list2 >> 2) | (list1 << 30);     \
         comp3 = (comp3 << 2) | (comp4 >> 30);     \
         list1 = (list1 >> 2) | (list0 << 30);     \
         list0 = (list0 >> 2) | (newbit << 30);    \
         comp4 = comp4 << 2;                       \
         break;                                    \
      case 3:                                      \
         comp0 = (comp0 << 3) | (comp1 >> 29);     \
         list4 = (list4 >> 3) | (list3 << 29);     \
         comp1 = (comp1 << 3) | (comp2 >> 29);     \
         list3 = (list3 >> 3) | (list2 << 29);     \
         comp2 = (comp2 << 3) | (comp3 >> 29);     \
         list2 = (list2 >> 3) | (list1 << 29);     \
         comp3 = (comp3 << 3) | (comp4 >> 29);     \
         list1 = (list1 >> 3) | (list0 << 29);     \
         list0 = (list0 >> 3) | (newbit << 29);    \
         comp4 = comp4 << 3;                       \
         break;                                    \
      case 4:                                      \
         comp0 = (comp0 << 4) | (comp1 >> 28);     \
         list4 = (list4 >> 4) | (list3 << 28);     \
         comp1 = (comp1 << 4) | (comp2 >> 28);     \
         list3 = (list3 >> 4) | (list2 << 28);     \
         comp2 = (comp2 << 4) | (comp3 >> 28);     \
         list2 = (list2 >> 4) | (list1 << 28);     \
         comp3 = (comp3 << 4) | (comp4 >> 28);     \
         list1 = (list1 >> 4) | (list0 << 28);     \
         list0 = (list0 >> 4) | (newbit << 28);    \
         comp4 = comp4 << 4;                       \
         break;                                    \
      case 5:                                      \
         comp0 = (comp0 << 5) | (comp1 >> 27);     \
         list4 = (list4 >> 5) | (list3 << 27);     \
         comp1 = (comp1 << 5) | (comp2 >> 27);     \
         list3 = (list3 >> 5) | (list2 << 27);     \
         comp2 = (comp2 << 5) | (comp3 >> 27);     \
         list2 = (list2 >> 5) | (list1 << 27);     \
         comp3 = (comp3 << 5) | (comp4 >> 27);     \
         list1 = (list1 >> 5) | (list0 << 27);     \
         list0 = (list0 >> 5) | (newbit << 27);    \
         comp4 = comp4 << 5;                       \
         break;                                    \
      case 6:                                      \
         comp0 = (comp0 << 6) | (comp1 >> 26);     \
         list4 = (list4 >> 6) | (list3 << 26);     \
         comp1 = (comp1 << 6) | (comp2 >> 26);     \
         list3 = (list3 >> 6) | (list2 << 26);     \
         comp2 = (comp2 << 6) | (comp3 >> 26);     \
         list2 = (list2 >> 6) | (list1 << 26);     \
         comp3 = (comp3 << 6) | (comp4 >> 26);     \
         list1 = (list1 >> 6) | (list0 << 26);     \
         list0 = (list0 >> 6) | (newbit << 26);    \
         comp4 = comp4 << 6;                       \
         break;                                    \
      case 7:                                      \
         comp0 = (comp0 << 7) | (comp1 >> 25);     \
         list4 = (list4 >> 7) | (list3 << 25);     \
         comp1 = (comp1 << 7) | (comp2 >> 25);     \
         list3 = (list3 >> 7) | (list2 << 25);     \
         comp2 = (comp2 << 7) | (comp3 >> 25);     \
         list2 = (list2 >> 7) | (list1 << 25);     \
         comp3 = (comp3 << 7) | (comp4 >> 25);     \
         list1 = (list1 >> 7) | (list0 << 25);     \
         list0 = (list0 >> 7) | (newbit << 25);    \
         comp4 = comp4 << 7;                       \
         break;                                    \
      case 8:                                      \
         comp0 = (comp0 << 8) | (comp1 >> 24);     \
         list4 = (list4 >> 8) | (list3 << 24);     \
         comp1 = (comp1 << 8) | (comp2 >> 24);     \
         list3 = (list3 >> 8) | (list2 << 24);     \
         comp2 = (comp2 << 8) | (comp3 >> 24);     \
         list2 = (list2 >> 8) | (list1 << 24);     \
         comp3 = (comp3 << 8) | (comp4 >> 24);     \
         list1 = (list1 >> 8) | (list0 << 24);     \
         list0 = (list0 >> 8) | (newbit << 24);    \
         comp4 = comp4 << 8;                       \
         break;                                    \
      case 9:                                      \
         comp0 = (comp0 << 9) | (comp1 >> 23);     \
         list4 = (list4 >> 9) | (list3 << 23);     \
         comp1 = (comp1 << 9) | (comp2 >> 23);     \
         list3 = (list3 >> 9) | (list2 << 23);     \
         comp2 = (comp2 << 9) | (comp3 >> 23);     \
         list2 = (list2 >> 9) | (list1 << 23);     \
         comp3 = (comp3 << 9) | (comp4 >> 23);     \
         list1 = (list1 >> 9) | (list0 << 23);     \
         list0 = (list0 >> 9) | (newbit << 23);    \
         comp4 = comp4 << 9;                       \
         break;                                    \
      case 10:                                     \
         comp0 = (comp0 << 10) | (comp1 >> 22);    \
         list4 = (list4 >> 10) | (list3 << 22);    \
         comp1 = (comp1 << 10) | (comp2 >> 22);    \
         list3 = (list3 >> 10) | (list2 << 22);    \
         comp2 = (comp2 << 10) | (comp3 >> 22);    \
         list2 = (list2 >> 10) | (list1 << 22);    \
         comp3 = (comp3 << 10) | (comp4 >> 22);    \
         list1 = (list1 >> 10) | (list0 << 22);    \
         list0 = (list0 >> 10) | (newbit << 22);   \
         comp4 = comp4 << 10;                      \
         break;                                    \
      case 11:                                     \
         comp0 = (comp0 << 11) | (comp1 >> 21);    \
         list4 = (list4 >> 11) | (list3 << 21);    \
         comp1 = (comp1 << 11) | (comp2 >> 21);    \
         list3 = (list3 >> 11) | (list2 << 21);    \
         comp2 = (comp2 << 11) | (comp3 >> 21);    \
         list2 = (list2 >> 11) | (list1 << 21);    \
         comp3 = (comp3 << 11) | (comp4 >> 21);    \
         list1 = (list1 >> 11) | (list0 << 21);    \
         list0 = (list0 >> 11) | (newbit << 21);   \
         comp4 = comp4 << 11;                      \
         break;                                    \
      case 12:                                     \
         comp0 = (comp0 << 12) | (comp1 >> 20);    \
         list4 = (list4 >> 12) | (list3 << 20);    \
         comp1 = (comp1 << 12) | (comp2 >> 20);    \
         list3 = (list3 >> 12) | (list2 << 20);    \
         comp2 = (comp2 << 12) | (comp3 >> 20);    \
         list2 = (list2 >> 12) | (list1 << 20);    \
         comp3 = (comp3 << 12) | (comp4 >> 20);    \
         list1 = (list1 >> 12) | (list0 << 20);    \
         list0 = (list0 >> 12) | (newbit << 20);   \
         comp4 = comp4 << 12;                      \
         break;                                    \
      case 13:                                     \
         comp0 = (comp0 << 13) | (comp1 >> 19);    \
         list4 = (list4 >> 13) | (list3 << 19);    \
         comp1 = (comp1 << 13) | (comp2 >> 19);    \
         list3 = (list3 >> 13) | (list2 << 19);    \
         comp2 = (comp2 << 13) | (comp3 >> 19);    \
         list2 = (list2 >> 13) | (list1 << 19);    \
         comp3 = (comp3 << 13) | (comp4 >> 19);    \
         list1 = (list1 >> 13) | (list0 << 19);    \
         list0 = (list0 >> 13) | (newbit << 19);   \
         comp4 = comp4 << 13;                      \
         break;                                    \
      case 14:                                     \
         comp0 = (comp0 << 14) | (comp1 >> 18);    \
         list4 = (list4 >> 14) | (list3 << 18);    \
         comp1 = (comp1 << 14) | (comp2 >> 18);    \
         list3 = (list3 >> 14) | (list2 << 18);    \
         comp2 = (comp2 << 14) | (comp3 >> 18);    \
         list2 = (list2 >> 14) | (list1 << 18);    \
         comp3 = (comp3 << 14) | (comp4 >> 18);    \
         list1 = (list1 >> 14) | (list0 << 18);    \
         list0 = (list0 >> 14) | (newbit << 18);   \
         comp4 = comp4 << 14;                      \
         break;                                    \
      case 15:                                     \
         comp0 = (comp0 << 15) | (comp1 >> 17);    \
         list4 = (list4 >> 15) | (list3 << 17);    \
         comp1 = (comp1 << 15) | (comp2 >> 17);    \
         list3 = (list3 >> 15) | (list2 << 17);    \
         comp2 = (comp2 << 15) | (comp3 >> 17);    \
         list2 = (list2 >> 15) | (list1 << 17);    \
         comp3 = (comp3 << 15) | (comp4 >> 17);    \
         list1 = (list1 >> 15) | (list0 << 17);    \
         list0 = (list0 >> 15) | (newbit << 17);   \
         comp4 = comp4 << 15;                      \
         break;                                    \
      case 16:                                     \
         comp0 = (comp0 << 16) | (comp1 >> 16);    \
         list4 = (list4 >> 16) | (list3 << 16);    \
         comp1 = (comp1 << 16) | (comp2 >> 16);    \
         list3 = (list3 >> 16) | (list2 << 16);    \
         comp2 = (comp2 << 16) | (comp3 >> 16);    \
         list2 = (list2 >> 16) | (list1 << 16);    \
         comp3 = (comp3 << 16) | (comp4 >> 16);    \
         list1 = (list1 >> 16) | (list0 << 16);    \
         list0 = (list0 >> 16) | (newbit << 16);   \
         comp4 = comp4 << 16;                      \
         break;                                    \
      case 17:                                     \
         comp0 = (comp0 << 17) | (comp1 >> 15);    \
         list4 = (list4 >> 17) | (list3 << 15);    \
         comp1 = (comp1 << 17) | (comp2 >> 15);    \
         list3 = (list3 >> 17) | (list2 << 15);    \
         comp2 = (comp2 << 17) | (comp3 >> 15);    \
         list2 = (list2 >> 17) | (list1 << 15);    \
         comp3 = (comp3 << 17) | (comp4 >> 15);    \
         list1 = (list1 >> 17) | (list0 << 15);    \
         list0 = (list0 >> 17) | (newbit << 15);   \
         comp4 = comp4 << 17;                      \
         break;                                    \
      case 18:                                     \
         comp0 = (comp0 << 18) | (comp1 >> 14);    \
         list4 = (list4 >> 18) | (list3 << 14);    \
         comp1 = (comp1 << 18) | (comp2 >> 14);    \
         list3 = (list3 >> 18) | (list2 << 14);    \
         comp2 = (comp2 << 18) | (comp3 >> 14);    \
         list2 = (list2 >> 18) | (list1 << 14);    \
         comp3 = (comp3 << 18) | (comp4 >> 14);    \
         list1 = (list1 >> 18) | (list0 << 14);    \
         list0 = (list0 >> 18) | (newbit << 14);   \
         comp4 = comp4 << 18;                      \
         break;                                    \
      case 19:                                     \
         comp0 = (comp0 << 19) | (comp1 >> 13);    \
         list4 = (list4 >> 19) | (list3 << 13);    \
         comp1 = (comp1 << 19) | (comp2 >> 13);    \
         list3 = (list3 >> 19) | (list2 << 13);    \
         comp2 = (comp2 << 19) | (comp3 >> 13);    \
         list2 = (list2 >> 19) | (list1 << 13);    \
         comp3 = (comp3 << 19) | (comp4 >> 13);    \
         list1 = (list1 >> 19) | (list0 << 13);    \
         list0 = (list0 >> 19) | (newbit << 13);   \
         comp4 = comp4 << 19;                      \
         break;                                    \
      case 20:                                     \
         comp0 = (comp0 << 20) | (comp1 >> 12);    \
         list4 = (list4 >> 20) | (list3 << 12);    \
         comp1 = (comp1 << 20) | (comp2 >> 12);    \
         list3 = (list3 >> 20) | (list2 << 12);    \
         comp2 = (comp2 << 20) | (comp3 >> 12);    \
         list2 = (list2 >> 20) | (list1 << 12);    \
         comp3 = (comp3 << 20) | (comp4 >> 12);    \
         list1 = (list1 >> 20) | (list0 << 12);    \
         list0 = (list0 >> 20) | (newbit << 12);   \
         comp4 = comp4 << 20;                      \
         break;                                    \
      case 21:                                     \
         comp0 = (comp0 << 21) | (comp1 >> 11);    \
         list4 = (list4 >> 21) | (list3 << 11);    \
         comp1 = (comp1 << 21) | (comp2 >> 11);    \
         list3 = (list3 >> 21) | (list2 << 11);    \
         comp2 = (comp2 << 21) | (comp3 >> 11);    \
         list2 = (list2 >> 21) | (list1 << 11);    \
         comp3 = (comp3 << 21) | (comp4 >> 11);    \
         list1 = (list1 >> 21) | (list0 << 11);    \
         list0 = (list0 >> 21) | (newbit << 11);   \
         comp4 = comp4 << 21;                      \
         break;                                    \
      case 22:                                     \
         comp0 = (comp0 << 22) | (comp1 >> 10);    \
         list4 = (list4 >> 22) | (list3 << 10);    \
         comp1 = (comp1 << 22) | (comp2 >> 10);    \
         list3 = (list3 >> 22) | (list2 << 10);    \
         comp2 = (comp2 << 22) | (comp3 >> 10);    \
         list2 = (list2 >> 22) | (list1 << 10);    \
         comp3 = (comp3 << 22) | (comp4 >> 10);    \
         list1 = (list1 >> 22) | (list0 << 10);    \
         list0 = (list0 >> 22) | (newbit << 10);   \
         comp4 = comp4 << 22;                      \
         break;                                    \
      case 23:                                     \
         comp0 = (comp0 << 23) | (comp1 >> 9);     \
         list4 = (list4 >> 23) | (list3 << 9);     \
         comp1 = (comp1 << 23) | (comp2 >> 9);     \
         list3 = (list3 >> 23) | (list2 << 9);     \
         comp2 = (comp2 << 23) | (comp3 >> 9);     \
         list2 = (list2 >> 23) | (list1 << 9);     \
         comp3 = (comp3 << 23) | (comp4 >> 9);     \
         list1 = (list1 >> 23) | (list0 << 9);     \
         list0 = (list0 >> 23) | (newbit << 9);    \
         comp4 = comp4 << 23;                      \
         break;                                    \
      case 24:                                     \
         comp0 = (comp0 << 24) | (comp1 >> 8);     \
         list4 = (list4 >> 24) | (list3 << 8);     \
         comp1 = (comp1 << 24) | (comp2 >> 8);     \
         list3 = (list3 >> 24) | (list2 << 8);     \
         comp2 = (comp2 << 24) | (comp3 >> 8);     \
         list2 = (list2 >> 24) | (list1 << 8);     \
         comp3 = (comp3 << 24) | (comp4 >> 8);     \
         list1 = (list1 >> 24) | (list0 << 8);     \
         list0 = (list0 >> 24) | (newbit << 8);    \
         comp4 = comp4 << 24;                      \
         break;                                    \
      case 25:                                     \
         comp0 = (comp0 << 25) | (comp1 >> 7);     \
         list4 = (list4 >> 25) | (list3 << 7);     \
         comp1 = (comp1 << 25) | (comp2 >> 7);     \
         list3 = (list3 >> 25) | (list2 << 7);     \
         comp2 = (comp2 << 25) | (comp3 >> 7);     \
         list2 = (list2 >> 25) | (list1 << 7);     \
         comp3 = (comp3 << 25) | (comp4 >> 7);     \
         list1 = (list1 >> 25) | (list0 << 7);     \
         list0 = (list0 >> 25) | (newbit << 7);    \
         comp4 = comp4 << 25;                      \
         break;                                    \
      case 26:                                     \
         comp0 = (comp0 << 26) | (comp1 >> 6);     \
         list4 = (list4 >> 26) | (list3 << 6);     \
         comp1 = (comp1 << 26) | (comp2 >> 6);     \
         list3 = (list3 >> 26) | (list2 << 6);     \
         comp2 = (comp2 << 26) | (comp3 >> 6);     \
         list2 = (list2 >> 26) | (list1 << 6);     \
         comp3 = (comp3 << 26) | (comp4 >> 6);     \
         list1 = (list1 >> 26) | (list0 << 6);     \
         list0 = (list0 >> 26) | (newbit << 6);    \
         comp4 = comp4 << 26;                      \
         break;                                    \
      case 27:                                     \
         comp0 = (comp0 << 27) | (comp1 >> 5);     \
         list4 = (list4 >> 27) | (list3 << 5);     \
         comp1 = (comp1 << 27) | (comp2 >> 5);     \
         list3 = (list3 >> 27) | (list2 << 5);     \
         comp2 = (comp2 << 27) | (comp3 >> 5);     \
         list2 = (list2 >> 27) | (list1 << 5);     \
         comp3 = (comp3 << 27) | (comp4 >> 5);     \
         list1 = (list1 >> 27) | (list0 << 5);     \
         list0 = (list0 >> 27) | (newbit << 5);    \
         comp4 = comp4 << 27;                      \
         break;                                    \
      case 28:                                     \
         comp0 = (comp0 << 28) | (comp1 >> 4);     \
         list4 = (list4 >> 28) | (list3 << 4);     \
         comp1 = (comp1 << 28) | (comp2 >> 4);     \
         list3 = (list3 >> 28) | (list2 << 4);     \
         comp2 = (comp2 << 28) | (comp3 >> 4);     \
         list2 = (list2 >> 28) | (list1 << 4);     \
         comp3 = (comp3 << 28) | (comp4 >> 4);     \
         list1 = (list1 >> 28) | (list0 << 4);     \
         list0 = (list0 >> 28) | (newbit << 4);    \
         comp4 = comp4 << 28;                      \
         break;                                    \
      case 29:                                     \
         comp0 = (comp0 << 29) | (comp1 >> 3);     \
         list4 = (list4 >> 29) | (list3 << 3);     \
         comp1 = (comp1 << 29) | (comp2 >> 3);     \
         list3 = (list3 >> 29) | (list2 << 3);     \
         comp2 = (comp2 << 29) | (comp3 >> 3);     \
         list2 = (list2 >> 29) | (list1 << 3);     \
         comp3 = (comp3 << 29) | (comp4 >> 3);     \
         list1 = (list1 >> 29) | (list0 << 3);     \
         list0 = (list0 >> 29) | (newbit << 3);    \
         comp4 = comp4 << 29;                      \
         break;                                    \
      case 30:                                     \
         comp0 = (comp0 << 30) | (comp1 >> 2);     \
         list4 = (list4 >> 30) | (list3 << 2);     \
         comp1 = (comp1 << 30) | (comp2 >> 2);     \
         list3 = (list3 >> 30) | (list2 << 2);     \
         comp2 = (comp2 << 30) | (comp3 >> 2);     \
         list2 = (list2 >> 30) | (list1 << 2);     \
         comp3 = (comp3 << 30) | (comp4 >> 2);     \
         list1 = (list1 >> 30) | (list0 << 2);     \
         list0 = (list0 >> 30) | (newbit << 2);    \
         comp4 = comp4 << 30;                      \
         break;                                    \
      case 31:                                     \
         comp0 = (comp0 << 31) | (comp1 >> 1);     \
         list4 = (list4 >> 31) | (list3 << 1);     \
         comp1 = (comp1 << 31) | (comp2 >> 1);     \
         list3 = (list3 >> 31) | (list2 << 1);     \
         comp2 = (comp2 << 31) | (comp3 >> 1);     \
         list2 = (list2 >> 31) | (list1 << 1);     \
         comp3 = (comp3 << 31) | (comp4 >> 1);     \
         list1 = (list1 >> 31) | (list0 << 1);     \
         list0 = (list0 >> 31) | (newbit << 1);    \
         comp4 = comp4 << 31;                      \
         break;                                    \
      case 32:                                     \
         comp0 = (comp0 << 32) | (comp1 >> 0);     \
         list4 = (list4 >> 32) | (list3 << 0);     \
         comp1 = (comp1 << 32) | (comp2 >> 0);     \
         list3 = (list3 >> 32) | (list2 << 0);     \
         comp2 = (comp2 << 32) | (comp3 >> 0);     \
         list2 = (list2 >> 32) | (list1 << 0);     \
         comp3 = (comp3 << 32) | (comp4 >> 0);     \
         list1 = (list1 >> 32) | (list0 << 0);     \
         list0 = (list0 >> 32) | (newbit << 0);    \
         comp4 = comp4 << 32;                      \
         break;                                    \
      }                                            \
   newbit = 0;                                     \
}
#else // OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 0
#define COMP_LEFT_LIST_RIGHT(lev, s) {             \
   int ss = 32 - s;                                \
   comp0 = (comp0 << s) | (comp1 >> ss);           \
   comp1 = (comp1 << s) | (comp2 >> ss);           \
   comp2 = (comp2 << s) | (comp3 >> ss);           \
   comp3 = (comp3 << s) | (comp4 >> ss);           \
   comp4 = comp4 << s;                             \
   list4 = (list4 >> s) | (list3 << ss);           \
   list3 = (list3 >> s) | (list2 << ss);           \
   list2 = (list2 >> s) | (list1 << ss);           \
   list1 = (list1 >> s) | (list0 << ss);           \
   list0 = (list0 >> s) | (newbit << ss);          \
   newbit = 0;                                     \
   }
#endif

/* shift by word size */
#define COMP_LEFT_LIST_RIGHT_32(lev) { \
   comp0 = comp1; comp1 = comp2; comp2 = comp3; comp3 = comp4; comp4 = 0;  \
   list4 = list3; list3 = list2; list2 = list1; list1 = list0; list0 = newbit; \
   newbit = 0; \
}
/* set the current mark and push a level to start a new mark */
#define PUSH_LEVEL_UPDATE_STATE(lev) { \
   lev->list[0] = list0; lev->list[1] = list1; lev->list[2] = list2; lev->list[3] = list3; lev->list[4] = list4;  \
   dist0 |= list0; dist1 |= list1; dist2 |= list2; dist3 |= list3; dist4 |= list4; \
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
#define POP_LEVEL(lev) { \
   limit = lev->limit; \
   list0 = lev->list[0], list1 = lev->list[1], list2 = lev->list[2], list3 = lev->list[3], list4 = lev->list[4]; \
   dist0 = dist0 & ~list0; dist1 = dist1 & ~list1; dist2 = dist2 & ~list2; dist3 = dist3 & ~list3; dist4 = dist4 & ~list4; \
   comp0 = lev->comp[0], comp1 = lev->comp[1], comp2 = lev->comp[2], comp3 = lev->comp[3], comp4 = lev->comp[4]; \
   newbit = 0; \
   cnt2 = lev->cnt2; \
}

/* save the local state variables */
#define SAVE_FINAL_STATE(state,lev) {   \
   lev->list[0] = list0; lev->list[1] = list1; lev->list[2] = list2; lev->list[3] = list3; lev->list[4] = list4;  \
   state->dist[0] = dist0; state->dist[1] = dist1; state->dist[2] = dist2; state->dist[3] = dist3; state->dist[4] = dist4; \
   lev->comp[0] = comp0; lev->comp[1] = comp1; lev->comp[2] = comp2; lev->comp[3] = comp3; lev->comp[4] = comp4; \
   lev->cnt2 = cnt2; \
}

#else /* support macros for the ogriginal ogr_cycle() routine */

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
  #define BITOFLIST(x) 0x80000000>>((x-1)&0x1f) /*0x80000000 >> ((x-1) % 32)*/
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
static int found_one(const struct State *oState)
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
#else
static int found_one(const struct State *oState)
{
   /* confirm ruler is golomb */
   int i, j;
   const int maximum = oState->max;
   const int maximum2 = maximum >> 1;     // shouldn't this be rounded up?
   const int maxdepth = oState->maxdepth;
   const struct Level *levels = &oState->Levels[0];
   char diffs[1024]; // first 64 entries will never be used!

   // always check for buffer overruns!
   if (maximum2 >= 1024)
      return CORE_E_MEMORY;

   memset( diffs, 0, maximum2 + 1 );

   for (i = 1; i < maxdepth; i++) {
      int levelICount = levels[i].cnt2;

      for (j = 0; j < i; j++) {
           int diff = levelICount - levels[j].cnt2;

         if (2*diff <= maximum) {      /* Principle 1 */

            if (diff <= 64)
               break;     /* 2 bitmaps always tracked */

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
               :"=r" (result), "=r" (temp) : "1" (input), "r" (ogr_first_blank_8bit));
      return result;
    }
  #else /* C code, no asm */
  static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int input)
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
#elif defined(__PPC__) || defined(ASM_PPC) || defined (__POWERPC__)/* CouNT Leading Zeros Word */
  #if defined(__GNUC__)
    static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i)
    { i = ~i; __asm__ ("cntlzw %0,%0" : "=r" (i) : "0" (i)); return ++i; }
  #elif (__MWERKS__) || (__MRC__)
    #define LOOKUP_FIRSTBLANK(x) (__cntlzw(~((unsigned int)(x)))+1)
  #else
    #error "Please check this (define OGR_TEST_FIRSTBLANK to test)"
  #endif
#elif defined(ASM_ALPHA) && defined(__GNUC__)
  #error "Please check this (define OGR_TEST_FIRSTBLANK to test)"
  static __inline__ int LOOKUP_FIRSTBLANK(register unsigned int i)
  { i = ~i; __asm__ ("cntlzw %0,%0" : "=r"(i) : "0" (i)); return i+1; }
#elif defined(ASM_X86) && defined(__GNUC__) || \
      defined(__386__) && defined(__WATCOMC__) || \
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
  { i = ~i; __asm__ ("bfffo %0,0,0,%0" : "=d" (i) : "0" (i)); return ++i; }
#else
  #error OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM is defined, and no code to match
#endif



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

  #if defined(FIRSTBLANK_ASM_TEST)
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

static int ogr_create(void *input, int inputlen, void *state, int statelen)
{
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

#else

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

     #if (OGROPT_ALTERNATE_CYCLE == 2)
       U dist0 = VEC_TO_INT(distV0,3);
     #endif

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
      
      #pragma warning This does not work
//      if (s <= (32*5))
//        if (lev->dist[(s-1)>>5] & (0x80000000>>((s-1)&0x1f)))//#define BITOFLIST(x) 0x80000000>>((x-1)&0x1f)
//          return CORE_E_STUB;
      #pragma warning This does not work
      
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
#endif

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

#if (OGROPT_ALTERNATE_CYCLE == 2) || (OGROPT_ALTERNATE_CYCLE == 1)
static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
{
   struct State *oState = (struct State *)state;
   int depth = oState->depth+1;      /* the depth of recursion */
   struct Level *lev = &oState->Levels[depth];
   int nodes = 0;
   int nodeslimit = *pnodes;
    const int oStateMax = oState->max;
    const int oStateMaxDepthM1 = oState->maxdepthm1;
    const int oStateHalfDepth2 = oState->half_depth2;
    const int oStateHalfDepth = oState->half_depth;
    const int oStateHalfLength = oState->half_length;
   struct Level *levHalfDepth = &oState->Levels[oStateHalfDepth];
   struct Level *levMaxM1 = &oState->Levels[oStateMaxDepthM1];
   int retval = CORE_S_CONTINUE;

   SETUP_TOP_STATE(oState,lev);

   for (;;) {

   //continue:
      #if (OGROPT_ALTERNATE_CYCLE == 2)
         U dist0 = VEC_TO_INT(distV0,3);
      #endif

      int maxMinusDepth = oStateMaxDepthM1 - depth;

      if (with_time_constraints) { /* if (...) is optimized away if unused */
         #if !defined(OGROPT_IGNORE_TIME_CONSTRAINT_ARG)
         if (nodes >= nodeslimit) {
           break;
         }  
         #endif  
      }

      if (depth <= oStateHalfDepth2) {
         if (depth <= oStateHalfDepth) {
         
            if (nodes >= nodeslimit) {
               break;
            }
            
            limit = oStateMax - OGR[maxMinusDepth];
            limit = (limit < oStateHalfLength) ? limit : oStateHalfLength;
         } else {
            limit = oStateMax - choose(dist0 >> ttmDISTBITS, maxMinusDepth);
            int tempLimit = oStateMax - levHalfDepth->cnt2 - 1;
            limit = (limit < tempLimit) ? limit : tempLimit;
         }
      } else {
         limit = oStateMax - choose(dist0 >> ttmDISTBITS, maxMinusDepth);
      }

      nodes++;

      /* Find the next available mark location for this level */

   stay:
      #if (OGROPT_ALTERNATE_CYCLE == 2)
         U comp0 = VEC_TO_INT(compV0,3);
      #endif
      if (comp0 < 0xfffffffe) {
         int s = LOOKUP_FIRSTBLANK( comp0 );
         if ((cnt2 += s) > limit)   goto up; /* no spaces left */
         COMP_LEFT_LIST_RIGHT(lev, s);
      } else { /* s>32 */
         U comp = comp0;
         if ((cnt2 += 32) > limit)  goto up; /* no spaces left */
         COMP_LEFT_LIST_RIGHT_32(lev)
         if (comp == 0xffffffff)    goto stay;
      }

      /* New ruler? */
      if (depth == oStateMaxDepthM1) {
         levMaxM1->cnt2 = cnt2;       /* not placed yet into list arrays! */
         retval = found_one(oState);
         if (retval != CORE_S_CONTINUE) {
            break;
         }
         goto stay;
      }

      /* Go Deeper */
      PUSH_LEVEL_UPDATE_STATE(lev);
      lev++;
      depth++;
      continue;

   up:
      lev--;
      depth--;
      POP_LEVEL(lev);
      if (depth <= oState->startdepth) {
         retval = CORE_S_OK;
         break;
      }
      goto stay; /* repeat this level till done */
   }

   SAVE_FINAL_STATE(oState,lev);
   /* oState->Nodes += nodes; (unused, count is returned through *pnodes) */
   oState->depth = depth-1;

   *pnodes = nodes;

   return retval;
}
#else
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
#endif

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
#endif

static int ogr_cleanup(void)
{
  return CORE_S_OK;
}

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
