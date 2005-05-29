/*
 * Copyright distributed.net 1999-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogr.cpp,v 1.2.4.36 2005/05/29 17:02:51 stream Exp $
 */
#include <string.h>   /* memset */

/* #define OGR_DEBUG           */ /* turn on miscellaneous debug information */ 
/* #define OGR_TEST_FIRSTBLANK */ /* test firstblank logic (table or asm)    */

#if defined(OGR_DEBUG) || defined(OGR_TEST_FIRSTBLANK)
#include <stdio.h>  /* printf for debugging */
#endif

/* --- various optimization option overrides ----------------------------- */

#if defined(NO_OGR_OPTIMIZATION) || defined(GIMME_BASELINE_OGR_CPP)
  /*
  ** baseline/reference == ogr.cpp without optimization == ~old ogr.cpp 
  */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - default is hw dependant */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - default is 1 ('yes') */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - default is 0 ('no')  */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - default is 0 ('no')  */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - default is 0 ('no')  */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0-2 - default is 0 ('no')  */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - default is 0         */
#endif

/* -- various optimization option defaults ------------------------------- */

/* ogr_cycle() method selection.
   OGROPT_ALTERNATE_CYCLE == 0 - Default (GARSP) ogr_cycle().
   OGROPT_ALTERNATE_CYCLE == 1 - Tuned ogr_cycle() for register rich
                                 architectures (ppc only ?) by dan and chris
   OGROPT_ALTERNATE_CYCLE == 2 - Fully customized ogr_cycle(). Implementors must
                                 provide all necessary macros.
*/
#ifndef OGROPT_ALTERNATE_CYCLE
#define OGROPT_ALTERNATE_CYCLE 0    /* 0 (GARSP) or 1 or 2 */
#endif


/* Because CHOOSE_MARKS == 12 we can strength reduce the multiply -- which is
   slow on MANY processors -- from "12*(x)" to "((x)<<3)+((x)<<2)" in
   choose(x,y).
   Note that very smart compilers can sometimes do a better job at replacing
   the original statement with intrinsics than we can do by inserting these
   shift operations (e.g.: MrC). Thanks to Chris Cox for this optimization.
   If CHOOSE_MARKS != 12 this setting will have no effect.
*/
#ifndef OGROPT_STRENGTH_REDUCE_CHOOSE
#define OGROPT_STRENGTH_REDUCE_CHOOSE 1     /* the default is "yes" */
#endif


/* These are alternatives for the COMP_LEFT_LIST_RIGHT macro to "shift the
   list to add or extend the first mark".

   OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 0 -> default COMP_LEFT_LIST_RIGHT
   OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1 -> Implementors may provide
      platform-specific or assembly versions of the bitmap manipulation macros
      that correspond to the OGROPT_ALTERNATE_CYCLE setting. Missing macros
      are still defined to reasonable defaults.
*/
#ifndef OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT
#define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0   /* 0 (no opt) or 1 */
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
#if defined(macintosh) || defined(__riscos)         \
    || defined(__NETWARE__) || defined(NETWARE)     \
    || defined(__WINDOWS386__) /* 16bit windows */  \
    || (defined(ASM_X86) && defined(GENERATE_ASM))
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
   you'll need to supply suitable code for the __BALIGN macro.
*/
#ifndef OGROPT_CYCLE_CACHE_ALIGN
#define OGROPT_CYCLE_CACHE_ALIGN 0      /* the default is "no" */
#endif

#if (OGROPT_CYCLE_CACHE_ALIGN == 1) && defined(__BALIGN)
  #define OGR_CYCLE_CACHE_ALIGN __BALIGN
#else
  #define OGR_CYCLE_CACHE_ALIGN { }
#endif


/* Some compilers (notably GCC 3.4) may choose to inline functions, which
   may have an adverse effect on performance of the OGR core.  This option
   currently only affects the found_one() function and will declare it so
   that the compiler will never inline it.  If you turn this option on,
   you'll need to supply suitable code for the NO_FUNCTION_INLINE macro.
*/
#ifndef OGROPT_NO_FUNCTION_INLINE
#define OGROPT_NO_FUNCTION_INLINE 0     /* the default is "no" */
#endif


/* ASM cores (implementation of the ogr_cycle() function) may need to call the
   function found_one() that cannot then be declared 'static'.
   OGROPT_HAVE_OGR_CYCLE_ASM == 0 -> No ASM core.
   OGROPT_HAVE_OGR_CYCLE_ASM == 1 -> ASM core, still need found_one()
   OGROPT_HAVE_OGR_CYCLE_ASM == 2 -> ASM core + embedded found_one()
*/
#ifndef OGROPT_HAVE_OGR_CYCLE_ASM
#define OGROPT_HAVE_OGR_CYCLE_ASM 0
#endif


/* Select the implementation for the LOOKUP_FIRSTBLANK macro.
   0 -> No hardware support
   1 -> Have ASM code, but still need the lookup table. You'll have to supply
        suitable code for the __CNTLZ_ARRAY_BASED(bitmap,bitarray) macro.
   2 -> Full featured ASM code. You'll have to supply code suitable for the
        __CNTLZ(bitmap) macro.
  
  NOTE : These macros shall arrange to return the number of leading 0 of
         "~bitmap" (i.e. the number of leading 1 of the "bitmap" argument) plus
         one. Since the bitmap argument is a 32-bit value, the valid range is
         [1; 31] (or optionally [1; 32]). Said otherwise :
         __CNTLZ(0xFFFFFFFC) == 31
         __CNTLZ(0xFFFFA427) == 18
         __CNTLZ(0x00000000) ==  1

  TEST : Define OGR_TEST_FIRSTBLANK to test the code at run time.
*/
#ifndef OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 0
#endif


/* ----------------------------------------------------------------------- */

#include "ogr.h"

#if defined(__cplusplus)
extern "C" {
#endif

static const
int OGR[] = {
  /*  1 */    0,   1,   3,   6,  11,  17,  25,  34,  44,  55,
  /* 11 */   72,  85, 106, 127, 151, 177, 199, 216, 246, 283,
  /* 21 */  333, 356, 372, 425, 480, 492, 553, 585, 623
};

/*
** CHOOSEDAT optimization (OGROPT_STRENGTH_REDUCE_CHOOSE)
*/
#define CHOOSE_MARKS       12     /* maximum number of marks supported    */
#define CHOOSE_DIST_BITS   12     /* number of bits to take into account  */

extern const unsigned char ogr_choose_dat[]; /* this is in ogr_dat.cpp */

#if (CHOOSE_MARKS == 12 && OGROPT_STRENGTH_REDUCE_CHOOSE == 1)
  // strength reduce the multiply -- which is slow on MANY processors
  #define choose(x,y) (ogr_choose_dat[((x)<<3)+((x)<<2)+(y)+3])
#else
  #define choose(x,y) (ogr_choose_dat[CHOOSE_MARKS*(x)+(y)+3])
#endif

#define ttmDISTBITS (32-CHOOSE_DIST_BITS)

/* ----------------------------------------------------------------------- */

#if (OGROPT_NO_FUNCTION_INLINE == 1)
   #if defined(__GNUC__) && (__GNUC__ >= 3)
      #define NO_FUNCTION_INLINE(x) x __attribute__ ((noinline))
   #else
      #error NO_FUNCTION_INLINE is defined, and no code to match
   #endif
#else
   #define NO_FUNCTION_INLINE(x) x
#endif

#if (OGROPT_HAVE_OGR_CYCLE_ASM == 1)
  #define FOUND_ONE_DECL(x) int found_one(x)
#else
  #define FOUND_ONE_DECL(x) static int found_one(x)
#endif


#ifndef __MRC__
static int init_load_choose(void);
static int ogr_init(void);
static int ogr_getresult(void *state, void *result, int resultlen);
static int ogr_destroy(void *state);
static int ogr_cleanup(void);
#if (OGROPT_HAVE_OGR_CYCLE_ASM == 0)
static int ogr_cycle(void *state, int *pnodes, int with_time_constraints);
#endif
static int ogr_create(void *input, int inputlen, void *state, int statelen,
                      int minpos);
#if (OGROPT_HAVE_OGR_CYCLE_ASM < 2)
  NO_FUNCTION_INLINE(FOUND_ONE_DECL(const struct State *oState));
#endif


#ifndef OGR_GET_DISPATCH_TABLE_FXN
  #define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table
#endif
extern CoreDispatchTable * OGR_GET_DISPATCH_TABLE_FXN (void);


#ifdef OGR_DEBUG
  /*
  ** Use DUMP_BITMAPS(depth, lev) or DUMP_RULER(state, depth) macros
  ** instead of calling the following functions directly, unless you
  ** want unpredictable results...
  */
  static void __dump(const struct Level *lev, int depth);
  static void __dump_ruler(const struct State *oState, int depth);

  /*
  ** Debugging macros
  */
  #if !defined(DUMP_BITMAPS)
    #define DUMP_BITMAPS(lev,depth)   \
      SAVE_FINAL_STATE(lev);          \
      __dump(lev, depth);
  #endif

  #if !defined(DUMP_RULER)
    #define DUMP_RULER(state,depth)   \
      __dump_ruler(state, depth);
  #endif
#else
  #undef  DUMP_BITMAPS
  #define DUMP_BITMAPS(foo,bar)
  #undef  DUMP_RULER
  #define DUMP_RULER(foo,bar)
#endif  /* OGR_DEBUG */


#ifdef OGR_TEST_FIRSTBLANK
  static int  __testFirstBlank(void);
#endif

#endif  /* __MRC__ */

#if defined(__cplusplus)
}
#endif

/* ----------------------------------------------------------------------- */
/***************************************************************************
 * The following macros define the BITLIST CLASS (bitmaps manipulation)
 * The variables defined here should only be manipulated within these class
 * macros.
 ***************************************************************************/

#if (OGROPT_ALTERNATE_CYCLE < 2) && (OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 0)
  /* Enforce the use of default code */
  #undef SETUP_TOP_STATE
  #undef COMP_LEFT_LIST_RIGHT
  #undef COMP_LEFT_LIST_RIGHT_32
  #undef PUSH_LEVEL_UPDATE_STATE
  #undef POP_LEVEL
  #undef SAVE_FINAL_STATE
#endif


#if (OGROPT_ALTERNATE_CYCLE == 1) 

  /*
  ** OGROPT_ALTERNATE_CYCLE == 1
  ** Initialize top state
  */
  #if !defined(SETUP_TOP_STATE)
    #define SETUP_TOP_STATE(lev)  \
      U comp0 = lev->comp[0],     \
        comp1 = lev->comp[1],     \
        comp2 = lev->comp[2],     \
        comp3 = lev->comp[3],     \
        comp4 = lev->comp[4];     \
      U list0 = lev->list[0],     \
        list1 = lev->list[1],     \
        list2 = lev->list[2],     \
        list3 = lev->list[3],     \
        list4 = lev->list[4];     \
      U dist0 = lev->dist[0],     \
        dist1 = lev->dist[1],     \
        dist2 = lev->dist[2],     \
        dist3 = lev->dist[3],     \
        dist4 = lev->dist[4];     \
      int newbit = 1;
  #endif

  /*
  ** OGROPT_ALTERNATE_CYCLE == 1
  ** Shift COMP and LIST bitmaps (default implementation)
  */
  #if !defined(COMP_LEFT_LIST_RIGHT)
    #define COMP_LEFT_LIST_RIGHT(lev, s) {  \
      U temp1, temp2;                       \
      int ss = 32 - (s);                    \
      comp0 <<= s;                          \
      temp1 = newbit << ss;                 \
      temp2 = list0 << ss;                  \
      list0 = (list0 >> (s)) | temp1;       \
      temp1 = list1 << ss;                  \
      list1 = (list1 >> (s)) | temp2;       \
      temp2 = list2 << ss;                  \
      list2 = (list2 >> (s)) | temp1;       \
      temp1 = list3 << ss;                  \
      list3 = (list3 >> (s)) | temp2;       \
      temp2 = comp1 >> ss;                  \
      list4 = (list4 >> (s)) | temp1;       \
      temp1 = comp2 >> ss;                  \
      comp0 |= temp2;                       \
      temp2 = comp3 >> ss;                  \
      comp1 = (comp1 << (s)) | temp1;       \
      temp1 = comp4 >> ss;                  \
      comp2 = (comp2 << (s)) | temp2;       \
      comp4 = comp4 << (s);                 \
      comp3 = (comp3 << (s)) | temp1;       \
      newbit = 0;                           \
    }
  #endif

  /*
  ** OGROPT_ALTERNATE_CYCLE == 1
  ** Shift COMP and LIST bitmaps by 32 (default implementation)
  */
  #if !defined(COMP_LEFT_LIST_RIGHT_32)
    #define COMP_LEFT_LIST_RIGHT_32(lev)  \
      list4 = list3;                      \
      list3 = list2;                      \
      list2 = list1;                      \
      list1 = list0;                      \
      list0 = newbit;                     \
      comp0 = comp1;                      \
      comp1 = comp2;                      \
      comp2 = comp3;                      \
      comp3 = comp4;                      \
      comp4 = 0;                          \
      newbit = 0;
  #endif

  /*
  ** OGROPT_ALTERNATE_CYCLE == 1
  ** Push level state. Update LIST and COMP bitmaps
  */
  #if !defined(PUSH_LEVEL_UPDATE_STATE)
    #define PUSH_LEVEL_UPDATE_STATE(lev)    \
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
      newbit = 1;
  #endif

  /*
  ** OGROPT_ALTERNATE_CYCLE == 1
  ** Pop level state (all bitmaps).
  */
  #if !defined(POP_LEVEL)
    #define POP_LEVEL(lev)  \
      list0 = lev->list[0]; \
      list1 = lev->list[1]; \
      list2 = lev->list[2]; \
      list3 = lev->list[3]; \
      list4 = lev->list[4]; \
      dist0 &= ~list0;      \
      comp0 = lev->comp[0]; \
      dist1 &= ~list1;      \
      comp1 = lev->comp[1]; \
      dist2 &= ~list2;      \
      comp2 = lev->comp[2]; \
      dist3 &= ~list3;      \
      comp3 = lev->comp[3]; \
      dist4 &= ~list4;      \
      comp4 = lev->comp[4]; \
      newbit = 0;
  #endif

  /*
  ** OGROPT_ALTERNATE_CYCLE == 1
  ** Save final state (all bitmaps)
  */
  #if !defined(SAVE_FINAL_STATE)
    #define SAVE_FINAL_STATE(lev) \
      lev->list[0] = list0;       \
      lev->list[1] = list1;       \
      lev->list[2] = list2;       \
      lev->list[3] = list3;       \
      lev->list[4] = list4;       \
      lev->dist[0] = dist0;       \
      lev->dist[1] = dist1;       \
      lev->dist[2] = dist2;       \
      lev->dist[3] = dist3;       \
      lev->dist[4] = dist4;       \
      lev->comp[0] = comp0;       \
      lev->comp[1] = comp1;       \
      lev->comp[2] = comp2;       \
      lev->comp[3] = comp3;       \
      lev->comp[4] = comp4;
  #endif

#elif (OGROPT_ALTERNATE_CYCLE == 0) /*======================================*/

  /*
  ** OGROPT_ALTERNATE_CYCLE == 0
  ** Initialize top state
  */
  #if !defined(SETUP_TOP_STATE)
    #define SETUP_TOP_STATE(lev)  \
      U comp0 = lev->comp[0];     \
      U dist0 = lev->dist[0];     \
      int newbit = 1;
  #endif

  /*
  ** OGROPT_ALTERNATE_CYCLE == 0
  ** Shift COMP and LIST bitmaps
  */
  #if !defined(COMP_LEFT_LIST_RIGHT)
    #define COMP_LEFT_LIST_RIGHT(lev, s) {          \
      U temp1, temp2;                               \
      register int ss = 32 - (s);                   \
      temp1 = newbit << ss;                         \
      temp2 = lev->list[0] << ss;                   \
      lev->list[0] = (lev->list[0] >> (s)) | temp1; \
      temp1 = lev->list[1] << ss;                   \
      lev->list[1] = (lev->list[1] >> (s)) | temp2; \
      temp2 = lev->list[2] << ss;                   \
      lev->list[2] = (lev->list[2] >> (s)) | temp1; \
      temp1 = lev->list[3] << ss;                   \
      lev->list[3] = (lev->list[3] >> (s)) | temp2; \
      temp2 = lev->comp[1] >> ss;                   \
      lev->list[4] = (lev->list[4] >> (s)) | temp1; \
      temp1 = lev->comp[2] >> ss;                   \
      comp0 = (lev->comp[0] << (s)) | temp2;        \
      lev->comp[0] = comp0;                         \
      temp2 = lev->comp[3] >> ss;                   \
      lev->comp[1] = (lev->comp[1] << (s)) | temp1; \
      temp1 = lev->comp[4] >> ss;                   \
      lev->comp[2] = (lev->comp[2] << (s)) | temp2; \
      lev->comp[4] = (lev->comp[4] << (s));         \
      lev->comp[3] = (lev->comp[3] << (s)) | temp1; \
      newbit = 0;                                   \
    }
  #endif

  /*
  ** OGROPT_ALTERNATE_CYCLE == 0
  ** Shift COMP and LIST bitmaps by 32
  */
  #if !defined(COMP_LEFT_LIST_RIGHT_32)
    #define COMP_LEFT_LIST_RIGHT_32(lev)  \
      lev->list[4] = lev->list[3];        \
      lev->list[3] = lev->list[2];        \
      lev->list[2] = lev->list[1];        \
      lev->list[1] = lev->list[0];        \
      lev->list[0] = newbit;              \
      comp0 = lev->comp[1];               \
      lev->comp[0] = comp0;               \
      lev->comp[1] = lev->comp[2];        \
      lev->comp[2] = lev->comp[3];        \
      lev->comp[3] = lev->comp[4];        \
      lev->comp[4] = 0;                   \
      newbit = 0;
  #endif

  #if !defined(PUSH_LEVEL_UPDATE_STATE)
    #define PUSH_LEVEL_UPDATE_STATE(lev) {            \
      U temp1, temp2;                                 \
      struct Level *lev2 = lev + 1;                   \
      dist0 = (lev2->list[0] = lev->list[0]);         \
      temp2 = (lev2->list[1] = lev->list[1]);         \
      dist0 = (lev2->dist[0] = lev->dist[0] | dist0); \
      temp2 = (lev2->dist[1] = lev->dist[1] | temp2); \
      comp0 = lev->comp[0] | dist0;                   \
      lev2->comp[0] = comp0;                          \
      lev2->comp[1] = lev->comp[1] | temp2;           \
      temp1 = (lev2->list[2] = lev->list[2]);         \
      temp2 = (lev2->list[3] = lev->list[3]);         \
      temp1 = (lev2->dist[2] = lev->dist[2] | temp1); \
      temp2 = (lev2->dist[3] = lev->dist[3] | temp2); \
      lev2->comp[2] = lev->comp[2] | temp1;           \
      lev2->comp[3] = lev->comp[3] | temp2;           \
      temp1 = (lev2->list[4] = lev->list[4]);         \
      temp1 = (lev2->dist[4] = lev->dist[4] | temp1); \
      lev2->comp[4] = lev->comp[4] | temp1;           \
      newbit = 1;                                     \
    }
  #endif

  /*
  ** OGROPT_ALTERNATE_CYCLE == 0
  ** Pop level state (all bitmaps).
  */
  #if !defined(POP_LEVEL)
    #define POP_LEVEL(lev)  \
      comp0 = lev->comp[0]; \
      newbit = 0;
  #endif

  /*
  ** OGROPT_ALTERNATE_CYCLE == 0
  ** Save final state (all bitmaps)
  */
  #if !defined(SAVE_FINAL_STATE)
    #define SAVE_FINAL_STATE(lev) \
      /* nothing */
  #endif

#endif  /* OGROPT_ALTERNATE_CYCLE */


/*-------------------------------------------------------------------------*/

#if defined(BYTE_ORDER) && (BYTE_ORDER == LITTLE_ENDIAN)
  #define FP_CLZ_LITTLEEND 1
#endif

#if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && defined(__CNTLZ)
  #define LOOKUP_FIRSTBLANK(x) __CNTLZ(x)
#elif (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 0) && defined(FP_CLZ_LITTLEEND)
  /*
  ** using the exponent in floating point double format
  ** Relies upon IEEE format (Little Endian)
  */
  static inline int LOOKUP_FIRSTBLANK(register unsigned int input)
  {
    unsigned int i;
    union {
      double d;
      int i[2];
    } u;
    i = ~input;
    u.d = i;
    return i == 0 ? 33 : 1055 - (u.i[1] >> 20);
  }
#else
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


  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 1) && defined(__CNTLZ_ARRAY_BASED)
    #define LOOKUP_FIRSTBLANK(x) __CNTLZ_ARRAY_BASED(x,ogr_first_blank_8bit)
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
#endif


/* ----------------------------------------------------------------------- */

static int init_load_choose(void)
{
  if (CHOOSE_DIST_BITS != ogr_choose_dat[2]
      || CHOOSE_MARKS != ogr_choose_dat[1])
  {
    /* Incompatible CHOOSE array - Give up */
    return CORE_E_INTERNAL;
  }

  return CORE_S_OK;
}


/*-------------------------------------------------------------------------*/
/*
** found_one() - Assert the ruler is Golomb.
** This function is *REALLY* seldom used and it has no impact at all on
** benchmarks or runtime node rates.
** However, some compilers may choose to inline this function into ogr_cycle()
** thus causing a noticeable slow down. The fix is then to prevent the
** compiler from inlining this function, *NOT* to optimize it.
**
** NOTE : Principle #1 states that we don't have to track distances larger
**    than half the length of the ruler.
*/

#if (OGROPT_HAVE_OGR_CYCLE_ASM < 2)   /* ASM cores (if any) need it */
FOUND_ONE_DECL(const struct State *oState)
{
  register int i, j;
  u32 diffs[((1024 - BITMAPS_LENGTH) + 31) / 32];
  register int max = oState->max;
  register int maxdepth = oState->maxdepth;
  const struct Level *levels = &oState->Levels[0];

  for (i = max >> (5+1); i >= 0; --i)
    diffs[i] = 0;

  for (i = 1; i < maxdepth; i++) {
    register int marks_i = levels[i].mark;

    for (j = 0; j < i; j++) {
      register int diff = marks_i - levels[j].mark;
      if (diff <= BITMAPS_LENGTH)
        break;

      if (diff+diff <= max) {       /* Principle #1 */
        register u32 mask = 1 << (diff & 31);
        diff = (diff >> 5) - (BITMAPS_LENGTH / 32);
        if ((diffs[diff] & mask) != 0)
          return CORE_S_CONTINUE;   /* Distance already taken = not Golomb */

        diffs[diff] |= mask;
      }
    }
  }
  return CORE_S_SUCCESS;
}
#endif  /* OGROPT_HAVE_OGR_CYCLE_ASM < 2 */


/* ----------------------------------------------------------------------- */

static int ogr_init(void)
{
  int r = init_load_choose();
  if (r != CORE_S_OK) {
    return r;
  }

  #if defined(OGR_TEST_FIRSTBLANK)
    if (__testFirstBlank())
      return CORE_E_INTERNAL;
  #endif

  return CORE_S_OK;
}


/* ----------------------------------------------------------------------- */
/*
** Merged implementation of ogr_create() (phase 1 + phase 2)
** This function is called each time a stub is (re)started, so optimizations
** are useless and not welcomed.
**
** OGR24-p2 :
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
** OGR25-p2 :
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
static int ogr_create(void *input, int inputlen, void *state, int statelen,
                      int minpos)
{
  struct State *oState;
  struct WorkStub *workstub = (struct WorkStub *)input;
  int finalization_stub = 0;

  if (input == NULL || inputlen != sizeof(struct WorkStub)
                    || (size_t)statelen < sizeof(struct State)) {
    return CORE_E_INTERNAL;
  }

  if ( (oState = (struct State *)state) == NULL) {
    return CORE_E_MEMORY;
  }

  memset(oState, 0, sizeof(struct State));
  oState->maxdepth   = workstub->stub.marks;
  oState->maxdepthm1 = oState->maxdepth-1;
  oState->max        = OGR[oState->maxdepthm1];

  if (((unsigned int)oState->maxdepth) > (sizeof(OGR)/sizeof(OGR[0]))) {
    return CORE_E_FORMAT;
  }  

  /*
  ** Sanity check
  ** Make sure supposed finalization stubs (that have less diffs than regular
  ** stubs) come with the corresponding starting point.
  ** OGR already handled all stubs upto (and including) length 70, so the
  ** starting point must be higher.
  */
  if (workstub->stub.marks == 24 && workstub->stub.length < 5) {
    if (minpos <= 70 || minpos > OGR[24-1] - OGR[(24-2) - workstub->stub.length]) {
      return CORE_E_STUB;         // Too low.
    }
    finalization_stub = 1;
  }
  else if (workstub->stub.marks == 25 && workstub->stub.length < 6) {
    if (minpos <= 70 || minpos > OGR[25-1] - OGR[(25-2) - workstub->stub.length]) {
      return CORE_E_STUB;         // Too low.
    }
    finalization_stub = 1;
  }
  else if (minpos != 0) {
    // Unsuspected starting point
    return CORE_E_STUB;
  }

  if (workstub->stub.length < workstub->worklength) {
    /* BUGFIX : Reset the flag if the stub has already been started to prevent
    ** inaccurate node counts when the user stop then restart the client.
    ** (the init loop performs one 'cycle' iteration for finalization stubs,
    ** and that iteration is not taken into account in the final node count) */
    finalization_stub = 0;
  }

  /* Mid-point reduction - First determine the middle mark (or the middle
  ** segment if the ruler is even).
  ** Note, marks are labled 0, 1...  so mark @ depth=1 is 2nd mark
  */
  oState->half_depth  = ((oState->maxdepth+1) >> 1) - 2;
  oState->half_depth2 = oState->half_depth + 2;
  if (!(oState->maxdepth % 2))
    oState->half_depth2++;  /* if even, use 2 marks */

  /*------------------
  Since:  half_depth2 = half_depth+2 (or 3 if maxdepth even) ...
  We get: half_length2 >= half_length + 3=OGR[2] (or 6=OGR[3] if maxdepth even)
  But:    half_length2 + half_length <= max-1    (our midpoint reduction)
  So:     half_length + 3 (6 if maxdepth even) + half_length <= max-1
          half_length <= (max-1 - 3 (6 if maxdepth even)) / 2
  ------------------*/
  if ( !(oState->maxdepth % 2) ) oState->half_length = (oState->max-7) >> 1;
  else                           oState->half_length = (oState->max-4) >> 1;

  {
    int depth, n;
    struct Level *lev = &oState->Levels[1];
    int mark = 0;
    SETUP_TOP_STATE(lev);

    n = workstub->worklength;
    if (n < workstub->stub.length) {
      n = workstub->stub.length;
    }
    if (n > STUB_MAX) {
      return CORE_E_FORMAT;
    }

    for (depth = 1; depth <= n + finalization_stub; depth++) {
      int limit = oState->max - choose(dist0 >> ttmDISTBITS, oState->maxdepthm1 - depth);

      /* Compute the maximum position for the current mark */
      if (depth <= oState->half_depth2) {
        if (depth <= oState->half_depth) {
          limit = oState->max - OGR[oState->maxdepthm1 - depth];
          if (limit > oState->half_length) {
            limit = oState->half_length;
          }
        }
        else if (limit >= oState->max - oState->Levels[oState->half_depth].mark) {
          limit = oState->max - oState->Levels[oState->half_depth].mark - 1;
        }
      }
      lev->limit = limit;

      if (depth > n) {
        /* Finalization stub
        ** Skip to the minimum position specified by the keyserver, then
        ** place the next mark at the first position available from here
        ** (including the specified minimum position).
        ** As a result, ogr_cycle() will start one level deeper, then
        ** proceed as usual until it backtracks to depth oState->startdepth.
        */
        int s = minpos - 1;
        if (mark < s) {
          int k = s - mark;
          while (k >= 32) {
            COMP_LEFT_LIST_RIGHT_32(lev);
            k -= 32;
          }
          if (k > 0) {
            COMP_LEFT_LIST_RIGHT(lev, k);
          }
          mark = s;
        }

      stay:
        if (comp0 < 0xfffffffe) {
          s = LOOKUP_FIRSTBLANK( comp0 );
          if ((mark += s) > limit)
            return CORE_E_STUB;
          COMP_LEFT_LIST_RIGHT(lev, s);
        } 
        else { /* s >= 32 */
          if ((mark += 32) > limit)
            return CORE_E_STUB;
          if (comp0 == ~0u) {
            COMP_LEFT_LIST_RIGHT_32(lev);
            goto stay;
          }
          COMP_LEFT_LIST_RIGHT_32(lev);
        }
      }
      else {
        /*
        ** Regular (phase #1) stubs
        */
        int s = workstub->stub.diffs[depth-1];
        if ((mark += s) > limit)
          return CORE_E_STUB;

        while (s >= 32) {
          if (s == 32 && (comp0 & 1u))
            return CORE_E_STUB;         /* invalid location */
          COMP_LEFT_LIST_RIGHT_32(lev);
          s -= 32;
        }
        if (s > 0) {
          if ( (comp0 & (1u << (32-s))) )
            return CORE_E_STUB;         /* invalid location */
          COMP_LEFT_LIST_RIGHT(lev, s);
        }
      }
      PUSH_LEVEL_UPDATE_STATE(lev);
      lev->mark = mark;
      lev++;
    }
    SAVE_FINAL_STATE(lev);
    lev->mark = mark;
    oState->depth = depth - 1;  // externally visible depth is one less than internal
  }

  oState->startdepth = workstub->stub.length;
  return CORE_S_OK;
}


/* ----------------------------------------------------------------------- */

#if (OGROPT_HAVE_OGR_CYCLE_ASM == 0)
static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
{
  int retval;
  struct State *oState = (struct State *)state;
  int depth = oState->depth + 1;
  struct Level *lev = &oState->Levels[depth];
  int nodes = 0;
  int mark = lev->mark;

  /*
  ** Copy useful datas into local variables to speed things up a bit. For
  ** register rich architectures, this gives a hint to the compiler. Otherwise,
  ** the compiler can fetch these datas from the stack instead of being forced
  ** to load the oState pointer, then the data we need.
  ** We also consider the first depth is 0, so that we don't have to compare
  ** the current depth with startdepth to determine whether the main loop must
  ** be exited. Other depth levels are updated accordingly.
  */
  const struct Level *levHalfDepth = &oState->Levels[oState->half_depth];
  int maxlength  = oState->max;
  int remdepth   = oState->maxdepthm1 - depth;
  int halfdepth  = oState->half_depth - oState->startdepth;
  int halfdepth2 = oState->half_depth2 - oState->startdepth;
  depth -= oState->startdepth;

  SETUP_TOP_STATE(lev);
  OGR_CYCLE_CACHE_ALIGN;

  for (;;) {
    int limit;

    if (with_time_constraints) { /* if (...) is optimized away if unused */
       #if !defined(OGROPT_IGNORE_TIME_CONSTRAINT_ARG)
       if (nodes >= *pnodes) {
         retval = CORE_S_CONTINUE;
         break;
       }  
       #endif  
    }

    limit = maxlength - choose(dist0 >> ttmDISTBITS, remdepth);

    /*
    ** Compute the maximum position for the current mark. The most common case
    ** is "depth <= halfdepth2" since we spend much time on the right side of
    ** the ruler.
    */
    if (depth <= halfdepth2) {
      if (depth <= halfdepth) {
        if (nodes >= *pnodes) {
          retval = CORE_S_CONTINUE;
          break;
        }
        limit = maxlength - OGR[remdepth];
        if (limit > oState->half_length)
          limit = oState->half_length;
      }
      else if (limit >= maxlength - levHalfDepth->mark) {
        limit = maxlength - levHalfDepth->mark - 1;
      }
    }

    nodes++;
    lev->limit = limit;

    /*
    ** Find the next available mark location for this level
    */
  stay:
    if (comp0 < 0xfffffffe) {
      int s = LOOKUP_FIRSTBLANK( comp0 );
      if ((mark += s) > limit) goto up;   /* no spaces left */
      COMP_LEFT_LIST_RIGHT(lev, s);
    }
    else {  /* s >= 32 */
      if ((mark += 32) > limit) goto up;
      if (comp0 == ~0u) {
        COMP_LEFT_LIST_RIGHT_32(lev);
        goto stay;  /* s == 33 : search next valid location */
      }
      COMP_LEFT_LIST_RIGHT_32(lev);
    }
    lev->mark = mark;

    if (remdepth == 0) {                  /* New ruler ? (last mark placed) */
      retval = found_one(oState);
      if (retval != CORE_S_CONTINUE) {
        break;                            /* Found a Golomb Ruler ! */
      }
      goto stay;                          /* Try all possible locations */
    }

    PUSH_LEVEL_UPDATE_STATE(lev);         /* Go deeper */
    lev++;
    depth++;
    remdepth--;
    continue;

    OGR_CYCLE_CACHE_ALIGN;
  up:
    lev--;
    depth--;
    remdepth++;
    POP_LEVEL(lev);
    limit = lev->limit;
    mark = lev->mark;
    if (depth <= 0) {
      retval = CORE_S_OK;
      break;
    }
    goto stay; /* repeat this level till done */
  }

  SAVE_FINAL_STATE(lev);
  lev->mark = mark;
  oState->depth = depth - 1 + oState->startdepth;
  *pnodes = nodes;
  return retval;
}
#endif  /* OGROPT_HAVE_OGR_CYCLE_ASM */


/* ----------------------------------------------------------------------- */

static int ogr_getresult(void *state, void *result, int resultlen)
{
  struct State *oState = (struct State *)state;
  struct WorkStub *workstub = (struct WorkStub *)result;
  int i;

  if (resultlen != sizeof(struct WorkStub)) {
    return CORE_E_INTERNAL;
  }
  workstub->stub.marks = (u16)oState->maxdepth;
  workstub->stub.length = (u16)oState->startdepth;
  for (i = 0; i < STUB_MAX; i++) {
    workstub->stub.diffs[i] = (u16)(oState->Levels[i+1].mark - oState->Levels[i].mark);
  }
  workstub->worklength = oState->depth;
  if (workstub->worklength > STUB_MAX) {
    workstub->worklength = STUB_MAX;
  }
  return CORE_S_OK;
}


/* ----------------------------------------------------------------------- */

static int ogr_destroy(void *state)
{
  state = state;
  return CORE_S_OK;
}


/* ----------------------------------------------------------------------- */

static int ogr_cleanup(void)
{
  return CORE_S_OK;
}


/* ----------------------------------------------------------------------- */

#if (OGROPT_HAVE_OGR_CYCLE_ASM == 0)
CoreDispatchTable * OGR_GET_DISPATCH_TABLE_FXN (void)
{
  static CoreDispatchTable dispatch_table;
  dispatch_table.init      = ogr_init;
  dispatch_table.create    = ogr_create;
  dispatch_table.cycle     = ogr_cycle;
  dispatch_table.getresult = ogr_getresult;
  dispatch_table.destroy   = ogr_destroy;
  dispatch_table.cleanup   = ogr_cleanup;
  return &dispatch_table;
}
#endif

/* ----------------------------------------------------------------------- */

#if defined(OGR_DEBUG)
static void __dump(const struct Level *lev, int depth)
{
  printf("--- depth %d, limit %d\n", depth, lev->limit);

#if  defined(OGROPT_OGR_CYCLE_ALTIVEC) \
  && (defined(__VEC__) || defined(__ALTIVEC__))
    printf("list=%08x %:08vlx\n", lev->list0, lev->listV);
    printf("dist=%08x %:08vlx\n", lev->dist0, lev->distV);
    printf("comp=%08x %:08vlx\n", lev->comp0, lev->compV);
#elif !defined(OGROPT_64BIT_IMPLEMENTATION)
  printf("list=%08x %08x %08x %08x %08x\n", lev->list[0], lev->list[1],
                                            lev->list[2], lev->list[3],
                                            lev->list[4]);
  printf("dist=%08x %08x %08x %08x %08x\n", lev->dist[0], lev->dist[1],
                                            lev->dist[2], lev->dist[3],
                                            lev->dist[4]);
  printf("comp=%08x %08x %08x %08x %08x\n", lev->comp[0], lev->comp[1],
                                            lev->comp[2], lev->comp[3],
                                            lev->comp[4]);

#elif (!defined(SIZEOF_SHORT) || (SIZEOF_SHORT < 8)) \
   && (!defined(SIZEOF_INT)   || (SIZEOF_INT   < 8)) \
   && (!defined(SIZEOF_LONG)  || (SIZEOF_LONG  < 8))
  // Assume ui64 is unsigned long long int
  printf("list=%016llx %016llx %016llx\n", lev->list[0], lev->list[1], lev->list[2]);
  printf("dist=%016llx %016llx %016llx\n", lev->dist[0], lev->dist[1], lev->dist[2]);
  printf("comp=%016llx %016llx %016llx\n", lev->comp[0], lev->comp[1], lev->comp[2]);
#else
  printf("list=%016lx %016lx %016lx\n", lev->list[0], lev->list[1], lev->list[2]);
  printf("dist=%016lx %016lx %016lx\n", lev->dist[0], lev->dist[1], lev->dist[2]);
  printf("comp=%016lx %016lx %016lx\n", lev->comp[0], lev->comp[1], lev->comp[2]);
#endif
}


static void __dump_ruler(const struct State *oState, int depth)
{
  int i;
  printf("max %d ruler : ", oState->max);
  for (i = 1; i < depth; i++) {
    printf("%d ", oState->Levels[i].mark - oState->Levels[i-1].mark);
  }
  printf("\n");
}
#endif  /* OGR_DEBUG */


#if defined(OGR_TEST_FIRSTBLANK)
static int __testFirstBlank(void)
{
  static int done_test = 0;
  static char ogr_first_blank[65537]; /* first blank in 16 bit COMP bitmap, range: 1..16 */
  /* first zero bit in 16 bits */
  int i, j, k = 0, m = 0x8000;
  unsigned int err_count = 0;

  for (i = 1; i <= 16; i++) {
    for (j = k; j < k+m; j++) ogr_first_blank[j] = (char)i;
    k += m;
    m >>= 1;
  }
  ogr_first_blank[0xffff] = 17;     /* just in case we use it */
  if (done_test == 0)
  {
    unsigned int q, first_fail = 0xffffffff;
    int last_s1 = 0, last_s2 = 0;
    done_test = 1;                  /* test only once */
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
  return (err_count) ? -1 : 0;
}
#endif
