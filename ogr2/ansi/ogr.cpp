/*
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * This file (ogr.cpp) contains 
 * - all the OGROPT optimization switch settings for all systems, cores, etc.
 * - routines common to all cores
 * - it #includes all neccessary .cor (core functions/macros), 
 *   .mac (general macros), .inc (general stuff) files
 */
#define __OGR_CPP__ "@(#)$Id: ogr.cpp,v 1.1.2.43.2.4 2001/05/03 11:14:24 andreasb Exp $"

#include <stdio.h>      /* printf for debugging */
#include <stdlib.h>     /* malloc (if using non-static choose dat) */
#include <string.h>     /* memset */

#define HAVE_STATIC_CHOOSEDAT /* choosedat table is static, pre-generated */
/* #define CRC_CHOOSEDAT_ANYWAY */ /* you'll need to link crc32 if this is defd */

/* #define OGR_TEST_FIRSTBLANK */ /* test firstblank logic (table or asm) */
/* #define OGR_TEST_BITOFLIST  */ /* test bitoflist table */

/* --- various optimization option overrides ----------------------------- */
//#define OGROPT_ALTERNATE_CYCLE 1
//#define OGROPT_ALTERNATE_CYCLE 2
#define OGROPT_ALTERNATE_CYCLE 3

/* baseline/reference == ogr.cpp without optimization == ~old ogr.cpp */
#if defined(NO_OGR_OPTIMIZATION) || defined(GIMME_BASELINE_OGR_CPP)
  #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 0/1 - default is 1 ('yes') */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0/1 - default is hw dependant */
  #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 0-2 - default is 1 */
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
#elif defined(__IA64__)
  #define OGROPT_BITOFLIST_DIRECT_BIT             0 /* 'no' irrelevant */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM     0 /* 'no' no asm */
  #define OGROPT_COPY_LIST_SET_BIT_JUMPS          1 /* 0-2 - default is 1 */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE           1 /* 0/1 - default is 1 ('yes') */
  #define OGROPT_ALTERNATE_CYCLE                  1 /* 0/1 - default is 0 ('no') */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT   0 /* 0-2 - default is 0 */
#elif defined(ASM_68K)
  #define OGROPT_BITOFLIST_DIRECT_BIT 0          /* we want 'no' */
#elif defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__)
  #if (__MWERKS__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* MWC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 2 /* use switch_asm   */
  #elif (__MRC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* MrC is better    */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* MrC is better    */
  #elif (__APPLE_CC__)//GCC with exclusive ppc, mach-o and ObjC extensions
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* ACC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* PPC optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* ACC is better    */
  #elif (__GNUC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
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
   OGROPT_ALTERNATE_CYCLE == 3 -> devel core by andreasb
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


/* ToDo: #undef all OGROPT_ defines not needed for selected ALTERNATE_CYCLE
*/

/* ----------------------------------------------------------------------- */

#if !defined(HAVE_STATIC_CHOOSEDAT) || defined(CRC_CHOOSEDAT_ANYWAY)
#include "crc32.h" /* only need to crc choose_dat if its not static */
#endif

#include "ogr.h"
#include "state.h"

// maximum number of marks supported by ogr_choose_dat
#define CHOOSE_MAX_DEPTH   12
// alignment musn't be equal to CHOOSE_MAX_MARKS
#define CHOOSE_ALIGNMENT   16
// number of bits from the beginning of dist bitmaps supported by ogr_choose_dat
#define CHOOSE_DIST_BITS   12
#define ttmDISTBITS (32-CHOOSE_DIST_BITS)



/* use the core as a stubmap generator ... */
#ifdef OGR_CALLBACK
  extern int ogr_callback_depth;

  /* returns 0 while ogr_cycle() should continue */
  int ogr_callback(const struct State *state);
  
  #define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table_callback
#endif



#if defined(__cplusplus)
extern "C" {
#endif

#ifdef HAVE_STATIC_CHOOSEDAT  /* choosedat table is static, pre-generated */
//  extern const unsigned char ogr_choose_dat2[]; /* this is in ogr_dat2.cpp */
/* choose(bitmap, depth) */
  #if (CHOOSE_ALIGNMENT == 16 && OGROPT_STRENGTH_REDUCE_CHOOSE == 1)
     // strength reduce the multiply -- which is slow on MANY processors
     #define choose(x,y) (ogr_choose_dat2[((x)<<4)+(y)])
  #else
     #define choose(x,y) (ogr_choose_dat2[CHOOSE_ALIGNMENT*(x)+(y)])
  #endif
#else
  #error OGROPT_NEW_CHOOSEDAT and not HAVE_STATIC_CHOOSEDAT ???   
#endif

static const int OGR_length[] = { /* use: OGR_length[depth] */ 
/* marks */
  /*  1 */    0,   1,   3,   6,  11,  17,  25,  34,  44,  55,
  /* 11 */   72,  85, 106, 127, 151, 177, 199, 216, 246, 283,
  /* 21 */  333, 356, 372, 425, 480, 492, 553, 585, 623
};

#ifndef __MRC__
static int init_load_choose(void);
static int ogr_init(void);
static const char* ogr_name(void);
static const char* ogr_core_id(void);
static int ogr_get_size(int* alignment);
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
/* Include general macros and functions                               */
/* ------------------------------------------------------------------ */

// Always include this. Defines several empty macros if not debugging.
#include "ogr_deb.inc"


// all methods of LOOKUP_FIRSTBLANK
#include "ogr_fb.mac"


/* ------------------------------------------------------------------ */
/* Include core specific macros and functions                         */
/* ------------------------------------------------------------------ */

#if (OGROPT_ALTERNATE_CYCLE == 0)
  #include "ogr_g_r.cor"
#elif ((OGROPT_ALTERNATE_CYCLE == 1) || (OGROPT_ALTERNATE_CYCLE == 2))
  #include "ogr_o_c.cor"
#elif (OGROPT_ALTERNATE_CYCLE == 3)
  #include "ogr_ab.cor"
#else
  #error unknown core selected!
#endif


/* ------------------------------------------------------------------ */
/* Functions common to all cores                                      */
/* ------------------------------------------------------------------ */

static int init_load_choose(void)
{
#ifndef HAVE_STATIC_CHOOSEDAT
  #error non static choosedat not supported with OGROPT_NEW_CHOOSEDAT
#endif
  if ( (CHOOSE_ALIGNMENT != choose_alignment) ||
       (CHOOSE_DIST_BITS != choose_dist_bits) ||
       (CHOOSE_MAX_DEPTH >  choose_max_depth) )
  {
    return CORE_E_FORMAT;
  }

  return CORE_S_OK;
}


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

  {
    static int printed = 0;
    if (printed == 0) {
      printf("%s\n%s\n", ogr_name(), ogr_core_id());
      printed = 1;
    }
  }
  return CORE_S_OK;
}


static const char* ogr_core_id()
{
  /* whats the most senseful format for this? char ** like argv ? */
  #define STRINGIFY2(x) #x
  #define STRINGIFY(x) STRINGIFY2(x)
  return STRINGIFY(OGR_GET_DISPATCH_TABLE_FXN) "\n" 
         __OGR_H__ "\n" __OGR_CPP__ "\n" __OGR_FB_MAC__ "\n" __OGR_CORE__ "\n";
  #undef STRINGIFY
  #undef STRINGIFY2
}


static int ogr_get_size(int *alignment)
{
  if (alignment)
    *alignment = OGR_MEM_ALIGNMENT;
  return sizeof(State);
}


#if 0 // old version
static int ogr_getresult(void *state, void *result, int resultlen)
{
  struct State *oState = (struct State *)state;
  struct WorkStub *workstub = (struct WorkStub *)result;
  int i;

  if (resultlen != sizeof(struct WorkStub)) {
    return CORE_E_FORMAT;
  }
  workstub->stub.marks = (u16)oState->maxmarks;
  workstub->stub.length = (u16)oState->startdepth;
  for (i = 0; i < STUB_MAX; i++) {
    #if (OGROPT_ALTERNATE_CYCLE == 0)
    workstub->stub.diffs[i] = (u16)(oState->markpos[i+1] - oState->markpos[i]);
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
#else
static int ogr_getresult(void *state, void *result, int resultlen)
/* DO NOT call ogr_getresult() for a state currently being processed 
   by ogr_cycle() */
{
  struct State *oState = (struct State *)state;
  struct WorkStub *workstub = (struct WorkStub *)result;
  int i;

  if (resultlen != sizeof(struct WorkStub)) {
    return CORE_E_FORMAT;
  }

  workstub->stub.marks = (u16)oState->maxmarks;
  workstub->stub.length = (u16)oState->startdepth;
  for (i = 0; i < STUB_MAX; i++) {
    workstub->stub.diffs[i] = (u16)(oState->Levels[i+1].cnt2 - oState->Levels[i].cnt2);
  }
  workstub->worklength = oState->depth;

  /* This causes node count differences !!! */
  /* the only senseful way to fix this is increasing STUB_MAX */
  if (workstub->worklength > STUB_MAX) {
    workstub->worklength = STUB_MAX;
  }

  /* will be needed later ... new struct Stub with nodecount
  workstub->nodeshi = oState->nodeshi;
  workstub->nodeslo = oState->nodeslo;
  */
  
  if (oState->stub_error != STUB_OK) {
    /* stub produced an error at load time */
    workstub->worklength = oState->stub_error;
    /* workstub->nodes.hi = workstub->nodes.lo = 0; */
  }

  return CORE_S_OK;
}
#endif


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
  dispatch_table.name      = ogr_name;
  dispatch_table.core_id   = ogr_core_id;
  dispatch_table.get_size  = ogr_get_size;
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
