/*
 * Copyright distributed.net 1999-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogrng_codebase.cpp,v 1.13 2012/05/17 17:57:25 stream Exp $
 */

#include <string.h>   /* memset */

/* #define OGR_DEBUG           */ /* turn on miscellaneous debug information */ 
/* #define OGR_TEST_FIRSTBLANK */ /* test firstblank logic (table or asm)    */

#if defined(OGR_DEBUG) || defined(OGR_TEST_FIRSTBLANK)
#include <stdio.h>  /* printf for debugging */
#endif


/* ogr_cycle_256() method selection.
 OGROPT_ALTERNATE_CYCLE == 0 - Default (FLEGE) ogr_cycle_256().
 OGROPT_ALTERNATE_CYCLE == 1 - Fully customized ogr_cycle_256(). Implementors
 must still provide all necessary macros for use in ogr_create().
 */
#ifndef OGROPT_ALTERNATE_CYCLE
#define OGROPT_ALTERNATE_CYCLE                  0 /* 0/1 - 'no' (default)  */
#endif


/* Select the implementation for the LOOKUP_FIRSTBLANK function/macro.
    0 -> No hardware support
    1 -> Have ASM code, but still need the lookup table. You'll have to supply
         suitable code for the __CNTLZ_ARRAY_BASED(bitmap,bitarray) macro.
    2 -> Full featured ASM code. You'll have to supply code suitable for the
         __CNTLZ(bitmap) macro.
 
    NOTE : These macros shall arrange to return the number of leading 0 of
           "~bitmap" (i.e. the number of leading 1 of the "bitmap" argument)
           plus one. For a 32-bit bitmap argument, the valid range is [1; 33].
           Said otherwise :
           __CNTLZ(0xFFFFFFFF) == 32 or 33  (Implementation defined).
           __CNTLZ(0xFFFFFFFE) == 32
           __CNTLZ(0xFFFFA427) == 18
           __CNTLZ(0x00000000) ==  1
 
    TEST : Define OGR_TEST_FIRSTBLANK to test the code at run time.
*/
#ifndef OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM     0 /* 0-2 - 'no' (default)  */
#endif


/* ----------------------------------------------------------------------- */

#include "ansi/first_blank.h"
#include "ansi/ogrng_corestate.h"


static const int OGR[] = {
  /*  1 */    0,   1,   3,   6,  11,  17,  25,  34,  44,  55,
  /* 11 */   72,  85, 106, 127, 151, 177, 199, 216, 246, 283,
  /* 21 */  333, 356, 372, 425, 480, 492, 553, 585, 623
};


/* ----------------------------------------------------------------------- */

static int ogr_init(void);
static int ogr_getresult(void *state, void *result, int resultlen);
static int ogr_destroy(void *state);
static int ogr_cycle_entry(void *state, int *pnodes, int dummy);
static int ogr_create(void *input, int inputlen, void *state, int statelen,
                      int maxlen);
static int found_one(const struct OgrState *oState);


/* The work horse...
** If you need to optimize something, here is the only one.
*/
static int ogr_cycle_256(struct OgrState *oState, int *pnodes, const u16* pchoose);


#ifdef OGR_TEST_FIRSTBLANK
static int  __testFirstBlank(void);
#endif


/*
** The dispatch table.
*/
extern CoreDispatchTable * OGR_NG_GET_DISPATCH_TABLE_FXN (void);


#ifdef OGR_DEBUG
  /*
  ** Use DUMP_BITMAPS(depth, lev) or DUMP_RULER(state, depth) macros instead of
  ** calling the following functions directly, unless you want unpredictable
  ** results...
  */
  static void __dump(const struct OgrLevel *lev, int depth);
  static void __dump_ruler(const struct OgrState *oState, int depth);

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


/*-------------------------------------------------------------------------*/
/*
** found_one() - Assert the ruler is Golomb.
** This function is *REALLY* seldom used and it has no impact at all on
** benchmarks or runtime node rates.
**
** NOTE : Principle #1 states that we don't have to track distances larger
**    than half the length of the ruler.
*/

static int found_one(const struct OgrState *oState)
{
  register int i, j;
  u32 diffs[((1024 - OGRNG_BITMAPS_LENGTH) + 31) / 32];
  register int max = oState->max;
  register int maxdepth = oState->maxdepth;
  const struct OgrLevel *levels = &oState->Levels[0];

  for (i = max >> (5+1); i >= 0; --i)
    diffs[i] = 0;

  for (i = 1; i < maxdepth; i++) {
    register int marks_i = levels[i].mark;

    for (j = 0; j < i; j++) {
      register int diff = marks_i - levels[j].mark;
      if (diff <= OGRNG_BITMAPS_LENGTH)
        break;

      if (diff+diff <= max) {       /* Principle #1 */
        register u32 mask = 1 << (diff & 31);
        diff = (diff >> 5) - (OGRNG_BITMAPS_LENGTH / 32);
        if ((diffs[diff] & mask) != 0)
          return 0;                 /* Distance already taken = not Golomb */

        diffs[diff] |= mask;
      }
    }
  }
  return -1;      /* Is golomb */
}


/* ----------------------------------------------------------------------- */

static int ogr_init(void)
{
  /* Enforce hard limits and sizes */
  STATIC_ASSERT( OGR_MAX_MARKS <= 28 );
  STATIC_ASSERT( OGR_STUB_MAX <= 28 );
  STATIC_ASSERT( sizeof(struct OgrStub) == 60 );
  STATIC_ASSERT( sizeof(struct OgrWorkStub) == 64 );

  /* Be sure we have enough memory in 'problem' even with wildest compiler alignment rules */
  STATIC_ASSERT( sizeof(struct OgrState) <= OGRNG_PROBLEM_SIZE );

  if (CHOOSE_DIST_BITS != ogrng_choose_bits || CHOOSE_MARKS != ogrng_choose_marks)
  {
    /* Incompatible CHOOSE array - Give up */
    return CORE_E_INTERNAL;
  }

#if defined(OGR_TEST_FIRSTBLANK)
    if (__testFirstBlank())
      return CORE_E_INTERNAL;
  #endif

  return CORE_S_OK;
}


/* ----------------------------------------------------------------------- */

#define choose(dist,seg) pchoose[(dist >> (SCALAR_BITS-CHOOSE_DIST_BITS)) * 32 + (seg)]

/*
** This function is called each time a stub is (re)started, so optimizations
** are useless and not welcomed.
*/

static int ogr_create(void *input, int inputlen, void *state, int statelen,
                      int lastSegment)
{
  struct OgrState *oState;
  struct OgrWorkStub *workstub = (struct OgrWorkStub *)input;
  int    midseg_size;


  if (input == NULL || inputlen != sizeof(struct OgrWorkStub)
                    || (size_t)statelen < sizeof(struct OgrState)) {
    return CORE_E_INTERNAL;
  }

  if ( (oState = (struct OgrState *)state) == NULL) {
    return CORE_E_MEMORY;
  }

  memset(oState, 0, sizeof(struct OgrState));
  oState->maxdepth   = workstub->stub.marks;
  oState->maxdepthm1 = oState->maxdepth-1;
  oState->max        = OGR[oState->maxdepthm1];

  if (oState->maxdepth < OGR_NG_MIN || oState->maxdepth > OGR_NG_MAX) {
    return CORE_E_FORMAT;
  }  

  if (0 == ogr_check_cache(oState->maxdepth)) {
    return CORE_E_CORRUPTED;
  }


  /* 
  ** Mid-segment reduction (one or two segments)
  */
  midseg_size         = 2 - (oState->maxdepthm1 & 1);
  oState->half_depth  = (oState->maxdepthm1 - midseg_size) / 2;
  oState->half_depth2 = oState->half_depth + midseg_size;


  {
    int n, limit;
    struct OgrLevel *lev = &oState->Levels[1];
    int mark = 0;
    int depth = 1;
    u16* pchoose = precomp_limits[oState->maxdepth - OGR_NG_MIN].choose_array;
    SETUP_TOP_STATE(lev);

    n = workstub->worklength;
    if (n < workstub->stub.length) {
      n = workstub->stub.length;
    }
    if (n >= OGR_STUB_MAX) {
      return CORE_E_FORMAT;
    }

    limit = choose((SCALAR) 0, depth);
    oState->Levels[depth].limit = limit;
    while (depth <= n) {
      int s = workstub->stub.diffs[depth-1];

      if ((mark += s) > limit) {
        return CORE_E_STUB;
      }

      while (s >= SCALAR_BITS) {
        if (s == SCALAR_BITS && (comp0 & 1)) {
          return CORE_E_STUB;         /* invalid location */
        }
        COMP_LEFT_LIST_RIGHT_WORD(lev);
        s -= SCALAR_BITS;
      }

      if (s > 0) {
        if ( (comp0 & ((SCALAR)1 << (SCALAR_BITS-s))) ) {
          return CORE_E_STUB;         /* invalid location */
        }
        COMP_LEFT_LIST_RIGHT(lev, s);
      }

      lev->mark = mark;

      PUSH_LEVEL_UPDATE_STATE(lev);
      lev++;
      depth++;

      /* Setup the next level */
      limit = choose(dist0, depth);

      if (depth > oState->half_depth && depth <= oState->half_depth2) {
        int temp = oState->max - 1 - oState->Levels[oState->half_depth].mark;
        
        /* The following part is only relevant for rulers with an odd number of
         ** marks. If the number of marks is even (as for OGR-26), then the
         ** condition is always false.
         */
        if (depth < oState->half_depth2) {
          temp -= LOOKUP_FIRSTBLANK(dist0);
        }
        
        if (limit > temp) {
          limit = temp;
        }
      }
      
      lev->limit = limit;
      lev->mark  = mark;
    }
    SAVE_FINAL_STATE(lev);
    oState->depth = depth - 1;
  }

  oState->startdepth = workstub->stub.length;
  oState->stopdepth  = (lastSegment != 0) ? oState->startdepth - 1 : oState->startdepth;
  return CORE_S_OK;
}


/* ----------------------------------------------------------------------- */

/* Note : pchoose points to a large array of pre-computed limits. The datas
** are organized as in : u16 pchoose[1 << CHOOSE_DIST_BITS][32]
*/
#if !defined(OGROPT_ALTERNATE_CYCLE) || (OGROPT_ALTERNATE_CYCLE == 0)
static int ogr_cycle_256(struct OgrState *oState, int *pnodes, const u16* pchoose)
{
  struct OgrLevel *lev = &oState->Levels[oState->depth];
  int depth       = oState->depth;
  int maxlen_m1   = oState->max - 1;
  int nodes       = *pnodes;

  SETUP_TOP_STATE(lev);

  do {
    int limit = lev->limit;
    int mark  = lev->mark;

    for (;;) {
      if (comp0 < (SCALAR)~1) {
        int s = LOOKUP_FIRSTBLANK(comp0);

        if ((mark += s) > limit) {
          break;
        }
        COMP_LEFT_LIST_RIGHT(lev, s);
      }
      else {         /* s >= SCALAR_BITS */
        if ((mark += SCALAR_BITS) > limit) {
          break;
        }
        if (comp0 == (SCALAR)~0) {
          COMP_LEFT_LIST_RIGHT_WORD(lev);
          continue;
        }
        COMP_LEFT_LIST_RIGHT_WORD(lev);
      }

      lev->mark = mark;
      if (depth == oState->maxdepthm1) {
        goto exit;         /* Ruler found */
      }

      /* Update the bitmaps for the next level */
      PUSH_LEVEL_UPDATE_STATE(lev);
      ++lev;
      ++depth;
      
      /* Compute the maximum position for the next level */
      limit = choose(dist0, depth);

      if (depth > oState->half_depth && depth <= oState->half_depth2) {
        int temp = maxlen_m1 - oState->Levels[oState->half_depth].mark;

        /* The following part is only relevant for rulers with an odd number of
        ** marks. If the number of marks is even (as for OGR-26), then the
        ** condition is always false.
        */
        if (depth < oState->half_depth2) {
          temp -= LOOKUP_FIRSTBLANK(dist0);
        }

        if (limit > temp) {
          limit = temp;
        }
      }

      lev->limit = limit;

      if (--nodes <= 0) {
        lev->mark = mark;
        goto exit;
      }
    } /* for (;;) */
    --lev;
    --depth;
    POP_LEVEL(lev);
  } while (depth > oState->stopdepth);

exit:
  SAVE_FINAL_STATE(lev);
  *pnodes -= nodes;
  return depth;
}
#endif  /* OGROPT_ALTERNATE_CYCLE */


/* ----------------------------------------------------------------------- */

/* Note to porters :
** The new implementation pulls the call to found_one() out of the main loop.
** Optimizations shall focus on ogr_cycle_256() only.
*/
static int ogr_cycle_entry(void *state, int *pnodes, int with_time_constraints)
{
  int retval = CORE_S_CONTINUE;
  struct OgrState *oState = (struct OgrState *)state;
  u16* pchoose = precomp_limits[oState->maxdepth - OGR_NG_MIN].choose_array;
  int safesize = OGRNG_BITMAPS_LENGTH * 2;
  int nodesDone;

  // Bug #4076 : Make sure a forked process has a chance to initialize itself.
  if (pchoose == NULL) {
    if (0 == ogr_check_cache(oState->maxdepth)) {
      return CORE_E_CORRUPTED;
    }
    pchoose = precomp_limits[oState->maxdepth - OGR_NG_MIN].choose_array;
  }
  
  /* Now that the core always exits when the specified number of nodes has
  ** been processed, the "with_time_constraints" setting is obsolete.
  */
  with_time_constraints = with_time_constraints;
  ++oState->depth;

  /* Invoke the main core.
  ** Except for OGR-26 (and shorter rulers), the core may still find rulers
  ** that are not Golomb. This loop asserts the Golombness of any ruler found
  ** when that makes sense.
  */
  for (nodesDone = 0;;) {

    int nodes, nodesCopy;

    nodes = nodesCopy = *pnodes - nodesDone;

    oState->depth = ogr_cycle_256(oState, &nodes, pchoose);
    nodesDone += nodes;

    /* Ruler is not complete if all nodes were exhausted.
    ** In this case last action was PUSH_LEVEL and last diff is zero.
    */
    if (oState->depth == oState->maxdepthm1 && nodes != nodesCopy) {
      /* For long rulers, check their golombness first */
      if (oState->Levels[oState->depth].mark > safesize && found_one(oState) == 0) {
        continue;
      }
      /* Simulate completion of the stub so ogr_getresult() will restore combined one */
      oState->depth = oState->stopdepth;
      retval = CORE_S_SUCCESS;
    }
    else if (oState->depth <= oState->stopdepth) {
      retval = CORE_S_OK;
    }
    else {
      retval = CORE_S_CONTINUE;
    }
    break;
  }

  *pnodes = nodesDone;
  --oState->depth;
  return retval;
}

/* ----------------------------------------------------------------------- */

static int ogr_getresult(void *state, void *result, int resultlen)
{
  struct OgrState *oState = (struct OgrState *)state;
  struct OgrWorkStub *workstub = (struct OgrWorkStub *)result;
  int i;

  if (resultlen != sizeof(struct OgrWorkStub)) {
    return CORE_E_INTERNAL;
  }

  workstub->stub.marks = (u16)oState->maxdepth;
  workstub->stub.length = (u16)oState->startdepth;
  for (i = 0; i < OGR_STUB_MAX; i++) {
    workstub->stub.diffs[i] = (u16)(oState->Levels[i+1].mark - oState->Levels[i].mark);
  }
  workstub->worklength = (u16)oState->depth;

  if (workstub->worklength > OGR_STUB_MAX) {
    workstub->worklength = OGR_STUB_MAX;
  }

  // Restore 'combined' stubs to their initial definition.
  if (oState->depth < oState->stopdepth && workstub->collapsed > 0) {
    workstub->stub.diffs[workstub->stub.length - 1] = workstub->collapsed;
  }

  return (ogr_check_cache(oState->maxdepth)) ? CORE_S_OK : CORE_E_CORRUPTED;
}


/* ----------------------------------------------------------------------- */

static int ogr_destroy(void *state)
{
  struct OgrState *oState = (struct OgrState*) state;

  return (ogr_check_cache(oState->maxdepth)) ? CORE_S_OK : CORE_E_CORRUPTED;
}


/* ----------------------------------------------------------------------- */

CoreDispatchTable * OGR_NG_GET_DISPATCH_TABLE_FXN (void)
{
  static CoreDispatchTable dispatch_table;
  dispatch_table.init      = ogr_init;
  dispatch_table.create    = ogr_create;
  dispatch_table.cycle     = ogr_cycle_entry;
  dispatch_table.getresult = ogr_getresult;
  dispatch_table.destroy   = ogr_destroy;
  return &dispatch_table;
}

/* ----------------------------------------------------------------------- */

#if defined(OGR_DEBUG)
static void __dump(const struct OgrLevel *lev, int depth)
{
  printf("--- depth %d, limit %d\n", depth, lev->limit);

#if (OGRNG_BITMAPS_WORDS == 2)
  #if defined(__VEC__) || defined(__ALTIVEC__)
    printf("list=%:08vlx %:08vlx\n", lev->list[0], lev->list[1]);
    printf("dist=%:08vlx %:08vlx\n", lev->dist[0], lev->dist[1]);
    printf("comp=%:08vlx %:08vlx\n", lev->comp[0], lev->comp[1]);
  #else
    #error fixme : Debugging vector bitmaps without vector support ?!?
  #endif
#elif (SCALAR_BITS == 32)
  printf("list=%08x %08x %08x %08x %08x %08x %08x %08x\n",
      lev->list[0], lev->list[1], lev->list[2], lev->list[3],
      lev->list[4], lev->list[5], lev->list[6], lev->list[7]);
  printf("dist=%08x %08x %08x %08x %08x %08x %08x %08x\n",
      lev->dist[0], lev->dist[1], lev->dist[2], lev->dist[3],
      lev->dist[4], lev->dist[5], lev->dist[6], lev->dist[7]);
  printf("comp=%08x %08x %08x %08x %08x %08x %08x %08x\n",
      lev->comp[0], lev->comp[1], lev->comp[2], lev->comp[3],
      lev->comp[4], lev->comp[5], lev->comp[6], lev->comp[7]);

#elif (!defined(SIZEOF_SHORT) || (SIZEOF_SHORT < 8)) \
   && (!defined(SIZEOF_INT)   || (SIZEOF_INT   < 8)) \
   && (!defined(SIZEOF_LONG)  || (SIZEOF_LONG  < 8))
  // Assume ui64 is unsigned long long int
  printf("list=%016llx %016llx %016llx %016llx\n",
      lev->list[0], lev->list[1], lev->list[2], lev->list[3]);
  printf("dist=%016llx %016llx %016llx %016llx\n",
      lev->dist[0], lev->dist[1], lev->dist[2], lev->dist[3]);
  printf("comp=%016llx %016llx %016llx %016llx\n",
      lev->comp[0], lev->comp[1], lev->comp[2], lev->comp[3]);
#else
  printf("list=%016lx %016lx %016lx %016lx\n",
      lev->list[0], lev->list[1], lev->list[2], lev->list[3]);
  printf("dist=%016lx %016lx %016lx %016lx\n",
      lev->dist[0], lev->dist[1], lev->dist[2], lev->dist[3]);
  printf("comp=%016lx %016lx %016lx %016lx\n",
      lev->comp[0], lev->comp[1], lev->comp[2], lev->comp[3]);
#endif
}


static void __dump_ruler(const struct OgrState *oState, int depth)
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
