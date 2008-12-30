/* 
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *ogrng_cell_spe_wrapper_cpp(void) {
return "@(#)$Id: ogrng-cell-spe-wrapper.c,v 1.5 2008/12/30 20:58:43 andreasb Exp $"; }

#include <spu_intrinsics.h>
#include "ccoreio.h"
#include "cputypes.h"
#include <spu_mfcio.h>

#define __CNTLZ__(n)  spu_extract(spu_cntlz(spu_promote(n, 0)), 0)

/*
 * Special version of LOOKUP_FIRSTBLANK which will never produce results
 * greater then SCALAR_BITS (32)
 */
#define LOOKUP_FIRSTBLANK_32(n) __CNTLZ__((~(n)) >> 1)

#include "ogrng-cell.h"
#include "ansi/first_blank.h"

#ifndef CORE_NAME
#error CORE_NAME not defined
#endif

#define SPE_CORE_FUNCTION(name) SPE_CORE_FUNCTION2(name)
#define SPE_CORE_FUNCTION2(name) ogrng_cycle_ ## name ## _spe_core

#ifdef __cplusplus
extern "C"
#endif
// s32 CDECL SPE_CORE_FUNCTION(CORE_NAME) ( struct State*, int*, const unsigned char* );

CellOGRCoreArgs myCellOGRCoreArgs __attribute__((aligned (128)));

int ogr_cycle_256(struct OgrState *oState, int *pnodes, /* const u16* */ u32 upchoose);

#define DMA_ID  31

/*
 * Select method to cache parts of huge pchoose array in local storage:
 *   0 - no caching, get one element per call (very slow).
 *   1 - also something trivial (may be broken now).
 *   2 - use hash function to divide search space on small arrays of 'keys',
 *       then use linear search to find 'key' in this array. Circular cache
 *       replacement policy.
 *   3 - improved version of (2), keys are stored in vector and compared
 *       in single vector SPU operation. No loops for search at all.
 */
#define PCHOOSE_FETCH_MODE  3

#ifdef GET_CACHE_STATS
  #define UPDATE_STAT(item, n) myCellOGRCoreArgs.cache_##item += n
#else
  #define UPDATE_STAT(item, n)
#endif

static void cleargroups(void), update_groups_stats(void);

int main(unsigned long long speid, addr64 argp, addr64 envp)
{
  // Check size of structures, these offsets must match assembly
  STATIC_ASSERT(sizeof(struct OgrLevel) == 6*16+16+16);
  STATIC_ASSERT(sizeof(struct OgrState) == 2*16 + 8*16*29);
  STATIC_ASSERT(sizeof(CellOGRCoreArgs) == 2*16 + 8*16*29 + 16 + 16 + 16);
  STATIC_ASSERT(offsetof(CellOGRCoreArgs, state.Levels) == 32);
  STATIC_ASSERT(sizeof(u16) == 2); /* DMA fetches of pchoose */
  
  (void) speid; (void) envp;

  // One DMA used in program
  mfc_write_tag_mask(1<<DMA_ID);

  // Fetch arguments from main memory
  mfc_get(&myCellOGRCoreArgs, argp.a32[1], sizeof(CellOGRCoreArgs), DMA_ID, 0, 0);
  mfc_read_tag_status_all();

  // Prepare arguments to be passed to the core
  struct OgrState* state = &myCellOGRCoreArgs.state;
  int* pnodes   = &myCellOGRCoreArgs.pnodes;
  u32  upchoose = myCellOGRCoreArgs.upchoose;
  
  static int cached_maxdepth;
  if (state->maxdepth != cached_maxdepth)
  {
    cached_maxdepth = state->maxdepth;
    cleargroups();
  }

  // Call the core
//  s32 retval = SPE_CORE_FUNCTION(CORE_NAME) (state, pnodes, ogr_choose_dat);
  s32 retval;
  myCellOGRCoreArgs.ret_depth = ogr_cycle_256(state, pnodes, upchoose);
  retval = 0;

  update_groups_stats();

  // Update changes in main memory
  mfc_put(&myCellOGRCoreArgs, argp.a32[1], sizeof(CellOGRCoreArgs), DMA_ID, 0, 0);
  mfc_read_tag_status_all();

  return retval; /* no status codes in ogr-ng, core info returned in ret_depth */
}

typedef vector unsigned int   v32_t;
typedef vector unsigned short v16_t;

#define vec_splat_u32(_a)   spu_splats((unsigned int)(_a))
#define vec_andc            spu_andc
#define vec_or              spu_or

/*
** Define the local variables used for the top recursion state
*/
#define comp0 spu_extract(compV0, 0)
#define dist0 spu_extract(distV0, 0)
#define SETUP_TOP_STATE(lev)                           \
  /* SCALAR comp0, dist0; */                           \
  v32_t compV0, compV1;                                \
  v32_t listV0, listV1;                                \
  v32_t distV0, distV1;                                \
  v32_t V_ZERO = vec_splat_u32(0);                     \
  v32_t V_ONES = vec_splat_u32(~0);                    \
  v32_t newbit = V_ZERO;                               \
  listV0 = lev->list[0].v;                             \
  listV1 = lev->list[1].v;                             \
  distV0 = lev->dist[0].v;                             \
  distV1 = lev->dist[1].v;                             \
  compV0 = lev->comp[0].v;                             \
  compV1 = lev->comp[1].v;                             \
  /* dist0 = spu_extract(distV0, 0); */                \
  /* comp0 = spu_extract(compV0, 0); */                \
  if (depth < maxdepthm1) {                            \
    newbit = vec_splat_u32(1);                         \
  }

/*
** Rotate/shift full vector by specific number of bits
*/
#define full_rl(v, n)    spu_rlqw(spu_rlqwbytebc(v, n), n)
#define full_sl(v, n)    spu_slqw(spu_slqwbytebc(v, n), n)
// #define full_sr(v, n)    spu_rlmaskqw(spu_rlmaskqwbytebc(v, 7-n), 0-n)

#define COMP_LEFT_LIST_RIGHT(lev, s)                   \
{                                                      \
  v32_t selMask;                                       \
  int   inv_s = 128-s;                                 \
  selMask = full_sl(V_ONES, inv_s);                    \
  compV0  = spu_sel(compV0, compV1, selMask);          \
  compV0  = full_rl(compV0, s);                        \
  compV1  = full_sl(compV1, s);                        \
  /* comp0   = spu_extract(compV0, 0); */              \
  selMask = full_sl(V_ONES, s);                        \
  listV1  = spu_sel(listV0, listV1, selMask);          \
  /* inv_s   = 128 - s;  */                            \
  listV1  = full_rl(listV1, inv_s);                    \
  listV0  = spu_sel(newbit, listV0, selMask);          \
  listV0  = full_rl(listV0, inv_s);                    \
  newbit  = V_ZERO;                                    \
}
  
#define COMP_LEFT_LIST_RIGHT_WORD(lev)  COMP_LEFT_LIST_RIGHT(lev, 32)


/*
** Set the current mark and push a level to start a new mark
*/
#define PUSH_LEVEL_UPDATE_STATE(lev)    \
  lev->list[0].v = listV0;              \
  distV0 = vec_or(distV0, listV0);      \
  lev->comp[0].v = compV0;              \
  compV0 = vec_or(compV0, distV0);      \
  lev->list[1].v = listV1;              \
  distV1 = vec_or(distV1, listV1);      \
  lev->comp[1].v = compV1;              \
  compV1 = vec_or(compV1, distV1);      \
  /* dist0  = spu_extract(distV0, 0); */     \
  /* comp0  = spu_extract(compV0, 0); */     \
  newbit = vec_splat_u32(1);


/*
** Pop a level to continue work on previous mark
*/
#define POP_LEVEL(lev)                  \
  listV0 = lev->list[0].v;              \
  listV1 = lev->list[1].v;              \
  distV0 = vec_andc(distV0, listV0);    \
  distV1 = vec_andc(distV1, listV1);    \
  compV0 = lev->comp[0].v;              \
  compV1 = lev->comp[1].v;              \
  /* comp0  = spu_extract(compV0, 0); */     \
  newbit = V_ZERO;


/*
** Save the local state variables
*/
#define SAVE_FINAL_STATE(lev)           \
  lev->list[0].v = listV0;              \
  lev->dist[0].v = distV0;              \
  lev->comp[0].v = compV0;              \
  lev->list[1].v = listV1;              \
  lev->dist[1].v = distV1;              \
  lev->comp[1].v = compV1;

#if PCHOOSE_FETCH_MODE == 0
static inline unsigned direct_dma_fetch(u32 upchoose, u32 index)
{
  u16 tempdma[8]; /* use array to fetch 16 bytes (transfer must be aligned) */
  STATIC_ASSERT(sizeof(tempdma) == 16);

  upchoose += index * 2;        /* get true full address in u16 *pchoose */
  index = (upchoose & 15) / 2;  /* index of u16 element within 16 bytes  */
  upchoose &= ~15;              /* align transfer to 16 bytes            */
  mfc_get(tempdma, upchoose, sizeof(tempdma), DMA_ID, 0, 0);
  mfc_read_tag_status_all();
  return tempdma[index];
}

static void cleargroups(void) {}

#endif

#if PCHOOSE_FETCH_MODE == 1

#define GROUPS_COUNT  256
#define GROUPS_LENGTH 64

static unsigned group_sizes[GROUPS_COUNT];
static struct element
{
   unsigned index, value;
} group_arrays[GROUPS_COUNT][GROUPS_LENGTH];

static inline unsigned direct_dma_fetch___(u32 upchoose, u32 index)
{
  u16 tempdma[8]; /* use array to fetch 16 bytes (transfer must be aligned) */
  STATIC_ASSERT(sizeof(tempdma) == 16);

  upchoose += index * 2;        /* get true full address in u16 *pchoose */
  index = (upchoose & 15) / 2;  /* index of u16 element within 16 bytes  */
  upchoose &= ~15;              /* align transfer to 16 bytes            */
  mfc_get(tempdma, upchoose, sizeof(tempdma), DMA_ID, 0, 0);
  mfc_read_tag_status_all();
  return tempdma[index];
}

static inline unsigned direct_dma_fetch(u32 upchoose, u32 index)
{
  unsigned group  = index & (GROUPS_COUNT-1);
  unsigned length = group_sizes[group];
  unsigned i;
  
  for (i = 0; i < length; i++)
    if (group_arrays[group][i].index == index)
       return group_arrays[group][i].value;
   
  if (length == GROUPS_LENGTH)
    length--;
  group_arrays[group][length].index = index;
  group_arrays[group][length].value = i = direct_dma_fetch___(upchoose, index);
  group_sizes[group] = ++length;
  return i;
}
#endif

#if PCHOOSE_FETCH_MODE == 2

#define GROUPS_COUNT   256
#define GROUPS_LENGTH  8
#define GROUP_ELEMENTS 32  /* const - because dist0 multiplied by 32 */

static unsigned group_sizes[GROUPS_COUNT];
static u32      group_keys[GROUPS_COUNT][GROUPS_LENGTH];
static u16      group_values[GROUPS_COUNT][GROUPS_LENGTH][GROUP_ELEMENTS];

static inline unsigned direct_dma_fetch(u32 upchoose, u32 group_of32, u32 index_in32)
{
  unsigned hash;
  unsigned fulllength, grplength;
  unsigned newpos;
  unsigned i;
  u16     *pvalues;
  
  STATIC_ASSERT(sizeof(group_values) == GROUPS_COUNT * GROUPS_LENGTH * GROUP_ELEMENTS * 2);
  
  upchoose += group_of32 * GROUP_ELEMENTS * 2; /* get true full address of group start in u16 *pchoose */
  
  hash       = (group_of32 ^ (group_of32 >> 8)) & (GROUPS_COUNT-1);
  fulllength = group_sizes[hash];
  grplength  = fulllength > GROUPS_LENGTH ? GROUPS_LENGTH : fulllength;
  
  for (i = 0; i < grplength; i++)
  {
    if (group_keys[hash][i] == upchoose)
    {
      UPDATE_STAT(hits, 1);
      UPDATE_STAT(search_iters, i+1);
      return group_values[hash][i][index_in32];
    }
  }
  UPDATE_STAT(misses, 1);
  UPDATE_STAT(search_iters, i);

#if 0
  if (fulllength >= GROUPS_LENGTH)
  {
    UPDATE_STAT(purges, 1);
  }
#else
  UPDATE_STAT(purges, (fulllength >= GROUPS_LENGTH));
#endif
  newpos  = fulllength & (GROUPS_LENGTH - 1);
  pvalues = group_values[hash][newpos];
  mfc_get(pvalues, upchoose, GROUP_ELEMENTS * 2, DMA_ID, 0, 0);
  group_keys[hash][newpos] = upchoose;
  group_sizes[hash] = ++fulllength;
  mfc_read_tag_status_all();
  return pvalues[index_in32];
}

static void cleargroups(void)
{
  unsigned i;

  for (i = 0; i < GROUPS_COUNT; i++)
    group_sizes[i] = 0;
}

static void update_groups_stats(void)
{
#ifdef GET_CACHE_STATS
  unsigned i;
  
  myCellOGRCoreArgs.cache_maxlen = GROUPS_COUNT * GROUPS_LENGTH;
  for (i = 0; i < GROUPS_COUNT; i++)
    myCellOGRCoreArgs.cache_curlen += (group_sizes[i] > GROUPS_LENGTH ? GROUPS_LENGTH : group_sizes[i]);
#endif
}

#define choose(dist, seg) direct_dma_fetch(upchoose, ((dist) >> (SCALAR_BITS-CHOOSE_DIST_BITS)), (seg))

#endif


#if PCHOOSE_FETCH_MODE == 3

#include <string.h>

#define GROUPS_COUNT   256
#define GROUPS_LENGTH  8   /* const - because 8 u16's can be stored in vector */
#define GROUP_ELEMENTS 32  /* const - because dist0 multiplied by 32 */

static v16_t    group_keysvectors[GROUPS_COUNT];
static v32_t    group_insertpos[GROUPS_COUNT]; /* store as vector for faster access */
static u16      group_values[GROUPS_COUNT][GROUPS_LENGTH][GROUP_ELEMENTS];
#ifdef GET_CACHE_STATS
static u32      group_length[GROUPS_COUNT];
#endif

/*
 * 'group' is a 16-bit value (top 16 bits of dist0) so 8 id's (keys) can be stored
 * in one SPU vector. When looking for specific key, SPU vector operations can be used
 * to compare all of them in parallel and find index of first match:
 *
 *    Compare Halfwords => 0xFFFF at matched positions
 *    Gather Bits From Halfwords => bits 24-31 in PS are set to '1' if corresp. vector matched.
 *    Count Leading Zeros => result is 24 if vector #0 matched, 25 if #1, ..., 32 if none.
 *    By substracting 24 from last value, we'll get index of matched key without single jump.
 */
static inline unsigned direct_dma_fetch(u32 upchoose, u32 group_of32, u32 index_in32)
{
  unsigned hash;
  u16     *pvalues;
  v16_t    keyvector;
  u32      eqbits, element;
  
  STATIC_ASSERT(sizeof(group_values) == GROUPS_COUNT * GROUPS_LENGTH * GROUP_ELEMENTS * 2);
  
  hash      = (group_of32 ^ (group_of32 >> 8)) & (GROUPS_COUNT-1);
  keyvector = group_keysvectors[hash];
  eqbits    = spu_extract(spu_cntlz(spu_gather(spu_cmpeq(keyvector, (u16)group_of32))), 0);
  element   = eqbits - 24;
  if (element == 8) /* 32 zeros => Out of bounds => no match */
  {
    v32_t tempvect = group_insertpos[hash];
    element        = spu_extract(tempvect, 0) & (GROUPS_LENGTH - 1);

    pvalues = group_values[hash][element];
    mfc_get(pvalues, upchoose + group_of32 * GROUP_ELEMENTS * 2, GROUP_ELEMENTS * 2, DMA_ID, 0, 0);

    group_insertpos[hash]   = spu_add(tempvect, 1);
    group_keysvectors[hash] = spu_insert((u16)group_of32, keyvector, element);

#ifdef GET_CACHE_STATS
    UPDATE_STAT(misses, 1);
    if (group_length[hash] != GROUPS_LENGTH)
      group_length[hash]++;
    else
      UPDATE_STAT(purges, 1);
#endif
    
    mfc_read_tag_status_all();
    
    return pvalues[index_in32];
  }

  UPDATE_STAT(hits, 1);

  return group_values[hash][element][index_in32];
}

static void cleargroups(void)
{
  unsigned i;

  for (i = 0; i < GROUPS_COUNT; i++)
  {
    group_keysvectors[i] = spu_splats((u16) 0);
    group_insertpos[i]   = spu_splats((u32) 0);
#ifdef GET_CACHE_STATS
    group_length[i]      = 0;
#endif
  }
  /* All vectors now points to group0, so fill all entries with true data for group 0 */
  mfc_get(group_values[0][0], myCellOGRCoreArgs.upchoose, GROUP_ELEMENTS * 2, DMA_ID, 0, 0);
  mfc_read_tag_status_all();
  for (i = 1; i < GROUPS_COUNT * GROUPS_LENGTH; i++)
    memcpy(group_values[0][i], group_values[0][0], GROUP_ELEMENTS * 2);
}

static void update_groups_stats(void)
{
#ifdef GET_CACHE_STATS
  unsigned i;
  
  myCellOGRCoreArgs.cache_maxlen = GROUPS_COUNT * GROUPS_LENGTH;
  for (i = 0; i < GROUPS_COUNT; i++)
    myCellOGRCoreArgs.cache_curlen += group_length[i];
#endif
}

#define choose(dist, seg) direct_dma_fetch(upchoose, ((dist) >> (SCALAR_BITS-CHOOSE_DIST_BITS)), (seg))

#endif

//#define getchoose(index)  direct_dma_fetch(upchoose, (index))

//#define choose(dist,seg) getchoose( (dist >> (SCALAR_BITS-CHOOSE_DIST_BITS)) * 32 + (seg) )

int ogr_cycle_256(struct OgrState * oState, int * pnodes, /* const u16* */ u32 upchoose)
{
  int nodes       = *pnodes;
  int maxlen_m1   = oState->max - 1;
  int maxdepthm1  = oState->maxdepthm1;
  int half_depth  = oState->half_depth;
  int half_depth2 = oState->half_depth2;
  int stopdepth   = oState->stopdepth;
  int depth       = oState->depth;
  struct OgrLevel *lev = &oState->Levels[depth];

  SETUP_TOP_STATE(lev);

  do {
    int limit = spu_extract(lev->limit, 0);
    int mark  = spu_extract(lev->mark, 0);

    for (;;) {
#if 0
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
#else
      int s;
      v32_t save_compV0;
      s = LOOKUP_FIRSTBLANK_32(comp0);
      if ((mark += s) > limit) {
        break;
      }
      save_compV0 = compV0;
      COMP_LEFT_LIST_RIGHT(lev, s);
      if (spu_extract(save_compV0, 0) == (SCALAR)~0) {
        continue;
      }
#endif

      lev->mark = spu_promote(mark, 0);
      if (depth == maxdepthm1) {
        goto exit;         /* Ruler found */
      }

      /* Update the bitmaps for the next level */
      PUSH_LEVEL_UPDATE_STATE(lev);
      ++lev;
      ++depth;
      
      /* Compute the maximum position for the next level */
      limit = choose(dist0, depth);

      if (depth > half_depth && depth <= half_depth2) {
        int temp = maxlen_m1 - spu_extract(oState->Levels[half_depth].mark, 0);

        if (limit > temp) {
          limit = temp;
        }

        /* The following part is only relevant for rulers with an odd number of
        ** marks. If the number of marks is even (as for OGR-26), then the
        ** condition is always false.
        ** LOOKUP_FIRSTBLANK(0xFF..FF) shall return the total number of bits
        ** set plus one. If not, selftest #32 will fail.
        */
        if (depth < half_depth2) {
          #if (SCALAR_BITS <= 32)
          limit -= LOOKUP_FIRSTBLANK(dist0);
          #else
          // Reduce the resolution for larger datatypes, otherwise the final
          // node count may not match that of 32-bit cores.
          limit -= LOOKUP_FIRSTBLANK(dist0 & -((SCALAR)1 << 32));
          #endif
        }
      }
      lev->limit = spu_promote(limit, 0);

      if (--nodes <= 0) {
        lev->mark = spu_promote(mark, 0);
        goto exit;
      }
    } /* for (;;) */
    --lev;
    --depth;
    POP_LEVEL(lev);
  } while (depth > stopdepth);

exit:
  SAVE_FINAL_STATE(lev);
  *pnodes -= nodes;
  return depth;
}
