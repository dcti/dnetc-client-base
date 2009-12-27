/* 
 * Copyright distributed.net 1997-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
/*
const char *ogrng_cell_spe_wrapper_cpp(void) {
return "@(#)$Id: ogrng-cellv2-spe-wrapper.c,v 1.4 2009/12/27 13:52:24 andreasb Exp $"; }
*/

#include <spu_intrinsics.h>
#include "ccoreio.h"
#include "cputypes.h"
#include <spu_mfcio.h>

#define __CNTLZ__(n)  spu_extract(spu_cntlz(spu_promote(n, 0)), 0)

#include "ogrng-cell.h"
#include "ansi/first_blank.h"

/*
#ifdef __cplusplus
extern "C"
#endif
s32 CDECL SPE_CORE_FUNCTION(CORE_NAME) ( struct State*, int*, const unsigned char* );
*/

CellOGRCoreArgs myCellOGRCoreArgs __attribute__((aligned (128)));

int ogr_cycle_256_test(struct OgrState *oState, int *pnodes, /* const u16* */ u32 upchoose);

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
  STATIC_ASSERT(sizeof(CellOGRCoreArgs) == 16 + 2*16 + 8*16*29 + 16 + 16 + 16);
  STATIC_ASSERT(offsetof(CellOGRCoreArgs, state       ) == 16);
  STATIC_ASSERT(offsetof(CellOGRCoreArgs, state.Levels) == 16 + 32);
  STATIC_ASSERT(sizeof(u16) == 2); /* DMA fetches of pchoose */
  
  (void) speid; (void) envp;

  // One DMA used in program
  mfc_write_tag_mask(1<<DMA_ID);

  // Fetch arguments from main memory
  mfc_get(&myCellOGRCoreArgs, argp.a32[1], sizeof(CellOGRCoreArgs), DMA_ID, 0, 0);
  mfc_read_tag_status_all();

  s32 retval;
  /* check for memory corruption in incoming arguments */
  if (myCellOGRCoreArgs.sign1 != SIGN_PPU_TO_SPU_1)
  {
    retval = RETVAL_ERR_BAD_SIGN1;
    goto done;
  }
  if (myCellOGRCoreArgs.sign2 != SIGN_PPU_TO_SPU_2)
  {
    retval = RETVAL_ERR_BAD_SIGN2;
    goto done;
  }

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
  if (*pnodes) /* core will not handle nodes == 0 */
    myCellOGRCoreArgs.ret_depth = ogr_cycle_256_test(state, pnodes, upchoose);

  // Check for memory corruption after core exit
  if (myCellOGRCoreArgs.sign1 != SIGN_PPU_TO_SPU_1)
    retval = RETVAL_ERR_TRASHED_SIGN1;
  else if (myCellOGRCoreArgs.sign2 != SIGN_PPU_TO_SPU_2)
    retval = RETVAL_ERR_TRASHED_SIGN2;
  else
    retval = 0;

  update_groups_stats();

done:
  // Update changes in main memory
  myCellOGRCoreArgs.sign1 = SIGN_SPU_TO_PPU_1;
  myCellOGRCoreArgs.sign2 = SIGN_SPU_TO_PPU_2;
  mfc_put(&myCellOGRCoreArgs, argp.a32[1], sizeof(CellOGRCoreArgs), DMA_ID, 0, 0);
  mfc_read_tag_status_all();

  return retval; /* no status codes in ogr-ng, core info returned in ret_depth */
}

typedef vector unsigned int   v32_t;
typedef vector unsigned short v16_t;

#define vec_splat_u32(_a)   spu_splats((unsigned int)(_a))
#define vec_andc            spu_andc
#define vec_or              spu_or

#if PCHOOSE_FETCH_MODE == 3

#include <string.h>

#define GROUPS_COUNT   256
#define GROUPS_LENGTH  8   /* const - because 8 u16's can be stored in vector */
#define GROUP_ELEMENTS 32  /* const - because dist0 multiplied by 32 */

v16_t    group_keysvectors[GROUPS_COUNT];
v32_t    group_insertpos[GROUPS_COUNT]; /* store as vector for faster access */
u16      group_values[GROUPS_COUNT][GROUPS_LENGTH][GROUP_ELEMENTS];
#ifdef GET_CACHE_STATS
u32      group_length[GROUPS_COUNT];
#endif

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

#endif
