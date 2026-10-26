/*
 * Copyright distributed.net 1999-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the "list", "dist" and "comp" bitmap is made of
 *  one 32-bit scalar part (left side), and two 128-bit vector parts, thus the
 *  "hybrid" name.
 *
 * $Id: ogrng-vec.cpp,v 1.7 2009/12/27 13:52:28 andreasb Exp $
*/

#include "ansi/ogrng.h"

#ifndef __SPU__
#if !defined(__VEC__) && !defined(__ALTIVEC__)
  #error fixme : No AltiVec support.
#endif

#if defined (__GNUC__) && !defined(__APPLE_CC__) && (__GNUC__ >= 3)
#include <altivec.h>
#endif
#endif // __SPU__

/*
** Bitmaps built with 128-bit vectors
*/

typedef union {
  vector unsigned int v;
  u32                 u[4];
} BMAP;                     /* Basic type for each bitmap array */

typedef u32 SCALAR;         /* Basic type for scalar operations on bitmaps */
#define OGRNG_BITMAPS_WORDS ((OGRNG_BITMAPS_LENGTH + 127) / 128)
#define SCALAR_BITS  32


#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM     2 /* 0-2 - '100% asm'      */
#define OGROPT_ALTERNATE_CYCLE                  1 /* 0-1 - '100% asm'      */

/*
** Define the name of the dispatch table.
*/
#ifndef IMPLEMENT_CELL_CORES
#define OGR_NG_GET_DISPATCH_TABLE_FXN    vec_ogrng_get_dispatch_table
#endif


/*========================================================================*/

#include "asm-ppc.h"

#if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && defined (__CNTLZ__)
  #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
#endif

#ifndef __SPU__

typedef vector unsigned char v8_t;
typedef vector unsigned int v32_t;


/*
** Define the local variables used for the top recursion state
*/
#define SETUP_TOP_STATE(lev)                           \
  v8_t vecShift[32];                                   \
  SCALAR comp0, dist0;                                 \
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
  dist0 = lev->dist[0].u[0];                           \
  comp0 = lev->comp[0].u[0];                           \
  { /* Initialize vecShift[] */                        \
    vector unsigned char val = vec_splat_u8(0);        \
    vector unsigned char one = vec_splat_u8(1);        \
    int i;                                             \
    for (i = 0; i < 32; i++) {                         \
      vecShift[i] = val;                               \
      val = vec_add(val, one);                         \
    }                                                  \
  }                                                    \
  if (depth < oState->maxdepthm1) {                    \
    newbit = vec_splat_u32(1);                         \
  }

        
/*
** Shift the list to add or move a mark
*/
#define COMP_LEFT_LIST_RIGHT(lev, s)                   \
{                                                      \
  v32_t shift_l = (v32_t) vecShift[s];                 \
  v32_t shift_r = vec_sub(V_ZERO, shift_l);            \
  v32_t mask_l  = vec_sr(V_ONES, shift_l);             \
  v32_t mask_r  = vec_sl(V_ONES, shift_l);             \
  v32_t temp1, temp2;                                  \
  temp1 = vec_sld(compV0, compV1, 4);                  \
  temp2 = vec_sld(listV0, listV1, 12);                 \
  temp1 = vec_sel(temp1, compV0, mask_l);              \
  temp2 = vec_sel(temp2, listV1, mask_r);              \
  compV0 = vec_rl(temp1, shift_l);                     \
  listV1 = vec_rl(temp2, shift_r);                     \
  lev->comp[0].v = compV0;                             \
  temp2 = vec_sld(newbit, listV0, 12);                 \
  temp1 = vec_slo(compV1, (v8_t) shift_l);             \
  temp2 = vec_sel(temp2, listV0, mask_r);              \
  compV1 = vec_sll(temp1, (v8_t) shift_l);             \
  listV0 = vec_rl(temp2, shift_r);                     \
  newbit = V_ZERO;                                     \
  comp0 = lev->comp[0].u[0];                           \
}


/*
** shift by word size
*/
#define COMP_LEFT_LIST_RIGHT_WORD(lev)  \
  compV0 = vec_sld(compV0, compV1, 4);  \
  lev->comp[0].v = compV0;              \
  compV1 = vec_sld(compV1, V_ZERO, 4);  \
  listV1 = vec_sld(listV0, listV1, 12); \
  listV0 = vec_sld(newbit, listV0, 12); \
  comp0  = lev->comp[0].u[0];           \
  newbit = V_ZERO;


/*
** Set the current mark and push a level to start a new mark
*/
#define PUSH_LEVEL_UPDATE_STATE(lev)    \
  lev->list[0].v = listV0;              \
  distV0 = vec_or(distV0, listV0);      \
  lev->dist[0].v = distV0;              \
  compV0 = vec_or(compV0, distV0);      \
  lev->list[1].v = listV1;              \
  distV1 = vec_or(distV1, listV1);      \
  lev->comp[1].v = compV1;              \
  compV1 = vec_or(compV1, distV1);      \
  dist0 = lev->dist[0].u[0];            \
  newbit = vec_splat_u32(1);            \
  comp0 |= dist0;


/*
** Pop a level to continue work on previous mark
*/
#define POP_LEVEL(lev)                  \
  listV0 = lev->list[0].v;              \
  listV1 = lev->list[1].v;              \
  comp0  = lev->comp[0].u[0];           \
  distV0 = vec_andc(distV0, listV0);    \
  distV1 = vec_andc(distV1, listV1);    \
  compV0 = lev->comp[0].v;              \
  compV1 = lev->comp[1].v;              \
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


//----------------------------------------------------------------------------

#ifndef IMPLEMENT_CELL_CORES
#include "ansi/ogrng_codebase.cpp"
#endif

#if !defined(OGRNG_BITMAPS_LENGTH) || (OGRNG_BITMAPS_LENGTH != 256)
#error OGRNG_BITMAPS_LENGTH must be 256 !!!
#endif


/*
** Check the settings again since we have to make sure ogr_create()
** produces compatible datas.
*/
#if defined(HAVE_FLEGE_PPC_CORES) && (OGROPT_ALTERNATE_CYCLE > 0)

  #ifdef __cplusplus
  extern "C" {
  #endif
  int cycle_ppc_hybrid_256(struct OgrState *state, int *pnodes,
                           const u16 *choose, const void* pShift);
  #ifdef __cplusplus
  }
  #endif

  #include <stddef.h> // offsetof
  static int ogr_cycle_256(struct OgrState *oState, int *pnodes,
                           const u16* pchoose)
  {
    v8_t vecShift[32];
    vector unsigned char val = vec_splat_u8(0);
    vector unsigned char one = vec_splat_u8(1);
    int i;
    
    STATIC_ASSERT(sizeof(struct OgrLevel) == 112);
    STATIC_ASSERT(offsetof(struct OgrLevel, list)  == 0);
    STATIC_ASSERT(offsetof(struct OgrLevel, dist)  == 32);
    STATIC_ASSERT(offsetof(struct OgrLevel, comp)  == 64);
    STATIC_ASSERT(offsetof(struct OgrLevel, mark)  == 96);
    STATIC_ASSERT(offsetof(struct OgrLevel, limit) == 100);
    
    STATIC_ASSERT(offsetof(struct OgrState, max)         == 0);
    STATIC_ASSERT(offsetof(struct OgrState, maxdepthm1)  == 8);
    STATIC_ASSERT(offsetof(struct OgrState, half_depth)  == 12);
    STATIC_ASSERT(offsetof(struct OgrState, half_depth2) == 16);
    STATIC_ASSERT(offsetof(struct OgrState, stopdepth)   == 24);
    STATIC_ASSERT(offsetof(struct OgrState, depth)       == 28);
    STATIC_ASSERT(offsetof(struct OgrState, Levels)      == 32);

    for (i = 0; i < 32; i++) {
      vecShift[i] = val;
      val = vec_add(val, one);
    }
    return cycle_ppc_hybrid_256(oState, pnodes, pchoose, vecShift);
  }
#endif

#endif // __SPU__
