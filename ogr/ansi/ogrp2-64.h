/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the three bitmaps "list", "dist" and "comp" is
 * made of three 64-bit scalars.
 * Beside, the OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT setting selects a
 * memory based implementation (0), or a register based implementation (1).
 *
 * $Id: ogrp2-64.h,v 1.1 2008/03/08 20:07:14 kakace Exp $
*/

#include "ansi/ogrp2.h"

#ifndef HAVE_I64
#error fixme: your compiler does not appear to support 64-bit datatypes
#endif


/*
** Bitmaps built with 64-bit words
*/

typedef ui64 BMAP;        /* Basic type for each bitmap array */
typedef u32  SCALAR;      /* Basic type for scalar operations on bitmaps */
#define OGR_BITMAPS_WORDS ((OGR_BITMAPS_LENGTH + 63) / 64)
#define SCALAR_BITS  32


//-------------------------- IMPLEMENTATION MACROS ---------------------------

/*
** Initialize top state
*/
#if !defined(SETUP_TOP_STATE)
  #define SETUP_TOP_STATE(lev)                      \
    register SCALAR comp0 = (SCALAR) lev->comp[0];  \
    register SCALAR dist0 = (SCALAR) lev->dist[0];  \
    lev->list[0] |= ((BMAP)1 << 32);
#endif


/*
** Shift COMP and LIST bitmaps
*/
#if !defined(COMP_LEFT_LIST_RIGHT)
 #define __COMP_LEFT_LIST_RIGHT(lev, s, ss) {               \
    BMAP temp1, temp2;                                      \
    temp2 = lev->list[0] << (ss);                           \
    lev->list[0] = (lev->list[0] >> (s));                   \
    temp1 = lev->list[1] << (ss);                           \
    lev->list[1] = (lev->list[1] >> (s)) | temp2;           \
    temp2 = lev->comp[1] >> (ss);                           \
    lev->list[2] = (lev->list[2] >> (s)) | temp1;           \
    temp1 = lev->comp[2] >> (ss);                           \
    comp0 = (SCALAR) (lev->comp[0] = (lev->comp[0] << (s)) | temp2); \
    lev->comp[1] = (lev->comp[1] << (s)) | temp1;           \
    lev->comp[2] = lev->comp[2] << (s);                     \
  }

  #define COMP_LEFT_LIST_RIGHT(lev, s) {    \
    register int ss = 64 - (s);             \
    __COMP_LEFT_LIST_RIGHT(lev, s, ss);     \
  }
#endif


/*
** Shift COMP and LIST bitmaps by 32
*/
#if !defined(COMP_LEFT_LIST_RIGHT_WORD)
  #define COMP_LEFT_LIST_RIGHT_WORD(lev)    \
    __COMP_LEFT_LIST_RIGHT(lev, 32, 32);
#endif


/*
** Update state then go deeper
*/
#if !defined(PUSH_LEVEL_UPDATE_STATE)
  #define PUSH_LEVEL_UPDATE_STATE(lev) {              \
    struct Level *lev2 = lev + 1;                     \
    BMAP temp = lev->list[0];                         \
    lev2->list[0] = temp | ((BMAP)1 << 32);           \
    dist0 = (SCALAR) (lev2->dist[0] = lev->dist[0] | temp);  \
    lev2->comp[0] = (comp0 |= dist0);                 \
    temp = (lev2->list[1] = lev->list[1]);            \
    temp = (lev2->dist[1] = lev->dist[1] | temp);     \
    lev2->comp[1] = lev->comp[1] | temp;              \
    temp = (lev2->list[2] = lev->list[2]);            \
    temp = (lev2->dist[2] = lev->dist[2] | temp);     \
    lev2->comp[2] = lev->comp[2] | temp;              \
  }
#endif


/*
** Pop level state (all bitmaps).
*/
#if !defined(POP_LEVEL)
  #define POP_LEVEL(lev)      \
    comp0 = (SCALAR) lev->comp[0];
#endif


/*
** Save final state (all bitmaps)
*/
#if !defined(SAVE_FINAL_STATE)
  #define SAVE_FINAL_STATE(lev)           /* nothing */
#endif


#if !defined(OGR_BITMAPS_LENGTH) || (OGR_BITMAPS_LENGTH != 160)
#error OGR_BITMAPS_LENGTH must be 160 !!!
#endif
