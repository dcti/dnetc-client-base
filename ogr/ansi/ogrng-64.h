/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the three bitmaps "list", "dist" and "comp" is
 * made of four 64-bit scalars, so that the bitmaps precision strictly matches
 * the regular 32-bit core.
 *
 * $Id: ogrng-64.h,v 1.1 2008/03/08 20:07:14 kakace Exp $
*/

#include "ansi/ogrng.h"

#ifndef HAVE_I64
#error fixme: your compiler does not appear to support 64-bit datatypes
#endif


/*
** Bitmaps built with 64-bit words
*/

typedef ui64 BMAP;        /* Basic type for each bitmap array */
typedef ui64 SCALAR;      /* Basic type for scalar operations on bitmaps */
#define OGRNG_BITMAPS_WORDS ((OGRNG_BITMAPS_LENGTH + 63) / 64)
#define SCALAR_BITS  64


//-------------------------- IMPLEMENTATION MACROS ---------------------------

/*
** Initialize top state
*/
#if !defined(SETUP_TOP_STATE)
  #define SETUP_TOP_STATE(lev)            \
    register BMAP comp0 = lev->comp[0];   \
    BMAP dist0;                           \
    BMAP newbit = (depth < oState->maxdepthm1) ? 1 : 0;
#endif

/*
** Shift COMP and LIST bitmaps
*/
#if !defined(COMP_LEFT_LIST_RIGHT)
 #define COMP_LEFT_LIST_RIGHT(lev, s)                            \
   register int ss = SCALAR_BITS - (s);                          \
   lev->list[3] = (lev->list[3] >> (s)) | (lev->list[2] << ss);  \
   lev->list[2] = (lev->list[2] >> (s)) | (lev->list[1] << ss);  \
   lev->list[1] = (lev->list[1] >> (s)) | (lev->list[0] << ss);  \
   lev->list[0] = (lev->list[0] >> (s)) | (newbit << ss);        \
   lev->comp[0] = comp0 = (comp0 << (s)) | (lev->comp[1] >> ss); \
   lev->comp[1] = (lev->comp[1] << (s)) | (lev->comp[2] >> ss);  \
   lev->comp[2] = (lev->comp[2] << (s)) | (lev->comp[3] >> ss);  \
   lev->comp[3] = (lev->comp[3] << (s));                         \
   newbit = 0;
#endif

/*
** Shift COMP and LIST bitmaps by 64
*/
#if !defined(COMP_LEFT_LIST_RIGHT_WORD)
 #define COMP_LEFT_LIST_RIGHT_WORD(lev)    \
   lev->comp[0] = comp0 = lev->comp[1];    \
   lev->comp[1] = lev->comp[2];            \
   lev->comp[2] = lev->comp[3];            \
   lev->comp[3] = 0;                       \
   lev->list[3] = lev->list[2];            \
   lev->list[2] = lev->list[1];            \
   lev->list[1] = lev->list[0];            \
   lev->list[0] = newbit;                  \
   newbit = 0;
#endif

/*
** Update state then go deeper
*/
#if !defined(PUSH_LEVEL_UPDATE_STATE)
 #define PUSH_LEVEL_UPDATE_STATE(lev) {          \
   struct OgrLevel *lev2 = lev + 1;              \
   BMAP temp;                                    \
   lev2->list[0] = temp = lev->list[0];          \
   lev2->dist[0] = dist0 = lev->dist[0] | temp;  \
   lev2->comp[0] = comp0 |= dist0;               \
   lev2->list[1] = temp = lev->list[1];          \
   lev2->dist[1] = temp |= lev->dist[1];         \
   lev2->comp[1] = lev->comp[1] | temp;          \
   lev2->list[2] = temp = lev->list[2];          \
   lev2->dist[2] = temp |= lev->dist[2];         \
   lev2->comp[2] = lev->comp[2] | temp;          \
   lev2->list[3] = temp = lev->list[3];          \
   lev2->dist[3] = temp |= lev->dist[3];         \
   lev2->comp[3] = lev->comp[3] | temp;          \
   newbit = 1;                                   \
 }
#endif

/*
** Pop level state (all bitmaps).
*/
#if !defined(POP_LEVEL)
 #define POP_LEVEL(lev)      \
   comp0 = lev->comp[0];     \
   newbit = 0;
#endif

/*
** Save final state (all bitmaps)
*/
#if !defined(SAVE_FINAL_STATE)
 #define SAVE_FINAL_STATE(lev)          /* nothing */
#endif


#if !defined(OGRNG_BITMAPS_LENGTH) || (OGRNG_BITMAPS_LENGTH != 256)
#error OGRNG_BITMAPS_LENGTH must be 256 !!!
#endif
