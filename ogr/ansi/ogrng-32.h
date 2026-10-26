/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the three bitmaps "list", "dist" and "comp" is
 * made of eight 32-bit scalars.
 *
 * $Id: ogrng-32.h,v 1.1 2008/03/08 20:07:14 kakace Exp $
*/

#include "ansi/ogrng.h"


/*
** Bitmaps built with 32-bit words
*/

typedef u32 BMAP;       /* Basic type for each bitmap array */
typedef u32 SCALAR;     /* Basic type for scalar operations on bitmaps */
#define OGRNG_BITMAPS_WORDS ((OGRNG_BITMAPS_LENGTH + 31) / 32)
#define SCALAR_BITS  32


//-------------------------- IMPLEMENTATION MACROS ---------------------------

/** Initialize top state.
 ** The macro shall at least define "comp0", "dist0" and "newbit". It can also
 ** define private datas used by the other macros.
 ** "newbit" acts as a flag to indicate whether we shall place a new mark on
 ** the ruler (newbit = 1), or move the last mark toward the right edge of the
 ** ruler (newbit = 0). The engine usually exits when "newbit == 1". The only
 ** exception occurs when a ruler is found.
 */
#if !defined(SETUP_TOP_STATE)
   #define SETUP_TOP_STATE(lev)                          \
      SCALAR comp0 = lev->comp[0];                       \
      SCALAR dist0;                                      \
      int newbit = (depth < oState->maxdepthm1) ? 1 : 0;
#endif


/** Shift COMP and LIST bitmaps.
 ** This macro implements two multi-precision shifts : a left shift (for the
 ** COMP bitmap), and a right shift (for the LIST bitmap). Note that the value
 ** of "newbit" is shifted in the LIST bitmap. On exit, the value of "newbit"
 ** shall be zero, and the value of "comp0" shall be reset to the leftmost
 ** word of the COMP bitmap.
 */
#if !defined(COMP_LEFT_LIST_RIGHT)
   #define COMP_LEFT_LIST_RIGHT(lev, s) {             \
      BMAP temp1, temp2;                              \
      register int ss = SCALAR_BITS - (s);            \
      temp1 = newbit << ss;                           \
      temp2 = lev->list[0] << ss;                     \
      lev->list[0] = (lev->list[0] >> (s)) | temp1;   \
      temp1 = lev->list[1] << ss;                     \
      lev->list[1] = (lev->list[1] >> (s)) | temp2;   \
      temp2 = lev->list[2] << ss;                     \
      lev->list[2] = (lev->list[2] >> (s)) | temp1;   \
      temp1 = lev->list[3] << ss;                     \
      lev->list[3] = (lev->list[3] >> (s)) | temp2;   \
      temp2 = lev->list[4] << ss;                     \
      lev->list[4] = (lev->list[4] >> (s)) | temp1;   \
      temp1 = lev->list[5] << ss;                     \
      lev->list[5] = (lev->list[5] >> (s)) | temp2;   \
      temp2 = lev->list[6] << ss;                     \
      lev->list[6] = (lev->list[6] >> (s)) | temp1;   \
      temp1 = lev->comp[1] >> ss;                     \
      lev->list[7] = (lev->list[7] >> (s)) | temp2;   \
      temp2 = lev->comp[2] >> ss;                     \
      comp0 = (lev->comp[0] << (s)) | temp1;          \
      lev->comp[0] = comp0;                           \
      temp1 = lev->comp[3] >> ss;                     \
      lev->comp[1] = (lev->comp[1] << (s)) | temp2;   \
      temp2 = lev->comp[4] >> ss;                     \
      lev->comp[2] = (lev->comp[2] << (s)) | temp1;   \
      temp1 = lev->comp[5] >> ss;                     \
      lev->comp[3] = (lev->comp[3] << (s)) | temp2;   \
      temp2 = lev->comp[6] >> ss;                     \
      lev->comp[4] = (lev->comp[4] << (s)) | temp1;   \
      temp1 = lev->comp[7] >> ss;                     \
      lev->comp[5] = (lev->comp[5] << (s)) | temp2;   \
      lev->comp[7] = (lev->comp[7] << (s));           \
      lev->comp[6] = (lev->comp[6] << (s)) | temp1;   \
      newbit = 0;                                     \
      }
#endif


/** Shift COMP and LIST bitmaps by one bitmap word.
 ** This macro implements a specialization of the preceeding macro.
 */
#if !defined(COMP_LEFT_LIST_RIGHT_WORD)
      #define COMP_LEFT_LIST_RIGHT_WORD(lev)    \
      lev->list[7] = lev->list[6];           \
      lev->list[6] = lev->list[5];           \
      lev->list[5] = lev->list[4];           \
      lev->list[4] = lev->list[3];           \
      lev->list[3] = lev->list[2];           \
      lev->list[2] = lev->list[1];           \
      lev->list[1] = lev->list[0];           \
      lev->list[0] = newbit;                 \
      comp0 = lev->comp[1];                  \
      lev->comp[0] = comp0;                  \
      lev->comp[1] = lev->comp[2];           \
      lev->comp[2] = lev->comp[3];           \
      lev->comp[3] = lev->comp[4];           \
      lev->comp[4] = lev->comp[5];           \
      lev->comp[5] = lev->comp[6];           \
      lev->comp[6] = lev->comp[7];           \
      lev->comp[7] = 0;                      \
      newbit = 0;
#endif


/** Update the COMP, DIST and LIST bitmaps.
 ** In pseudo-code :
 **   LIST[lev+1] = LIST[lev]
 **   DIST[lev+1] = (DIST[lev] | LIST[lev+1])
 **   COMP[lev+1] = (COMP[lev] | DIST[lev+1])
 **   newbit = 1;
 ** Note that "dist0" and "comp0" shall be updated to the new values of the
 ** leftmost words of the corresponding bitmaps.
 */
#if !defined(PUSH_LEVEL_UPDATE_STATE)
   #define PUSH_LEVEL_UPDATE_STATE(lev) {             \
      BMAP temp1, temp2;                              \
      struct OgrLevel *lev2 = lev + 1;                \
      temp1 = (lev2->list[0] = lev->list[0]);         \
      temp2 = (lev2->list[1] = lev->list[1]);         \
      dist0 = (lev2->dist[0] = lev->dist[0] | temp1); \
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
      temp2 = (lev2->list[5] = lev->list[5]);         \
      temp1 = (lev2->dist[4] = lev->dist[4] | temp1); \
      temp2 = (lev2->dist[5] = lev->dist[5] | temp2); \
      lev2->comp[4] = lev->comp[4] | temp1;           \
      lev2->comp[5] = lev->comp[5] | temp2;           \
      temp1 = (lev2->list[6] = lev->list[6]);         \
      temp2 = (lev2->list[7] = lev->list[7]);         \
      temp1 = (lev2->dist[6] = lev->dist[6] | temp1); \
      temp2 = (lev2->dist[7] = lev->dist[7] | temp2); \
      lev2->comp[6] = lev->comp[6] | temp1;           \
      lev2->comp[7] = lev->comp[7] | temp2;           \
      newbit = 1;                                     \
      }
#endif


/** Pop level state (all bitmaps).
 ** Reload the state of the specified level. "newbit" shall be reset to zero.
 */
#if !defined(POP_LEVEL)
   #define POP_LEVEL(lev)     \
      comp0 = lev->comp[0];   \
      dist0 = lev->dist[0];   \
      newbit = 0;
#endif


/** Save final state (all bitmaps)
 */
#if !defined(SAVE_FINAL_STATE)
   #define SAVE_FINAL_STATE(lev)
      /* nothing */
#endif


#if !defined(OGRNG_BITMAPS_LENGTH) || (OGRNG_BITMAPS_LENGTH != 256)
#error OGRNG_BITMAPS_LENGTH must be 256 !!!
#endif
