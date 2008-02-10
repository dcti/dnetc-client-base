/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/
#ifndef __OGRP2_H__
#define __OGRP2_H__ "@(#)$Id: ogrp2.h,v 1.1 2008/02/10 00:07:41 kakace Exp $"

#include "ogr-interface.h"

/* ===================================================================== */

#define OGR_BITMAPS_LENGTH  160


#ifndef OGROPT_64BIT_IMPLEMENTATION
  /* Bitmaps built with 32-bit words */
  typedef u32 U;
  #define OGR_BITMAPS_WORDS ((OGR_BITMAPS_LENGTH + 31) / 32)

#else   /* OGROPT_64BIT_IMPLEMENTATION */
  /* Bitmaps built with 64-bit words */
  typedef ui64 U;
  #define OGR_BITMAPS_WORDS ((OGR_BITMAPS_LENGTH + 63) / 64)

#endif


/* ===================================================================== */

#ifndef OGROPT_OGR_CYCLE_ALTIVEC
/*
** Standard (32-bit scalar) implementation
*/
  struct Level {
    U list[OGR_BITMAPS_WORDS];
    U dist[OGR_BITMAPS_WORDS];
    U comp[OGR_BITMAPS_WORDS];
    int mark;
    int limit;
  };
#else   
  struct Level {
    VECTOR listV, compV, distV;
    int limit;
    U comp0, dist0, list0;  /* list0 *MUST* be the 4th integer */
    int mark;
  };
#endif


/* Internal stuff that's not part of the interface but we need for declaring
** the problem work area size.
**
** I have to reserve memory for all possible OGR cruncher setups because
** memory reservation happens inside problem.h/.c and I cannot know what
** cruncher is going to get used :(
*/
#define OGR_LEVEL_SIZE_SCALAR (((((OGR_BITMAPS_LENGTH+63)/64)*3*8)+(OGR_INT_SIZE*2)+8)&(-8))
#define OGR_LEVEL_SIZE_VECTOR (((16*3)+(4*3)+(OGR_INT_SIZE*2)+15)&(-16))

#define OGR_LEVEL_SIZE (OGR_LEVEL_SIZE_SCALAR > OGR_LEVEL_SIZE_VECTOR ? \
          OGR_LEVEL_SIZE_SCALAR : OGR_LEVEL_SIZE_VECTOR)


struct State {
  int max;                  /* maximum length of ruler */
  int maxdepth;             /* maximum number of marks in ruler */
  int maxdepthm1;           /* maxdepth-1 */
  int half_length;          /* half of max */
  int half_depth;           /* half of maxdepth */
  int half_depth2;          /* half of maxdepth, adjusted for 2nd mark */
  int startdepth;
  int depth;
  struct Level Levels[OGR_MAXDEPTH];
  int node_offset;          /* node count cache for non-preemptive OS */
};

#define OGR_PROBLEM_SIZE (((8*OGR_INT_SIZE+15)&(-16))+(OGR_LEVEL_SIZE*OGR_MAXDEPTH))
                         /* sizeof(struct State) */

#endif /* __OGRP2_H__ */

