/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogrp2_corestate.h,v 1.1 2008/03/08 20:07:14 kakace Exp $
 */

#ifndef ogrp2_corestate_H
#define ogrp2_corestate_H

#ifndef OGROPT_SPECIFIC_LEVEL_STRUCT
struct Level {
   BMAP list[OGR_BITMAPS_WORDS];
   BMAP dist[OGR_BITMAPS_WORDS];
   BMAP comp[OGR_BITMAPS_WORDS];
   int mark;
   int limit;
};
#endif


#ifndef OGROPT_SPECIFIC_STATE_STRUCT
struct State {
   int max;                  /* maximum length of ruler */
   int maxdepth;             /* maximum number of marks in ruler */
   int maxdepthm1;           /* maxdepth-1 */
   int half_length;          /* half of max */
   int half_depth;           /* half of maxdepth */
   int half_depth2;          /* half of maxdepth, adjusted for 2nd mark */
   int startdepth;
   int depth;
   struct Level Levels[MAXDEPTH];
   int node_offset;          /* node count cache for non-preemptive OS */
};
#endif

#endif	/* ogrp2_corestate_H */
