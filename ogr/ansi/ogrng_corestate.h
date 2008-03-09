/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogrng_corestate.h,v 1.1 2008/03/09 13:22:34 kakace Exp $
 */

#ifndef ogrng_corestate_H
#define ogrng_corestate_H

#ifndef OGROPT_SPECIFIC_LEVEL_STRUCT
/*
 ** Level datas.
 */
struct OgrLevel {
   BMAP list[OGRNG_BITMAPS_WORDS];
   BMAP dist[OGRNG_BITMAPS_WORDS];
   BMAP comp[OGRNG_BITMAPS_WORDS];
   int mark;
   int limit;
};
#endif


#ifndef OGROPT_SPECIFIC_STATE_STRUCT
/*
 ** Full state.
 */
struct OgrState {
   int max;                  /* Maximum length of the ruler */
   int maxdepth;             /* maximum number of marks in ruler */
   int maxdepthm1;           /* maxdepth-1 */
   int half_depth;           /* half of maxdepth */
   int half_depth2;          /* half of maxdepth, adjusted for 2nd mark */
   int startdepth;           /* Initial depth */
   int stopdepth;            /* May be lower than startdepth */
   int depth;                /* Current depth */
   struct OgrLevel Levels[OGR_MAXDEPTH];
};
#endif

#endif	/* ogrng_corestate_H */
