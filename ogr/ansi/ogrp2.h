/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/
#ifndef __OGRP2_H__
#define __OGRP2_H__ "@(#)$Id: ogrp2.h,v 1.2 2008/02/13 22:06:53 kakace Exp $"

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

/* specifies the number of ruler diffs can be represented.
** Warning: increasing this will cause all structures based
** on workunit_t in packets.h to change, possibly breaking
** network and buffer structure operations.
*/
#define STUB_MAX 10
#define MAXDEPTH 26

#ifndef __SUNPRO_CC
  #include "pack1.h"
#else
  #undef DNETC_PACKED
  #define DNETC_PACKED
#endif

struct Stub {           /* size is 24 */
  u16 marks;            /* N-mark ruler to which this stub applies */
  u16 length;           /* number of valid elements in the stub[] array */
  u16 diffs[STUB_MAX];  /* first <length> differences in ruler */
} DNETC_PACKED;

struct WorkStub {       /* size is 28 */
  struct Stub stub;     /* stub we're working on */
  u32 worklength;       /* depth of current state */
} DNETC_PACKED;

#ifndef __SUNPRO_CC
  #include "pack0.h"
#else
  #undef DNETC_PACKED
#endif



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
  struct Level Levels[MAXDEPTH];
  int node_offset;          /* node count cache for non-preemptive OS */
};

#define OGR_PROBLEM_SIZE (((8*OGR_INT_SIZE+15)&(-16))+(OGR_LEVEL_SIZE*MAXDEPTH))
                         /* sizeof(struct State) */

#endif /* __OGRP2_H__ */

