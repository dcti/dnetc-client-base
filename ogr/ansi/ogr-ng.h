/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/
#ifndef __OGR_NG_H__
#define __OGR_NG_H__ "@(#)$Id: ogr-ng.h,v 1.5 2008/03/03 22:29:32 kakace Exp $"

#include "ogr-interface.h"

/* ===================================================================== */

#define OGRNG_BITMAPS_LENGTH  256


#ifndef OGROPT_64BIT_IMPLEMENTATION
  /* Bitmaps built with 32-bit words */
  typedef u32 U;
  #define OGRNG_BITMAPS_WORDS ((OGRNG_BITMAPS_LENGTH + 31) / 32)

#else   /* OGROPT_64BIT_IMPLEMENTATION */
  /* Bitmaps built with 64-bit words */
  typedef ui64 U;
  #define OGRNG_BITMAPS_WORDS ((OGRNG_BITMAPS_LENGTH + 63) / 64)

#endif


/* ===================================================================== */

/*
** CHOOSEDAT optimization (OGROPT_STRENGTH_REDUCE_CHOOSE)
*/
#define CHOOSE_MARKS       16     /* maximum number of marks supported    */
#define CHOOSE_DIST_BITS   16     /* number of bits to take into account  */
#define CHOOSE_ELEMS (CHOOSE_MARKS << CHOOSE_DIST_BITS)

#define ttmDISTBITS (32-CHOOSE_DIST_BITS)


/* These are defined in ogrng_dat.cpp */
extern const unsigned char ogrng_choose_datas[];
extern const int           ogr_ng_choose_bits;
extern const int           ogr_ng_choose_marks;


struct choose_datas {
   u16* choose_array;
   u32  checksum;
};


#define OGR_MAX_MARKS 28      /* NEVER EVER INCREASE THIS VALUE */
#define OGR_STUB_MAX  28      /* DITTO */

#define OGR_MAXDEPTH  (OGR_MAX_MARKS+1)
#define OGR_NG_MIN    21      /* OGR-21 (test cases) */
#define OGR_NG_MAX    28      /* Max set to OGR-28 */


/* ===================================================================== */

#ifndef __SUNPRO_CC
  #include "pack1.h"
#else
  #undef DNETC_PACKED
  #define DNETC_PACKED
#endif

/* Specifies the number of ruler diffs can be represented.
** For OGR-NG, this structure is large enough to record every marks. Also note
** that it can be seen as a derivative from the Stub structure (i.e., an
** OgrStub object IS a Stub object).
*/

struct OgrStub {           /* size is 60 */
  u16 marks;               /* N-mark ruler to which this stub applies */
  u16 length;              /* number of valid elements in the diffs[] array */
  u16 diffs[OGR_STUB_MAX]; /* first <length> differences in ruler */
} DNETC_PACKED;


/* CAUTION : assert sizeof(OgrWorkStub) <= 64
** Otherwise, the ContestWork structure will break the 80 bytes limit !
*/
struct OgrWorkStub {       /* size is 64 */
  struct OgrStub stub;     /* stub we're working on */
  u16 worklength;          /* depth of current state */
  u16 collapsed;           /* If != 0, then it's the last segment in stub (backup) */
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
  struct OgrLevel {
    U list[OGRNG_BITMAPS_WORDS];
    U dist[OGRNG_BITMAPS_WORDS];
    U comp[OGRNG_BITMAPS_WORDS];
    int mark;
    int limit;
  };
#else   
  struct OgrLevel {
    VECTOR listV0, listV1;
    VECTOR distV0, distV1;
    VECTOR compV0, compV1;
    int mark;
    int limit;
  };
#endif


/* Internal stuff that's not part of the interface but we need for declaring
** the problem work area size.
**
** I have to reserve memory for all possible OGR cruncher setups because
** memory reservation happens inside problem.h/.c and I cannot know what
** cruncher is going to get used :(
*/
#define OGRNG_LEVEL_SIZE_SCALAR (((((OGRNG_BITMAPS_LENGTH+63)/64)*3*8)+(OGR_INT_SIZE*2)+8)&(-8))
#define OGRNG_LEVEL_SIZE_VECTOR (((16*6)+(OGR_INT_SIZE*2)+15)&(-16))

#define OGRNG_LEVEL_SIZE (OGRNG_LEVEL_SIZE_SCALAR > OGRNG_LEVEL_SIZE_VECTOR ? \
          OGRNG_LEVEL_SIZE_SCALAR : OGRNG_LEVEL_SIZE_VECTOR)


struct OgrState {
  int max;                  /* maximum length of ruler */
  int maxdepth;             /* maximum number of marks in ruler */
  int maxdepthm1;           /* maxdepth-1 */
  int half_depth;           /* half of maxdepth */
  int half_depth2;          /* half of maxdepth, adjusted for 2nd mark */
  int startdepth;           /* Initial depth */
  int stopdepth;            /* May be lower than startdepth */
  int depth;                /* Current depth */
  struct OgrLevel Levels[OGR_MAXDEPTH];
};

#define OGRNG_PROBLEM_SIZE (((9*OGR_INT_SIZE+15)&(-16))+(OGRNG_LEVEL_SIZE*OGR_MAXDEPTH))
                         /* sizeof(struct State) */


/*
** ogrng_init.cpp
*/
extern struct choose_datas precomp_limits[OGR_NG_MAX - OGR_NG_MIN + 1];

int  ogr_init_choose(void);
void ogr_cleanup_choose(void);
int  ogr_check_cache(int nMarks);

#endif /* __OGR_NG_H__ */

