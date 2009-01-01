/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/
#ifndef __OGR_NG_H__
#define __OGR_NG_H__ "@(#)$Id: ogrng.h,v 1.2 2009/01/01 13:37:14 andreasb Exp $"

#include "ansi/ogr-interface.h"

/* ===================================================================== */

/*
** The number of bits in each of the three bitmaps.
*/
#define OGRNG_BITMAPS_LENGTH  256


/*
** CHOOSEDAT optimization.
*/
#define CHOOSE_MARKS       16     /* maximum number of marks supported    */
#define CHOOSE_DIST_BITS   16     /* number of bits to take into account  */
#define CHOOSE_ELEMS (CHOOSE_MARKS << CHOOSE_DIST_BITS)


/*
** OGR-NG basic parameters.
*/
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


/* Ruler stub.
** For OGR-NG, this structure is large enough to record every marks. Also note
** that it can be seen as a derivative from the legacy Stub structure (i.e., an
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


/* Internal stuff that's not part of the interface but we need for declaring
** the problem work area size.
** For each level, we need to record the three bitmaps plus two words to store
** the actual limit and mark position. Assuming a 32-byte padding, the overall
** size of the required structure is 32*4 bytes.
** To record the entire core state, we have to store the datas of every level,
** plus eight words (assumed to be 64-bit wide each) for ancillary datas.
*/
#define OGRNG_PROBLEM_SIZE (32 * 4 * OGR_MAXDEPTH + 8 * 8)


/*-----------------------------------------------------------------------------
** Declarations for ogrng_init.cpp
*/

/* Lookup tables (These are defined in ogrng_dat.cpp) */
extern const unsigned char ogrng_choose_datas[];
extern const int           ogrng_choose_bits;
extern const int           ogrng_choose_marks;


struct choose_datas {
  u16* choose_array;
  u32 checksum;
};


/* Caches for pre-computed limits.
** The cache is built dynamically whenever necessary.
*/
extern struct choose_datas precomp_limits[OGR_NG_MAX - OGR_NG_MIN + 1];


int  ogr_init_choose(void);
void ogr_cleanup_choose(void);
void ogr_cleanup_cache();
int  ogr_check_cache(int nMarks);

#endif /* __OGR_NG_H__ */

