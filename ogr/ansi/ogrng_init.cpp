/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogrng_init.cpp,v 1.7 2008/11/22 20:25:50 stream Exp $
 */

#include <stdlib.h>     /* calloc */
#include "clisync.h"    /* fastlock_* */
#include "ansi/ogrng.h"

static void cache_limits(u16* pDatas, int nMarks);
static int  min_length(int nSegs, u32 distSet);
static int  build_cache(int nMarks);


static const int OGR[] = {
  /*  1 */    0,   1,   3,   6,  11,  17,  25,  34,  44,  55,
  /* 11 */   72,  85, 106, 127, 151, 177, 199, 216, 246, 283,
  /* 21 */  333, 356, 372, 425, 480, 492, 553, 585, 623, 680,
  /* 31 */  747, 784
};


static fastlock_t memlock;

/*-------------------------------------------------------------------------*/

static u32 fletcher16(const u16* pDatas, size_t nElems)
{
   u32 sum1 = 0xFFFF;
   u32 sum2 = 0xFFFF;


   while (nElems) {
      size_t block_size = (nElems > 360) ? 360 : nElems;

      nElems -= block_size;
      do {
         sum1 += *pDatas++;
         sum2 += sum1;
      } while (--block_size);

      sum1 = (sum1 & 0xFFFF) + (sum1 >> 16);
      sum2 = (sum2 & 0xFFFF) + (sum2 >> 16);
   }

   sum1 = (sum1 & 0xFFFF) + (sum1 >> 16);
   sum2 = (sum2 & 0xFFFF) + (sum2 >> 16);

   return (sum2 << 16) | sum1;
}

/* ----------------------------------------------------------------------- */

/*
** ogrng_dat.cpp only contains a somewhat compressed dataset that cannot be
** used "as is". The real datas are recovered here.
*/
static struct choose_datas choose_dat = {NULL, 0x4328E149};


/*
** Pre-computed limits.
** ogr_init_choose() initialize the tables for OGR-NG_MIN up to OGR_NG_MAX. Each
** table is (2 x 2^16 x 32) bytes wide, and it is only created when required
** (see ogr_create()).
*/
struct choose_datas precomp_limits[OGR_NG_MAX - OGR_NG_MIN + 1] = {
   {NULL, 0},       /* OGR-21 */
   {NULL, 0},       /* OGR-22 */
   {NULL, 0},       /* OGR-23 */
   {NULL, 0},       /* OGR-24 */
   {NULL, 0},       /* OGR-25 */
   {NULL, 0},       /* OGR-26 */
   {NULL, 0},       /* OGR-27 */
   {NULL, 0}        /* OGR-28 */
};

/*
 * This is mostly for Cell because DMA transfer must be aligned to 16 bytes
 */
static void *aligned_malloc(size_t size)
{
  void *varray;
#if CLIENT_CPU == CPU_CELLBE
  if (posix_memalign(&varray, 16, size))
    varray = NULL;
#else
  varray = malloc(size);
#endif
  return varray;
}

/* Rebuild the 'choose' array. The memory area is protected by a checksum
 * (Fletcher16). That checksum is always verified after the datas have been
 * used to initialize a new set of pre-computed limits.
 * Returns :
 * 0 if successful
 * -1 if the choose array cannot be allocated
 * -2 if the choose array mismatch.
 */
int ogr_init_choose(void)
{
  int m, set;
  u16* array = choose_dat.choose_array;

  fastlock_init(&memlock);
  if (CHOOSE_DIST_BITS != ogrng_choose_bits || CHOOSE_MARKS != ogrng_choose_marks)
  {
    /* Incompatible CHOOSE array - Give up */
    return -2;
  }

  if (NULL != array) {
   return 0;                     /* Array already allocated */
  }


  /* Allocate and setup the choose array */
  array = (u16*) aligned_malloc(CHOOSE_ELEMS * sizeof(u16));

  if (NULL == array) {
    return -1;
  }

  for (set = 0; set < (1 << CHOOSE_DIST_BITS); set++) {
    array[CHOOSE_MARKS * set] = 0;
    for (m = 1; m < CHOOSE_MARKS; m++) {
       unsigned long wo = CHOOSE_MARKS * set + m;
       unsigned long ro = (CHOOSE_MARKS - 1) * set + (m - 1);
       array[wo] = array[wo - 1] + ogrng_choose_datas[ro];
    }
  }

  if (fletcher16(array, CHOOSE_ELEMS) != choose_dat.checksum) {
    return -2;          /* Modified choose_dat - Give up ! */
  }
  choose_dat.choose_array = array;

  return 0;
}


void ogr_cleanup_choose(void)
{
  int m;

  /* Release the choose array */
  if (NULL != choose_dat.choose_array) {
    free(choose_dat.choose_array);
    choose_dat.choose_array = NULL;
  }

  /* Release the cached limits arrays */
  for (m = OGR_NG_MIN; m <= OGR_NG_MAX; m++) {
    struct choose_datas* p = &precomp_limits[m - OGR_NG_MIN];

    if (p->choose_array) {
      free(p->choose_array);
      p->choose_array = NULL;
    }
  }
}


static int build_cache(int nMarks)
{
  int success = 0;

  fastlock_lock(&memlock);

  if (nMarks >= OGR_NG_MIN && nMarks <= OGR_NG_MAX) {
    struct choose_datas* p = &precomp_limits[nMarks - OGR_NG_MIN];
    size_t n_elems = 32 << CHOOSE_DIST_BITS;
    u16* array = (u16*) aligned_malloc(n_elems * sizeof(u16));

    if (array) {
      p->choose_array = array;
      cache_limits(p->choose_array, nMarks);
      p->checksum = fletcher16(p->choose_array, n_elems);
      success = (fletcher16(choose_dat.choose_array, CHOOSE_ELEMS) == choose_dat.checksum);
    }
  }

  fastlock_unlock(&memlock);
  return success;
}


/*  Check the pre-computed limits table. Create the table on the fly if it has
 *  not been allocated yet.
 *  Returns :
 *  True if the cache is healthy
 *  False if the cache has been trashed
 */
int ogr_check_cache(int nMarks)
{
  if (nMarks >= OGR_NG_MIN && nMarks <= OGR_NG_MAX) {
    struct choose_datas* p = &precomp_limits[nMarks - OGR_NG_MIN];

    if (p->choose_array) {
      return (p->checksum == fletcher16(p->choose_array, (32 << CHOOSE_DIST_BITS)));
    }
    else {
      return build_cache(nMarks);
    }
  }
  return 0;
}


/* Cache the limits
 */
static void cache_limits(u16* pDatas, int nMarks)
{
   u32  dist;
   int  depth;
   int  nsegs       = nMarks - 1;
   int  midseg_size = 2 - (nsegs & 1);
   int  midseg_pos  = (nsegs - midseg_size) / 2;


   for (dist = 0; dist < (1 << CHOOSE_DIST_BITS); dist++) {
      int limit, temp;
      u16* table = &pDatas[dist * 32];

      table[0] = 0;
      for (depth = 1; depth <= nsegs; depth++) {
         limit = OGR[nsegs] - min_length(nsegs - depth, dist);
         if (depth <= midseg_pos) {
            temp = (OGR[nsegs] - 1 - min_length(midseg_size, dist)) / 2
                 - min_length(midseg_pos - depth, dist);
            if (temp < limit) {
               limit = temp;
            }
         }

         table[depth] = (limit < 0) ? 0 : (u16)limit;
      }
   }
}


static int min_length(int nSegs, u32 distSet)
{
   int len;

   if (nSegs < CHOOSE_MARKS) {
      len = choose_dat.choose_array[distSet * CHOOSE_MARKS + nSegs];
   }
   else {   /* Compute an estimate */
      int min_length = 0;
      int distance   = 1;
      u32 mask       = 1 << (CHOOSE_DIST_BITS - 1);

      len = choose_dat.choose_array[distSet * CHOOSE_MARKS + CHOOSE_MARKS - 1]
          + choose_dat.choose_array[distSet * CHOOSE_MARKS + nSegs - (CHOOSE_MARKS - 1)];

      if (len < OGR[nSegs]) {
         len = OGR[nSegs];
      }

      while (nSegs > 0) {
         if (0 == (distSet & mask)) {
            min_length += distance;
            --nSegs;
         }
         distSet <<= 1;
         ++distance;
      }

      if (min_length > len) {
         len = min_length;
      }
   }

   return len;
}
