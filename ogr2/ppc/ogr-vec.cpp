/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * @(#)$Id: ogr-vec.cpp,v 1.1.2.2 2000/02/02 07:39:47 gregh Exp $
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "client2.h"
#include "crc32.h"
#include "ogr-vec.h"

/* this is in choosedat.c */
extern unsigned char choose_dat[];

#define CHOOSEBITS 12
#define MAXBITS    12
#define ttmMAXBITS (32-MAXBITS)

#define choose(x,y) (choosedat[CHOOSEBITS*(x)+(y)])

static unsigned char *choosedat;
static int OGR[] = {
  /*  1 */    0,   1,   3,   6,  11,  17,  25,  34,  44,  55,
  /* 11 */   72,  85, 106, 127, 151, 177, 199, 216, 246, 283,
  /* 21 */  333, 356, 372, 425, 480, 492, 553, 585, 623
};
static char first[65537];  /* first blank in 16 bit COMP bitmap, range: 1..16 */
static U bit[200];         /* which bit of LIST to update */

#define COMP_LEFT_LIST_RIGHT(lev,s)                             \
  {                                                             \
    int ss = 32 - s;                                            \
    v_u32 s_vec,ss_vec,vec1,vec2;								\
    s_vec.sca[0] = s;											\
    ss_vec.sca[0] = ss;    										\
    s_vec.vec = vec_splat(s_vec.vec,0);							\
    ss_vec.vec = vec_splat(ss_vec.vec,0);						\
    vec1.vec = vec_sl(lev->zeroed_comp_vec.vec,s_vec.vec); 		\
    vec2.vec = vec_sr(lev->offset_comp_vec.vec,ss_vec.vec); 	\
    lev->zeroed_comp_vec.vec = vec_or(vec1.vec,vec2.vec); 		\
    lev->comp[4] <<= s;                                         \
	vec1.vec = vec_sr(lev->offset_list_vec.vec,s_vec.vec); 		\
	vec2.vec = vec_sl(lev->zeroed_list_vec.vec,ss_vec.vec); 	\
	lev->offset_list_vec.vec = vec_or(vec1.vec,vec2.vec); 		\
    lev->list[0] >>= s;                                         \
  }

#define COMP_LEFT_LIST_RIGHT_32(lev)              		\
  lev->zeroed_comp_vec.vec = lev->offset_comp_vec.vec;	\
  lev->comp[4] = 0;                               		\
  lev->offset_list_vec.vec = lev->zeroed_list_vec.vec;	\
  lev->list[0] = 0;

#define COPY_LIST_SET_BIT(lev2,lev,bitindex)      \
  {                                               \
    int d = bitindex;                             \
    if (d <= 32) {                                \
       lev2->list[0] = lev->list[0] | bit[ d ];   \
	   lev2->offset_list_vec.vec = lev->offset_list_vec.vec; \
    } else if (d <= 64) {                         \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1] | bit[ d ];   \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 96) {                         \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2] | bit[ d ];   \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 128) {                        \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3] | bit[ d ];   \
       lev2->list[4] = lev->list[4];              \
    } else if (d <= 160) {                        \
	   lev2->zeroed_list_vec.vec = lev->zeroed_list_vec.vec; \
       lev2->list[4] = lev->list[4] | bit[ d ];   \
    } else {                                      \
	   lev2->zeroed_list_vec.vec = lev->zeroed_list_vec.vec; \
       lev2->list[4] = lev->list[4];              \
    }                                             \
  }

#define COPY_DIST_COMP(lev2,lev)                  \
  lev2->zeroed_dist_vec.vec = vec_or(lev->zeroed_dist_vec.vec,lev2->zeroed_list_vec.vec); \
  lev2->dist[4] = lev->dist[4] | lev2->list[4];   \
  lev2->zeroed_comp_vec.vec = vec_or(lev->zeroed_comp_vec.vec,lev2->zeroed_dist_vec.vec); \
  lev2->comp[4] = lev->comp[4] | lev2->dist[4];

static CoreDispatchTable dispatch_table;

static const unsigned chooseCRC32[24] = {
  0x00000000,   /* 0 */
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,   /* 5 */
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,   /* 10 */
  0x00000000,
  0x01138a7d,
  0x00000000,
  0x00000000,
  0x00000000,   /* 15 */
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,
  0x00000000,   /* 20 */
  0x00000000,
  0x00000000,
  0x00000000
};

static int init_load_choose()
{
  if (MAXBITS != choose_dat[2]) {
    return CORE_E_FORMAT;
  }
  /* skip over the choose.dat header */
  choosedat = &choose_dat[3];

  /* CRC32 check */
  {
    int i, j;
    unsigned crc32 = 0xffffffff;
    crc32 = CRC32(crc32, choose_dat[0]);
    crc32 = CRC32(crc32, choose_dat[1]);
    crc32 = CRC32(crc32, choose_dat[2]);           /* This varies a lot */
    for (j = 0; j < (1 << MAXBITS); j++) {
      for (i = 0; i < CHOOSEBITS; ++i) crc32 = CRC32(crc32, choose(j, i));
    }
    crc32 = ~crc32;
    if (chooseCRC32[MAXBITS] != crc32) {
      /* printf("Your choose.dat (CRC=%08x) is corrupted! Oh well, continuing anyway.\n", crc32); */
      return CORE_E_FORMAT;
    }
  }

  return CORE_S_OK;
}

/*-----------------------------------------*/
/*  found_one() - print out golomb rulers  */
/*-----------------------------------------*/
static int found_one(struct State *oState)
{
  /* confirm ruler is golomb */
  {
    int diff, i, j;
    char diffs[1024];
    for (i = 1; i <= oState->max/2; i++) diffs[i] = 0;
    for (i = 1; i < oState->maxdepth; i++) {
      for (j = 0; j < i; j++) {
        diff = oState->marks[i] - oState->marks[j];
        if (diff+diff <= oState->max) {        /* Principle 1 */
          if (diff <= 64) break;      /* 2 bitmaps always tracked */
          if (diffs[diff]) return 0;
          diffs[diff] = 1;
        }
      }
    }
  }
  return 1;
}

static int ogr_init()
{
  int r, i, j, k, m;
  
  r = init_load_choose();
  if (r != CORE_S_OK) {
    return r;
  }

  for( i=1; i < 200; i++) {
     bit[i] = 0x80000000 >> ((i-1) % 32);
  }

  /* first zero bit in 16 bits */
  k = 0; m = 0x8000;
  for (i = 1; i <= 16; i++) {
    for (j = k; j < k+m; j++) first[j] = (char)i;
    k += m;
    m >>= 1;
  }
  first[0xffff] = 17;     /* just in case we use it */

  return CORE_S_OK;
}

#ifdef OGR_DEBUG
static void dump(int depth, struct Level *lev, int limit)
{
  printf("--- depth %d\n", depth);
  printf("list=%08x%08x%08x%08x%08x\n", lev->list[0], lev->list[1], lev->list[2], lev->list[3], lev->list[4]);
  printf("dist=%08x%08x%08x%08x%08x\n", lev->dist[0], lev->dist[1], lev->dist[2], lev->dist[3], lev->dist[4]);
  printf("comp=%08x%08x%08x%08x%08x\n", lev->comp[0], lev->comp[1], lev->comp[2], lev->comp[3], lev->comp[4]);
  printf("cnt1=%d cnt2=%d limit=%d\n", lev->cnt1, lev->cnt2, limit);
  //sleep(1);
}
#endif

static int ogr_create(void *input, int inputlen, void *state, int statelen)
{
  struct State *oState;
  struct WorkStub *workstub = (struct WorkStub *)input;

  if (input == NULL || inputlen != sizeof(struct WorkStub)) {
    return CORE_E_FORMAT;
  }

  if (((unsigned int)statelen) < sizeof(struct State)) {
    return CORE_E_FORMAT;
  }
  oState = (struct State *)state;
  if (oState == NULL) {
    return CORE_E_MEMORY;
  }

  memset(oState, 0, sizeof(struct State));

  oState->maxdepth = workstub->stub.marks;
  oState->maxdepthm1 = oState->maxdepth-1;

  if (((unsigned int)oState->maxdepth) > (sizeof(OGR)/sizeof(OGR[0]))) {
    return CORE_E_FORMAT;
  }

  oState->max = OGR[oState->maxdepth-1];

  /* Note, marks are labled 0, 1...  so mark @ depth=1 is 2nd mark */
  oState->half_depth2 = oState->half_depth = ((oState->maxdepth+1) >> 1) - 1;
  if (!(oState->maxdepth % 2)) oState->half_depth2++;  /* if even, use 2 marks */

  /* Simulate GVANT's "KTEST=1" */
  oState->half_depth--;
  oState->half_depth2++;
  /*------------------
  Since:  half_depth2 = half_depth+2 (or 3 if maxdepth even) ...
  We get: half_length2 >= half_length + 3 (or 6 if maxdepth even)
  But:    half_length2 + half_length <= max-1    (our midpoint reduction)
  So:     half_length + 3 (6 if maxdepth even) + half_length <= max-1
  ------------------*/
                               oState->half_length = (oState->max-4) >> 1;
  if ( !(oState->maxdepth%2) ) oState->half_length = (oState->max-7) >> 1;

  oState->depth = 1;

  {
    int i, n;
    struct Level *lev, *lev2;

    n = workstub->worklength;
    if (n < workstub->stub.length) {
      n = workstub->stub.length;
    }
    if (n > STUB_MAX) {
      return CORE_E_FORMAT;
    }
    lev = &oState->Levels[1];
    for (i = 0; i < n; i++) {
      int limit;
      if (oState->depth <= oState->half_depth2) {
        if (oState->depth <= oState->half_depth) {
          limit = oState->max - OGR[oState->maxdepthm1 - oState->depth];
          limit = limit < oState->half_length ? limit : oState->half_length;
        } else {
          limit = oState->max - choose(lev->dist[0] >> ttmMAXBITS, oState->maxdepthm1 - oState->depth);
          limit = limit < oState->max - oState->marks[oState->half_depth]-1 ? limit : oState->max - oState->marks[oState->half_depth]-1;
        }
      } else {
        limit = oState->max - choose(lev->dist[0] >> ttmMAXBITS, oState->maxdepthm1 - oState->depth);
      }
      lev->limit = limit;
      int s = workstub->stub.diffs[i];
      //dump(oState->depth, lev, 0);
      oState->marks[i+1] = oState->marks[i] + s;
      lev->cnt2 += s;
      int t = s;
      while (t >= 32) {
        COMP_LEFT_LIST_RIGHT_32(lev);
        t -= 32;
      }
      if (t > 0) {
        COMP_LEFT_LIST_RIGHT(lev, t);
      }
      lev2 = lev + 1;
      COPY_LIST_SET_BIT(lev2, lev, s);
      COPY_DIST_COMP(lev2, lev);
      lev2->cnt1 = lev->cnt2;
      lev2->cnt2 = lev->cnt2;
      lev++;
      oState->depth++;
    }
    oState->depth--; // externally visible depth is one less than internal
  }

  oState->startdepth = workstub->stub.length;

/*
  printf("sizeof      = %d\n", sizeof(struct State));
  printf("max         = %d\n", oState->max);
  printf("maxdepth    = %d\n", oState->maxdepth);
  printf("maxdepthm1  = %d\n", oState->maxdepthm1);
  printf("half_length = %d\n", oState->half_length);
  printf("half_depth  = %d\n", oState->half_depth);
  printf("half_depth2 = %d\n", oState->half_depth2);
  {
    int i;
    printf("marks       = ");
    for (i = 1; i < oState->depth; i++) {
      printf("%d ", oState->marks[i]-oState->marks[i-1]);
    }
    printf("\n");
  }
*/

  return CORE_S_OK;
}

#ifdef OGR_DEBUG
static void dump_ruler(struct State *oState, int depth)
{
  int i;
  printf("max %d ruler ", oState->max);
  for (i = 1; i < depth; i++) {
    printf("%d ", oState->marks[i] - oState->marks[i-1]);
  }
  printf("\n");
}
#endif

static int ogr_cycle(void *state, int *pnodes)
{
  struct State *oState = (struct State *)state;
  int depth = oState->depth+1;      /* the depth of recursion */
  struct Level *lev = &oState->Levels[depth];
  struct Level *lev2;
  int nodes = 0;
  int nodeslimit = *pnodes;
  int retval = CORE_S_CONTINUE;
  int limit;
  int s;
  U comp0;

#ifdef OGR_DEBUG
  oState->LOGGING = 1;
#endif
  for (;;) {

    oState->marks[depth-1] = lev->cnt2;
#ifdef OGR_DEBUG
    if (oState->LOGGING) dump_ruler(oState, depth);
#endif
    if (depth <= oState->half_depth2) {
      if (depth <= oState->half_depth) {
        //dump_ruler(oState, depth);
        if (nodes >= nodeslimit) {
          break;
        }
        limit = oState->max - OGR[oState->maxdepthm1 - depth];
        limit = limit < oState->half_length ? limit : oState->half_length;
      } else {
        limit = oState->max - choose(lev->dist[0] >> ttmMAXBITS, oState->maxdepthm1 - depth);
        limit = limit < oState->max - oState->marks[oState->half_depth]-1 ? limit : oState->max - oState->marks[oState->half_depth]-1;
      }
    } else {
      limit = oState->max - choose(lev->dist[0] >> ttmMAXBITS, oState->maxdepthm1 - depth);
    }

#ifdef OGR_DEBUG
    if (oState->LOGGING) dump(depth, lev, limit);
#endif

    nodes++;

    /* Find the next available mark location for this level */
stay:
    comp0 = lev->comp[0];
#ifdef OGR_DEBUG
    if (oState->LOGGING) printf("comp0=%08x\n", comp0);
#endif
    if (comp0 < 0xffff0000) {
      s = first[comp0 >> 16];
    } else {
      if (comp0 < 0xfffffffe) {
        /* s = 16 + first[comp0 & 0x0000ffff]; slow code */
        s = 16 + first[comp0 - 0xffff0000];
      } else {
        /* s>32 */
        if ((lev->cnt2 += 32) > limit) goto up; /* no spaces left */
        COMP_LEFT_LIST_RIGHT_32(lev);
        if (comp0 == 0xffffffff) goto stay;
        goto skip_out;
      }
    }
#ifdef OGR_DEBUG
    if (oState->LOGGING) printf("depth=%d s=%d len=%d limit=%d\n", depth, s+(lev->cnt2-lev->cnt1), lev->cnt2+s, limit);
#endif
    if ((lev->cnt2 += s) > limit) goto up; /* no spaces left */

    COMP_LEFT_LIST_RIGHT(lev, s);
skip_out:

    /* New ruler? */
    if (depth == oState->maxdepthm1) {
      oState->marks[oState->maxdepthm1] = lev->cnt2;       /* not placed yet into list arrays! */
      if (found_one(oState)) {
        retval = CORE_S_SUCCESS;
        break;
      }
      goto stay;
    }

    /* Go Deeper */
    lev2 = lev + 1;
    COPY_LIST_SET_BIT(lev2, lev, lev->cnt2-lev->cnt1);
    COPY_DIST_COMP(lev2, lev);
    lev2->cnt1 = lev->cnt2;
    lev2->cnt2 = lev->cnt2;
    lev->limit = limit;
    lev++;
    depth++;
    continue;

up:
    lev--;
    depth--;
    if (depth <= oState->startdepth) {
      retval = CORE_S_OK;
      break;
    }
    limit = lev->limit;

    goto stay; /* repeat this level till done */
  }

  oState->Nodes += nodes;
  oState->depth = depth-1;

  *pnodes = nodes;

  return retval;
}

static int ogr_getresult(void *state, void *result, int resultlen)
{
  struct State *oState = (struct State *)state;
  struct WorkStub *workstub = (struct WorkStub *)result;
  int i;

  if (resultlen != sizeof(struct WorkStub)) {
    return CORE_E_FORMAT;
  }
  workstub->stub.marks = (u16)oState->maxdepth;
  workstub->stub.length = (u16)oState->startdepth;
  for (i = 0; i < STUB_MAX; i++) {
    workstub->stub.diffs[i] = (u16)(oState->marks[i+1] - oState->marks[i]);
  }
  workstub->worklength = oState->depth;
  if (workstub->worklength > STUB_MAX) {
    workstub->worklength = STUB_MAX;
  }
  return CORE_S_OK;
}

static int ogr_destroy(void *state)
{
  state = state;
  return CORE_S_OK;
}

#if 0
static int ogr_count(void *state)
{
  return sizeof(struct State);
}

static int ogr_save(void *state, void *buffer, int buflen)
{
  if (buflen < sizeof(struct State)) {
    return CORE_E_MEMORY;
  }
  memcpy(buffer, state, sizeof(struct State));
  return CORE_S_OK;
}

static int ogr_load(void *buffer, int buflen, void **state)
{
  if (buflen < sizeof(struct State)) {
    return CORE_E_FORMAT;
  }
  *state = malloc(sizeof(struct State));
  if (*state == NULL) {
    return CORE_E_MEMORY;
  }
  memcpy(*state, buffer, sizeof(struct State));
  return CORE_S_OK;
}
#endif

static int ogr_cleanup()
{
  return CORE_S_OK;
}

CoreDispatchTable *ogr_get_dispatch_table()
{
  dispatch_table.init      = &ogr_init;
  dispatch_table.create    = &ogr_create;
  dispatch_table.cycle     = &ogr_cycle;
  dispatch_table.getresult = &ogr_getresult;
  dispatch_table.destroy   = &ogr_destroy;
#if 0
  dispatch_table.count     = &ogr_count;
  dispatch_table.save      = &ogr_save;
  dispatch_table.load      = &ogr_load;
#endif
  dispatch_table.cleanup   = &ogr_cleanup;
  return &dispatch_table;
}
