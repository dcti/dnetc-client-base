// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: ogr.cpp,v $
// Revision 1.2  1999/03/20 18:14:51  gregh
// Fix ogr.h includes.
//
// Revision 1.1  1999/03/18 07:45:40  gregh
// Renamed from .c
//
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "client2.h"
#include "crc32.h"
#include "ogr.h"

/* this is in choosedat.c */
extern unsigned char choose_dat[];

typedef unsigned long U;

#define CHOOSEBITS 12
#define BITMAPS     5       /* need to change macros when changing this */
#define MAXDEPTH   40
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

struct Level {
  U list[BITMAPS];
  U dist[BITMAPS];
  U comp[BITMAPS];
  int cnt1;
  int cnt2;
  int limit;
};

struct State {
  double Nodes;                   /* counts "tree branches" */
  int max;                        /* maximum length of ruler */
  int maxdepth;                   /* maximum number of marks in ruler */
  int maxdepthm1;                 /* maxdepth-1 */
  int half_length;                /* half of max */
  int half_depth;                 /* half of maxdepth */
  int half_depth2;                /* half of maxdepth, adjusted for 2nd mark */
  int marks[MAXDEPTH+1];          /* current length */
  int startdepth;
  int depth;
  int limit;
  int LOGGING;
  struct Level Levels[MAXDEPTH];
};

#define COMP_LEFT_LIST_RIGHT(lev,s)                             \
  {                                                             \
    int ss = 32 - s;                                            \
    lev->comp[0] = (lev->comp[0] << s) | (lev->comp[1] >> ss);  \
    lev->comp[1] = (lev->comp[1] << s) | (lev->comp[2] >> ss);  \
    lev->comp[2] = (lev->comp[2] << s) | (lev->comp[3] >> ss);  \
    lev->comp[3] = (lev->comp[3] << s) | (lev->comp[4] >> ss);  \
    lev->comp[4] <<= s;                                         \
    lev->list[4] = (lev->list[4] >> s) | (lev->list[3] << ss);  \
    lev->list[3] = (lev->list[3] >> s) | (lev->list[2] << ss);  \
    lev->list[2] = (lev->list[2] >> s) | (lev->list[1] << ss);  \
    lev->list[1] = (lev->list[1] >> s) | (lev->list[0] << ss);  \
    lev->list[0] >>= s;                                         \
  }

#define COMP_LEFT_LIST_RIGHT_32(lev)              \
  lev->comp[0] = lev->comp[1];                    \
  lev->comp[1] = lev->comp[2];                    \
  lev->comp[2] = lev->comp[3];                    \
  lev->comp[3] = lev->comp[4];                    \
  lev->comp[4] = 0;                               \
  lev->list[4] = lev->list[3];                    \
  lev->list[3] = lev->list[2];                    \
  lev->list[2] = lev->list[1];                    \
  lev->list[1] = lev->list[0];                    \
  lev->list[0] = 0;

#define COPY_LIST_SET_BIT(lev2,lev,bitindex)      \
  {                                               \
    int d = bitindex;                             \
    if (d <= 32) {                                \
       lev2->list[0] = lev->list[0] | bit[ d ];   \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
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
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4] | bit[ d ];   \
    } else {                                      \
       lev2->list[0] = lev->list[0];              \
       lev2->list[1] = lev->list[1];              \
       lev2->list[2] = lev->list[2];              \
       lev2->list[3] = lev->list[3];              \
       lev2->list[4] = lev->list[4];              \
    }                                             \
  }

#define COPY_DIST_COMP(lev2,lev)                  \
  lev2->dist[0] = lev->dist[0] | lev2->list[0];   \
  lev2->dist[1] = lev->dist[1] | lev2->list[1];   \
  lev2->dist[2] = lev->dist[2] | lev2->list[2];   \
  lev2->dist[3] = lev->dist[3] | lev2->list[3];   \
  lev2->dist[4] = lev->dist[4] | lev2->list[4];   \
  lev2->comp[0] = lev->comp[0] | lev2->dist[0];   \
  lev2->comp[1] = lev->comp[1] | lev2->dist[1];   \
  lev2->comp[2] = lev->comp[2] | lev2->dist[2];   \
  lev2->comp[3] = lev->comp[3] | lev2->dist[3];   \
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
static int found_one(struct State *State)
{
  /* confirm ruler is golomb */
  {
    int diff, i, j;
    char diffs[1024];
    for (i = 1; i <= State->max/2; i++) diffs[i] = 0;
    for (i = 1; i < State->maxdepth; i++) {
      for (j = 0; j < i; j++) {
        diff = State->marks[i] - State->marks[j];
        if (diff+diff <= State->max) {        /* Principle 1 */
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
    for (j = k; j < k+m; j++) first[j] = i;
    k += m;
    m >>= 1;
  }
  first[0xffff] = 17;     /* just in case we use it */

  return CORE_S_OK;
}

static void dump(int depth, struct Level *lev, int limit)
{
  printf("--- depth %d\n", depth);
  printf("list=%08lx%08lx%08lx%08lx%08lx\n", lev->list[0], lev->list[1], lev->list[2], lev->list[3], lev->list[4]);
  printf("dist=%08lx%08lx%08lx%08lx%08lx\n", lev->dist[0], lev->dist[1], lev->dist[2], lev->dist[3], lev->dist[4]);
  printf("comp=%08lx%08lx%08lx%08lx%08lx\n", lev->comp[0], lev->comp[1], lev->comp[2], lev->comp[3], lev->comp[4]);
  printf("cnt1=%d cnt2=%d limit=%d\n", lev->cnt1, lev->cnt2, limit);
}

static int ogr_create(void *input, int inputlen, void **state)
{
  struct State *State;
  struct Stub *stub = (struct Stub *)input;

  if (input == NULL) {
    return CORE_E_FORMAT;
  }

  State = (struct State *)malloc(sizeof(struct State));
  if (State == NULL) {
    return CORE_E_MEMORY;
  }
  *state = State;

  memset(State, 0, sizeof(struct State));

  State->maxdepth = stub->marks;
  State->maxdepthm1 = State->maxdepth-1;

  if (State->maxdepth > sizeof(OGR)/sizeof(OGR[0])) {
    return CORE_E_FORMAT;
  }

  State->max = OGR[State->maxdepth-1];

  /* Note, marks are labled 0, 1...  so mark @ depth=1 is 2nd mark */
  State->half_depth2 = State->half_depth = ((State->maxdepth+1) >> 1) - 1;
  if (!(State->maxdepth % 2)) State->half_depth2++;  /* if even, use 2 marks */

  /* Simulate GVANT's "KTEST=1" */
  State->half_depth--;
  State->half_depth2++;
  /*------------------
  Since:  half_depth2 = half_depth+2 (or 3 if maxdepth even) ...
  We get: half_length2 >= half_length + 3 (or 6 if maxdepth even)
  But:    half_length2 + half_length <= max-1    (our midpoint reduction)
  So:     half_length + 3 (6 if maxdepth even) + half_length <= max-1
  ------------------*/
                              State->half_length = (State->max-4) >> 1;
  if ( !(State->maxdepth%2) ) State->half_length = (State->max-7) >> 1;

  State->depth = 1;

  {
    int i, n;
    struct Level *lev, *lev2;

    n = stub->length;
    lev = &State->Levels[1];
    for (i = 0; i < n; i++) {
      int t;
      int s = stub->stub[i];
      //dump(State->depth, lev, 0);
      State->marks[i+1] = State->marks[i] + s;
      lev->cnt2 += s;
      t = s;
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
      State->depth++;
    }
  }

  State->startdepth = State->depth;

/*
  printf("sizeof      = %d\n", sizeof(struct State));
  printf("max         = %d\n", State->max);
  printf("maxdepth    = %d\n", State->maxdepth);
  printf("maxdepthm1  = %d\n", State->maxdepthm1);
  printf("half_length = %d\n", State->half_length);
  printf("half_depth  = %d\n", State->half_depth);
  printf("half_depth2 = %d\n", State->half_depth2);
  {
    int i;
    printf("marks       = ");
    for (i = 1; i < State->startdepth; i++) {
      printf("%d ", State->marks[i]-State->marks[i-1]);
    }
    printf("\n");
  }
*/

  return CORE_S_OK;
}

static void dump_ruler(struct State *State, int depth)
{
  int i;
  printf("max %d ruler ", State->max);
  for (i = 1; i < depth; i++) {
    printf("%d ", State->marks[i] - State->marks[i-1]);
  }
  printf("\n");
}

static int ogr_cycle(void *state, int *pnodes)
{
  struct State *State = (struct State *)state;
  int depth = State->depth;      /* the depth of recursion */
  struct Level *lev = &State->Levels[depth];
  struct Level *lev2;
  int nodes = 0;
  int nodeslimit = *pnodes;
  int retval = CORE_S_CONTINUE;
  int limit;
  int s;
  U comp0;

  //State->LOGGING = 1;
  while (1) {

    State->marks[depth-1] = lev->cnt2;
    if (State->LOGGING) dump_ruler(State, depth);
    if (depth <= State->half_depth2) {
      if (depth <= State->half_depth) {
        //dump_ruler(State, depth);
        if (nodes >= nodeslimit) {
          break;
        }
        limit = State->max - OGR[State->maxdepthm1 - depth];
        limit = limit < State->half_length ? limit : State->half_length;
      } else {
        limit = State->max - choose(lev->dist[0] >> ttmMAXBITS, State->maxdepthm1 - depth);
        limit = limit < State->max - State->marks[State->half_depth]-1 ? limit : State->max - State->marks[State->half_depth]-1;
      }
    } else {
      limit = State->max - choose(lev->dist[0] >> ttmMAXBITS, State->maxdepthm1 - depth);
    }

    if (State->LOGGING) dump(depth, lev, limit);

    nodes++;

    /* Find the next available mark location for this level */
stay:
    comp0 = lev->comp[0];
    if (State->LOGGING) printf("comp0=%08lx\n", comp0);
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
    if (State->LOGGING) printf("depth=%d s=%d len=%d limit=%d\n", depth, s+(lev->cnt2-lev->cnt1), lev->cnt2+s, limit);
    if ((lev->cnt2 += s) > limit) goto up; /* no spaces left */

    COMP_LEFT_LIST_RIGHT(lev, s);
skip_out:

    /* New ruler? */
    if (depth == State->maxdepthm1) {
      State->marks[State->maxdepthm1] = lev->cnt2;       /* not placed yet into list arrays! */
      if (found_one(State)) {
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
    if (depth < State->startdepth) {
      retval = CORE_S_OK;
      break;
    }
    limit = lev->limit;

    goto stay; /* repeat this level till done */
  }

  State->Nodes += nodes;
  State->depth = depth;

  *pnodes = nodes;

  return retval;
}

static int ogr_getresult(void *state, void *result, int resultlen)
{
  struct State *State = (struct State *)state;
  struct Stub *stub = (struct Stub *)result;
  int i;

  if (resultlen != sizeof(struct Stub)) {
    return CORE_E_FORMAT;
  }
  stub->marks = State->maxdepth;
  stub->length = State->depth;
  if (stub->length > STUB_MAX) {
    stub->length = STUB_MAX;
  }
  for (i = 0; i < STUB_MAX; i++) {
    stub->stub[i] = State->marks[i+1] - State->marks[i];
  }
  return CORE_S_OK;
}

static int ogr_destroy(void *state)
{
  free(state);
  return CORE_S_OK;
}

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
  dispatch_table.count     = &ogr_count;
  dispatch_table.save      = &ogr_save;
  dispatch_table.load      = &ogr_load;
  dispatch_table.cleanup   = &ogr_cleanup;
  return &dispatch_table;
}
