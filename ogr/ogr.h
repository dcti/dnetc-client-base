// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#ifndef __OGR_H__
#define __OGR_H__ "@(#)$Id: ogr.h,v 1.3.2.14 2000/07/02 17:09:19 cyp Exp $"

#ifndef u16
#include "cputypes.h"
#endif
#include "client2.h"
#include <limits.h>

#if (UINT_MAX == 0xffff)
  #define OGR_INT_SIZE 2
#elif (UINT_MAX == 0xffffffff)
  #define OGR_INT_SIZE 4
#elif (UINT_MAX == 0xffffffffffffffff)
  #define OGR_INT_SIZE 8
#else
  #error "What's up Doc?"
#endif

// define this to enable LOGGING code
#undef OGR_DEBUG

// specifies the number of ruler diffs can be represented.
// Warning: increasing this will cause all structures based
// on workunit_t in packets.h to change, possibly breaking
// network and buffer structure operations.
#define STUB_MAX 10

struct Stub { /* size is 24 */
  u16 marks;           /* N-mark ruler to which this stub applies */
  u16 length;          /* number of valid elements in the stub[] array */
  u16 diffs[STUB_MAX]; /* first <length> differences in ruler */
};

struct WorkStub { /* size is 28 */
  Stub stub;           /* stub we're working on */
  u32 worklength;      /* depth of current state */
};

// Internal stuff that's not part of the interface but we need for
// declaring the problem work area size.
#define BITMAPS     5       /* need to change macros when changing this */
#define MAXDEPTH   40

typedef u32 U;

struct Level {
  U list[BITMAPS];
  U dist[BITMAPS];
  U comp[BITMAPS];
  int cnt1;
  int cnt2;
  int limit;
};

#define OGR_LEVEL_SIZE (((4*BITMAPS)*3)+(OGR_INT_SIZE*3))

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
#ifdef OGR_DEBUG
  int LOGGING;
#endif
  struct Level Levels[MAXDEPTH];
};

#define OGR_PROBLEM_SIZE (16+ (6*OGR_INT_SIZE)+(OGR_INT_SIZE*(MAXDEPTH+1))+ \
                         (4*OGR_INT_SIZE)+(OGR_LEVEL_SIZE*MAXDEPTH) + 64)
                         //sizeof(struct State)

unsigned long ogr_nodecount(const struct Stub *);
const char *ogr_stubstr_r(const struct Stub *stub, char *buffer, unsigned int bufflen);
const char *ogr_stubstr(const struct Stub *stub);

#endif

