// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#ifndef __OGR_H__
#define __OGR_H__

#ifndef u16
#include "cputypes.h"
#endif
#include "client2.h"

#define STUB_MAX 10 /* change ogr_packet_t in packets.h when changing this */

struct Stub { /* size is 24 */
  u16 marks;           /* N-mark ruler to which this stub applies */
  u16 length;          /* number of valid elements in the stub[] array */
  u16 diffs[STUB_MAX]; /* first <length> differences in ruler */
};

struct WorkStub { /* size is 28 */
  Stub stub;           /* stub we're working on */
  u32 worklength;      /* depth of current state */
};

/*
 * Internal stuff that's not part of the interface but we need for
 * declaring the problem work area size.
 */

#define BITMAPS     5       /* need to change macros when changing this */
#define MAXDEPTH   40

typedef unsigned long U;

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

#define OGR_PROBLEM_SIZE sizeof(struct State)

#endif
