#ifndef __OGR_CELL_H__
#define __OGR_CELL_H__

#include "cputypes.h"
#include "ansi/ogr.h"

typedef struct
{
  struct State state;
  int pnodes;
  unsigned signature;
} CellOGRCoreArgs;

#define CELL_OGR_SIGNATURE   0x12345678

typedef union {
    ui64 a64;
    u32 a32[2];
} addr64;

#define STATIC_ASSERT(cond) { typedef int foo[(cond) ? 1 : -1]; }

#endif // __OGR_CELL_H__
