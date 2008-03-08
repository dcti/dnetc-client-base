#ifndef __OGR_CELL_H__
#define __OGR_CELL_H__


#define  IMPLEMENT_CELL_CORES

#include "cputypes.h"
#include "ppc/ogr-vec.cpp"
#include "ansi/ogrp2_corestate.h"

typedef struct
{
  struct State state;
  int pnodes;
} CellOGRCoreArgs;

typedef union {
    ui64 a64;
    u32 a32[2];
} addr64;

#define STATIC_ASSERT(cond) { typedef int foo[(cond) ? 1 : -1]; }

#endif // __OGR_CELL_H__
