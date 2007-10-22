#ifndef __OGR_CELL_H__
#define __OGR_CELL_H__

#include "cputypes.h"
#include "ansi/ogr.h"

typedef struct
{
  struct State state;
  int pnodes;
} CellOGRCoreArgs;

typedef union {
    ui64 a64;
    u32 a32[2];
} addr64;


#endif // __OGR_CELL_H__
