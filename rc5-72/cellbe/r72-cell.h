#ifndef __R72_CELL_H__
#define __R72_CELL_H__

#include "ccoreio.h"
#include "cputypes.h"

typedef struct
{
  RC5_72UnitWork rc5_72unitwork;
  u32 iterations;
} CellR72CoreArgs;

typedef union {
    ui64 a64;
    u32 a32[2];
} addr64;

#endif // __R72_CELL_H__
