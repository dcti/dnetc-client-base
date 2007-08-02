#ifndef __R72_CELL_H__
#define __R72_CELL_H__

#include "ccoreio.h"
#include "cputypes.h"

struct CellR72CoreArgs
{
  RC5_72UnitWork rc5_72unitwork;
  u32 iterations;
};

union addr64 {
    ui64 a64;
    u32 a32[2];
};


#endif // __R72_CELL_H__
