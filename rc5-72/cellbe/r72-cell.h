#ifndef __R72_CELL_H__
#define __R72_CELL_H__

#include "ccoreio.h"
#include "cputypes.h"

typedef struct
{
  RC5_72UnitWork rc5_72unitwork;  /* 44 bytes */
  u32 iterations;                 /* 48 bytes */
  u32 signature;                  /* 52 bytes */
  u32 pad1, pad2, pad3;           /* pad to 16 */
} CellR72CoreArgs;

#define CELL_RC5_72_SIGNATURE  0x98765432

#define STATIC_ASSERT(foo)  { typedef int footype[(foo) ? 1 : -1]; }

typedef union {
    ui64 a64;
    u32 a32[2];
} addr64;

#endif // __R72_CELL_H__
