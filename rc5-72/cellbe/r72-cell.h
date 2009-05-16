#ifndef __R72_CELL_H__
#define __R72_CELL_H__

#include "ccoreio.h"
#include "cputypes.h"

typedef struct
{
  u32 sign1, pad1[3];             /* 16+...   */
  RC5_72UnitWork rc5_72unitwork;  /* 44 bytes */
  u32 iterations;                 /* 48 bytes */
  u32 sign2, pad2[3];             /* pad to 16 */
} CellR72CoreArgs;

#define STATIC_ASSERT(foo)  { typedef int footype[(foo) ? 1 : -1]; }

#define SIGN_PPU_TO_SPU_1   0xDEADFACE
#define SIGN_PPU_TO_SPU_2   0xC0FFEE11
#define SIGN_SPU_TO_PPU_1   0xFEEDBEEF
#define SIGN_SPU_TO_PPU_2   0xC0DAC01A

/* Return codes for internal bugchecks. Must be different from normal RESULT_xxx codes. */

#define RETVAL_ERR_BAD_SIGN1      55
#define RETVAL_ERR_BAD_SIGN2      66
#define RETVAL_ERR_TRASHED_SIGN1  77
#define RETVAL_ERR_TRASHED_SIGN2  88

typedef union {
    ui64 a64;
    u32 a32[2];
} addr64;

#endif // __R72_CELL_H__
