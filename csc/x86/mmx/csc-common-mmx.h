// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#ifndef __CSC_COMMON_MMX_H
#define __CSC_COMMON_MMX_H "@(#)$Id: csc-common-mmx.h,v 1.1 1999/12/08 05:35:46 remi Exp $"

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "cputypes.h" /* u32, s32 */

#define _FORCE64_

  #define CSC_BIT_64
  #if defined(__GNUC__)
    typedef unsigned long long ulong;
    #define CASTNUM64(n) n##ull
  #elif ( defined(_MSC_VER) /*>=11*/ || defined(__WATCOMC__) /*>=10*/)
    typedef unsigned __int64 ulong;
    #define CASTNUM64(n) n##ul
  #else
    #error something missing
  #endif

#define _0 ( (ulong)0)
#define _1 (~(_0))

#define PASTE( xx, yy ) PASTE1( xx, yy )
#define PASTE1( xx, yy ) xx##yy

// ------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
// bitslice version of c0..c8
extern const ulong csc_tabc_mmx[9][64];

// bitslice version of e and e'
extern const ulong csc_tabe_mmx[2][64];

// table-lookup implementation of csc_transP()
// (defined in csc-comon.cpp)
extern const u8 csc_tabp[256];

extern const ulong mmNOT;

#ifdef __cplusplus
}
#endif

// ------------------------------------------------------------------

typedef struct {
  ulong in[8];
  ulong *out[8];
} csc_mmxParameters;

// parameters are passed in %mm0..%mm7 and in %edx
extern "C" void csc_transP2( void ) asm ("csc_transP2");

#endif /* ifndef __CSC_COMMON_MMX_H */

// ------------------------------------------------------------------
