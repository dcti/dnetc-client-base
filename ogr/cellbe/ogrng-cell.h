#ifndef __OGR_CELL_H__
#define __OGR_CELL_H__

#define CORE_NAME cellv1

#define IMPLEMENT_CELL_CORES         /* Required by ogr-vec.cpp */
#undef  HAVE_FLEGE_PPC_CORES         /* Don't use PPC assembly  */
#include "ppc/ogrng-vec.cpp"         /* Vectored stub structures */

#ifdef __SPU__
  #include "ansi/ogrng_corestate.h"  /* Get only "State" structure */
#else
  #include "ansi/ogrng_codebase.cpp" /* Get "State" structure and basic code */
#endif

typedef struct
{
  struct OgrState state;
  int pnodes;
  int ret_depth;
  u32 upchoose;
} CellOGRCoreArgs;

typedef union {
    ui64 a64;
    u32 a32[2];
} addr64;

// #define STATIC_ASSERT(cond) { typedef int foo[(cond) ? 1 : -1]; }

#endif // __OGR_CELL_H__
