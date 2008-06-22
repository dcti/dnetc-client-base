#ifndef __OGR_CELL_H__
#define __OGR_CELL_H__

#define IMPLEMENT_CELL_CORES         /* Required by ogr-vec.cpp */
#include "ppc/ogr-vec.cpp"           /* Vectored stub structures */
#ifdef __SPU__
  #include "ansi/ogrp2_corestate.h"  /* Get only "State" structure */
#else
  #include "ansi/ogrp2_codebase.cpp" /* Get "State" structure and basic code */
#endif

typedef struct
{
  struct State state;
  int pnodes;
} CellOGRCoreArgs;

typedef union {
    ui64 a64;
    u32 a32[2];
} addr64;

// #define STATIC_ASSERT(cond) { typedef int foo[(cond) ? 1 : -1]; }

#endif // __OGR_CELL_H__
