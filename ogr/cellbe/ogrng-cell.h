#ifndef __OGR_CELL_H__
#define __OGR_CELL_H__

#define IMPLEMENT_CELL_CORES         /* Required by ogr-vec.cpp */
#undef  HAVE_FLEGE_PPC_CORES         /* Don't use PPC assembly  */
#include "ppc/ogrng-vec.cpp"         /* Vectored stub structures */

#define OGROPT_SPECIFIC_LEVEL_STRUCT
/*
 ** Level datas - mark and limit can be loaded as SPU vectors 
 */
#ifdef __SPU__
  #define fake_vector(var)  __vector int var
#else
  #define fake_vector(var)  int var; int var##pad[3]
#endif
struct OgrLevel {
   BMAP list[OGRNG_BITMAPS_WORDS];
   BMAP dist[OGRNG_BITMAPS_WORDS];
   BMAP comp[OGRNG_BITMAPS_WORDS];
   fake_vector(mark);
   fake_vector(limit);
};

#ifdef __SPU__
  #include "ansi/ogrng_corestate.h"  /* Get only "State" structure */
#else
  #include "ansi/ogrng_codebase.cpp" /* Get "State" structure and basic code */
#endif

#define SIGN_PPU_TO_SPU_1   0xDEADFACE
#define SIGN_PPU_TO_SPU_2   0xC0FFEE11
#define SIGN_SPU_TO_PPU_1   0xFEEDBEEF
#define SIGN_SPU_TO_PPU_2   0xC0DAC01A

/* Return codes for internal bugchecks. (OGR-NG-cycle normal exit code is always zero) */

#define RETVAL_ERR_BAD_SIGN1      55
#define RETVAL_ERR_BAD_SIGN2      66
#define RETVAL_ERR_TRASHED_SIGN1  77
#define RETVAL_ERR_TRASHED_SIGN2  88

typedef struct
{
  u32 sign1, pad1[3]; /* force padding of 'state' to 16 */
  
  struct OgrState state;
  int pnodes;
  int ret_depth;
  u32 upchoose;
  
  u32 sign2;
  
  u32 cache_misses;  /* have to load new element */
  u32 cache_hits;    /* cached data used */
  u32 cache_purges;  /* have to purge old entry before load of new one */
  u32 cache_search_iters; /* elements checked before decision */
  u32 cache_maxlen;  /* storage blocks in cache */
  u32 cache_curlen;  /* used storage blocks */
  
} CellOGRCoreArgs;

typedef union {
    ui64 a64;
    u32 a32[2];
} addr64;

/*
 * Enable run-time collection of LS cache history usage.
 * Use for debugging only, may seriously affect performance.
 */
// #define GET_CACHE_STATS

// #define STATIC_ASSERT(cond) { typedef int foo[(cond) ? 1 : -1]; }

#endif // __OGR_CELL_H__
