#include "ansi/ogrng-64.h"

#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'            */
#define OGROPT_ALTERNATE_CYCLE                1 /* 0/1 - 'yes'           */
#define OGR_NG_GET_DISPATCH_TABLE_FXN  ogrng64_get_dispatch_table_cj1_generic
#define OGROPT_SPECIFIC_LEVEL_STRUCT

/*
 ** Level datas.
 */
struct OgrLevel {
   BMAP list[OGRNG_BITMAPS_WORDS];
   BMAP dist[OGRNG_BITMAPS_WORDS];
   BMAP comp[OGRNG_BITMAPS_WORDS];
   int mark;
   int limit;
   int pad0;
   int pad1;
};

#include "ansi/ogrng_codebase.cpp"

#include "ccoreio.h"       /* CDECL */
#include <stddef.h>        /* offsetof */

extern "C" int CDECL ogrng64_cycle_256_cj1_generic(struct OgrState *oState, int *pnodes, const u16* pchoose);

static int ogr_cycle_256(struct OgrState *oState, int *pnodes, const u16* pchoose)
{
    /* Check structures layout and alignment to match assembly */

    STATIC_ASSERT(offsetof(struct OgrState, max)         == 0 );
    STATIC_ASSERT(offsetof(struct OgrState, maxdepthm1)  == 8 );
    STATIC_ASSERT(offsetof(struct OgrState, half_depth)  == 12);
    STATIC_ASSERT(offsetof(struct OgrState, half_depth2) == 16);
    STATIC_ASSERT(offsetof(struct OgrState, stopdepth)   == 24);
    STATIC_ASSERT(offsetof(struct OgrState, depth)       == 28);
    STATIC_ASSERT(offsetof(struct OgrState, Levels)      == 32);

    STATIC_ASSERT(sizeof(struct OgrLevel) == 112);
    STATIC_ASSERT(sizeof(oState->Levels)  == 112 * OGR_MAXDEPTH);

    STATIC_ASSERT(offsetof(struct OgrLevel, list)  ==   0);
    STATIC_ASSERT(offsetof(struct OgrLevel, dist)  ==  32);
    STATIC_ASSERT(offsetof(struct OgrLevel, comp)  ==  64);
    STATIC_ASSERT(offsetof(struct OgrLevel, mark)  ==  96);
    STATIC_ASSERT(offsetof(struct OgrLevel, limit) == 100);

    return ogrng64_cycle_256_cj1_generic(oState, pnodes, pchoose);
}
