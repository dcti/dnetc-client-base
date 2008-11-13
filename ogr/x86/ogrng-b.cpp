#include "ansi/ogrng-64.h"

#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'            */
#define OGROPT_ALTERNATE_CYCLE                1 /* 0/1 - 'yes'           */
#define OGR_NG_GET_DISPATCH_TABLE_FXN  ogrng_get_dispatch_table_mmx

#include "ansi/ogrng_codebase.cpp"

#include "ccoreio.h"

extern "C" int CDECL ogr_cycle_256_rt1_mmx(struct OgrState *oState, int *pnodes, const u16* pchoose);

static int ogr_cycle_256(struct OgrState *oState, int *pnodes, const u16* pchoose)
{
    return ogr_cycle_256_rt1_mmx(oState, pnodes, pchoose);
}
