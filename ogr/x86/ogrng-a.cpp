#include "ansi/ogrng-32.h"

#include "x86/asm-x86.h"

#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
#define OGROPT_ALTERNATE_CYCLE                1 /* 0/1 - 'yes'           */
#define OGR_NG_GET_DISPATCH_TABLE_FXN  ogrng_get_dispatch_table_asm1

#include "ansi/ogrng_codebase.cpp"

extern "C" int cdecl ogr_cycle_256_rt1(struct OgrState *oState, int *pnodes, const u16* pchoose);

static int ogr_cycle_256(struct OgrState *oState, int *pnodes, const u16* pchoose)
{
    return ogr_cycle_256_rt1(oState, pnodes, pchoose);
}
