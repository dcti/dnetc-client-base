/*
 * Copyright distributed.net 2001-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Wrapper around ogr64.cpp for assembly x86 64-bit cores.
 *
 * $Id: ogr-e.cpp,v 1.5 2008/12/30 20:58:44 andreasb Exp $
*/

#include <stddef.h>
#include "cputypes.h"      /* HAVE_I64 */
#include "ccoreio.h"       /* CDECL    */

#ifdef HAVE_I64

/*
 * Intensive optimizations not required, we need only support functions.
 * Default settings are Ok, only OGROPT_HAVE_OGR_CYCLE_ASM changed.
 */

#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
#define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
#define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
#define OGROPT_HAVE_OGR_CYCLE_ASM             1 /* 0-2 - 'yes', need found_one() */

#define OGR_GET_DISPATCH_TABLE_FXN    ogr_get_dispatch_table_asm_mmx_amd

#include "asm-x86-p2.h"
#include "ansi/ogrp2-64.h"
#include "ansi/ogrp2_codebase.cpp"

/*
 * This is a manager for all 64-bit assembly X86 OGR cores.
 *
 * See comments in ogr-c.cpp regarding calling conventions and function names.
 */

#if defined(__cplusplus)
extern "C" {
#endif

#define CYCLE_THUNK(func) \
extern "C" int CDECL ogr_##func##_asm( \
    void *state, \
    int *pnodes, \
    int with_time_constraints, \
    unsigned char const *choose_dat, \
    int (CDECL *found_one_cdecl_func)(const struct State *oState) \
); \
static int ogr_cycle(void *state, int *pnodes, int with_time_constraints) \
{ \
    return ogr_##func##_asm(state, pnodes, with_time_constraints, ogr_choose_dat, found_one_cdecl_thunk); \
}

static int CDECL found_one_cdecl_thunk(const struct State *oState)
{
    STATIC_ASSERT( sizeof(struct Level) == 0x50 );
    STATIC_ASSERT( offsetof(struct State, Levels) == 32 );
    STATIC_ASSERT( offsetof(struct State, node_offset) == 0x980 );

    return found_one(oState);
}

CYCLE_THUNK(watcom_rt1_mmx64_amd);

#if defined(__cplusplus)
}
#endif

#endif // HAVE_I64
