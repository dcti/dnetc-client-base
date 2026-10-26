/*
 * Copyright distributed.net 2001-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Wrapper around ogr.cpp for all processor WITH a fast bsr instruction.
 * (ie, PPro, PII, PIII)
 *
 * $Id: ogr-c.cpp,v 1.5 2008/12/30 20:58:44 andreasb Exp $
*/

#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
#define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
#define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
#define OGROPT_HAVE_OGR_CYCLE_ASM             1 /* 0-2 - 'yes, partial'  */
#define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */

#define OGR_GET_DISPATCH_TABLE_FXN    ogr_get_dispatch_table_asm_gen

#include "asm-x86-p2.h"
#include "ansi/ogrp2-32.h"
#include "ansi/ogrp2_codebase.cpp"

#include "ccoreio.h"       /* CDECL    */
#include <stddef.h>        /* offsetof */

/*
 * This is a manager for all assembly X86 OGR cores.
 *
 * We have two "C" cores "A" and "B" numbered 0 and 1, so assembly cores will
 * be numbered from 2 and above.
 *
 * All functions, except for ogr_cycle, are taken directly from ANSI source.
 * Since they are called not so often, we'll not care much about optimization
 * options above. Defaults from core "A" are acceptable.
 *
 * Assembly ogr_cycle routine must use "cdecl" calling convention. To avoid mess
 * in declarations inside CoreDispatchTable and possible slowdown of ANSI
 * cores, we'll use special thunking function. A macro CYCLE_THUNK(foo) will
 * create necessary definitions. Thunking function will be called
 * "cycle_thunk_foo" and have default compiler's calling convention. Assembly
 * function must be called "_ogr_foo_asm" or "ogr_foo_asm" (depending on
 * compiler) and accept arguments using "cdecl" calling convention.
 *
 * Likewise, we're creating thunking function "found_one_cdecl_thunk" which is
 * guaranteed to be cdecl. Assembly code must call "found_one_cdecl_thunk"
 * instead of just "found_one" because "found_one" has generally undefined
 * compiler-specific calling convention.  Assembly code must save
 * ebx, ecx, and edx if they're used before this call.
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
    STATIC_ASSERT( sizeof(struct Level) == 0x44 );
    STATIC_ASSERT( offsetof(struct State, Levels) == 32 );
    STATIC_ASSERT( offsetof(struct State, node_offset) == 0x818 );

    return found_one(oState);
}

CYCLE_THUNK(watcom_rt1);

#if defined(__cplusplus)
}
#endif
