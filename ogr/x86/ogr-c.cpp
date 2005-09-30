/*
 * Copyright distributed.net 2001-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Wrapper around ogr.cpp for all processor WITH a fast bsr instruction.
 * (ie, PPro, PII, PIII)
 *
 * $Id: ogr-c.cpp,v 1.1.2.3 2005/09/30 05:37:23 stream Exp $
*/

#define OGR_GET_DISPATCH_TABLE_FXN    ogr_get_dispatch_table_foo /* shutup declaration in ogr.cpp */

#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
#define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
#define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
#define OGROPT_HAVE_OGR_CYCLE_ASM             1 /* 0-2 - 'yes, partial'  */
#define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
#define OGROPT_ALTERNATE_CYCLE                0 /* 0-2 - 'no'  (default) */
#define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */

#include "asm-x86.h"

#if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && !defined(__CNTLZ)
  #warning Macro __CNTLZ not defined. OGROPT_FFZ reset to 0.
  #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0
#endif

#include "ansi/ogr.cpp"
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

#define FIRST_ASM_CORE   2
#define TOTAL_ASM_CORES  2

#define CYCLE_THUNK(func) \
extern "C" int CDECL ogr_##func##_asm( \
    void *state, \
    int *pnodes, \
    int with_time_constraints, \
    unsigned char const *choose_dat, \
    int (CDECL *found_one_cdecl_func)(const struct State *oState) \
); \
static int cycle_thunk_##func(void *state, int *pnodes, int with_time_constraints) \
{ \
    return ogr_##func##_asm(state, pnodes, with_time_constraints, ogr_choose_dat, found_one_cdecl_thunk); \
}

static int CDECL found_one_cdecl_thunk(const struct State *oState)
{
    return found_one(oState);
}

CYCLE_THUNK(watcom_rt1);

extern void setup_asm64_ogr_core(CoreDispatchTable *table, int index);

CoreDispatchTable * ogr_get_dispatch_table_asm (int coresel)
{
  static CoreDispatchTable dispatch_table[TOTAL_ASM_CORES];
  int    i;

  STATIC_ASSERT( sizeof(struct Level) == 0x44 );
  STATIC_ASSERT( offsetof(struct State, Levels) == 32 );

  for (i = 0; i < TOTAL_ASM_CORES; i++)
  {
    dispatch_table[i].init      = ogr_init;
    dispatch_table[i].create    = ogr_create;
    dispatch_table[i].getresult = ogr_getresult;
    dispatch_table[i].destroy   = ogr_destroy;
    dispatch_table[i].cleanup   = ogr_cleanup;
  }
  dispatch_table[0].cycle  = cycle_thunk_watcom_rt1;

/*
 * Sorry for a little mess (dispatch_table[1] unnecessary and incorrectly
 * filled in loop above, then completely overwritten in setup_asm64_ogr_core()).
 * It's simpler because only one 64-bit asm core exist now.
 */
  setup_asm64_ogr_core(&dispatch_table[1], 0);

  coresel -= FIRST_ASM_CORE;
  if (coresel < 0 || coresel >= TOTAL_ASM_CORES)
    return NULL;

  return &dispatch_table[coresel];
}

#if defined(__cplusplus)
}
#endif
