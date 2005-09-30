/*
 * Copyright distributed.net 2001-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Wrapper around ogr64.cpp for assembly x86 64-bit cores.
 *
 * $Id: ogr-d.cpp,v 1.1.2.1 2005/09/30 05:37:23 stream Exp $
*/

#include <stddef.h>
#include "cputypes.h"
#include "ccoreio.h"       /* CDECL    */

#ifdef HAVE_I64

/*
 * Intensive optimizations not required, we need only support functions.
 * Default settings are Ok, only OGROPT_HAVE_OGR_CYCLE_ASM changed.
 */

#define OVERWRITE_DEFAULT_OPTIMIZATIONS  1

#include "x86/asm-x86.h"
#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
#define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
#define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
#define OGROPT_HAVE_OGR_CYCLE_ASM             1 /* 0-2 - 'yes', need found_one() */
#define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
#define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */

#if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && !defined(__CNTLZ)
  #warning Macro __CNTLZ not defined. OGROPT_FFZ reset to 0.
  #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0
#endif

#include "ansi/ogr-64.cpp"

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
static int cycle_thunk_##func(void *state, int *pnodes, int with_time_constraints) \
{ \
    return ogr_##func##_asm(state, pnodes, with_time_constraints, ogr_choose_dat, found_one_cdecl_thunk); \
}

static int CDECL found_one_cdecl_thunk(const struct State *oState)
{
    return found_one(oState);
}

CYCLE_THUNK(watcom_rt1_mmx64);

void setup_asm64_ogr_core(CoreDispatchTable *table, int index)
{
  (void) index;  /* To be used later, only one 64-bit asm function exist now */

  STATIC_ASSERT( sizeof(struct Level) == 0x50 );
  STATIC_ASSERT( offsetof(struct State, Levels) == 32 );

  table->init      = ogr_init;
  table->create    = ogr_create;
  table->getresult = ogr_getresult;
  table->destroy   = ogr_destroy;
  table->cleanup   = ogr_cleanup;
  table->cycle     = cycle_thunk_watcom_rt1_mmx64;
}

#if defined(__cplusplus)
}
#endif

#else // HAVE_I64

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * Only need definition of CoreDispatchTable. Other OGR setup and optimization
 * options are totally incorrect but we don't care.
 */
#include "ansi/ogr.h"

/*
 * No int64 support. Well, these pointers shouldn't be called at all.
 * Reset all fields to NULL (they've got garbage in ogr-c.cpp).
 * If we've got a GPF, fix core selection routines!
 */
void setup_asm64_ogr_core(CoreDispatchTable *table, int index)
{
  (void) index;

  table->init      = NULL;
  table->create    = NULL;
  table->getresult = NULL;
  table->destroy   = NULL;
  table->cleanup   = NULL;
  table->cycle     = NULL;
}

#if defined(__cplusplus)
}
#endif

#endif // HAVE_I64
