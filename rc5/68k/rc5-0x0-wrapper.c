/*
 * This is a wrapper for the rc5-0x0 cores as needed by Metrowerks CodeWarrior
 * in order to link to the rc5-0x0 object files built by MPW Asm.
 */

#include "cputypes.h"
#include "problem.h"

#ifdef __cplusplus
extern "C" {
#endif

   #pragma parameter __d0 _rc5_unit_func_000_010re(__a0, __d0)
   u32 _rc5_unit_func_000_010re( RC5UnitWork *, u32 );
   u32 rc5_unit_func_000_010re( RC5UnitWork *, u32 );

   #pragma parameter __d0 _rc5_unit_func_020_030(__a0, __d0)
   u32 _rc5_unit_func_020_030( RC5UnitWork *, u32 );
   u32 rc5_unit_func_020_030( RC5UnitWork *, u32 );

   #pragma parameter __d0 _rc5_unit_func_060re(__a0, __d0)
   u32 _rc5_unit_func_060re( RC5UnitWork *, u32 );
   u32 rc5_unit_func_060re( RC5UnitWork *, u32 );

#ifdef __cplusplus
}
#endif

u32 rc5_unit_func_000_010re( RC5UnitWork *work, u32 iter )
{
   return (u32) _rc5_unit_func_000_010re( work, iter );
}

u32 rc5_unit_func_020_030( RC5UnitWork *work, u32 iter )
{
   return (u32) _rc5_unit_func_020_030( work, iter );
}

u32 rc5_unit_func_060re( RC5UnitWork *work, u32 iter )
{
   return (u32) _rc5_unit_func_060re( work, iter );
}

