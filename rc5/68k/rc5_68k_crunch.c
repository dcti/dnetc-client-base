/*  "@(#)$Id: rc5_68k_crunch.c,v 1.3 1999/12/08 01:28:52 cyp Exp $" */

#include "cputypes.h"
#include "problem.h"


#ifdef __cplusplus
extern "C" {
#endif
  extern __asm u32 rc5_unit_func_000_030_asm( register __a0 RC5UnitWork *, register __d0 unsigned long );
  extern __asm u32 rc5_unit_func_040_060_asm( register __a0 RC5UnitWork *, register __d0 unsigned long );
  u32 rc5_unit_func_000_030( RC5UnitWork * , u32 );   /* this */
  u32 rc5_unit_func_040_060( RC5UnitWork * , u32 );   /* this */
#ifdef __cplusplus
}
#endif

u32 rc5_unit_func_000_030( RC5UnitWork *work, u32 iter )
{
  return u32 rc5_unit_func_000_030_asm( work, iter );
}

u32 rc5_unit_func_040_060( RC5UnitWork *work, u32 iter )
{
  return u32 rc5_unit_func_040_060_asm( work, iter );
}
