/*
// Just declare it so we can call the assebler version
*/
#include "cputypes.h" /* u32 */
#include "problem.h" /* RC5UnitWork */

extern "C" {
u32 rc5_unit_func_mips_crunch_asm(register RC5UnitWork *, u32 iter); /* that */
u32 rc5_unit_func_mips_crunch(RC5UnitWork *, u32 iterations); /* this */
}

u32 rc5_unit_func_mips_crunch(RC5UnitWork *work, u32 iterations)
{
  return rc5_unit_func_mips_crunch_asm( work, iterations );
}  
