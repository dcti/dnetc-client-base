#include "problem.h"

extern "C" {
int crunch_allitnil( RC5UnitWork *work, unsigned long iterations ); /* asm */
s32 rc5_unit_func_allitnil( RC5UnitWork *, u32 *timeslice, void *); /* this */
u32 rc5_unit_func_allitnil_compat( RC5UnitWork *, u32 iterations ); /* this */
}

u32 rc5_unit_func_allitnil_compat( RC5UnitWork *work, u32 iterations )
{
  return crunch_allitnil( work, iterations );
}  

s32 rc5_unit_func_allitnil( RC5UnitWork *work, u32 *timeslice, void *savstate )
{
  u32 kiter = 0;
  savstate = savstate; /* not needed. shaddup compiler */

  kiter = crunch_allitnil( work, *timeslice );
  if (*timeslice == kiter)
    return RESULT_WORKING;
  if (*timeslice < kiter) {
    *timeslice = kiter;
    return RESULT_FOUND;
  }    
  /* shouldn't happen with this core */
  *timeslice = kiter; 
  return -1; /* error */
}

