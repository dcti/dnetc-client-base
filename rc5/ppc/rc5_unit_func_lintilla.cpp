#include "problem.h"

extern "C" {
int crunch_lintilla( RC5UnitWork *work, unsigned long iterations ); /* asm */
int crunch_allitnil( RC5UnitWork *work, unsigned long iterations ); /* asm */
s32 rc5_unit_func_lintilla( RC5UnitWork *, u32 *timeslice, void *); /* this */ 
u32 rc5_unit_func_lintilla_compat( RC5UnitWork *, u32 iterations ); /* this */ 
}  

u32 rc5_unit_func_lintilla_compat( RC5UnitWork *work, u32 iterations )
{                                         /* old style calling convention */
  u32 kiter = 0; /* keys processed so far */
  do 
  {
    kiter += crunch_lintilla(work, iterations-kiter);
    if (kiter < iterations)
    {
      if (crunch_allitnil(work,1) == 0) 
        break;
      kiter++;
    }
  } while (kiter < iterations);
  return kiter;
}

s32 rc5_unit_func_lintilla( RC5UnitWork *work, u32 *timeslice, void * savstate)
{                                         /* new style calling convention */
  u32 kiter = rc5_unit_func_lintilla_compat( work, *timeslice );
  savstate = savstate; /* not used. shaddup compiler */
  
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
