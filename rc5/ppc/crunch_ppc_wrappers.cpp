#include "problem.h"
#include "logstuff.h"

extern "C" int crunch_allitnil( RC5UnitWork *work, unsigned long iterations );
extern "C" int crunch_lintilla( RC5UnitWork *work, unsigned long iterations );

extern "C" {
s32 rc5_unit_func_g1( RC5UnitWork *work, u32 *timeslice /* , void *scratch_area 
*/) {
  u32 kiter = 0;

  kiter = crunch_allitnil( work, *timeslice );
  if (*timeslice == kiter) {
    *timeslice = kiter;
    return RESULT_WORKING;
  } else if (*timeslice < kiter) {
    *timeslice = kiter;
    return RESULT_FOUND;
  } else { /* shouldn't happen with this core */
    *timeslice = kiter; 
    return -1; /* error */
  }
}

s32 rc5_unit_func_g2_g3( RC5UnitWork *work, u32 *timeslice /* , void *scratch_area */) {
  u32 kiter = 0;

  do {
    kiter = crunch_lintilla( work, *timeslice - kiter );
    if (kiter < *timeslice) {
      u32 kiter_allitnil = crunch_allitnil( work, 256 );
      if (kiter_allitnil) 
        kiter += kiter_allitnil;
      else
        break;
    }
  } while (kiter < *timeslice);
  if (*timeslice == kiter) {
    return RESULT_WORKING;
  } else if (*timeslice < kiter) {
    *timeslice = kiter;
    return RESULT_FOUND;
  } else { /* shouldn't happen with this core */
    *timeslice = kiter; 
    return -1; /* error */
  }
}
};
