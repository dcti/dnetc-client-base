#include "problem.h"
#include "logstuff.h"

extern "C" int crunch_allitnil( RC5UnitWork *work, unsigned long iterations );

extern "C" {
s32 rc5_unit_func_allitnil( RC5UnitWork *work, u32 *timeslice /* , void *scratch_area 
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
} /* extern "C" */
