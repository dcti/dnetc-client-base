#include "problem.h"
#include "logstuff.h"

#ifndef NO_ALTIVEC
extern "C" int crunch_vec( RC5UnitWork *work, unsigned long iterations );

s32 rc5_unit_func_vec( RC5UnitWork *work, u32 *timeslice /* , void *scratch_area 
*/) {
  u32 kiter = 0; /* keys processed so far */
  do {
    kiter += crunch_vec(work, *timeslice-kiter);
    if (kiter < *timeslice)
    {
      if (crunch_allitnil(work,1) == 0) break;
      kiter++;
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
}
#endif /* NO_ALTIVEC */
