/*
 * This is the wrapper around rc5_unit_func
 * @(#)$Id: rc5_68k_gcc_crunch.c,v 1.1.2.3 2000/05/06 22:14:55 mfeiri Exp $
*/

#include "cputypes.h"
#include "problem.h"
#define PIPELINE_COUNT 1 /* this is how many the core is expecting */
#define USE_ANSI_INCREMENT
#include "rotate.h"

#ifdef __cplusplus
extern "C" {
#endif
 u32 rc5_68k_crunch_unit_func( RC5UnitWork * , u32 );   /* this */
 u32 rc5_unit_func( RC5UnitWork * );                    /* that */
#ifdef __cplusplus
} 
#endif

u32 rc5_68k_crunch_unit_func ( RC5UnitWork * rc5unitwork, u32 iterations )
{                                
  u32 kiter = 0;
  int keycount = iterations;
  int pipeline_count = PIPELINE_COUNT;
  
  //LogScreenf ("rc5unitwork = %08X:%08X (%X)\n", rc5unitwork.L0.hi, rc5unitwork.L0.lo, keycount);
  while ( keycount-- ) // iterations ignores the number of pipelines
  {
    u32 result = rc5_unit_func( rc5unitwork );
    if ( result )
    {
      kiter += result-1;
      break;
    }
    else
    {
      ansi_increment(rc5unitwork);
      kiter += pipeline_count;
    }
  }
  return kiter;
}  
