/*
 * This is the wrapper around rc5_unit_func
 * @(#)$Id: rc5_68k_gcc_crunch.c,v 1.3 1999/12/08 01:56:48 cyp Exp $
*/

#include "cputypes.h"
#include "problem.h"
#define PIPELINE_COUNT 1 /* this is how many the core is expecting */

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
      /* note: we switch the order */  
      register u32 tempkeylo = rc5unitwork->L0.hi; 
      register u32 tempkeyhi = rc5unitwork->L0.lo;
      rc5unitwork->L0.lo =
        ((tempkeylo >> 24) & 0x000000FFL) |                               
        ((tempkeylo >>  8) & 0x0000FF00L) |                               
        ((tempkeylo <<  8) & 0x00FF0000L) |                               
        ((tempkeylo << 24) & 0xFF000000L);                                
      rc5unitwork->L0.hi = 
        ((tempkeyhi >> 24) & 0x000000FFL) |                               
        ((tempkeyhi >>  8) & 0x0000FF00L) |                               
        ((tempkeyhi <<  8) & 0x00FF0000L) |                               
        ((tempkeyhi << 24) & 0xFF000000L);                                
      rc5unitwork->L0.lo += pipeline_count;
      if (rc5unitwork->L0.lo < ((u32)pipeline_count))
        rc5unitwork->L0.hi++;
      tempkeylo = rc5unitwork->L0.hi; 
      tempkeyhi = rc5unitwork->L0.lo;
      rc5unitwork->L0.lo =
        ((tempkeylo >> 24) & 0x000000FFL) |                               
        ((tempkeylo >>  8) & 0x0000FF00L) |                               
        ((tempkeylo <<  8) & 0x00FF0000L) |                               
        ((tempkeylo << 24) & 0xFF000000L);                                
      rc5unitwork->L0.hi = 
        ((tempkeyhi >> 24) & 0x000000FFL) |                               
        ((tempkeyhi >>  8) & 0x0000FF00L) |                               
        ((tempkeyhi <<  8) & 0x00FF0000L) |                               
        ((tempkeyhi << 24) & 0xFF000000L);                                
      kiter += pipeline_count;
    }
  }
  return kiter;
}  
