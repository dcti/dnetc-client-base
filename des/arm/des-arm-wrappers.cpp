/* @(#)$Id: des-arm-wrappers.cpp,v 1.3.2.1 1999/12/07 23:56:29 cyp Exp $ */

#include "cputypes.h" /* u32 */
#include "problem.h" /* RC5UnitWork */

#ifdef __cplusplus
extern "C" {
#endif
  u32 des_unit_func_arm_asm( RC5UnitWork * , unsigned long );
  u32 des_unit_func_strongarm_asm( RC5UnitWork * , unsigned long );
#ifdef __cplusplus
}  
#endif

u32 des_unit_func_arm( RC5UnitWork *rc5unitwork, u32 *iterations, char *)
{
  unsigned long nbits = 8;
  while (*iterations > (1ul << nbits)) 
    nbits++;
  if (nbits > 24)
    nbits = 24;    
  *iterations = (1ul << nbits);
  return des_unit_func_arm_asm( rc5unitwork, nbits );
}  

u32 des_unit_func_strongarm( RC5UnitWork *rc5unitwork, u32 *iterations, char *)
{
  unsigned long nbits = 8;
  while (*iterations > (1ul << nbits)) 
    nbits++;
  if (nbits > 24)
    nbits = 24;    
  *iterations = (1ul << nbits);
  return des_unit_func_strongarm_asm( rc5unitwork, nbits );
}  
