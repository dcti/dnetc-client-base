/*
 * Sunset's ScalarFusion RC5 ARM64 core. Messy but works.
 * Parts of this file (but not scalarfusion.S) based on ANSI 1-pipe. 
*/

#include "ccoreio.h"


#ifdef __cplusplus
extern "C" s32 CDECL rc5_72_unit_func_scalarfusion ( RC5_72UnitWork *, u32 *, void * );
#endif

extern "C" ui64 scalarFusionEntry(u32 A, u32 B, u32 lo, u32 mid, u32 hi);

s32 CDECL rc5_72_unit_func_scalarfusion (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  u32 A, B;
  ui64 scalarFusionPackedReturn;
  u32 kiter = *iterations;
  while (kiter--)
  {
    scalarFusionPackedReturn = scalarFusionEntry(rc5_72unitwork->plain.lo,
	rc5_72unitwork->plain.hi,
	rc5_72unitwork->L0.lo,
	rc5_72unitwork->L0.mid,
	rc5_72unitwork->L0.hi);
    A = scalarFusionPackedReturn & 0xFFFFFFFF;
    B = (scalarFusionPackedReturn >> 32) & 0xFFFFFFFF;

    if (A == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (B == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1);
        return RESULT_FOUND;
      }
    }

    #define key rc5_72unitwork->L0
    key.hi = (key.hi + 0x01) & 0x000000FF;
    if (!key.hi)
    {
      key.mid = key.mid + 0x01000000;
      if (!(key.mid & 0xFF000000u))
      {
        key.mid = (key.mid + 0x00010000) & 0x00FFFFFF;
        if (!(key.mid & 0x00FF0000))
        {
          key.mid = (key.mid + 0x00000100) & 0x0000FFFF;
          if (!(key.mid & 0x0000FF00))
          {
            key.mid = (key.mid + 0x00000001) & 0x000000FF;
            if (!key.mid)
            {
              key.lo = key.lo + 0x01000000;
              if (!(key.lo & 0xFF000000u))
              {
                key.lo = (key.lo + 0x00010000) & 0x00FFFFFF;
                if (!(key.lo & 0x00FF0000))
                {
                  key.lo = (key.lo + 0x00000100) & 0x0000FFFF;
                  if (!(key.lo & 0x0000FF00))
                  {
                    key.lo = (key.lo + 0x00000001) & 0x000000FF;
                  }
                }
              }
            }
          }
        }
      }
    }
    #undef key
  }
  return RESULT_NOTHING;
}
