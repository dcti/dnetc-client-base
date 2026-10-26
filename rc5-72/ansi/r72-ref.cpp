/* 
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *r72_ref_cpp(void) {
return "@(#)$Id: r72-ref.cpp,v 1.1 2002/10/20 22:13:52 andreasb Exp $"; }

#include "ccoreio.h"
#include "rotate.h"

#define P 0xB7E15163
#define Q 0x9E3779B9

#ifdef __cplusplus
extern "C" s32 rc5_72_unit_func_ansi_ref ( RC5_72UnitWork *, u32 *, void * );
#endif

s32 rc5_72_unit_func_ansi_ref (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  u32 i, j, k;
  u32 A, B;
  u32 S[26];
  u32 L[3];
  u32 kiter = *iterations;
  while (kiter--)
  {
    L[2] = rc5_72unitwork->L0.hi;
    L[1] = rc5_72unitwork->L0.mid;
    L[0] = rc5_72unitwork->L0.lo;
    for (S[0] = P, i = 1; i < 26; i++)
      S[i] = S[i-1] + Q;
      
    for (A = B = i = j = k = 0;
         k < 3*26; k++, i = (i + 1) % 26, j = (j + 1) % 3)
    {
      A = S[i] = ROTL3(S[i]+(A+B));
      B = L[j] = ROTL(L[j]+(A+B),(A+B));
    }
    A = rc5_72unitwork->plain.lo + S[0];
    B = rc5_72unitwork->plain.hi + S[1];
    for (i=1; i<=12; i++)
    {
      A = ROTL(A^B,B)+S[2*i];
      B = ROTL(B^A,A)+S[2*i+1];
    }
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
