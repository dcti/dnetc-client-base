#include "problem.h"
#define P 0xB7E15163
#define Q 0x9E3779B9
#define ROTL(x,y) (((x)<<(y&(0x1F))) | ((x)>>(32-(y&(0x1F)))))

#ifdef __cplusplus
extern "C" u32 rc5_72_unit_func_ansi_1 ( RC5_72UnitWork *, u32 );
#endif

u32 rc5_72_unit_func_ansi_1 (RC5_72UnitWork *rc5_72unitwork, u32 timeslice)
{
  u32 i, j, k;
  u32 A, B;
  u32 S[26], L[3];
  u32 kiter = 0;
  while (timeslice--)
  {
    L[2] = rc5_72unitwork->L0.hi;
    L[1] = rc5_72unitwork->L0.mid;
    L[0] = rc5_72unitwork->L0.lo;
    for (S[0] = P, i = 1; i < 26; i++)
      S[i] = S[i-1] + Q;
      
    for (A = B = i = j = k = 0;
         k < 3*26; k++, i = (i + 1) % 26, j = (j + 1) % 3)
    {
      A = S[i] = ROTL(S[i]+(A+B),3);
      B = L[j] = ROTL(L[j]+(A+B),(A+B));
    }
    A = rc5_72unitwork->plain.lo + S[0];
    B = rc5_72unitwork->plain.hi + S[1];
    for (i=1; i<=12; i++)
    {
      A = ROTL(A^B,B1)+S[2*i];
      B = ROTL(B^A,A1)+S[2*i+1];
    }
    if (A == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (B == rc5_72unitwork->cypher.hi)
        return kiter;
    }

    ++kiter;
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
  }
  return kiter;
}
