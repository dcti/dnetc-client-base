/* 
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *rc5ansi4_cpp(void) {
return "@(#)$Id: r72ansi4.cpp,v 1.7 2002/09/25 21:24:18 acidblood Exp $"; }

#include "problem.h"
#define P 0xB7E15163
#define Q 0x9E3779B9
#define ROTL(x,y) (((x)<<(y&(0x1F))) | ((x)>>(32-(y&(0x1F)))))

#ifdef __cplusplus
extern "C" u32 rc5_72_unit_func_ansi_4 ( RC5_72UnitWork *, u32 );
#endif

u32 rc5_72_unit_func_ansi_4 (RC5_72UnitWork *rc5_72unitwork, u32 timeslice)
{
  u32 i, j, k;
  u32 A1, A2, A3, A4, B1, B2, B3, B4;
  u32 S1[26], S2[26], S3[26], S4[26], L1[3], L2[3], L3[3], L4[3];
  u32 kiter = 0;
  while (timeslice--)
  {
    L1[2] = rc5_72unitwork->L0.hi;
    L2[2] = 0x01 + L1[2];
    L3[2] = 0x01 + L2[2];
    L4[2] = 0x01 + L3[2];

    L1[1] = L2[1] = L3[1] = L4[1] = rc5_72unitwork->L0.mid;
    L1[0] = L2[0] = L3[0] = L4[0] = rc5_72unitwork->L0.lo;

    for (S1[0] = S2[0] = S3[0] = S4[0] = P, i = 1; i < 26; i++)
      S1[i] = S2[i] = S3[i] = S4[i] = S1[i-1] + Q;

    for (A1 = A2 = A3 = A4 = B1 = B2 = B3 = B4 = i = j = k = 0;
         k < 3*26; k++, i = (i + 1) % 26, j = (j + 1) % 3)
    {
      A1 = S1[i] = ROTL(S1[i]+(A1+B1),3);
      A2 = S2[i] = ROTL(S2[i]+(A2+B2),3);
      A3 = S3[i] = ROTL(S3[i]+(A3+B3),3);
      A4 = S4[i] = ROTL(S4[i]+(A4+B4),3);
      B1 = L1[j] = ROTL(L1[j]+(A1+B1),(A1+B1));
      B2 = L2[j] = ROTL(L2[j]+(A2+B2),(A2+B2));
      B3 = L3[j] = ROTL(L3[j]+(A3+B3),(A3+B3));
      B4 = L4[j] = ROTL(L4[j]+(A4+B4),(A4+B4));
    }
    A1 = rc5_72unitwork->plain.lo + S1[0];
    A2 = rc5_72unitwork->plain.lo + S2[0];
    A3 = rc5_72unitwork->plain.lo + S3[0];
    A4 = rc5_72unitwork->plain.lo + S4[0];
    B1 = rc5_72unitwork->plain.hi + S1[1];
    B2 = rc5_72unitwork->plain.hi + S2[1];
    B3 = rc5_72unitwork->plain.hi + S3[1];
    B4 = rc5_72unitwork->plain.hi + S4[1];
    for (i=1; i<=12; i++)
    {
      A1 = ROTL(A1^B1,B1)+S1[2*i];
      A2 = ROTL(A2^B2,B2)+S2[2*i];
      A3 = ROTL(A3^B3,B3)+S3[2*i];
      A4 = ROTL(A4^B4,B4)+S4[2*i];
      B1 = ROTL(B1^A1,A1)+S1[2*i+1];
      B2 = ROTL(B2^A2,A2)+S2[2*i+1];
      B3 = ROTL(B3^A3,A3)+S3[2*i+1];
      B4 = ROTL(B4^A4,A4)+S4[2*i+1];
    }
    if (A1 == rc5_72unitwork->cypher.lo)
	{
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (B1 == rc5_72unitwork->cypher.hi)
        return kiter;
    }

    if (A2 == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x01;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (B2 == rc5_72unitwork->cypher.hi)
        return kiter + 1;
    }

    if (A3 == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x02;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (B3 == rc5_72unitwork->cypher.hi)
        return kiter + 2;
    }

    if (A4 == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi = rc5_72unitwork->L0.hi + 0x03;
      rc5_72unitwork->check.lo = rc5_72unitwork->L0.lo;
      rc5_72unitwork->check.hi = rc5_72unitwork->L0.hi;
      if (B4 == rc5_72unitwork->cypher.hi)
        return kiter + 3;
    }

    kiter += 4;
    #define key rc5_72unitwork->L0
    key.hi = (key.hi + 0x04) & 0x000000FF;
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
