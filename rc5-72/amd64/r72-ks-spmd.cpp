#include "ccoreio.h"
#include <stdlib.h>
#ifdef __cplusplus
extern "C" s32 CDECL rc5_72_unit_func_ispc_16 ( RC5_72UnitWork *, u32 *, void * );
#endif

#if defined(_MSC_VER)
#pragma warning(disable:4307)   // integral constant overflow
#elif defined(__WATCOMC__)
#pragma warning 130 9           // arithmetic overflow in a constant expression
#endif

#include "spmd.h"

s32 CDECL rc5_72_unit_func_ispc_16 (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  u32 kiter = *iterations/16;
  uint32_t *bufferA = (uint32_t *)malloc(sizeof(uint32_t) * 16);
  uint32_t *bufferB = (uint32_t *)malloc(sizeof(uint32_t) * 16);
  while (kiter--)
  {
    ispc::rc5_16pipe(rc5_72unitwork->L0.hi, rc5_72unitwork->L0.mid, rc5_72unitwork->L0.lo, rc5_72unitwork->plain.hi, rc5_72unitwork->plain.lo, 0, bufferA, bufferB);


    if (bufferA[0] == rc5_72unitwork->cypher.lo)
	{
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[0] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16;
        return RESULT_FOUND;
      }
    }

    if (bufferA[1] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x01;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[1] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 1;
        return RESULT_FOUND;
      }
    }

    if (bufferA[2] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x02;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[2] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 2;
        return RESULT_FOUND;
      }
    }

    if (bufferA[3] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x03;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[3] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 3;
        return RESULT_FOUND;
      }
    }

    if (bufferA[4] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x04;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[4] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 4;
        return RESULT_FOUND;
      }
    }

    if (bufferA[5] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x05;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[5] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 5;
        return RESULT_FOUND;
      }
    }

    if (bufferA[6] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x06;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[6] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 6;
        return RESULT_FOUND;
      }
    }

    if (bufferA[7] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x07;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[7] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 7;
        return RESULT_FOUND;
      }
    }

    if (bufferA[8] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x08;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[8] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 8;
        return RESULT_FOUND;
      }
    }

    if (bufferA[9] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x09;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[9] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 9;
        return RESULT_FOUND;
      }
    }

    if (bufferA[10] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x0A;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[10] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 10;
        return RESULT_FOUND;
      }
    }

    if (bufferA[11] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x0B;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[11] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 11;
        return RESULT_FOUND;
      }
    }

    if (bufferA[12] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x0C;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[12] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 12;
        return RESULT_FOUND;
      }
    }

    if (bufferA[13] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x0D;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
      if (bufferB[13] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 13;
        return RESULT_FOUND;
      }
    }

    if (bufferA[14] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x0E;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo; 
      if (bufferB[14] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 14;
        return RESULT_FOUND;
      }
    }

    if (bufferA[15] == rc5_72unitwork->cypher.lo)
    {
      ++rc5_72unitwork->check.count;
      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x0F;
      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo; 
      if (bufferB[15] == rc5_72unitwork->cypher.hi)
      {
        *iterations -= (kiter + 1)*16 - 15;
        return RESULT_FOUND;
      }
    }

    #define key rc5_72unitwork->L0
    key.hi = (key.hi + 16) & 0x000000FF;
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


