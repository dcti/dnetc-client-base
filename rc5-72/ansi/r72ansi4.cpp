/* 
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *rc5ansi4_cpp(void) {
return "@(#)$Id: r72ansi4.cpp,v 1.10 2002/10/15 20:47:27 acidblood Exp $"; }

#include "problem.h"
#define P 0xB7E15163
#define Q 0x9E3779B9
#define ROTL(x,y) (((x)<<(y&(0x1F))) | ((x)>>(32-(y&(0x1F)))))

#ifdef __cplusplus
extern "C" u32 rc5_72_unit_func_ansi_4 ( RC5_72UnitWork *, u32 );
#endif

u32 rc5_72_unit_func_ansi_4 (RC5_72UnitWork *rc5_72unitwork, u32 timeslice)
{
  u32 A1, A2, A3, A4, B1, B2, B3, B4;
  u32 S1[26], S2[26], S3[26], S4[26];
  u32 L1[3], L2[3], L3[3], L4[3];
  u32 kiter = 0;
  while (timeslice--)
  {
    L1[2] = rc5_72unitwork->L0.hi;
    L2[2] = 0x01 + L1[2];
    L3[2] = 0x01 + L2[2];
    L4[2] = 0x01 + L3[2];

    L1[1] = L2[1] = L3[1] = L4[1] = rc5_72unitwork->L0.mid;
    L1[0] = L2[0] = L3[0] = L4[0] = rc5_72unitwork->L0.lo;

#define KEY_INIT(i) S1[i] = S2[i] = S3[i] = S4[i] = P + i*Q;

	KEY_INIT(0);
	KEY_INIT(1);
	KEY_INIT(2);
	KEY_INIT(3);
	KEY_INIT(4);
	KEY_INIT(5);
	KEY_INIT(6);
	KEY_INIT(7);
	KEY_INIT(8);
	KEY_INIT(9);
	KEY_INIT(10);
	KEY_INIT(11);
	KEY_INIT(12);
	KEY_INIT(13);
	KEY_INIT(14);
	KEY_INIT(15);
	KEY_INIT(16);
	KEY_INIT(17);
	KEY_INIT(18);
	KEY_INIT(19);
	KEY_INIT(20);
	KEY_INIT(21);
	KEY_INIT(22);
	KEY_INIT(23);
	KEY_INIT(24);
	KEY_INIT(25);

#define ROTL_BLOCK(i,j) ROTL_BLOCK_j##j (i)

#define ROTL_BLOCK_i0_j1 \
    S1[0] = ROTL(S1[0]+(S1[25]+L1[0]),3); \
    S2[0] = ROTL(S2[0]+(S2[25]+L2[0]),3); \
    S3[0] = ROTL(S3[0]+(S3[25]+L3[0]),3); \
    S4[0] = ROTL(S4[0]+(S4[25]+L4[0]),3); \
    L1[1] = ROTL(L1[1]+(S1[0]+L1[0]),(S1[0]+L1[0])); \
    L2[1] = ROTL(L2[1]+(S2[0]+L2[0]),(S2[0]+L2[0])); \
    L3[1] = ROTL(L3[1]+(S3[0]+L3[0]),(S3[0]+L3[0])); \
    L4[1] = ROTL(L4[1]+(S4[0]+L4[0]),(S4[0]+L4[0])); 

#define ROTL_BLOCK_i0_j2 \
    S1[0] = ROTL(S1[0]+(S1[25]+L1[1]),3); \
    S2[0] = ROTL(S2[0]+(S2[25]+L2[1]),3); \
    S3[0] = ROTL(S3[0]+(S3[25]+L3[1]),3); \
    S4[0] = ROTL(S4[0]+(S4[25]+L4[1]),3); \
    L1[2] = ROTL(L1[2]+(S1[0]+L1[1]),(S1[0]+L1[1])); \
    L2[2] = ROTL(L2[2]+(S2[0]+L2[1]),(S2[0]+L2[1])); \
    L3[2] = ROTL(L3[2]+(S3[0]+L3[1]),(S3[0]+L3[1])); \
    L4[2] = ROTL(L4[2]+(S4[0]+L4[1]),(S4[0]+L4[1])); 

#define ROTL_BLOCK_j0(i) \
    S1[i] = ROTL(S1[i]+(S1[i-1]+L1[2]),3); \
    S2[i] = ROTL(S2[i]+(S2[i-1]+L2[2]),3); \
    S3[i] = ROTL(S3[i]+(S3[i-1]+L3[2]),3); \
    S4[i] = ROTL(S4[i]+(S4[i-1]+L4[2]),3); \
    L1[0] = ROTL(L1[0]+(S1[i]+L1[2]),(S1[i]+L1[2])); \
    L2[0] = ROTL(L2[0]+(S2[i]+L2[2]),(S2[i]+L2[2])); \
    L3[0] = ROTL(L3[0]+(S3[i]+L3[2]),(S3[i]+L3[2])); \
    L4[0] = ROTL(L4[0]+(S4[i]+L4[2]),(S4[i]+L4[2])); 

#define ROTL_BLOCK_j1(i) \
    S1[i] = ROTL(S1[i]+(S1[i-1]+L1[0]),3); \
    S2[i] = ROTL(S2[i]+(S2[i-1]+L2[0]),3); \
    S3[i] = ROTL(S3[i]+(S3[i-1]+L3[0]),3); \
    S4[i] = ROTL(S4[i]+(S4[i-1]+L4[0]),3); \
    L1[1] = ROTL(L1[1]+(S1[i]+L1[0]),(S1[i]+L1[0])); \
    L2[1] = ROTL(L2[1]+(S2[i]+L2[0]),(S2[i]+L2[0])); \
    L3[1] = ROTL(L3[1]+(S3[i]+L3[0]),(S3[i]+L3[0])); \
    L4[1] = ROTL(L4[1]+(S4[i]+L4[0]),(S4[i]+L4[0])); 

#define ROTL_BLOCK_j2(i) \
    S1[i] = ROTL(S1[i]+(S1[i-1]+L1[1]),3); \
    S2[i] = ROTL(S2[i]+(S2[i-1]+L2[1]),3); \
    S3[i] = ROTL(S3[i]+(S3[i-1]+L3[1]),3); \
    S4[i] = ROTL(S4[i]+(S4[i-1]+L4[1]),3); \
    L1[2] = ROTL(L1[2]+(S1[i]+L1[1]),(S1[i]+L1[1])); \
    L2[2] = ROTL(L2[2]+(S2[i]+L2[1]),(S2[i]+L2[1])); \
    L3[2] = ROTL(L3[2]+(S3[i]+L3[1]),(S3[i]+L3[1])); \
    L4[2] = ROTL(L4[2]+(S4[i]+L4[1]),(S4[i]+L4[1])); 

    S1[0] = ROTL(S1[0],3);
    S2[0] = ROTL(S2[0],3);
    S3[0] = ROTL(S3[0],3);
    S4[0] = ROTL(S4[0],3);
    L1[0] = ROTL(L1[0]+S1[0],S1[0]);
    L2[0] = ROTL(L2[0]+S2[0],S2[0]);
    L3[0] = ROTL(L3[0]+S3[0],S3[0]);
    L4[0] = ROTL(L4[0]+S4[0],S4[0]); 

    ROTL_BLOCK(1,1);
    ROTL_BLOCK(2,2);
    ROTL_BLOCK(3,0);
    ROTL_BLOCK(4,1);
    ROTL_BLOCK(5,2);
    ROTL_BLOCK(6,0);
    ROTL_BLOCK(7,1);
    ROTL_BLOCK(8,2);
    ROTL_BLOCK(9,0);
    ROTL_BLOCK(10,1);
    ROTL_BLOCK(11,2);
    ROTL_BLOCK(12,0);
    ROTL_BLOCK(13,1);
    ROTL_BLOCK(14,2);
    ROTL_BLOCK(15,0);
    ROTL_BLOCK(16,1);
    ROTL_BLOCK(17,2);
    ROTL_BLOCK(18,0);
    ROTL_BLOCK(19,1);
    ROTL_BLOCK(20,2);
    ROTL_BLOCK(21,0);
    ROTL_BLOCK(22,1);
    ROTL_BLOCK(23,2);
    ROTL_BLOCK(24,0);
    ROTL_BLOCK(25,1);

    ROTL_BLOCK_i0_j2;
    ROTL_BLOCK(1,0);
    ROTL_BLOCK(2,1);
    ROTL_BLOCK(3,2);
    ROTL_BLOCK(4,0);
    ROTL_BLOCK(5,1);
    ROTL_BLOCK(6,2);
    ROTL_BLOCK(7,0);
    ROTL_BLOCK(8,1);
    ROTL_BLOCK(9,2);
    ROTL_BLOCK(10,0);
    ROTL_BLOCK(11,1);
    ROTL_BLOCK(12,2);
    ROTL_BLOCK(13,0);
    ROTL_BLOCK(14,1);
    ROTL_BLOCK(15,2);
    ROTL_BLOCK(16,0);
    ROTL_BLOCK(17,1);
    ROTL_BLOCK(18,2);
    ROTL_BLOCK(19,0);
    ROTL_BLOCK(20,1);
    ROTL_BLOCK(21,2);
    ROTL_BLOCK(22,0);
    ROTL_BLOCK(23,1);
    ROTL_BLOCK(24,2);
    ROTL_BLOCK(25,0);

    ROTL_BLOCK_i0_j1;
    ROTL_BLOCK(1,2);
    ROTL_BLOCK(2,0);
    ROTL_BLOCK(3,1);
    ROTL_BLOCK(4,2);
    ROTL_BLOCK(5,0);
    ROTL_BLOCK(6,1);
    ROTL_BLOCK(7,2);
    ROTL_BLOCK(8,0);
    ROTL_BLOCK(9,1);
    ROTL_BLOCK(10,2);
    ROTL_BLOCK(11,0);
    ROTL_BLOCK(12,1);
    ROTL_BLOCK(13,2);
    ROTL_BLOCK(14,0);
    ROTL_BLOCK(15,1);
    ROTL_BLOCK(16,2);
    ROTL_BLOCK(17,0);
    ROTL_BLOCK(18,1);
    ROTL_BLOCK(19,2);
    ROTL_BLOCK(20,0);
    ROTL_BLOCK(21,1);
    ROTL_BLOCK(22,2);
    ROTL_BLOCK(23,0);
    ROTL_BLOCK(24,1);
    ROTL_BLOCK(25,2);

    A1 = rc5_72unitwork->plain.lo + S1[0];
    A2 = rc5_72unitwork->plain.lo + S2[0];
    A3 = rc5_72unitwork->plain.lo + S3[0];
    A4 = rc5_72unitwork->plain.lo + S4[0];
    B1 = rc5_72unitwork->plain.hi + S1[1];
    B2 = rc5_72unitwork->plain.hi + S2[1];
    B3 = rc5_72unitwork->plain.hi + S3[1];
    B4 = rc5_72unitwork->plain.hi + S4[1];

#define FINAL_BLOCK(i) \
    A1 = ROTL(A1^B1,B1)+S1[2*i]; \
    A2 = ROTL(A2^B2,B2)+S2[2*i]; \
    A3 = ROTL(A3^B3,B3)+S3[2*i]; \
    A4 = ROTL(A4^B4,B4)+S4[2*i]; \
    B1 = ROTL(B1^A1,A1)+S1[2*i+1]; \
    B2 = ROTL(B2^A2,A2)+S2[2*i+1]; \
    B3 = ROTL(B3^A3,A3)+S3[2*i+1]; \
    B4 = ROTL(B4^A4,A4)+S4[2*i+1];

    FINAL_BLOCK(1);
    FINAL_BLOCK(2);
    FINAL_BLOCK(3);
    FINAL_BLOCK(4);
    FINAL_BLOCK(5);
    FINAL_BLOCK(6);
    FINAL_BLOCK(7);
    FINAL_BLOCK(8);
    FINAL_BLOCK(9);
    FINAL_BLOCK(10);
    FINAL_BLOCK(11);
    FINAL_BLOCK(12);

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
