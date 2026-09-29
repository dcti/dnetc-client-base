/*
* Copyright distributed.net 2012-2026 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: 
*/
//CORENAME=ocl_rc572_4pipe_src
#if defined(__NVPTX__) && defined(NV_SM) // NVIDIA OPTIMIZED
  #if (NV_SM >= 32) // funnel shift supported
    #define ROTL(x, s) ({ \
        uint4 _x = (uint4)(x), _s = (uint4)(s), _res; \
        __asm__ ("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(_res.s0) : "r"(_x.s0), "r"(_s.s0)); \
        __asm__ ("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(_res.s1) : "r"(_x.s1), "r"(_s.s1)); \
        __asm__ ("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(_res.s2) : "r"(_x.s2), "r"(_s.s2)); \
        __asm__ ("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(_res.s3) : "r"(_x.s3), "r"(_s.s3)); \
        _res; \
    })
    #define ROTL3(x) ({ \
        uint4 _x = (uint4)(x), _res; \
        __asm__ ("shf.l.wrap.b32 %0, %1, %1, 3;" : "=r"(_res.s0) : "r"(_x.s0)); \
        __asm__ ("shf.l.wrap.b32 %0, %1, %1, 3;" : "=r"(_res.s1) : "r"(_x.s1)); \
        __asm__ ("shf.l.wrap.b32 %0, %1, %1, 3;" : "=r"(_res.s2) : "r"(_x.s2)); \
        __asm__ ("shf.l.wrap.b32 %0, %1, %1, 3;" : "=r"(_res.s3) : "r"(_x.s3)); \
        _res; \
    })
    #define ROTL1(x, s) ({ \
        uint _x = (uint)(x), _s = (uint)(s), _res; \
        __asm__ ("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(_res) : "r"(_x), "r"(_s)); \
        _res; \
    })
  #endif
  #if (NV_SM >= 20) // permute supported
    #define SWAP(x) ({ \
        uint _x = (uint)(x), _res; \
        __asm__ ("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(_res) : "r"(_x)); \
        _res; \
    })
  #endif
#elif defined(cl_amd_media_ops) && !defined(__clang__) // AMD LEGACY OPTIMIZED
  #pragma OPENCL EXTENSION cl_amd_media_ops : enable
  #define ROTL(x, s)  amd_bitalign((uint4)(x), (uint4)(x), (uint4)(32u) - (uint4)(s))
  #define ROTL3(x)    amd_bitalign((uint4)(x), (uint4)(x), (uint4)(29u))
  #define ROTL1(x, s) amd_bitalign((uint)(x), (uint)(x), 32u - (uint)(s))
  #define SWAP(x)     bitselect(amd_bytealign((uint)(x), (uint)(x), 3u), amd_bytealign((uint)(x), (uint)(x), 1u), 0xFF00FF00u)
#endif

#ifndef ROTL // STANDARD OPENCL
  #define ROTL(x, s) rotate((uint4)(x), (uint4)(s))
#endif
#ifndef ROTL3
  #define ROTL3(x) rotate((uint4)(x), (uint4)(3u))
#endif
#ifndef ROTL1
  #define ROTL1(x, s) rotate((uint)(x), (uint)(s))
#endif
#ifndef SWAP
  #define SWAP(x) (((uint)(x) << 24) | (((uint)(x) & 0x0000FF00u) << 8) | (((uint)(x) >> 8) & 0x0000FF00u) | ((uint)(x) >> 24))
#endif

#if (defined(NV_SM) && NV_SM >= 75 && NV_SM <= 89) || (defined(AMD_VGPR) && AMD_VGPR >= 128) // ENABLE ROUND1 OPTIMIZATION
  #define ROUND1_OPT 1
#endif

#define P 0xB7E15163u
#define Q 0x9E3779B9u

#define ROUND1(a, b, c, d) \
  S[a] = ROTL3(S[b] + (P + a * Q) + L[c]); \
  t = S[a] + L[c]; \
  L[d] = ROTL(L[d] + t, t)

#define ROUND23(a, b, c, d) \
  S[a] = ROTL3(S[a] + S[b] + L[c]); \
  t = S[a] + L[c]; \
  L[d] = ROTL(L[d] + t, t)

#define ENCRYPT(a) \
  A = ROTL(A^B, B) + S[a]; \
  B = ROTL(B^A, A) + S[a+1]

__kernel void ocl_rc572_4pipe( __constant const uint *rc5_72unitwork, volatile __global uint *outbuf)
{
  uint4 L[3];
  uint4 S[26];
  uint4 A, B;
  uint4 t;

  L[2].x = rc5_72unitwork[0];   //L0hi;
  L[1].x = rc5_72unitwork[1];   //L0mid;
  L[0] = (uint4)rc5_72unitwork[8];   
  S[1] = (uint4)rc5_72unitwork[9];

  L[2].x += (uint)get_global_id(0) * 4;
  uint l1_t1 = L[1].x;
  uint l1_t2 = l1_t1 + (L[2].x >> 8);
  L[2].x &= 0x000000ff;
  if(l1_t2 < l1_t1)
  {
    uint l0_t = SWAP(rc5_72unitwork[2]);
    l0_t += 1;
    L[0] = (uint4)ROTL1(0xBF0A8B1D + SWAP(l0_t), 0x1d);
    S[1] = (uint4)ROTL1(L[0].x + 0xBF0A8B1D + 0x5618cb1c, 3u);
  }
  L[1].x = SWAP(l1_t2);
  
  S[0] = (uint4)0xBF0A8B1D;
  t.x = S[1].x + L[0].x;
  L[1] = (uint4)ROTL1(L[1].x + t.x, t.x);
  L[2].y = L[2].x + 1;
  L[2].z = L[2].x + 2;
  L[2].w = L[2].x + 3;

  #if defined(ROUND1_OPT)
    S[2] = (uint4)ROTL1(S[1].x + (P + 2 * Q) + L[1].x, 3u);
    t = (uint4)(S[2].x + L[1].x);
    L[2] = ROTL(L[2] + t, t);
    S[3] = ROTL3((uint4)(S[2].x + (P + 3 * Q)) + L[2]);
    t = S[3] + L[2];
    L[0] = ROTL(L[0] + t, t);
  #else
    ROUND1( 2,  1, 1, 2);
    ROUND1( 3,  2, 2, 0);
  #endif

  ROUND1( 4,  3, 0, 1);
  ROUND1( 5,  4, 1, 2);
  ROUND1( 6,  5, 2, 0);
  ROUND1( 7,  6, 0, 1);
  ROUND1( 8,  7, 1, 2);
  ROUND1( 9,  8, 2, 0);
  ROUND1(10,  9, 0, 1);
  ROUND1(11, 10, 1, 2);
  ROUND1(12, 11, 2, 0);
  ROUND1(13, 12, 0, 1);
  ROUND1(14, 13, 1, 2);
  ROUND1(15, 14, 2, 0);
  ROUND1(16, 15, 0, 1);
  ROUND1(17, 16, 1, 2);
  ROUND1(18, 17, 2, 0);
  ROUND1(19, 18, 0, 1);
  ROUND1(20, 19, 1, 2);
  ROUND1(21, 20, 2, 0);
  ROUND1(22, 21, 0, 1);
  ROUND1(23, 22, 1, 2);
  ROUND1(24, 23, 2, 0);
  ROUND1(25, 24, 0, 1);

  ROUND23(0, 25, 1, 2);
  ROUND23(1,  0, 2, 0);
  ROUND23(2,  1, 0, 1);
  ROUND23(3,  2, 1, 2);
  ROUND23(4,  3, 2, 0);
  ROUND23(5,  4, 0, 1);
  ROUND23(6,  5, 1, 2);
  ROUND23(7,  6, 2, 0);
  ROUND23(8,  7, 0, 1);
  ROUND23(9,  8, 1, 2);
  ROUND23(10, 9, 2, 0);
  ROUND23(11, 10, 0, 1);
  ROUND23(12, 11, 1, 2);
  ROUND23(13, 12, 2, 0);
  ROUND23(14, 13, 0, 1);
  ROUND23(15, 14, 1, 2);
  ROUND23(16, 15, 2, 0);
  ROUND23(17, 16, 0, 1);
  ROUND23(18, 17, 1, 2);
  ROUND23(19, 18, 2, 0);
  ROUND23(20, 19, 0, 1);
  ROUND23(21, 20, 1, 2);
  ROUND23(22, 21, 2, 0);
  ROUND23(23, 22, 0, 1);
  ROUND23(24, 23, 1, 2);
  ROUND23(25, 24, 2, 0);

  ROUND23(0, 25, 0, 1);
  ROUND23(1,  0, 1, 2);
  ROUND23(2,  1, 2, 0);
  ROUND23(3,  2, 0, 1);
  ROUND23(4,  3, 1, 2);
  ROUND23(5,  4, 2, 0);
  ROUND23(6,  5, 0, 1);
  ROUND23(7,  6, 1, 2);
  ROUND23(8,  7, 2, 0);
  ROUND23(9,  8, 0, 1);
  ROUND23(10, 9, 1, 2);
  ROUND23(11, 10, 2, 0);
  ROUND23(12, 11, 0, 1);
  ROUND23(13, 12, 1, 2);
  ROUND23(14, 13, 2, 0);
  ROUND23(15, 14, 0, 1);
  ROUND23(16, 15, 1, 2);
  ROUND23(17, 16, 2, 0);
  ROUND23(18, 17, 0, 1);
  ROUND23(19, 18, 1, 2);
  ROUND23(20, 19, 2, 0);
  ROUND23(21, 20, 0, 1);
  ROUND23(22, 21, 1, 2);
  ROUND23(23, 22, 2, 0);

  S[24] = ROTL3(S[24] + S[23] + L[0]); 

  A = rc5_72unitwork[4] + S[0];	//plain_lo
  B = rc5_72unitwork[5] + S[1]; //plain_hi

  ENCRYPT(2);
  ENCRYPT(4);
  ENCRYPT(6);
  ENCRYPT(8);
  ENCRYPT(10);
  ENCRYPT(12);
  ENCRYPT(14);
  ENCRYPT(16);
  ENCRYPT(18);
  ENCRYPT(20);
  ENCRYPT(22);

  A = ROTL(A^B, B) + S[24]; 

  if((A.x == rc5_72unitwork[6]) || (A.y == rc5_72unitwork[6]) || (A.z == rc5_72unitwork[6]) || (A.w == rc5_72unitwork[6]))
  {
    uint idx, val, attrib;

    t = S[24] + L[0]; 
    L[1] = ROTL(L[1] + t, t);

    S[25] = ROTL3(S[25] + S[24] + L[1]);
    B = ROTL(B^A, A) + S[25];

    if(A.x == rc5_72unitwork[6])
    {
      idx = atomic_add(&outbuf[0], 1) * 2 + 1;
      val = get_global_id(0) * 4 + rc5_72unitwork[3]; //keyN+offset
      attrib = (B.x == rc5_72unitwork[7]) ? 0x80000000 : 0;
      outbuf[idx] = attrib;
      outbuf[idx+1] = val;
    }
    if(A.y == rc5_72unitwork[6])
    {
      idx = atomic_add(&outbuf[0], 1) * 2 + 1;
      val = get_global_id(0) * 4 + rc5_72unitwork[3] + 1; //keyN+offset
      attrib = (B.y == rc5_72unitwork[7]) ? 0x80000000 : 0;
      outbuf[idx] = attrib;
      outbuf[idx+1] = val;
    }
    if(A.z == rc5_72unitwork[6])
    {
      idx = atomic_add(&outbuf[0], 1) * 2 + 1;
      val = get_global_id(0) * 4 + rc5_72unitwork[3] + 2; //keyN+offset
      attrib = (B.z == rc5_72unitwork[7]) ? 0x80000000 : 0;
      outbuf[idx] = attrib;
      outbuf[idx+1] = val;
    }
    if(A.w == rc5_72unitwork[6])
    {
      idx = atomic_add(&outbuf[0], 1) * 2 + 1;
      val = get_global_id(0) * 4 + rc5_72unitwork[3] + 3; //keyN+offset
      attrib = (B.w == rc5_72unitwork[7]) ? 0x80000000 : 0;
      outbuf[idx] = attrib;
      outbuf[idx+1] = val;
    }
  }
}
