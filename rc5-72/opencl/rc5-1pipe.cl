/*
* Copyright distributed.net 2012 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: 
*/
//CORENAME=ocl_rc572_1pipe_src
#ifdef  cl_amd_media_ops
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#define ROTL(x, s) amd_bitalign(x, x, (uint)(32 - (s)))
#define SWAP(x) ((amd_bytealign(x, x, 1) & 0xff00ff00) | (amd_bytealign(x, x, 3) & 0x00ff00ff))
#else
#define ROTL(x, s) rotate((uint)x, (uint)s)
#define SWAP(x) ((x >> 24) | (x << 24) | ((x&0x00ff0000) >> 8) | ((x&0x0000ff00)<<8))
#endif //cl_amd_media_ops

#define ROTL3(x) ROTL(x,3)

#define P 0xB7E15163
#define Q 0x9E3779B9

#define ROUND1(a, b, c, d) \
  S[a] = ROTL3(S[b] + P + a*Q + L[c]); \
  t = S[a] + L[c]; \
  L[d] = ROTL(L[d] + t, t)

#define ROUND23(a, b, c, d) \
  S[a] = ROTL3(S[a] + S[b] +L[c]); \
  t = S[a] + L[c]; \
  L[d] = ROTL(L[d] + t, t)


#define ENCRYPT(a) \
    A = ROTL(A^B,B)+S[a]; \
    B = ROTL(B^A,A)+S[a+1]

__kernel void ocl_rc572_1pipe( __constant uint *rc5_72unitwork, __global uint *outbuf)
{
  uint L[3];
  uint S[26];
  uint A,B;
  uint t;

  L[2] = rc5_72unitwork[0];   //L0hi;
  L[1] = rc5_72unitwork[1];   //L0mid;
  L[0] = rc5_72unitwork[8];   
  S[1] = rc5_72unitwork[9];

  L[2] += get_global_id(0);
  uint l1_t1 = L[1];
  uint l1_t2 = l1_t1 + (L[2] >> 8);
  L[2] &= 0x000000ff;
  if(l1_t2 < l1_t1)
  {
    uint l0_t = SWAP(rc5_72unitwork[2]);
    l0_t +=1;
    L[0] = ROTL(0xBF0A8B1D + SWAP(l0_t), (uint)0x1d);
    S[1] = ROTL(L[0] + 0xBF0A8B1D + 0x5618cb1c, (uint)3);
  }
  L[1] = SWAP(l1_t2);
  
  S[0] = 0xBF0A8B1D;
  t = S[1] + L[0];
  L[1] = ROTL(L[1] + t, t);

  ROUND1( 2,  1, 1, 2);
  ROUND1( 3,  2, 2, 0);
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
  ROUND23(11,10, 0, 1);
  ROUND23(12,11, 1, 2);
  ROUND23(13,12, 2, 0);
  ROUND23(14,13, 0, 1);
  ROUND23(15,14, 1, 2);
  ROUND23(16,15, 2, 0);
  ROUND23(17,16, 0, 1);
  ROUND23(18,17, 1, 2);
  ROUND23(19,18, 2, 0);
  ROUND23(20,19, 0, 1);
  ROUND23(21,20, 1, 2);
  ROUND23(22,21, 2, 0);
  ROUND23(23,22, 0, 1);
  ROUND23(24,23, 1, 2);
  ROUND23(25,24, 2, 0);

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
  ROUND23(11,10, 2, 0);
  ROUND23(12,11, 0, 1);
  ROUND23(13,12, 1, 2);
  ROUND23(14,13, 2, 0);
  ROUND23(15,14, 0, 1);
  ROUND23(16,15, 1, 2);
  ROUND23(17,16, 2, 0);
  ROUND23(18,17, 0, 1);
  ROUND23(19,18, 1, 2);
  ROUND23(20,19, 2, 0);
  ROUND23(21,20, 0, 1);
  ROUND23(22,21, 1, 2);
  ROUND23(23,22, 2, 0);

  S[24] = ROTL3(S[24] + S[23] +L[0]); 

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

  A = ROTL(A^B,B)+S[24]; 

  if(A == rc5_72unitwork[6])
  {
    t = S[24] + L[0]; 
    L[1] = ROTL(L[1] + t, t);

    S[25] = ROTL3(S[25] + S[24] +L[1]);
    B = ROTL(B^A,A)+S[25];

    uint idx = atomic_add(&outbuf[0], 1)*2+1;
    uint val = get_global_id(0) + rc5_72unitwork[3]; //keyN+offset
    uint attrib = (B == rc5_72unitwork[7])?0x80000000:0;
    outbuf[idx] = attrib;
    outbuf[idx+1] = val;
  }
}
