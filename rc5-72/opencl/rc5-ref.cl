#define SHL(x, s) ((uint) ((x) << ((s) & 31)))
#define SHR(x, s) ((uint) ((x) >> (32 - ((s) & 31))))
#define ROTL(x, s) rotate((uint)x, (uint)s)
#define ROTL3(x) rotate((uint)x,(uint)3)
#define SWAP(x) ((x >> 24) | (x << 24) | ((x&0x00ff0000) >> 8) | ((x&0x0000ff00)<<8))

#define P 0xB7E15163
#define Q 0x9E3779B9

__kernel void ocl_rc572_ref( __constant uint *rc5_72unitwork, __global uint *outbuf)
{
  uint L[3];
  uint S[26];
  uint A, B;
  uint i,j,k;

  L[2] = rc5_72unitwork[0];   //L0hi;
  L[1] = rc5_72unitwork[1];   //L0mid;
  L[0] = rc5_72unitwork[2];   //L0lo;

  L[2] += get_global_id(0);
  uint l1_t1 = SWAP(L[1]);
  uint l1_t2 = l1_t1 + (L[2] >> 8);
  L[2] &= 0x000000ff;
  if(l1_t2 < l1_t1)
  {
    uint l0_t = SWAP(L[0]);
    l0_t +=1;
    L[0] = SWAP(l0_t);
  }
  L[1] = SWAP(l1_t2);


  for (S[0] = P, i = 1; i < 26; i++)
    S[i] = S[i-1] + Q;

  for (A = B = i = j = k = 0;
    k < 3*26; k++, i = (i + 1) % 26, j = (j + 1) % 3)
  {
    A = S[i] = ROTL3(S[i]+(A+B));
    B = L[j] = ROTL(L[j]+(A+B),(A+B));
  }
  A = rc5_72unitwork[4] + S[0];	//plain_lo
  B = rc5_72unitwork[5] + S[1]; //plain_hi
  for (i=1; i<=12; i++)
  {
    A = ROTL(A^B,B)+S[2*i];
    B = ROTL(B^A,A)+S[2*i+1];
  }
 
  if(A == rc5_72unitwork[6])
  {
    uint idx = atomic_add(&outbuf[0], 1) + 1;
    uint val = get_global_id(0);
    if(B == rc5_72unitwork[7])
      val |= 0x80000000; //Full match
    outbuf[idx] = val;
  }
}
