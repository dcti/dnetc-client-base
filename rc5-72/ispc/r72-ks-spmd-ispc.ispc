typedef unsigned int32 uint32;

#define P 0xB7E15163
#define Q 0x9E3779B9

static void rc5pipe(uint32 iter, uint32 hi, uint32 mid, uint32 lo, uint32 plainhi, uint32 plainlo, uniform uint32 outputA[], uniform uint32 outputB[])
{
	#define KEY_INIT(i) S[i] = P + i * Q;
	#define SHL(x, s) ((uint32) ((x) << (s) ))
	#define SHR(x, s) ((uint32) ((x) >> (32 - (s)) ))
	#define ROTL(x, s) ((uint32) (SHL((x), (s & 0x1f)) | SHR((x), (s & 0x1f))))
	#define ROTL3(x) ROTL(x, 3)

	/* 2 is hi, 0 is lo */
	uint32 A;
	uint32 B;
	uint32 S[26];
	uint32 L[3];

	L[2] = hi + iter;
	L[1] = mid;
	L[0] = lo;

	S[0] = P;
	S[1] = P + Q;
	S[2] = P + 2 * Q;
	S[3] = P + 3 * Q;
	S[4] = P + 4 * Q;
	S[5] = P + 5 * Q;
	S[6] = P + 6 * Q;
	S[7] = P + 7 * Q;
	S[8] = P + 8 * Q;
	S[9] = P + 9 * Q;
	S[10] = P + 10 * Q;
	S[11] = P + 11 * Q;
	S[12] = P + 12 * Q;
	S[13] = P + 13 * Q;
	S[14] = P + 14 * Q;
	S[15] = P + 15 * Q;
	S[16] = P + 16 * Q;
	S[17] = P + 17 * Q;
	S[18] = P + 18 * Q;
	S[19] = P + 19 * Q;
	S[20] = P + 20 * Q;
	S[21] = P + 21 * Q;
	S[22] = P + 22 * Q;
	S[23] = P + 23 * Q;
	S[24] = P + 24 * Q;
	S[25] = P + 25 * Q;

	#define ROTL_BLOCK(i,j) ROTL_BLOCK_j##j (i)

	#define ROTL_BLOCK_i0_j1 \
    		S[0] = ROTL3(S[0]+(S[25]+L[0])); \
    		L[1] = ROTL(L[1]+(S[0]+L[0]),(S[0]+L[0])); \

	#define ROTL_BLOCK_i0_j2 \
    		S[0] = ROTL3(S[0]+(S[25]+L[1])); \
    		L[2] = ROTL(L[2]+(S[0]+L[1]),(S[0]+L[1])); \

	#define ROTL_BLOCK_j0(i) \
    		S[i] = ROTL3(S[i]+(S[i-1]+L[2])); \
    		L[0] = ROTL(L[0]+(S[i]+L[2]),(S[i]+L[2])); \

	#define ROTL_BLOCK_j1(i) \
    		S[i] = ROTL3(S[i]+(S[i-1]+L[0])); \
    		L[1] = ROTL(L[1]+(S[i]+L[0]),(S[i]+L[0])); \

	#define ROTL_BLOCK_j2(i) \
    		S[i] = ROTL3(S[i]+(S[i-1]+L[1])); \
    		L[2] = ROTL(L[2]+(S[i]+L[1]),(S[i]+L[1]));
	
	S[0] = ROTL3(S[0]);
    	
	L[0] = ROTL(L[0]+S[0],S[0]);

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

	A = S[0] + plainlo;
	B = S[1] + plainhi;

	#define FINAL_BLOCK(i) \
    		A = ROTL(A^B,B)+S[2*i]; B = ROTL(B^A,A)+S[2*i+1];

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

	outputA[iter] = A;
	outputB[iter] = B;
}

export void rc5_16pipe(uniform uint32 hi, uniform uint32 mid, uniform uint32 lo, uniform uint32 plainhi, uniform uint32 plainlo,
        uniform uint32 cipherlo, uniform uint32 outputA[], uniform uint32 outputB[])
{
        foreach (i = 0 ... 16)
        {
                rc5pipe(i, hi, mid, lo, plainhi, plainlo, outputA, outputB);
        }
}
