//
// $Log: gentestscsc.cpp,v $
// Revision 1.4  2000/06/02 06:32:56  jlawson
// sync, copy files from release branch to head
//
// Revision 1.1.2.1  1999/10/07 18:41:14  cyp
// sync'd from head
//
// Revision 1.1  1999/07/23 02:43:06  fordbr
// CSC cores added
//
//

char *id="@(#)$Id: gentestscsc.cpp,v 1.4 2000/06/02 06:32:56 jlawson Exp $";

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned char      u8;
typedef unsigned int       u32;
typedef unsigned long long u64;

#define CSC_C00 0xb7
#define CSC_C01 0xe1
#define CSC_C02 0x51
#define CSC_C03 0x62
#define CSC_C04 0x8a
#define CSC_C05 0xed
#define CSC_C06 0x2a
#define CSC_C07 0x6a
#define CSC_C10 0xbf
#define CSC_C11 0x71
#define CSC_C12 0x58
#define CSC_C13 0x80
#define CSC_C14 0x9c
#define CSC_C15 0xf4
#define CSC_C16 0xf3
#define CSC_C17 0xc7

u8 tbp[256] = {
  0x29,0x0d,0x61,0x40,0x9c,0xeb,0x9e,0x8f,
  0x1f,0x85,0x5f,0x58,0x5b,0x01,0x39,0x86,
  0x97,0x2e,0xd7,0xd6,0x35,0xae,0x17,0x16,
  0x21,0xb6,0x69,0x4e,0xa5,0x72,0x87,0x08,
  0x3c,0x18,0xe6,0xe7,0xfa,0xad,0xb8,0x89,
  0xb7,0x00,0xf7,0x6f,0x73,0x84,0x11,0x63,
  0x3f,0x96,0x7f,0x6e,0xbf,0x14,0x9d,0xac,
  0xa4,0x0e,0x7e,0xf6,0x20,0x4a,0x62,0x30,
  0x03,0xc5,0x4b,0x5a,0x46,0xa3,0x44,0x65,
  0x7d,0x4d,0x3d,0x42,0x79,0x49,0x1b,0x5c,
  0xf5,0x6c,0xb5,0x94,0x54,0xff,0x56,0x57,
  0x0b,0xf4,0x43,0x0c,0x4f,0x70,0x6d,0x0a,
  0xe4,0x02,0x3e,0x2f,0xa2,0x47,0xe0,0xc1,
  0xd5,0x1a,0x95,0xa7,0x51,0x5e,0x33,0x2b,
  0x5d,0xd4,0x1d,0x2c,0xee,0x75,0xec,0xdd,
  0x7c,0x4c,0xa6,0xb4,0x78,0x48,0x3a,0x32,
  0x98,0xaf,0xc0,0xe1,0x2d,0x09,0x0f,0x1e,
  0xb9,0x27,0x8a,0xe9,0xbd,0xe3,0x9f,0x07,
  0xb1,0xea,0x92,0x93,0x53,0x6a,0x31,0x10,
  0x80,0xf2,0xd8,0x9b,0x04,0x36,0x06,0x8e,
  0xbe,0xa9,0x64,0x45,0x38,0x1c,0x7a,0x6b,
  0xf3,0xa1,0xf0,0xcd,0x37,0x25,0x15,0x81,
  0xfb,0x90,0xe8,0xd9,0x7b,0x52,0x19,0x28,
  0x26,0x88,0xfc,0xd1,0xe2,0x8c,0xa0,0x34,
  0x82,0x67,0xda,0xcb,0xc7,0x41,0xe5,0xc4,
  0xc8,0xef,0xdb,0xc3,0xcc,0xab,0xce,0xed,
  0xd0,0xbb,0xd3,0xd2,0x71,0x68,0x13,0x12,
  0x9a,0xb3,0xc2,0xca,0xde,0x77,0xdc,0xdf,
  0x66,0x83,0xbc,0x8d,0x60,0xc6,0x22,0x23,
  0xb2,0x8b,0x91,0x05,0x76,0xcf,0x74,0xc9,
  0xaa,0xf1,0x99,0xa8,0x59,0x50,0x3b,0x2a,
  0xfe,0xf9,0x24,0xb0,0xba,0xfd,0xf8,0x55,
};

// ------------------------------------------------------------------
void cscipher_encrypt( u8 m[8], const u8 *k )
{
  u8 tmpx,tmprx,tmpy;
  int i;

#define APPLY_M(cl, cr, adl, adr) \
	tmpx = m[adl] ^ cl; \
	tmprx = (tmpx<<1) | (tmpx>>7); \
	tmpy = m[adr] ^ cr; \
	m[adl] = tbp[(tmprx & 0x55) ^ tmpx ^ tmpy]; \
	m[adr] = tbp[tmprx ^ tmpy]

  for( i=0; i<8; i++,k+=8 ) {
    APPLY_M( k[0], k[1], 0, 1);
    APPLY_M( k[2], k[3], 2, 3);
    APPLY_M( k[4], k[5], 4, 5);
    APPLY_M( k[6], k[7], 6, 7);
    APPLY_M( CSC_C00, CSC_C01, 0, 2);
    APPLY_M( CSC_C02, CSC_C03, 4, 6);
    APPLY_M( CSC_C04, CSC_C05, 1, 3);
    APPLY_M( CSC_C06, CSC_C07, 5, 7);
    APPLY_M( CSC_C10, CSC_C11, 0, 4);
    APPLY_M( CSC_C12, CSC_C13, 1, 5);
    APPLY_M( CSC_C14, CSC_C15, 2, 6);
    APPLY_M( CSC_C16, CSC_C17, 3, 7);
  }
  for( i=0; i<8; i++ )
    m[i] ^= k[i];
#undef APPLY_M
}

// ------------------------------------------------------------------
void cscipher_decrypt( u8 m[8], const u8 *k )
{
  u8 pyr, pxorp, xl;

#define APPLY_M1(cl, cr, adl, adr)				\
	pyr = tbp[m[adr]];					\
	pxorp = tbp[m[adl]] ^ pyr;				\
	xl = (((pxorp << 1) | (pxorp >> 7)) & 0xAA) ^ pxorp;	\
	m[adr] = ((xl << 1) | (xl >> 7)) ^ pyr ^ cr;		\
	m[adl] = xl ^ cl

  k += 64;
  for( int i=0; i<8; i++ )
    m[i] ^= k[i];

  for( int i=0; i<8; i++ ) {
    k -= 8;
    APPLY_M1( CSC_C10, CSC_C11, 0, 4);
    APPLY_M1( CSC_C12, CSC_C13, 1, 5);
    APPLY_M1( CSC_C14, CSC_C15, 2, 6);
    APPLY_M1( CSC_C16, CSC_C17, 3, 7);
    APPLY_M1( CSC_C00, CSC_C01, 0, 2);
    APPLY_M1( CSC_C02, CSC_C03, 4, 6);
    APPLY_M1( CSC_C04, CSC_C05, 1, 3);
    APPLY_M1( CSC_C06, CSC_C07, 5, 7);
    APPLY_M1( k[0], k[1], 0, 1);
    APPLY_M1( k[2], k[3], 2, 3);
    APPLY_M1( k[4], k[5], 4, 5);
    APPLY_M1( k[6], k[7], 6, 7); 
  }
#undef APPLY_M1
}

// ------------------------------------------------------------------
void cscipher_keyschedule( u8 subkey[9+2][8] )
{
  for( int sk=2; sk<9+2; sk++ ) {
    //  8  i-1      i
    // P (k    xor c )
    u8 p8kxorc[8];
    *(u64*)&p8kxorc = *(u64*)&subkey[sk-1] ^ *(u64*)&tbp[(sk-2)*8];
    //for( int i=7; i>=0; i-- )
    //  p8kxorc[i] = tbp[p8kxorc[i]];
    p8kxorc[0] = tbp[p8kxorc[0]];
    p8kxorc[1] = tbp[p8kxorc[1]];
    p8kxorc[2] = tbp[p8kxorc[2]];
    p8kxorc[3] = tbp[p8kxorc[3]];
    p8kxorc[4] = tbp[p8kxorc[4]];
    p8kxorc[5] = tbp[p8kxorc[5]];
    p8kxorc[6] = tbp[p8kxorc[6]];
    p8kxorc[7] = tbp[p8kxorc[7]];
    //    8  i-1      i
    // T(P (k    xor c ))
    u8 tp8[8], mask = 0x01;
    for( int i=7; i>=0; i--,mask<<=1 ) {
      u8 t8 = 0;
      if( p8kxorc[7] & mask ) t8 |= 0x01;
      if( p8kxorc[6] & mask ) t8 |= 0x02;
      if( p8kxorc[5] & mask ) t8 |= 0x04;
      if( p8kxorc[4] & mask ) t8 |= 0x08;
      if( p8kxorc[3] & mask ) t8 |= 0x10;
      if( p8kxorc[2] & mask ) t8 |= 0x20;
      if( p8kxorc[1] & mask ) t8 |= 0x40;
      if( p8kxorc[0] & mask ) t8 |= 0x80;
      tp8[i] = t8;
    }
    //  i-2        8  i-1      i
    // k    xor T(P (k    xor c ))
    *(u64*)&subkey[sk] = *(u64*)&subkey[sk-2] ^ *(u64*)&tp8;
  }
}

// ------------------------------------------------------------------
void cscipher_cbc_encrypt( u8 *message, int nblocks, const u8 key[2][8], const u8 iv[8] )
{
  u8 subkey[9+2][8];
  u8 feedback[8];

  // first, schedule the key
  memcpy( subkey[0], key[1], sizeof(subkey[0]) );
  memcpy( subkey[1], key[0], sizeof(subkey[1]) );
  cscipher_keyschedule( subkey );
/*
  // print the generated subkeys
  for( int i=0; i<9+2; i++ ) {
    printf( "subkey %2d = 0x", i-2 );
    for( int j=0; j<8; j++ )
      printf( "%02x", (int) subkey[i][j] );
    printf( "\n" );
  }
  printf( "\n" );
*/
  // then, copy iv to the feedback register
  memcpy( feedback, iv, sizeof(feedback) );
  
  // and now, encrypt the whole thing
  for(; nblocks; nblocks--, message += 8 ) {
    for( int i=0; i<8; i++ ) message[i] ^= feedback[i];
    cscipher_encrypt( message, &subkey[2][0] );
    memcpy( feedback, message, sizeof(feedback) );
  }
  
}

// ------------------------------------------------------------------
void cscipher_cbc_decrypt( u8 *message, int nblocks, const u8 key[2][8], const u8 iv[8] )
{
  u8 subkey[9+2][8];
  u8 feedback1[8], *fb1 = feedback1;
  u8 feedback2[8], *fb2 = feedback2;

  // first, schedule the key
  memcpy( subkey[0], key[1], sizeof(subkey[0]) );
  memcpy( subkey[1], key[0], sizeof(subkey[1]) );
  cscipher_keyschedule( subkey );

  // then, copy iv to the feedback register
  memcpy( feedback1, iv, sizeof(feedback1) );
  
  // and now, decrypt the whole thing
  for(; nblocks; nblocks--, message += 8 ) {
    memcpy( fb2, message, 8 );
    cscipher_decrypt( message, &subkey[2][0] );
    for( int i=0; i<8; i++ ) message[i] ^= fb1[i];
    u8 *t = fb1; fb1 = fb2; fb2 = t;
  }
}

#define TEST_CASE_COUNT 32

// CSC test cases -- key/iv/plain/cypher
// note this is in .lo, .hi, .lo... order...
u32 csc_test_cases[TEST_CASE_COUNT][8];

// ------------------------------------------------------------------
// ------------------------------------------------------------------
int main (void) 
{
  /*static const u8 key[2][8] = { 
    {0x5f, 0x64, 0x3e, 0xc4, 0x78, 0x5d, 0x91, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}    
  };
  static const u8 iv[8] = { 0x13, 0x24, 0xfe, 0x68, 0x57, 0xbd, 0x90, 0xac };
  static const u8 plain[8] = { 0x54, 0x68, 0x65, 0x20, 0x73, 0x65, 0x63, 0x72 };
  static const u8 ciphertext[8] = { 0x35, 0x32, 0x8d, 0x4e, 0x5c, 0xc4, 0x2f, 0xb0 };*/

  u8 key[2][8], iv[8], plain[8], cypher[8], plain2[8];
  int i,j;
  unsigned int seed;
  
  seed = time( NULL );
  printf( "static const u32 csc_test_cases[TEST_CASE_COUNT][8] = { // seed = 0x%08x\n", seed );
  srandom( seed );
  key[0][7] = 0;
  memset( key[1], 0, sizeof(key[1]) );

  for( i=0; i<TEST_CASE_COUNT; i++ ) {

    // first, set up some random keys,plain,iv
    for (j=0; j<7; j++) key[0][j] = random() & 0xFF;
    csc_test_cases[i][0] = (key[0][7]<< 0) | (key[0][6]<< 8) |
                           (key[0][5]<<16) | (key[0][4]<<24); // low
    csc_test_cases[i][1] = (key[0][3]<< 0) | (key[0][2]<< 8) |
                           (key[0][1]<<16) | (key[0][0]<<24); // high
    for (j=0; j<8; j++) {
      plain[j] = random() & 0xFF;
      iv[j] = random() & 0xFF;
    }
    csc_test_cases[i][2] = iv[7] | (iv[6]<<8) | (iv[5]<<16) | (iv[4]<<24); // low
    csc_test_cases[i][3] = iv[3] | (iv[2]<<8) | (iv[1]<<16) | (iv[0]<<24); // high
    csc_test_cases[i][4] = plain[7] | (plain[6]<<8) | (plain[5]<<16) | (plain[4]<<24);
    csc_test_cases[i][5] = plain[3] | (plain[2]<<8) | (plain[1]<<16) | (plain[0]<<24);

    // now encrypt all this and store cyphertext
    memcpy( cypher, plain, sizeof(cypher) );
    cscipher_cbc_encrypt( cypher, 1, key, iv);
    csc_test_cases[i][6] = cypher[7] | (cypher[6]<<8) | (cypher[5]<<16) | (cypher[4]<<24);
    csc_test_cases[i][7] = cypher[3] | (cypher[2]<<8) | (cypher[1]<<16) | (cypher[0]<<24);

    // try a decrypt, just in case
    memcpy( plain2, cypher, sizeof(plain2) );
    cscipher_cbc_decrypt( plain2, 1, key, iv);
    if (memcmp(plain, plain2, sizeof(plain2) ) != 0) 
      printf( "cbc_encrypt decrypt error\n" );

    // write everything in C++ syntax
    printf( "  {0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x},\n",
	    csc_test_cases[i][0],csc_test_cases[i][1],
	    csc_test_cases[i][2],csc_test_cases[i][3],
	    csc_test_cases[i][4],csc_test_cases[i][5],
	    csc_test_cases[i][6],csc_test_cases[i][7] );
  }
  printf( "};\n" );
  return 0;
}
