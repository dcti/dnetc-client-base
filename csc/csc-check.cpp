#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

  // print the generated subkeys
  for( int i=0; i<9+2; i++ ) {
    printf( "subkey %2d = 0x", i-2 );
    for( int j=0; j<8; j++ )
      printf( "%02x", (int) subkey[i][j] );
    printf( "\n" );
  }
  printf( "\n" );

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

/*
 * cs-cipher is a 64-bit block cipher, with variable key length (0..128)
 * Internally, the key is viewed as two 64-bit numbers, independantly from
 * the real key-length. For example, a 56-bit CSC key might be :
 *   0x21, 0x53, 0xad, 0xf9, 0x46, 0x7c, 0xb9, 0x00,
 *   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
 *
 * Since we're on a 56-bit contest, we will only convert the first 64-bit number
 *
 * Here's how I'm counting bits in all CSC stuff :
 * 0x21 == bits 63..56, 0x21 & 0x80 == bit 63
 * 0x53 == bits 55..48, 0x53 & 0x01 == bit 48
 * etc...
 *
 * NOTE: Those routines are probably slower than their DES conterparts
 * but I think they will be far easier to maintain if we want to change
 * the bit order.
 */

const int csc_bit_order[64] = {
  // bits internal to the bitslicer and/or the driver
  22,30,38,46,54,62, 8,9,10,11,12,13,
  // other bits
                    14,15,
  16,17,18,19,20,21,   23,
  24,25,26,27,28,29,   31,
  32,33,34,35,36,37,   39,
  40,41,42,43,44,45,   47,
  48,49,50,51,52,53,   55,
  56,57,58,59,60,61,   63,
  // unused bits (in case of a 56-bit key)
   0, 1, 2, 3, 4, 5, 6, 7,
};

// ------------------------------------------------------------------
// Convert a key from CSC format to incrementable format
//
void convert_key_from_csc_to_inc (u32 *cschi, u32 *csclo)
{
  u32 tmphi, tmplo;
  
  tmphi = tmplo = 0;
  for( int i=0; i<64; i++ ) {
    int src = csc_bit_order[i];
    u32 bitval;
    if( src <= 31 )
      bitval = (*csclo >> src) & 1;
    else
      bitval = (*cschi >> (src-32)) & 1;
    if( i <= 31 )
      tmplo |= bitval << i;
    else
      tmphi |= bitval << (i-32);
  }
  *cschi = tmphi;
  *csclo = tmplo;
}

// ------------------------------------------------------------------
// Convert a key from incrementable format to CSC format
//
void convert_key_from_inc_to_csc (u32 *cschi, u32 *csclo)
{
  u32 tmphi, tmplo;
  
  tmphi = tmplo = 0;
  for( int i=0; i<64; i++ ) {
    u32 bitval;
    int src;
    for( src=0; src<64 && csc_bit_order[src] != i; src++ );
    if( src <= 31 )
      bitval = (*csclo >> src) & 1;
    else
      bitval = (*cschi >> (src-32)) & 1;
    if( i <= 31 )
      tmplo |= bitval << i;
    else
      tmphi |= bitval << (i-32);
  }
  *cschi = tmphi;
  *csclo = tmplo;
}

// ------------------------------------------------------------------
int main( int argc, char **argv ) 
{
#if 0
  // CS-CIPHER test 1
  // ----------------
  // key = 003E91F447076477 (incrementable format)
  // key = 0x5f643ec4785d91 (CSC format)
  static u8 key[2][8];
  static const u8 iv[8] = { 0x13, 0x24, 0xfe, 0x68, 0x57, 0xbd, 0x90, 0xac };
  #define KNOWN_PLAINTEXT_SIZE 23
  static const u8 plaintext[] = {
    0x54,0x68,0x65,0x20,0x73,0x65,0x63,0x72,0x65,0x74,0x20,0x6d,0x65,0x73,0x73,0x61,
    0x67,0x65,0x20,0x69,0x73,0x3a,0x20,0x74,0x68,0x65,0x20,0x77,0x69,0x6e,0x6e,0x65,
    0x72,0x20,0x69,0x73,0x20,0x2e,0x2e,0x2e,0x20,0x43,0x53,0x2d,0x43,0x49,0x50,0x48,
    0x45,0x52,0x21,0x20,0x20,0x20,0x20,0x20
  };
  static const u8 ciphertext[] = {
    0x35,0x32,0x8d,0x4e,0x5c,0xc4,0x2f,0xb0,0x5f,0x96,0x63,0x32,0xa8,0xc9,0x72,0xb7,
    0x3b,0xca,0x17,0xd9,0x22,0x4c,0x4e,0x93,0x05,0x17,0x65,0x20,0xbc,0x5b,0x91,0x1b,
    0x36,0xac,0xb9,0xd7,0x1a,0xb7,0xdc,0xa7,0xe2,0x09,0x8a,0xc4,0xab,0x8e,0xe4,0xe6,
    0x0c,0x0b,0xc8,0x9b,0xe7,0x56,0x3d,0x30
  };
#endif
  
#if 0
  // CS-CIPHER test 2
  // ----------------
  // key = 00424F6F90CF2E57 (incrementable format)
  // key = 0x2153adf9467cb9 (CSC format)
  static u8 key[2][8];
  static const u8 iv[8] = { 0xf3, 0x67, 0xb8, 0x35, 0x99, 0x22, 0xa6, 0x44 };
  #define KNOWN_PLAINTEXT_SIZE 23
  static const u8 plaintext[] = {
    0x54,0x68,0x65,0x20,0x73,0x65,0x63,0x72,0x65,0x74,0x20,0x6d,0x65,0x73,0x73,0x61,
    0x67,0x65,0x20,0x69,0x73,0x3a,0x20,0x74,0x68,0x65,0x20,0x6e,0x65,0x74,0x69,0x7a,
    0x65,0x6e,0x73,0x20,0x61,0x72,0x65,0x20,0x74,0x68,0x65,0x20,0x62,0x65,0x73,0x74,
    0x21,0x20,0x20,0x20,0x20,0x20,0x20,0x20
  };
  static const u8 ciphertext[] = {
    0x7C,0x9b,0xb5,0xe5,0x9c,0xa6,0xa3,0xd7,0x9d,0x59,0xf0,0x96,0x68,0x49,0x5d,0xab,
    0x21,0x27,0xa7,0xf4,0xe5,0xbb,0xa2,0x12,0x8c,0xd0,0x94,0xf7,0x2c,0xa7,0x7f,0x1c,
    0xa3,0xb5,0x15,0x76,0x19,0xf7,0x92,0xcc,0x3d,0x81,0x5b,0x95,0x3e,0x1a,0xb8,0x49,
    0x97,0xa2,0x6f,0x21,0x31,0x14,0x3f,0x88
  };
#endif

  //#if 0
  // REAL CS-CIPHER CHALLENGE
  // ------------------------
  static u8 key[2][8];
  static const u8 iv[8] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
  static const u8 ciphertext[] = {
    0x80,0xf3,0xf6,0x43,0xb8,0x12,0xc0,0x5d, 0xbd,0x52,0x3e,0x14,0xeb,0x26,0x5f,0xc1,
    0xa6,0xfa,0xd7,0x15,0x1d,0x58,0xdd,0xb2, 0x70,0xb0,0x8f,0x2d,0xbd,0x67,0xd5,0x8a,
    0xe0,0x84,0x1f,0x96,0x5d,0x95,0xae,0x54, 0x83,0x36,0x78,0xf0,0xc8,0x34,0xe5,0x10,
    0x3f,0xba,0x73,0xac,0xfe,0xa7,0x51,0x49, 0x01,0x63,0x56,0xfd,0xb5,0xc3,0xdc,0x95,
    0x9f,0x73,0xcc,0x2c,0xe1,0x69,0x6e,0x97, 0x77,0xba,0x60,0x85,0x63,0x38,0x39,0x21,
    0x2b,0x80,0x49,0x87,0x2f,0x79,0x0d,0xf2, 0x2f,0xa6,0xae,0xf8,0x87,0x30,0xf0,0xe4,
    0x74,0x5c,0x1c,0xdc,0xd9,0xb4,0xda,0x51, 0xaa,0x42,0xab,0x66,0x20,0x20,0x3a,0xea,
    0x78,0xfa,0xb1,0x9a,0x94,0x71,0xca,0x1c, 0x18,0x4a,0x8c,0xff,0xc4,0x83,0xc1,0x82,
  };
  // "The secret message is: "
  static u8 plaintext[sizeof(ciphertext)] = {
    0x54,0x68,0x65,0x20,0x73,0x65,0x63,0x72,0x65,0x74,0x20,0x6d,0x65,0x73,0x73,0x61,
    0x67,0x65,0x20,0x69,0x73,0x3a,0x20
  };
  #define KNOWN_PLAINTEXT_SIZE 23
  //#endif

  static u8 message[sizeof(plaintext)];


  if( argc != 2 ) {
    fprintf( stderr, "%s success-key-to-test-in-hex (and incrementable format)\n", argv[0] );
    return -1;
  }

  memset( key, 0, sizeof(key) );
  int j = 0;
  if( strncmp( argv[1], "0x", 2 ) == 0 )
    j = 2;
  for( int i=0; i<8; i++,j+=2 ) {
    unsigned char z;

    char c = argv[1][j];
    if( c >= '0' && c <= '9' ) 
      z = (c - '0') * 16;
    else {
      c = toupper( c );
      if( c >= 'A' && c <= 'F' )
	z = ((c - 'A') + 10) * 16;
    }

    c = argv[1][j+1];
    if( c >= '0' && c <= '9' ) 
      z += (c - '0');
    else if( toupper(c) >= 'A' && toupper(c) <= 'F' )
      z += ((toupper(c) - 'A') + 10);
    else {
      c = toupper( c );
      if( c >= 'A' && c <= 'F' )
	z += (((c - 'A') + 10) * 16);
    }

    key[0][i] = z;
  }

  printf( "tested key : 0x");
  for( int i=0; i<16; i++ )
    printf( "%02X", key[0][i] );
  printf( " (incrementable format)\n" );
  
  u32 keyhi = 
    (key[0][0] << 24) |
    (key[0][1] << 16) |
    (key[0][2] <<  8) |
    (key[0][3] <<  0);
  u32 keylo = 
    (key[0][4] << 24) |
    (key[0][5] << 16) |
    (key[0][6] <<  8) |
    (key[0][7] <<  0);
  convert_key_from_inc_to_csc( &keyhi, &keylo );
  key[0][0] = (keyhi >> 24) & 0xFF;
  key[0][1] = (keyhi >> 16) & 0xFF;
  key[0][2] = (keyhi >>  8) & 0xFF;
  key[0][3] = (keyhi >>  0) & 0xFF;
  key[0][4] = (keylo >> 24) & 0xFF;
  key[0][5] = (keylo >> 16) & 0xFF;
  key[0][6] = (keylo >>  8) & 0xFF;
  key[0][7] = (keylo >>  0) & 0xFF;

  printf( "tested key : 0x");
  for( int i=0; i<16; i++ )
    printf( "%02X", key[0][i] );
  printf( " (CSC format)\n" );

  memcpy( message, ciphertext, sizeof(message) );
  cscipher_cbc_decrypt( message, sizeof(message)/8, key, iv );
  bool isok = true;
  printf( "plaintext (decrypted) :\n\t" );
  for( int i=0; i<(int)sizeof(message); i++ ) {
    if( isascii( message[i] ) )
      printf( " %c ", message[i] );
    else
      printf( "%02x ", (int)message[i] );
    if( i % 8 == 7 ) printf( "\n\t" );
    if( i < KNOWN_PLAINTEXT_SIZE && message[i] != plaintext[i] )
      isok = false;
  }
  printf( "\n" );
  if( !isok )
    printf( "WARNING !! decryption failed !!\n\n" );
  else {
    printf( "MOOOOO!! SEND THIS KEY TO CS : " );
    for( int i=0; i<7; i++ )
      printf( "%02X ", key[0][i] );
    printf( "\n\n" );
  }

  return 0;
}
