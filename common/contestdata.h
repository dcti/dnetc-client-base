// RC5-56:
// Contest identifier: RC5-32/12/7 
// Cipher: RC5-32/12/7 (RC5 with 32-bit wordsize, 12 rounds, and
// 7*8=56-bit key) 
// Start of contest: 28 January 1997, 9 am PST 
// State of contest: finished 
// IV: 7b 32 f0 8a e6 17 de 8c 
// Hexadecimal ciphertext:
//   82 d3 4e a7 b3 24 86 0b c6 d8 61 5c e9 f9 e4 79
//   88 5c 98 f1 d2 92 4c 59 ee 47 51 31 01 3e a8 ab
//   d6 f0 4d c8 19 97 af 01 5e af f8 3f cd 61 b3 c2
//   66 89 7c 82 09 87 4d fb 07 f2 56 03 8d d5 1b 01
//   ca e3 41 c2 8d d7 18 1d             
#define RC556_IVLO 0x8AF0327BL
#define RC556_IVHI 0x8CDE17E6L
#define RC556_CYPHERLO 0xA74ED382L
#define RC556_CYPHERHI 0x0B8624B3L
#define RC556_PLAINLO 0x20656854L
#define RC556_PLAINHI 0x6E6B6E75L

// RC5-64:
// Contest identifier: RC5-32/12/8 
// Cipher: RC5-32/12/8 (RC5 with 32-bit wordsize, 12 rounds, and
// 8*8=64-bit key) 
// Start of contest: 28 January 1997, 9 am PST 
// State of contest: ongoing 
// IV: 79 ce d5 d5 50 75 ea fc 
// Hexadecimal ciphertext:
//   bf 55 01 55 dc 26 f2 4b 26 e4 85 4d f9 0a d6 79
//   66 93 ab 92 3c 72 f1 37 c8 b7 0d 1f 60 11 0c 92
//   ae 2e cd fd 70 d3 fd 17 df b0 42 12 b9 7d cf 22
//   18 6b a7 15 ce 2c 84 bf ce 0d d0 4d 00 6b e1 46
#define RC564_IVLO 0xD5D5CE79L
#define RC564_IVHI 0xFCEA7550L
#define RC564_CYPHERLO 0x550155BFL
#define RC564_CYPHERHI 0x4BF226DCL
#define RC564_PLAINLO 0x20656854L
#define RC564_PLAINHI 0x6E6B6E75L

// DES-II-1:
// Contest Identifier: DES-II-1
// Cipher: DES
// Start of contest: 13 January 1998, 9 am PST
// State of contest: finished
// IV: fc 69 a5 54 ca d6 42 b1
// Hex ciphertext:
//   09 86 1f 2d 37 36 d5 f1 88 e0 b5 4c d3 2c 12 e3
//   ee 0f 89 7e 00 12 e7 40 df c2 81 a1 df 6d 29 3d
//   c5 af e1 9c 4a 08 ce 49 f2 21 95 4d de 17 5c 1b
//   8c d9 d8 20 24 6b b4 89
#define DESII1_CYPHERLO 0x3736d5f1
#define DESII1_CYPHERHI 0x09861f2d
#define DESII1_IVLO 0xcad642b1
#define DESII1_IVHI 0xfc69a554
#define DESII1_PLAINLO 0x756e6b6e
#define DESII1_PLAINHI 0x54686520

// DES-II-2:
// Contest identifier: DES-II-2
// Cipher: DES
// Start of the contest: 13 July 1998, 9 am PDT
// State of the contest: finished
// IV: 09 f0 04 15 c7 c3 6d 8e
// Hexadecimal ciphertext:
//   1d 49 50 87 a7 68 f5 ac a6 63 4b 90 b0 fa 02 e9 2f 12 c4 0f
//   5f 3f 07 c7 04 72 76 64 0d 37 fa 20 3e ae 18 39 c7 83 1d 11
//   d4 87 a9 18 5e 72 3f 98 a7 59 26 d6 21 45 10 7d eb b3 46 2f
//   54 43 09 bf 03 68 63 e3 d7 2c f6 9d 40 e2 65 76 3b d6 6c 38
#define DESII2_CYPHERLO 0xa768f5ac
#define DESII2_CYPHERHI 0x1d495087
#define DESII2_IVLO 0xc7c36d8e
#define DESII2_IVHI 0x09f00415
#define DESII2_PLAINLO 0x756e6b6e
#define DESII2_PLAINHI 0x54686520

// DES-II-3: (this is fake data here since it is not yet known)
#define DESII3_CYPHERLO 0
#define DESII3_CYPHERHI 0
#define DESII3_IVLO 0
#define DESII3_IVHI 0
#define DESII3_CYPHERLO 0
#define DESII3_CYPHERHI 0
#define DESII3_PLAINLO 0x756e6b6e
#define DESII3_PLAINHI 0x54686520

// Signals used to indiciate contest closure.
#define RC5CLOSEDCODE 0x1337F00DL
#define DESCLOSEDCODE 0xBEEFF00DL

const int contestclosedcodes[2]=
  {
  RC5CLOSEDCODE,
  DESCLOSEDCODE
  };
