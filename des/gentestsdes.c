/*
 * $Log: gentestsdes.c,v $
 * Revision 1.6  1999/12/12 15:39:16  cyp
 * sync
 *
 * Revision 1.5.2.1  1999/12/12 15:11:56  cyp
 * removed reference to u64
 *
 * Revision 1.5  1998/06/18 22:50:15  remi
 * This is a C file, not a C++ one.
 *
 * Revision 1.4  1998/06/14 10:44:19  remi
 * Indentation.
 *
 * Revision 1.3  1998/06/14 08:27:09  friedbait
 * 'Id' tags added in order to support 'ident' command to display a bill of
 * material of the binary executable
 *
 * Revision 1.2  1998/06/14 08:13:25  friedbait
 * 'Log' keywords added to maintain automatic change history
 *
 */

static char *id="@(#)$Id: gentestsdes.c,v 1.6 1999/12/12 15:39:16 cyp Exp $";


/*
Date sent:        Sun, 11 Jan 1998 19:37:00 +0100
From:             Remi Guyomarch <rguyom@mail.dotcom.fr>
Organization:     Me ? Organized ?
To:               Andrew Meggs <insect@antennahead.com>
Copies to:        tcharron@interlog.com, jlawson@hmc.edu, beberg@distributed.net
Subject:          Re: (Fwd) DES x86-core news
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "des.h"

typedef unsigned int u32;
typedef unsigned char u8;

static const u8 odd_parity[256]={
  1,  1,  2,  2,  4,  4,  7,  7,  8,  8, 11, 11, 13, 13, 14, 14,
 16, 16, 19, 19, 21, 21, 22, 22, 25, 25, 26, 26, 28, 28, 31, 31,
 32, 32, 35, 35, 37, 37, 38, 38, 41, 41, 42, 42, 44, 44, 47, 47,
 49, 49, 50, 50, 52, 52, 55, 55, 56, 56, 59, 59, 61, 61, 62, 62,
 64, 64, 67, 67, 69, 69, 70, 70, 73, 73, 74, 74, 76, 76, 79, 79,
 81, 81, 82, 82, 84, 84, 87, 87, 88, 88, 91, 91, 93, 93, 94, 94,
 97, 97, 98, 98,100,100,103,103,104,104,107,107,109,109,110,110,
112,112,115,115,117,117,118,118,121,121,122,122,124,124,127,127,
128,128,131,131,133,133,134,134,137,137,138,138,140,140,143,143,
145,145,146,146,148,148,151,151,152,152,155,155,157,157,158,158,
161,161,162,162,164,164,167,167,168,168,171,171,173,173,174,174,
176,176,179,179,181,181,182,182,185,185,186,186,188,188,191,191,
193,193,194,194,196,196,199,199,200,200,203,203,205,205,206,206,
208,208,211,211,213,213,214,214,217,217,218,218,220,220,223,223,
224,224,227,227,229,229,230,230,233,233,234,234,236,236,239,239,
241,241,242,242,244,244,247,247,248,248,251,251,253,253,254,254};

#define TEST_CASE_COUNT 32

// DES test cases -- key/iv/plain/cypher
// note this is in .lo, .hi, .lo... order...
u32 des_test_cases[TEST_CASE_COUNT][8];

// ------------------------------------------------------------------
// ------------------------------------------------------------------
void main (void) 
{
    u8 key[8], iv[8], plain[8], cypher[8], plain2[8];
    des_key_schedule ks;
    int i,j;
    u32 hi, lo;

    printf ("u32 des_test_cases[TEST_CASE_COUNT][8] = {\n");
    srandom (time());

    for (i=0; i<TEST_CASE_COUNT; i++) {

	// first, set up some random keys,plain,iv
	for (j=0; j<8; j++) key[j] = odd_parity[(random() & 0x7F) << 1];
	des_test_cases[i][0] = key[7] | (key[6] << 8) | (key[5] << 16) | (key[4] << 24); // low
	des_test_cases[i][1] = key[3] | (key[2] << 8) | (key[1] << 16) | (key[0] << 24); // high
	for (j=0; j<8; j++) iv[j] = random() & 0xFF;
	des_test_cases[i][2] = iv[7] | (iv[6] << 8) | (iv[5] << 16) | (iv[4] << 24); // low
	des_test_cases[i][3] = iv[3] | (iv[2] << 8) | (iv[1] << 16) | (iv[0] << 24); // high
	for (j=0; j<8; j++) plain[j] = random() & 0xFF;
	des_test_cases[i][4] = plain[7] | (plain[6] << 8) | (plain[5] << 16) | (plain[4] << 24);
	des_test_cases[i][5] = plain[3] | (plain[2] << 8) | (plain[1] << 16) | (plain[0] << 24);

	// now encrypt all that and store cyphertext
	if ((j=key_sched((C_Block*)key,ks)) != 0) printf("Key error %d\n",j);
	des_cbc_encrypt((C_Block*)plain,(C_Block*)cypher, 8,ks, (C_Block*)iv,DES_ENCRYPT);
	des_test_cases[i][6] = cypher[7] | (cypher[6]<<8) | (cypher[5]<<16) | (cypher[4]<<24);
	des_test_cases[i][7] = cypher[3] | (cypher[2]<<8) | (cypher[1]<<16) | (cypher[0]<<24);
	des_cbc_encrypt((C_Block*)cypher,(C_Block*)plain2, 8,ks, (C_Block*)iv,DES_DECRYPT);
	if (memcmp(plain,plain2,8) != 0) printf("cbc_encrypt decrypt error\n");

	// write everything in C++ syntax
	printf ("  {0x%08X,0x%08X,0x%08X,0x%08X,0x%08X,0x%08X,0x%08X,0x%08X},\n",
		des_test_cases[i][0],des_test_cases[i][1],
		des_test_cases[i][2],des_test_cases[i][3],
		des_test_cases[i][4],des_test_cases[i][5],
		des_test_cases[i][6],des_test_cases[i][7]);
    }
    printf ("};\n");
}

