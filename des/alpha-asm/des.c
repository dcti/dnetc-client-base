/*
Copyright 1999 by Christoph Dworzak, Biel, Switzerland - All Rights Reserved
For use in distributed.net projects only.
Any other distribution or use of this source violates copyright.
*/

#include <sys/types.h>
#include <memory.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>
#include <sys/signal.h>

extern unsigned long PT, CT, SK;
extern unsigned long checkKey(const unsigned long *K, unsigned long init);
unsigned long cd,ccm=0,cc=~0UL,ccs=0,n=0;

static void finish(int ignore){
    signal(SIGINT, SIG_IGN);
	n *= 64;
    printf("\n%ld,%ld,%ld Cycles/Key (%ld,%ld,%ld Cycles/Round)\n", cc>>6, ccs/n, ccm>>6, cc, ccs/(n>>6), ccm);
    printf("%ld,%ld Key/s (@500MHz)\n", 64UL*500000000UL/cc, n*500000000UL/ccs);
    printf("%ld Keys tested\n", n);
    exit(0);
}

char htoi(char c){
    if('0' <= c && c <= '9')
        return c - '0';
    if('a' <= c && c <= 'f')
        return 10 + c - 'a';
    if('A' <= c && c <= 'F')
        return 10 + c - 'A';
    return 0;
}

unsigned long atob(char *hex) {
    int i;
    unsigned long l=0;

    for (i=0; i<8; i++){
	l <<= 8;
	l |= htoi(hex[2*i]) << 4;
	l |= htoi(hex[2*i+1]);
    }
    return(l);
}

unsigned long scrambleCT(unsigned long l) {
    unsigned long binary;

    binary  = ((l >>  6) & 1UL) <<  0;
    binary |= ((l >> 14) & 1UL) <<  1;
    binary |= ((l >> 22) & 1UL) <<  2;
    binary |= ((l >> 30) & 1UL) <<  3;
    binary |= ((l >> 38) & 1UL) <<  4;
    binary |= ((l >> 46) & 1UL) <<  5;
    binary |= ((l >> 54) & 1UL) <<  6;
    binary |= ((l >> 62) & 1UL) <<  7;
    binary |= ((l >>  4) & 1UL) <<  8;
    binary |= ((l >> 12) & 1UL) <<  9;
    binary |= ((l >> 20) & 1UL) << 10;
    binary |= ((l >> 28) & 1UL) << 11;
    binary |= ((l >> 36) & 1UL) << 12;
    binary |= ((l >> 44) & 1UL) << 13;
    binary |= ((l >> 52) & 1UL) << 14;
    binary |= ((l >> 60) & 1UL) << 15;
    binary |= ((l >>  2) & 1UL) << 16;
    binary |= ((l >> 10) & 1UL) << 17;
    binary |= ((l >> 18) & 1UL) << 18;
    binary |= ((l >> 26) & 1UL) << 19;
    binary |= ((l >> 34) & 1UL) << 20;
    binary |= ((l >> 42) & 1UL) << 21;
    binary |= ((l >> 50) & 1UL) << 22;
    binary |= ((l >> 58) & 1UL) << 23;
    binary |= ((l >>  0) & 1UL) << 24;
    binary |= ((l >>  8) & 1UL) << 25;
    binary |= ((l >> 16) & 1UL) << 26;
    binary |= ((l >> 24) & 1UL) << 27;
    binary |= ((l >> 32) & 1UL) << 28;
    binary |= ((l >> 40) & 1UL) << 29;
    binary |= ((l >> 48) & 1UL) << 30;
    binary |= ((l >> 56) & 1UL) << 31;
    binary |= ((l >>  7) & 1UL) << 32;
    binary |= ((l >> 15) & 1UL) << 33;
    binary |= ((l >> 23) & 1UL) << 34;
    binary |= ((l >> 31) & 1UL) << 35;
    binary |= ((l >> 39) & 1UL) << 36;
    binary |= ((l >> 47) & 1UL) << 37;
    binary |= ((l >> 55) & 1UL) << 38;
    binary |= ((l >> 63) & 1UL) << 39;
    binary |= ((l >>  5) & 1UL) << 40;
    binary |= ((l >> 13) & 1UL) << 41;
    binary |= ((l >> 21) & 1UL) << 42;
    binary |= ((l >> 29) & 1UL) << 43;
    binary |= ((l >> 37) & 1UL) << 44;
    binary |= ((l >> 45) & 1UL) << 45;
    binary |= ((l >> 53) & 1UL) << 46;
    binary |= ((l >> 61) & 1UL) << 47;
    binary |= ((l >>  3) & 1UL) << 48;
    binary |= ((l >> 11) & 1UL) << 49;
    binary |= ((l >> 19) & 1UL) << 50;
    binary |= ((l >> 27) & 1UL) << 51;
    binary |= ((l >> 35) & 1UL) << 52;
    binary |= ((l >> 43) & 1UL) << 53;
    binary |= ((l >> 51) & 1UL) << 54;
    binary |= ((l >> 59) & 1UL) << 55;
    binary |= ((l >>  1) & 1UL) << 56;
    binary |= ((l >>  9) & 1UL) << 57;
    binary |= ((l >> 17) & 1UL) << 58;
    binary |= ((l >> 25) & 1UL) << 59;
    binary |= ((l >> 33) & 1UL) << 60;
    binary |= ((l >> 41) & 1UL) << 61;
    binary |= ((l >> 49) & 1UL) << 62;
    binary |= ((l >> 57) & 1UL) << 63;
    return(binary);
}

unsigned long scramblePT(unsigned long l) {
    unsigned long binary;

    binary  = ((l >>  7) & 1UL) <<  0;
    binary |= ((l >> 15) & 1UL) <<  1;
    binary |= ((l >> 23) & 1UL) <<  2;
    binary |= ((l >> 31) & 1UL) <<  3;
    binary |= ((l >> 39) & 1UL) <<  4;
    binary |= ((l >> 47) & 1UL) <<  5;
    binary |= ((l >> 55) & 1UL) <<  6;
    binary |= ((l >> 63) & 1UL) <<  7;
    binary |= ((l >>  5) & 1UL) <<  8;
    binary |= ((l >> 13) & 1UL) <<  9;
    binary |= ((l >> 21) & 1UL) << 10;
    binary |= ((l >> 29) & 1UL) << 11;
    binary |= ((l >> 37) & 1UL) << 12;
    binary |= ((l >> 45) & 1UL) << 13;
    binary |= ((l >> 53) & 1UL) << 14;
    binary |= ((l >> 61) & 1UL) << 15;
    binary |= ((l >>  3) & 1UL) << 16;
    binary |= ((l >> 11) & 1UL) << 17;
    binary |= ((l >> 19) & 1UL) << 18;
    binary |= ((l >> 27) & 1UL) << 19;
    binary |= ((l >> 35) & 1UL) << 20;
    binary |= ((l >> 43) & 1UL) << 21;
    binary |= ((l >> 51) & 1UL) << 22;
    binary |= ((l >> 59) & 1UL) << 23;
    binary |= ((l >>  1) & 1UL) << 24;
    binary |= ((l >>  9) & 1UL) << 25;
    binary |= ((l >> 17) & 1UL) << 26;
    binary |= ((l >> 25) & 1UL) << 27;
    binary |= ((l >> 33) & 1UL) << 28;
    binary |= ((l >> 41) & 1UL) << 29;
    binary |= ((l >> 49) & 1UL) << 30;
    binary |= ((l >> 57) & 1UL) << 31;
    binary |= ((l >>  6) & 1UL) << 32;
    binary |= ((l >> 14) & 1UL) << 33;
    binary |= ((l >> 22) & 1UL) << 34;
    binary |= ((l >> 30) & 1UL) << 35;
    binary |= ((l >> 38) & 1UL) << 36;
    binary |= ((l >> 46) & 1UL) << 37;
    binary |= ((l >> 54) & 1UL) << 38;
    binary |= ((l >> 62) & 1UL) << 39;
    binary |= ((l >>  4) & 1UL) << 40;
    binary |= ((l >> 12) & 1UL) << 41;
    binary |= ((l >> 20) & 1UL) << 42;
    binary |= ((l >> 28) & 1UL) << 43;
    binary |= ((l >> 36) & 1UL) << 44;
    binary |= ((l >> 44) & 1UL) << 45;
    binary |= ((l >> 52) & 1UL) << 46;
    binary |= ((l >> 60) & 1UL) << 47;
    binary |= ((l >>  2) & 1UL) << 48;
    binary |= ((l >> 10) & 1UL) << 49;
    binary |= ((l >> 18) & 1UL) << 50;
    binary |= ((l >> 26) & 1UL) << 51;
    binary |= ((l >> 34) & 1UL) << 52;
    binary |= ((l >> 42) & 1UL) << 53;
    binary |= ((l >> 50) & 1UL) << 54;
    binary |= ((l >> 58) & 1UL) << 55;
    binary |= ((l >>  0) & 1UL) << 56;
    binary |= ((l >>  8) & 1UL) << 57;
    binary |= ((l >> 16) & 1UL) << 58;
    binary |= ((l >> 24) & 1UL) << 59;
    binary |= ((l >> 32) & 1UL) << 60;
    binary |= ((l >> 40) & 1UL) << 61;
    binary |= ((l >> 48) & 1UL) << 62;
    binary |= ((l >> 56) & 1UL) << 63;
    return(binary);
}

unsigned long delParity(unsigned long SK) {
	SK = ((SK&0xFEUL)>>1) |
		 ((SK&0xFE00UL)>>2) |
		 ((SK&0xFE0000UL)>>3) |
		 ((SK&0xFE000000UL)>>4) |
		 ((SK&0xFE00000000UL)>>5) |
		 ((SK&0xFE0000000000UL)>>6) |
		 ((SK&0xFE000000000000UL)>>7) |
		 ((SK&0xFE00000000000000UL)>>8);
	return(SK);
}

void findKey(unsigned long pt, unsigned long iv, unsigned long ct, unsigned long sk){
    unsigned long K[56], i, j, c1, c2, init;

	pt ^= iv;
	PT = scramblePT(pt);
	CT = scrambleCT(ct);
	SK = delParity(sk);

    for (j=2;j;j--){
	printf("PT=%016lX CT=%016lX SK=%014lX\n", PT, CT, SK);
    for (i=0; i<56; i++) if (SK & (1UL << i)) K[i] = ~0UL; else K[i] = 0;
    K[0] = 0xFFFFFFFF00000000UL; K[1] = 0xFFFF0000FFFF0000UL;
    K[2] = 0xFF00FF00FF00FF00UL; K[3] = 0xF0F0F0F0F0F0F0F0UL;
    K[4] = 0xCCCCCCCCCCCCCCCCUL; K[5] = 0xAAAAAAAAAAAAAAAAUL;
    K[ 8] = ~0UL; K[10] = ~0UL; K[11] = ~0UL; K[12] = ~0UL;
    K[15] = ~0UL; K[18] = ~0UL; K[42] = ~0UL; K[43] = ~0UL;
    K[45] = ~0UL; K[46] = ~0UL; K[49] = ~0UL; K[50] = ~0UL;
    K[40] = ~0UL; K[41] = ~0UL; K[ 6] = ~0UL; K[ 7] = ~0UL;

    init = 2;
    asm volatile("rpcc %0" : "=r"(c1) : : "memory");
    c1 = (c1+(c1<<32))>>32;

    do{do{do{do{do{do{do{do{do{do{do{do{do{do{do{do{

    	n++;
    	if (i = checkKey(K, init)) goto found;
    	init = 0;

    	asm volatile("rpcc %0" : "=r"(c2) : : "memory");
    	c2 = (c2+(c2<<32))>>32; cd = (c2>c1) ? c2-c1 : c2+(1UL<<32)-c1;
    	c1 = c2; ccs += cd; if (cd<cc) cc = cd; if (cd>ccm) ccm = cd;
    
    K[10] = ~K[10];} while (~K[10]);
    K[12] = ~K[12];} while (~K[12]);
    K[15] = ~K[15];} while (~K[15]);
    K[18] = ~K[18];} while (~K[18]);
    K[45] = ~K[45];} while (~K[45]);
    K[46] = ~K[46];} while (~K[46]);
    K[49] = ~K[49];} while (~K[49]);
    K[50] = ~K[50];} while (~K[50]);
	init =1;
    K[ 8] = ~K[ 8];} while (~K[ 8]);
    K[11] = ~K[11];} while (~K[11]);
    K[42] = ~K[42];} while (~K[42]);
    K[43] = ~K[43];} while (~K[43]);
    K[40] = ~K[40];} while (~K[40]);
    K[41] = ~K[41];} while (~K[41]);
    K[ 6] = ~K[ 6];} while (~K[ 6]);
    K[ 7] = ~K[ 7];} while (~K[ 7]);
    SK=(~SK)&0x00ffffffffffffffUL;
    }

    printf(" -> Key not found!\n");
	return;

found:
    for (c1=c2=0;c1<56;c1++)
	if (K[c1]&i)
	    c2 |= 1UL<<c1;
    printf(" -> Key=%014lX\n", c2);
}

int main(int argc, char **argv) {
    char line[255], *tmp;
	unsigned long pt, iv, ct, sk;

    signal(SIGINT, finish);
    while (1){
	fgets(line, 255, stdin);
	if (feof(stdin)) break;
		tmp = &line[5];
		pt = atob(tmp);
		fgets(line, 255, stdin);
  		tmp = &line[5];
		iv = atob(tmp);
  		fgets(line, 255, stdin);
		tmp = &line[5];
	    ct = atob(tmp);
		fgets(line, 255, stdin);
	    tmp = &line[5];
		sk = atob(tmp);
	    printf("PT=%016lX IV=%016lX CT=%016lX SK=%016lX\n", pt, iv, ct, sk);
	    findKey(pt, iv, ct, sk);
    }
    finish(0);
}
