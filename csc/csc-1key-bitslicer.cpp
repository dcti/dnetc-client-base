// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-1key-bitslicer.cpp,v $
// Revision 1.5  2000/06/02 06:32:55  jlawson
// sync, copy files from release branch to head
//
// Revision 1.1.2.4  1999/11/01 17:23:23  cyp
// renamed transX(...) to csc_transX(...) to avoid potential (future) symbol
// collisions.
//
// Revision 1.1.2.3  1999/10/24 23:54:53  remi
// Use Problem::core_membuffer instead of stack for CSC cores.
// Align frequently used memory to 16-byte boundary in CSC cores.
//
// Revision 1.1.2.2  1999/10/08 00:07:00  cyp
// made (mostly) all extern "C" {}
//
// Revision 1.1.2.1  1999/10/07 18:41:13  cyp
// sync'd from head
//
// Revision 1.1  1999/07/23 02:43:05  fordbr
// CSC cores added
//
//

#if (!defined(lint) && defined(__showids__))
const char * PASTE(csc_1key_bitslicer_,CSC_SUFFIX) (void) {
return "@(#)$Id: csc-1key-bitslicer.cpp,v 1.5 2000/06/02 06:32:55 jlawson Exp $"; }
#endif

// ------------------------------------------------------------------
//            -O2      -O3
// 	
// 486    :  33.67
// K5     :  68.89    88.56
// K6     :          297.47
// K6-2   :          321.87 (gcc 2.7.2.1)
// K6-2   :          389.60
// alpha  :  77.16    67.07
//
#ifdef __cplusplus
extern "C" {
ulong 
PASTE(cscipher_bitslicer_,CSC_SUFFIX) 
( const ulong key[2][64], const ulong msg[64], const ulong cipher[64], char *membuffer );
}
#endif

ulong 
PASTE(cscipher_bitslicer_,CSC_SUFFIX) 
( const ulong key[2][64], const ulong msg[64], const ulong cipher[64], char *membuffer )
{
  //ulong subkey[9+2][64];
  ulong (*subkey)[9+2][64] = (ulong (*)[9+2][64])membuffer;
  ulong *skp;  // subkey[n]
  ulong *skp1; // subkey[n-1]
  const ulong *tcp; // pointer to tabc[] (bitslice values of c0..c8)
  const ulong *tep; // pointer to tabe[] (bitslice values of e and e')
  // temp. variables used in the encryption process
  ulong x0,x1,x2,x3,x4,x5,x6,x7, y1,y3,y5,y7;
  ulong xy56, xy34, xy12, xy70;
  ulong cfr[64];

#define APPLY_Ms(adr, adl)						\
  x0 = cfr[adl+0] ^ (skp[0+8] ^= skp[0+8-128]);				\
  x1 = cfr[adl+1] ^ (skp[1+8] ^= skp[1+8-128]);				\
  x2 = cfr[adl+2] ^ (skp[2+8] ^= skp[2+8-128]);				\
  x3 = cfr[adl+3] ^ (skp[3+8] ^= skp[3+8-128]);				\
  x4 = cfr[adl+4] ^ (skp[4+8] ^= skp[4+8-128]);				\
  x5 = cfr[adl+5] ^ (skp[5+8] ^= skp[5+8-128]);				\
  x6 = cfr[adl+6] ^ (skp[6+8] ^= skp[6+8-128]);				\
  x7 = cfr[adl+7] ^ (skp[7+8] ^= skp[7+8-128]);				\
  csc_transP(                                                           \
          x7 ^ (y7   =      cfr[adr+7] ^ (skp[7] ^= skp[7-128])),	\
	  x6 ^ (xy56 = x5 ^ cfr[adr+6] ^ (skp[6] ^= skp[6-128])),	\
	  x5 ^ (y5   =      cfr[adr+5] ^ (skp[5] ^= skp[5-128])),	\
	  x4 ^ (xy34 = x3 ^ cfr[adr+4] ^ (skp[4] ^= skp[4-128])),	\
	  x3 ^ (y3   =      cfr[adr+3] ^ (skp[3] ^= skp[3-128])),	\
	  x2 ^ (xy12 = x1 ^ cfr[adr+2] ^ (skp[2] ^= skp[2-128])),	\
	  x1 ^ (y1   =      cfr[adr+1] ^ (skp[1] ^= skp[1-128])),	\
	  x0 ^ (xy70 = x7 ^ cfr[adr+0] ^ (skp[0] ^= skp[0-128])),	\
	  cfr[adl+7], cfr[adl+6], cfr[adl+5], cfr[adl+4],		\
	  cfr[adl+3], cfr[adl+2], cfr[adl+1], cfr[adl+0] );		\
  csc_transP(                          					\
          x6 ^ y7, xy56, x4 ^ y5, xy34,					\
	  x2 ^ y3, xy12, x0 ^ y1, xy70,					\
	  cfr[adr+7], cfr[adr+6], cfr[adr+5], cfr[adr+4],		\
	  cfr[adr+3], cfr[adr+2], cfr[adr+1], cfr[adr+0] );		\
  skp += 16;

#define APPLY_Me(adr, adl)						\
  x0 = cfr[adl+0] ^ tep[0+8]; x1 = cfr[adl+1] ^ tep[1+8];		\
  x2 = cfr[adl+2] ^ tep[2+8]; x3 = cfr[adl+3] ^ tep[3+8];		\
  x4 = cfr[adl+4] ^ tep[4+8]; x5 = cfr[adl+5] ^ tep[5+8];		\
  x6 = cfr[adl+6] ^ tep[6+8]; x7 = cfr[adl+7] ^ tep[7+8];		\
  csc_transP(                      					\
          x7 ^ (y7   =      cfr[adr+7] ^ tep[7]),			\
	  x6 ^ (xy56 = x5 ^ cfr[adr+6] ^ tep[6]),			\
	  x5 ^ (y5   =      cfr[adr+5] ^ tep[5]),			\
	  x4 ^ (xy34 = x3 ^ cfr[adr+4] ^ tep[4]),			\
	  x3 ^ (y3   =      cfr[adr+3] ^ tep[3]),			\
	  x2 ^ (xy12 = x1 ^ cfr[adr+2] ^ tep[2]),			\
	  x1 ^ (y1   =      cfr[adr+1] ^ tep[1]),			\
	  x0 ^ (xy70 = x7 ^ cfr[adr+0] ^ tep[0]),			\
	  cfr[adl+7], cfr[adl+6], cfr[adl+5], cfr[adl+4],		\
	  cfr[adl+3], cfr[adl+2], cfr[adl+1], cfr[adl+0] );		\
  csc_transP( 								\
          x6 ^ y7, xy56, x4 ^ y5, xy34,					\
	  x2 ^ y3, xy12, x0 ^ y1, xy70,					\
	  cfr[adr+7], cfr[adr+6], cfr[adr+5], cfr[adr+4],		\
	  cfr[adr+3], cfr[adr+2], cfr[adr+1], cfr[adr+0] );		\
  tep += 16;


  // global initializations
  tcp = &csc_tabc[0][0];
  memcpy( &(*subkey)[0], &key[1], sizeof((*subkey)[0]) );
  memcpy( &(*subkey)[1], &key[0], sizeof((*subkey)[1]) );
  skp  = &(*subkey)[2][0];
  skp1 = &(*subkey)[1][0];
  memcpy( cfr, msg, sizeof(cfr) );

  // the first 8 rounds
  for( int sk=8; sk; sk-- ) {
    for( int n=8; n; n--,tcp+=8,skp1+=8,skp++ )
      csc_transP( 
              skp1[7] ^ tcp[7], skp1[6] ^ tcp[6], skp1[5] ^ tcp[5], skp1[4] ^ tcp[4],
	      skp1[3] ^ tcp[3], skp1[2] ^ tcp[2], skp1[1] ^ tcp[1], skp1[0] ^ tcp[0],
	      skp[56], skp[48], skp[40], skp[32], skp[24], skp[16], skp[ 8], skp[ 0] );
    skp -= 8;

    APPLY_Ms(  0,  8);
    APPLY_Ms( 16, 24);
    APPLY_Ms( 32, 40);
    APPLY_Ms( 48, 56);

    tep = &csc_tabe[0][0];
    APPLY_Me(  0, 16);
    APPLY_Me( 32, 48);
    APPLY_Me(  8, 24);
    APPLY_Me( 40, 56);
    APPLY_Me(  0, 32);
    APPLY_Me(  8, 40);
    APPLY_Me( 16, 48);
    APPLY_Me( 24, 56);
  }

  // the last round
  ulong result = _1;
  for( int n=0; n<8; n++,tcp+=8,skp1+=8,skp++ ) {
    csc_transP( 
            skp1[7] ^ tcp[7], skp1[6] ^ tcp[6], skp1[5] ^ tcp[5], skp1[4] ^ tcp[4],
	    skp1[3] ^ tcp[3], skp1[2] ^ tcp[2], skp1[1] ^ tcp[1], skp1[0] ^ tcp[0],
	    skp[56], skp[48], skp[40], skp[32], skp[24], skp[16], skp[ 8], skp[ 0] );
    result &= ~(cipher[56+n] ^ cfr[56+n] ^ skp[56] ^ skp[56-128]); if( !result ) break;
    result &= ~(cipher[48+n] ^ cfr[48+n] ^ skp[48] ^ skp[48-128]); if( !result ) break;
    result &= ~(cipher[40+n] ^ cfr[40+n] ^ skp[40] ^ skp[40-128]); if( !result ) break;
    result &= ~(cipher[32+n] ^ cfr[32+n] ^ skp[32] ^ skp[32-128]); if( !result ) break;
    result &= ~(cipher[24+n] ^ cfr[24+n] ^ skp[24] ^ skp[24-128]); if( !result ) break;
    result &= ~(cipher[16+n] ^ cfr[16+n] ^ skp[16] ^ skp[16-128]); if( !result ) break;
    result &= ~(cipher[ 8+n] ^ cfr[ 8+n] ^ skp[ 8] ^ skp[ 8-128]); if( !result ) break;
    result &= ~(cipher[ 0+n] ^ cfr[ 0+n] ^ skp[ 0] ^ skp[ 0-128]); if( !result ) break;
  }
  return result;
}
