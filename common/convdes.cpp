// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: convdes.cpp,v $
// Revision 1.12  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.11  1998/10/04 11:35:32  remi
// Id tags fun.
//
// Revision 1.10  1998/07/07 21:55:33  cyruspatel
// Serious house cleaning - client.h has been split into client.h (Client
// class, FileEntry struct etc - but nothing that depends on anything) and
// baseincs.h (inclusion of generic, also platform-specific, header files).
// The catchall '#include "client.h"' has been removed where appropriate and
// replaced with correct dependancies. cvs Ids have been encapsulated in
// functions which are later called from cliident.cpp. Corrected other
// compile-time warnings where I caught them. Removed obsolete timer and
// display code previously def'd out with #if NEW_STATS_AND_LOGMSG_STUFF.
// Made MailMessage in the client class a static object (in client.cpp) in
// anticipation of global log functions.
//
// Revision 1.9  1998/06/29 08:44:09  jlawson
// More OS_WIN32S/OS_WIN16 differences and long constants added.
//
// Revision 1.8  1998/06/29 04:22:21  jlawson
// Updates for 16-bit Win16 support
//
// Revision 1.7  1998/06/15 12:03:56  kbracey
// Lots of consts.
//
// Revision 1.6  1998/06/14 08:26:46  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.5  1998/06/14 08:12:48  friedbait
// 'Log' keywords added to maintain automatic change history
//
//

#if (!defined(lint) && defined(__showids__))
const char *convdes_cpp(void) {
return "@(#)$Id: convdes.cpp,v 1.12 1999/01/01 02:45:15 cramer Exp $"; }
#endif

// DES convertion routines

#include <stdio.h>
#include <string.h>
#include "problem.h"
#include "convdes.h"

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif

//#define DEBUG

#if (CLIENT_CPU != CPU_ARM)

const u8 odd_parity[256]={
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

// ------------------------------------------------------------------
// Convert a key from DES format to incrementable format
//
// Accept a deshi:deslo pair in 32+32 bits, DES order format with parity bits.
// Returns the key in 24+32, incrementable format, without parity bits
//
// Fixed bits in bryd_des are (for a 64 bits key with parity bits) :
//               31                                0
//               |              'lo'               |
// 0x00247450 or 00000000 00100100 01110100 01010000
// bits 4,6,10,12,13,14,18,21
//
// And if we strip parity bits :
//               27                            0
//               |             'lo'            |
// 0x00049D28 or 0000 00000100 10011101 00101000
// bits 3,5,8,10,11,12,15,18
//
//
// Fixed bits for Meggs' bitslice core :
//               31                                0
//               |               'hi'              |
// 0x0636C000    00000110 00110110 11000000 00000000
//
// without parity bits :
//               27                            0
//               |             'hi'            |
//               0000 01100110 11110000 00000000
void convert_key_from_des_to_inc (u32 *deshi, u32 *deslo)
{

#ifdef DEBUG
    printf ("convert key from %08X:%08X (des to inc)\n",*deshi,*deslo);
#endif

// strip parity bits
    *deshi =
      ((*deshi & 0x000000FEL) >> 1) |
      ((*deshi & 0x0000FE00L) >> 2) |
      ((*deshi & 0x00FE0000L) >> 3) |
      ((*deshi & 0xFE000000L) >> 4);
   *deslo =
      ((*deslo & 0x000000FEL) >> 1) |
      ((*deslo & 0x0000FE00L) >> 2) |
      ((*deslo & 0x00FE0000L) >> 3) |
      ((*deslo & 0xFE000000L) >> 4);
// there is now 28 bits in *deshi, and 28 bits in *deslo
#ifdef DEBUG
    printf ("              to %07X:%07X (strip parity bits)\n",*deshi,*deslo);
#endif


// move fixed bits to the least significant positions
    u32 templo =

// shift bits 3,5,8,10,11,12,15,18 to the right
// to fill the 8 least significant bits
//
// 27           lo                  0
// |                                |
// 0000 0000 0100 1001 1101 0010 1000
//            f   e  d dd c   b  a     original position
//                          fedd dcba  new position

      ((*deslo & 0x0000008L) >> 3) |  // 'a' bit
      ((*deslo & 0x0000020L) >> 4) |  // 'b' bit
      ((*deslo & 0x0000100L) >> 6) |  // 'c' bit
      ((*deslo & 0x0001C00L) >> 7) |  // 'd' bits
      ((*deslo & 0x0008000L) >> 9) |  // 'e' bit
      ((*deslo & 0x0040000L) >>11) |  // 'f' bit

// to be compatible with Meggs' bitslice code we also need
// to move these bits from deshi to deslo
//
//    27                               0
//    |              'hi'              |
//    0000 0110 0110 1111 0000 0000 0000
//          cc   bb  aaaa                original position (in deshi)
//                   ccbb aaaa **** **** new position      (in deslo)

      ((*deshi & 0x000F000L) >> 4) |  // 'a' bits
      ((*deshi & 0x0060000L) >> 5) |  // 'b' bits
      ((*deshi & 0x0600000L) >> 7) |  // 'c' bits

// shift all other bits to the left
//
// 27                               0
// |              'lo'              |
// 0000 0000 0100 1001 1101 0010 1000
// .... .... g ff  ee    d  cc b  aaa  original position
// gffe edcc baaa ---- ---- **** ****  new position

      ((*deslo & 0x0000007L) <<16) |  // 'a' bits
      ((*deslo & 0x0000010L) <<15) |  // 'b' bit
      ((*deslo & 0x00000C0L) <<14) |  // 'c' bits
      ((*deslo & 0x0000200L) <<13) |  // 'd' bit
      ((*deslo & 0x0006000L) <<10) |  // 'e' bits
      ((*deslo & 0x0030000L) << 9) |  // 'f' bits
      ((*deslo & 0x0080000L) << 8);   // 'g' bits

// To be compatible with Meggs' bitslice code we also need
// to move these bits
//
//    27                               0
//    |              'hi'              |
//    0000 0110 0110 1111 0000 0000 0000
//    #### #  c c  b      aaaa aaaa aaaa original position
//    #### #ccb aaaa aaaa aaaa .... .... new position

    u32 temphi =
      ((*deshi & 0x0000FFFL) << 8) |  // 'a' bits
      ((*deshi & 0x0010000L) << 4) |  // 'b' bits
      ((*deshi & 0x0180000L) << 2) |  // 'c' bits
      ((*deshi & 0xF800000L)     ) |  // '#' bits (not moved)

// and also these bits moved from deslo to deshi :
//
// 27                               0
// |              'lo'              |
// 0000 0000 0100 1001 1101 0010 1000
// aaaa aaaa                           original position (in deslo)
//                          aaaa aaaa  new position (in deshi)

    ((*deslo & 0xFF00000L) >> 20);

    *deshi = temphi;
    *deslo = templo;

#ifdef DEBUG
    printf ("              to %07X:%07X (move bits)\n",*deshi,*deslo);
#endif

// now pack the key (remember that it's now without parity
// and so *deshi & *deslo have only 28 bits each)
    *deslo |= (*deshi & 0x0F) << 28;
    *deshi >>= 4;

#ifdef DEBUG
    printf ("              to %06X:%08X (incrementable format)\n",*deshi,*deslo);
#endif
}

// ------------------------------------------------------------------
// Convert a key from incrementable format to DES
// Revert what has done the previous function
//
// Accept a deshi:deslo pair in 24+32 bits, incrementable format
// Returns the key in 32+32, DES bit order, with parity bits
//
void convert_key_from_inc_to_des (u32 *deshi, u32 *deslo)
{

#ifdef DEBUG
    printf ("convert key from %06X:%08X (inc to des)\n",*deshi,*deslo);
#endif
// convert to 28+28 bits
    *deshi = (*deshi << 4) | ((*deslo >> 28) & 0x0F);
    *deslo &= 0x0FFFFFFFL;
// there is now 28 bits in *deshi, and 28 bits in *deslo
#ifdef DEBUG
    printf ("              to %07X:%07X (28+28)\n",*deshi,*deslo);
#endif

// shift the 8 least significant bits so that they
// will land in bits no 3,5,8,10,11,12,15,18
//
// 27                               0
// |              'lo'              |
// 0000 0000 0100 1001 1101 0010 1000
//                          fedd dcba  current position
//            f   e  d dd c   b  a     original position

    u32 templo =
        ((*deslo & 0x00000001L) << 3) |  // 'a' bit
        ((*deslo & 0x00000002L) << 4) |  // 'b' bit
        ((*deslo & 0x00000004L) << 6) |  // 'c' bit
        ((*deslo & 0x00000038L) << 7) |  // 'd' bits
        ((*deslo & 0x00000040L) << 9) |  // 'e' bit
        ((*deslo & 0x00000080L) <<11) |  // 'f' bit

// shift all other bits to the right
//
// 27                               0
// |              'lo'              |
// 0000 0000 0100 1001 1101 0010 1000
// gffe edcc baaa           **** ****  current position
// .... .... g ff  ee    d  cc b  aaa  original position

        ((*deslo & 0x0070000L) >>16) |  // 'a' bits
        ((*deslo & 0x0080000L) >>15) |  // 'b' bit
        ((*deslo & 0x0300000L) >>14) |  // 'c' bits
        ((*deslo & 0x0400000L) >>13) |  // 'd' bit
        ((*deslo & 0x1800000L) >>10) |  // 'e' bits
        ((*deslo & 0x6000000L) >> 9) |  // 'f' bits
        ((*deslo & 0x8000000L) >> 8) |  // 'g' bit

// and also these bits from deshi to deslo
//
// 27                               0
// |              'hi'              |
// 0000 0110 0110 1111 0000 0000 0000
//                          aaaa aaaa current position (in deshi)
// aaaa aaaa                          original position (in deslo)

        ((*deshi & 0x00000FFL) << 20);  // 'a' bits

// To be compatible with Meggs' bitslice code we also need
// to move these bits
//
// 27                               0
// |              'hi'              |
// 0000 0110 0110 1111 0000 0000 0000
// #### #ccb aaaa aaaa aaaa .... .... current position
// #### #  c c  b      aaaa aaaa aaaa original position

    u32 temphi =
        ((*deshi & 0x00FFF00L) >> 8) |  // 'a' bits
        ((*deshi & 0x0100000L) >> 4) |  // 'b' bit
        ((*deshi & 0x0600000L) >> 2) |  // 'c' bits
        ((*deshi & 0xF800000L)     ) |  // '#' bits (not moved)

// to be compatible with Meggs' bitslice code we also need
// to move these bits from deslo to deshi
//
// 27                               0
// |              'lo'              |
// 0000 0110 0110 1111 0000 0000 0000
//                ccbb aaaa **** **** current position  (in deslo)
//       cc   bb  aaaa                original position (in deshi)

        ((*deslo & 0x0000F00L) << 4) |  // 'a' bits
        ((*deslo & 0x0003000L) << 5) |  // 'b' bits
        ((*deslo & 0x000C000L) << 7);

    *deshi = temphi;
    *deslo = templo;

#ifdef DEBUG
    printf ("              to %07X:%07X (move bits)\n",*deshi,*deslo);
#endif

// now add parity bits
// take each group of 7 bits, and add parity bit with a table lookup
    *deshi =
        (((u32) odd_parity[(int) ((*deshi & 0x0000007FL) << 1)]) << 0) |
        (((u32) odd_parity[(int) ((*deshi & 0x00003F80L) >> 6)]) << 8) |
        (((u32) odd_parity[(int) ((*deshi & 0x001FC000L) >>13)]) <<16) |
        (((u32) odd_parity[(int) ((*deshi & 0x0FE00000L) >>20)]) <<24);
    *deslo =
        (((u32) odd_parity[(int) ((*deslo & 0x0000007FL) << 1)]) << 0) |
        (((u32) odd_parity[(int) ((*deslo & 0x00003F80L) >> 6)]) << 8) |
        (((u32) odd_parity[(int) ((*deslo & 0x001FC000L) >>13)]) <<16) |
        (((u32) odd_parity[(int) ((*deslo & 0x0FE00000L) >>20)]) <<24);

#ifdef DEBUG
    printf ("              to %08X:%08X (with parity)\n",*deshi,*deslo);
#endif
}

// ------------------------------------------------------------------


#endif // CLIENT_CPU != CPU_ARM

