// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: des-x86.cpp,v $
// Revision 1.21  1998/12/02 20:56:42  silby
// Changed Multithread -> CLIENT_SUPPORTS_SMP
//
// Revision 1.20  1998/11/28 19:21:40  silby
// Fixed nasty define that broke win32 (and others?)
//
// Revision 1.19  1998/11/25 06:12:54  dicamillo
// Update for BeOS R4 for Intel because elf format is used now.
//
// Revision 1.18  1998/11/07 08:29:46  remi
// Changed a #ifdef to be a bit more human-readable.
//
// Revision 1.17  1998/10/19 00:32:08  daa
// add a FREEBSD/X86/ELF set to the bryd-* defines
//
// Revision 1.16  1998/10/07 17:46:44  snake
//
// Modified for BSD/OS 4.0 and ELF
//
// Revision 1.15  1998/07/08 23:42:08  remi
// Added support for CliIdentifyModules().
//
// Revision 1.14  1998/07/08 18:53:06  remi
// Added function for CliIdentifyModules().
//
// Revision 1.13  1998/07/07 07:42:20  jlawson
// added lint tags around cvs id to supress unused variable warning
//
// Revision 1.12  1998/06/23 21:59:08  remi
// Use only two x86 DES cores (P5 & PPro) when not multithreaded.
//
// Revision 1.11  1998/06/18 00:18:26  silby
// p1bdespro.asm and p2bdespro.asm aren't being removed.
//
// Revision 1.9  1998/06/17 22:12:51  remi
// No need of more than two bryd_key_found and bryd_continue C routines.
//
// Revision 1.8  1998/06/17 21:56:07  remi
// Fixed p?bryd_des routines naming.
//
// Revision 1.7  1998/06/16 22:12:29  silby
// fixed second pro thread not working right
//
// Revision 1.5  1998/06/16 21:49:46  silby
// Added p1bdespro and p2bdespro, really the "old" p5 bryddes cores until
// the pro ones are ported to .s
//
// des-x86.cpp has been modified for the dual x86 core support (p5/pro)
//
// Revision 1.4  1998/06/14 08:27:04  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.3  1998/06/14 08:13:19  friedbait
// 'Log' keywords added to maintain automatic change history
//
//

// encapsulate the BrydDES library

#if (!defined(lint) && defined(__showids__))
const char *des_x86_cpp(void) {
return "@(#)$Id: des-x86.cpp,v 1.21 1998/12/02 20:56:42 silby Exp $"; }
#endif


#include <stdio.h>
#include <string.h>
#include "../common/problem.h"
#include "../common/convdes.h"

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif

//#define DEBUG

#if defined(__WATCOMC__) || \
    ((CLIENT_OS == OS_SOLARIS) && (CLIENT_CPU == CPU_X86)) || \
    ((CLIENT_OS == OS_LINUX) && (CLIENT_CPU == CPU_X86) && defined(__ELF__)) || \
    ((CLIENT_OS == OS_BSDI) && (CLIENT_CPU == CPU_X86) && defined(__ELF__)) || \
    ((CLIENT_OS == OS_FREEBSD) && (CLIENT_CPU == CPU_X86) && defined(__ELF__)) || \
    ((CLIENT_OS == OS_BEOS) && (CLIENT_CPU == CPU_X86) && defined(__ELF__))
 #define bryd_des _bryd_des
 #define bbryd_des _bbryd_des
 #define p1bryd_des _p1bryd_des
 #define p2bryd_des _p2bryd_des
 #define bryd_continue _bryd_continue
 #define bryd_key_found _bryd_key_found
 #define bbryd_continue _bbryd_continue
 #define bbryd_key_found _bbryd_key_found
#endif

// this is in high..low format
// bitmasks[][0] & 0x80 = most significant bit
// bitmasks[][7] & 0x01 = least significant bit
static const u8 bitmasks [][8] = {
// least significant bit of each byte should be set to 0
// bits set to 1 which must be 0 in 'bitmask' :
//  00   00   00   00   00   24   74   50
// bits that should be set (incrementaly) to 1 when timeslice > 256 (Meggs' compatibility)
//  06   36   C0   00   00   00   00   00
{ 0xFE,0xFE,0xFE,0xFE,0xFE,0xDA,0x8A,0xAE }, // timeslice =    256
{ 0xFE,0xFE,0xBE,0xFE,0xFE,0xDA,0x8A,0xAE }, // timeslice =    512
{ 0xFE,0xFE,0x3E,0xFE,0xFE,0xDA,0x8A,0xAE }, // timeslice =   1024
{ 0xFE,0xFC,0x3E,0xFE,0xFE,0xDA,0x8A,0xAE }, // timeslice =   2048
{ 0xFE,0xF8,0x3E,0xFE,0xFE,0xDA,0x8A,0xAE }, // timeslice =   4096
{ 0xFE,0xE8,0x3E,0xFE,0xFE,0xDA,0x8A,0xAE }, // timeslice =   8192
{ 0xFE,0xC8,0x3E,0xFE,0xFE,0xDA,0x8A,0xAE }, // timeslice =  16384
{ 0xFC,0xC8,0x3E,0xFE,0xFE,0xDA,0x8A,0xAE }, // timeslice =  32768
{ 0xF8,0xC8,0x3E,0xFE,0xFE,0xDA,0x8A,0xAE }, // timeslice =  65536
{ 0xF8,0xC8,0x3E,0xFE,0xFE,0xDA,0x8A,0xAC }, // timeslice =   2^17
{ 0xF8,0xC8,0x3E,0xFE,0xFE,0xDA,0x8A,0xA8 }, // timeslice =   2^18
{ 0xF8,0xC8,0x3E,0xFE,0xFE,0xDA,0x8A,0xA0 }, // timeslice =   2^19
{ 0xF8,0xC8,0x3E,0xFE,0xFE,0xDA,0x8A,0x80 }, // timeslice =   2^20
{ 0xF8,0xC8,0x3E,0xFE,0xFE,0xDA,0x8A,0x00 }, // timeslice =   2^21
{ 0xF8,0xC8,0x3E,0xFE,0xFE,0xDA,0x88,0x00 }, // timeslice =   2^22
{ 0xF8,0xC8,0x3E,0xFE,0xFE,0xDA,0x80,0x00 }, // timeslice =   2^23
{ 0xF8,0xC8,0x3E,0xFE,0xFE,0xDA,0x00,0x00 }, // timeslice =   2^24
};

extern "C" int bryd_des (u8 *plain, u8 *cypher, u8 *iv, u8 *key, const u8 *bitmask);
extern "C" int p1bryd_des (u8 *plain, u8 *cypher, u8 *iv, u8 *key, const u8 *bitmask);
#if defined(CLIENT_SUPPORTS_SMP)
extern "C" int bbryd_des (u8 *plain, u8 *cypher, u8 *iv, u8 *key, const u8 *bitmask);
extern "C" int p2bryd_des (u8 *plain, u8 *cypher, u8 *iv, u8 *key, const u8 *bitmask);
#endif

static u8 key_found[8];
static bool key_is_found;
#if defined(CLIENT_SUPPORTS_SMP)
static u8 Bkey_found[8];
static bool Bkey_is_found;
#endif

// ------------------------------------------------------------------
// in case we find the key
extern "C" void bryd_key_found (u8 *bytekey);
void bryd_key_found (u8 *bytekey)
{
#ifdef DEBUG
  printf ("key found!\n");
#endif
  memcpy (key_found, bytekey, 8);
  key_is_found = true;
  return;
}

#if defined(CLIENT_SUPPORTS_SMP)
extern "C" void bbryd_key_found (u8 *bytekey);
void bbryd_key_found (u8 *bytekey)
{
#ifdef DEBUG
  printf ("key found!\n");
#endif
  memcpy (Bkey_found, bytekey, 8);
  Bkey_is_found = true;
  return;
}
#endif

// ------------------------------------------------------------------
// Called before keys are tested, and each time 2^16 (65536) keys are tested.
// (in fact, it depends on the bitmask used...)
extern "C" int bryd_continue (void);
int bryd_continue (void)
{
#ifdef DEBUG
  printf ("bryd_continue_called\n");
#endif
  return key_is_found ? 0 : 1;
}

#if defined(CLIENT_SUPPORTS_SMP)
extern "C" int bbryd_continue (void);
int bbryd_continue (void)
{
#ifdef DEBUG
  printf ("Bbryd_continue_called\n");
#endif
  return Bkey_is_found ? 0 : 1;
}
#endif

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 'timeslice-something' else
// note : timeslice will be rounded to the upper power of two
//        and can't be less than 256

// rc5unitwork.LO in lo:hi 24+32 incrementable format

u32 p1des_unit_func_p5( RC5UnitWork * rc5unitwork, u32 nbbits )
{
  const u8 *bitmask;
  u8 key[8];
  u8 plain[8];
  u8 cypher[8];
  u8 iv[8] = {0,0,0,0,0,0,0,0}; // fake IV, plaintext already xor'ed with it
  u32 i;

  // convert the starting key from incrementable format
  // to DES format
  u32 keyhi = rc5unitwork->L0.hi;
  u32 keylo = rc5unitwork->L0.lo;
  convert_key_from_inc_to_des (&keyhi, &keylo);

  // adjust bitmask
  bitmask = &(bitmasks[nbbits-8][0]);

  // convert key, plaintext and cyphertext to bryddes flavor
  u32 kk = keylo;
  u32 pp = rc5unitwork->plain.lo;
  u32 cc = rc5unitwork->cypher.lo;
  for (i=0; i<8; i++)
  {
    key[7-i] = (u8) (kk & 0xFF); kk >>= 8;
    plain[7-i] = (u8) (pp & 0xFF); pp >>= 8;
    cypher[7-i] = (u8) (cc & 0xFF); cc >>= 8;
    if (i == 3)
    {
      kk = keyhi;
      pp = rc5unitwork->plain.hi;
      cc = rc5unitwork->cypher.hi;
    }
  }
  // key[] is now in 64 bits, DES ordering format

#ifdef DEBUG
  printf (" plain  = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    plain[0],plain[1],plain[2],plain[3],plain[4],plain[5],plain[6],plain[7]);
  printf (" cypher = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    cypher[0],cypher[1],cypher[2],cypher[3],cypher[4],cypher[5],cypher[6],cypher[7]);
  printf ("key     = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7]);
  printf ("bitmask = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    bitmask[0],bitmask[1],bitmask[2],bitmask[3],bitmask[4],bitmask[5],bitmask[6],bitmask[7]);
#endif

  // launch bryddes
  key_is_found = false;
//  int result = bryd_des (plain, cypher, iv, key, bitmask);
  int result = bryd_des (plain, cypher, iv, key, bitmask);

  // have we found something ?
  if (result == 0 || key_is_found)
  {
  #ifdef DEBUG
      printf ("found = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
        key_found[0],key_found[1],key_found[2],key_found[3],
        key_found[4],key_found[5],key_found[6],key_found[7]);
  #endif

    // have we found the complementary key ?
    // we can test key_found[3] or key_found[4]
    // but no other bytes
    if ((u32)key_found[3] == (~keyhi & 0xFF))
    {
      // report it as beeing on the non-complementary key
      *(u32*)(&key_found[0]) = ~(*(u32*)(&key_found[0]));
      *(u32*)(&key_found[4]) = ~(*(u32*)(&key_found[4]));
    }

    // convert key from 64 bits DES ordering with parity
    // to incrementable format (to do arithmetic on it)
    keyhi =
        (key_found[0] << 24) |
        (key_found[1] << 16) |
        (key_found[2] <<  8) |
        (key_found[3]      );
    keylo =
        (key_found[4] << 24) |
        (key_found[5] << 16) |
        (key_found[6] <<  8) |
        (key_found[7]      );
    convert_key_from_des_to_inc (&keyhi, &keylo);

  #ifdef DEBUG
    printf ("found = %08X:%08X\n",keyhi, keylo);
  #endif
    u32 nbkeys = keylo - rc5unitwork->L0.lo;
    rc5unitwork->L0.lo = keylo;
    rc5unitwork->L0.hi = keyhi;

    return nbkeys;

  } else {
    rc5unitwork->L0.lo += 1 << nbbits;
    return 1 << nbbits;
  }
}

// ------------------------------------------------------------------

#if defined(CLIENT_SUPPORTS_SMP)
u32 p2des_unit_func_p5( RC5UnitWork * rc5unitwork, u32 nbbits )
{
  const u8 *bitmask;
  u8 key[8];
  u8 plain[8];
  u8 cypher[8];
  u8 iv[8] = {0,0,0,0,0,0,0,0}; // fake IV, plaintext already xor'ed with it
  u32 i;

  // convert the starting key from incrementable format
  // to DES format
  u32 keyhi = rc5unitwork->L0.hi;
  u32 keylo = rc5unitwork->L0.lo;
  convert_key_from_inc_to_des (&keyhi, &keylo);

  // adjust bitmask
  bitmask = &(bitmasks[nbbits-8][0]);

  // convert key, plaintext and cyphertext to bryddes flavor
  u32 kk = keylo;
  u32 pp = rc5unitwork->plain.lo;
  u32 cc = rc5unitwork->cypher.lo;
  for (i=0; i<8; i++)
  {
    key[7-i] = (u8) (kk & 0xFF); kk >>= 8;
    plain[7-i] = (u8) (pp & 0xFF); pp >>= 8;
    cypher[7-i] = (u8) (cc & 0xFF); cc >>= 8;
    if (i == 3)
    {
      kk = keyhi;
      pp = rc5unitwork->plain.hi;
      cc = rc5unitwork->cypher.hi;
    }
  }
  // key[] is now in 64 bits, DES ordering format

#ifdef DEBUG
  printf (" plain  = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    plain[0],plain[1],plain[2],plain[3],plain[4],plain[5],plain[6],plain[7]);
  printf (" cypher = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    cypher[0],cypher[1],cypher[2],cypher[3],cypher[4],cypher[5],cypher[6],cypher[7]);
  printf ("key     = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7]);
  printf ("bitmask = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    bitmask[0],bitmask[1],bitmask[2],bitmask[3],bitmask[4],bitmask[5],bitmask[6],bitmask[7]);
#endif

  // launch bryddes
  Bkey_is_found = false;
  int result = bbryd_des (plain, cypher, iv, key, bitmask);

  // have we found something ?
  if (result == 0 || Bkey_is_found)
  {

  #ifdef DEBUG
    printf ("found = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
      Bkey_found[0],Bkey_found[1],Bkey_found[2],Bkey_found[3],
      Bkey_found[4],Bkey_found[5],Bkey_found[6],Bkey_found[7]);
  #endif

    // have we found the complementary key ?
    // we can test key_found[3] or key_found[4]
    // but no other bytes
    if ((u32)Bkey_found[3] == (~keyhi & 0xFF))
    {
      // report it as beeing on the non-complementary key
      *(u32*)(&Bkey_found[0]) = ~(*(u32*)(&Bkey_found[0]));
      *(u32*)(&Bkey_found[4]) = ~(*(u32*)(&Bkey_found[4]));
    }

    // convert key from 64 bits DES ordering with parity
    // to incrementable format (to do arithmetic on it)
    keyhi =
      (Bkey_found[0] << 24) |
      (Bkey_found[1] << 16) |
      (Bkey_found[2] <<  8) |
      (Bkey_found[3]      );
    keylo =
      (Bkey_found[4] << 24) |
      (Bkey_found[5] << 16) |
      (Bkey_found[6] <<  8) |
      (Bkey_found[7]      );
    convert_key_from_des_to_inc (&keyhi, &keylo);

  #ifdef DEBUG
    printf ("found = %08X:%08X\n",keyhi, keylo);
  #endif
    u32 nbkeys = keylo - rc5unitwork->L0.lo;
    rc5unitwork->L0.lo = keylo;
    rc5unitwork->L0.hi = keyhi;

    return nbkeys;

  } else {
    rc5unitwork->L0.lo += 1 << nbbits;
    return 1 << nbbits;
  }
}
#endif

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 'timeslice-something' else
// note : timeslice will be rounded to the upper power of two
//        and can't be less than 256

// rc5unitwork.LO in lo:hi 24+32 incrementable format

u32 p1des_unit_func_pro( RC5UnitWork * rc5unitwork, u32 nbbits )
{
  const u8 *bitmask;
  u8 key[8];
  u8 plain[8];
  u8 cypher[8];
  u8 iv[8] = {0,0,0,0,0,0,0,0}; // fake IV, plaintext already xor'ed with it
  u32 i;

  // convert the starting key from incrementable format
  // to DES format
  u32 keyhi = rc5unitwork->L0.hi;
  u32 keylo = rc5unitwork->L0.lo;
  convert_key_from_inc_to_des (&keyhi, &keylo);

  // adjust bitmask
  bitmask = &(bitmasks[nbbits-8][0]);

  // convert key, plaintext and cyphertext to bryddes flavor
  u32 kk = keylo;
  u32 pp = rc5unitwork->plain.lo;
  u32 cc = rc5unitwork->cypher.lo;
  for (i=0; i<8; i++)
  {
    key[7-i] = (u8) (kk & 0xFF); kk >>= 8;
    plain[7-i] = (u8) (pp & 0xFF); pp >>= 8;
    cypher[7-i] = (u8) (cc & 0xFF); cc >>= 8;
    if (i == 3)
    {
      kk = keyhi;
      pp = rc5unitwork->plain.hi;
      cc = rc5unitwork->cypher.hi;
    }
  }
  // key[] is now in 64 bits, DES ordering format

#ifdef DEBUG
  printf (" plain  = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    plain[0],plain[1],plain[2],plain[3],plain[4],plain[5],plain[6],plain[7]);
  printf (" cypher = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    cypher[0],cypher[1],cypher[2],cypher[3],cypher[4],cypher[5],cypher[6],cypher[7]);
  printf ("key     = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7]);
  printf ("bitmask = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    bitmask[0],bitmask[1],bitmask[2],bitmask[3],bitmask[4],bitmask[5],bitmask[6],bitmask[7]);
#endif

  // launch bryddes
  key_is_found = false;
  int result = p1bryd_des (plain, cypher, iv, key, bitmask);

  // have we found something ?
  if (result == 0 || key_is_found)
  {
  #ifdef DEBUG
      printf ("found = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
        key_found[0],key_found[1],key_found[2],key_found[3],
        key_found[4],key_found[5],key_found[6],key_found[7]);
  #endif

    // have we found the complementary key ?
    // we can test key_found[3] or key_found[4]
    // but no other bytes
    if ((u32)key_found[3] == (~keyhi & 0xFF))
    {
      // report it as beeing on the non-complementary key
      *(u32*)(&key_found[0]) = ~(*(u32*)(&key_found[0]));
      *(u32*)(&key_found[4]) = ~(*(u32*)(&key_found[4]));
    }

    // convert key from 64 bits DES ordering with parity
    // to incrementable format (to do arithmetic on it)
    keyhi =
        (key_found[0] << 24) |
        (key_found[1] << 16) |
        (key_found[2] <<  8) |
        (key_found[3]      );
    keylo =
        (key_found[4] << 24) |
        (key_found[5] << 16) |
        (key_found[6] <<  8) |
        (key_found[7]      );
    convert_key_from_des_to_inc (&keyhi, &keylo);

  #ifdef DEBUG
    printf ("found = %08X:%08X\n",keyhi, keylo);
  #endif
    u32 nbkeys = keylo - rc5unitwork->L0.lo;
    rc5unitwork->L0.lo = keylo;
    rc5unitwork->L0.hi = keyhi;

    return nbkeys;

  } else {
    rc5unitwork->L0.lo += 1 << nbbits;
    return 1 << nbbits;
  }
}

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 'timeslice-something' else
// note : timeslice will be rounded to the upper power of two
//        and can't be less than 256

// rc5unitwork.LO in lo:hi 24+32 incrementable format

#if defined(CLIENT_SUPPORTS_SMP)
u32 p2des_unit_func_pro( RC5UnitWork * rc5unitwork, u32 nbbits )
{
  const u8 *bitmask;
  u8 key[8];
  u8 plain[8];
  u8 cypher[8];
  u8 iv[8] = {0,0,0,0,0,0,0,0}; // fake IV, plaintext already xor'ed with it
  u32 i;

  // convert the starting key from incrementable format
  // to DES format
  u32 keyhi = rc5unitwork->L0.hi;
  u32 keylo = rc5unitwork->L0.lo;
  convert_key_from_inc_to_des (&keyhi, &keylo);

  // adjust bitmask
  bitmask = &(bitmasks[nbbits-8][0]);

  // convert key, plaintext and cyphertext to bryddes flavor
  u32 kk = keylo;
  u32 pp = rc5unitwork->plain.lo;
  u32 cc = rc5unitwork->cypher.lo;
  for (i=0; i<8; i++)
  {
    key[7-i] = (u8) (kk & 0xFF); kk >>= 8;
    plain[7-i] = (u8) (pp & 0xFF); pp >>= 8;
    cypher[7-i] = (u8) (cc & 0xFF); cc >>= 8;
    if (i == 3)
    {
      kk = keyhi;
      pp = rc5unitwork->plain.hi;
      cc = rc5unitwork->cypher.hi;
    }
  }
  // key[] is now in 64 bits, DES ordering format

#ifdef DEBUG
  printf (" plain  = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    plain[0],plain[1],plain[2],plain[3],plain[4],plain[5],plain[6],plain[7]);
  printf (" cypher = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    cypher[0],cypher[1],cypher[2],cypher[3],cypher[4],cypher[5],cypher[6],cypher[7]);
  printf ("key     = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7]);
  printf ("bitmask = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
    bitmask[0],bitmask[1],bitmask[2],bitmask[3],bitmask[4],bitmask[5],bitmask[6],bitmask[7]);
#endif

  // launch bryddes
  Bkey_is_found = false;
  int result = p2bryd_des (plain, cypher, iv, key, bitmask);

  // have we found something ?
  if (result == 0 || Bkey_is_found)
  {

  #ifdef DEBUG
    printf ("found = %02X%02X%02X%02X:%02X%02X%02X%02X\n",
      Bkey_found[0],Bkey_found[1],Bkey_found[2],Bkey_found[3],
      Bkey_found[4],Bkey_found[5],Bkey_found[6],Bkey_found[7]);
  #endif

    // have we found the complementary key ?
    // we can test key_found[3] or key_found[4]
    // but no other bytes
    if ((u32)Bkey_found[3] == (~keyhi & 0xFF))
    {
      // report it as beeing on the non-complementary key
      *(u32*)(&Bkey_found[0]) = ~(*(u32*)(&Bkey_found[0]));
      *(u32*)(&Bkey_found[4]) = ~(*(u32*)(&Bkey_found[4]));
    }

    // convert key from 64 bits DES ordering with parity
    // to incrementable format (to do arithmetic on it)
    keyhi =
      (Bkey_found[0] << 24) |
      (Bkey_found[1] << 16) |
      (Bkey_found[2] <<  8) |
      (Bkey_found[3]      );
    keylo =
      (Bkey_found[4] << 24) |
      (Bkey_found[5] << 16) |
      (Bkey_found[6] <<  8) |
      (Bkey_found[7]      );
    convert_key_from_des_to_inc (&keyhi, &keylo);

  #ifdef DEBUG
    printf ("found = %08X:%08X\n",keyhi, keylo);
  #endif
    u32 nbkeys = keylo - rc5unitwork->L0.lo;
    rc5unitwork->L0.lo = keylo;
    rc5unitwork->L0.hi = keyhi;

    return nbkeys;

  } else {
    rc5unitwork->L0.lo += 1 << nbbits;
    return 1 << nbbits;
  }
}
#endif
