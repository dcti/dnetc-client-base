// Copyright distributed.net 1997-2008 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// ------------------------------------------------------------------
// The following is adapted from source in
//   "Inner Loops -- A sourcebook for fast 32-bit software development"
//                                                      by Rick Booth.
// Get a random 32bit number with Random() 
// If called continuously, the sequence IL_MixRandom() generates 
// wouldn't start to repeat for about 37 trillion years.
//
//                                - Added by Tim Charron 12.15.1997
// -----------------------------------------------------------------

const char *random_cpp(void) {
return "@(#)$Id: random.cpp,v 1.9 2008/04/11 04:49:15 jlawson Exp $"; }

#include "cputypes.h" /* u32 */
#include <time.h>     /* time() */

#if (CLIENT_OS == OS_DEC_UNIX)
#include "baseincs.h"
#endif

#define IL_RMULT 1103515245L
static u32 IL_StandardRandom_seed;

struct IL_MixRandom_struct
{
  s32 trseed32[32*2];
  s32 trseed31[31*2];
  s32 trseed29[29*2];
  s32 ptr32;
  s32 ptr31;
  s32 ptr29;
};

static IL_MixRandom_struct IL_MixRandom_seeds;

//! Internal method used to validate the generator seed.
/*!
 * \param source
 * \param count
 * \return Returns 0 on failure, 1 on success.
 */
static s32 IL_AreBitColumnsOK(u32 *source, s32 count)
{
  int n;
  u32 l, b;

  for (n = 0, l = 0; n < count; n++)
      l |= source[n];
  if (l != (u32)-1L)
      return 0;                               // a column of zeroes fails
  b = 1;
  do
  {
    l = -1 ^ b;
    for (n = 0; (n < count) && l; n++)
    {
      if (source[n] & b)
        l &=  source[n];
      else
        l &= ~source[n];
    }
    if (l)
        return 0;                           // equal columns fail
    b <<= 1;
  } while (b);
  return 1;                                   // success
}

//! Internal method used to seed the generator.
/*!
 * \param rand_fn Pointer to function used for seeding.
 */
static void IL_MixRandomSeed(register s32 (*rand_fn)())
{
  int n;
  struct IL_MixRandom_struct *p;

  p = (struct IL_MixRandom_struct *)&IL_MixRandom_seeds;
  p->ptr32 = p->ptr31 = p->ptr29 = 0;
  do {
    for (n = 0; n < 32; n++)
      p->trseed32[n] = p->trseed32[n + 32] = (u32) rand_fn();
  } while (!IL_AreBitColumnsOK((u32 *)p->trseed32, 32));
  do {
    for (n = 0; n < 31; n++)
      p->trseed31[n] = p->trseed31[n + 31] = (u32) rand_fn();
  } while (!IL_AreBitColumnsOK((u32 *)p->trseed31, 31));
  do {
    for (n = 0; n < 29; n++)
      p->trseed29[n] = p->trseed29[n + 29] = (u32) rand_fn();
  } while (!IL_AreBitColumnsOK((u32 *)p->trseed29, 29));
}

//! Internal method that returns a generated random value.
static u32 IL_MixRandom()
{
  s32 sum, temp, i32, i31, i29;
  s32 *iptr;
  struct IL_MixRandom_struct *p;

  p    = (struct IL_MixRandom_struct *)&IL_MixRandom_seeds;
  i32      = ((p->ptr32 >> 2) + 1) &  0x1f;
  if ((i31 =  (p->ptr31 >> 2) + 1) ==   31)
      i31  = 0;
  if ((i29 =  (p->ptr29 >> 2) + 1) ==   29)
      i29  = 0;
  p->ptr32 = i32 << 2;
  p->ptr31 = i31 << 2;
  p->ptr29 = i29 << 2;

  iptr     = &p->trseed32[i32];
  temp     = iptr[31] ^ iptr[6] ^ iptr[4] ^ iptr[2] ^ iptr[1] ^ iptr[0];
  iptr[0]  = iptr[32] = temp;
  sum      = temp;

  iptr     = &p->trseed31[i31];
  temp     = iptr[30] ^ iptr[2];
  iptr[0]  = iptr[31] = temp;
  sum     ^= temp;

  iptr     = &p->trseed29[i29];
  temp     = iptr[28] ^ iptr[1];
  iptr[0]  = iptr[29] = temp;
  sum     ^= temp;

  return (u32) sum;
}

//! Internal callback method used for seeding the generator.
/*!
 * This is the usual random number generator most packages use,
 * except that it's been modified to return 32 bits instead of 16.
 */
static s32 IL_StandardRandom()
{
  u32 lo, hi, ll, lh, hh, hl;

  lo = IL_StandardRandom_seed & 0xffffL;
  hi = IL_StandardRandom_seed >> 16;
  IL_StandardRandom_seed = IL_StandardRandom_seed * IL_RMULT + 12345;
  ll = lo * (IL_RMULT  & 0xffffL);
  lh = lo * (IL_RMULT >> 16    );
  hl = hi * (IL_RMULT  & 0xffffL);
  hh = hi * (IL_RMULT >> 16    );
  return ((ll + 12345L) >> 16) + lh + hl + (hh << 16);
}

//! Initialize the random number generator without entropy.
/*!
 * The random number generator uses a single static seed, and may only
 * be seeded once.  If it has already been seeded, then this method
 * does nothing.
 *
 * \return Does not return a value.
 */
void InitRandom(void)
{
  static int random_initialized = 0;
  int i;
  if (random_initialized)
    return;
  random_initialized = 1;

  IL_StandardRandom_seed = (u32)time((time_t *)0);  // 32 bit seed value

  for ( i = 0; i < 64; i++ )
  {
    IL_MixRandom_seeds.trseed32[i] = -1;
    if (i<62) IL_MixRandom_seeds.trseed31[i] = -1;
    if (i<58) IL_MixRandom_seeds.trseed29[i] = -1;
  }
  IL_MixRandom_seeds.ptr32 = 0;
  IL_MixRandom_seeds.ptr31 = 0;
  IL_MixRandom_seeds.ptr29 = 0;

  IL_MixRandomSeed( IL_StandardRandom );
}


//! Initialize the random number generator with entropy.
/*!
 * The random number generator uses a single static seed, and may only
 * be seeded once.  If it has already been seeded, then this method
 * does nothing.
 *
 * \param p Pointer to a buffer used for additional entropy.  The buffer
 *     must be 4 bytes in size.  If NULL, then no additional entropy
 *     will be used.
 * \return Does not return a value.
 */
void InitRandom2(const char *p)
{
  static int random_2_initialized = 0;
  int i;

  InitRandom(); /* does nothing if already seeded */
  if (random_2_initialized || !p)
    return;
  random_2_initialized = 1;
    
  IL_StandardRandom_seed ^= ((p[0]<<24) + (p[1]<<16) + (p[2]<<8) + p[3]);
  for ( i=0 ; i< 64 ; i++ )
  {
    IL_MixRandom_seeds.trseed32[i] = -1;
    if (i<62) IL_MixRandom_seeds.trseed31[i] = -1;
    if (i<58) IL_MixRandom_seeds.trseed29[i] = -1;
  }
  IL_MixRandom_seeds.ptr32 = 0;
  IL_MixRandom_seeds.ptr31 = 0;
  IL_MixRandom_seeds.ptr29 = 0;

  IL_MixRandomSeed( IL_StandardRandom );
}

//! Generate 32-bits of random data.
/*!
 * If the random number generator has not already been seeded, then it
 * will be automatically seeded.
 *
 * \param u32data Pointer to buffer that will contribute more entropy.
 *       If NULL, then no additonal entropy will be mixed in.
 * \param u32count Size of the buffer, measured in 32-bit words.
 * \return Returns the generated random value.
 */
u32 Random( const u32 * u32data, unsigned int u32count )
{
  register u32 tmp;
  InitRandom2( (const char *)u32data ); /* does nothing if already seeded */
  
  tmp = IL_MixRandom();
  if (u32data && u32count) 
  {
    register unsigned int i;
    for( i = 0 ; i < u32count ; i++ ) 
      tmp ^= u32data[i];
  }
  return( tmp );
}


