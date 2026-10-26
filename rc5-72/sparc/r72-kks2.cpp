/* 
** Copyright distributed.net 1997-2003 - All Rights Reserved
** For use in distributed.net projects only.
** Any other distribution or use of this source violates copyright.
**
** 2-pipe core optimized for Sparc processors written by
** Didier Levet (kakace@wanadoo.fr)
**
** By modifying the assembly macros below, it can be adapted to other platforms
**
** Compiled to assembly with gcc and hand optimized further
**
** version 0.2 (03/29/2003) : Rescheduled instructions for better dispatching
**
** Inspired in part by the RC5-64 sparc core by Lawrence Butcher
** (lbutcher@eng.sun.com), Remi Guyomarch (rguyom@mail.dotcom.fr), et. al.
**
** Sparc's architecture :
**   - 2 independant integer units.
**   - 1 load/store unit.
**   - Can dispatch 3 instructions per clock.
**
** Bottlenecks :
**   - 2 shifts cannot be dispatched in the same cycle.
**   - A shift must be the first instruction to dispatch
**     from an integer instruction pair.
*/

#include "ccoreio.h"
#include "rotate.h"

#define P 0xB7E15163
#define Q 0x9E3779B9

#ifdef __cplusplus
extern "C" s32 CDECL rc5_72_unit_func_KKS_2 ( RC5_72UnitWork *, u32 *, void * );
#endif


/*
** Inline assembly macros.
** Note that PowerPC CPUs use the lower 6 bits as shift count
** when using slw/srw instructions, so the shift count shall
** be masked to [0; 31] values.
*/

#if defined(__GNUC__)

  #define ASM_ADD(res, op1, op2) \
    __asm__ volatile ("add %1, %2, %0" : "=r" (res) : "r" (op1), "r" (op2));
  #define ASM_SUB(res, wsize, op) \
    __asm__ volatile ("sub %1, %2, %0" : "=r" (res) : "rc" (wsize), "rc" (op));
  #define ASM_OR(res, op1, op2) \
    __asm__ volatile ("or %1, %2, %0" : "=r" (res) : "r" (op1), "r" (op2));
  #define ASM_XOR(res, op1, op2) \
    __asm__ volatile ("xor %1, %2, %0" : "=r" (res) : "r" (op1), "r" (op2));
  #define ASM_SL(res, op, dist) \
    __asm__ volatile ("sll %1, %2, %0" : "=r" (res) : "r" (op), "rc" (dist));
  #define ASM_SR(res, op, dist) \
    __asm__ volatile ("srl %1, %2, %0" : "=r" (res) : "r" (op), "rc" (dist));
  #define ASM_SL3(res, op) \
    __asm__ volatile ("sll %1, 3, %0" : "=r" (res) : "r" (op));
  #define ASM_SR29(res, op) \
    __asm__ volatile ("srl %1, 29, %0" : "=r" (res) : "r" (op));

#else

  #define ASM_ADD(res, op1, op2) (res) = (op1) + (op2);
  #define ASM_SUB(res, wsize, op) (res) = (wsize) - (op);
  #define ASM_OR(res, op1, op2) (res) = (op1) | (op2);
  #define ASM_XOR(res, op1, op2) (res) = (op1) ^ (op2);
  #define ASM_SL(res, op, dist) (res) = (op) << (dist & 31);
  #define ASM_SR(res, op, dist) (res) = (op) >> (dist & 31);
  #define ASM_SL3(res, op) (res) = (op) << 3u;
  #define ASM_SR29(res, op) (res) = (op) >> 29u;

#endif


/*
** MIX_1 macro :
** PIPE A                         PIPE B
** Ta = SaA + Ba                  Tb = SbA + Bb
** Sa[store] = SaA                Sb[store] = SbA
** SKa = Sa[load]                 SKb = SKa
** La += Ta                       Lb += Tb
** Ba = La = ROTL(La, Ta)         Bb = Lb = ROTL(Lb, Tb)
** Sa = SKa + SaA                 Sb = SKb + SbA
** Sa += Ba                       Sb += Bb
** SaA = Sa = ROTL3(Sa)           SbA = Sb = ROTL3(Sb)
*/

#define MIX_1(load, store, La, Lb, Ba, Bb) \
  /* ASM_ADD(Ta, SaA, Ba) */ \
  ASM_OR(SbA, SbA, TSb) \
  ASM_ADD(TSa, La, Ta) \
  Sa[store] = SaA; \
  \
  ASM_ADD(Tb, SbA, Bb) \
  ASM_SUB(SCa, word_size, Ta) \
  Sb[store] = SbA; \
  \
  ASM_SL(La, TSa, Ta) \
  ASM_ADD(TSb, Lb, Tb) \
  SKa = Sa[load]; \
  \
  ASM_SR(TSa, TSa, SCa) \
  ASM_SUB(SCb, word_size, Tb) \
  \
  ASM_SL(Lb, TSb, Tb) \
  ASM_OR(La, La, TSa) \
  \
  ASM_SR(TSb, TSb, SCb) \
  ASM_ADD(SaA, SaA, SKa) \
  \
  ASM_OR(Lb, Lb, TSb) \
  ASM_ADD(SaA, SaA, La) \
  \
  ASM_SL3(TSa, SaA) \
  ASM_ADD(SbA, SbA, SKa) \
  \
  ASM_SR29(SaA, SaA) \
  ASM_ADD(SbA, SbA, Lb) \
  \
  ASM_SL3(TSb, SbA) \
  ASM_OR(SaA, SaA, TSa) \
  \
  ASM_SR29(SbA, SbA) \
  /* ASM_OR(SbA, SbA, TSb) */ \
  ASM_ADD(Ta, SaA, La)


/*
** MIX_2 macro :
** PIPE A                         PIPE B
** Ta = SaA + Ba                  Tb = SbA + Bb
** Sa[store] = SaA                Sb[store] = SbA
** SKa = Sa[load]                 SKb = Sb[load]
** La += Ta                       Lb += Tb
** Ba = La = ROTL(La, Ta)         Bb = Lb = ROTL(Lb, Tb)
** Sa = SKa + SaA                 Sb = SKb + SbA
** Sa += Ba                       Sb += Bb
** SaA = Sa = ROTL3(Sa)           SbA = Sb = ROTL3(Sb)
*/

#define MIX_2(load, store, La, Lb, Ba, Bb) \
  /* ASM_ADD(Ta, SaA, Ba) */ \
  ASM_OR(SbA, SbA, TSb) \
  ASM_ADD(TSa, La, Ta) \
  Sa[store] = SaA; \
  \
  ASM_ADD(Tb, SbA, Bb) \
  ASM_SUB(SCa, word_size, Ta) \
  Sb[store] = SbA; \
  \
  ASM_SL(La, TSa, Ta) \
  ASM_ADD(TSb, Lb, Tb) \
  SKa = Sa[load]; \
  \
  ASM_SR(TSa, TSa, SCa) \
  ASM_SUB(SCb, word_size, Tb) \
  SKb = Sb[load]; \
  \
  ASM_SL(Lb, TSb, Tb) \
  ASM_OR(La, La, TSa) \
  \
  ASM_SR(TSb, TSb, SCb) \
  ASM_ADD(SaA, SaA, SKa) \
  \
  ASM_OR(Lb, Lb, TSb) \
  ASM_ADD(SaA, SaA, La) \
  \
  ASM_SL3(TSa, SaA) \
  ASM_ADD(SbA, SbA, SKb) \
  \
  ASM_SR29(SaA, SaA) \
  ASM_ADD(SbA, SbA, Lb) \
  \
  ASM_SL3(TSb, SbA) \
  ASM_OR(SaA, SaA, TSa) \
  \
  ASM_SR29(SbA, SbA) \
  /* ASM_OR(SbA, SbA, TSb) */ \
  ASM_ADD(Ta, SaA, La)


/*
** ROUND_1 macro :
** PIPE A                         PIPE B
** SKa = Sa[load]                 SKb = Sb[load]
** Ta = SaA + Ba                  Tb = SbA + Bb
** La += Ta                       Lb += Tb
** Ba = La = ROTL(La, Ta)         Bb = Lb = ROTL(Lb, Tb)
** Sa = SKa + SaA                 Sb = SKb + SbA
** Sa += Ba                       Sb += Bb
** SaA = Sa = ROTL3(Sa)           SbA = Sb = ROTL3(Sb)
** RaA ^= RaB                     RbA ^= RbB
** RaA = ROTL(RaA, RaB)           RbA = ROTL(RbA, RbB)
** RaA += SaA                     RbA += SbA
*/

#define ROUND_1(load, La, Lb, Ba, Bb) \
  ASM_ADD(Ta, SaA, Ba) \
  ASM_ADD(Tb, SbA, Bb) \
  SKa = Sa[load]; \
  \
  ASM_ADD(TSa, La, Ta) \
  ASM_SUB(SCa, word_size, Ta) \
  SKb = Sb[load]; \
  \
  ASM_SL(La, TSa, Ta) \
  ASM_ADD(TSb, Lb, Tb) \
  \
  ASM_SR(TSa, TSa, SCa) \
  ASM_SUB(SCb, word_size, Tb) \
  \
  ASM_SL(Lb, TSb, Tb) \
  ASM_OR(La, La, TSa) \
  \
  ASM_SR(TSb, TSb, SCb) \
  ASM_ADD(SaA, SaA, SKa) \
  \
  ASM_OR(Lb, Lb, TSb) \
  ASM_ADD(SaA, SaA, La) \
  \
  ASM_SL3(TSa, SaA) \
  ASM_ADD(SbA, SbA, SKb) \
  \
  ASM_SR29(SaA, SaA) \
  ASM_ADD(SbA, SbA, Lb) \
  \
  ASM_SL3(TSb, SbA) \
  ASM_XOR(RaA, RaA, RaB) \
  \
  ASM_SR29(SbA, SbA) \
  ASM_XOR(RbA, RbA, RbB) \
  \
  ASM_SL(Ta, RaA, RaB) \
  ASM_SUB(SCa, word_size, RaB) \
  \
  ASM_SL(Tb, RbA, RbB) \
  ASM_SUB(SCb, word_size, RbB) \
  \
  ASM_SR(RaA, RaA, SCa) \
  ASM_OR(SaA, SaA, TSa) \
  \
  ASM_SR(RbA, RbA, SCb) \
  ASM_OR(SbA, SbA, TSb) \
  \
  ASM_OR(RaA, RaA, Ta) \
  ASM_OR(RbA, RbA, Tb) \
  \
  ASM_ADD(RaA, RaA, SaA) \
  ASM_ADD(RbA, RbA, SbA)


/*
** ROUND_2 macro :
** PIPE A                         PIPE B
** SKa = Sa[load]                 SKb = Sb[load]
** Ta = SaA + Ba                  Tb = SbA + Bb
** La += Ta                       Lb += Tb
** Ba = La = ROTL(La, Ta)         Bb = Lb = ROTL(Lb, Tb)
** Sa = SKa + SaA                 Sb = SKb + SbA
** Sa += Ba                       Sb += Bb
** SaA = Sa = ROTL3(Sa)           SbA = Sb = ROTL3(Sb)
** RaB ^= RaA                     RbB ^= RbA
** RaB = ROTL(RaB, RaA)           RbB = ROTL(RbB, RbA)
** RaB += SaA                     RbB += SbA
*/

#define ROUND_2(load, La, Lb, Ba, Bb) \
  ASM_ADD(Ta, SaA, Ba) \
  ASM_ADD(Tb, SbA, Bb) \
  SKa = Sa[load]; \
  \
  ASM_ADD(TSa, La, Ta) \
  ASM_SUB(SCa, word_size, Ta) \
  SKb = Sb[load]; \
  \
  ASM_SL(La, TSa, Ta) \
  ASM_ADD(TSb, Lb, Tb) \
  \
  ASM_SR(TSa, TSa, SCa) \
  ASM_SUB(SCb, word_size, Tb) \
  \
  ASM_SL(Lb, TSb, Tb) \
  ASM_OR(La, La, TSa) \
  \
  ASM_SR(TSb, TSb, SCb) \
  ASM_ADD(SaA, SaA, SKa) \
  \
  ASM_OR(Lb, Lb, TSb) \
  ASM_ADD(SaA, SaA, La) \
  \
  ASM_SL3(TSa, SaA) \
  ASM_ADD(SbA, SbA, SKb) \
  \
  ASM_SR29(SaA, SaA) \
  ASM_ADD(SbA, SbA, Lb) \
  \
  ASM_SL3(TSb, SbA) \
  ASM_XOR(RaB, RaB, RaA) \
  \
  ASM_SR29(SbA, SbA) \
  ASM_XOR(RbB, RbB, RbA) \
  \
  ASM_SL(Ta, RaB, RaA) \
  ASM_SUB(SCa, word_size, RaA) \
  \
  ASM_SL(Tb, RbB, RbA) \
  ASM_SUB(SCb, word_size, RbA) \
  \
  ASM_SR(RaB, RaB, SCa) \
  ASM_OR(SaA, SaA, TSa) \
  \
  ASM_SR(RbB, RbB, SCb) \
  ASM_OR(SbA, SbA, TSb) \
  \
  ASM_OR(RaB, RaB, Ta) \
  ASM_OR(RbB, RbB, Tb) \
  \
  ASM_ADD(RaB, RaB, SaA) \
  ASM_ADD(RbB, RbB, SbA)



s32 CDECL rc5_72_unit_func_KKS_2 (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  u32 Sa[52], Sb[52];
  u32 kiter;
  u32 SKa, SKb, cached_s0, cached_s1, cached_s2; 
  u32 cached_l0, cached_l1;
  register u32 La0, La1, La2, Ta, TSa, SCa;
  register u32 Lb0, Lb1, Lb2, Tb, TSb, SCb;
  register u32 SaA, SbA;
  register u32 RaA, RaB, RbA, RbB;

  u32 key_lo      = rc5_72unitwork->L0.lo;
  u32 key_mid     = rc5_72unitwork->L0.mid;
  u32 key_hi      = rc5_72unitwork->L0.hi;
  u32 plain_lo    = rc5_72unitwork->plain.lo;
  u32 plain_hi    = rc5_72unitwork->plain.hi;
  u32 cypher_lo   = rc5_72unitwork->cypher.lo;
  u32 cypher_hi   = rc5_72unitwork->cypher.hi;
  u32 check_count = rc5_72unitwork->check.count;
  u32 check_lo    = rc5_72unitwork->check.lo;
  u32 check_mid   = rc5_72unitwork->check.mid;
  u32 check_hi    = rc5_72unitwork->check.hi;

  u32 word_size = 32;

  /*
  ** Initialize the expanded key table.
  */
	
  for (kiter = 1; kiter < 26; kiter++)
    Sa[kiter] = Sb[kiter] = P + kiter * Q;
 
  kiter = *iterations / 2;

  /*
  ** Compute the part which is independant from L0.hi
  ** Also initialize S[0], S[1] and S[2].
  */

  cached_s0 = ROTL3(P);
  cached_l0 = ROTL(key_lo + cached_s0, cached_s0);
  cached_s1 = ROTL3(Sa[1] + cached_s0 + cached_l0);
	
  Ta = cached_s1 + cached_l0;
  cached_l1 = ROTL(key_mid + Ta, Ta);
  cached_s2 = ROTL3(Sa[2] + cached_s1 + cached_l1);

  do {
    Sa[26] = Sb[26] = cached_s0;
    Sa[27] = Sb[27] = cached_s1;
    SaA = SbA = cached_s2;
    La0 = Lb0 = cached_l0;
    La1 = Lb1 = cached_l1;
    La2 = key_hi;
    Lb2 = La2 + 1;

    ASM_ADD(Ta, SaA, La1);
    TSb = 0;  /* Makes the ASM_OR(SbA, SbA, TSb) operation */
              /* a no-op in the first MIX_1 expansion */

    MIX_1( 3, 28, La2, Lb2, La1, Lb1);
    MIX_1( 4, 29, La0, Lb0, La2, Lb2);
    MIX_1( 5, 30, La1, Lb1, La0, Lb0);
    MIX_1( 6, 31, La2, Lb2, La1, Lb1);
    MIX_1( 7, 32, La0, Lb0, La2, Lb2);
    MIX_1( 8, 33, La1, Lb1, La0, Lb0);
    MIX_1( 9, 34, La2, Lb2, La1, Lb1);
    MIX_1(10, 35, La0, Lb0, La2, Lb2);
    MIX_1(11, 36, La1, Lb1, La0, Lb0);
    MIX_1(12, 37, La2, Lb2, La1, Lb1);
    MIX_1(13, 38, La0, Lb0, La2, Lb2);
    MIX_1(14, 39, La1, Lb1, La0, Lb0);
    MIX_1(15, 40, La2, Lb2, La1, Lb1);
    MIX_1(16, 41, La0, Lb0, La2, Lb2);
    MIX_1(17, 42, La1, Lb1, La0, Lb0);
    MIX_1(18, 43, La2, Lb2, La1, Lb1);
    MIX_1(19, 44, La0, Lb0, La2, Lb2);
    MIX_1(20, 45, La1, Lb1, La0, Lb0);
    MIX_1(21, 46, La2, Lb2, La1, Lb1);
    MIX_1(22, 47, La0, Lb0, La2, Lb2);
    MIX_1(23, 48, La1, Lb1, La0, Lb0);
    MIX_1(24, 49, La2, Lb2, La1, Lb1);
    MIX_1(25, 50, La0, Lb0, La2, Lb2);

    MIX_2(26, 51, La1, Lb1, La0, Lb0);
    MIX_2(27, 26, La2, Lb2, La1, Lb1);
    MIX_2(28, 27, La0, Lb0, La2, Lb2);
    MIX_2(29, 28, La1, Lb1, La0, Lb0);
    MIX_2(30, 29, La2, Lb2, La1, Lb1);
    MIX_2(31, 30, La0, Lb0, La2, Lb2);
    MIX_2(32, 31, La1, Lb1, La0, Lb0);
    MIX_2(33, 32, La2, Lb2, La1, Lb1);
    MIX_2(34, 33, La0, Lb0, La2, Lb2);
    MIX_2(35, 34, La1, Lb1, La0, Lb0);
    MIX_2(36, 35, La2, Lb2, La1, Lb1);
    MIX_2(37, 36, La0, Lb0, La2, Lb2);
    MIX_2(38, 37, La1, Lb1, La0, Lb0);
    MIX_2(39, 38, La2, Lb2, La1, Lb1);
    MIX_2(40, 39, La0, Lb0, La2, Lb2);
    MIX_2(41, 40, La1, Lb1, La0, Lb0);
    MIX_2(42, 41, La2, Lb2, La1, Lb1);
    MIX_2(43, 42, La0, Lb0, La2, Lb2);
    MIX_2(44, 43, La1, Lb1, La0, Lb0);
    MIX_2(45, 44, La2, Lb2, La1, Lb1);
    MIX_2(46, 45, La0, Lb0, La2, Lb2);
    MIX_2(47, 46, La1, Lb1, La0, Lb0);
    MIX_2(48, 47, La2, Lb2, La1, Lb1);
    MIX_2(49, 48, La0, Lb0, La2, Lb2);
    MIX_2(50, 49, La1, Lb1, La0, Lb0);
    MIX_2(51, 50, La2, Lb2, La1, Lb1);

    /*
    ** MIX_2( 0, La0, Lb0, La2, Lb2);
    ** MIX_2( 1, La1, Lb1, La0, Lb0);
    */

    ASM_OR(SbA, SbA, TSb);  /* Stage #52 */
    ASM_ADD(TSa, La0, Ta);
    Sa[51] = SaA;

    ASM_ADD(Tb, SbA, Lb2);
    ASM_SUB(SCa, word_size, Ta);
    Sb[51] = SbA;

    ASM_SL(La0, TSa, Ta);
    ASM_ADD(TSb, Lb0, Tb);
    SKa = Sa[26];

    ASM_SR(TSa, TSa, SCa);
    ASM_SUB(SCb, word_size, Tb);
    SKb = Sb[26];

    ASM_SL(Lb0, TSb, Tb);
    ASM_OR(La0, La0, TSa);

    ASM_SR(TSb, TSb, SCb);
    ASM_ADD(SaA, SaA, SKa);

    ASM_OR(Lb0, Lb0, TSb);
    ASM_ADD(SaA, SaA, La0);

    ASM_SL3(TSa, SaA);
    ASM_ADD(SbA, SbA, SKb);

    ASM_SR29(SaA, SaA);
    ASM_ADD(SbA, SbA, Lb0);

    ASM_SL3(TSb, SbA);
    ASM_OR(SaA, SaA, TSa);

    ASM_SR29(SbA, SbA);
    ASM_ADD(Ta, SaA, La0);
		

    ASM_OR(SbA, SbA, TSb);  /* Stage #53 */
    ASM_ADD(TSa, La1, Ta);
    RaA = plain_lo;

    ASM_ADD(Tb, SbA, Lb0);
    ASM_SUB(SCa, word_size, Ta);
    RbA = plain_lo;

    ASM_SL(La1, TSa, Ta);
    ASM_ADD(TSb, Lb1, Tb);
    SKa = Sa[27];

    ASM_SR(TSa, TSa, SCa);
    ASM_SUB(SCb, word_size, Tb);
    SKb = Sb[27];

    ASM_SL(Lb1, TSb, Tb);
    ASM_OR(La1, La1, TSa);
    RaB = plain_hi;

    RaA += SaA;
    RbA += SbA;
    RbB = plain_hi;

    ASM_SR(TSb, TSb, SCb);
    ASM_ADD(SaA, SaA, SKa);

    ASM_OR(Lb1, Lb1, TSb);
    ASM_ADD(SaA, SaA, La1);

    ASM_SL3(TSa, SaA);
    ASM_ADD(SbA, SbA, SKb);

    ASM_SR29(SaA, SaA);
    ASM_ADD(SbA, SbA, Lb1);

    ASM_SL3(TSb, SbA);
    ASM_OR(SaA, SaA, TSa);

    ASM_SR29(SbA, SbA);
    RaB += SaA;

    ASM_OR(SbA, SbA, TSb);
    RbB += SbA;

    ROUND_1(28, La2, Lb2, La1, Lb1);
    ROUND_2(29, La0, Lb0, La2, Lb2);
    ROUND_1(30, La1, Lb1, La0, Lb0);
    ROUND_2(31, La2, Lb2, La1, Lb1);
    ROUND_1(32, La0, Lb0, La2, Lb2);
    ROUND_2(33, La1, Lb1, La0, Lb0);
    ROUND_1(34, La2, Lb2, La1, Lb1);
    ROUND_2(35, La0, Lb0, La2, Lb2);
    ROUND_1(36, La1, Lb1, La0, Lb0);
    ROUND_2(37, La2, Lb2, La1, Lb1);
    ROUND_1(38, La0, Lb0, La2, Lb2);
    ROUND_2(39, La1, Lb1, La0, Lb0);
    ROUND_1(40, La2, Lb2, La1, Lb1);
    ROUND_2(41, La0, Lb0, La2, Lb2);
    ROUND_1(42, La1, Lb1, La0, Lb0);
    ROUND_2(43, La2, Lb2, La1, Lb1);
    ROUND_1(44, La0, Lb0, La2, Lb2);
    ROUND_2(45, La1, Lb1, La0, Lb0);
    ROUND_1(46, La2, Lb2, La1, Lb1);
    ROUND_2(47, La0, Lb0, La2, Lb2);
    ROUND_1(48, La1, Lb1, La0, Lb0);
    ROUND_2(49, La2, Lb2, La1, Lb1);
    ROUND_1(50, La0, Lb0, La2, Lb2);

    if (RaA == cypher_lo || RbA == cypher_lo) {
      ROUND_2(51, La1, Lb1, La0, Lb0);
    }

    if (RaA == cypher_lo) {
      ++check_count;
      check_hi  = key_hi;
      check_mid = key_mid;
      check_lo  = key_lo;

      if (RaB == cypher_hi) {
        *iterations -= kiter * 2;

        rc5_72unitwork->L0.lo           = key_lo;
        rc5_72unitwork->L0.mid          = key_mid;
        rc5_72unitwork->L0.hi           = key_hi;
        rc5_72unitwork->check.count     = check_count;
        rc5_72unitwork->check.lo        = check_lo;
        rc5_72unitwork->check.mid       = check_mid;
        rc5_72unitwork->check.hi        = check_hi;

        return RESULT_FOUND;
      }
    }

    if (RbA == cypher_lo) {
      ++check_count;
      check_hi  = key_hi + 1;
      check_mid = key_mid;
      check_lo  = key_lo;

      if (RbB == cypher_hi) {
        *iterations -= kiter * 2 - 1;

        rc5_72unitwork->L0.lo           = key_lo;
        rc5_72unitwork->L0.mid          = key_mid;
        rc5_72unitwork->L0.hi           = key_hi;
        rc5_72unitwork->check.count     = check_count;
        rc5_72unitwork->check.lo        = check_lo;
        rc5_72unitwork->check.mid       = check_mid;
        rc5_72unitwork->check.hi        = check_hi;

        return RESULT_FOUND;
      }
    }

    key_hi = (key_hi + 0x02) & 0x000000FF;
    if (!key_hi) {
      key_mid = key_mid + 0x01000000;
      if (!(key_mid & 0xFF000000u)) {
        key_mid = (key_mid + 0x00010000) & 0x00FFFFFF;
        if (!(key_mid & 0x00FF0000)) {
          key_mid = (key_mid + 0x00000100) & 0x0000FFFF;
          if (!(key_mid & 0x0000FF00)) {
            key_mid = (key_mid + 0x00000001) & 0x000000FF;
            if (!key_mid) {
              key_lo = key_lo + 0x01000000;
              if (!(key_lo & 0xFF000000u)) {
                key_lo = (key_lo + 0x00010000) & 0x00FFFFFF;
                if (!(key_lo & 0x00FF0000)) {
                  key_lo = (key_lo + 0x00000100) & 0x0000FFFF;
                  if (!(key_lo & 0x0000FF00)) {
                    key_lo = (key_lo + 0x00000001) & 0x000000FF;
                  }
                }
              }
              /* key_lo changed */
              cached_l0 = ROTL(key_lo + cached_s0, cached_s0);
              cached_s1 = ROTL3(Sa[1] + cached_s0 + cached_l0);
            }
          }
        }
      }
      /* key_mid changed */
      Ta = cached_s1 + cached_l0;
      cached_l1 = ROTL(key_mid + Ta, Ta);
      cached_s2 = ROTL3(Sa[2] + cached_s1 + cached_l1);
    }
  } while (--kiter);

  rc5_72unitwork->L0.lo         = key_lo;
  rc5_72unitwork->L0.mid        = key_mid;
  rc5_72unitwork->L0.hi         = key_hi;
  rc5_72unitwork->check.count   = check_count;
  rc5_72unitwork->check.lo      = check_lo;
  rc5_72unitwork->check.mid     = check_mid;
  rc5_72unitwork->check.hi      = check_hi;

  return RESULT_NOTHING;

}
