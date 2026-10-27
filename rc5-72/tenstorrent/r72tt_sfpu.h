/*
 * Copyright distributed.net 2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * RC5-72 on the Tensix Vector Unit (SFPU), 32 keys at a time: one key per
 * lane, lane i checking low key byte 'base + i'. Driven by TRISC1, which
 * pushes the instruction stream to Tensix thread 1 and can read Dst back.
 *
 * The SFPU only computes the encrypted low word for every lane and records
 * which lanes match cypher_lo, accumulating one bit per lane and batch in
 * LReg ACC. The firmware then re-checks flagged keys (partial match
 * candidates, roughly one in 2^32) with the scalar core.
 *
 * Register use (LReg) during the key schedule and the encryption:
 *   0  A        / eA      5  x (temp)
 *   1  B        / eB      6  y (temp)
 *   2  L[0]     / s       7  ACC (match bits)
 *   3  L[1]
 *   4  L[2]
 *   9  zero (fixed)      11  31 (configured)      12  cypher_lo (configured)
 *  15  2 * lane (fixed)
 *
 * The key schedule passes and the encryption run as short loops whose
 * Tensix instructions are the same every iteration, so they stay in TRISC1's
 * small instruction cache: their Dst accesses advance through the S[] slots
 * using the Dst read/write counter (RWC), via the address modes below.
 *
 * Dst slots of 32 values (four rows, even columns), at address 4 * slot + RWC:
 *   0..25  S[]       26..51  P + i*Q       52  ACC readback
 *   53     plain_lo  54      plain_hi      55  eA readback (verify mode)
 *
 * $Id:
*/

#ifndef R72TT_SFPU_H
#define R72TT_SFPU_H

#include <stdint.h>

/* --- Tensix instruction encodings ------------------------------------- */

#define TT_OP(op, params) ((uint32_t)(((uint32_t)(op) << 24) + (params)))

#define SFPLOAD_AM(d, am, addr)   TT_OP(0x70, ((d) << 20) + (SFP_INT32 << 16) + ((am) << 13) + (addr))
#define SFPSTORE_AM(d, am, addr)  TT_OP(0x72, ((d) << 20) + (SFP_INT32 << 16) + ((am) << 13) + (addr))
#define SFPLOAD(d, addr)          SFPLOAD_AM(d, AM_KEEP, addr)
#define SFPSTORE(d, addr)         SFPSTORE_AM(d, AM_KEEP, addr)
#define SFPLOADI(d, mod, imm16)   TT_OP(0x71, ((d) << 20) + ((mod) << 16) + ((imm16) & 0xFFFFu))
#define SFPOP(op, imm12, c, d, mod) \
  TT_OP(op, ((uint32_t)((imm12) & 0xFFF) << 12) + ((c) << 8) + ((d) << 4) + (mod))
#define SFPIADD(imm12, c, d, mod) SFPOP(0x79, imm12, c, d, mod)
#define SFPSHFT(imm12, c, d, mod) SFPOP(0x7A, imm12, c, d, mod)
#define SFPAND(imm12, c, d, mod)  SFPOP(0x7E, imm12, c, d, mod)
#define SFPOR(imm12, c, d, mod)   SFPOP(0x7F, imm12, c, d, mod)
#define SFPLZ(c, d)               SFPOP(0x81, 0, c, d, 0)
#define SFPXOR(c, d)              SFPOP(0x8D, 0, c, d, 0)
#define SFPCONFIG(d)              TT_OP(0x91, ((d) << 4))
#define SFPSHFT2(imm12, c, d, mod) SFPOP(0x94, imm12, c, d, mod)
#define STALLWAIT(block, cond)    TT_OP(0xA2, ((block) << 15) + (cond))
#define SETDMAREG(value, reg16)   TT_OP(0x45, ((value) << 8) + (reg16))
#define SETC16(reg, value)        TT_OP(0xB2, ((reg) << 16) + (value))
#define SETRWC_DST0               TT_OP(0x37, 4)   /* Dst RWC = 0 */

/* Address modes (ADDR_MOD slots) for SFPLOAD/SFPSTORE: what happens to the
   Dst RWC after the access. Configured per thread with SETC16. */
#define AM_KEEP          0
#define AM_NEXT          1   /* advance one slot */
#define AM_CLEAR         2   /* back to 0 */
#define THCFG_ADDR_MOD_AB(am)    (12 + (am))
#define THCFG_ADDR_MOD_DST(am)   (28 + (am))
#define THCFG_ADDR_MOD_BIAS(am)  (47 + (am))
#define ADDR_MOD_DST_INCR(n)     (n)
#define ADDR_MOD_DST_CLEAR       (1 << 11)

#define SFP_INT32        4   /* SFPLOAD/SFPSTORE: 32-bit integer, no conversion */
#define SFPLOADI_LO16   10   /* write the low 16 bits, keep the high 16 */
#define SFPLOADI_HI16    8   /* write the high 16 bits, keep the low 16 */
#define SFPLOADI_U16     2   /* zero-extended 16-bit value */
#define IADD_REG         4   /* VD = VC + VD, flags untouched */
#define IADD_IMM         5   /* VD = VC + Imm12, flags untouched */
#define SHFT_REG         0   /* VD = VD << VC (VC < 0: logical >> -VC) */
#define SHFT_IMM_VC      5   /* VD = VC << Imm12 (Imm12 < 0: logical >> -Imm12) */
#define SHFT2_VB         5   /* VD = L[Imm12] << VC (VC < 0: logical >> -VC) */
#define LOGIC_VD         0   /* VD = VD op VC */
#define LOGIC_VB         1   /* VD = L[Imm12] op VC */
#define STALL_THCON      (1 << 5)    /* block ThCon (SETDMAREG) ... */
#define WAIT_SFPU        (1 << 11)   /* ... until the SFPU is idle */

#define LR_A    0
#define LR_B    1
#define LR_L0   2
#define LR_L1   3
#define LR_L2   4
#define LR_X    5
#define LR_Y    6
#define LR_ACC  7
#define LR_ZERO 9
#define LR_31   11
#define LR_CLO  12
#define LR_LANE2 15
#define LR_EA   LR_A
#define LR_EB   LR_B
#define LR_S    LR_L0

#define SLOT(n)         (4 * (n))
#define SLOT_S(i)       SLOT(i)
#define SLOT_SINIT(i)   SLOT(26 + (i))
#define SLOT_ACC        SLOT(52)
#define SLOT_PLAIN_LO   SLOT(53)
#define SLOT_PLAIN_HI   SLOT(54)
#define SLOT_EA         SLOT(55)

/* TRISC1 views */
#define TENSIX_INSTRN_BUF 0xFFE40000u
#define TENSIX_GPR63      (0xFFE00000u + 63 * 4)
#define TENSIX_GPR63_LO16 126
#define DST_BASE          0xFFBD8000u
#define DST_SLOT_LANE(slot_addr, lane) \
  (DST_BASE + ((((slot_addr) + (lane) / 8) * 16 + 2 * ((lane) & 7)) * 4))

/* Pushes a compile-time constant instruction as a single .ttinsn */
#define TTI_ROT2(w) ((((uint32_t)(w)) << 2) | (((uint32_t)(w)) >> 30))
#define TTI(w) __asm__ volatile(".word %0" :: "i"(TTI_ROT2(w)) : "memory")

/* Pushes an instruction computed at run time */
static inline void tt_push(uint32_t w)
{
  *(volatile uint32_t *)TENSIX_INSTRN_BUF = w;
}

static inline void sfpu_loadi32(uint32_t d, uint32_t v)
{
  tt_push(SFPLOADI(d, SFPLOADI_LO16, v & 0xFFFF));
  tt_push(SFPLOADI(d, SFPLOADI_HI16, v >> 16));
}

/* Waits for all SFPU instructions pushed so far to complete */
static void sfpu_wait(void)
{
  volatile uint32_t *done = (volatile uint32_t *)TENSIX_GPR63;
  *done = 0;
  while (*done)
    ;
  TTI(STALLWAIT(STALL_THCON, WAIT_SFPU));
  TTI(SETDMAREG(1, TENSIX_GPR63_LO16));
  while (!*done)
    ;
}

/* --- Key schedule -------------------------------------------------------- */

/* A = S[i] = rotl3(src + A + B); 'src' and S[i] are Dst slots offset by
   the RWC, which address mode 'am' updates after the store */
#define V_ASTEP(src, i, am) \
  TTI(SFPLOAD(LR_X, src)); \
  TTI(SFPIADD(0, LR_A, LR_X, IADD_REG)); \
  TTI(SFPIADD(0, LR_B, LR_X, IADD_REG)); \
  TTI(SFPSHFT(3, LR_X, LR_Y, SHFT_IMM_VC)); \
  TTI(SFPSHFT(-29, LR_X, LR_A, SHFT_IMM_VC)); \
  TTI(SFPOR(0, LR_Y, LR_A, LOGIC_VD)); \
  TTI(SFPSTORE_AM(LR_A, am, SLOT_S(i)));

/* B = Lj = rotl(Lj + A + B, A + B); B doubles as the shift amount */
#define V_BSTEP(Lj) \
  TTI(SFPIADD(0, LR_A, LR_B, IADD_REG)); \
  TTI(SFPIADD(0, LR_B, Lj, IADD_REG)); \
  TTI(SFPAND(0, LR_31, LR_B, LOGIC_VD)); \
  TTI(SFPSHFT2(Lj, LR_B, LR_Y, SHFT2_VB)); \
  TTI(SFPIADD(-32, LR_B, LR_B, IADD_IMM)); \
  TTI(SFPSHFT(0, LR_B, Lj, SHFT_REG)); \
  TTI(SFPOR(0, LR_Y, Lj, LOGIC_VD)); \
  TTI(SFPOR(Lj, Lj, LR_B, LOGIC_VB));

/* Key schedule steps for passes starting at S[first], RWC counting from 0 */
#define V_PASS1(first, Lj, am) V_ASTEP(SLOT_SINIT(first), first, am) V_BSTEP(Lj)
#define V_PASS23(Lj, am) V_ASTEP(SLOT_S(0), 0, am) V_BSTEP(Lj)

/* --- Encryption ------------------------------------------------------------ */

/* d = rotl(d ^ o, o) + S[i], with S[i] offset by the RWC */
#define V_HALF(d, o, i, am) \
  TTI(SFPXOR(o, d)); \
  TTI(SFPAND(o, LR_31, LR_S, LOGIC_VB)); \
  TTI(SFPSHFT2(d, LR_S, LR_Y, SHFT2_VB)); \
  TTI(SFPIADD(-32, LR_S, LR_S, IADD_IMM)); \
  TTI(SFPSHFT(0, LR_S, d, SHFT_REG)); \
  TTI(SFPOR(0, LR_Y, d, LOGIC_VD)); \
  TTI(SFPLOAD_AM(LR_X, am, SLOT_S(i))); \
  TTI(SFPIADD(0, LR_X, d, IADD_REG));

/* One-time setup for a run: constants and the plaintext */
static void sfpu_setup(uint32_t plain_lo, uint32_t plain_hi, uint32_t cypher_lo)
{
  uint32_t i;

  TTI(SETC16(THCFG_ADDR_MOD_AB(AM_KEEP), 0));
  TTI(SETC16(THCFG_ADDR_MOD_DST(AM_KEEP), 0));
  TTI(SETC16(THCFG_ADDR_MOD_BIAS(AM_KEEP), 0));
  TTI(SETC16(THCFG_ADDR_MOD_AB(AM_NEXT), 0));
  TTI(SETC16(THCFG_ADDR_MOD_DST(AM_NEXT), ADDR_MOD_DST_INCR(SLOT(1))));
  TTI(SETC16(THCFG_ADDR_MOD_BIAS(AM_NEXT), 0));
  TTI(SETC16(THCFG_ADDR_MOD_AB(AM_CLEAR), 0));
  TTI(SETC16(THCFG_ADDR_MOD_DST(AM_CLEAR), ADDR_MOD_DST_CLEAR));
  TTI(SETC16(THCFG_ADDR_MOD_BIAS(AM_CLEAR), 0));
  /* Nothing has run on the SFPU yet in this run, so this cannot race */
  TTI(SETRWC_DST0);

  TTI(SFPLOADI(0, SFPLOADI_U16, 31));
  TTI(SFPCONFIG(LR_31));
  sfpu_loadi32(0, cypher_lo);
  TTI(SFPCONFIG(LR_CLO));
  for (i = 0; i < 26; i++)
  {
    sfpu_loadi32(LR_X, 0xB7E15163u + i * 0x9E3779B9u);
    tt_push(SFPSTORE(LR_X, SLOT_SINIT(i)));
  }
  sfpu_loadi32(LR_X, plain_lo);
  TTI(SFPSTORE(LR_X, SLOT_PLAIN_LO));
  sfpu_loadi32(LR_X, plain_hi);
  TTI(SFPSTORE(LR_X, SLOT_PLAIN_HI));
  TTI(SFPLOADI(LR_ACC, SFPLOADI_U16, 0));
}

/*
 * Queues one batch: lanes check low key bytes base..base+31 for the
 * (mid, lo) whose key schedule prefix is 'p'. Shifts ACC left by one and
 * sets bit 0 of the lanes whose encrypted low word equals cypher_lo.
 * In verify mode the encrypted low words are also stored to SLOT_EA.
 */
static void sfpu_batch(const prefix_t *p, uint32_t base, int verify)
{
  const uint32_t r = p->ab & 31;
  uint32_t i;

  /* B = L[2] = rotl(hi + ab, ab), with hi = base + lane */
  TTI(SFPSHFT(-1, LR_LANE2, LR_L2, SHFT_IMM_VC));
  tt_push(SFPIADD(base, LR_L2, LR_L2, IADD_IMM));
  sfpu_loadi32(LR_X, p->ab);
  TTI(SFPIADD(0, LR_X, LR_L2, IADD_REG));
  if (r)
  {
    tt_push(SFPSHFT(r, LR_L2, LR_Y, SHFT_IMM_VC));
    tt_push(SFPSHFT(r - 32, LR_L2, LR_L2, SHFT_IMM_VC));
    TTI(SFPOR(0, LR_Y, LR_L2, LOGIC_VD));
  }
  TTI(SFPOR(LR_L2, LR_L2, LR_B, LOGIC_VB));
  sfpu_loadi32(LR_A, p->s2);
  sfpu_loadi32(LR_L0, p->l0);
  sfpu_loadi32(LR_L1, p->l1);
  sfpu_loadi32(LR_X, p->s0);
  TTI(SFPSTORE(LR_X, SLOT_S(0)));
  sfpu_loadi32(LR_X, p->s1);
  TTI(SFPSTORE(LR_X, SLOT_S(1)));
  TTI(SFPSTORE(LR_A, SLOT_S(2)));

  /* First pass: steps 3..25, cycling L[0], L[1], L[2] */
#pragma GCC unroll 1
  for (i = 0; i < 7; i++)
  {
    V_PASS1(3, LR_L0, AM_NEXT) V_PASS1(3, LR_L1, AM_NEXT) V_PASS1(3, LR_L2, AM_NEXT)
  }
  V_PASS1(3, LR_L0, AM_NEXT) V_PASS1(3, LR_L1, AM_CLEAR)

  /* Second pass: steps 0..25, starting with L[2] */
#pragma GCC unroll 1
  for (i = 0; i < 8; i++)
  {
    V_PASS23(LR_L2, AM_NEXT) V_PASS23(LR_L0, AM_NEXT) V_PASS23(LR_L1, AM_NEXT)
  }
  V_PASS23(LR_L2, AM_NEXT) V_PASS23(LR_L0, AM_CLEAR)

  /* Third pass, starting with L[1]: S[25] and the last L update only feed
     the encrypted high word, which the SFPU does not compute */
#pragma GCC unroll 1
  for (i = 0; i < 8; i++)
  {
    V_PASS23(LR_L1, AM_NEXT) V_PASS23(LR_L2, AM_NEXT) V_PASS23(LR_L0, AM_NEXT)
  }
  V_ASTEP(SLOT_S(0), 0, AM_CLEAR)

  /* eA = plain_lo + S[0], eB = plain_hi + S[1] */
  TTI(SFPLOAD(LR_EA, SLOT_PLAIN_LO));
  TTI(SFPLOAD(LR_X, SLOT_S(0)));
  TTI(SFPIADD(0, LR_X, LR_EA, IADD_REG));
  TTI(SFPLOAD(LR_EB, SLOT_PLAIN_HI));
  TTI(SFPLOAD(LR_X, SLOT_S(1)));
  TTI(SFPIADD(0, LR_X, LR_EB, IADD_REG));
  /* Rounds 1..11, then the last round's low word, using S[2..24] */
#pragma GCC unroll 1
  for (i = 0; i < 11; i++)
  {
    V_HALF(LR_EA, LR_EB, 2, AM_NEXT) V_HALF(LR_EB, LR_EA, 2, AM_NEXT)
  }
  V_HALF(LR_EA, LR_EB, 2, AM_CLEAR)

  if (verify)
    TTI(SFPSTORE(LR_EA, SLOT_EA));

  /* ACC = (ACC << 1) | (eA == cypher_lo), via clz(eA ^ cypher_lo) >> 5 */
  TTI(SFPXOR(LR_CLO, LR_EA));
  TTI(SFPLZ(LR_EA, LR_EA));
  TTI(SFPSHFT(-5, LR_EA, LR_EA, SHFT_IMM_VC));
  TTI(SFPSHFT(1, LR_ACC, LR_ACC, SHFT_IMM_VC));
  TTI(SFPOR(0, LR_EA, LR_ACC, LOGIC_VD));
}

/* Stores ACC, clears it, and waits until the stored copy can be read */
static void sfpu_flush_acc(void)
{
  TTI(SFPSTORE(LR_ACC, SLOT_ACC));
  TTI(SFPLOADI(LR_ACC, SFPLOADI_U16, 0));
  sfpu_wait();
}

#endif /* R72TT_SFPU_H */
