/*
 * RC5-72 NEON core for arm64 - Oct 2026
 *
 * This core runs 32 keys at once as 8 groups of uint32x4_t (4 keys/lane).
 * Variable left-rotate uses USHL with a per-lane signed shift:
 * ROTL(x,s) = ushl(x, s&31) | ushl(x, (s&31)-32). The schedule/encryption are
 * bit-identical to the ANSI core (validated by `dnetc -test rc5-72`).
 * 32 keys/block and 32 divides 256, so the hi byte never crosses a block
 * boundary within a group assuming the start is aligned -- the ANSI key
 * counter applies as-is.
 */
#include "ccoreio.h"
#include <arm_neon.h>

#define P 0xB7E15163u
#define Q 0x9E3779B9u
#define NG 8                 /* vector groups (4 keys each) */
#define NP (4 * NG)          /* 32 keys per block */

static inline uint32x4_t VROTL(uint32x4_t x, uint32x4_t s)
{
  int32x4_t m = vreinterpretq_s32_u32(vandq_u32(s, vdupq_n_u32(31)));
  return vorrq_u32(vshlq_u32(x, m), vshlq_u32(x, vsubq_s32(m, vdupq_n_s32(32))));
}
static inline uint32x4_t VROTL3(uint32x4_t x)
{ return vorrq_u32(vshlq_n_u32(x, 3), vshrq_n_u32(x, 29)); }

extern "C" s32 rc5_72_unit_func_neon_8x4(RC5_72UnitWork *w, u32 *iterations, void *)
{
  uint32x4_t S[NG][26], L[NG][3], A[NG], B[NG];
  const uint32x4_t lane = {0, 1, 2, 3};
  u32 kiter = *iterations / NP;
  while (kiter--) {
    _Pragma("clang loop unroll(full)")
    for (int g = 0; g < NG; g++) {
      L[g][2] = vaddq_u32(vdupq_n_u32(w->L0.hi), vaddq_u32(lane, vdupq_n_u32(4 * g)));
      L[g][1] = vdupq_n_u32(w->L0.mid);
      L[g][0] = vdupq_n_u32(w->L0.lo);
      for (int i = 0; i < 26; i++) S[g][i] = vdupq_n_u32(P + (u32)i * Q);
      A[g] = vdupq_n_u32(0); B[g] = vdupq_n_u32(0);
    }
    int i = 0, j = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 78; k++) {
      _Pragma("clang loop unroll(full)")
      for (int g = 0; g < NG; g++) {
        uint32x4_t ab = vaddq_u32(A[g], B[g]);
        A[g] = S[g][i] = VROTL3(vaddq_u32(S[g][i], ab));
        uint32x4_t ab2 = vaddq_u32(A[g], B[g]);
        B[g] = L[g][j] = VROTL(vaddq_u32(L[g][j], ab2), ab2);
      }
      if (++i == 26) i = 0;
      if (++j == 3) j = 0;
    }
    _Pragma("clang loop unroll(full)")
    for (int g = 0; g < NG; g++) {
      uint32x4_t a = vaddq_u32(vdupq_n_u32(w->plain.lo), S[g][0]);
      uint32x4_t b = vaddq_u32(vdupq_n_u32(w->plain.hi), S[g][1]);
      _Pragma("clang loop unroll(full)")
      for (int r = 1; r <= 12; r++) {
        a = vaddq_u32(VROTL(veorq_u32(a, b), b), S[g][2 * r]);
        b = vaddq_u32(VROTL(veorq_u32(b, a), a), S[g][2 * r + 1]);
      }
      u32 al[4], bl[4];
      vst1q_u32(al, a); vst1q_u32(bl, b);
      for (int p = 0; p < 4; p++) {
        if (al[p] == w->cypher.lo) {
          ++w->check.count;
          w->check.hi = w->L0.hi + (u32)(4 * g + p);
          w->check.mid = w->L0.mid;
          w->check.lo = w->L0.lo;
          if (bl[p] == w->cypher.hi) { *iterations -= (kiter + 1) * NP - (4 * g + p); return RESULT_FOUND; }
        }
      }
    }
    /* key += NP (byte hi, full ANSI carry into mid/lo) */
    w->L0.hi = (w->L0.hi + NP) & 0xFF;
    if (!w->L0.hi) {
      w->L0.mid = w->L0.mid + 0x01000000;
      if (!(w->L0.mid & 0xFF000000u)) { w->L0.mid = (w->L0.mid + 0x00010000) & 0x00FFFFFF;
        if (!(w->L0.mid & 0x00FF0000)) { w->L0.mid = (w->L0.mid + 0x00000100) & 0x0000FFFF;
          if (!(w->L0.mid & 0x0000FF00)) { w->L0.mid = (w->L0.mid + 1) & 0xFF;
            if (!w->L0.mid) { w->L0.lo = w->L0.lo + 0x01000000;
              if (!(w->L0.lo & 0xFF000000u)) { w->L0.lo = (w->L0.lo + 0x00010000) & 0x00FFFFFF;
                if (!(w->L0.lo & 0x00FF0000)) { w->L0.lo = (w->L0.lo + 0x00000100) & 0x0000FFFF;
                  if (!(w->L0.lo & 0x0000FF00)) { w->L0.lo = (w->L0.lo + 0x00000001) & 0x000000FF; } } } } } } }
    }
  }
  return RESULT_NOTHING;
}
