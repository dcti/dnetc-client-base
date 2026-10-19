/*
 * RC5-72 NEON core for arm64 - 2026
 *
 * This core runs 32 keys at once as 8 groups of uint32x4_t (4 keys/lane).
 * Variable left-rotate uses USHL with a per-lane signed shift:
 * ROTL(x,s) = ushl(x, s&31) | ushl(x, (s&31)-32). The schedule/encryption are
 * bit-identical to the ANSI core (validated by `dnetc -test rc5-72`). Each
 * lane's key is built with the full ANSI carry, so blocks need no alignment;
 * the core reports pipeline_count 16 and masks the lanes of a short final
 * block.
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

static inline void key_inc(u32 &hi, u32 &mid, u32 &lo)
{
  hi = (hi + 1) & 0xFF;
  if (!hi) {
    mid = mid + 0x01000000;
    if (!(mid & 0xFF000000u)) { mid = (mid + 0x00010000) & 0x00FFFFFF;
      if (!(mid & 0x00FF0000)) { mid = (mid + 0x00000100) & 0x0000FFFF;
        if (!(mid & 0x0000FF00)) { mid = (mid + 1) & 0xFF;
          if (!mid) { lo = lo + 0x01000000;
            if (!(lo & 0xFF000000u)) { lo = (lo + 0x00010000) & 0x00FFFFFF;
              if (!(lo & 0x00FF0000)) { lo = (lo + 0x00000100) & 0x0000FFFF;
                if (!(lo & 0x0000FF00)) { lo = (lo + 0x00000001) & 0x000000FF; } } } } } } }
  }
}

extern "C" s32 rc5_72_unit_func_neon_8x4(RC5_72UnitWork *w, u32 *iterations, void *)
{
  uint32x4_t S[NG][26], L[NG][3], A[NG], B[NG];
  u32 khi[NP + 1], kmid[NP + 1], klo[NP + 1];
  u32 todo = *iterations, done = 0;
  khi[0] = w->L0.hi; kmid[0] = w->L0.mid; klo[0] = w->L0.lo;
  while (done < todo) {
    u32 valid = (todo - done < NP) ? (todo - done) : NP;
    bool wrap = khi[0] + NP > 0x100;
    if (!wrap) {
      for (int n = 1; n <= NP; n++) { khi[n] = khi[0] + n; kmid[n] = kmid[0]; klo[n] = klo[0]; }
      if (khi[NP] == 0x100) { khi[NP] = khi[NP - 1]; key_inc(khi[NP], kmid[NP], klo[NP]); }
    } else {
      for (int n = 0; n < NP; n++) {
        khi[n + 1] = khi[n]; kmid[n + 1] = kmid[n]; klo[n + 1] = klo[n];
        key_inc(khi[n + 1], kmid[n + 1], klo[n + 1]);
      }
    }
    const uint32x4_t lane = {0, 1, 2, 3};
    _Pragma("clang loop unroll(full)")
    for (int g = 0; g < NG; g++) {
      if (!wrap) {
        L[g][2] = vaddq_u32(vdupq_n_u32(khi[0] + 4 * g), lane);
        L[g][1] = vdupq_n_u32(kmid[0]);
        L[g][0] = vdupq_n_u32(klo[0]);
      } else {
        L[g][2] = vld1q_u32(&khi[4 * g]);
        L[g][1] = vld1q_u32(&kmid[4 * g]);
        L[g][0] = vld1q_u32(&klo[4 * g]);
      }
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
        u32 n = 4 * g + p;
        if (al[p] == w->cypher.lo && n < valid) {
          ++w->check.count;
          w->check.hi = khi[n];
          w->check.mid = kmid[n];
          w->check.lo = klo[n];
          if (bl[p] == w->cypher.hi) {
            w->L0.hi = khi[n]; w->L0.mid = kmid[n]; w->L0.lo = klo[n];
            *iterations = done + n;
            return RESULT_FOUND;
          }
        }
      }
    }
    done += valid;
    khi[0] = khi[valid]; kmid[0] = kmid[valid]; klo[0] = klo[valid];
  }
  w->L0.hi = khi[0]; w->L0.mid = kmid[0]; w->L0.lo = klo[0];
  return RESULT_NOTHING;
}
