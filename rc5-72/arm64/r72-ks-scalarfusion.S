// RC5-72 scalar ARM64 core
// Brought to you by Sunset Ash
// Returns completed A and B, needs C++ support file for final checks

.text
#if defined(__APPLE__)
.globl _scalarFusionEntry
_scalarFusionEntry:
#else
.globl scalarFusionEntry
scalarFusionEntry:
#endif
  // push fp and lr
  stp x29, x30, [sp, #-16]!

  // push the callee saved regs
  stp x19, x20, [sp, #-16]!
  stp x21, x22, [sp, #-16]!
  stp x23, x24, [sp, #-16]!
  stp x25, x26, [sp, #-16]!
  stp x27, x28, [sp, #-16]!

  // we're going to want w0 and w1 later as A and B, so push them now
  stp x0, x1, [sp, #-16]!

  // load initial values; we used to precompute these, but now we don't
  ldr w0, constant_p
  ldr w1, constant_q
  mov w5, w0
  add w6, w0, w1

  mov w30, #2
  madd w7, w1, w30, w0

  mov w30, #3
  madd w8, w1, w30, w0

  mov w30, #4
  madd w9, w1, w30, w0

  mov w30, #5
  madd w10, w1, w30, w0

  mov w30, #6
  madd w11, w1, w30, w0

  mov w30, #7
  madd w12, w1, w30, w0

  mov w30, #8
  madd w13, w1, w30, w0

  mov w30, #9
  madd w14, w1, w30, w0

  mov w30, #10
  madd w15, w1, w30, w0

  mov w30, #11
  madd w16, w1, w30, w0

  mov w30, #12
  madd w17, w1, w30, w0

  mov w30, #13
  madd w18, w1, w30, w0

  mov w30, #14
  madd w19, w1, w30, w0

  mov w30, #15
  madd w20, w1, w30, w0

  mov w30, #16
  madd w21, w1, w30, w0

  mov w30, #17
  madd w22, w1, w30, w0

  mov w30, #18
  madd w23, w1, w30, w0

  mov w30, #19
  madd w24, w1, w30, w0

  mov w30, #20
  madd w25, w1, w30, w0

  mov w30, #21
  madd w26, w1, w30, w0

  mov w30, #22
  madd w27, w1, w30, w0

  mov w30, #23
  madd w28, w1, w30, w0

  mov w30, #24
  madd w29, w1, w30, w0

  mov w30, #25
  madd w30, w1, w30, w0

  ror w5, w5, #29 // s[0] = rotl3(S[0])

  add w2, w2, w5
  mov w1, #32
  sub w1, w1, w5
  ror w2, w2, w1

  #define ROTL_BLOCK_J1(regElement, prevElement) \
    add regElement, regElement, prevElement; \
    add regElement, regElement, w2; \
    ror regElement, regElement, #29; \
    add w3, w3, regElement; \
    add w3, w3, w2; \
    add w1, regElement, w2; \
    mov w0, #32; \
    sub w1, w0, w1; \
    ror w3, w3, w1;

  #define ROTL_BLOCK_J2(regElement, prevElement) \
    add regElement, regElement, prevElement; \
    add regElement, regElement, w3; \
    ror regElement, regElement, #29; \
    add w4, w4, regElement; \
    add w4, w4, w3; \
    add w1, regElement, w3; \
    mov w0, #32; \
    sub w1, w0, w1; \
    ror w4, w4, w1;

  #define ROTL_BLOCK_J0(regElement, prevElement) \
    add regElement, regElement, prevElement; \
    add regElement, regElement, w4; \
    ror regElement, regElement, #29; \
    add w2, w2, regElement; \
    add w2, w2, w4; \
    add w1, regElement, w4; \
    mov w0, #32; \
    sub w1, w0, w1; \
    ror w2, w2, w1;

  #define FINAL_BLOCK(regElement, nextElement) \
    eor w0, w0, w1; \
    mov w5, #32; \
    sub w5, w5, w1; \
    ror w0, w0, w5; \
    add w0, w0, regElement; \
    eor w1, w1, w0; \
    mov w5, #32; \
    sub w5, w5, w0; \
    ror w1, w1, w5; \
    add w1, w1, nextElement;

  // block j1, element 1 - here we go!
  ROTL_BLOCK_J1(w6, w5)

  // block j2, element 2
  ROTL_BLOCK_J2(w7, w6)

  // block j0, element 3
  ROTL_BLOCK_J0(w8, w7)

  // block j1, element 4. deeply thankful i opted to use the preprocessor.
  ROTL_BLOCK_J1(w9, w8)

  ROTL_BLOCK_J2(w10, w9) // j2, element 5
  ROTL_BLOCK_J0(w11, w10) // j0, element 6
  ROTL_BLOCK_J1(w12, w11) // j1, element 7
  ROTL_BLOCK_J2(w13, w12) // j2, element 8
  ROTL_BLOCK_J0(w14, w13) // j0, element 9
  ROTL_BLOCK_J1(w15, w14) // J1, element 10
  ROTL_BLOCK_J2(w16, w15) // J2, element 11
  ROTL_BLOCK_J0(w17, w16) // J0, element 12
  ROTL_BLOCK_J1(w18, w17) // J1, element 13
  ROTL_BLOCK_J2(w19, w18) // J2, element 14
  ROTL_BLOCK_J0(w20, w19) // J0, element 15
  ROTL_BLOCK_J1(w21, w20) // J1, element 16
  ROTL_BLOCK_J2(w22, w21) // J2, element 17
  ROTL_BLOCK_J0(w23, w22) // J0, element 18
  ROTL_BLOCK_J1(w24, w23) // J1, element 19
  ROTL_BLOCK_J2(w25, w24) // J2, element 20
  ROTL_BLOCK_J0(w26, w25) // J0, element 21
  ROTL_BLOCK_J1(w27, w26) // J1, element 22
  ROTL_BLOCK_J2(w28, w27) // J2, element 23
  ROTL_BLOCK_J0(w29, w28) // J0, element 24
  ROTL_BLOCK_J1(w30, w29) // J1, element 25

  ROTL_BLOCK_J2(w5, w30) // J2 element 0 - i0_j2 in the original

  ROTL_BLOCK_J0(w6, w5) // now on the second part
  ROTL_BLOCK_J1(w7, w6)
  ROTL_BLOCK_J2(w8, w7)
  ROTL_BLOCK_J0(w9, w8)
  ROTL_BLOCK_J1(w10, w9)
  ROTL_BLOCK_J2(w11, w10)
  ROTL_BLOCK_J0(w12, w11)
  ROTL_BLOCK_J1(w13, w12)
  ROTL_BLOCK_J2(w14, w13)
  ROTL_BLOCK_J0(w15, w14)
  ROTL_BLOCK_J1(w16, w15)
  ROTL_BLOCK_J2(w17, w16)
  ROTL_BLOCK_J0(w18, w17)
  ROTL_BLOCK_J1(w19, w18)
  ROTL_BLOCK_J2(w20, w19)
  ROTL_BLOCK_J0(w21, w20)
  ROTL_BLOCK_J1(w22, w21)
  ROTL_BLOCK_J2(w23, w22)
  ROTL_BLOCK_J0(w24, w23)
  ROTL_BLOCK_J1(w25, w24)
  ROTL_BLOCK_J2(w26, w25)
  ROTL_BLOCK_J0(w27, w26)
  ROTL_BLOCK_J1(w28, w27)
  ROTL_BLOCK_J2(w29, w28)
  ROTL_BLOCK_J0(w30, w29)

  // third (and final) part before final block

  ROTL_BLOCK_J1(w5, w30)
  ROTL_BLOCK_J2(w6, w5)
  ROTL_BLOCK_J0(w7, w6)
  ROTL_BLOCK_J1(w8, w7)
  ROTL_BLOCK_J2(w9, w8)
  ROTL_BLOCK_J0(w10, w9)
  ROTL_BLOCK_J1(w11, w10)
  ROTL_BLOCK_J2(w12, w11)
  ROTL_BLOCK_J0(w13, w12)
  ROTL_BLOCK_J1(w14, w13)
  ROTL_BLOCK_J2(w15, w14)
  ROTL_BLOCK_J0(w16, w15)
  ROTL_BLOCK_J1(w17, w16)
  ROTL_BLOCK_J2(w18, w17)
  ROTL_BLOCK_J0(w19, w18)
  ROTL_BLOCK_J1(w20, w19)
  ROTL_BLOCK_J2(w21, w20)
  ROTL_BLOCK_J0(w22, w21)
  ROTL_BLOCK_J1(w23, w22)
  ROTL_BLOCK_J2(w24, w23)
  ROTL_BLOCK_J0(w25, w24)
  ROTL_BLOCK_J1(w26, w25)
  ROTL_BLOCK_J2(w27, w26)
  ROTL_BLOCK_J0(w28, w27)
  ROTL_BLOCK_J1(w29, w28)
  ROTL_BLOCK_J2(w30, w29)

  // restore A and B from the stack
  ldp x0, x1, [sp], #16

  // now, for the final blocks...
  add w0, w0, w5
  add w1, w1, w6

  FINAL_BLOCK(w7, w8)
  FINAL_BLOCK(w9, w10)
  FINAL_BLOCK(w11, w12)
  FINAL_BLOCK(w13, w14)
  FINAL_BLOCK(w15, w16)
  FINAL_BLOCK(w17, w18)
  FINAL_BLOCK(w19, w20)
  FINAL_BLOCK(w21, w22)
  FINAL_BLOCK(w23, w24)
  FINAL_BLOCK(w25, w26)
  FINAL_BLOCK(w27, w28)
  FINAL_BLOCK(w29, w30)

  mov x30, xzr // zero out x30, just in case
  add x30, x30, w0, uxtw
  add x30, x30, x1, LSL #32
  mov x0, x30

  ldp x27, x28, [sp], #16
  ldp x25, x26, [sp], #16
  ldp x23, x24, [sp], #16
  ldp x21, x22, [sp], #16
  ldp x19, x20, [sp], #16

  ldp x29, x30, [sp], #16

  ret

.balign 8
constant_p: .word 0xb7e15163

.balign 8
constant_q: .word 0x9e3779b9

