// RC5-72 NEON core for AArch64, 32 keys per call (MIX NEON)
// Created by Mix. A port of the 64-key AVX-512F core (MIX ZMM):
// 8 pipes x 4 lanes, run as two lockstep groups of 4.
//
// NEON has no vector rotate: x <<< s = ushl(x, s & 31) | ushl(x, s | ~31)
// (USHL takes the low byte of each lane as a signed shift); x <<< 3 = shl + sri.
// Each pipe keeps A and its current B word in registers; the other two B
// words live on the stack. Needs MINIMUM_ITERATIONS to be a multiple of 32.

#if defined(__APPLE__)
#define FUNC(name) _##name
#define L(name) L##name
#else
#define FUNC(name) name
#define L(name) .L##name
#endif

        .text
        .p2align 4
        .globl  FUNC(rc5_72_unit_func_neon_mix)
#if defined(__ELF__)
        .type   rc5_72_unit_func_neon_mix, %function
#endif
FUNC(rc5_72_unit_func_neon_mix):
// x0 = RC5_72UnitWork *, x1 = u32 *iterations
// w9 batches left  w13/w14/w15 L0 hi/mid/lo  x10 S source  x11 S dest  w3 round
// x12 &S[1], x8 S-table base for the k=52 wrap
// w2 w5 w4 w6 w7 w17: the scalar precompute of k=0-2 (kept across batches)
// v8-v15 A (C in encryption)  v16-v23 current B word (D)  v0-v7,v24-v27 scratch
// v28 cypher.lo  v29 cypher.hi  v30 #31  v31 ~31
// Stack, 64-byte aligned: d8-d15 | L words 3x128 | A+B_1 128 | S table 26x128

        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        sub     x16, x29, #3904
        and     sp, x16, #-64
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]

        ldr     w9, [x1]
        lsr     w9, w9, #5                 // batches of 32 keys
        ldr     w13, [x0, #16]             // L0.hi
        ldr     w14, [x0, #20]             // L0.mid
        ldr     w15, [x0, #24]             // L0.lo
        add     x12, sp, #704            // &S[1]

        movi    v30.4s, #31
        mvni    v31.4s, #31              // 0xFFFFFFE0
        add     x16, x0, #12
        ld1r    {v28.4s}, [x16]          // cypher.lo
        add     x16, x0, #8
        ld1r    {v29.4s}, [x16]          // cypher.hi

L(outer_loop):
        // k=0: pre_B_0 = (L0lo + S0) <<< 29
        mov     w16, #0x8B1D
        movk    w16, #0xBF0A, lsl #16
        add     w2, w15, w16
        ror     w2, w2, #3
        // k=1: pre_S1 = (P+Q + S0 + pre_B_0) <<< 3
        mov     w16, #0x5639
        movk    w16, #0x1523, lsl #16
        add     w5, w2, w16
        ror     w5, w5, #29
        add     w4, w2, w5

L(mid_loop):
        // k=1: pre_B_1 = (L0mid + w4) <<< w4
        add     w6, w14, w4
        neg     w16, w4
        ror     w6, w6, w16
        // k=2: pre_S2 = (P+2Q + pre_S1 + pre_B_1) <<< 3; pre_AB_2 = pre_S2 + pre_B_1
        mov     w16, #0x44D5
        movk    w16, #0xF450, lsl #16
        add     w7, w5, w6
        add     w7, w7, w16
        ror     w7, w7, #29
        add     w17, w6, w7

        .p2align 4
L(inner_loop):
        dup     v8.4s, w7
        dup     v9.4s, w7
        dup     v10.4s, w7
        dup     v11.4s, w7
        dup     v12.4s, w7
        dup     v13.4s, w7
        dup     v14.4s, w7
        dup     v15.4s, w7
        dup     v0.4s, w2                // L[0] = pre_B_0
        dup     v1.4s, w6                // L[1] = pre_B_1
        dup     v2.4s, w5                // S[1] = pre_S1
        stp     q0, q0, [sp, #64]
        stp     q0, q0, [sp, #96]
        stp     q0, q0, [sp, #128]
        stp     q0, q0, [sp, #160]
        stp     q1, q1, [sp, #192]
        stp     q1, q1, [sp, #224]
        stp     q1, q1, [sp, #256]
        stp     q1, q1, [sp, #288]
        stp     q2, q2, [sp, #704]
        stp     q2, q2, [sp, #736]
        stp     q2, q2, [sp, #768]
        stp     q2, q2, [sp, #800]
        stp     q8, q9, [sp, #832]
        stp     q10, q11, [sp, #864]
        stp     q12, q13, [sp, #896]
        stp     q14, q15, [sp, #928]

        // k=2: B_2 = (L0hi + lane + pre_AB_2) <<< pre_AB_2
        dup     v24.4s, w17
        and     v25.16b, v24.16b, v30.16b
        orr     v26.16b, v24.16b, v31.16b
        adr     x16, L(lanes)
        dup     v16.4s, w13
        dup     v17.4s, w13
        dup     v18.4s, w13
        dup     v19.4s, w13
        ldp     q0, q1, [x16, #0]
        ldp     q2, q3, [x16, #32]
        add     v16.4s, v16.4s, v0.4s
        add     v17.4s, v17.4s, v1.4s
        add     v18.4s, v18.4s, v2.4s
        add     v19.4s, v19.4s, v3.4s
        add     v16.4s, v16.4s, v24.4s
        add     v17.4s, v17.4s, v24.4s
        add     v18.4s, v18.4s, v24.4s
        add     v19.4s, v19.4s, v24.4s
        ushl    v4.4s, v16.4s, v25.4s
        ushl    v5.4s, v17.4s, v25.4s
        ushl    v6.4s, v18.4s, v25.4s
        ushl    v7.4s, v19.4s, v25.4s
        ushl    v0.4s, v16.4s, v26.4s
        ushl    v1.4s, v17.4s, v26.4s
        ushl    v2.4s, v18.4s, v26.4s
        ushl    v3.4s, v19.4s, v26.4s
        orr     v16.16b, v4.16b, v0.16b
        orr     v17.16b, v5.16b, v1.16b
        orr     v18.16b, v6.16b, v2.16b
        orr     v19.16b, v7.16b, v3.16b
        dup     v20.4s, w13
        dup     v21.4s, w13
        dup     v22.4s, w13
        dup     v23.4s, w13
        ldp     q0, q1, [x16, #64]
        ldp     q2, q3, [x16, #96]
        add     v20.4s, v20.4s, v0.4s
        add     v21.4s, v21.4s, v1.4s
        add     v22.4s, v22.4s, v2.4s
        add     v23.4s, v23.4s, v3.4s
        add     v20.4s, v20.4s, v24.4s
        add     v21.4s, v21.4s, v24.4s
        add     v22.4s, v22.4s, v24.4s
        add     v23.4s, v23.4s, v24.4s
        ushl    v4.4s, v20.4s, v25.4s
        ushl    v5.4s, v21.4s, v25.4s
        ushl    v6.4s, v22.4s, v25.4s
        ushl    v7.4s, v23.4s, v25.4s
        ushl    v0.4s, v20.4s, v26.4s
        ushl    v1.4s, v21.4s, v26.4s
        ushl    v2.4s, v22.4s, v26.4s
        ushl    v3.4s, v23.4s, v26.4s
        orr     v20.16b, v4.16b, v0.16b
        orr     v21.16b, v5.16b, v1.16b
        orr     v22.16b, v6.16b, v2.16b
        orr     v23.16b, v7.16b, v3.16b

        adr     x10, L(IS3)
        add     x11, sp, #960
        add     x8, sp, #448             // wrap target for csel
        mov     w3, #0
        b       L(round1)

        .p2align 4
L(round):
        cmp     w3, #7
        csel    x11, x12, x11, eq          // k=26 writes S[0]
        // === STEP A: A += B_1, B_2 <<< ===
        // 1. Load S
        ldp     q0, q1, [x10, #-128]
        ldp     q2, q3, [x10, #-96]
        // 2. Add S and B_1 into A
        add     v0.4s, v0.4s, v8.4s
        add     v1.4s, v1.4s, v9.4s
        add     v2.4s, v2.4s, v10.4s
        add     v3.4s, v3.4s, v11.4s
        add     v0.4s, v0.4s, v16.4s
        add     v1.4s, v1.4s, v17.4s
        add     v2.4s, v2.4s, v18.4s
        add     v3.4s, v3.4s, v19.4s
        // 3. Fixed rotate (shl + sri)
        shl     v8.4s, v0.4s, #3
        shl     v9.4s, v1.4s, #3
        shl     v10.4s, v2.4s, #3
        shl     v11.4s, v3.4s, #3
        sri     v8.4s, v0.4s, #29
        sri     v9.4s, v1.4s, #29
        sri     v10.4s, v2.4s, #29
        sri     v11.4s, v3.4s, #29
        // 4. Store S
        stp     q8, q9, [x11, #-128]
        stp     q10, q11, [x11, #-96]
        // 5. T = B_1 + A
        add     v0.4s, v16.4s, v8.4s
        add     v1.4s, v17.4s, v9.4s
        add     v2.4s, v18.4s, v10.4s
        add     v3.4s, v19.4s, v11.4s
        // 6. Load B_2 and store B_1 (the L area); add T into B_2
        ldp     q24, q25, [sp, #320]
        ldp     q26, q27, [sp, #352]
        stp     q16, q17, [sp, #192]
        stp     q18, q19, [sp, #224]
        add     v24.4s, v24.4s, v0.4s
        add     v25.4s, v25.4s, v1.4s
        add     v26.4s, v26.4s, v2.4s
        add     v27.4s, v27.4s, v3.4s
        // 7. Variable rotate B_2 by T
        and     v4.16b, v0.16b, v30.16b
        and     v5.16b, v1.16b, v30.16b
        and     v6.16b, v2.16b, v30.16b
        and     v7.16b, v3.16b, v30.16b
        orr     v0.16b, v0.16b, v31.16b
        orr     v1.16b, v1.16b, v31.16b
        orr     v2.16b, v2.16b, v31.16b
        orr     v3.16b, v3.16b, v31.16b
        ushl    v4.4s, v24.4s, v4.4s
        ushl    v5.4s, v25.4s, v5.4s
        ushl    v6.4s, v26.4s, v6.4s
        ushl    v7.4s, v27.4s, v7.4s
        ushl    v0.4s, v24.4s, v0.4s
        ushl    v1.4s, v25.4s, v1.4s
        ushl    v2.4s, v26.4s, v2.4s
        ushl    v3.4s, v27.4s, v3.4s
        orr     v16.16b, v4.16b, v0.16b
        orr     v17.16b, v5.16b, v1.16b
        orr     v18.16b, v6.16b, v2.16b
        orr     v19.16b, v7.16b, v3.16b
        // same for pipes 4-7
        ldp     q0, q1, [x10, #-64]
        ldp     q2, q3, [x10, #-32]
        add     v0.4s, v0.4s, v12.4s
        add     v1.4s, v1.4s, v13.4s
        add     v2.4s, v2.4s, v14.4s
        add     v3.4s, v3.4s, v15.4s
        add     v0.4s, v0.4s, v20.4s
        add     v1.4s, v1.4s, v21.4s
        add     v2.4s, v2.4s, v22.4s
        add     v3.4s, v3.4s, v23.4s
        shl     v12.4s, v0.4s, #3
        shl     v13.4s, v1.4s, #3
        shl     v14.4s, v2.4s, #3
        shl     v15.4s, v3.4s, #3
        sri     v12.4s, v0.4s, #29
        sri     v13.4s, v1.4s, #29
        sri     v14.4s, v2.4s, #29
        sri     v15.4s, v3.4s, #29
        stp     q12, q13, [x11, #-64]
        stp     q14, q15, [x11, #-32]
        add     v0.4s, v20.4s, v12.4s
        add     v1.4s, v21.4s, v13.4s
        add     v2.4s, v22.4s, v14.4s
        add     v3.4s, v23.4s, v15.4s
        ldp     q24, q25, [sp, #384]
        ldp     q26, q27, [sp, #416]
        stp     q20, q21, [sp, #256]
        stp     q22, q23, [sp, #288]
        add     v24.4s, v24.4s, v0.4s
        add     v25.4s, v25.4s, v1.4s
        add     v26.4s, v26.4s, v2.4s
        add     v27.4s, v27.4s, v3.4s
        and     v4.16b, v0.16b, v30.16b
        and     v5.16b, v1.16b, v30.16b
        and     v6.16b, v2.16b, v30.16b
        and     v7.16b, v3.16b, v30.16b
        orr     v0.16b, v0.16b, v31.16b
        orr     v1.16b, v1.16b, v31.16b
        orr     v2.16b, v2.16b, v31.16b
        orr     v3.16b, v3.16b, v31.16b
        ushl    v4.4s, v24.4s, v4.4s
        ushl    v5.4s, v25.4s, v5.4s
        ushl    v6.4s, v26.4s, v6.4s
        ushl    v7.4s, v27.4s, v7.4s
        ushl    v0.4s, v24.4s, v0.4s
        ushl    v1.4s, v25.4s, v1.4s
        ushl    v2.4s, v26.4s, v2.4s
        ushl    v3.4s, v27.4s, v3.4s
        orr     v20.16b, v4.16b, v0.16b
        orr     v21.16b, v5.16b, v1.16b
        orr     v22.16b, v6.16b, v2.16b
        orr     v23.16b, v7.16b, v3.16b
        add     w3, w3, #1

L(round1):
        cmp     w3, #8
        csel    x10, x11, x10, eq          // ROM constants -> S table
        // === STEP B: A += B_2, B_0 <<< ===
        ldp     q0, q1, [x10, #0]
        ldp     q2, q3, [x10, #32]
        add     v0.4s, v0.4s, v8.4s
        add     v1.4s, v1.4s, v9.4s
        add     v2.4s, v2.4s, v10.4s
        add     v3.4s, v3.4s, v11.4s
        add     v0.4s, v0.4s, v16.4s
        add     v1.4s, v1.4s, v17.4s
        add     v2.4s, v2.4s, v18.4s
        add     v3.4s, v3.4s, v19.4s
        shl     v8.4s, v0.4s, #3
        shl     v9.4s, v1.4s, #3
        shl     v10.4s, v2.4s, #3
        shl     v11.4s, v3.4s, #3
        sri     v8.4s, v0.4s, #29
        sri     v9.4s, v1.4s, #29
        sri     v10.4s, v2.4s, #29
        sri     v11.4s, v3.4s, #29
        stp     q8, q9, [x11, #0]
        stp     q10, q11, [x11, #32]
        add     v0.4s, v16.4s, v8.4s
        add     v1.4s, v17.4s, v9.4s
        add     v2.4s, v18.4s, v10.4s
        add     v3.4s, v19.4s, v11.4s
        ldp     q24, q25, [sp, #64]
        ldp     q26, q27, [sp, #96]
        stp     q16, q17, [sp, #320]
        stp     q18, q19, [sp, #352]
        add     v24.4s, v24.4s, v0.4s
        add     v25.4s, v25.4s, v1.4s
        add     v26.4s, v26.4s, v2.4s
        add     v27.4s, v27.4s, v3.4s
        and     v4.16b, v0.16b, v30.16b
        and     v5.16b, v1.16b, v30.16b
        and     v6.16b, v2.16b, v30.16b
        and     v7.16b, v3.16b, v30.16b
        orr     v0.16b, v0.16b, v31.16b
        orr     v1.16b, v1.16b, v31.16b
        orr     v2.16b, v2.16b, v31.16b
        orr     v3.16b, v3.16b, v31.16b
        ushl    v4.4s, v24.4s, v4.4s
        ushl    v5.4s, v25.4s, v5.4s
        ushl    v6.4s, v26.4s, v6.4s
        ushl    v7.4s, v27.4s, v7.4s
        ushl    v0.4s, v24.4s, v0.4s
        ushl    v1.4s, v25.4s, v1.4s
        ushl    v2.4s, v26.4s, v2.4s
        ushl    v3.4s, v27.4s, v3.4s
        orr     v16.16b, v4.16b, v0.16b
        orr     v17.16b, v5.16b, v1.16b
        orr     v18.16b, v6.16b, v2.16b
        orr     v19.16b, v7.16b, v3.16b
        ldp     q0, q1, [x10, #64]
        ldp     q2, q3, [x10, #96]
        add     v0.4s, v0.4s, v12.4s
        add     v1.4s, v1.4s, v13.4s
        add     v2.4s, v2.4s, v14.4s
        add     v3.4s, v3.4s, v15.4s
        add     v0.4s, v0.4s, v20.4s
        add     v1.4s, v1.4s, v21.4s
        add     v2.4s, v2.4s, v22.4s
        add     v3.4s, v3.4s, v23.4s
        shl     v12.4s, v0.4s, #3
        shl     v13.4s, v1.4s, #3
        shl     v14.4s, v2.4s, #3
        shl     v15.4s, v3.4s, #3
        sri     v12.4s, v0.4s, #29
        sri     v13.4s, v1.4s, #29
        sri     v14.4s, v2.4s, #29
        sri     v15.4s, v3.4s, #29
        stp     q12, q13, [x11, #64]
        stp     q14, q15, [x11, #96]
        add     v0.4s, v20.4s, v12.4s
        add     v1.4s, v21.4s, v13.4s
        add     v2.4s, v22.4s, v14.4s
        add     v3.4s, v23.4s, v15.4s
        ldp     q24, q25, [sp, #128]
        ldp     q26, q27, [sp, #160]
        stp     q20, q21, [sp, #384]
        stp     q22, q23, [sp, #416]
        add     v24.4s, v24.4s, v0.4s
        add     v25.4s, v25.4s, v1.4s
        add     v26.4s, v26.4s, v2.4s
        add     v27.4s, v27.4s, v3.4s
        and     v4.16b, v0.16b, v30.16b
        and     v5.16b, v1.16b, v30.16b
        and     v6.16b, v2.16b, v30.16b
        and     v7.16b, v3.16b, v30.16b
        orr     v0.16b, v0.16b, v31.16b
        orr     v1.16b, v1.16b, v31.16b
        orr     v2.16b, v2.16b, v31.16b
        orr     v3.16b, v3.16b, v31.16b
        ushl    v4.4s, v24.4s, v4.4s
        ushl    v5.4s, v25.4s, v5.4s
        ushl    v6.4s, v26.4s, v6.4s
        ushl    v7.4s, v27.4s, v7.4s
        ushl    v0.4s, v24.4s, v0.4s
        ushl    v1.4s, v25.4s, v1.4s
        ushl    v2.4s, v26.4s, v2.4s
        ushl    v3.4s, v27.4s, v3.4s
        orr     v20.16b, v4.16b, v0.16b
        orr     v21.16b, v5.16b, v1.16b
        orr     v22.16b, v6.16b, v2.16b
        orr     v23.16b, v7.16b, v3.16b

        cmp     w3, #16
        csel    x11, x8, x11, eq           // k=52 uses S[0]
        csel    x10, x8, x10, eq
        // === STEP C: A += B_0, B_1 <<< ===
        ldp     q0, q1, [x10, #128]
        ldp     q2, q3, [x10, #160]
        add     v0.4s, v0.4s, v8.4s
        add     v1.4s, v1.4s, v9.4s
        add     v2.4s, v2.4s, v10.4s
        add     v3.4s, v3.4s, v11.4s
        add     v0.4s, v0.4s, v16.4s
        add     v1.4s, v1.4s, v17.4s
        add     v2.4s, v2.4s, v18.4s
        add     v3.4s, v3.4s, v19.4s
        shl     v8.4s, v0.4s, #3
        shl     v9.4s, v1.4s, #3
        shl     v10.4s, v2.4s, #3
        shl     v11.4s, v3.4s, #3
        sri     v8.4s, v0.4s, #29
        sri     v9.4s, v1.4s, #29
        sri     v10.4s, v2.4s, #29
        sri     v11.4s, v3.4s, #29
        stp     q8, q9, [x11, #128]
        stp     q10, q11, [x11, #160]
        add     v0.4s, v16.4s, v8.4s
        add     v1.4s, v17.4s, v9.4s
        add     v2.4s, v18.4s, v10.4s
        add     v3.4s, v19.4s, v11.4s
        ldp     q24, q25, [sp, #192]
        ldp     q26, q27, [sp, #224]
        stp     q16, q17, [sp, #64]
        stp     q18, q19, [sp, #96]
        add     v24.4s, v24.4s, v0.4s
        add     v25.4s, v25.4s, v1.4s
        add     v26.4s, v26.4s, v2.4s
        add     v27.4s, v27.4s, v3.4s
        and     v4.16b, v0.16b, v30.16b
        and     v5.16b, v1.16b, v30.16b
        and     v6.16b, v2.16b, v30.16b
        and     v7.16b, v3.16b, v30.16b
        orr     v0.16b, v0.16b, v31.16b
        orr     v1.16b, v1.16b, v31.16b
        orr     v2.16b, v2.16b, v31.16b
        orr     v3.16b, v3.16b, v31.16b
        ushl    v4.4s, v24.4s, v4.4s
        ushl    v5.4s, v25.4s, v5.4s
        ushl    v6.4s, v26.4s, v6.4s
        ushl    v7.4s, v27.4s, v7.4s
        ushl    v0.4s, v24.4s, v0.4s
        ushl    v1.4s, v25.4s, v1.4s
        ushl    v2.4s, v26.4s, v2.4s
        ushl    v3.4s, v27.4s, v3.4s
        orr     v16.16b, v4.16b, v0.16b
        orr     v17.16b, v5.16b, v1.16b
        orr     v18.16b, v6.16b, v2.16b
        orr     v19.16b, v7.16b, v3.16b
        ldp     q0, q1, [x10, #192]
        ldp     q2, q3, [x10, #224]
        add     v0.4s, v0.4s, v12.4s
        add     v1.4s, v1.4s, v13.4s
        add     v2.4s, v2.4s, v14.4s
        add     v3.4s, v3.4s, v15.4s
        add     v0.4s, v0.4s, v20.4s
        add     v1.4s, v1.4s, v21.4s
        add     v2.4s, v2.4s, v22.4s
        add     v3.4s, v3.4s, v23.4s
        shl     v12.4s, v0.4s, #3
        shl     v13.4s, v1.4s, #3
        shl     v14.4s, v2.4s, #3
        shl     v15.4s, v3.4s, #3
        sri     v12.4s, v0.4s, #29
        sri     v13.4s, v1.4s, #29
        sri     v14.4s, v2.4s, #29
        sri     v15.4s, v3.4s, #29
        stp     q12, q13, [x11, #192]
        stp     q14, q15, [x11, #224]
        add     v0.4s, v20.4s, v12.4s
        add     v1.4s, v21.4s, v13.4s
        add     v2.4s, v22.4s, v14.4s
        add     v3.4s, v23.4s, v15.4s
        ldp     q24, q25, [sp, #256]
        ldp     q26, q27, [sp, #288]
        stp     q20, q21, [sp, #128]
        stp     q22, q23, [sp, #160]
        add     v24.4s, v24.4s, v0.4s
        add     v25.4s, v25.4s, v1.4s
        add     v26.4s, v26.4s, v2.4s
        add     v27.4s, v27.4s, v3.4s
        and     v4.16b, v0.16b, v30.16b
        and     v5.16b, v1.16b, v30.16b
        and     v6.16b, v2.16b, v30.16b
        and     v7.16b, v3.16b, v30.16b
        orr     v0.16b, v0.16b, v31.16b
        orr     v1.16b, v1.16b, v31.16b
        orr     v2.16b, v2.16b, v31.16b
        orr     v3.16b, v3.16b, v31.16b
        ushl    v4.4s, v24.4s, v4.4s
        ushl    v5.4s, v25.4s, v5.4s
        ushl    v6.4s, v26.4s, v6.4s
        ushl    v7.4s, v27.4s, v7.4s
        ushl    v0.4s, v24.4s, v0.4s
        ushl    v1.4s, v25.4s, v1.4s
        ushl    v2.4s, v26.4s, v2.4s
        ushl    v3.4s, v27.4s, v3.4s
        orr     v20.16b, v4.16b, v0.16b
        orr     v21.16b, v5.16b, v1.16b
        orr     v22.16b, v6.16b, v2.16b
        orr     v23.16b, v7.16b, v3.16b

        add     x10, x10, #384
        add     x11, x11, #384
        cmp     w3, #24
        b.ne    L(round)

L(encrypt_setup):
        // save A + B_1 for the rare path's k=77
        add     v0.4s, v8.4s, v16.4s
        add     v1.4s, v9.4s, v17.4s
        add     v2.4s, v10.4s, v18.4s
        add     v3.4s, v11.4s, v19.4s
        stp     q0, q1, [sp, #448]
        stp     q2, q3, [sp, #480]
        add     v0.4s, v12.4s, v20.4s
        add     v1.4s, v13.4s, v21.4s
        add     v2.4s, v14.4s, v22.4s
        add     v3.4s, v15.4s, v23.4s
        stp     q0, q1, [sp, #512]
        stp     q2, q3, [sp, #544]
        // C = plain.lo + S[0], D = plain.hi + S[1]
        ldr     w16, [x0, #4]
        dup     v24.4s, w16
        ldr     w16, [x0, #0]
        dup     v25.4s, w16
        add     x10, sp, #576
        ldp     q0, q1, [x10, #0]
        ldp     q2, q3, [x10, #32]
        add     v8.4s, v0.4s, v24.4s
        add     v9.4s, v1.4s, v24.4s
        add     v10.4s, v2.4s, v24.4s
        add     v11.4s, v3.4s, v24.4s
        ldp     q0, q1, [x10, #128]
        ldp     q2, q3, [x10, #160]
        add     v16.4s, v0.4s, v25.4s
        add     v17.4s, v1.4s, v25.4s
        add     v18.4s, v2.4s, v25.4s
        add     v19.4s, v3.4s, v25.4s
        ldp     q0, q1, [x10, #64]
        ldp     q2, q3, [x10, #96]
        add     v12.4s, v0.4s, v24.4s
        add     v13.4s, v1.4s, v24.4s
        add     v14.4s, v2.4s, v24.4s
        add     v15.4s, v3.4s, v24.4s
        ldp     q0, q1, [x10, #192]
        ldp     q2, q3, [x10, #224]
        add     v20.4s, v0.4s, v25.4s
        add     v21.4s, v1.4s, v25.4s
        add     v22.4s, v2.4s, v25.4s
        add     v23.4s, v3.4s, v25.4s
        add     x10, x10, #256
        mov     w3, #0

        .p2align 4
L(encrypt):
        add     w3, w3, #1
        // C = ((C ^ D) <<< D) + S[2r]
        eor     v8.16b, v8.16b, v16.16b
        eor     v9.16b, v9.16b, v17.16b
        eor     v10.16b, v10.16b, v18.16b
        eor     v11.16b, v11.16b, v19.16b
        and     v4.16b, v16.16b, v30.16b
        and     v5.16b, v17.16b, v30.16b
        and     v6.16b, v18.16b, v30.16b
        and     v7.16b, v19.16b, v30.16b
        orr     v0.16b, v16.16b, v31.16b
        orr     v1.16b, v17.16b, v31.16b
        orr     v2.16b, v18.16b, v31.16b
        orr     v3.16b, v19.16b, v31.16b
        ushl    v4.4s, v8.4s, v4.4s
        ushl    v5.4s, v9.4s, v5.4s
        ushl    v6.4s, v10.4s, v6.4s
        ushl    v7.4s, v11.4s, v7.4s
        ushl    v0.4s, v8.4s, v0.4s
        ushl    v1.4s, v9.4s, v1.4s
        ushl    v2.4s, v10.4s, v2.4s
        ushl    v3.4s, v11.4s, v3.4s
        orr     v8.16b, v4.16b, v0.16b
        orr     v9.16b, v5.16b, v1.16b
        orr     v10.16b, v6.16b, v2.16b
        orr     v11.16b, v7.16b, v3.16b
        ldp     q0, q1, [x10, #0]
        ldp     q2, q3, [x10, #32]
        add     v8.4s, v8.4s, v0.4s
        add     v9.4s, v9.4s, v1.4s
        add     v10.4s, v10.4s, v2.4s
        add     v11.4s, v11.4s, v3.4s
        eor     v12.16b, v12.16b, v20.16b
        eor     v13.16b, v13.16b, v21.16b
        eor     v14.16b, v14.16b, v22.16b
        eor     v15.16b, v15.16b, v23.16b
        and     v4.16b, v20.16b, v30.16b
        and     v5.16b, v21.16b, v30.16b
        and     v6.16b, v22.16b, v30.16b
        and     v7.16b, v23.16b, v30.16b
        orr     v0.16b, v20.16b, v31.16b
        orr     v1.16b, v21.16b, v31.16b
        orr     v2.16b, v22.16b, v31.16b
        orr     v3.16b, v23.16b, v31.16b
        ushl    v4.4s, v12.4s, v4.4s
        ushl    v5.4s, v13.4s, v5.4s
        ushl    v6.4s, v14.4s, v6.4s
        ushl    v7.4s, v15.4s, v7.4s
        ushl    v0.4s, v12.4s, v0.4s
        ushl    v1.4s, v13.4s, v1.4s
        ushl    v2.4s, v14.4s, v2.4s
        ushl    v3.4s, v15.4s, v3.4s
        orr     v12.16b, v4.16b, v0.16b
        orr     v13.16b, v5.16b, v1.16b
        orr     v14.16b, v6.16b, v2.16b
        orr     v15.16b, v7.16b, v3.16b
        ldp     q0, q1, [x10, #64]
        ldp     q2, q3, [x10, #96]
        add     v12.4s, v12.4s, v0.4s
        add     v13.4s, v13.4s, v1.4s
        add     v14.4s, v14.4s, v2.4s
        add     v15.4s, v15.4s, v3.4s
        // D = ((D ^ C) <<< C) + S[2r+1]
        eor     v16.16b, v16.16b, v8.16b
        eor     v17.16b, v17.16b, v9.16b
        eor     v18.16b, v18.16b, v10.16b
        eor     v19.16b, v19.16b, v11.16b
        and     v4.16b, v8.16b, v30.16b
        and     v5.16b, v9.16b, v30.16b
        and     v6.16b, v10.16b, v30.16b
        and     v7.16b, v11.16b, v30.16b
        orr     v0.16b, v8.16b, v31.16b
        orr     v1.16b, v9.16b, v31.16b
        orr     v2.16b, v10.16b, v31.16b
        orr     v3.16b, v11.16b, v31.16b
        ushl    v4.4s, v16.4s, v4.4s
        ushl    v5.4s, v17.4s, v5.4s
        ushl    v6.4s, v18.4s, v6.4s
        ushl    v7.4s, v19.4s, v7.4s
        ushl    v0.4s, v16.4s, v0.4s
        ushl    v1.4s, v17.4s, v1.4s
        ushl    v2.4s, v18.4s, v2.4s
        ushl    v3.4s, v19.4s, v3.4s
        orr     v16.16b, v4.16b, v0.16b
        orr     v17.16b, v5.16b, v1.16b
        orr     v18.16b, v6.16b, v2.16b
        orr     v19.16b, v7.16b, v3.16b
        ldp     q0, q1, [x10, #128]
        ldp     q2, q3, [x10, #160]
        add     v16.4s, v16.4s, v0.4s
        add     v17.4s, v17.4s, v1.4s
        add     v18.4s, v18.4s, v2.4s
        add     v19.4s, v19.4s, v3.4s
        eor     v20.16b, v20.16b, v12.16b
        eor     v21.16b, v21.16b, v13.16b
        eor     v22.16b, v22.16b, v14.16b
        eor     v23.16b, v23.16b, v15.16b
        and     v4.16b, v12.16b, v30.16b
        and     v5.16b, v13.16b, v30.16b
        and     v6.16b, v14.16b, v30.16b
        and     v7.16b, v15.16b, v30.16b
        orr     v0.16b, v12.16b, v31.16b
        orr     v1.16b, v13.16b, v31.16b
        orr     v2.16b, v14.16b, v31.16b
        orr     v3.16b, v15.16b, v31.16b
        ushl    v4.4s, v20.4s, v4.4s
        ushl    v5.4s, v21.4s, v5.4s
        ushl    v6.4s, v22.4s, v6.4s
        ushl    v7.4s, v23.4s, v7.4s
        ushl    v0.4s, v20.4s, v0.4s
        ushl    v1.4s, v21.4s, v1.4s
        ushl    v2.4s, v22.4s, v2.4s
        ushl    v3.4s, v23.4s, v3.4s
        orr     v20.16b, v4.16b, v0.16b
        orr     v21.16b, v5.16b, v1.16b
        orr     v22.16b, v6.16b, v2.16b
        orr     v23.16b, v7.16b, v3.16b
        ldp     q0, q1, [x10, #192]
        ldp     q2, q3, [x10, #224]
        add     v20.4s, v20.4s, v0.4s
        add     v21.4s, v21.4s, v1.4s
        add     v22.4s, v22.4s, v2.4s
        add     v23.4s, v23.4s, v3.4s
        add     x10, x10, #256
        cmp     w3, #11
        b.ne    L(encrypt)

L(encrypt_end):
        // final half-round: C = ((C ^ D) <<< D) + S[24]; x10 = &S[24]
        eor     v8.16b, v8.16b, v16.16b
        eor     v9.16b, v9.16b, v17.16b
        eor     v10.16b, v10.16b, v18.16b
        eor     v11.16b, v11.16b, v19.16b
        and     v4.16b, v16.16b, v30.16b
        and     v5.16b, v17.16b, v30.16b
        and     v6.16b, v18.16b, v30.16b
        and     v7.16b, v19.16b, v30.16b
        orr     v0.16b, v16.16b, v31.16b
        orr     v1.16b, v17.16b, v31.16b
        orr     v2.16b, v18.16b, v31.16b
        orr     v3.16b, v19.16b, v31.16b
        ushl    v4.4s, v8.4s, v4.4s
        ushl    v5.4s, v9.4s, v5.4s
        ushl    v6.4s, v10.4s, v6.4s
        ushl    v7.4s, v11.4s, v7.4s
        ushl    v0.4s, v8.4s, v0.4s
        ushl    v1.4s, v9.4s, v1.4s
        ushl    v2.4s, v10.4s, v2.4s
        ushl    v3.4s, v11.4s, v3.4s
        orr     v8.16b, v4.16b, v0.16b
        orr     v9.16b, v5.16b, v1.16b
        orr     v10.16b, v6.16b, v2.16b
        orr     v11.16b, v7.16b, v3.16b
        ldp     q0, q1, [x10, #0]
        ldp     q2, q3, [x10, #32]
        add     v8.4s, v8.4s, v0.4s
        add     v9.4s, v9.4s, v1.4s
        add     v10.4s, v10.4s, v2.4s
        add     v11.4s, v11.4s, v3.4s
        eor     v12.16b, v12.16b, v20.16b
        eor     v13.16b, v13.16b, v21.16b
        eor     v14.16b, v14.16b, v22.16b
        eor     v15.16b, v15.16b, v23.16b
        and     v4.16b, v20.16b, v30.16b
        and     v5.16b, v21.16b, v30.16b
        and     v6.16b, v22.16b, v30.16b
        and     v7.16b, v23.16b, v30.16b
        orr     v0.16b, v20.16b, v31.16b
        orr     v1.16b, v21.16b, v31.16b
        orr     v2.16b, v22.16b, v31.16b
        orr     v3.16b, v23.16b, v31.16b
        ushl    v4.4s, v12.4s, v4.4s
        ushl    v5.4s, v13.4s, v5.4s
        ushl    v6.4s, v14.4s, v6.4s
        ushl    v7.4s, v15.4s, v7.4s
        ushl    v0.4s, v12.4s, v0.4s
        ushl    v1.4s, v13.4s, v1.4s
        ushl    v2.4s, v14.4s, v2.4s
        ushl    v3.4s, v15.4s, v3.4s
        orr     v12.16b, v4.16b, v0.16b
        orr     v13.16b, v5.16b, v1.16b
        orr     v14.16b, v6.16b, v2.16b
        orr     v15.16b, v7.16b, v3.16b
        ldp     q0, q1, [x10, #64]
        ldp     q2, q3, [x10, #96]
        add     v12.4s, v12.4s, v0.4s
        add     v13.4s, v13.4s, v1.4s
        add     v14.4s, v14.4s, v2.4s
        add     v15.4s, v15.4s, v3.4s

L(check_key):
        cmeq    v0.4s, v8.4s, v28.4s
        cmeq    v1.4s, v9.4s, v28.4s
        cmeq    v2.4s, v10.4s, v28.4s
        cmeq    v3.4s, v11.4s, v28.4s
        cmeq    v4.4s, v12.4s, v28.4s
        cmeq    v5.4s, v13.4s, v28.4s
        cmeq    v6.4s, v14.4s, v28.4s
        cmeq    v7.4s, v15.4s, v28.4s
        orr     v24.16b, v0.16b, v1.16b
        orr     v25.16b, v2.16b, v3.16b
        orr     v26.16b, v4.16b, v5.16b
        orr     v27.16b, v6.16b, v7.16b
        orr     v24.16b, v24.16b, v25.16b
        orr     v26.16b, v26.16b, v27.16b
        orr     v24.16b, v24.16b, v26.16b
        umaxv   s24, v24.4s
        fmov    w16, s24
        cbnz    w16, L(check_hi)

L(next_key):
        subs    w9, w9, #1
        b.eq    L(finished_nothing)
        add     w13, w13, #32
        tbnz    w13, #8, L(carry_hi)
        str     w13, [x0, #16]
        b       L(inner_loop)
L(carry_hi):
        and     w13, w13, #0xFF
        str     w13, [x0, #16]
        rev     w14, w14
        adds    w14, w14, #1
        rev     w14, w14
        b.cc    L(mid_loop)
        rev     w15, w15
        add     w15, w15, #1
        rev     w15, w15
        b       L(outer_loop)

L(check_hi):
        // rare path (about 1 batch in 2^27); must not touch the precompute registers
        add     x8, x10, #128              // &S[25] (x10 = &S[24])
        adr     x16, L(weights)
        ldr     q24, [x16]
        and     v25.16b, v0.16b, v24.16b
        addv    s25, v25.4s
        fmov    w10, s25
        and     v25.16b, v1.16b, v24.16b
        addv    s25, v25.4s
        fmov    w11, s25
        orr     w10, w10, w11, lsl #4
        and     v25.16b, v2.16b, v24.16b
        addv    s25, v25.4s
        fmov    w11, s25
        orr     w10, w10, w11, lsl #8
        and     v25.16b, v3.16b, v24.16b
        addv    s25, v25.4s
        fmov    w11, s25
        orr     w10, w10, w11, lsl #12
        and     v25.16b, v4.16b, v24.16b
        addv    s25, v25.4s
        fmov    w11, s25
        orr     w10, w10, w11, lsl #16
        and     v25.16b, v5.16b, v24.16b
        addv    s25, v25.4s
        fmov    w11, s25
        orr     w10, w10, w11, lsl #20
        and     v25.16b, v6.16b, v24.16b
        addv    s25, v25.4s
        fmov    w11, s25
        orr     w10, w10, w11, lsl #24
        and     v25.16b, v7.16b, v24.16b
        addv    s25, v25.4s
        fmov    w11, s25
        orr     w10, w10, w11, lsl #28
        // w10 = 32-bit low-word match mask, bit n = key n

        // k=77: S[25] = (S[25] + A + B_1) <<< 3, then D = ((D ^ C) <<< C) + S[25]
        ldp     q24, q25, [x8, #0]
        ldp     q26, q27, [x8, #32]
        ldp     q0, q1, [sp, #448]
        ldp     q2, q3, [sp, #480]
        add     v0.4s, v0.4s, v24.4s
        add     v1.4s, v1.4s, v25.4s
        add     v2.4s, v2.4s, v26.4s
        add     v3.4s, v3.4s, v27.4s
        shl     v24.4s, v0.4s, #3
        shl     v25.4s, v1.4s, #3
        shl     v26.4s, v2.4s, #3
        shl     v27.4s, v3.4s, #3
        sri     v24.4s, v0.4s, #29
        sri     v25.4s, v1.4s, #29
        sri     v26.4s, v2.4s, #29
        sri     v27.4s, v3.4s, #29
        eor     v16.16b, v16.16b, v8.16b
        eor     v17.16b, v17.16b, v9.16b
        eor     v18.16b, v18.16b, v10.16b
        eor     v19.16b, v19.16b, v11.16b
        and     v4.16b, v8.16b, v30.16b
        and     v5.16b, v9.16b, v30.16b
        and     v6.16b, v10.16b, v30.16b
        and     v7.16b, v11.16b, v30.16b
        orr     v0.16b, v8.16b, v31.16b
        orr     v1.16b, v9.16b, v31.16b
        orr     v2.16b, v10.16b, v31.16b
        orr     v3.16b, v11.16b, v31.16b
        ushl    v4.4s, v16.4s, v4.4s
        ushl    v5.4s, v17.4s, v5.4s
        ushl    v6.4s, v18.4s, v6.4s
        ushl    v7.4s, v19.4s, v7.4s
        ushl    v0.4s, v16.4s, v0.4s
        ushl    v1.4s, v17.4s, v1.4s
        ushl    v2.4s, v18.4s, v2.4s
        ushl    v3.4s, v19.4s, v3.4s
        orr     v16.16b, v4.16b, v0.16b
        orr     v17.16b, v5.16b, v1.16b
        orr     v18.16b, v6.16b, v2.16b
        orr     v19.16b, v7.16b, v3.16b
        add     v16.4s, v16.4s, v24.4s
        add     v17.4s, v17.4s, v25.4s
        add     v18.4s, v18.4s, v26.4s
        add     v19.4s, v19.4s, v27.4s
        ldp     q24, q25, [x8, #64]
        ldp     q26, q27, [x8, #96]
        ldp     q0, q1, [sp, #512]
        ldp     q2, q3, [sp, #544]
        add     v0.4s, v0.4s, v24.4s
        add     v1.4s, v1.4s, v25.4s
        add     v2.4s, v2.4s, v26.4s
        add     v3.4s, v3.4s, v27.4s
        shl     v24.4s, v0.4s, #3
        shl     v25.4s, v1.4s, #3
        shl     v26.4s, v2.4s, #3
        shl     v27.4s, v3.4s, #3
        sri     v24.4s, v0.4s, #29
        sri     v25.4s, v1.4s, #29
        sri     v26.4s, v2.4s, #29
        sri     v27.4s, v3.4s, #29
        eor     v20.16b, v20.16b, v12.16b
        eor     v21.16b, v21.16b, v13.16b
        eor     v22.16b, v22.16b, v14.16b
        eor     v23.16b, v23.16b, v15.16b
        and     v4.16b, v12.16b, v30.16b
        and     v5.16b, v13.16b, v30.16b
        and     v6.16b, v14.16b, v30.16b
        and     v7.16b, v15.16b, v30.16b
        orr     v0.16b, v12.16b, v31.16b
        orr     v1.16b, v13.16b, v31.16b
        orr     v2.16b, v14.16b, v31.16b
        orr     v3.16b, v15.16b, v31.16b
        ushl    v4.4s, v20.4s, v4.4s
        ushl    v5.4s, v21.4s, v5.4s
        ushl    v6.4s, v22.4s, v6.4s
        ushl    v7.4s, v23.4s, v7.4s
        ushl    v0.4s, v20.4s, v0.4s
        ushl    v1.4s, v21.4s, v1.4s
        ushl    v2.4s, v22.4s, v2.4s
        ushl    v3.4s, v23.4s, v3.4s
        orr     v20.16b, v4.16b, v0.16b
        orr     v21.16b, v5.16b, v1.16b
        orr     v22.16b, v6.16b, v2.16b
        orr     v23.16b, v7.16b, v3.16b
        add     v20.4s, v20.4s, v24.4s
        add     v21.4s, v21.4s, v25.4s
        add     v22.4s, v22.4s, v26.4s
        add     v23.4s, v23.4s, v27.4s
        cmeq    v16.4s, v16.4s, v29.4s
        cmeq    v17.4s, v17.4s, v29.4s
        cmeq    v18.4s, v18.4s, v29.4s
        cmeq    v19.4s, v19.4s, v29.4s
        cmeq    v20.4s, v20.4s, v29.4s
        cmeq    v21.4s, v21.4s, v29.4s
        cmeq    v22.4s, v22.4s, v29.4s
        cmeq    v23.4s, v23.4s, v29.4s
        adr     x16, L(weights)
        ldr     q24, [x16]
        and     v25.16b, v16.16b, v24.16b
        addv    s25, v25.4s
        fmov    w11, s25
        and     v25.16b, v17.16b, v24.16b
        addv    s25, v25.4s
        fmov    w16, s25
        orr     w11, w11, w16, lsl #4
        and     v25.16b, v18.16b, v24.16b
        addv    s25, v25.4s
        fmov    w16, s25
        orr     w11, w11, w16, lsl #8
        and     v25.16b, v19.16b, v24.16b
        addv    s25, v25.4s
        fmov    w16, s25
        orr     w11, w11, w16, lsl #12
        and     v25.16b, v20.16b, v24.16b
        addv    s25, v25.4s
        fmov    w16, s25
        orr     w11, w11, w16, lsl #16
        and     v25.16b, v21.16b, v24.16b
        addv    s25, v25.4s
        fmov    w16, s25
        orr     w11, w11, w16, lsl #20
        and     v25.16b, v22.16b, v24.16b
        addv    s25, v25.4s
        fmov    w16, s25
        orr     w11, w11, w16, lsl #24
        and     v25.16b, v23.16b, v24.16b
        addv    s25, v25.4s
        fmov    w16, s25
        orr     w11, w11, w16, lsl #28
        ands    w16, w11, w10              // keys matching both words
        b.ne    L(save_found_key)

        // partial match(es) only: count them, record the last (as the reference does)
        fmov    s24, w10
        cnt     v24.8b, v24.8b
        addv    b24, v24.8b
        fmov    w11, s24
        ldr     w16, [x0, #28]
        add     w16, w16, w11
        str     w16, [x0, #28]
        clz     w16, w10
        mov     w11, #31
        sub     w16, w11, w16              // index of the highest set bit
        add     w16, w16, w13
        str     w16, [x0, #32]
        str     w14, [x0, #36]
        str     w15, [x0, #40]
        b       L(next_key)

L(save_found_key):
        // f = first full match; count partial matches at or before f, record f
        rbit    w16, w16
        clz     w3, w16                    // w3 = f
        mov     w11, #2
        lsl     w11, w11, w3
        sub     w11, w11, #1               // mask of keys 0..f (f=31 gives all ones)
        and     w10, w10, w11
        fmov    s24, w10
        cnt     v24.8b, v24.8b
        addv    b24, v24.8b
        fmov    w11, s24
        ldr     w16, [x0, #28]
        add     w16, w16, w11
        str     w16, [x0, #28]
        add     w16, w3, w13
        str     w16, [x0, #32]
        str     w14, [x0, #36]
        str     w15, [x0, #40]
        // *iterations = keys before the winning one
        ldr     w16, [x1]
        lsr     w16, w16, #5
        sub     w16, w16, w9
        lsl     w16, w16, #5
        add     w16, w16, w3
        str     w16, [x1]
        str     w14, [x0, #20]
        str     w15, [x0, #24]
        mov     w0, #2                     // RESULT_FOUND
        b       L(finished)

L(finished_nothing):
        add     w13, w13, #32
        ubfx    w16, w13, #8, #1           // carry out of the L0hi byte
        and     w13, w13, #0xFF
        rev     w14, w14
        rev     w15, w15
        adds    w14, w14, w16
        adc     w15, w15, wzr
        rev     w14, w14
        rev     w15, w15
        str     w13, [x0, #16]
        str     w14, [x0, #20]
        str     w15, [x0, #24]
        mov     w0, #1                     // RESULT_NOTHING

L(finished):
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        mov     sp, x29
        ldp     x29, x30, [sp], #16
        ret
#if defined(__ELF__)
        .size   rc5_72_unit_func_neon_mix, . - rc5_72_unit_func_neon_mix
#endif

        // constants in .text so adr reaches them on ELF and Mach-O
        .p2align 6
L(lanes):
        .word   0, 1, 2, 3
        .word   4, 5, 6, 7
        .word   8, 9, 10, 11
        .word   12, 13, 14, 15
        .word   16, 17, 18, 19
        .word   20, 21, 22, 23
        .word   24, 25, 26, 27
        .word   28, 29, 30, 31
L(weights):
        .word   1, 2, 4, 8
// pass-1 subkeys, one row per subkey; IS0 after IS25 so the pointer reads S[0] at k=26
        .p2align 6
L(IS3):
        .word   0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E
        .word   0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E, 0x9287BE8E
L(IS4):
        .word   0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847
        .word   0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847, 0x30BF3847
L(IS5):
        .word   0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200
        .word   0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200, 0xCEF6B200
L(IS6):
        .word   0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9
        .word   0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9, 0x6D2E2BB9
L(IS7):
        .word   0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572
        .word   0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572, 0x0B65A572
L(IS8):
        .word   0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B
        .word   0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B, 0xA99D1F2B
L(IS9):
        .word   0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4
        .word   0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4, 0x47D498E4
L(IS10):
        .word   0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D
        .word   0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D, 0xE60C129D
L(IS11):
        .word   0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56
        .word   0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56, 0x84438C56
L(IS12):
        .word   0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F
        .word   0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F, 0x227B060F
L(IS13):
        .word   0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8
        .word   0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8, 0xC0B27FC8
L(IS14):
        .word   0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981
        .word   0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981, 0x5EE9F981
L(IS15):
        .word   0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A
        .word   0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A, 0xFD21733A
L(IS16):
        .word   0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3
        .word   0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3, 0x9B58ECF3
L(IS17):
        .word   0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC
        .word   0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC, 0x399066AC
L(IS18):
        .word   0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065
        .word   0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065, 0xD7C7E065
L(IS19):
        .word   0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E
        .word   0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E, 0x75FF5A1E
L(IS20):
        .word   0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7
        .word   0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7, 0x1436D3D7
L(IS21):
        .word   0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90
        .word   0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90, 0xB26E4D90
L(IS22):
        .word   0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749
        .word   0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749, 0x50A5C749
L(IS23):
        .word   0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102
        .word   0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102, 0xEEDD4102
L(IS24):
        .word   0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB
        .word   0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB, 0x8D14BABB
L(IS25):
        .word   0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474
        .word   0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474, 0x2B4C3474
L(IS0):
        .word   0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D
        .word   0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D

#if defined(__ELF__)
        .section .note.GNU-stack,"",%progbits
#endif
