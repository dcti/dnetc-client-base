// Multipipe Optimized NEON Implementation of Key Analysis
// Brought to you by Sunset Ash
// Needs no C++ shim, unlike ScalarFusion, and should be considerably faster on most hardware
// 2-pipe

.text
.global rc5_72_unit_func_monika_2pipe
rc5_72_unit_func_monika_2pipe:
        ldr w16, [x1]
        asr w16, w16, #1
        stp d8, d9, [sp, #-16]!
        stp d10, d11, [sp, #-16]!
        stp d12, d13, [sp, #-16]!
        stp d14, d15, [sp, #-16]!
        ldr w2, constant_p
        ldr w3, constant_q

loopStart:
        sub w16, w16, #1
        mov w9, #29
        sub w7, wzr, w9
        mov w5, #-32
        //add w5, w9, #3 // Use this to hold 32 later.
        //dup v30.4s, w7
        ldr w10, [x0, #16] // high
        ldr w14, [x0, #20] // mid
        ldr w15, [x0, #24] // lo

        dup v0.2s, w10
        mov w10, 0x1f
        // This is an ugly way of adding 1-3 through elements 1-3. Might be
        // worth investigating doing it as a load instead of a runtime calc
        // at some point, although we experimented with something similar in
        // SF and it hurt perf by ~20%.
        dup v1.2s, wzr
        mov w11, #1
        ins v1.2s[1], w11
        add v0.2s, v0.2s, v1.2s
        // good through here - KS

        dup v1.2s, w14 // dump mid in the vreg
        dup v2.2s, w15 // lo too

        // good through here

        dup v3.2s, w2 // KEY_INIT(0)

        add w4, w2, w3 // KEY_INIT(1)
        dup v4.2s, w4

        mov w4, #2
        madd w4, w3, w4, w2
        dup v5.2s, w4 // 2

        mov w4, #3
        madd w4, w3, w4, w2
        dup v6.2s, w4 // 3

        mov w4, #4
        madd w4, w3, w4, w2
        dup v7.2s, w4

        mov w4, #5
        madd w4, w3, w4, w2
        dup v8.2s, w4

        mov w4, #6
        madd w4, w3, w4, w2
        dup v9.2s, w4

        mov w4, #7
        madd w4, w3, w4, w2
        dup v10.2s, w4

        mov w4, #8
        madd w4, w3, w4, w2
        dup v11.2s, w4

        mov w4, #9
        madd w4, w3, w4, w2
        dup v12.2s, w4

        mov w4, #10
        madd w4, w3, w4, w2
        dup v13.2s, w4

        mov w4, #11
        madd w4, w3, w4, w2
        dup v14.2s, w4

        mov w4, #12
        madd w4, w3, w4, w2
        dup v15.2s, w4

        mov w4, #13
        madd w4, w3, w4, w2
        dup v16.2s, w4

        mov w4, #14
        madd w4, w3, w4, w2
        dup v17.2s, w4

        mov w4, #15
        madd w4, w3, w4, w2
        dup v18.2s, w4

        mov w4, #16
        madd w4, w3, w4, w2
        dup v19.2s, w4

        mov w4, #17
        madd w4, w3, w4, w2
        dup v20.2s, w4

        mov w4, #18
        madd w4, w3, w4, w2
        dup v21.2s, w4

        mov w4, #19
        madd w4, w3, w4, w2
        dup v22.2s, w4

        mov w4, #20
        madd w4, w3, w4, w2
        dup v23.2s, w4

        mov w4, #21
        madd w4, w3, w4, w2
        dup v24.2s, w4

        mov w4, #22
        madd w4, w3, w4, w2
        dup v25.2s, w4

        mov w4, #23
        madd w4, w3, w4, w2
        dup v26.2s, w4

        mov w4, #24
        madd w4, w3, w4, w2
        dup v27.2s, w4

        mov w4, #25
        madd w4, w3, w4, w2
        dup v28.2s, w4

#define VROR(v, vb) \
        dup v30.2s, w7; \
        shl v31.2s, v, #3; \
        ushl v29.2s, v, v30.2s; \
        eor vb, v29.8b, v31.8b;

#define VROLV(firstV, secondV, firstVB, secondVB) \
        dup v30.2s, w10; \
        and v30.8b, v30.8b, secondVB; \
        dup v31.2s, w5; \
        add v31.2s, v31.2s, v30.2s; \
        ushl v31.2s, firstV, v31.2s; \
        ushl firstV, firstV, v30.2s; \
        eor firstVB, firstVB, v31.8b;

#define ROTL_BLOCK_i0_j1 \
        add v29.2s, v2.2s, v28.2s; \
        add v3.2s, v29.2s, v3.2s; \
        VROR(v3.2s, v3.8b); \
        add v29.2s, v3.2s, v2.2s; \
        add v1.2s, v1.2s, v29.2s; \
        VROLV(v1.2s, v29.2s, v1.8b, v29.8b);

#define ROTL_BLOCK_i0_j2 \
        add v29.2s, v1.2s, v28.2s; \
        add v3.2s, v29.2s, v3.2s; \
        VROR(v3.2s, v3.8b); \
        add v29.2s, v1.2s, v3.2s; \
        add v0.2s, v0.2s, v29.2s; \
        VROLV(v0.2s, v29.2s, v0.8b, v29.8b);

#define ROTL_BLOCK_j0(highReg, lowReg, highRegB) \
        add highReg, highReg, lowReg; \
        add highReg, highReg, v0.2s; \
        VROR(highReg, highRegB); \
        add v29.2s, highReg, v0.2s; \
        add v2.2s, v2.2s, v29.2s; \
        VROLV(v2.2s, v29.2s, v2.8b, v29.8b);

#define ROTL_BLOCK_j1(highReg, lowReg, highRegB) \
        add highReg, highReg, lowReg; \
        add highReg, highReg, v2.2s; \
        VROR(highReg, highRegB); \
        add v29.2s, highReg, v2.2s; \
        add v1.2s, v1.2s, v29.2s; \
        VROLV(v1.2s, v29.2s, v1.8b, v29.8b);

#define ROTL_BLOCK_j2(highReg, lowReg, highRegB) \
        add highReg, highReg, lowReg; \
        add highReg, highReg, v1.2s; \
        VROR(highReg, highRegB); \
        add v29.2s, highReg, v1.2s; \
        add v0.2s, v0.2s, v29.2s; \
        VROLV(v0.2s, v29.2s, v0.8b, v29.8b);

VROR(v3.2s, v3.8b); // LOC 122 in r72ansi4.cpp - bookmark
add v2.2s, v2.2s, v3.2s; // 2-pipe conversion bookmark
VROLV(v2.2s, v3.2s, v2.8b, v3.8b);

ROTL_BLOCK_j1(v4.2s, v3.2s, v4.8b);
ROTL_BLOCK_j2(v5.2s, v4.2s, v5.8b);
ROTL_BLOCK_j0(v6.2s, v5.2s, v6.8b);
ROTL_BLOCK_j1(v7.2s, v6.2s, v7.8b);
ROTL_BLOCK_j2(v8.2s, v7.2s, v8.8b);
ROTL_BLOCK_j0(v9.2s, v8.2s, v9.8b);
ROTL_BLOCK_j1(v10.2s, v9.2s, v10.8b);
ROTL_BLOCK_j2(v11.2s, v10.2s, v11.8b);
ROTL_BLOCK_j0(v12.2s, v11.2s, v12.8b);
ROTL_BLOCK_j1(v13.2s, v12.2s, v13.8b);
ROTL_BLOCK_j2(v14.2s, v13.2s, v14.8b);
ROTL_BLOCK_j0(v15.2s, v14.2s, v15.8b);
ROTL_BLOCK_j1(v16.2s, v15.2s, v16.8b);
ROTL_BLOCK_j2(v17.2s, v16.2s, v17.8b);
ROTL_BLOCK_j0(v18.2s, v17.2s, v18.8b);
ROTL_BLOCK_j1(v19.2s, v18.2s, v19.8b);
ROTL_BLOCK_j2(v20.2s, v19.2s, v20.8b);
ROTL_BLOCK_j0(v21.2s, v20.2s, v21.8b);
ROTL_BLOCK_j1(v22.2s, v21.2s, v22.8b);
ROTL_BLOCK_j2(v23.2s, v22.2s, v23.8b);
ROTL_BLOCK_j0(v24.2s, v23.2s, v24.8b);
ROTL_BLOCK_j1(v25.2s, v24.2s, v25.8b);
ROTL_BLOCK_j2(v26.2s, v25.2s, v26.8b);
ROTL_BLOCK_j0(v27.2s, v26.2s, v27.8b);
ROTL_BLOCK_j1(v28.2s, v27.2s, v28.8b);

ROTL_BLOCK_i0_j2;

ROTL_BLOCK_j0(v4.2s, v3.2s, v4.8b);
ROTL_BLOCK_j1(v5.2s, v4.2s, v5.8b);
ROTL_BLOCK_j2(v6.2s, v5.2s, v6.8b);
ROTL_BLOCK_j0(v7.2s, v6.2s, v7.8b);
ROTL_BLOCK_j1(v8.2s, v7.2s, v8.8b);
ROTL_BLOCK_j2(v9.2s, v8.2s, v9.8b);
ROTL_BLOCK_j0(v10.2s, v9.2s, v10.8b);
ROTL_BLOCK_j1(v11.2s, v10.2s, v11.8b);
ROTL_BLOCK_j2(v12.2s, v11.2s, v12.8b);
ROTL_BLOCK_j0(v13.2s, v12.2s, v13.8b);
ROTL_BLOCK_j1(v14.2s, v13.2s, v14.8b);
ROTL_BLOCK_j2(v15.2s, v14.2s, v15.8b);
ROTL_BLOCK_j0(v16.2s, v15.2s, v16.8b);
ROTL_BLOCK_j1(v17.2s, v16.2s, v17.8b);
ROTL_BLOCK_j2(v18.2s, v17.2s, v18.8b);
ROTL_BLOCK_j0(v19.2s, v18.2s, v19.8b);
ROTL_BLOCK_j1(v20.2s, v19.2s, v20.8b);
ROTL_BLOCK_j2(v21.2s, v20.2s, v21.8b);
ROTL_BLOCK_j0(v22.2s, v21.2s, v22.8b);
ROTL_BLOCK_j1(v23.2s, v22.2s, v23.8b);
ROTL_BLOCK_j2(v24.2s, v23.2s, v24.8b);
ROTL_BLOCK_j0(v25.2s, v24.2s, v25.8b);
ROTL_BLOCK_j1(v26.2s, v25.2s, v26.8b);
ROTL_BLOCK_j2(v27.2s, v26.2s, v27.8b);
ROTL_BLOCK_j0(v28.2s, v27.2s, v28.8b);

ROTL_BLOCK_i0_j1;

ROTL_BLOCK_j2(v4.2s, v3.2s, v4.8b);
ROTL_BLOCK_j0(v5.2s, v4.2s, v5.8b);
ROTL_BLOCK_j1(v6.2s, v5.2s, v6.8b);
ROTL_BLOCK_j2(v7.2s, v6.2s, v7.8b);
ROTL_BLOCK_j0(v8.2s, v7.2s, v8.8b);
ROTL_BLOCK_j1(v9.2s, v8.2s, v9.8b);
ROTL_BLOCK_j2(v10.2s, v9.2s, v10.8b);
ROTL_BLOCK_j0(v11.2s, v10.2s, v11.8b);
ROTL_BLOCK_j1(v12.2s, v11.2s, v12.8b);
ROTL_BLOCK_j2(v13.2s, v12.2s, v13.8b);
ROTL_BLOCK_j0(v14.2s, v13.2s, v14.8b);
ROTL_BLOCK_j1(v15.2s, v14.2s, v15.8b);
ROTL_BLOCK_j2(v16.2s, v15.2s, v16.8b);
ROTL_BLOCK_j0(v17.2s, v16.2s, v17.8b);
ROTL_BLOCK_j1(v18.2s, v17.2s, v18.8b);
ROTL_BLOCK_j2(v19.2s, v18.2s, v19.8b);
ROTL_BLOCK_j0(v20.2s, v19.2s, v20.8b);
ROTL_BLOCK_j1(v21.2s, v20.2s, v21.8b);
ROTL_BLOCK_j2(v22.2s, v21.2s, v22.8b);
ROTL_BLOCK_j0(v23.2s, v22.2s, v23.8b);
ROTL_BLOCK_j1(v24.2s, v23.2s, v24.8b);
ROTL_BLOCK_j2(v25.2s, v24.2s, v25.8b);
ROTL_BLOCK_j0(v26.2s, v25.2s, v26.8b);
ROTL_BLOCK_j1(v27.2s, v26.2s, v27.8b);
ROTL_BLOCK_j2(v28.2s, v27.2s, v28.8b);

ldr w11, [x0]; // plain.hi
ldr w12, [x0, #4]; // plain.lo

dup v0.2s, w12;
add v0.2s, v0.2s, v3.2s; // A

dup v1.2s, w11;
add v1.2s, v1.2s, v4.2s; // B

#define FINAL_BLOCK(lowerReg, upperReg) \
        eor v0.8b, v0.8b, v1.8b; \
        VROLV(v0.2s, v1.2s, v0.8b, v1.8b); \
        add v0.2s, v0.2s, lowerReg; \
        eor v1.8b, v1.8b, v0.8b; \
        VROLV(v1.2s, v0.2s, v1.8b, v0.8b); \
        add v1.2s, v1.2s, upperReg;

FINAL_BLOCK(v5.2s, v6.2s);
FINAL_BLOCK(v7.2s, v8.2s);
FINAL_BLOCK(v9.2s, v10.2s);
FINAL_BLOCK(v11.2s, v12.2s);
FINAL_BLOCK(v13.2s, v14.2s);
FINAL_BLOCK(v15.2s, v16.2s);
FINAL_BLOCK(v17.2s, v18.2s);
FINAL_BLOCK(v19.2s, v20.2s);
FINAL_BLOCK(v21.2s, v22.2s);
FINAL_BLOCK(v23.2s, v24.2s);
FINAL_BLOCK(v25.2s, v26.2s);
FINAL_BLOCK(v27.2s, v28.2s);

//str q0, [x1]
//str q1, [x1, #16]

// kira c++ shim removal starts here
// notes:
// cypher.hi - #8
// cypher.lo - #12
// L0.hi - #16
// L0.mid - #20
// L0.lo - #24
// check.count - #28
// check.hi - #32
// check.mid - #36
// check.lo - #40
umov w5, v0.2s[0];
ldr q3, [x0, #8];
// v3.4s[0] = cypher.hi
// v3.4s[1] = cypher.lo
// v3.4s[2] = L0.hi
// v3.4s[3] = L0.mid
umov w13, v3.2s[1];
//ldr w13, [x0, #12];
cmp w5, w13;
b.ne A2Check;

ldr w6, [x0, #28];
add w6, w6, #1;
str w6, [x0, #28];

umov x6, v3.2d[1];
str x6, [x0, #32];

//str w4, [x0, #32]; removed to use store pair

ldr w4, [x0, #24];
str w4, [x0, #40];

umov w17, v3.2s[0];
umov w6, v1.2s[0];
cmp w6, w17;
b.ne A2Check;

// whoaaa, we found a result!
ldr w17, [x1];
add w16, w16, #1;
lsl w16, w16, #1;
sub w17, w17, w16;
str w17, [x1];
mov x0, #2;

b programEnd;

A2Check:

umov w5, v0.2s[1];
dup v6.4s, wzr;
mov w17, #1;
ins v6.4s[2], w17;

add v3.4s, v3.4s, v6.4s;

umov w13, v3.4s[1];
//ldr w13, [x0, #12];
cmp w5, w13;
b.ne keyTransform;

ldr w6, [x0, #28];
add w6, w6, #1;
str w6, [x0, #28];

umov x6, v3.2d[1];
str x6, [x0, #32];
//str w4, [x0, #32]; removed to use store pair

ldr w4, [x0, #24];
str w4, [x0, #40];

umov w17, v3.4s[0];
umov w6, v1.4s[1];
cmp w6, w17;
b.ne keyTransform;

// whoaaa, we found a result!
ldr w17, [x1];
add w16, w16, #1;
lsl w16, w16, #1;
sub w16, w16, #1;
sub w17, w17, w16;
str w17, [x1];
mov x0, #2;

b programEnd;
// program end

keyTransform:
// l0 hi/mid/lo: 16/20/24

mov w7, 0xFF;
ldr w17, [x0, #16];
add w17, w17, #2;
and w17, w17, w7;

cbnz w17, storeHi;
ldr w4, [x0, #20]; // mid
mov w11, #1;
lsl w7, w11, #24;
add w4, w4, w7;

mov w10, 0xff; // if (!(key.mid & 0xFF000000u))
lsl w9, w10, #24;
and w9, w9, w4;
cbnz w9, storeMid;

add w4, w4, w11, LSL #16;
add w12, w10, w10, LSL #8;
add w12, w12, w10, LSL #16;
and w4, w4, w12; // key.mid = (key.mid + 0x00010000) & 0x00FFFFFF;

lsl w9, w10, #16; // if (!(key.mid & 0x00FF0000))
and w9, w9, w4;
cbnz w9, storeMid;

add w4, w4, 0x100;
add w12, w10, w10, LSL #8;
and w4, w4, w12;

lsl w9, w10, #8; // if (!(key.mid & 0x0000FF00))
and w9, w9, w4;
cbnz w9, storeMid;

add w4, w4, #1;
and w4, w4, w10;

cbnz w4, storeMid;

ldr w5, [x0, #24];

add w5, w5, w11, LSL #24;

lsl w9, w10, #24;
and w9, w9, w5;
cbnz w9, storeLo;

add w5, w5, w11, LSL #16;
add w12, w10, w10, LSL #8;
add w12, w12, w10, LSL #16;
and w5, w5, w12;

and w9, w5, w10, LSL #16;
cbnz w9, storeLo;

add w12, w10, w10, LSL #8;
add w5, w5, 0x100;
and w5, w5, w12;
str w5, [x0, #24];

and w9, w5, w10, LSL #8;
cbnz w9, storeLo;

add w5, w5, #1;
and w5, w5, w10;

storeLo:
str w5, [x0, #24];

storeMid:
str w4, [x0, #20];

storeHi:
str w17, [x0, #16];

transformExit:

cmp w16, wzr;
b.gt loopStart;

mov x0, #0;

programEnd:
ldp d14, d15, [sp], #16
ldp d12, d13, [sp], #16
ldp d10, d11, [sp], #16
ldp d8, d9, [sp], #16

ret

.balign 8
constant_p: .word 0xb7e15163

.balign 8
constant_q: .word 0x9e3779b9

