; Copyright distributed.net 1997-2002 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; Author: Peter Teichmann <dnet@peter-teichmann.de>
; $Id: r72arm1-gccsdk.S,v 1.2 2003/09/12 23:08:52 mweiser Exp $

r0	RN	0
r1	RN	1
r2	RN	2
r3	RN	3
r4	RN	4
r5	RN	5
r6	RN	6
r7	RN	7
r8	RN	8
r9	RN	9
r10	RN	10
r11	RN	11
r12	RN	12
r13	RN	13
r14	RN	14
r15	RN	15
sp	RN	13
lr	RN	14
pc	RN	15

WORK	RN	0
L0	RN	2
L1	RN	3
L2	RN	4
S1	RN	5
S2	RN	6
S3	RN	7
B	RN	7
A	RN	8
C	RN	9
Q	RN	10

; S[10], S[11] and S[12] are kept in registers. This saves 12 cycles
; for 2x3 loads and 2x3 stores.
S_10    RN	11
S_11    RN	12
S_12    RN	14


	AREA	|C$$CODE|, CODE, READONLY

	ALIGN
	EXPORT	rc5_72_unit_func_arm1
rc5_72_unit_func_arm1
	stmfd	sp!, {r1, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}
	sub	sp, sp, #26*4	; S[0]...S[25]
	ldr	r1, [r1]	; r1=kiter
	ldr	L2, [WORK, #4*4]
	b	loop_end

loop_start
	ldr	L1, [WORK, #5*4]
	ldr	L0, [WORK, #6*4]

; Macros for init and round 1
; SA generate, SB generate+store, SC use
; LA generate, LB use

	MACRO
	BLOCKA0
	ldr	S1, data+0*4	; P
	ldr	Q, data+1*4	; Q
	mov	C, S1, ror#29
	add	S2, S1, Q
	add	L0, L0, C
	rsb	C, C, #32
	mov	L0, L0, ror C
	str	S1, [sp, #0*4]
	MEND

	MACRO
	BLOCKA	$i, $SA, $SB, $SC, $LA, $LB
	add	$SA, $SB, Q
	add	$SB, $SB, $SC, ror#29
	add	$SB, $SB, $LB
	add	C, $LB, $SB, ror#29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	str	$SB, [sp, #$i*4]
	MEND

	MACRO
	BLOCKA25 $SA, $SB, $SC, $LA, $LB
	add	$SB, $SB, $SC, ror#29
	ldr	$SC, [sp, #0*4]
	add	$SB, $SB, $LB
	add	C, $LB, $SB, ror#29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	str	$SB, [sp, #25*4]
	MEND

	BLOCKA0
	BLOCKA	1, S3, S2, S1, L1, L0

	stmdb	sp, {L0, L1, S1, S2, S3, Q}
	b	skip
loop_start2
	ldmdb	sp, {L0, L1, S1, S2, S3, Q}
	stmia	sp, {S1, S2}
skip
	BLOCKA	2, S1, S3, S2, L2, L1
	BLOCKA	3, S2, S1, S3, L0, L2
	BLOCKA	4, S3, S2, S1, L1, L0 
	BLOCKA	5, S1, S3, S2, L2, L1
	BLOCKA	6, S2, S1, S3, L0, L2
	BLOCKA	7, S3, S2, S1, L1, L0
	BLOCKA	8, S1, S3, S2, L2, L1
	BLOCKA	9, S2, S1, S3, L0, L2

;       BLOCKA  10, S3, S2, S1, L1, L0
        add     S3, S2, Q
        add     S2, S2, S1, ror#29
        add     S_10 /*S2*/, S2, L0
        add     C, L0, S_10 /*S2*/, ror#29
        add     L1, L1, C
        rsb     C, C, #32
        mov     L1, L1, ror C
//      str     S2, [sp, #10*4]

;       BLOCKA  11, S1, S3, S2, L2, L1
        add     S1, S3, Q
        add     S3, S3, S_10 /*S2*/, ror#29
        add     S_11 /*S3*/, S3, L1
        add     C, L1, S_11 /*S3*/, ror#29
        add     L2, L2, C
        rsb     C, C, #32
        mov     L2, L2, ror C
//      str     S3, [sp, #11*4]

;       BLOCKA  12, S2, S1, S3, L0, L2
        add     S2, S1, Q
        add     S1, S1, S_11 /*S3*/, ror#29
        add     S_12 /*S1*/, S1, L2
        add     C, L2, S_12 /*S1*/, ror#29
        add     L0, L0, C
        rsb     C, C, #32
        mov     L0, L0, ror C
//      str     S1, [sp, #12*4]

;       BLOCKA  13, S3, S2, S1, L1, L0
        add     S3, S2, Q
        add     S2, S2, S_12 /*S1*/, ror#29
        add     S2, S2, L0
        add     C, L0, S2, ror#29
        add     L1, L1, C
        rsb     C, C, #32
        mov     L1, L1, ror C
        str     S2, [sp, #13*4]

	BLOCKA	14, S1, S3, S2, L2, L1
	BLOCKA	15, S2, S1, S3, L0, L2
	BLOCKA	16, S3, S2, S1, L1, L0
	BLOCKA	17, S1, S3, S2, L2, L1
	BLOCKA	18, S2, S1, S3, L0, L2
	BLOCKA	19, S3, S2, S1, L1, L0
	BLOCKA	20, S1, S3, S2, L2, L1
	BLOCKA	21, S2, S1, S3, L0, L2
	BLOCKA	22, S3, S2, S1, L1, L0
	BLOCKA	23, S1, S3, S2, L2, L1
	BLOCKA	24, S2, S1, S3, L0, L2
	BLOCKA25    S3, S2, S1, L1, L0

; Macros for round 2
; SA generate+store, SB use+preload
; LA generate, LB use

	MACRO
	BLOCKB0 $SA, $SB, $LA, $LB
	add	$SA, $LB, $SA, ror #29
	add	$SA, $SA, $SB, ror #29
	ldr	$SB, [sp, #1*4]
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	str	$SA, [sp, #0*4]
	MEND

	MACRO
	BLOCKB	$i, $SA, $SB, $LA, $LB
	add	$SA, $LB, $SA, ror #29
	add	$SA, $SA, $SB, ror #29
	ldr	$SB, [sp, #($i+1)*4]
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	str	$SA, [sp, #$i*4]
	MEND

	MACRO
	BLOCKB25 $SA, $SB, $LA, $LB
	add	$SA, $LB, $SA, ror #29
	add	$SA, $SA, $SB, ror #29
	ldr	$SB, [sp, #0*4]
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	str	$SA, [sp, #25*4]
	MEND

	BLOCKB0	   S1, S2, L2, L1	
	BLOCKB	1, S2, S1, L0, L2	
	BLOCKB	2, S1, S2, L1, L0	
	BLOCKB	3, S2, S1, L2, L1	
	BLOCKB	4, S1, S2, L0, L2	
	BLOCKB	5, S2, S1, L1, L0	
	BLOCKB	6, S1, S2, L2, L1	
	BLOCKB	7, S2, S1, L0, L2	
	BLOCKB	8, S1, S2, L1, L0	

;       BLOCKB  9, S2, S1, L2, L1
        add     S2, L1, S2, ror #29
        add     S2, S2, S1, ror #29
//      ldr     S1, [sp, #10*4]
        add     C, L1, S2, ror #29
        add     L2, L2, C
        rsb     C, C, #32
        mov     L2, L2, ror C
        str     S2, [sp, #9*4]

;       BLOCKB  10, S1, S2, L0, L2
        add     S1, L2, S_10 /*S1*/, ror #29
        add     S_10 /*S1*/, S1, S2, ror #29
//      ldr     S2, [sp, #11*4]
        add     C, L2, S_10 /*S1*/, ror #29
        add     L0, L0, C
        rsb     C, C, #32
        mov     L0, L0, ror C
//      str     S1, [sp, #10*4]

;       BLOCKB  11, S2, S1, L1, L0
        add     S2, L0, S_11 /*S2*/, ror #29
        add     S_11 /*S2*/, S2, S_10 /*S1*/, ror #29
//      ldr     S1, [sp, #12*4]
        add     C, L0, S_11 /*S2*/, ror #29
        add     L1, L1, C
        rsb     C, C, #32
        mov     L1, L1, ror C
//      str     S2, [sp, #11*4]

;       BLOCKB  12, S1, S2, L2, L1
        add     S1, L1, S_12 /*S1*/, ror #29
        add     S_12 /*S1*/, S1, S_11 /*S2*/, ror #29
        ldr     S2, [sp, #13*4]
        add     C, L1, S_12 /*S1*/, ror #29
        add     L2, L2, C
        rsb     C, C, #32
        mov     L2, L2, ror C
//      str     S1, [sp, #12*4]

;       BLOCKB  13, S2, S1, L0, L2
        add     S2, L2, S2, ror #29
        add     S2, S2, S_12 /*S1*/, ror #29
        ldr     S1, [sp, #14*4]
        add     C, L2, S2, ror #29
        add     L0, L0, C
        rsb     C, C, #32
        mov     L0, L0, ror C
        str     S2, [sp, #13*4]

	BLOCKB	14, S1, S2, L1, L0	
	BLOCKB	15, S2, S1, L2, L1	
	BLOCKB	16, S1, S2, L0, L2	
	BLOCKB	17, S2, S1, L1, L0	
	BLOCKB	18, S1, S2, L2, L1	
	BLOCKB	19, S2, S1, L0, L2	
	BLOCKB	20, S1, S2, L1, L0	
	BLOCKB	21, S2, S1, L2, L1	
	BLOCKB	22, S1, S2, L0, L2	
	BLOCKB	23, S2, S1, L1, L0	
	BLOCKB	24, S1, S2, L2, L1	
	BLOCKB25    S2, S1, L0, L2

; Macros for round 3 and finish
; SA generate+store, SB use+preload
; LA generate, LB use	

	MACRO
	BLOCKC0 $SA, $SB, $LA, $LB
	ldmia	WORK, {B, A}
	add	$SA, $LB, $SA, ror #29
	add	$SA, $SA, $SB, ror #29
	ldr	$SB, [sp, #1*4]
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	add	A, A, $SA, ror #29
	MEND

	MACRO
	BLOCKC1 $SA, $SB, $LA, $LB
	add	$SA, $LB, $SA, ror #29
	add	$SA, $SA, $SB, ror #29
	ldr	$SB, [sp, #2*4]
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	add	B, B, $SA, ror #29
	MEND

	MACRO
	BLOCKC	$i, $SA, $SB, $LA, $LB, $XA, $XB
	add	$SA, $LB, $SA, ror #29
	add	$SA, $SA, $SB, ror #29
	ldr	$SB, [sp, #($i+1)*4]
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	eor	$XA, $XA, $XB
	rsb	C, $XB, #32
	mov	$XA, $XA, ror C
	add	$XA, $XA, $SA, ror #29
	MEND

	MACRO
	BLOCKC24_1 $SA, $SB, $LA, $LB
	add	$SA, $LB, $SA, ror #29
	add	$SA, $SA, $SB, ror #29
	eor	A, A, B
	rsb	C, B, #32
	mov	A, A, ror C
	add	A, A, $SA, ror #29
	MEND

	MACRO
	BLOCKC24_2 $SA, $SB, $LA, $LB
	ldr	$SB, [sp, #25*4]
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	MEND

	MACRO
	BLOCKC25 $SA, $SB, $LA, $LB
	add	$SA, $LB, $SA, ror #29
	add	$SA, $SA, $SB, ror #29
	eor	B, B, A
	rsb	C, A, #32
	mov	B, B, ror C
	add	B, B, $SA, ror #29
	MEND

	BLOCKC0	   S1, S2, L1, L0
	BLOCKC1	   S2, S1, L2, L1
	BLOCKC	2, S1, S2, L0, L2, A, B
	BLOCKC	3, S2, S1, L1, L0, B, A
	BLOCKC	4, S1, S2, L2, L1, A, B
	BLOCKC	5, S2, S1, L0, L2, B, A
	BLOCKC	6, S1, S2, L1, L0, A, B
	BLOCKC	7, S2, S1, L2, L1, B, A
	BLOCKC	8, S1, S2, L0, L2, A, B

;       BLOCKC  9, S2, S1, L1, L0, B, A
        add     S2, L0, S2, ror #29
        add     S2, S2, S1, ror #29
//      ldr     S1, [sp, #10*4]
        add     C, L0, S2, ror #29
        add     L1, L1, C
        rsb     C, C, #32
        mov     L1, L1, ror C
        eor     B, B, A
        rsb     C, A, #32
        mov     B, B, ror C
        add     B, B, S2, ror #29

;       BLOCKC  10, S1, S2, L2, L1, A, B
        add     S1, L1, S_10 /*S1*/, ror #29
        add     S1, S1, S2, ror #29
//      ldr     S2, [sp, #11*4]
        add     C, L1, S1, ror #29
        add     L2, L2, C
        rsb     C, C, #32
        mov     L2, L2, ror C
        eor     A, A, B
        rsb     C, B, #32
        mov     A, A, ror C
        add     A, A, S1, ror #29

;       BLOCKC  11, S2, S1, L0, L2, B, A
        add     S2, L2, S_11 /*S2*/, ror #29
        add     S2, S2, S1, ror #29
//      ldr     S1, [sp, #12*4]
        add     C, L2, S2, ror #29
        add     L0, L0, C
        rsb     C, C, #32
        mov     L0, L0, ror C
        eor     B, B, A
        rsb     C, A, #32
        mov     B, B, ror C
        add     B, B, S2, ror #29

;       BLOCKC  12, S1, S2, L1, L0, A, B
        add     S1, L0, S_12 /*S1*/, ror #29
        add     S1, S1, S2, ror #29
        ldr     S2, [sp, #13*4]
        add     C, L0, S1, ror #29
        add     L1, L1, C
        rsb     C, C, #32
        mov     L1, L1, ror C
        eor     A, A, B
        rsb     C, B, #32
        mov     A, A, ror C
        add     A, A, S1, ror #29

	BLOCKC	13, S2, S1, L2, L1, B, A
	BLOCKC	14, S1, S2, L0, L2, A, B
	BLOCKC	15, S2, S1, L1, L0, B, A
	BLOCKC	16, S1, S2, L2, L1, A, B
	BLOCKC	17, S2, S1, L0, L2, B, A
	BLOCKC	18, S1, S2, L1, L0, A, B
	BLOCKC	19, S2, S1, L2, L1, B, A
	BLOCKC	20, S1, S2, L0, L2, A, B
	BLOCKC	21, S2, S1, L1, L0, B, A
	BLOCKC	22, S1, S2, L2, L1, A, B
	BLOCKC	23, S2, S1, L0, L2, B, A
	ldr	Q, [WORK, #3*4]
	BLOCKC24_1  S1, S2, L1, L0
	cmp	A, Q
	ldr	L2, [WORK, #4*4]
	beq	found_a

increment_hi
	add	L2, L2, #1
	ands	L2, L2, #0xff
	str	L2, [WORK, #4*4]
	beq	increment_mid
	
	subs	r1,r1,#1
	bcs	loop_start2
	
	mov	r0,#1
	add	sp, sp, #26*4
	ldmfd	sp!, {r1, r4, r5, r6, r7, r8, r9, r10, r11, r12, pc}

increment_mid
	ldr	A, data+2*4
	ldr	L0, [WORK, #5*4]
	mvn	B, A
	and	L1, A, L0, ror #8
	and	L0, B, L0, ror #24
	orr	L0, L0, L1		; swap byte order

	adds	L0, L0, #1	

	and	L1, A, L0, ror #8
	and	L0, B, L0, ror #24
	orr	L0, L0, L1		; swap byte order
	
	str	L0, [WORK, #5*4]
	bcc	loop_end

increment_lo
	ldr	L0, [WORK, #6*4]
	and	L1, A, L0, ror #8
	and	L0, B, L0, ror #24
	orr	L0, L0, L1		; swap byte order

	add	L0, L0, #1	

	and	L1, A, L0, ror #8
	and	L0, B, L0, ror #24
	orr	L0, L0, L1		; swap byte order
	
	str	L0, [WORK, #6*4]
	
loop_end
	subs	r1,r1,#1
	bcs	loop_start
	
	mov	r0,#1
	add	sp, sp, #26*4
	ldmfd	sp!, {r1, r4, r5, r6, r7, r8, r9, r10, r11, r12, pc}

found_a
	BLOCKC24_2  S1, S2, L1, L0
	BLOCKC25    S2, S1, L2, L1

	ldr	r2,[WORK, #7*4]
	ldr	r3,[WORK, #4*4]
	add	r2,r2,#1
	str	r2,[WORK, #7*4]
	str	r3,[WORK, #8*4]
	ldr	r2,[WORK, #5*4]
	ldr	r3,[WORK, #6*4]
	str	r2,[WORK, #9*4]
	str	r3,[WORK, #10*4]

	ldr	Q, [WORK, #2*4]
	cmp	B, Q
	bne	increment_hi	
	
	add	sp, sp, #26*4
	ldmfd	sp!, {r3}
	add	r2, r1, #1
	ldr	r1, [r3]
	mov	r0, #2
	sub	r1, r1, r2
	str	r1, [r3]
	ldmfd	sp!, {r4, r5, r6, r7, r8, r9, r10, r11, r12, pc}

	ALIGN
data
	DCD	&b7e15163	; P
	DCD	&9e3779b9	; Q
	DCD	&ff00ff00
