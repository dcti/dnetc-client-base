; Copyright distributed.net 1997-2015 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; Author: Peter Teichmann <dnet@peter-teichmann.de>

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
S_1	RN	5
S_2	RN	6
S_3	RN	7
S_4	RN	8
S_5	RN	9
S_6	RN	10	; don not use together with B
S_7	RN	11	; don not use together with A
S_8	RN	12	; don not use together with Q
B	RN	10
A	RN	11
Q	RN	12
C	RN	14

	AREA	|C$$CODE|, CODE, READONLY

	ALIGN
	EXPORT	rc5_72_unit_func_arm2
rc5_72_unit_func_arm2
	stmfd	sp!, {r1, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}
	sub	sp, sp, #26*4	; S[0]...S[25]
	ldr	r1, [r1]	; r1=kiter
	ldr	L2, [WORK, #4*4]
	b	loop_end

loop_start
	ldr	L1, [WORK, #5*4]
	ldr	L0, [WORK, #6*4]

; Macros for init and round 1
; SA add Q, SB generate+store, SC use (last SB)
; LA generate, LB use

	MACRO
	BLOCKA0
	ldr	S_1, data+0*4	; P
	ldr	Q, data+1*4	; Q
	mov	C, S_1, ror#29
	add	S_7, S_1, Q
	add	L0, L0, C
	rsb	C, C, #32
	mov	L0, L0, ror C
	MEND

	MACRO
	BLOCKA	$SB, $SC, $LA, $LB
	add	$SB, S_7, $SC, ror#29
	add	S_7, S_7, Q
	add	$SB, $SB, $LB
	add	C, $LB, $SB, ror#29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	MEND

	MACRO
	BLOCKA25 $SB, $SC, $LA, $LB
	add	$SB, S_7, $SC, ror#29
	add	$SB, $SB, $LB
	add	C, $LB, $SB, ror#29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	MEND

	BLOCKA0
	BLOCKA	S_2, S_1, L1, L0

	stmdb	sp, {L0, L1, S_1, S_2, S_7, Q}
	b	skip
loop_start2
	ldmdb	sp, {L0, L1, S_1, S_2, S_7, Q}
skip
	BLOCKA	S_3, S_2, L2, L1
	BLOCKA	S_4, S_3, L0, L2
	BLOCKA	S_5, S_4, L1, L0 
	BLOCKA	S_6, S_5, L2, L1
	stmia	sp!, {S_1, S_2, S_3, S_4, S_5, S_6}
	BLOCKA	S_1, S_6, L0, L2
	BLOCKA	S_2, S_1, L1, L0
	BLOCKA	S_3, S_2, L2, L1
	BLOCKA	S_4, S_3, L0, L2
	BLOCKA	S_5, S_4, L1, L0
	BLOCKA	S_6, S_5, L2, L1
	stmia	sp!, {S_1, S_2, S_3, S_4, S_5, S_6}
	BLOCKA	S_1, S_6, L0, L2
	BLOCKA	S_2, S_1, L1, L0
	BLOCKA	S_3, S_2, L2, L1
	BLOCKA	S_4, S_3, L0, L2
	BLOCKA	S_5, S_4, L1, L0
	BLOCKA	S_6, S_5, L2, L1
	stmia	sp!, {S_1, S_2, S_3, S_4, S_5, S_6}
	BLOCKA	S_1, S_6, L0, L2
	BLOCKA	S_2, S_1, L1, L0
	BLOCKA	S_3, S_2, L2, L1
	BLOCKA	S_4, S_3, L0, L2
	BLOCKA	S_5, S_4, L1, L0
	BLOCKA	S_6, S_5, L2, L1
	stmia	sp!, {S_1, S_2, S_3, S_4, S_5, S_6}
	BLOCKA	S_1, S_6, L0, L2
	BLOCKA25 S_8, S_1, L1, L0
	stmia	sp!, {S_1, S_8}

	sub	sp, sp, #26*4

; Macros for round 2
; SA generate+store, SB use (last SA), SD use from last round
; LA generate, LB use

	MACRO
	BLOCKB	$SA, $SB, $SD, $LA, $LB
	add	$SA, $LB, $SD, ror #29
	add	$SA, $SA, $SB, ror #29
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	MEND

	ldmia	sp, {S_1, S_2, S_3, S_4, S_5, S_6, S_7}
	BLOCKB	S_1, S_8, S_1, L2, L1	
	BLOCKB	S_2, S_1, S_2, L0, L2	
	BLOCKB	S_3, S_2, S_3, L1, L0	
	BLOCKB	S_4, S_3, S_4, L2, L1	
	BLOCKB	S_5, S_4, S_5, L0, L2	
	BLOCKB	S_6, S_5, S_6, L1, L0	
	BLOCKB	S_8, S_6, S_7, L2, L1
	stmia	sp!, {S_1, S_2, S_3, S_4, S_5, S_6, S_8}
	ldmia	sp, {S_1, S_2, S_3, S_4, S_5, S_6, S_7}
	BLOCKB	S_1, S_8, S_1, L0, L2	
	BLOCKB	S_2, S_1, S_2, L1, L0	
	BLOCKB	S_3, S_2, S_3, L2, L1	
	BLOCKB	S_4, S_3, S_4, L0, L2	
	BLOCKB	S_5, S_4, S_5, L1, L0	
	BLOCKB	S_6, S_5, S_6, L2, L1	
	BLOCKB	S_8, S_6, S_7, L0, L2	
	stmia	sp!, {S_1, S_2, S_3, S_4, S_5, S_6, S_8}
	ldmia	sp, {S_1, S_2, S_3, S_4, S_5, S_6, S_7}
	BLOCKB	S_1, S_8, S_1, L1, L0	
	BLOCKB	S_2, S_1, S_2, L2, L1	
	BLOCKB	S_3, S_2, S_3, L0, L2	
	BLOCKB	S_4, S_3, S_4, L1, L0	
	BLOCKB	S_5, S_4, S_5, L2, L1	
	BLOCKB	S_6, S_5, S_6, L0, L2	
	BLOCKB	S_8, S_6, S_7, L1, L0	
	stmia	sp!, {S_1, S_2, S_3, S_4, S_5, S_6, S_8}
	ldmia	sp, {S_1, S_2, S_3, S_4, S_5}
	BLOCKB	S_1, S_8, S_1, L2, L1	
	BLOCKB	S_2, S_1, S_2, L0, L2	
	BLOCKB	S_3, S_2, S_3, L1, L0	
	BLOCKB	S_4, S_3, S_4, L2, L1	
	BLOCKB	S_5, S_4, S_5, L0, L2
	stmia	sp!, {S_1, S_2, S_3, S_4, S_5}

	sub	sp, sp, #26*4
	mov	S_8, S_5

; Macros for round 3 and finish
; SA generate+store, SB use (last SA), SC use from last round
; LA generate, LB use	

	MACRO
	BLOCKC0 $SA, $SB, $SC, $LA, $LB
	ldmia	WORK, {B, A}
	add	$SA, $LB, $SC, ror #29
	add	$SA, $SA, $SB, ror #29
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	add	A, A, $SA, ror #29
	MEND

	MACRO
	BLOCKC1 $SA, $SB, $SC, $LA, $LB
	add	$SA, $LB, $SC, ror #29
	add	$SA, $SA, $SB, ror #29
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	add	B, B, $SA, ror #29
	MEND

	MACRO
	BLOCKC	$SA, $SB, $SC, $LA, $LB, $XA, $XB
	add	$SA, $LB, $SC, ror #29
	add	$SA, $SA, $SB, ror #29
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
	BLOCKC24_1 $SA, $SB, $SC, $LA, $LB
	add	$SA, $LB, $SC, ror #29
	add	$SA, $SA, $SB, ror #29
	eor	A, A, B
	rsb	C, B, #32
	mov	A, A, ror C
	add	A, A, $SA, ror #29
	MEND

	MACRO
	BLOCKC24_2 $SA, $SB, $SC, $LA, $LB
	add	C, $LB, $SA, ror #29
	add	$LA, $LA, C
	rsb	C, C, #32
	mov	$LA, $LA, ror C
	MEND

	MACRO
	BLOCKC25 $SA, $SB, $SC, $LA, $LB
	add	$SA, $LB, $SC, ror #29
	add	$SA, $SA, $SB, ror #29
	eor	B, B, A
	rsb	C, A, #32
	mov	B, B, ror C
	add	B, B, $SA, ror #29
	MEND
	
	ldmia	sp!, {S_1, S_2, S_3, S_4, S_5}
	BLOCKC0	S_1, S_8, S_1, L1, L0
	BLOCKC1	S_2, S_1, S_2, L2, L1
	BLOCKC	S_3, S_2, S_3, L0, L2, A, B
	BLOCKC	S_4, S_3, S_4, L1, L0, B, A
	BLOCKC	S_8, S_4, S_5, L2, L1, A, B
	ldmia	sp!, {S_1, S_2, S_3, S_4, S_5}
	BLOCKC	S_1, S_8, S_1, L0, L2, B, A
	BLOCKC	S_2, S_1, S_2, L1, L0, A, B
	BLOCKC	S_3, S_2, S_3, L2, L1, B, A
	BLOCKC	S_4, S_3, S_4, L0, L2, A, B
	BLOCKC	S_8, S_4, S_5, L1, L0, B, A
	ldmia	sp!, {S_1, S_2, S_3, S_4, S_5}
	BLOCKC	S_1, S_8, S_1, L2, L1, A, B
	BLOCKC	S_2, S_1, S_2, L0, L2, B, A
	BLOCKC	S_3, S_2, S_3, L1, L0, A, B
	BLOCKC	S_4, S_3, S_4, L2, L1, B, A
	BLOCKC	S_8, S_4, S_5, L0, L2, A, B
	ldmia	sp!, {S_1, S_2, S_3, S_4, S_5}
	BLOCKC	S_1, S_8, S_1, L1, L0, B, A
	BLOCKC	S_2, S_1, S_2, L2, L1, A, B
	BLOCKC	S_3, S_2, S_3, L0, L2, B, A
	BLOCKC	S_4, S_3, S_4, L1, L0, A, B
	BLOCKC	S_8, S_4, S_5, L2, L1, B, A
	ldmia	sp!, {S_1, S_2, S_3, S_4, S_5}
	BLOCKC	S_1, S_8, S_1, L0, L2, A, B
	BLOCKC	S_2, S_1, S_2, L1, L0, B, A
	BLOCKC	S_3, S_2, S_3, L2, L1, A, B
	BLOCKC	S_4, S_3, S_4, L0, L2, B, A
	BLOCKC24_1 S_8, S_4, S_5, L1, L0
	ldr	C, [WORK, #3*4]
	cmp	A, C
	beq	found_a

	sub	sp, sp, #25*4
increment_hi
	ldr	L2, [WORK, #4*4]
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
	BLOCKC24_2  S_8, S_4, S_5, L1, L0
	ldr	S_1, [sp],#-25*4
	BLOCKC25    S_1, S_8, S_1, L2, L1

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
    pop	r3 ; ldmfd	sp!, {r3} gives error A2193
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

    END

