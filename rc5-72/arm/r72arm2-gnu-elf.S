@ Copyright distributed.net 1997-2002 - All Rights Reserved
@ For use in distributed.net projects only.
@ Any other distribution or use of this source violates copyright.
@
@ Author: Peter Teichmann <dnet@peter-teichmann.de>
@ $Id: r72arm2-gnu-elf.S,v 1.1.2.2 2003/02/25 19:23:01 teichp Exp $

#define	WORK	r0
#define	L0	r2
#define	L1	r3
#define	L2	r4
#define	S1	r5
#define	S2	r6
#define	S3	r7
#define S4	r8
#define	S5	r9
#define S6	r10	// don't use together with B
#define S7	r11	// don't use together with A
#define S8	r12	// don't use together with Q
#define	B	r10
#define	A	r11
#define	Q	r12
#define	C	lr

.text
	.align	5
	.global	rc5_72_unit_func_arm2
rc5_72_unit_func_arm2:
	stmfd	sp!, {r1, r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}
	sub	sp, sp, #26*4	@ S[0]...S[25]
	ldr	r1, [r1]	@ r1=kiter
	ldr	L2, [WORK, #4*4]
	b	loop_end

loop_start:
	ldr	L1, [WORK, #5*4]
	ldr	L0, [WORK, #6*4]

@ Macros for init and round 1
@ SA add Q, SB generate+store, SC use (last SB)
@ LA generate, LB use

.MACRO	 BLOCKA0
	ldr	S1, data+0*4	@ P
	ldr	Q, data+1*4	@ Q
	mov	C, S1, ror#29
	add	S7, S1, Q
	add	L0, L0, C
	rsb	C, C, #32
	mov	L0, L0, ror C
.ENDM

.MACRO	BLOCKA	SB, SC, LA, LB
	add	\SB, S7, \SC, ror#29
	add	S7, S7, Q
	add	\SB, \SB, \LB
	add	C, \LB, \SB, ror#29
	add	\LA, \LA, C
	rsb	C, C, #32
	mov	\LA, \LA, ror C
.ENDM

.MACRO	BLOCKA25 SB, SC, LA, LB
	add	\SB, S7, \SC, ror#29
	add	\SB, \SB, \LB
	add	C, \LB, \SB, ror#29
	add	\LA, \LA, C
	rsb	C, C, #32
	mov	\LA, \LA, ror C
.ENDM

	BLOCKA0
	BLOCKA	S2, S1, L1, L0

	stmdb	sp, {L0, L1, S1, S2, S7, Q}
	b	skip
loop_start2:
	ldmdb	sp, {L0, L1, S1, S2, S7, Q}
skip:
	BLOCKA	S3, S2, L2, L1
	BLOCKA	S4, S3, L0, L2
	BLOCKA	S5, S4, L1, L0 
	BLOCKA	S6, S5, L2, L1
	stmia	sp!, {S1, S2, S3, S4, S5, S6}
	BLOCKA	S1, S6, L0, L2
	BLOCKA	S2, S1, L1, L0
	BLOCKA	S3, S2, L2, L1
	BLOCKA	S4, S3, L0, L2
	BLOCKA	S5, S4, L1, L0
	BLOCKA	S6, S5, L2, L1
	stmia	sp!, {S1, S2, S3, S4, S5, S6}
	BLOCKA	S1, S6, L0, L2
	BLOCKA	S2, S1, L1, L0
	BLOCKA	S3, S2, L2, L1
	BLOCKA	S4, S3, L0, L2
	BLOCKA	S5, S4, L1, L0
	BLOCKA	S6, S5, L2, L1
	stmia	sp!, {S1, S2, S3, S4, S5, S6}
	BLOCKA	S1, S6, L0, L2
	BLOCKA	S2, S1, L1, L0
	BLOCKA	S3, S2, L2, L1
	BLOCKA	S4, S3, L0, L2
	BLOCKA	S5, S4, L1, L0
	BLOCKA	S6, S5, L2, L1
	stmia	sp!, {S1, S2, S3, S4, S5, S6}
	BLOCKA	S1, S6, L0, L2
	BLOCKA25 S8, S1, L1, L0
	stmia	sp!, {S1, S8}

	sub	sp, sp, #26*4

@ Macros for round 2
@ SA generate+store, SB use (last SA), SD use from last round
@ LA generate, LB use

.MACRO	BLOCKB	SA, SB, SD, LA, LB
	add	\SA, \LB, \SD, ror #29
	add	\SA, \SA, \SB, ror #29
	add	C, \LB, \SA, ror #29
	add	\LA, \LA, C
	rsb	C, C, #32
	mov	\LA, \LA, ror C
.ENDM

.MACRO	BLOCKB25 SA, SB, SD, LA, LB
	add	\SA, \LB, \SD, ror #29
	add	\SA, \SA, \SB, ror #29
	add	C, \LB, \SA, ror #29
	add	\LA, \LA, C
	rsb	C, C, #32
	mov	\LA, \LA, ror C
.ENDM

	ldmia	sp, {S1, S2, S3, S4, S5, S6, S7}
	BLOCKB	S1, S8, S1, L2, L1	
	BLOCKB	S2, S1, S2, L0, L2	
	BLOCKB	S3, S2, S3, L1, L0	
	BLOCKB	S4, S3, S4, L2, L1	
	BLOCKB	S5, S4, S5, L0, L2	
	BLOCKB	S6, S5, S6, L1, L0	
	BLOCKB	S8, S6, S7, L2, L1
	stmia	sp!, {S1, S2, S3, S4, S5, S6, S8}
	ldmia	sp, {S1, S2, S3, S4, S5, S6, S7}
	BLOCKB	S1, S8, S1, L0, L2	
	BLOCKB	S2, S1, S2, L1, L0	
	BLOCKB	S3, S2, S3, L2, L1	
	BLOCKB	S4, S3, S4, L0, L2	
	BLOCKB	S5, S4, S5, L1, L0	
	BLOCKB	S6, S5, S6, L2, L1	
	BLOCKB	S8, S6, S7, L0, L2	
	stmia	sp!, {S1, S2, S3, S4, S5, S6, S8}
	ldmia	sp, {S1, S2, S3, S4, S5, S6, S7}
	BLOCKB	S1, S8, S1, L1, L0	
	BLOCKB	S2, S1, S2, L2, L1	
	BLOCKB	S3, S2, S3, L0, L2	
	BLOCKB	S4, S3, S4, L1, L0	
	BLOCKB	S5, S4, S5, L2, L1	
	BLOCKB	S6, S5, S6, L0, L2	
	BLOCKB	S8, S6, S7, L1, L0	
	stmia	sp!, {S1, S2, S3, S4, S5, S6, S8}
	ldmia	sp, {S1, S2, S3, S4, S5}
	BLOCKB	S1, S8, S1, L2, L1	
	BLOCKB	S2, S1, S2, L0, L2	
	BLOCKB	S3, S2, S3, L1, L0	
	BLOCKB	S4, S3, S4, L2, L1	
	BLOCKB	S5, S4, S5, L0, L2
	stmia	sp!, {S1, S2, S3, S4, S5}

	sub	sp, sp, #26*4
	mov	S8, S5

@ Macros for round 3 and finish
@ SA generate+store, SB use (last SA), SC use from last round
@ LA generate, LB use	

.MACRO	BLOCKC0 SA, SB, SC, LA, LB
	ldmia	WORK, {B, A}
	add	\SA, \LB, \SC, ror #29
	add	\SA, \SA, \SB, ror #29
	add	C, \LB, \SA, ror #29
	add	\LA, \LA, C
	rsb	C, C, #32
	mov	\LA, \LA, ror C
	add	A, A, \SA, ror #29
.ENDM

.MACRO BLOCKC1 SA, SB, SC, LA, LB
	add	\SA, \LB, \SC, ror #29
	add	\SA, \SA, \SB, ror #29
	add	C, \LB, \SA, ror #29
	add	\LA, \LA, C
	rsb	C, C, #32
	mov	\LA, \LA, ror C
	add	B, B, \SA, ror #29
.ENDM

.MACRO BLOCKC	SA, SB, SC, LA, LB, XA, XB
	add	\SA, \LB, \SC, ror #29
	add	\SA, \SA, \SB, ror #29
	add	C, \LB, \SA, ror #29
	add	\LA, \LA, C
	rsb	C, C, #32
	mov	\LA, \LA, ror C
	eor	\XA, \XA, \XB
	rsb	C, \XB, #32
	mov	\XA, \XA, ror C
	add	\XA, \XA, \SA, ror #29
.ENDM

.MACRO BLOCKC24_1 SA, SB, SC, LA, LB
	add	\SA, \LB, \SC, ror #29
	add	\SA, \SA, \SB, ror #29
	eor	A, A, B
	rsb	C, B, #32
	mov	A, A, ror C
	add	A, A, \SA, ror #29
.ENDM

.MACRO BLOCKC24_2 SA, SB, SC, LA, LB
	add	C, \LB, \SA, ror #29
	add	\LA, \LA, C
	rsb	C, C, #32
	mov	\LA, \LA, ror C
.ENDM

.MACRO BLOCKC25	SA, SB, SC, LA, LB
	add	\SA, \LB, \SC, ror #29
	add	\SA, \SA, \SB, ror #29
	eor	B, B, A
	rsb	C, A, #32
	mov	B, B, ror C
	add	B, B, \SA, ror #29
.ENDM
	
	ldmia	sp!, {S1, S2, S3, S4, S5}
	BLOCKC0	S1, S8, S1, L1, L0
	BLOCKC1	S2, S1, S2, L2, L1
	BLOCKC	S3, S2, S3, L0, L2, A, B
	BLOCKC	S4, S3, S4, L1, L0, B, A
	BLOCKC	S8, S4, S5, L2, L1, A, B
	ldmia	sp!, {S1, S2, S3, S4, S5}
	BLOCKC	S1, S8, S1, L0, L2, B, A
	BLOCKC	S2, S1, S2, L1, L0, A, B
	BLOCKC	S3, S2, S3, L2, L1, B, A
	BLOCKC	S4, S3, S4, L0, L2, A, B
	BLOCKC	S8, S4, S5, L1, L0, B, A
	ldmia	sp!, {S1, S2, S3, S4, S5}
	BLOCKC	S1, S8, S1, L2, L1, A, B
	BLOCKC	S2, S1, S2, L0, L2, B, A
	BLOCKC	S3, S2, S3, L1, L0, A, B
	BLOCKC	S4, S3, S4, L2, L1, B, A
	BLOCKC	S8, S4, S5, L0, L2, A, B
	ldmia	sp!, {S1, S2, S3, S4, S5}
	BLOCKC	S1, S8, S1, L1, L0, B, A
	BLOCKC	S2, S1, S2, L2, L1, A, B
	BLOCKC	S3, S2, S3, L0, L2, B, A
	BLOCKC	S4, S3, S4, L1, L0, A, B
	BLOCKC	S8, S4, S5, L2, L1, B, A
	ldmia	sp!, {S1, S2, S3, S4, S5}
	BLOCKC	S1, S8, S1, L0, L2, A, B
	BLOCKC	S2, S1, S2, L1, L0, B, A
	BLOCKC	S3, S2, S3, L2, L1, A, B
	BLOCKC	S4, S3, S4, L0, L2, B, A
	BLOCKC24_1 S8, S4, S5, L1, L0
	ldr	C, [WORK, #3*4]
	cmp	A, C
	beq	found_a

	sub	sp, sp, #25*4
increment_hi:
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

increment_mid:
	ldr	A, data+2*4
	ldr	L0, [WORK, #5*4]
	mvn	B, A
	and	L1, A, L0, ror #8
	and	L0, B, L0, ror #24
	orr	L0, L0, L1		@ swap byte order

	adds	L0, L0, #1	

	and	L1, A, L0, ror #8
	and	L0, B, L0, ror #24
	orr	L0, L0, L1		@ swap byte order
	
	str	L0, [WORK, #5*4]
	bcc	loop_end

increment_lo:
	ldr	L0, [WORK, #6*4]
	and	L1, A, L0, ror #8
	and	L0, B, L0, ror #24
	orr	L0, L0, L1		@ swap byte order

	add	L0, L0, #1	

	and	L1, A, L0, ror #8
	and	L0, B, L0, ror #24
	orr	L0, L0, L1		@ swap byte order
	
	str	L0, [WORK, #6*4]
	
loop_end:
	subs	r1,r1,#1
	bcs	loop_start
	
	mov	r0,#1
	add	sp, sp, #26*4
	ldmfd	sp!, {r1, r4, r5, r6, r7, r8, r9, r10, r11, r12, pc}

found_a:
	BLOCKC24_2  S8, S4, S5, L1, L0
	ldr	S1, [sp],#-25*4
	BLOCKC25    S1, S8, S1, L2, L1

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

	.align	5
data:
	.word	0xb7e15163	@ P
	.word	0x9e3779b9	@ Q
	.word	0xff00ff00
