; ARM optimised RC5-64 core, 1 key at a time (OBJASM format)
;
; Steve Lee, Chris Berry, Tim Dobson 1997-2000
;
; $Id: rc5-arm-1.s,v 1.2.2.1 2000/09/12 12:34:08 cyp Exp $
;

	AREA	fastrc5area, CODE, READONLY


	EXPORT	rc5_unit_func_arm_1

R	*	12
T	*	2*(R+1)
	ASSERT	R=12
; current use of macros does not allow variable number of rounds.

P32	*	&B7E15163
Q32	*	&9E3779B9
P32_ROR29 *	(P32 :SHL: 3) :OR: (P32 :SHR: 29)

rA	RN	10
rB	RN	11
rC	RN	12

	GBLS	lastreg

	MACRO
	Round1a	$reg,$a
	ADD	$reg, $reg, $a
	ADD	r0, r2, $reg, ROR #29
	RSB	r0, r0, #0
	SUB	r3, r3, r0
	MOV	r3, r3, ROR r0
	ADD	r0, r3, $reg, ROR #29
	MEND

	MACRO
	Round1b	$reg,$a
	ADD	$reg, $reg, $a
	ADD	r0, r3, $reg, ROR #29
	RSB	r0, r0, #0
	SUB	r2, r2, r0
	MOV	r2, r2, ROR r0
	ADD	r0, r2, $reg, ROR #29
	MEND


	MACRO
	Round2a $reg

	ADD	$reg, r0, $reg, ROR #29
	ADD	r0, r2, $reg, ROR #29

	RSB	r0, r0, #0
	SUB	r3, r3, r0
	MOV	r3, r3, ROR r0
	ADD	r0, r3, $reg, ROR #29
	MEND


	MACRO
	Round2b $reg
	ADD	$reg, r0, $reg, ROR #29

	ADD	r0, r3, $reg, ROR #29

	RSB	r0, r0, #0
	SUB	r2, r2, r0
	MOV	r2, r2, ROR r0
	ADD	r0, r2, $reg, ROR #29

	MEND


	MACRO
	Round3a $reg

	ADD	$reg, r0, $reg, ROR #29
	ADD	r7,r7,$reg, ROR #29

	ADD	r0, r2, $reg, ROR #29

	RSB	r0, r0, #0
	SUB	r3, r3, r0
	MOV	r3, r3, ROR r0

	ADD	r0, r3, $reg,ROR #29
	MEND

	MACRO
	Crypta	$reg
	EOR	r7,r7,r6
	RSB	$reg,r6,#0
	MOV	r7,r7,ROR $reg
	MEND


	MACRO
	Round3b $reg

	ADD	$reg, r0, $reg, ROR #29
	ADD	r6,r6,$reg,ROR #29

	ADD	r0, r3, $reg, ROR #29
	RSB	r0, r0, #0
	SUB	r2, r2, r0
	MOV	r2, r2, ROR r0
	ADD	r0, r2, $reg, ROR #29
	MEND

	MACRO
	Cryptb	$reg
	EOR	r6,r6,r7
	RSB	$reg,r7,#0
	MOV	r6,r6,ROR $reg
	MEND


	GBLA	TMP
	GBLA	TMP2
	GBLA	Inner
	GBLA	CNT

CNT	SETA	0
TMP	SETA	P32

pqtable
	WHILE	CNT < T
	&	TMP
TMP	SETA	TMP + Q32
CNT	SETA	CNT + 1
	WEND


rc5_unit_func_arm_1
	STMFD	r13!, {r4-r12,r14}

	mov	r14, r1

	LDMIA	r0!,{r4-r7}
	LDMIA	r0, {r2-r3}
	STR	r14,[r13,#-(T*4-24+32)]!
	STR	r14,[r13,#-4]!
	STR	r0,[r13,#8]
	ADD	r14,r13,#32+4
	STMFD	r13!, {r4-r7}

timingloop
	str	r3,[r13,#28+4]

	ADR	r1, pqtable
	LDMIA	r1!,{r4-r6}

	ADD	r3, r3, r4, ROR #29
	MOV	r3, r3, ROR #(31 :AND: -P32_ROR29)

	ADD	r0, r5, r4, ROR #29
	ADD	r5, r0, r3

	ADD	r0, r3, r5, ROR #29
	RSB	r0, r0, #0

	ADD	r6, r6, r5, ROR #29
	STR	r5,[r14,#-16]
	STMDB	r14,{r0,r3,r6}
	B	skippy

timingloop2
	ADR	r1, pqtable+12
	LDMDB	r14,{r0,r3,r6}
skippy
	LDMIA	r1!, {r7-r9,rA,rB,rC}
	ADD	r14,r14,#8

	STR	r2,[r13,#24+4]

	SUB	r2, r2, r0
	MOV	r2, r2, ROR r0

	Round1a	r6,r2
	Round1b	r7,r0
	Round1a	r8,r0
	Round1b	r9,r0
	Round1a	rA,r0
	Round1b	rB,r0
	Round1a	rC,r0
	STMIA	r14!,{r6-r9,rA,rB,rC}
	LDMIA	r1!,{r4-r9,rA,rB,rC}
	Round1b	r4,r0
	Round1a	r5,r0
	Round1b	r6,r0
	Round1a	r7,r0
	Round1b	r8,r0
	Round1a	r9,r0
	Round1b	rA,r0
	Round1a	rB,r0
	STMIA	r14!,{r4-r9,rA,rB}
	Round1b	rC,r0
	STR	rC,[r14],#32+4-(T)*4
	LDMIA	r1!,{r4,r5,r8,r9,rA,rB,rC}
	Round1a	r4,r0
	Round1b	r5,r0
	Round1a	r8,r0
	Round1b	r9,r0
	Round1a	rA,r0
	Round1b	rB,r0
	LDR	r1,[r1]
	Round1a	rC,r0
	Round1b	r1,r0

	LDR	r6,pqtable
	LDR	r7,[r14,#-16]

	Round2a	r6
	Round2b	r7
	STMIA	r14!,{r6,r7}
	LDMIA	r14,{r6,r7}
	Round2a	r6
	Round2b	r7
	STMIA	r14!,{r6,r7}
	LDMIA	r14,{r6,r7}
	Round2a	r6
	Round2b	r7
	STMIA	r14!,{r6,r7}
	LDMIA	r14,{r6,r7}
	Round2a	r6
	Round2b	r7
	STMIA	r14!,{r6,r7}
	LDMIA	r14,{r6,r7}
	Round2a	r6
	Round2b	r7
	STMIA	r14!,{r6,r7}
	LDMIA	r14,{r6,r7}
	Round2a	r6
	Round2b	r7
	STMIA	r14!,{r6,r7}
	LDMIA	r14,{r6,r7}
	Round2a	r6
	Round2b	r7
	STMIA	r14!,{r6,r7}
	LDMIA	r14,{r6,r7}
	Round2a	r6
	Round2b	r7
	STMIA	r14!,{r6,r7}
	LDMIA	r14,{r6,r7}
	Round2a	r6
	Round2b	r7
	STMIA	r14!,{r6,r7}
	Round2a	r4
	STR	r4,[r14],#28-24
	Round2b	r5
	Round2a	r8
	Round2b	r9
	Round2a	rA
	Round2b	rB
	Round2a	rC
	Round2b	r1
	STR	r1,[r14],#4-T*4+24

	LDR	r4,[r14],#4
	LDMIA	r13,{r6,r7}

	Round3a	r4
	LDMIA	r14!,{r1,r4}
	Round3b	r1
	Crypta	r1
	Round3a	r4
	Cryptb	r4
	LDMIA	r14!,{r1,r4}
	Round3b	r1
	Crypta	r1
	Round3a	r4
	Cryptb	r4
	LDMIA	r14!,{r1,r4}
	Round3b	r1
	Crypta	r1
	Round3a	r4
	Cryptb	r4
	LDMIA	r14!,{r1,r4}
	Round3b	r1
	Crypta	r1
	Round3a	r4
	Cryptb	r4
	LDMIA	r14!,{r1,r4}
	Round3b	r1
	Crypta	r1
	Round3a	r4
	Cryptb	r4
	LDMIA	r14!,{r1,r4}
	Round3b	r1
	Crypta	r1
	Round3a	r4
	Cryptb	r4
	LDMIA	r14!,{r1,r4}
	Round3b	r1
	Crypta	r1
	Round3a	r4
	Cryptb	r4
	LDMIA	r14!,{r1,r4}
	Round3b	r1
	Crypta	r1
	Round3a	r4
	Cryptb	r4
	LDMIA	r14!,{r1,r4}
	Round3b	r1
	Crypta	r1
	Round3a	r4
	Cryptb	r4
	Round3b	r5
	Crypta	r5
	Round3a	r8
	Cryptb	r8
	Round3b	r9
	Crypta	r9
	Round3a	rA
	Cryptb	rA
	Round3b	rB

lastreg	SETS	"rC"

	LDR	r14,[r13,#12]
	ADD	$lastreg, r0, $lastreg, ROR #29

	SUB	r0,r14,$lastreg,ROR #29
	EOR	r0,r6,r0,ROR r6

check_r7
	TEQ	r0,r7
	beq	check_r6
missed
	ldr	r14,[r13,#16]
	ldr	r2,[r13,#24+4]
	subs	r14,r14,#1
	beq	the_end
	str	r14,[r13,#16]
	ADD	r14,r13,#32+16+4


; increments 32107654
inc_1st
	adds	r2,r2,#&01000000
	bcc	timingloop2

	add	r2,r2,#&00010000
	tst	r2,   #&00ff0000
	bne	timingloop2

	sub	r2,r2,#&01000000
	add	r2,r2,#&00000100
	tst	r2,   #&0000ff00
	bne	timingloop2

	add	r2,r2,#&00000001
	ands	r2,r2,#&000000ff
	bne	timingloop2

; not likely to happen very often...
	ldr	r3,[r13,#28+4]
	adds	r3,r3,#&01000000
	bcc	timingloop

	add	r3,r3,#&00010000
	tst	r3,   #&00ff0000
	bne	timingloop
	sub	r3,r3,#&01000000
	add	r3,r3,#&00000100
	tst	r3,   #&0000ff00
	bne	timingloop
	add	r3,r3,#&00000001
	and	r3,r3,#&000000ff
	b	timingloop


; increments 32107654 before leaving
the_end
	add	r13,r13,#4*4+4
	ldmfd	r13!,{r0-r3}
	adds	r2,r2,#&01000000
	bcc	function_exit

	add	r2,r2,#&00010000
	tst	r2,   #&00ff0000
	bne	function_exit

	sub	r2,r2,#&01000000
	add	r2,r2,#&00000100
	tst	r2,   #&0000ff00
	bne	function_exit

	add	r2,r2,#&00000001
	ands	r2,r2,#&000000ff
	bne	function_exit

; not likely to happen very often...
	adds	r3,r3,#&01000000
	bcc	function_exit

	add	r3,r3,#&00010000
	tst	r3,   #&00ff0000
	bne	function_exit
	sub	r3,r3,#&01000000
	add	r3,r3,#&00000100
	tst	r3,   #&0000ff00
	bne	function_exit
	add	r3,r3,#&00000001
	and	r3,r3,#&000000ff

function_exit
	sub	r0,r0,r14
	stmia	r1, {r2-r3}
	ADD	r13,r13,#T*4+32-16-24
	LDMIA	r13!,{r4-r12, pc}^

check_r6
	ADD	r0,r2,$lastreg,ROR #29

	RSB	r0,r0,#0
	SUB	r3,r3,r0
	MOV	r3,r3,ROR r0

	ADD	r0,r3,$lastreg,ROR #29
	LDR	$lastreg,[r13,#32+16+4+T*4-24-4]

	ADD	r0,r0,$lastreg,ROR #29
	MOV	$lastreg,r0,ROR #29

	LDR	r0,[r13,#8]
	SUB	r0,r0,$lastreg
	EOR	r0,r14,r0,ROR r14

	TEQ	r0, r6
	bne	missed

	ldr	r14,[r13,#16]
	add	r13,r13,#4*4+4
	ldmfd	r13!,{r0-r3}

	b	function_exit

	END

