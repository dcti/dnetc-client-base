;-------------------------------------------------------------------
; ARM optimised RC5-64 core, 1 key at a time
;
; Steve Lee, Chris Berry, Tim Dobson 1997,1998
;
;
; $Log: rc5-arm-1.s,v $
; Revision 1.2  1999/01/06 14:45:21  chrisb
; changed to allow cores to increment the high word of the key, so they pass -test. rc5-arm-3.s coming soon.
;
; Revision 1.1  1998/09/25 11:33:32  chrisb
; rc5-arm-1.s rc5-arm-2.s and rc5-arm-3.s replace rc5-arm.s and rc5-sa.s
;
;
;
	
	AREA	fastrc5area, CODE

        DCB     "@(#)$Id: rc5-arm-1.s,v 1.2 1999/01/06 14:45:21 chrisb Exp $", 0
        ALIGN

	EXPORT	rc5_unit_func_arm_1

R	*	12
T	*	2*(R+1)

P32	*	&B7E15163
Q32	*	&9E3779B9
P32_ROR29 *	(P32 :SHL: 3) :OR: (P32 :SHR: 29)

rA	RN	10
rB	RN	11
rC	RN	12

	GBLS	regpool
	GBLS	pool2
	GBLS	regtodo
	GBLS	regload
	GBLS	lastreg
	GBLS	thisreg
	GBLA	rcount

; does a LDR or LDM depending on number of registers required
; assumes inc is IA or DB
	MACRO
	MemTran	$type, $inc, $reg, $list
 [ (:LEN: "$list") > 4
	$type.M$inc	$reg,{$list}
 |
	LCLL	doinc

doinc	SETL	("$reg" :RIGHT: 1) = "!"
  [ doinc
	LCLS	realreg
realreg	SETS	"$reg" :LEFT: ((:LEN: "$reg")-1)
   [ "$inc" = "IA"
	$type.R	$list, [$realreg],#4
   |
	$type.R	$list, [$realreg,#-4]!
   ]
  |
   [ "$inc" = "IA"
	$type.R	$list, [$reg]
   |
	$type.R	$list, [$reg,#-4]
   ]
  ]
 ]
	MEND


; This macro fills as many registers as needed in $list
; from the memory pointed to by $reg.
	MACRO
	LoadRegs	$inc, $reg, $list
	ASSERT	rcount>0
 [ rcount >= (1+:LEN:$list)/3
rcount	SETA	rcount-(1+:LEN:$list)/3
regload	SETS	$list
 |
regload	SETS	$list:LEFT:(rcount*3-1)
rcount	SETA	0
 ]
	MemTran	LD,$inc,$reg,"$regload"
	MEND


; When the current register pool is completely processed, stores
; it out.
	MACRO
	StoreRegs	$reg
 [ :LEN:regload > 1 :LAND: :LEN:regtodo = 0
	MemTran	ST,IA,$reg,"$regload"
 ]
	MEND


; Sets thisreg to the next available register from the loaded pool.
; if none are available then loads some more. This version works left
; to right.
	MACRO
	CycleRegsR	$reg
lastreg SETS	thisreg
 [ :LEN:regtodo = 0
	ASSERT	:LEN:pool2 = 0 :LOR: :LEN:pool2 = 5
  [ :LEN:pool2 = 5
; we have a pool2. This is a pool of the last two available
; registers. only one of these is filled each time round so the
; last one from one filling is available after the next filling
   [ :LEN: regpool = 0
regtodo	SETS	pool2:LEFT:2
   |
regtodo	SETS	regpool:CC:",":CC:(pool2:LEFT:2)
   ]
pool2	SETS	(pool2:RIGHT:2):CC:",":CC:(pool2:LEFT:2)
  |
regtodo	SETS	regpool
  ]
	LoadRegs	IA, $reg, regtodo
regtodo	SETS	regload
 ]

thisreg	SETS	regtodo:LEFT:2
 [ :LEN:regtodo > 3
regtodo	SETS	regtodo:RIGHT:(:LEN:regtodo-3)
 |
regtodo	SETS	""
 ]
	MEND


; this macro does not use pool2 or set lastreg
	MACRO
	CycleRegsL	$reg
 [ :LEN:regtodo = 0
	LoadRegs	DB, $reg, regpool
regtodo	SETS	regload
 ]
thisreg	SETS	regtodo:RIGHT:2
 [ :LEN:regtodo > 3
regtodo	SETS	regtodo:LEFT:(:LEN:regtodo-3)
 |
regtodo	SETS	""
 ]
	MEND


	MACRO
	Round1	$inner
 [ Inner = 0
	ADR	r1, pqtable
regpool	SETS	"r4,r5,r6,r7,r8,r9"
pool2	SETS	"rA,rB"
rcount	SETA	T

	CycleRegsR	r1!

	ADD	r3, r3, $thisreg, ROR #29
	MOV	r3, r3, ROR #(31 :AND: -P32_ROR29)

	CycleRegsR	r1!
	ADD	r0, $thisreg, $lastreg, ROR #29
	ADD	$thisreg, r0, r3

	ADD	r0, r3, $thisreg, ROR #29
	RSB	r0, r0, #0

	CycleRegsR	r1!
	ADD	$thisreg, $thisreg, $lastreg, ROR #29
	STR	$lastreg,[r12,#-16]
	STMDB	r12,{r0,r3,$thisreg}
	B	skippy
timingloop2
	ADR	r1, pqtable+12
	LDMIA	r1!, {$regtodo}
	LDMDB	r12,{r0,r3,$thisreg}
skippy
	ADD	r12,r12,#8
regload	SETS	thisreg:CC:",":CC:regtodo

	STR	r2,[r13,#24]
 |
  [ Inner = 1
	ADD	$thisreg, $thisreg, r2
  |
	CycleRegsR	r1!
	ADD	$thisreg, $thisreg, $lastreg, ROR #29
	ADD	$thisreg, $thisreg, r2
  ]
	StoreRegs	r12!

	ADD	r0, r2, $thisreg, ROR #29
	RSB	r0, r0, #0
	SUB	r3, r3, r0
	MOV	r3, r3, ROR r0

	CycleRegsR	r1!
	ADD	r0, $thisreg, $lastreg, ROR #29
	ADD	$thisreg, r0, r3
	StoreRegs	r12!

	ADD	r0, r3, $thisreg, ROR #29
	RSB	r0, r0, #0
 ]
	SUB	r2, r2, r0
	MOV	r2, r2, ROR r0

	MEND


	MACRO
	Round2 $inner

 [ Inner = 0
	SUB	r12,r12,#T*4
regpool	SETS	"r1,r4,r5,r6,r7,r8,r9,rA,rB"
pool2	SETS	""
rcount	SETA	T
 ]

	ADD	r0, r2, $thisreg, ROR #29
 [ Inner = 0
	LDR	r1,pqtable
	LDR	r4,[r12,#-16]
regload	SETS	"r1,r4"
regtodo	SETS	regload
rcount	SETA	rcount-2
 ]
	CycleRegsR	r12
	ADD	$thisreg, r0, $thisreg, ROR #29
	ADD	r0, r2, $thisreg, ROR #29
	StoreRegs	r12!

	RSB	r0, r0, #0
	SUB	r3, r3, r0
	MOV	r3, r3, ROR r0

	ADD	r0, r3, $thisreg, ROR #29
	CycleRegsR	r12
	ADD	$thisreg, r0, $thisreg, ROR #29

	ADD	r0, r3, $thisreg, ROR #29
	StoreRegs	r12!

	RSB	r0, r0, #0
	SUB	r2, r2, r0
	MOV	r2, r2, ROR r0

	MEND


	MACRO
	Round3 $inner

 [ Inner = 0
	SUB	r12,r12,#T*4
regpool	SETS	"r1,r4,r5,r8,r9,rA,rB"
pool2	SETS	""
rcount	SETA	T

	LDMIA	r13,{r6,r7}
 ]
	ADD	r0, r2, $thisreg, ROR #29
	CycleRegsR	r12!
 [ Inner = T/2-1
	LDR	r12,[r13,#12]
 ]
	ADD	$thisreg, r0, $thisreg, ROR #29
 [ Inner <> T/2-1
  [ Inner <> 0
	EOR	r7,r7,r6
	RSB	r0,r6,#0
	MOV	r7,r7,ROR r0
  ]
	ADD	r7,r7,$thisreg, ROR #29

	ADD	r0, r2, $thisreg, ROR #29

	RSB	r0, r0, #0
	SUB	r3, r3, r0
	MOV	r3, r3, ROR r0

	ADD	r0, r3, $thisreg,ROR #29

	CycleRegsR	r12!
	ADD	$thisreg, r0, $thisreg, ROR #29

  [ Inner <> 0
	EOR	r6,r6,r7
	RSB	r0,r7,#0
	MOV	r6,r6,ROR r0
  ]
	ADD	r6,r6,$thisreg,ROR #29

	ADD	r0, r3, $thisreg, ROR #29
	RSB	r0, r0, #0
	SUB	r2, r2, r0
	MOV	r2, r2, ROR r0
 ]
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
	STR	r14,[r13,#-(T*4+32)]!
	STR	r0,[r13,#4]
	ADD	r12,r13,#32
	STMFD	r13!, {r4-r7}
timingloop
	str	r3,[r13,#28]

Inner	SETA	0
	WHILE	Inner < T/2
	Round1	Inner
Inner	SETA	Inner + 1
	WEND
Inner	SETA	0
	WHILE	Inner < T/2
	Round2	Inner
Inner	SETA	Inner + 1
	WEND
Inner	SETA	0
	WHILE	Inner < T/2
	Round3	Inner
Inner	SETA	Inner + 1
	WEND

	SUB	r0,r12,$thisreg,ROR #29
	EOR	r0,r6,r0,ROR r6

check_r7
	TEQ	r0,r7
	beq	check_r6
missed
	ADD	r12,r13,#32+16
	ldr	r2,[r13,#24]
	subs	r14,r14,#1
	beq	the_end


; increments 32107654
inc_1st
	adds	r2,r2,#&01000000
	bcc	timingloop2

carry_1st
	add	r2,r2,#&00010000
	tst	r2,   #&00ff0000
	bne	timingloop2
	sub	r2,r2,#&01000000

	add	r2,r2,#&00000100
	tst	r2,   #&0000ff00
	bne	timingloop2
	sub	r2,r2,#&00010000

	add	r2,r2,#&00000001
	ands	r2,r2,#&000000ff
; will never increment high word
	;	b	timingloop2	
	bne	timingloop2
;
;; not likely to happen very often...
	ldr	r3,[r13,#28]
	adds	r3,r3,#&01000000
	bcc	timingloop
;
carry_1st_again
	add	r3,r3,#&00010000
	tst	r3,   #&00ff0000
	bne	timingloop
	sub	r3,r3,#&01000000
	add	r3,r3,#&00000100
	tst	r3,   #&0000ff00
	bne	timingloop
	sub	r3,r3,#&00010000
	add	r3,r3,#&00000001
	and	r3,r3,#&000000ff
	b	timingloop




; increments 32107654 before leaving
the_end
	add	r13,r13,#4*4
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
	sub	r2,r2,#&00010000

	add	r2,r2,#&00000001
	ands	r2,r2,#&000000ff
; increment of high word only at end of block - result does not matter then.
	;; 	b	function_exit
	bne	function_exit
;
;; not likely to happen very often...
	adds	r3,r3,#&01000000
	bcc	function_exit
;
	add	r3,r3,#&00010000
	tst	r3,   #&00ff0000
	bne	function_exit
	sub	r3,r3,#&01000000
	add	r3,r3,#&00000100
	tst	r3,   #&0000ff00
	bne	function_exit
	sub	r3,r3,#&00010000
	add	r3,r3,#&00000001
	and	r3,r3,#&000000ff

function_exit
	sub	r0,r0,r14
	stmia	r1, {r2-r3}
	ADD	r13,r13,#T*4+32-16
	LDMIA	r13!,{r4-r12, pc}^

check_r6
	ADD	r0,r2,$thisreg,ROR #29

	RSB	r0,r0,#0
	SUB	r3,r3,r0
	MOV	r3,r3,ROR r0

	ADD	r0,r3,$thisreg,ROR #29
	LDR	$thisreg,[r13,#32+16+T*4-4]
	ADD	r0,r0,$thisreg,ROR #29
	MOV	$thisreg,r0,ROR #29

	LDR	r0,[r13,#8]
	SUB	r0,r0,$thisreg
	EOR	r0,r12,r0,ROR r12

	TEQ	r0, r6
	bne	missed

	add	r13,r13,#4*4
	ldmfd	r13!,{r0-r3}

	b	function_exit

	END

