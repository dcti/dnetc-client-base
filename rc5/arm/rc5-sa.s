;-------------------------------------------------------------------
; StrongARM optimised RC5-64 core
;
; Steve Lee, Chris Berry, Tim Dobson 1997,1998
;
; $Log: rc5-sa.s,v $
; Revision 1.7  1998/08/07 10:57:22  cberry
; Removed unnecessary branch instruction
;
; Revision 1.6  1998/08/05 13:08:23  cberry
; New cores from Steve Lee.
;
; Revision 1.5  1998/06/15 12:04:16  kbracey
; Lots of consts.
;
; Revision 1.4  1998/06/15 10:49:42  kbracey
; Added ident strings
;
; Revision 1.3  1998/06/14 10:30:44  friedbait
; 'Log' keyword added.
;
;

        AREA    fastrc5area, CODE, READONLY

        DCB     "@(#)$Id: rc5-sa.s,v 1.7 1998/08/07 10:57:22 cberry Exp $", 0
        ALIGN

        EXPORT  rc5_unit_func_strongarm

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
	GBLS	last1
	GBLS	last2
	GBLS	this1
	GBLS	this2
	GBLS	tmpreg
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
	LoadRegs	$inc, $reg, $list, $alt
	ASSERT	rcount>0
 [ rcount >= (1+:LEN:$list)/3
rcount	SETA	rcount-(1+:LEN:$list)/3
regload	SETS	$list
 |
regload	SETS	$list:LEFT:(rcount*3-1)
rcount	SETA	0
 ]
 [ $alt = 0
	MemTran	LD,$inc,$reg,"$regload"
 |
	LCLS	allreg
	LCLS	altload
	ASSERT	(1+:LEN:regload)/3 :MOD: 2 = 0
allreg	SETS	regload
altload	SETS	""
	WHILE	:LEN:allreg>6
altload	SETS	altload:CC:(allreg:RIGHT:3)
allreg	SETS	allreg:LEFT:(:LEN:allreg - 6)
	WEND
altload	SETS	(allreg:RIGHT:2):CC:altload
	MemTran	LD,$inc,$reg,"$altload"
 ]
regtodo	SETS	regload
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
	CycleRegsR	$reg, $alt
lastreg	SETS	thisreg
 [ :LEN:regtodo = 0
	ASSERT	:LEN:pool2 = 0 :LOR: :LEN:pool2 = 5
  [ :LEN:pool2 = 5
; we have a pool2. This is a pool of the last two available
; registers. only one of these is filled each time round so the
; last one from one filling is available after the next filling
regtodo	SETS	regpool:CC:",":CC:(pool2:LEFT:2)
pool2	SETS	(pool2:RIGHT:2):CC:",":CC:(pool2:LEFT:2)
  |
regtodo	SETS	regpool
  ]
	LoadRegs	IA, $reg, regtodo, $alt
 ]

thisreg	SETS	regtodo:LEFT:2
 [ :LEN:regtodo > 3
regtodo	SETS	regtodo:RIGHT:(:LEN:regtodo-3)
 |
regtodo	SETS	""
 ]
	MEND


; this macro must only be called when the total regpool (with pool2
; halved) is even. last1 and last2 should only be used if alternate
; loading is set and a pool2 is available.
; when alternate loading is used, only this2 contains valid input data.
	MACRO
	CycleRegsR2	$reg, $alt
last1	SETS	this1
last2	SETS	this2
	CycleRegsR	$reg, $alt
this1	SETS	thisreg
	CycleRegsR	$reg, $alt
this2	SETS	thisreg
	MEND


; this macro does not use pool2 or set lastreg
	MACRO
	CycleRegsL	$reg
 [ :LEN:regtodo = 0
	LoadRegs	DB, $reg, regpool, 0
 ]
thisreg	SETS	regtodo:RIGHT:2
 [ :LEN:regtodo > 3
regtodo	SETS	regtodo:LEFT:(:LEN:regtodo-3)
 |
regtodo	SETS	""
 ]
	MEND


	MACRO
	CycleRegsL2	$reg
	CycleRegsL	$reg
this2	SETS	thisreg
	CycleRegsL	$reg
this1	SETS	thisreg
	MEND


	MACRO
	Round1 $inner

 [ Inner = 0
	ADR	r6, pqtable
regpool	SETS	"r7,r8,r9"  ; odd number of registers
pool2	SETS	"rA,rB"     ; plus one of these makes it even
rcount	SETA	T*2

	CycleRegsR2	r6!, 1
	ADD	r3, r3, $this2, ROR #29
	MOV	r3, r3, ROR #(31 :AND: -P32_ROR29)

	CycleRegsR2	r6!, 1
	ADD	r1, $this2, $last2, ROR #29
	ADD	$this2, r1, r3

	ADD	r1, r3, $this2, ROR #29
tmpreg	SETS	last2
	LoadRegs	IA, r6!, """$last1,$this1""", 1
	CycleRegsR2	r6!, 1
	RSB	r1, r1, #0

	ADD	$this2, $this2, $last2, ROR #29

	STMDB	r12,{r1,r3,$last2,$this2}
	B	skippy
timingloop2
	ADR	r6, pqtable+12
	LDR	$tmpreg,[r6,#-12]
	LDMDB	r12,{r1,r3,$last2,$this2}
skippy
	str	r2,[r13,#24]
	ORR	r4,r2,#&01000000
	STMIA	r12!,{$tmpreg,$last2}
	ADD	r12,r12,#8

	SUB	r2, r2, r1
	SUB	r4, r4, r1
	MOV	r2, r2, ROR r1
	MOV	r4, r4, ROR r1
 |
  [ Inner = 1
	ADD	$this1, $this2, r2
	ADD	$this2, $this2, r4
  |
	CycleRegsR2	r6!, 1
	ADD	$this1, $this2, $last1, ROR #29
	ADD	$this2, $this2, $last2, ROR #29
	ADD	$this1, $this1, r2
	ADD	$this2, $this2, r4
  ]
	StoreRegs	r12!

	ADD	r0, r2, $this1, ROR #29
	ADD	r1, r4, $this2, ROR #29
	RSB	r0, r0, #0
	RSB	r1, r1, #0
  [ Inner = 1
	SUB	r5, r3, r1
	SUB	r3, r3, r0
  |
	SUB	r3, r3, r0
	SUB	r5, r5, r1
  ]
	CycleRegsR2	r6!, 1
	MOV	r3, r3, ROR r0
	MOV	r5, r5, ROR r1

	ADD	r0, $this2, $last1, ROR #29
	ADD	r1, $this2, $last2, ROR #29
	ADD	$this1, r0, r3
	ADD	$this2, r1, r5

	StoreRegs	r12!
	ADD	r0, r3, $this1, ROR #29
	ADD	r1, r5, $this2, ROR #29
	RSB	r0, r0, #0
	RSB	r1, r1, #0

	SUB	r2, r2, r0
	SUB	r4, r4, r1
	MOV	r2, r2, ROR r0
	MOV	r4, r4, ROR r1
 ]
	MEND


	MACRO
	Round2 $inner

	ADD	r0, r2, $this1, ROR #29
	ADD	r1, r4, $this2, ROR #29
 [ Inner = 0
	SUB	r12,r12,#T*4*2

regpool	SETS	"r6,r7,r8,r9,rA,rB"  ; even number of registers
pool2	SETS	""
rcount	SETA	4
	CycleRegsR2	r12, 1
rcount	SETA	T*2-4

	ADD	$this1, r0, $this2, ROR #29
 |
	CycleRegsR2	r12, 0
	ADD	$this1, r0, $this1, ROR #29
 ]
	ADD	$this2, r1, $this2, ROR #29
	ADD	r0, r2, $this1, ROR #29
	ADD	r1, r4, $this2, ROR #29
	StoreRegs	r12!

	RSB	r0, r0, #0
	RSB	r1, r1, #0
	SUB	r3, r3, r0
	SUB	r5, r5, r1
	MOV	r3, r3, ROR r0
	MOV	r5, r5, ROR r1

	ADD	r0, r3, $this1, ROR #29
	ADD	r1, r5, $this2, ROR #29
 [ Inner = 0
	CycleRegsR2	r12, 1
	ADD	$this1, r0, $this2, ROR #29
 |
	CycleRegsR2	r12, 0
	ADD	$this1, r0, $this1, ROR #29
 ]
	ADD	$this2, r1, $this2, ROR #29

	ADD	r0, r3, $this1, ROR #29
	ADD	r1, r5, $this2, ROR #29
	StoreRegs	r12!

	RSB	r0, r0, #0
	RSB	r1, r1, #0
	SUB	r2, r2, r0
	SUB	r4, r4, r1
	MOV	r2, r2, ROR r0
	MOV	r4, r4, ROR r1

	MEND


	MACRO
	Round3 $inner

 [ Inner = 0
	SUB	r12,r12,#T*4*2
regpool	SETS	"rA,rB"  ; even number of registers
pool2	SETS	""
rcount	SETA	T*2

	LDMIA	r13,{r8,r9}
 ]
 [ Inner <= 1
	ADD	r0, r2, $this1, ROR #29
	ADD	r1, r4, $this2, ROR #29
 |
	ADD	r0, r2, $this1
	ADD	r1, r4, $this2
 ]
	CycleRegsR2	r12!, 0
 [ Inner = T/2-1
	LDR	r12,[r13,#12]
 ]
	ADD	$this1, r0, $this1, ROR #29
	ADD	$this2, r1, $this2, ROR #29
 [ Inner <> T/2-1
  [ Inner = 0
	ADD	r7,r9,$this1, ROR #29
	ADD	r9,r9,$this2, ROR #29

	ADD	r0, r2, $this1, ROR #29
	ADD	r1, r4, $this2, ROR #29
  |
	MOV	$this1, $this1, ROR #29
	MOV	$this2, $this2, ROR #29
	EOR	r7,r7,r6
	EOR	r9,r9,r8
	RSB	r0,r6,#0
	RSB	r1,r8,#0
	ADD	r7,$this1,r7,ROR r0
	ADD	r9,$this2,r9,ROR r1

	ADD	r0, r2, $this1
	ADD	r1, r4, $this2
  ]


	RSB	r0, r0, #0
	RSB	r1, r1, #0
	SUB	r3, r3, r0
	SUB	r5, r5, r1
	MOV	r3, r3, ROR r0
	MOV	r5, r5, ROR r1

  [ Inner = 0
	ADD	r0, r3, $this1, ROR #29
	ADD	r1, r5, $this2, ROR #29
  |
	ADD	r0, r3, $this1
	ADD	r1, r5, $this2
  ]
	CycleRegsR2	r12!, 0
	ADD	$this1, r0, $this1, ROR #29
	ADD	$this2, r1, $this2, ROR #29
  [ Inner = 0
	ADD	r6,r8,$this1, ROR #29
	ADD	r8,r8,$this2, ROR #29

	ADD	r0, r3, $this1, ROR #29
	ADD	r1, r5, $this2, ROR #29
  |
	MOV	$this1, $this1, ROR #29
	MOV	$this2, $this2, ROR #29
	EOR	r6,r6,r7
	EOR	r8,r8,r9
	RSB	r0,r7,#0
	RSB	r1,r9,#0
	ADD	r6,$this1,r6,ROR r0
	ADD	r8,$this2,r8,ROR r1

	ADD	r0, r3, $this1
	ADD	r1, r5, $this2
  ]

	RSB	r0, r0, #0
	RSB	r1, r1, #0
	SUB	r2, r2, r0
	SUB	r4, r4, r1
	MOV	r2, r2, ROR r0
	MOV	r4, r4, ROR r1
 ]
	MEND


	GBLA	TMP
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


	IMPORT	__rt_stkovf_split_big
rc5_unit_func_strongarm
	mov	r12,r13
	STMFD	r13!, {r4-r12,r14,pc}
	sub	r11,r12,#4
	sub	r12,r13,#&200
	cmps	r12,r10
	bllt	__rt_stkovf_split_big

	mov	r14, r1
	stmfd	r13!,{r11}

	LDMIA	r0!,{r4-r7}
	LDMIA	r0, {r2-r3}

	TST	r2,#&01000000
	BNE	odd

	MOV	r12,r14,ASL#1
	STR	r12,[r13,#-(T*4*2 +32)]!
	STR	r0,[r13,#4]
	ADD	r12,r13,#32
	STMFD	r13!, {r4-r7}

timingloop
	STR	r3,[r13,#28]
;key expansion
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


	SUB	r0,r12,$this1,ROR #29
	SUB	r1,r12,$this2,ROR #29
	EOR	r0,r6,r0,ROR r6
	EOR	r1,r8,r1,ROR r8


; check
check_r9
	teq	r1,r9
	beq	check_r8
check_r7
	TEQ	r0,r7
	beq	check_r6
missed
	ADD	r12,r13,#32+16
	subs	r14,r14,#1
	beq	the_end

	ldr	r2,[r13,#24]

; increments 32107654
inc_1st
	adds	r2,r2,#&02000000
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
	b	timingloop2
;	bne	timingloop2
;
;; not likely to happen very often...
;	ldr	r3,[r13,#28]
;	adds	r3,r3,#&01000000
;	bcc	timingloop
;
;	add	r3,r3,#&00010000
;	tst	r3,   #&00ff0000
;	bne	timingloop
;	sub	r3,r3,#&01000000
;	add	r3,r3,#&00000100
;	tst	r3,   #&0000ff00
;	bne	timingloop
;	sub	r3,r3,#&00010000
;	add	r3,r3,#&00000001
;	and	r3,r3,#&000000ff
;	b	timingloop



the_end
; increments 32107654 before leaving
	add	r13,r13,#16
	ldmia	r13!,{r0-r3}

	add	r2,r2,#&02000000
	tst	r2,   #&ff000000
	bne	function_exit

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
;	b	function_exit
;	bne	function_exit
;
;; not likely to happen very often...
;	adds	r3,r3,#&01000000
;	tst	r3,   #&ff000000
;	bne	function_exit
;
;	add	r3,r3,#&00010000
;	tst	r3,   #&00ff0000
;	bne	function_exit
;	sub	r3,r3,#&01000000
;	add	r3,r3,#&00000100
;	tst	r3,   #&0000ff00
;	bne	function_exit
;	sub	r3,r3,#&00010000
;	add	r3,r3,#&00000001
;	and	r3,r3,#&000000ff


function_exit
	ldr	r11,[r13,#(T*2)*4+32-16]
	stmia	r1,{r2,r3}
	ldmdb	r11, {r4-r11,r13, pc}

check_r8
	ADD	r1, r4, $this2,ROR #29

	RSB	r1, r1, #0
	SUB	r5, r5, r1
	MOV	r5, r5, ROR r1

	ADD	r1, r5, $this2,ROR #29
	LDR	$this2,[r13,#32+16+T*2*4-4]
	ADD	r1, r1, $this2, ROR #29
	MOV	$this2, r1, ROR #29

	LDR	r1,[r13,#8]
	SUB	r1,r1,$this2
	EOR	r1,r12,r1,ROR r12

	TEQ	r1, r8
	bne	check_r7

;it's r4,r5!
	add	r13,r13,#16
	ldmia	r13!,{r0-r3}
	sub	r0,r0,r14,ASL#1
	add	r0,r0,#1
	orr	r2,r2,#&01000000
	b	function_exit


check_r6
	ADD	r0, r2, $this1,ROR #29

	RSB	r0, r0, #0
	SUB	r3, r3, r0
	MOV	r3, r3, ROR r0

	ADD	r0, r3, $this1,ROR #29
	LDR	$this1,[r13,#32+16+T*2*4-8]
	ADD	r0, r0, $this1, ROR #29
	MOV	$this1, r0, ROR #29

	LDR	r0,[r13,#8]
	SUB	r0,r0,$this1
	EOR	r0,r12,r0,ROR r12


	TEQ	r0, r6
	bne	missed

; it's r2,r3!
	add	r13,r13,#16
	ldmia	r13!,{r0-r3}
	sub	r0,r0,r14,ASL#1
	b	function_exit


	IMPORT	printf
odd	ADR	r0,odd_message
	MOV	r1,r3
	BL	printf
infinite_loop
	B	infinite_loop

odd_message
	DCB	"Hey! I've got an odd starting key: %08lx:%08lx\nSTOP THIS AT ONCE!!!\0"

	END

