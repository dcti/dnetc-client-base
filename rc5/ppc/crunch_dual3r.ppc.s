	
	EXPORT	crunch_dual3r[DS]
	EXPORT	.crunch_dual3r[PR]
	
RC5UnitWork	RECORD	; Note: data is now in RC5/platform useful form
plain_hi	ds.l	1	; plaintext, already mixed with IV
plain_lo	ds.l	1
cypher_hi	ds.l	1	; target cyphertext
cypher_lo	ds.l	1
L0_hi		ds.l	1	; key, changes with every unit * PIPELINE_COUNT
L0_lo		ds.l	1
		ENDR
		
	CSECT	initKS[RO]
		dc.l	0xbf0a8b1d
		dc.l	0x5618cb1c
		dc.l	0xf45044d5
		dc.l	0x9287be8e
		dc.l	0x30bf3847
		dc.l	0xcef6b200
		dc.l	0x6d2e2bb9
		dc.l	0x0b65a572
		dc.l	0xa99d1f2b
		dc.l	0x47d498e4
		dc.l	0xe60c129d
		dc.l	0x84438c56
		dc.l	0x227b060f
		dc.l	0xc0b27fc8
		dc.l	0x5ee9f981
		dc.l	0xfd21733a
		dc.l	0x9b58ecf3
		dc.l	0x399066ac
		dc.l	0xd7c7e065
		dc.l	0x75ff5a1e
		dc.l	0x1436d3d7
		dc.l	0xb26e4d90
		dc.l	0x50a5c749
		dc.l	0xeedd4102
		dc.l	0x8d14babb
		dc.l	0x2b4c3474

	TOC
		tc	crunch_dual3r[TC],crunch_dual3r[DS]
initKS	tc	initKS[TC],initKS[RO]
	
	CSECT	crunch_dual3r[DS]
		dc.l	.crunch_dual3r[PR]
		dc.l	TOC[tc0]
		dc.l	0
		
	WITH RC5UnitWork
	CSECT	.crunch_dual3r[PR]
; int crunch( RC5UnitWork *data, unsigned long iterations )
		
dataptr		set	r3
iterations	set	r4

s0X			set	r5
s1X			set	r6
s2X			set	r7

crX			set	cr0

s0Y			set	r8
s1Y			set	r9
s2Y			set	r10

crY			set	cr1

ival		set	r0
AX			set	r11
BX			set r12
AY			set r13
BY			set r14
tX			set	r15
tY			set	r16
key0X		set	r17
key1X		set	r18
key0Y		set	r19
key1Y		set	r20
keylo		set	r21
keyhi		set	r22
plain0		set	r23
plain1		set	r24
cypher0		set	r25
cypher1		set	key1X
initptr		set	iterations

ss1Y		set cypher0
ss2Y		set r2
ss5Y		set keylo
ss6Y		set keyhi
ss9Y		set r26
ss10Y		set r27
ss13Y		set r28
ss14Y		set r29
ss17Y		set r30
ss18Y		set r31

schedX		set	0
schedY		set	26*4

prologue
		subi	SP,SP,2*(schedY-schedX)
		stw		r31,-4(SP)
		stw		r30,-8(SP)
		stw		r29,-12(SP)
		stw		r28,-16(SP)
		stw		r27,-20(SP)
		stw		r26,-24(SP)
		stw		r25,-28(SP)
		stw		r24,-32(SP)
		stw		r23,-36(SP)
		stw		r22,-40(SP)
		stw		r21,-44(SP)
		stw		r20,-48(SP)
		stw		r19,-52(SP)
		stw		r18,-56(SP)
		stw		r17,-60(SP)
		stw		r16,-64(SP)
		stw		r15,-68(SP)
		stw		r14,-72(SP)
		stw		r2,-76(SP)
		
		stw		iterations,-80(SP)
		
		rlwinm	r5,r4,31,1,31
		
		lwz		keylo,L0_lo(dataptr)
		lwz		keyhi,L0_hi(dataptr)
		lwz		plain0,plain_lo(dataptr)
		lwz		plain1,plain_hi(dataptr)
		lwz		cypher0,cypher_lo(dataptr)
		lwz		initptr,initKS(RTOC)
		
		
		mtctr	r5
checkkey

pass1
		lwz		s0X,0*4(initptr)
		lwz		ival,1*4(initptr)
		add		s0Y,keylo,s0X
		
		add		s1X,ival,s0X
		rlwinm	s0Y,s0Y,29,0,31
		add		ss1Y,ival,s0X
		add		s1X,s1X,s0Y
		add		ss1Y,ss1Y,s0Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	ss1Y,ss1Y,3,0,31
		lwz		ival,2*4(initptr)
		addis	key1Y,keyhi,1
		add		tX,s1X,s0Y
		add		tY,ss1Y,s0Y
		add		key1X,keyhi,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s1X,1*4+schedX(SP)
		add		s2X,ival,s1X
		add		ss2Y,ival,ss1Y
		add		s2X,s2X,key1X
		add		ss2Y,ss2Y,key1Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	ss2Y,ss2Y,3,0,31
		lwz		ival,3*4(initptr)
		add		tX,s2X,key1X
		add		tY,ss2Y,key1Y
		add		key0X,s0Y,tX
		add		key0Y,s0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s2X,2*4+schedX(SP)
		add		s0X,ival,s2X
		add		s0Y,ival,ss2Y
		add		s0X,s0X,key0X
		add		s0Y,s0Y,key0Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		lwz		ival,4*4(initptr)
		add		tX,s0X,key0X
		add		tY,s0Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s0X,3*4+schedX(SP)
		add		s1X,ival,s0X
		add		s1Y,ival,s0Y
		add		s1X,s1X,key1X
		add		s1Y,s1Y,key1Y
		lwz		ival,5*4(initptr)
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		add		tX,s1X,key1X
		add		tY,s1Y,key1Y
		stw		s0Y,3*4+schedY(SP)
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s1X,4*4+schedX(SP)
		add		s2X,ival,s1X
		add		ss5Y,ival,s1Y
		add		s2X,s2X,key0X
		add		ss5Y,ss5Y,key0Y
		lwz		ival,6*4(initptr)
		rlwinm	s2X,s2X,3,0,31
		rlwinm	ss5Y,ss5Y,3,0,31
		add		tX,s2X,key0X
		add		tY,ss5Y,key0Y
		stw		s1Y,4*4+schedY(SP)
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s2X,5*4+schedX(SP)
		add		s0X,ival,s2X
		add		ss6Y,ival,ss5Y
		add		s0X,s0X,key1X
		add		ss6Y,ss6Y,key1Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	ss6Y,ss6Y,3,0,31
		lwz		ival,7*4(initptr)
		add		tX,s0X,key1X
		add		tY,ss6Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s0X,6*4+schedX(SP)
		add		s1X,ival,s0X
		add		s1Y,ival,ss6Y
		add		s1X,s1X,key0X
		add		s1Y,s1Y,key0Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		lwz		ival,8*4(initptr)
		add		tX,s1X,key0X
		add		tY,s1Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s1X,7*4+schedX(SP)
		add		s2X,ival,s1X
		add		s2Y,ival,s1Y
		add		s2X,s2X,key1X
		add		s2Y,s2Y,key1Y
		lwz		ival,9*4(initptr)
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		add		tX,s2X,key1X
		add		tY,s2Y,key1Y
		stw		s1Y,7*4+schedY(SP)
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s2X,8*4+schedX(SP)
		add		s0X,ival,s2X
		add		ss9Y,ival,s2Y
		add		s0X,s0X,key0X
		add		ss9Y,ss9Y,key0Y
		lwz		ival,10*4(initptr)
		rlwinm	s0X,s0X,3,0,31
		rlwinm	ss9Y,ss9Y,3,0,31
		add		tX,s0X,key0X
		add		tY,ss9Y,key0Y
		stw		s2Y,8*4+schedY(SP)
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s0X,9*4+schedX(SP)
		add		s1X,ival,s0X
		add		ss10Y,ival,ss9Y
		add		s1X,s1X,key1X
		add		ss10Y,ss10Y,key1Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	ss10Y,ss10Y,3,0,31
		lwz		ival,11*4(initptr)
		add		tX,s1X,key1X
		add		tY,ss10Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s1X,10*4+schedX(SP)
		add		s2X,ival,s1X
		add		s2Y,ival,ss10Y
		add		s2X,s2X,key0X
		add		s2Y,s2Y,key0Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		lwz		ival,12*4(initptr)
		add		tX,s2X,key0X
		add		tY,s2Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s2X,11*4+schedX(SP)
		add		s0X,ival,s2X
		add		s0Y,ival,s2Y
		add		s0X,s0X,key1X
		add		s0Y,s0Y,key1Y
		lwz		ival,13*4(initptr)
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		add		tX,s0X,key1X
		add		tY,s0Y,key1Y
		stw		s2Y,11*4+schedY(SP)
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s0X,12*4+schedX(SP)
		add		s1X,ival,s0X
		add		ss13Y,ival,s0Y
		add		s1X,s1X,key0X
		add		ss13Y,ss13Y,key0Y
		lwz		ival,14*4(initptr)
		rlwinm	s1X,s1X,3,0,31
		rlwinm	ss13Y,ss13Y,3,0,31
		add		tX,s1X,key0X
		add		tY,ss13Y,key0Y
		stw		s0Y,12*4+schedY(SP)
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s1X,13*4+schedX(SP)
		add		s2X,ival,s1X
		add		ss14Y,ival,ss13Y
		add		s2X,s2X,key1X
		add		ss14Y,ss14Y,key1Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	ss14Y,ss14Y,3,0,31
		lwz		ival,15*4(initptr)
		add		tX,s2X,key1X
		add		tY,ss14Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s2X,14*4+schedX(SP)
		add		s0X,ival,s2X
		add		s0Y,ival,ss14Y
		add		s0X,s0X,key0X
		add		s0Y,s0Y,key0Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		lwz		ival,16*4(initptr)
		add		tX,s0X,key0X
		add		tY,s0Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s0X,15*4+schedX(SP)
		add		s1X,ival,s0X
		add		s1Y,ival,s0Y
		add		s1X,s1X,key1X
		add		s1Y,s1Y,key1Y
		lwz		ival,17*4(initptr)
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		add		tX,s1X,key1X
		add		tY,s1Y,key1Y
		stw		s0Y,15*4+schedY(SP)
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s1X,16*4+schedX(SP)
		add		s2X,ival,s1X
		add		ss17Y,ival,s1Y
		add		s2X,s2X,key0X
		add		ss17Y,ss17Y,key0Y
		lwz		ival,18*4(initptr)
		rlwinm	s2X,s2X,3,0,31
		rlwinm	ss17Y,ss17Y,3,0,31
		add		tX,s2X,key0X
		add		tY,ss17Y,key0Y
		stw		s1Y,16*4+schedY(SP)
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s2X,17*4+schedX(SP)
		add		s0X,ival,s2X
		add		ss18Y,ival,ss17Y
		add		s0X,s0X,key1X
		add		ss18Y,ss18Y,key1Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	ss18Y,ss18Y,3,0,31
		lwz		ival,19*4(initptr)
		add		tX,s0X,key1X
		add		tY,ss18Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s0X,18*4+schedX(SP)
		add		s1X,ival,s0X
		add		s1Y,ival,ss18Y
		add		s1X,s1X,key0X
		add		s1Y,s1Y,key0Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		lwz		ival,20*4(initptr)
		add		tX,s1X,key0X
		add		tY,s1Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s1X,19*4+schedX(SP)
		add		s2X,ival,s1X
		add		s2Y,ival,s1Y
		add		s2X,s2X,key1X
		add		s2Y,s2Y,key1Y
		lwz		ival,21*4(initptr)
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		add		tX,s2X,key1X
		add		tY,s2Y,key1Y
		stw		s1Y,19*4+schedY(SP)
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s2X,20*4+schedX(SP)
		add		s0X,ival,s2X
		add		s0Y,ival,s2Y
		add		s0X,s0X,key0X
		add		s0Y,s0Y,key0Y
		lwz		ival,22*4(initptr)
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		add		tX,s0X,key0X
		add		tY,s0Y,key0Y
		stw		s2Y,20*4+schedY(SP)
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s0X,21*4+schedX(SP)
		add		s1X,ival,s0X
		add		s1Y,ival,s0Y
		add		s1X,s1X,key1X
		add		s1Y,s1Y,key1Y
		lwz		ival,23*4(initptr)
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		add		tX,s1X,key1X
		add		tY,s1Y,key1Y
		stw		s0Y,21*4+schedY(SP)
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s1X,22*4+schedX(SP)
		add		s2X,ival,s1X
		add		s2Y,ival,s1Y
		add		s2X,s2X,key0X
		add		s2Y,s2Y,key0Y
		lwz		ival,24*4(initptr)
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		add		tX,s2X,key0X
		add		tY,s2Y,key0Y
		stw		s1Y,22*4+schedY(SP)
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		stw		s2X,23*4+schedX(SP)
		add		s0X,ival,s2X
		add		s0Y,ival,s2Y
		add		s0X,s0X,key1X
		add		s0Y,s0Y,key1Y
		lwz		ival,25*4(initptr)
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		add		tX,s0X,key1X
		add		tY,s0Y,key1Y
		stw		s2Y,23*4+schedY(SP)
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		stw		s0X,24*4+schedX(SP)
		add		s1X,ival,s0X
		add		s1Y,ival,s0Y
		add		s1X,s1X,key0X
		add		s1Y,s1Y,key0Y
		lwz		ival,0*4(initptr)
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		add		tX,s1X,key0X
		add		tY,s1Y,key0Y
		stw		s0Y,24*4+schedY(SP)
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
pass2
		lwz		s0X,1*4+schedX(SP)
		add		s2X,ival,s1X
		add		s2Y,ival,s1Y
		add		s2X,s2X,key1X
		add		s2Y,s2Y,key1Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		stw		s1X,25*4+schedX(SP)
		add		tX,s2X,key1X
		add		tY,s2Y,key1Y
		add		key0X,key0X,tX
		stw		s1Y,25*4+schedY(SP)
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s1X,2*4+schedX(SP)
		add		s0X,s0X,s2X
		add		ss1Y,ss1Y,s2Y
		add		s0X,s0X,key0X
		add		ss1Y,ss1Y,key0Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	ss1Y,ss1Y,3,0,31
		stw		s2X,0*4+schedX(SP)
		add		tX,s0X,key0X
		add		tY,ss1Y,key0Y
		add		key1X,key1X,tX
		stw		s2Y,0*4+schedY(SP)
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s2X,3*4+schedX(SP)
		add		s1X,s1X,s0X
		add		ss2Y,ss2Y,ss1Y
		add		s1X,s1X,key1X
		lwz		s2Y,3*4+schedY(SP)
		add		ss2Y,ss2Y,key1Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	ss2Y,ss2Y,3,0,31
		stw		s0X,1*4+schedX(SP)
		add		tX,s1X,key1X
		add		tY,ss2Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s0X,4*4+schedX(SP)
		add		s2X,s2X,s1X
		add		s2Y,s2Y,ss2Y
		add		s2X,s2X,key0X
		lwz		s0Y,4*4+schedY(SP)
		add		s2Y,s2Y,key0Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		stw		s1X,2*4+schedX(SP)
		add		tX,s2X,key0X
		add		tY,s2Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s1X,5*4+schedX(SP)
		add		s0X,s0X,s2X
		add		s0Y,s0Y,s2Y
		add		s0X,s0X,key1X
		add		s0Y,s0Y,key1Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		stw		s2X,3*4+schedX(SP)
		add		tX,s0X,key1X
		add		tY,s0Y,key1Y
		add		key0X,key0X,tX
		stw		s2Y,3*4+schedY(SP)
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s2X,6*4+schedX(SP)
		add		s1X,s1X,s0X
		add		ss5Y,ss5Y,s0Y
		add		s1X,s1X,key0X
		add		ss5Y,ss5Y,key0Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	ss5Y,ss5Y,3,0,31
		stw		s0X,4*4+schedX(SP)
		add		tX,s1X,key0X
		add		tY,ss5Y,key0Y
		add		key1X,key1X,tX
		stw		s0Y,4*4+schedY(SP)
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s0X,7*4+schedX(SP)
		add		s2X,s2X,s1X
		add		ss6Y,ss6Y,ss5Y
		add		s2X,s2X,key1X
		lwz		s0Y,7*4+schedY(SP)
		add		ss6Y,ss6Y,key1Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	ss6Y,ss6Y,3,0,31
		stw		s1X,5*4+schedX(SP)
		add		tX,s2X,key1X
		add		tY,ss6Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s1X,8*4+schedX(SP)
		add		s0X,s0X,s2X
		add		s0Y,s0Y,ss6Y
		add		s0X,s0X,key0X
		lwz		s1Y,8*4+schedY(SP)
		add		s0Y,s0Y,key0Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		stw		s2X,6*4+schedX(SP)
		add		tX,s0X,key0X
		add		tY,s0Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s2X,9*4+schedX(SP)
		add		s1X,s1X,s0X
		add		s1Y,s1Y,s0Y
		add		s1X,s1X,key1X
		add		s1Y,s1Y,key1Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		stw		s0X,7*4+schedX(SP)
		add		tX,s1X,key1X
		add		tY,s1Y,key1Y
		add		key0X,key0X,tX
		stw		s0Y,7*4+schedY(SP)
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s0X,10*4+schedX(SP)
		add		s2X,s2X,s1X
		add		ss9Y,ss9Y,s1Y
		add		s2X,s2X,key0X
		add		ss9Y,ss9Y,key0Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	ss9Y,ss9Y,3,0,31
		stw		s1X,8*4+schedX(SP)
		add		tX,s2X,key0X
		add		tY,ss9Y,key0Y
		add		key1X,key1X,tX
		stw		s1Y,8*4+schedY(SP)
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s1X,11*4+schedX(SP)
		add		s0X,s0X,s2X
		add		ss10Y,ss10Y,ss9Y
		add		s0X,s0X,key1X
		lwz		s1Y,11*4+schedY(SP)
		add		ss10Y,ss10Y,key1Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	ss10Y,ss10Y,3,0,31
		stw		s2X,9*4+schedX(SP)
		add		tX,s0X,key1X
		add		tY,ss10Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s2X,12*4+schedX(SP)
		add		s1X,s1X,s0X
		add		s1Y,s1Y,ss10Y
		add		s1X,s1X,key0X
		lwz		s2Y,12*4+schedY(SP)
		add		s1Y,s1Y,key0Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		stw		s0X,10*4+schedX(SP)
		add		tX,s1X,key0X
		add		tY,s1Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s0X,13*4+schedX(SP)
		add		s2X,s2X,s1X
		add		s2Y,s2Y,s1Y
		add		s2X,s2X,key1X
		add		s2Y,s2Y,key1Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		stw		s1X,11*4+schedX(SP)
		add		tX,s2X,key1X
		add		tY,s2Y,key1Y
		add		key0X,key0X,tX
		stw		s1Y,11*4+schedY(SP)
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s1X,14*4+schedX(SP)
		add		s0X,s0X,s2X
		add		ss13Y,ss13Y,s2Y
		add		s0X,s0X,key0X
		add		ss13Y,ss13Y,key0Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	ss13Y,ss13Y,3,0,31
		stw		s2X,12*4+schedX(SP)
		add		tX,s0X,key0X
		add		tY,ss13Y,key0Y
		add		key1X,key1X,tX
		stw		s2Y,12*4+schedY(SP)
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s2X,15*4+schedX(SP)
		add		s1X,s1X,s0X
		add		ss14Y,ss14Y,ss13Y
		add		s1X,s1X,key1X
		lwz		s2Y,15*4+schedY(SP)
		add		ss14Y,ss14Y,key1Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	ss14Y,ss14Y,3,0,31
		stw		s0X,13*4+schedX(SP)
		add		tX,s1X,key1X
		add		tY,ss14Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s0X,16*4+schedX(SP)
		add		s2X,s2X,s1X
		add		s2Y,s2Y,ss14Y
		add		s2X,s2X,key0X
		lwz		s0Y,16*4+schedY(SP)
		add		s2Y,s2Y,key0Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		stw		s1X,14*4+schedX(SP)
		add		tX,s2X,key0X
		add		tY,s2Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s1X,17*4+schedX(SP)
		add		s0X,s0X,s2X
		add		s0Y,s0Y,s2Y
		add		s0X,s0X,key1X
		add		s0Y,s0Y,key1Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		stw		s2X,15*4+schedX(SP)
		add		tX,s0X,key1X
		add		tY,s0Y,key1Y
		add		key0X,key0X,tX
		stw		s2Y,15*4+schedY(SP)
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s2X,18*4+schedX(SP)
		add		s1X,s1X,s0X
		add		ss17Y,ss17Y,s0Y
		add		s1X,s1X,key0X
		add		ss17Y,ss17Y,key0Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	ss17Y,ss17Y,3,0,31
		stw		s0X,16*4+schedX(SP)
		add		tX,s1X,key0X
		add		tY,ss17Y,key0Y
		add		key1X,key1X,tX
		stw		s0Y,16*4+schedY(SP)
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s0X,19*4+schedX(SP)
		add		s2X,s2X,s1X
		add		ss18Y,ss18Y,ss17Y
		add		s2X,s2X,key1X
		lwz		s0Y,19*4+schedY(SP)
		add		ss18Y,ss18Y,key1Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	ss18Y,ss18Y,3,0,31
		stw		s1X,17*4+schedX(SP)
		add		tX,s2X,key1X
		add		tY,ss18Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s1X,20*4+schedX(SP)
		add		s0X,s0X,s2X
		add		s0Y,s0Y,ss18Y
		add		s0X,s0X,key0X
		lwz		s1Y,20*4+schedY(SP)
		add		s0Y,s0Y,key0Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		stw		s2X,18*4+schedX(SP)
		add		tX,s0X,key0X
		add		tY,s0Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s2X,21*4+schedX(SP)
		add		s1X,s1X,s0X
		add		s1Y,s1Y,s0Y
		add		s1X,s1X,key1X
		lwz		s2Y,21*4+schedY(SP)
		add		s1Y,s1Y,key1Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		stw		s0X,19*4+schedX(SP)
		add		tX,s1X,key1X
		add		tY,s1Y,key1Y
		add		key0X,key0X,tX
		stw		s0Y,19*4+schedY(SP)
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s0X,22*4+schedX(SP)
		add		s2X,s2X,s1X
		add		s2Y,s2Y,s1Y
		add		s2X,s2X,key0X
		lwz		s0Y,22*4+schedY(SP)
		add		s2Y,s2Y,key0Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		stw		s1X,20*4+schedX(SP)
		add		tX,s2X,key0X
		add		tY,s2Y,key0Y
		add		key1X,key1X,tX
		stw		s1Y,20*4+schedY(SP)
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s1X,23*4+schedX(SP)
		add		s0X,s0X,s2X
		add		s0Y,s0Y,s2Y
		add		s0X,s0X,key1X
		lwz		s1Y,23*4+schedY(SP)
		add		s0Y,s0Y,key1Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		stw		s2X,21*4+schedX(SP)
		add		tX,s0X,key1X
		add		tY,s0Y,key1Y
		add		key0X,key0X,tX
		stw		s2Y,21*4+schedY(SP)
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s2X,24*4+schedX(SP)
		add		s1X,s1X,s0X
		add		s1Y,s1Y,s0Y
		add		s1X,s1X,key0X
		lwz		s2Y,24*4+schedY(SP)
		add		s1Y,s1Y,key0Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		stw		s0X,22*4+schedX(SP)
		add		tX,s1X,key0X
		add		tY,s1Y,key0Y
		add		key1X,key1X,tX
		stw		s0Y,22*4+schedY(SP)
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s0X,25*4+schedX(SP)
		add		s2X,s2X,s1X
		add		s2Y,s2Y,s1Y
		add		s2X,s2X,key1X
		lwz		s0Y,25*4+schedY(SP)
		add		s2Y,s2Y,key1Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		stw		s1X,23*4+schedX(SP)
		add		tX,s2X,key1X
		add		tY,s2Y,key1Y
		add		key0X,key0X,tX
		stw		s1Y,23*4+schedY(SP)
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s1X,0*4+schedX(SP)
		add		s0X,s0X,s2X
		add		s0Y,s0Y,s2Y
		add		s0X,s0X,key0X
		lwz		s1Y,0*4+schedY(SP)
		add		s0Y,s0Y,key0Y
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		stw		s2X,24*4+schedX(SP)
		add		tX,s0X,key0X
		add		tY,s0Y,key0Y
		add		key1X,key1X,tX
		stw		s2Y,24*4+schedY(SP)
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
pass3
		lwz		s2X,1*4+schedX(SP)
		add		s1X,s1X,s0X
		add		s1Y,s1Y,s0Y
		add		s1X,s1X,key1X
		add		s1Y,s1Y,key1Y
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		add		AX,plain0,s1X
		add		AY,plain0,s1Y
		stw		s0X,25*4+schedX(SP)
		add		tX,s1X,key1X
		add		tY,s1Y,key1Y
		add		key0X,key0X,tX
		stw		s0Y,25*4+schedY(SP)
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s0X,2*4+schedX(SP)
		add		s2X,s2X,s1X
		add		ss1Y,ss1Y,s1Y
		add		s2X,s2X,key0X
		add		ss1Y,ss1Y,key0Y
		rlwinm	s2X,s2X,3,0,31
		rlwinm	ss1Y,ss1Y,3,0,31
		add		BX,plain1,s2X
		add		BY,plain1,ss1Y
		add		tX,s2X,key0X
		add		tY,ss1Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s1X,3*4+schedX(SP)
		add		s0X,s0X,s2X
		add		ss2Y,ss2Y,ss1Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s0X,s0X,key1X
		add		ss2Y,ss2Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s0X,s0X,3,0,31
		rlwinm	ss2Y,ss2Y,3,0,31
		lwz		s1Y,3*4+schedY(SP)
		add		AX,AX,s0X
		add		AY,AY,ss2Y
		add		tX,s0X,key1X
		add		tY,ss2Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s2X,4*4+schedX(SP)
		add		s1X,s1X,s0X
		add		s1Y,s1Y,ss2Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s1X,s1X,key0X
		add		s1Y,s1Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		lwz		s2Y,4*4+schedY(SP)
		add		BX,BX,s1X
		add		BY,BY,s1Y
		add		tX,s1X,key0X
		add		tY,s1Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s0X,5*4+schedX(SP)
		add		s2X,s2X,s1X
		add		s2Y,s2Y,s1Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s2X,s2X,key1X
		add		s2Y,s2Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		add		AX,AX,s2X
		add		AY,AY,s2Y
		add		tX,s2X,key1X
		add		tY,s2Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s1X,6*4+schedX(SP)
		add		s0X,s0X,s2X
		add		ss5Y,ss5Y,s2Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s0X,s0X,key0X
		add		ss5Y,ss5Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s0X,s0X,3,0,31
		rlwinm	ss5Y,ss5Y,3,0,31
		add		BX,BX,s0X
		add		BY,BY,ss5Y
		add		tX,s0X,key0X
		add		tY,ss5Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s2X,7*4+schedX(SP)
		add		s1X,s1X,s0X
		add		ss6Y,ss6Y,ss5Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s1X,s1X,key1X
		add		ss6Y,ss6Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s1X,s1X,3,0,31
		rlwinm	ss6Y,ss6Y,3,0,31
		lwz		s2Y,7*4+schedY(SP)
		add		AX,AX,s1X
		add		AY,AY,ss6Y
		add		tX,s1X,key1X
		add		tY,ss6Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s0X,8*4+schedX(SP)
		add		s2X,s2X,s1X
		add		s2Y,s2Y,ss6Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s2X,s2X,key0X
		add		s2Y,s2Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		lwz		s0Y,8*4+schedY(SP)
		add		BX,BX,s2X
		add		BY,BY,s2Y
		add		tX,s2X,key0X
		add		tY,s2Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s1X,9*4+schedX(SP)
		add		s0X,s0X,s2X
		add		s0Y,s0Y,s2Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s0X,s0X,key1X
		add		s0Y,s0Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		add		AX,AX,s0X
		add		AY,AY,s0Y
		add		tX,s0X,key1X
		add		tY,s0Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s2X,10*4+schedX(SP)
		add		s1X,s1X,s0X
		add		ss9Y,ss9Y,s0Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s1X,s1X,key0X
		add		ss9Y,ss9Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s1X,s1X,3,0,31
		rlwinm	ss9Y,ss9Y,3,0,31
		add		BX,BX,s1X
		add		BY,BY,ss9Y
		add		tX,s1X,key0X
		add		tY,ss9Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s0X,11*4+schedX(SP)
		add		s2X,s2X,s1X
		add		ss10Y,ss10Y,ss9Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s2X,s2X,key1X
		add		ss10Y,ss10Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s2X,s2X,3,0,31
		rlwinm	ss10Y,ss10Y,3,0,31
		lwz		s0Y,11*4+schedY(SP)
		add		AX,AX,s2X
		add		AY,AY,ss10Y
		add		tX,s2X,key1X
		add		tY,ss10Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s1X,12*4+schedX(SP)
		add		s0X,s0X,s2X
		add		s0Y,s0Y,ss10Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s0X,s0X,key0X
		add		s0Y,s0Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		lwz		s1Y,12*4+schedY(SP)
		add		BX,BX,s0X
		add		BY,BY,s0Y
		add		tX,s0X,key0X
		add		tY,s0Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s2X,13*4+schedX(SP)
		add		s1X,s1X,s0X
		add		s1Y,s1Y,s0Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s1X,s1X,key1X
		add		s1Y,s1Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		add		AX,AX,s1X
		add		AY,AY,s1Y
		add		tX,s1X,key1X
		add		tY,s1Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s0X,14*4+schedX(SP)
		add		s2X,s2X,s1X
		add		ss13Y,ss13Y,s1Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s2X,s2X,key0X
		add		ss13Y,ss13Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s2X,s2X,3,0,31
		lwz		cypher0,cypher_lo(dataptr)
		rlwinm	ss13Y,ss13Y,3,0,31
		add		BX,BX,s2X
		add		BY,BY,ss13Y
		add		tX,s2X,key0X
		add		tY,ss13Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s1X,15*4+schedX(SP)
		add		s0X,s0X,s2X
		add		ss14Y,ss14Y,ss13Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s0X,s0X,key1X
		add		ss14Y,ss14Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s0X,s0X,3,0,31
		rlwinm	ss14Y,ss14Y,3,0,31
		lwz		s1Y,15*4+schedY(SP)
		add		AX,AX,s0X
		add		AY,AY,ss14Y
		add		tX,s0X,key1X
		add		tY,ss14Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s2X,16*4+schedX(SP)
		add		s1X,s1X,s0X
		add		s1Y,s1Y,ss14Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s1X,s1X,key0X
		add		s1Y,s1Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		lwz		s2Y,16*4+schedY(SP)
		add		BX,BX,s1X
		add		BY,BY,s1Y
		add		tX,s1X,key0X
		add		tY,s1Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s0X,17*4+schedX(SP)
		add		s2X,s2X,s1X
		add		s2Y,s2Y,s1Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s2X,s2X,key1X
		add		s2Y,s2Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s2X,s2X,3,0,31
		lwz		keylo,L0_lo(dataptr)
		rlwinm	s2Y,s2Y,3,0,31
		add		AX,AX,s2X
		add		AY,AY,s2Y
		add		tX,s2X,key1X
		add		tY,s2Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s1X,18*4+schedX(SP)
		add		s0X,s0X,s2X
		add		ss17Y,ss17Y,s2Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s0X,s0X,key0X
		add		ss17Y,ss17Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s0X,s0X,3,0,31
		lwz		keyhi,L0_hi(dataptr)
		rlwinm	ss17Y,ss17Y,3,0,31
		add		BX,BX,s0X
		add		BY,BY,ss17Y
		add		tX,s0X,key0X
		add		tY,ss17Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s2X,19*4+schedX(SP)
		add		s1X,s1X,s0X
		add		ss18Y,ss18Y,ss17Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s1X,s1X,key1X
		add		ss18Y,ss18Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s1X,s1X,3,0,31
		rlwinm	ss18Y,ss18Y,3,0,31
		lwz		s2Y,19*4+schedY(SP)
		add		AX,AX,s1X
		add		AY,AY,ss18Y
		add		tX,s1X,key1X
		add		tY,ss18Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s0X,20*4+schedX(SP)
		add		s2X,s2X,s1X
		add		s2Y,s2Y,ss18Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s2X,s2X,key0X
		add		s2Y,s2Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		lwz		s0Y,20*4+schedY(SP)
		add		BX,BX,s2X
		add		BY,BY,s2Y
		add		tX,s2X,key0X
		add		tY,s2Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s1X,21*4+schedX(SP)
		add		s0X,s0X,s2X
		add		s0Y,s0Y,s2Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s0X,s0X,key1X
		add		s0Y,s0Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		lwz		s1Y,21*4+schedY(SP)
		add		AX,AX,s0X
		add		AY,AY,s0Y
		add		tX,s0X,key1X
		add		tY,s0Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s2X,22*4+schedX(SP)
		add		s1X,s1X,s0X
		add		s1Y,s1Y,s0Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s1X,s1X,key0X
		add		s1Y,s1Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		lwz		s2Y,22*4+schedY(SP)
		add		BX,BX,s1X
		add		BY,BY,s1Y
		add		tX,s1X,key0X
		add		tY,s1Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		lwz		s0X,23*4+schedX(SP)
		add		s2X,s2X,s1X
		add		s2Y,s2Y,s1Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s2X,s2X,key1X
		add		s2Y,s2Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s2X,s2X,3,0,31
		rlwinm	s2Y,s2Y,3,0,31
		lwz		s0Y,23*4+schedY(SP)
		add		AX,AX,s2X
		add		AY,AY,s2Y
		add		tX,s2X,key1X
		add		tY,s2Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		
		lwz		s1X,24*4+schedX(SP)
		add		s0X,s0X,s2X
		add		s0Y,s0Y,s2Y
		xor		BX,BX,AX
		xor		BY,BY,AY
		add		s0X,s0X,key0X
		add		s0Y,s0Y,key0Y
		rlwnm	BX,BX,AX,0,31
		rlwnm	BY,BY,AY,0,31
		rlwinm	s0X,s0X,3,0,31
		rlwinm	s0Y,s0Y,3,0,31
		lwz		s1Y,24*4+schedY(SP)
		add		BX,BX,s0X
		add		BY,BY,s0Y
		add		tX,s0X,key0X
		add		tY,s0Y,key0Y
		add		key1X,key1X,tX
		add		key1Y,key1Y,tY
		rlwnm	key1X,key1X,tX,0,31
		rlwnm	key1Y,key1Y,tY,0,31
		
		add		s1X,s1X,s0X
		add		s1Y,s1Y,s0Y
		xor		AX,AX,BX
		xor		AY,AY,BY
		add		s1X,s1X,key1X
		add		s1Y,s1Y,key1Y
		rlwnm	AX,AX,BX,0,31
		rlwnm	AY,AY,BY,0,31
		rlwinm	s1X,s1X,3,0,31
		rlwinm	s1Y,s1Y,3,0,31
		add		AX,AX,s1X
		add		AY,AY,s1Y
		cmplw	crX,AX,cypher0
		add		tX,s1X,key1X
		cmplw	crY,AY,cypher0
		add		tY,s1Y,key1Y
		add		key0X,key0X,tX
		add		key0Y,key0Y,tY
		rlwnm	key0X,key0X,tX,0,31
		rlwnm	key0Y,key0Y,tY,0,31
		beq-	crX,checkBX
		beq-	crY,checkBY
		
notfound
		addis	keyhi,keyhi,0x200
		rlwinm.	tX,keyhi,0,0,7
		bne+	cycle
		
                addis	keyhi,keyhi,1
		rlwinm.	tX,keyhi,0,8,15
		rlwinm	keyhi,keyhi,0,8,31
		bne+	cycle
		
		addi	keyhi,keyhi,0x100
		rlwinm.	tX,keyhi,0,16,23
		rlwinm	keyhi,keyhi,0,16,31
		bne+	cycle
		
		addi	keyhi,keyhi,1
		rlwinm.	keyhi,keyhi,0,24,31
		bne+	cycle
		
		addis	keylo,keylo,0x100
		rlwinm.	tX,keylo,0,0,7
		bne+	storelow
		
		addis	keylo,keylo,1
		rlwinm.	tX,keylo,0,8,15
		rlwinm	keylo,keylo,0,8,31
		bne+	storelow
		
		addi	keylo,keylo,0x100
		rlwinm.	tX,keylo,0,16,23
		rlwinm	keylo,keylo,0,16,31
		bne+	storelow
		
		addi	keylo,keylo,1
		rlwinm	keylo,keylo,0,24,31
storelow
		stw		keylo,L0_lo(dataptr)
cycle
		stw		keyhi,L0_hi(dataptr)
		bdnz+	checkkey
		
epilogue
		lwz		iterations,-80(SP)
keyfound
		stw		keyhi,L0_hi(dataptr)
		stw		keylo,L0_lo(dataptr)
		
		mfctr	r5
		rlwinm	r5,r5,1,0,30
		subf	r3,r5,iterations
		
		lwz		r31,-4(SP)
		lwz		r30,-8(SP)
		lwz		r29,-12(SP)
		lwz		r28,-16(SP)
		lwz		r27,-20(SP)
		lwz		r26,-24(SP)
		lwz		r25,-28(SP)
		lwz		r24,-32(SP)
		lwz		r23,-36(SP)
		lwz		r22,-40(SP)
		lwz		r21,-44(SP)
		lwz		r20,-48(SP)
		lwz		r19,-52(SP)
		lwz		r18,-56(SP)
		lwz		r17,-60(SP)
		lwz		r16,-64(SP)
		lwz		r15,-68(SP)
		lwz		r14,-72(SP)
		lwz		r2,-76(SP)
		
		addi	SP,SP,2*(schedY-schedX)
		
		blr
		
		
checkBX
		lwz		s2X,25*4+schedX(SP)
		lwz		cypher1,cypher_hi(dataptr)
		add		s2X,s2X,s1X
		xor		BX,BX,AX
		add		s2X,s2X,key0X
		rlwnm	BX,BX,AX,0,31
		rlwinm	s2X,s2X,3,0,31
		add		BX,BX,s2X
		cmplw	crX,BX,cypher1
		beq-	crX,epilogue
		bne+	crY,notfound
checkBY
		lwz		s2Y,25*4+schedY(SP)
		lwz		cypher1,cypher_hi(dataptr)
		add		s2Y,s2Y,s1Y
		xor		BY,BY,AY
		add		s2Y,s2Y,key0Y
		rlwnm	BY,BY,AY,0,31
		rlwinm	s2Y,s2Y,3,0,31
		add		BY,BY,s2Y
		cmplw	crY,BY,cypher1
		bne+	crY,notfound
		lwz		iterations,-80(SP)
		addi	iterations,iterations,1
		b		keyfound
		
		ENDWITH
		END
