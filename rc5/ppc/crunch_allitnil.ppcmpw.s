;allitnil -- the 601 companion for lintilla

; stratagy: straight code in inner loop, 256 key cycle, load S0_n from pre-calculated constants
;			runtime calculation of round 1 constants for key.lo
;				r0	r1	r2	r3	r4	...					r24	r25	r26	r27	r28	r29	r30	r31
; registers:	tmp	SP	Sr2	Sr3	Sr4	...					24	25	Lr0	Lr1	key1 --	Sr0	Sr1
;							ptr	cnt	Lt0	Lt1	S0r	Qr	key0						Ax	Bx

	EXPORT	crunch_allitnil[DS]
	EXPORT	.crunch_allitnil[PR]
	
RC5UnitWork	RECORD	; Note: data is now in RC5/platform useful form
plain_hi	ds.l	1	;0		;plaintext, already mixed with IV
plain_lo	ds.l	1	;4
cypher_hi	ds.l	1	;8		;target cyphertext
cypher_lo	ds.l	1	;12
L0_hi		ds.l	1	;16		;key, changes with every unit * PIPELINE_COUNT
L0_lo		ds.l	1	;20
		ENDR

Local_Frame	RECORD 32,DECR ; Stack frame and Local variables (offset from SP)
itterations	ds.l	1	;28(SP)
work_ptr	ds.l	1	;24(SP)
			ds.l	1	;20(SP)	callers RTOC
			ds.l	1	;16(SP)	reserved
			ds.l	1	;12(SP)	reserved
			ds.l	1	;8(SP)	save LR
			ds.l	1	;4(SP)	save CR
			ds.l	1	;0(SP)	callers saved SP
save_regs	ds.l	19	;-76(SP) GPR13-31
			
save_RTOC	ds.l	1	;save local RTOC
count		ds.l	1	; remaining itterations
P_0		ds.l	1
P_1		ds.l	1
;C_0		ds.l	1
C_1		ds.l	1
;L0_0	ds.l	1	; key is in work record
;L0_1	ds.l	1
Sr_0	ds.l	1	; key0 constants computed out of loop
Sr_1	ds.l	1
con0	ds.l	1
con1	ds.l	1
con2	ds.l	1
S0_3	ds.l	1	; round 0 constants computed once only
S0_4	ds.l	1	;(could use static data section)
S0_5	ds.l	1
S0_6	ds.l	1
S0_7	ds.l	1
S0_8	ds.l	1
S0_9	ds.l	1
S0_10	ds.l	1
S0_11	ds.l	1
S0_12	ds.l	1
S0_13	ds.l	1
S0_14	ds.l	1
S0_15	ds.l	1
S0_16	ds.l	1
S0_17	ds.l	1
S0_18	ds.l	1
S0_19	ds.l	1
S0_20	ds.l	1
S0_21	ds.l	1
S0_22	ds.l	1
S0_23	ds.l	1
S0_24	ds.l	1
S0_25	ds.l	1
	ENDR

	TOC
		tc	crunch_allitnil[TC],crunch_allitnil[DS]
	
	CSECT	crunch_allitnil[DS]
		dc.l	.crunch_allitnil[PR]
		dc.l	TOC[tc0]
		dc.l	0
		
	CSECT	.crunch_allitnil[PR]

tmp		set r0
Sr2		set r2
Sr3		set r3
ptr			set r3
Sr4		set r4
cnt			set r4
Sr5		set r5
Lt0			set r5
Sr6		set r6
Lt0			set r6
Sr7		set r7
Lt1			set r7
Sr8		set r8
Sr9		set r9
S0r			set r9
Sr10	set r10
Qr			set r10
Sr11	set r11
key0		set r11
Sr12	set r12
tmp12		set r12
Sr13	set r13
Sr14	set r14
Sr15	set r15
Sr16	set r16
Sr17	set r17
Sr18	set r18
Sr19	set r19
Sr20	set r20
Sr21	set r21
Sr22	set r22
Sr23	set r23
Sr24	set r24
Sr25	set r25	; Note register overloading
Lr0		set r26
Lr1		set r27
key1	set r28
C_0		set r29
Sr0		set r30
Ax			set r30
Sr1		set r31
Bx			set r31

	WITH Local_Frame ; base register (SP)

	stmw     r13,-76(SP)

	stw		r2,save_RTOC(SP)
	stw      r3,work_ptr(SP); 24(SP)	; work
	stw      r4,itterations(SP);	28(SP)	; iterations
	stw      r4,count(SP);			-148(SP)	; save_iterations

	WITH RC5UnitWork ; base register ptr

; Save work parameters on the local stack
	lwz		tmp,plain_hi(ptr)
	stw		tmp,P_1(SP)
	lwz		tmp,plain_lo(ptr)
	stw		tmp,P_0(SP)
	lwz		tmp,cypher_hi(ptr)
	stw		tmp,C_1(SP)
	lwz		C_0,cypher_lo(ptr)
;	stw		tmp,C_0(SP)
	lwz		key1,L0_hi(ptr)
;	stw		key1,L0_1(SP)
	lwz		key0,L0_lo(ptr)
;	stw		key0,L0_0(SP)
	ENDWITH

; Precompute the round 0 constants
	lis      S0r,0xB7E1
	lis      Qr,0x9E37
	addi     S0r,S0r,0x5163	;S0r = P;
	addi     Qr,Qr,0x79B9	;Qr = Q;
	add     S0r,S0r,Qr
	add     S0r,S0r,Qr
	add     S0r,S0r,Qr
	stw		S0r,S0_3(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_4(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_5(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_6(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_7(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_8(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_9(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_10(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_11(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_12(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_13(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_14(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_15(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_16(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_17(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_18(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_19(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_20(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_21(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_22(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_23(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_24(SP)
	add     S0r,S0r,Qr
	stw		S0r,S0_25(SP)

start:	; Return here to recompute the key0 invariants
	lis      S0r,0xB7E1
	lis      Qr,0x9E37
	addi     S0r,S0r,0x5163	;S0r = P;
	addi     Qr,Qr,0x79B9	;Qr = Q;

	lwz		ptr,work_ptr(SP)		; probably already loaded
	WITH RC5UnitWork ; base register ptr
	lwz		key0,L0_lo(ptr)
	ENDWITH

; round 1.0 even
	rotlwi  Sr0,S0r,3
	stw		Sr0,Sr_0(SP);		; Sr_0 = ...
	add     Lr0,key0,Sr0
	rotlw	Lr0,Lr0,Sr0
	stw		Lr0,con0(SP)		; con0 = Lr_0
	add     S0r,S0r,Qr

; round 1.1 odd
	add     tmp,S0r,Sr0
	add     tmp,tmp,Lr0	
	rotlwi  Sr1,tmp,3
	stw		Sr1,Sr_1(SP)		; Sr_1 = rotl3(...)
	add     tmp12,Sr1,Lr0
	stw		tmp12,con1(SP)		; con1 =  Sr_1 + Lr_0
	add     Lr1,key1,tmp12
	rotlw	Lr1,Lr1,tmp12
	add     S0r,S0r,Qr

; round 1.2 even
	add     tmp,S0r,Sr1
	stw		tmp,con2(SP)		; con2 = S0_2 + Sr_1


reloop:
; registers: cnt, key1

; compute the loop count until the next event
	srwi	tmp,key1,24		; shift right logical 24, mask 0xFF
	subfic	tmp,tmp,256
	cmpw	tmp,cnt
	ble		@1
	mr		tmp,cnt
@1:
;// save remaining count, setup cnt
	sub		cnt,cnt,tmp
	stw		cnt,count(SP)
	mtctr	tmp

loop:
;// Uses saved constants 

	lwz     Sr2,con2(SP)	;	 = S0_2 + Sr_1
; round 1.1 odd
	add     Lr1,key1,tmp12
	rotlw	Lr1,Lr1,tmp12

	lwz		Sr3,S0_3(SP)
; round 1.2 even
	add     Sr2,Sr2,Lr1
	rotlwi  Sr2,Sr2,3
	add     tmp,Sr2,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr4,S0_4(SP)
; round 1.3 odd
	add     Sr3,Sr3,Sr2
	add     Sr3,Sr3,Lr0	
	rotlwi  Sr3,Sr3,3
	add     tmp,Sr3,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr5,S0_5(SP)
; round 1.4 even
	add     Sr4,Sr4,Sr3
	add     Sr4,Sr4,Lr1
	rotlwi  Sr4,Sr4,3
	add     tmp,Sr4,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr6,S0_6(SP)
; round 1.5 odd
	add     Sr5,Sr5,Sr4
	add     Sr5,Sr5,Lr0	
	rotlwi  Sr5,Sr5,3
	add     tmp,Sr5,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr7,S0_7(SP)
; round 1.6 even
	add     Sr6,Sr6,Sr5
	add     Sr6,Sr6,Lr1
	rotlwi  Sr6,Sr6,3
	add     tmp,Sr6,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr8,S0_8(SP)
; round 1.7 odd
	add     Sr7,Sr7,Sr6
	add     Sr7,Sr7,Lr0	
	rotlwi  Sr7,Sr7,3
	add     tmp,Sr7,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr9,S0_9(SP)
; round 1.8 even
	add     Sr8,Sr8,Sr7
	add     Sr8,Sr8,Lr1
	rotlwi  Sr8,Sr8,3
	add     tmp,Sr8,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr10,S0_10(SP)
; round 1.9 odd
	add     Sr9,Sr9,Sr8
	add     Sr9,Sr9,Lr0	
	rotlwi  Sr9,Sr9,3
	add     tmp,Sr9,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr11,S0_11(SP)
; round 1.10 even
	add     Sr10,Sr10,Sr9
	add     Sr10,Sr10,Lr1
	rotlwi  Sr10,Sr10,3
	add     tmp,Sr10,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr12,S0_12(SP)
; round 1.11 odd
	add     Sr11,Sr11,Sr10
	add     Sr11,Sr11,Lr0	
	rotlwi  Sr11,Sr11,3
	add     tmp,Sr11,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr13,S0_13(SP)
; round 1.12 even
	add     Sr12,Sr12,Sr11
	add     Sr12,Sr12,Lr1
	rotlwi  Sr12,Sr12,3
	add     tmp,Sr12,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr14,S0_14(SP)
; round 1.13 odd
	add     Sr13,Sr13,Sr12
	add     Sr13,Sr13,Lr0	
	rotlwi  Sr13,Sr13,3
	add     tmp,Sr13,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr15,S0_15(SP)
; round 1.14 even
	add     Sr14,Sr14,Sr13
	add     Sr14,Sr14,Lr1
	rotlwi  Sr14,Sr14,3
	add     tmp,Sr14,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr16,S0_16(SP)
; round 1.15 odd
	add     Sr15,Sr15,Sr14
	add     Sr15,Sr15,Lr0	
	rotlwi  Sr15,Sr15,3
	add     tmp,Sr15,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr17,S0_17(SP)
; round 1.16 even
	add     Sr16,Sr16,Sr15
	add     Sr16,Sr16,Lr1
	rotlwi  Sr16,Sr16,3
	add     tmp,Sr16,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr18,S0_18(SP)
; round 1.17 odd
	add     Sr17,Sr17,Sr16
	add     Sr17,Sr17,Lr0	
	rotlwi  Sr17,Sr17,3
	add     tmp,Sr17,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr19,S0_19(SP)
; round 1.18 even
	add     Sr18,Sr18,Sr17
	add     Sr18,Sr18,Lr1
	rotlwi  Sr18,Sr18,3
	add     tmp,Sr18,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr20,S0_20(SP)
; round 1.19 odd
	add     Sr19,Sr19,Sr18
	add     Sr19,Sr19,Lr0	
	rotlwi  Sr19,Sr19,3
	add     tmp,Sr19,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr21,S0_21(SP)
; round 1.20 even
	add     Sr20,Sr20,Sr19
	add     Sr20,Sr20,Lr1
	rotlwi  Sr20,Sr20,3
	add     tmp,Sr20,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr22,S0_22(SP)
; round 1.21 odd
	add     Sr21,Sr21,Sr20
	add     Sr21,Sr21,Lr0	
	rotlwi  Sr21,Sr21,3
	add     tmp,Sr21,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr23,S0_23(SP)
; round 1.22 even
	add     Sr22,Sr22,Sr21
	add     Sr22,Sr22,Lr1
	rotlwi  Sr22,Sr22,3
	add     tmp,Sr22,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr24,S0_24(SP)
; round 1.23 odd
	add     Sr23,Sr23,Sr22
	add     Sr23,Sr23,Lr0	
	rotlwi  Sr23,Sr23,3
	add     tmp,Sr23,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr25,S0_25(SP)
; round 1.24 even
	add     Sr24,Sr24,Sr23
	add     Sr24,Sr24,Lr1
	rotlwi  Sr24,Sr24,3
	add     tmp,Sr24,Lr1
	add     Lr0,Lr0,tmp
	rotlw	Lr0,Lr0,tmp

	lwz		Sr0,Sr_0(SP)
; round 1.25 odd
	add     Sr25,Sr25,Sr24
	add     Sr25,Sr25,Lr0	
	rotlwi  Sr25,Sr25,3
	add     tmp,Sr25,Lr0
	add     Lr1,Lr1,tmp
	rotlw	Lr1,Lr1,tmp

	lwz		Sr1,Sr_1(SP)
; round 2.0 even
	add     Sr0,Sr0,Sr25
	add     Sr0,Sr0,Lr1
	rotlwi  Sr0,Sr0,3
	add     tmp,Sr0,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.1 odd
	add     Sr1,Sr1,Sr0
	add     Sr1,Sr1,Lr0
	rotlwi  Sr1,Sr1,3
	add     tmp,Sr1,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.2 even
	add     Sr2,Sr2,Sr1
	add     Sr2,Sr2,Lr1
	rotlwi  Sr2,Sr2,3
	add     tmp,Sr2,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.3 odd
	add     Sr3,Sr3,Sr2
	add     Sr3,Sr3,Lr0
	rotlwi  Sr3,Sr3,3
	add     tmp,Sr3,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.4 even
	add     Sr4,Sr4,Sr3
	add     Sr4,Sr4,Lr1
	rotlwi  Sr4,Sr4,3
	add     tmp,Sr4,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.5 odd
	add     Sr5,Sr5,Sr4
	add     Sr5,Sr5,Lr0
	rotlwi  Sr5,Sr5,3
	add     tmp,Sr5,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.6 even
	add     Sr6,Sr6,Sr5
	add     Sr6,Sr6,Lr1
	rotlwi  Sr6,Sr6,3
	add     tmp,Sr6,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.7 odd
	add     Sr7,Sr7,Sr6
	add     Sr7,Sr7,Lr0
	rotlwi  Sr7,Sr7,3
	add     tmp,Sr7,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.8 even
	add     Sr8,Sr8,Sr7
	add     Sr8,Sr8,Lr1
	rotlwi  Sr8,Sr8,3
	add     tmp,Sr8,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.9 odd
	add     Sr9,Sr9,Sr8
	add     Sr9,Sr9,Lr0
	rotlwi  Sr9,Sr9,3
	add     tmp,Sr9,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.10 even
	add     Sr10,Sr10,Sr9
	add     Sr10,Sr10,Lr1
	rotlwi  Sr10,Sr10,3
	add     tmp,Sr10,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.11 odd
	add     Sr11,Sr11,Sr10
	add     Sr11,Sr11,Lr0
	rotlwi  Sr11,Sr11,3
	add     tmp,Sr11,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.12 even
	add     Sr12,Sr12,Sr11
	add     Sr12,Sr12,Lr1
	rotlwi  Sr12,Sr12,3
	add     tmp,Sr12,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.13 odd
	add     Sr13,Sr13,Sr12
	add     Sr13,Sr13,Lr0
	rotlwi  Sr13,Sr13,3
	add     tmp,Sr13,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.14 even
	add     Sr14,Sr14,Sr13
	add     Sr14,Sr14,Lr1
	rotlwi  Sr14,Sr14,3
	add     tmp,Sr14,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.15 odd
	add     Sr15,Sr15,Sr14
	add     Sr15,Sr15,Lr0
	rotlwi  Sr15,Sr15,3
	add     tmp,Sr15,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.16 even
	add     Sr16,Sr16,Sr15
	add     Sr16,Sr16,Lr1
	rotlwi  Sr16,Sr16,3
	add     tmp,Sr16,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.17 odd
	add     Sr17,Sr17,Sr16
	add     Sr17,Sr17,Lr0
	rotlwi  Sr17,Sr17,3
	add     tmp,Sr17,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.18 even
	add     Sr18,Sr18,Sr17
	add     Sr18,Sr18,Lr1
	rotlwi  Sr18,Sr18,3
	add     tmp,Sr18,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.19 odd
	add     Sr19,Sr19,Sr18
	add     Sr19,Sr19,Lr0
	rotlwi  Sr19,Sr19,3
	add     tmp,Sr19,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.20 even
	add     Sr20,Sr20,Sr19
	add     Sr20,Sr20,Lr1
	rotlwi  Sr20,Sr20,3
	add     tmp,Sr20,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.21 odd
	add     Sr21,Sr21,Sr20
	add     Sr21,Sr21,Lr0
	rotlwi  Sr21,Sr21,3
	add     tmp,Sr21,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.22 even
	add     Sr22,Sr22,Sr21
	add     Sr22,Sr22,Lr1
	rotlwi  Sr22,Sr22,3
	add     tmp,Sr22,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.23 odd
	add     Sr23,Sr23,Sr22
	add     Sr23,Sr23,Lr0
	rotlwi  Sr23,Sr23,3
	add     tmp,Sr23,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; round 2.24 even
	add     Sr24,Sr24,Sr23
	add     Sr24,Sr24,Lr1
	rotlwi  Sr24,Sr24,3
	add     tmp,Sr24,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

; round 2.25 odd
	add     Sr25,Sr25,Sr24
	add     Sr25,Sr25,Lr0
	rotlwi  Sr25,Sr25,3
	add     tmp,Sr25,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

; registers in use: Sr0 ... Sr25, Lr0, Lr1, key1
; we might save a load if we throw away Sr25!

; round 3.0 even
	add     Sr0,Sr0,Sr25
	add     Sr0,Sr0,Lr1
	rotlwi  Sr0,Sr0,3
	add     tmp,Sr0,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp

	lwz		tmp,P_0(SP)
; round 3.1 odd
	add     Sr1,Sr1,Sr0
	add     Sr1,Sr1,Lr0
	rotlwi  Sr1,Sr1,3
	add		Ax,tmp,Sr0
	add     tmp,Sr1,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp

	lwz		tmp,P_1(SP)
; round 3.2 even
	add     Sr2,Sr2,Sr1
	add     Sr2,Sr2,Lr1
	rotlwi  Sr2,Sr2,3
	add		Bx,tmp,Sr1
	add     tmp,Sr2,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr2

; round 3.3 odd
	add     Sr3,Sr3,Sr2
	add     Sr3,Sr3,Lr0
	rotlwi  Sr3,Sr3,3
	add     tmp,Sr3,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr3

; round 3.4 even
	add     Sr4,Sr4,Sr3
	add     Sr4,Sr4,Lr1
	rotlwi  Sr4,Sr4,3
	add     tmp,Sr4,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr4

; round 3.5 odd
	add     Sr5,Sr5,Sr4
	add     Sr5,Sr5,Lr0
	rotlwi  Sr5,Sr5,3
	add     tmp,Sr5,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr5

; round 3.6 even
	add     Sr6,Sr6,Sr5
	add     Sr6,Sr6,Lr1
	rotlwi  Sr6,Sr6,3
	add     tmp,Sr6,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr6

; round 3.7 odd
	add     Sr7,Sr7,Sr6
	add     Sr7,Sr7,Lr0
	rotlwi  Sr7,Sr7,3
	add     tmp,Sr7,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr7

; round 3.8 even
	add     Sr8,Sr8,Sr7
	add     Sr8,Sr8,Lr1
	rotlwi  Sr8,Sr8,3
	add     tmp,Sr8,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr8

; round 3.9 odd
	add     Sr9,Sr9,Sr8
	add     Sr9,Sr9,Lr0
	rotlwi  Sr9,Sr9,3
	add     tmp,Sr9,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr9

; round 3.10 even
	add     Sr10,Sr10,Sr9
	add     Sr10,Sr10,Lr1
	rotlwi  Sr10,Sr10,3
	add     tmp,Sr10,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr10

; round 3.11 odd
	add     Sr11,Sr11,Sr10
	add     Sr11,Sr11,Lr0
	rotlwi  Sr11,Sr11,3
	add     tmp,Sr11,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr11

; round 3.12 even
	add     Sr12,Sr12,Sr11
	add     Sr12,Sr12,Lr1
	rotlwi  Sr12,Sr12,3
	add     tmp,Sr12,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr12

; round 3.13 odd
	add     Sr13,Sr13,Sr12
	add     Sr13,Sr13,Lr0
	rotlwi  Sr13,Sr13,3
	add     tmp,Sr13,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr13

; round 3.14 even
	add     Sr14,Sr14,Sr13
	add     Sr14,Sr14,Lr1
	rotlwi  Sr14,Sr14,3
	add     tmp,Sr14,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr14

; round 3.15 odd
	add     Sr15,Sr15,Sr14
	add     Sr15,Sr15,Lr0
	rotlwi  Sr15,Sr15,3
	add     tmp,Sr15,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr15

; round 3.16 even
	add     Sr16,Sr16,Sr15
	add     Sr16,Sr16,Lr1
	rotlwi  Sr16,Sr16,3
	add     tmp,Sr16,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr16

; round 3.17 odd
	add     Sr17,Sr17,Sr16
	add     Sr17,Sr17,Lr0
	rotlwi  Sr17,Sr17,3
	add     tmp,Sr17,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr17

; round 3.18 even
	add     Sr18,Sr18,Sr17
	add     Sr18,Sr18,Lr1
	rotlwi  Sr18,Sr18,3
	add     tmp,Sr18,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr18

; round 3.19 odd
	add     Sr19,Sr19,Sr18
	add     Sr19,Sr19,Lr0
	rotlwi  Sr19,Sr19,3
	add     tmp,Sr19,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr19

; round 3.20 even
	add     Sr20,Sr20,Sr19
	add     Sr20,Sr20,Lr1
	rotlwi  Sr20,Sr20,3
	add     tmp,Sr20,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lr0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr20

; round 3.21 odd
	add     Sr21,Sr21,Sr20
	add     Sr21,Sr21,Lr0
	rotlwi  Sr21,Sr21,3
	add     tmp,Sr21,Lr0		
	add     Lr1,tmp,Lr1
	rotlw	Lr1,Lr1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr21

; round 3.22 even
	add     Sr22,Sr22,Sr21
	add     Sr22,Sr22,Lr1
	rotlwi  Sr22,Sr22,3
	add     tmp,Sr22,Lr1
	add     Lr0,tmp,Lr0
	rotlw	Lt0,Lr0,tmp
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr22

; round 3.23 odd
	add     Sr23,Sr23,Sr22
	add     Sr23,Sr23,Lt0
	rotlwi  Sr23,Sr23,3
	add     tmp,Sr23,Lt0		
	add     Lt1,tmp,Lr1
	rotlw	Lt1,Lt1,tmp
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr23

;	lwz		tmp,C_0(SP)
; round 3.24 even
	add     Sr24,Sr24,Sr23
	add     Sr24,Sr24,Lt1
	rotlwi  Sr24,Sr24,3
	xor		Ax,Ax,Bx
	rotlw	Ax,Ax,Bx
	add		Ax,Ax,Sr24

; Preserve these registers to finish 3.24 and 3.25
; Ax, Bx, Sr24, Sr25, Lt0, Lt1, Key1

	cmpw	C_0,Ax

	addis	key1,key1,256	; Increment high byte of key

; --fill from round 1
; round 1.0 even
	lwz     tmp12,con1(SP)	; = Sr1 + Lr0
	lwz		Lr0,con0(SP)

	bdnzf	2,loop

	lwz		cnt,count(SP)
	mfctr	tmp
	add		cnt,cnt,tmp

	bne @2
; round 3.24 (continued)
	add     tmp,Sr24,Lt1
	add     Lt0,tmp,Lt0
	rotlw	Lt0,Lt0,tmp
	lwz		tmp,C_1(SP)
; round 3.25 odd
	add     Sr25,Sr25,Sr24
	add     Sr25,Sr25,Lt0
	rotlwi  Sr25,Sr25,3
	xor		Bx,Bx,Ax
	rotlw	Bx,Bx,Ax
	add		Bx,Bx,Sr25
	cmpw	tmp,Bx
	bne		@2

;!!!! Found it !!!!
; undo the last key increment and return the result
	addi	cnt,cnt,1
	subis	key1,key1,256
	b		exit

@2:

; registers: cnt, key1, temps for round 1
; check for rollovers, update key in work record
	srwi.   tmp,key1,24	;// logical shift right ??
	bne @4					;// high bits of key1 != 0

	lwz		ptr,work_ptr(SP)
	WITH RC5UnitWork ; base register ptr
	li		Lt1,L0_hi
	lwbrx	key1,ptr,Lt1	; L0_hi reversed
	ori		key1,key1,0xff
	addic.	key1,key1,1
	stwbrx	key1,ptr,Lt1
	bnz		@3
	li		Lt0,L0_lo
	lwbrx	key0,ptr,Lt0	; L0_lo reversed
	addi	key0,key0,1
	stwbrx	key0,ptr,Lt0
@3:	
	lwz		key1,L0_hi(ptr)
	ENDWITH


@4:
; check if done
	cmpwi	cnt,0
	beq		exit

;// be sure the registers are setup for relooping!!
	cmpwi	key1,0
	beq		start	; recalc key0 constants
	b		reloop	; just restart the loop
	
exit:
; save the last key tested in the work record
	lwz		ptr,work_ptr(SP)
	WITH RC5UnitWork ; base register ptr
	stw		key1,L0_hi(ptr)
	ENDWITH

; return the count of keys tested
	lwz		r3,itterations(SP)
	sub		r3,r3,cnt
	
	lwz		r2,save_RTOC(SP)
	lmw      r13,-76(SP)
	blr

	ENDWITH
	END

