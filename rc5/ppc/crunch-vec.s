;rc5loop.s -- Optimized for the Velocity Engine of the G4 PowerPC

	EXPORT	crunch_vec[DS]
	EXPORT	.crunch_vec[PR]

; Assume key is mod 4 and won't wrap 2^32 boundary
; Compute and save round 0 constants
; Compute constants for 1.0 and 1.1
; Compute through round 1.15 for vectors S1_n.v
; Compute through round 1.2 for next iteration integer

RC5UnitWork	RECORD	; Note: data is now in RC5/platform useful form
plain_hi	ds.l	1	;0		;plaintext, already mixed with IV
plain_lo	ds.l	1	;4
cypher_hi	ds.l	1	;8		;target cyphertext
cypher_lo	ds.l	1	;12
L0_hi		ds.l	1	;16		;key, changes with every unit * PIPELINE_COUNT
L0_lo		ds.l	1	;20
		ENDR

vec_data RECORD
a	ds.l	1
b	ds.l	1
c	ds.l	1
d	ds.l	1
	ENDR

New_Frame	RECORD; Stack frame for calls and Local variables (offset from SP)
;(red zone to -224(SP))
new_sp		ds.l	1	;0(SP)	save SP (16 byte aligned SP points here after update)
			ds.l	1	;4(SP)	save CR?
			ds.l	1	;8(SP)	save LR?
			ds.l	1	;12(SP)	reserved
			ds.l	1	;16(SP)	reserved
			ds.l	1	;20(SP)	save RTOC
;param1		ds.l	1	;24(SP) pass parameters here
;param2		ds.l	1	;28(SP) ...
	ORG 0x20
L0_1		ds.l	1	; hi key index by 0,sp20x
C_1			ds.l	1	; used only after C_0 is matched
;			ds.l	1	; dummy for alignment
;			ds.l	1	; dummy for alignment
	ORG 0x30
L0_0		ds.l	1	; lo key index by 0,sp30x
;			ds.l	1	; dummy for alignment
;			ds.l	1	; dummy for alignment
;			ds.l	1	; dummy for alignment

; local variables
; NOTE: the folowing vector data is loaded using pairs of index registers
; Do NOT alter the order!!
	ALIGN 4
tVEC 		ds vec_data		; tmpx,sp Temp vector word
tS0_2		EQU tVEC		; initial constatants for setup
tQ			EQU tVEC+4
tLr_0		EQU tVEC+8
tS1_1		EQU tVEC+12
L_0	 		ds vec_data		; tmpx,sp10x
L_1	 		ds vec_data		; tmpx,sp20x
G_data		ds vec_data		; tmpx,sp30x - vector guard data
S0_15		ds.l	1		; rcon,sp
S0_16		ds.l	1
S0_17		ds.l	1
S0_18		ds.l	1
S0_19		ds.l	1		; rcon,sp10x
S0_20		ds.l	1
S0_21		ds.l	1
S0_22		ds.l	1
S0_23		ds.l	1		; rcon,sp20x
S0_24		ds.l	1
S0_25		ds.l	1
S1_0		ds.l	1
S1_1		ds.l	1		; rcon,sp30x
P_0 		ds.l	1
P_1 		ds.l	1
C_0 		ds.l	1
S1_2		ds vec_data		; rvec1(sp)
S1_3		ds vec_data		; rvec1,sp10x
S1_4		ds vec_data
S1_5		ds vec_data

S1_6		ds vec_data		; rvec2,sp
S1_7		ds vec_data
S1_8		ds vec_data
S1_9		ds vec_data
S1_10		ds vec_data		; rvec3(sp)
S1_11		ds vec_data
S1_12		ds vec_data
S1_13		ds vec_data
; saved vector registers
save_vrs	ds.l	12*4	; 48 words, 192 bytes
frame_end	EQU 	*		; offset to end of frame [must be aligned for vector]
	ENDR

Old_Frame	RECORD	{old_sp}	;
save_vrsave	ds.l	1	;-80(xsp) saved vrsave
save_gprs	ds.l	19	;-76(xsp) saved GPR13-31
						; saved fprs (not used)
old_sp		ds.l	1	;0(xsp)	old SP pointed here
save_cr		ds.l	1	;4(xsp)
save_lr		ds.l	1	;8(xsp)
			ds.l	1	;12(xsp)	reserved
			ds.l	1	;16(xsp)	reserved
			ds.l	1	;20(xsp)	callers RTOC
; calling parameter storage
work_ptr	ds.l	1	;24(xsp)
iterations	ds.l	1	;28(xsp)
	ENDR
stack_offset	EQU		New_Frame.frame_end - Old_Frame.save_vrsave

	TOC
		tc	crunch_vec[TC],crunch_vec[DS]

	CSECT	crunch_vec[DS]
		dc.l	.crunch_vec[PR]
		dc.l	TOC[tc0]
		dc.l	0

	CSECT	.crunch_vec[PR]

tmp 	EQU r0	; (can't use as first index)
;SP 	EQU r1
;RTOC	EQU r2
workptr	EQU	r3
iter	EQU r4
tmpx	EQU	r5	; index for temps
sp10x	EQU r6	; SP+0x10 - offsets for loading vector data
sp20x	EQU r7	; SP+0x20
sp30x	EQU r8	; SP+0x30
rcon	EQU r9	; vector data group pointers
rvec1	EQU r10
rvec2	EQU r11
rvec3	EQU r12

con0	EQU r13	; constants for integer pre-processing
key0	EQU r13	; (same as con0)
con1	EQU r14
con2	EQU r15
con3	EQU r16
Qr  	EQU r17
key1	EQU r18

S0r 	EQU r19
L0a 	EQU r20	; ordered for stwm
L0b 	EQU r21
L0c 	EQU r22
L0d 	EQU r23
L1a 	EQU r24
L1b 	EQU r25
L1c 	EQU r26
L1d 	EQU r27
Sr_a 	EQU r28
Sr_b 	EQU r29
Sr_c 	EQU r30
Sr_d 	EQU r31

Sr0 	EQU r28
Sr1 	EQU r29

vS0 	EQU v0
vS1 	EQU v1
vS2 	EQU v2
vS3 	EQU v3
vS4 	EQU v4
vS5 	EQU v5
vS6 	EQU v6
vS7 	EQU v7
vS8 	EQU v8
vS9 	EQU v9
vS10	EQU v10
vS11	EQU v11
vS12	EQU v12
vS13	EQU v13
vS14	EQU v14
vS15	EQU v15
vS16	EQU v16
vS17	EQU v17
vS18	EQU v18
vS19	EQU v19
vS20	EQU v20
vS21	EQU v21
vS22	EQU v22
vS23	EQU v23
vS24	EQU v24
vS25	EQU v25
vGuard	EQU v26		; Vector guard register
vtmp	EQU v27
val3  	EQU v28		; constant #3
vcon	EQU v29		; loaded constants
vL0 	EQU v30
vL1 	EQU v31
vAx 	EQU v0			; overwrites vS0
vBx 	EQU v1			; overwrites vS1
vC	EQU v19		; overwrites vS19
vL0t	EQU v20		; overwrites vS20
vL1t	EQU v21		; overwrites vS21
vhit	EQU v22		; overwrites vC

prolog:
	WITH Old_Frame	; base register sp (moved to r6)
	mflr	r0
	stw		r0,save_lr(sp)
	stmw	r13,save_gprs(sp)	# sage gp registers
								# save CR here
	stw		r3,work_ptr(sp)		# save parameters
	stw		r4,iterations(sp)
	mfspr	r0,vrsave
	stw		r0,save_vrsave(sp)
	li		r0,-1				# uses all vector registers
	mtspr	vrsave,r0
	ENDWITH #Old_Frame
	mr		r6,sp				# old frame is now accessed with r6
	rlwinm	r12,sp,0,28,31		# old stack alignment padding
	subfic	r12,r12,-stack_offset
	stwux	sp,sp,r12			# establish new stack frame
	WITH New_Frame ; base register sp
	li  	r0,save_vrs			# save vector registers in the alligned frame
	stvx	v20,sp,r0
	li  	r0,save_vrs+0x10
	stvx	v21,sp,r0
	li  	r0,save_vrs+0x20
	stvx	v22,sp,r0
	li  	r0,save_vrs+0x30
	stvx	v23,sp,r0
	li  	r0,save_vrs+0x40
	stvx	v24,sp,r0
	li  	r0,save_vrs+0x50
	stvx	v25,sp,r0
	li  	r0,save_vrs+0x60
	stvx	v26,sp,r0
	li  	r0,save_vrs+0x70
	stvx	v27,sp,r0
	li  	r0,save_vrs+0x80
	stvx	v28,sp,r0
	li  	r0,save_vrs+0x90
	stvx	v29,sp,r0
	li  	r0,save_vrs+0xa0
	stvx	v30,sp,r0
	li  	r0,save_vrs+0xb0
	stvx	v31,sp,r0

; Setup the vector constant and guard register
	vspltisw	val3,3

; Save work parameters on the local stack
	WITH RC5UnitWork ; base register workptr
	lwz		tmp,plain_hi(workptr)
	stw		tmp,P_1(SP)
	lwz		tmp,plain_lo(workptr)
	stw		tmp,P_0(SP)
	lwz		tmp,cypher_hi(workptr)
	stw		tmp,C_1(SP)
	lwz		tmp,cypher_lo(workptr)
	stw		tmp,C_0(SP)
	li		tmpx,L0_hi
	lwbrx	key1,tmpx,workptr	# keep key1 byte reversed in register
	li  	tmpx,L0_lo
	lwbrx	key0,tmpx,workptr
	ENDWITH

; setup the index registers for addressing vector data
	addi	sp10x,sp,0x10
	addi	sp20x,sp,0x20
	addi	sp30x,sp,0x30
	li		rcon,S0_15
	li		rvec1,S1_2
	li		rvec2,S1_6
	li		rvec3,S1_10
	li		tmpx,tVEC

;// Setup the vector guard register. This register contains the starting key
	stw		key0,G_data.a(sp)
	stw 	key1,G_data.b(sp)
	lvx 	vGuard,tmpx,sp30x

; remove the low 2 bits from key1 and add to iter
	clrlwi	tmp,key1,30
	add		iter,iter,tmp
	clrrwi	key1,key1,2

start:	; resume processing keys after key0 update
;		 iter, key0, key1 in registers (keys byte reversed)

	stw key0,L0_0(sp)
	stw key1,L0_1(sp)

;**** setup the count // check this code!!
; limit count so key1 will not pass roll over
; NOTE: iter=0 is treated as 2^32
	addi	cnt,iter,-1		; make 0 a big value
	ori 	cnt,cnt,3		; force to mod 4
	not		tmp,key1		; count to rollover
	cmplw	tmp,cnt
	bge 	@1				; keep the smaller count
	mr  	cnt,tmp
@1	srwi	cnt,cnt,2
	addi	cnt,cnt,1
	mtctr	cnt				; set loop counter
	slwi	cnt,cnt,2
	sub 	iter,iter,cnt	; unused count or -overcount

; Setup the round 0 constants
	lis      S0r,0xB7E1
	lis      Qr,0x9E37
	addi     S0r,S0r,0x5163	;S0r = P;
	addi     Qr,Qr,0x79B9	;Qr = Q;

	lwbrx	con0,0,sp30x	; L0_0
; round 1.0 even con0
	rotlwi  Sr0,S0r,3
	add     con0,con0,Sr0
	rotlw	con0,con0,Sr0
	add     S0r,S0r,Qr

; round 1.1 odd
	add     tmp,S0r,Sr0
	add     tmp,tmp,con0
	rotlwi  Sr1,tmp,3
	add		con1,con0,Sr1
	add     S0r,S0r,Qr
	add 	con2,S0r,Sr1

	stw		S0r,tS0_2(SP)		; S0_2 = P+2Q
	stw		Qr,tQ(SP)		; Q
	stw		con0,tLr_0(SP)		; Lr_0
	stw		Sr1,tS1_1(SP)		; S1[1] = rotl3(...)

	stw		Sr0,S1_0(SP)		; constants for round 2
	stw		Sr1,S1_1(SP)

; setup 4 keys for the first vector loop
	lwbrx	L1a,0,sp20x		; L0_1
	stw		L1a,L_1.a(sp)
	addis	L1b,L1a,256
	stw		L1b,L_1.b(sp)
	addis	L1c,L1b,256
	stw		L1c,L_1.c(sp)
	addis	L1d,L1c,256
	stw		L1d,L_1.d(sp)

; load setup values for P, Q, key.lo
vS0r	SET vS0
vQr 	SET vS23
	lvx			vcon,tmpx,sp		; initial loop
constants
	vspltw		vS0,vcon,0		; P+2Q
	vspltw		vQr,vcon,1		; Q
	vspltw		vL0,vcon,2		; L[0] (1.0)
	vspltw		vS1,vcon,3 		; S[1] (1.1)
	lvx			vL1,tmpx,sp20x			; key1

; compute vector work through r15
	vadduwm		vtmp,vS1,vL0	; finish r1.1

	vadduwm		vL1,vL1,vtmp

	vrlw		vL1,vL1,vtmp

	vadduwm		vS2,vS0r,vS1	; r1.2

	vadduwm		vS2,vS2,vL1

	vrlw		vS2,vS2,val3
;		stvx		vS2,rvec1,sp	; S1_2
	vadduwm		vtmp,vS2,vL1

	vadduwm		vL0,vL0,vtmp

	vrlw		vL0,vL0,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS3,vS0r,vS2	; r1.3

	vadduwm		vS3,vS3,vL0

	vrlw		vS3,vS3,val3
;		stvx	vS3,rvec1,sp10x ; S1_3
	vadduwm		vtmp,vS3,vL0

	vadduwm		vL1,vL1,vtmp

	vrlw		vL1,vL1,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS4,vS0r,vS3	; r1.4

	vadduwm		vS4,vS4,vL1

	vrlw		vS4,vS4,val3
;		stvx	vS4,rvec1,sp20x ; S1_4
	vadduwm		vtmp,vS4,vL1

	vadduwm		vL0,vL0,vtmp

	vrlw		vL0,vL0,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS5,vS0r,vS4	; r1.5

	vadduwm		vS5,vS5,vL0

	vrlw		vS5,vS5,val3
;		stvx	vS5,rvec1,sp30x ; S1_5
	vadduwm		vtmp,vS5,vL0

	vadduwm		vL1,vL1,vtmp

	vrlw		vL1,vL1,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS6,vS0r,vS5	; r1.6

	vadduwm		vS6,vS6,vL1

	vrlw		vS6,vS6,val3
		stvx	vS6,rvec2,sp ; S1_6
	vadduwm		vtmp,vS6,vL1

	vadduwm		vL0,vL0,vtmp

	vrlw		vL0,vL0,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS7,vS0r,vS6	; r1.7

	vadduwm		vS7,vS7,vL0

	vrlw		vS7,vS7,val3
		stvx	vS7,rvec2,sp10x ; S1_7
	vadduwm		vtmp,vS7,vL0

	vadduwm		vL1,vL1,vtmp

	vrlw		vL1,vL1,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS8,vS0r,vS7	; r1.8

	vadduwm		vS8,vS8,vL1

	vrlw		vS8,vS8,val3
		stvx	vS8,rvec2,sp20x ; S1_8
	vadduwm		vtmp,vS8,vL1

	vadduwm		vL0,vL0,vtmp

	vrlw		vL0,vL0,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS9,vS0r,vS8	; r1.9

	vadduwm		vS9,vS9,vL0

	vrlw		vS9,vS9,val3
		stvx	vS9,rvec2,sp30x ; S1_9
	vadduwm		vtmp,vS9,vL0

	vadduwm		vL1,vL1,vtmp

	vrlw		vL1,vL1,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS10,vS0r,vS9	; r1.10
	vadduwm		vS10,vS10,vL1

	vrlw		vS10,vS10,val3
		stvx	vS10,rvec3,sp ; S1_10

	vadduwm		vtmp,vS10,vL1

	vadduwm		vL0,vL0,vtmp

	vrlw		vL0,vL0,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS11,vS0r,vS10	; r1.11

	vadduwm		vS11,vS11,vL0

	vrlw		vS11,vS11,val3
		stvx	vS11,rvec3,sp10x ; S1_11
	vadduwm		vtmp,vS11,vL0

	vadduwm		vL1,vL1,vtmp

	vrlw		vL1,vL1,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS12,vS0r,vS11	; r1.12

	vadduwm		vS12,vS12,vL1

	vrlw		vS12,vS12,val3
		stvx	vS12,rvec3,sp20x ; S1_12
	vadduwm		vtmp,vS12,vL1

	vadduwm		vL0,vL0,vtmp

	vrlw		vL0,vL0,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS13,vS0r,vS12	; r1.13

	vadduwm		vS13,vS13,vL0

	vrlw		vS13,vS13,val3
		stvx	vS13,rvec3,sp30x ; S1_13
	vadduwm		vtmp,vS13,vL0

	vadduwm		vL1,vL1,vtmp

	vrlw		vL1,vL1,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS14,vS0r,vS13	; r1.14

	vadduwm		vS14,vS14,vL1

	vrlw		vS14,vS14,val3
	vadduwm		vtmp,vS14,vL1

	vadduwm		vL0,vL0,vtmp

	vrlw		vL0,vL0,vtmp

		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS15,vS0r,vS14	; r1.15

	vadduwm		vS15,vS15,vL0

	vrlw		vS15,vS15,val3
	vadduwm		vtmp,vS15,vL0

	vadduwm		vL1,vL1,vtmp

	vrlw		vL1,vL1,vtmp

		vadduwm	vS16,vS0r,vQr

; increment the key for the next group
;//	lwz 	key1,L0_1(sp)
	addi	key1,key1,4
	stw 	key1,L0_1(sp)

;Load the keys for the integer rounds
	lwbrx 	L1a,0,sp20x	; L0_1

	add     con3,S0r,Qr		; S0_3 == con3

; need -- key1, con0, con1, con2,
	add 	L1a,L1a,con1	; round 1.1 odd
	addis	L1b,L1a,256
	addis	L1c,L1b,256
	addis	L1d,L1c,256
	rotlw	L1a,L1a,con1
	rotlw	L1b,L1b,con1
	rotlw	L1c,L1c,con1
	rotlw	L1d,L1d,con1
	add 	Sr_a,con2,L1a	; round 1.2 even
	rotlwi	Sr_a,Sr_a,3
	stw 	Sr_a,S1_2.a(sp)
	add 	Sr_b,con2,L1b
	rotlwi	Sr_b,Sr_b,3
	stw 	Sr_b,S1_2.b(sp)
	add 	Sr_c,con2,L1c
	rotlwi	Sr_c,Sr_c,3
	stw 	Sr_c,S1_2.c(sp)
	add 	Sr_d,con2,L1d
	rotlwi	Sr_d,Sr_d,3
	stw 	Sr_d,S1_2.d(sp)
	add 	tmp,Sr_a,L1a
	add 	L0a,con0,tmp
	rotlw	L0a,L0a,tmp
	add 	tmp,Sr_b,L1b
	add 	L0b,con0,tmp
	rotlw	L0b,L0b,tmp
	add 	tmp,Sr_c,L1c
	add 	L0c,con0,tmp
	rotlw	L0c,L0c,tmp
	add 	tmp,Sr_d,L1d
	add 	L0d,con0,tmp
	rotlw	L0d,L0d,tmp
; current: Lr0*, Sr*, Lr1*, S1_2.*

; Generate the S0 constants for vectors rounds
; needs Qr=Q, con3=P+3Q
	add     S0r,con3,Qr		; S0_4
	add     S0r,S0r,Qr		; S0_5
	add     S0r,S0r,Qr		; S0_6
	add     S0r,S0r,Qr		; S0_7
	add     S0r,S0r,Qr		; S0_8
	add     S0r,S0r,Qr		; S0_9
	add     S0r,S0r,Qr		; S0_10
	add     S0r,S0r,Qr		; S0_11
	add     S0r,S0r,Qr		; S0_12
	add     S0r,S0r,Qr		; S0_13
	add     S0r,S0r,Qr		; S0_14
	add     S0r,S0r,Qr		; S0_15
	stw		S0r,S0_15(SP)
	add     S0r,S0r,Qr		; S0_16
	stw		S0r,S0_16(SP)
	add     S0r,S0r,Qr		; S0_17
	stw		S0r,S0_17(SP)
	add     S0r,S0r,Qr		; S0_18
	stw		S0r,S0_18(SP)
	add     S0r,S0r,Qr		; S0_19
	stw		S0r,S0_19(SP)
	add     S0r,S0r,Qr		; S0_20
	stw		S0r,S0_20(SP)
	add     S0r,S0r,Qr		; S0_21
	stw		S0r,S0_21(SP)
	add     S0r,S0r,Qr		; S0_22
	stw		S0r,S0_22(SP)
	add     S0r,S0r,Qr		; S0_23
	stw		S0r,S0_23(SP)
	add     S0r,S0r,Qr		; S0_24
	stw		S0r,S0_24(SP)
	add     S0r,S0r,Qr		; S0_25
	stw		S0r,S0_25(SP)
; current: S0_15 through S0_25
		lvx		vcon,rcon,sp ; S0_15


; The next iteration has been computed through r1.15

; results in vS2, vS14, vS15, vL0, vL1
; S1_3 through S1_13 have been saved to memory
; constants loaded in vS16, vcon, val3

; The key is incremented to the begining of the succeding iteration


; Lr0, Lr1, Sr [a-d] are computed through r1.2
; S1_2 is saved to memory

; Integer registers in use:
; Qr = Q, con0, con1, con2, con3 = P+2Q
; Sr[a-d], Lr0[a-d], Lr1[a-d] from r1.2

LOOP:
	vadduwm		vS16,vS16,vS15	; r1.16
		vspltw	vS17,vcon,2
	vadduwm		vS16,vS16,vL1
		vspltw	vS18,vcon,3
	vrlw		vS16,vS16,val3
		lvx		vcon,rcon,sp10x ; S0_19
	vadduwm		vtmp,vS16,vL1


	;
	vadduwm		vL0,vL0,vtmp


	add 	Sr_a,con3,Sr_a
	vrlw		vL0,vL0,vtmp


	add 	Sr_a,Sr_a,L0a
	vadduwm		vS17,vS17,vS16	; r1.17


	rotlwi	Sr_a,Sr_a,3
	vadduwm		vS17,vS17,vL0


	stw 	Sr_a,S1_3.a(sp)
	vrlw		vS17,vS17,val3


	add 	Sr_b,con3,Sr_b
	vadduwm		vtmp,vS17,vL0


	add 	Sr_b,Sr_b,L0b
	vadduwm		vL1,vL1,vtmp


	rotlwi	Sr_b,Sr_b,3
	vrlw		vL1,vL1,vtmp


	stw 	Sr_b,S1_3.b(sp)
	vadduwm		vS18,vS18,vS17	; r1.18


	add 	Sr_c,con3,Sr_c
	vadduwm		vS18,vS18,vL1


	add 	Sr_c,Sr_c,L0c
	vrlw		vS18,vS18,val3
		vspltw	vS19,vcon,0
	vadduwm		vtmp,vS18,vL1


	rotlwi	Sr_c,Sr_c,3
	vadduwm		vL0,vL0,vtmp


	stw 	Sr_c,S1_3.c(sp)
	vrlw		vL0,vL0,vtmp


	add 	Sr_d,con3,Sr_d
	vadduwm		vS19,vS19,vS18	; r1.19


	add 	Sr_d,Sr_d,L0d
	vadduwm		vS19,vS19,vL0


	rotlwi	Sr_d,Sr_d,3
	vrlw		vS19,vS19,val3
		vspltw	vS20,vcon,1
	vadduwm		vtmp,vS19,vL0


	stw 	Sr_d,S1_3.d(sp)
	vadduwm		vL1,vL1,vtmp


	add 	tmp,Sr_a,L0a
	vrlw		vL1,vL1,vtmp


	add 	L1a,L1a,tmp
	vadduwm		vS20,vS20,vS19	; r1.20


	rotlw	L1a,L1a,tmp
	vadduwm		vS20,vS20,vL1
		vspltw	vS21,vcon,2
	vrlw		vS20,vS20,val3
		lvx		vS6,rvec2,sp ; S1_6
	vadduwm		vtmp,vS20,vL1


	add 	tmp,Sr_b,L0b
	vadduwm		vL0,vL0,vtmp


	add 	L1b,L1b,tmp
	vrlw		vL0,vL0,vtmp


	rotlw	L1b,L1b,tmp
	vadduwm		vS21,vS21,vS20	; r1.21
		vspltw	vS22,vcon,3
	vadduwm		vS21,vS21,vL0
		lvx		vcon,rcon,sp20x ; S0_23
	vrlw		vS21,vS21,val3


	add 	tmp,Sr_c,L0c
	vadduwm		vtmp,vS21,vL0


	add 	L1c,L1c,tmp
	vadduwm		vL1,vL1,vtmp


	rotlw	L1c,L1c,tmp
	vrlw		vL1,vL1,vtmp


	add 	tmp,Sr_d,L0d
	vadduwm		vS22,vS22,vS21	; r1.22


	add 	L1d,L1d,tmp
	vadduwm		vS22,vS22,vL1
		lvx		vS7,rvec2,sp10x ; S1_7
	vrlw		vS22,vS22,val3


	rotlw	L1d,L1d,tmp
	vadduwm		vtmp,vS22,vL1
		vspltw	vS23,vcon,0
	vadduwm		vL0,vL0,vtmp
		vspltw	vS24,vcon,1
	vrlw		vL0,vL0,vtmp
		vspltw	vS25,vcon,2
	vadduwm		vS23,vS23,vS22	; r1.23
		vspltw	vS0,vcon,3
	vadduwm		vS23,vS23,vL0
		lvx		vcon,rcon,sp30x ; S1_1
	vrlw		vS23,vS23,val3


	add 	S0r,con3,Qr	; round 1.4 even
	vadduwm		vtmp,vS23,vL0


	add 	Sr_a,S0r,Sr_a
	vadduwm		vL1,vL1,vtmp


	add 	Sr_a,Sr_a,L1a
	vrlw		vL1,vL1,vtmp


	rotlwi	Sr_a,Sr_a,3
	vadduwm		vS24,vS24,vS23	; r1.24


	stw 	Sr_a,S1_4.a(sp)
	vadduwm		vS24,vS24,vL1


	add 	Sr_b,S0r,Sr_b
	vrlw		vS24,vS24,val3
		vspltw	vS1,vcon,0
	vadduwm		vtmp,vS24,vL1


	add 	Sr_b,Sr_b,L1b
	vadduwm		vL0,vL0,vtmp


	rotlwi	Sr_b,Sr_b,3
	vrlw		vL0,vL0,vtmp


	stw 	Sr_b,S1_4.b(sp)
	vadduwm		vS25,vS25,vS24	; r1.25


	add 	Sr_c,S0r,Sr_c
	vadduwm		vS25,vS25,vL0


	add 	Sr_c,Sr_c,L1c
	vrlw		vS25,vS25,val3


	rotlwi	Sr_c,Sr_c,3
	vadduwm		vtmp,vS25,vL0


	stw 	Sr_c,S1_4.c(sp)
	vadduwm		vL1,vL1,vtmp


	add 	Sr_d,S0r,Sr_d
	vrlw		vL1,vL1,vtmp


	add 	Sr_d,Sr_d,L1d
	vadduwm		vS0,vS0,vS25	; r2.0


	rotlwi	Sr_d,Sr_d,3
	vadduwm		vS0,vS0,vL1


	stw 	Sr_d,S1_4.d(sp)
	vrlw		vS0,vS0,val3


	add 	tmp,Sr_a,L1a
	vadduwm		vtmp,vS0,vL1


	add 	L0a,L0a,tmp
	vadduwm		vL0,vtmp,vL0


	rotlw	L0a,L0a,tmp
	vrlw		vL0,vL0,vtmp
		lvx		vS8,rvec2,sp20x ; S1_8
	vadduwm		vS1,vS1,vS0	; r2.1


	add 	tmp,Sr_b,L1b
	vadduwm		vS1,vS1,vL0


	add 	L0b,L0b,tmp
	vrlw		vS1,vS1,val3


	rotlw	L0b,L0b,tmp
	vadduwm		vtmp,vS1,vL0


	add 	tmp,Sr_c,L1c
	vadduwm		vL1,vtmp,vL1


	add 	L0c,L0c,tmp
	vrlw		vL1,vL1,vtmp
		lvx		vS9,rvec2,sp30x ; S1_9
	vadduwm		vS2,vS2,vS1	; r2.2


	rotlw	L0c,L0c,tmp
	vadduwm		vS2,vS2,vL1


	add 	tmp,Sr_d,L1d
	vrlw		vS2,vS2,val3


	add 	L0d,L0d,tmp
	vadduwm		vtmp,vS2,vL1


	rotlw	L0d,L0d,tmp
	vadduwm		vL0,vtmp,vL0


	add 	S0r,S0r,Qr	; round 1.5 odd
	vrlw		vL0,vL0,vtmp
		lvx		vS10,rvec3,sp ; S1_10

	vadduwm		vS3,vS3,vS2	; r2.3


	add 	Sr_a,S0r,Sr_a
	vadduwm		vS3,vS3,vL0


	add 	Sr_a,Sr_a,L0a
	vrlw		vS3,vS3,val3


	rotlwi	Sr_a,Sr_a,3
	vadduwm		vtmp,vS3,vL0


	stw 	Sr_a,S1_5.a(sp)
	vadduwm		vL1,vtmp,vL1


	add 	Sr_b,S0r,Sr_b
	vrlw		vL1,vL1,vtmp


	add 	Sr_b,Sr_b,L0b
	vadduwm		vS4,vS4,vS3	; r2.4


	rotlwi	Sr_b,Sr_b,3
	vadduwm		vS4,vS4,vL1


	stw 	Sr_b,S1_5.b(sp)
	vrlw		vS4,vS4,val3


	add 	Sr_c,S0r,Sr_c
	vadduwm		vtmp,vS4,vL1


	add 	Sr_c,Sr_c,L0c
	vadduwm		vL0,vtmp,vL0


	rotlwi	Sr_c,Sr_c,3
	vrlw		vL0,vL0,vtmp


	stw 	Sr_c,S1_5.c(sp)
	vadduwm		vS5,vS5,vS4	; r2.5


	add 	Sr_d,S0r,Sr_d
	vadduwm		vS5,vS5,vL0


	add 	Sr_d,Sr_d,L0d
	vrlw		vS5,vS5,val3


	rotlwi	Sr_d,Sr_d,3
	vadduwm		vtmp,vS5,vL0


	stw 	Sr_d,S1_5.d(sp)
	vadduwm		vL1,vtmp,vL1


	add 	tmp,Sr_a,L0a
	vrlw		vL1,vL1,vtmp


	add 	L1a,L1a,tmp
	vadduwm		vS6,vS6,vS5	; r2.6


	rotlw	L1a,L1a,tmp
	vadduwm		vS6,vS6,vL1
		lvx		vS11,rvec3,sp10x ; S1_11
	vrlw		vS6,vS6,val3


	add 	tmp,Sr_b,L0b
	vadduwm		vtmp,vS6,vL1


	add 	L1b,L1b,tmp
	vadduwm		vL0,vtmp,vL0


	rotlw	L1b,L1b,tmp
	vrlw		vL0,vL0,vtmp


	add 	tmp,Sr_c,L0c
	vadduwm		vS7,vS7,vS6	; r2.7


	add 	L1c,L1c,tmp
	vadduwm		vS7,vS7,vL0
		lvx		vS12,rvec3,sp20x ; S1_12
	vrlw		vS7,vS7,val3


	rotlw	L1c,L1c,tmp
	vadduwm		vtmp,vS7,vL0


	add 	tmp,Sr_d,L0d
	vadduwm		vL1,vtmp,vL1


	add 	L1d,L1d,tmp
	vrlw		vL1,vL1,vtmp


	rotlw	L1d,L1d,tmp
	vadduwm		vS8,vS8,vS7	; r2.8


	add 	S0r,S0r,Qr	; round 1.6 even
	vadduwm		vS8,vS8,vL1


	add 	Sr_a,S0r,Sr_a
	vrlw		vS8,vS8,val3


	add 	Sr_a,Sr_a,L1a
	vadduwm		vtmp,vS8,vL1


	rotlwi	Sr_a,Sr_a,3
	vadduwm		vL0,vtmp,vL0


	stw 	Sr_a,S1_6.a(sp)
	vrlw		vL0,vL0,vtmp


	add 	Sr_b,S0r,Sr_b
	vadduwm		vS9,vS9,vS8	; r2.9


	add 	Sr_b,Sr_b,L1b
	vadduwm		vS9,vS9,vL0					

	rotlwi	Sr_b,Sr_b,3
	vrlw		vS9,vS9,val3


	stw 	Sr_b,S1_6.b(sp)
	vadduwm		vtmp,vS9,vL0


	add 	Sr_c,S0r,Sr_c
	vadduwm		vL1,vtmp,vL1


	add 	Sr_c,Sr_c,L1c
	vrlw		vL1,vL1,vtmp


	rotlwi	Sr_c,Sr_c,3
	vadduwm		vS10,vS10,vS9	; r2.10


	stw 	Sr_c,S1_6.c(sp)
	vadduwm		vS10,vS10,vL1


	add 	Sr_d,S0r,Sr_d
	vrlw		vS10,vS10,val3


	add 	Sr_d,Sr_d,L1d
	vadduwm		vtmp,vS10,vL1


	rotlwi	Sr_d,Sr_d,3
	vadduwm		vL0,vtmp,vL0


	stw 	Sr_d,S1_6.d(sp)
	vrlw		vL0,vL0,vtmp


	add 	tmp,Sr_a,L1a
	vadduwm		vS11,vS11,vS10	; r2.11


	add 	L0a,L0a,tmp
	vadduwm		vS11,vS11,vL0


	rotlw	L0a,L0a,tmp
	vrlw		vS11,vS11,val3


	add 	tmp,Sr_b,L1b
	vadduwm		vtmp,vS11,vL0
		lvx		vS13,rvec3,sp30x ; S1_13
	vadduwm		vL1,vtmp,vL1


	add 	L0b,L0b,tmp
	vrlw		vL1,vL1,vtmp


	rotlw	L0b,L0b,tmp
	vadduwm		vS12,vS12,vS11	; r2.12


	add 	tmp,Sr_c,L1c
	vadduwm		vS12,vS12,vL1


	add 	L0c,L0c,tmp
	vrlw		vS12,vS12,val3


	rotlw	L0c,L0c,tmp
	vadduwm		vtmp,vS12,vL1


	add 	tmp,Sr_d,L1d
	vadduwm		vL0,vtmp,vL0


	add 	L0d,L0d,tmp
	vrlw		vL0,vL0,vtmp


	rotlw	L0d,L0d,tmp
	vadduwm		vS13,vS13,vS12	; r2.13


	add 	S0r,S0r,Qr	; round 1.7 odd
	vadduwm		vS13,vS13,vL0


	add 	Sr_a,S0r,Sr_a
	vrlw		vS13,vS13,val3


	add 	Sr_a,Sr_a,L0a
	vadduwm		vtmp,vS13,vL0


	rotlwi	Sr_a,Sr_a,3
	vadduwm		vL1,vtmp,vL1


	stw 	Sr_a,S1_7.a(sp)
	vrlw		vL1,vL1,vtmp


	add 	Sr_b,S0r,Sr_b
	vadduwm		vS14,vS14,vS13	; r2.14


	add 	Sr_b,Sr_b,L0b
	vadduwm		vS14,vS14,vL1


	rotlwi	Sr_b,Sr_b,3
	vrlw		vS14,vS14,val3


	stw 	Sr_b,S1_7.b(sp)
	vadduwm		vtmp,vS14,vL1


	add 	Sr_c,S0r,Sr_c
	vadduwm		vL0,vtmp,vL0


	add 	Sr_c,Sr_c,L0c
	vrlw		vL0,vL0,vtmp


	rotlwi	Sr_c,Sr_c,3
	vadduwm		vS15,vS15,vS14	; r2.15


	stw 	Sr_c,S1_7.c(sp)
	vadduwm		vS15,vS15,vL0


	add 	Sr_d,S0r,Sr_d
	vrlw		vS15,vS15,val3


	add 	Sr_d,Sr_d,L0d
	vadduwm		vtmp,vS15,vL0


	rotlwi	Sr_d,Sr_d,3
	vadduwm		vL1,vtmp,vL1


	stw 	Sr_d,S1_7.d(sp)
	vrlw		vL1,vL1,vtmp


	add 	tmp,Sr_a,L0a
	vadduwm		vS16,vS16,vS15	; r2.16


	add 	L1a,L1a,tmp
	vadduwm		vS16,vS16,vL1


	rotlw	L1a,L1a,tmp
	vrlw		vS16,vS16,val3


	add 	tmp,Sr_b,L0b
	vadduwm		vtmp,vS16,vL1


	add 	L1b,L1b,tmp
	vadduwm		vL0,vtmp,vL0


	rotlw	L1b,L1b,tmp
	vrlw		vL0,vL0,vtmp


	add 	tmp,Sr_c,L0c
	vadduwm		vS17,vS17,vS16	; r2.17


	add 	L1c,L1c,tmp
	vadduwm		vS17,vS17,vL0


	rotlw	L1c,L1c,tmp
	vrlw		vS17,vS17,val3


	add 	tmp,Sr_d,L0d
	vadduwm		vtmp,vS17,vL0


	add 	L1d,L1d,tmp
	vadduwm		vL1,vtmp,vL1


	rotlw	L1d,L1d,tmp
	vrlw		vL1,vL1,vtmp


	add 	S0r,S0r,Qr	; round 1.8 even
	vadduwm		vS18,vS18,vS17	; r2.18


	add 	Sr_a,S0r,Sr_a
	vadduwm		vS18,vS18,vL1


	add 	Sr_a,Sr_a,L1a
	vrlw		vS18,vS18,val3


	rotlwi	Sr_a,Sr_a,3
	vadduwm		vtmp,vS18,vL1


	stw 	Sr_a,S1_8.a(sp)
	vadduwm		vL0,vtmp,vL0


	add 	Sr_b,S0r,Sr_b
	vrlw		vL0,vL0,vtmp


	add 	Sr_b,Sr_b,L1b
	vadduwm		vS19,vS19,vS18	; r2.19


	rotlwi	Sr_b,Sr_b,3
	vadduwm		vS19,vS19,vL0


	stw 	Sr_b,S1_8.b(sp)
	vrlw		vS19,vS19,val3


	add 	Sr_c,S0r,Sr_c
	vadduwm		vtmp,vS19,vL0


	add 	Sr_c,Sr_c,L1c
	vadduwm		vL1,vtmp,vL1


	rotlwi	Sr_c,Sr_c,3
	vrlw		vL1,vL1,vtmp


	stw 	Sr_c,S1_8.c(sp)
	vadduwm		vS20,vS20,vS19	; r2.20


	add 	Sr_d,S0r,Sr_d
	vadduwm		vS20,vS20,vL1


	add 	Sr_d,Sr_d,L1d
	vrlw		vS20,vS20,val3


	rotlwi	Sr_d,Sr_d,3
	vadduwm		vtmp,vS20,vL1			

	stw 	Sr_d,S1_8.d(sp)
	vadduwm		vL0,vtmp,vL0


	add 	tmp,Sr_a,L1a
	vrlw		vL0,vL0,vtmp


	add 	L0a,L0a,tmp
	vadduwm		vS21,vS21,vS20	; r2.21


	rotlw	L0a,L0a,tmp
	vadduwm		vS21,vS21,vL0


	add 	tmp,Sr_b,L1b
	vrlw		vS21,vS21,val3


	add 	L0b,L0b,tmp
	vadduwm		vtmp,vS21,vL0


	rotlw	L0b,L0b,tmp
	vadduwm		vL1,vtmp,vL1


	add 	tmp,Sr_c,L1c
	vrlw		vL1,vL1,vtmp


	add 	L0c,L0c,tmp
	vadduwm		vS22,vS22,vS21	; r2.22


	rotlw	L0c,L0c,tmp
	vadduwm		vS22,vS22,vL1


	add 	tmp,Sr_d,L1d
	vrlw		vS22,vS22,val3


	add 	L0d,L0d,tmp
	vadduwm		vtmp,vS22,vL1


	rotlw	L0d,L0d,tmp
	vadduwm		vL0,vtmp,vL0


	add 	S0r,S0r,Qr	; round 1.9 odd
	vrlw		vL0,vL0,vtmp


	add 	Sr_a,S0r,Sr_a
	vadduwm		vS23,vS23,vS22	; r2.23


	add 	Sr_a,Sr_a,L0a
	vadduwm		vS23,vS23,vL0


	rotlwi	Sr_a,Sr_a,3
	vrlw		vS23,vS23,val3


	stw 	Sr_a,S1_9.a(sp)
	vadduwm		vtmp,vS23,vL0


	add 	Sr_b,S0r,Sr_b
	vadduwm		vL1,vtmp,vL1


	add 	Sr_b,Sr_b,L0b
	vrlw		vL1,vL1,vtmp


	rotlwi	Sr_b,Sr_b,3
	vadduwm		vS24,vS24,vS23	; r2.24


	stw 	Sr_b,S1_9.b(sp)
	vadduwm		vS24,vS24,vL1


	add 	Sr_c,S0r,Sr_c
	vrlw		vS24,vS24,val3


	add 	Sr_c,Sr_c,L0c
	vadduwm		vtmp,vS24,vL1


	rotlwi	Sr_c,Sr_c,3
	vadduwm		vL0,vtmp,vL0


	stw 	Sr_c,S1_9.c(sp)
	vrlw		vL0,vL0,vtmp


	add 	Sr_d,S0r,Sr_d
	vadduwm		vS25,vS25,vS24	; r2.25


	add 	Sr_d,Sr_d,L0d
	vadduwm		vS25,vS25,vL0


	rotlwi	Sr_d,Sr_d,3
	vrlw		vS25,vS25,val3


	stw 	Sr_d,S1_9.d(sp)
	vadduwm		vtmp,vS25,vL0


	add 	tmp,Sr_a,L0a
	vadduwm		vL1,vtmp,vL1


	add 	L1a,L1a,tmp
	vrlw		vL1,vL1,vtmp


	rotlw	L1a,L1a,tmp
	vadduwm		vS0,vS0,vS25	; r3.0


	add 	tmp,Sr_b,L0b
	vadduwm		vS0,vS0,vL1


	add 	L1b,L1b,tmp
	vrlw		vS0,vS0,val3


	rotlw	L1b,L1b,tmp
	vadduwm		vtmp,vS0,vL1


	add 	tmp,Sr_c,L0c
	vadduwm		vL0,vtmp,vL0


	add 	L1c,L1c,tmp
	vrlw		vL0,vL0,vtmp
		vspltw	vtmp,vcon,1 ;P_0(SP)
	vadduwm		vS1,vS1,vS0	; r3.1


	rotlw	L1c,L1c,tmp
	vadduwm		vS1,vS1,vL0


	add 	tmp,Sr_d,L0d
	vrlw		vS1,vS1,val3


	add 	L1d,L1d,tmp
	vadduwm		vAx,vtmp,vS0


	rotlw	L1d,L1d,tmp
	vadduwm		vtmp,vS1,vL0


	add 	S0r,S0r,Qr	; round 1.10 even
	vadduwm		vL1,vtmp,vL1


	add 	Sr_a,S0r,Sr_a
	vrlw		vL1,vL1,vtmp
		vspltw	vtmp,vcon,2	;P_1

	vadduwm		vS2,vS2,vS1	; r3.2


	add 	Sr_a,Sr_a,L1a
	vadduwm		vS2,vS2,vL1


	rotlwi	Sr_a,Sr_a,3
	vrlw		vS2,vS2,val3


	stw 	Sr_a,S1_10.a(sp)
	vadduwm		vBx,vtmp,vS1


	add 	Sr_b,S0r,Sr_b
	vadduwm		vtmp,vS2,vL1


	add 	Sr_b,Sr_b,L1b
	vadduwm		vL0,vtmp,vL0


	rotlwi	Sr_b,Sr_b,3
	vrlw		vL0,vL0,vtmp


	stw 	Sr_b,S1_10.b(sp)
	vxor 		vAx,vAx,vBx


	add 	Sr_c,S0r,Sr_c
	vrlw		vAx,vAx,vBx


	add 	Sr_c,Sr_c,L1c
	vadduwm		vAx,vAx,vS2


	rotlwi	Sr_c,Sr_c,3
	vadduwm		vS3,vS3,vS2	; r3.3


	stw 	Sr_c,S1_10.c(sp)
	vadduwm		vS3,vS3,vL0	; vS2 free


	add 	Sr_d,S0r,Sr_d
	vrlw		vS3,vS3,val3


	add 	Sr_d,Sr_d,L1d
	vadduwm		vtmp,vS3,vL0


	rotlwi	Sr_d,Sr_d,3
	vadduwm		vL1,vtmp,vL1


	stw 	Sr_d,S1_10.d(sp)
	vrlw		vL1,vL1,vtmp


	add 	tmp,Sr_a,L1a
	vxor 		vBx,vBx,vAx


	add 	L0a,L0a,tmp
	vrlw		vBx,vBx,vAx


	rotlw	L0a,L0a,tmp
	vadduwm		vBx,vBx,vS3


	add 	tmp,Sr_b,L1b
	vadduwm		vS4,vS4,vS3	; r3.4


	add 	L0b,L0b,tmp
	vadduwm		vS4,vS4,vL1


	rotlw	L0b,L0b,tmp
	vrlw		vS4,vS4,val3


	add 	tmp,Sr_c,L1c
	vadduwm		vtmp,vS4,vL1


	add 	L0c,L0c,tmp
	vadduwm		vL0,vtmp,vL0
		lvx		vS2,rvec1,sp ; S1_2	; load now before it's overwriten

	vrlw		vL0,vL0,vtmp


	rotlw	L0c,L0c,tmp
	vxor 		vAx,vAx,vBx


	add 	tmp,Sr_d,L1d
	vrlw		vAx,vAx,vBx


	add 	L0d,L0d,tmp
	vadduwm		vAx,vAx,vS4


	rotlw	L0d,L0d,tmp
	vadduwm		vS5,vS5,vS4	; r3.5


	add 	S0r,S0r,Qr	; round 1.11 odd
	vadduwm		vS5,vS5,vL0
		lvx		vS3,rvec1,sp10x ; S1_3
	vrlw		vS5,vS5,val3


	add 	Sr_a,S0r,Sr_a
	vadduwm		vtmp,vS5,vL0


	add 	Sr_a,Sr_a,L0a
	vadduwm		vL1,vtmp,vL1


	rotlwi	Sr_a,Sr_a,3
	vrlw		vL1,vL1,vtmp


	stw 	Sr_a,S1_11.a(sp)
	vxor 		vBx,vBx,vAx


	add 	Sr_b,S0r,Sr_b
	vrlw		vBx,vBx,vAx


	add 	Sr_b,Sr_b,L0b
	vadduwm		vBx,vBx,vS5


	rotlwi	Sr_b,Sr_b,3
	vadduwm		vS6,vS6,vS5	; r3.6


	stw 	Sr_b,S1_11.b(sp)
	vadduwm		vS6,vS6,vL1


	add 	Sr_c,S0r,Sr_c
	vrlw		vS6,vS6,val3


	add 	Sr_c,Sr_c,L0c
	vadduwm		vtmp,vS6,vL1


	rotlwi	Sr_c,Sr_c,3
	vadduwm		vL0,vtmp,vL0


	stw 	Sr_c,S1_11.c(sp)
	vrlw		vL0,vL0,vtmp


	add 	Sr_d,S0r,Sr_d
	vxor 		vAx,vAx,vBx


	add 	Sr_d,Sr_d,L0d
	vrlw		vAx,vAx,vBx


	rotlwi	Sr_d,Sr_d,3
	vadduwm		vAx,vAx,vS6


	stw 	Sr_d,S1_11.d(sp)
	vadduwm		vS7,vS7,vS6	; r3.7


	add 	tmp,Sr_a,L0a
	vadduwm		vS7,vS7,vL0


	add 	L1a,L1a,tmp
	vrlw		vS7,vS7,val3


	rotlw	L1a,L1a,tmp
	vadduwm		vtmp,vS7,vL0

									;//
	lwz 	key1,L0_1(sp)
	vadduwm		vL1,vtmp,vL1


	add 	tmp,Sr_b,L0b
	vrlw		vL1,vL1,vtmp


	add 	L1b,L1b,tmp
	vxor 		vBx,vBx,vAx


	rotlw	L1b,L1b,tmp
	vrlw		vBx,vBx,vAx


	add 	tmp,Sr_c,L0c
	vadduwm		vBx,vBx,vS7


	addi	key1,key1,4
	vadduwm		vS8,vS8,vS7	; r3.8


	stw 	key1,L0_1(sp)
	vadduwm		vS8,vS8,vL1


	add 	L1c,L1c,tmp
	vrlw		vS8,vS8,val3


	rotlw	L1c,L1c,tmp
	vadduwm		vtmp,vS8,vL1


	add 	tmp,Sr_d,L0d
	vadduwm		vL0,vtmp,vL0


	add 	L1d,L1d,tmp
	vrlw		vL0,vL0,vtmp


	rotlw	L1d,L1d,tmp
	vxor 		vAx,vAx,vBx


	vrlw		vAx,vAx,vBx


	add 	S0r,S0r,Qr	; round 1.12 even
	vadduwm		vAx,vAx,vS8


	add 	Sr_a,S0r,Sr_a
	vadduwm		vS9,vS9,vS8	; r3.9


	add 	Sr_a,Sr_a,L1a
	vadduwm		vS9,vS9,vL0


	rotlwi	Sr_a,Sr_a,3
	vrlw		vS9,vS9,val3


	stw 	Sr_a,S1_12.a(sp)
	vadduwm		vtmp,vS9,vL0


	add 	Sr_b,S0r,Sr_b
	vadduwm		vL1,vtmp,vL1


	add 	Sr_b,Sr_b,L1b
	vrlw		vL1,vL1,vtmp


	rotlwi	Sr_b,Sr_b,3
	vxor 		vBx,vBx,vAx


	stw 	Sr_b,S1_12.b(sp)
	vrlw		vBx,vBx,vAx


	add 	Sr_c,S0r,Sr_c
	vadduwm		vBx,vBx,vS9


	add 	Sr_c,Sr_c,L1c
	vadduwm		vS10,vS10,vS9	; r3.10


	rotlwi	Sr_c,Sr_c,3
	vadduwm		vS10,vS10,vL1


	stw 	Sr_c,S1_12.c(sp)
	vrlw		vS10,vS10,val3


	add 	Sr_d,S0r,Sr_d
	vadduwm		vtmp,vS10,vL1


	add 	Sr_d,Sr_d,L1d
	vadduwm		vL0,vtmp,vL0


	rotlwi	Sr_d,Sr_d,3
	vrlw		vL0,vL0,vtmp


	stw 	Sr_d,S1_12.d(sp)
	vxor 		vAx,vAx,vBx


	add 	tmp,Sr_a,L1a
	vrlw		vAx,vAx,vBx


	add 	L0a,L0a,tmp
	vadduwm		vAx,vAx,vS10


	rotlw	L0a,L0a,tmp
	vadduwm		vS11,vS11,vS10	; r3.11


	add 	tmp,Sr_b,L1b
	vadduwm		vS11,vS11,vL0

		lvx		vS4,rvec1,sp20x ; S1_4
	vrlw		vS11,vS11,val3


	add 	L0b,L0b,tmp
	vadduwm		vtmp,vS11,vL0


	rotlw	L0b,L0b,tmp
	vadduwm		vL1,vtmp,vL1


	add 	tmp,Sr_c,L1c
	vrlw		vL1,vL1,vtmp


	add 	L0c,L0c,tmp
	vxor 		vBx,vBx,vAx


	rotlw	L0c,L0c,tmp
	vrlw		vBx,vBx,vAx
		lvx		vS5,rvec1,sp30x ; S1_5
	vadduwm		vBx,vBx,vS11


	add 	tmp,Sr_d,L1d
	vadduwm		vS12,vS12,vS11	; r3.12


	add 	L0d,L0d,tmp
	vadduwm		vS12,vS12,vL1


	rotlw	L0d,L0d,tmp
	vrlw		vS12,vS12,val3

	add 	S0r,S0r,Qr	;round 1.13 odd
	vadduwm		vtmp,vS12,vL1


	add 	Sr_a,S0r,Sr_a
	vadduwm		vL0,vtmp,vL0


	add 	Sr_a,Sr_a,L0a
	vrlw		vL0,vL0,vtmp


	rotlwi	Sr_a,Sr_a,3
	vxor 		vAx,vAx,vBx


	stw 	Sr_a,S1_13.a(sp)
	vrlw		vAx,vAx,vBx


	add 	Sr_b,S0r,Sr_b
	vadduwm		vAx,vAx,vS12


	add 	Sr_b,Sr_b,L0b
	vadduwm		vS13,vS13,vS12	; r3.13


	rotlwi	Sr_b,Sr_b,3
	vadduwm		vS13,vS13,vL0


	stw 	Sr_b,S1_13.b(sp)
	vrlw		vS13,vS13,val3


	add 	Sr_c,S0r,Sr_c
	vadduwm		vtmp,vS13,vL0


	add 	Sr_c,Sr_c,L0c
	vadduwm		vL1,vtmp,vL1


	rotlwi	Sr_c,Sr_c,3
	vrlw		vL1,vL1,vtmp


	stw 	Sr_c,S1_13.c(sp)
	vxor 		vBx,vBx,vAx


	add 	Sr_d,S0r,Sr_d
	vrlw		vBx,vBx,vAx


	add 	Sr_d,Sr_d,L0d
	vadduwm		vBx,vBx,vS13


	rotlwi	Sr_d,Sr_d,3
	vadduwm		vS14,vS14,vS13	; r3.14


	stw 	Sr_d,S1_13.d(sp)
	vadduwm		vS14,vS14,vL1


	add 	tmp,Sr_a,L0a
	vrlw		vS14,vS14,val3


	add 	L1a,L1a,tmp
	vadduwm		vtmp,vS14,vL1


	rotlw	L1a,L1a,tmp
	vadduwm		vL0,vtmp,vL0


	stw 	L1a,L_1.a(sp)
	vrlw		vL0,vL0,vtmp


	add 	tmp,Sr_b,L0b
	vxor 		vAx,vAx,vBx


	add 	L1b,L1b,tmp
	vrlw		vAx,vAx,vBx


	rotlw	L1b,L1b,tmp
	vadduwm		vAx,vAx,vS14


	stw 	L1b,L_1.b(sp)
	vadduwm		vS15,vS15,vS14	; r3.15


	add 	tmp,Sr_c,L0c
	vadduwm		vS15,vS15,vL0


	add 	L1c,L1c,tmp
	vrlw		vS15,vS15,val3


	rotlw	L1c,L1c,tmp
	vadduwm		vtmp,vS15,vL0


	stw 	L1c,L_1.c(sp)
	vadduwm		vL1,vtmp,vL1


	add 	tmp,Sr_d,L0d
	vrlw		vL1,vL1,vtmp


	add 	L1d,L1d,tmp
	vxor 		vBx,vBx,vAx


	rotlw	L1d,L1d,tmp
	vrlw		vBx,vBx,vAx


	stw 	L1d,L_1.d(sp)
	vadduwm		vBx,vBx,vS15


	add 	S0r,S0r,Qr	; round 1.14 even
	vadduwm		vS16,vS16,vS15	; r3.16


	add 	Sr_a,S0r,Sr_a
	vadduwm		vS16,vS16,vL1


	add 	Sr_a,Sr_a,L1a
	vrlw		vS16,vS16,val3


	rotlwi	Sr_a,Sr_a,3
	vadduwm		vtmp,vS16,vL1


	stw 	Sr_a,tVEC.a(sp)
	vadduwm		vL0,vtmp,vL0


	add 	Sr_b,S0r,Sr_b
	vrlw		vL0,vL0,vtmp


	add 	Sr_b,Sr_b,L1b
	vxor 		vAx,vAx,vBx


	rotlwi	Sr_b,Sr_b,3
	vrlw		vAx,vAx,vBx


	stw 	Sr_b,tVEC.b(sp)
	vadduwm		vAx,vAx,vS16


	add 	Sr_c,S0r,Sr_c
	vadduwm		vS17,vS17,vS16	; r3.17


	add 	Sr_c,Sr_c,L1c
	vadduwm		vS17,vS17,vL0


	rotlwi	Sr_c,Sr_c,3
	vrlw		vS17,vS17,val3


	stw 	Sr_c,tVEC.c(sp)
	vadduwm		vtmp,vS17,vL0


	add 	Sr_d,S0r,Sr_d
	vadduwm		vL1,vtmp,vL1


	add 	Sr_d,Sr_d,L1d
	vrlw		vL1,vL1,vtmp


	rotlwi	Sr_d,Sr_d,3
	vxor 		vBx,vBx,vAx


	stw 	Sr_d,tVEC.d(sp)
	vrlw		vBx,vBx,vAx


	add 	tmp,Sr_a,L1a
	vadduwm		vBx,vBx,vS17


	add 	L0a,L0a,tmp
	vadduwm		vS18,vS18,vS17	; r3.18


	rotlw	L0a,L0a,tmp
	vadduwm		vS18,vS18,vL1


	stw 	L0a,L_0.a(sp)
	vrlw		vS18,vS18,val3


	add 	tmp,Sr_b,L1b
	vadduwm		vtmp,vS18,vL1


	add 	L0b,L0b,tmp
	vadduwm		vL0,vtmp,vL0


	rotlw	L0b,L0b,tmp
	vrlw		vL0,vL0,vtmp


	stw 	L0b,L_0.b(sp)
	vxor 		vAx,vAx,vBx


	add 	tmp,Sr_c,L1c
	vrlw		vAx,vAx,vBx


	add 	L0c,L0c,tmp
	vadduwm		vAx,vAx,vS18


	rotlw	L0c,L0c,tmp
	vadduwm		vS19,vS19,vS18	; r3.19


	stw 	L0c,L_0.c(sp)
	vadduwm		vS19,vS19,vL0


	add 	tmp,Sr_d,L1d
	vrlw		vS19,vS19,val3


	add 	L0d,L0d,tmp
	vadduwm		vtmp,vS19,vL0


	rotlw	L0d,L0d,tmp
	vadduwm		vL1,vtmp,vL1


	stw 	L0d,L_0.d(sp)
	vrlw		vL1,vL1,vtmp


	lwbrx	L1a,0,sp20x	; L0_1 key1

	add 	L1a,L1a,con1	; round 1.1 odd
	vxor 		vBx,vBx,vAx


	addis	L1b,L1a,256
	vrlw		vBx,vBx,vAx


	addis	L1c,L1b,256
	vadduwm		vBx,vBx,vS19


	addis	L1d,L1c,256
	vadduwm		vS20,vS20,vS19	; r3.20
		vspltw	vC,vcon,3	; = v19
	vadduwm		vS20,vS20,vL1
		lvx		vcon,rcon,sp ; S0_15
	vrlw		vS20,vS20,val3


	rotlw	L1a,L1a,con1
	vadduwm		vtmp,vS20,vL1


	rotlw	L1b,L1b,con1
	vadduwm		vL0,vtmp,vL0


	rotlw	L1c,L1c,con1
	vrlw		vL0,vL0,vtmp


	rotlw	L1d,L1d,con1
	vxor 		vAx,vAx,vBx


	add 	Sr_a,con2,L1a	; round 1.2 even
	vrlw		vAx,vAx,vBx


	rotlwi	Sr_a,Sr_a,3
	vadduwm		vAx,vAx,vS20


	stw 	Sr_a,S1_2.a(sp)
	vadduwm		vS21,vS21,vS20	; r3.21


	add 	Sr_b,con2,L1b
	vadduwm		vS21,vS21,vL0


	rotlwi	Sr_b,Sr_b,3
	vrlw		vS21,vS21,val3


	stw 	Sr_b,S1_2.b(sp)
	vadduwm		vtmp,vS21,vL0


	add 	Sr_c,con2,L1c
	vadduwm		vL1,vtmp,vL1


	rotlwi	Sr_c,Sr_c,3
	vrlw		vL1,vL1,vtmp


	stw 	Sr_c,S1_2.c(sp)
	vxor 		vBx,vBx,vAx


	add 	Sr_d,con2,L1d
	vrlw		vBx,vBx,vAx


	rotlwi	Sr_d,Sr_d,3
	vadduwm		vBx,vBx,vS21


	stw 	Sr_d,S1_2.d(sp)
	vadduwm		vS22,vS22,vS21	; r3.22


	add 	tmp,Sr_a,L1a
	vadduwm		vS22,vS22,vL1


	add 	L0a,con0,tmp
	vrlw		vS22,vS22,val3


	rotlw	L0a,L0a,tmp
	vadduwm		vtmp,vS22,vL1


	add 	tmp,Sr_b,L1b
	vadduwm		vL0t,vtmp,vL0	;->v20
		lvx		vL0,tmpx,sp10x ; L_0

	vrlw		vL0t,vL0t,vtmp


	add 	L0b,con0,tmp
	vxor 		vAx,vAx,vBx


	rotlw	L0b,L0b,tmp
	vrlw		vAx,vAx,vBx


	add 	tmp,Sr_c,L1c
	vadduwm		vAx,vAx,vS22


	add 	L0c,con0,tmp
	vadduwm		vS23,vS23,vS22	; r3.23


	rotlw	L0c,L0c,tmp
	vadduwm		vS23,vS23,vL0t


	add 	tmp,Sr_d,L1d
	vrlw		vS23,vS23,val3


	add 	L0d,con0,tmp
	vadduwm		vtmp,vS23,vL0t


	rotlw	L0d,L0d,tmp
	vadduwm		vL1t,vtmp,vL1	;->v21
		lvx 	vL1,tmpx,sp20x ; L_1
	vrlw		vL1t,vL1t,vtmp

	vxor 		vBx,vBx,vAx
	vrlw		vBx,vBx,vAx

	vadduwm		vBx,vBx,vS23

	vadduwm		vS24,vS24,vS23	; r3.24
		lvx 	vS14,tmpx,sp ; tVEC = S1_14
	vadduwm		vS24,vS24,vL1t
	vrlw		vS24,vS24,val3
	vxor 		vAx,vAx,vBx
	vrlw		vAx,vAx,vBx
	vadduwm		vAx,vAx,vS24
	vcmpequw.	vhit,vAx,vC	;->v22
		vspltw	vS15,vcon,0
;Start processing next vector group
	vadduwm		vS15,vS15,vS14	; r1.15
	vadduwm		vS15,vS15,vL0
	vrlw		vS15,vS15,val3
	vadduwm		vtmp,vS15,vL0
	vadduwm		vL1,vL1,vtmp
	vrlw		vL1,vL1,vtmp
		vspltw	vS16,vcon,1
		bdnzt	26,loop			; cnt > 0 and vhit == all 0

;	441	cycles for vector integer instructions

;	35	interleaved vector load/splat instructions

;	393	interleaved integer instructions
;	13	holes
;
; These registers are available to finish 3.24 and 3.25

; vAx=v0, vBx=v1, vS24, vS25, vL0t=v20, vL1t=v21, vhit=v22


; Preserve registers vS2-vS5, vS15, vS16, vL0, vL1, vcon
; Sr[a-d], Lr0[a-d], Lr1[a-d], con1, con2

; Registers available for use:
;	v6-v14,v17-v19,v23,v26,v27,v29
	mfctr	cnt
	bt	26,done	; vhit == all 0 so ctr must have expired

; finish last round
	vadduwm		vtmp,vS24,vL1t

	vadduwm		vL0t,vtmp,vL0t
	vrlw		vL0t,vL0t,vtmp

	vadduwm		vS25,vS25,vS24	; r3.25

	vadduwm		vS25,vS25,vL0t

	vrlw		vS25,vS25,val3

	vxor 		vBx,vBx,vAx
	vrlw		vBx,vBx,vAx

	vadduwm		vBx,vBx,vS25

	lvx			vC,0,sp20x
	vspltw		vC,vC,1
	vcmpequw	vtmp,vBx,vC
	vand		vhit,vhit,vtmp
	vspltisw	vtmp,-1
	vcmpequw.	vtmp,vtmp,vhit
	bf  		26,found
	cmplwi		cnt,0
	bne 		loop
done:
; cnt is 0, fix rollover, check for remaining iterations
; Exit phase registers
; key1	EQU r18
cnt 	EQU r19		; (was S0r)
;key0	EQU	r13		; (was con0)
vzero	EQU v1

; // handle remaining iterations
; fixup key for not found
;	key1 is in register
	lwz 	key0,L0_0(sp)
	addi	key1,key1,-5	; backup last speculative iteration
	addic	key1,key1,1	; set carry if key1 rolled to 0
	addze	key0,key0	; add carry to key0
	neg 	tmp,iter	; test iter for -3..0
	clrrwi.	cnt,tmp,2
	bne		start		; finish keys after roll over
	subfc 	key1,tmp,key1	; fix the overrun
	addme	key0,key0
	li		iter,0
	b exit

found:		; find first winning vector
	;note - possible key1 rollover is undone by first decrement
;//	lwz 	key1,L0_1(sp)
	addi	key1,key1,-4	; backup last speculative iteration
	lwz 	key0,L0_0(sp)
	vspltisw	vzero,0
	sli 	cnt,cnt,2
	add 	iter,iter,cnt
shift_out:
	addi	key1,key1,-1		# decrement key1 and shift the high result off
	addi	iter,iter,1
	vsldoi	vhit,vzero,vhit,12	# -> 0,key+0,key+1,key+2
	vcmpequw.	vtmp,vhit,vzero
	bf		24,shift_out	# repeat until the lowest hit is removed (max 4)

exit:	# common exit code
;// check that the guard words have not been disturbed
	lvx 		vhit,tmpx,sp30x
	vspltisw	v0,3
	vcmpequw	v0,v0,val3	#(sets condition)
	vcmpequw	vhit,vhit,vGuard
	vand		vhit,vhit,v0
	vnot		vhit,vhit
	vspltisw	vzero,0
	vcmpequw.	vhit,vhit,vzero

	li  	r0,save_vrs			# restore vector registers
	lvx 	v20,sp,r0
	li  	r0,save_vrs+0x10
	lvx 	v21,sp,r0
	li  	r0,save_vrs+0x20
	lvx 	v22,sp,r0
	li  	r0,save_vrs+0x30
	lvx 	v23,sp,r0
	li  	r0,save_vrs+0x40
	lvx 	v24,sp,r0
	li  	r0,save_vrs+0x50
	lvx 	v25,sp,r0
	li  	r0,save_vrs+0x60
	lvx 	v26,sp,r0
	li  	r0,save_vrs+0x70
	lvx 	v27,sp,r0
	li  	r0,save_vrs+0x80
	lvx 	v28,sp,r0
	li  	r0,save_vrs+0x90
	lvx 	v29,sp,r0
	li  	r0,save_vrs+0xa0
	lvx 	v30,sp,r0
	li  	r0,save_vrs+0xb0
	lvx 	v31,sp,r0

	lwz		sp,new_sp(sp)				# restor the old stack frame pointer
	ENDWITH #New_Frame
	WITH Old_Frame	; base register sp
	lwz 	workptr,work_ptr(sp)
	WITH RC5UnitWork
	li		tmpx,L0_hi
	stwbrx	key1,tmpx,workptr	# save last key in work record
	li  	tmpx,L0_lo
	stwbrx	key0,tmpx,workptr
	ENDWITH	#RC5UnitWork
	lwz		r3,iterations(sp)		# NOTE: r3 was workptr
	sub		r3,r3,iter;	return number of keys processed

	blt  	cr6,@2	# (compare was way back)
	li  	r3,-1	# Error, vector register was clobbered
@2:
	mfspr	r0,vrsave
	cmpwi	r0,-1
	beq		@3
	li  	r3,-1	# Error, vrsave has been changed
@3:
	lwz		r0,save_vrsave(sp)
	mtspr	vrsave,r0
	lwz		r0,save_lr(sp)
	mtlr	r0
	lmw 	r13,save_gprs(sp)	# saved gp register
	ENDWITH #Old_Frame
	blr

	END
	
