;rc5_7450.s -- Optimized for the Velocity Engine of the G4 PowerPC

	EXPORT	crunch_vec_7450[DS]
	EXPORT	.crunch_vec_7450[PR]

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
;			ds.l	1	; used only after C_0 is matched
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
S1_0		ds.l	1		; load for vcon
S1_1		ds.l	1
key_0		ds.l	1		; 	(keys used for vector safety check only)
key_1		ds.l	1
P_0 		ds.l	1		; load for vtxt
P_1 		ds.l	1
C_0 		ds.l	1
C_1 		ds.l	1
; Transfer integer registers to vectors
L_0	 		ds vec_data		; rvec1(sp) 
L_1	 		ds vec_data		; rvec1,sp10x
tS0r 		ds.l	1		; rvec1,sp20x
tQr	 		ds.l	1
tL0 		ds.l	1
tS1 		ds.l	1
	ORG L_0					; Reuse space to save index registers
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
	ORG L_0
S1_14		ds vec_data		; rvec1(sp)
S1_15		ds vec_data		; rvec1,sp10x
S1_16		ds vec_data
S1_17		ds vec_data
S1_18		ds vec_data		; rvec2,sp
S1_19		ds vec_data
S1_20		ds vec_data
S1_21		ds vec_data
S1_22		ds vec_data		; rvec3(sp)
S1_23		ds vec_data
S1_24		ds vec_data
S1_25		ds vec_data
	
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
		tc	crunch_vec_7450[TC],crunch_vec_7450[DS]
	
	CSECT	crunch_vec_7450[DS]
		dc.l	.crunch_vec_7450[PR]
		dc.l	TOC[tc0]
		dc.l	0
		
	CSECT	.crunch_vec_7450[PR]
	
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
vL0		EQU v26
vL1		EQU v27		; cipher and plain text
vS25t  	EQU v28		; vS25 from round 1
vcon	EQU v29		; loaded constants
vtxt 	EQU v30		; plain and cypher text
val3 	EQU v31		; constant for rotations
vAx 	EQU vS0		; overwrites vS0
vBx 	EQU vS1		; overwrites vS1
vL0t	EQU vS20		; overwrites vS20
vL1t	EQU vS21		; overwrites vS21
vS0t	EQU vS22		; overwrites vS22
vhit	EQU vS23		; overwrites vC
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
; Setup the vector constant and text register
	vspltisw	val3,3
	li			tmpx,P_0
	lvx			vtxt,tmpx,sp
; setup the index registers for addressing vector data
	addi	sp10x,sp,0x10
	addi	sp20x,sp,0x20
	addi	sp30x,sp,0x30
	li		rvec1,S1_2
	li		rvec2,S1_6
	li		rvec3,S1_10
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
cnt	SET	S0r
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
	rotlwi  Sr0,S0r,3		;S1.0 
	add     con0,con0,Sr0	
	rotlw	con0,con0,Sr0
	add     S0r,S0r,Qr
; round 1.1 odd
	add     tmp,S0r,Sr0
	add     tmp,tmp,con0	
	rotlwi  Sr1,tmp,3		; S1.1
	add		con1,con0,Sr1
	add     S0r,S0r,Qr
	add 	con2,S0r,Sr1
	
	stw		S0r,tS0r(SP)		; S0_2 = P+2Q
	stw		Qr,tQr(SP)			; Q
	stw		con0,tL0(SP)		; Lr_0
	stw		Sr1,tS1(SP)			; S1[1] = rotl3(...)
	
	stw		Sr0,S1_0(SP)		; constants for round 2
	stw		Sr1,S1_1(SP)
; load the round 2 vector constants
	li			tmpx,S1_0
	lvx			vcon,tmpx,sp
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
vS0r	SET vS25t		; set temporary registers
vQr 	SET vS0
vtmp	SET	vS25
	lvx			vtmp,rvec1,sp20x		; initial loop constants
	vspltw		vS0r,vtmp,0		; P+2Q
	vspltw		vQr,vtmp,1		; Q
	vspltw		vL0,vtmp,2		; L[0] (1.0)
	vspltw		vS1,vtmp,3 		; S[1] (1.1)
	lvx			vL1,rvec1,sp10x			; key1
; compute vector work through r1.25
	vadduwm		vtmp,vS1,vL0	; finish r1.1					
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1,vL1,vtmp						
	vadduwm		vS2,vS0r,vS1	; r1.2					
	vadduwm		vS2,vS2,vL1						
	vrlw		vS2,vS2,val3
	vadduwm		vtmp,vS2,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS3,vS0r,vS2	; r1.3					
	vadduwm		vS3,vS3,vL0						
	vrlw		vS3,vS3,val3		
	vadduwm		vtmp,vS3,vL0						
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1,vL1,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS4,vS0r,vS3	; r1.4					
	vadduwm		vS4,vS4,vL1						
	vrlw		vS4,vS4,val3		
	vadduwm		vtmp,vS4,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS5,vS0r,vS4	; r1.5					
	vadduwm		vS5,vS5,vL0						
	vrlw		vS5,vS5,val3		
	vadduwm		vtmp,vS5,vL0						
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1,vL1,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS6,vS0r,vS5	; r1.6					
	vadduwm		vS6,vS6,vL1						
	vrlw		vS6,vS6,val3		
	vadduwm		vtmp,vS6,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS7,vS0r,vS6	; r1.7					
	vadduwm		vS7,vS7,vL0						
	vrlw		vS7,vS7,val3		
	vadduwm		vtmp,vS7,vL0						
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1,vL1,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS8,vS0r,vS7	; r1.8					
	vadduwm		vS8,vS8,vL1						
	vrlw		vS8,vS8,val3		
	vadduwm		vtmp,vS8,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS9,vS0r,vS8	; r1.9					
	vadduwm		vS9,vS9,vL0						
	vrlw		vS9,vS9,val3		
	vadduwm		vtmp,vS9,vL0						
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1,vL1,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS10,vS0r,vS9	; r1.10				
	vadduwm		vS10,vS10,vL1						
	vrlw		vS10,vS10,val3		
	vadduwm		vtmp,vS10,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS11,vS0r,vS10	; r1.11					
	vadduwm		vS11,vS11,vL0						
	vrlw		vS11,vS11,val3		
	vadduwm		vtmp,vS11,vL0						
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1,vL1,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS12,vS0r,vS11	; r1.12					
	vadduwm		vS12,vS12,vL1						
	vrlw		vS12,vS12,val3		
	vadduwm		vtmp,vS12,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS13,vS0r,vS12	; r1.13					
	vadduwm		vS13,vS13,vL0						
	vrlw		vS13,vS13,val3		
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
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS16,vS0r,vS15	; r1.16					
	vadduwm		vS16,vS16,vL1						
	vrlw		vS16,vS16,val3		
	vadduwm		vtmp,vS16,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS17,vS0r,vS16	; r1.17					
	vadduwm		vS17,vS17,vL0						
	vrlw		vS17,vS17,val3		
	vadduwm		vtmp,vS17,vL0						
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1,vL1,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS18,vS0r,vS17	; r1.18					
	vadduwm		vS18,vS18,vL1						
	vrlw		vS18,vS18,val3		
	vadduwm		vtmp,vS18,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS19,vS0r,vS18	; r1.19					
	vadduwm		vS19,vS19,vL0						
	vrlw		vS19,vS19,val3		
	vadduwm		vtmp,vS19,vL0						
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1,vL1,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS20,vS0r,vS19	; r1.20				
	vadduwm		vS20,vS20,vL1						
	vrlw		vS20,vS20,val3		
	stvx	vS20,rvec2,sp20x
	vadduwm		vtmp,vS20,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS21,vS0r,vS20	; r1.21					
	vadduwm		vS21,vS21,vL0						
	vrlw		vS21,vS21,val3		
	stvx	vS21,rvec2,sp30x
	vadduwm		vtmp,vS21,vL0						
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1,vL1,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS22,vS0r,vS21	; r1.22					
	vadduwm		vS22,vS22,vL1						
	vrlw		vS22,vS22,val3		
	stvx	vS22,rvec3,sp
	vadduwm		vtmp,vS22,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS23,vS0r,vS22	; r1.23					
	vadduwm		vS23,vS23,vL0						
	vrlw		vS23,vS23,val3		
	stvx	vS23,rvec3,sp10x
	vadduwm		vtmp,vS23,vL0						
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1,vL1,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS24,vS0r,vS23	; r1.24					
	vadduwm		vS24,vS24,vL1						
	vrlw		vS24,vS24,val3		
	stvx	vS24,rvec3,sp20x
	vadduwm		vtmp,vS24,vL1						
	vadduwm		vL0,vL0,vtmp						
	vrlw		vL0t,vL0,vtmp						
		vadduwm	vS0r,vS0r,vQr
	vadduwm		vS25t,vS0r,vS24	; r1.25					
	vadduwm		vS25t,vS25t,vL0t						
	vrlw		vS25t,vS25t,val3
	vadduwm		vtmp,vS25t,vL0t						
	vadduwm		vL1,vL1,vtmp						
	vrlw		vL1t,vL1,vtmp						
	vspltw	vS0t,vcon,0


; increment the key for the next group
	addi	key1,key1,4	
	stw 	key1,L0_1(sp)
;Load the keys for the integer rounds 
	lwbrx 	L1a,0,sp20x	; L0_1
	
;		integer constants
;			con0 = L0 (from round 1.0)
;			con1 = S1_1+con0 (from round 1.1)
;			con2 = S1_1+S0_2 (from round 1.2)
;			con3 = S0_3 = P+3Q
;
;		vector constants
;			vcon.0 = S1.0
;			vcon.1 = S1.1
;	State at beginning of loop:
;		Current cycle computed through round 1.25
;		in vectors vL0t, vL1t, vS0t, vS2-vS19, vS25t,
;		S1_20 through S1_24 in memory (need S1_1 from vcon)
;		Next cycle started in integer registers into round 1.2
;		in registers L1a-d.
;	
	add     con3,S0r,Qr		; S0_3 == con3

	add 	L1a,L1a,con1       		;	*				; round 1.1 odd	
	addis	L1b,L1a,256        		;		+			;
	addis	L1c,L1a,512        		;			+		;
	addis	L1d,L1a,768        		;				+	;
	rotlw	L1a,L1a,con1       		;	r				;
	rotlw	L1b,L1b,con1       		;		r			;
	rotlw	L1c,L1c,con1       		;			r		;
	rotlw	L1d,L1d,con1       		;				r	;
	add 	Sr_a,con2,L1a       	;	+				;	; round 1.2 even
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	add 	Sr_b,con2,L1b      		;		+			;
	add 	tmp,Sr_a,L1a       		;	+^				;
	add 	L0a,con0,tmp       		;	+|				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	rotlw	L0a,L0a,tmp        		;	rv				;
	add 	tmp,Sr_b,L1b       		;		+^			;
	add 	Sr_c,con2,L1c      		;			+		;
	add 	L0b,con0,tmp       		;		+|			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;

LOOP:	                    		;	a	b	c	d	;
vtmp	SET	vS25		; set temporary register

	vspltw	vS1,vcon,1
	vadduwm		vS0,vS0t,vS25t	; r2.0						// vS0t, vS25t used here

	rotlw	L0b,L0b,tmp        		;		rv			;
	add 	tmp,Sr_c,L1c       		;			+^		;
	vadduwm		vS0,vS0,vL1t	;							// vL1t used here

	add 	Sr_d,con2,L1d      		;				+	;
	add 	L0c,con0,tmp       		;			+|		;
	vrlw		vS0,vS0,val3

	rotlwi	Sr_d,Sr_d,3        		;				r	;
	rotlw	L0c,L0c,tmp        		;			rv		;
	vadduwm		vtmp,vS0,vL1t

	stw 	Sr_a,S1_2.a(sp)    		;	s				;
	add 	tmp,Sr_d,L1d       		;				+^	;
	vadduwm		vL0,vtmp,vL0t	;							// vL0t used here

	stw 	Sr_b,S1_2.b(sp)    		;		s			;
	add 	L0d,con0,tmp       		;				+|	;
	vrlw		vL0,vL0,vtmp

	stw 	Sr_c,S1_2.c(sp)    		;			s		;
	rotlw	L0d,L0d,tmp        		;				rv	;
	vadduwm		vS1,vS1,vS0	; r2.1							// vS1 used here

	stw 	Sr_d,S1_2.d(sp)    		;				s	;
	add 	Sr_a,con3,Sr_a     		;	+				;	; round 1.3 odd (add con3)
	vadduwm		vS1,vS1,vL0

	add 	Sr_b,con3,Sr_b     		;		+			;
	add 	Sr_a,Sr_a,L0a      		;	+				;
	vrlw		vS1,vS1,val3

	add 	Sr_b,Sr_b,L0b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vtmp,vS1,vL0

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L0a       		;	+^				;
	vadduwm		vL1,vtmp,vL1t

	add 	Sr_c,con3,Sr_c     		;			+		;
	add 	L1a,L1a,tmp        		;	+|				;
	vrlw		vL1,vL1,vtmp

	add 	Sr_c,Sr_c,L0c      		;			+		;
	rotlw	L1a,L1a,tmp        		;	rv				;
	vadduwm		vS2,vS2,vS1	; r2.2

	add 	tmp,Sr_b,L0b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vadduwm		vS2,vS2,vL1

	add 	L1b,L1b,tmp        		;		+|			;
	add 	Sr_d,con3,Sr_d     		;				+	;
	vrlw		vS2,vS2,val3

	rotlw	L1b,L1b,tmp        		;		rv			;
	add 	tmp,Sr_c,L0c       		;			+^		;
	vadduwm		vtmp,vS2,vL1

	add 	Sr_d,Sr_d,L0d      		;				+	;
	add 	L1c,L1c,tmp        		;			+|		;
	vadduwm		vL0,vtmp,vL0

	rotlwi	Sr_d,Sr_d,3        		;				r	;
	rotlw	L1c,L1c,tmp        		;			rv		;
	vrlw		vL0,vL0,vtmp

	stw 	Sr_a,S1_3.a(sp)    		;	s				;
	add 	tmp,Sr_d,L0d       		;				+^	;
	vadduwm		vS3,vS3,vS2	; r2.3

	stw 	Sr_b,S1_3.b(sp)    		;		s			;
	add 	L1d,L1d,tmp        		;				+|	;
	vadduwm		vS3,vS3,vL0

	stw 	Sr_c,S1_3.c(sp)    		;			s		;
	add 	S0r,con3,Qr				;	*				; Set S0r for round 1.4
	vrlw		vS3,vS3,val3

	stw 	Sr_d,S1_3.d(sp)    		;				s	;
	rotlw	L1d,L1d,tmp        		;				rv	;
	vadduwm		vtmp,vS3,vL0

	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.4 even
	add 	Sr_b,S0r,Sr_b      		;		+			;
	vadduwm		vL1,vtmp,vL1

	add 	Sr_a,Sr_a,L1a      		;	+				;
	add 	Sr_b,Sr_b,L1b      		;		+			;
	vrlw		vL1,vL1,vtmp

	rotlwi	Sr_a,Sr_a,3        		;	r				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	vadduwm		vS4,vS4,vS3	; r2.4

	add 	tmp,Sr_a,L1a       		;	+^				;
	add 	Sr_c,S0r,Sr_c      		;			+		;
	vadduwm		vS4,vS4,vL1

	add 	L0a,L0a,tmp        		;	+|				;
	add 	Sr_c,Sr_c,L1c      		;			+		;
	vrlw		vS4,vS4,val3

	rotlw	L0a,L0a,tmp        		;	rv				;
	add 	tmp,Sr_b,L1b       		;		+^			;
	vadduwm		vtmp,vS4,vL1

	rotlwi	Sr_c,Sr_c,3        		;			r		;
	add 	L0b,L0b,tmp        		;		+|			;
	vadduwm		vL0,vtmp,vL0

	add 	Sr_d,S0r,Sr_d      		;				+	;
	rotlw	L0b,L0b,tmp        		;		rv			;
	vrlw		vL0,vL0,vtmp

	add 	tmp,Sr_c,L1c       		;			+^		;
	add 	Sr_d,Sr_d,L1d      		;				+	;
	vadduwm		vS5,vS5,vS4	; r2.5

	add 	L0c,L0c,tmp        		;			+|		;
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vadduwm		vS5,vS5,vL0

	add 	S0r,S0r,Qr          	;	*				;
	rotlw	L0c,L0c,tmp        		;			rv		;
	vrlw		vS5,vS5,val3

	stw 	Sr_a,S1_4.a(sp)    		;	s				;
	add 	tmp,Sr_d,L1d       		;				+^	;
	vadduwm		vtmp,vS5,vL0

	stw 	Sr_b,S1_4.b(sp)    		;		s			;
	add 	L0d,L0d,tmp        		;				+|	;
	vadduwm		vL1,vtmp,vL1

	stw 	Sr_c,S1_4.c(sp)    		;			s		;
	rotlw	L0d,L0d,tmp        		;				rv	;
	vrlw		vL1,vL1,vtmp

	stw 	Sr_d,S1_4.d(sp)    		;				s	;
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.5 odd
	vadduwm		vS6,vS6,vS5	; r2.6

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L0a      		;	+				;
	vadduwm		vS6,vS6,vL1

	add 	Sr_b,Sr_b,L0b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vrlw		vS6,vS6,val3

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L0a       		;	+^				;
	vadduwm		vtmp,vS6,vL1

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L1a,L1a,tmp        		;	+|				;
	vadduwm		vL0,vtmp,vL0

	add 	Sr_c,Sr_c,L0c      		;			+		;
	rotlw	L1a,L1a,tmp        		;	rv				;
	vrlw		vL0,vL0,vtmp

	add 	tmp,Sr_b,L0b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vadduwm		vS7,vS7,vS6	; r2.7

	add 	L1b,L1b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vS7,vS7,vL0

	rotlw	L1b,L1b,tmp        		;		rv			;
	add 	tmp,Sr_c,L0c       		;			+^		;
	vrlw		vS7,vS7,val3

	add 	Sr_d,Sr_d,L0d      		;				+	;
	add 	L1c,L1c,tmp        		;			+|		;
	vadduwm		vtmp,vS7,vL0

	rotlwi	Sr_d,Sr_d,3        		;				r	;
	rotlw	L1c,L1c,tmp        		;			rv		;
	vadduwm		vL1,vtmp,vL1

	stw 	Sr_a,S1_5.a(sp)    		;	s				;
	add 	tmp,Sr_d,L0d       		;				+^	;
	vrlw		vL1,vL1,vtmp

	stw 	Sr_b,S1_5.b(sp)    		;		s			;
	add 	L1d,L1d,tmp        		;				+|	;
	vadduwm		vS8,vS8,vS7	; r2.8

	stw 	Sr_c,S1_5.c(sp)    		;			s		;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vS8,vS8,vL1

	stw 	Sr_d,S1_5.d(sp)    		;				s	;
	rotlw	L1d,L1d,tmp        		;				rv	;
	vrlw		vS8,vS8,val3

	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.6 even
	add 	Sr_b,S0r,Sr_b      		;		+			;
	vadduwm		vtmp,vS8,vL1

	add 	Sr_a,Sr_a,L1a      		;	+				;
	add 	Sr_b,Sr_b,L1b      		;		+			;
	vadduwm		vL0,vtmp,vL0

	rotlwi	Sr_a,Sr_a,3        		;	r				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	vrlw		vL0,vL0,vtmp

	add 	tmp,Sr_a,L1a       		;	+^				;
	add 	Sr_c,S0r,Sr_c      		;			+		;
	vadduwm		vS9,vS9,vS8	; r2.9

	add 	L0a,L0a,tmp        		;	+|				;
	add 	Sr_c,Sr_c,L1c      		;			+		;
	vadduwm		vS9,vS9,vL0

	rotlw	L0a,L0a,tmp        		;	rv				;
	add 	tmp,Sr_b,L1b       		;		+^			;
	vrlw		vS9,vS9,val3

	rotlwi	Sr_c,Sr_c,3        		;			r		;
	add 	L0b,L0b,tmp        		;		+|			;
	vadduwm		vtmp,vS9,vL0

	add 	Sr_d,S0r,Sr_d      		;				+	;
	rotlw	L0b,L0b,tmp        		;		rv			;
	vadduwm		vL1,vtmp,vL1

	add 	tmp,Sr_c,L1c       		;			+^		;
	add 	Sr_d,Sr_d,L1d      		;				+	;
	vrlw		vL1,vL1,vtmp

	add 	L0c,L0c,tmp        		;			+|		;
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vadduwm		vS10,vS10,vS9	; r2.10

	add 	S0r,S0r,Qr          	;	*				;
	rotlw	L0c,L0c,tmp        		;			rv		;
	vadduwm		vS10,vS10,vL1

	stw 	Sr_a,S1_6.a(sp)    		;	s				;
	add 	tmp,Sr_d,L1d       		;				+^	;
	vrlw		vS10,vS10,val3

	stw 	Sr_b,S1_6.b(sp)    		;		s			;
	add 	L0d,L0d,tmp        		;				+|	;
	vadduwm		vtmp,vS10,vL1

	stw 	Sr_c,S1_6.c(sp)    		;			s		;
	rotlw	L0d,L0d,tmp        		;				rv	;
	vadduwm		vL0,vtmp,vL0

	stw 	Sr_d,S1_6.d(sp)    		;				s	;
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.7 odd
	vrlw		vL0,vL0,vtmp

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L0a      		;	+				;
	vadduwm		vS11,vS11,vS10	; r2.11

	add 	Sr_b,Sr_b,L0b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vS11,vS11,vL0

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L0a       		;	+^				;
	vrlw		vS11,vS11,val3

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L1a,L1a,tmp        		;	+|				;
	vadduwm		vtmp,vS11,vL0

	add 	Sr_c,Sr_c,L0c      		;			+		;
	rotlw	L1a,L1a,tmp        		;	rv				;
	vadduwm		vL1,vtmp,vL1

	add 	tmp,Sr_b,L0b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vrlw		vL1,vL1,vtmp

	add 	L1b,L1b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vS12,vS12,vS11	; r2.12

	rotlw	L1b,L1b,tmp        		;		rv			;
	add 	tmp,Sr_c,L0c       		;			+^		;
	vadduwm		vS12,vS12,vL1

	add 	Sr_d,Sr_d,L0d      		;				+	;
	add 	L1c,L1c,tmp        		;			+|		;
	vrlw		vS12,vS12,val3

	rotlwi	Sr_d,Sr_d,3        		;				r	;
	rotlw	L1c,L1c,tmp        		;			rv		;
	vadduwm		vtmp,vS12,vL1

	stw 	Sr_a,S1_7.a(sp)    		;	s				;
	add 	tmp,Sr_d,L0d       		;				+^	;
	vadduwm		vL0,vtmp,vL0

	stw 	Sr_b,S1_7.b(sp)    		;		s			;
	add 	L1d,L1d,tmp        		;				+|	;
	vrlw		vL0,vL0,vtmp

	stw 	Sr_c,S1_7.c(sp)    		;			s		;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vS13,vS13,vS12	; r2.13

	stw 	Sr_d,S1_7.d(sp)    		;				s	;
	rotlw	L1d,L1d,tmp        		;				rv	;
	vadduwm		vS13,vS13,vL0

	lvx	vS20,rvec2,sp20x	; vS20 loaded !!!!
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.8 even
	vrlw		vS13,vS13,val3

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L1a      		;	+				;
	vadduwm		vtmp,vS13,vL0

	add 	Sr_b,Sr_b,L1b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vL1,vtmp,vL1

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L1a       		;	+^				;
	vrlw		vL1,vL1,vtmp

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L0a,L0a,tmp        		;	+|				;
	vadduwm		vS14,vS14,vS13	; r2.14

	add 	Sr_c,Sr_c,L1c      		;			+		;
	rotlw	L0a,L0a,tmp        		;	rv				;
	vadduwm		vS14,vS14,vL1

	add 	tmp,Sr_b,L1b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vrlw		vS14,vS14,val3

	add 	L0b,L0b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vtmp,vS14,vL1

	rotlw	L0b,L0b,tmp        		;		rv			;
	add 	tmp,Sr_c,L1c       		;			+^		;
	vadduwm		vL0,vtmp,vL0

	add 	Sr_d,Sr_d,L1d      		;				+	;
	add 	L0c,L0c,tmp        		;			+|		;
	vrlw		vL0,vL0,vtmp

	rotlwi	Sr_d,Sr_d,3        		;				r	;
	rotlw	L0c,L0c,tmp        		;			rv		;
	vadduwm		vS15,vS15,vS14	; r2.15

	stw 	Sr_a,S1_8.a(sp)    		;	s				;	; Load vS20 before here
	add 	tmp,Sr_d,L1d       		;				+^	;
	vadduwm		vS15,vS15,vL0

	stw 	Sr_b,S1_8.b(sp)    		;		s			;
	add 	L0d,L0d,tmp        		;				+|	;
	vrlw		vS15,vS15,val3

	stw 	Sr_c,S1_8.c(sp)    		;			s		;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vtmp,vS15,vL0

	stw 	Sr_d,S1_8.d(sp)    		;				s	;
	rotlw	L0d,L0d,tmp        		;				rv	;
	vadduwm		vL1,vtmp,vL1

	lvx	vS21,rvec2,sp30x	; vS21 loaded !!!!
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.9 odd
	vrlw		vL1,vL1,vtmp

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L0a      		;	+				;
	vadduwm		vS16,vS16,vS15	; r2.16

	add 	Sr_b,Sr_b,L0b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vS16,vS16,vL1

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L0a       		;	+^				;
	vrlw		vS16,vS16,val3

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L1a,L1a,tmp        		;	+|				;
	vadduwm		vtmp,vS16,vL1

	add 	Sr_c,Sr_c,L0c      		;			+		;
	rotlw	L1a,L1a,tmp        		;	rv				;
	vadduwm		vL0,vtmp,vL0

	add 	tmp,Sr_b,L0b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vrlw		vL0,vL0,vtmp

	add 	L1b,L1b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vS17,vS17,vS16	; r2.17

	rotlw	L1b,L1b,tmp        		;		rv			;
	add 	tmp,Sr_c,L0c       		;			+^		;
	vadduwm		vS17,vS17,vL0

	add 	Sr_d,Sr_d,L0d      		;				+	;
	add 	L1c,L1c,tmp        		;			+|		;
	vrlw		vS17,vS17,val3

	rotlwi	Sr_d,Sr_d,3        		;				r	;
	rotlw	L1c,L1c,tmp        		;			rv		;
	vadduwm		vtmp,vS17,vL0

	stw 	Sr_a,S1_9.a(sp)    		;	s				;	; Load vS21 before here
	add 	tmp,Sr_d,L0d       		;				+^	;
	vadduwm		vL1,vtmp,vL1

	stw 	Sr_b,S1_9.b(sp)    		;		s			;
	add 	L1d,L1d,tmp        		;				+|	;
	vrlw		vL1,vL1,vtmp

	stw 	Sr_c,S1_9.c(sp)    		;			s		;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vS18,vS18,vS17	; r2.18

	stw 	Sr_d,S1_9.d(sp)    		;				s	;
	rotlw	L1d,L1d,tmp        		;				rv	;
	vadduwm		vS18,vS18,vL1

	lvx	vS22,rvec3,sp	; vS22 loaded
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.10 even
	vrlw		vS18,vS18,val3

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L1a      		;	+				;
	vadduwm		vtmp,vS18,vL1

	add 	Sr_b,Sr_b,L1b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vL0,vtmp,vL0

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L1a       		;	+^				;
	vrlw		vL0,vL0,vtmp

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L0a,L0a,tmp        		;	+|				;
	vadduwm		vS19,vS19,vS18	; r2.19

	add 	Sr_c,Sr_c,L1c      		;			+		;
	rotlw	L0a,L0a,tmp        		;	rv				;
	vadduwm		vS19,vS19,vL0

	add 	tmp,Sr_b,L1b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vrlw		vS19,vS19,val3

	add 	L0b,L0b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vtmp,vS19,vL0

	rotlw	L0b,L0b,tmp        		;		rv			;
	add 	tmp,Sr_c,L1c       		;			+^		;
	vadduwm		vL1,vtmp,vL1

	add 	Sr_d,Sr_d,L1d      		;				+	;
	add 	L0c,L0c,tmp        		;			+|		;
	vrlw		vL1,vL1,vtmp

	rotlwi	Sr_d,Sr_d,3        		;				r	;
	rotlw	L0c,L0c,tmp        		;			rv		;
	vadduwm		vS20,vS20,vS19	; r2.20

	stw 	Sr_a,S1_10.a(sp)   		;	s				;	; Load vS22 before here
	add 	tmp,Sr_d,L1d       		;				+^	;
	vadduwm		vS20,vS20,vL1

	stw 	Sr_b,S1_10.b(sp)   		;		s			;
	add 	L0d,L0d,tmp        		;				+|	;
	vrlw		vS20,vS20,val3

	stw 	Sr_c,S1_10.c(sp)   		;			s		;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vtmp,vS20,vL1

	stw 	Sr_d,S1_10.d(sp)   		;				s	;
	rotlw	L0d,L0d,tmp        		;				rv	;
	vadduwm		vL0,vtmp,vL0

	lvx	vS23,rvec3,sp10x	; vS23 loaded
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.11 odd
	vrlw		vL0,vL0,vtmp

	lvx	vS24,rvec3,sp20x
	add 	Sr_b,S0r,Sr_b      		;		+			;
	vadduwm		vS21,vS21,vS20	; r2.21

	add 	Sr_a,Sr_a,L0a      		;	+				;
	add 	Sr_b,Sr_b,L0b      		;		+			;
	vadduwm		vS21,vS21,vL0

	rotlwi	Sr_a,Sr_a,3        		;	r				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	vrlw		vS21,vS21,val3

	add 	tmp,Sr_a,L0a       		;	+^				;
	add 	Sr_c,S0r,Sr_c      		;			+		;
	vadduwm		vtmp,vS21,vL0

	add 	L1a,L1a,tmp        		;	+|				;
	add 	Sr_c,Sr_c,L0c      		;			+		;
	vadduwm		vL1,vtmp,vL1

	rotlw	L1a,L1a,tmp        		;	rv				;
	add 	tmp,Sr_b,L0b       		;		+^			;
	vrlw		vL1,vL1,vtmp

	rotlwi	Sr_c,Sr_c,3        		;			r		;
	add 	L1b,L1b,tmp        		;		+|			;
	vadduwm		vS22,vS22,vS21	; r2.22

	add 	Sr_d,S0r,Sr_d      		;				+	;
	rotlw	L1b,L1b,tmp        		;		rv			;
	vadduwm		vS22,vS22,vL1

	add 	tmp,Sr_c,L0c       		;			+^		;
	add 	Sr_d,Sr_d,L0d      		;				+	;
	vrlw		vS22,vS22,val3

	add 	L1c,L1c,tmp        		;			+|		;
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vadduwm		vtmp,vS22,vL1

	add 	S0r,S0r,Qr          	;	*				;
	rotlw	L1c,L1c,tmp        		;			rv		;
	vadduwm		vL0,vtmp,vL0

	stw 	Sr_a,S1_11.a(sp)   		;	s				;	; Load vS23 before here
	add 	tmp,Sr_d,L0d       		;				+^	;
	vrlw		vL0,vL0,vtmp

	stw 	Sr_b,S1_11.b(sp)   		;		s			;
	add 	L1d,L1d,tmp        		;				+|	;
	vadduwm		vS23,vS23,vS22	; r2.23

	stw 	Sr_c,S1_11.c(sp)   		;			s		;
	rotlw	L1d,L1d,tmp        		;				rv	;
	vadduwm		vS23,vS23,vL0

	stw 	Sr_d,S1_11.d(sp)   		;				s	;
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.12 even
	vrlw		vS23,vS23,val3

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L1a      		;	+				;
	vadduwm		vtmp,vS23,vL0

	add 	Sr_b,Sr_b,L1b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vL1,vtmp,vL1

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L1a       		;	+^				;
	vrlw		vL1,vL1,vtmp

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L0a,L0a,tmp        		;	+|				;
	vadduwm		vS24,vS24,vS23	; r2.24			need vS24

	add 	Sr_c,Sr_c,L1c      		;			+		;
	rotlw	L0a,L0a,tmp        		;	rv				;
	vadduwm		vS24,vS24,vL1

	add 	tmp,Sr_b,L1b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vrlw		vS24,vS24,val3

	add 	L0b,L0b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vtmp,vS24,vL1

	rotlw	L0b,L0b,tmp        		;		rv			;
	add 	tmp,Sr_c,L1c       		;			+^		;
	vadduwm		vL0,vtmp,vL0

	add 	Sr_d,Sr_d,L1d      		;				+	;
	add 	L0c,L0c,tmp        		;			+|		;
	vrlw		vL0,vL0,vtmp

	rotlwi	Sr_d,Sr_d,3        		;				r	;
	rotlw	L0c,L0c,tmp        		;			rv		;
	vadduwm		vS25,vS25t,vS24	; r2.25
vtmp	SET	vS25t		; switch to alternate temporary
	stw 	Sr_a,S1_12.a(sp)   		;	s				;	; Load vS24 before here
	add 	tmp,Sr_d,L1d       		;				+^	;
	vadduwm		vS25,vS25,vL0

	stw 	Sr_b,S1_12.b(sp)   		;		s			;
	add 	L0d,L0d,tmp        		;				+|	;
	vrlw		vS25,vS25,val3

	stw 	Sr_c,S1_12.c(sp)   		;			s		;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vtmp,vS25,vL0

	stw 	Sr_d,S1_12.d(sp)   		;				s	;
	rotlw	L0d,L0d,tmp        		;				rv	;
	vadduwm		vL1,vtmp,vL1

	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.13 odd
	add 	Sr_b,S0r,Sr_b      		;		+			;
	vrlw		vL1,vL1,vtmp

	add 	Sr_a,Sr_a,L0a      		;	+				;
	add 	Sr_b,Sr_b,L0b      		;		+			;
	vadduwm		vS0,vS0,vS25	; r3.0

	rotlwi	Sr_a,Sr_a,3        		;	r				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	vadduwm		vS0,vS0,vL1

	add 	tmp,Sr_a,L0a       		;	+^				;
	add 	Sr_c,S0r,Sr_c      		;			+		;
	vrlw		vS0,vS0,val3

	add 	L1a,L1a,tmp        		;	+|				;
	add 	Sr_c,Sr_c,L0c      		;			+		;
	vadduwm		vtmp,vS0,vL1

	rotlw	L1a,L1a,tmp        		;	rv				;
	add 	tmp,Sr_b,L0b       		;		+^			;
	vadduwm		vL0,vtmp,vL0

	rotlwi	Sr_c,Sr_c,3        		;			r		;
	add 	L1b,L1b,tmp        		;		+|			;
	vrlw		vL0,vL0,vtmp				;vtmp free

	vspltw	vtmp,vtxt,0 	;P_0(SP)
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vS1,vS1,vS0	; r3.1

	rotlw	L1b,L1b,tmp        		;		rv			;
	add 	tmp,Sr_c,L0c       		;			+^		;
	vadduwm		vS1,vS1,vL0

	add 	Sr_d,Sr_d,L0d      		;				+	;
	add 	L1c,L1c,tmp        		;			+|		;
	vrlw		vS1,vS1,val3

	rotlwi	Sr_d,Sr_d,3        		;				r	;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vAx,vtmp,vS0	; vAx reuses vS0			// need P_0 in vtmp

	stw 	Sr_a,S1_13.a(sp)   		;	s				;
	rotlw	L1c,L1c,tmp        		;			rv		;
	vadduwm		vtmp,vS1,vL0

	stw 	Sr_b,S1_13.b(sp)   		;		s			;
	add 	tmp,Sr_d,L0d       		;				+^	;
	vadduwm		vL1,vtmp,vL1

	stw 	Sr_c,S1_13.c(sp)   		;			s		;
	add 	L1d,L1d,tmp        		;				+|	;
	vrlw		vL1,vL1,vtmp				;vtmp free

	stw 	Sr_d,S1_13.d(sp)   		;				s	;
	rotlw	L1d,L1d,tmp        		;				rv	;
	vadduwm		vS2,vS2,vS1	; r3.2

	vspltw	vtmp,vtxt,1	;P_1
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.14 even
	vadduwm		vS2,vS2,vL1

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L1a      		;	+				;
	vrlw		vS2,vS2,val3

	add 	Sr_b,Sr_b,L1b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vBx,vtmp,vS1	; vBx reuses vS1			// need P_1 in vtmp

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L1a       		;	+^				;
	vadduwm		vtmp,vS2,vL1

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L0a,L0a,tmp        		;	+|				;
	vadduwm		vL0,vtmp,vL0

	add 	Sr_c,Sr_c,L1c      		;			+		;
	rotlw	L0a,L0a,tmp        		;	rv				;
	vrlw		vL0,vL0,vtmp

	add 	tmp,Sr_b,L1b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vxor 		vAx,vAx,vBx

	add 	L0b,L0b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vrlw		vAx,vAx,vBx

	rotlw	L0b,L0b,tmp        		;		rv			;
	add 	tmp,Sr_c,L1c       		;			+^		;
	vadduwm		vAx,vAx,vS2

	add 	Sr_d,Sr_d,L1d      		;				+	;
	add 	L0c,L0c,tmp        		;			+|		;
	vadduwm		vS3,vS3,vS2	; r3.3			vS2 free

	rotlwi	Sr_d,Sr_d,3        		;				r	;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vS3,vS3,vL0		; At this point S1_2 through S1_13 are stored

	lvx	vS2,rvec1,sp	; S1_2 is loaded here so the storage can be reused
	rotlw	L0c,L0c,tmp        		;			rv		;
	vrlw		vS3,vS3,val3

	stw 	Sr_a,S1_14.a(sp)   		;	s				;	; Overwrites vS2
	add 	tmp,Sr_d,L1d       		;				+^	;
	vadduwm		vtmp,vS3,vL0

	stw 	Sr_b,S1_14.b(sp)   		;		s			;
	add 	L0d,L0d,tmp        		;				+|	;
	vadduwm		vL1,vtmp,vL1

	stw 	Sr_c,S1_14.c(sp)   		;			s		;
	rotlw	L0d,L0d,tmp        		;				rv	;
	vrlw		vL1,vL1,vtmp

	stw 	Sr_d,S1_14.d(sp)   		;				s	;
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.15 odd
	vxor 		vBx,vBx,vAx

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L0a      		;	+				;
	vrlw		vBx,vBx,vAx

	add 	Sr_b,Sr_b,L0b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vBx,vBx,vS3

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L0a       		;	+^				;
	vadduwm		vS4,vS4,vS3	; r3.4			vS3 free

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L1a,L1a,tmp        		;	+|				;
	vadduwm		vS4,vS4,vL1

	add 	Sr_c,Sr_c,L0c      		;			+		;
	rotlw	L1a,L1a,tmp        		;	rv				;
	vrlw		vS4,vS4,val3

	add 	tmp,Sr_b,L0b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vadduwm		vtmp,vS4,vL1

	add 	L1b,L1b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vL0,vtmp,vL0

	rotlw	L1b,L1b,tmp        		;		rv			;
	add 	tmp,Sr_c,L0c       		;			+^		;
	vrlw		vL0,vL0,vtmp

	add 	Sr_d,Sr_d,L0d      		;				+	;
	add 	L1c,L1c,tmp        		;			+|		;
	vxor 		vAx,vAx,vBx

	lvx	vS3,rvec1,sp10x
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vrlw		vAx,vAx,vBx

	stw 	Sr_a,S1_15.a(sp)   		;	s				;	; Overwrites vS3
	rotlw	L1c,L1c,tmp        		;			rv		;
	vadduwm		vAx,vAx,vS4

	stw 	Sr_b,S1_15.b(sp)   		;		s			;
	add 	tmp,Sr_d,L0d       		;				+^	;
	vadduwm		vS5,vS5,vS4	; r3.5			vS4 free

	stw 	Sr_c,S1_15.c(sp)   		;			s		;
	add 	L1d,L1d,tmp        		;				+|	;
	vadduwm		vS5,vS5,vL0

	stw 	Sr_d,S1_15.d(sp)   		;				s	;
	add 	S0r,S0r,Qr          	;	*				;
	vrlw		vS5,vS5,val3

	lvx	vS4,rvec1,sp20x
	rotlw	L1d,L1d,tmp        		;				rv	;
	vadduwm		vtmp,vS5,vL0

	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.16 even
	add 	Sr_b,S0r,Sr_b      		;		+			;
	vadduwm		vL1,vtmp,vL1

	add 	Sr_a,Sr_a,L1a      		;	+				;
	add 	Sr_b,Sr_b,L1b      		;		+			;
	vrlw		vL1,vL1,vtmp

	rotlwi	Sr_a,Sr_a,3        		;	r				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	vxor 		vBx,vBx,vAx

	add 	tmp,Sr_a,L1a       		;	+^				;
	add 	Sr_c,S0r,Sr_c      		;			+		;
	vrlw		vBx,vBx,vAx

	add 	L0a,L0a,tmp        		;	+|				;
	add 	Sr_c,Sr_c,L1c      		;			+		;
	vadduwm		vBx,vBx,vS5

	rotlw	L0a,L0a,tmp        		;	rv				;
	add 	tmp,Sr_b,L1b       		;		+^			;
	vadduwm		vS6,vS6,vS5	; r3.6			vS5 free

	rotlwi	Sr_c,Sr_c,3        		;			r		;
	add 	L0b,L0b,tmp        		;		+|			;
	vadduwm		vS6,vS6,vL1

	add 	Sr_d,S0r,Sr_d      		;				+	;
	rotlw	L0b,L0b,tmp        		;		rv			;
	vrlw		vS6,vS6,val3

	add 	tmp,Sr_c,L1c       		;			+^		;
	add 	Sr_d,Sr_d,L1d      		;				+	;
	vadduwm		vtmp,vS6,vL1

	add 	L0c,L0c,tmp        		;			+|		;
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vadduwm		vL0,vtmp,vL0

	lvx	vS5,rvec1,sp30x
	add 	S0r,S0r,Qr          	;	*				;
	vrlw		vL0,vL0,vtmp

	stw 	Sr_a,S1_16.a(sp)   		;	s				;
	rotlw	L0c,L0c,tmp        		;			rv		;
	vxor 		vAx,vAx,vBx

	stw 	Sr_b,S1_16.b(sp)   		;		s			;
	add 	tmp,Sr_d,L1d       		;				+^	;
	vrlw		vAx,vAx,vBx

	stw 	Sr_c,S1_16.c(sp)   		;			s		;
	add 	L0d,L0d,tmp        		;				+|	;
	vadduwm		vAx,vAx,vS6

	stw 	Sr_d,S1_16.d(sp)   		;				s	;
	rotlw	L0d,L0d,tmp        		;				rv	;
	vadduwm		vS7,vS7,vS6	; r3.7			vS6 free

	lvx	vS6,rvec2,sp
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.17 odd
	vadduwm		vS7,vS7,vL0

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L0a      		;	+				;
	vrlw		vS7,vS7,val3

	add 	Sr_b,Sr_b,L0b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vtmp,vS7,vL0

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L0a       		;	+^				;
	vadduwm		vL1,vtmp,vL1

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L1a,L1a,tmp        		;	+|				;
	vrlw		vL1,vL1,vtmp

	add 	Sr_c,Sr_c,L0c      		;			+		;
	rotlw	L1a,L1a,tmp        		;	rv				;
	vxor 		vBx,vBx,vAx

	add 	tmp,Sr_b,L0b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vrlw		vBx,vBx,vAx

	add 	L1b,L1b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vBx,vBx,vS7

	rotlw	L1b,L1b,tmp        		;		rv			;
	add 	tmp,Sr_c,L0c       		;			+^		;
	vadduwm		vS8,vS8,vS7	; r3.8			vS7 free

	add 	Sr_d,Sr_d,L0d      		;				+	;
	add 	L1c,L1c,tmp        		;			+|		;
	vadduwm		vS8,vS8,vL1

	lvx	vS7,rvec2,sp10x
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vrlw		vS8,vS8,val3

	stw 	Sr_a,S1_17.a(sp)   		;	s				;
	rotlw	L1c,L1c,tmp        		;			rv		;
	vadduwm		vtmp,vS8,vL1

	stw 	Sr_b,S1_17.b(sp)   		;		s			;
	add 	tmp,Sr_d,L0d       		;				+^	;
	vadduwm		vL0,vtmp,vL0

	stw 	Sr_c,S1_17.c(sp)   		;			s		;
	add 	L1d,L1d,tmp        		;				+|	;
	vrlw		vL0,vL0,vtmp

	stw 	Sr_d,S1_17.d(sp)   		;				s	;
	add 	S0r,S0r,Qr          	;	*				;
	vxor 		vAx,vAx,vBx

	rotlw	L1d,L1d,tmp        		;				rv	;
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.18 even
	vrlw		vAx,vAx,vBx

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L1a      		;	+				;
	vadduwm		vAx,vAx,vS8

	add 	Sr_b,Sr_b,L1b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vS9,vS9,vS8	; r3.9			vS8 free

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L1a       		;	+^				;
	vadduwm		vS9,vS9,vL0

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L0a,L0a,tmp        		;	+|				;
	vrlw		vS9,vS9,val3

	add 	Sr_c,Sr_c,L1c      		;			+		;
	rotlw	L0a,L0a,tmp        		;	rv				;
	vadduwm		vtmp,vS9,vL0

	add 	tmp,Sr_b,L1b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vadduwm		vL1,vtmp,vL1

	add 	L0b,L0b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vrlw		vL1,vL1,vtmp

	rotlw	L0b,L0b,tmp        		;		rv			;
	add 	tmp,Sr_c,L1c       		;			+^		;
	vxor 		vBx,vBx,vAx

	add 	Sr_d,Sr_d,L1d      		;				+	;
	add 	L0c,L0c,tmp        		;			+|		;
	vrlw		vBx,vBx,vAx

	lvx	vS8,rvec2,sp20x
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vadduwm		vBx,vBx,vS9

	stw 	Sr_a,S1_18.a(sp)   		;	s
	rotlw	L0c,L0c,tmp        		;			rv		;
	vadduwm		vS10,vS10,vS9	; r3.10			vS9 free

	stw 	Sr_b,S1_18.b(sp)   		;		s			;
	add 	tmp,Sr_d,L1d       		;				+^	;
	vadduwm		vS10,vS10,vL1

	stw 	Sr_c,S1_18.c(sp)   		;			s		;
	add 	L0d,L0d,tmp        		;				+|	;
	vrlw		vS10,vS10,val3

	stw 	Sr_d,S1_18.d(sp)   		;				s	;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vtmp,vS10,vL1

	lvx	vS9,rvec2,sp30x
	rotlw	L0d,L0d,tmp        		;				rv	;
	vadduwm		vL0,vtmp,vL0

	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.19 odd
	add 	Sr_b,S0r,Sr_b      		;		+			;
	vrlw		vL0,vL0,vtmp

	add 	Sr_a,Sr_a,L0a      		;	+				;
	add 	Sr_b,Sr_b,L0b      		;		+			;
	vxor 		vAx,vAx,vBx

	rotlwi	Sr_a,Sr_a,3        		;	r				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	vrlw		vAx,vAx,vBx

	add 	tmp,Sr_a,L0a       		;	+^				;
	add 	Sr_c,S0r,Sr_c      		;			+		;
	vadduwm		vAx,vAx,vS10

	add 	L1a,L1a,tmp        		;	+|				;
	add 	Sr_c,Sr_c,L0c      		;			+		;
	vadduwm		vS11,vS11,vS10	; r3.11			vS10 free

	rotlw	L1a,L1a,tmp        		;	rv				;
	add 	tmp,Sr_b,L0b       		;		+^			;
	vadduwm		vS11,vS11,vL0

	rotlwi	Sr_c,Sr_c,3        		;			r		;
	add 	L1b,L1b,tmp        		;		+|			;
	vrlw		vS11,vS11,val3

	add 	Sr_d,S0r,Sr_d      		;				+	;
	rotlw	L1b,L1b,tmp        		;		rv			;
	vadduwm		vtmp,vS11,vL0

	add 	tmp,Sr_c,L0c       		;			+^		;
	add 	Sr_d,Sr_d,L0d      		;				+	;
	vadduwm		vL1,vtmp,vL1

	add 	L1c,L1c,tmp        		;			+|		;
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vrlw		vL1,vL1,vtmp

	lvx	vS10,rvec3,sp
	add 	S0r,S0r,Qr          	;	*				;
	vxor 		vBx,vBx,vAx

	stw 	Sr_a,S1_19.a(sp)   		;	s				;
	rotlw	L1c,L1c,tmp        		;			rv		;
	vrlw		vBx,vBx,vAx

	stw 	Sr_b,S1_19.b(sp)   		;		s			;
	add 	tmp,Sr_d,L0d       		;				+^	;
	vadduwm		vBx,vBx,vS11

	stw 	Sr_c,S1_19.c(sp)   		;			s		;
	add 	L1d,L1d,tmp        		;				+|	;
	vadduwm		vS12,vS12,vS11	; r3.12			vS11 free

	stw 	Sr_d,S1_19.d(sp)   		;				s	;
	rotlw	L1d,L1d,tmp        		;				rv	;
	vadduwm		vS12,vS12,vL1

	lvx	vS11,rvec3,sp10x
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.20 even
	vrlw		vS12,vS12,val3

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L1a      		;	+				;
	vadduwm		vtmp,vS12,vL1

	add 	Sr_b,Sr_b,L1b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vL0,vtmp,vL0

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L1a       		;	+^				;
	vrlw		vL0,vL0,vtmp

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L0a,L0a,tmp        		;	+|				;
	vxor 		vAx,vAx,vBx

	add 	Sr_c,Sr_c,L1c      		;			+		;
	rotlw	L0a,L0a,tmp        		;	rv				;
	vrlw		vAx,vAx,vBx

	add 	tmp,Sr_b,L1b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vadduwm		vAx,vAx,vS12

	add 	L0b,L0b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vS13,vS13,vS12	; r3.13			vS12 free

	rotlw	L0b,L0b,tmp        		;		rv			;
	add 	tmp,Sr_c,L1c       		;			+^		;
	vadduwm		vS13,vS13,vL0

	add 	Sr_d,Sr_d,L1d      		;				+	;
	add 	L0c,L0c,tmp        		;			+|		;
	vrlw		vS13,vS13,val3

	lvx	vS12,rvec3,sp20x
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vadduwm		vtmp,vS13,vL0

	stw 	Sr_a,S1_20.a(sp)   		;	s				;
	rotlw	L0c,L0c,tmp        		;			rv		;
	vadduwm		vL1,vtmp,vL1

	stw 	Sr_b,S1_20.b(sp)   		;		s			;
	add 	tmp,Sr_d,L1d       		;				+^	;
	vrlw		vL1,vL1,vtmp

	stw 	Sr_c,S1_20.c(sp)   		;			s		;
	add 	L0d,L0d,tmp        		;				+|	;
	vxor 		vBx,vBx,vAx

	stw 	Sr_d,S1_20.d(sp)   		;				s	;
	add 	S0r,S0r,Qr          	;	*				;
	vrlw		vBx,vBx,vAx

	rotlw	L0d,L0d,tmp        		;				rv	;
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.21 odd
	vadduwm		vBx,vBx,vS13

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L0a      		;	+				;
	vadduwm		vS14,vS14,vS13	; r3.14			vS13 free

	add 	Sr_b,Sr_b,L0b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vadduwm		vS14,vS14,vL1

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L0a       		;	+^				;
	vrlw		vS14,vS14,val3

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L1a,L1a,tmp        		;	+|				;
	vadduwm		vtmp,vS14,vL1

	add 	Sr_c,Sr_c,L0c      		;			+		;
	rotlw	L1a,L1a,tmp        		;	rv				;
	vadduwm		vL0,vtmp,vL0

	add 	tmp,Sr_b,L0b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vrlw		vL0,vL0,vtmp

	add 	L1b,L1b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vxor 		vAx,vAx,vBx

	rotlw	L1b,L1b,tmp        		;		rv			;
	add 	tmp,Sr_c,L0c       		;			+^		;
	vrlw		vAx,vAx,vBx

	add 	Sr_d,Sr_d,L0d      		;				+	;
	add 	L1c,L1c,tmp        		;			+|		;
	vadduwm		vAx,vAx,vS14

	lvx	vS13,rvec3,sp30x
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vadduwm		vS15,vS15,vS14	; r3.15			vS14 free

	lvx	vS14,rvec1,sp
	rotlw	L1c,L1c,tmp        		;			rv		;
	vadduwm		vS15,vS15,vL0

	stw 	Sr_a,S1_21.a(sp)   		;	s				;
	add 	tmp,Sr_d,L0d       		;				+^	;
	vrlw		vS15,vS15,val3

	stw 	Sr_b,S1_21.b(sp)   		;		s			;
	add 	L1d,L1d,tmp        		;				+|	;
	vadduwm		vtmp,vS15,vL0

	stw 	Sr_c,S1_21.c(sp)   		;			s		;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vL1,vtmp,vL1

	stw 	Sr_d,S1_21.d(sp)   		;				s	;
	rotlw	L1d,L1d,tmp        		;				rv	;
	vrlw		vL1,vL1,vtmp

	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.22 even
	add 	Sr_b,S0r,Sr_b      		;		+			;
	vxor 		vBx,vBx,vAx

	add 	Sr_a,Sr_a,L1a      		;	+				;
	add 	Sr_b,Sr_b,L1b      		;		+			;
	vrlw		vBx,vBx,vAx

	rotlwi	Sr_a,Sr_a,3        		;	r				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	vadduwm		vBx,vBx,vS15

	add 	tmp,Sr_a,L1a       		;	+^				;
	add 	Sr_c,S0r,Sr_c      		;			+		;
	vadduwm		vS16,vS16,vS15	; r3.16			vS15 free

	add 	L0a,L0a,tmp        		;	+|				;
	add 	Sr_c,Sr_c,L1c      		;			+		;
	vadduwm		vS16,vS16,vL1

	rotlw	L0a,L0a,tmp        		;	rv				;
	add 	tmp,Sr_b,L1b       		;		+^			;
	vrlw		vS16,vS16,val3

	rotlwi	Sr_c,Sr_c,3        		;			r		;
	add 	L0b,L0b,tmp        		;		+|			;
	vadduwm		vtmp,vS16,vL1

	add 	Sr_d,S0r,Sr_d      		;				+	;
	rotlw	L0b,L0b,tmp        		;		rv			;
	vadduwm		vL0,vtmp,vL0

	add 	tmp,Sr_c,L1c       		;			+^		;
	add 	Sr_d,Sr_d,L1d      		;				+	;
	vrlw		vL0,vL0,vtmp

	add 	L0c,L0c,tmp        		;			+|		;
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vxor 		vAx,vAx,vBx

	lvx	vS15,rvec1,sp10x
	add 	S0r,S0r,Qr          	;	*				;
	vrlw		vAx,vAx,vBx

	stw 	Sr_a,S1_22.a(sp)   		;	s				;
	rotlw	L0c,L0c,tmp        		;			rv		;
	vadduwm		vAx,vAx,vS16

	stw 	Sr_b,S1_22.b(sp)   		;		s			;
	add 	tmp,Sr_d,L1d       		;				+^	;
	vadduwm		vS17,vS17,vS16	; r3.17			vS16 free

	stw 	Sr_c,S1_22.c(sp)   		;			s		;
	add 	L0d,L0d,tmp        		;				+|	;
	vadduwm		vS17,vS17,vL0

	stw 	Sr_d,S1_22.d(sp)   		;				s	;
	rotlw	L0d,L0d,tmp        		;				rv	;
	vrlw		vS17,vS17,val3

	lvx	vS16,rvec1,sp20x
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.23 odd
	vadduwm		vtmp,vS17,vL0

	add 	Sr_b,S0r,Sr_b      		;		+			;
	add 	Sr_a,Sr_a,L0a      		;	+				;
	vadduwm		vL1,vtmp,vL1

	add 	Sr_b,Sr_b,L0b      		;		+			;
	rotlwi	Sr_a,Sr_a,3        		;	r				;
	vrlw		vL1,vL1,vtmp

	rotlwi	Sr_b,Sr_b,3        		;		r			;
	add 	tmp,Sr_a,L0a       		;	+^				;
	vxor 		vBx,vBx,vAx

	add 	Sr_c,S0r,Sr_c      		;			+		;
	add 	L1a,L1a,tmp        		;	+|				;
	vrlw		vBx,vBx,vAx

	add 	Sr_c,Sr_c,L0c      		;			+		;
	rotlw	L1a,L1a,tmp        		;	rv				;
	vadduwm		vBx,vBx,vS17

	add 	tmp,Sr_b,L0b       		;		+^			;
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vadduwm		vS18,vS18,vS17	; r3.18			vS17 free

	add 	L1b,L1b,tmp        		;		+|			;
	add 	Sr_d,S0r,Sr_d      		;				+	;
	vadduwm		vS18,vS18,vL1

	rotlw	L1b,L1b,tmp        		;		rv			;
	add 	tmp,Sr_c,L0c       		;			+^		;
	vrlw		vS18,vS18,val3

	add 	Sr_d,Sr_d,L0d      		;				+	;
	add 	L1c,L1c,tmp        		;			+|		;
	vadduwm		vtmp,vS18,vL1

	lvx	vS17,rvec1,sp30x
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vadduwm		vL0,vtmp,vL0

	stw 	Sr_a,S1_23.a(sp)   		;	s				;
	rotlw	L1c,L1c,tmp        		;			rv		;
	vrlw		vL0,vL0,vtmp

	stw 	Sr_b,S1_23.b(sp)   		;		s			;
	add 	tmp,Sr_d,L0d       		;				+^	;
	vxor 		vAx,vAx,vBx

	stw 	Sr_c,S1_23.c(sp)   		;			s		;
	add 	L1d,L1d,tmp        		;				+|	;
	vrlw		vAx,vAx,vBx

	stw 	Sr_d,S1_23.d(sp)   		;				s	;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vAx,vAx,vS18

	rotlw	L1d,L1d,tmp        		;				rv	;
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.24 even
	vadduwm		vS19,vS19,vS18	; r3.19			vS18 free

	addi	key1,key1,4        	;//	;					; Update key for next cycle
	add 	Sr_b,S0r,Sr_b      		;		+			;
	vadduwm		vS19,vS19,vL0

	add 	Sr_a,Sr_a,L1a      		;	+				;
	add 	Sr_b,Sr_b,L1b      		;		+			;
	vrlw		vS19,vS19,val3

	rotlwi	Sr_a,Sr_a,3        		;	r				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	vadduwm		vtmp,vS19,vL0

	add 	tmp,Sr_a,L1a       		;	+^				;
	add 	Sr_c,S0r,Sr_c      		;			+		;
	vadduwm		vL1,vtmp,vL1

	add 	L0a,L0a,tmp        		;	+|				;
	add 	Sr_c,Sr_c,L1c      		;			+		;
	vrlw		vL1,vL1,vtmp

	rotlw	L0a,L0a,tmp        		;	rv				;
	add 	tmp,Sr_b,L1b       		;		+^			;
	vxor 		vBx,vBx,vAx

	rotlwi	Sr_c,Sr_c,3        		;			r		;
	add 	L0b,L0b,tmp        		;		+|			;
	vrlw		vBx,vBx,vAx

	add 	Sr_d,S0r,Sr_d      		;				+	;
	rotlw	L0b,L0b,tmp        		;		rv			;
	vadduwm		vBx,vBx,vS19

	lvx	vS18,rvec2,sp
	add 	tmp,Sr_c,L1c       		;			+^		;
	vadduwm		vS20,vS20,vS19	; r3.20			vS19 free

	lvx	vS19,rvec2,sp10x
	add 	Sr_d,Sr_d,L1d      		;				+	;
	vadduwm		vS20,vS20,vL1

	stw 	key1,L0_1(sp)      	;//	;	s				; Store updated key
	add 	L0c,L0c,tmp        		;			+|		;
	vrlw		vS20,vS20,val3

	stw 	Sr_a,S1_24.a(sp)   		;	s				;
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vadduwm		vtmp,vS20,vL1

	stw 	Sr_b,S1_24.b(sp)   		;		s			;
	add 	S0r,S0r,Qr          	;	*				;
	vadduwm		vL0,vtmp,vL0

	stw 	Sr_c,S1_24.c(sp)   		;			s		;
	rotlw	L0c,L0c,tmp        		;			rv		;
	vrlw		vL0,vL0,vtmp

	stw 	Sr_d,S1_24.d(sp)   		;				s	;
	add 	tmp,Sr_d,L1d       		;				+^	;
	vxor 		vAx,vAx,vBx

	stw 	L0a,L_0.a(sp)      		;	s				;	; Store L0 from 1.24
	add 	L0d,L0d,tmp        		;				+|	;
	vrlw		vAx,vAx,vBx

	stw 	L0b,L_0.b(sp)      		;		s			;
	rotlw	L0d,L0d,tmp        		;				rv	;
	vadduwm		vAx,vAx,vS20

	stw 	L0c,L_0.c(sp)      		;			s		;
	add 	Sr_a,S0r,Sr_a      		;	+				;	; round 1.25 odd
	vadduwm		vS21,vS21,vS20	; r3.21

	stw 	L0d,L_0.d(sp)      		;				s	; // only 2 cycles from data!
	add 	Sr_b,S0r,Sr_b      		;		+			;
	vadduwm		vS21,vS21,vL0	; Load L0t after here

	add 	Sr_a,Sr_a,L0a      		;	+				;
	add 	Sr_b,Sr_b,L0b      		;		+			;
	vrlw		vS21,vS21,val3

	rotlwi	Sr_a,Sr_a,3        		;	r				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	vadduwm		vtmp,vS21,vL0

	add 	tmp,Sr_a,L0a       		;	+^				; // L0a is free
	add 	Sr_c,S0r,Sr_c      		;			+		;
	vadduwm		vL1,vtmp,vL1

	add 	L1a,L1a,tmp        		;	+|				;
	add 	Sr_c,Sr_c,L0c      		;			+		;
	vrlw		vL1,vL1,vtmp

	rotlw	L1a,L1a,tmp        		;	rv				;
	add 	tmp,Sr_b,L0b       		;		+^			;
	vxor 		vBx,vBx,vAx

	rotlwi	Sr_c,Sr_c,3        		;			r		;
	add 	L1b,L1b,tmp        		;		+|			;
	vrlw		vBx,vBx,vAx

	add 	Sr_d,S0r,Sr_d      		;				+	;
	rotlw	L1b,L1b,tmp        		;		rv			;
	vadduwm		vBx,vBx,vS21

	add 	tmp,Sr_c,L0c       		;			+^		;
	add 	Sr_d,Sr_d,L0d      		;				+	;
	vadduwm		vS22,vS22,vS21	; r3.22

	lvx		vL0t,rvec1,sp      								; // wait 9 cycles after stw L0d
	add 	L1c,L1c,tmp        		;			+|		;
	vadduwm		vS22,vS22,vL1	; Load L1t after here

	lwbrx 	L0a,0,sp20x      	;//	;					; Load key1 reversed for next cycle
	rotlwi	Sr_d,Sr_d,3        		;				r	;
	vrlw		vS22,vS22,val3

	stw 	Sr_a,S1_25.a(sp)   		;	s				;	; Store S1_25
	rotlw	L1c,L1c,tmp        		;			rv		;
	vadduwm		vtmp,vS22,vL1

	stw 	Sr_b,S1_25.b(sp)   		;		s			;
	add 	tmp,Sr_d,L0d       		;				+^	;
	vadduwm		vL0,vtmp,vL0	;->v20

	stw 	Sr_c,S1_25.c(sp)   		;			s		;
	add 	L1d,L1d,tmp        		;				+|	;
	vrlw		vL0,vL0,vtmp

	stw 	Sr_d,S1_25.d(sp)   		;				s	;
	rotlw	L1d,L1d,tmp        		;				rv	;
	vxor 		vAx,vAx,vBx

	stw 	L1a,L_1.a(sp)      		;	s				;	; Store L1 from 1.25
	add 	L0a,L0a,con1       	;//	;	*				;	; Begin round 1.1 of next cycle 
	vrlw		vAx,vAx,vBx

	stw 	L1b,L_1.b(sp)      		;		s			;
	addis	L0b,L0a,256        	;//	;		+			;	; increment high byte
	vadduwm		vAx,vAx,vS22

	stw 	L1c,L_1.c(sp)      		;			s		;
	addis	L0c,L0a,512        	;//	;			+		;	; twice
	vadduwm		vS23,vS23,vS22	; r3.23 // splat vS0t after here

	stw 	L1d,L_1.d(sp)      		;				s	;
	addis	L0d,L0a,768        	;//	;				+	;	; thrice
	vadduwm		vS23,vS23,vL0

	rotlw	L1a,L0a,con1       	;//	;	r				;
	vspltw	vS0t,vcon,0			;->v22
	vrlw		vS23,vS23,val3

	add 	Sr_a,con2,L1a       	;	+				;	; round 1.2 even
	rotlw	L1b,L0b,con1       	;//	;		r			;
	vadduwm		vtmp,vS23,vL0

	rotlwi	Sr_a,Sr_a,3        		;	r				;
	rotlw	L1c,L0c,con1       	;//	;			r		;
	vadduwm		vL1,vtmp,vL1	;->v21

	add 	tmp,Sr_a,L1a       		;	+^				;
	rotlw	L1d,L0d,con1       	;//	;				r	;
	vrlw		vL1,vL1,vtmp								; Load vS25t after here

	add 	L0a,con0,tmp       		;	+|				;
	add 	Sr_b,con2,L1b      		;		+			;
	vxor 		vBx,vBx,vAx

	rotlw	L0a,L0a,tmp        		;	rv				;
	rotlwi	Sr_b,Sr_b,3        		;		r			;
	vrlw		vBx,vBx,vAx

	add 	Sr_c,con2,L1c      		;			+		;
	add 	tmp,Sr_b,L1b       		;		+^			;
	vadduwm		vBx,vBx,vS23	;

	add 	L0b,con0,tmp       		;		+|			;
	vadduwm		vS24,vS24,vS23	; r3.24						// splat C_0 after here

	lvx		vS25t,rvec3,sp30x	; 							// wait 9 cycles after stw Sr_d
	vspltw vhit,vtxt,2	; C_0								// replaces vS23
	vadduwm		vS24,vS24,vL1

	lvx		vL1t,rvec1,sp10x	; 							// wait 9 cycles after stw L1d !!!
	rotlwi	Sr_c,Sr_c,3        		;			r		;
	vrlw		vS24,vS24,val3

	vxor 		vAx,vAx,vBx

	vrlw		vAx,vAx,vBx

	vadduwm		vAx,vAx,vS24

	vcmpequw.	vhit,vAx,vhit	;->v23
	
;										// rotate fill for branch resolution
	bdnzt	26,loop			; cnt > 0 and vhit == all 0
	
;	State at end of loop:
;		Computed through top of 3.24 results in vS24, vS25, vL0, vL1, vAx, vBx
;		vhit is mask of encryption results matching C_0
;		(all vector registers are in use. vS0t, vS25t, vL0t, vL1t can be reloaded)
;//Finish
;// if hit
;//		finish encryption and compare C_1
;//		if still hit return success
;//		restore temporary registers used
;//		if not done yet
;//			jump to loop
;// Compare guard vectors to saved values
;// if not the same return error
;// return not found
cnt	SET S0r
	mfctr	cnt
	bt	26,done	; vhit == all 0 so ctr must have expired
; finish last round
vtmp set vS0t	; use as temp
	vadduwm		vtmp,vS24,vL1						
	vadduwm		vL0,vtmp,vL0
	vrlw		vL0,vL0,vtmp						
	vadduwm		vS25,vS25,vS24	; r3.25					
	vadduwm		vS25,vS25,vL0						
	vrlw		vS25,vS25,val3						
	vxor 		vBx,vBx,vAx					
	vrlw		vBx,vBx,vAx						
	vadduwm		vBx,vBx,vS25						
	vspltw		vtmp,vtxt,3		; C_0
	vcmpequw	vtmp,vBx,vtmp
	vand		vhit,vhit,vtmp
	vspltisw	vtmp,-1
	vcmpequw.	vtmp,vtmp,vhit
	bf  		26,found
	vspltw		vS0t,vcon,0		; restore temp register
	cmplwi		cnt,0
	bne 		loop
done:
; cnt is 0, fix rollover, check for remaining iterations
; Exit phase registers
; key1	EQU r18
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
;// NOTE: vcon and vtxt are reset if key1 wraps around
;// 	(this test is only here becuse some hacked os's
;//		might not properly handle context switches)
	vspltisw	vS2,3
	vcmpequw	vS2,vS2,val3
	li			tmpx,S1_0
	lvx			vS3,tmpx,sp
	vcmpequw	vS3,vS3,vcon
	li			tmpx,P_0
	lvx			vS4,tmpx,sp
	vcmpequw	vS4,vS4,vtxt
	vand		vS5,vS2,vS3
	vand		vS5,vS5,vS4
	vspltisw	vS2,-1
	vcmpequw.	vS2,vS2,vS5		#(sets condition)
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
	