#
# RC5-72 Altivec core - MPC744x/MPC745x (MacOS ABI)
# TAB = 4
#
# Written by Didier Levet (kakace@wanadoo.fr)
# Copyright distributed.net 1997-2003 - All Rights Reserved
# For use in distributed.net projects only.
# Any other distribution or use of this source violates copyright.
#
# Hacked from Mac OS X assembler format to 'gas' 2.x format
# by Monroe Williams <monroe@pobox.com>.
#
# This implementation is optimized for the MPC7450 family (G4+). The core
# checks 4 keys per loop using the AltiVec units while computing parts
# of the next loop with the integer units.
# The main goal is to use otherwise unused clock cycles to do some work
# with non-AltiVec units, while maintaining the maximum throughput from
# AltiVec. Since the MPC7450 can dispatch or retire three instructions
# per cycle, we can insert two integer instructions (but only one load
# or store) between each AltiVec pair.
#
# Dependencies : 
#
#	struct rc5_72UnitWork (ccoreio.h) :
#		typedef struct {
#  			struct {u32 hi,lo;} plain;
#  			struct {u32 hi,lo;} cypher;
#  			struct {u32 hi,mid,lo;} L0;
#  			struct {u32 count; u32 hi,mid,lo;} check;
#		} RC5_72UnitWork;
#
#	MINIMUM_ITERATIONS (problem.cpp) :
#		The number of iterations to perform is always an even multiple of
#		MINIMUM_ITERATIONS, and the first key to checked is also an even
#		multiple of this constant.
#		Therefore, it is assumed that the number of iterations is never
#		equal to zero (otherwise it would be interpreted as 2^32). 
#		The current value of 24 also ensure that we can process 1, 2, 4 or
#		8 keys at once, all keys (within such "block") having the same mid
#		and lo values).
#
# $Id: r72-KKS7450.gas.s,v 1.1.2.1 2003/03/12 12:13:15 snake Exp $
#
# $Log: r72-KKS7450.gas.s,v $
# Revision 1.1.2.1  2003/03/12 12:13:15  snake
# Added optimized gas compliant cores (like those for MacOS X)
#
# Revision 1.1.2.1  2003/01/20 00:28:29  mfeiri
# New AltiVec core by kakace for 7450 and 7455
#
# Revision 1.5  2003/01/05 18:47:37  kakace
# check.count was trashed (epilog)
#
# Revision 1.4  2003/01/05 01:19:05  kakace
# Little speedup
#
# Revision 1.3  2003/01/05 01:04:00  kakace
# Fixed stage 29 to avoid LSU stalls
#
# Revision 1.2  2003/01/04 22:46:13  kakace
# Fixed a bug that makes the core miss potential matches against specific keys.
# Enhanced unrolled code in an attempt to get more speed.
#
# Revision 1.1  2003/01/01 16:36:27  kakace
# Initial release
#
#============================================================================
 
        .text
		.align	2
		.globl	rc5_72_unit_func_KKS7450

# Register aliases

.set    r0,0
.set    r1,1
.set    r2,2
.set    r3,3
.set    r4,4
.set    r5,5
.set    r6,6
.set    r7,7
.set    r8,8
.set    r9,9
.set    r10,10
.set    r11,11
.set    r12,12
.set    r13,13
.set    r14,14
.set    r15,15
.set    r16,16
.set    r17,17
.set    r18,18
.set    r19,19
.set    r20,20
.set    r21,21
.set    r22,22
.set    r23,23
.set    r24,24
.set    r25,25
.set    r26,26
.set    r27,27
.set    r28,28
.set    r29,29
.set    r30,30
.set    r31,31

.set    v0,0
.set    v1,1
.set    v2,2
.set    v3,3
.set    v4,4
.set    v5,5
.set    v6,6
.set    v7,7
.set    v8,8
.set    v9,9
.set    v10,10
.set    v11,11
.set    v12,12
.set    v13,13
.set    v14,14
.set    v15,15
.set    v16,16
.set    v17,17
.set    v18,18
.set    v19,19
.set    v20,20
.set    v21,21
.set    v22,22
.set    v23,23
.set    v24,24
.set    v25,25
.set    v26,26
.set    v27,27
.set    v28,28
.set    v29,29
.set    v30,30
.set    v31,31

# Result values (see ccoreio.h)	
	
.set 	RESULT_NOTHING	, 1
.set 	RESULT_FOUND	, 2
	

# struct rc5_72unitwork

.set 	plain_hi	,	 0
.set 	plain_lo	,	 4
.set 	cypher_hi	,	 8
.set 	cypher_lo	,	12
.set 	L0_hi		,	16
.set 	L0_mid		,	20
.set 	L0_lo		,	24
.set 	check_count	,	28
.set 	check_hi	,	32
.set 	check_mid	,	36
.set 	check_lo	,	40


# RSA constants

.set 	P	, 0xB7E15163
.set 	Q	, 0x9E3779B9


# stack frame (MacOS ABI)
# The floating-point register save area and the parameter list
# area remain empty. Therefore, the stack frame looks like this :
#	General register save area
#	VRsave save word
#	Alignment padding			: Computed at run time
#	Vector register save area	: Must be quad-word aligned
#	Local variable space		: Shall be quad-word aligned.
#	Linkage area
#
# Tricks :
# The linkage area is extended to be an even multiple of 16, so that the 
# local variable space is properly aligned on a quad-word boundary. 
# The local area size is rounded to an even multiple of 16. Thus, there's 
# only one alignment padding area that need to be computed at run time to
# ensure the correct alignment of the whole stack frame.

.set 	oldLR		,	8			# 8(sp) stores the old LR (caller's frame)
.set 	numGPRs		,	32-13		# r13 to r31
.set 	numVRs		,	32-20		# v20 to v31

	# The linkageArea size, as defined by Apple, is equal to 24. Since we
	# have to align our local variable space to an address that is an
	# even multiple of 16, we shall add an extra padding. Of course, the
	# padding area could be used to save some local variables.

.set 	reserved	,	24			# Size of the linkageArea.
.set 	saveR2		,	24			# Storage for r2 (defined as volatile under
								# MacOS X, but usually points to RTOC).
.set 	iterations	,	28			# u32 *iterations

.set 	linkageArea	,	iterations + 4

# Local variables area
# Offsets (from the callee's SP) to the local variable space and
# the vector register save area.

.set 	localBase	,	linkageArea
	
	# Don't change the order of the following offsets because
	# they can be accessed relatively to each other by means of
	# four pointers and four indexes :
	# Indexes  : -16, -32, -48, -64 (not counting r0 = 0)
	# Pointers : vS_00, vS_05, vS_10, vS_15
.set 	vKey	,	localBase	# (key.hi, key.mid, key.lo, -check.count)
.set 	vL_00	,	vKey + 16	# vector (key0, key1, key2, key3)
.set 	vL_01	,	vL_00 + 16
.set 	vL_02	,	vL_01 + 16

.set 	vS_00	,	vL_02 + 16	# (ptr1) vector (key0, key1, key2, key3)
.set 	vS_01	,	vS_00 + 16
.set 	vS_02	,	vS_01 + 16
.set 	vS_03	,	vS_02 + 16
.set 	vS_04	,	vS_03 + 16
.set 	vS_05	,	vS_04 + 16	# ptr2
.set 	vS_06	,	vS_05 + 16
.set 	vS_07	,	vS_06 + 16
.set 	vS_08	,	vS_07 + 16
.set 	vS_09	,	vS_08 + 16
.set 	vS_10	,	vS_09 + 16	# ptr3
.set 	vS_11	,	vS_10 + 16
.set 	vS_12	,	vS_11 + 16
.set 	vS_13	,	vS_12 + 16
.set 	vS_14	,	vS_13 + 16
.set 	vS_15	,	vS_14 + 16	# ptr4
.set 	vS_16	,	vS_15 + 16
.set 	vS_17	,	vS_16 + 16
.set 	vS_18	,	vS_17 + 16
.set 	vS_19	,	vS_18 + 16
.set 	vS_20	,	vS_19 + 16
.set 	vS_21	,	vS_20 + 16
.set 	vS_22	,	vS_21 + 16
.set 	vS_23	,	vS_22 + 16
.set 	vS_24	,	vS_23 + 16
.set 	vS_25	,	vS_24 + 16

.set 	vCheck	,	vS_25 + 16	# (check.hi, check.mid, check.lo, -count)
.set 	vText	,	vCheck + 16 # (plain.hi, plain.lo, cypher.hi, cypher.lo)

.set 	rc5_ptr	,	vText + 16	# (ptr) Copy of rc5_72unitwork. 
	
	# The "key" elements below hold the current rc5 key for the integer
	# work. *THIS* key is incremented by 4 after each iteration, propagating
	# the carry as required. Since AltiVec works with the previous 4 keys,
	# key_lo, key_mid and key_hi shall be copied into vKey before being
	# updated.
.set 	key_hi	,	rc5_ptr + 4			# key.hi, key.mid and key.lo
.set 	key_mid	,	key_hi + 4			# for the integer pass.
.set 	key_lo	,	key_mid + 4			# DON'T CHANGE THEIR RESPECTIVE ORDER

.set 	S1_cached	,	key_lo + 4		# Save intermediate (constant) results
.set 	S2_cached	,	S1_cached + 4	# for use by the integer unit code when
.set 	L0_cached	,	S2_cached + 4	# only key.hi changed.
.set 	L1_cached	,	L0_cached + 4

.set 	localTOP	,	L1_cached + 4	# Last offset
.set 	vectorBase	, 	(linkageArea + localTOP + 15) & (-16)

# Total frame size (minus the optional alignment padding).

.set 	frameSize	, vectorBase + numVRs * 16 + 4 + numGPRs * 4
	
# Size of the general register save area. This value is used as a
# (negative) offset from the caller's SP to save all non-volatile
# GPRs, as well as the VRsave register (which is saved at offset
# -GPRSave-4).

.set 	GPRSave		, numGPRs * 4
.set	VRsave, 0x100

	
#============================================================================
#
# s32 rc5_72_unit_func_AltiVec_MPC7450 (RC5_72UnitWork *rc5_72unitwork (r3), 
#  r3									u32 *iterations (r4), 
#										void * /*memblk (r5) */)

rc5_72_unit_func_KKS7450:

	# Prolog
	# Save r2, LR, GPRs, VRs and VRsave to the stack frame.
	# Although the Apple ABI states that there's a red zone at negative 
	# offsets from the stack pointer, I don't use this feature to help 
	# porting the code to other ABI.
	# Said otherwise, nothing is written at negative offsets from the 
	# stack pointer. 
	#
	# NOTES : 
	#	- The stack pointer is assumed to be aligned on a 8-byte 
	#	  boundary.
		
		mflr	r0
		stw		r0,oldLR(r1)		# LR (to caller's stack frame)
		mr		r6,r1				# Caller's SP
				
		rlwinm	r5,r1,0,28,28		# Adjust the frame size (compute the size
		subfic	r5,r5,-frameSize	# of the optional alignment padding),
		stwux	r1,r1,r5			# then create the new stack frame.

	# Save all non-volatile registers (r13-r31, v20-v31 and VRsave)
		
		stmw	r13,-GPRSave(r6)	# Save r13-r31
		mfspr	r0,VRsave
		stw		r0,-4-GPRSave(r6)	# Old VRsave
		li		r0,-1				# We'll use all vector registers.
		mtspr	VRsave,r0
		stw		r2,saveR2(r1)
		
	# Save the arguments to free 2 registers.
	
		stw		r4,iterations(r1)
		stw		r3,rc5_ptr(r1)
	
	# Save all non-volatile vectors

		li		r6,vectorBase
		stvx	v31,r6,r1
		addi	r6,r6,16
		stvx	v30,r6,r1
		addi	r6,r6,16
		stvx	v29,r6,r1
		addi	r6,r6,16
		stvx	v28,r6,r1
		addi	r6,r6,16
		stvx	v27,r6,r1
		addi	r6,r6,16
		stvx	v26,r6,r1
		addi	r6,r6,16
		stvx	v25,r6,r1
		addi	r6,r6,16
		stvx	v24,r6,r1
		addi	r6,r6,16
		stvx	v23,r6,r1
		addi	r6,r6,16
		stvx	v22,r6,r1
		addi	r6,r6,16
		stvx	v21,r6,r1
		addi	r6,r6,16
		stvx	v20,r6,r1
		
	#--------------------------------------------------------------------
	# Cruncher setup
	# r3 := struct rc5_72UnitWork *rc5_72unitwork
	# r4 := u32 *iterations
	#
	# According to problem.cpp/MINIMUM_ITERATIONS, the number of keys to
	# check (iterations) is always an even multiple of 24. The cruncher
	# is usually asked to check thousands of 24-key blocks (no performance
	# issues involved here).
	
	# Initialize vKey. Encoding summary :
	# 	vKey := (key.hi, key.mid, key.lo, -check.count)
		
		lwz		r14,L0_hi(r3)		# (L0_hi % 24) == 0 ==> (L0_hi & 7) == 0
		lwz		r13,L0_mid(r3)
		lwz		r12,L0_lo(r3)
		lwz		r8,check_count(r3)	# check.count
		stw		r14,vKey(r1)		# key.hi
		subfic	r8,r8,0				# -check.count
		stw		r13,vKey+4(r1)		# key.mid
		stw		r12,vKey+8(r1)		# key.lo
		stw		r8,vKey+12(r1)		# -check.count
		
	# Initialize vCheck. Encoding summary :
	# vCheck := {check.hi; check.mid; check.lo; -check.count}

		lwz		r5,check_hi(r3)		# check.hi
		lwz		r6,check_mid(r3)	# check.mid
		lwz		r7,check_lo(r3)		# check.lo
		stw		r5,vCheck(r1)
		stw		r6,vCheck+4(r1)
		stw		r7,vCheck+8(r1)
		stw		r8,vCheck+12(r1)	# -check.count
									
	# Initialize vText. Encoding summary :
	#	vText := {plain.hi; plain.lo; cypher.hi; cypher.lo}
		
		lwz		r6,plain_hi(r3)
		lwz		r7,plain_lo(r3)
		stw		r6,vText(r1)		# plain.hi
		stw		r7,vText+4(r1)		# plain.lo 
		lwz		r6,cypher_hi(r3)
		lwz		r7,cypher_lo(r3)
		stw		r6,vText+8(r1)		# cypher.hi
		stw		r7,vText+12(r1)		# cypher.lo

	# Initialize invariant registers

		li		 r6,vText
		vspltisw v31,3				# ROTL count (constant)
		lvx		 v30,r6,r1			# vText (constant)

		lwz		r5,0(r4)
		srwi	r5,r5,2				# iterations / 4
		mtctr	r5					# Setup the loop counter

		lis		r5,(expanded_key)@h
		ori		r2,r5,(expanded_key)@l

	# Offsets to local vector variables.
		
		li		r0,-64
		li		r4,-48
		li		r5,-32
		li		r6,-16
		
	# Pointers to local vector variables.
	
		addi	r7,r1,vS_00
		addi	r8,r1,vS_05
		addi	r9,r1,vS_10
		addi	r10,r1,vS_15

	# Initialize SVec[0] to SVec[15], LVec[] and save them to memory.
	#
	# Assigned registers :
	# r0  := -64 (const)
	# r2  := &expanded_key table (const)
	# r4  := -48 (const)
	# r5  := -32 (const)
	# r6  := -16 (const)
	# r7-r10 := Reserved.
	# r12,r13,r14 := key.lo, key.mid, key.hi
	# CTR := Loop count (== iterations / 4)
	# v30 := {plain.hi; plain.lo; cypher.hi; cypher.lo} (const)
	# v31 := ROTL count (== 3, const)
	#
	# Register used :
	# r15-r19 := Pointers to the static expanded key table.
	# v0-v15  := S[0]-S[15]
	# v26-v28 := L[0]-L[2]
	# v29 := temporary storage

		# Init S[0], S[1], L[0], L[1], L[2]
		# and some local constants
		lvx		v29,r6,r2			# vector(0,1,2,3)
		addi	r15,r2,80			# -> S[5]
		addi	r16,r2,160			# -> S[10]
		lvx		v27,r7,r0			# vKey(hi, mid, lo)
		addi	r17,r2,240			# -> S[15]
		addi	r18,r2,320			# -> S[20]
		lvx		v0,0,r2 			# load S[0]
		addi	r19,r2,400			# -> S[25]
		vspltw	v28,v27,0			# get L[2]
		lvx		v1,r15,r0			# load S[1]
		vadduwm	v28,v28,v29			# L[2] = key.hi + (0,1,2,3)
		vspltw	v26,v27,2			# get L[0] = key.lo

		#------ Stage 0. Mix S[0] and L[0]
		lvx		v2,r15,r4			# load S[2]
		vspltw	v27,v27,1			# L[1] = key.mid
		vrlw	v0,v0,v31			# A = S[0] = ROTL3(...)
		lvx		v3,r15,r5			# load S[3]
		addi	r20,r1,vS_20			# &vS_20
		vadduwm	v26,v26,v0			# L[0] += A
		stvx	v0,0,r7 			# save S[0]
		addi	r21,r1,vS_25			# &vS[25]
		vrlw	v26,v26,v0			# B = L[0] = ROTL(...)
		lvx		v4,r15,r6			# load S[4]

		#------ Stage 1. Mix S[1] and L[1]
		vadduwm	v29,v26,v0			# A + B
		lvx		v5,0,r15			# load S[5]
		vadduwm	v1,v1,v29			# S[1] += (A + B)
		lvx		v6,r16,r0			# load S[6]
		vrlw	v1,v1,v31			# A = S[1] = ROTL3(...)
		lvx		v7,r16,r4			# load S[7]
		vadduwm	v29,v1,v26			# A + B
		stvx	v1,r8,r0			# save s[1]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		lvx		v8,r16,r5			# load S[8]
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		lvx		v9,r16,r6			# load S[9]

		#------ Stage 2. Mix S[2] and L[2]
		vadduwm	v29,v27,v1			# A + B
		lvx		v10,0,r16			# load S[10]
		vadduwm	v2,v2,v29			# S[2] += (A + B)
		lvx		v11,r17,r0			# load S[11]
		vrlw	v2,v2,v31			# A = S[2] = ROTL3(...)
		lvx		v12,r17,r4			# load S[12]
		vadduwm	v29,v2,v27			# A + B
		stvx	v2,r8,r4			# save s[2]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		lvx		v13,r17,r5			# load S[13]
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		lvx		v14,r17,r6			# load S[14]

		#------ Stage 3. Mix S[3] and L[0]
		vadduwm	v29,v28,v2			# A + B
		lvx		v15,0,r17			# load S[15]
		vadduwm	v3,v3,v29			# S[3] += (A + B)
		lvx		v16,r18,r0			# load S[16]
		vrlw	v3,v3,v31			# A = S[3] = ROTL3(...)
		lvx		v17,r18,r4			# load S[17]
		vadduwm	v29,v3,v28			# A + B
		stvx	v3,r8,r5			# save s[3]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		lvx		v18,r18,r5			# load S[18]
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		lvx		v19,r18,r6			# load S[19]

		#------ Stage 4. Mix S[4] and L[1]
		vadduwm	v29,v26,v3			# A + B
		lvx		v20,0,r18			# load S[20]
		vadduwm	v4,v4,v29			# S[4] += (A + B)
		lvx		v21,r19,r0			# load S[21]
		vrlw	v4,v4,v31			# A = S[4] = ROTL3(...)
		lvx		v22,r19,r4			# load S[22]
		vadduwm	v29,v4,v26			# A + B
		stvx	v4,r8,r6			# save s[4]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		lvx		v23,r19,r5			# load S[23]
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		lvx		v24,r19,r6			# load S[24]

		#------ Stage 5. Mix S[5] and L[2]
		vadduwm	v29,v27,v4			# A + B
		lvx		v25,0,r19			# load S[25]
		vadduwm	v5,v5,v29			# S[5] += (A + B)
		vrlw	v5,v5,v31			# A = S[5] = ROTL3(...)
		vadduwm	v29,v5,v27			# A + B
		stvx	v5,0,r8 			# save s[5]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)

		#------ Stage 6. Mix S[6] and L[0]
		vadduwm	v29,v28,v5			# A + B
		vadduwm	v6,v6,v29			# S[6] += (A + B)
		vrlw	v6,v6,v31			# A = S[6] = ROTL3(...)
		vadduwm	v29,v6,v28			# A + B
		stvx	v6,r9,r0			# save s[6]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)

		#------ Stage 7. Mix S[7] and L[1]
		vadduwm	v29,v26,v6			# A + B
		vadduwm	v7,v7,v29			# S[7] += (A + B)
		vrlw	v7,v7,v31			# A = S[7] = ROTL3(...)
		vadduwm	v29,v7,v26			# A + B
		stvx	v7,r9,r4			# save s[7]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)

		#------ Stage 8. Mix S[8] and L[2]
		vadduwm	v29,v27,v7			# A + B
		vadduwm	v8,v8,v29			# S[8] += (A + B)
		vrlw	v8,v8,v31			# A = S[8] = ROTL3(...)
		vadduwm	v29,v8,v27			# A + B
		stvx	v8,r9,r5			# save s[8]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)

		#------ Stage 9. Mix S[9] and L[0]
		vadduwm	v29,v28,v8			# A + B
		vadduwm	v9,v9,v29			# S[9] += (A + B)
		vrlw	v9,v9,v31			# A = S[9] = ROTL3(...)
		vadduwm	v29,v9,v28			# A + B
		stvx	v9,r9,r6			# save s[9]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)

		#------ Stage 10. Mix S[10] and L[1]
		vadduwm	v29,v26,v9			# A + B
		vadduwm	v10,v10,v29			# S[10] += (A + B)
		vrlw	v10,v10,v31			# A = S[10] = ROTL3(...)
		vadduwm	v29,v10,v26			# A + B
		stvx	v10,0,r9 			# save s[10]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)

		#------ Stage 11. Mix S[11] and L[2]
		vadduwm	v29,v27,v10			# A + B
		vadduwm	v11,v11,v29			# S[11] += (A + B)
		vrlw	v11,v11,v31			# A = S[11] = ROTL3(...)
		vadduwm	v29,v11,v27			# A + B
		stvx	v11,r10,r0			# save s[11]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)

		#------ Stage 12. Mix S[12] and L[0]
		vadduwm	v29,v28,v11			# A + B
		vadduwm	v12,v12,v29			# S[12] += (A + B)
		vrlw	v12,v12,v31			# A = S[12] = ROTL3(...)
		vadduwm	v29,v12,v28			# A + B
		stvx	v12,r10,r4			# save s[12]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)

		#------ Stage 13. Mix S[13] and L[1]
		vadduwm	v29,v26,v12			# A + B
		vadduwm	v13,v13,v29			# S[13] += (A + B)
		vrlw	v13,v13,v31			# A = S[13] = ROTL3(...)
		vadduwm	v29,v13,v26			# A + B
		stvx	v13,r10,r5			# save s[13]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)

		#------ Stage 14. Mix S[14] and L[2]
		vadduwm	v29,v27,v13			# A + B
		vadduwm	v14,v14,v29			# S[14] += (A + B)
		vrlw	v14,v14,v31			# A = S[14] = ROTL3(...)
		vadduwm	v29,v14,v27			# A + B
		stvx	v14,r10,r6			# save s[14]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)

		#------ Stage 15. Mix S[15] and L[0]
		vadduwm	v29,v28,v14			# A + B
		vadduwm	v15,v15,v29			# S[15] += (A + B)
		vrlw	v15,v15,v31			# A = S[15] = ROTL3(...)
		vadduwm	v29,v15,v28			# A + B
		stvx	v15,0,r10 			# save s[15]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)

		#------ Stage 16. Mix S[16] and L[1]
		vadduwm	v29,v26,v15			# A + B
		vadduwm	v16,v16,v29			# S[16] += (A + B)
		vrlw	v16,v16,v31			# A = S[16] = ROTL3(...)
		vadduwm	v29,v16,v26			# A + B
		stvx	v16,r20,r0			# save s[16]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)

		#------ Stage 17. Mix S[17] and L[2]
		vadduwm	v29,v27,v16			# A + B
		vadduwm	v17,v17,v29			# S[17] += (A + B)
		vrlw	v17,v17,v31			# A = S[17] = ROTL3(...)
		vadduwm	v29,v17,v27			# A + B
		stvx	v17,r20,r4			# save s[17]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)

		#------ Stage 18. Mix S[18] and L[0]
		vadduwm	v29,v28,v17			# A + B
		vadduwm	v18,v18,v29			# S[18] += (A + B)
		vrlw	v18,v18,v31			# A = S[18] = ROTL3(...)
		vadduwm	v29,v18,v28			# A + B
		stvx	v18,r20,r5			# save s[18]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)

		#------ Stage 19. Mix S[19] and L[1]
		vadduwm	v29,v26,v18			# A + B
		vadduwm	v19,v19,v29			# S[19] += (A + B)
		vrlw	v19,v19,v31			# A = S[19] = ROTL3(...)
		vadduwm	v29,v19,v26			# A + B
		stvx	v19,r20,r6			# save s[19]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)

		#------ Stage 20. Mix S[20] and L[2]
		vadduwm	v29,v27,v19			# A + B
		vadduwm	v20,v20,v29			# S[20] += (A + B)
		vrlw	v20,v20,v31			# A = S[20] = ROTL3(...)
		vadduwm	v29,v20,v27			# A + B
		stvx	v20,0,r20			# save s[20]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)

		#------ Stage 21. Mix S[21] and L[0]
		vadduwm	v29,v28,v20			# A + B
		vadduwm	v21,v21,v29			# S[21] += (A + B)
		vrlw	v21,v21,v31			# A = S[21] = ROTL3(...)
		vadduwm	v29,v21,v28			# A + B
		stvx	v21,r21,r0			# save s[21]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)

		#------ Stage 22. Mix S[22] and L[1]
		vadduwm	v29,v26,v21			# A + B
		vadduwm	v22,v22,v29			# S[22] += (A + B)
		vrlw	v22,v22,v31			# A = S[22] = ROTL3(...)
		vadduwm	v29,v22,v26			# A + B
		stvx	v22,r21,r4			# save s[22]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)

		#------ Stage 23. Mix S[23] and L[2]
		vadduwm	v29,v27,v22			# A + B
		vadduwm	v23,v23,v29			# S[23] += (A + B)
		vrlw	v23,v23,v31			# A = S[23] = ROTL3(...)
		vadduwm	v29,v23,v27			# A + B
		stvx	v23,r21,r5			# save s[23]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		stvx	v28,r7,r6			# save L[2]

		#------ Stage 24. Mix S[24] and L[0]
		vadduwm	v29,v28,v23			# A + B
		vadduwm	v24,v24,v29			# S[24] += (A + B)
		vrlw	v24,v24,v31			# A = S[24] = ROTL3(...)
		vadduwm	v29,v24,v28			# A + B
		stvx	v24,r21,r6			# save s[24]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		stvx	v26,r7,r4			# save L[0]

		#------ Stage 25. Mix S[25] and L[1]
		vadduwm	v29,v26,v24			# A + B
		vadduwm	v25,v25,v29			# S[25] += (A + B)
		vrlw	v25,v25,v31			# A = S[25] = ROTL3(...)
		vadduwm	v29,v25,v26			# A + B
		stvx	v25,0,r21			# save s[25]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		stvx	v27,r7,r5			# save L[1]

		addi	r14,r14,4			# IU checks the next 4 keys (no carry)
		stw		r12,key_lo(r1)		# Save current integer key.
		stw		r13,key_mid(r1)
		stw		r14,key_hi(r1)
		
		lis		r3,(Q)@h
		ori		r3,r3,(Q)@l
		

	#---------------------------------------------------------------------
	# Main loop. Registers assigned :
	# r0  := -64 (const)
	# r2  := &expanded_key[0] table (const)
	# r3  := Q (0x9E3779B9)
	# r4  := -48 (const)
	# r5  := -32 (const)
	# r6  := -16 (const)
	# r7  := &vS_00 (const)
	# r8  := &vS_05 (const)
	# r9  := &vS_10 (const)
	# r10 := &vS_15 (const)
	# CTR := Loop count (== iterations / 4)
	# v30 := {plain.hi; plain.lo; cypher.hi; cypher.lo} (const)
	# v31 := ROTL count (== 3, const)
	#
	# Registers used :
	# r11 := current s[i]
	# r12-r14 := l0[0], l0[1], l0[2]
	# r15-r17 := l1[0], l1[1], l1[2]
	# r18-r20 := l2[0], l2[1], l2[2]
	# r21-r23 := l3[0], l3[1], l3[2]
	# r24-r27 := s0[x], s1[x], s2[x], s3[x]
	# r28-r31 := t0, t1, t2, t3
	# v0-v25  := S[0]-S[25]
	# v26-v28 := L[0]-L[2]
	# v29 := temporary storage
	#
	# NOTES :
	# - key.mid changes only every 64 loops, and key.lo 
	#	probably never change. Therefore, the first rounds are
	#	as follows :
	#		A = ROTL3(S[0]);
	#		B = L[0] = ROTL(L[0] + A, A);
	#		A = ROTL3(S[1] + A + B);
	#		B = L[1] = ROTL(L[1] + (A + B), (A + B));
	#		A = ROTL3(S[2] + A + B);
	#	where everything remains constant until key.mid or
	#	key.lo change.
	#
	# BUG-FIX:
	#	Since S[1] and S[2] are initialized by the integer code
	#	when key.mid and/or key.lo change, the AltiVec code shall
	#	have to find the right values elsewhere (AltiVec works on
	#	the 4 previous keys). Therefore, v1 and v2 MUST be reset
	#	to their initial values at the end of the loop, where they
	#	can be initialized from vS_01 and vS_02.
	#	vS_00 is constant.


inner_loop_lo:
		# key.lo changed. Preconditions :
		# r12, r13, r14 := key.lo, key.mid, key.hi
		# v1, v2 := S[1], S[2]
		#
		# Compute s[0], l[0] and s[1].

		lwz		r24,vS_00(r1)		# load S[0]
		lis		r11,(P+Q)@h		# s[1] = P + Q
		ori		r11,r11,(P+Q)@l

		add		r21,r12,r24			# key.lo += a
		rotlw	r21,r21,r24			# b = l0[0] = ROTL(...)
		stw		r21,L0_cached(r1)

		add		r28,r24,r21			# a + b
		add		r24,r28,r11			# s[1] += (a + b)
		rotlwi	r24,r24,3			# a = s[1] = ROTL3(...)
		stw		r24,vS_01(r1)		# Write back s[1]
		stw		r24,vS_01+4(r1)
		stw		r24,vS_01+8(r1)
		stw		r24,vS_01+12(r1)
		stw		r24,S1_cached(r1)
		
		add		r11,r11,r3			# s[2] = s[1] + Q


inner_loop_mid:
		# key.mid changed, key.lo unchanged. Preconditions :
		# r21, r13, r14 := l[0], key.mid, key.hi
		# r11 := s[2] = P + 2Q
		# r24 := a = s[1]
		# v1, v2 := S[1], S[2]
		# vS_00 and vS_01 are correct.
		#
		# Compute l[1] and s[2].

		add		r28,r24,r21			# a + b = s[1] + l[0]
		add		r22,r13,r28			# key.mid += a + b
		rotlw	r22,r22,r28			# b = l0[1] = ROTL(...)
		stw		r22,L1_cached(r1)
		
		add		r28,r24,r22			# a + b
		add		r24,r28,r11			# s0[2] += (a + b)
		rotlwi	r24,r24,3			# a = s0[2] = ROTL3(...)
		stw		r24,vS_02(r1)		# Write back s[2]
		stw		r24,vS_02+4(r1)
		stw		r24,vS_02+8(r1)
		stw		r24,vS_02+12(r1)
		stw		r24,S2_cached(r1)

		add		r11,r11,r3			# s[3] = s[2] + Q
		
		
inner_loop_hi:
		# key.mid and key.lo unchanged. Preconditions :
		# r21, r22, r14 := l[0], l[1], key.hi
		# r11 := s[3] = P + 3Q.
		# r24 := a = s[2]
		# v1, v2 := S[1], S[2]
		# vS_00, vS_01 and vS_02 are correct.
		#
		# Registers v0, v3-v25 and v26-v28 will be loaded from memory
		# with the work done by the integer units during the previous
		# loop iteration.

		lvx		v27,r7,r5			# load B
		addi	r13,r1,vS_20		# &vS[20]
		addi	r16,r1,vS_25		# &vS[25]
		lvx		v0,0,r7 			# load s[0]
		add		r28,r24,r22			# a0+b0 = s0[2]+l0[1]
		add		r14,r14,r28			# l0[2] += a0+b0
		lvx		v25,0,r16			# load A
		addi	r17,r14,1			# l1[2] = l0[2]+1
		addi	r20,r14,2			# l2[2] = l0[2]+2

		#------ Stage 26. Mix S[0] and L[2]
		vadduwm	v29,v27,v25			# A + B
		lvx		v28,r7,r6			# load L[2]
		addi	r23,r14,3			# l3[2] = l0[2]+3
		vadduwm	v0,v0,v29			# S[0] += (A + B)
		lvx		v26,r7,r4			# load L[0]
		rotlw	r14,r14,r28			# b0 = l0[2] = ROTL(...)
		vrlw	v0,v0,v31			# A = S[0] = ROTL3(...)
		lvx		v3,r8,r5			# load S[3]
		rotlw	r17,r17,r28			# b1 = l1[2] = ROTL(...)
		vadduwm	v29,v0,v27			# A + B
		lvx		v4,r8,r6			# load S[4]
		rotlw	r20,r20,r28			# b2 = l2[2] = ROTL(...)
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		lvx		v5,0,r8 			# load S[5]
		rotlw	r23,r23,r28			# b3 = l3[2] = ROTL(...)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		lvx		v6,r9,r0			# load S[6]
		add		r28,r24,r14			# a0+b0 = s0[2]+l0[2]

		#------ Stage 27. Mix S[1] and L[0]
		vadduwm	v29,v28,v0			# A + B
		lvx		v7,r9,r4			# load S[7]
		add		r29,r24,r17			# a1+b1 = s0[2]+l1[2]
		vadduwm	v1,v1,v29			# S[1] += (A + B)
		lvx		v8,r9,r5			# load S[8]
		add		r30,r24,r20			# a2+b2 = s0[2]+l1[2]
		vrlw	v1,v1,v31			# A = S[1] = ROTL3(...)
		lvx		v9,r9,r6			# load S[9]
		add		r31,r24,r23			# a3+b3 = s0[2]+l1[2]
		vadduwm	v29,v1,v28			# A + B
		lvx		v10,0,r9 			# load S[10]
		add		r24,r28,r11			# s0[3] += a0+b0
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		lvx		v11,r10,r0			# load S[11]
		add		r25,r29,r11			# s1[3] += a1+b1
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		lvx		v12,r10,r4			# load S[12]
		add		r26,r30,r11			# s2[3] += a2+b2

		#------ Stage 28. Mix S[2] and L[1]
		vadduwm	v29,v26,v1			# A + B
		lvx		v13,r10,r5			# load S[13]
		add		r27,r31,r11			# s3[3] += a3+b3
		vadduwm	v2,v2,v29			# S[2] += (A + B)
		lvx		v14,r10,r6			# load S[14]
		add		r11,r11,r3			# s[4] = P + Q * 4
		vrlw	v2,v2,v31			# A = S[2] = ROTL3(...)
		lvx		v15,0,r10 			# load S[15]
		rotlwi	r24,r24,3			# a0 = s0[3] = ROTL3(...)
		vadduwm	v29,v2,v26			# A + B
		lvx		v16,r13,r0			# load S[16]
		rotlwi	r25,r25,3			# a1 = s1[3] = ROTL3(...)
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		lvx		v17,r13,r4			# load S[17]
		rotlwi	r26,r26,3			# a2 = s2[3] = ROTL3(...)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		lvx		v18,r13,r5			# load S[18]
		rotlwi	r27,r27,3			# a3 = s3[3] = ROTL3(...)

		#------ Stage 29. Mix S[3] and L[2]
		vadduwm	v29,v27,v2			# A + B
		add		r28,r24,r14			# a0+b0 = s0[3]+l0[2]	
		stw		r24,vS_03(r1)
		vadduwm	v3,v3,v29			# S[3] += (A + B)
		add		r29,r25,r17			# a1+b1 = s1[3]+l1[2]
		stw		r25,vS_03+4(r1)
		vrlw	v3,v3,v31			# A = S[3] = ROTL3(...)
		add		r30,r26,r20			# a2+b2 = s2[3]+l2[2]
		stw		r26,vS_03+8(r1)
		vadduwm	v29,v3,v27			# A + B
		add		r31,r27,r23			# a3+b3 = s3[3]+l3[2]
		stw		r27,vS_03+12(r1)
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		# Empty slot to avoid CQ stalls.
		lvx		v19,r13,r6			# load S[19]
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		add		r12,r21,r28			# l0[0] += a0+b0
		lvx		v20,0,r13			# load S[20]

		#------ Stage 30. Mix S[4] and L[0]
		vadduwm	v29,v28,v3			# A + B
		add		r15,r21,r29			# l1[0] = l0[0] + a1+b1
		lvx		v21,r16,r0			# load S[21]
		vadduwm	v4,v4,v29			# S[4] += (A + B)
		add		r18,r21,r30			# l2[0] = l0[0] + a2+b2
		lvx		v22,r16,r4			# load S[22]
		vrlw	v4,v4,v31			# A = S[4] = ROTL3(...)
		add		r21,r21,r31			# l1[0] = l0[0] + a3+b3
		lvx		v23,r16,r5			# load S[23]
		vadduwm	v29,v4,v28			# A + B
		rotlw	r12,r12,r28			# b0 = l0[0] = ROTL(...)
		lvx		v24,r16,r6			# load S[24]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		rotlw	r15,r15,r29			# b1 = l1[0] = ROTL(...)
		rotlw	r18,r18,r30			# b2 = l2[0] = ROTL(...)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		rotlw	r21,r21,r31			# b3 = l3[0] = ROTL(...)
		add		r28,r24,r12			# a0+b0 = s0[3]+l0[0]

		#------ Stage 31. Mix S[5] and L[1]
		vadduwm	v29,v26,v4			# A + B
		add		r29,r25,r15			# a1+b1 = s1[3]+l1[0]
		add		r30,r26,r18			# a2+b2 = s2[3]+l2[0]
		vadduwm	v5,v5,v29			# S[5] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[3]+l3[0]
		add		r24,r28,r11			# s0[4] += a0+b0
		vrlw	v5,v5,v31			# A = S[5] = ROTL3(...)
		add		r25,r29,r11			# s1[4] += a1+b1
		add		r26,r30,r11			# s2[4] += a2+b2
		vadduwm	v29,v5,v26			# A + B
		add		r27,r31,r11			# s3[4] += a3+b3
		add		r11,r11,r3			# s[5] = P + Q * 5
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[4] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[4] = ROTL3(...)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		rotlwi	r26,r26,3			# a2 = s2[4] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[4] = ROTL3(...)

		#------ Stage 32. Mix S[6] and L[2]
		vadduwm	v29,v27,v5			# A + B
		stw		r24,vS_04(r1)
		add		r28,r24,r12			# a0+b0 = s0[4]+l0[0]
		vadduwm	v6,v6,v29			# S[6] += (A + B)
		stw		r25,vS_04+4(r1)
		add		r29,r25,r15			# a1+b1 = s1[4]+l1[0]
		vrlw	v6,v6,v31			# A = S[6] = ROTL3(...)
		stw		r26,vS_04+8(r1)
		add		r30,r26,r18			# a2+b2 = s2[4]+l2[0]
		vadduwm	v29,v6,v27			# A + B
		stw		r27,vS_04+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[4]+l3[0]

		add		r13,r22,r28			# l0[1] += a0+b0
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		add		r16,r22,r29			# l1[1] = l0[1] + a1+b1
		add		r19,r22,r30			# l2[1] = l0[1] + a2+b2

		#------ Stage 33. Mix S[7] and L[0]
		vadduwm	v29,v28,v6			# A + B
		add		r22,r22,r31			# l1[1] = l0[1] + a3+b3
		rotlw	r13,r13,r28			# b0 = l0[1] = ROTL(...)
		vadduwm	v7,v7,v29			# S[7] += (A + B)
		rotlw	r16,r16,r29			# b1 = l1[1] = ROTL(...)
		rotlw	r19,r19,r30			# b2 = l2[1] = ROTL(...)
		vrlw	v7,v7,v31			# A = S[7] = ROTL3(...)
		rotlw	r22,r22,r31			# b3 = l3[1] = ROTL(...)
		add		r28,r24,r13			# a0+b0 = s0[4]+l0[1]
		vadduwm	v29,v7,v28			# A + B
		add		r29,r25,r16			# a1+b1 = s1[4]+l1[1]
		add		r30,r26,r19			# a2+b2 = s2[4]+l2[1]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[4]+l3[1]
		add		r24,r28,r11			# s0[5] += a0+b0
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		add		r25,r29,r11			# s1[5] += a1+b1
		add		r26,r30,r11			# s2[5] += a2+b2

		#------ Stage 34. Mix S[8] and L[1]
		vadduwm	v29,v26,v7			# A + B
		add		r27,r31,r11			# s3[5] += a3+b3
		add		r11,r11,r3			# s[6] = P + Q * 6
		vadduwm	v8,v8,v29			# S[8] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[5] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[5] = ROTL3(...)
		vrlw	v8,v8,v31			# A = S[8] = ROTL3(...)
		rotlwi	r26,r26,3			# a2 = s2[5] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[5] = ROTL3(...)
		vadduwm	v29,v8,v26			# A + B
		stw		r24,vS_05(r1)
		add		r28,r24,r13			# a0+b0 = s0[5]+l0[1]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		stw		r25,vS_05+4(r1)
		add		r29,r25,r16			# a1+b1 = s1[5]+l1[1]
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		stw		r26,vS_05+8(r1)
		add		r30,r26,r19			# a2+b2 = s2[5]+l2[1]

		#------ Stage 35. Mix S[9] and L[2]
		vadduwm	v29,v27,v8			# A + B
		stw		r27,vS_05+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v9,v9,v29			# S[9] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[5]+l3[1]

		add		r14,r14,r28			# l0[2] += a0+b0
		vrlw	v9,v9,v31			# A = S[9] = ROTL3(...)
		add		r17,r17,r29			# l1[2] += a1+b1
		add		r20,r20,r30			# l2[2] += a2+b2
		vadduwm	v29,v9,v27			# A + B
		add		r23,r23,r31			# l3[2] += a3+b3
		rotlw	r14,r14,r28			# b0 = l0[2] = ROTL(...)
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		rotlw	r17,r17,r29			# b1 = l1[2] = ROTL(...)
		rotlw	r20,r20,r30			# b2 = l2[2] = ROTL(...)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		rotlw	r23,r23,r31			# b3 = l3[2] = ROTL(...)
		add		r28,r24,r14			# a0+b0 = s0[5]+l0[2]

		#------ Stage 36. Mix S[10] and L[0]
		vadduwm	v29,v28,v9			# A + B
		add		r29,r25,r17			# a1+b1 = s1[5]+l1[2]
		add		r30,r26,r20			# a2+b2 = s2[5]+l2[2]
		vadduwm	v10,v10,v29			# S[10] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[5]+l3[2]
		add		r24,r28,r11			# s0[6] += a0+b0
		vrlw	v10,v10,v31			# A = S[10] = ROTL3(...)
		add		r25,r29,r11			# s1[6] += a1+b1
		add		r26,r30,r11			# s2[6] += a2+b2
		vadduwm	v29,v10,v28			# A + B
		add		r27,r31,r11			# s3[6] += a3+b3
		add		r11,r11,r3			# s[7] = P + Q * 7
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[6] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[6] = ROTL3(...)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		rotlwi	r26,r26,3			# a2 = s2[6] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[6] = ROTL3(...)

		#------ Stage 37. Mix S[11] and L[1]
		vadduwm	v29,v26,v10			# A + B
		stw		r24,vS_06(r1)
		add		r28,r24,r14			# a0+b0 = s0[6]+l0[2]
		vadduwm	v11,v11,v29			# S[11] += (A + B)
		stw		r25,vS_06+4(r1)
		add		r29,r25,r17			# a1+b1 = s1[6]+l1[2]
		vrlw	v11,v11,v31			# A = S[11] = ROTL3(...)
		stw		r26,vS_06+8(r1)
		add		r30,r26,r20			# a2+b2 = s2[6]+l2[2]
		vadduwm	v29,v11,v26			# A + B
		stw		r27,vS_06+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[6]+l3[2]

		add		r12,r12,r28			# l0[0] += a0+b0
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		add		r15,r15,r29			# l1[0] += a1+b1
		add		r18,r18,r30			# l2[0] += a2+b2

		#------ Stage 38. Mix S[12] and L[2]
		vadduwm	v29,v27,v11			# A + B
		add		r21,r21,r31			# l3[0] += a3+b3
		rotlw	r12,r12,r28			# b0 = l0[0] = ROTL(...)
		vadduwm	v12,v12,v29			# S[12] += (A + B)
		rotlw	r15,r15,r29			# b1 = l1[0] = ROTL(...)
		rotlw	r18,r18,r30			# b2 = l2[0] = ROTL(...)
		vrlw	v12,v12,v31			# A = S[12] = ROTL3(...)
		rotlw	r21,r21,r31			# b3 = l3[0] = ROTL(...)
		add		r28,r24,r12			# a0+b0 = s0[6]+l0[0]
		vadduwm	v29,v12,v27			# A + B
		add		r29,r25,r15			# a1+b1 = s1[6]+l1[0]
		add		r30,r26,r18			# a2+b2 = s2[6]+l2[0]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[6]+l3[0]
		add		r24,r28,r11			# s0[7] += a0+b0
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		add		r25,r29,r11			# s1[7] += a1+b1
		add		r26,r30,r11			# s2[7] += a2+b2

		#------ Stage 39. Mix S[13] and L[0]
		vadduwm	v29,v28,v12			# A + B
		add		r27,r31,r11			# s3[7] += a3+b3
		add		r11,r11,r3			# s[8] = P + Q * 8
		vadduwm	v13,v13,v29			# S[13] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[7] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[7] = ROTL3(...)
		vrlw	v13,v13,v31			# A = S[13] = ROTL3(...)
		rotlwi	r26,r26,3			# a2 = s2[7] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[7] = ROTL3(...)
		vadduwm	v29,v13,v28			# A + B
		stw		r24,vS_07(r1)
		add		r28,r24,r12			# a0+b0 = s0[7]+l0[0]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		stw		r25,vS_07+4(r1)
		add		r29,r25,r15			# a1+b1 = s1[7]+l1[0]
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		stw		r26,vS_07+8(r1)
		add		r30,r26,r18			# a2+b2 = s2[7]+l2[0]

		#------ Stage 40. Mix S[14] and L[1]
		vadduwm	v29,v26,v13			# A + B
		stw		r27,vS_07+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v14,v14,v29			# S[14] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[7]+l3[0]

		add		r13,r13,r28			# l0[1] += a0+b0
		vrlw	v14,v14,v31			# A = S[14] = ROTL3(...)
		add		r16,r16,r29			# l1[1] += a1+b1
		add		r19,r19,r30			# l2[1] += a2+b2
		vadduwm	v29,v14,v26			# A + B
		add		r22,r22,r31			# l3[1] += a3+b3
		rotlw	r13,r13,r28			# b0 = l0[1] = ROTL(...)
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		rotlw	r16,r16,r29			# b1 = l1[1] = ROTL(...)
		rotlw	r19,r19,r30			# b2 = l2[1] = ROTL(...)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		rotlw	r22,r22,r31			# b3 = l3[1] = ROTL(...)
		add		r28,r24,r13			# a0+b0 = s0[7]+l0[1]

		#------ Stage 41. Mix S[15] and L[2]
		vadduwm	v29,v27,v14			# A + B
		add		r29,r25,r16			# a1+b1 = s1[7]+l1[1]
		add		r30,r26,r19			# a2+b2 = s2[7]+l2[1]
		vadduwm	v15,v15,v29			# S[15] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[7]+l3[1]
		add		r24,r28,r11			# s0[8] += a0+b0
		vrlw	v15,v15,v31			# A = S[15] = ROTL3(...)
		add		r25,r29,r11			# s1[8] += a1+b1
		add		r26,r30,r11			# s2[8] += a2+b2
		vadduwm	v29,v15,v27			# A + B
		add		r27,r31,r11			# s3[8] += a3+b3
		add		r11,r11,r3			# s[9] = P + Q * 9
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[8] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[8] = ROTL3(...)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		rotlwi	r26,r26,3			# a2 = s2[8] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[8] = ROTL3(...)

		#------ Stage 42. Mix S[16] and L[0]
		vadduwm	v29,v28,v15			# A + B
		stw		r24,vS_08(r1)
		add		r28,r24,r13			# a0+b0 = s0[8]+l0[1]
		vadduwm	v16,v16,v29			# S[16] += (A + B)
		stw		r25,vS_08+4(r1)
		add		r29,r25,r16			# a1+b1 = s1[8]+l1[1]
		vrlw	v16,v16,v31			# A = S[16] = ROTL3(...)
		stw		r26,vS_08+8(r1)
		add		r30,r26,r19			# a2+b2 = s2[8]+l2[1]
		vadduwm	v29,v16,v28			# A + B
		stw		r27,vS_08+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[8]+l3[1]

		add		r14,r14,r28			# l0[2] += a0+b0
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		add		r17,r17,r29			# l1[2] += a1+b1
		add		r20,r20,r30			# l2[2] += a2+b2

		#------ Stage 43. Mix S[17] and L[1]
		vadduwm	v29,v26,v16			# A + B
		add		r23,r23,r31			# l3[2] += a3+b3
		rotlw	r14,r14,r28			# b0 = l0[2] = ROTL(...)
		vadduwm	v17,v17,v29			# S[17] += (A + B)
		rotlw	r17,r17,r29			# b1 = l1[2] = ROTL(...)
		rotlw	r20,r20,r30			# b2 = l2[2] = ROTL(...)
		vrlw	v17,v17,v31			# A = S[17] = ROTL3(...)
		rotlw	r23,r23,r31			# b3 = l3[2] = ROTL(...)
		add		r28,r24,r14			# a0+b0 = s0[8]+l0[2]
		vadduwm	v29,v17,v26			# A + B
		add		r29,r25,r17			# a1+b1 = s1[8]+l1[2]
		add		r30,r26,r20			# a2+b2 = s2[8]+l2[2]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[8]+l3[2]
		add		r24,r28,r11			# s0[9] += a0+b0
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		add		r25,r29,r11			# s1[9] += a1+b1
		add		r26,r30,r11			# s2[9] += a2+b2

		#------ Stage 44. Mix S[18] and L[2]
		vadduwm	v29,v27,v17			# A + B
		add		r27,r31,r11			# s3[9] += a3+b3
		add		r11,r11,r3			# s[10] = P + Q * 10
		vadduwm	v18,v18,v29			# S[18] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[9] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[9] = ROTL3(...)
		vrlw	v18,v18,v31			# A = S[18] = ROTL3(...)
		rotlwi	r26,r26,3			# a2 = s2[9] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[9] = ROTL3(...)
		vadduwm	v29,v18,v27			# A + B
		stw		r24,vS_09(r1)
		add		r28,r24,r14			# a0+b0 = s0[9]+l0[2]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		stw		r25,vS_09+4(r1)
		add		r29,r25,r17			# a1+b1 = s1[9]+l1[2]
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		stw		r26,vS_09+8(r1)
		add		r30,r26,r20			# a2+b2 = s2[9]+l2[2]

		#------ Stage 45. Mix S[19] and L[0]
		vadduwm	v29,v28,v18			# A + B
		stw		r27,vS_09+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v19,v19,v29			# S[19] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[9]+l3[2]

		add		r12,r12,r28			# l0[0] += a0+b0
		vrlw	v19,v19,v31			# A = S[19] = ROTL3(...)
		add		r15,r15,r29			# l1[0] += a1+b1
		add		r18,r18,r30			# l2[0] += a2+b2
		vadduwm	v29,v19,v28			# A + B
		add		r21,r21,r31			# l3[0] += a3+b3
		rotlw	r12,r12,r28			# b0 = l0[0] = ROTL(...)
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		rotlw	r15,r15,r29			# b1 = l1[0] = ROTL(...)
		rotlw	r18,r18,r30			# b2 = l2[0] = ROTL(...)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		rotlw	r21,r21,r31			# b3 = l3[0] = ROTL(...)
		add		r28,r24,r12			# a0+b0 = s0[9]+l0[0]

		#------ Stage 46. Mix S[20] and L[1]
		vadduwm	v29,v26,v19			# A + B
		add		r29,r25,r15			# a1+b1 = s1[9]+l1[0]
		add		r30,r26,r18			# a2+b2 = s2[9]+l2[0]
		vadduwm	v20,v20,v29			# S[20] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[9]+l3[0]
		add		r24,r28,r11			# s0[10] += a0+b0
		vrlw	v20,v20,v31			# A = S[20] = ROTL3(...)
		add		r25,r29,r11			# s1[10] += a1+b1
		add		r26,r30,r11			# s2[10] += a2+b2
		vadduwm	v29,v20,v26			# A + B
		add		r27,r31,r11			# s3[10] += a3+b3
		add		r11,r11,r3			# s[11] = P + Q * 11
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[10] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[10] = ROTL3(...)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		rotlwi	r26,r26,3			# a2 = s2[10] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[10] = ROTL3(...)

		#------ Stage 47. Mix S[21] and L[2]
		vadduwm	v29,v27,v20			# A + B
		stw		r24,vS_10(r1)
		add		r28,r24,r12			# a0+b0 = s0[10]+l0[0]
		vadduwm	v21,v21,v29			# S[21] += (A + B)
		stw		r25,vS_10+4(r1)
		add		r29,r25,r15			# a1+b1 = s1[10]+l1[0]
		vrlw	v21,v21,v31			# A = S[21] = ROTL3(...)
		stw		r26,vS_10+8(r1)
		add		r30,r26,r18			# a2+b2 = s2[10]+l2[0]
		vadduwm	v29,v21,v27			# A + B
		stw		r27,vS_10+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[10]+l3[0]

		add		r13,r13,r28			# l0[1] += a0+b0
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		add		r16,r16,r29			# l1[1] += a1+b1
		add		r19,r19,r30			# l2[1] += a2+b2

		#------ Stage 48. Mix S[22] and L[0]
		vadduwm	v29,v28,v21			# A + B
		add		r22,r22,r31			# l3[1] += a3+b3
		rotlw	r13,r13,r28			# b0 = l0[1] = ROTL(...)
		vadduwm	v22,v22,v29			# S[22] += (A + B)
		rotlw	r16,r16,r29			# b1 = l1[1] = ROTL(...)
		rotlw	r19,r19,r30			# b2 = l2[1] = ROTL(...)
		vrlw	v22,v22,v31			# A = S[22] = ROTL3(...)
		rotlw	r22,r22,r31			# b3 = l3[1] = ROTL(...)
		add		r28,r24,r13			# a0+b0 = s0[10]+l0[1]
		vadduwm	v29,v22,v28			# A + B
		add		r29,r25,r16			# a1+b1 = s1[10]+l1[1]
		add		r30,r26,r19			# a2+b2 = s2[10]+l2[1]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[10]+l3[1]
		add		r24,r28,r11			# s0[11] += a0+b0
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		add		r25,r29,r11			# s1[11] += a1+b1
		add		r26,r30,r11			# s2[11] += a2+b2

		#------ Stage 49. Mix S[23] and L[1]
		vadduwm	v29,v26,v22			# A + B
		add		r27,r31,r11			# s3[11] += a3+b3
		add		r11,r11,r3			# s[12] = P + Q * 12
		vadduwm	v23,v23,v29			# S[23] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[11] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[11] = ROTL3(...)
		vrlw	v23,v23,v31			# A = S[23] = ROTL3(...)
		rotlwi	r26,r26,3			# a2 = s2[11] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[11] = ROTL3(...)
		vadduwm	v29,v23,v26			# A + B
		stw		r24,vS_11(r1)
		add		r28,r24,r13			# a0+b0 = s0[11]+l0[1]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		stw		r25,vS_11+4(r1)
		add		r29,r25,r16			# a1+b1 = s1[11]+l1[1]
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		stw		r26,vS_11+8(r1)
		add		r30,r26,r19			# a2+b2 = s2[11]+l2[1]

		#------ Stage 50. Mix S[24] and L[2]
		vadduwm	v29,v27,v23			# A + B
		stw		r27,vS_11+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v24,v24,v29			# S[24] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[11]+l3[1]

		add		r14,r14,r28			# l0[2] += a0+b0
		vrlw	v24,v24,v31			# A = S[24] = ROTL3(...)
		add		r17,r17,r29			# l1[2] += a1+b1
		add		r20,r20,r30			# l2[2] += a2+b2
		vadduwm	v29,v24,v27			# A + B
		add		r23,r23,r31			# l3[2] += a3+b3
		rotlw	r14,r14,r28			# b0 = l0[2] = ROTL(...)
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		rotlw	r17,r17,r29			# b1 = l1[2] = ROTL(...)
		rotlw	r20,r20,r30			# b2 = l2[2] = ROTL(...)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		rotlw	r23,r23,r31			# b3 = l3[2] = ROTL(...)
		add		r28,r24,r14			# a0+b0 = s0[11]+l0[2]

		#------ Stage 51. Mix S[25] and L[0]
		vadduwm	v29,v28,v24			# A + B
		add		r29,r25,r17			# a1+b1 = s1[11]+l1[2]
		add		r30,r26,r20			# a2+b2 = s2[11]+l2[2]
		vadduwm	v25,v25,v29			# S[25] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[11]+l3[2]
		add		r24,r28,r11			# s0[12] += a0+b0
		vrlw	v25,v25,v31			# A = S[25] = ROTL3(...)
		add		r25,r29,r11			# s1[12] += a1+b1
		add		r26,r30,r11			# s2[12] += a2+b2
		vadduwm	v29,v25,v28			# A + B
		add		r27,r31,r11			# s3[12] += a3+b3
		add		r11,r11,r3			# s[13] = P + Q * 13
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[12] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[12] = ROTL3(...)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		rotlwi	r26,r26,3			# a2 = s2[12] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[12] = ROTL3(...)

		#------ Stage 52. Mix S[0] and L[1]
		vadduwm	v29,v26,v25			# A + B
		stw		r24,vS_12(r1)
		add		r28,r24,r14			# a0+b0 = s0[12]+l0[2]
		vadduwm	v0,v0,v29			# S[0] += (A + B)
		stw		r25,vS_12+4(r1)
		add		r29,r25,r17			# a1+b1 = s1[12]+l1[2]
		vrlw	v0,v0,v31			# A = S[0] = ROTL3(...)
		stw		r26,vS_12+8(r1)
		add		r30,r26,r20			# a2+b2 = s2[12]+l2[2]
		vadduwm	v29,v0,v26			# A + B
		stw		r27,vS_12+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[12]+l3[2]

		add		r12,r12,r28			# l0[0] += a0+b0
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		add		r15,r15,r29			# l1[0] += a1+b1
		add		r18,r18,r30			# l2[0] += a2+b2

		#------ Stage 53. Mix S[1] and L[2]
		vadduwm	v29,v27,v0			# A + B
		add		r21,r21,r31			# l3[0] += a3+b3
		rotlw	r12,r12,r28			# b0 = l0[0] = ROTL(...)
		vadduwm	v1,v1,v29			# S[1] += (A + B)
		rotlw	r15,r15,r29			# b1 = l1[0] = ROTL(...)
		rotlw	r18,r18,r30			# b2 = l2[0] = ROTL(...)
		vrlw	v1,v1,v31			# A = S[1] = ROTL3(...)
		rotlw	r21,r21,r31			# b3 = l3[0] = ROTL(...)
		add		r28,r24,r12			# a0+b0 = s0[12]+l0[0]
		vadduwm	v29,v1,v27			# A + B
		add		r29,r25,r15			# a1+b1 = s1[12]+l1[0]
		add		r30,r26,r18			# a2+b2 = s2[12]+l2[0]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[12]+l3[0]
		add		r24,r28,r11			# s0[13] += a0+b0
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		add		r25,r29,r11			# s1[13] += a1+b1
		add		r26,r30,r11			# s2[13] += a2+b2

		#------ Stage 54. Mix S[2] and L[0]
		vadduwm	v29,v28,v1			# A + B
		add		r27,r31,r11			# s3[13] += a3+b3
		add		r11,r11,r3			# s[14] = P + Q * 14
		vadduwm	v2,v2,v29			# S[2] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[13] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[13] = ROTL3(...)
		vrlw	v2,v2,v31			# A = S[2] = ROTL3(...)
		rotlwi	r26,r26,3			# a2 = s2[13] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[13] = ROTL3(...)
		vadduwm	v29,v2,v28			# A + B
		stw		r24,vS_13(r1)
		add		r28,r24,r12			# a0+b0 = s0[13]+l0[0]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		stw		r25,vS_13+4(r1)
		add		r29,r25,r15			# a1+b1 = s1[13]+l1[0]
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		stw		r26,vS_13+8(r1)
		add		r30,r26,r18			# a2+b2 = s2[13]+l2[0]

		#------ Stage 55. Mix S[3] and L[1]
		vadduwm	v29,v26,v2			# A + B
		stw		r27,vS_13+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v3,v3,v29			# S[3] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[13]+l3[0]

		add		r13,r13,r28			# l0[1] += a0+b0
		vrlw	v3,v3,v31			# A = S[3] = ROTL3(...)
		add		r16,r16,r29			# l1[1] += a1+b1
		add		r19,r19,r30			# l2[1] += a2+b2
		vadduwm	v29,v3,v26			# A + B
		add		r22,r22,r31			# l3[1] += a3+b3
		rotlw	r13,r13,r28			# b0 = l0[1] = ROTL(...)
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		rotlw	r16,r16,r29			# b1 = l1[1] = ROTL(...)
		rotlw	r19,r19,r30			# b2 = l2[1] = ROTL(...)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		rotlw	r22,r22,r31			# b3 = l3[1] = ROTL(...)
		add		r28,r24,r13			# a0+b0 = s0[13]+l0[1]

		#------ Stage 56. Mix S[4] and L[2]
		vadduwm	v29,v27,v3			# A + B
		add		r29,r25,r16			# a1+b1 = s1[13]+l1[1]
		add		r30,r26,r19			# a2+b2 = s2[13]+l2[1]
		vadduwm	v4,v4,v29			# S[4] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[13]+l3[1]
		add		r24,r28,r11			# s0[14] += a0+b0
		vrlw	v4,v4,v31			# A = S[4] = ROTL3(...)
		add		r25,r29,r11			# s1[14] += a1+b1
		add		r26,r30,r11			# s2[14] += a2+b2
		vadduwm	v29,v4,v27			# A + B
		add		r27,r31,r11			# s3[14] += a3+b3
		add		r11,r11,r3			# s[15] = P + Q * 15
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[14] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[14] = ROTL3(...)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		rotlwi	r26,r26,3			# a2 = s2[14] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[14] = ROTL3(...)

		#------ Stage 57. Mix S[5] and L[0]
		vadduwm	v29,v28,v4			# A + B
		stw		r24,vS_14(r1)
		add		r28,r24,r13			# a0+b0 = s0[14]+l0[1]
		vadduwm	v5,v5,v29			# S[5] += (A + B)
		stw		r25,vS_14+4(r1)
		add		r29,r25,r16			# a1+b1 = s1[14]+l1[1]
		vrlw	v5,v5,v31			# A = S[5] = ROTL3(...)
		stw		r26,vS_14+8(r1)
		add		r30,r26,r19			# a2+b2 = s2[14]+l2[1]
		vadduwm	v29,v5,v28			# A + B
		stw		r27,vS_14+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[14]+l3[1]

		add		r14,r14,r28			# l0[2] += a0+b0
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		add		r17,r17,r29			# l1[2] += a1+b1
		add		r20,r20,r30			# l2[2] += a2+b2

		#------ Stage 58. Mix S[6] and L[1]
		vadduwm	v29,v26,v5			# A + B
		add		r23,r23,r31			# l3[2] += a3+b3
		rotlw	r14,r14,r28			# b0 = l0[2] = ROTL(...)
		vadduwm	v6,v6,v29			# S[6] += (A + B)
		rotlw	r17,r17,r29			# b1 = l1[2] = ROTL(...)
		rotlw	r20,r20,r30			# b2 = l2[2] = ROTL(...)
		vrlw	v6,v6,v31			# A = S[6] = ROTL3(...)
		rotlw	r23,r23,r31			# b3 = l3[2] = ROTL(...)
		add		r28,r24,r14			# a0+b0 = s0[14]+l0[2]
		vadduwm	v29,v6,v26			# A + B
		add		r29,r25,r17			# a1+b1 = s1[14]+l1[2]
		add		r30,r26,r20			# a2+b2 = s2[14]+l2[2]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[14]+l3[2]
		add		r24,r28,r11			# s0[15] += a0+b0
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		add		r25,r29,r11			# s1[15] += a1+b1
		add		r26,r30,r11			# s2[15] += a2+b2

		#------ Stage 59. Mix S[7] and L[2]
		vadduwm	v29,v27,v6			# A + B
		add		r27,r31,r11			# s3[15] += a3+b3
		add		r11,r11,r3			# s[16] = P + Q * 16
		vadduwm	v7,v7,v29			# S[7] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[15] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[15] = ROTL3(...)
		vrlw	v7,v7,v31			# A = S[7] = ROTL3(...)
		rotlwi	r26,r26,3			# a2 = s2[15] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[15] = ROTL3(...)
		vadduwm	v29,v7,v27			# A + B
		stw		r24,vS_15(r1)
		add		r28,r24,r14			# a0+b0 = s0[15]+l0[2]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		stw		r25,vS_15+4(r1)
		add		r29,r25,r17			# a1+b1 = s1[15]+l1[2]
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		stw		r26,vS_15+8(r1)
		add		r30,r26,r20			# a2+b2 = s2[15]+l2[2]

		#------ Stage 60. Mix S[8] and L[0]
		vadduwm	v29,v28,v7			# A + B
		stw		r27,vS_15+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v8,v8,v29			# S[8] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[15]+l3[2]

		add		r12,r12,r28			# l0[0] += a0+b0
		vrlw	v8,v8,v31			# A = S[8] = ROTL3(...)
		add		r15,r15,r29			# l1[0] += a1+b1
		add		r18,r18,r30			# l2[0] += a2+b2
		vadduwm	v29,v8,v28			# A + B
		add		r21,r21,r31			# l3[0] += a3+b3
		rotlw	r12,r12,r28			# b0 = l0[0] = ROTL(...)
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		rotlw	r15,r15,r29			# b1 = l1[0] = ROTL(...)
		rotlw	r18,r18,r30			# b2 = l2[0] = ROTL(...)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		rotlw	r21,r21,r31			# b3 = l3[0] = ROTL(...)
		add		r28,r24,r12			# a0+b0 = s0[15]+l0[0]

		#------ Stage 61. Mix S[9] and L[1]
		vadduwm	v29,v26,v8			# A + B
		add		r29,r25,r15			# a1+b1 = s1[15]+l1[0]
		add		r30,r26,r18			# a2+b2 = s2[15]+l2[0]
		vadduwm	v9,v9,v29			# S[9] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[15]+l3[0]
		add		r24,r28,r11			# s0[16] += a0+b0
		vrlw	v9,v9,v31			# A = S[9] = ROTL3(...)
		add		r25,r29,r11			# s1[16] += a1+b1
		add		r26,r30,r11			# s2[16] += a2+b2
		vadduwm	v29,v9,v26			# A + B
		add		r27,r31,r11			# s3[16] += a3+b3
		add		r11,r11,r3			# s[17] = P + Q * 17
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[16] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[16] = ROTL3(...)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		rotlwi	r26,r26,3			# a2 = s2[16] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[16] = ROTL3(...)

		#------ Stage 62. Mix S[10] and L[2]
		vadduwm	v29,v27,v9			# A + B
		stw		r24,vS_16(r1)
		add		r28,r24,r12			# a0+b0 = s0[16]+l0[0]
		vadduwm	v10,v10,v29			# S[10] += (A + B)
		stw		r25,vS_16+4(r1)
		add		r29,r25,r15			# a1+b1 = s1[16]+l1[0]
		vrlw	v10,v10,v31			# A = S[10] = ROTL3(...)
		stw		r26,vS_16+8(r1)
		add		r30,r26,r18			# a2+b2 = s2[16]+l2[0]
		vadduwm	v29,v10,v27			# A + B
		stw		r27,vS_16+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[16]+l3[0]

		add		r13,r13,r28			# l0[1] += a0+b0
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		add		r16,r16,r29			# l1[1] += a1+b1
		add		r19,r19,r30			# l2[1] += a2+b2

		#------ Stage 63. Mix S[11] and L[0]
		vadduwm	v29,v28,v10			# A + B
		add		r22,r22,r31			# l3[1] += a3+b3
		rotlw	r13,r13,r28			# b0 = l0[1] = ROTL(...)
		vadduwm	v11,v11,v29			# S[11] += (A + B)
		rotlw	r16,r16,r29			# b1 = l1[1] = ROTL(...)
		rotlw	r19,r19,r30			# b2 = l2[1] = ROTL(...)
		vrlw	v11,v11,v31			# A = S[11] = ROTL3(...)
		rotlw	r22,r22,r31			# b3 = l3[1] = ROTL(...)
		add		r28,r24,r13			# a0+b0 = s0[16]+l0[1]
		vadduwm	v29,v11,v28			# A + B
		add		r29,r25,r16			# a1+b1 = s1[16]+l1[1]
		add		r30,r26,r19			# a2+b2 = s2[16]+l2[1]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[16]+l3[1]
		add		r24,r28,r11			# s0[17] += a0+b0
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		add		r25,r29,r11			# s1[17] += a1+b1
		add		r26,r30,r11			# s2[17] += a2+b2

		#------ Stage 64. Mix S[12] and L[1]
		vadduwm	v29,v26,v11			# A + B
		add		r27,r31,r11			# s3[17] += a3+b3
		add		r11,r11,r3			# s[18] = P + Q * 18
		vadduwm	v12,v12,v29			# S[12] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[17] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[17] = ROTL3(...)
		vrlw	v12,v12,v31			# A = S[12] = ROTL3(...)
		rotlwi	r26,r26,3			# a2 = s2[17] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[17] = ROTL3(...)
		vadduwm	v29,v12,v26			# A + B
		stw		r24,vS_17(r1)
		add		r28,r24,r13			# a0+b0 = s0[17]+l0[1]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		stw		r25,vS_17+4(r1)
		add		r29,r25,r16			# a1+b1 = s1[17]+l1[1]
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		stw		r26,vS_17+8(r1)
		add		r30,r26,r19			# a2+b2 = s2[17]+l2[1]

		#------ Stage 65. Mix S[13] and L[2]
		vadduwm	v29,v27,v12			# A + B
		stw		r27,vS_17+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v13,v13,v29			# S[13] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[17]+l3[1]

		add		r14,r14,r28			# l0[2] += a0+b0
		vrlw	v13,v13,v31			# A = S[13] = ROTL3(...)
		add		r17,r17,r29			# l1[2] += a1+b1
		add		r20,r20,r30			# l2[2] += a2+b2
		vadduwm	v29,v13,v27			# A + B
		add		r23,r23,r31			# l3[2] += a3+b3
		rotlw	r14,r14,r28			# b0 = l0[2] = ROTL(...)
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		rotlw	r17,r17,r29			# b1 = l1[2] = ROTL(...)
		rotlw	r20,r20,r30			# b2 = l2[2] = ROTL(...)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		rotlw	r23,r23,r31			# b3 = l3[2] = ROTL(...)
		add		r28,r24,r14			# a0+b0 = s0[17]+l0[2]

		#------ Stage 66. Mix S[14] and L[0]
		vadduwm	v29,v28,v13			# A + B
		add		r29,r25,r17			# a1+b1 = s1[17]+l1[2]
		add		r30,r26,r20			# a2+b2 = s2[17]+l2[2]
		vadduwm	v14,v14,v29			# S[14] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[17]+l3[2]
		add		r24,r28,r11			# s0[18] += a0+b0
		vrlw	v14,v14,v31			# A = S[14] = ROTL3(...)
		add		r25,r29,r11			# s1[18] += a1+b1
		add		r26,r30,r11			# s2[18] += a2+b2
		vadduwm	v29,v14,v28			# A + B
		add		r27,r31,r11			# s3[18] += a3+b3
		add		r11,r11,r3			# s[19] = P + Q * 19
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[18] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[18] = ROTL3(...)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		rotlwi	r26,r26,3			# a2 = s2[18] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[18] = ROTL3(...)

		#------ Stage 67. Mix S[15] and L[1]
		vadduwm	v29,v26,v14			# A + B
		stw		r24,vS_18(r1)
		add		r28,r24,r14			# a0+b0 = s0[18]+l0[2]
		vadduwm	v15,v15,v29			# S[15] += (A + B)
		stw		r25,vS_18+4(r1)
		add		r29,r25,r17			# a1+b1 = s1[18]+l1[2]
		vrlw	v15,v15,v31			# A = S[15] = ROTL3(...)
		stw		r26,vS_18+8(r1)
		add		r30,r26,r20			# a2+b2 = s2[18]+l2[2]
		vadduwm	v29,v15,v26			# A + B
		stw		r27,vS_18+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[18]+l3[2]

		add		r12,r12,r28			# l0[0] += a0+b0
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		add		r15,r15,r29			# l1[0] += a1+b1
		add		r18,r18,r30			# l2[0] += a2+b2

		#------ Stage 68. Mix S[16] and L[2]
		vadduwm	v29,v27,v15			# A + B
		add		r21,r21,r31			# l3[0] += a3+b3
		rotlw	r12,r12,r28			# b0 = l0[0] = ROTL(...)
		vadduwm	v16,v16,v29			# S[16] += (A + B)
		rotlw	r15,r15,r29			# b1 = l1[0] = ROTL(...)
		rotlw	r18,r18,r30			# b2 = l2[0] = ROTL(...)
		vrlw	v16,v16,v31			# A = S[16] = ROTL3(...)
		rotlw	r21,r21,r31			# b3 = l3[0] = ROTL(...)
		add		r28,r24,r12			# a0+b0 = s0[18]+l0[0]
		vadduwm	v29,v16,v27			# A + B
		add		r29,r25,r15			# a1+b1 = s1[18]+l1[0]
		add		r30,r26,r18			# a2+b2 = s2[18]+l2[0]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[18]+l3[0]
		add		r24,r28,r11			# s0[19] += a0+b0
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		add		r25,r29,r11			# s1[19] += a1+b1
		add		r26,r30,r11			# s2[19] += a2+b2

		#------ Stage 69. Mix S[17] and L[0]
		vadduwm	v29,v28,v16			# A + B
		add		r27,r31,r11			# s3[19] += a3+b3
		add		r11,r11,r3			# s[20] = P + Q * 20
		vadduwm	v17,v17,v29			# S[17] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[19] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[19] = ROTL3(...)
		vrlw	v17,v17,v31			# A = S[17] = ROTL3(...)
		rotlwi	r26,r26,3			# a2 = s2[19] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[19] = ROTL3(...)
		vadduwm	v29,v17,v28			# A + B
		stw		r24,vS_19(r1)
		add		r28,r24,r12			# a0+b0 = s0[19]+l0[0]
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		stw		r25,vS_19+4(r1)
		add		r29,r25,r15			# a1+b1 = s1[19]+l1[0]
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		stw		r26,vS_19+8(r1)
		add		r30,r26,r18			# a2+b2 = s2[19]+l2[0]

		#------ Stage 70. Mix S[18] and L[1]
		vadduwm	v29,v26,v17			# A + B
		stw		r27,vS_19+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v18,v18,v29			# S[18] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[19]+l3[0]

		add		r13,r13,r28			# l0[1] += a0+b0
		vrlw	v18,v18,v31			# A = S[18] = ROTL3(...)
		add		r16,r16,r29			# l1[1] += a1+b1
		add		r19,r19,r30			# l2[1] += a2+b2
		vadduwm	v29,v18,v26			# A + B
		add		r22,r22,r31			# l3[1] += a3+b3
		rotlw	r13,r13,r28			# b0 = l0[1] = ROTL(...)
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		rotlw	r16,r16,r29			# b1 = l1[1] = ROTL(...)
		rotlw	r19,r19,r30			# b2 = l2[1] = ROTL(...)
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		rotlw	r22,r22,r31			# b3 = l3[1] = ROTL(...)
		add		r28,r24,r13			# a0+b0 = s0[19]+l0[1]

		#------ Stage 71. Mix S[19] and L[2]
		vadduwm	v29,v27,v18			# A + B
		add		r29,r25,r16			# a1+b1 = s1[19]+l1[1]
		add		r30,r26,r19			# a2+b2 = s2[19]+l2[1]
		vadduwm	v19,v19,v29			# S[19] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[19]+l3[1]
		add		r24,r28,r11			# s0[20] += a0+b0
		vrlw	v19,v19,v31			# A = S[19] = ROTL3(...)
		add		r25,r29,r11			# s1[20] += a1+b1
		add		r26,r30,r11			# s2[20] += a2+b2
		vadduwm	v29,v19,v27			# A + B
		add		r27,r31,r11			# s3[20] += a3+b3
		add		r11,r11,r3			# s[21] = P + Q * 21
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[20] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[20] = ROTL3(...)
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		rotlwi	r26,r26,3			# a2 = s2[20] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[20] = ROTL3(...)

		#------ Stage 72. Mix S[20] and L[0]
		vadduwm	v29,v28,v19			# A + B
		stw		r24,vS_20(r1)
		add		r28,r24,r13			# a0+b0 = s0[20]+l0[1]
		vadduwm	v20,v20,v29			# S[20] += (A + B)
		stw		r25,vS_20+4(r1)
		add		r29,r25,r16			# a1+b1 = s1[20]+l1[1]
		vrlw	v20,v20,v31			# A = S[20] = ROTL3(...)
		stw		r26,vS_20+8(r1)
		add		r30,r26,r19			# a2+b2 = s2[20]+l2[1]
		vadduwm	v29,v20,v28			# A + B
		stw		r27,vS_20+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		add		r31,r27,r22			# a3+b3 = s3[20]+l3[1]

		add		r14,r14,r28			# l0[2] += a0+b0
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		add		r17,r17,r29			# l1[2] += a1+b1
		add		r20,r20,r30			# l2[2] += a2+b2

		#------ Stage 73. Mix S[21] and L[1]
		vadduwm	v29,v26,v20			# A + B
		add		r23,r23,r31			# l3[2] += a3+b3
		rotlw	r14,r14,r28			# b0 = l0[2] = ROTL(...)
		vadduwm	v21,v21,v29			# S[21] += (A + B)
		rotlw	r17,r17,r29			# b1 = l1[2] = ROTL(...)
		rotlw	r20,r20,r30			# b2 = l2[2] = ROTL(...)
		vrlw	v21,v21,v31			# A = S[21] = ROTL3(...)
		rotlw	r23,r23,r31			# b3 = l3[2] = ROTL(...)
		add		r28,r24,r14			# a0+b0 = s0[20]+l0[2]
		vadduwm	v29,v21,v26			# A + B
		add		r29,r25,r17			# a1+b1 = s1[20]+l1[2]
		add		r30,r26,r20			# a2+b2 = s2[20]+l2[2]
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[20]+l3[2]
		add		r24,r28,r11			# s0[21] += a0+b0
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		add		r25,r29,r11			# s1[21] += a1+b1
		add		r26,r30,r11			# s2[21] += a2+b2

		#------ Stage 74. Mix S[22] and L[2]
		vadduwm	v29,v27,v21			# A + B
		add		r27,r31,r11			# s3[21] += a3+b3
		add		r11,r11,r3			# s[22] = P + Q * 22
		vadduwm	v22,v22,v29			# S[22] += (A + B)
		rotlwi	r24,r24,3			# a0 = s0[21] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[21] = ROTL3(...)
		vrlw	v22,v22,v31			# A = S[22] = ROTL3(...)
		rotlwi	r26,r26,3			# a2 = s2[21] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[21] = ROTL3(...)
		vadduwm	v29,v22,v27			# A + B
		stw		r24,vS_21(r1)
		add		r28,r24,r14			# a0+b0 = s0[21]+l0[2]
		vadduwm	v28,v28,v29			# L[2] += (A + B)
		stw		r25,vS_21+4(r1)
		add		r29,r25,r17			# a1+b1 = s1[21]+l1[2]
		vrlw	v28,v28,v29			# B = L[2] = ROTL(...)
		stw		r26,vS_21+8(r1)
		add		r30,r26,r20			# a2+b2 = s2[21]+l2[2]

		#------ Stage 75. Mix S[23] and L[0]
		vadduwm	v29,v28,v22			# A + B
		stw		r27,vS_21+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v23,v23,v29			# S[23] += (A + B)
		add		r31,r27,r23			# a3+b3 = s3[21]+l3[2]

		add		r12,r12,r28			# l0[0] += a0+b0
		vrlw	v23,v23,v31			# A = S[23] = ROTL3(...)
		add		r15,r15,r29			# l1[0] += a1+b1
		add		r18,r18,r30			# l2[0] += a2+b2
		vadduwm	v29,v23,v28			# A + B
		add		r21,r21,r31			# l3[0] += a3+b3
		rotlw	r12,r12,r28			# b0 = l0[0] = ROTL(...)
		vadduwm	v26,v26,v29			# L[0] += (A + B)
		rotlw	r15,r15,r29			# b1 = l1[0] = ROTL(...)
		rotlw	r18,r18,r30			# b2 = l2[0] = ROTL(...)
		vrlw	v26,v26,v29			# B = L[0] = ROTL(...)
		rotlw	r21,r21,r31			# b3 = l3[0] = ROTL(...)
		add		r28,r24,r12			# a0+b0 = s0[21]+l0[0]

		#------ Stage 76. Mix S[24] and L[1]
		vadduwm	v29,v26,v23			# A + B
		add		r29,r25,r15			# a1+b1 = s1[21]+l1[0]
		add		r30,r26,r18			# a2+b2 = s2[21]+l2[0]
		vadduwm	v24,v24,v29			# S[24] += (A + B)
		add		r31,r27,r21			# a3+b3 = s3[21]+l3[0]
		add		r24,r28,r11			# s0[22] += a0+b0
		vrlw	v24,v24,v31			# A = S[24] = ROTL3(...)
		add		r25,r29,r11			# s1[22] += a1+b1
		add		r26,r30,r11			# s2[22] += a2+b2
		vadduwm	v29,v24,v26			# A + B
		vspltw	v26,v30,1			# plain.lo
		add		r27,r31,r11			# s3[22] += a3+b3
		vadduwm	v27,v27,v29			# L[1] += (A + B)
		vspltw	v28,v30,0			# plain.hi
		add		r11,r11,r3			# s[23] = P + Q * 23
		vrlw	v27,v27,v29			# B = L[1] = ROTL(...)
		rotlwi	r24,r24,3			# a0 = s0[22] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[22] = ROTL3(...)

		#------ Stage 77. Mix S[25] and L[2]
		vadduwm	v29,v27,v24			# A + B
		rotlwi	r26,r26,3			# a2 = s2[22] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[22] = ROTL3(...)
		vadduwm	v25,v25,v29			# S[25] += (A + B)
		stw		r24,vS_22(r1)
		add		r28,r24,r12			# a0+b0 = s0[22]+l0[0]
		vrlw	v25,v25,v31			# A = S[25] = ROTL3(...)
		stw		r25,vS_22+4(r1)
		add		r29,r25,r15			# a1+b1 = s1[22]+l1[0]


		vadduwm	v26,v26,v0			# A = S[0] + plain.lo
		stw		r26,vS_22+8(r1)
		add		r30,r26,r18			# a2+b2 = s2[22]+l2[0]
		vadduwm	v28,v28,v1			# B = S[1] + plain.hi
		stw		r27,vS_22+12(r1)
		# Empty slot to avoid CQ stalls.

		#------ Round 1
		vxor	v26,v26,v28			# A ^= B
		add		r31,r27,r21			# a3+b3 = s3[22]+l3[0]

		add		r13,r13,r28			# l0[1] += a0+b0
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		add		r16,r16,r29			# l1[1] += a1+b1
		add		r19,r19,r30			# l2[1] += a2+b2
		vadduwm	v26,v26,v2			# A += S[2]
		add		r22,r22,r31			# l3[1] += a3+b3
		rotlw	r13,r13,r28			# b0 = l0[1] = ROTL(...)
		vxor	v28,v28,v26			# B ^= A
		rotlw	r16,r16,r29			# b1 = l1[1] = ROTL(...)
		rotlw	r19,r19,r30			# b2 = l2[1] = ROTL(...)
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		rotlw	r22,r22,r31			# b3 = l3[1] = ROTL(...)
		add		r28,r24,r13			# a0+b0 = s0[22]+l0[1]
		vadduwm	v28,v28,v3			# B += S[3]
		add		r29,r25,r16			# a1+b1 = s1[22]+l1[1]
		add		r30,r26,r19			# a2+b2 = s2[22]+l2[1]

		#------ Round 2
		vxor	v26,v26,v28			# A ^= B
		add		r31,r27,r22			# a3+b3 = s3[22]+l3[1]
		add		r24,r28,r11			# s0[23] += a0+b0
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		add		r25,r29,r11			# s1[23] += a1+b1
		add		r26,r30,r11			# s2[23] += a2+b2
		vadduwm	v26,v26,v4			# A += S[4]
		add		r27,r31,r11			# s3[23] += a3+b3
		add		r11,r11,r3			# s[24] = P + Q * 24
		vxor	v28,v28,v26			# B ^= A
		rotlwi	r24,r24,3			# a0 = s0[23] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[23] = ROTL3(...)
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		rotlwi	r26,r26,3			# a2 = s2[23] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[23] = ROTL3(...)
		vadduwm	v28,v28,v5			# B += S[5]
		stw		r24,vS_23(r1)
		add		r28,r24,r13			# a0+b0 = s0[23]+l0[1]

		#------ Round 3
		vxor	v26,v26,v28			# A ^= B
		stw		r25,vS_23+4(r1)
		add		r29,r25,r16			# a1+b1 = s1[23]+l1[1]
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		stw		r26,vS_23+8(r1)
		add		r30,r26,r19			# a2+b2 = s2[23]+l2[1]
		vadduwm	v26,v26,v6			# A += S[6]
		stw		r27,vS_23+12(r1)
		# Empty slot to avoid CQ stalls.
		vxor	v28,v28,v26			# B ^= A
		add		r31,r27,r22			# a3+b3 = s3[23]+l3[1]

		add		r14,r14,r28			# l0[2] += a0+b0
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		add		r17,r17,r29			# l1[2] += a1+b1
		add		r20,r20,r30			# l2[2] += a2+b2
		vadduwm	v28,v28,v7			# B += S[7]
		add		r23,r23,r31			# l3[2] += a3+b3
		rotlw	r14,r14,r28			# b0 = l0[2] = ROTL(...)

		#------ Round 4
		vxor	v26,v26,v28			# A ^= B
		rotlw	r17,r17,r29			# b1 = l1[2] = ROTL(...)
		rotlw	r20,r20,r30			# b2 = l2[2] = ROTL(...)
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		rotlw	r23,r23,r31			# b3 = l3[2] = ROTL(...)
		add		r28,r24,r14			# a0+b0 = s0[23]+l0[2]
		vadduwm	v26,v26,v8			# A += S[8]
		add		r29,r25,r17			# a1+b1 = s1[23]+l1[2]
		add		r30,r26,r20			# a2+b2 = s2[23]+l2[2]
		vxor	v28,v28,v26			# B ^= A
		add		r31,r27,r23			# a3+b3 = s3[23]+l3[2]
		add		r24,r28,r11			# s0[24] += a0+b0
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		stw		r14,vL_02(r1)		# save l0[2]
		add		r25,r29,r11			# s1[24] += a1+b1
		vadduwm	v28,v28,v9			# B += S[9]
		stw		r17,vL_02+4(r1)		# save l1[2]
		add		r26,r30,r11			# s2[24] += a2+b2

		#------ Round 5
		vxor	v26,v26,v28			# A ^= B
		stw		r20,vL_02+8(r1)		# save l2[2]
		add		r27,r31,r11			# s3[24] += a3+b3
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		stw		r23,vL_02+12(r1)	# save l3[2]
		add		r11,r11,r3			# s[25] = P + Q * 25
		vadduwm	v26,v26,v10			# A += S[10]
		rotlwi	r24,r24,3			# a0 = s0[24] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[24] = ROTL3(...)
		vxor	v28,v28,v26			# B ^= A
		rotlwi	r26,r26,3			# a2 = s2[24] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[24] = ROTL3(...)
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		stw		r24,vS_24(r1)
		add		r28,r24,r14			# a0+b0 = s0[24]+l0[2]
		vadduwm	v28,v28,v11			# B += S[11]
		stw		r25,vS_24+4(r1)
		add		r29,r25,r17			# a1+b1 = s1[24]+l1[2]

		#------ Round 6
		vxor	v26,v26,v28			# A ^= B
		stw		r26,vS_24+8(r1)
		add		r30,r26,r20			# a2+b2 = s2[24]+l2[2]
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		stw		r27,vS_24+12(r1)
		# Empty slot to avoid CQ stalls.
		vadduwm	v26,v26,v12			# A += S[12]
		add		r31,r27,r23			# a3+b3 = s3[24]+l3[2]

		add		r12,r12,r28			# l0[0] += a0+b0
		vxor	v28,v28,v26			# B ^= A
		add		r15,r15,r29			# l1[0] += a1+b1
		add		r18,r18,r30			# l2[0] += a2+b2
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		add		r21,r21,r31			# l3[0] += a3+b3
		rotlw	r12,r12,r28			# b0 = l0[0] = ROTL(...)
		vadduwm	v28,v28,v13			# B += S[13]
		rotlw	r15,r15,r29			# b1 = l1[0] = ROTL(...)
		rotlw	r18,r18,r30			# b2 = l2[0] = ROTL(...)

		#------ Round 7
		vxor	v26,v26,v28			# A ^= B
		rotlw	r21,r21,r31			# b3 = l3[0] = ROTL(...)
		add		r28,r24,r12			# a0+b0 = s0[24]+l0[0]
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		add		r29,r25,r15			# a1+b1 = s1[24]+l1[0]
		add		r30,r26,r18			# a2+b2 = s2[24]+l2[0]
		vadduwm	v26,v26,v14			# A += S[14]
		add		r31,r27,r21			# a3+b3 = s3[24]+l3[0]
		add		r24,r28,r11			# s0[25] += a0+b0
		vxor	v28,v28,v26			# B ^= A
		stw		r12,vL_00(r1)		# save l0[0]
		add		r25,r29,r11			# s1[25] += a1+b1
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		stw		r15,vL_00+4(r1)		# save l1[0]
		add		r26,r30,r11			# s2[25] += a2+b2
		vadduwm	v28,v28,v15			# B += S[15]
		stw		r18,vL_00+8(r1)		# save l2[0]
		add		r27,r31,r11			# s3[25] += a3+b3

		#------ Round 8
		vxor	v26,v26,v28			# A ^= B
		stw		r21,vL_00+12(r1)	# save l3[0]
		add		r11,r11,r3			# s[0] = P + Q * 0
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		rotlwi	r24,r24,3			# a0 = s0[25] = ROTL3(...)
		rotlwi	r25,r25,3			# a1 = s1[25] = ROTL3(...)
		vadduwm	v26,v26,v16			# A += S[16]
		rotlwi	r26,r26,3			# a2 = s2[25] = ROTL3(...)
		rotlwi	r27,r27,3			# a3 = s3[25] = ROTL3(...)
		vxor	v28,v28,v26			# B ^= A
		stw		r24,vS_25(r1)
		add		r28,r24,r12			# a0+b0 = s0[25]+l0[0]
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		stw		r25,vS_25+4(r1)
		add		r29,r25,r15			# a1+b1 = s1[25]+l1[0]
		vadduwm	v28,v28,v17			# B += S[17]
		stw		r26,vS_25+8(r1)
		add		r30,r26,r18			# a2+b2 = s2[25]+l2[0]

		#------ Round 9
		vxor	v26,v26,v28			# A ^= B
		stw		r27,vS_25+12(r1)
		# Empty slot to avoid CQ stalls.
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		add		r31,r27,r21			# a3+b3 = s3[25]+l3[0]

		add		r13,r13,r28			# l0[1] += a0+b0
		vadduwm	v26,v26,v18			# A += S[18]
		add		r16,r16,r29			# l1[1] += a1+b1
		add		r19,r19,r30			# l2[1] += a2+b2
		vxor	v28,v28,v26			# B ^= A
		add		r22,r22,r31			# l3[1] += a3+b3
		rotlw	r13,r13,r28			# b0 = l0[1] = ROTL(...)
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		rotlw	r16,r16,r29			# b1 = l1[1] = ROTL(...)
		rotlw	r19,r19,r30			# b2 = l2[1] = ROTL(...)
		vadduwm	v28,v28,v19			# B += S[19]
		rotlw	r22,r22,r31			# b3 = l3[1] = ROTL(...)
		stw		r13,vL_01+0(r1)		# save l0[1]

		#------ Round 10
		vxor	v26,v26,v28			# A ^= B
		vspltw	v27,v30,3			# cypher.lo
		stw		r16,vL_01+4(r1)		# save l1[1]
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		addi	r17,r1,vKey			# &vKey (AltiVec key)
		stw		r19,vL_01+8(r1)		# save l2[1]
		vadduwm	v26,v26,v20			# A += S[20]
		addi	r18,r1,key_hi		# &key_hi
		stw		r22,vL_01+12(r1)	# save l3[1]
		vxor	v28,v28,v26			# B ^= A
		li		r15,4
		lwbrx	r28,0,r18			# key_hi
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		li		r16,8
		lvx		v1,r8,r0			# pre-load S[1]
		vadduwm	v28,v28,v21			# B += S[21]
		lis		r22,0x0400
		lvx		v2,r8,r4			# pre-load S[2]

		#------ Round 11
		vxor	v26,v26,v28			# A ^= B
		lwbrx	r25,r16,r18			# key_lo

		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		lwbrx	r29,r15,r18			# key_mid

		vadduwm	v26,v26,v22			# A += S[22]
		lvx		 v5,r7,r0			# vKey

		vxor	v28,v28,v26			# B ^= A
		stwbrx	r28,0,r17			# vKey:key.hi
		addc	r28,r28,r22			# 72-bit key += 4
		vrlw	v28,v28,v26			# B = ROTL(B ^ A, A)
		stwbrx	r29,r15,r17			# vKey:key.mid
		addze	r29,r29
		vadduwm	v28,v28,v23			# B += S[23]
		stwbrx	r25,r16,r17			# vKey:key.lo
		addze	r30,r25

		#------ Round 12
		vxor	v26,v26,v28			# A ^= B
		lis		r11,(P+3*Q)@h
		stwbrx	r28,0,r18			# save key_hi
		vrlw	v26,v26,v28			# A = ROTL(A ^ B, B)
		ori		r11,r11,(P+3*Q)@l	# s[3]
		lwz		r22,L1_cached(r1)
		vadduwm	v26,v26,v24			# A += S[24]
		cmp		cr5,r25,r30			# key.lo has changed ?
		lwz		r24,S2_cached(r1)
		vcmpequw. v4,v27,v26
		cmpi	cr7,r28,0			# key.mid has changed ?
		lwz		r14,key_hi(r1)

		bne-	cr6,check_keys		# At least one key matches
		

next_keys:
		lwz		r21,L0_cached(r1)

		bne-	cr5,new_key_lo
		beq-	cr7,new_key_mid
		
		bdnz	inner_loop_hi
		
		li		r3,RESULT_NOTHING	# Not found.
		b		epilog


	# key.lo has been changed (also implies a new key.mid)

new_key_lo:
		stwbrx	r30,r16,r18			# save key.lo
		stwbrx	r29,r15,r18			# save key.mid

		lwz		r12,key_lo(r1)
		lwz		r13,key_mid(r1)
		bdnz	inner_loop_lo
		
		li		r3,RESULT_NOTHING	# Not found
		b		epilog

	
	# key.mid has been changed.
	
new_key_mid:
		stwbrx	r29,r15,r18			# save key.mid

		lis		r11,(P+2*Q)@h		# s[2]
		lwz		r13,key_mid(r1)
		ori		r11,r11,(P+2*Q)@l
		lwz		r24,S1_cached(r1)
		bdnz	inner_loop_mid

		li		r3,RESULT_NOTHING	# Not found
		b		epilog


	# Update counter measure checks (big tricks involved).
	# Each element of v4 is set to either 0 (FALSE) or -1 (TRUE),
	# so the sum of these elements can be used to update the
	# "check.count" value. But the comparison between B and
	# "cypher.hi" comes into play because we shall not count
	# any false positive past the first real matching key.
	# Note that B can also have false positives, which might be
	# different than the false positives in A.
				
check_keys:
		vxor	 v28,v28,v26		# B ^= A
		vspltw	 v3,v30,2			# cypher.hi
		vrlw	 v28,v28,v26		# B = ROTL(B ^ A, A)
		vspltw	 v7,v5,0			# key.hi
		vadduwm	 v28,v28,v25		# B += S[25]
		vspltisw v6,0
		vcmpequw v9,v3,v28			# cypher.hi, B
		vspltisw v11,-1
		vand	 v9,v9,v4			# Identifies matching key(s)
		vcmpequw. v10,v9,v11		# != 0 => Key found.

		# The first element in v10 that is equal to -1 identifies the
		# matching key. All elements before this one (in v4) are false 
		# positives. The number of these false positives, including the
		# first matching key, shall be added to check.count. The value
		# of the first matching key shall then be written to check.hi,
		# check.mid and check.lo.

		vsldoi	v11,v6,v10,4		# v11 = v10 >> 32 
		vsldoi	v12,v6,v10,8		# v12 = v10 >> 64
		vsldoi	v13,v6,v10,12		# v13 = v10 >> 96
		lvx		v8,r2,r6			# vector(0,1,2,3)
		vor		v11,v11,v12
		vnor	v11,v11,v13
		vand	v4,v4,v11			# Mask out unwanted matches.
		
		vsumsws	v6,v4,v6			# v6.lsw := [-1, -2, -3, -4]
		vadduwm	v7,v7,v8			# (key0.hi,key1.hi,key2.hi,key3.hi)
		vaddsws	v5,v5,v6			# update -check.count
		vand	v7,v7,v4			# Select matching key.hi values.
		stvx	v5,r7,r0			# save the updated vKey (check.count)
		
		addi	r31,r1,vCheck
		vsldoi	v8,v7,v7,8			# (key2.hi,key3.hi,key0.hi,key1.hi)
		vmaxuw	v8,v8,v7			# => {max(key2,key0); max(key3,key1);
									#	  max(key0,key2); max(key1,key3)}
		vsldoi	v7,v8,v8,4
		stvx	v5,0,r31
		vmaxuw	v8,v8,v7			# All elements == last matching key.
		stvewx	v8,0,r31			# overwrite check.hi
		
		beq-	cr6,next_keys		# False positives : continue.
		
	# A matching key has been found.
		lwz		r4,iterations(r1)
		lwz		r5,0(r4)			# iterations
		lwz		r6,vCheck(r1)		# check.hi
		mfctr	r7
		clrlwi	r6,r6,30			# keep only bits 30 and 31.
		slwi	r7,r7,2				# count * 4
		add		r5,r5,r6
		subf	r5,r7,r5
		stw		r5,0(r4)			# update iterations count

		li		r3,RESULT_FOUND		# At least one key matched


		#--------------------------------------------------------------------
		# Epilog
		# Restore r2, LR, GPRs, VRs and VRsave from the stack frame.
		# r3 := RESULT_FOUND or RESULT_NOTHING

epilog:	
		lwz		r6,vCheck(r1)		# Copy the sanity check values
		lwz		r7,vCheck+4(r1)
		lwz		r8,vCheck+8(r1)
		lwz		r9,vCheck+12(r1)
		lwz		r4,rc5_ptr(r1)		# get back rc5_72unitwork ptr

		stw		r6,check_hi(r4)
		stw		r7,check_mid(r4)
		stw		r8,check_lo(r4)
		subfic	r6,r9,0
		stw		r6,check_count(r4)
	
		lwz		r6,vKey(r1)			# Copy the last key checked back
		lwz		r7,vKey+4(r1)		# in rc5_72unitwork
		lwz		r8,vKey+8(r1)
		stw		r6,L0_hi(r4)
		stw		r7,L0_mid(r4)
		stw		r8,L0_lo(r4)
	
		li		r6,vectorBase		# Restore vector registers
		lvx		v31,r6,r1
		addi	r6,r6,16
		lvx		v30,r6,r1
		addi	r6,r6,16
		lvx		v29,r6,r1
		addi	r6,r6,16
		lvx		v28,r6,r1
		addi	r6,r6,16
		lvx		v27,r6,r1
		addi	r6,r6,16
		lvx		v26,r6,r1
		addi	r6,r6,16
		lvx		v25,r6,r1
		addi	r6,r6,16
		lvx		v24,r6,r1
		addi	r6,r6,16
		lvx		v23,r6,r1
		addi	r6,r6,16
		lvx		v22,r6,r1
		addi	r6,r6,16
		lvx		v21,r6,r1
		addi	r6,r6,16
		lvx		v20,r6,r1
		
		lwz		r2,saveR2(r1)		# Restore r2
		lwz		r6,0(r1)			# Caller's SP
		lwz		r0,-4-GPRSave(r6)	# VRsave
		mtspr	VRsave,r0
		lmw		r13,-GPRSave(r6)	# Restore GPRs
		mr		r1,r6				# Restore caller's stack pointer.
		lwz		r0,oldLR(r6)		# LR (from caller's frame)
		mtlr	r0
		blr
		
		
#============================================================================

		.data
#		.const
		.align	4

	.macro	gen_vec arg0
		.long	P+Q*\arg0
		.long	P+Q*\arg0
		.long	P+Q*\arg0
		.long	P+Q*\arg0
	.endm

		.long	0		# Vector {0; 1; 2; 3}, accessed as expanded_key[-1] 
		.long	1
		.long	2
		.long	3
		
expanded_key:			# Static expanded key datas S[]

		gen_vec	0
		gen_vec	1
		gen_vec 2
		gen_vec 3
		gen_vec 4
		gen_vec 5
		gen_vec 6
		gen_vec 7
		gen_vec 8
		gen_vec 9
		gen_vec 10
		gen_vec 11
		gen_vec 12
		gen_vec 13
		gen_vec 14
		gen_vec 15
		gen_vec 16
		gen_vec 17
		gen_vec 18
		gen_vec 19
		gen_vec 20
		gen_vec 21
		gen_vec 22
		gen_vec 23
		gen_vec 24
		gen_vec 25
