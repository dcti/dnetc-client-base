#
# RC5-72 Dual key core - MPC604e (MacOS ABI)
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
# This implementation is optimized for MPC604e. 
# The core checks 2 keys per loop using the integer units.
#
# Specifications :
#	MPC604e
#		- Fetch/dispatch/retire 4 instructions per clock cycle.
#		- 2 stages LSU, 2 SCIU.
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
# $Id: r72-KKS604e.gas.s,v 1.1.2.2 2003/03/12 20:31:44 oliver Exp $
#
# $Log: r72-KKS604e.gas.s,v $
# Revision 1.1.2.2  2003/03/12 20:31:44  oliver
# Linker errors when compiled with gas 2.11 - fixed
# Moved executed_key data from .data to .rodata
#
# Revision 1.1.2.1  2003/03/12 12:13:15  snake
# Added optimized gas compliant cores (like those for MacOS X)
#
# Revision 1.1.2.1  2003/01/20 00:26:42  mfeiri
# New core by kakace for 604
#
# Revision 1.3  2003/01/12 21:21:41  kakace
# The core now uses a static expanded key table instead of computing
# S[] value dynamically. This frees up the integer units (at the expense
# of extra LSU instructions) but this should be faster since the MPC604e
# can retire as much as 4 instructions per cycle, thus reducing the
# probability of completion queue stalls due to the extra LSU instructions.
#
# Revision 1.2  2003/01/12 20:53:57  kakace
# Fixed key.mid incrementation
#
# Revision 1.1  2003/01/12 19:44:36  kakace
# First implementation
#
#============================================================================
 
        .text
		.align	2
		.globl	rc5_72_unit_func_KKS604e

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
#	Local variable space
#	Linkage area

.set 	oldLR		,	8			# 8(sp) stores the old LR (caller's frame)
.set 	numGPRs		,	32-13		# r13 to r31

	# The linkageArea size, as defined by Apple, is equal to 24. Since we
	# have to align our local variable space to an address that is an
	# even multiple of 16, we shall add an extra padding. Of course, the
	# padding area could be used to save some local variables.

.set 	linkageArea	,	24			# Size of the linkageArea.


# Local variables area
# Offsets (from the callee's SP) to the local variable space.

.set 	saveR2	,	linkageArea		# Storage for r2 (defined as volatile under
								# MacOS X, but usually points to RTOC).
.set 	Key		,	saveR2 + 4		# key.hi, key.mid, key.lo

.set 	S_00	,	Key + 12
.set 	S_01	,	S_00 + 8
.set 	S_02	,	S_01 + 8
.set 	S_03	,	S_02 + 8
.set 	S_04	,	S_03 + 8
.set 	S_05	,	S_04 + 8
.set 	S_06	,	S_05 + 8
.set 	S_07	,	S_06 + 8
.set 	S_08	,	S_07 + 8
.set 	S_09	,	S_08 + 8
.set 	S_10	,	S_09 + 8
.set 	S_11	,	S_10 + 8
.set 	S_12	,	S_11 + 8
.set 	S_13	,	S_12 + 8
.set 	S_14	,	S_13 + 8
.set 	S_15	,	S_14 + 8
.set 	S_16	,	S_15 + 8
.set 	S_17	,	S_16 + 8
.set 	S_18	,	S_17 + 8
.set 	S_19	,	S_18 + 8
.set 	S_20	,	S_19 + 8
.set 	S_21	,	S_20 + 8
.set 	S_22	,	S_21 + 8
.set 	S_23	,	S_22 + 8
.set 	S_24	,	S_23 + 8
.set 	S_25	,	S_24 + 8
	
.set 	localTOP	,	(S_25 + 15 + 8) & (-16)

# Total frame size (minus the optional alignment padding).

.set 	frameSize	, localTOP + numGPRs * 4
	
# Size of the general register save area. This value is used as a
# (negative) offset from the caller's SP to save all non-volatile GPRs.

.set 	GPRSave	, numGPRs * 4
	
	
#============================================================================
#
# s32 rc5_72_unit_func_MPC604e (RC5_72UnitWork *rc5_72unitwork (r3), 
#  r3						 	u32 *iterations (r4), 
#							 	void * /*memblk (r5) */)

rc5_72_unit_func_KKS604e:

	# Prolog
	# Save r2, LR and GPRs to the stack frame.
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

	# Save all non-volatile registers (r13-r31)
		
		stmw	r13,-GPRSave(r6)	# Save r13-r31
		stw		r2,saveR2(r1)
		

	#--------------------------------------------------------------------
	# Cruncher setup
	# r3 := struct rc5_72UnitWork *rc5_72unitwork
	# r4 := u32 *iterations
	#
	# According to problem.cpp/MINIMUM_ITERATIONS, the number of keys to
	# check (iterations) is always an even multiple of 24. The cruncher
	# is usually asked to check thousands of 24-key blocks (no performance
	# issues involved here).
	

		# Save the current key, and keep a big-endian
		# copy of key.mid and key.lo in r7-r8.

		li		r21,L0_mid
		li		r22,L0_lo
		lwz		r6,L0_hi(r3)		# key.hi
		lwbrx	r7,r3,r21			# key.mid (big-endian)
		lwbrx	r8,r3,r22			# key.lo (big-endian)

		li		r21,Key+4			# key.mid
		li		r22,Key+8			# key.lo
		stw		r6,Key(r1)			# save key.hi
		stwbrx	r7,r1,r21			# save key.mid
		stwbrx	r8,r1,r22			# save key.lo
		
		# Compute the number of inner loop iterations for the 
		# first pass (assigned to CTR)
		
		lwz		r5,0(r4)			# iterations
		subfic	r24,r6,0x100		# 0x100 - key.hi
		cmpl	cr0,r5,r24
		bgt		set_counter
		
		mr		r24,r5
	
set_counter:
		subf	r5,r24,r5			# Remaining iterations.
		srwi	r24,r24,1			# inner loop count
		mtctr	r24					# = min(iterations, 0x100 - key.hi) / 2

		# Pre-load contest's values
		
		lwz		r2,cypher_lo(r3)
		
		lis		r10,(expanded_key)@ha
		la		r10,(expanded_key)@l(r10)
		

	#---------------------------------------------------------------------
	# Main loop. Preconditions :
	# r2  := cypher.lo
	# r3  := struct rc5_72UnitWork *rc5_72unitwork
	# r4  := u32 *iterations
	# r5  := Remaining iterations after the first pass in the inner loop.
	# r10 := &expanded_key[0] (constant)
	# CTR := Inner loop count
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

inner_loop_lo:
		# key.lo changed. Compute s[0], l[0] and s[1].
		# Preconditions :
		#	r2  := cypher.lo
		#	r10 := Q
		#
		# Results :
		#	r11 := l0[0]
		#	r13 := s[0]
		#	r14 := s[1]
		
		lwz		r13,0(r10)			# S[0]	
		lwz		r11,Key+8(r1)		# key.lo
		lwz		r14,4(r10)			# S[1]

		add		r11,r11,r13			# key.lo += a
		rotlw	r11,r11,r13			# b = L[0] = ROTL(...)

		add		r24,r13,r11			# a + b
		add		r14,r14,r24			# s[1] = S[1] + (a + b)
		rotlwi	r14,r14,3			# a = s[1] = ROTL3(...)
		

inner_loop_mid:
		# key.mid changed, key.lo unchanged. Compute l[1] and s[2]
		# Preconditions :
		#	r2  := cypher.lo
		#	r10 := &expanded_key[0]
		#	r11 := l0[0] (b)
		#	r14 := s[1] (a)
		#	
		# Results :
		#	r9  := S[3] (P + 3Q)
		#	r12 := l0[1]
		#	r15 := s[2]

		lwz		r12,Key+4(r1)		# key.mid
		add		r24,r14,r11			# a + b = s[1] + L[0]
		add		r12,r12,r24			# key.mid += a + b
		lwz		r15,8(r10)			# S[2]
		rotlw	r12,r12,r24			# b = L[1] = ROTL(...)
		
		add		r24,r12,r14			# a + b
		add		r15,r15,r24			# s[2] = S[2] + (a + b)
		lwz		r9,12(r10)			# S[3]
		rotlwi	r15,r15,3			# a = s[2] = ROTL3(...)

		
inner_loop_hi:
		# key.mid and key.lo unchanged. 
		# Preconditions :
		#	r2  := cypher.lo
		#	r9  := S[3]
		#	r10 := &expanded_key[0]
		#	r11 := L[0]
		#	r12 := L[1] (== b)
		#	r13 := s[0]
		#	r14 := s[1]
		#	r15 := s[2] (== a)
		#
		# Registers used :
		#	r16,r17 := l0[0], l1[0]
		#	r18,r19 := l0[1], l1[1]
		#	r20,r21 := l0[2], l1[2]
		#	r22,r23 := s0[x], s1[x]
		#	r24,r25 := t0, t1
		#	r26,r27 := A0, A1 (round stage)
		#	r28,r29 := B0, B1 (round stage)
		#
		# The first steps require specific code since l0[0], l1[0],
		# l0[1] and l1[1] are not initialized yet. This is done
		# dynamically by using L[0] and L[1] which are constant
		# values when key.mid and key.lo remain the same.

		# Step 1.2 : Compute l[2]

		add		r24,r15,r12			# a + b = s[2] + L[1]
		addi	r21,r6,1			# l1[2] = key.hi + 1
		add		r20,r6,r24			# l0[2] = key.hi + a + b
		add		r21,r21,r24			# l1[2] += a + b
		rotlw	r20,r20,r24			# b0 = l0[2] = ROTL(...) 
		rotlw	r21,r21,r24			# b1 = l1[2] = ROTL(...)

		# Step 1.3 : Compute s[3] and l[0]
		
		add		r24,r15,r20			# t0 = a + b0
		add		r25,r15,r21			# t1 = a + b1
		add		r22,r9,r24			# s0[3] = S[3] + t0
		add		r23,r9,r25			# s1[3] = S[3] + t1
		rotlwi	r22,r22,3			# a0 = s0[3] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[3] = ROTL3(...)
		
		add		r24,r22,r20			# t0 = a0 + b0
		add		r25,r23,r21			# t1 = a1 + b1
		lwz		r9,16(r10)			# S[4]
		add		r16,r11,r24			# l0[0] = L[0] + t0
		add		r17,r11,r25			# l1[0] = L[0] + t1
		stw		r22,S_03(r1)		# save s0[3]
		rotlw	r16,r16,r24			# b0 = l0[0] = ROTL(...)
		rotlw	r17,r17,r25			# b1 = l1[0] = ROTL(...)
		stw		r23,S_03+4(r1)		# save s1[3]
		
		# Step 1.4 : Compute s[4] and l[1]
		
		add		r24,r22,r16			# t0 = a0 + b0
		add		r25,r23,r17			# t1 = a1 + b1
		add		r22,r9,r24			# s0[4] = S[4] + t0
		add		r23,r9,r25			# s1[4] = S[4] + t1
		rotlwi	r22,r22,3			# a0 = s0[4] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[4] = ROTL3(...)
		
		add		r24,r22,r16			# t0 = a0 + b0
		add		r25,r23,r17			# t1 = a1 + b1
		lwz		r9,20(r10)			# S[5]
		add		r18,r12,r24			# l0[1] = L[1] + t0
		add		r19,r12,r25			# l1[1] = L[1] + t1
		stw		r22,S_04(r1)		# save s0[4]
		rotlw	r18,r18,r24			# b0 = l0[1] = ROTL(...)
		rotlw	r19,r19,r25			# b1 = l1[1] = ROTL(...)
		stw		r23,S_04+4(r1)		# save s1[4]
		
		# Step 1.5 (partial) : Compute s[5].
		
		add		r24,r22,r18			# t0 = a0 + b0
		add		r25,r23,r19			# t1 = a1 + b1
		add		r22,r9,r24			# s0[5] = S[5] + t0
		add		r23,r9,r25			# s1[5] = S[5] + t1
		rotlwi	r22,r22,3			# a0 = s0[5] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[5] = ROTL3(...)
		
		add		r24,r22,r18			# t0 = a0 + b0
		add		r25,r23,r19			# t1 = a1 + b1
		
		# The next steps (upto 25) can be implemented
		# using a macro. Note that in order to reduce the
		# number of arguments, the code compute l[j] and
		# s[i] and save s[i-1]
	
	.macro	mix1 arg0, arg1, arg2, arg3
									# l0[j], l1[j], S_kk
		lwz		r9,\arg3*4(r10)		# S[z]
		add		\arg0,\arg0,r24			# l0[j] += t0
		add		\arg1,\arg1,r25			# l1[j] += t1
		stw		r22,\arg2(r1)			# save s0[k]
		rotlw	\arg0,\arg0,r24			# b0 = l0[j] = ROTL(...)
		rotlw	\arg1,\arg1,r25			# b1 = l1[j] = ROTL(...)
		stw		r23,\arg2+4(r1)		# save s1[k]
		
		add		r24,r22,\arg0			# t0 = a0 + b0
		add		r25,r23,\arg1			# t1 = a1 + b1
		add		r22,r9,r24			# s0[i] = S[i] + t0
		add		r23,r9,r25			# s1[i] = S[i] + t1
		rotlwi	r22,r22,3			# a0 = s0[i] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[i] = ROTL3(...)
		
		add		r24,r22,\arg0			# t0 = a0 + b0
		add		r25,r23,\arg1			# t1 = a1 + b1
	.endm

		mix1	r20,r21,S_05,6		# Step  1.6 : l[2] and s[6]
		mix1	r16,r17,S_06,7		# Step  1.7 : l[0] and s[7]
		mix1	r18,r19,S_07,8		# Step  1.8 : l[1] and s[8]
		mix1	r20,r21,S_08,9		# Step  1.9 : l[2] and s[9]
		mix1	r16,r17,S_09,10		# Step 1.10 : l[0] and s[10]
		mix1	r18,r19,S_10,11		# Step 1.11 : l[1] and s[11]
		mix1	r20,r21,S_11,12		# Step 1.12 : l[2] and s[12]
		mix1	r16,r17,S_12,13		# Step 1.13 : l[0] and s[13]
		mix1	r18,r19,S_13,14		# Step 1.14 : l[1] and s[14]
		mix1	r20,r21,S_14,15		# Step 1.15 : l[2] and s[15]
		mix1	r16,r17,S_15,16		# Step 1.16 : l[0] and s[16]
		mix1	r18,r19,S_16,17		# Step 1.17 : l[1] and s[17]
		mix1	r20,r21,S_17,18		# Step 1.18 : l[2] and s[18]
		mix1	r16,r17,S_18,19		# Step 1.19 : l[0] and s[19]
		mix1	r18,r19,S_19,20		# Step 1.20 : l[1] and s[20]
		mix1	r20,r21,S_20,21		# Step 1.21 : l[2] and s[21]
		mix1	r16,r17,S_21,22		# Step 1.22 : l[0] and s[22]
		mix1	r18,r19,S_22,23		# Step 1.23 : l[1] and s[23]
		mix1	r20,r21,S_23,24		# Step 1.24 : l[2] and s[24]
		mix1	r16,r17,S_24,25		# Step 1.25 : l[0] and s[25]
				
		# Complete step 1.25 (compute l[1])
		
		add		r18,r18,r24			# l0[1] += a0 + b0
		add		r19,r19,r25			# l1[1] += a1 + b1
		stw		r22,S_25(r1)		# save s0[25]
		rotlw	r18,r18,r24			# b0 = l0[1] = ROTL(...)
		rotlw	r19,r19,r25			# b1 = l1[1] = ROTL(...)
		stw		r23,S_25+4(r1)		# save s1[25]

	#== Pass 2 : Update s[0] ... s[25]
	#	The first steps require specific code to load
	#	s[0], s[1] and s[2]. The remaining steps can be
	#	implemented using a macro.
	
		# Step 2.0 : Compute s[0] and l[2]
	
		add		r24,r22,r18			# t0 = a0 + b0
		add		r25,r23,r19			# t1 = a1 + b1
		add		r22,r13,r24			# s0[0] = s[0] + t0
		add		r23,r13,r25			# s1[0] = s[0] + t1
		rotlwi	r22,r22,3			# a0 = s0[0] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[0] = ROTL3(...)
		
		add		r24,r22,r18			# t0 = a0 + b0
		add		r25,r23,r19			# t1 = a1 + b1
		add		r20,r20,r24			# l0[2] += t0
		add		r21,r21,r25			# l1[2] += t1
		stw		r22,S_00(r1)		# save s0[0]
		rotlw	r20,r20,r24			# b0 = l0[2] = ROTL(...)
		rotlw	r21,r21,r25			# b1 = l1[2] = ROTL(...)
		stw		r23,S_00+4(r1)		# save s1[0]
		
		# Step 2.1 : Compute s[1] and l[0]
		
		add		r24,r22,r20			# t0 = a0 + b0
		add		r25,r23,r21			# t1 = a1 + b1
		add		r22,r14,r24			# s0[1] = s[1] + t0
		add		r23,r14,r25			# s1[1] = s[1] + t1
		rotlwi	r22,r22,3			# a0 = s0[1] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[1] = ROTL3(...)
		
		add		r24,r22,r20			# t0 = a0 + b0
		add		r25,r23,r21			# t1 = a1 + b1
		add		r16,r16,r24			# l0[0] += t0
		add		r17,r17,r25			# l1[0] += t1
		stw		r22,S_01(r1)		# save s0[1]
		rotlw	r16,r16,r24			# b0 = l0[0] = ROTL(...)
		rotlw	r17,r17,r25			# b1 = l1[0] = ROTL(...)
		stw		r23,S_01+4(r1)		# save s1[1]

		# Step 2.2 : Compute s[2] and l[1]
		
		add		r24,r22,r16			# t0 = a0 + b0
		add		r25,r23,r17			# t1 = a1 + b1
		add		r22,r15,r24			# s0[2] = s[2] + t0
		add		r23,r15,r25			# s1[2] = s[2] + t1
		rotlwi	r22,r22,3			# a0 = s0[2] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[2] = ROTL3(...)
		
		lwz		r30,S_03(r1)		# pre-load s0[3]
		add		r24,r22,r16			# t0 = a0 + b0
		add		r25,r23,r17			# t1 = a1 + b1
		lwz		r31,S_03+4(r1)		# pre-load s1[3]
		add		r18,r18,r24			# l0[1] += t0
		add		r19,r19,r25			# l1[1] += t1
		stw		r22,S_02(r1)		# save s0[2]
		rotlw	r18,r18,r24			# b0 = l0[1] = ROTL(...)
		rotlw	r19,r19,r25			# b1 = l1[1] = ROTL(...)
		stw		r23,S_02+4(r1)		# save s1[2]

		# Begin step 2.3 : Compute s[3]
		
		add		r24,r22,r18			# t0 = a0 + b0
		add		r25,r23,r19			# t1 = a1 + b1
		add		r22,r30,r24			# s0[3] += t0
		add		r23,r31,r25			# s1[3] += t1
		rotlwi	r22,r22,3			# a0 = s0[3] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[3] = ROTL3(...)
		
		add		r24,r22,r18			# t0 = a0 + b0
		add		r25,r23,r19			# t1 = a1 + b1


	.macro	mix2 arg0, arg1, arg2
									# l0[j], l1[j], S_kk[]
		lwz		r30,\arg2+8(r1)		# pre-load s0[i+1]
		add		\arg0,\arg0,r24			# l0[j] += t0
		add		\arg1,\arg1,r25			# l1[j] += t1
		lwz		r31,\arg2+12(r1)		# pre-load s1[i+1]
		rotlw	\arg0,\arg0,r24			# b0 = l0[j] = ROTL(...)
		rotlw	\arg1,\arg1,r25			# b1 = l1[j] = ROTL(...)
		stw		r22,\arg2(r1)			# save s0[i]
		add		r24,r22,\arg0			# t0 = a0 + b0
		add		r25,r23,\arg1			# t1 = a1 + b1
		stw		r23,\arg2+4(r1)		# save s1[i]
		add		r22,r30,r24			# s0[i] += t0
		add		r23,r31,r25			# s1[i] += t1
		rotlwi	r22,r22,3			# a0 = s0[i] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[i] = ROTL3(...)
		add		r24,r22,\arg0			# t0 = a0 + b0
		add		r25,r23,\arg1			# t1 = a1 + b1
	.endm

		mix2	r20,r21,S_03		# Step  2.4 : l[2] and s[4]
		mix2	r16,r17,S_04		# Step  2.5 : l[0] and s[5]
		mix2	r18,r19,S_05		# Step  2.6 : l[1] and s[6]
		mix2	r20,r21,S_06		# Step  2.7 : l[2] and s[7]
		mix2	r16,r17,S_07		# Step  2.8 : l[0] and s[8]
		mix2	r18,r19,S_08		# Step  2.9 : l[1] and s[9]
		mix2	r20,r21,S_09		# Step 2.10 : l[2] and s[10]
		mix2	r16,r17,S_10		# Step 2.11 : l[0] and s[11]
		mix2	r18,r19,S_11		# Step 2.12 : l[1] and s[12]
		mix2	r20,r21,S_12		# Step 2.13 : l[2] and s[13]
		mix2	r16,r17,S_13		# Step 2.14 : l[0] and s[14]
		mix2	r18,r19,S_14		# Step 2.15 : l[1] and s[15]
		mix2	r20,r21,S_15		# Step 2.16 : l[2] and s[16]
		mix2	r16,r17,S_16		# Step 2.17 : l[0] and s[17]
		mix2	r18,r19,S_17		# Step 2.18 : l[1] and s[18]
		mix2	r20,r21,S_18		# Step 2.19 : l[2] and s[19]
		mix2	r16,r17,S_19		# Step 2.20 : l[0] and s[20]
		mix2	r18,r19,S_20		# Step 2.21 : l[1] and s[21]
		mix2	r20,r21,S_21		# Step 2.22 : l[2] and s[22]
		mix2	r16,r17,S_22		# Step 2.23 : l[0] and s[23]
		mix2	r18,r19,S_23		# Step 2.24 : l[1] and s[24]

		# Terminate step 24 : Compute l[2]
		
		lwz		r30,S_25(r1)		# pre-load s0[25]
		add		r20,r20,r24			# l0[2] += t0
		add		r21,r21,r25			# l1[2] += t1
		lwz		r31,S_25+4(r1)		# pre-load s1[25]
		rotlw	r20,r20,r24			# b0 = l0[2] = ROTL(...)
		rotlw	r21,r21,r25			# b1 = l1[2] = ROTL(...)
		stw		r22,S_24(r1)		# save s0[24]
		
		# Step 2.25 : Compute s[25] and l[0]

		add		r24,r22,r20			# t0 = a0 + b0
		add		r25,r23,r21			# r1 = a1 + b1
		stw		r23,S_24+4(r1)		# save s1[24]
		add		r22,r30,r24			# s0[25] += t0
		add		r23,r31,r25			# s1[25] += t1
		rotlwi	r22,r22,3			# a0 = s0[25] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[25] = ROTL3(...)
		
		lwz		r30,S_00(r1)		# pre-load s0[0]
		add		r24,r22,r20			# t0 = a0 + b0
		add		r25,r23,r21			# r1 = a1 + b1
		lwz		r31,S_00+4(r1)		# pre-load s1[0]
		add		r16,r16,r24			# l0[0] += t0
		add		r17,r17,r25			# l1[0] += t1
		stw		r22,S_25(r1)		# save s0[25]
		rotlw	r16,r16,r24			# b0 = l0[0] = ROTL(...)
		rotlw	r17,r17,r25			# b1 = l1[0] = ROTL(...)
		stw		r23,S_25+4(r1)		# save s1[25]

	#== Pass 3 : Update s[0] ... s[25]
	#	Also perform the round pass
	# r26-r27 := A0, A1
	# r28-r29 := B0,B1

		# Step 3.0 : Compute s[0], l[1] and A
		
		add		r24,r22,r16			# t0 = a0 + b0
		add		r25,r23,r17			# t1 = a1 + b1
		add		r22,r30,r24			# s0[0] += t0
		add		r23,r31,r25			# s1[0] += t1
		lwz		r30,S_01(r1)		# pre-load s0[1]
		rotlwi	r22,r22,3			# a0 = s0[0] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[0] = ROTL3(...)
		lwz		r31,S_01+4(r1)		# pre-load s1[1]
		
		add		r24,r22,r16			# t0 = a0 + b0
		add		r25,r23,r17			# t1 = a1 + b1
		lwz		r27,plain_lo(r3)	# plain.lo
		add		r18,r18,r24			# l0[1] += t0
		add		r19,r19,r25			# l1[1] += t1
		lwz		r29,plain_hi(r3)	# plain.hi
		rotlw	r18,r18,r24			# b0 = l0[1] = ROTL(...)
		add		r26,r27,r22			# A0 = plain.lo + s0[0]
		rotlw	r19,r19,r25			# b1 = l1[1] = ROTL(...)
		add		r27,r27,r23			# A1 = plain.lo + s1[0]

		# Step 3.1 : Compute s[1], l[2] and B
		
		add		r24,r22,r18			# t0 = a0 + b0
		add		r25,r23,r19			# t1 = a1 + b1
		add		r22,r30,r24			# s0[1] += t0
		add		r23,r31,r25			# s1[1] += t1
		lwz		r30,S_02(r1)		# pre-load s0[2]
		rotlwi	r22,r22,3			# a0 = s0[1] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[1] = ROTL3(...)
		lwz		r31,S_02+4(r1)		# pre-load s1[2]

		add		r24,r22,r18			# t0 = a0 + b0
		add		r25,r23,r19			# t1 = a1 + b1
		add		r20,r20,r24			# l0[2] += t0
		add		r21,r21,r25			# l1[2] += t1
		rotlw	r20,r20,r24			# b0 = l0[2] = ROTL(...)
		add		r28,r29,r22			# B0 = plain.hi + s0[1]
		rotlw	r21,r21,r25			# b1 = l1[2] = ROTL(...)
		add		r29,r29,r23			# B1 = plain.hi + s1[1]


	.macro	mix3 arg0, arg1, arg2, arg3, arg4, arg5, arg6
					# l0[i-1], l1[i-1], l0[i], l1[i], l0[i+1], l1[i+1], S_xx
					#    $0       $1      $2     $3      $4       $5     $6
		
		add		r24,r22,\arg0			# t0 = a0 + b0
		xor		r26,r26,r28			# A0 = A0 ^ B0
		add		r25,r23,\arg1			# t1 = a1 + b1
		xor		r27,r27,r29			# A1 = A1 ^ B1
		add		r22,r30,r24			# s0[j] += t0
		rotlw	r26,r26,r28			# A0 = ROTL(A0, B0)
		add		r23,r31,r25			# s1[j] += t1
		lwz		r30,\arg6(r1)			# pre-load s0[j+1]
		rotlwi	r22,r22,3			# a0 = s0[j] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[j] = ROTL3(...)
		lwz		r31,\arg6+4(r1)		# pre-load s1[j+1]
		rotlw	r27,r27,r29			# A1 = ROTL(A1, B1)

		add		r24,r22,\arg0			# t0 = a0 + b0
		add		r25,r23,\arg1			# t1 = a1 + b1
		add		\arg2,\arg2,r24			# l0[i] += t0
		add		\arg3,\arg3,r25			# l1[i] += t1
		rotlw	\arg2,\arg2,r24			# b0 = l0[i] = ROTL(...)
		add		r26,r26,r22			# A0 += s0[j]
		rotlw	\arg3,\arg3,r25			# b1 = l1[i] = ROTL(...)
		add		r27,r27,r23			# A1 += s1[j]

		add		r24,r22,\arg2			# t0 = a0 + b0
		xor		r28,r28,r26			# B0 = B0 ^ A0
		add		r25,r23,\arg3			# t1 = a1 + b1
		xor		r29,r29,r27			# B1 = B1 ^ A1
		add		r22,r30,r24			# s0[j+1] += t0
		rotlw	r28,r28,r26			# B0 = ROTL(B0, A0)
		add		r23,r31,r25			# s1[j+1] += t1
		lwz		r30,\arg6+8(r1)		# pre-load s0[j+2]
		rotlwi	r22,r22,3			# a0 = s0[j+1] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[j+1] = ROTL3(...)
		lwz		r31,\arg6+12(r1)		# pre-load s1[j+2]
		rotlw	r29,r29,r27			# B1 = ROTL(B1, A1)
		
		add		r24,r22,\arg2			# t0 = a0 + b0
		add		r25,r23,\arg3			# t1 = a1 + b1
		add		\arg4,\arg4,r24			# l0[i+1] += t0
		add		\arg5,\arg5,r25			# l1[i+1] += t1
		rotlw	\arg4,\arg4,r24			# b0 = l0[i+1] = ROTL(...)
		add		r28,r28,r22			# B0 += s0[j+1]
		rotlw	\arg5,\arg5,r25			# b1 = l1[i+1] = ROTL(...)
		add		r29,r29,r23			# B1 += s1[j+1]
	.endm

		mix3	r20,r21,r16,r17,r18,r19,S_03
		mix3	r18,r19,r20,r21,r16,r17,S_05
		mix3	r16,r17,r18,r19,r20,r21,S_07
		mix3	r20,r21,r16,r17,r18,r19,S_09
		mix3	r18,r19,r20,r21,r16,r17,S_11
		mix3	r16,r17,r18,r19,r20,r21,S_13
		mix3	r20,r21,r16,r17,r18,r19,S_15
		mix3	r18,r19,r20,r21,r16,r17,S_17
		mix3	r16,r17,r18,r19,r20,r21,S_19
		mix3	r20,r21,r16,r17,r18,r19,S_21
		mix3	r18,r19,r20,r21,r16,r17,S_23

		# Step 3.24 : Compute s[24] and l[1]

		add		r24,r22,r16			# t0 = a0 + b0
		xor		r26,r26,r28			# A0 = A0 ^ B0
		add		r25,r23,r17			# t1 = a1 + b1
		xor		r27,r27,r29			# A1 = A1 ^ B1
		add		r22,r30,r24			# s0[24] += t0
		rotlw	r26,r26,r28			# A0 = ROTL(A0, B0)
		add		r23,r31,r25			# s1[24] += t1
		lwz		r30,S_25(r1)		# pre-load s0[25]
		rotlwi	r22,r22,3			# a0 = s0[24] = ROTL3(...)
		rotlwi	r23,r23,3			# a1 = s1[24] = ROTL3(...)
		lwz		r31,S_25+4(r1)		# pre-load s1[25]
		rotlw	r27,r27,r29			# A1 = ROTL(A1, B1)

		add		r24,r22,r16			# t0 = a0 + b0
		add		r25,r23,r17			# t1 = a1 + b1
		add		r18,r18,r24			# l0[1] += t0
		add		r19,r19,r25			# l1[1] += t1
		add		r26,r26,r22			# A0 += s0[24]
		rotlw	r18,r18,r24			# b0 = l0[1] = ROTL(...)
		cmpl	cr0,r26,r2			# A0 == cypher.lo ?
		add		r27,r27,r23			# A1 += s1[24]
		rotlw	r19,r19,r25			# b1 = l1[1] = ROTL(...)
		cmpl	cr1,r27,r2			# A1 == cypher.lo ?
		
		lwz		r9,12(r10)			# S[3]
		beq		cr0,check_key0
		beq		cr1,check_key1

next_key:
		addi	r6,r6,2
		bdnz	inner_loop_hi
		
	# Increment key.mid
	
		srwi	r0,r6,8				# key.hi / 256
		cmpli	cr0,r5,0
		add		r7,r7,r0			# Increment key.mid (if key.hi overflowed)
		clrlwi	r6,r6,24
		cmpli	cr1,r5,0xFF
		li		r0,0x100			# Preset inner loop count.
		beq		cr0,exit			# iteration == 0 : Exit

		bgt		cr1,set_CTR
		mr		r0,r5				# Select inner loop count
	
set_CTR:
		subf	r5,r0,r5			# Update iterations count
		srwi	r0,r0,1				# Check 2 keys per loop
		cmpli	cr0,r7,0			# TRUE => Increment key.lo
		mtctr	r0					# inner loop counter
		li		r0,Key+4
		stwbrx	r7,r1,r0			# Store key.mid
		bne		cr0,inner_loop_mid
		

	# Increment key.lo

inc_key_lo:
		li		r0,Key+8
		addi	r8,r8,1
		stwbrx	r8,r1,r0			# Store key.lo
		b		inner_loop_lo
	
exit:	li		r4,RESULT_NOTHING	# Not found.
		b		epilog


	# Check whether key #0 matches.

check_key0:							# Step 3.25 : compute s0[25]		
		add		r24,r22,r18			# t0 = a0 + b0
		xor		r28,r28,r26			# B0 = B0 ^ A0
		add		r22,r30,r24			# s0[25] += t0
		rotlw	r28,r28,r26			# B0 = ROTL(B0, A0)
		lwz		r26,cypher_hi(r3)
		rotlwi	r22,r22,3			# a0 = s0[25] = ROTL3(...)
		lwz		r24,check_count(r3)
		add		r28,r28,r22			# B0 += s0[25]
		addi	r24,r24,1
		cmpl	cr0,r26,r28			# B0 == cypher.hi ?
		stw		r24,check_count(r3)
		
		li		r0,0				# key #0
		addi	r24,r6,0			# key.hi
		li		r25,check_mid
		li		r26,check_lo
		stw		r24,check_hi(r3)	# check.hi
		stwbrx	r7,r25,r3			# check.mid
		stwbrx	r8,r26,r3			# check.lo

		beq		cr0,key_found
		bne		cr1,next_key		# A1 != cypher.lo

	# Check whether key #1 matches.
	
check_key1:							# Step 3.25 : compute s1[25]
		add		r25,r23,r19			# t1 = a1 + b1
		xor		r29,r29,r27			# B1 = B1 ^ A1
		add		r23,r31,r25			# s1[25] += t1
		rotlw	r29,r29,r27			# B1 = ROTL(B1, A1)
		lwz		r27,cypher_hi(r3)
		rotlwi	r23,r23,3			# a1 = s1[25] = ROTL3(...)
		lwz		r24,check_count(r3)	
		add		r29,r29,r23			# B1 += s1[25]
		addi	r24,r24,1
		cmpl	cr1,r27,r29			# B1 == cypher_hi ?
		stw		r24,check_count(r3)
	
		li		r0,1				# key #1
		addi	r24,r6,1			# key.hi
		li		r25,check_mid
		li		r26,check_lo
		stw		r24,check_hi(r3)	# check.hi
		stwbrx	r7,r25,r3			# check.mid
		stwbrx	r8,r26,r3			# check.lo

		bne		cr1,next_key

	# Key found. r0 := key offset (0 or 1)

key_found:
		mfctr	r9
		slwi	r9,r9,1
		lwz		r10,0(r4)			# Initial iterations
		add		r5,r5,r9
		subf	r5,r0,r5			# r5 := remaining iterations.
		subf	r10,r5,r10
		stw		r10,0(r4)			# How many iterations done.

		li		r4,RESULT_FOUND


		#--------------------------------------------------------------------
		# Epilog
		# Restore r2, LR and GPRs from the stack frame.
		# r4 := RESULT_FOUND or RESULT_NOTHING

epilog:	
		li		r10,L0_mid
		li		r11,L0_lo
		stw		r6,L0_hi(r3)		# L0_hi
		stwbrx	r7,r10,r3			# L0_mid
		stwbrx	r8,r11,r3			# L0_lo
		
		lwz		r2,saveR2(r1)		# Restore r2
		lwz		r6,0(r1)			# Caller's SP
		lmw		r13,-GPRSave(r6)	# Restore GPRs
		mr		r1,r6				# Restore caller's stack pointer.
		mr		r3,r4				# Return code
		lwz		r0,oldLR(r6)		# LR (from caller's frame)
		mtlr	r0
		blr

#============================================================================

		.rodata
		.align	4

		
expanded_key:					# Static expanded key table S[]

		.long	0xBF0A8B1D		# ROTL3(P)
		.long	P+Q
		.long	P+Q*2
		.long	P+Q*3
		.long	P+Q*4
		.long	P+Q*5
		.long	P+Q*6
		.long	P+Q*7
		.long	P+Q*8
		.long	P+Q*9
		.long	P+Q*10
		.long	P+Q*11
		.long	P+Q*12
		.long	P+Q*13
		.long	P+Q*14
		.long	P+Q*15
		.long	P+Q*16
		.long	P+Q*17
		.long	P+Q*18
		.long	P+Q*19
		.long	P+Q*20
		.long	P+Q*21
		.long	P+Q*22
		.long	P+Q*23
		.long	P+Q*24
		.long	P+Q*25
