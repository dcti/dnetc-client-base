# Copyright distributed.net 1997-2003 - All Rights Reserved
# For use in distributed.net projects only.
# Any other distribution or use of this source violates copyright.
#
# Author: Decio Luiz Gazzoni Filho <decio@distributed.net>
# $Id: r72-cellv1-spe.s,v 1.2 2007/10/22 16:48:35 jlawson Exp $

	# A NOTE ON OPTIMIZING THE CORE
	#
	# The core is already basically optimal if you use conventional
	# implementation techniques. There may be a couple of cycles to
	# shave in key incrementation or comparison, but don't expect even
	# 1% improvement that way. This is because all RC5 instructions
	# (addition, 32-bit rotates and XOR) are even pipeline instructions,
	# and the occasional loads and stores that may be required are odd
	# pipeline instructions and already paired with even pipeline
	# instructions. To attain significant speedups, you'll have to think
	# outside the box.
	#
	# One observes that most (say 90% or 95%) of the odd pipeline
	# instruction slots are free in this core. If there were a way to
	# use them, significant gains might be realized, but at first sight
	# none of the odd pipeline instructions are relevant to the
	# computations being performed.
	#
	# However, it turns out that there are other ways of computing
	# 32-bit rotates without using the rot/roti instruction. I'll focus
	# on a replacement for rotation by 3, which accounts for half of the
	# rotations performed on the key schedule of the algorithm. Although
	# it's possible to come up with a replacement for rotation by a
	# variable amount, the performance is not worthwhile.
	#
	# Our main tool is the 128-bit rotate instruction, rotqbii. Suppose
	# first that you replicate a value X across all 32-bit slots of a
	# vector register, i.e. let reg = X | X | X | X. It's not hard to
	# see that a 128-bit rotation on that register would be equivalent
	# to a scalar 32-bit rotation by the same amount, due to the
	# cyclical nature of the data. That suggests a scheme where an
	# arbitrary vector register with contents X | Y | Z | W is copied to
	# four registers of the form X | X | X | X, Y | Y | Y | Y,
	# Z | Z | Z | Z and W | W | W | W, then rotations are performed on
	# each of them individually, and results are assembled back in a
	# vector register. As luck would have it, the instruction we use for
	# packing and unpacking registers is the shuffle bytes (shufb)
	# instruction, which is also an odd pipeline instruction, so we
	# don't have to waste any odd pipeline slots on this instruction
	# sequence.
	#
	# However, if you assume the same rotation amount for all 32-bit
	# slots, some optimizations are applicable. Consider a register
	# assembled as reg = X | X | Y | Y. Applying a 128-bit rotation by R
	# to reg produced X rotated by R in the first slot (from left to
	# right) and Y rotated by R in the third slot, so that only two
	# 128-bit rotations suffice for emulating a vector 32-bit rotation.
	# More explicitly, let
	#
	# $1:     target register for 32-bit vector rotation
	# $2, $3: temporaries
	# $4:     shuffle pattern 0x00010203000102030405060704050607 for
	#         extracting X | X | Y | Y from X | Y | Z | W
	# $5:     shuffle pattern 0x08090A0B08090A0B0C0D0E0F0C0D0E0F for
	#         extracting Z | Z | W | W from X | Y | Z | W
	# $6:     shuffle pattern 0x0001020308090A0B1011121318191A1B for
	#         reassembling X | Y | Z | W from X | _ | Y | _ and
	#         Z | _ | W | _
	#
	# Then we can compute a functionally equivalent version of the even
	# pipeline instruction
	#
	# 	roti	$1, $1, rot_amount
	#
	# using the following code snippet, which employs only odd pipeline
	# instructions:
	#
	# 	shufb	$2, $1, $1, $4
	#	shufb	$3, $1, $1, $5
	#	rotqbii	$2, $2, rot_amount
	#	rotqbii	$3, $3, rot_amount
	#	shufb	$1, $2, $3, $6
	#
	# In theory, each vector pipe in the key schedule requires 6
	# instructions per round, and under this scheme one of these
	# instructions (the rotation by 3) can be replaced by 5 odd pipeline
	# instructions, which if properly paired would only require 5
	# cycles. Unfortunately, an impractical number of pipes would be
	# required to mask latencies in the code above, which would spill
	# part of the key schedule array to memory (currently this array is
	# completely stored in registers), and this would render most if not
	# all the improvements moot. A second possibility that wouldn't
	# require as many pipes, but is extremely difficult to implement, is
	# software pipelining, similar to the way the PowerPC cores by
	# kakace operate. So this is your mission, should you choose to
	# accept it: take the ideas above and apply them in a way that
	# produces actual performance improvements.
	#
	# There's an improvement for the case of rotation by 3 or any other
	# small (< 8) rotation amount. Again let reg = X | Y | Z | W and
	# consider its decomposition into bytes:
	#
	# reg = X0 X1 X2 X3 | Y0 Y1 Y2 Y3 | Z0 Z1 Z2 Z3 | W0 W1 W2 W3
	#
	# Now shuffle the contents of this register into this form (there
	# are other equivalent forms, you are free to choose a different one
	# if it suits you better):
	#
	# reg' = X0 X1 X2 X3 | X0 Y0 Y1 Y2 | Y3 Y0 Z0 Z1 | Z2 Z3 Z0 __
	#
	# Rotating reg' by 3 will yield X, Y, Z rotated by R in byte slots
	# 0-3, 5-8, 10-13 respectively (from left to right). More
	# explicitly, let
	#
	# $1, $2,
	# $3:     target registers for 32-bit vector rotation
	# $4, $5,
	# $6, $7: temporaries
	# $8:     shuffle pattern 0x0001020300040506070408090A0B0880
	# $9:     shuffle pattern 0x0C0D0E0F0C1011121310141516171480
	# $10:    shuffle pattern 0x08090A0B080C0D0E0F0C101112131080
	# $11:    shuffle pattern 0x040506070408090A0B080C0D0E0F0C80
	# $12:    shuffle pattern 0x00010203050607080A0B0C0D10111213
	# $13:    shuffle pattern 0x050607080A0B0C0D1011121315161718
	# $14:    shuffle pattern 0x0A0B0C0D10111213151617181A1B1C1D
	#
	# Then we can compute a functionally equivalent version of the
	# following set of rotations, which uses even pipeline instructions:
	#
	# 	roti	$1, $1, rot_amount
	#	roti	$2, $2, rot_amount
	#	roti	$3, $3, rot_amount
	#
	# using the following code snippet, which employs only odd pipeline
	# instructions:
	#
	# 	shufb	$4, $1, $1,  $8
	#	shufb	$5, $1, $2,  $9
	#	shufb	$6, $2, $3, $10
	#	shufb	$7, $3, $3, $11
	#	rotqbii	$4, $4, rot_amount
	#	rotqbii	$5, $5, rot_amount
	#	rotqbii	$6, $6, rot_amount
	#	rotqbii	$7, $7, rot_amount
	#	shufb	$1, $4, $5, $12
	#	shufb	$2, $5, $6, $13
	#	shufb	$3, $6, $7, $14
	#
	# Hence we replace 3 even pipeline instructions by 10 odd pipeline
	# instructions, a ratio of 10/3 ~ 3.33, which is better than the 5
	# instructions above, but again we run into latency issues. Note
	# that this arrangement is better suited to a 3 vector pipe core
	# instead of the current 4 vector pipe arrangement. Again, it might
	# only be possible to exploit this technique using something like
	# software pipelining, but you're free to experiment. A 6-pipe core
	# would probably allow the use of this technique without any
	# software pipelining, but it runs into spill issues as well since
	# 4 pipes are the maximum one can implement without running out of
	# registers.

	.section bss
	.align	4

	# Area for saving registers when leaving the main loop
	.lcomm	save_118,		16
	.lcomm	save_119,		16
	.lcomm	save_120,		16
	.lcomm	save_121,		16
	.lcomm	save_122,		16

	# The rc5_72unitwork struct
	.lcomm	rc5_72unitwork,		16

	# Pointer to iteration count in memory
	.lcomm	iterations_ptr,		16

	# The iteration count itself
	.lcomm	iterations,		16

	# Data read from the rc5_72unitwork struct
	.lcomm	plain_hi,		16
	.lcomm	plain_lo,		16
	.lcomm	cypher_hi,		16
	.lcomm	cypher_lo,		16
	.lcomm	L0_hi,			16
	.lcomm	L0_mid,			16
	.lcomm	L0_lo,			16
	.lcomm	check_count,		16
	.lcomm	check_hi,		16
	.lcomm	check_mid,		16
	.lcomm	check_lo,		16

	# Cached value of L[0] (recomputed when key.lo changes)
	.lcomm	L0,			16

	# Cached values of S[1] and L[1] (recomputed when key.mid changes)
	.lcomm	S1,			16
	.lcomm	L1,			16

	# Cached values of S[2] (recomputed when key.mid changes)
	.lcomm	S2_1,			16
	.lcomm	S2_2,			16
	.lcomm	S2_3,			16
	.lcomm	S2_4,			16

	# Cached value of A+B in key setup round 2 (recomputed when key.mid
	# changes)
	.lcomm	AB2_1,			16
	.lcomm	AB2_2,			16
	.lcomm	AB2_3,			16
	.lcomm	AB2_4,			16

	# Saved value of L[1] after key setup round 76, used for computing
	# S[25] in key setup round 77, which is only done when a partial
	# match happens)
	.lcomm	L1_76_1,		16
	.lcomm	L1_76_2,		16
	.lcomm	L1_76_3,		16
	.lcomm	L1_76_4,		16

	# P and Q constants from the RC5 algorithm
	.set	P,		0xB7E15163
	.set	Q,		0x9E3779B9

	# Generates initial values of S[0..25]; we have S[i] = P+i*Q
	.macro	gen_S_const	i
	.int	(P+\i*Q) & 0xFFFFFFFF, (P+\i*Q) & 0xFFFFFFFF, (P+\i*Q) & 0xFFFFFFFF, (P+\i*Q) & 0xFFFFFFFF
	.if	\i < 26
	gen_S_const	"(\i+1)"
	.endif
	.endm

	.data
	.align	4
	# The initial values for S[0..25]
S_const:
	gen_S_const	0

	# The value of S[0] after key setup round 0; it's constant
S0:
	.int	0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D

	# Constants for incrementing the keys
inc_const:
	.int	0, 1, 2, 3

	# Byte-swapped value of 16 (= 0x10) for incrementing the keys
key_inc_const:
	.int	0x10000000, 0x10000000, 0x10000000, 0x10000000

	# Byte-swap permute mask
bswap:
	.int	0x03020100, 0x03020100, 0x03020100, 0x03020100

	.text

	# Return values for the core
	.set	RESULT_WORKING,		 0
	.set	RESULT_NOTHING,		 1
	.set	RESULT_FOUND,		 2

	# Offsets of struct RC5_72UnitWork members
	.set	offset_plain_hi,	 0
	.set	offset_plain_lo,	 4
	.set	offset_cypher_hi,	 8
	.set	offset_cypher_lo,	12
	.set	offset_L0_hi,		16
	.set	offset_L0_mid,		20
	.set	offset_L0_lo,		24
	.set	offset_check_count,	28
	.set	offset_check_hi,	32
	.set	offset_check_mid,	36
	.set	offset_check_lo,	40

	# register allocation:
	#   $2- $27: S1
	#  $28- $30: L1
	#
	#  $31- $56: S2
	#  $57- $59: L2
	#
	#  $60- $85: S3
	#  $86- $88: L3
	#
	#  $89-$114: S4
	# $115-$117: L4
	#
	# $118-$127:temp

	# Performs the key setup for the r-th round
.macro	KEY_SETUP	round
	# suffix 'p' as in previous
	.set	L1p,	 28+((\round+ 2) %  3)
	.set	L2p,	 57+((\round+ 2) %  3)
	.set	L3p,	 86+((\round+ 2) %  3)
	.set	L4p,	115+((\round+ 2) %  3)

	# suffix 'c' as in current
	.set	L1c,	 28+((\round   ) %  3)
	.set	L2c,	 57+((\round   ) %  3)
	.set	L3c,	 86+((\round   ) %  3)
	.set	L4c,	115+((\round   ) %  3)

	# suffix 'p' as in previous
	.set	S1p,	  2+((\round+25) % 26)
	.set	S2p,	 31+((\round+25) % 26)
	.set	S3p,	 60+((\round+25) % 26)
	.set	S4p,	 89+((\round+25) % 26)

	# suffix 'c' as in current
	.set	S1c,	  2+((\round   ) % 26)
	.set	S2c,	 31+((\round   ) % 26)
	.set	S3c,	 60+((\round   ) % 26)
	.set	S4c,	 89+((\round   ) % 26)

	# S[i+1] = (S[i+1] + S[i] + L[j]) <<< 3

	# For the first pass through the S[] array (i.e. when round <= 25),
	# read S[] from the constants in the S_const array; after that, read
	# the data from the S[] array itself, stored in registers.
.if \round <= 25
	a	$S1c, $118, $S1p
	a	$S2c, $118, $S2p
	a	$S3c, $118, $S3p
	a	$S4c, $118, $S4p
.else
	a	$S1c, $S1c, $S1p
	a	$S2c, $S2c, $S2p
	a	$S3c, $S3c, $S3p
	a	$S4c, $S4c, $S4p
.endif

	a	$S1c, $S1c, $L1p
	a	$S2c, $S2c, $L2p
	a	$S3c, $S3c, $L3p
	a	$S4c, $S4c, $L4p

.if	\round < 25
	roti	$S1c, $S1c,    3
	# Load S[] array entry for the next round
	lqa	$118, S_const+(\round+1)*16

	roti	$S2c, $S2c,    3
	lnop
.else
	roti	$S1c, $S1c,    3
	roti	$S2c, $S2c,    3
.endif
	roti	$S3c, $S3c,    3
	roti	$S4c, $S4c,    3

	# L[j+1] = (L[j+1] + S[i+1] + L[j]) <<< (S[i+1] + L[j])
	a	$126, $S1c, $L1p
	a	$127, $S2c, $L2p
	a	$L1c, $L1c, $126
	a	$L2c, $L2c, $127

	rot	$L1c, $L1c, $126
	rot	$L2c, $L2c, $127

	a	$126, $S3c, $L3p
	a	$127, $S4c, $L4p
	a	$L3c, $L3c, $126
	a	$L4c, $L4c, $127

	rot	$L3c, $L3c, $126
	rot	$L4c, $L4c, $127
.endm

.macro	ENCRYPTION	round
	# S1x = S[2i], S2x = S[2i + 1]
	.set	S11,	  4+2*\round
	.set	S12,	  4+2*\round+1
	.set	S21,	 33+2*\round
	.set	S22,	 33+2*\round+1
	.set	S31,	 62+2*\round
	.set	S32,	 62+2*\round+1
	.set	S41,	 91+2*\round
	.set	S42,	 91+2*\round+1

	# S[0] = ((S[0] ^ S[1]) <<< S[1]) + S[2i]
	xor	  $4,   $4,   $5
	xor	 $33,  $33,  $34
	xor	 $62,  $62,  $63
	xor	 $91,  $91,  $92

	rot	  $4,   $4,   $5
	rot	 $33,  $33,  $34
	rot	 $62,  $62,  $63
	rot	 $91,  $91,  $92

	a	  $4,   $4, $S11
	a	 $33,  $33, $S21
	a	 $62,  $62, $S31
	a	 $91,  $91, $S41

	# S[1] = ((S[1] ^ S[0]) <<< S[0]) + S[2i+1]
	xor	  $5,   $5,   $4
	xor	 $34,  $34,  $33
	xor	 $63,  $63,  $62
	xor	 $92,  $92,  $91

	rot	  $5,   $5,   $4
	rot	 $34,  $34,  $33
	rot	 $63,  $63,  $62
	rot	 $92,  $92,  $91

	a	  $5,   $5, $S12
	a	 $34,  $34, $S22
	a	 $63,  $63, $S32
	a	 $92,  $92, $S42
.endm

	.global rc5_72_unit_func_cellv1_spe_core
	.global _rc5_72_unit_func_cellv1_spe_core

rc5_72_unit_func_cellv1_spe_core:
_rc5_72_unit_func_cellv1_spe_core:

	# Allocate stack space
	il	  $2, -16*(127-80+1)
	a	 $SP,  $SP,   $2

	# Save non-volatile registers
	stqd	 $80,   0($SP)
	stqd	 $81,  16($SP)
	stqd	 $82,  32($SP)
	stqd	 $83,  48($SP)
	stqd	 $84,  64($SP)
	stqd	 $85,  80($SP)
	stqd	 $86,  96($SP)
	stqd	 $87, 112($SP)
	stqd	 $88, 128($SP)
	stqd	 $89, 144($SP)
	stqd	 $90, 160($SP)
	stqd	 $91, 176($SP)
	stqd	 $92, 192($SP)
	stqd	 $93, 208($SP)
	stqd	 $94, 224($SP)
	stqd	 $95, 240($SP)
	stqd	 $96, 256($SP)
	stqd	 $97, 272($SP)
	stqd	 $98, 288($SP)
	stqd	 $99, 304($SP)
	stqd	$100, 320($SP)
	stqd	$101, 336($SP)
	stqd	$102, 352($SP)
	stqd	$103, 368($SP)
	stqd	$104, 384($SP)
	stqd	$105, 400($SP)
	stqd	$106, 416($SP)
	stqd	$107, 432($SP)
	stqd	$108, 448($SP)
	stqd	$109, 464($SP)
	stqd	$110, 480($SP)
	stqd	$111, 496($SP)
	stqd	$112, 512($SP)
	stqd	$113, 528($SP)
	stqd	$114, 544($SP)
	stqd	$115, 560($SP)
	stqd	$116, 576($SP)
	stqd	$117, 592($SP)
	stqd	$118, 608($SP)
	stqd	$119, 624($SP)
	stqd	$120, 640($SP)
	stqd	$121, 656($SP)
	stqd	$122, 672($SP)
	stqd	$123, 688($SP)
	stqd	$124, 704($SP)
	stqd	$125, 720($SP)
	stqd	$126, 736($SP)
	stqd	$127, 752($SP)

	# Save a copy of rc5_72unitwork and iterations to memory
	stqa	  $3, rc5_72unitwork
	stqa	  $4, iterations_ptr

	# Load loop iteration count, initialize loop variable and let the
	# fist encryption round save a copy to memory
	lqd	$120, 0($4)
	rotqby	$120, $120,   $4
	stqa	$120, iterations

	# Constant for splats that will be needed in the following loads
	ilhu	$127, 0x0001
	iohl	$127, 0x0203

.macro	ReadStructMember	address, offset
	# Read value from memory
	lqd	$126, \offset($3)

	# Compute effective address offset($3)
	ai	$125,   $3, \offset

	# Move the desired member of the struct to the preferred slot
	rotqby	$126, $126, $125

	# Splat the desired member to remaining words of the register
	shufb	$126, $126, $126, $127

	# Save it to memory
	stqa	$126, \address
.endm

	# Load data from struct RC5_72UnitWork
	ReadStructMember	   plain_hi, offset_plain_hi
	ReadStructMember	   plain_lo, offset_plain_lo
	ReadStructMember	  cypher_hi, offset_cypher_hi
	ReadStructMember	  cypher_lo, offset_cypher_lo	
	ReadStructMember	      L0_hi, offset_L0_hi
	ReadStructMember	     L0_mid, offset_L0_mid
	ReadStructMember	      L0_lo, offset_L0_lo
	ReadStructMember	check_count, offset_check_count
	ReadStructMember	   check_hi, offset_check_hi
	ReadStructMember	  check_mid, offset_check_mid
	ReadStructMember	   check_lo, offset_check_lo

	lqa	$123, L0_hi
	lqa	$124, L0_mid
	lqa	$125, L0_lo
	lqa	$126, inc_const

	# Initialize S[0]
	lqa	  $2, S0
	lqa	 $31, S0
	lqa	 $60, S0
	lqa	 $89, S0

	# increment each word of key_hi in L1 by 0,...,3
	a	 $30, $123, $126
	# Load S[] array entry for key setup round 2
	lqa	$118, S_const+2*16

	# Each key in L2[2] is 4 more than the corresponding key in L1[2].
	# Similarly, each key in L3[2] is 4 more than the corresponding key
	# in L2[2], or 8 more than the corresponding key in L1[2]. And
	# finally, each key in L4[2] is 4 more than the corresponding key in
	# L3[2], which is 12 more than the corresponding key in L1[2].
	ai	 $59,  $30,    4
	ai	 $88,  $30,    8
	ai	$117,  $30,   12

	# Make sure the loop starts on an address == 0 mod 8, or instruction
	# scheduling goes out the window
	.align 3
new_key_lo:
	# Recompute and cache L[0] when key.lo changes; practically, key.lo
	# will never change within a block, so this code will only be
	# visited once. Hence its runtime is irrelevant to the keyrate of
	# the core. But I still went to the trouble of optimizing it (:

	#    L[0] = (L[0] + S[0] + 0) <<< (S[0] + 0)
	# => L[0] = (L[0] + S[0]) <<< S[0]
	#
	# This is computed directly for L1 and L2, then copied to L3 and L4
	a	 $28, $125,   $2
	a	 $57, $125,  $31

	rot	 $28,  $28,   $2
	rot	 $57,  $57,  $31

	nop
	nop

	# Copy to L3[0] and L4[0] and cache it in memory
	lr	 $86,  $28
	stqa	 $28, L0

	lr	$115,  $28
	lnop

	.align	3
new_key_mid:
	# Recompute and cache S[1] and L[1] when key.mid changes; this will
	# happen every 256 keys divided by 16 pipes = 16 iterations. Hence,
	# if this code snippet runs in time t, it adds t/16 to the execution
	# time of the core, so we need to be a little more careful with
	# scheduling.
	#
	# It also computes and caches S[2] and A + B = S[2] + L[1] for use
	# in key setup round 2.

	# Load the initial value of S[1] = P+Q
	ilhu	$118, (P+Q)@h
	# Load key.mid into L1[1]. We could use lr but it is a mnemonic for
	# an even pipeline instruction, and we need an odd pipeline
	# instruction to pair with the instruction above
	rotqbyi	 $29, $124,    0

	iohl	$118, (P+Q)@l
	# Load key.mid into L2[1]
	lr	 $58, $124

	# S[1] = (S[1] + S[0] + L[0]) <<< 3
	a	  $3,   $2, $118
	a	 $32,  $31, $118

	a	  $3,   $3,  $28
	a	 $32,  $32,  $57

	roti	  $3,   $3,    3
	roti	 $32,  $32,    3

	# Load S[] array entry for key setup round 2
	lqa	$118, S_const + 2*16
	lnop

	# L[1] = (L[1] + S[1] + L[0]) <<< (S[1] + L[0])
	a	$126,   $3,  $28
	# Copy L1[1] into L3[1] with an even pipeline instruction
	rotqbyi	 $61,   $3,    0

	a	$127,  $32,  $57
	# Copy L1[] into L4[1] with an even pipeline instruction
	rotqbyi	 $90,   $3,    0

	a	 $29,  $29, $126
	# Cache S[1] in memory
	stqa	  $3, S1

	a	 $58,  $58, $127
	lnop

	rot	 $29,  $29, $126	
	rot	 $58,  $58, $127

	nop
	nop

	# Copy L[1] to L3[1] and L4[1] and cache it in memory
	lr	 $87,  $29
	stqa	 $29, L1

	lr	$116,  $29
	lnop

	# S[2] = (S[2] + S[1] + L[1]) <<< 3
	a	  $4, $118,   $3
	a	 $33, $118,  $32
	a	 $62, $118,  $61
	a	 $91, $118,  $90

	a	  $4,   $4,  $29
	a	 $33,  $33,  $58
	a	 $62,  $62,  $87
	a	 $91,  $91, $116

	roti	  $4,   $4,    3
	roti	 $33,  $33,    3
	roti	 $62,  $62,    3
	roti	 $91,  $91,    3

	# Compute and cache S[1] + L[1] (i.e. A+B) for key setup round 2.
	#
	# Also cache S[2] in memory.
	a	  $7,   $4,  $29
	stqa	  $4, S2_1

	a	 $36,  $33,  $58
	stqa	 $33, S2_2

	a	 $65,  $62,  $87
	stqa	 $62, S2_3

	a	 $94,  $91, $116
	stqa	 $91, S2_4

	stqa	  $7, AB2_1
	stqa	 $36, AB2_2
	stqa	 $65, AB2_3
	stqa	 $94, AB2_4

	.align	3
new_key_hi:

	# Key setup round 2

	# L[2] = (L[2] + S[2] + L[1]) <<< (S[2] + L[1]), where S[2] + L[1]
	# is cached. This is interleaved with a bunch of loads and stores.
	a	 $30,  $30,   $7
	# Load S[] array entry for next round of key setup
	lqa	$118, S_const + 3*16

	a	 $59,  $59,  $36
	# Store incremented key.hi
	stqa	$123, L0_hi

	rot	 $30,  $30,   $7
	# Store incremented key.mid
	stqa	$124, L0_mid

	rot	 $59,  $59,  $36
	# Store incremented key.lo
	stqa	$125, L0_lo

	a	 $88,  $88,  $65
	# Load plain.lo, needed for encryption later
	lqa	$124, plain_lo

	a	$117, $117,  $94
	# Load plain.hi, needed for encryption later
	lqa	$125, plain_hi

	rot	 $88,  $88,  $65
	rot	$117, $117,  $94

	# Key setup rounds 3-76
	.align	3
	KEY_SETUP	  3
	KEY_SETUP	  4
	KEY_SETUP	  5
	KEY_SETUP	  6
	KEY_SETUP	  7
	KEY_SETUP	  8
	KEY_SETUP	  9
	KEY_SETUP	 10
	KEY_SETUP	 11
	KEY_SETUP	 12
	KEY_SETUP	 13
	KEY_SETUP	 14
	KEY_SETUP	 15
	KEY_SETUP	 16
	KEY_SETUP	 17
	KEY_SETUP	 18
	KEY_SETUP	 19
	KEY_SETUP	 20
	KEY_SETUP	 21
	KEY_SETUP	 22
	KEY_SETUP	 23
	KEY_SETUP	 24
	KEY_SETUP	 25
	KEY_SETUP	 26
	KEY_SETUP	 27
	KEY_SETUP	 28
	KEY_SETUP	 29
	KEY_SETUP	 30
	KEY_SETUP	 31
	KEY_SETUP	 32
	KEY_SETUP	 33
	KEY_SETUP	 34
	KEY_SETUP	 35
	KEY_SETUP	 36
	KEY_SETUP	 37
	KEY_SETUP	 38
	KEY_SETUP	 39
	KEY_SETUP	 40
	KEY_SETUP	 41
	KEY_SETUP	 42
	KEY_SETUP	 43
	KEY_SETUP	 44
	KEY_SETUP	 45
	KEY_SETUP	 46
	KEY_SETUP	 47
	KEY_SETUP	 48
	KEY_SETUP	 49
	KEY_SETUP	 50
	KEY_SETUP	 51
	KEY_SETUP	 52
	KEY_SETUP	 53
	KEY_SETUP	 54
	KEY_SETUP	 55
	KEY_SETUP	 56
	KEY_SETUP	 57
	KEY_SETUP	 58
	KEY_SETUP	 59
	KEY_SETUP	 60
	KEY_SETUP	 61
	KEY_SETUP	 62
	KEY_SETUP	 63
	KEY_SETUP	 64
	KEY_SETUP	 65
	KEY_SETUP	 66
	KEY_SETUP	 67
	KEY_SETUP	 68
	KEY_SETUP	 69
	KEY_SETUP	 70
	KEY_SETUP	 71
	KEY_SETUP	 72
	KEY_SETUP	 73
	KEY_SETUP	 74
	KEY_SETUP	 75
	KEY_SETUP	 76

	# Key setup round 77 is not done here. We don't need S[25] except to
	# compute B in the last encryption round, and we don't even bother
	# comparing B to cypher_hi unless A == cypher.lo, which happens with
	# low probability (1 in 2^32). If it matches, then we compute the
	# last key setup round and the second half of the last encryption
	# round outside the main loop.

	# Encryption pre-setup

	.align	3

	# S[0] = plain_lo + S[0], interleaved with a bunch of loads

	a	  $2, $124,   $2
	lqa	$126, key_inc_const

	a	 $31, $124,  $31
	lqa	$127, bswap

	a	 $60, $124,  $60
	lqa	$123, L0_hi

	a	 $89, $124,  $89
	lqa	$124, L0_mid

	# S[1] = plain_hi + S[1], interleaved with a bunch of loads and
	# stores

	a	  $3, $125,   $3
	# Store L1[1] at key setup round 76, needed for computing key setup
	# round 77 if required later.
	stqa	 $29, L1_76_1

	a	 $32, $125,  $32
	# Store L2[1] at key setup round 76, needed for computing key setup
	# round 77 if required later.
	stqa	 $58, L1_76_2

	a	 $61, $125,  $61
	# Store L3[1] at key setup round 76, needed for computing key setup
	# round 77 if required later.
	stqa	 $87, L1_76_3

	a	 $90, $125,  $90
	lqa	$125, L0_lo

	# Encryption round 0

	.align	3

	# S[0] = ((S[0] ^ S[1]) <<< S[1]) + S[2], interleaved with key
	# increment code
	xor	  $2,   $2,   $3
	# Byte-swap key.hi
	shufb	$123, $123, $123,  $127

	xor	 $31,  $31,  $32
	# Byte-swap key.mid
	shufb	$124, $124, $124,  $127

	xor	 $60,  $60,  $61
	# Load S[2] for next iteration of the loop
	lqa	$118, S_const+2*16

	xor	 $89,  $89,  $90
	# Load cached L[0] into L1[0]
	lqa	 $28, L0

	# Generate carry from key.hi increment
	cg	$121, $123, $126
	# Store L4[1] at key setup round 76, needed for computing key setup
	# round 77 if required later.
	stqa	$116, L1_76_4

	# Increment key.hi
	a	$123, $123, $126
	# Byte-swap key.lo
	shufb	$125, $125, $125,  $127

	# Generate carry from key.mid increment
	cg	$122, $124, $121

	# Load cached L[0] into L2[0]
	lqa	 $57, L0

	# Add carry from key.hi increment to key.mid
	a	$124, $124, $121
	# Byte-swap back key.hi
	shufb	$123, $123, $123, $127

	rot	  $2,   $2,   $3
	lqa	$119, inc_const

	# Add carry from key.mid increment to key.lo
	a	$125, $125, $122
	# Byte-swap back key.mid
	shufb	$124, $124, $124, $127

	rot	 $31,  $31,  $32
	# Load cached L[0] into L3[0]
	lqa	 $86, L0

	rot	 $60,  $60,  $61
	# Byte-swap back key.lo
	shufb	$125, $125, $125, $127

	rot	 $89,  $89,  $90
	# Move key.hi to L1[2] using an even pipeline instruction
	rotqbyi	 $30, $123,    0

	a	  $2,   $2,   $4
	# Load cached L[0] into L4[0]
	lqa	$115, L0

	a	 $31,  $31,  $33
	# Load cached L[1] into L1[1]
	lqa	 $29, L1

	a	 $60,  $60,  $62
	# Load cached L[1] into L2[1]
	lqa	 $58, L1

	# increment each word of key_hi in L1 by 0,...,3
	a	 $30,  $30, $119
	# Load cached L[1] into L3[1]
	lqa	 $87, L1

	a	 $89,  $89,  $91
	# Load cached L[1] into L4[1]
	lqa	$116, L1

	# Increment keys: L2[2] = L1[2] + {  4,  4,  4,  4 }
	ai	 $59,  $30,    4
	lqa	$120, iterations

	# Increment keys: L3[2] = L1[2] + {  8,  8,  8,  8 }
	ai	 $88,  $30,    8
	lqa	$126, cypher_lo

	# Increment keys: L4[2] = L1[2] + { 12, 12, 12, 12 }
	ai	$117,  $30,   12
	lqa	$127, cypher_hi

	# S[1] = ((S[1] ^ S[0]) <<< S[0]) + S[3]
	xor	  $3,   $3,   $2
	xor	 $32,  $32,  $31
	xor	 $61,  $61,  $60
	xor	 $90,  $90,  $89

	# Decrement iteration count
	ai	$120, $120,  -16

	rot	  $3,   $3,   $2
	rot	 $32,  $32,  $31
	rot	 $61,  $61,  $60
	rot	 $90,  $90,  $89
	# Store iteration count
	stqa	$120, iterations

	a	  $3,   $3,   $5
	a	 $32,  $32,  $34
	a	 $61,  $61,  $63
	a	 $90,  $90,  $92

	# Encryption round 1

	# S[0] = ((S[0] ^ S[1]) <<< S[1]) + S[2i], interleaved with a bunch
	# of loads.
	#
	# From now on we store S[0] and S[1] in S[2] and S[3] instead, since
	# we need to load S[0] and S[1] to be used in the next iteration of
	# the loop.
	xor	  $4,   $2,   $3
	# Load cached S[0] into S1[0]
	lqa	  $2, S0

	xor	 $33,  $31,  $32
	# Load cached S[0] into S2[0]
	lqa	 $31, S0

	xor	 $62,  $60,  $61
	# Load cached S[0] into S3[0]
	lqa	 $60, S0

	xor	 $91,  $89,  $90
	# Load cached S[0] into S4[0]
	lqa	 $89, S0

	rot	  $4,   $4,   $3
	rot	 $33,  $33,  $32
	rot	 $62,  $62,  $61
	rot	 $91,  $91,  $90

	a	  $4,   $4,   $6
	a	 $33,  $33,  $35
	a	 $62,  $62,  $64
	a	 $91,  $91,  $93

	# S[1] = ((S[1] ^ S[0]) <<< S[0]) + S[2i+1], interleaved with a
	# bunch of loads.
	xor	  $5,   $3,   $4
	# Load cached S[1] into S1[1]
	lqa	  $3, S1

	xor	 $34,  $32,  $33
	# Load cached S[1] into S2[1]
	lqa	 $32, S1

	xor	 $63,  $61,  $62
	# Load cached S[1] into S3[1]
	lqa	 $61, S1

	xor	 $92,  $90,  $91
	# Load cached S[1] into S4[1]
	lqa	 $90, S1

	rot	  $5,   $5,   $4
	rot	 $34,  $34,  $33
	rot	 $63,  $63,  $62
	rot	 $92,  $92,  $91

	a	  $5,   $5,   $7
	a	 $34,  $34,  $36
	a	 $63,  $63,  $65
	a	 $92,  $92,  $94

	# Encryption rounds 2-10
	.align	3
	ENCRYPTION	 2
	ENCRYPTION	 3
	ENCRYPTION	 4
	ENCRYPTION	 5
	ENCRYPTION	 6
	ENCRYPTION	 7
	ENCRYPTION	 8
	ENCRYPTION	 9
	ENCRYPTION	10

	# Encryption round 11, or rather, only the first half of it
	# (computation of A). As explained above, B is computed (outside the
	# main loop) only if the comparison A == cypher.lo succeeds.

	# Here's how the comparison is done. We compare A1, A2, A3 and A4 to
	# cypher.lo, so that each word partition of these registers contains
	# either 0x00000000 or 0xFFFFFFFF, depending on whether the result
	# was equal to cypher.lo or not. Then we use the gb (gather bits
	# from words) to copy the least-significant bits from each word
	# partition of the registers to the preferred slot of the
	# destination register. Then we branch based on that. If we do take
	# the branch and have to test matching keys, we can just test
	# whether bits 0-3 of the destination register are set.

	# S[0] = ((S[0] ^ S[1]) <<< S[1]) + S[2i]
	xor	  $4,   $4,   $5
	# Hint for the unconditional branch to the beginning of the loop.
	# It's the only branch we expect to take.
	#
	# Actually, we expect to take the branch to new_key_mid whenever the
	# carry from incrementing key.hi is set, but I have no idea how to
	# use this information to actually speed up the core. Remember, that
	# branch is only taken once every 16 loop iterations, so the
	# amortized cost due to pipeline flushen is 18/16 = 1.125 cycles.
	# If you can insert a hint for that branch without increasing the
	# execution time of the current code by more than 1 cycle, feel free
	# to do it.
	hbra	branch_new_key_hi, new_key_hi

	xor	 $33,  $33,  $34
	lnop

	rot	  $4,   $4,   $5
	# Load cached A+B for pipeline 1
	lqa	  $7, AB2_1

	rot	 $33,  $33,  $34
	# Load cached A+B for pipeline 2
	lqa	 $36, AB2_2

	xor	 $62,  $62,  $63
	# Load cached A+B for pipeline 3
	lqa	 $65, AB2_3

	xor	 $91,  $91,  $92
	# Load cached A+B for pipeline 4
	lqa	 $94, AB2_4

	rot	 $62,  $62,  $63
	rot	 $91,  $91,  $92

	a	  $4,   $4,  $26
	a	 $33,  $33,  $55

	# Test A1 == cypher.lo
	ceq	  $6,   $4, $126
	# Load cached S[2] into S1[2]
	lqa	  $4, S2_1

	# Test A2 == cypher.lo
	ceq	 $35,  $33, $126
	# Load cached S[2] into S2[2]
	lqa	 $33, S2_2

	a	 $62,  $62,  $84
	# Set preferred slot of comparison output register 1 with comparison
	# results from every word partition of A1 (see above)
 	gb	  $6,   $6

	a	 $91,  $91, $113
	# Set preferred slot of comparison output register 2 with comparison
	# results from every word partition of A2 (see above)
	gb	 $35,  $35

	# Test A3 == cypher.lo
	ceq	 $64,  $62, $126
	# Load cached S[2] into S3[2]
	lqa	 $62, S2_3

	# Test A4 == cypher.lo
	ceq	 $93,  $91, $126
	# Load cached S[2] into S4[2]
	lqa	 $91, S2_4

	# Set preferred slot of comparison output register 3 with comparison
	# results from every word partition of A3 (see above)
	gb	 $64,  $64
	# Set preferred slot of comparison output register 4 with comparison
	# results from every word partition of A4 (see above)
	gb	 $93,  $93

	.align	3
cmp_cypher:

test_key_1:
	# Branch if there was a partial match in one of the keys from A1
	brnz	  $6, match_key_1
test_key_2:
	# Branch if there was a partial match in one of the keys from A2
	brnz	 $35, match_key_2
test_key_3:
	# Branch if there was a partial match in one of the keys from A3
	brnz	 $64, match_key_3
test_key_4:
	# Branch if there was a partial match in one of the keys from A4
	brnz	 $93, match_key_4

	.align	3
loop:
	# Finished the loop; pack up and return to main()
	brz	$120, set_retval

	# key.lo was incremented; re-compute the cache
	brnz	$121, new_key_lo

	# key.mid was incremented; re-compute the cache
	brnz	$122, new_key_mid

branch_new_key_hi:
	# If key.lo and key.mid didn't change, start from key setup round 2
	# with cached values from previous rounds
	br	new_key_hi

set_retval:
	# Found nothing
	il	$127, RESULT_NOTHING

epilogue:
	# Save incremented keys to struct RC5_72UnitWork
	lqa	  $3, rc5_72unitwork

.macro	WriteStructMember	data, offset
	# Read original values
	lqd	  $4, \offset($3)

	# Generate insertion mask for the struct member
	cwd	  $5, \offset($3)

	# Insert the struct member
	shufb	  $4, \data,   $4,   $5

	# Write new values
	stqd	  $4, \offset($3)
.endm

	WriteStructMember	$123, offset_L0_hi
	WriteStructMember	$124, offset_L0_mid
	WriteStructMember	$125, offset_L0_lo

	# Return value goes on $3
	lr	  $3, $127

	# Restore registers
	lqd	 $80,   0($SP)
	lqd	 $81,  16($SP)
	lqd	 $82,  32($SP)
	lqd	 $83,  48($SP)
	lqd	 $84,  64($SP)
	lqd	 $85,  80($SP)
	lqd	 $86,  96($SP)
	lqd	 $87, 112($SP)
	lqd	 $88, 128($SP)
	lqd	 $89, 144($SP)
	lqd	 $90, 160($SP)
	lqd	 $91, 176($SP)
	lqd	 $92, 192($SP)
	lqd	 $93, 208($SP)
	lqd	 $94, 224($SP)
	lqd	 $95, 240($SP)
	lqd	 $96, 256($SP)
	lqd	 $97, 272($SP)
	lqd	 $98, 288($SP)
	lqd	 $99, 304($SP)
	lqd	$100, 320($SP)
	lqd	$101, 336($SP)
	lqd	$102, 352($SP)
	lqd	$103, 368($SP)
	lqd	$104, 384($SP)
	lqd	$105, 400($SP)
	lqd	$106, 416($SP)
	lqd	$107, 432($SP)
	lqd	$108, 448($SP)
	lqd	$109, 464($SP)
	lqd	$110, 480($SP)
	lqd	$111, 496($SP)
	lqd	$112, 512($SP)
	lqd	$113, 528($SP)
	lqd	$114, 544($SP)
	lqd	$115, 560($SP)
	lqd	$116, 576($SP)
	lqd	$117, 592($SP)
	lqd	$118, 608($SP)
	lqd	$119, 624($SP)
	lqd	$120, 640($SP)
	lqd	$121, 656($SP)
	lqd	$122, 672($SP)
	lqd	$123, 688($SP)
	lqd	$124, 704($SP)
	lqd	$125, 720($SP)
	lqd	$126, 736($SP)
	lqd	$127, 752($SP)

	# Restore stack pointer
	il	  $2, 16*(127-80+1)
	a	 $SP,  $SP,   $2

	# Return to caller
	bi	$LR


	# Save registers $118-$122
.macro SaveRegs
	stqa	$118, save_118
	stqa	$119, save_119
	stqa	$120, save_120
	stqa	$121, save_121
	stqa	$122, save_122
.endm

	# Restore registers $118-$122
.macro RestoreRegs
	lqa	$118, save_118
	lqa	$119, save_119
	lqa	$120, save_120
	lqa	$121, save_121
	lqa	$122, save_122
.endm

	# Due to the way comparisons are handled, we have that the register
	# corresponding to pipeline i has:
	# -bit 3 set if key 4*(i-1) out of 16 matched
	# -bit 2 set if key 4*(i-1)+1 out of 16 matched
	# -bit 1 set if key 4*(i-1)+2 out of 16 matched
	# -bit 0 set if key 4*(i-1)+3 out of 16 matched

match_key_1:
	# Compute key setup round 77 and second half of encryption round 11
	# for the first pipeline
	lqa	 $29, L1_76_1

	a	 $27,  $27,  $26
	a	 $27,  $27,  $29
	roti	 $27,  $27,    3

	xor	  $5,   $5, $126
	rot	  $5,   $5, $126
	a	  $5,   $5,  $27

	SaveRegs

test_match_key_11:
	andi	$121,   $6,    8
	brz	$121, test_match_key_12

	# Key 0 of 16
	il	$118,    0

	brsl	$119, report_cmc

test_match_key_12:
	andi	$121,   $6,    4
	brz	$121, test_match_key_13

	# Key 1 of 16
	il	$118,    1

	brsl	$119, report_cmc

test_match_key_13:
	andi	$121,   $6,    2
	brz	$121, test_match_key_14

	# Key 2 of 16
	il	$118,    2

	brsl	$119, report_cmc

test_match_key_14:
	andi	$121,   $6,    1
	brz	$121, test_full_match_key_1

	# Key 3 of 16
	il	$118,    3

	brsl	$119, report_cmc

test_full_match_key_1:
	# Keys 0-3
	il	$118,    0

	# Test B1 == cypher.hi
	ceq	$120,   $5, $127
	gb	$120, $120
	brnz	$120, found_key

	RestoreRegs

	# Back to the main loop
	br	test_key_2


match_key_2:
	# Compute key setup round 77 and second half of encryption round 11
	# for the second pipeline
	lqa	 $58, L1_76_2

	a	 $56,  $56,  $55
	a	 $56,  $56,  $58
	roti	 $56,  $56,    3

	xor	 $34,  $34, $126
	rot	 $34,  $34, $126
	a	 $34,  $34,  $56

	SaveRegs

test_match_key_21:
	andi	$121,  $35,    8
	brz	$121, test_match_key_22

	# Key 4 of 16
	il	$118,    4

	brsl	$119, report_cmc

test_match_key_22:
	andi	$121,  $35,    4
	brz	$121, test_match_key_23

	# Key 5 of 16
	il	$118,    5

	brsl	$119, report_cmc

test_match_key_23:
	andi	$121,  $35,    2
	brz	$121, test_match_key_24

	# Key 6 of 16
	il	$118,    6

	brsl	$119, report_cmc

test_match_key_24:
	andi	$121,  $35,    1
	brz	$121, test_full_match_key_2

	# Key 7 of 16
	il	$118,    7

	brsl	$119, report_cmc

test_full_match_key_2:
	# Keys 4-7
	il	$118,    4

	# Test B2 == cypher.hi
	ceq	$120,  $34, $127
	gb	$120, $120
	brnz	$120, found_key

	RestoreRegs

	# Back to the main loop
	br	test_key_3


match_key_3:
	# Compute key setup round 77 and second half of encryption round 11
	# for the third pipeline
	lqa	 $87, L1_76_3

	a	 $85,  $85,  $84
	a	 $85,  $85,  $87
	roti	 $85,  $85,    3

	xor	 $63,  $63, $126
	rot	 $63,  $63, $126
	a	 $63,  $63,  $85

	SaveRegs

test_match_key_31:
	andi	$121,  $64,    8
	brz	$121, test_match_key_32

	# Key 8 of 16
	il	$118,    8

	brsl	$119, report_cmc

test_match_key_32:
	andi	$121,  $64,    4
	brz	$121, test_match_key_33

	# Key 9 of 16
	il	$118,    9

	brsl	$119, report_cmc

test_match_key_33:
	andi	$121,  $64,    2
	brz	$121, test_match_key_34

	# Key 10 of 16
	il	$118,   10

	brsl	$119, report_cmc

test_match_key_34:
	andi	$121,  $64,    1
	brz	$121, test_full_match_key_3

	# Key 11 of 16
	il	$118,   11

	brsl	$119, report_cmc

test_full_match_key_3:

	# Keys 8-11
	il	$118,    8

	# Test B3 == cypher.hi
	ceq	$120,  $63, $127
	gb	$120, $120
	brnz	$120, found_key

	RestoreRegs

	# Back to the main loop
	br	test_key_4


match_key_4:
	# Compute key setup round 77 and second half of encryption round 11
	# for the fourth pipeline
	lqa	$116, L1_76_4

	a	$114, $114, $113
	a	$114, $114, $116
	roti	$114, $114,    3

	xor	 $92,  $92, $126
	rot	 $92,  $92, $126
	a	 $92,  $92, $114

	SaveRegs

test_match_key_41:
	andi	$121,  $93,    8
	brz	$121, test_match_key_42

	# Key 12 of 16
	il	$118,   12

	brsl	$119, report_cmc

test_match_key_42:
	andi	$121,  $93,    4
	brz	$121, test_match_key_43

	# Key 13 of 16
	il	$118,   13

	brsl	$119, report_cmc

test_match_key_43:
	andi	$121,  $93,    2
	brz	$121, test_match_key_44

	# Key 14 of 16
	il	$118,   14

	brsl	$119, report_cmc

test_match_key_44:
	andi	$121,  $93,    1
	brz	$121, test_full_match_key_4

	# Key 15 of 16
	il	$118,   15

	brsl	$119, report_cmc

test_full_match_key_4:
	# Keys 12-15
	il	$118,   12

	# Test B3 == cypher.hi
	ceq	$120,  $92, $127
	gb	$120, $120
	brnz	$120, found_key

	RestoreRegs

	# Back to the main loop
	br	loop


.macro	WriteStructMemberCMC	data, offset
	# Read original values
	lqd	$120, \offset($121)

	# Generate insertion mask for the struct member
	cwd	$122, \offset($121)

	# Insert the struct member
	shufb	$120, \data, $120, $122

	# Write new values
	stqd	$120, \offset($121)
.endm

report_cmc:
	lqa	$121, rc5_72unitwork

	lqa	$120, L0_hi

	# Add the key index (out of 16 keys checked per iteration)
	a	$118, $120, $118

	WriteStructMemberCMC	$118, offset_check_hi

	lqa	$118, L0_mid

	WriteStructMemberCMC	$118, offset_check_mid

	lqa	$118, L0_lo

	WriteStructMemberCMC	$118, offset_check_lo

	lqa	$118, check_count

	# check_count++
	ai	$118, $118,    1

	stqa	$118, check_count

	WriteStructMemberCMC	$118, offset_check_count

	# $119 is the link register for this function
	bi	$119

found_key:
test_found_key_1:
	andi	$121, $120,    8
	brz	$121, test_found_key_2

	# Keys 0,4,8,12/16 -- just branch to the end
	br	end_found_key

test_found_key_2:
	andi	$121, $120,    4
	brz	$121, test_found_key_3

	# Keys 1,5,9,13/16 -- add 1 and branch to the end
	ai	$118, $118,    1
	br	end_found_key

test_found_key_3:
	andi	$121, $120,    2
	brz	$121, test_found_key_4

	# Keys 2,6,10,14/16 -- add 2 and branch to the end
	ai	$118, $118,    2
	br	end_found_key

test_found_key_4:
	andi	$121, $120,    1
	brz	$121, end_found_key

	# Keys 3,7,11,15/16 -- add 3 and branch to the end
	ai	$118, $118,    3
	br	end_found_key

end_found_key:
	lqa	$121, iterations_ptr

	lqa	$120, iterations

	lqd	$119, 0($121)
	rotqby	$119, $119, $121

	sf	$119, $120, $119
	a	$118, $119, $118
	ai	$118, $118, -16

	WriteStructMemberCMC	$118,    0

	# We found it!
	il	$127, RESULT_FOUND

	# Return from the function
	br	epilogue
