# Copyright distributed.net 1997-2003 - All Rights Reserved
# For use in distributed.net projects only.
# Any other distribution or use of this source violates copyright.
#
# Author: Decio Luiz Gazzoni Filho <decio@distributed.net>
# $Id: r72-cellv1-spe.s,v 1.1.2.1 2007/08/02 08:08:37 decio Exp $

	.section bss
	.align	4

	.lcomm	save_118,		16
	.lcomm	save_119,		16
	.lcomm	save_120,		16
	.lcomm	save_121,		16
	.lcomm	save_122,		16
	.lcomm	rc5_72unitwork,		16
	.lcomm	iterations_ptr,		16
	.lcomm	iterations,		16
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
	.lcomm	L0,			16
	.lcomm	S1,			16
	.lcomm	L1,			16

	.set	P,		0xB7E15163
	.set	Q,		0x9E3779B9

	.macro	gen_S_const	i
	.int	(P+\i*Q) & 0xFFFFFFFF, (P+\i*Q) & 0xFFFFFFFF, (P+\i*Q) & 0xFFFFFFFF, (P+\i*Q) & 0xFFFFFFFF
	.if	\i < 26
	gen_S_const	"(\i+1)"
	.endif
	.endm

	.data
	.align	4
S_const:
	gen_S_const	0
S0:
	.int	0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D, 0xBF0A8B1D
inc_const:
	.int	0, 1, 2, 3
key_inc_const:
	.int	0x10000000, 0x10000000, 0x10000000, 0x10000000
bswap:
	.int	0x03020100, 0x03020100, 0x03020100, 0x03020100

	.text

	.set	RESULT_WORKING,		 0
	.set	RESULT_NOTHING,		 1
	.set	RESULT_FOUND,		 2

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

#  $31- $56: S2
#  $57- $59: L2

#  $60- $85: S3
#  $86- $88: L3

#  $89-$114: S4
# $115-$117: L4

# $118-$127:temp

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

.if	\round < 11
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
.else
	# S[1] = ((S[1] ^ S[0]) <<< S[0]) + S[2i+1]
	xor	  $5,   $5,   $4
	hbra	branch_new_key_hi, new_key_hi

	xor	 $34,  $34,  $33

	rot	  $5,   $5,   $4

	ceq	  $4,   $4, $126

	rot	 $34,  $34,  $33

	ceq	 $33,  $33, $126
 	gb	  $4,   $4

	xor	 $63,  $63,  $62
	xor	 $92,  $92,  $91

	rot	 $63,  $63,  $62
	gb	 $33,  $33

	ceq	 $62,  $62, $126
	rot	 $92,  $92,  $91

	ceq	 $91,  $91, $126
	gb	 $62,  $62

	a	  $5,   $5, $S12
	a	 $34,  $34, $S22

	a	 $63,  $63, $S32
	lnop

	a	 $92,  $92, $S42
	gb	 $91,  $91
.endif
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

	lqa	  $2, S0
	lqa	 $31, S0
	lqa	 $60, S0
	lqa	 $89, S0

#	lqa	 $28, L0
#	lqa	 $57, L0
#	lqa	 $86, L0
#	lqa	$115, L0

#	lqa	  $3, S1
#	lqa	 $32, S1
#	lqa	 $61, S1
#	lqa	 $90, S1

#	lqa	 $29, L1
#	lqa	 $58, L1
#	lqa	 $87, L1
#	lqa	$116, L1

	# increment each word of key_hi in L1 by 0,...,3
	a	 $30, $123, $126
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

	a	 $28, $125,   $2
	a	 $57, $125,  $31

	rot	 $28,  $28,   $2
	rot	 $57,  $57,  $31

	nop
	nop

	lr	 $86,  $28
	stqa	 $28, L0

	lr	$115,  $28
	lnop

	.align	3
new_key_mid:

	ilhu	$118, (P+Q)@h
	rotqbyi	 $29, $124,    0

	iohl	$118, (P+Q)@l
	lr	 $58, $124

	a	  $3,   $2, $118
	a	 $32,  $31, $118

	a	  $3,   $3,  $28
	a	 $32,  $32,  $57

	roti	  $3,   $3,    3
	roti	 $32,  $32,    3

	lqa	$118, S_const + 2*16
	lnop

	a	$126,   $3,  $28
	rotqbyi	 $61,   $3,    0

	a	$127,  $32,  $57
	rotqbyi	 $90,   $3,    0

	a	 $29,  $29, $126
	stqa	  $3, S1

	a	 $58,  $58, $127
	lnop

	rot	 $29,  $29, $126	
	rot	 $58,  $58, $127

	nop
	nop

	lr	 $87,  $29
	stqa	 $29, L1

	lr	$116,  $29
	lnop

	.align	3
new_key_hi:

	# Key setup round 2
	a	  $4, $118,   $3
	stqa	$123, L0_hi

	a	 $33, $118,  $32
	stqa	$124, L0_mid

	a	 $62, $118,  $61
	stqa	$125, L0_lo

	a	 $91, $118,  $90
	lqa	$124, plain_lo

	a	  $4,   $4,  $29
	lqa	$125, plain_hi

	a	 $33,  $33,  $58
	lqa	$118, S_const + 3*16

	a	 $62,  $62,  $87
	a	 $91,  $91, $116

	roti	  $4,   $4,    3
	roti	 $33,  $33,    3
	roti	 $62,  $62,    3
	roti	 $91,  $91,    3

	a	$126,   $4,  $29
	a	$127,  $33,  $58
	a	 $30,  $30, $126
	a	 $59,  $59, $127

	rot	 $30,  $30, $126
	rot	 $59,  $59, $127

	a	$126,  $62,  $87
	a	$127,  $91, $116
	a	 $88,  $88, $126
	a	$117, $117, $127

	rot	 $88,  $88, $126
	rot	$117, $117, $127

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
	KEY_SETUP	 77

	# Encryption pre-setup

	.align	3
	# S[0] = plain_lo + S[0]
	a	  $2, $124,   $2
	lqa	$126, key_inc_const

	a	 $31, $124,  $31
	lqa	$127, bswap

	a	 $60, $124,  $60
	lqa	$123, L0_hi

	a	 $89, $124,  $89
	lqa	$124, L0_mid

	# S[1] = plain_hi + S[1]

	a	  $3, $125,   $3
	a	 $32, $125,  $32

	a	 $61, $125,  $61
	lnop

	a	 $90, $125,  $90
	lqa	$125, L0_lo

	.align	3
	# Encryption round 0
	xor	  $2,   $2,   $3
	shufb	$123, $123, $123,  $127

	xor	 $31,  $31,  $32
	shufb	$124, $124, $124,  $127

	xor	 $60,  $60,  $61
	lqa	$118, S_const+2*16

	xor	 $89,  $89,  $90
	lqa	 $28, L0

	cg	$121, $123, $126
	lnop

	a	$123, $123, $126
	shufb	$125, $125, $125,  $127

	cg	$122, $124, $121
	lqa	 $57, L0

	a	$124, $124, $121
	shufb	$123, $123, $123, $127

	rot	  $2,   $2,   $3
	lqa	$119, inc_const

	a	$125, $125, $122
	shufb	$124, $124, $124, $127

	rot	 $31,  $31,  $32
	lqa	 $86, L0

	rot	 $60,  $60,  $61
	shufb	$125, $125, $125, $127

	rot	 $89,  $89,  $90
	rotqbyi	 $30, $123,    0

	a	  $2,   $2,   $4
	lqa	$115, L0

	a	 $31,  $31,  $33
	lqa	 $29, L1

	a	 $60,  $60,  $62
	lqa	 $58, L1

	# increment each word of key_hi in L1 by 0,...,3
	a	 $30,  $30, $119
	lqa	 $87, L1

	a	 $89,  $89,  $91
	lqa	$116, L1

	ai	 $59,  $30,    4
	lqa	$120, iterations

	ai	 $88,  $30,    8
	lqa	$126, cypher_lo

	ai	$117,  $30,   12
	lqa	$127, cypher_hi

	xor	  $3,   $3,   $2
	xor	 $32,  $32,  $31
	xor	 $61,  $61,  $60
	xor	 $90,  $90,  $89

	ai	$120, $120,  -16

	rot	  $3,   $3,   $2
	rot	 $32,  $32,  $31
	rot	 $61,  $61,  $60
	rot	 $90,  $90,  $89
	stqa	$120, iterations

	a	  $3,   $3,   $5
	a	 $32,  $32,  $34
	a	 $61,  $61,  $63
	a	 $90,  $90,  $92

	# Encryption round 1

	# S[0] = ((S[0] ^ S[1]) <<< S[1]) + S[2i]
	xor	  $4,   $2,   $3
	lqa	  $2, S0

	xor	 $33,  $31,  $32
	lqa	 $31, S0

	xor	 $62,  $60,  $61
	lqa	 $60, S0

	xor	 $91,  $89,  $90
	lqa	 $89, S0

	rot	  $4,   $4,   $3
	rot	 $33,  $33,  $32
	rot	 $62,  $62,  $61
	rot	 $91,  $91,  $90

	a	  $4,   $4,   $6
	a	 $33,  $33,  $35
	a	 $62,  $62,  $64
	a	 $91,  $91,  $93

	# S[1] = ((S[1] ^ S[0]) <<< S[0]) + S[2i+1]
	xor	  $5,   $3,   $4
	lqa	  $3, S1

	xor	 $34,  $32,  $33
	lqa	 $32, S1

	xor	 $63,  $61,  $62
	lqa	 $61, S1

	xor	 $92,  $90,  $91
	lqa	 $90, S1

	rot	  $5,   $5,   $4
	rot	 $34,  $34,  $33
	rot	 $63,  $63,  $62
	rot	 $92,  $92,  $91

	a	  $5,   $5,   $7
	a	 $34,  $34,  $36
	a	 $63,  $63,  $65
	a	 $92,  $92,  $94

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
	ENCRYPTION	11

	.align	3
cmp_cypher:

test_key_1:
	brnz	  $4, match_key_1
test_key_2:
	brnz	 $33, match_key_2
test_key_3:
	brnz	 $62, match_key_3
test_key_4:
	brnz	 $91, match_key_4

	.align	3
loop:

	brz	$120, set_retval

	brnz	$121, new_key_lo

	brnz	$122, new_key_mid

branch_new_key_hi:
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



.macro SaveRegs
	stqa	$118, save_118
	stqa	$119, save_119
	stqa	$120, save_120
	stqa	$121, save_121
	stqa	$122, save_122
.endm

.macro RestoreRegs
	lqa	$118, save_118
	lqa	$119, save_119
	lqa	$120, save_120
	lqa	$121, save_121
	lqa	$122, save_122
.endm

match_key_1:
	SaveRegs

test_match_key_11:
	andi	$121,   $4,    8
	brz	$121, test_match_key_12

	# Key 0 of 16
	il	$118,    0

	brsl	$119, report_cmc

test_match_key_12:
	andi	$121,   $4,    4
	brz	$121, test_match_key_13

	# Key 1 of 16
	il	$118,    1

	brsl	$119, report_cmc

test_match_key_13:
	andi	$121,   $4,    2
	brz	$121, test_match_key_14

	# Key 2 of 16
	il	$118,    2

	brsl	$119, report_cmc

test_match_key_14:
	andi	$121,   $4,    1
	brz	$121, test_full_match_key_1

	# Key 3 of 16
	il	$118,    3

	brsl	$119, report_cmc

test_full_match_key_1:
	# Keys 0-3
	il	$118,    0

	ceq	$120,   $5, $127
	gb	$120, $120
	brnz	$120, found_key

	RestoreRegs

	br	test_key_2


match_key_2:
	SaveRegs

test_match_key_21:
	andi	$121,  $33,    8
	brz	$121, test_match_key_22

	# Key 4 of 16
	il	$118,    4

	brsl	$119, report_cmc

test_match_key_22:
	andi	$121,  $33,    4
	brz	$121, test_match_key_23

	# Key 5 of 16
	il	$118,    5

	brsl	$119, report_cmc

test_match_key_23:
	andi	$121,  $33,    2
	brz	$121, test_match_key_24

	# Key 6 of 16
	il	$118,    6

	brsl	$119, report_cmc

test_match_key_24:
	andi	$121,  $33,    1
	brz	$121, test_full_match_key_2

	# Key 7 of 16
	il	$118,    7

	brsl	$119, report_cmc

test_full_match_key_2:
	# Keys 4-7
	il	$118,    4

	ceq	$120,  $34, $127
	gb	$120, $120
	brnz	$120, found_key

	RestoreRegs

	br	test_key_3


match_key_3:
	SaveRegs

test_match_key_31:
	andi	$121,  $62,    8
	brz	$121, test_match_key_32

	# Key 8 of 16
	il	$118,    8

	brsl	$119, report_cmc

test_match_key_32:
	andi	$121,  $62,    4
	brz	$121, test_match_key_33

	# Key 9 of 16
	il	$118,    9

	brsl	$119, report_cmc

test_match_key_33:
	andi	$121,  $62,    2
	brz	$121, test_match_key_34

	# Key 10 of 16
	il	$118,   10

	brsl	$119, report_cmc

test_match_key_34:
	andi	$121,  $62,    1
	brz	$121, test_full_match_key_3

	# Key 11 of 16
	il	$118,   11

	brsl	$119, report_cmc

test_full_match_key_3:

	# Keys 8-11
	il	$118,    8

	ceq	$120,  $63, $127
	gb	$120, $120
	brnz	$120, found_key

	RestoreRegs

	br	test_key_4


match_key_4:
	SaveRegs

test_match_key_41:
	andi	$121,  $91,    8
	brz	$121, test_match_key_42

	# Key 12 of 16
	il	$118,   12

	brsl	$119, report_cmc

test_match_key_42:
	andi	$121,  $91,    4
	brz	$121, test_match_key_43

	# Key 13 of 16
	il	$118,   13

	brsl	$119, report_cmc

test_match_key_43:
	andi	$121,  $91,    2
	brz	$121, test_match_key_44

	# Key 14 of 16
	il	$118,   14

	brsl	$119, report_cmc

test_match_key_44:
	andi	$121,  $91,    1
	brz	$121, test_full_match_key_4

	# Key 15 of 16
	il	$118,   15

	brsl	$119, report_cmc

test_full_match_key_4:
	# Keys 12-15
	il	$118,   12

	ceq	$120,  $92, $127
	gb	$120, $120
	brnz	$120, found_key

	RestoreRegs

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

	il	$127, RESULT_FOUND

	br	epilogue
