/*
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version
2 of the License, or (at your option) any later version.

A non-transferrable, royalty free license to this code
is granted to distributed.net for use exclusively in the DES Challenge II.

Copyright Biel/Switzerland, 21. Janaury 1999 by Christoph Dworzak

Changes for WinNT by Mike Marcelais, 25. January 1999

The s-boxes are based on the work of Matthew Kwan.

Version 3.2 (8.1MKey/s on PC164@500MHz)

$Log: deseval-dworz3.S,v $
Revision 1.4  1999/02/09 04:17:47  dworz
included the keyschedule into this asm-file
improved the keyschedule. an average of 4 sboxes less per key

Revision 1.3  1999/01/28 22:38:48  remi
Added $Log tag.

*/

	.set noat
//	.set noreorder
.text

#define K     0*8($9)
#define PT   56*8($9)
#define CT  120*8($9)
#define ct  184*8($9)
#define t   280*8($9)
#define pt  344*8($9)

	.align 5
	.globl checkKey
	.ent checkKey
checkKey:
	ldgp $29,0($27)
	lda $30,-64($30)
	stq $26,0($30)
	stq $9, 8($30)
	stq $10,16($30)
	stq $11,24($30)
	stq $12,32($30)
	stq $13,40($30)
	stq $14,48($30)
	stq $15,56($30)
	bis $16,$16,$9

again:
	bis $31,$31,$23		/* loopmask */
	ornot $31,$31,$24	/* headmask */
	ornot $31,$31,$25	/* tailmask */
	lda $27,pt

next:
	bsr $22,round1
	lda $27,96*8($27)

bit5:
	blbs $23,bit3
	ldq $0, 5*8+K
	xor $23,1,$23
	bis $31,2,$25
	eqv $0,$31,$0
	stq $0, 5*8+K
	br next
	
bit3:
	srl $23,1,$23
	blbs $23,bit11
	ldq $0, 3*8+K
	xor $23,1,$23
	bis $31,4,$25
	eqv $0,$31,$0
	sll $23,1,$23
	stq $0, 3*8+K
	br next

bit11:
	srl $23,1,$23
	blbs $23,bit8
	ldq $0,11*8+K
	xor $23,1,$23
	bis $31,4,$25
	eqv $0,$31,$0
	sll $23,2,$23
	stq $0,11*8+K
	br next

bit8:
	srl $23,1,$23
	blbs $23,bit43
	ldq $0, 8*8+K
	xor $23,1,$23
	bis $31,8,$25
	eqv $0,$31,$0
	sll $23,3,$23
	stq $0, 8*8+K
	br next

bit43:
	srl $23,1,$23
	blbs $23,bit42
	ldq $0,43*8+K
	xor $23,1,$23
	bis $31,8,$25
	eqv $0,$31,$0
	sll $23,4,$23
	stq $0,43*8+K
	br next

bit42:
	srl $23,1,$23
	blbs $23,head
	ldq $0,42*8+K
	xor $23,1,$23
	bis $31,16,$25
	eqv $0,$31,$0
	sll $23,5,$23
	stq $0,42*8+K
	br next

head:
	ldq $0,42*8+K
	ornot $31,$31,$25
	lda $27,pt
	eqv $0,$31,$0
	stq $0,42*8+K

bit10:
	srl $23,1,$23
	blbs $23,bit18
	ldq $0,10*8+K
	xor $23,1,$23
	bis $31,2,$24
	eqv $0,$31,$0
	sll $23,6,$23
	stq $0,10*8+K
	br next

bit18:
	srl $23,1,$23
	blbs $23,bit46
	ldq $0,18*8+K
	xor $23,1,$23
	bis $31,2,$24
	eqv $0,$31,$0
	sll $23,7,$23
	stq $0,18*8+K
	br next

bit46:
	srl $23,1,$23
	blbs $23,bit12
	ldq $0,46*8+K
	xor $23,1,$23
	bis $31, 4,$24
	eqv $0,$31,$0
	sll $23,8,$23
	stq $0,46*8+K
	br next

bit12:
	srl $23,1,$23
	blbs $23,bit15
	ldq $0,12*8+K
	xor $23,1,$23
	bis $31, 8,$24
	eqv $0,$31,$0
	sll $23,9,$23
	stq $0,12*8+K
	br next

bit15:
	srl $23,1,$23
	blbs $23,bit49
	ldq $0,15*8+K
	xor $23,1,$23
	bis $31,16,$24
	eqv $0,$31,$0
	sll $23,10,$23
	stq $0,15*8+K
	br next

bit49:
	srl $23,1,$23
	blbs $23,bit45
	ldq $0,49*8+K
	xor $23,1,$23
	bis $31,32,$24
	eqv $0,$31,$0
	sll $23,11,$23
	stq $0,49*8+K
	br next

bit45:
	srl $23,1,$23
	blbs $23,bit50
	ldq $0,45*8+K
	xor $23,1,$23
	bis $31,32,$24
	eqv $0,$31,$0
	sll $23,12,$23
	stq $0,45*8+K
	br next

bit50:
	srl $23,1,$23
	blbs $23,inv
	ldq $0,50*8+K
	xor $23,1,$23
	bis $31,64,$24
	eqv $0,$31,$0
	sll $23,13,$23
	stq $0,50*8+K
	br next

inv:
	srl $23,1,$23
	blbs $23,found
	xor $23,1,$23
	sll $23,14,$23
	ornot $31,$31,$24
	bis $31,56,$0
	lda $1,K
nextkey:
	ldq $2,0($1)
	eqv $2,$31,$2
	subq $0,1,$0
	stq $2,0($1)
	lda $1,8($1)
	bne $0,nextkey
	br next

found:
	bis $28,$28,$0
	ldq $26,0($30)
	ldq $9, 8($30)
	ldq $10,16($30)
	ldq $11,24($30)
	ldq $12,32($30)
	ldq $13,40($30)
	ldq $14,48($30)
	ldq $15,56($30)
	lda $30,64($30)
	ret $31,($26),1


round1:
	blbc $24,round2

	lda $13,32*8+PT
	lda $14, 0*8+PT
	lda $15, 0*8($27)

	ldq $16,47*8+K
	ldq $17,11*8+K
	ldq $18,26*8+K
	ldq $19, 3*8+K
	ldq $20,13*8+K
	ldq $21,41*8+K
	bsr $26,s1
	ldq $16,27*8+K
	ldq $17, 6*8+K
	ldq $18,54*8+K
	ldq $19,48*8+K
	ldq $20,39*8+K
	ldq $21,19*8+K
	bsr $26,s2
	ldq $16,53*8+K
	ldq $17,25*8+K
	ldq $18,33*8+K
	ldq $19,34*8+K
	ldq $20,17*8+K
	ldq $21, 5*8+K
	bsr $26,s3
	ldq $16, 4*8+K
	ldq $17,55*8+K
	ldq $18,24*8+K
	ldq $19,32*8+K
	ldq $20,40*8+K
	ldq $21,20*8+K
	bsr $26,s4
	ldq $16,36*8+K
	ldq $17,31*8+K
	ldq $18,21*8+K
	ldq $19, 8*8+K
	ldq $20,23*8+K
	ldq $21,52*8+K
	bsr $26,s5
	ldq $16,14*8+K
	ldq $17,29*8+K
	ldq $18,51*8+K
	ldq $19, 9*8+K
	ldq $20,35*8+K
	ldq $21,30*8+K
	bsr $26,s6
	ldq $16, 2*8+K
	ldq $17,37*8+K
	ldq $18,22*8+K
	ldq $19, 0*8+K
	ldq $20,42*8+K
	ldq $21,38*8+K
	bsr $26,s7
	ldq $16,16*8+K
	ldq $17,43*8+K
	ldq $18,44*8+K
	ldq $19, 1*8+K
	ldq $20, 7*8+K
	ldq $21,28*8+K
	bsr $26,s8

	lda $13, 0*8($27)
	lda $14,32*8+PT
	lda $15,32*8($27)

	ldq $16,54*8+K
	ldq $17,18*8+K
	ldq $18,33*8+K
	ldq $19,10*8+K
	ldq $20,20*8+K
	ldq $21,48*8+K
	bsr $26,s1
	ldq $16,34*8+K
	ldq $17,13*8+K
	ldq $18, 4*8+K
	ldq $19,55*8+K
	ldq $20,46*8+K
	ldq $21,26*8+K
	bsr $26,s2
	ldq $16, 3*8+K
	ldq $17,32*8+K
	ldq $18,40*8+K
	ldq $19,41*8+K
	ldq $20,24*8+K
	ldq $21,12*8+K
	bsr $26,s3
	ldq $16,11*8+K
	ldq $17, 5*8+K
	ldq $18, 6*8+K
	ldq $19,39*8+K
	ldq $20,47*8+K
	ldq $21,27*8+K
	bsr $26,s4
	ldq $16,43*8+K
	ldq $17,38*8+K
	ldq $18,28*8+K
	ldq $19,15*8+K
	ldq $20,30*8+K
	ldq $21, 0*8+K
	bsr $26,s5
	ldq $16,21*8+K
	ldq $17,36*8+K
	ldq $18,31*8+K
	ldq $19,16*8+K
	ldq $20,42*8+K
	ldq $21,37*8+K
	bsr $26,s6
	ldq $16, 9*8+K
	ldq $17,44*8+K
	ldq $18,29*8+K
	ldq $19, 7*8+K
	ldq $20,49*8+K
	ldq $21,45*8+K
	bsr $26,s7
	ldq $16,23*8+K
	ldq $17,50*8+K
	ldq $18,51*8+K
	ldq $19, 8*8+K
	ldq $20,14*8+K
	ldq $21,35*8+K
	bsr $26,s8

	br round3

round2:
	lda $13, 0*8($27)
	lda $14,32*8+PT
	lda $15,32*8($27)

round2.1:
	and $24,2,$0
	beq $0,round2.2

	ldq $16,54*8+K
	ldq $17,18*8+K
	ldq $18,33*8+K
	ldq $19,10*8+K
	ldq $20,20*8+K
	ldq $21,48*8+K
	bsr $26,s1

	br round3

round2.2:
	and $24,4,$0
	beq $0,round2.3

	ldq $16,34*8+K
	ldq $17,13*8+K
	ldq $18, 4*8+K
	ldq $19,55*8+K
	ldq $20,46*8+K
	ldq $21,26*8+K
	bsr $26,s2

	br round3

round2.3:
	and $24,8,$0
	beq $0,round2.4

	ldq $16, 3*8+K
	ldq $17,32*8+K
	ldq $18,40*8+K
	ldq $19,41*8+K
	ldq $20,24*8+K
	ldq $21,12*8+K
	bsr $26,s3

	br round3

round2.4:
	and $24,16,$0
	beq $0,round2.5

	ldq $16,43*8+K
	ldq $17,38*8+K
	ldq $18,28*8+K
	ldq $19,15*8+K
	ldq $20,30*8+K
	ldq $21, 0*8+K
	bsr $26,s5

	br round3

round2.5:
	and $24,32,$0
	beq $0,round2.6

	ldq $16, 9*8+K
	ldq $17,44*8+K
	ldq $18,29*8+K
	ldq $19, 7*8+K
	ldq $20,49*8+K
	ldq $21,45*8+K
	bsr $26,s7

	br round3

round2.6:
	and $24,64,$0
	beq $0,round3

	ldq $16,23*8+K
	ldq $17,50*8+K
	ldq $18,51*8+K
	ldq $19, 8*8+K
	ldq $20,14*8+K
	ldq $21,35*8+K
	bsr $26,s8

round3:
	lda $13,32*8($27)
	lda $14, 0*8($27)
	lda $15,64*8($27)

	ldq $16,25*8+K
	ldq $17,19*8+K
	ldq $18,20*8+K
	ldq $19,53*8+K
	ldq $20, 4*8+K
	ldq $21,41*8+K
	bsr $26,s4

	and $24,4+16+32+64,$0
	beq $0,round3.1

	ldq $16,11*8+K
	ldq $17,32*8+K
	ldq $18,47*8+K
	ldq $19,24*8+K
	ldq $20,34*8+K
	ldq $21, 5*8+K
	bsr $26,s1

round3.1:
	and $24,2+8+16+32+64,$0
	beq $0,round3.2

	ldq $16,48*8+K
	ldq $17,27*8+K
	ldq $18,18*8+K
	ldq $19,12*8+K
	ldq $20, 3*8+K
	ldq $21,40*8+K
	bsr $26,s2

round3.2:
	and $24,2+4+16+32,$0
	beq $0,round3.3

	ldq $16,17*8+K
	ldq $17,46*8+K
	ldq $18,54*8+K
	ldq $19,55*8+K
	ldq $20,13*8+K
	ldq $21,26*8+K
	bsr $26,s3

round3.3:
	and $24,2+4+8+64,$0
	beq $0,round3.4

	ldq $16, 2*8+K
	ldq $17,52*8+K
	ldq $18,42*8+K
	ldq $19,29*8+K
	ldq $20,44*8+K
	ldq $21,14*8+K
	bsr $26,s5

round3.4:
	and $24,2+8+16+32+64,$0
	beq $0,round3.5

	ldq $16,35*8+K
	ldq $17,50*8+K
	ldq $18,45*8+K
	ldq $19,30*8+K
	ldq $20, 1*8+K
	ldq $21,51*8+K
	bsr $26,s6

round3.5:
	and $24,4+8+16+64,$0
	beq $0,round3.6

	ldq $16,23*8+K
	ldq $17,31*8+K
	ldq $18,43*8+K
	ldq $19,21*8+K
	ldq $20, 8*8+K
	ldq $21, 0*8+K
	bsr $26,s7

round3.6:
	and $24,2+4+8+32,$0
	beq $0,round16

	ldq $16,37*8+K
	ldq $17, 9*8+K
	ldq $18,38*8+K
	ldq $19,22*8+K
	ldq $20,28*8+K
	ldq $21,49*8+K
	bsr $26,s8

round16:
	and $25,1,$0
	beq $0,round15

	lda $13,32*8+CT
	lda $14, 0*8+CT
	lda $15, 0*8+ct

	ldq $16,40*8+K
	ldq $17, 4*8+K
	ldq $18,19*8+K
	ldq $19,53*8+K
	ldq $20, 6*8+K
	ldq $21,34*8+K
	bsr $26,s1
	ldq $16,20*8+K
	ldq $17,24*8+K
	ldq $18,47*8+K
	ldq $19,41*8+K
	ldq $20,32*8+K
	ldq $21,12*8+K
	bsr $26,s2
	ldq $16,46*8+K
	ldq $17,18*8+K
	ldq $18,26*8+K
	ldq $19,27*8+K
	ldq $20,10*8+K
	ldq $21,55*8+K
	bsr $26,s3
	ldq $16,54*8+K
	ldq $17,48*8+K
	ldq $18,17*8+K
	ldq $19,25*8+K
	ldq $20,33*8+K
	ldq $21,13*8+K
	bsr $26,s4
	ldq $16,29*8+K
	ldq $17,51*8+K
	ldq $18,14*8+K
	ldq $19, 1*8+K
	ldq $20,16*8+K
	ldq $21,45*8+K
	bsr $26,s5
	ldq $16, 7*8+K
	ldq $17,22*8+K
	ldq $18,44*8+K
	ldq $19, 2*8+K
	ldq $20,28*8+K
	ldq $21,23*8+K
	bsr $26,s6
	ldq $16,50*8+K
	ldq $17,30*8+K
	ldq $18,15*8+K
	ldq $19,52*8+K
	ldq $20,35*8+K
	ldq $21,31*8+K
	bsr $26,s7
	ldq $16, 9*8+K
	ldq $17,36*8+K
	ldq $18,37*8+K
	ldq $19,49*8+K
	ldq $20, 0*8+K
	ldq $21,21*8+K
	bsr $26,s8

	lda $13, 0*8+ct
	lda $14,32*8+CT
	lda $15,32*8+ct

	ldq $16,33*8+K
	ldq $17,54*8+K
	ldq $18,12*8+K
	ldq $19,46*8+K
	ldq $20,24*8+K
	ldq $21,27*8+K
	bsr $26,s1
	ldq $16,13*8+K
	ldq $17,17*8+K
	ldq $18,40*8+K
	ldq $19,34*8+K
	ldq $20,25*8+K
	ldq $21, 5*8+K
	bsr $26,s2
	ldq $16,39*8+K
	ldq $17,11*8+K
	ldq $18,19*8+K
	ldq $19,20*8+K
	ldq $20, 3*8+K
	ldq $21,48*8+K
	bsr $26,s3
	ldq $16,47*8+K
	ldq $17,41*8+K
	ldq $18,10*8+K
	ldq $19,18*8+K
	ldq $20,26*8+K
	ldq $21, 6*8+K
	bsr $26,s4
	ldq $16,22*8+K
	ldq $17,44*8+K
	ldq $18, 7*8+K
	ldq $19,49*8+K
	ldq $20, 9*8+K
	ldq $21,38*8+K
	bsr $26,s5
	ldq $16, 0*8+K
	ldq $17,15*8+K
	ldq $18,37*8+K
	ldq $19,50*8+K
	ldq $20,21*8+K
	ldq $21,16*8+K
	bsr $26,s6
	ldq $16,43*8+K
	ldq $17,23*8+K
	ldq $18, 8*8+K
	ldq $19,45*8+K
	ldq $20,28*8+K
	ldq $21,51*8+K
	bsr $26,s7
	ldq $16, 2*8+K
	ldq $17,29*8+K
	ldq $18,30*8+K
	ldq $19,42*8+K
	ldq $20,52*8+K
	ldq $21,14*8+K
	bsr $26,s8

	br round14

round15:
	lda $13, 0*8+ct
	lda $14,32*8+CT
	lda $15,32*8+ct

round15.1:
	and $25,2,$0
	beq $0,round15.2

	ldq $16,13*8+K
	ldq $17,17*8+K
	ldq $18,40*8+K
	ldq $19,34*8+K
	ldq $20,25*8+K
	ldq $21, 5*8+K
	bsr $26,s2

	br round14

round15.2:
	and $25,4,$0
	beq $0,round15.3

	ldq $16,39*8+K
	ldq $17,11*8+K
	ldq $18,19*8+K
	ldq $19,20*8+K
	ldq $20, 3*8+K
	ldq $21,48*8+K
	bsr $26,s3

	br round14

round15.3:
	and $25,8,$0
	beq $0,round15.4

	ldq $16,43*8+K
	ldq $17,23*8+K
	ldq $18, 8*8+K
	ldq $19,45*8+K
	ldq $20,28*8+K
	ldq $21,51*8+K
	bsr $26,s7

	br round14

round15.4:
	and $25,16,$0
	beq $0,round14

	ldq $16, 2*8+K
	ldq $17,29*8+K
	ldq $18,30*8+K
	ldq $19,42*8+K
	ldq $20,52*8+K
	ldq $21,14*8+K
	bsr $26,s8

round14:
	lda $13,32*8+ct
	lda $14, 0*8+ct
	lda $15,64*8+ct

	ldq $16,33*8+K
	ldq $17,27*8+K
	ldq $18,53*8+K
	ldq $19, 4*8+K
	ldq $20,12*8+K
	ldq $21,17*8+K
	bsr $26,s4

	and $25,2+8+16,$0
	beq $0,round14.1

	ldq $16,19*8+K
	ldq $17,40*8+K
	ldq $18,55*8+K
	ldq $19,32*8+K
	ldq $20,10*8+K
	ldq $21,13*8+K
	bsr $26,s1

round14.1:
	and $25,4+8+16,$0
	beq $0,round14.2

	ldq $16,24*8+K
	ldq $17, 3*8+K
	ldq $18,26*8+K
	ldq $19,20*8+K
	ldq $20,11*8+K
	ldq $21,48*8+K
	bsr $26,s2

round14.2:
	and $25,2+8,$0
	beq $0,round14.3

	ldq $16,25*8+K
	ldq $17,54*8+K
	ldq $18, 5*8+K
	ldq $19, 6*8+K
	ldq $20,46*8+K
	ldq $21,34*8+K
	bsr $26,s3

round14.3:
	and $25,2+4+8+16,$0
	beq $0,round14.4

	ldq $16, 8*8+K
	ldq $17,30*8+K
	ldq $18,52*8+K
	ldq $19,35*8+K
	ldq $20,50*8+K
	ldq $21,51*8+K
	bsr $26,s5

round14.4:
	and $25,4+8+16,$0
	beq $0,round14.5

	ldq $16,45*8+K
	ldq $17, 1*8+K
	ldq $18,23*8+K
	ldq $19,36*8+K
	ldq $20, 7*8+K
	ldq $21, 2*8+K
	bsr $26,s6

round14.5:
	and $25,2+4+16,$0
	beq $0,round14.6

	ldq $16,29*8+K
	ldq $17, 9*8+K
	ldq $18,49*8+K
	ldq $19,31*8+K
	ldq $20,14*8+K
	ldq $21,37*8+K
	bsr $26,s7

round14.6:
	and $25,2+4+8,$0
	beq $0,round13

	ldq $16,43*8+K
	ldq $17,15*8+K
	ldq $18,16*8+K
	ldq $19,28*8+K
	ldq $20,38*8+K
	ldq $21, 0*8+K
	bsr $26,s8

round13:
	lda $13,64*8+ct
	lda $14,32*8+ct
	lda $15, 0*8+t

	ldq $16, 5*8+K
	ldq $17,26*8+K
	ldq $18,41*8+K
	ldq $19,18*8+K
	ldq $20,53*8+K
	ldq $21,24*8+K
	bsr $26,s1
	ldq $16,10*8+K
	ldq $17,46*8+K
	ldq $18,12*8+K
	ldq $19, 6*8+K
	ldq $20,54*8+K
	ldq $21,34*8+K
	bsr $26,s2
	ldq $16,11*8+K
	ldq $17,40*8+K
	ldq $18,48*8+K
	ldq $19,17*8+K
	ldq $20,32*8+K
	ldq $21,20*8+K
	bsr $26,s3
	ldq $16,19*8+K
	ldq $17,13*8+K
	ldq $18,39*8+K
	ldq $19,47*8+K
	ldq $20,55*8+K
	ldq $21, 3*8+K
	bsr $26,s4
	ldq $16,49*8+K
	ldq $17,16*8+K
	ldq $18,38*8+K
	ldq $19,21*8+K
	ldq $20,36*8+K
	ldq $21,37*8+K
	bsr $26,s5
	ldq $16,31*8+K
	ldq $17,42*8+K
	ldq $18, 9*8+K
	ldq $19,22*8+K
	ldq $20,52*8+K
	ldq $21,43*8+K
	bsr $26,s6
	ldq $16,15*8+K
	ldq $17,50*8+K
	ldq $18,35*8+K
	ldq $19,44*8+K
	ldq $20, 0*8+K
	ldq $21,23*8+K
	bsr $26,s7
	ldq $16,29*8+K
	ldq $17, 1*8+K
	ldq $18, 2*8+K
	ldq $19,14*8+K
	ldq $20,51*8+K
	ldq $21,45*8+K
	bsr $26,s8

round12:
	lda $13, 0*8+t
	lda $14,64*8+ct
	lda $15,32*8+t

	ldq $16,48*8+K
	ldq $17,12*8+K
	ldq $18,27*8+K
	ldq $19, 4*8+K
	ldq $20,39*8+K
	ldq $21,10*8+K
	bsr $26,s1
	ldq $16,53*8+K
	ldq $17,32*8+K
	ldq $18,55*8+K
	ldq $19,17*8+K
	ldq $20,40*8+K
	ldq $21,20*8+K
	bsr $26,s2
	ldq $16,54*8+K
	ldq $17,26*8+K
	ldq $18,34*8+K
	ldq $19, 3*8+K
	ldq $20,18*8+K
	ldq $21, 6*8+K
	bsr $26,s3
	ldq $16, 5*8+K
	ldq $17,24*8+K
	ldq $18,25*8+K
	ldq $19,33*8+K
	ldq $20,41*8+K
	ldq $21,46*8+K
	bsr $26,s4
	ldq $16,35*8+K
	ldq $17, 2*8+K
	ldq $18,51*8+K
	ldq $19, 7*8+K
	ldq $20,22*8+K
	ldq $21,23*8+K
	bsr $26,s5
	ldq $16,44*8+K
	ldq $17,28*8+K
	ldq $18,50*8+K
	ldq $19, 8*8+K
	ldq $20,38*8+K
	ldq $21,29*8+K
	bsr $26,s6
	ldq $16, 1*8+K
	ldq $17,36*8+K
	ldq $18,21*8+K
	ldq $19,30*8+K
	ldq $20,45*8+K
	ldq $21, 9*8+K
	bsr $26,s7
	ldq $16,15*8+K
	ldq $17,42*8+K
	ldq $18,43*8+K
	ldq $19, 0*8+K
	ldq $20,37*8+K
	ldq $21,31*8+K
	bsr $26,s8

round11:
	lda $13,32*8+t
	lda $14, 0*8+t
	lda $15, 0*8+t

	ldq $16,34*8+K
	ldq $17,55*8+K
	ldq $18,13*8+K
	ldq $19,47*8+K
	ldq $20,25*8+K
	ldq $21,53*8+K
	bsr $26,s1
	ldq $16,39*8+K
	ldq $17,18*8+K
	ldq $18,41*8+K
	ldq $19, 3*8+K
	ldq $20,26*8+K
	ldq $21, 6*8+K
	bsr $26,s2
	ldq $16,40*8+K
	ldq $17,12*8+K
	ldq $18,20*8+K
	ldq $19,46*8+K
	ldq $20, 4*8+K
	ldq $21,17*8+K
	bsr $26,s3
	ldq $16,48*8+K
	ldq $17,10*8+K
	ldq $18,11*8+K
	ldq $19,19*8+K
	ldq $20,27*8+K
	ldq $21,32*8+K
	bsr $26,s4
	ldq $16,21*8+K
	ldq $17,43*8+K
	ldq $18,37*8+K
	ldq $19,52*8+K
	ldq $20, 8*8+K
	ldq $21, 9*8+K
	bsr $26,s5
	ldq $16,30*8+K
	ldq $17,14*8+K
	ldq $18,36*8+K
	ldq $19,49*8+K
	ldq $20,51*8+K
	ldq $21,15*8+K
	bsr $26,s6
	ldq $16,42*8+K
	ldq $17,22*8+K
	ldq $18, 7*8+K
	ldq $19,16*8+K
	ldq $20,31*8+K
	ldq $21,50*8+K
	bsr $26,s7
	ldq $16, 1*8+K
	ldq $17,28*8+K
	ldq $18,29*8+K
	ldq $19,45*8+K
	ldq $20,23*8+K
	ldq $21,44*8+K
	bsr $26,s8

round10:
	lda $13, 0*8+t
	lda $14,32*8+t
	lda $15,32*8+t

	ldq $16,20*8+K
	ldq $17,41*8+K
	ldq $18,24*8+K
	ldq $19,33*8+K
	ldq $20,11*8+K
	ldq $21,39*8+K
	bsr $26,s1
	ldq $16,25*8+K
	ldq $17, 4*8+K
	ldq $18,27*8+K
	ldq $19,46*8+K
	ldq $20,12*8+K
	ldq $21,17*8+K
	bsr $26,s2
	ldq $16,26*8+K
	ldq $17,55*8+K
	ldq $18, 6*8+K
	ldq $19,32*8+K
	ldq $20,47*8+K
	ldq $21, 3*8+K
	bsr $26,s3
	ldq $16,34*8+K
	ldq $17,53*8+K
	ldq $18,54*8+K
	ldq $19, 5*8+K
	ldq $20,13*8+K
	ldq $21,18*8+K
	bsr $26,s4
	ldq $16, 7*8+K
	ldq $17,29*8+K
	ldq $18,23*8+K
	ldq $19,38*8+K
	ldq $20,49*8+K
	ldq $21,50*8+K
	bsr $26,s5
	ldq $16,16*8+K
	ldq $17, 0*8+K
	ldq $18,22*8+K
	ldq $19,35*8+K
	ldq $20,37*8+K
	ldq $21, 1*8+K
	bsr $26,s6
	ldq $16,28*8+K
	ldq $17, 8*8+K
	ldq $18,52*8+K
	ldq $19, 2*8+K
	ldq $20,44*8+K
	ldq $21,36*8+K
	bsr $26,s7
	ldq $16,42*8+K
	ldq $17,14*8+K
	ldq $18,15*8+K
	ldq $19,31*8+K
	ldq $20, 9*8+K
	ldq $21,30*8+K
	bsr $26,s8

round9:
	lda $13,32*8+t
	lda $14, 0*8+t
	lda $15, 0*8+t

	ldq $16, 6*8+K
	ldq $17,27*8+K
	ldq $18,10*8+K
	ldq $19,19*8+K
	ldq $20,54*8+K
	ldq $21,25*8+K
	bsr $26,s1
	ldq $16,11*8+K
	ldq $17,47*8+K
	ldq $18,13*8+K
	ldq $19,32*8+K
	ldq $20,55*8+K
	ldq $21, 3*8+K
	bsr $26,s2
	ldq $16,12*8+K
	ldq $17,41*8+K
	ldq $18,17*8+K
	ldq $19,18*8+K
	ldq $20,33*8+K
	ldq $21,46*8+K
	bsr $26,s3
	ldq $16,20*8+K
	ldq $17,39*8+K
	ldq $18,40*8+K
	ldq $19,48*8+K
	ldq $20,24*8+K
	ldq $21, 4*8+K
	bsr $26,s4
	ldq $16,52*8+K
	ldq $17,15*8+K
	ldq $18, 9*8+K
	ldq $19,51*8+K
	ldq $20,35*8+K
	ldq $21,36*8+K
	bsr $26,s5
	ldq $16, 2*8+K
	ldq $17,45*8+K
	ldq $18, 8*8+K
	ldq $19,21*8+K
	ldq $20,23*8+K
	ldq $21,42*8+K
	bsr $26,s6
	ldq $16,14*8+K
	ldq $17,49*8+K
	ldq $18,38*8+K
	ldq $19,43*8+K
	ldq $20,30*8+K
	ldq $21,22*8+K
	bsr $26,s7
	ldq $16,28*8+K
	ldq $17, 0*8+K
	ldq $18, 1*8+K
	ldq $19,44*8+K
	ldq $20,50*8+K
	ldq $21,16*8+K
	bsr $26,s8

round8:
	lda $13, 0*8+t
	lda $14,32*8+t
	lda $15,32*8+t

	ldq $16,24*8+K
	ldq $17,20*8+K
	ldq $18, 3*8+K
	ldq $19,12*8+K
	ldq $20,47*8+K
	ldq $21,18*8+K
	bsr $26,s1
	ldq $16, 4*8+K
	ldq $17,40*8+K
	ldq $18, 6*8+K
	ldq $19,25*8+K
	ldq $20,48*8+K
	ldq $21,53*8+K
	bsr $26,s2
	ldq $16, 5*8+K
	ldq $17,34*8+K
	ldq $18,10*8+K
	ldq $19,11*8+K
	ldq $20,26*8+K
	ldq $21,39*8+K
	bsr $26,s3
	ldq $16,13*8+K
	ldq $17,32*8+K
	ldq $18,33*8+K
	ldq $19,41*8+K
	ldq $20,17*8+K
	ldq $21,54*8+K
	bsr $26,s4
	ldq $16,45*8+K
	ldq $17, 8*8+K
	ldq $18, 2*8+K
	ldq $19,44*8+K
	ldq $20,28*8+K
	ldq $21,29*8+K
	bsr $26,s5
	ldq $16,50*8+K
	ldq $17,38*8+K
	ldq $18, 1*8+K
	ldq $19,14*8+K
	ldq $20,16*8+K
	ldq $21,35*8+K
	bsr $26,s6
	ldq $16, 7*8+K
	ldq $17,42*8+K
	ldq $18,31*8+K
	ldq $19,36*8+K
	ldq $20,23*8+K
	ldq $21,15*8+K
	bsr $26,s7
	ldq $16,21*8+K
	ldq $17,52*8+K
	ldq $18,49*8+K
	ldq $19,37*8+K
	ldq $20,43*8+K
	ldq $21, 9*8+K
	bsr $26,s8

round7:
	lda $13,32*8+t
	lda $14, 0*8+t
	lda $15, 0*8+t

	ldq $16,10*8+K
	ldq $17, 6*8+K
	ldq $18,46*8+K
	ldq $19,55*8+K
	ldq $20,33*8+K
	ldq $21, 4*8+K
	bsr $26,s1
	ldq $16,47*8+K
	ldq $17,26*8+K
	ldq $18,17*8+K
	ldq $19,11*8+K
	ldq $20,34*8+K
	ldq $21,39*8+K
	bsr $26,s2
	ldq $16,48*8+K
	ldq $17,20*8+K
	ldq $18,53*8+K
	ldq $19,54*8+K
	ldq $20,12*8+K
	ldq $21,25*8+K
	bsr $26,s3
	ldq $16,24*8+K
	ldq $17,18*8+K
	ldq $18,19*8+K
	ldq $19,27*8+K
	ldq $20, 3*8+K
	ldq $21,40*8+K
	bsr $26,s4
	ldq $16,31*8+K
	ldq $17,49*8+K
	ldq $18,43*8+K
	ldq $19,30*8+K
	ldq $20,14*8+K
	ldq $21,15*8+K
	bsr $26,s5
	ldq $16,36*8+K
	ldq $17,51*8+K
	ldq $18,42*8+K
	ldq $19, 0*8+K
	ldq $20, 2*8+K
	ldq $21,21*8+K
	bsr $26,s6
	ldq $16,52*8+K
	ldq $17,28*8+K
	ldq $18,44*8+K
	ldq $19,22*8+K
	ldq $20, 9*8+K
	ldq $21, 1*8+K
	bsr $26,s7
	ldq $16, 7*8+K
	ldq $17,38*8+K
	ldq $18,35*8+K
	ldq $19,23*8+K
	ldq $20,29*8+K
	ldq $21,50*8+K
	bsr $26,s8

round6:
	lda $13, 0*8+t
	lda $14,32*8+t
	lda $15,32*8+t

	ldq $16,53*8+K
	ldq $17,17*8+K
	ldq $18,32*8+K
	ldq $19,41*8+K
	ldq $20,19*8+K
	ldq $21,47*8+K
	bsr $26,s1
	ldq $16,33*8+K
	ldq $17,12*8+K
	ldq $18, 3*8+K
	ldq $19,54*8+K
	ldq $20,20*8+K
	ldq $21,25*8+K
	bsr $26,s2
	ldq $16,34*8+K
	ldq $17, 6*8+K
	ldq $18,39*8+K
	ldq $19,40*8+K
	ldq $20,55*8+K
	ldq $21,11*8+K
	bsr $26,s3
	ldq $16,10*8+K
	ldq $17, 4*8+K
	ldq $18, 5*8+K
	ldq $19,13*8+K
	ldq $20,46*8+K
	ldq $21,26*8+K
	bsr $26,s4
	ldq $16,44*8+K
	ldq $17,35*8+K
	ldq $18,29*8+K
	ldq $19,16*8+K
	ldq $20, 0*8+K
	ldq $21, 1*8+K
	bsr $26,s5
	ldq $16,22*8+K
	ldq $17,37*8+K
	ldq $18,28*8+K
	ldq $19,45*8+K
	ldq $20,43*8+K
	ldq $21, 7*8+K
	bsr $26,s6
	ldq $16,38*8+K
	ldq $17,14*8+K
	ldq $18,30*8+K
	ldq $19, 8*8+K
	ldq $20,50*8+K
	ldq $21,42*8+K
	bsr $26,s7
	ldq $16,52*8+K
	ldq $17,51*8+K
	ldq $18,21*8+K
	ldq $19, 9*8+K
	ldq $20,15*8+K
	ldq $21,36*8+K
	bsr $26,s8

round5:
	lda $13,32*8+t
	lda $14, 0*8+t
	lda $15, 0*8+t

	ldq $16,39*8+K
	ldq $17, 3*8+K
	ldq $18,18*8+K
	ldq $19,27*8+K
	ldq $20, 5*8+K
	ldq $21,33*8+K
	bsr $26,s1

	ornot $31,$31,$28
	ldq $1, 1*8+64*8($27)
	ldq $2, 1*8($15)
	ldq $3,24*8+64*8($27)
	ldq $4,24*8($15)
	ldq $5, 2*8+64*8($27)
	eqv $1,$2,$1
	ldq $6, 2*8($15)
	eqv $3,$4,$3
	ldq $7,25*8+64*8($27)
	ldq $8,25*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,19*8+K
	ldq $17,55*8+K
	ldq $18,46*8+K
	ldq $19,40*8+K
	ldq $20, 6*8+K
	ldq $21,11*8+K
	bsr $26,s2

	ldq $1,18*8+64*8($27)
	ldq $2,18*8($15)
	ldq $3, 7*8+64*8($27)
	ldq $4, 7*8($15)
	ldq $5, 5*8+64*8($27)
	eqv $1,$2,$1
	ldq $6, 5*8($15)
	eqv $3,$4,$3
	ldq $7,12*8+64*8($27)
	ldq $8,12*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,20*8+K
	ldq $17,17*8+K
	ldq $18,25*8+K
	ldq $19,26*8+K
	ldq $20,41*8+K
	ldq $21,54*8+K
	bsr $26,s3

	ldq $1,23*8+64*8($27)
	ldq $2,23*8($15)
	ldq $3,20*8+64*8($27)
	ldq $4,20*8($15)
	ldq $5,30*8+64*8($27)
	eqv $1,$2,$1
	ldq $6,30*8($15)
	eqv $3,$4,$3
	ldq $7,29*8+64*8($27)
	ldq $8,29*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,53*8+K
	ldq $17,47*8+K
	ldq $18,48*8+K
	ldq $19,24*8+K
	ldq $20,32*8+K
	ldq $21,12*8+K
	bsr $26,s4

	ldq $1, 4*8+64*8($27)
	ldq $2, 4*8($15)
	ldq $3,13*8+64*8($27)
	ldq $4,13*8($15)
	ldq $5, 6*8+64*8($27)
	eqv $1,$2,$1
	ldq $6, 6*8($15)
	eqv $3,$4,$3
	ldq $7, 3*8+64*8($27)
	ldq $8, 3*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,30*8+K
	ldq $17,21*8+K
	ldq $18,15*8+K
	ldq $19, 2*8+K
	ldq $20,45*8+K
	ldq $21,42*8+K
	bsr $26,s5

	ldq $1,11*8+64*8($27)
	ldq $2,11*8($15)
	ldq $3,22*8+64*8($27)
	ldq $4,22*8($15)
	ldq $5, 0*8+64*8($27)
	eqv $1,$2,$1
	ldq $6, 0*8($15)
	eqv $3,$4,$3
	ldq $7,31*8+64*8($27)
	ldq $8,31*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16, 8*8+K
	ldq $17,23*8+K
	ldq $18,14*8+K
	ldq $19,31*8+K
	ldq $20,29*8+K
	ldq $21,52*8+K
	bsr $26,s6

	ldq $1,15*8+64*8($27)
	ldq $2,15*8($15)
	ldq $3, 9*8+64*8($27)
	ldq $4, 9*8($15)
	ldq $5,10*8+64*8($27)
	eqv $1,$2,$1
	ldq $6,10*8($15)
	eqv $3,$4,$3
	ldq $7,16*8+64*8($27)
	ldq $8,16*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,51*8+K
	ldq $17, 0*8+K
	ldq $18,16*8+K
	ldq $19,49*8+K
	ldq $20,36*8+K
	ldq $21,28*8+K
	bsr $26,s7

	ldq $1,28*8+64*8($27)
	ldq $2,28*8($15)
	ldq $3,14*8+64*8($27)
	ldq $4,14*8($15)
	ldq $5,21*8+64*8($27)
	eqv $1,$2,$1
	ldq $6,21*8($15)
	eqv $3,$4,$3
	ldq $7,27*8+64*8($27)
	ldq $8,27*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,38*8+K
	ldq $17,37*8+K
	ldq $18, 7*8+K
	ldq $19,50*8+K
	ldq $20, 1*8+K
	ldq $21,22*8+K
	bsr $26,s8

	ldq $1,19*8+64*8($27)
	ldq $2,19*8($15)
	ldq $3,26*8+64*8($27)
	ldq $4,26*8($15)
	ldq $5, 8*8+64*8($27)
	eqv $1,$2,$1
	ldq $6, 8*8($15)
	eqv $3,$4,$3
	ldq $7,17*8+64*8($27)
	ldq $8,17*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return


round4:
	lda $13, 0*8+t
	lda $14,32*8+t
	lda $15,32*8+t

	ldq $16,25*8+K
	ldq $17,46*8+K
	ldq $18, 4*8+K
	ldq $19,13*8+K
	ldq $20,48*8+K
	ldq $21,19*8+K
	bsr $26,s1

	ldq $1, 1*8+32*8($27)
	ldq $2, 1*8($15)
	ldq $3,24*8+32*8($27)
	ldq $4,24*8($15)
	ldq $5, 2*8+32*8($27)
	eqv $1,$2,$1
	ldq $6, 2*8($15)
	eqv $3,$4,$3
	ldq $7,25*8+32*8($27)
	ldq $8,25*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16, 5*8+K
	ldq $17,41*8+K
	ldq $18,32*8+K
	ldq $19,26*8+K
	ldq $20,17*8+K
	ldq $21,54*8+K
	bsr $26,s2

	ldq $1,18*8+32*8($27)
	ldq $2,18*8($15)
	ldq $3, 7*8+32*8($27)
	ldq $4, 7*8($15)
	ldq $5, 5*8+32*8($27)
	eqv $1,$2,$1
	ldq $6, 5*8($15)
	eqv $3,$4,$3
	ldq $7,12*8+32*8($27)
	ldq $8,12*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16, 6*8+K
	ldq $17, 3*8+K
	ldq $18,11*8+K
	ldq $19,12*8+K
	ldq $20,27*8+K
	ldq $21,40*8+K
	bsr $26,s3

	ldq $1,23*8+32*8($27)
	ldq $2,23*8($15)
	ldq $3,20*8+32*8($27)
	ldq $4,20*8($15)
	ldq $5,30*8+32*8($27)
	eqv $1,$2,$1
	ldq $6,30*8($15)
	eqv $3,$4,$3
	ldq $7,29*8+32*8($27)
	ldq $8,29*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,39*8+K
	ldq $17,33*8+K
	ldq $18,34*8+K
	ldq $19,10*8+K
	ldq $20,18*8+K
	ldq $21,55*8+K
	bsr $26,s4

	ldq $1, 4*8+32*8($27)
	ldq $2, 4*8($15)
	ldq $3,13*8+32*8($27)
	ldq $4,13*8($15)
	ldq $5, 6*8+32*8($27)
	eqv $1,$2,$1
	ldq $6, 6*8($15)
	eqv $3,$4,$3
	ldq $7, 3*8+32*8($27)
	ldq $8, 3*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,16*8+K
	ldq $17, 7*8+K
	ldq $18, 1*8+K
	ldq $19,43*8+K
	ldq $20,31*8+K
	ldq $21,28*8+K
	bsr $26,s5

	ldq $1,11*8+32*8($27)
	ldq $2,11*8($15)
	ldq $3,22*8+32*8($27)
	ldq $4,22*8($15)
	ldq $5, 0*8+32*8($27)
	eqv $1,$2,$1
	ldq $6, 0*8($15)
	eqv $3,$4,$3
	ldq $7,31*8+32*8($27)
	ldq $8,31*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,49*8+K
	ldq $17, 9*8+K
	ldq $18, 0*8+K
	ldq $19,44*8+K
	ldq $20,15*8+K
	ldq $21,38*8+K
	bsr $26,s6

	ldq $1,15*8+32*8($27)
	ldq $2,15*8($15)
	ldq $3, 9*8+32*8($27)
	ldq $4, 9*8($15)
	ldq $5,10*8+32*8($27)
	eqv $1,$2,$1
	ldq $6,10*8($15)
	eqv $3,$4,$3
	ldq $7,16*8+32*8($27)
	ldq $8,16*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,37*8+K
	ldq $17,45*8+K
	ldq $18, 2*8+K
	ldq $19,35*8+K
	ldq $20,22*8+K
	ldq $21,14*8+K
	bsr $26,s7

	ldq $1,28*8+32*8($27)
	ldq $2,28*8($15)
	ldq $3,14*8+32*8($27)
	ldq $4,14*8($15)
	ldq $5,21*8+32*8($27)
	eqv $1,$2,$1
	ldq $6,21*8($15)
	eqv $3,$4,$3
	ldq $7,27*8+32*8($27)
	ldq $8,27*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	ldq $16,51*8+K
	ldq $17,23*8+K
	ldq $18,52*8+K
	ldq $19,36*8+K
	ldq $20,42*8+K
	ldq $21, 8*8+K
	bsr $26,s8

	ldq $1,19*8+32*8($27)
	ldq $2,19*8($15)
	ldq $3,26*8+32*8($27)
	ldq $4,26*8($15)
	ldq $5, 8*8+32*8($27)
	eqv $1,$2,$1
	ldq $6, 8*8($15)
	eqv $3,$4,$3
	ldq $7,17*8+32*8($27)
	ldq $8,17*8($15)
	eqv $5,$6,$5
	and $1,$3,$1
	eqv $7,$8,$7
	and $1,$28,$28
	and $5,$7,$5
	and $5,$28,$28
	beq $28,$return

	br found

$return:
	ret $31,($22),1
	.end checkKey



	.align 9
	.ent s1
s1:
/* 66 + 14 */
	ldq $3, 3*8($13)
	ldq $6,28*8($13)
	ldq $5,11*8($13)
	ldq $4, 7*8($13)
	xor $17,$3,$17
	xor $16,$6,$16
	ldq $7,19*8($13)
	ldq $6,15*8($13)
	xor $19,$5,$19
	xor $18,$4,$18
	xor $21,$7,$21
	xor $20,$6,$20
	bic $18,$19,$11
	bic $18,$20,$5
	bis $11,$20,$0
	bic $19,$20,$1
	and $21,$0,$2
	xor $5,$19,$6
	xor $18,$19,$10
	xor $20,$2,$4
	and $4,$10,$4
	bic $20,$19,$19
	xor $11,$4,$7
	bic $21,$10,$12
	bis $7,$21,$7
	xor $1,$12,$3
	xor $7,$19,$7
	bis $3,$17,$3
	bis $17,$7,$19
	xor $6,$2,$8
	xor $19,$4,$19
	bis $5,$2,$5
	and $19,$16,$19
	ldq $4, 1*8($14)
	xor $19,$3,$19
	xor $5,$10,$5
	eqv $8,$19,$19
	bic $7,$6,$6
	xor $19,$4,$19
	bis $8,$1,$8
	eqv $5,$12,$4
	stq $19, 1*8($15)
	and $4,$7,$4
	ldq $3,24*8($14)
	and $17,$4,$1
	ldq $10, 2*8($14)
	xor $1,$8,$1
	and $18,$8,$8
	xor $1,$6,$1
	bic $17,$6,$6
	and $16,$1,$19
	bis $18,$4,$18
	xor $6,$19,$6
	bic $7,$8,$19
	xor $6,$5,$6
	bis $17,$11,$4
	bic $5,$19,$5
	xor $6,$3,$6
	bis $5,$11,$5
	xor $0,$8,$0
	ornot $17,$5,$8
	eqv $4,$19,$3
	bis $5,$7,$5
	xor $18,$8,$18
	eqv $5,$20,$5
	and $18,$16,$18
	and $5,$17,$5
	bic $0,$2,$2
	xor $18,$3,$18
	xor $5,$2,$5
	xor $18,$10,$18
	bis $5,$16,$5
	bic $1,$12,$12
	ldq $0,25*8($14)
	xor $5,$12,$5
	stq $6,24*8($15)
	xor $5,$3,$5
	stq $18, 2*8($15)
	xor $5,$0,$5
	stq $5,25*8($15) 
	ret $31,($26),1
	.end s1

	.align 6
	.ent s2
s2:
/* 60 + 14  */
	ldq $2,15*8($13)
	ldq $3,19*8($13)
	ldq $6,23*8($13)
	ldq $5,27*8($13)
	xor $17,$3,$17
	xor $16,$2,$16
	ldq $3,31*8($13)
	ldq $2, 2*8($13)
	xor $18,$6,$18
	xor $19,$5,$19
	xor $20,$3,$20
	xor $21,$2,$21
	and $21,$20,$0
	xor $16,$21,$8
	bic $16,$0,$1
	xor $8,$20,$10
	bic $17,$1,$2
	bic $20,$1,$7
	xor $10,$2,$5
	bis $7,$17,$11
	bis $0,$2,$2
	and $19,$11,$3
	bic $2,$8,$8
	xor $5,$7,$7
	bis $18,$8,$4
	ldq $2,18*8($14)
	xor $4,$5,$4
	and $7,$17,$7
	eqv $4,$3,$3
	bis $5,$16,$5
	xor $3,$2,$2
	bic $21,$1,$21
	xor $20,$17,$20
	xor $5,$17,$5
	stq $2,18*8($15)
	bic $20,$8,$8
	bic $18,$5,$6
	xor $21,$7,$2
	xor $1,$3,$1
	xor $6,$8,$6
	bic $1,$17,$1
	bis $5,$7,$5
	and $2,$18,$2
	bis $19,$6,$3
	xor $2,$1,$2
	and $20,$5,$20
	ldq $1, 7*8($14)
	xor $2,$3,$2
	bic $11,$20,$20
	xor $2,$10,$2
	bis $21,$8,$21
	bis $20,$19,$20
	and $18,$5,$5
	xor $2,$1,$2
	eqv $21,$4,$21
	xor $20,$5,$20
	xor $17,$10,$17
	bis $0,$7,$7
	stq $2, 7*8($15)
	xor $20,$21,$20
	ldq $2, 5*8($14)
	and $17,$21,$21
	and $18,$0,$0
	eqv $6,$21,$6
	eqv $0,$7,$0
	xor $20,$2,$20
	and $0,$19,$0
	and $6,$18,$6
	ldq $2,12*8($14)
	xor $0,$6,$0
	eqv $17,$0,$17
	stq $20, 5*8($15)
	xor $17,$2,$17
	stq $17,12*8($15)
	ret $31,($26),1
	.end s2

	.align 5
	.ent s3
s3:
/* 63 + 14 */
	ldq $3, 2*8($13)
	ldq $0,31*8($13)
	ldq $4,10*8($13)
	ldq $5, 6*8($13)
	xor $17,$3,$17
	xor $16,$0,$16
	ldq $0,18*8($13)
	ldq $3,14*8($13)
	xor $19,$4,$19
	xor $18,$5,$18
	xor $21,$0,$21
	xor $17,$18,$10
	xor $20,$3,$20
	xor $10,$21,$10
	and $17,$10,$3
	and $18,$21,$6
	bic $21,$3,$7
	xor $18,$3,$11
	xor $7,$20,$1
	bic $11,$20,$2
	and $16,$1,$8
	bic $10,$2,$12
	xor $8,$20,$8
	bis $20,$3,$5
	bis $16,$2,$0
	bis $8,$19,$8
	xor $10,$5,$4
	bis $6,$3,$3
	xor $8,$0,$8
	ldq $0,23*8($14)
	bis $1,$12,$1
	xor $8,$4,$8
	xor $8,$0,$0
	stq $0,23*8($15)
	xor $3,$20,$0
	xor $1,$3,$1
	xor $21,$3,$3
	bic $3,$4,$4
	bis $17,$2,$3
	and $4,$16,$4
	bis $7,$1,$7
	bis $16,$1,$1
	xor $1,$3,$1
	xor $4,$7,$4
	bic $2,$18,$2
	eqv $5,$1,$5
	xor $12,$6,$6
	and $4,$19,$4
	bis $2,$16,$2
	and $5,$19,$5
	bis $6,$16,$6
	xor $4,$2,$4
	and $18,$20,$18
	ldq $1,20*8($14)
	xor $5,$6,$5
	xor $4,$18,$4
	xor $4,$10,$4
	eqv $0,$5,$5
	and $21,$8,$21
	xor $5,$1,$5
	xor $21,$11,$21
	bic $4,$3,$3
	ldq $1,30*8($14)
	bic $21,$16,$21
	bis $3,$16,$3
	bis $17,$12,$17
	xor $3,$21,$3
	ldq $6,29*8($14)
	xor $4,$1,$1
	and $3,$19,$3
	stq $1,30*8($15)
	xor $3,$17,$3
	xor $3,$0,$3
	xor $21,$6,$21
	stq $5,20*8($15)
	eqv $21,$3,$21
	stq $21,29*8($15)
	ret $31,($26),1
	.end s3

	.align 5
	.ent s4
s4:
/* 49 + 14 */
	ldq $2,14*8($13)
	ldq $3,18*8($13)
	ldq $4,22*8($13)
	ldq $5,26*8($13)
	xor $16,$2,$16
	xor $17,$3,$17
	ldq $2,30*8($13)
	ldq $3, 1*8($13)
	xor $18,$4,$18
	xor $19,$5,$19
	bis $16,$18,$5
	xor $20,$2,$20
	bis $17,$18,$6
	and $5,$20,$5
	xor $21,$3,$21
	xor $16,$5,$8
	bic $18,$16,$16
	xor $18,$5,$5
	bis $16,$8,$16
	ldq $0, 4*8($14)
	and $17,$16,$4
	ldq $1,13*8($14)
	xor $20,$4,$3
	ldq $10, 6*8($14)
	and $19,$3,$2
	xor $18,$20,$20
	xor $6,$2,$6
	bic $20,$17,$20
	xor $6,$8,$6
	xor $18,$4,$18
	bis $5,$8,$8
	ldq $2, 3*8($14)
	xor $8,$20,$4
	bic $17,$5,$5
	bis $4,$19,$4
	eqv $20,$18,$20
	xor $4,$5,$4
	and $17,$3,$3
	xor $4,$16,$4
	and $20,$19,$20
	bis $21,$4,$5
	xor $20,$3,$20
	xor $5,$6,$5
	and $21,$4,$4
	xor $5,$0,$0
	eqv $6,$4,$3
	xor $20,$8,$20
	stq $0, 4*8($15)
	xor $6,$20,$6
	xor $3,$1,$3
	bic $17,$6,$17
	bic $6,$19,$19
	xor $17,$19,$17
	stq $3,13*8($15)
	xor $17,$5,$17
	bis $21,$17,$21
	xor $4,$17,$4
	eqv $20,$21,$20
	xor $20,$10,$10
	xor $4,$20,$4
	stq $10, 6*8($15)
	xor $4,$2,$4
	stq $4, 3*8($15)
	ret $31,($26),1
	.end s4

	.align 5
	.ent s5
s5:
/* 66 + 14 */
	ldq $6,30*8($13)
	ldq $3, 1*8($13)
	ldq $4, 5*8($13)
	ldq $5, 9*8($13)
	xor $16,$6,$16
	xor $17,$3,$17
	xor $18,$4,$18
	xor $19,$5,$19
	xor $19,$16,$12
	bic $18,$19,$1
	ldq $3,17*8($13)
	xor $1,$16,$8
	bic $16,$18,$7
	bis $12,$1,$1
	xor $21,$3,$21
	and $18,$1,$5
	bis $21,$7,$4
	xor $5,$19,$5
	ldq $3,13*8($13)
	xor $19,$7,$11
	bic $1,$21,$10
	bis $11,$21,$11
	bic $5,$7,$7
	xor $20,$3,$20
	xor $7,$11,$7
	xor $18,$10,$0
	bis $20,$7,$2
	xor $8,$4,$6
	xor $2,$5,$2
	xor $16,$0,$16
	bic $2,$17,$21
	bis $20,$0,$3
	xor $3,$21,$3
	ldq $21,11*8($14)
	and $8,$16,$8
	xor $3,$6,$3
	and $19,$4,$4
	bic $20,$8,$8
	xor $3,$21,$21
	xor $8,$4,$8
	xor $8,$7,$8
	bis $19,$16,$4
	stq $21,11*8($15)
	and $7,$6,$7
	ldq $21,22*8($14)
	bic $4,$17,$6
	xor $0,$7,$0
	xor $6,$8,$6
	xor $6,$21,$6
	eqv $0,$3,$3
	bis $20,$0,$21
	ornot $7,$1,$1
	bic $10,$19,$10
	bis $5,$11,$5
	and $3,$20,$3
	xor $10,$18,$10
	and $2,$1,$2
	and $12,$0,$0
	xor $5,$21,$5
	and $3,$4,$3
	bis $5,$17,$5
	and $20,$10,$4
	xor $3,$0,$3
	bis $8,$2,$8
	ldq $21, 0*8($14)
	xor $5,$4,$5
	xor $3,$10,$3
	eqv $12,$8,$12
	xor $5,$1,$5
	bis $3,$17,$3
	and $12,$20,$12
	xor $5,$21,$5
	xor $3,$12,$3
	ldq $21,31*8($14)
	xor $3,$2,$3
	xor $3,$16,$3
	xor $3,$21,$3
	stq $6,22*8($15)
	stq $5, 0*8($15)
	stq $3,31*8($15)
	ret $31,($26),1
	.end s5

	.align 5
	.ent s6
s6:
/* 63 + 14 */
	ldq $2,13*8($13)
	ldq $3,17*8($13)
	ldq $4,21*8($13)
	ldq $5,25*8($13)
	xor $16,$2,$16
	xor $17,$3,$17
	ldq $2,29*8($13)
	ldq $3, 0*8($13)
	xor $18,$4,$18
	xor $19,$5,$19
	xor $20,$2,$20
	xor $21,$3,$21
	xor $20,$16,$5
	and $16,$21,$16
	xor $5,$21,$11
	bic $16,$20,$6
	xor $21,$16,$10
	bic $19,$6,$8
	bis $6,$10,$7
	xor $11,$8,$1
	bic $7,$19,$7
	bis $21,$1,$3
	xor $7,$10,$7
	bic $3,$20,$3
	bis $6,$7,$6
	and $17,$7,$4
	bic $17,$6,$2
	xor $16,$1,$16
	xor $3,$2,$3
	xor $1,$4,$4
	ldq $2,15*8($14)
	bic $3,$18,$3
	bic $16,$7,$7
	eqv $4,$3,$3
	xor $3,$2,$2
	bic $3,$5,$3
	stq $2,15*8($15)
	xor $3,$6,$3
	bic $21,$3,$0
	bis $20,$21,$6
	xor $0,$1,$0
	bic $6,$5,$5
	bic $17,$0,$4
	and $6,$7,$6
	ornot $4,$17,$2
	xor $6,$11,$6
	eqv $3,$4,$4
	xor $5,$2,$5
	xor $21,$4,$21
	ldq $2, 9*8($14)
	and $20,$21,$21
	and $5,$18,$5
	bis $0,$21,$0
	xor $5,$4,$5
	and $6,$17,$6
	xor $5,$2,$5
	bic $3,$20,$3
	xor $6,$0,$6
	and $20,$10,$20
	bic $17,$21,$2
	bis $3,$18,$3
	eqv $8,$6,$8
	ornot $20,$19,$20
	xor $3,$2,$3
	and $8,$18,$8
	bic $20,$17,$20
	ldq $4,10*8($14)
	ldq $2,16*8($14)
	xor $8,$20,$8
	bis $21,$11,$21
	eqv $7,$3,$3
	xor $8,$21,$8
	xor $3,$4,$3
	xor $8,$2,$8
	stq $5, 9*8($15)
	stq $3,10*8($15)
	stq $8,16*8($15)
	ret $31,($26),1
	.end s6

	.align 5
	.ent s7
s7:
/* 61 + 14 */
	ldq $2,29*8($13)
	ldq $3, 0*8($13)
	ldq $4, 4*8($13)
	ldq $5, 8*8($13)
	xor $16,$2,$16
	xor $17,$3,$17
	ldq $0,12*8($13)
	ldq $2,16*8($13)
	xor $18,$4,$18
	xor $19,$5,$19
	xor $20,$0,$20
	and $17,$19,$6
	xor $21,$2,$21
	xor $6,$20,$6
	bis $17,$19,$5
	and $19,$6,$7
	bic $20,$17,$2
	xor $7,$17,$11
	bis $2,$18,$2
	bic $18,$11,$8
	bis $5,$20,$5
	xor $6,$8,$6
	xor $5,$2,$5
	xor $18,$8,$8
	bis $18,$7,$4
	bic $21,$8,$1
	bic $19,$18,$12
	xor $1,$6,$1
	ldq $2,28*8($14)
	xor $7,$6,$6
	xor $19,$11,$19
	bis $21,$6,$3
	xor $18,$7,$7
	xor $5,$3,$5
	and $7,$17,$7
	and $5,$16,$5
	xor $4,$19,$4
	xor $5,$1,$5
	bic $21,$7,$7
	xor $5,$2,$2
	xor $7,$4,$7
	and $17,$21,$10
	bic $10,$12,$10
	stq $2,28*8($15)
	bis $16,$7,$2
	xor $10,$1,$0
	ldq $3,14*8($14)
	eqv $0,$2,$2
	bic $4,$18,$4
	eqv $6,$17,$6
	xor $2,$3,$2
	bis $17,$19,$3
	bis $4,$7,$4
	and $6,$19,$6
	xor $3,$5,$3
	and $6,$21,$6
	stq $2,14*8($15)
	bic $3,$21,$3
	bic $4,$16,$4
	and $18,$10,$18
	ldq $2,21*8($14)
	xor $8,$7,$8
	xor $3,$4,$3
	xor $18,$6,$18
	xor $3,$8,$3
	bis $18,$16,$18
	bis $20,$12,$20
	xor $18,$6,$18
	ldq $0,27*8($14)
	xor $18,$8,$18
	xor $3,$2,$3
	xor $18,$20,$18
	stq $3,21*8($15)
	xor $18,$0,$18
	stq $18,27*8($15)
	ret $31,($26),1
	.end s7

	.align 5
	.ent s8
s8:
/* 60 + 14 */
	ldq $2,12*8($13)
	ldq $3,16*8($13)
	ldq $4,20*8($13)
	ldq $5,24*8($13)
	xor $16,$2,$16
	xor $17,$3,$17
	ldq $2,28*8($13)
	ldq $3, 3*8($13)
	xor $18,$4,$18
	xor $19,$5,$19
	xor $20,$2,$20
	bic $16,$18,$11
	xor $21,$3,$21
	xor $11,$19,$4
	xor $18,$16,$8
	bis $20,$4,$7
	xor $7,$8,$7
	bic $7,$16,$0
	bis $0,$19,$3
	xor $0,$18,$10
	xor $3,$8,$3
	bic $10,$20,$1
	xor $3,$20,$3
	xor $1,$19,$1
	bic $17,$1,$5
	bic $4,$3,$6
	xor $5,$7,$5
	eqv $6,$10,$6
	bis $7,$20,$7
	and $17,$6,$2
	xor $2,$3,$2
	bic $5,$19,$10
	ldq $12,19*8($14)
	bis $2,$21,$2
	eqv $2,$5,$2
	and $16,$7,$16
	xor $2,$12,$2
	xor $7,$4,$4
	and $20,$11,$12
	stq $2,19*8($15)
	xor $12,$10,$12
	ldq $2,26*8($14)
	and $12,$17,$12
	bic $17,$10,$10
	xor $12,$16,$12
	bic $11,$3,$11
	bic $12,$21,$12
	and $18,$6,$6
	xor $12,$10,$12
	bis $17,$0,$0
	xor $12,$4,$12
	bis $6,$1,$6
	bis $4,$12,$4
	xor $12,$2,$2
	bic $17,$4,$4
	stq $2,26*8($15)
	xor $4,$11,$4
	xor $18,$20,$18
	ldq $2, 8*8($14)
	xor $6,$0,$6
	bis $4,$21,$4
	xor $18,$11,$18
	bic $8,$20,$8
	eqv $4,$6,$4
	bic $18,$17,$17
	bis $8,$19,$8
	xor $4,$2,$4
	xor $17,$8,$17
	ldq $2,17*8($14)
	and $17,$21,$17
	eqv $17,$5,$17
	stq $4, 8*8($15)
	xor $17,$2,$17
	stq $17,17*8($15)
	ret $31,($26),1
	.end s8
