
	OPT	O+,W-

	XDEF	_rc5_unit_func_000_030_asm

;-------------------- 
;@(#)$Id: rc5-000_030-jg.s,v 1.1.2.1 1999/12/08 01:27:56 cyp Exp $

	; Define P, Q

P:	equ	$b7e15163
Q:	equ	$9e3779b9
P0QR3:	equ	$bf0a8b1d	;(0*Q + P) <<< 3

;--------------------

RUF_CODEALIGN:	MACRO
	;Ensure next instruction is aligned on quadword boundary
.align\@:	equ	*-_rc5_unit_func_000_030_asm
	IFNE	.align\@&7
	nop
	RUF_CODEALIGN
	ENDC
	ENDM

;--------------------

	;680x0 RC5 core function
	;Dual key, rolled loops, 020/030 optimised
	;
	;Entry: a0=rc5unitwork structure:
	;	 0(a0) = plain.hi	- plaintext
	;	 4(a0) = plain.lo
	;	 8(a0) = cypher.hi	- cyphertext
	;	12(a0) = cypher.lo
	;	16(a0) = L0.hi	- key
	;	20(a0) = L0.lo
	;       d0=number of iterations to run for
	;
	;Exit:  d0=return code

_rc5_unit_func_000_030_asm:	movem.l	d2-7/a2-6,-(a7)

	move.l	d0,-(a7)	;Save initial key iterations counter
	move.l	#Q,a4	;a4=constant (P+nQ increment)
	move.l	16(a0),d2	;d2=L0.hi=L1a
	lea	-212(a7),a1	;a1=Sx[] storage
	move.l	20(a0),d1	;d1=L0.lo=L0a and L0b for a while
	move.l	d0,a6	;Keep loop counter in a6

	move.l	#P0QR3,d0	;Calculate initial L0x and Ax
	add.l	d0,d1
	ror.l	#3,d1
	move.l	d1,-216(a7)

	add.l	#P+Q,d0
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,-220(a7)

	RUF_CODEALIGN
.ruf_mainloop:
 	;---- Round 1 of key expansion ----
	;A : d0 d4
	;L0: d1 d5
	;L1: d2 d6

	move.l	a1,a2	;a2=Sa[] array
	move.l	-216(a7),d1	;d1=L0x
	move.l	#P0QR3,d7	;d7=Sx[00]
	move.l	d2,d6	;d6=L1b=L1a
	move.l	d7,(a2)+	;Sa[00]=Aa (first round special case)
	lea	104(a1),a3	;a3=Sb[] array
	add.l	#1<<24,d6	;d6=L1b=L1a+1<<24
	move.l	d7,(a3)+	;Sb[00]=Ab (first round special case)
	move.l	#P+Q+Q,a5	;a5=Initial P+nQ
	move.l	-220(a7),d0	;d0=Ax
	moveq	#12-1,d7	;d7=loop counter
	move.l	d0,d4	;Ab=Aa
	move.l	d0,(a2)+	;Sa[01]=Aa
	move.l	d1,d3	;d3=L0x
	move.l	d1,d5	;L0b=L0a
	add.l	d0,d3	;d3=Ax+L0x
	move.l	d0,(a3)+	;Sb[01]=Ab
	add.l	d3,d2	;L1a=L1a+Aa+L0a
	add.l	d3,d6	;L1b=L1b+Ab+L0b
	rol.l	d3,d2	;L1a=L1a<<<(Aa+L0a)
	rol.l	d3,d6	;L1b=L1b<<<(Ab+L0b)

	RUF_CODEALIGN
.ruf_r1loop:	add.l	a5,d0	;d0=Aa=Aa+P+nQ
	add.l	a5,d4	;d4=Ab=Ab+P+nQ
	add.l	d2,d0	;d0=Aa=Aa+P+nQ+L1a
	add.l	d6,d4	;d4=Ab=Ab+P+nQ+L1b
	rol.l	#3,d0	;d0=Aa=(Aa+P+nQ+L1a)<<<3
	move.l	d2,d3	;d3=L1a
	move.l	d0,(a2)+	;Sa[n]=Aa
	rol.l	#3,d4	;d4=Ab=(Ab+P+nQ+L1b)<<<3
	add.l	d0,d3	;d3=Aa+L1a
	add.l	a4,a5	;Update P+nQ
	add.l	d3,d1	;d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;Sb[n]=Ab
	rol.l	d3,d1	;d1=L0a=L0a<<<(Aa+L1a)
	add.l	a5,d0	;d0=Aa=Aa+P+nQ
	move.l	d6,d3	;d3=L1b
	add.l	d1,d0	;d0=Aa=Aa+P+nQ+L0a
	add.l	d4,d3	;d3=Ab+L1b
	rol.l	#3,d0	;d0=Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d3,d5	;d5=L0b=L0b+Ab+L1b
	move.l	d0,(a2)+	;Sa[n]=Aa
	rol.l	d3,d5	;d5=L0b=L0b<<<(Ab+L1b)
	add.l	a5,d4	;d4=Ab=Ab+P+nQ
	move.l	d1,d3	;d3=L0a
	add.l	d5,d4	;d4=Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;d3=Aa+L0a
	rol.l	#3,d4	;d4=Ab=(Ab+P+nQ+L0b)<<<3
	add.l	d3,d2	;d2=L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;Sb[n]=Ab
	rol.l	d3,d2	;d2=L1a=L1a<<<(Aa+L0a)
	add.l	a4,a5	;Update P+nQ
	move.l	d5,d3	;d3=L0b

	add.l	d4,d3	;d3=Ab+L0b
	add.l	d3,d6	;d6=L1b=L1b+Ab+L0b
	rol.l	d3,d6	;d6=L1b=L1b<<<(Ab+L0b)

	dbf	d7,.ruf_r1loop

	;---- Round 2 of key expansion ----

	move.l	a1,a2	;a2=SArray
	moveq	#13-1,d7	;d7=loop counter
	lea	104(a1),a3	;a3=Sb[] array

	RUF_CODEALIGN
.ruf_r2loop:	add.l	(a2),d0	;Aa=Aa+Sa[n]
	add.l	d6,d4	;Ab=Ab+L1b
	add.l	d2,d0	;Aa=Aa+Sa[n]+L1a
	add.l	(a3),d4	;Ab=Ab+Sb[n]+L1b
	rol.l	#3,d0	;Aa=(Aa+Sa[n]+L1a)<<<3
	move.l	d2,d3	;d3=L1a
	move.l	d0,(a2)+	;Sa[n]=Aa
	rol.l	#3,d4	;Ab=(Ab+Sb[n]+L1b)<<<3
	add.l	d0,d3	;d3=Aa+L1a
	move.l	d4,(a3)+	;Sb[n]=Ab
	add.l	d3,d1	;L0a=L0a+Aa+L1a

	rol.l	d3,d1	;L0a=L0a<<<(Aa+L1a)
	add.l	(a2),d0	;Aa=Aa+Sa[n]
	move.l	d6,d3	;d3=L1b
	add.l	d1,d0	;Aa=Aa+Sa[n]+L0a
	add.l	d4,d3	;d3=Ab+L1b
	rol.l	#3,d0	;Aa=(Aa+Sa[n]+L0a)<<<3
	add.l	d3,d5	;L0b=L0b+Ab+L1b
	add.l	(a3),d4	;Ab=Ab+Sb[n]
	rol.l	d3,d5	;L0b=L0b<<<(Ab+L1b)
	move.l	d0,(a2)+	;Sa[n]=Aa
	add.l	d5,d4	;Ab=Ab+Sb[n]+L0b

	rol.l	#3,d4	;Ab=(Ab+Sb[n]+L0b)<<<3
	move.l	d4,(a3)+	;Sb[n]=Ab
	move.l	d1,d3	;d3=L0a
	add.l	d0,d3	;d3=Aa+L0a
	add.l	d3,d2	;L1a=L1a+Aa+L0a
	rol.l	d3,d2	;L1a=L1a<<<(Aa+L0a)
	move.l	d5,d3	;d3=L0b
	add.l	d4,d3	;d3=Ab+L0b
	add.l	d3,d6	;L1b=L1b+Ab+L0b
	rol.l	d3,d6	;L1b=L1b<<<(Ab+L0b)

	dbf	d7,.ruf_r2loop

	;---- Combined round 3 of key expansion and encryption round ----

	move.l	d4,a2	;a2=Ab
	move.l	d5,a3	;a3=L0b

	bsr.s	.ruf_round3	;Do round 3 for 'a' key
	beq	.ruf_gotit1	;Skip if we found a key!

	move.l	a2,d0	;d0=Ab
	move.l	a3,d1	;d1=L0b
	move.l	d6,d2	;d2=L1b
	addq.l	#8,a1	;a1=Sb[]
	
	bsr.s	.ruf_round3	;Do round 3 for 'b' key
	beq	.ruf_gotit2

;----------	;Mangle-increment current key by PIPELINE_COUNT

	addq.b	#2,16(a0)	;L1=L1+PIPELINE_COUNT
	bcs.s	.ruf_l1carry	;Skip if carry

.ruf_midone: move.l	16(a0),d2	;d2=updated L1
	subq.l	#1,a6	;Update loop counter
	lea	-212(a7),a1	;a1=Sx[] storage for next iteration

	move.l	a6,d0	;Loop back for next key
	bpl	.ruf_mainloop

;---------- 	;Didnt find key :(

	move.l	(a7)+,d0	;Return iterations*pipeline_count
	add.l	d0,d0
	movem.l	(a7)+,d2-7/a2-6
	rts

;--------------------

	RUF_CODEALIGN
.ruf_l1carry:
	addq.b	#1,17(a0)	;Every 256^1
	bcc.s	.ruf_midone
	addq.b	#1,18(a0)	;Every 256^2
	bcc.s	.ruf_midone
	addq.b	#1,19(a0)	;Every 256^3
	bcc.s	.ruf_midone

	move.l	20(a0),d1	;d1=L0
	rol.w	#8,d1	;L0:abdc
	swap	d1	;L0:dcab
	rol.w	#8,d1	;L0:dcba
	addq.l	#1,d1	;L0=L0+1
	rol.w	#8,d1	;L0:dcab
	swap	d1	;L0:abdc
	rol.w	#8,d1	;L0:abcd
	move.l	d1,20(a0)	;Store updated L0

	move.l	#P0QR3,d0	;Recalculate initial L0x and Ax
	add.l	d0,d1
	ror.l	#3,d1
	move.l	d1,-216(a7)

	add.l	#P+Q,d0
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,-220(a7)

	bra.s	.ruf_midone

;--------------------

	RUF_CODEALIGN
.ruf_round3:	;RC5 round 3
	;Entry:	a0=RC5UnitWork structure
	;	a1=S array to use
	;	d0=A
	;	d1=L0
	;	d2=L1
	;Exit:	EQ if key found, NE otherwise
	;	d1-5/d7 corrupt, a1=96(a1)

	add.l	(a1)+,d0	;d0=A=Sa[00]
	move.l	d2,d3	;d3=L1
	add.l	d2,d0	;d0=A=Sa[00+L1]
	move.l	4(a0),d4	;d4=eA=plain.lo
	rol.l	#3,d0	;d0=A=(Sa[00]+L1)<<<3
	add.l	d0,d4	;d4=eA=plain.lo+A
	add.l	d0,d3	;d3=A+L1

	add.l	d3,d1	;d1=L0=L0+A+L1
	add.l	(a1)+,d0	;d0=A=A+Sa[01]
	rol.l	d3,d1	;d1=L0=(L0+A+L1)<<<(A+L1)
	moveq	#11-1,d7	;d7=loop counter
	add.l	d1,d0	;d0=A=A+Sa[01]+L0
	move.l	d1,d3	;d3=L0
	rol.l	#3,d0	;d0=A=(A+Sa[01]+L0)<<<3
	move.l	(a0),d5	;d5=eB=plain.hi
	add.l	d0,d3	;d3=A+L0
	add.l	d0,d5	;d5=eB=plain.hi+A
	add.l	d3,d2	;d2=L1=L1+A+L0

	rol.l	d3,d2	;d2=L1=(L1+A+L0)<<<(A+L0)

	RUF_CODEALIGN
.ruf_r3_r3loop: 
	add.l	(a1)+,d0	;d0=A=A+Sn
	eor.l	d5,d4	;d4=eA=eA^eB
	add.l	d2,d0	;d0=A=A+Sn+L1
	rol.l	d5,d4	;d4=eA=((eA^eB)<<<eB)
	rol.l	#3,d0	;d0=A=(A+Sn+L1)<<<3
	move.l	d2,d3	;d3=L1
	add.l	d0,d4	;d4=eA=((eA^eB)<<<eB)+A

	add.l	d0,d3	;d3=A+L1
	add.l	d3,d1	;d1=L0=L0+A+L1
	add.l	(a1)+,d0	;d0=A=A+Sn
	rol.l	d3,d1	;d1=L0=(L0+A+L1)<<<(A+L1)

	eor.l	d4,d5	;d5=eB=eB^eA
	add.l	d1,d0	;d0=A=A+Sn+L0
	rol.l	d4,d5	;d5=eB=((eB^eA)<<<eA)
	rol.l	#3,d0	;d0=A=(A+Sn+L0)<<<3
	move.l	d1,d3	;d3=L0
	add.l	d0,d5	;d5=eB=((eB^eA)<<<eA)+A

	add.l	d0,d3	;d3=A+L0
	add.l	d3,d2	;d2=L1=L1+A+L0
	rol.l	d3,d2	;d2=L1=L1<<<(A+L0)

	dbf	d7,.ruf_r3_r3loop

	add.l	(a1),d0	;d0=A=A+S24
	eor.l	d5,d4	;d4=eA=eA^eB
	add.l	d2,d0	;d0=A=A+S24+L1
	rol.l	d5,d4	;d4=eA=((eA^eB)<<<eB)
	rol.l	#3,d0	;d0=A=(A+S24+L1)<<<3
	move.l	12(a0),d3	;d3=cypher.lo
	add.l	d0,d4	;d4=eA=((eA^eB)<<<eB)+A
	cmp.l	d3,d4	;d4=eA=cypher.lo?
	beq.s	.ruf_r3_tryeb

	rts		 ;Not the right key

	RUF_CODEALIGN
.ruf_r3_tryeb:
	move.l	d0,d3	;d3=A
	add.l	d2,d3	;d3=A+L1
	add.l	d3,d1	;d1=L0=L0+A+L1
	add.l	4(a1),d0	;d0=A=A+S25
	rol.l	d3,d1	;d1=L0=(L0+A+L1)<<<(A+L1)
	eor.l	d4,d5	;d5=eB=eB^eA
	add.l	d1,d0	;d0=A=A+S25+L0
	rol.l	d4,d5	;d5=eB=((eB^eA)<<<eA)
	rol.l	#3,d0	;d0=A=(A+S25+L0)<<<3
	move.l	8(a0),d3	;d3=cypher.hi
	add.l	d0,d5	;d5=eB=((eB^eA)<<<eA)+A
	cmp.l	d3,d5	;eB=cypher.hi?

	rts		;EQ if right key, NE otherwise

;--------------------

.ruf_gotit1:	;Key found on pipeline 1!
	move.l	(a7)+,d0	;d0=initial iteration count
	sub.l	a6,d0	;Return number of keys checked
	add.l	d0,d0	; = loops*pipeline_count

	movem.l	(a7)+,d2-7/a2-6
	rts

.ruf_gotit2:	;Key found on pipeline 2!
	move.l	(a7)+,d0	;d0=initial iteration count
	sub.l	a6,d0	;Return number of keys checked
	add.l	d0,d0	; = loops*pipeline_count+1
	addq.b	#1,16(a0)	;Adjust L0.hi for pipeline 2
	addq.l	#1,d0

	movem.l	(a7)+,d2-7/a2-6
	rts

;--------------------
