
	OPT	O+,W-

	XDEF	_rc5_unit_func_040_060_asm

;--------------------
;@(#)$Id: rc5-040_060-jg.s,v 1.2 1999/12/08 01:28:52 cyp Exp $

	RSRESET
	;Offsets into .ruf_vars

RUFV_LOOP:	rs.l	1	;Loop counter
RUFV_KITER:	rs.l	1	;Total iterations
RUFV_L0X:	rs.l	1	;L0x constant
RUFV_AX:	rs.l	1	;Ax constant
RUFV_L0XAX:	rs.l	1	;L0x+Ax
RUFV_SIZE:	equ	__RS

;--------------------

	; Define P, Q

P:	equ	$b7e15163
Q:	equ	$9e3779b9
P0QR3:	equ	$bf0a8b1d	;(0*Q + P) <<< 3
PR3Q:	equ	$15235639	;P0QR3+P+Q

;--------------------

RUF_CODEALIGN:	MACRO
	;Ensure next instruction is aligned on quadword boundary
.align\@:	equ	*-_rc5_unit_func_040_060_asm
	IFNE	.align\@&7
	nop
	RUF_CODEALIGN
	ENDC
	ENDM

;--------------------

	CNOP	0,8

	;680x0 RC5 core function
	;Dual key, unrolled loops, 040/060 optimised
	;
	;Entry: a0=rc5unitwork structure:
	;	 0(a0) = plain.hi  - plaintext
	;	 4(a0) = plain.lo
	;	 8(a0) = cypher.hi - cyphertext
	;	12(a0) = cypher.lo
	;	16(a0) = L0.hi     - key
	;	20(a0) = L0.lo
	;       d0=number of iterations to run for

_rc5_unit_func_040_060_asm:	movem.l	d2-7/a2-6,-(a7)

	move.l	#P0QR3,a5	;a5=handy constant
	move.l	20(a0),d1	;d1=L0a
	lea	.ruf_pnq(pc),a4	;a4=P+nQ lookup
	add.l	a5,d1	;L0x=L0x+Sx[00]
	lea	.ruf_vars(pc),a6	;a6=core variables
	ror.l	#3,d1	;L0x=(L0x+Ax)>>>3 = (L0x+Ax)<<<P0QR3
	move.l	a6,a1	;a1=scratch core vars
	move.l	#PR3Q,d2	;d2=Ax=Ax+P+nQ
	move.l	d0,(a1)+	;Initialise loop counter
	move.l	d0,(a1)+	;Save initial kiter
	move.l	d1,(a1)+	;Save (L0x+Ax)<<<P0QR3
	add.l	d1,d2	;d2=Ax=Ax+P+nQ+L0x
	lea	.ruf_sarray+8(pc),a2	;a2=&Sa[02]
	rol.l	#3,d2	;Ax=(Ax+P+nQ+L0x)<<<3
	lea	104(a2),a3	;a3=&Sb[02]
	move.l	d2,(a1)+	;Store constant Ax
	add.l	d1,d2

	move.l	d2,(a1)
	lea	RUFV_L0X(a6),a1

	RUF_CODEALIGN

.ruf_mainloop:
	;A : d0 d4
	;L0: d1 d5
	;L1: d2 d6

	move.l	(a1)+,d1	;d1=(L0x+Ax)<<<P0QR3
	move.l	16(a0),d2	;d2=Updated L1 for next
	move.l	(a1)+,d0	;d0=Ax=(Ax+P+nQ+L0x)<<<3
	move.l	d2,d6	;d6=L1b=L1a
	move.l	(a1)+,d3	;d3=Ax+L0x
	add.l	(a1),d6	;d6=L1b=L1a+1<<24

	; Remainder of first odd round

	add.l	d3,d2	;L1a=L1a+Aa+L0a
	add.l	d3,d6	;L1b=L1b+Ab+L0b
	add.l	(a4)+,d0	;d0=Ax=Ax+P+nQ, update P+nQ
	rol.l	d3,d2	;L1a=L1a<<<(Aa+L0a)
	rol.l	d3,d6	;L1b=L1b<<<(Ab+L0b)

	; Repeated round 1 even-odd-even-odd rounds

	move.l	d0,d4	;Ab=Aa
	move.l	d1,d5	;L0b=L0a
	add.l	d2,d0	;d0=Aa=Aa+P+nQ+L1a
	add.l	d6,d4	;d4=Ab=Ab+P+nQ+L1b
	rol.l	#3,d0	;d0=Aa=(Aa+P+nQ+L1a)<<<3
	move.l	d2,d3	;d3=L1a
	move.l	d0,(a2)+	;Sa[n]=Aa
	rol.l	#3,d4	;d4=Ab=(Ab+P+nQ+L1b)<<<3
	add.l	d0,d3	;d3=Aa+L1a
	move.l	d6,d7	;d7=L1b
	add.l	d3,d1	;d1=L0a=L0a+Aa+L1a
	add.l	d4,d7	;d7=Ab+L1b
	move.l	d4,(a3)+	;Sb[n]=Ab
	rol.l	d3,d1	;d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;d5=L0b=L0b+Ab+L1b
	add.l	(a4),d0	;Aa=Aa+P+nQ
	rol.l	d7,d5	;d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;Aa=Aa+P+nQ+L0a
	move.l	d1,d3	;d3=L0a
	add.l	(a4)+,d4	;Ab=Ab+P+nQ, update P+nQ
	rol.l	#3,d0	;Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;Ab=Ab+P+nQ+L0b
	move.l	d0,(a2)+	;Sa[n]=Aa
	rol.l	#3,d4	;Ab=(Ab+P+nQ+L0b)<<<3
	add.l	d0,d3	;d3=Aa+L0a
	move.l	d5,d7	;d7=L0b
	add.l	d3,d2	;L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;Sb[n]=Ab
	add.l	d4,d7	;d7=Ab+L0b
	rol.l	d3,d2	;L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;L1b=L1b+Ab+L0b
	add.l	(a4),d0	;d0=Aa=Aa+P+nQ
	rol.l	d7,d6	;L1b=L1b<<<(Ab+L0b)	(sOEP)

	REPT 10

	add.l	(a4)+,d4	;d4=Ab=Ab+P+nQ, update P+nQ	(pOEP)
	add.l	d2,d0	;d0=Aa=Aa+P+nQ+L1a
	add.l	d6,d4	;d4=Ab=Ab+P+nQ+L1b
	rol.l	#3,d0	;d0=Aa=(Aa+P+nQ+L1a)<<<3
	move.l	d2,d3	;d3=L1a
	move.l	d0,(a2)+	;Sa[n]=Aa
	rol.l	#3,d4	;d4=Ab=(Ab+P+nQ+L1b)<<<3
	add.l	d0,d3	;d3=Aa+L1a
	move.l	d6,d7	;d7=L1b
	add.l	d3,d1	;d1=L0a=L0a+Aa+L1a
	add.l	d4,d7	;d7=Ab+L1b
	move.l	d4,(a3)+	;Sb[n]=Ab
	rol.l	d3,d1	;d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;d5=L0b=L0b+Ab+L1b
	add.l	(a4),d0	;Aa=Aa+P+nQ
	rol.l	d7,d5	;d5=L0b=L0b<<<(Ab+L1b)
	move.l	d1,d3	;d3=L0a
	add.l	d1,d0	;Aa=Aa+P+nQ+L0a
	add.l	(a4)+,d4	;Ab=Ab+P+nQ, update P+nQ
	rol.l	#3,d0	;Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;d3=Aa+L0a
	move.l	d0,(a2)+	;Sa[n]=Aa
	rol.l	#3,d4	;Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;d7=L0b
	add.l	d3,d2	;L1a=L1a+Aa+L0a
	add.l	d4,d7	;d7=Ab+L0b
	move.l	d4,(a3)+	;Sb[n]=Ab
	rol.l	d3,d2	;L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;L1b=L1b+Ab+L0b
	add.l	(a4),d0	;d0=Aa=Aa+P+nQ
	rol.l	d7,d6	;L1b=L1b<<<(Ab+L0b)	(sOEP)

	ENDR

	; Final round 1 even-odd iteration

	add.l	(a4)+,d4	;d4=Ab=Ab+P+nQ, update P+nQ	(pOEP)
	add.l	d2,d0	;d0=Aa=Aa+P+nQ+L1a
	add.l	d6,d4	;d4=Ab=Ab+P+nQ+L1b
	rol.l	#3,d0	;d0=Aa=(Aa+P+nQ+L1a)<<<3
	move.l	d2,d3	;d3=L1a
	move.l	d0,(a2)+	;Sa[24]=Aa
	rol.l	#3,d4	;d4=Ab=(Ab+P+nQ+L1b)<<<3
	add.l	d0,d3	;d3=Aa+L1a
	move.l	d6,d7	;d7=L1b
	add.l	d3,d1	;d1=L0a=L0a+Aa+L1a
	add.l	d4,d7	;d7=Ab+L1b
	move.l	d4,(a3)+	;Sb[24]=Ab
	rol.l	d3,d1	;d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;d5=L0b=L0b+Ab+L1b
	add.l	(a4),d0	;Aa=Aa+P+nQ
	rol.l	d7,d5	;d5=L0b=L0b<<<(Ab+L1b)
	add.l	(a4),d4	;Ab=Ab+P+nQ
	add.l	d1,d0	;Aa=Aa+P+nQ+L0a
	add.l	d5,d4	;Ab=Ab+P+nQ+L0b
	rol.l	#3,d0	;Aa=(Aa+P+nQ+L0a)<<<3
	move.l	d1,d3	;d3=L0a
	move.l	d0,(a2)	;Sa[25]=Aa
	rol.l	#3,d4	;Ab=(Ab+P+nQ+L0b)<<<3
	add.l	d0,d3	;d3=Aa+L0a
	move.l	d5,d7	;d7=L0b
	add.l	d3,d2	;L1a=L1a+Aa+L0a
	add.l	d4,d7	;d7=Ab+L0b
	rol.l	d3,d2	;L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;L1b=L1b+Ab+L0b
	move.l	d4,(a3)	;Sb[25]=Ab
	lea	.ruf_sarray(pc),a1	;a1=Sa[] array
	rol.l	d7,d6	;L1b=L1b<<<(Ab+L0b)	(pOEP)

	;---- Round 2 of key expansion ----
	; First iteration: Sx[00] is constant P0QR3
	;                  Sx[01] is constant RUFV_AX

	move.l	a1,a2	;a2=Sa[] array
	lea	104(a1),a3	;a3=Sb[] array
	add.l	a5,d0	;Aa=Aa+Sa[00]
	add.l	d6,d4	;Ab=Ab+Sb[00]+L1b
	add.l	d2,d0	;Aa=Aa+Sa[00]+L1a
	add.l	a5,d4	;Ab=Ab+Sb[00]
	rol.l	#3,d0	;Aa=(Aa+Sa[00]+L1a)<<<3
	move.l	d2,d3	;d3=L1a
	rol.l	#3,d4	;Ab=(Ab+Sb[00]+L1b)<<<3
	move.l	d0,(a2)+	;Sa[00]=Aa
	move.l	d6,d7	;d7=L1b
	add.l	d0,d3	;d3=Aa+L1a
	add.l	d4,d7	;d7=Ab+L1b
	add.l	d3,d1	;L0a=L0a+Aa+L1a
	add.l	d7,d5	;L0b=L0b+Ab+L1b
	rol.l	d3,d1	;L0a=L0a<<<(Aa+L1a)
	move.l	d4,(a3)+	;Sb[00]=Ab
	rol.l	d7,d5	;L0b=L0b<<<(Ab+L1b)
	add.l	RUFV_AX(a6),d0	;Aa=Aa+Sa[01]
	add.l	d5,d4	;Ab=Ab+Sb[01]+L0b
	add.l	d1,d0	;Aa=Aa+Sa[01]+L0a
	add.l	RUFV_AX(a6),d4	;Ab=Ab+Sb[01]
	rol.l	#3,d0	;Aa=(Aa+Sa[01]+L0a)<<<3
	move.l	d1,d3	;d3=L0a
	rol.l	#3,d4	;Ab=(Ab+Sb[01]+L0b)<<<3
	move.l	d0,(a2)+	;Sa[01]=Aa
	move.l	d5,d7	;d7=L0b
	add.l	d0,d3	;d3=Aa+L0a
	add.l	d4,d7	;d7=Ab+L0b
	add.l	d3,d2	;L1a=L1a+Aa+L0a
	add.l	d7,d6	;L1b=L1b+Ab+L0b
	rol.l	d3,d2	;L1a=L1a<<<(Aa+L0a)
	move.l	d4,(a3)+	;Sb[01]=Ab
	rol.l	d7,d6	;L1b=L1b<<<(Ab+L0b)	(pOEP)

	; Repeated even-odd-even-odd rounds

	REPT	11

	add.l	(a2),d0	;Aa=Aa+Sa[n]	(sOEP)
	add.l	d6,d4	;Ab=Ab+Sb[n]+L1b
	add.l	d2,d0	;Aa=Aa+Sa[n]+L1a
	add.l	(a3),d4	;Ab=Ab+Sb[n]
	rol.l	#3,d0	;Aa=(Aa+Sa[n]+L1a)<<<3
	move.l	d2,d3	;d3=L1a
	rol.l	#3,d4	;Ab=(Ab+Sb[n]+L1b)<<<3
	move.l	d0,(a2)+	;Sa[n]=Aa
	move.l	d6,d7	;d7=L1b
	add.l	d0,d3	;d3=Aa+L1a
	add.l	d4,d7	;d7=Ab+L1b
	add.l	d3,d1	;L0a=L0a+Aa+L1a
	add.l	d7,d5	;L0b=L0b+Ab+L1b
	rol.l	d3,d1	;L0a=L0a<<<(Aa+L1a)
	move.l	d4,(a3)+	;Sb[n]=Ab
	rol.l	d7,d5	;L0b=L0b<<<(Ab+L1b)
	add.l	(a2),d0	;Aa=Aa+Sa[n]
	add.l	d5,d4	;Ab=Ab+Sb[n]+L0b
	add.l	d1,d0	;Aa=Aa+Sa[n]+L0a
	add.l	(a3),d4	;Ab=Ab+Sb[n]
	rol.l	#3,d0	;Aa=(Aa+Sa[n]+L0a)<<<3
	move.l	d1,d3	;d3=L0a
	rol.l	#3,d4	;Ab=(Ab+Sb[n]+L0b)<<<3
	move.l	d0,(a2)+	;Sa[n]=Aa
	move.l	d5,d7	;d7=L0b
	add.l	d0,d3	;d3=Aa+L0a
	add.l	d4,d7	;d7=Ab+L0b
	add.l	d3,d2	;L1a=L1a+Aa+L0a
	add.l	d7,d6	;L1b=L1b+Ab+L0b
	rol.l	d3,d2	;L1a=L1a<<<(Aa+L0a)
	move.l	d4,(a3)+	;Sb[n]=Ab
	rol.l	d7,d6	;L1b=L1b<<<(Ab+L0b)	(pOEP)

	ENDR

	; Final round 2 even-odd iteration

	add.l	(a2),d0	;Aa=Aa+Sa[24]	(sOEP)
	add.l	d6,d4	;Ab=Ab+Sb[24]+L1b
	add.l	d2,d0	;Aa=Aa+Sa[24]+L1a
	add.l	(a3),d4	;Ab=Ab+Sb[24]
	rol.l	#3,d0	;Aa=(Aa+Sa[24]+L1a)<<<3
	move.l	d2,d3	;d3=L1a
	rol.l	#3,d4	;Ab=(Ab+Sb[24]+L1b)<<<3
	move.l	d0,(a2)+	;Sa[24]=Aa
	move.l	d6,d7	;d7=L1b
	add.l	d0,d3	;d3=Aa+L1a
	add.l	d4,d7	;d7=Ab+L1b
	add.l	d3,d1	;L0a=L0a+Aa+L1a
	add.l	d7,d5	;L0b=L0b+Ab+L1b
	rol.l	d3,d1	;L0a=L0a<<<(Aa+L1a)
	move.l	d4,(a3)+	;Sb[24]=Ab
	rol.l	d7,d5	;L0b=L0b<<<(Ab+L1b)
	add.l	(a2),d0	;Aa=Aa+Sa[25]
	add.l	d5,d4	;Ab=Ab+Sb[25]+L0b
	add.l	d1,d0	;Aa=Aa+Sa[25]+L0a
	add.l	(a3),d4	;Ab=Ab+Sb[25]
	rol.l	#3,d0	;Aa=(Aa+Sa[25]+L0a)<<<3
	move.l	d1,d3	;d3=L0a
	rol.l	#3,d4	;Ab=(Ab+Sb[25]+L0b)<<<3
	move.l	d0,(a2)	;Sa[25]=Aa
	move.l	d5,d7	;d7=L0b
	add.l	d0,d3	;d3=Aa+L0a
	add.l	d4,d7	;d7=Ab+L0b
	add.l	d3,d2	;L1a=L1a+Aa+L0a
	move.l	d4,(a3)	;Sb[25]=Ab
	add.l	d7,d6	;L1b=L1b+Ab+L0b
	rol.l	d3,d2	;L1a=L1a<<<(Aa+L0a)
	rol.l	d7,d6	;L1b=L1b<<<(Ab+L0b)	(pOEP)

	;---- Combined round 3 of key expansion and encryption round ----

	move.l	d4,a4	;a4=Ab	(sOEP)

	;-- Perform round 3 for 'a' key --

	add.l	(a1)+,d0	;d0=Aa=Sa[00]
	move.l	d2,d3	;d3=L1a
	add.l	d2,d0	;d0=Aa=Sa[00]+L1a
	move.l	4(a0),d4	;d4=eAa=plain.lo
	rol.l	#3,d0	;d0=Aa=(Sa[00]+L1a)<<<3
	move.l	(a0),d7	;d7=eBa=plain.hi
	add.l	d0,d3	;d3=Aa+L1a
	add.l	d0,d4	;d4=eAa=plain.lo+Aa
	add.l	d3,d1	;d1=L0a=L0a+Aa+L1a
	add.l	(a1)+,d0	;d0=Aa=Aa+Sa[01]
	rol.l	d3,d1	;d1=L0a=(L0a+Aa+L1a)<<<(Aa+L1a)
	move.l	a1,a2	;a2=&Sa[02] for next
	add.l	d6,a4	;a4=Ab=Ab+L1b
	add.l	d1,d0	;d0=Aa=Aa+Sa[01]+L0a
	move.l	d1,d3	;d3=L0a
	rol.l	#3,d0	;d0=Aa=(Aa+Sa[01]+L0a)<<<3
	add.l	104-8(a1),a4	;a4=Ab=Ab+L1b+Sb[00]
	add.l	d0,d3	;d3=Aa+L0a
	add.l	d0,d7	;d7=eBa=plain.hi+Aa
	add.l	d3,d2	;d2=L1a=L1a+Aa+L0a
	add.l	(a1)+,d0	;d0=Aa=Aa+Sa[01]
	rol.l	d3,d2	;d2=L1a=(L1a+Aa+L0a)<<<(Aa+L0a)

	; Repeated round 3 even-odd-even-odd rounds
	REPT	11
	eor.l	d7,d4	;d4=eAa=eAa^eBa
	add.l	d2,d0	;d0=Aa=Aa+Sa[n]+L1a
	rol.l	d7,d4	;d4=eAa=((eAa^eBa)<<<eBa)
	rol.l	#3,d0	;d0=Aa=(Aa+Sa[n]+L1a)<<<3
	move.l	d2,d3	;d3=L1a
	add.l	d0,d4	;d4=eAa=((eAa^eBa)<<<eBa)+Aa
	add.l	d0,d3	;d3=Aa+L1a
	eor.l	d4,d7	;d7=eBa=eBa^eAa
	add.l	d3,d1	;d1=L0a=L0a+Aa+L1a
	add.l	(a1)+,d0	;d0=Aa=Aa+Sa[n]
	rol.l	d3,d1	;d1=L0a=(L0a+Aa+L1a)<<<(Aa+L1a)
	rol.l	d4,d7	;d7=eBa=((eBa^eAa)<<<eAa)
	add.l	d1,d0	;d0=Aa=Aa+Sa[n]+L0a
	move.l	d1,d3	;d3=L0a
	rol.l	#3,d0	;d0=Aa=(Aa+Sa[n]+L0a)<<<3

	add.l	d0,d3	;d3=Aa+L0a
	add.l	d0,d7	;d7=eBa=((eBa^eAa)<<<eAa)+Aa
	add.l	d3,d2	;d2=L1a=L1a+Aa+L0a
	add.l	(a1)+,d0	;d0=Aa=Aa+Sa[n]
	rol.l	d3,d2	;d2=L1a=L1a<<<(Aa+L0a)
	ENDR

	eor.l	d7,d4	;d4=eAa=eAa^eBa
	add.l	d2,d0	;d0=Aa=Aa+Sa[24]+L1a
	rol.l	d7,d4	;d4=eAa=((eAa^eBa)<<<eBa)
	rol.l	#3,d0	;d0=Aa=(Aa+Sa[24]+L1a)<<<3

	add.l	d0,d4	;d4=eAa=((eAa^eBa)<<<eBa)+Aa
	addq.l	#8,a1	;a1=Sb[]
	cmp.l	12(a0),d4	;d4=eAa=cypher.lo?
	bne.s	.ruf_notfounda	;Skip if not

	move.l	d0,d3	;d3=Aa

	add.l	d2,d3	;d3=Aa+L1a

	add.l	d3,d1	;d1=L0a=L0a+Aa+L1a
	add.l	-8(a1),d0	;d0=Aa=Aa+Sa[25]
	rol.l	d3,d1	;d1=L0a=(L0a+Aa+L1a)<<<(Aa+L1a)
	eor.l	d4,d7	;d7=eBa=eBa^eAa
	add.l	d1,d0	;d0=Aa=Aa+Sa[25]+L0a
	rol.l	d4,d7	;d7=eBa=((eBa^eAa)<<<eAa)
	rol.l	#3,d0	;d0=Aa=(Aa+Sa[25]+L0a)<<<3

	add.l	d0,d7	;d7=eBa=((eBa^eAa)<<<eAa)+Aa

	cmp.l	8(a0),d7	;eBa=cypher.hi?
	bne.s	.ruf_notfounda	;Skip if not
	;---------------------------------

	;Key found on pipeline 1!

	movem.l	RUFV_LOOP(a6),d0-1	;d0=loop count, d1=kiter

	sub.l	d0,d1	;Return number of keys checked
	add.l	d1,d1	; = loops*pipeline_count
	move.l	d1,d0

	movem.l	(a7)+,d2-7/a2-6
	rts

	CNOP	0,8
.ruf_notfounda:
	;-- Perform round 3 for 'b' key --

	move.l	a4,d0	;d0=Ab=Ab+L1b
	move.l	4(a0),d4	;d4=eAb=plain.lo
	rol.l	#3,d0	;d0=Ab=(Sb[00]+L1b)<<<3
	move.l	d6,d3	;d3=L1b
	add.l	d0,d4	;d4=eAb=plain.lo+Ab
	move.l	(a0),d7	;d7=eBb=plain.hi
	add.l	d0,d3	;d3=Ab+L1b
	add.l	d3,d5	;d5=L0b=L0b+Ab+L1b
	add.l	(a1)+,d0	;d0=Ab=Ab+Sb[01]
	rol.l	d3,d5	;d5=L0b=(L0b+Ab+L1b)<<<(Ab+L1b)

	add.l	d5,d0	;d0=Ab=Ab+Sb[01]+L0b
	move.l	d5,d3	;d3=L0b
	rol.l	#3,d0	;d0=Ab=(Ab+Sb[01]+L0b)<<<3
	move.l	a1,a3	;a3=&Sb[02]
	add.l	d0,d3	;d3=Ab+L0b
	add.l	d0,d7	;d7=eBb=plain.hi+Ab
	add.l	d3,d6	;d6=L1b=L1b+Ab+L0b
	add.l	(a1)+,d0	;d0=Ab=Ab+Sb[01]
	rol.l	d3,d6	;d6=L1b=(L1b+Ab+L0b)<<<(Ab+L0b)

	; Repeated round 3 even-odd-even-odd rounds
	REPT	11
	eor.l	d7,d4	;d4=eAb=eAb^eBb
	add.l	d6,d0	;d0=Ab=Ab+Sb[n]+L1a
	rol.l	d7,d4	;d4=eAb=((eAb^eBb)<<<eBb)
	rol.l	#3,d0	;d0=Ab=(Ab+Sb[n]+L1a)<<<3
	move.l	d6,d3	;d3=L1a
	add.l	d0,d4	;d4=eAb=((eAb^eBb)<<<eBb)+Ab
	add.l	d0,d3	;d3=Ab+L1b
	eor.l	d4,d7	;d7=eBb=eBb^eAb
	add.l	d3,d5	;d5=L0b=L0b+Ab+L1b
	add.l	(a1)+,d0	;d0=Ab=Ab+Sb[n]
	rol.l	d3,d5	;d5=L0b=(L0b+Ab+L1b)<<<(Ab+L1b)
	rol.l	d4,d7	;d7=eBb=((eBb^eAb)<<<eAb)
	add.l	d5,d0	;d0=Ab=Ab+Sb[n]+L0b
	move.l	d5,d3	;d3=L0b
	rol.l	#3,d0	;d0=Ab=(Ab+Sb[n]+L0b)<<<3

	add.l	d0,d3	;d3=Ab+L0b
	add.l	d0,d7	;d7=eBb=((eBb^eAb)<<<eAb)+Ab
	add.l	d3,d6	;d6=L1b=L1b+Ab+L0b
	add.l	(a1)+,d0	;d0=Ab=Ab+Sb[n]
	rol.l	d3,d6	;d6=L1b=L1b<<<(Ab+L0b)
	ENDR

	eor.l	d7,d4	;d4=eAb=eAb^eBb
	add.l	d6,d0	;d0=Ab=Ab+Sb[24]+L1b
	rol.l	d7,d4	;d4=eAb=((eAb^eBb)<<<eBb)
	rol.l	#3,d0	;d0=Ab=(Ab+Sb[24]+L1b)<<<3
	lea	.ruf_pnq(pc),a4	;a4=P+nQ lookup for next
	add.l	d0,d4	;d4=eAb=((eAb^eBb)<<<eBb)+Ab

	cmp.l	12(a0),d4	;d4=eAb=cypher.lo?
	bne.s	.ruf_notfoundb	;Skip if not

	move.l	d0,d3	;d3=Ab

	add.l	d6,d3	;d3=Ab+L1b

	add.l	d3,d5	;d5=L0b=L0b+Ab+L1b
	add.l	(a1),d0	;d0=Ab=Ab+Sb[25]
	rol.l	d3,d5	;d5=L0b=(L0b+Ab+L1b)<<<(Ab+L1b)
	eor.l	d4,d7	;d7=eBb=eBb^eAb
	add.l	d5,d0	;d0=Ab=Ab+Sb[25]+L0b
	rol.l	d4,d7	;d7=eBb=((eBb^eAb)<<<eAb)
	rol.l	#3,d0	;d0=Ab=(Ab+Sb[25]+L0b)<<<3

	add.l	d0,d7	;d7=eBb=((eBb^eAb)<<<eAb)+Ab

	cmp.l	8(a0),d7	;eBb=cypher.hi?
	bne.s	.ruf_notfoundb	;Skip if not

	;---------------------------------
	;Key found on pipeline 2!

	movem.l	RUFV_LOOP(a6),d0-1	;d0=loop count, d1=kiter

	sub.l	d0,d1	;Return number of keys checked
	add.l	d1,d1	; = loops*pipeline_count+1
	addq.l	#1,d1
	addq.b	#1,16(a0)	;Adjust L0.hi for pipeline 2
	move.l	d1,d0

	movem.l	(a7)+,d2-7/a2-6
	rts

	CNOP	0,8
.ruf_notfoundb:
	;Mangle-increment current key

	addq.b	#2,16(a0)	;Increment L1 by PIPELINE_COUNT
	lea	RUFV_L0X(a6),a1
	bcc.s	.ruf_midone
	addq.b	#1,17(a0)	;Every 256^1 (256)
	bcc.s	.ruf_midone
	addq.b	#1,18(a0)	;Every 256^2 (65536)
	bcc.s	.ruf_midone
	addq.b	#1,19(a0)	;Every 256^3 (16777216)
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

	add.l	a5,d1	;L0x=L0x+Sx[00]
	ror.l	#3,d1	;L0x=(L0x+Ax)>>>3 = (L0x+Ax)<<<P0QR3
	move.l	d1,RUFV_L0X(a6)	;Save (L0x+Ax)<<<P0QR3

	add.l	#PR3Q,d1
	rol.l	#3,d1
	move.l	d1,RUFV_AX(a6)

	add.l	RUFV_L0X(a6),d1
	move.l	d1,RUFV_L0XAX(a6)

	RUF_CODEALIGN
.ruf_midone:	subq.l	#1,RUFV_LOOP(a6)	;Loop back for next key
	bpl	.ruf_mainloop

	;---------------------------------
	;Key not found on either pipeline

	move.l	RUFV_KITER(a6),d0	;Return iterations*pipeline_count
	add.l	d0,d0

	movem.l	(a7)+,d2-7/a2-6 	;Didnt find a key
	rts

;--------------------

	CNOP	0,8
.ruf_sarray:	dcb.l	26*2	;Sx[] storage

	CNOP	0,8
.ruf_vars:	dcb.b	RUFV_SIZE	;Variables
	dc.l	1<<24	;Must follow variables!

	CNOP	0,8
.ruf_pnq:	dc.l	$F45044D5,$9287BE8E	;Table of P+nQ values for 2<=n<=26
	dc.l	$30BF3847,$CEF6B200
	dc.l	$6D2E2BB9,$0B65A572
	dc.l	$A99D1F2B,$47D498E4
	dc.l	$E60C129D,$84438C56
	dc.l	$227B060F,$C0B27FC8
	dc.l	$5EE9F981,$FD21733A
	dc.l	$9B58ECF3,$399066AC
	dc.l	$D7C7E065,$75FF5A1E
	dc.l	$1436D3D7,$B26E4D90
	dc.l	$50A5C749,$EEDD4102
	dc.l	$8D14BABB,$2B4C3474

;--------------------
