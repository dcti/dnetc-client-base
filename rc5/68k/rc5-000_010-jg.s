
	SECTION	rc5core,CODE
	OPT	O+,W-

;--------------------

	XDEF	_rc5_unit_func_000_010

	INCLUDE	rc5-0x0-jg.i

;--------------------
;@(#)$Id: rc5-000_010-jg.s,v 1.1.2.1 2000/06/05 15:19:08 oliver Exp $

	; $VER: MC68000/68010 RC5 core 24-Jan-2000
	;
	; MC680x0 RC5 key checking function
	; Dual key, unrolled loops, MC68000/MC68010 optimised
	;
	; Written by John Girvin <girv@girvnet.freeserve.co.uk>
	;
	; Entry: a0=rc5unitwork structure:
	;	 0(a0) = plain.hi  - plaintext
	;	 4(a0) = plain.lo
	;	 8(a0) = cypher.hi - cyphertext
	;	12(a0) = cypher.lo
	;	16(a0) = L0.hi     - key
	;	20(a0) = L0.lo
	;        d0=number of iterations to run for
	;


_rc5_unit_func_000_010:
	movem.l	d2-7/a2-6,-(a7)	;[ ]

	lea	ruf000_vars(pc),a6	;[ ] a6=core variables
	move.l	d0,-(a7)	;[ ] Save initial loop counter
	move.l	#P0QR3,a5	;[ ] a5=handy constant
	move.l	d0,-(a7)	;[ ] Set loop counter
	move.l	20(a0),d1	;[ ] d1=L0
	add.l	a5,d1	;[ ] L0=L0+S[00]
	ror.l	#3,d1	;[ ] L0=(L0+A)>>>3 = (L0+A)<<<P0QR3
	move.l	d1,RUFV_L0X(a6)	;[ ] Set (L0+A)<<<P0QR3
	move.l	#PR3Q,d2	;[ ] d2=Ax=Ax+P+Q
	add.l	d1,d2	;[ ] d2=Ax=Ax+P+Q+L0x
	rol.l	#3,d2	;[ ] d2=Ax=(Ax+P+Q+L0x)<<<3
	lea	RUFV_L0X(a6),a1	;[ ] a1=vars for loop
	move.l	d2,RUFV_AX(a6)	;[ ] Set Ax
	move.l	d2,d0	;[ ] d0=Ax
	lea	ruf000_sarray+8(pc),a2	;[ ] a2=&Sa[02]
	add.l	d1,d2	;[ ] d2=L0x+Ax
	lea	104(a2),a3	;[ ] a3=&Sb[02]
	move.l	d2,RUFV_L0XAX(a6)	;[ ] Set L0x+Ax
	add.l	#P2Q,d0	;[ ] d0=Ax+P+2Q
	move.l	d0,RUFV_AXP2Q(a6)	;[ ] Set Ax+P+2Q

ruf000_mainloop:
	;A : d0 d4
	;L0: d1 d5
	;L1: d2 d6

	move.l	16(a0),d2	;[ ] d2=L1x
	move.l	(a1)+,d1	;[ ] d1=L0a=RUFV_L0X
	move.l	d1,d5	;[ ] d5=L0b=L0a
	move.l	(a1)+,d3	;[ ] d3=Ax+L0x=RUFV_L0XAX
	add.l	d3,d2	;[ ] d2=L1a=L1x+Ax+L0x
	move.l	(a1)+,d0	;[ ] d0=Aa=Ax+P+2Q=RUFV_AXP2Q
	move.l	d2,d6	;[ ] d6=L1a
	add.l	(a1)+,d6	;[ ] d6=L1b=L1a+1<<24, a1=&Sa[00]
	rol.l	d3,d2	;[ ] d2=L1a=L1x<<<(Ax+L0x)
	move.l	d0,d4	;[ ] d4=Ab=Aa
	rol.l	d3,d6	;[ ] d6=L1b=L1b<<<(Ab+L0b)

	; Repeated round 1 even-odd-even-odd rounds

	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	move.l	d2,d3	;[ ] d3=L1a
	move.l	d0,(a2)+	;[ ] Sa[02]=Aa, a2=&Sa[03]
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[02]=Ab, a3=&Sb[03]
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$9287BE8E,d0	;[ ] Aa=Aa+P+3Q
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	move.l	d1,d3	;[ ] d3=L0a
	add.l	#$9287BE8E,d4	;[ ] Ab=Ab+P+3Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	rol.l	#3,d4	;p] [1] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d0,(a2)+	;[ ] Sa[03]=Aa, a2=&Sa[04]
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[03]=Ab, a3=&Sb[04]
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$30BF3847,d0	;[ ] Aa=Aa+P+4Q

	; 1
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$30BF3847,d4	;[ ] d4=Ab=Ab+P+4Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$CEF6B200,d0	;[ ] Aa=Aa+P+5Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$CEF6B200,d4	;[ ] Ab=Ab+P+5Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$6D2E2BB9,d0	;[ ] d0=Aa=Aa+P+6Q

	; 2
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$6D2E2BB9,d4	;[ ] d4=Ab=Ab+P+6Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$0B65A572,d0	;[ ] Aa=Aa+P+7Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$0B65A572,d4	;[ ] Ab=Ab+P+7Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$A99D1F2B,d0	;[ ] d0=Aa=Aa+P+8Q

	; 3
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$A99D1F2B,d4	;[ ] d4=Ab=Ab+P+8Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$47D498E4,d0	;[ ] Aa=Aa+P+9Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$47D498E4,d4	;[ ] Ab=Ab+P+9Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$E60C129D,d0	;[ ] d0=Aa=Aa+P+10Q

	; 4
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$E60C129D,d4	;[ ] d4=Ab=Ab+P+10Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$84438C56,d0	;[ ] Aa=Aa+P+11Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$84438C56,d4	;[ ] Ab=Ab+P+11Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$227B060F,d0	;[ ] d0=Aa=Aa+P+12Q

	; 5
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$227B060F,d4	;[ ] d4=Ab=Ab+P+12Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$C0B27FC8,d0	;[ ] Aa=Aa+P+13Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$C0B27FC8,d4	;[ ] Ab=Ab+P+13Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$5EE9F981,d0	;[ ] d0=Aa=Aa+P+14Q

	; 6
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$5EE9F981,d4	;[ ] d4=Ab=Ab+P+14Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$FD21733A,d0	;[ ] Aa=Aa+P+15Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$FD21733A,d4	;[ ] Ab=Ab+P+15Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$9B58ECF3,d0	;[ ] d0=Aa=Aa+P+16Q

	; 7
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$9B58ECF3,d4	;[ ] d4=Ab=Ab+P+16Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$399066AC,d0	;[ ] Aa=Aa+P+17Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$399066AC,d4	;[ ] Ab=Ab+P+17Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$D7C7E065,d0	;[ ] d0=Aa=Aa+P+18Q

	; 8
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$D7C7E065,d4	;[ ] d4=Ab=Ab+P+18Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$75FF5A1E,d0	;[ ] Aa=Aa+P+19Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$75FF5A1E,d4	;[ ] Ab=Ab+P+19Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$1436D3D7,d0	;[ ] d0=Aa=Aa+P+20Q

	; 9
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$1436D3D7,d4	;[ ] d4=Ab=Ab+P+20Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$B26E4D90,d0	;[ ] Aa=Aa+P+21Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$B26E4D90,d4	;[ ] Ab=Ab+P+21Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$50A5C749,d0	;[ ] d0=Aa=Aa+P+22Q

	; 10
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$50A5C749,d4	;[ ] d4=Ab=Ab+P+22Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$EEDD4102,d0	;[ ] Aa=Aa+P+23Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$EEDD4102,d4	;[ ] Ab=Ab+P+23Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	add.l	#$8D14BABB,d0	;[ ] d0=Aa=Aa+P+24Q

	; Final round 1 even-odd iteration

	move.l	d2,d3	;[ ] d3=L1a
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;[ ] d0=Aa=Aa+P+nQ+L1a
	add.l	#$8D14BABB,d4	;[ ] d4=Ab=Ab+P+24Q
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;[ ] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	d0,(a2)+	;[ ] Sa[24]=Aa
	rol.l	#3,d4	;[ ] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)	;[ ] Sb[24]=Ab
	add.l	d4,d7	;[ ] d7=Ab+L1b
	rol.l	d3,d1	;[ ] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	#$2B4C3474,d0	;[ ] Aa=Aa+P+25Q
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	d7,d5	;[ ] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;[ ] Aa=Aa+P+nQ+L0a
	add.l	#$2B4C3474,d4	;[ ] Ab=Ab+P+25Q
	rol.l	#3,d0	;[ ] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;[ ] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	move.l	d0,(a2)	;[ ] Sa[25]=Aa
	rol.l	#3,d4	;[ ] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	add.l	d4,d7	;[ ] d7=Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	move.l	d4,a4	;[ ] Sb[25]=Ab
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)

	;---- Round 2 of key expansion ----
	; First iteration: Sx[00] is constant P0QR3
	;                  Sx[01] is constant RUFV_AX

	move.l	a1,a2	;[ ] a2=Sa[] array
	lea	104(a1),a3	;[ ] a3=Sb[] array
	add.l	a5,d0	;[ ] Aa=Aa+Sa[00]
	add.l	d6,d4	;[ ] Ab=Ab+Sb[00]+L1b
	add.l	d2,d0	;[ ] Aa=Aa+Sa[00]+L1a
	add.l	a5,d4	;[ ] Ab=Ab+Sb[00]
	rol.l	#3,d0	;[ ] Aa=(Aa+Sa[00]+L1a)<<<3
	move.l	d2,d3	;[ ] d3=L1a
	rol.l	#3,d4	;[ ] Ab=(Ab+Sb[00]+L1b)<<<3
	move.l	d0,(a2)+	;[ ] Sa[00]=Aa, a2=&Sa[01]
	move.l	d6,d7	;[ ] d7=L1b
	add.l	d0,d3	;[ ] d3=Aa+L1a
	add.l	d4,d7	;[ ] d7=Ab+L1b
	add.l	d3,d1	;[ ] L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[00]=Ab, a3=&Sb[01]
	add.l	d7,d5	;[ ] L0b=L0b+Ab+L1b
	rol.l	d3,d1	;[ ] L0a=L0a<<<(Aa+L1a)
	rol.l	d7,d5	;[ ] L0b=L0b<<<(Ab+L1b)
	add.l	RUFV_AX(a6),d0	;[ ] Aa=Aa+Sa[01]
	move.l	d1,d3	;[ ] d3=L0a
	add.l	d5,d4	;[ ] Ab=Ab+Sb[01]+L0b
	add.l	d1,d0	;[ ] Aa=Aa+Sa[01]+L0a
	add.l	RUFV_AX(a6),d4	;[ ] Ab=Ab+Sb[01]
	rol.l	#3,d0	;[ ] Aa=(Aa+Sa[01]+L0a)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	rol.l	#3,d4	;[ ] Ab=(Ab+Sb[01]+L0b)<<<3
	move.l	d0,(a2)+	;[ ] Sa[01]=Aa, a2=&Sa[02]
	add.l	d0,d3	;[ ] d3=Aa+L0a
	add.l	d4,d7	;[ ] d7=Ab+L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[01]=Ab, a3=&Sb[02]
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	(a2),d0	;[ ] Aa=Aa+Sa[n]

	; Repeated round 2 even-odd-even-odd rounds

	REPT	11
	move.l	d2,d3	;[ ] d3=L1a
	add.l	d6,d4	;[ ] Ab=Ab+Sb[n]+L1b
	add.l	d2,d0	;[ ] Aa=Aa+Sa[n]+L1a
	add.l	(a3),d4	;[ ] Ab=Ab+Sb[n]
	rol.l	#3,d0	;[ ] Aa=(Aa+Sa[n]+L1a)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	rol.l	#3,d4	;[ ] Ab=(Ab+Sb[n]+L1b)<<<3
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	add.l	d0,d3	;[ ] d3=Aa+L1a
	add.l	d4,d7	;[ ] d7=Ab+L1b
	add.l	d3,d1	;[ ] L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d7,d5	;[ ] L0b=L0b+Ab+L1b
	rol.l	d3,d1	;[ ] L0a=L0a<<<(Aa+L1a)
	rol.l	d7,d5	;[ ] L0b=L0b<<<(Ab+L1b)
	add.l	(a2),d0	;[ ] Aa=Aa+Sa[n]
	move.l	d1,d3	;[ ] d3=L0a
	add.l	d5,d4	;[ ] Ab=Ab+Sb[n]+L0b
	add.l	d1,d0	;[ ] Aa=Aa+Sa[n]+L0a
	add.l	(a3),d4	;[ ] Ab=Ab+Sb[n]
	rol.l	#3,d0	;[ ] Aa=(Aa+Sa[n]+L0a)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	rol.l	#3,d4	;[ ] Ab=(Ab+Sb[n]+L0b)<<<3
	move.l	d0,(a2)+	;[ ] Sa[n]=Aa
	add.l	d0,d3	;[ ] d3=Aa+L0a
	add.l	d4,d7	;[ ] d7=Ab+L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;[ ] Sb[n]=Ab
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)
	add.l	(a2),d0	;[ ] Aa=Aa+Sa[24]
	ENDR

	; Final round 2 even-odd iteration

	move.l	d2,d3	;[ ] d3=L1a
	add.l	d6,d4	;[ ] Ab=Ab+Sb[24]+L1b
	add.l	d2,d0	;[ ] Aa=Aa+Sa[24]+L1a
	add.l	(a3),d4	;[ ] Ab=Ab+Sb[24]
	rol.l	#3,d0	;[ ] Aa=(Aa+Sa[24]+L1a)<<<3
	move.l	d6,d7	;[ ] d7=L1b
	rol.l	#3,d4	;[ ] Ab=(Ab+Sb[24]+L1b)<<<3
	move.l	d0,(a2)+	;[ ] Sa[24]=Aa, a2=&Sa[25]
	add.l	d0,d3	;[ ] d3=Aa+L1a
	add.l	d4,d7	;[ ] d7=Ab+L1b
	add.l	d3,d1	;[ ] L0a=L0a+Aa+L1a
	move.l	d4,(a3)	;[ ] Sb[24]=Ab
	add.l	d7,d5	;[ ] L0b=L0b+Ab+L1b
	rol.l	d3,d1	;[ ] L0a=L0a<<<(Aa+L1a)
	rol.l	d7,d5	;[ ] L0b=L0b<<<(Ab+L1b)
	add.l	(a2),d0	;[ ] Aa=Aa+Sa[25]
	add.l	d5,d4	;[ ] Ab=Ab+L0b
	add.l	d1,d0	;[ ] Aa=Aa+Sa[25]L0a
	add.l	a4,d4	;[ ] Ab=Ab+L0b+Sb[25]
	rol.l	#3,d0	;[ ] Aa=(Aa+Sa[25]+L0a)<<<3
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	#3,d4	;[ ] Ab=(Ab+Sb[25]+L0b)<<<3
	move.l	d5,d7	;[ ] d7=L0b
	add.l	d0,d3	;[ ] d3=Aa+L0a
	add.l	d4,d7	;[ ] d7=Ab+L0b
	add.l	d3,d2	;[ ] L1a=L1a+Aa+L0a
	add.l	(a1)+,d0	;[ ] d0=Aa=Sa[00]
	add.l	d7,d6	;[ ] L1b=L1b+Ab+L0b
	rol.l	d3,d2	;[ ] L1a=L1a<<<(Aa+L0a)
	rol.l	d7,d6	;[ ] L1b=L1b<<<(Ab+L0b)

	;---- Combined round 3 of key expansion and encryption round ----

	add.l	d2,d0	;[ ] d0=Aa=Sa[00]+L1a
	move.l	d4,a4	;[ ] a4=Ab (save Ab in a4)
	rol.l	#3,d0	;[ ] d0=Aa=(Sa[00]+L1a)<<<3
	move.l	4(a0),d4	;[ ] d4=eAa=plain.lo
	move.l	d2,d3	;[ ] d3=L1a
	add.l	d0,d4	;[ ] d4=eAa=plain.lo+Aa
	add.l	d0,d3	;[ ] d3=Aa+L1a
	move.l	(a0),d7	;[ ] d7=eBa=plain.hi
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	lea	4(a1),a2	;[ ] a2=next iter &Sa[02]
	rol.l	d3,d1	;[ ] d1=L0a=(L0a+Aa+L1a)<<<(Aa+L1a)
	add.l	(a1)+,d0	;[ ] d0=Aa=Aa+Sa[01], a1=&Sa[02]
	add.l	d1,d0	;[ ] d0=Aa=Aa+Sa[01]+L0a
	move.l	d1,d3	;[ ] d3=L0a
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+Sa[01]+L0a)<<<3
	add.l	104-8(a1),a4	;[ ] a4=Ab=Ab+L1b+Sb[00]
	add.l	d0,d3	;[ ] d3=Aa+L0a
	add.l	d0,d7	;[ ] d7=eBa=plain.hi+Aa
	add.l	d3,d2	;[ ] d2=L1a=L1a+Aa+L0a
	add.l	(a1)+,d0	;[ ] d0=Aa=Aa+Sa[02], a1=&Sa[03]
	rol.l	d3,d2	;[ ] d2=L1a=(L1a+Aa+L0a)<<<(Aa+L0a)
	eor.l	d7,d4	;[ ] d4=eAa=eAa^eBa

	; Repeated round 3 even-odd-even-odd rounds
	REPT	11
	add.l	d2,d0	;[ ] d0=Aa=Aa+Sa[n]+L1a
	rol.l	d7,d4	;[ ] d4=eAa=((eAa^eBa)<<<eBa)
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+Sa[n]+L1a)<<<3
	move.l	d2,d3	;[ ] d3=L1a
	add.l	d0,d3	;[ ] d3=Aa+L1a
	add.l	d0,d4	;[ ] d4=eAa=((eAa^eBa)<<<eBa)+Aa
	add.l	d3,d1	;[ ] d1=L0a=L0a+Aa+L1a
	add.l	(a1)+,d0	;[ ] d0=Aa=Aa+Sa[n], a1=&Sa[n+1]
	rol.l	d3,d1	;[ ] d1=L0a=(L0a+Aa+L1a)<<<(Aa+L1a)
	eor.l	d4,d7	;[ ] d7=eBa=eBa^eAa
	add.l	d1,d0	;[ ] d0=Aa=Aa+Sa[n]+L0a
	rol.l	d4,d7	;[ ] d7=eBa=((eBa^eAa)<<<eAa)
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+Sa[n]+L0a)<<<3
	move.l	d1,d3	;[ ] d3=L0a
	add.l	d0,d3	;[ ] d3=Aa+L0a
	add.l	d0,d7	;[ ] d7=eBa=((eBa^eAa)<<<eAa)+Aa
	add.l	d3,d2	;[ ] d2=L1a=L1a+Aa+L0a
	add.l	(a1)+,d0	;[ ] d0=Aa=Aa+Sa[n], a1=&Sa[n+1]
	rol.l	d3,d2	;[ ] d2=L1a=(L1a+Aa+L0a)<<<(Aa+L0a)
	eor.l	d7,d4	;[ ] d4=eAa=eAa^eBa
	ENDR

	;-- Generate and check low 32 bits of result --
	add.l	d2,d0	;[ ] d0=Aa=Aa+Sa[24]+L1a
	rol.l	d7,d4	;[ ] d4=eAa=((eAa^eBa)<<<eBa)
	rol.l	#3,d0	;[ ] d0=Aa=(Aa+Sa[24]+L1a)<<<3
	addq.l	#8,a1	;[ ] a1=Sb[]
	add.l	d0,d4	;[ ] d4=eAa=((eAa^eBa)<<<eBa)+Aa
	add.l        d6,a4        	;[ ] a4=Ab=Ab+L1b
	move.l	d6,d3	;[ ] d3=L1b
	cmp.l	12(a0),d4	;[ ] eAa == cypher.lo?
	bne.s	ruf000_notfounda	;[ ] Skip if not

	;-- Low 32 bits match! (1 in 2^32 keys!) --
	; Need to completely re-check key but
	; generating high 32 bits as well...

	moveq	#0,d0
	bsr	ruf000_check64
	bne.s	ruf000_notfounda

	;---------------------------------------
	; 'Interesting' key found on pipeline 1

	move.l	(a7)+,d0	;d0=loop count
	move.l	(a7)+,d1	;d1=initial loop count
	sub.l	d0,d1	;Return number of keys checked
	add.l	d1,d1	; = loops*pipeline_count
	move.l	d1,d0

	movem.l	(a7)+,d2-7/a2-6
	rts

ruf000_notfounda:
	;-- Perform round 3 for 'b' key --

	move.l	a4,d0	;[ ] d0=Ab=Ab+L1b
	rol.l	#3,d0	;[ ] d0=Ab=(Sb[00]+L1b)<<<3
	move.l	4(a0),d4	;[ ] d4=eAb=plain.lo
	add.l	d0,d3	;[ ] d3=Ab+L1b
	add.l	d0,d4	;[ ] d4=eAb=plain.lo+Ab
	add.l	d3,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	(a1)+,d0	;[ ] d0=Ab=Ab+Sb[01]
	rol.l	d3,d5	;[ ] d5=L0b=(L0b+Ab+L1b)<<<(Ab+L1b)
	move.l	a1,a3	;[ ] a3=&Sb[02]
	add.l	d5,d0	;[ ] d0=Ab=Ab+Sb[01]+L0b
	move.l	d5,d3	;[ ] d3=L0b
	rol.l	#3,d0	;[ ] d0=Ab=(Ab+Sb[01]+L0b)<<<3
	move.l	(a0),d7	;[ ] d7=eBb=plain.hi
	add.l	d0,d3	;[ ] d3=Ab+L0b
	add.l	d0,d7	;[ ] d7=eBb=plain.hi+Ab
	add.l	d3,d6	;[ ] d6=L1b=L1b+Ab+L0b
	add.l	(a1)+,d0	;[ ] d0=Ab=Ab+Sb[01]
	rol.l	d3,d6	;[ ] d6=L1b=(L1b+Ab+L0b)<<<(Ab+L0b)
	eor.l	d7,d4	;[ ] d4=eAb=eAb^eBb

	; Repeated round 3 even-odd-even-odd rounds
	REPT	11
	add.l	d6,d0	;[ ] d0=Ab=Ab+Sb[n]+L1a
	rol.l	d7,d4	;[ ] d4=eAb=((eAb^eBb)<<<eBb)
	rol.l	#3,d0	;[ ] d0=Ab=(Ab+Sb[n]+L1a)<<<3
	move.l	d6,d3	;[ ] d3=L1a
	add.l	d0,d4	;[ ] d4=eAb=((eAb^eBb)<<<eBb)+Ab
	add.l	d0,d3	;[ ] d3=Ab+L1b
	eor.l	d4,d7	;[ ] d7=eBb=eBb^eAb
	add.l	d3,d5	;[ ] d5=L0b=L0b+Ab+L1b
	add.l	(a1)+,d0	;[ ] d0=Ab=Ab+Sb[n]
	rol.l	d3,d5	;[ ] d5=L0b=(L0b+Ab+L1b)<<<(Ab+L1b)
	rol.l	d4,d7	;[ ] d7=eBb=((eBb^eAb)<<<eAb)
	add.l	d5,d0	;[ ] d0=Ab=Ab+Sb[n]+L0b
	rol.l	#3,d0	;[ ] d0=Ab=(Ab+Sb[n]+L0b)<<<3
	move.l	d5,d3	;[ ] d3=L0b
	add.l	d0,d3	;[ ] d3=Ab+L0b
	add.l	d0,d7	;[ ] d7=eBb=((eBb^eAb)<<<eAb)+Ab
	add.l	d3,d6	;[ ] d6=L1b=L1b+Ab+L0b
	add.l	(a1)+,d0	;[ ] d0=Ab=Ab+Sb[n]
	rol.l	d3,d6	;[ ] d6=L1b=L1b<<<(Ab+L0b)
	eor.l	d7,d4	;[ ] d4=eAb=eAb^eBb
	ENDR

	add.l	d6,d0	;[ ] d0=Ab=Ab+Sb[24]+L1b
	rol.l	d7,d4	;[ ] d4=eAb=((eAb^eBb)<<<eBb)
	rol.l	#3,d0	;[ ] d0=Ab=(Ab+Sb[24]+L1b)<<<3
	add.l	d0,d4	;[ ] d4=eAb=((eAb^eBb)<<<eBb)+Ab
	cmp.l	12(a0),d4	;[ ] eAb == cypher.lo?
	bne.s	ruf000_notfoundb	;[ ] Skip if not

	;-- Low 32 bits match! (1 in 2^32 keys!) --
	; Need to completely re-check key but
	; generating high 32 bits as well...

	moveq	#1,d0
	bsr	ruf000_check64
	bne.s	ruf000_notfoundb

	;---------------------------------------
	; 'Interesting' key found on pipeline 2

	move.l	(a7)+,d0	;d0=loop count
	move.l	(a7)+,d1	;d1=initial loop count
	sub.l	d0,d1	;Return number of keys checked
	add.l	d1,d1	; = loops*pipeline_count+1
	addq.l	#1,d1
	addq.b	#1,16(a0)	;Adjust L0.hi for pipeline 2
	move.l	d1,d0
	movem.l	(a7)+,d2-7/a2-6
	rts

ruf000_notfoundb:
	;Mangle-increment current key

	addq.b	#2,16(a0)	;Increment L1 by pipeline count
	lea	RUFV_L0X(a6),a1
	bcc.s	ruf000_midone
	addq.b	#1,17(a0)	;Every 256^1 (256)
	bcc.s	ruf000_midone
	addq.b	#1,18(a0)	;Every 256^2 (65536)
	bcc.s	ruf000_midone
	addq.b	#1,19(a0)	;Every 256^3 (16777216)
	bcc.s	ruf000_midone

	addq.b	#1,20(a0)	;Every 256^4 (4294967296)
	bcc.s	ruf000_l0chg
	addq.b	#1,21(a0)	;Every 256^5 (1099511627776)
	bcc.s	ruf000_l0chg
	addq.b	#1,22(a0)	;Every 256^6 (281474976710656)
	bcc.s	ruf000_l0chg
	addq.b	#1,23(a0)	;Every 256^7 (72057594037927936)

	; Need to do anything special wrapping 0xff..f -> 0x00..0 ?

ruf000_l0chg:	; L0 has changed so recalculate "constants"

	add.l	a5,d1	;d1=L0=L0+S[00]
	ror.l	#3,d1	;d1=L0=(L0+A)>>>3 = (L0+A)<<<P0QR3
	move.l	d1,RUFV_L0X(a6)	;Set L0x

	add.l	#PR3Q,d1	;d1=A+P+Q
	rol.l	#3,d1	;d1=(A+P+Q)<<<3
	move.l	d1,RUFV_AX(a6)	;Set Ax

	add.l	RUFV_L0X(a6),d1	;d1=A+L0
	move.l	d1,RUFV_L0XAX(a6)	;Set A+L0

	move.l	RUFV_AX(a6),d1	;d1=A
	add.l	#P2Q,d1	;d1=A+P+2Q
	move.l	d1,RUFV_AXP2Q(a6)	;Set A+P+2Q

ruf000_midone:	subq.l	#1,(a7)	;Loop back for next key
	bne	ruf000_mainloop

	;---------------------------------
	; Key not found on either pipeline
	; Return iterations*pipeline_count

	addq.l	#4,a7	;Forget loop counter
	move.l	(a7)+,d0	;d0=initial loop counter
	add.l	d0,d0
	movem.l	(a7)+,d2-7/a2-6
	rts

;--------------------

	;
	; Completely (all 64 bits) check a single RC5-64 key.
	;
	; Entry:	a0=rc5unitwork structure
	;	 0(a0) = plain.hi  - plaintext
	;	 4(a0) = plain.lo
	;	 8(a0) = cypher.hi - cyphertext
	;	12(a0) = cypher.lo
	;	16(a0) = L0.hi     - key
	;	20(a0) = L0.lo
	;        d0=pipeline number 0/1
	;
	; Exit:	d0=0 if key is correct, !=0 otherwise
	;	Other registers preserved.
	;
	; NOTES:
	;   Speed not important since this will be called on
	;   average once for every 2^32 keys checked by the
	;   main core.
	;

ruf000_check64:
	movem.l	d2-3/d6-7/a1-3,-(a7)

	;---- Initialise S[] array ----

	lea	ruf000_sarray(pc),a1	;a1=aligned S[] storage

	;---- Mix secret key into S[] ----

	move.l	20(a0),d1	;d1=L0=L0.lo
	lea	ruf000_p2q(pc),a3	;a3=&S[02] read address
	ror.l	#8,d0	;d0=pipeline << 24
	move.l	16(a0),d2	;d2=L1=L0.hi
	move.l	a1,a2	;a2=&S[00] write address
	add.l	d0,d2	;Adjust L0 for pipeline

	;First iteration special case

	move.l	#P0QR3,d0	;d0=A=P<<<3

	move.l	d0,(a2)+	;S[00]=P<<<3
	add.l	d0,d1	;d1=L0=L0+A
	rol.l	d0,d1	;d1=L0=(L0+A)<<<A

	add.l	#P+Q,d0	;A=A+P+Q
	add.l	d1,d0	;A=A+P+Q+L0
	rol.l	#3,d0	;A=(A+P+Q+L0)<<<3
	move.l	d0,(a2)+	;S[01]=A

	; Begin proper iterations now we have initial L0 and L1

	moveq	#3-1,d7	;d7=outer loop counter
	moveq	#13-1-1,d6	;d6=initial inner loop counter
	bra.s	.c64_mixkey2

.c64_mixkey1:
	move.l	a1,a2	;a2=S[] storage
	move.l	a1,a3	;a3=S[] storage
	moveq	#13-1,d6	;d6=inner loop counter

.c64_mixkey2:
	;d0=A d1=L1 d2=L2 a2=&S[]

	move.l	d1,d3	;d3=L0
	add.l	d0,d3	;d3=A+L0
	add.l	d3,d2	;L1=L1+A+L0

	add.l	(a3)+,d0	;A=A+S[n]
	rol.l	d3,d2	;L1=L1<<<(A+L0)
	add.l	d2,d0	;A=A+S[n]+L1
	rol.l	#3,d0	;A=(A+S[n]+L1)<<<3
	move.l	d2,d3	;d3=L1
	move.l	d0,(a2)+	;S[n]=A

	add.l	d0,d3	;d3=A+L1
	add.l	d3,d1	;L0=L0+A+L1
	add.l	(a3)+,d0	;A=A+S[n]
	rol.l	d3,d1	;L0=L0<<<(A+L1)

	add.l	d1,d0	;A=A+S[n]+L0
	rol.l	#3,d0	;A=(A+S[n]+L0)<<<3
	move.l	d0,(a2)+	;S[n]=A

	dbf	d6,.c64_mixkey2
	dbf	d7,.c64_mixkey1

	;---- Perform the encryption ----

	move.l	(a1)+,d0	;d0=A=S[00]
	add.l	4(a0),d0	;d0=A=S[00]+plain.lo

	move.l	(a1)+,d1	;d1=B=S[01]
	moveq	#5-1,d7	;d7=loop counter
	add.l	(a0),d1	;d1=B=S[01]+plain.hi

	eor.l	d1,d0	;d0=A=A^B
	rol.l	d1,d0	;d0=A=A<<<B
	add.l	(a1)+,d0	;d0=A=A+S[n]

	eor.l	d0,d1	;d1=B=B^A
	rol.l	d0,d1	;d1=B=B<<<A
	add.l	(a1)+,d1	;d1=B=B+S[n]

.c64_encrypt:
	REPT	2
	eor.l	d1,d0	;d0=A=A^B
	rol.l	d1,d0	;d0=A=A<<<B
	add.l	(a1)+,d0	;d0=A=A+S[n]

	eor.l	d0,d1	;d1=B=B^A
	rol.l	d0,d1	;d1=B=B<<<A
	add.l	(a1)+,d1	;d1=B=B+S[n]
	ENDR

	dbf	d7,.c64_encrypt

	eor.l	d1,d0	;d0=A=A^B
	move.l	12(a0),d2	;d2=cypher.lo
	rol.l	d1,d0	;d0=A=A<<<B
	add.l	(a1),d0	;d0=A=A+S[24]

	eor.l	d0,d1	;d1=B=B^A
	move.l	8(a0),d2	;d2=cypher.hi
	rol.l	d0,d1	;d1=B=B<<<A
	add.l	4(a1),d1	;d1=B=B+S[25]
	cmp.l	d2,d1	;B=cypher.hi?
	beq.s	.c64_found

	moveq	#1,d0	;Didn't find the key this time...
	bra.s	.c64_done

.c64_found:	moveq	#0,d0	;Found the key!

.c64_done:	movem.l	(a7)+,d2-3/d6-7/a1-3
	tst.l	d0
	rts

;--------------------

	; Data for 32 bit check check core
	;
	; Must be in the order shown below with no gaps
	; between the different parts - core relies on it!

	CNOP	0,8
ruf000_vars:	dcb.b	RUFV_SIZE	;Variables
	dc.l	1<<24	;Must follow variables!

ruf000_sarray:	dcb.l	26*2	;Sx[] storage

;--------------------

	; Data for 64 bit check core

	CNOP	0,8
ruf000_p2q:	;Table of P+nQ values for 2<=n<=26
	dc.l	$F45044D5,$9287BE8E
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
