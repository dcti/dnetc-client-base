
	SECTION	rc5core_060,CODE
	OPT	O+,W-

	; $VER: MC68060 reentrant RC5 core 04-Feb-2001

	;
	; MC680x0 RC5 key checking function
	; for distributed.net RC5-64 clients.
	;
	; Dual key, unrolled loops, MC68060 optimised, rentrant
	;
	; Written by John Girvin <girv@girvnet.org.uk>
	;
	; TODO: some optimisation of initial setup may be possible
	;

;--------------------

	INCLUDE	rc5-0x0-common-jg.i

	XREF	_rc5_check64

	XDEF	_rc5_unit_func_060re

;--------------------

	;
	; MC680x0 RC5 key checking function
	; Dual key, unrolled loops, 060 optimised, rentrant
	;
	; Entry: a0=rc5unitwork structure:
	;	 0(a0) = plain.hi  - plaintext
	;	 4(a0) = plain.lo
	;	 8(a0) = cypher.hi - cyphertext
	;	12(a0) = cypher.lo
	;	16(a0) = L0.hi     - key
	;	20(a0) = L0.lo
	;        d0=number of iterations to run for
	;	=number of keys to check / 2
	;
	; NOTES:
	;   All cycle counts [n] are in 060 cycles.
	;
	;   'p'  indicates instruction runs in pOEP
	;   's'  indicates instruction runs in sOEP
	;
	;   'P'  indicates instruction is pOEP-only
	;      (ie: no superscalar pairing possible)
	;
	;   's]' indicates instruction pair would stall
	;   'p]'  if first was in pOEP instead of sOEP.
	;

	CNOP	0,8
_rc5_unit_func_060re:
	movem.l	d2-7/a2-6,-(a7)	;P  [11]

	lea	-208(a7),a7	;p  [ ] save space for Sx[] storage
	move.l	#$2B4C3474,d1	;   [0] d1=P+26Q
	move.l	#Q,d2	;p  [ ] d2=Q
	moveq	#26-3-1,d3	;   [0] d3=loop counter
.ruf_initp3q_loop:
	move.l	d1,-(a7)	;p  [ ] initialise P+nQ lookup
	sub.l	d2,d1
	dbf	d3,.ruf_initp3q_loop

	move.l	#1<<24,-(a7)	;       initialise core variables
	lea	-RUFV_SIZE(a7),a7

	move.l	a7,a6	;p  [1] a6=core variables
	move.l	d0,-(a7)	;   [0] Save initial loop counter
	move.l	#P0QR3,a5	;p  [1] a5=handy constant
	move.l	d0,-(a7)	;   [0] Set loop counter
	move.l	20(a0),d1	;p  [1] d1=L0
	add.l	a5,d1	;   [0] L0=L0+S[00]
	ror.l	#3,d1	;p  [1] L0=(L0+A)>>>3 = (L0+A)<<<P0QR3
	move.l	d1,RUFV_L0X(a6)	;   [0] Set (L0+A)<<<P0QR3
	move.l	#PR3Q,d2	;p  [1] d2=Ax=Ax+P+Q
	add.l	d1,d2	;   [0] d2=Ax=Ax+P+Q+L0x
	rol.l	#3,d2	;p  [1] d2=Ax=(Ax+P+Q+L0x)<<<3
	lea	RUFV_L0X(a6),a1	;   [0] a1=vars for loop
	move.l	d2,RUFV_AX(a6)	;p  [1] Set Ax
	move.l	d2,d0	;   [0] d0=Ax
	lea	RUFV_SIZE+4+92+8(a6),a2	;p  [1] a2=&Sa[02]
	add.l	d1,d2	;   [0] d2=L0x+Ax
	lea	104(a2),a3	;p  [1] a3=&Sb[02]
	move.l	d2,RUFV_L0XAX(a6)	;   [0] Set L0x+Ax
	add.l	#P2Q,d0	;p  [1] d0=Ax+P+2Q
	; ** STALL **
	move.l	d0,RUFV_AXP2Q(a6)	;p  [1] Set Ax+P+2Q
	; ** STALL **


	RUF_ALIGN	d1,_rc5_unit_func_060re	;Align to 8 bytes and pOEP
.ruf_mainloop:
	;A : d0 d4
	;L0: d1 d5
	;L1: d2 d6

	move.l	16(a0),d2	;p  [1] d2=L1x
	; ** STALL **
	move.l	(a1)+,d1	;p  [1] d1=L0a=RUFV_L0X
	move.l	d1,d5	;   [0] d5=L0b=L0a
	move.l	(a1)+,d3	;p  [1] d3=Ax+L0x=RUFV_L0XAX
	add.l	d3,d2	;   [0] d2=L1a=L1x+Ax+L0x
	move.l	(a1)+,d0	;p  [1] d0=Aa=Ax+P+2Q=RUFV_AXP2Q
	move.l	d2,d6	;   [0] d6=L1a
	add.l	(a1)+,d6	;p  [1] d6=L1b=L1a+1<<24, a1=P+3Q lookup
	rol.l	d3,d2	;   [0] d2=L1a=L1x<<<(Ax+L0x)
	move.l	d0,d4	;p  [1] d4=Ab=Aa
	rol.l	d3,d6	;   [0] d6=L1b=L1b<<<(Ab+L0b)
	; 6 cycles

	; Repeated round 1 even-odd-even-odd rounds

	add.l	d2,d0	;p  [1] d0=Aa=Aa+P+nQ+L1a
	add.l	d6,d4	;   [0] d4=Ab=Ab+P+nQ+L1b
	rol.l	#3,d0	;p  [1] d0=Aa=(Aa+P+nQ+L1a)<<<3
	move.l	d2,d3	;   [0] d3=L1a
	move.l	d0,(a2)+	;p  [1] Sa[02]=Aa, a2=&Sa[03]
	rol.l	#3,d4	;   [0] d4=Ab=(Ab+P+nQ+L1b)<<<3
	add.l	d0,d3	;p  [1] d3=Aa+L1a
	move.l	d6,d7	;   [0] d7=L1b
	add.l	d3,d1	;p  [1] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;   [0] Sb[02]=Ab, a3=&Sb[03]
	add.l	d4,d7	;p  [1] d7=Ab+L1b
	rol.l	d3,d1	;   [0] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;p  [1] d5=L0b=L0b+Ab+L1b
	add.l	(a1),d0	;   [0] Aa=Aa+P+nQ
	rol.l	d7,d5	;p  [1] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;   [0] Aa=Aa+P+nQ+L0a
	move.l	d1,d3	;p  [1] d3=L0a
	add.l	(a1)+,d4	;   [0] Ab=Ab+P+nQ, update P+nQ
	rol.l	#3,d0	;p  [1] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;s] [0] Ab=Ab+P+nQ+L0b
	rol.l	#3,d4	;p] [1] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d0,(a2)+	;   [0] Sa[03]=Aa, a2=&Sa[04]
	add.l	d0,d3	;p  [1] d3=Aa+L0a
	move.l	d5,d7	;   [0] d7=L0b
	add.l	d3,d2	;p  [1] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;   [0] Sb[03]=Ab, a3=&Sb[04]
	add.l	d4,d7	;p  [1] d7=Ab+L0b
	rol.l	d3,d2	;   [0] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;p  [1] L1b=L1b+Ab+L0b
	add.l	(a1),d0	;   [0] Aa=Aa+P+nQ
	; 15 cycles

	REPT 10
	move.l	d2,d3	;p  [1] d3=L1a
	rol.l	d7,d6	;   [0] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;p  [1] d0=Aa=Aa+P+nQ+L1a
	add.l	(a1)+,d4	;   [0] d4=Ab=Ab+P+nQ, update P+nQ
	rol.l	#3,d0	;p  [1] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;   [0] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;p  [1] d3=Aa+L1a
	move.l	d0,(a2)+	;   [0] Sa[n]=Aa
	rol.l	#3,d4	;p  [1] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;   [0] d7=L1b
	add.l	d3,d1	;p  [1] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;   [0] Sb[n]=Ab
	add.l	d4,d7	;p  [1] d7=Ab+L1b
	rol.l	d3,d1	;   [0] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;p  [1] d5=L0b=L0b+Ab+L1b
	add.l	(a1),d0	;   [0] Aa=Aa+P+nQ
	move.l	d1,d3	;p  [1] d3=L0a
	rol.l	d7,d5	;   [0] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;p  [1] Aa=Aa+P+nQ+L0a
	add.l	(a1)+,d4	;   [0] Ab=Ab+P+nQ, update P+nQ
	rol.l	#3,d0	;p  [1] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;   [0] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;p  [1] d3=Aa+L0a
	move.l	d0,(a2)+	;   [0] Sa[n]=Aa
	rol.l	#3,d4	;p  [1] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;   [0] d7=L0b
	add.l	d3,d2	;p  [1] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;   [0] Sb[n]=Ab
	add.l	d4,d7	;p  [1] d7=Ab+L0b
	rol.l	d3,d2	;   [0] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;p  [1] L1b=L1b+Ab+L0b
	add.l	(a1),d0	;   [0] d0=Aa=Aa+P+nQ
	ENDR
	; 16 cycles x 10 = 160 cycles

	; Final round 1 even-odd iteration

	move.l	d2,d3	;p  [1] d3=L1a
	rol.l	d7,d6	;   [0] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;p  [1] d0=Aa=Aa+P+nQ+L1a
	add.l	(a1)+,d4	;   [0] d4=Ab=Ab+P+nQ, update P+nQ
	rol.l	#3,d0	;p  [1] d0=Aa=(Aa+P+nQ+L1a)<<<3
	add.l	d6,d4	;   [0] d4=Ab=Ab+P+nQ+L1b
	add.l	d0,d3	;p  [1] d3=Aa+L1a
	move.l	d0,(a2)+	;   [0] Sa[24]=Aa
	rol.l	#3,d4	;p  [1] d4=Ab=(Ab+P+nQ+L1b)<<<3
	move.l	d6,d7	;   [0] d7=L1b
	add.l	d3,d1	;p  [1] d1=L0a=L0a+Aa+L1a
	move.l	d4,(a3)	;   [0] Sb[24]=Ab
	add.l	d4,d7	;p  [1] d7=Ab+L1b
	rol.l	d3,d1	;   [0] d1=L0a=L0a<<<(Aa+L1a)
	add.l	d7,d5	;p  [1] d5=L0b=L0b+Ab+L1b
	add.l	(a1),d0	;   [0] Aa=Aa+P+nQ
	move.l	d1,d3	;p  [1] d3=L0a
	rol.l	d7,d5	;   [0] d5=L0b=L0b<<<(Ab+L1b)
	add.l	d1,d0	;p  [1] Aa=Aa+P+nQ+L0a
	add.l	(a1)+,d4	;   [0] Ab=Ab+P+nQ, a1=&Sa[00]
	rol.l	#3,d0	;p  [1] Aa=(Aa+P+nQ+L0a)<<<3
	add.l	d5,d4	;   [0] Ab=Ab+P+nQ+L0b
	add.l	d0,d3	;p  [1] d3=Aa+L0a
	move.l	d0,(a2)	;   [0] Sa[25]=Aa
	rol.l	#3,d4	;p  [1] Ab=(Ab+P+nQ+L0b)<<<3
	move.l	d5,d7	;   [0] d7=L0b
	add.l	d3,d2	;p  [1] L1a=L1a+Aa+L0a
	add.l	d4,d7	;   [0] d7=Ab+L0b
	rol.l	d3,d2	;p  [1] L1a=L1a<<<(Aa+L0a)
	add.l	d7,d6	;   [0] L1b=L1b+Ab+L0b
	move.l	a1,a2	;p  [1] a2=Sa[] array
	rol.l	d7,d6	;   [0] L1b=L1b<<<(Ab+L0b)
	move.l	d4,a4	;p  [1] Sb[25]=Ab
	lea	104(a1),a3	;   [0] a3=Sb[] array
	; 17 cycles

	;---- Round 2 of key expansion ----
	; First iteration: Sx[00] is constant P0QR3
	;                  Sx[01] is constant RUFV_AX

	add.l	a5,d0	;p  [1] Aa=Aa+Sa[00]
	add.l	d6,d4	;   [0] Ab=Ab+Sb[00]+L1b
	add.l	d2,d0	;p  [1] Aa=Aa+Sa[00]+L1a
	add.l	a5,d4	;   [0] Ab=Ab+Sb[00]
	rol.l	#3,d0	;p  [1] Aa=(Aa+Sa[00]+L1a)<<<3
	move.l	d2,d3	;   [0] d3=L1a
	rol.l	#3,d4	;p  [1] Ab=(Ab+Sb[00]+L1b)<<<3
	move.l	d0,(a2)+	;   [0] Sa[00]=Aa, a2=&Sa[01]
	move.l	d6,d7	;p  [1] d7=L1b
	add.l	d0,d3	;   [0] d3=Aa+L1a
	add.l	d4,d7	;p  [1] d7=Ab+L1b
	add.l	d3,d1	;   [0] L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;p  [1] Sb[00]=Ab, a3=&Sb[01]
	add.l	d7,d5	;   [0] L0b=L0b+Ab+L1b
	rol.l	d3,d1	;p  [1] L0a=L0a<<<(Aa+L1a)
	rol.l	d7,d5	;   [0] L0b=L0b<<<(Ab+L1b)
	add.l	RUFV_AX(a6),d0	;p  [1] Aa=Aa+Sa[01]
	move.l	d1,d3	;   [0] d3=L0a
	add.l	d5,d4	;p  [1] Ab=Ab+Sb[01]+L0b
	add.l	d1,d0	;   [0] Aa=Aa+Sa[01]+L0a
	add.l	RUFV_AX(a6),d4	;p  [1] Ab=Ab+Sb[01]
	rol.l	#3,d0	;   [0] Aa=(Aa+Sa[01]+L0a)<<<3
	move.l	d5,d7	;p  [1] d7=L0b
	rol.l	#3,d4	;   [0] Ab=(Ab+Sb[01]+L0b)<<<3
	move.l	d0,(a2)+	;p  [1] Sa[01]=Aa, a2=&Sa[02]
	add.l	d0,d3	;   [0] d3=Aa+L0a
	add.l	d4,d7	;p  [1] d7=Ab+L0b
	add.l	d3,d2	;   [0] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;p  [1] Sb[01]=Ab, a3=&Sb[02]
	add.l	d7,d6	;   [0] L1b=L1b+Ab+L0b
	rol.l	d3,d2	;p  [1] L1a=L1a<<<(Aa+L0a)
	rol.l	d7,d6	;   [0] L1b=L1b<<<(Ab+L0b)
	add.l	(a2),d0	;p  [1] Aa=Aa+Sa[2]
	move.l	d2,d3	;   [0] d3=L1a
	; 17 cycles

	; Repeated round 2 even-odd-even-odd rounds

	REPT	11
	add.l	d6,d4	;p  [1] Ab=Ab+Sb[n]+L1b
	add.l	d2,d0	;   [0] Aa=Aa+Sa[n]+L1a
	add.l	(a3),d4	;p  [1] Ab=Ab+Sb[n]
	rol.l	#3,d0	;   [0] Aa=(Aa+Sa[n]+L1a)<<<3
	move.l	d6,d7	;p  [1] d7=L1b
	rol.l	#3,d4	;   [0] Ab=(Ab+Sb[n]+L1b)<<<3
	move.l	d0,(a2)+	;p  [1] Sa[n]=Aa
	add.l	d0,d3	;   [0] d3=Aa+L1a
	add.l	d4,d7	;p  [1] d7=Ab+L1b
	add.l	d3,d1	;   [0] L0a=L0a+Aa+L1a
	move.l	d4,(a3)+	;p  [1] Sb[n]=Ab
	add.l	d7,d5	;   [0] L0b=L0b+Ab+L1b
	rol.l	d3,d1	;p  [1] L0a=L0a<<<(Aa+L1a)
	rol.l	d7,d5	;   [0] L0b=L0b<<<(Ab+L1b)
	add.l	(a2),d0	;p  [1] Aa=Aa+Sa[n]
	move.l	d1,d3	;   [0] d3=L0a
	add.l	d5,d4	;p  [1] Ab=Ab+Sb[n]+L0b
	add.l	d1,d0	;   [0] Aa=Aa+Sa[n]+L0a
	add.l	(a3),d4	;p  [1] Ab=Ab+Sb[n]
	rol.l	#3,d0	;   [0] Aa=(Aa+Sa[n]+L0a)<<<3
	move.l	d5,d7	;p  [1] d7=L0b
	rol.l	#3,d4	;   [0] Ab=(Ab+Sb[n]+L0b)<<<3
	move.l	d0,(a2)+	;p  [1] Sa[n]=Aa
	add.l	d0,d3	;   [0] d3=Aa+L0a
	add.l	d4,d7	;p  [1] d7=Ab+L0b
	add.l	d3,d2	;   [0] L1a=L1a+Aa+L0a
	move.l	d4,(a3)+	;p  [1] Sb[n]=Ab
	add.l	d7,d6	;   [0] L1b=L1b+Ab+L0b
	rol.l	d3,d2	;p  [1] L1a=L1a<<<(Aa+L0a)
	rol.l	d7,d6	;   [0] L1b=L1b<<<(Ab+L0b)
	add.l	(a2),d0	;p  [1] Aa=Aa+Sa[24]
	move.l	d2,d3	;   [0] d3=L1a
	ENDR
	; 16 cycles x 11 = 176 cycles

	; Final round 2 even-odd iteration

	add.l	d6,d4	;p  [1] Ab=Ab+Sb[24]+L1b
	add.l	d2,d0	;   [0] Aa=Aa+Sa[24]+L1a
	add.l	(a3),d4	;p  [1] Ab=Ab+Sb[24]
	rol.l	#3,d0	;   [0] Aa=(Aa+Sa[24]+L1a)<<<3
	move.l	d6,d7	;p  [1] d7=L1b
	rol.l	#3,d4	;   [0] Ab=(Ab+Sb[24]+L1b)<<<3
	move.l	d0,(a2)+	;p  [1] Sa[24]=Aa, a2=&Sa[25]
	add.l	d0,d3	;   [0] d3=Aa+L1a
	add.l	d4,d7	;p  [1] d7=Ab+L1b
	add.l	d3,d1	;   [0] L0a=L0a+Aa+L1a
	move.l	d4,(a3)	;p  [1] Sb[24]=Ab
	add.l	d7,d5	;   [0] L0b=L0b+Ab+L1b
	rol.l	d3,d1	;p  [1] L0a=L0a<<<(Aa+L1a)
	rol.l	d7,d5	;   [0] L0b=L0b<<<(Ab+L1b)
	add.l	(a2),d0	;p  [1] Aa=Aa+Sa[25]
	add.l	d5,d4	;   [0] Ab=Ab+L0b
	add.l	d1,d0	;p  [1] Aa=Aa+Sa[25]L0a
	add.l	a4,d4	;   [0] Ab=Ab+L0b+Sb[25]
	rol.l	#3,d0	;p  [1] Aa=(Aa+Sa[25]+L0a)<<<3
	move.l	d1,d3	;   [0] d3=L0a
	rol.l	#3,d4	;p  [1] Ab=(Ab+Sb[25]+L0b)<<<3
	move.l	d5,d7	;   [0] d7=L0b
	add.l	d0,d3	;p  [1] d3=Aa+L0a
	add.l	d4,d7	;   [0] d7=Ab+L0b
	add.l	d3,d2	;p  [1] L1a=L1a+Aa+L0a
	add.l	(a1)+,d0	;   [0] d0=Aa=Sa[00]
	add.l	d7,d6	;p  [1] L1b=L1b+Ab+L0b
	rol.l	d3,d2	;   [0] L1a=L1a<<<(Aa+L0a)
	rol.l	d7,d6	;p  [1] L1b=L1b<<<(Ab+L0b)
	add.l	d2,d0	;   [0] d0=Aa=Sa[00]+L1a
	; 15 cycles

	;---- Combined round 3 of key expansion and encryption round ----

	move.l	d4,a4	;p  [1] a4=Ab (save Ab in a4)
	rol.l	#3,d0	;   [0] d0=Aa=(Sa[00]+L1a)<<<3
	move.l	d2,d3	;p  [1] d3=L1a
	move.l	4(a0),d4	;   [0] d4=eAa=plain.lo
	add.l	d0,d3	;p  [1] d3=Aa+L1a
	add.l	d0,d4	;   [0] d4=eAa=plain.lo+Aa
	add.l	d3,d1	;p  [1] d1=L0a=L0a+Aa+L1a
	lea	4(a1),a2	;   [0] a2=next iter &Sa[02]
	rol.l	d3,d1	;p  [1] d1=L0a=(L0a+Aa+L1a)<<<(Aa+L1a)
	add.l	(a1)+,d0	;   [0] d0=Aa=Aa+Sa[01], a1=&Sa[02]
	move.l	d1,d3	;p  [1] d3=L0a
	add.l	d1,d0	;   [0] d0=Aa=Aa+Sa[01]+L0a
	move.l	(a0),d7	;p  [1] d7=eBa=plain.hi
	rol.l	#3,d0	;   [0] d0=Aa=(Aa+Sa[01]+L0a)<<<3
	add.l	104-8(a1),a4	;p  [1] a4=Ab=Ab+L1b+Sb[00]
	add.l	d0,d3	;   [0] d3=Aa+L0a
	add.l	d0,d7	;p  [1] d7=eBa=plain.hi+Aa
	add.l	d3,d2	;   [0] d2=L1a=L1a+Aa+L0a
	add.l	(a1)+,d0	;p  [1] d0=Aa=Aa+Sa[02], a1=&Sa[03]
	rol.l	d3,d2	;   [0] d2=L1a=(L1a+Aa+L0a)<<<(Aa+L0a)
	eor.l	d7,d4	;p  [1] d4=eAa=eAa^eBa
	add.l	d2,d0	;   [0] d0=Aa=Aa+Sa[03]+L1a
	; 11 cycles

	; Repeated round 3 even-odd-even-odd rounds
	REPT	11
	rol.l	d7,d4	;p  [1] d4=eAa=((eAa^eBa)<<<eBa)
	rol.l	#3,d0	;   [0] d0=Aa=(Aa+Sa[n]+L1a)<<<3
	move.l	d2,d3	;p  [1] d3=L1a    \ no stall
	add.l	d0,d3	;   [0] d3=Aa+L1a / (move opt)
	add.l	d0,d4	;p  [1] d4=eAa=((eAa^eBa)<<<eBa)+Aa
	add.l	d3,d1	;   [0] d1=L0a=L0a+Aa+L1a
	add.l	(a1)+,d0	;p  [1] d0=Aa=Aa+Sa[n], a1=&Sa[n+1]
	rol.l	d3,d1	;   [0] d1=L0a=(L0a+Aa+L1a)<<<(Aa+L1a)
	eor.l	d4,d7	;p  [1] d7=eBa=eBa^eAa
	add.l	d1,d0	;   [0] d0=Aa=Aa+Sa[n]+L0a
	rol.l	d4,d7	;p  [1] d7=eBa=((eBa^eAa)<<<eAa)
	rol.l	#3,d0	;   [0] d0=Aa=(Aa+Sa[n]+L0a)<<<3
	move.l	d1,d3	;p  [1] d3=L0a    \ no stall
	add.l	d0,d3	;   [0] d3=Aa+L0a / (move opt)
	add.l	d0,d7	;p  [1] d7=eBa=((eBa^eAa)<<<eAa)+Aa
	add.l	d3,d2	;   [0] d2=L1a=L1a+Aa+L0a
	add.l	(a1)+,d0	;p  [1] d0=Aa=Aa+Sa[n], a1=&Sa[n+1]
	rol.l	d3,d2	;   [0] d2=L1a=(L1a+Aa+L0a)<<<(Aa+L0a)
	eor.l	d7,d4	;p  [1] d4=eAa=eAa^eBa
	add.l	d2,d0	;   [0] d0=Aa=Aa+Sa[24]+L1a
	ENDR
	;10 cycles x 11 = 110 cycles

	;-- Generate and check low 32 bits of result --
	rol.l	d7,d4	;p  [1] d4=eAa=((eAa^eBa)<<<eBa)
	rol.l	#3,d0	;   [0] d0=Aa=(Aa+Sa[24]+L1a)<<<3
	addq.l	#8,a1	;p  [1] a1=Sb[]
	add.l	d0,d4	;   [0] d4=eAa=((eAa^eBa)<<<eBa)+Aa
	add.l        d6,a4        	;p  [1] a4=Ab=Ab+L1b
	cmp.l	12(a0),d4	;   [0] eAa == cypher.lo?
	bne.s	.ruf_notfounda	;P  [0] Skip if not
	; 3 cycles

	;-- Low 32 bits match! (1 in 2^32 keys!) --
	; Need to completely re-check key but
	; generating high 32 bits as well...

	moveq	#0,d0
	jsr	_rc5_check64
	bne.s	.ruf_notfounda

	;---------------------------------------
	; 'Interesting' key found on pipeline 1

	move.l	(a7)+,d0	;d0=loop count
	move.l	(a7)+,d1	;d1=initial loop count
	sub.l	d0,d1	;Return number of keys checked
	add.l	d1,d1	; = loops*pipeline_count
	move.l	d1,d0

	lea	RUFV_SIZE+4+92+208(a7),a7

	movem.l	(a7)+,d2-7/a2-6
	rts

	CNOP	0,8
.ruf_notfounda:
	;-- Perform round 3 for 'b' key --

	move.l	a4,d0	;p  [1] d0=Ab=Ab+L1b
	move.l	d6,d3	;   [0] d3=L1b
	rol.l	#3,d0	;p  [1] d0=Ab=(Sb[00]+L1b)<<<3
	move.l	4(a0),d4	;   [0] d4=eAb=plain.lo
	add.l	d0,d3	;p  [1] d3=Ab+L1b
	add.l	d0,d4	;   [0] d4=eAb=plain.lo+Ab
	add.l	d3,d5	;p  [1] d5=L0b=L0b+Ab+L1b
	add.l	(a1)+,d0	;   [0] d0=Ab=Ab+Sb[01]
	rol.l	d3,d5	;p  [1] d5=L0b=(L0b+Ab+L1b)<<<(Ab+L1b)
	move.l	a1,a3	;   [0] a3=&Sb[02]
	add.l	d5,d0	;p  [1] d0=Ab=Ab+Sb[01]+L0b
	move.l	d5,d3	;   [0] d3=L0b
	rol.l	#3,d0	;p  [1] d0=Ab=(Ab+Sb[01]+L0b)<<<3
	move.l	(a0),d7	;   [0] d7=eBb=plain.hi
	add.l	d0,d3	;p  [1] d3=Ab+L0b
	add.l	d0,d7	;   [0] d7=eBb=plain.hi+Ab
	add.l	d3,d6	;p  [1] d6=L1b=L1b+Ab+L0b
	add.l	(a1)+,d0	;   [0] d0=Ab=Ab+Sb[01]
	rol.l	d3,d6	;p  [1] d6=L1b=(L1b+Ab+L0b)<<<(Ab+L0b)
	eor.l	d7,d4	;   [0] d4=eAb=eAb^eBb
	add.l	d6,d0	;p  [1] d0=Ab=Ab+Sb[n]+L1a
	rol.l	d7,d4	;   [0] d4=eAb=((eAb^eBb)<<<eBb)
	; 10 cycles

	; Repeated round 3 even-odd-even-odd rounds
	REPT	11
	rol.l	#3,d0	;p  [1] d0=Ab=(Ab+Sb[n]+L1a)<<<3
	move.l	d6,d3	;   [0] d3=L1a
	add.l	d0,d4	;p  [1] d4=eAb=((eAb^eBb)<<<eBb)+Ab
	add.l	d0,d3	;   [0] d3=Ab+L1b
	eor.l	d4,d7	;p  [1] d7=eBb=eBb^eAb
	add.l	d3,d5	;   [0] d5=L0b=L0b+Ab+L1b
	add.l	(a1)+,d0	;p  [1] d0=Ab=Ab+Sb[n]
	rol.l	d3,d5	;   [0] d5=L0b=(L0b+Ab+L1b)<<<(Ab+L1b)
	rol.l	d4,d7	;p  [1] d7=eBb=((eBb^eAb)<<<eAb)
	add.l	d5,d0	;   [0] d0=Ab=Ab+Sb[n]+L0b
	move.l	d5,d3	;p  [1] d3=L0b
	rol.l	#3,d0	;s] [0] d0=Ab=(Ab+Sb[n]+L0b)<<<3
	add.l	d0,d3	;p] [1] d3=Ab+L0b
	add.l	d0,d7	;   [0] d7=eBb=((eBb^eAb)<<<eAb)+Ab
	add.l	d3,d6	;p  [1] d6=L1b=L1b+Ab+L0b
	add.l	(a1)+,d0	;   [0] d0=Ab=Ab+Sb[n]
	rol.l	d3,d6	;p  [1] d6=L1b=L1b<<<(Ab+L0b)
	eor.l	d7,d4	;   [0] d4=eAb=eAb^eBb
	add.l	d6,d0	;p  [1] d0=Ab=Ab+Sb[n]+L1b
	rol.l	d7,d4	;   [0] d4=eAb=((eAb^eBb)<<<eBb)
	ENDR
	; 10 cycles x 11 = 110 cycles

	rol.l	#3,d0	;p  [1] d0=Ab=(Ab+Sb[24]+L1b)<<<3
	; ** STALL **
	add.l	d0,d4	;p  [1] d4=eAb=((eAb^eBb)<<<eBb)+Ab
	; ** STALL **
	cmp.l	12(a0),d4	;p  [1] eAb == cypher.lo?
	; ** STALL **
	bne.s	.ruf_notfoundb	;P  [0] Skip if not
	; 3 cycles

	;-- Low 32 bits match! (1 in 2^32 keys!) --
	; Need to completely re-check key but
	; generating high 32 bits as well...

	moveq	#1,d0
	jsr	_rc5_check64
	bne.s	.ruf_notfoundb

	;---------------------------------------
	; 'Interesting' key found on pipeline 2

	move.l	(a7)+,d0	;d0=loop count
	move.l	(a7)+,d1	;d1=initial loop count
	sub.l	d0,d1	;Return number of keys checked
	add.l	d1,d1	; = loops*pipeline_count+1
	addq.l	#1,d1
	addq.b	#1,16(a0)	;Adjust L0.hi for pipeline 2
	move.l	d1,d0
	lea	RUFV_SIZE+4+92+208(a7),a7
	movem.l	(a7)+,d2-7/a2-6
	rts

	CNOP	0,8
.ruf_notfoundb:
	;Mangle-increment current key

	addq.b	#2,16(a0)	;Increment L1 by pipeline count
	lea	RUFV_L0X(a6),a1
	bcc.s	.ruf_midone
	addq.b	#1,17(a0)	;Every 256^1 (256)
	bcc.s	.ruf_midone
	addq.b	#1,18(a0)	;Every 256^2 (65536)
	bcc.s	.ruf_midone
	addq.b	#1,19(a0)	;Every 256^3 (16777216)
	bcc.s	.ruf_midone

	addq.b	#1,20(a0)	;Every 256^4 (4294967296)
	bcc.s	.ruf_l0chg
	addq.b	#1,21(a0)	;Every 256^5 (1099511627776)
	bcc.s	.ruf_l0chg
	addq.b	#1,22(a0)	;Every 256^6 (281474976710656)
	bcc.s	.ruf_l0chg
	addq.b	#1,23(a0)	;Every 256^7 (72057594037927936)

	; Need to do anything special wrapping 0xff..f -> 0x00..0 ?

.ruf_l0chg:	; L0 has changed so recalculate "constants"

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

	RUF_ALIGN	d1,_rc5_unit_func_060re	;Align to 8 bytes and pOEP
.ruf_midone:	subq.l	#1,(a7)	;Loop back for next key
	bne	.ruf_mainloop

	;---------------------------------
	; Key not found on either pipeline
	; Return iterations*pipeline_count

	addq.l	#4,a7	;Forget loop counter
	move.l	(a7)+,d0	;d0=initial loop counter
	add.l	d0,d0
	lea	RUFV_SIZE+4+92+208(a7),a7
	movem.l	(a7)+,d2-7/a2-6
	rts

;--------------------
