
;	OPT	O+,W-
	XDEF	@ruf_mu

;--------------------

	; Define P, Q

P:	equ	$b7e15163
Q:	equ	$9e3779b9
P0QR3:	equ	$bf0a8b1d	;(0*Q + P) <<< 3

KEYCNT:      equ          256          ;Must match with PIPELINE_COUNT

;--------------------

	;68k RC5 core
	;Many keys, unrolled loops, 040/060 optimisations
	;Based on actual client core
	;
	;Entry: a0=rc5unitwork structure:
	;	 0(a0) = plain.hi	- plaintext
	;	 4(a0) = plain.lo
	;	 8(a0) = cypher.hi	- cyphertext
	;	12(a0) = cypher.lo
	;	16(a0) = L0.hi	- key
	;	20(a0) = L0.lo
	;
	;Exit:  d0=return code

	SECTION rc564,CODE

@ruf_mu:	movem.l	d2-d7/a2-a6,-(a7)

	move.l	#KEYCNT-1,d7	;d7=number of keys
	move.l	#P0QR3+P+Q,a2
	move.l	#P0QR3,a3
	move.l	#Q,a4
	move.l	16(a0),a5	;a5=L1=L0.hi (increments)
	move.l	20(a0),a6	;a6=L0=L0.lo	(constant)

.ruf_mainloop:
	move.l	a6,d1	;d1=L0=L0.lo
	move.l	a2,d0	;d0=P<<<3+P+Q
	add.l	a3,d1	;d1=L0=L0.lo+(P<<<3)
	move.l	a5,d2	;d2=L1
	ror.l	#3,d1	;d1=(L0.lo+(P<<<3)>>>3)

	lea	-$0064(a7),a1	;a1=&S[01]
	add.l	d1,d0
	move.l	a4,d5
	rol.l	#3,d0
	move.l	#P+Q+Q,d4
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2

	add.l	d4,d0
	rol.l	d3,d2
	add.l	d5,d4
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1

	add.l	d4,d0
	rol.l	d3,d1
	add.l	d5,d4
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)
	add.l	d3,d2

	add.l	#P0QR3,d0
	rol.l	d3,d2
	add.l	d2,d0
	lea	-$0068(a7),a1
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)+
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d2,d3
	move.l	d0,(a1)+
	add.l	d3,d1
	add.l	(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	move.l	d0,(a1)
	add.l	d3,d2
	add.l	-$0068(a7),d0
	rol.l	d3,d2
	add.l	d2,d0
	move.l	4(a0),d4
	rol.l	#3,d0
	lea	-$0064(a7),a1
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	move.l	(a0),d5
	rol.l	#3,d0
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1)+,d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	d0,d3
	add.l	d2,d3
	add.l	d0,d4
	add.l	d3,d1
	add.l	(a1)+,d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	d0,d3
	add.l	d1,d3
	add.l	d0,d5
	add.l	d3,d2
	add.l	(a1),d0
	rol.l	d3,d2
	add.l	d2,d0
	eor.l	d5,d4
	rol.l	#3,d0
	rol.l	d5,d4
	move.l	12(a0),d3
	add.l	d0,d4
	cmp.l	d3,d4
	bne.s	.ruf_next

	move.l	d0,d3
	add.l	d2,d3
	add.l	d3,d1
	add.l	4(a1),d0
	rol.l	d3,d1
	add.l	d1,d0
	eor.l	d4,d5
	rol.l	#3,d0
	rol.l	d4,d5
	move.l	8(a0),d3
	add.l	d0,d5
	cmp.l	d3,d5
	bne.s	.ruf_next

	; Found a possible key!

	move.l	#KEYCNT,d0	;Get d0="pipeline" number
	sub.l	d7,d0
	movem.l	(a7)+,d2-d7/a2-a6
	rts

.ruf_next:	add.l	#$01000000,a5	;Loop for next key
	dbf	d7,.ruf_mainloop

	moveq	#0,d0
	movem.l	(a7)+,d2-d7/a2-a6
	rts


