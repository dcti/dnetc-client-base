
	| $VER: MC680x0 RC5 core common functions 04-Feb-2001

	|
	| MC680x0 RC5 core common functions
	| for distributed.net RC5-64 clients.
	|
	| Written by John Girvin <girv@girvnet.org.uk>
	|
	| Converted from Amiga Devpac assembler notation to GAS
	| notation by Oliver Roberts <oliver@futuara.co.uk>
	|

|--------------------

	.globl		_rc5_check64

	.include	"rc5/68k/rc5-0x0-common-jg.gcc.i"

|--------------------


	| Completely (all 64 bits) check a single RC5-64 key.
	|
	| Entry:	a0=rc5unitwork structure
	|	 0(a0) = plain.hi  - plaintext
	|	 4(a0) = plain.lo
	|	 8(a0) = cypher.hi - cyphertext
	|	12(a0) = cypher.lo
	|	16(a0) = L0.hi     - key
	|	20(a0) = L0.lo
	|        d0=pipeline number 0/1
	|
	| Exit:	d0=0 if key is correct, !=0 otherwise
	|	Other registers preserved.
	|
	| NOTES:
	|   Speed not important since this will be called on
	|   average once for every 2^32 keys checked by the
	|   main core. Instead, keep the code-size small so
	|   as to not flush the main core from the i-cache.
	|

	CNOP	0,8
_rc5_check64:
	movem.l	d2-d3/d6-d7/a1-a3,-(a7)

	|---- Initialise S[] array ----

	lea -208(a7),a7 |a7=Sx[] storage
	move.l  a7,a1   |a1=Sx[] storage

	|---- Mix secret key into S[] ----

	move.l	20(a0),d1	|d1=L0=L0.lo
	lea	.c64_p2q(pc),a3	|a3=&S[02] read address
	ror.l	#8,d0	|d0=pipeline << 24
	move.l	16(a0),d2	|d2=L1=L0.hi
	move.l	a1,a2	|a2=&S[00] write address
	add.l	d0,d2	|Adjust L0 for pipeline

	|First iteration special case

	move.l	#P0QR3,d0	|d0=A=P<<<3

	move.l	d0,(a2)+	|S[00]=P<<<3
	add.l	d0,d1	|d1=L0=L0+A
	rol.l	d0,d1	|d1=L0=(L0+A)<<<A

	add.l	#P+Q,d0	|A=A+P+Q
	add.l	d1,d0	|A=A+P+Q+L0
	rol.l	#3,d0	|A=(A+P+Q+L0)<<<3
	move.l	d0,(a2)+	|S[01]=A

	| Begin proper iterations now we have initial L0 and L1

	moveq	#3-1,d7	|d7=outer loop counter
	moveq	#13-1-1,d6	|d6=initial inner loop counter
	bra.s	.c64_mixkey2

.c64_mixkey1:
	move.l	a1,a2	|a2=S[] storage
	move.l	a1,a3	|a3=S[] storage
	moveq	#13-1,d6	|d6=inner loop counter

.c64_mixkey2:
	|d0=A d1=L1 d2=L2 a2=&S[]

	move.l	d1,d3	|d3=L0
	add.l	d0,d3	|d3=A+L0
	add.l	d3,d2	|L1=L1+A+L0

	add.l	(a3)+,d0	|A=A+S[n]
	rol.l	d3,d2	|L1=L1<<<(A+L0)
	add.l	d2,d0	|A=A+S[n]+L1
	rol.l	#3,d0	|A=(A+S[n]+L1)<<<3
	move.l	d2,d3	|d3=L1
	move.l	d0,(a2)+	|S[n]=A

	add.l	d0,d3	|d3=A+L1
	add.l	d3,d1	|L0=L0+A+L1
	add.l	(a3)+,d0	|A=A+S[n]
	rol.l	d3,d1	|L0=L0<<<(A+L1)

	add.l	d1,d0	|A=A+S[n]+L0
	rol.l	#3,d0	|A=(A+S[n]+L0)<<<3
	move.l	d0,(a2)+	|S[n]=A

	dbf	d6,.c64_mixkey2
	dbf	d7,.c64_mixkey1

	|---- Perform the encryption ----

	move.l	(a1)+,d0	|d0=A=S[00]
	add.l	4(a0),d0	|d0=A=S[00]+plain.lo

	move.l	(a1)+,d1	|d1=B=S[01]
	moveq	#5-1,d7	|d7=loop counter
	add.l	(a0),d1	|d1=B=S[01]+plain.hi

	eor.l	d1,d0	|d0=A=A^B
	rol.l	d1,d0	|d0=A=A<<<B
	add.l	(a1)+,d0	|d0=A=A+S[n]

	eor.l	d0,d1	|d1=B=B^A
	rol.l	d0,d1	|d1=B=B<<<A
	add.l	(a1)+,d1	|d1=B=B+S[n]

.c64_encrypt:
	.REPT	2
	eor.l	d1,d0	|d0=A=A^B
	rol.l	d1,d0	|d0=A=A<<<B
	add.l	(a1)+,d0	|d0=A=A+S[n]

	eor.l	d0,d1	|d1=B=B^A
	rol.l	d0,d1	|d1=B=B<<<A
	add.l	(a1)+,d1	|d1=B=B+S[n]
	.ENDR

	dbf	d7,.c64_encrypt

	eor.l	d1,d0	|d0=A=A^B
	move.l	12(a0),d2	|d2=cypher.lo
	rol.l	d1,d0	|d0=A=A<<<B
	add.l	(a1),d0	|d0=A=A+S[24]

	eor.l	d0,d1	|d1=B=B^A
	move.l	8(a0),d2	|d2=cypher.hi
	rol.l	d0,d1	|d1=B=B<<<A
	add.l	4(a1),d1	|d1=B=B+S[25]
	cmp.l	d2,d1	|B=cypher.hi?
	beq.s	.c64_found

	moveq	#1,d0	|Didn't find the key this time...
	bra.s	.c64_done

.c64_found:	moveq	#0,d0	|Found the key!

.c64_done:
	lea 208(a7),a7  |forget Sx[] storage
	movem.l	(a7)+,d2-d3/d6-d7/a1-a3
	tst.l	d0
	rts

|----------

	| Data for 64 bit check core

	CNOP	0,8
.c64_p2q:	|Table of P+nQ values for 2<=n<=26
	dc.l	0xF45044D5,0x9287BE8E
	dc.l	0x30BF3847,0xCEF6B200
	dc.l	0x6D2E2BB9,0x0B65A572
	dc.l	0xA99D1F2B,0x47D498E4
	dc.l	0xE60C129D,0x84438C56
	dc.l	0x227B060F,0xC0B27FC8
	dc.l	0x5EE9F981,0xFD21733A
	dc.l	0x9B58ECF3,0x399066AC
	dc.l	0xD7C7E065,0x75FF5A1E
	dc.l	0x1436D3D7,0xB26E4D90
	dc.l	0x50A5C749,0xEEDD4102
	dc.l	0x8D14BABB,0x2B4C3474

|--------------------
