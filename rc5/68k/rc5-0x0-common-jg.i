
	; $VER: MC680x0 RC5 core common definitions 04-Feb-2001

             ;
	; MC680x0 RC5 core common definitions
	; for distributed.net RC5-64 clients.
	;
	; Written by John Girvin <girv@girvnet.org.uk>
	;

;--------------------

	RSRESET
	;Offsets into .ruf_vars
RUFV_AX:	rs.l	1	;A       \-- only change when
RUFV_L0X:	rs.l	1	;L0       |  L0 changes
RUFV_L0XAX:	rs.l	1	;L0+A     |
RUFV_AXP2Q:	rs.l	1	;A+P+2Q  /
RUFV_SIZE:	equ	__RS

;--------------------

	;Define P, Q and related constants

P:	equ	$b7e15163	;P
Q:	equ	$9e3779b9	;Q
P0QR3:	equ	$bf0a8b1d	;(0*Q + P) <<< 3
PR3Q:	equ	$15235639	;P0QR3+P+Q
P2Q:	equ	$f45044d5	;(2*Q + P)

;--------------------

	;
	; Ensure next instruction is aligned on
	; quadword boundary and runs in pOEP.
	;
	; \1 is scratch data register
	; \2 is label to align relative to
	;

RUF_ALIGN:	MACRO
	;BEGIN RUF_ALIGN
	  IFNE NARG-2
	    FAIL "RUF_ALIGN: usage"
	    MEXIT
	  ENDC
.align\@:	  equ	*-\2
	  IFNE	.align\@&7
	    swap	\1
	    RUF_ALIGN \1,\2
	  ENDC
	;END RUF_ALIGN
	ENDM

;--------------------
