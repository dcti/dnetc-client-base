;--------------------
;@(#)$Id: rc5-0x0-jg.i,v 1.1.2.1 2000/06/05 15:25:29 oliver Exp $

	RSRESET
	;Offsets into ruf???_vars

RUFV_AX:	rs.l	1	;A       \-- only change when (if?)
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
	; quadword (8 byte) boundary
	;
	; \1 is base label to align respective to
	; \2 is scratch data register, will be = 0
	;

RUF_ALIGN:	MACRO
	;BEGIN RUF_ALIGN
	  IFNE NARG-2
	    FAIL "RUF_ALIGN: usage"
	    MEXIT
	  ENDC
\1align\@:	  equ	*-\1
	  IFNE	\1align\@&7
	    moveq	#0,\2
	    RUF_ALIGN \1,\2
	  ENDC
	;END RUF_ALIGN
	ENDM

;--------------------
