
	; $VER: MC680x0 RC5 core common definitions 04-Feb-2001

    ;
	; MC680x0 RC5 core common definitions
	; for distributed.net RC5-64 clients.
	;
	; Written by John Girvin <girv@girvnet.org.uk>
	;
	; Converted from Amiga Devpac assembler to MPW Asm
	; notation by Michael Feiri <michael@feiri.de>

;--------------------

	;Offsets into .ruf_vars
RUFV_AX:	equ 0   ;A       \-- only change when
RUFV_L0X:	equ 4   ;L0       |  L0 changes
RUFV_L0XAX:	equ 8   ;L0+A     |
RUFV_AXP2Q:	equ 12  ;A+P+2Q  /
RUFV_SIZE:	equ 16

;--------------------

	;Define P, Q and related constants

P:	    equ	$b7e15163
Q:	    equ	$9e3779b9	;Q
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

;	MACRO
; 	RUF_ALIGN &1,&2
;
;	@ALIGN:	equ	*-&2
;
;	IF	'@ALIGN&&7'<>'0' THEN
;		swap	&1
;		RUF_ALIGN &1,&2
;	ENDIF
;	
;	;WHILE	'@ALIGN&&7'<>'0' DO
;	;	swap	&1
;	;	;GOTO .lab1 ;RUF_ALIGN &1,&2
;	;ENDWHILE
;
;	ENDM

;--------------------

	MACRO
	CNOP &OFFSET,&ALIGNMENT
		ALIGN &ALIGNMENT
	ENDM

;--------------------
