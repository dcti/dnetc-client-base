
	; Copyright distributed.net 1997-2003 - All Rights Reserved
	; For use in distributed.net projects only.
	; Any other distribution or use of this source violates copyright.

        ; $VER: MC680x0 RC5 core common definitions 04-Feb-2001
	; $Id: r72-0x0-common-mh.i,v 1.2 2003/09/12 23:08:52 mweiser Exp $

        ; MC680x0 RC5 core common definitions
        ; for distributed.net RC5-64 clients.
        ;
        ; Written by John Girvin <girv@girvnet.org.uk>
        ;

;--------------------
        ;Constants
        ;
        ;These values only change when some of the key words change
        ;Therefore they can be precalculated and re-used 256 or 256^5 times

        ;AX, L0X, L0XAX and AXP2Q only depend on L0
        ;L1X, AXP2QR, L1P2QR and P32QR depend on L0 and L1

        ;The order they occur in memory depends on when they are needed
        ;by the core

        RSRESET
        ;Offsets into .ruf_vars
RUFV_LC:        rs.l    1       ;Loop count
RUFV_ILC:       rs.l    1       ;Initial loop count address

;Only used for re-initialising some L1-constants
RUFV_L0XAX:     rs.l    1       ;L0+A
RUFV_AXP2Q:     rs.l    1       ;A+P+2Q

;Used in round 2 (in this order!)
RUFV_AX:        rs.l    1       ;A
RUFV_AXP2QR:    rs.l    1       ;(AXP2Q + L1X) <<< 3

;Used in round 1 (in this order!)
RUFV_L1P2QR:    rs.l    1       ;L1X + AXP2QR
RUFV_P32QR:     rs.l    1       ;P+3Q+L1P2QR
RUFV_L1X:       rs.l    1       ;L1
RUFV_L0X:       rs.l    1       ;L0

RUFV_SIZE:      equ     __RS

;--------------------

        RSRESET
        ; RC5_72UnitWork structure
plain_hi:       rs.l    1
plain_lo:       rs.l    1
cypher_hi:      rs.l    1
cypher_lo:      rs.l    1
L0_hi:          rs.l    1
L0_mid:         rs.l    1
L0_lo:          rs.l    1
check_count     rs.l    1
check_hi        rs.l    1
check_mid       rs.l    1
check_lo        rs.l    1

;--------------------

        ;Define P, Q and related constants

P:      equ     $b7e15163       ;P
Q:      equ     $9e3779b9       ;Q
P0QR3:  equ     $bf0a8b1d       ;(0*Q + P) <<< 3
PR3Q:   equ     $15235639       ;P0QR3+P+Q
P2Q:    equ     $f45044d5       ;(2*Q + P)
P3Q:    equ     $9287be8e       ;(3*Q + P)

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
