
	| Copyright distributed.net 1997-2003 - All Rights Reserved
	| For use in distributed.net projects only.
	| Any other distribution or use of this source violates copyright.
	|
        | $VER: MC680x0 RC5 core common definitions 04-Feb-2001
	|
        | MC680x0 RC5 core common definitions
        | for distributed.net RC5-64 clients.
        |
        | Written by John Girvin <girv@girvnet.org.uk>
        |
	| Converted from Amiga Devpac assembler notation to GAS
	| notation by Oliver Roberts <oliver@futaura.co.uk>
	|
	| $Id: r72-0x0-common-mh.gas.i,v 1.1.2.1 2003/04/03 22:18:21 oliver Exp $
	|
	| $Log: r72-0x0-common-mh.gas.i,v $
	| Revision 1.1.2.1  2003/04/03 22:18:21  oliver
	| gcc/gas compilable versions of all the 68k optimized cores
	|
	|

|--------------------
        |Constants
        |
        |These values only change when some of the key words change
        |Therefore they can be precalculated and re-used 256 or 256^5 times

        |AX, L0X, L0XAX and AXP2Q only depend on L0
        |L1X, AXP2QR, L1P2QR and P32QR depend on L0 and L1

        |The order they occur in memory depends on when they are needed
        |by the core

        |Offsets into .ruf_vars
.equ RUFV_LC,		0	|Loop count
.equ RUFV_ILC,		4	|Initial loop count address

|Only used for re-initialising some L1-constants
.equ RUFV_L0XAX,	8	|L0+A
.equ RUFV_AXP2Q,	12	|A+P+2Q

|Used in round 2 (in this order!)
.equ RUFV_AX,		16	|A
.equ RUFV_AXP2QR,	20	|(AXP2Q + L1X) <<< 3

|Used in round 1 (in this order!)
.equ RUFV_L1P2QR,	24	|L1X + AXP2QR
.equ RUFV_P32QR,	28	|P+3Q+L1P2QR
.equ RUFV_L1X,		32	|L1
.equ RUFV_L0X,		36	|L0

.equ RUFV_SIZE,		40

|--------------------

        | RC5_72UnitWork structure
.equ plain_hi,		0
.equ plain_lo,		4
.equ cypher_hi,		8
.equ cypher_lo,		12
.equ L0_hi,		16
.equ L0_mid,		20
.equ L0_lo,		24
.equ check_count,	28
.equ check_hi,		32
.equ check_mid,		36
.equ check_lo,		40

|--------------------

        |Define P, Q and related constants

.equ P,     0xb7e15163       |P
.equ Q,     0x9e3779b9       |Q
.equ P0QR3, 0xbf0a8b1d       |(0*Q + P) <<< 3
.equ PR3Q,  0x15235639       |P0QR3+P+Q
.equ P2Q,   0xf45044d5       |(2*Q + P)
.equ P3Q,   0x9287be8e       |(3*Q + P)

|--------------------

        |
        | Ensure next instruction is aligned on
        | quadword boundary and runs in pOEP.
        |
        | \1 is scratch data register
        | \2 is label to align relative to
        |

.macro RUF_ALIGN dregnum
.balignw 8,(0x4840+\dregnum)		| pad with SWAP dn instructions
.endm

|--------------------

.macro CNOP offset,alignment
  .balign \alignment
.endm

|--------------------
