
	| $VER: MC680x0 RC5 core common definitions 04-Feb-2001

	|
	| MC680x0 RC5 core common definitions
	| for distributed.net RC5-64 clients.
	|
	| Written by John Girvin <girv@girvnet.org.uk>
	|
	| Converted from Amiga Devpac assembler notation to GAS
	| notation by Oliver Roberts <oliver@futuara.co.uk>
	|

|--------------------

	|Offsets into .ruf_vars
.equ RUFV_AX,		0	|A       \-- only change when
.equ RUFV_L0X,		4	|L0       |  L0 changes
.equ RUFV_L0XAX,	8	|L0+A     |
.equ RUFV_AXP2Q,	12	|A+P+2Q  /
.equ RUFV_SIZE,		16

|--------------------

	|Define P, Q and related constants

.equ P,		0xb7e15163	|P
.equ Q,		0x9e3779b9	|Q
.equ P0QR3,	0xbf0a8b1d	|(0*Q + P) <<< 3
.equ PR3Q,	0x15235639	|P0QR3+P+Q
.equ P2Q,	0xf45044d5	|(2*Q + P)

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
