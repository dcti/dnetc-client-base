| Copyright distributed.net 1997-2003 - All Rights Reserved
| For use in distributed.net projects only.
| Any other distribution or use of this source violates copyright.

| $VER: MC680x0 RC5 core common definitions 04-Feb-2001
| $Id: r72-0x0-common-mh.ns.i,v 1.1.2.2 2004/05/15 08:31:09 mweiser Exp $

| MC680x0 RC5 core common definitions
| for distributed.net RC5-64 clients.
|
| Written by John Girvin <girv@girvnet.org.uk>
| Adapted to funny NeXTstep assembler syntax by
|  Michael Weiser <michael@weiser.dinsnail.net>

| --------------------
| Constants
|
| These values only change when some of the key words change
| Therefore they can be precalculated and re-used 256 or 256^5 times

| AX, L0X, L0XAX and AXP2Q only depend on L0
| L1X, AXP2QR, L1P2QR and P32QR depend on L0 and L1

| The order they occur in memory depends on when they are needed
| by the core

.data
        .align 2
| Offsets into .ruf_vars
RUFV_LC         = 0     | Loop count
RUFV_ILC        = 4     | Initial loop count address

| Only used for re-initialising some L1-constants
RUFV_L0XAX      = 8     | L0+A
RUFV_AXP2Q      = 12    | A+P+2Q

| Used in round 2 (in this order!)
RUFV_AX         = 16    | A
RUFV_AXP2QR     = 20    | (AXP2Q + L1X) <<< 3

| Used in round 1 (in this order!)
RUFV_L1P2QR     = 24    | L1X + AXP2QR
RUFV_P32QR      = 28    | P+3Q+L1P2QR
RUFV_L1X        = 32	| L1
RUFV_L0X        = 36	| L0

RUFV_SIZE       = 40

| RC5_72UnitWork structure
plain_hi        = 0
plain_lo        = 4
cypher_hi       = 8
cypher_lo       = 12
L0_hi           = 16
L0_mid          = 20
L0_lo           = 24
check_count     = 28
check_hi        = 32
check_mid       = 36
check_lo        = 40

| Define P, Q and related constants
P       = 0xb7e15163
Q       = 0x9e3779b9
P0QR3   = 0xbf0a8b1d    | (0*Q + P) <<< 3
PR3Q    = 0x15235639    | P0QR3+P+Q
P2Q     = 0xf45044d5    | (2*Q + P)
P3Q     = 0x9287be8e    | (3*Q + P)

|
| Ensure next instruction is aligned on
| quadword boundary and runs in pOEP.
|
| \1 is scratch data register
| \2 is label to align relative to
|
|.macro RUF_ALIGN
|.if . & 7
|            swap        $0
|            RUF_ALIGN   $0
|.endif
|.endmacro
