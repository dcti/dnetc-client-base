| Copyright distributed.net 1997-2003 - All Rights Reserved
| For use in distributed.net projects only.
| Any other distribution or use of this source violates copyright.

| $VER: MC680x0 RC5 core common functions 04-Feb-2001
| $Id: r72-0x0-common-mh.ns.s,v 1.1.2.1 2003/08/09 12:25:27 mweiser Exp $

|
| MC680x0 RC5 core common functions
| for distributed.net RC5-64 clients.
|
| Written by John Girvin <girv@girvnet.org.uk>
| Adapted to RC5-72 by Malcolm Howell <coreblimey@rottingscorpion.com>
| Adapted to funny NeXTstep assembler syntax by
|  Michael Weiser <michael@weiser.saale-net.de>

| Completely (all 64 bits) check a single RC5-72 key.
|
| Entry:        a0=rc5unitwork structure
|        0(a0) = plain.hi  - plaintext
|        4(a0) = plain.lo
|        8(a0) = cypher.hi - cyphertext
|       12(a0) = cypher.lo
|       16(a0) = L0.hi     - key
|       20(a0) = L0.mid
|       24(a0) = L0.lo
|        d0=pipeline number 0/1
|
| Exit: d0=0 if key is correct, !=0 otherwise
|       Other registers preserved.
|
| NOTES:
|   Speed not important since this will be called on
|   average once for every 2^32 keys checked by the
|   main core. Instead, keep the code-size small so
|   as to not flush the main core from the i-cache.
|

.include "r72-0x0-common-mh.ns.i"

WS      =       312

.globl _rc5_check64

.align 3                        | align to 8 byte boundary
_rc5_check64:
        moveml d2-d4/d6-d7/a1-a3,a7@-

        |---- Initialise S[] array ----

        lea     a7@(-WS),a7     | a7=Sx[] storage - three copies of table
        movel   a7,a1           | a1=Sx[] storage

        |---- Mix secret key into S[] ----
        movel   a0@(L0_hi),d3   | d3=L2=L0.hi
        lea     .c64_p2q,a3     | a3=&S[02] read address
        movel   a0@(L0_mid),d2  | d2=L1=L0.mid
        movel   a1,a2           | a2=&S[00] write address
        movel   a0@(L0_lo),d1   | d1=L0=L0.lo
        
        addb    d0,d3           | Adjust key byte in L2 for pipeline

        | First iteration special case

        movel   #P0QR3,d0       | d0=A=P<<<3
        movel   d0,a2@+         | S[00]=P<<<3

        addl    d0,d1           | d1=L0=L0+A
        roll    d0,d1           | d1=L0=(L0+A)<<<A

        addl    #P+Q,d0         | A=A+P+Q
        addl    d1,d0           | A=A+P+Q+L0
        roll    #3,d0           | A=(A+P+Q+L0)<<<3
        movel   d0,a2@+         | S[01]=A

        movel   d1,d4           | d4=L0
        addl    d0,d4           | d4=A+L0
        addl    d4,d2           | L1=L1+A+L0
        roll    d4,d2           | L1=L1<<<(A+L0)

        | Begin proper iterations now we have initial L0 and L1

        moveq   #2-1,d7         | d7=outer loop counter
        moveq   #8-1,d6         | d6=initial inner loop counter
        bras    .c64_mixkey2

.c64_mixkey1:
        | When arriving back here, a3 has just finished passing over the
        | P+nQ lookup table. Reset it to the start of the just-written S

        movel   a1,a3           | a3=S[] storage
        moveq   #17-1,d6        | d6=inner loop counter

.c64_mixkey2:
        | d0=A d1=L0 d2=L1 d3=L2 a2=&S[] writes a3=&S[] reads

        addl    a3@+,d0 | A=A+S[n]
        addl    d2,d0   | A=A+S[n]+L1
        roll    #3,d0   | A=(A+S[n]+L1)<<<3
        movel   d2,d4   | d4=L1
        movel   d0,a2@+ | S[n]=A

        addl    d0,d4   | d4=A+L1
        addl    d4,d3   | L2=L2+A+L1
        addl    a3@+,d0 | A=A+S[n]
        roll    d4,d3   | L2=L2<<<(A+L1)

        addl    d3,d0   | A=A+S[n]+L2
        roll    #3,d0   | A=(A+S[n]+L2)<<<3
        movel   d3,d4   | d4=L2
        movel   d0,a2@+ | S[n]=A

        addl    d0,d4   | d4=A+L2
        addl    d4,d1   | L0=L0+A+L2
        addl    a3@+,d0 | A=A+S[n]
        roll    d4,d1   | L0=L0<<<(A+L2)

        addl    d1,d0   | A=A+S[n]+L0
        roll    #3,d0   | A=(A+S[n]+L0)<<<3
        movel   d0,a2@+ | S[n]=A

        movel   d1,d4   | d4=L0
        addl    d0,d4   | d4=A+L0
        addl    d4,d2   | L1=L1+A+L0
        roll    d4,d2   | L1=L1<<<(A+L0)

        dbf     d6,.c64_mixkey2
        dbf     d7,.c64_mixkey1

        |  Finish up key expansion manually

        addl    a3@+,d0         | A=A+S[25]
        addl    d2,d0           | A=A+S[25]+L1
        roll    #3,d0           | A=A<<<3
        movel   d0,a2@          | S[25]=A

        | ---- Perform the encryption ----

        lea     a1@(208),a1             | a1=S[] (third copy of table)
        movel   a1@+,d0                 | d0=A=S[00]
        addl    a0@(plain_lo),d0        | d0=A=S[00]+plain.lo

        movel   a1@+,d1 | d1=B=S[01]
        moveq   #5-1,d7 | d7=loop counter
        addl    a0@,d1 | d1=B=S[01]+plain.hi

        eorl    d1,d0   | d0=A=A^B
        roll    d1,d0   | d0=A=A<<<B
        addl    a1@+,d0 | d0=A=A+S[n]

        eorl    d0,d1   | d1=B=B^A
        roll    d0,d1   | d1=B=B<<<A
        addl    a1@+,d1 | d1=B=B+S[n]

.c64_encrypt:
.macro c64_encrypt_REPT1
        eorl    d1,d0   | d0=A=A^B
        roll    d1,d0   | d0=A=A<<<B
        addl    a1@+,d0 | d0=A=A+S[n]

        eorl    d0,d1   | d1=B=B^A
        roll    d0,d1   | d1=B=B<<<A
        addl    a1@+,d1 | d1=B=B+S[n]
.endmacro

        c64_encrypt_REPT1
        c64_encrypt_REPT1

        dbf     d7,.c64_encrypt

        eorl    d1,d0                   | d0=A=A^B
        movel   a0@(cypher_lo),d2       | d2=cypher.lo
        roll    d1,d0                   | d0=A=A<<<B
        addl    a1@,d0                  | d0=A=A+S[24]

        eorl    d0,d1                   | d1=B=B^A
        movel   a0@(cypher_hi),d2       | d2=cypher.hi
        roll    d0,d1                   | d1=B=B<<<A
        addl    a1@(4),d1               | d1=B=B+S[25]
        cmpl    d2,d1                   | B=cypher.hi?
        beqs    .c64_found

        moveq   #1,d0           | Didn't find the key this time...
        bras    .c64_done

.c64_found:     moveq   #0,d0   | Found the key!

.c64_done:
        lea a7@(WS),a7  | forget Sx[] storage
        moveml a7@+,d2-d4/d6-d7/a1-a3
        tstl    d0
        rts

|----------

        | Data for 64 bit check core

.align 3
.c64_p2q:       | Table of P+nQ values for 2<=n<=25
        .long   0xF45044D5,0x9287BE8E
        .long   0x30BF3847,0xCEF6B200
        .long   0x6D2E2BB9,0x0B65A572
        .long   0xA99D1F2B,0x47D498E4
        .long   0xE60C129D,0x84438C56
        .long   0x227B060F,0xC0B27FC8
        .long   0x5EE9F981,0xFD21733A
        .long   0x9B58ECF3,0x399066AC
        .long   0xD7C7E065,0x75FF5A1E
        .long   0x1436D3D7,0xB26E4D90
        .long   0x50A5C749,0xEEDD4102
        .long   0x8D14BABB,0x2B4C3474

|--------------------
