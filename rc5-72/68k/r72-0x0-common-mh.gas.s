
	| Copyright distributed.net 1997-2003 - All Rights Reserved
	| For use in distributed.net projects only.
	| Any other distribution or use of this source violates copyright.
	|
        | $VER: MC680x0 RC5 core common functions 04-Feb-2001
        |
        | MC680x0 RC5 core common functions
        | for distributed.net RC5-64 clients.
        |
        | Written by John Girvin <girv@girvnet.org.uk>
        | Adapted to RC5-72 by Malcolm Howell <coreblimey@rottingscorpion.com>
	|
	| Converted from Amiga Devpac assembler notation to GAS
	| notation by Oliver Roberts <oliver@futuara.co.uk>
	|
	| $Id: r72-0x0-common-mh.gas.s,v 1.1.2.3 2003/04/04 12:41:43 oliver Exp $
	|
	| $Log: r72-0x0-common-mh.gas.s,v $
	| Revision 1.1.2.3  2003/04/04 12:41:43  oliver
	| ooops - missed replacing pc with %pc
	|
	| Revision 1.1.2.2  2003/04/04 12:13:45  oliver
	| changed regname syntax (d0 now %d0, etc) to provide better compatibility
	| with varying gas versions
	|
	| Revision 1.1.2.1  2003/04/03 22:18:21  oliver
	| gcc/gas compilable versions of all the 68k optimized cores
	|
	|

|--------------------

	.globl		_rc5_check64

        .include	"r72-0x0-common-mh.gas.i"

|--------------------


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

        CNOP    0,8
_rc5_check64:
        movem.l %d2-%d4/%d6-%d7/%a1-%a3,-(%a7)

        |---- Initialise S[] array ----

        lea -312(%a7),%a7 |%a7=Sx[] storage - three copies of table
        move.l  %a7,%a1   |%a1=Sx[] storage

        |---- Mix secret key into S[] ----
        move.l  L0_hi(%a0),%d3        |%d3=L2=L0.hi
        lea     L.c64_p2q(%pc),%a3 |%a3=&S[02] read address
        move.l  L0_mid(%a0),%d2       |%d2=L1=L0.mid
        move.l  %a1,%a2   |%a2=&S[00] write address
        move.l  L0_lo(%a0),%d1        |%d1=L0=L0.lo
        
        add.b   %d0,%d3   |Adjust key byte in L2 for pipeline

        |First iteration special case

        move.l  #P0QR3,%d0       |%d0=A=P<<<3
        move.l  %d0,(%a2)+        |S[00]=P<<<3

        add.l   %d0,%d1   |%d1=L0=L0+A
        rol.l   %d0,%d1   |%d1=L0=(L0+A)<<<A

        add.l   #P+Q,%d0 |A=A+P+Q
        add.l   %d1,%d0   |A=A+P+Q+L0
        rol.l   #3,%d0   |A=(A+P+Q+L0)<<<3
        move.l  %d0,(%a2)+        |S[01]=A

        move.l  %d1,%d4   |%d4=L0
        add.l   %d0,%d4   |%d4=A+L0
        add.l   %d4,%d2   |L1=L1+A+L0
        rol.l   %d4,%d2   |L1=L1<<<(A+L0)

        | Begin proper iterations now we have initial L0 and L1

        moveq   #2-1,%d7 |%d7=outer loop counter
        moveq   #8-1,%d6      |%d6=initial inner loop counter
        bra.s   L.c64_mixkey2

L.c64_mixkey1:
        | When arriving back here, %a3 has just finished passing over the
        | P+nQ lookup table. Reset it to the start of the just-written S

        move.l  %a1,%a3   |%a3=S[] storage
        moveq   #17-1,%d6        |%d6=inner loop counter

L.c64_mixkey2:
        |%d0=A %d1=L0 %d2=L1 %d3=L2 %a2=&S[] writes %a3=&S[] reads

        add.l   (%a3)+,%d0        |A=A+S[n]
        add.l   %d2,%d0   |A=A+S[n]+L1
        rol.l   #3,%d0   |A=(A+S[n]+L1)<<<3
        move.l  %d2,%d4   |%d4=L1
        move.l  %d0,(%a2)+        |S[n]=A

        add.l   %d0,%d4   |%d4=A+L1
        add.l   %d4,%d3   |L2=L2+A+L1
        add.l   (%a3)+,%d0        |A=A+S[n]
        rol.l   %d4,%d3   |L2=L2<<<(A+L1)

        add.l   %d3,%d0   |A=A+S[n]+L2
        rol.l   #3,%d0   |A=(A+S[n]+L2)<<<3
        move.l  %d3,%d4   |%d4=L2
        move.l  %d0,(%a2)+        |S[n]=A

        add.l   %d0,%d4   |%d4=A+L2
        add.l   %d4,%d1   |L0=L0+A+L2
        add.l   (%a3)+,%d0        |A=A+S[n]
        rol.l   %d4,%d1   |L0=L0<<<(A+L2)

        add.l   %d1,%d0   |A=A+S[n]+L0
        rol.l   #3,%d0   |A=(A+S[n]+L0)<<<3
        move.l  %d0,(%a2)+        |S[n]=A

        move.l  %d1,%d4   |%d4=L0
        add.l   %d0,%d4   |%d4=A+L0
        add.l   %d4,%d2   |L1=L1+A+L0
        rol.l   %d4,%d2   |L1=L1<<<(A+L0)

        dbf     %d6,L.c64_mixkey2
        dbf     %d7,L.c64_mixkey1

        | Finish up key expansion manually

        add.l   (%a3)+,%d0        |A=A+S[25]
        add.l   %d2,%d0           |A=A+S[25]+L1
        rol.l   #3,%d0           |A=A<<<3
        move.l  %d0,(%a2)         |S[25]=A

        |---- Perform the encryption ----

        lea     208(%a1),%a1      |%a1=S[] (third copy of table)
        move.l  (%a1)+,%d0        |%d0=A=S[00]
        add.l   plain_lo(%a0),%d0        |%d0=A=S[00]+plain.lo

        move.l  (%a1)+,%d1        |%d1=B=S[01]
        moveq   #5-1,%d7 |%d7=loop counter
        add.l   (%a0),%d1 |%d1=B=S[01]+plain.hi

        eor.l   %d1,%d0   |%d0=A=A^B
        rol.l   %d1,%d0   |%d0=A=A<<<B
        add.l   (%a1)+,%d0        |%d0=A=A+S[n]

        eor.l   %d0,%d1   |%d1=B=B^A
        rol.l   %d0,%d1   |%d1=B=B<<<A
        add.l   (%a1)+,%d1        |%d1=B=B+S[n]

L.c64_encrypt:
        .rept    2
        eor.l   %d1,%d0   |%d0=A=A^B
        rol.l   %d1,%d0   |%d0=A=A<<<B
        add.l   (%a1)+,%d0        |%d0=A=A+S[n]

        eor.l   %d0,%d1   |%d1=B=B^A
        rol.l   %d0,%d1   |%d1=B=B<<<A
        add.l   (%a1)+,%d1        |%d1=B=B+S[n]
        .endr

        dbf     %d7,L.c64_encrypt

        eor.l   %d1,%d0   |%d0=A=A^B
        move.l  cypher_lo(%a0),%d2       |%d2=cypher.lo
        rol.l   %d1,%d0   |%d0=A=A<<<B
        add.l   (%a1),%d0 |%d0=A=A+S[24]

        eor.l   %d0,%d1   |%d1=B=B^A
        move.l  cypher_hi(%a0),%d2        |%d2=cypher.hi
        rol.l   %d0,%d1   |%d1=B=B<<<A
        add.l   4(%a1),%d1        |%d1=B=B+S[25]
        cmp.l   %d2,%d1   |B=cypher.hi?
        beq.s   L.c64_found

        moveq   #1,%d0   |Didn't find the key this time...
        bra.s   L.c64_done

L.c64_found:     moveq   #0,%d0   |Found the key!

L.c64_done:
        lea 312(%a7),%a7  |forget Sx[] storage
        movem.l (%a7)+,%d2-%d4/%d6-%d7/%a1-%a3
        tst.l   %d0
        rts

|----------

        | Data for 64 bit check core

        CNOP    0,8
L.c64_p2q:       |Table of P+nQ values for 2<=n<=25
        dc.l    0xF45044D5,0x9287BE8E
        dc.l    0x30BF3847,0xCEF6B200
        dc.l    0x6D2E2BB9,0x0B65A572
        dc.l    0xA99D1F2B,0x47D498E4
        dc.l    0xE60C129D,0x84438C56
        dc.l    0x227B060F,0xC0B27FC8
        dc.l    0x5EE9F981,0xFD21733A
        dc.l    0x9B58ECF3,0x399066AC
        dc.l    0xD7C7E065,0x75FF5A1E
        dc.l    0x1436D3D7,0xB26E4D90
        dc.l    0x50A5C749,0xEEDD4102
        dc.l    0x8D14BABB,0x2B4C3474

|--------------------
