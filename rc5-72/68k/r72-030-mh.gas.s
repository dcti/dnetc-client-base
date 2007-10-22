
| Copyright distributed.net 1997-2003 - All Rights Reserved
| For use in distributed.net projects only.
| Any other distribution or use of this source violates copyright.
|
| RC5-72 core function for 68020/030
| 1 pipeline, rolled loops, P+nQ calculated on the fly
| Both critical loop code and data are < 256 bytes
|
| Malcolm Howell <coreblimey@rottingscorpion.com>, 27th Jan 2003
|
| Converted from Amiga Devpac assembler notation to GAS notation
| by Oliver Roberts <oliver@futaura.co.uk>
|
| $Id: r72-030-mh.gas.s,v 1.3 2007/10/22 16:48:33 jlawson Exp $
|
| $Log: r72-030-mh.gas.s,v $
| Revision 1.3  2007/10/22 16:48:33  jlawson
| overwrite head with contents of release-2-90xx
|
| Revision 1.1.2.3  2003/04/04 19:45:54  snake
| make the assembler cores link with both, elf and a.out (eg. with or with underscore prepend to the main symbols), thanx to Oliver Roberts
|
| Revision 1.1.2.2  2003/04/04 12:13:45  oliver
| changed regname syntax (d0 now %d0, etc) to provide better compatibility
| with varying gas versions
|
| Revision 1.1.2.1  2003/04/03 22:18:21  oliver
| gcc/gas compilable versions of all the 68k optimized cores
|
|

	.globl  rc5_72_unit_func_030_mh_1	| elf
	.globl	_rc5_72_unit_func_030_mh_1	| a.out

	.extern	_rc5_check64

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

| Stack layout

.equ L1XS2X,	0	| L1X + S2X
.equ S2XP3Q,	4	| S2X + P + 3Q
.equ L0X,	8	| L0 after iteration 0 of round 1
.equ L1X,	12	| L1 after iteration 1 of round 1
.equ S0X,	16	| Constant P << 3, which is S00 after round 1
.equ S1X,	20	| S01 after round 1
.equ S2X,	24	| S02 after round 1
.equ S03,	28	| Memory for S, S03-S25 from R1 and S00-S24 from R2
.equ L0XS1X,	220	| L0X + S1X
.equ S1XP2Q,	224	| S1X + P + 2Q
.equ LC,	228	| Loop count
.equ var_size,	232

|Define P, Q and related constants

.equ P,		0xb7e15163       |P
.equ Q,		0x9e3779b9       |Q
.equ P0QR3,	0xbf0a8b1d       |(0*Q + P) <<< 3
.equ PR3Q,	0x15235639       |P0QR3+P+Q
.equ P2Q,	0xf45044d5       |(2*Q + P)
.equ P3Q,	0x9287be8e       |(3*Q + P)
.equ P4Q,	0x30bf3847	|(4*Q + P)

.equ RESULT_NOTHING,	1
.equ RESULT_FOUND,	2

|---- Main key checking function ----
| Args: a0, pointer to RC5_72UnitWork
|       a1, pointer to u32 iteration count
|      (a2, void * memblk| unused)

rc5_72_unit_func_030_mh_1:	| elf
_rc5_72_unit_func_030_mh_1:	| a.out
    move.l  4(%a7),%a0
    move.l  8(%a7),%a1

    movem.l %d2-%d7/%a2-%a6,-(%a7)
    lea     -var_size(%a7),%a7

    | Calculate constants
    move.l  L0_lo(%a0),%d1

    move.l  #P0QR3,%d0
    move.l  %d0,S0X(%a7)

    add.l   %d0,%d1   |%d1=L0=L0+S[00]
    ror.l   #3,%d1       |%d1=L0=(L0+A)>>>3 = (L0+A)<<<P0QR3
    move.l  %d1,L0X(%a7)  |Set L0X

    add.l   #PR3Q,%d1    |%d1=A+P+Q
    rol.l   #3,%d1       |%d1=(A+P+Q)<<<3
    move.l  %d1,S1X(%a7)  |Set S1X

    add.l   L0X(%a7),%d1      |%d1=A+L0
    move.l  %d1,L0XS1X(%a7)   |Set A+L0

    move.l  S1X(%a7),%d1      |%d1=A
    addi.l  #P2Q,%d1         |%d1=A+P+2Q
    move.l  %d1,S1XP2Q(%a7)   |Set A+P+2Q

    move.l  L0_mid(%a0),%d2

    move.l  L0XS1X(%a7),%d3
    add.l   %d3,%d2           |%d3=L1+L0x+Ax
    move.l  S1XP2Q(%a7),%d0
    rol.l   %d3,%d2   |%d2=L1x=(L1+L0x+Ax)<<<(L0x+Ax)

    move.l  %d2,L1X(%a7)
    add.l   %d2,%d0
    rol.l   #3,%d0
    move.l  %d0,S2X(%a7)

    move.l  %d0,%d4   |Copy S2X into %d4
    add.l   %d2,%d4   |Add L1X
    add.l   #P3Q,%d0
    move.l  %d4,L1XS2X(%a7)
    move.l  %d0,S2XP3Q(%a7)

    | Put initial loop count in %d7

    move.l  #256,%d7
    move.l  L0_hi(%a0),%d3
    sub.l   %d3,%d7

    move.l  (%a1),%d6
    sub.l   %d7,%d6
    bcc.b   L.use_d7

    add.l   %d6,%d7
    moveq.l #0,%d6
L.use_d7:

    move.l  %d6,LC(%a7)
    subq.l  #1,%d7

    | Final constant setup
    move.l  #P4Q,%a4
    move.l  #Q,%a5
    bra.b   L.enter_mainloop

    .balign 8

    | Registers:
    | %d0 = A, %d1-%d3 = L0-L2, %d4 = scratch, %d7 = loop counter
    | %a0 = UnitWork, %a2 = S read ptr, %a3 = S write ptr
    | %a4 = P4Q, %a5 = Q, %a6 = loop count

L.mainloop:

    | Key increment
    addq.b  #1,L0_hi+3(%a0)  | Increment byte without carrying

L.enter_mainloop:
    move.l  L0_hi(%a0),%d3    | Fetch whole word into %d3

    | Save loop count
    move    %d7,%a6

    | Setup, and iteration 1.2
    movea.l %a7,%a2           | %a2 = &round 1 constants (L1XS2X is at offset 0)

    move.l  (%a2)+,%d4        | %d4 = S2X + L1X

    lea     S03(%a7),%a3      | %a3 = &S[03] for writing
    move.l  (%a2)+,%d0        | Get S2XP3Q ready in %d0

    add.l   %d4,%d3
    move.l  (%a2)+,%d1        | %d1 = L0X
    rol.l   %d4,%d3
    move.l  (%a2)+,%d2        | %d2 = L1X, %a2 = S read ptr ready on S0X
    move.l  %a4,%d5           | Set up %d5 = P+4Q for round 1


    moveq.l #8-1,%d7
    bra.b   L.r1_enter_loop

L.r1_loop:
    add.l   %a5,%d5

    move.l  %d1,%d4
    add.l   %d0,%d4
    add.l   %d4,%d2
    rol.l   %d4,%d2

    add.l   %d5,%d0
    add.l   %d2,%d0
    rol.l   #3,%d0
    move.l  %d0,(%a3)+
    add.l   %a5,%d5

    move.l  %d2,%d4
    add.l   %d0,%d4
    add.l   %d4,%d3
    rol.l   %d4,%d3

    add.l   %d5,%d0
    add.l   %a5,%d5

L.r1_enter_loop:
    add.l   %d3,%d0
    rol.l   #3,%d0
    move.l  %d0,(%a3)+

    move.l  %d3,%d4
    add.l   %d0,%d4
    add.l   %d4,%d1
    rol.l   %d4,%d1

    add.l   %d5,%d0
    add.l   %d1,%d0
    rol.l   #3,%d0
    move.l  %d0,(%a3)+

    dbf     %d7,L.r1_loop

    | Round 2 (starting with the very tail end of round 1,
    | and finishing with the beginning of iteration 25)

    moveq.l #9-1,%d7
    bra.b   L.r2_enter_loop

L.r2_loop:
    move.l  %d0,(%a3)+

    move.l  %d3,%d4
    add.l   %d0,%d4
    add.l   %d4,%d1
    rol.l   %d4,%d1

    add.l   (%a2)+,%d0
    add.l   %d1,%d0
    rol.l   #3,%d0
    move.l  %d0,(%a3)+

L.r2_enter_loop:
    move.l  %d1,%d4
    add.l   %d0,%d4
    add.l   %d4,%d2
    rol.l   %d4,%d2

    add.l   (%a2)+,%d0
    add.l   %d2,%d0
    rol.l   #3,%d0
    move.l  %d0,(%a3)+

    move.l  %d2,%d4
    add.l   %d0,%d4
    add.l   %d4,%d3
    rol.l   %d4,%d3

    add.l   (%a2)+,%d0
    add.l   %d3,%d0
    rol.l   #3,%d0

    dbf     %d7,L.r2_loop

    | Don't bother to store S25

    move.l  %d3,%d4
    add.l   %d0,%d4
    add.l   %d4,%d1
    rol.l   %d4,%d1

    | Round 3/Encryption
    | %d5=eA, %d6=eB

    move.l  plain_lo(%a0),%d5
    add.l   %d1,%d0
    add.l   (%a2)+,%d0
    rol.l   #3,%d0
    add.l   %d0,%d5

    move.l  (%a0),%d6     | plain_hi

    | Now the repetitions, 2-24a

    moveq   #8-1,%d7
    bra.b   L.r3_enter_loop

L.r3_loop:

    eor.l   %d6,%d5
    rol.l   %d6,%d5

    exg     %d5,%d6           | Swapping eA & eB lets us re-use the loop

L.r3_enter_loop:

    move.l  %d1,%d4
    add.l   %d0,%d4
    add.l   %d4,%d2
    rol.l   %d4,%d2

    add.l   (%a2)+,%d0
    add.l   %d2,%d0
    rol.l   #3,%d0
    add.l   %d0,%d6

    move.l  %d2,%d4
    add.l   %d0,%d4
    add.l   %d4,%d3
    rol.l   %d4,%d3

    add.l   (%a2)+,%d0
    eor.l   %d6,%d5
    add.l   %d3,%d0
    rol.l   %d6,%d5
    rol.l   #3,%d0
    add.l   %d0,%d5

    move.l  %d3,%d4
    add.l   %d0,%d4
    add.l   %d4,%d1
    rol.l   %d4,%d1

    add.l   (%a2)+,%d0
    eor.l   %d5,%d6
    add.l   %d1,%d0
    rol.l   %d5,%d6
    rol.l   #3,%d0
    add.l   %d0,%d6

    dbf     %d7,L.r3_loop

    | Arrive here with cypher_lo in %d6

    move    %a6,%d7           | Get loop count

    cmp.l   cypher_lo(%a0),%d6

    dbeq    %d7,L.mainloop

    | Reaching here, either the inner loop count has run out (highly likely)
    | or eA matches cypher_lo (might happen)

    | If loop exited because eA was a match, go check eB
    beq.w   L.keycheck

L.ctzero:
    | We either get here because the text didn't match and the beq fell through,
    | or the keycheck routine found the key didn't match and the loop count was
    | zero and it jumped here.

    | Now mangle-increment the key.
    | Normally the loop count has run out so L0_hi = 0, but if the
    | iteration count has run out we need to do one last increment
    | which may or may not carry

    | Registers: %d7 = loop count (to be reset), %d6 = overall loop count
    | %a0 and %a1 are still initial function parameters!
    | %a4, %a5 are handy constants to be preserved

    addq.b  #1,L0_hi+3(%a0)
    bcc.w   L.no_carry

    addq.b  #1,L0_mid(%a0)
    bcc.b   L.l1chg
    addq.b  #1,L0_mid+1(%a0)
    bcc.b   L.l1chg
    addq.b  #1,L0_mid+2(%a0)
    bcc.b   L.l1chg
    addq.b  #1,L0_mid+3(%a0)
    bcc.b   L.l1chg

    addq.b  #1,L0_lo(%a0)
    bcc.b   L.l0chg
    addq.b  #1,L0_lo+1(%a0)
    bcc.b   L.l0chg
    addq.b  #1,L0_lo+2(%a0)
    bcc.b   L.l0chg
    addq.b  #1,L0_lo+3(%a0)

L.l0chg:
    move.l  L0_lo(%a0),%d1

    addi.l  #P0QR3,%d1   |%d1=L0=L0+S[00]
    ror.l   #3,%d1       |%d1=L0=(L0+A)>>>3 = (L0+A)<<<P0QR3
    move.l  %d1,L0X(%a7)  |Set L0X
    move.l  %d1,%d2	|Keep L0X in %d2

    addi.l  #PR3Q,%d1    |%d1=A+P+Q
    rol.l   #3,%d1       |%d1=(A+P+Q)<<<3
    move.l  %d1,S1X(%a7)  |Set S1X
    move.l  %d1,%d4	|Keep S1X in %d4

    add.l   %d2,%d1      	    |%d1=S1X+L0X
    move.l  %d1,L0XS1X(%a7)

    addi.l  #P2Q,%d4         |%d4=S1X+P+2Q
    move.l  %d4,S1XP2Q(%a7)

L.l1chg:     | Get L0_mid and redo constants

    move.l  L0_mid(%a0),%d2

    move.l  L0XS1X(%a7),%d4
    add.l   %d4,%d2           |%d2=L1+L0x+Ax
    move.l  S1XP2Q(%a7),%d0
    rol.l   %d4,%d2   |%d2=L1x=(L1+L0x+Ax)<<<(L0x+Ax)

    move.l  %d2,L1X(%a7)
    add.l   %d2,%d0
    rol.l   #3,%d0
    move.l  %d0,S2X(%a7)

    move.l  %d0,%d4   |Copy S2X into %d4
    add.l   %d2,%d4   |Add L1X
    add.l   #P3Q,%d0
    move.l  %d4,L1XS2X(%a7)
    move.l  %d0,S2XP3Q(%a7)

L.no_carry:

    | Now to check the overall loop count.
    | If it has reached zero, time to exit
    | (Constant calculations were wasted, but key increment necessary)

    move.l  LC(%a7),%d6       | move sets condition flags
    beq.b   L.exit_nothing

    | Work out next loop count
    | This is min(LC, 256)

    move.l  #256,%d7
    sub.l   %d7,%d6

    bcc.w   L.use_256

    | If LC < 256, use LC as the inner loop count

    add.l   %d6,%d7       | If %d6 < 0, reduce loop count %d7 and set %d6 = 0
    moveq   #0,%d6

L.use_256:
    move.l  %d6,LC(%a7)

    dbf     %d7,L.enter_mainloop  | Reduce loop count and re-enter main loop

    | dbf never falls through, as %d7 >= 1

L.exit_nothing:

    moveq   #RESULT_NOTHING,%d0

L.ruf_exit:

    lea     var_size(%a7),%a7
    movem.l (%a7)+,%d2-%d7/%a2-%a6
    rts

L.keycheck:

    | Do the countermeasure check first
    addq.l  #1,check_count(%a0)
    move.l  L0_hi(%a0),check_hi(%a0)
    move.l  L0_mid(%a0),check_mid(%a0)
    move.l  L0_lo(%a0),check_lo(%a0)

    | Use common 68k check routine to test cypher_hi

    moveq   #0,%d0
    jsr     _rc5_check64

    beq.b   L.exit_success

    | eB didn't match, so figure out where to return to.
    | The dbeq at the end of the main loop fell through
    | because of eq, so %d7 wasn't decremented.
    | So, decrement it here and branch back to the main loop or ctzero,
    | as appropriate
 
    dbf     %d7,L.mainloop
    bra.w   L.ctzero

L.exit_success:
    | Return number of iterations done before success
    move.l  LC(%a7),%d6   | Get remaining outer loop count
    addq.l  #1,%d7       | Inner loop count is 1 less than you'd think
    add.l   %d6,%d7       | Total loops not done yet in %d7
    sub.l   %d7,(%a1)     | Subtract this from original count to get loops done

    moveq   #RESULT_FOUND,%d0
    bra.b   L.ruf_exit
