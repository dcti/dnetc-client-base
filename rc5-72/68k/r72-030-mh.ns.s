| Copyright distributed.net 1997-2003 - All Rights Reserved
| For use in distributed.net projects only.
| Any other distribution or use of this source violates copyright.
|
| RC5-72 core function for 68020/030
| 1 pipeline, rolled loops, P+nQ calculated on the fly
| Both critical loop code and data are < 256 bytes
|
| Malcolm Howell <coreblimey@rottingscorpion.com>, 27th Jan 2003
| Adapted to funny NeXTstep assembler syntax by
|  Michael Weiser <michael@weiser.saale-net.de>
|
| $Id: r72-030-mh.ns.s,v 1.2 2003/09/12 23:08:52 mweiser Exp $
|
| $Log: r72-030-mh.ns.s,v $
| Revision 1.2  2003/09/12 23:08:52  mweiser
| add new files from release-2-90xx
|
| Revision 1.1.2.1  2003/08/09 12:25:27  mweiser
| add NeXTstep as versions of 68k assembler cores
|
| Revision 1.1.2.1  2003/04/03 22:12:10  oliver
| new cores from Malcolm Howell
|
|

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

| Stack layout

L1XS2X          = 0     | L1X + S2X
S2XP3Q          = 4     | S2X + P + 3Q
L0X             = 8     | L0 after iteration 0 of round 1
L1X             = 12    | L1 after iteration 1 of round 1
S0X             = 16    | Constant P << 3, which is S00 after round 1
S1X             = 20    | S01 after round 1
S2X             = 24    | S02 after round 1
S03             = 28    | Memory for S, S03-S25 from R1 and S00-S24 from R2
L0XS1X          = 220   | L0X + S1X
S1XP2Q          = 224   | S1X + P + 2Q
LC              = 228   | Loop count
var_size        = 232

| Define P, Q and related constants

P               = 0xb7e15163    | P
Q               = 0x9e3779b9    | Q
P0QR3           = 0xbf0a8b1d    | (0*Q + P) <<< 3
PR3Q            = 0x15235639    | P0QR3+P+Q
P2Q             = 0xf45044d5    | (2*Q + P)
P3Q             = 0x9287be8e    | (3*Q + P)
P4Q             = 0x30bf3847    | (4*Q + P)

RESULT_NOTHING  = 1
RESULT_FOUND    = 2

.globl _rc5_72_unit_func_030_mh_1
.text
.align 3                        | align to 8 byte boundary
_rc5_72_unit_func_030_mh_1:

        | get our arguments from stack - unfortunately only stack
        | based calling on nextstep because of lack of support for
        | __regargs or __attribute__((regparm(3))) in gcc on m68k
        |
        | extern "C" s32 CDECL 
        |  rc5_72_unit_func_030_mh_1(
        |   RC5_72UnitWork *, u32 *, void *);
        |
        | translates to:
        |  sp:          return address
        |  sp + 4:      RC5_72UnitWork *
        |  sp + 8:      u32 *
        |  sp + 12:     void *

        moveal  a7@(4),a0
        moveal  a7@(8),a1

        moveml  d2-d7/a2-a6,a7@-
        lea     a7@(-var_size),a7

        | Calculate constants
        movel   a0@(L0_lo),d1

        movel   #P0QR3,d0
        movel   d0,a7@(S0X)

        addl    d0,d1           | d1=L0=L0+S[00]
        rorl    #3,d1           | d1=L0=(L0+A)>>>3 = (L0+A)<<<P0QR3
        movel   d1,a7@(L0X)     | Set L0X

        addl    #PR3Q,d1        | d1=A+P+Q
        roll    #3,d1           | d1=(A+P+Q)<<<3
        movel   d1,a7@(S1X)     | Set S1X

        addl    a7@(L0X),d1     | d1=A+L0
        movel   d1,a7@(L0XS1X)  | Set A+L0

        movel   a7@(S1X),d1     | d1=A
        addil   #P2Q,d1         | d1=A+P+2Q
        movel   d1,a7@(S1XP2Q)  | Set A+P+2Q

        movel   a0@(L0_mid),d2

        movel   a7@(L0XS1X),d3
        addl    d3,d2           | d3=L1+L0x+Ax
        movel   a7@(S1XP2Q),d0
        roll    d3,d2           | d2=L1x=(L1+L0x+Ax)<<<(L0x+Ax)

        movel   d2,a7@(L1X)
        addl    d2,d0
        roll    #3,d0
        movel   d0,a7@(S2X)

        movel   d0,d4           | Copy S2X into d4
        addl    d2,d4           | Add L1X
        addl    #P3Q,d0
        movel   d4,a7@(L1XS2X)
        movel   d0,a7@(S2XP3Q)

        | Put initial loop count in d7

        movel   #256,d7
        movel   a0@(L0_hi),d3
        subl    d3,d7

        movel   a1@,d6
        subl    d7,d6
        bccs    .use_d7

        addl    d6,d7
        moveql  #0,d6
.use_d7:

        movel   d6,a7@(LC)
        subql   #1,d7

        | Final constant setup
        movel   #P4Q,a4
        movel   #Q,a5
        bras    .enter_mainloop

        | Registers:
        | d0 = A, d1-d3 = L0-L2, d4 = scratch, d7 = loop counter
        | a0 = UnitWork, a2 = S read ptr, a3 = S write ptr
        | a4 = P4Q, a5 = Q, a6 = loop count

.align 3                        | align to 8 byte boundary
.mainloop:

        | Key increment
        addqb   #1,a0@(L0_hi+3) | Increment byte without carrying

.enter_mainloop:
        movel   a0@(L0_hi),d3   | Fetch whole word into d3

        | Save loop count
        movel   d7,a6

        | Setup, and iteration 1.2
        moveal  a7,a2           | a2 = &round 1 constants (L1XS2X is at offset 0)

        movel   a2@+,d4         | d4 = S2X + L1X

        lea     a7@(S03),a3     | a3 = &S[03] for writing
        movel   a2@+,d0         | Get S2XP3Q ready in d0

        addl    d4,d3
        movel   a2@+,d1         | d1 = L0X
        roll    d4,d3
        movel   a2@+,d2         | d2 = L1X, a2 = S read ptr ready on S0X
        movel   a4,d5           | Set up d5 = P+4Q for round 1


        moveql  #8-1,d7
        bras    .r1_enter_loop

.r1_loop:
        addl    a5,d5

        movel   d1,d4
        addl    d0,d4
        addl    d4,d2
        roll    d4,d2

        addl    d5,d0
        addl    d2,d0
        roll    #3,d0
        movel   d0,a3@+
        addl    a5,d5

        movel   d2,d4
        addl    d0,d4
        addl    d4,d3
        roll    d4,d3

        addl    d5,d0
        addl    a5,d5

.r1_enter_loop:
        addl    d3,d0
        roll    #3,d0
        movel   d0,a3@+

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        addl    d5,d0
        addl    d1,d0
        roll    #3,d0
        movel   d0,a3@+

        dbf     d7,.r1_loop

        | Round 2 (starting with the very tail end of round 1,
        | and finishing with the beginning of iteration 25)

        moveql #9-1,d7
        bras    .r2_enter_loop

.r2_loop:
        movel   d0,a3@+

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        addl    a2@+,d0
        addl    d1,d0
        roll    #3,d0
        movel   d0,a3@+

.r2_enter_loop:
        movel   d1,d4
        addl    d0,d4
        addl    d4,d2
        roll    d4,d2

        addl    a2@+,d0
        addl    d2,d0
        roll    #3,d0
        movel   d0,a3@+

        movel   d2,d4
        addl    d0,d4
        addl    d4,d3
        roll    d4,d3

        addl    a2@+,d0
        addl    d3,d0
        roll    #3,d0

        dbf     d7,.r2_loop

        | Don't bother to store S25

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        | Round 3/Encryption
        | d5=eA, d6=eB

        movel   a0@(plain_lo),d5
        addl    d1,d0
        addl    a2@+,d0
        roll    #3,d0
        addl    d0,d5

        movel   a0@,d6  | plain_hi

        | Now the repetitions, 2-24a

        moveq   #8-1,d7
        bras    .r3_enter_loop

.r3_loop:

        eorl    d6,d5
        roll    d6,d5

        exg     d5,d6   | Swapping eA & eB lets us re-use the loop

.r3_enter_loop:

        movel   d1,d4
        addl    d0,d4
        addl    d4,d2
        roll    d4,d2

        addl    a2@+,d0
        addl    d2,d0
        roll    #3,d0
        addl    d0,d6

        movel   d2,d4
        addl    d0,d4
        addl    d4,d3
        roll    d4,d3

        addl    a2@+,d0
        eorl    d6,d5
        addl    d3,d0
        roll    d6,d5
        roll    #3,d0
        addl    d0,d5

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        addl    a2@+,d0
        eorl    d5,d6
        addl    d1,d0
        roll    d5,d6
        roll    #3,d0
        addl    d0,d6

        dbf     d7,.r3_loop

        | Arrive here with cypher_lo in d6

        movel   a6,d7   | Get loop count

        cmpl    a0@(cypher_lo),d6

        dbeq    d7,.mainloop

        | Reaching here, either the inner loop count has run out (highly likely)
        | or eA matches cypher_lo (might happen)

        | If loop exited because eA was a match, go check eB
        beq     .keycheck

.ctzero:
        | We either get here because the text didn't match and the beq fell through,
        | or the keycheck routine found the key didn't match and the loop count was
        | zero and it jumped here.

        | Now mangle-increment the key.
        | Normally the loop count has run out so L0_hi = 0, but if the
        | iteration count has run out we need to do one last increment
        | which may or may not carry

        | Registers: d7 = loop count (to be reset), d6 = overall loop count
        | a0 and a1 are still initial function parameters!
        | a4, a5 are handy constants to be preserved

        addqb   #1,a0@(L0_hi+3)
        bcc     .no_carry

        addqb   #1,a0@(L0_mid)
        bccs    .l1chg
        addqb   #1,a0@(L0_mid+1)
        bccs    .l1chg
        addqb   #1,a0@(L0_mid+2)
        bccs    .l1chg
        addqb   #1,a0@(L0_mid+3)
        bccs    .l1chg

        addqb   #1,a0@(L0_lo)
        bccs    .l0chg
        addqb   #1,a0@(L0_lo+1)
        bccs    .l0chg
        addqb   #1,a0@(L0_lo+2)
        bccs    .l0chg
        addqb   #1,a0@(L0_lo+3)

.l0chg:
        movel   a0@(L0_lo),d1

        addil   #P0QR3,d1       | d1=L0=L0+S[00]
        rorl    #3,d1           | L0+A@(d1=L0=)>>>3 = (L0+A)<<<P0QR3
        movel   d1,a7@(L0X)     | Set L0X
        movel   d1,d2           | Keep L0X in d2

        addil   #PR3Q,d1        | d1=A+P+Q
        roll    #3,d1           | A+P+Q@(d1=)<<<3
        movel   d1,a7@(S1X)     | Set S1X
        movel   d1,d4           | Keep S1X in d4

        addl    d2,d1           | d1=S1X+L0X
        movel   d1,a7@(L0XS1X)

        addil   #P2Q,d4         | d4=S1X+P+2Q
        movel   d4,a7@(S1XP2Q)

.l1chg: | Get L0_mid and redo constants

        movel   a0@(L0_mid),d2

        movel   a7@(L0XS1X),d4
        addl    d4,d2           | d2=L1+L0x+Ax
        movel   a7@(S1XP2Q),d0
        roll    d4,d2           | d2=L1x=(L1+L0x+Ax)<<<(L0x+Ax)

        movel   d2,a7@(L1X)
        addl    d2,d0
        roll    #3,d0
        movel   d0,a7@(S2X)

        movel   d0,d4           | Copy S2X into d4
        addl    d2,d4           | Add L1X
        addl    #P3Q,d0
        movel   d4,a7@(L1XS2X)
        movel   d0,a7@(S2XP3Q)

.no_carry:

        | Now to check the overall loop count.
        | If it has reached zero, time to exit
        | (Constant calculations were wasted, but key increment necessary)

        movel   a7@(LC),d6      | move sets condition flags
        beqs    .exit_nothing

        | Work out next loop count
        | This is min(LC, 256)

        movel   #256,d7
        subl    d7,d6

        bccs    .use_256

        | If LC < 256, use LC as the inner loop count

        addl    d6,d7           | If d6 < 0, reduce loop count d7 and set d6 = 0
        moveq   #0,d6

.use_256:
        movel   d6,a7@(LC)

        dbf     d7,.enter_mainloop      | Reduce loop count and re-enter main loop

        | dbf never falls through, as d7 >= 1

.exit_nothing:

        moveq   #RESULT_NOTHING,d0

.ruf_exit:

        lea     a7@(var_size),a7
        moveml  a7@+,d2-d7/a2-a6
        rts

.keycheck:

        | Do the countermeasure check first
        addql   #1,a0@(check_count)
        movel   a0@(L0_hi),a0@(check_hi)
        movel   a0@(L0_mid),a0@(check_mid)
        movel   a0@(L0_lo),a0@(check_lo)

        | Use common 68k check routine to test cypher_hi

        moveq   #0,d0
        jsr     _rc5_check64

        beqs    .exit_success

        | eB didn't match, so figure out where to return to.
        | The dbeq at the end of the main loop fell through
        | because of eq, so d7 wasn't decremented.
        | So, decrement it here and branch back to the main loop or ctzero,
        | as appropriate
        
        dbf     d7,.mainloop
        bra     .ctzero

.exit_success:
        | Return number of iterations done before success
        movel   a7@(LC),d6      | Get remaining outer loop count
        addql   #1,d7           | Inner loop count is 1 less than you'd think
        addl    d6,d7           | Total loops not done yet in d7
        subl    d7,a1@          | Subtract this from original count to get loops done

        moveq   #RESULT_FOUND,d0
        bras    .ruf_exit
