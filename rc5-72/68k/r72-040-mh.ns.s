| Copyright distributed.net 1997-2003 - All Rights Reserved
| For use in distributed.net projects only.
| Any other distribution or use of this source violates copyright.
|
| RC5-72 core function for 68040
| 1 pipeline, unrolled loops, P+nQ lookup table
|
| Malcolm Howell <coreblimey@rottingscorpion.com>, 29th Jan 2003
| Adapted to funny NeXTstep assembler syntax by
|  Michael Weiser <michael@weiser.dinsnail.net>
|
| $Id: r72-040-mh.ns.s,v 1.1.2.2 2004/05/15 08:31:09 mweiser Exp $
|
| $Log: r72-040-mh.ns.s,v $
| Revision 1.1.2.2  2004/05/15 08:31:09  mweiser
| new email address
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

L1XS2X          = 0             | L1X + S2X
S2XP3Q          = 4             | S2X + P + 3Q
L0X             = 8             | L0 after iteration 0 of round 1
L1X             = 12            | L1 after iteration 1 of round 1
P4QLU           = 16            | P + nQ lookup, 4 <= n <= 25
S0X             = 104           | Constant P << 3, which is S00 after round 1
S1X             = 108           | S01 after round 1
S2X             = 112           | S02 after round 1
S07             = 116           | Memory for S, S07-S25 from R1 and S02,S07-S24 from R2
L0XS1X          = 268           | L0X + S1X
S1XP2Q          = 274           | S1X + P + 2Q
InLC            = 278           | Inner loop count
LC              = 282           | Loop count
var_size        = 286

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

.globl _rc5_72_unit_func_040_mh_1
.text
.align 3                        | align to 8 byte boundary
_rc5_72_unit_func_040_mh_1:

        | get our arguments from stack - unfortunately only stack
        | based calling on nextstep because of lack of support for
        | __regargs or __attribute__((regparm(3))) in gcc on m68k
        |
        | extern "C" s32 CDECL 
        |  rc5_72_unit_func_040_mh_1(
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

        | Calculate constants & lookup table
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

        movel   a7@(S1X),d1             | d1=A
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

        movel   #P4Q,d4
        movel   #Q,d5
        lea     a7@(P4QLU),a3
        moveq   #7-1,d7

.pnq_loop:
        movel   d4,a3@+
        addl    d5,d4
        movel   d4,a3@+
        addl    d5,d4
        movel   d4,a3@+
        addl    d5,d4
        dbf     d7,.pnq_loop

        movel   d4,a3@

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

        bras    .enter_mainloop

        | Registers:
        | d0 = A, d1-d3 = L0-L2, d4 = scratch, d7 = S03
        | d5,d6 = S00, S01, then eA, eB
        | a0 = UnitWork, a2 = S read ptr, a3 = S write ptr
        | a4-6 = S04-6

.align 3                        | align to 8 byte boundary
.mainloop:

        | Key increment
        addqb   #1,a0@(L0_hi+3) | Increment byte without carrying
.enter_mainloop:
        movel   a0@(L0_hi),d3   | Fetch whole word into d3

        | Setup, and iteration 1.2
        moveal  a7,a2           | a2 = &round 1 constants (L1XS2X is at offset 0)

        movel   d7,a7@(InLC)    | Save loop count

        movel   a2@+,d4         | d4 = S2X + L1X
        addl    d4,d3
        movel   a2@+,d0         | Get S2XP3Q ready in d0
        roll    d4,d3
        lea     a7@(S07),a3     | a3 = &S[07] for writing
        movel   a2@+,d1         | d1 = L0X

        | 1.3
        addl    d3,d0
        movel   a2@+,d2         | d2 = L1X, a2 = S read ptr ready on P+nQ LUT

        roll    #3,d0
        movel   d0,d7           | Store S03 in d7

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        | 1.4
        addl    a2@+,d0
        addl    d1,d0
        roll    #3,d0
        movel   d0,a4           | Store S04 in a4

        movel   d1,d4
        addl    d0,d4
        addl    d4,d2
        roll    d4,d2

        | 1.5
        addl    a2@+,d0
        addl    d2,d0
        roll    #3,d0
        movel   d0,a5           | Store S05 in a5

        movel   d2,d4
        addl    d0,d4
        addl    d4,d3
        roll    d4,d3

        | 1.6
        addl    a2@+,d0
        addl    d3,d0
        roll    #3,d0
        movel   d0,a6           | Store S06 in a6

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        | Iterations 1.7 to 1.24

.macro mainloop_REPT1
        addl    a2@+,d0
        addl    d1,d0
        roll    #3,d0
        movel   d0,a3@+

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
        movel   d0,a3@+

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1
.endmacro
        mainloop_REPT1
        mainloop_REPT1
        mainloop_REPT1
        mainloop_REPT1
        mainloop_REPT1
        mainloop_REPT1
        
        | 1.25
        addl    a2@+,d0
        addl    d1,d0
        roll    #3,d0
        movel   d0,a3@+

        movel   d1,d4
        addl    d0,d4
        addl    d4,d2
        roll    d4,d2

        | 2.0
        addl    a2@+,d0
        addl    d2,d0
        roll    #3,d0
        movel   d0,d5           | Store S00 in d5

        movel   d2,d4
        addl    d0,d4
        addl    d4,d3
        roll    d4,d3

        | 2.1
        addl    a2@+,d0
        addl    d3,d0
        roll    #3,d0
        movel   d0,d6           | Store S01 in d6

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        | 2.2
        addl    a2@+,d0
        addl    d1,d0
        roll    #3,d0
        movel   d0,a3@+

        movel   d1,d4
        addl    d0,d4
        addl    d4,d2
        roll    d4,d2

        | 2.3
        addl    d7,d0
        addl    d2,d0
        roll    #3,d0
        movel   d0,d7

        movel   d2,d4
        addl    d0,d4
        addl    d4,d3
        roll    d4,d3

        | 2.4
        addl    a4,d0
        addl    d3,d0
        roll    #3,d0
        movel   d0,a4

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        | 2.5
        addl    a5,d0
        addl    d1,d0
        roll    #3,d0
        movel   d0,a5

        movel   d1,d4
        addl    d0,d4
        addl    d4,d2
        roll    d4,d2

        | 2.6
        addl    a6,d0
        addl    d2,d0
        roll    #3,d0
        movel   d0,a6

        movel   d2,d4
        addl    d0,d4
        addl    d4,d3
        roll    d4,d3

        | 2.7 to 2.24
.macro mainloop_REPT2
        addl    a2@+,d0
        addl    d3,d0
        roll    #3,d0
        movel   d0,a3@+

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        addl    a2@+,d0
        addl    d1,d0
        roll    #3,d0
        movel   d0,a3@+

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
.endmacro
        mainloop_REPT2
        mainloop_REPT2
        mainloop_REPT2
        mainloop_REPT2
        mainloop_REPT2
        mainloop_REPT2

        addl    a2@+,d0
        addl    d3,d0
        roll    #3,d0

        | Don't bother to store S25

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        | Round 3/Encryption
        | d5=eA, d6=eB
        | Start by calculating S00 in d5, S01 in d6, then move back to using d0
        | and add in plaintext

        | 3.0
        addl    d1,d0
        addl    d0,d5
        roll    #3,d5

        movel   d1,d4
        addl    d5,d4
        addl    d4,d2
        roll    d4,d2

        | 3.1
        addl    d5,d6
        addl    d2,d6
        roll    #3,d6

        movel   d2,d4
        addl    d6,d4
        addl    d4,d3
        roll    d4,d3

        addl    a0@(plain_lo),d5
        movel   d6,d0
        addl    a0@,d6

        | 3.2
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

        | 3.3
        addl    d7,d0
        eorl    d5,d6
        addl    d1,d0
        roll    d5,d6
        roll    #3,d0
        addl    d0,d6

        movel   d1,d4
        addl    d0,d4
        addl    d4,d2
        roll    d4,d2

        | 3.4
        addl    a4,d0
        eorl    d6,d5
        addl    d2,d0
        roll    d6,d5
        roll    #3,d0
        addl    d0,d5

        movel   d2,d4
        addl    d0,d4
        addl    d4,d3
        roll    d4,d3

        | 3.5
        addl    a5,d0
        eorl    d5,d6
        addl    d3,d0
        roll    d5,d6
        roll    #3,d0
        addl    d0,d6

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        | 3.6
        addl    a6,d0
        eorl    d6,d5
        addl    d1,d0
        roll    d6,d5
        roll    #3,d0
        addl    d0,d5

.macro mainloop_REPT3
        movel   d1,d4
        addl    d0,d4
        addl    d4,d2
        roll    d4,d2

        addl    a2@+,d0
        eorl    d5,d6
        addl    d2,d0
        roll    d5,d6
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

        movel   d1,d4
        addl    d0,d4
        addl    d4,d2
        roll    d4,d2

        addl    a2@+,d0
        eorl    d6,d5
        addl    d2,d0
        roll    d6,d5
        roll    #3,d0
        addl    d0,d5

        movel   d2,d4
        addl    d0,d4
        addl    d4,d3
        roll    d4,d3

        addl    a2@+,d0
        eorl    d5,d6
        addl    d3,d0
        roll    d5,d6
        roll    #3,d0
        addl    d0,d6

        movel   d3,d4
        addl    d0,d4
        addl    d4,d1
        roll    d4,d1

        addl    a2@+,d0
        eorl    d6,d5
        addl    d1,d0
        roll    d6,d5
        roll    #3,d0
        addl    d0,d5
.endmacro
        mainloop_REPT3
        mainloop_REPT3
        mainloop_REPT3

        | Now cypher_lo is in d5

        movel   a7@(InLC),d7            | Get loop count

        cmpl    a0@(cypher_lo),d5

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
        | Normally the loop count has run out so L0_hi will hit 0, but if the
        | iteration count has run out we need to do one last increment
        | which may or may not carry

        | Registers: d7 = loop count (to be reset), d6 = overall loop count
        | a0 and a1 are still initial function parameters!

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
        rorl    #3,d1           | d1=L0=(L0+A)>>>3 = (L0+A)<<<P0QR3
        movel   d1,a7@(L0X)     | Set L0X
        movel   d1,d2           | Keep L0X in d2

        addl    #PR3Q,d1        | d1=A+P+Q
        roll    #3,d1           | d1=(A+P+Q)<<<3
        movel   d1,a7@(S1X)     | Set S1X
        movel   d1,d4           | Keep S1X in d4

        addl    d2,d1           | d1=S1X+L0X
        movel   d1,a7@(L0XS1X)

        addil   #P2Q,d4         | d4=S1X+P+2Q
        movel   d4,a7@(S1XP2Q)

.l1chg:         | Get L0_mid and redo constants

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

        movel   a7@(LC),d6              | move sets condition flags
        beqs    .exit_nothing

        | Work out next loop count
        | This is min(LC, 256)

        movel   #256,d7
        subl    d7,d6

        bccs    .use_256

        | If LC < 256, use LC as the inner loop count

        addl    d6,d7                   | If d6 < 0, reduce loop count d7 and set d6 = 0
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
        bras    .ctzero

.exit_success:
        | Return number of iterations done before success
        movel   a7@(LC),d6      | Get remaining outer loop count
        addql   #1,d7           | Inner loop count is 1 less than you'd think
        addl    d6,d7           | Total loops not done yet in d7
        subl    d7,a1@          | Subtract this from original count to get loops done

        moveq   #RESULT_FOUND,d0
        bras    .ruf_exit
