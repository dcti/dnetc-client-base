| Copyright distributed.net 1997-2003 - All Rights Reserved
| For use in distributed.net projects only.
| Any other distribution or use of this source violates copyright.

| $VER: MC68060 reentrant RC5 core 17-Dec-2002
| $Id: r72-060-mh.ns.s,v 1.1.2.2 2004/05/15 08:31:09 mweiser Exp $

|
| MC680x0 RC5 key checking function
| for distributed.net RC5-72 clients.
|
| Dual key, unrolled loops, MC68060 optimised, rentrant
|
| Written by John Girvin <girv@girvnet.org.uk>
| Adapted to 72-bit by Malcolm Howell <coreblimey@rottingscorpion.com>
| Adapted to funny NeXTstep assembler syntax
|  by Michael Weiser <michael@weiser.dinsnail.net>
|
| TODO: further optimisation may be possible
|
|
| MC680x0 RC5 key checking function
| Dual key, unrolled loops, 060 optimised, rentrant
|
| Entry: a0=rc5unitwork structure:
|        0(a0) = plain.hi  - plaintext
|        4(a0) = plain.lo
|        8(a0) = cypher.hi - cyphertext
|       12(a0) = cypher.lo
|       16(a0) = L0.hi     - key
|       20(a0) = L0.mid
|       24(a0) = L0.lo
|        a1=pointer to number of iterations to run for
|       =number of keys to check / 2
|
|       (see ccoreio.h for detailed interface description)
|
| NOTES:
|   All cycle counts [n] are in 060 cycles.
|
|   'p'  indicates instruction runs in pOEP
|   's'  indicates instruction runs in sOEP
|
|   'P'  indicates instruction is pOEP-only
|      (ie: no superscalar pairing possible)
|
|   's]' indicates instruction pair would stall
|   'p]'  if first was in pOEP instead of sOEP.
|

.include "r72-0x0-common-mh.ns.i"

WS      =       204

.globl _rc5_72_unit_func_060_mh_2
.text
.align 3                        | align to 8 byte boundary
_rc5_72_unit_func_060_mh_2:

        | get our arguments from stack - unfortunately only stack
        | based calling on nextstep because of lack of support for
        | __regargs or __attribute__((regparm(3))) in gcc on m68k
        |
        | extern "C" s32 CDECL
        |  rc5_72_unit_func_020_030_mh_2(
        |   RC5_72UnitWork *, u32 *, void *);
        |
        | translates to:
        |  sp:          return address
        |  sp + 4:      RC5_72UnitWork *
        |  sp + 8:      u32 *
        |  sp + 12:     void *

        moveal  a7@(4),a0
        moveal  a7@(8),a1

        moveml  d2-d7/a2-a6,a7@-        | P  [11]

        | Following is 204 bytes because Sa[25] is never written to mem

        lea     a7@(-WS),a7             | p  [ ] save space for Sx[] storage
        movel   #0x2B4C3474,d1          |    [0] d1=P+25Q
        movel   a1@,d0                  | Get iteration count
        movel   #Q,d2                   | d2=Q
        moveq   #26-4-1,d3              |    [0] d3=loop counter
.ruf_initp3q_loop:
        movel   d1,a7@-                 | p  [ ] initialise P+nQ lookup
        subl    d2,d1
        dbf     d3,.ruf_initp3q_loop

        lea     a7@(-RUFV_SIZE),a7

        lsrl    #1,d0                   | Initial loop counter (half of key count)
        movel   a1,a7@(RUFV_ILC)        | Save initial loop counter pointer
        addql   #1,d0                   | Jigger loop counter, as we start by
                                        | jumping to end of loop
        movel   #P0QR3,a5               | a5=handy constant
        movel   d0,a7@(RUFV_LC)         | Set loop counter

        lea     a7@(RUFV_L1P2QR),a1     | a1=vars for loop
        lea     a7@(RUFV_SIZE+88+12),a3 | a3=&Sb[03]
        lea     a3@(104),a2             | a2=&Sa[03]

        bra     .ruf_l0chg

        | Upon reaching/jumping to mainloop, a1-a3 must be set up with right
        | addresses: a2=&Sa[03], a3=&Sb[03], a1=RUFV_L1P2QR (First constant)

.align 3                        | align to 8 byte boundary
.ruf_mainloop:
        | A : d0 d5
        | L0: d1          \
        | L1: d2          |- d6,d7,a6; order varies
        | L2: d3          /

        | Pipelines take turns to use d4 scratch register

        movel   a0@(L0_hi),d3   | d3=L2a=key_hi
        movel   d3,d5           | Use d5 as L2b for now

        movel   a1@+,d4         | Use d4 as handy constant L1P2QR
        addqb   #1,d5           | Next key (no carry?)

        addl    d4,d3           | Perform calculation of L2 after iteration 2
        addl    d4,d5
        roll    d4,d3
        roll    d4,d5

        movel   a1@+,d0         | d0=P32QR
        movel   d5,a6           | Preserve L2b in a6
        movel   a1@+,d2         | Get L1X in d2
        addl    d0,d5           | Perform calculation of Sb[3] in d5
        movel   a1@+,d6         | L0X=Value of L0 going into iteration 3
        roll    #3,d5
        | a1 now points to P+4*Q lookup

        addl    d3,d0   | Perform calculation of Sa[3] in d0
        movel   d5,a3@+
        roll    #3,d0
        movel   d3,d4   | Manually do second half of iter 3 on pipeline 1

        movel   d0,a2@+
        addl    d0,d4

        movel   d6,d1   | Copy L0X into d1
        addl    d4,d1
        movel   d2,d7   | Copy L1X into d7
        roll    d4,d1

        movel   a6,d4   | Set up scratch reg with L2b for first macro
        | STALL??

        | Repeated round 1 even-odd-even-odd rounds

        | This macro does a complete iteration on pipeline 1 simultaneously
        | with the end of the preceding iteration and first half of the next
        | on pipeline 2

        | args: 1 is L[j-1] reg on pipeline 1
        |       2 is L[j]   reg on pipeline 1
        |       3 is L      reg on pipeline 2
        |
        | Upon entry, the scratch register is in use by pipeline 2

.macro ROUND1

        addl    $0,d0
        addl    d5,d4   | Assume d4 already contains prev Lb[n]

        addl    a1@,d0
        addl    d4,$2

        roll    #3,d0
        roll    d4,$2
        | Pipeline 2 now finished with d4

        movel   d0,a2@+

        | Pipeline 1 claims d4
        movel   $0,d4

        addl    $2,d5
        addl    d0,d4
        addl    a1@+,d5 | Use and advance P+nQ lookup
        addl    d4,$1
        roll    #3,d5
        roll    d4,$1
        | Pipeline 1 finished with d4

        movel   d5,a3@+
        movel   $2,d4   | Set up scratch for next macro use
.endmacro

        | Now the iterations
        | Pipeline 1 does one "whole" iteration in each macro,
        | pipeline 2 is always half an iteration behind

        ROUND1  d1,d2,d6        | Iteration 4
        ROUND1  d2,d3,d7        | 5
        exg     d7,a6           | d6=L0b,a6=L1b,d7=L2b
        ROUND1  d3,d1,d7        | 6
        ROUND1  d1,d2,d6        | 7
        exg     d6,a6           | a6=L0b,d6=L1b,d7=L2b
        ROUND1  d2,d3,d6        | 8
        ROUND1  d3,d1,d7        | 9
        exg     d7,a6           | d7=L0b,d6=L1b,a6=L2b
        ROUND1  d1,d2,d7        | 10
        ROUND1  d2,d3,d6
        exg     d6,a6           | d7=L0b,a6=L1b,d6=L2b
        ROUND1  d3,d1,d6        | 12
        ROUND1  d1,d2,d7
        exg     d7,a6           | a6=L0b,d7=L1b,d6=L2b
        ROUND1  d2,d3,d7        | 14
        ROUND1  d3,d1,d6
        exg     d6,a6           | d6=L0b,d7=L1b,a6=L2b
        ROUND1  d1,d2,d6        | 16
        ROUND1  d2,d3,d7
        exg     d7,a6           | d6=L0b,a6=L1b,d7=L2b
        ROUND1  d3,d1,d7        | 18
        ROUND1  d1,d2,d6
        exg     d6,a6           | a6=L0b,d6=L0b,d7=L2b
        ROUND1  d2,d3,d6        | 20
        ROUND1  d3,d1,d7
        exg     d7,a6           | d7=L0b,d6=L1b,a6=L2b
        ROUND1  d1,d2,d7        | 22
        ROUND1  d2,d3,d6
        exg     d6,a6           | d7=L0b,a6=L1b,d6=L2b
        ROUND1  d3,d1,d6        | 24

        | Finish round 1

        | Round 1, Iteration 25 on pipeline 1
        addl    d1,d0
        | Iteration 24, second half on pipeline 2
        addl    d5,d4

        addl    a1@,d0
        addl    d4,d7

        roll    #3,d0
        roll    d4,d7   | End of pipeline 2 iter24

        movel   d0,a4   | Store Sa[25] in a4
        addl    d7,d5   | Begin pipe2 iter25

        addl    a1@+,d5 | a1=&Sb[00]
        movel   d1,d4   | Pipeline 1 claims d4
        roll    #3,d5
        addl    d0,d4
        movel   d5,a3@+ | Store Sb[25], a3=&Sa[00]
        addl    d4,d2
        exg     d7,a6   | a6=L0b,d7=L1b,d6=L2b
        roll    d4,d2   | Pipeline 1 finished with d4, round 1 ended

        movel   a6,d4   | Pipe 2 claims d4
        addl    d2,d0   | Start of round2 on pipe 1
        addl    d5,d4
        addl    a5,d0   | a5=P0QR3=S[00]
        addl    d4,d7
        roll    #3,d0
        roll    d4,d7   | End of round1 on pipe 2 - finishes with d4

        | Now a3=Sa[], a1=Sb[] - no point moving pointers around now!

        lea     a7@(RUFV_AX),a2         | Get a2 ready with pointer to constants


        movel   d0,a3@+ | Sa[00]=A
        movel   d2,d4   | pipe 1 claims d4

        addl    d7,d5   | Start of round2 on pipe 2
        addl    d0,d4
        addl    a5,d5
        addl    d4,d3
        roll    #3,d5
        roll    d4,d3   | pipe 1 finishes with d4

        movel   d5,a1@+
        movel   d7,d4   | pipe2 claims d4

        addl    a2@,d0  | pipe1 iter1, S[01]=RUFV_AX
        addl    d5,d4
        addl    d3,d0
        addl    d4,d6
        roll    #3,d0
        roll    d4,d6   | pipe2 finishes with d4

        movel   d0,a3@+
        movel   d3,d4   | pipe1 claims d4
        addl    d6,d5
        addl    d0,d4
        addl    a2@+,d5 | Worth caching (a2) = AX in a register?
        addl    d4,d1
        roll    #3,d5
        roll    d4,d1   | pipe1 finished with d4

        movel   d5,a1@+
        movel   d6,d4   | pipe2 claims d4

        addl    d1,d0   | pipe1 iter2
        exg     d6,a6   | d6=L0b,d7=L1b,a6=L2b
        addl    a2@,d0  | S[02]=AXP2QR
        addl    d5,d4
        roll    #3,d0
        addl    d4,d6
        movel   d0,a3@+
        roll    d4,d6   | pipe2 finished with d4

        | Iteration 2 (only half of it for pipeline 2)

        addl    a2@,d5  | pipe2 iter2
        movel   d1,d4   | pipe1 claims d4
        addl    d6,d5
        addl    d0,d4

        roll    #3,d5
        addl    d4,d2
        movel   d5,a1@+
        roll    d4,d2   | pipe1 finishes with d4

        movel   d6,d4   | Ready scratch register for macro repetitions
        | Again, do we stall here?

        | ROUND2 macro works the same as ROUND1, except table pointers differ
.macro ROUND2

        addl    a3@,d0
        addl    d5,d4   | Assume d4 already contains prev Lb[n]
        addl    $0,d0
        addl    d4,$2
        roll    #3,d0
        roll    d4,$2   | Pipeline 2 now finished with d4

        movel   d0,a3@+ 
        movel   $0,d4   | Pipeline 1 claims d4
        addl    a1@,d5
        addl    d0,d4
        addl    $2,d5
        addl    d4,$1
        roll    #3,d5
        roll    d4,$1   | Pipeline 1 finished with d4

        movel   d5,a1@+
        movel   $2,d4   | Set up scratch for next macro use
.endmacro

        ROUND2  d2,d3,d7        | Iteration 3
        exg     d7,a6           | d6=L0b,a6=L1b,d7=L2b
        ROUND2  d3,d1,d7        | 4
        ROUND2  d1,d2,d6
        exg     d6,a6           | a6=L0b,d6=L1b,d7=L2b
        ROUND2  d2,d3,d6        | 6
        ROUND2  d3,d1,d7
        exg     d7,a6           | d7=L0b,d6=L1b,a6=L2b
        ROUND2  d1,d2,d7        | 8
        ROUND2  d2,d3,d6
        exg     d6,a6           | d7=L0b,a6=L1b,d6=L2b
        ROUND2  d3,d1,d6        | 10
        ROUND2  d1,d2,d7
        exg     d7,a6           | a6=L0b,d7=L1b,d6=L2b
        ROUND2  d2,d3,d7        | 12
        ROUND2  d3,d1,d6
        exg     d6,a6           | d6=L0b,d7=L1b,a6=L2b
        ROUND2  d1,d2,d6        | 14
        ROUND2  d2,d3,d7
        exg     d7,a6           | d6=L0b,a6=L1b,d7=L2b
        ROUND2  d3,d1,d7        | 16
        ROUND2  d1,d2,d6
        exg     d6,a6           | a6=L0b,d6=L1b,d7=L2b
        ROUND2  d2,d3,d6        | 18
        ROUND2  d3,d1,d7
        exg     d7,a6           | d7=L0b,d6=L1b,a6=L2b
        ROUND2  d1,d2,d7        | 20
        ROUND2  d2,d3,d6
        exg     d6,a6           | d7=L0b,a6=L1b,d6=L2b
        ROUND2  d3,d1,d6        | 22
        ROUND2  d1,d2,d7
        exg     d7,a6           | a6=L0b,d7=L1b,d6=L2b
        ROUND2  d2,d3,d7        | 24

        addl    a4,d0   | Sa[25] is stored in a4
        addl    d5,d4   | Finish 24 manually on pipeline 2
        addl    d3,d0
        addl    d4,d6
        roll    #3,d0   | Don't need to store Sa[25]
        roll    d4,d6

        movel   d3,d4   | pipe1 claims d4
        addl    d6,d5
        addl    d0,d4
        addl    a1@+,d5 | a1=&Sa[00]
        addl    d4,d1
        roll    #3,d5   | Don't need to store Sb[25]
        roll    d4,d1   | pipe1 finished with d4
        | Maybe can skip following exg by waiting until round 3
        exg     d6,a6   | d6=L0b,d7=L1b,a6=L2b

        movel   a6,d4
        movel   d7,a3   | Store pipe2 state in address registers
        addl    d5,d4
        addl    d1,d0   | Begin round3/encryption on pipe1
        addl    d4,d6
        addl    a1@+,d0
        roll    d4,d6
        roll    #3,d0
 
        | ---- Combined round 3 of key expansion and encryption round ----

        | Pipeline 1 only

        movel   d6,a2   | Store pipe2 state: a2=L0b,a3=L1b,a6=L2b
        movel   d1,d4
        | Now d6=eAa, d7=eBa

        | Iteration 0
        movel   a0@(plain_lo),d6
        addl    d0,d4
        addl    d4,d2
        addl    d0,d6   | eA=plain_lo + S[00]
        addl    a1@+,d0
        roll    d4,d2

        | Iteration 1
        movel   a0@(plain_hi),d7
        addl    d2,d0
        movel   d2,d4
        roll    #3,d0
        addl    d0,d4
        addl    d0,d7
        addl    d4,d3

.macro ROUND3
        roll    d4,$0   | L[j-1] <<= scr
        addl    a1@+,d0 | A+=S[i]
        eorl    $3,$2   | eA^=eB (or vice versa)
        addl    $0,d0   | A+=L[j-1]
        movel   $0,d4   | scr <- L[j-1]
        roll    #3,d0   | A <<= 3
        roll    $3,$2   | eA <<= eB (or vv)
        addl    d0,d4   | scr += A
        addl    d0,$2   | eA += A
        addl    d4,$1   | L[j] += scr
.endmacro

.macro ROUND3_REPT1
        ROUND3  d3,d1,d6,d7     | Iterations 2-19 inclusive
        ROUND3  d1,d2,d7,d6
        ROUND3  d2,d3,d6,d7
        ROUND3  d3,d1,d7,d6
        ROUND3  d1,d2,d6,d7
        ROUND3  d2,d3,d7,d6
.endmacro
        ROUND3_REPT1
        ROUND3_REPT1
        ROUND3_REPT1

        ROUND3  d3,d1,d6,d7     | Iteration 20
        ROUND3  d1,d2,d7,d6     | 21
        ROUND3  d2,d3,d6,d7     | 22
        ROUND3  d3,d1,d7,d6     | 23

        roll    d4,d1
        addl    a1@,d0          | Iteration 24 (as much of it as needed)

        eorl    d7,d6
        addl    d1,d0

        roll    d7,d6
        roll    #3,d0
        addl    d0,d6

        cmpl    a0@(cypher_lo),d6       | eAa == cypher.lo?
        bnes    .ruf_notfounda

        | -- Low 32 bits match! (1 in 2^32 keys!) --
        | Need to completely re-check key but
        | generating high 32 bits as well...

        | Fill in "check" portion of RC5_72UnitWork
        addql   #1,a0@(check_count)
        movel   a0@(L0_hi),a0@(check_hi)
        movel   a0@(L0_mid),a0@(check_mid)
        movel   a0@(L0_lo),a0@(check_lo)

        moveq   #0,d0
        jsr     _rc5_check64
        bnes    .ruf_notfounda

        | ---------------------------------------
        | 'Interesting' key found on pipeline 1

        movel   a7@(RUFV_LC),d0         | d0=loop count
        movel   a7@(RUFV_ILC),a1        | a1=initial loop count address
        addl    d0,d0                   | 2*remaining loops = keys unchecked
        movel   a1@,d1
        subl    d0,d1                   | Find number of keys checked
        movel   d1,a1@                  | Return keys checked in in/out parameter

        lea     a7@(RUFV_SIZE+88+WS),a7

        moveml  a7@+,d2-d7/a2-a6
        moveq   #2,d0           | RESULT_FOUND
        rts

.align 3                        | align to 8 byte boundary
.ruf_notfounda:
        | -- Perform round 3 for 'b' key --
        | Now d6=eAa, d7=eBa
        | d1=L0b,d2=L1b,d3=L2b and d5=Ab
        | Retrieve stored state as we go - a2=L0,a3=L1,a6=L2

        addl    a2,d5
        lea     a7@(RUFV_SIZE+88),a1    | a1=&Sb[00]

        addl    a1@+,d5
        movel   a2,d4

        movel   a0@(plain_lo),d6
        roll    #3,d5

        addl    d5,d6           | eA=plain_lo + S[00]
        movel   a3,d2

        addl    d5,d4
        movel   a6,d3

        addl    d4,d2
        movel   a2,d1

        roll    d4,d2

        | Iteration 1
        addl    a1@+,d5

        movel   a0@(plain_hi),d7
        addl    d2,d5
        movel   d2,d4
        roll    #3,d5

        addl    d5,d4
        addl    d5,d7
        addl    d4,d3
        lea     a7@(RUFV_SIZE+88+12),a3 | a3=&Sb[03]

.macro ROUND3b
        roll    d4,$0
        addl    a1@+,d5
        eorl    $3,$2
        addl    $0,d5
        movel   $0,d4
        roll    #3,d5
        roll    $3,$2
        addl    d5,d4
        addl    d5,$2
        addl    d4,$1
.endmacro

.macro ROUND3b_REPT1
        ROUND3b  d3,d1,d6,d7            | Iterations 2-19 inclusive
        ROUND3b  d1,d2,d7,d6
        ROUND3b  d2,d3,d6,d7
        ROUND3b  d3,d1,d7,d6
        ROUND3b  d1,d2,d6,d7
        ROUND3b  d2,d3,d7,d6
.endmacro
        ROUND3b_REPT1
        ROUND3b_REPT1
        ROUND3b_REPT1
        
        ROUND3b  d3,d1,d6,d7            | Iteration 20
        ROUND3b  d1,d2,d7,d6            | 21
        ROUND3b  d2,d3,d6,d7            | 22
        ROUND3b  d3,d1,d7,d6            | 23

        | Iteration 24 (as much of it as needed)
        roll    d4,d1
        addl    a1@,d5
        addl    d1,d5
        eorl    d7,d6
        roll    #3,d5
        roll    d7,d6
        addl    d5,d6
        lea     a3@(104),a2             | a2=&Sa[03]

        cmpl    a0@(cypher_lo),d6       | eAb == cypher.lo?
        bnes    .ruf_notfoundb

        | -- Low 32 bits match! (1 in 2^32 keys!) --
        | Need to completely re-check key but
        | generating high 32 bits as well...

        | Fill in "check" portion of RC5_72UnitWork
        addql   #1,a0@(check_count)
        movel   a0@(L0_hi),d0
        addql   #1,d0
        movel   d0,a0@(check_hi)
        movel   a0@(L0_mid),a0@(check_mid)
        movel   a0@(L0_lo),a0@(check_lo)

        | Now test cypher_hi (by redoing whole key)
        moveq   #1,d0
        jsr     _rc5_check64
        bnes    .ruf_notfoundb

        | ---------------------------------------
        | 'Interesting' key found on pipeline 2

        movel   a7@(RUFV_LC),d0         | d0=loop count
        movel   a7@(RUFV_ILC),a1        | a1=initial loop count address
        addl    d0,d0                   | 2*loops to go = keys unchecked
        movel   a1@,d1
        subl    d0,d1                   | Find number of keys checked
        addql   #1,d1                   | Add one for second pipeline
        movel   d1,a1@                  | Return keys checked in in/out parameter

        lea     a7@(RUFV_SIZE+88+WS),a7

        moveml  a7@+,d2-d7/a2-a6
        moveq   #2,d0           | RESULT_FOUND
        rts

.align 3                        | align to 8 byte boundary
.ruf_notfoundb:
        | Mangle-increment current key

        addqb   #2,a0@(19)      | Increment L2 by pipeline count
        lea     a7@(RUFV_L1P2QR),a1     | First constant in main loop
        bcc     .ruf_midone
        addqb   #1,a0@(20)      | Every 256^1 (256)
        bccs    .ruf_l1chg
        addqb   #1,a0@(21)      | Every 256^2 (65536)
        bccs    .ruf_l1chg
        addqb   #1,a0@(22)      | Every 256^3 (16777216)
        bccs    .ruf_l1chg
        addqb   #1,a0@(23)      | Every 256^4 (4294967296)
        bccs    .ruf_l1chg

        addqb   #1,a0@(24)      | Every 256^5 (1099511627776)
        bccs    .ruf_l0chg
        addqb   #1,a0@(25)      | Every 256^6 (281474976710656)
        bccs    .ruf_l0chg
        addqb   #1,a0@(26)      | Every 256^7 (72057594037927936)
        bccs    .ruf_l0chg
        addqb   #1,a0@(27)      | Every 256^8

        | Need to do anything special wrapping 0xff..f -> 0x00..0 ?

.ruf_l0chg:     | L0 has changed so recalculate "constants"
                | Only called every 256^5 keys so not worth optimising
        movel   a0@(L0_lo),d1
        addl    a5,d1                   | d1=L0=L0+S[00]
        rorl    #3,d1                   | d1=L0=(L0+A)>>>3 = (L0+A)<<<P0QR3
        movel   d1,a7@(RUFV_L0X)        | Set L0x

        addl    #PR3Q,d1                | d1=A+P+Q
        roll    #3,d1                   | d1=(A+P+Q)<<<3
        movel   d1,a7@(RUFV_AX)         | Set Ax

        addl    a7@(RUFV_L0X),d1        | d1=A+L0
        movel   d1,a7@(RUFV_L0XAX)      | Set A+L0

        movel   a7@(RUFV_AX),d1         | d1=A  | Use different dn, work in parallel?
        addl    #P2Q,d1 | d1=A+P+2Q
        movel   d1,a7@(RUFV_AXP2Q)      | Set A+P+2Q

.ruf_l1chg:     | L1 change requires some recalculation
                | Every 256 keys = 128 loops, so worth making it quick

        movel   a7@(RUFV_L0XAX),d2
        movel   a0@(L0_mid),d3
        addl    d2,d3                   | d3=L1+L0x+Ax
        movel   a7@(RUFV_AXP2Q),d0
        roll    d2,d3                   | d3=L1x=(L1+L0x+Ax)<<<(L0x+Ax)
        movel   d3,a7@(RUFV_L1X)
        addl    d3,d0
        roll    #3,d0
        movel   d0,a7@(RUFV_AXP2QR)
        movel   d0,d4                   | Copy AxP2QR into d4
        addl    d3,d4                   | Add L1x
        addl    #P3Q,d0
        movel   d4,a7@(RUFV_L1P2QR)
        movel   d0,a7@(RUFV_P32QR)

.ruf_midone:
        subql   #1,a7@(RUFV_LC) | Loop back for next key
        bne     .ruf_mainloop

        | ---------------------------------
        | Key not found on either pipeline
        | Return RESULT_NOTHING: iteration count needs no adjustment

        lea     a7@(RUFV_SIZE+88+WS),a7
        moveml  a7@+,d2-d7/a2-a6
        moveq   #1,d0           | RESULT_NOTHING
        rts

| --------------------
