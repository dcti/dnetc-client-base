# Copyright distributed.net 1997-2003 - All Rights Reserved
# For use in distributed.net projects only.
# Any other distribution or use of this source violates copyright.
#
# PowerPC RC5-72 core
# 2-pipeline version, optimised for 603e
#
# Written by Malcolm Howell <coreblimey@rottingscorpion.com>
# 4th Jan 2003
#
# $Id: r72-ppc-mh-2.gas.s,v 1.2 2003/09/12 23:08:52 mweiser Exp $

gcc2_compiled.:

.section    .text

.global     rc5_72_unit_func_ppc_mh_2  # Exported

# Register aliases

.set    r0,0
.set    r1,1
.set    r3,3
.set    r4,4
.set    r5,5
.set    r6,6
.set    r7,7
.set    r8,8
.set    r9,9
.set    r10,10
.set    r11,11
.set    r12,12
.set    r13,13
.set    r14,14
.set    r15,15
.set    r16,16
.set    r17,17
.set    r18,18
.set    r19,19
.set    r20,20
.set    r21,21
.set    r22,22
.set    r23,23
.set    r24,24
.set    r25,25
.set    r26,26
.set    r27,27
.set    r28,28
.set    r29,29
.set    r30,30
.set    r31,31

.set    PnQ,r3
.set    rQ,r0

.set    Aa,r8
.set    L0a,r9
.set    L1a,r10
.set    L2a,r11

.set    eAa,r4
.set    eBa,r5

.set    S00a,r4
.set    S01a,r5

.set    Ab,r12
.set    L0b,r13
.set    L1b,r14
.set    L2b,r15

.set    eAb,r0
.set    eBb,r3

.set    S00b,r0
.set    S01b,r3

.set    scr,r6
.set    memr,r7

.set    S02a,r16
.set    S03a,r17
.set    S04a,r18
.set    S05a,r19
.set    S06a,r20
.set    S07a,r21
.set    S08a,r22
.set    S09a,r23
.set    S10a,r24
.set    S11a,r25
.set    S12a,r26
.set    S13a,r27
.set    S14a,r28
.set    S15a,r29
.set    S16a,r30
.set    S17a,r31

# Stack variables
.set    save_ret,4      # Function return address
.set    ILCP,8
.set    UWP,12
.set    L0XAX,16
.set    AXP2Q,20
.set    AX,24
.set    AXP2QR,28
.set    L1P2QR,32
.set    P32QR,36
.set    L0X,40
.set    L1X,44
.set    mS02b,48
.set    mS03b,52
.set    mS04b,56
.set    mS05b,60
.set    mS06b,64
.set    mS07b,68
.set    mS08b,72
.set    mS09b,76
.set    mS10b,80
.set    mS11b,84
.set    mS12b,88
.set    mS13b,92
.set    mS14b,96
.set    mS15b,100
.set    mS16b,104
.set    mS17b,108
.set    mS18b,112
.set    mS18a,116
.set    mS19b,120
.set    mS19a,124
.set    mS20b,128
.set    mS20a,132
.set    mS21b,136
.set    mS21a,140
.set    mS22b,144
.set    mS22a,148
.set    mS23b,152
.set    mS23a,156
.set    mS24b,160
.set    mS24a,164
.set    mS25b,168
.set    mS25a,172
.set    UW_copy,176
.set    save_regs,UW_copy + 44
.set    var_size,save_regs + 76

# Constants
.set    P,0xb7e15163
.set    Q,0x9e3779b9
.set    P0QR3,0xbf0a8b1d    # P<<3
.set    PR3Q,0x15235639     # P0QR3 + P + Q
.set    P2Q,0xf45044d5      # P+2Q
.set    P3Q,0x9287be8e      # P+3Q
.set    P4Q,0x30bf3847      # P+4Q

.set    RESULT_NOTHING,1
.set    RESULT_FOUND,2

# RC5_72UnitWork struct offsets
.set    plain_hi,UW_copy+0
.set    plain_lo,UW_copy+4
.set    cypher_hi,UW_copy+8
.set    cypher_lo,UW_copy+12
.set    L0_hi,UW_copy+16
.set    L0_mid,UW_copy+20
.set    L0_lo,UW_copy+24
.set    check_count,UW_copy+28
.set    check_hi,UW_copy+32
.set    check_mid,UW_copy+36
.set    check_lo,UW_copy+40

# Args: r3=RC5_72UnitWork *, r4=u32 * iteration_count

rc5_72_unit_func_ppc_mh_2:
    stwu    r1,-var_size(r1)
    stmw    r13,save_regs(r1)
    lwz     r7,0(r4)            # Get initial key count
    mflr    r6
    srwi    r7,r7,1             # Divide by pipeline count
    stw     r3,UWP(r1)          # Save params
    stw     r4,ILCP(r1)

    # The loop count is incremented by one, as we start by jumping to the
    # end of the loop, where the constants are calculated.

    addi    r7,r7,1
    stw     r6,save_ret(r1)     # Save return address
    mtctr   r7

    # Make copy of UnitWork on stack
    lmw     r21,0(r3)       # 11 words, r21-r31
    stmw    r21,UW_copy(r1)

    # Some constants are expected in registers on entry
    lis     r23,P4Q@h
    lwz     r20,L0_hi(r1)

    b       l0chg

.align  3

mainloop:
    # Begin round 1, halfway into iteration 2
    # At start of loop:

    # r20 contains L0_hi
    # r23 contains P4Q@h

    lwz     r28,L1P2QR(r1)      # Get L1P2QR into scratch reg r28
    lis     rQ,Q@h
    ori     PnQ,r23,P4Q@l       # P4Q@h is preloaded into r23
    addi    L2b,r20,1

    lwz     r21,P32QR(r1)
    add     L2b,L2b,r28
    rotlw   L2b,L2b,r28
    add     L2a,r20,r28
    rotlw   L2a,L2a,r28
    add     Ab,L2b,r21

    # 3b: L0 = ROTL(L0 + L2 + A, L2 + A)

    lwz     r30,L0X(r1)
    rotlwi  Ab,Ab,3
    ori     rQ,rQ,Q@l
    add     S03a,L2a,r21
    rotlwi  S03a,S03a,3
    add     scr,L2b,Ab
    stw     Ab,mS03b(r1)
    add     L0b,scr,r30
    add     memr,L2a,S03a   # Use memr as a second scratch register

    # 4a: S[04] = A = ROTL(A + L0 + P+4Q, 3)

    rotlw   L0b,L0b,scr
    add     S04a,S03a,PnQ
    add     L0a,memr,r30
    add     Ab,Ab,PnQ

    rotlw   L0a,L0a,memr
    add     Ab,Ab,L0b

    # 4b: L1 = ROTL(L1 + A + L0, A + L0)

    lwz     r31,L1X(r1)
    rotlwi  Ab,Ab,3
    add     S04a,S04a,L0a
    add     scr,Ab,L0b      # scr = A + L0
    stw     Ab,mS04b(r1)
    add     L1b,scr,r31     # L1 = L1X + A + L0

    # Now pipe a is waiting to rotate A (to find S04), and pipe b has
    # done S04 already and is waiting to rotlw L1 to find the new L1

    rotlw   L1b,L1b,scr
    add     PnQ,PnQ,rQ          # Advance to P + 5Q
    rotlwi  S04a,S04a,3         # Find S04a
    add     Ab,Ab,L1b
    add     scr,S04a,L0a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L1a,r31,scr
    stw     Ab,mS05b(r1)        # Store S05b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     S05a,S04a,PnQ
    add     L2b,L2b,scr
    add     S05a,S05a,L1a

    rotlw   L2b,L2b,scr
    add     PnQ,PnQ,rQ          # Advance to P + 6Q
    rotlwi  S05a,S05a,3         # Find S05a
    add     Ab,Ab,L2b           # Start iter 6 on b
    add     scr,S05a,L1a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS06b(r1)        # Store S06b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     S06a,S05a,PnQ       # Start iter 6 on a
    add     L0b,L0b,scr
    add     S06a,S06a,L2a

    rotlw   L0b,L0b,scr
    add     PnQ,PnQ,rQ          # P + 7Q
    rotlwi  S06a,S06a,3         # Find S06a
    add     Ab,Ab,L0b           # Start iter 7 on b
    add     scr,S06a,L2a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS07b(r1)        # Store S07b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     S07a,S06a,PnQ
    add     L1b,L1b,scr
    add     S07a,S07a,L0a

    rotlw   L1b,L1b,scr
    add     PnQ,PnQ,rQ          # P + 8Q
    rotlwi  S07a,S07a,3         # Find S07a
    add     Ab,Ab,L1b           # Start iter 8 on b
    add     scr,S07a,L0a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS08b(r1)        # Store S08b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     S08a,S07a,PnQ
    add     L2b,L2b,scr
    add     S08a,S08a,L1a

    rotlw   L2b,L2b,scr
    add     PnQ,PnQ,rQ          # P + 9Q
    rotlwi  S08a,S08a,3         # Find S08a
    add     Ab,Ab,L2b           # Start iter 9 on b
    add     scr,S08a,L1a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS09b(r1)        # Store S09b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     S09a,S08a,PnQ       # Start iter 9 on a
    add     L0b,L0b,scr
    add     S09a,S09a,L2a

    rotlw   L0b,L0b,scr
    add     PnQ,PnQ,rQ          # P + 10Q
    rotlwi  S09a,S09a,3         # Find S09a
    add     Ab,Ab,L0b           # Start iter 10 on b
    add     scr,S09a,L2a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS10b(r1)        # Store S10b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     S10a,S09a,PnQ       # Start iter 10 on a
    add     L1b,L1b,scr
    add     S10a,S10a,L0a

    rotlw   L1b,L1b,scr
    add     PnQ,PnQ,rQ          # P + 11Q
    rotlwi  S10a,S10a,3         # Find S10a
    add     Ab,Ab,L1b           # Start iter 11 on b
    add     scr,S10a,L0a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS11b(r1)        # Store S11b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     S11a,S10a,PnQ       # Start iter 11 on a
    add     L2b,L2b,scr
    add     S11a,S11a,L1a

    rotlw   L2b,L2b,scr
    add     PnQ,PnQ,rQ          # P + 12Q
    rotlwi  S11a,S11a,3         # Find S11a
    add     Ab,Ab,L2b           # Start iter 12 on b
    add     scr,S11a,L1a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS12b(r1)        # Store S12b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     S12a,S11a,PnQ       # Start iter 12 on a
    add     L0b,L0b,scr
    add     S12a,S12a,L2a

    rotlw   L0b,L0b,scr
    add     PnQ,PnQ,rQ          # P + 13Q
    rotlwi  S12a,S12a,3         # Find S12a
    add     Ab,Ab,L0b           # Start iter 13 on b
    add     scr,S12a,L2a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS13b(r1)        # Store S13b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     S13a,S12a,PnQ       # Start iter 13 on a
    add     L1b,L1b,scr
    add     S13a,S13a,L0a

    rotlw   L1b,L1b,scr
    add     PnQ,PnQ,rQ          # P + 14Q
    rotlwi  S13a,S13a,3         # Find S13a
    add     Ab,Ab,L1b           # Start iter 14 on b
    add     scr,S13a,L0a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS14b(r1)        # Store S14b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     S14a,S13a,PnQ       # Start iter 14 on a
    add     L2b,L2b,scr
    add     S14a,S14a,L1a

    rotlw   L2b,L2b,scr
    add     PnQ,PnQ,rQ          # P + 15Q
    rotlwi  S14a,S14a,3         # Find S14a
    add     Ab,Ab,L2b           # Start iter 15 on b
    add     scr,S14a,L1a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS15b(r1)        # Store S15b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     S15a,S14a,PnQ       # Start iter 15 on a
    add     L0b,L0b,scr
    add     S15a,S15a,L2a

    rotlw   L0b,L0b,scr
    add     PnQ,PnQ,rQ          # P + 16Q
    rotlwi  S15a,S15a,3         # Find S15a
    add     Ab,Ab,L0b           # Start iter 16 on b
    add     scr,S15a,L2a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS16b(r1)        # Store S16b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     S16a,S15a,PnQ       # Start iter 16 on a
    add     L1b,L1b,scr
    add     S16a,S16a,L0a

    rotlw   L1b,L1b,scr
    add     PnQ,PnQ,rQ          # P + 17Q
    rotlwi  S16a,S16a,3         # Find S16a
    add     Ab,Ab,L1b           # Start iter 17 on b
    add     scr,S16a,L0a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS17b(r1)        # Store S17b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     S17a,S16a,PnQ       # Start iter 17 on a
    add     L2b,L2b,scr
    add     S17a,S17a,L1a

    # Transition iteration: now exhausted Snna regs, start using Aa
    # and storing results in mem

    rotlw   L2b,L2b,scr
    add     PnQ,PnQ,rQ          # P + 18Q
    rotlwi  S17a,S17a,3         # Find S17a
    add     Ab,Ab,L2b           # Start iter 18 on b
    add     scr,S17a,L1a
    add     Ab,Ab,PnQ
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS18b(r1)        # Store S18b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     Aa,S17a,PnQ         # Start iter 18 on a
    add     L0b,L0b,scr
    add     Aa,Aa,L2a

    rotlw   L0b,L0b,scr
    add     PnQ,PnQ,rQ          # P + 19Q
    rotlwi  Aa,Aa,3             # Find S18a
    add     Ab,Ab,L0b           # Start iter 19 on b
    add     scr,Aa,L2a
    add     Ab,Ab,PnQ
    stw     Aa,mS18a(r1)
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS19b(r1)        # Store S19b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     L1b,L1b,scr
    add     Aa,Aa,PnQ           # Start iter 19 on a

    rotlw   L1b,L1b,scr
    add     Aa,Aa,L0a
    rotlwi  Aa,Aa,3             # Find S19a
    add     PnQ,PnQ,rQ          # P + 20Q
    add     Ab,Ab,L1b           # Start iter 20 on b
    stw     Aa,mS19a(r1)
    add     Ab,Ab,PnQ
    add     scr,Aa,L0a
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS20b(r1)        # Store S20b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     Aa,Aa,PnQ           # Start iter 20 on a
    add     L2b,L2b,scr
    add     Aa,Aa,L1a

    rotlw   L2b,L2b,scr
    add     PnQ,PnQ,rQ          # P + 21Q
    rotlwi  Aa,Aa,3             # Find S20a
    add     Ab,Ab,L2b           # Start iter 21 on b
    add     scr,Aa,L1a
    add     Ab,Ab,PnQ
    stw     Aa,mS20a(r1)
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS21b(r1)    # Store S21b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     Aa,Aa,PnQ         # Start iter 21 on a
    add     L0b,L0b,scr

    rotlw   L0b,L0b,scr
    add     Aa,Aa,L2a
    rotlwi  Aa,Aa,3             # Find S21a
    add     PnQ,PnQ,rQ          # P + 22Q
    add     Ab,Ab,L0b           # Start iter 22 on b
    stw     Aa,mS21a(r1)
    add     Ab,Ab,PnQ
    add     scr,Aa,L2a
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS22b(r1)    # Store S22b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     Aa,Aa,PnQ           # Start iter 22 on a
    add     L1b,L1b,scr
    add     Aa,Aa,L0a

    rotlw   L1b,L1b,scr
    add     PnQ,PnQ,rQ          # P + 23Q
    rotlwi  Aa,Aa,3             # Find S22a
    add     Ab,Ab,L1b           # Start iter 23 on b
    add     scr,Aa,L0a
    add     Ab,Ab,PnQ
    stw     Aa,mS22a(r1)
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS23b(r1)        # Store S23b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     Aa,Aa,PnQ           # Start iter 23 on a
    add     L2b,L2b,scr

    rotlw   L2b,L2b,scr
    add     Aa,Aa,L1a
    rotlwi  Aa,Aa,3             # Find S23a
    add     Ab,Ab,L2b           # Start iter 24 on b
    add     PnQ,PnQ,rQ          # P + 24Q
    stw     Aa,mS23a(r1)
    add     Ab,Ab,PnQ
    add     scr,Aa,L1a
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS24b(r1)        # Store S24b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     Aa,Aa,PnQ           # Start iter 24 on a
    add     L0b,L0b,scr
    add     Aa,Aa,L2a

    rotlwi  Aa,Aa,3             # Find S24a
    add     PnQ,PnQ,rQ          # P + 25Q
    rotlw   L0b,L0b,scr
    stw     Aa,mS24a(r1)
    add     Ab,Ab,L0b           # Start iter 25 on b
    add     scr,Aa,L2a
    add     Ab,Ab,PnQ
    add     L0a,L0a,scr
    rotlwi  Ab,Ab,3
    add     Aa,Aa,PnQ           # Start iter 25 on a
    add     r7,Ab,L0b           # Use r7 as a scratch reg
    stw     Ab,mS25b(r1)        # Store S25b
    rotlw   L0a,L0a,scr
    add     L1b,L1b,r7
    rotlw   L1b,L1b,r7          # Finished round 1 on b
    add     Aa,Aa,L0a

    # ---- ROUND 2 begins ----
    # (Somewhere near here)

# Recap on registers:

# r8-r11:   Aa, L0a-L2a
# r12-r15:  Ab, L0b-L2b
# r17-r31:  S03a-S17a
# r4,r5,r16:still unused, waiting for S00a-S02a

# r0,r3:    Were PnQ, rQ; now waiting for S00b,S01b
# r7:       memr, still unused (except for occasional temp usage)
# r6:       scr

    # Iteration 0a: S[00] = A = ROTL(A + L1 + P0QR3,3)

    rotlwi  Aa,Aa,3             # Find S25a
    add     Ab,Ab,L1b
    lis     r7,P0QR3@h
    stw     Aa,mS25a(r1)
    ori     r7,r7,P0QR3@l       # Both S[00] values are P0QR3
    add     scr,Aa,L0a

    add     Ab,Ab,r7
    add     L1a,L1a,scr
    rotlw   L1a,L1a,scr         # Finish round 1 on a
    add     Aa,Aa,r7
    rotlwi  S00b,Ab,3
    add     Aa,Aa,L1a

    rotlwi  S00a,Aa,3
    add     scr,S00b,L1b
    add     L2b,L2b,scr
    lwz     r7,AX(r1)

    # 0b: L2 = ROTL(L2 + A + L1, A + L1)

    rotlw   L2b,L2b,scr
    add     scr,S00a,L1a
    add     L2a,L2a,scr
    add     S01b,S00b,L2b
    rotlw   L2a,L2a,scr
    add     S01b,S01b,r7

    # 1a: S[01] = A = ROTL(A + L2 + AX,3)

    rotlwi  S01b,S01b,3
    add     S01a,S00a,L2a

    # 1b: L0 = ROTL(L0 + A + L2, A + L2)

    add     scr,S01b,L2b
    add     S01a,S01a,r7
    add     L0b,L0b,scr
    lwz     r7,AXP2QR(r1)
    rotlw   L0b,L0b,scr
    add     Ab,S01b,L0b

    # 2a: S[02] = A = ROTL(A + L0 + AXP2QR,3)

    rotlwi  S01a,S01a,3
    # Oops, stalled again
    add     scr,S01a,L2a
    add     Ab,Ab,r7
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS02b(r1)       # Store S02b in memory

    # 2b: L1 = ROTL(L1 + A + L0, A + L0)

    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     S02a,S01a,L0a
    add     L1b,L1b,scr
    add     S02a,S02a,r7

#---- Start the round 2 repeating code

    rotlw   L1b,L1b,scr
    lwz     memr,mS03b(r1)          # Fetch S03b from round 1
    rotlwi  S02a,S02a,3             # Calculate S02a
    add     Ab,Ab,L1b               # Begin iter 3 on pipe b
    add     scr,S02a,L0a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS03b(r1)            # Store S03b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     S03a,S02a,S03a          # Begin iter 3 on pipe a
    add     L2b,L2b,scr
    add     S03a,S03a,L1a

    rotlw   L2b,L2b,scr
    lwz     memr,mS04b(r1)          # Fetch S04b from round 1
    rotlwi  S03a,S03a,3             # Calculate S03a
    add     Ab,Ab,L2b               # Begin iter 4 on pipe b
    add     scr,S03a,L1a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS04b(r1)            # Store S04b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     S04a,S03a,S04a          # Begin iter 4 on pipe a
    add     L0b,L0b,scr
    add     S04a,S04a,L2a

    rotlw   L0b,L0b,scr
    lwz     memr,mS05b(r1)          # Fetch S05b from round 1
    rotlwi  S04a,S04a,3             # Calculate S04a
    add     Ab,Ab,L0b               # Begin iter 5 on pipe b
    add     scr,S04a,L2a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS05b(r1)            # Store S05b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     S05a,S04a,S05a          # Begin iter 5 on pipe a
    add     L1b,L1b,scr
    add     S05a,S05a,L0a

    rotlw   L1b,L1b,scr
    lwz     memr,mS06b(r1)          # Fetch S06b from round 1
    rotlwi  S05a,S05a,3             # Calculate S05a
    add     Ab,Ab,L1b               # Begin iter 6 on pipe b
    add     scr,S05a,L0a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS06b(r1)            # Store S06b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     S06a,S05a,S06a          # Begin iter 6 on pipe a
    add     L2b,L2b,scr
    add     S06a,S06a,L1a

    rotlw   L2b,L2b,scr
    lwz     memr,mS07b(r1)          # Fetch S07b from round 1
    rotlwi  S06a,S06a,3             # Calculate S06a
    add     Ab,Ab,L2b               # Begin iter 7 on pipe b
    add     scr,S06a,L1a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS07b(r1)            # Store S07b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     S07a,S06a,S07a          # Begin iter 7 on pipe a
    add     L0b,L0b,scr
    add     S07a,S07a,L2a

    rotlw   L0b,L0b,scr
    lwz     memr,mS08b(r1)          # Fetch S08b from round 1
    rotlwi  S07a,S07a,3             # Calculate S07a
    add     Ab,Ab,L0b               # Begin iter 8 on pipe b
    add     scr,S07a,L2a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS08b(r1)            # Store S08b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     S08a,S07a,S08a          # Begin iter 8 on pipe a
    add     L1b,L1b,scr
    add     S08a,S08a,L0a

    rotlw   L1b,L1b,scr
    lwz     memr,mS09b(r1)          # Fetch S09b from round 1
    rotlwi  S08a,S08a,3             # Calculate S08a
    add     Ab,Ab,L1b               # Begin iter 9 on pipe b
    add     scr,S08a,L0a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS09b(r1)            # Store S09b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     S09a,S08a,S09a          # Begin iter 9 on pipe a
    add     L2b,L2b,scr
    add     S09a,S09a,L1a

    rotlw   L2b,L2b,scr
    lwz     memr,mS10b(r1)          # Fetch S10b from round 1
    rotlwi  S09a,S09a,3             # Calculate S09a
    add     Ab,Ab,L2b               # Begin iter 10 on pipe b
    add     scr,S09a,L1a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS10b(r1)            # Store S10b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     S10a,S09a,S10a          # Begin iter 10 on pipe a
    add     L0b,L0b,scr
    add     S10a,S10a,L2a

    rotlw   L0b,L0b,scr
    lwz     memr,mS11b(r1)          # Fetch S11b from round 1
    rotlwi  S10a,S10a,3             # Calculate S10a
    add     Ab,Ab,L0b               # Begin iter 11 on pipe b
    add     scr,S10a,L2a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS11b(r1)            # Store S11b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     S11a,S10a,S11a          # Begin iter 11 on pipe a
    add     L1b,L1b,scr
    add     S11a,S11a,L0a

    rotlw   L1b,L1b,scr
    lwz     memr,mS12b(r1)          # Fetch S12b from round 1
    rotlwi  S11a,S11a,3             # Calculate S11a
    add     Ab,Ab,L1b               # Begin iter 12 on pipe b
    add     scr,S11a,L0a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS12b(r1)            # Store S12b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     S12a,S11a,S12a          # Begin iter 12 on pipe a
    add     L2b,L2b,scr
    add     S12a,S12a,L1a

    rotlw   L2b,L2b,scr
    lwz     memr,mS13b(r1)          # Fetch S13b from round 1
    rotlwi  S12a,S12a,3             # Calculate S12a
    add     Ab,Ab,L2b               # Begin iter 13 on pipe b
    add     scr,S12a,L1a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS13b(r1)            # Store S13b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     S13a,S12a,S13a          # Begin iter 13 on pipe a
    add     L0b,L0b,scr
    add     S13a,S13a,L2a

    rotlw   L0b,L0b,scr
    lwz     memr,mS14b(r1)          # Fetch S14b from round 1
    rotlwi  S13a,S13a,3             # Calculate S13a
    add     Ab,Ab,L0b               # Begin iter 14 on pipe b
    add     scr,S13a,L2a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS14b(r1)            # Store S14b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     S14a,S13a,S14a          # Begin iter 14 on pipe a
    add     L1b,L1b,scr
    add     S14a,S14a,L0a

    rotlw   L1b,L1b,scr
    lwz     memr,mS15b(r1)          # Fetch S15b from round 1
    rotlwi  S14a,S14a,3             # Calculate S14a
    add     Ab,Ab,L1b               # Begin iter 15 on pipe b
    add     scr,S14a,L0a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS15b(r1)            # Store S15b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     S15a,S14a,S15a          # Begin iter 15 on pipe a
    add     L2b,L2b,scr
    add     S15a,S15a,L1a

    rotlw   L2b,L2b,scr
    lwz     memr,mS16b(r1)          # Fetch S16b from round 1
    rotlwi  S15a,S15a,3             # Calculate S15a
    add     Ab,Ab,L2b               # Begin iter 16 on pipe b
    add     scr,S15a,L1a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS16b(r1)            # Store S16b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     S16a,S15a,S16a          # Begin iter 16 on pipe a
    add     L0b,L0b,scr
    add     S16a,S16a,L2a

    rotlw   L0b,L0b,scr
    lwz     memr,mS17b(r1)          # Fetch S17b from round 1
    rotlwi  S16a,S16a,3             # Calculate S16a
    add     Ab,Ab,L0b               # Begin iter 17 on pipe b
    add     scr,S16a,L2a
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS17b(r1)            # Store S17b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     S17a,S16a,S17a          # Begin iter 17 on pipe a
    add     L1b,L1b,scr
    add     S17a,S17a,L0a

    # This block fetches S18a from memory
    rotlw   L1b,L1b,scr
    lwz     memr,mS18b(r1)          # Fetch S18b from round 1
    rotlwi  S17a,S17a,3             # Calculate S17a
    add     Ab,Ab,L1b               # Begin iter 18 on pipe b
    add     scr,S17a,L0a
    add     Ab,Ab,memr
    lwz     memr,mS18a(r1)          # Fetch S18a
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS18b(r1)            # Store S18b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     Aa,S17a,memr            # Begin iter 18 on pipe a
    add     L2b,L2b,scr

    # This block stores S18a and fetches S19a
    add     Aa,Aa,L1a
    rotlw   L2b,L2b,scr
    lwz     memr,mS19b(r1)          # Fetch S19b from round 1
    rotlwi  Aa,Aa,3                 # Calculate S18a
    add     Ab,Ab,L2b               # Begin iter 19 on pipe b
    add     scr,Aa,L1a
    stw     Aa,mS18a(r1)            # Store S18a
    add     Ab,Ab,memr
    lwz     memr,mS19a(r1)          # Fetch S19a
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS19b(r1)            # Store S19b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     Aa,Aa,memr              # Begin iter 19 on pipe a
    add     L0b,L0b,scr

    add     Aa,Aa,L2a
    rotlw   L0b,L0b,scr
    lwz     memr,mS20b(r1)          # Fetch S20b from round 1
    rotlwi  Aa,Aa,3                 # Calculate S19a
    add     Ab,Ab,L0b               # Begin iter 20 on pipe b
    add     scr,Aa,L2a
    stw     Aa,mS19a(r1)            # Store S19a
    add     Ab,Ab,memr
    lwz     memr,mS20a(r1)          # Fetch S20a
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS20b(r1)            # Store S20b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     Aa,Aa,memr              # Begin iter 20 on pipe a
    add     L1b,L1b,scr

    add     Aa,Aa,L0a
    rotlw   L1b,L1b,scr
    lwz     memr,mS21b(r1)          # Fetch S21b from round 1
    rotlwi  Aa,Aa,3                 # Calculate S20a
    add     Ab,Ab,L1b               # Begin iter 21 on pipe b
    add     scr,Aa,L0a
    stw     Aa,mS20a(r1)            # Store S20a
    add     Ab,Ab,memr
    lwz     memr,mS21a(r1)          # Fetch S21a
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS21b(r1)            # Store S21b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     Aa,Aa,memr              # Begin iter 21 on pipe a
    add     L2b,L2b,scr

    add     Aa,Aa,L1a
    rotlw   L2b,L2b,scr
    lwz     memr,mS22b(r1)          # Fetch S22b from round 1
    rotlwi  Aa,Aa,3                 # Calculate S21a
    add     Ab,Ab,L2b               # Begin iter 22 on pipe b
    add     scr,Aa,L1a
    stw     Aa,mS21a(r1)            # Store S21a
    add     Ab,Ab,memr
    lwz     memr,mS22a(r1)          # Fetch S22a
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS22b(r1)            # Store S22b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     Aa,Aa,memr              # Begin iter 22 on pipe a
    add     L0b,L0b,scr

    add     Aa,Aa,L2a
    rotlw   L0b,L0b,scr
    lwz     memr,mS23b(r1)          # Fetch S23b from round 1
    rotlwi  Aa,Aa,3                 # Calculate S22a
    add     Ab,Ab,L0b               # Begin iter 23 on pipe b
    add     scr,Aa,L2a
    stw     Aa,mS22a(r1)            # Store S22a
    add     Ab,Ab,memr
    lwz     memr,mS23a(r1)          # Fetch S23a
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    stw     Ab,mS23b(r1)            # Store S23b
    rotlw   L0a,L0a,scr
    add     scr,Ab,L0b
    add     Aa,Aa,memr              # Begin iter 23 on pipe a
    add     L1b,L1b,scr

    add     Aa,Aa,L0a
    rotlw   L1b,L1b,scr
    lwz     memr,mS24b(r1)          # Fetch S24b from round 1
    rotlwi  Aa,Aa,3                 # Calculate S23a
    add     Ab,Ab,L1b               # Begin iter 24 on pipe b
    add     scr,Aa,L0a
    stw     Aa,mS23a(r1)            # Store S23a
    add     Ab,Ab,memr
    lwz     memr,mS24a(r1)          # Fetch S24a
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    stw     Ab,mS24b(r1)            # Store S24b
    rotlw   L1a,L1a,scr
    add     scr,Ab,L1b
    add     Aa,Aa,memr              # Begin iter 24 on pipe a
    add     L2b,L2b,scr

    add     Aa,Aa,L1a
    rotlw   L2b,L2b,scr
    lwz     memr,mS25b(r1)          # Fetch S25b from round 1
    rotlwi  Aa,Aa,3                 # Calculate S24a
    add     Ab,Ab,L2b               # Begin iter 25 on pipe b
    add     scr,Aa,L1a
    stw     Aa,mS24a(r1)            # Store S24a
    add     Ab,Ab,memr
    lwz     memr,mS25a(r1)          # Fetch S25a
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    stw     Ab,mS25b(r1)            # Store S25b
    rotlw   L2a,L2a,scr
    add     scr,Ab,L2b
    add     Aa,Aa,memr              # Begin iter 25 on pipe a
    add     L0b,L0b,scr

    rotlw   L0b,L0b,scr             # End of round 2 on pipe b
    add     Aa,Aa,L2a
    rotlwi  Aa,Aa,3                 # Calculate S25a
    add     Ab,Ab,S00b              # r0=S00b now becomes eAb
    lwz     memr,plain_lo(r1)
    add     scr,Aa,L2a
    add     Ab,Ab,L0b
    add     L0a,L0a,scr
    stw     Aa,mS25a(r1)            # Store S25a
    rotlw   L0a,L0a,scr             # End of round 2 on pipe a

# --- End of round 2

# Registers:

# r0,r3:        S00b,S01b
# r4,r5:        S00a,S01a
# r6,r7:        available (scr and memr)
# r8-r15:       A,L0-L2
# r16-r31:      S02a-S17a

    # ---- ROUND 3 / ENCRYPTION ----

    rotlwi  Ab,Ab,3
    add     Aa,Aa,S00a  # r4=S00a now becomes eAa
    add     scr,Ab,L0b
    add     Aa,Aa,L0a
    rotlwi  Aa,Aa,3
    add     L1b,L1b,scr
    rotlw   L1b,L1b,scr
    add     eAa,memr,Aa  # eA = plain_lo + S[00]
    add     scr,Aa,L0a
    add     eAb,memr,Ab  # Finish with memr

    # L1 = ROTL(L1 + A + L0, A + L0)

    add     L1a,L1a,scr
    add     Aa,Aa,S01a  # r5=S01a becomes eBa
    rotlw   L1a,L1a,scr
    lwz     memr,plain_hi(r1)

    # Iteration 1

    add     Ab,Ab,S01b  # r3=S01b becomes eBb
    add     Aa,Aa,L1a
    rotlwi  Aa,Aa,3
    add     Ab,Ab,L1b

    rotlwi  Ab,Ab,3
    add     scr,Aa,L1a

    add     eBa,memr,Aa # eB = plain_hi + S[01]
    add     L2a,L2a,scr
    rotlw   L2a,L2a,scr
    add     eBb,memr,Ab

    # L2 = ROTL(L2 + A + L1, A + L1)

    add     scr,Ab,L1b
    add     S02a,S02a,Aa
    add     L2b,L2b,scr
    add     S02a,S02a,L2a

    # After a nifty bit of overtaking, pipe a is now ahead of pipe b

    rotlw   L2b,L2b,scr
    lwz     memr,mS02b(r1)      # Fetch S02b
    rotlwi  S02a,S02a,3         # Calculate S02a
    add     Ab,Ab,L2b           # Begin iteration 2 on b
    xor     eAa,eAa,eBa
    add     scr,S02a,L2a
    xor     eAb,eAb,eBb
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     L0a,L0a,scr
    rotlwi  Ab,Ab,3
    add     eAa,eAa,S02a
    rotlw   L0a,L0a,scr
    add     S03a,S03a,S02a      # Begin iteration 3 on a
    rotlw   eAb,eAb,eBb
    add     scr,Ab,L2b
    xor     eBa,eBa,eAa
    add     S03a,S03a,L0a
    add     L0b,L0b,scr
    lwz     memr,mS03b(r1)      # Fetch S03b
    rotlw   L0b,L0b,scr
    add     eAb,eAb,Ab
    rotlwi  S03a,S03a,3         # Calculate S03a
    add     Ab,Ab,L0b           # Begin iteration 3 on b
    xor     eBb,eBb,eAb
    add     scr,S03a,L0a
    rotlw   eBa,eBa,eAa
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,S03a
    rotlw   L1a,L1a,scr
    add     S04a,S04a,S03a      # Begin iteration 4 on a
    add     scr,Ab,L0b
    add     eBb,eBb,Ab          # Add S03b to text B
    add     L1b,L1b,scr
    add     S04a,S04a,L1a

    rotlw   L1b,L1b,scr
    lwz     memr,mS04b(r1)      # Fetch S04b
    rotlwi  S04a,S04a,3         # Calculate S04a
    add     Ab,Ab,L1b           # Begin iteration 4 on b
    xor     eAa,eAa,eBa
    add     scr,S04a,L1a
    xor     eAb,eAb,eBb
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     L2a,L2a,scr
    rotlwi  Ab,Ab,3
    add     eAa,eAa,S04a
    rotlw   L2a,L2a,scr
    add     S05a,S05a,S04a      # Begin iteration 5 on a
    rotlw   eAb,eAb,eBb
    add     scr,Ab,L1b
    xor     eBa,eBa,eAa
    add     S05a,S05a,L2a
    add     L2b,L2b,scr
    lwz     memr,mS05b(r1)      # Fetch S05b
    rotlw   L2b,L2b,scr
    add     eAb,eAb,Ab
    rotlwi  S05a,S05a,3         # Calculate S05a
    add     Ab,Ab,L2b           # Begin iteration 5 on b
    xor     eBb,eBb,eAb
    add     scr,S05a,L2a
    rotlw   eBa,eBa,eAa
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,S05a
    rotlw   L0a,L0a,scr
    add     S06a,S06a,S05a      # Begin iteration 6 on a
    add     scr,Ab,L2b
    add     eBb,eBb,Ab
    add     L0b,L0b,scr
    add     S06a,S06a,L0a

    rotlw   L0b,L0b,scr
    lwz     memr,mS06b(r1)      # Fetch S06b
    rotlwi  S06a,S06a,3         # Calculate S06a
    add     Ab,Ab,L0b           # Begin iteration 6 on b
    xor     eAa,eAa,eBa
    add     scr,S06a,L0a
    xor     eAb,eAb,eBb
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     L1a,L1a,scr
    rotlwi  Ab,Ab,3
    add     eAa,eAa,S06a
    rotlw   L1a,L1a,scr
    add     S07a,S07a,S06a      # Begin iteration 7 on a
    rotlw   eAb,eAb,eBb
    add     scr,Ab,L0b
    xor     eBa,eBa,eAa
    add     S07a,S07a,L1a
    add     L1b,L1b,scr
    lwz     memr,mS07b(r1)      # Fetch S07b
    rotlw   L1b,L1b,scr
    add     eAb,eAb,Ab
    rotlwi  S07a,S07a,3         # Calculate S07a
    add     Ab,Ab,L1b           # Begin iteration 7 on b
    xor     eBb,eBb,eAb
    add     scr,S07a,L1a
    rotlw   eBa,eBa,eAa
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,S07a
    rotlw   L2a,L2a,scr
    add     S08a,S08a,S07a      # Begin iteration 8 on a
    add     scr,Ab,L1b
    add     eBb,eBb,Ab
    add     L2b,L2b,scr
    add     S08a,S08a,L2a

    rotlw   L2b,L2b,scr
    lwz     memr,mS08b(r1)      # Fetch S08b
    rotlwi  S08a,S08a,3         # Calculate S08a
    add     Ab,Ab,L2b           # Begin iteration 8 on b
    xor     eAa,eAa,eBa
    add     scr,S08a,L2a
    xor     eAb,eAb,eBb
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     L0a,L0a,scr
    rotlwi  Ab,Ab,3
    add     eAa,eAa,S08a
    rotlw   L0a,L0a,scr
    add     S09a,S09a,S08a      # Begin iteration 9 on a
    rotlw   eAb,eAb,eBb
    add     scr,Ab,L2b
    xor     eBa,eBa,eAa
    add     S09a,S09a,L0a
    add     L0b,L0b,scr
    lwz     memr,mS09b(r1)      # Fetch S09b
    rotlw   L0b,L0b,scr
    add     eAb,eAb,Ab
    rotlwi  S09a,S09a,3         # Calculate S09a
    add     Ab,Ab,L0b           # Begin iteration 9 on b
    xor     eBb,eBb,eAb
    add     scr,S09a,L0a
    rotlw   eBa,eBa,eAa
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,S09a
    rotlw   L1a,L1a,scr
    add     S10a,S10a,S09a      # Begin iteration 10 on a
    add     scr,Ab,L0b
    add     eBb,eBb,Ab
    add     L1b,L1b,scr
    add     S10a,S10a,L1a

    rotlw   L1b,L1b,scr
    lwz     memr,mS10b(r1)      # Fetch S10b
    rotlwi  S10a,S10a,3         # Calculate S10a
    add     Ab,Ab,L1b           # Begin iteration 10 on b
    xor     eAa,eAa,eBa
    add     scr,S10a,L1a
    xor     eAb,eAb,eBb
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     L2a,L2a,scr
    rotlwi  Ab,Ab,3
    add     eAa,eAa,S10a
    rotlw   L2a,L2a,scr
    add     S11a,S11a,S10a      # Begin iteration 11 on a
    rotlw   eAb,eAb,eBb
    add     scr,Ab,L1b
    xor     eBa,eBa,eAa
    add     S11a,S11a,L2a
    add     L2b,L2b,scr
    lwz     memr,mS11b(r1)      # Fetch S11b
    rotlw   L2b,L2b,scr
    add     eAb,eAb,Ab
    rotlwi  S11a,S11a,3         # Calculate S11a
    add     Ab,Ab,L2b           # Begin iteration 11 on b
    xor     eBb,eBb,eAb
    add     scr,S11a,L2a
    rotlw   eBa,eBa,eAa
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L0a,L0a,scr
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,S11a
    rotlw   L0a,L0a,scr
    add     S12a,S12a,S11a      # Begin iteration 12 on a
    add     scr,Ab,L2b
    add     eBb,eBb,Ab
    add     L0b,L0b,scr
    add     S12a,S12a,L0a

    rotlw   L0b,L0b,scr
    lwz     memr,mS12b(r1)      # Fetch S12b
    rotlwi  S12a,S12a,3         # Calculate S12a
    add     Ab,Ab,L0b           # Begin iteration 12 on b
    xor     eAa,eAa,eBa
    add     scr,S12a,L0a
    xor     eAb,eAb,eBb
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     L1a,L1a,scr
    rotlwi  Ab,Ab,3
    add     eAa,eAa,S12a
    rotlw   L1a,L1a,scr
    add     S13a,S13a,S12a      # Begin iteration 13 on a
    rotlw   eAb,eAb,eBb
    add     scr,Ab,L0b
    xor     eBa,eBa,eAa
    add     S13a,S13a,L1a
    add     L1b,L1b,scr
    lwz     memr,mS13b(r1)      # Fetch S13b
    rotlw   L1b,L1b,scr
    add     eAb,eAb,Ab
    rotlwi  S13a,S13a,3         # Calculate S13a
    add     Ab,Ab,L1b           # Begin iteration 13 on b
    xor     eBb,eBb,eAb
    add     scr,S13a,L1a
    rotlw   eBa,eBa,eAa
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L2a,L2a,scr
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,S13a
    rotlw   L2a,L2a,scr
    add     S14a,S14a,S13a      # Begin iteration 14 on a
    add     scr,Ab,L1b
    add     eBb,eBb,Ab
    add     L2b,L2b,scr
    add     S14a,S14a,L2a

    rotlw   L2b,L2b,scr
    lwz     memr,mS14b(r1)      # Fetch S14b
    rotlwi  S14a,S14a,3         # Calculate S14a
    add     Ab,Ab,L2b           # Begin iteration 14 on b
    xor     eAa,eAa,eBa
    add     scr,S14a,L2a
    xor     eAb,eAb,eBb
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     L0a,L0a,scr
    rotlwi  Ab,Ab,3
    add     eAa,eAa,S14a
    rotlw   L0a,L0a,scr
    add     S15a,S15a,S14a      # Begin iteration 15 on a
    rotlw   eAb,eAb,eBb
    add     scr,Ab,L2b
    xor     eBa,eBa,eAa
    add     S15a,S15a,L0a
    add     L0b,L0b,scr
    lwz     memr,mS15b(r1)      # Fetch S15b
    rotlw   L0b,L0b,scr
    add     eAb,eAb,Ab
    rotlwi  S15a,S15a,3         # Calculate S15a
    add     Ab,Ab,L0b           # Begin iteration 15 on b
    xor     eBb,eBb,eAb
    add     scr,S15a,L0a
    rotlw   eBa,eBa,eAa
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     L1a,L1a,scr
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,S15a
    rotlw   L1a,L1a,scr
    add     S16a,S16a,S15a      # Begin iteration 16 on a
    add     scr,Ab,L0b
    add     eBb,eBb,Ab
    add     L1b,L1b,scr
    add     S16a,S16a,L1a

    rotlw   L1b,L1b,scr
    lwz     memr,mS16b(r1)      # Fetch S16b
    rotlwi  S16a,S16a,3         # Calculate S16a
    add     Ab,Ab,L1b           # Begin iteration 16 on b
    xor     eAa,eAa,eBa
    add     scr,S16a,L1a
    xor     eAb,eAb,eBb
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     L2a,L2a,scr
    rotlwi  Ab,Ab,3
    add     eAa,eAa,S16a
    lwz     r20,L0_hi(r1)       # Load some useful data in advance
    lis     r23,P4Q@h           # Load half of PnQ, ready for next mainloop
    rotlw   L2a,L2a,scr
    add     S17a,S17a,S16a      # Begin iteration 17 on a
    rotlw   eAb,eAb,eBb
    add     scr,Ab,L1b
    xor     eBa,eBa,eAa
    add     S17a,S17a,L2a
    add     L2b,L2b,scr
    lwz     memr,mS17b(r1)      # Fetch S17b
    rotlw   L2b,L2b,scr
    add     eAb,eAb,Ab
    rotlwi  S17a,S17a,3         # Calculate S17a
    add     Ab,Ab,L2b           # Begin iteration 17 on b
    xor     eBb,eBb,eAb
    add     scr,S17a,L2a
    rotlw   eBa,eBa,eAa
    add     Ab,Ab,memr

    rotlwi  Ab,Ab,3
    lwz     memr,mS18a(r1)      # Fetch S18a
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,S17a
    add     L0a,L0a,scr
    add     eBb,eBb,Ab
    rotlw   L0a,L0a,scr
    add     scr,Ab,L2b
    add     Aa,S17a,memr
    add     L0b,L0b,scr
    lwz     memr,mS18b(r1)      # Fetch S18b
    xor     eAa,eAa,eBa
    rotlw   L0b,L0b,scr
    add     Aa,Aa,L0a
    xor     eAb,eAb,eBb
    add     Ab,Ab,L0b
    rotlwi  Aa,Aa,3
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     scr,Aa,L0a

    rotlwi  Ab,Ab,3
    lwz     memr,mS19a(r1)      # Fetch S19a
    rotlw   eAb,eAb,eBb
    add     eAa,eAa,Aa
    add     L1a,L1a,scr
    add     eAb,eAb,Ab
    rotlw   L1a,L1a,scr
    add     scr,Ab,L0b
    add     Aa,Aa,memr
    add     L1b,L1b,scr
    lwz     memr,mS19b(r1)      # Fetch S19b
    xor     eBa,eBa,eAa
    rotlw   L1b,L1b,scr
    add     Aa,Aa,L1a
    xor     eBb,eBb,eAb
    add     Ab,Ab,L1b
    rotlwi  Aa,Aa,3
    add     Ab,Ab,memr
    rotlw   eBa,eBa,eAa
    add     scr,Aa,L1a

    rotlwi  Ab,Ab,3
    lwz     memr,mS20a(r1)      # Fetch S20a
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,Aa
    add     L2a,L2a,scr
    add     eBb,eBb,Ab
    rotlw   L2a,L2a,scr
    add     scr,Ab,L1b
    add     Aa,Aa,memr
    add     L2b,L2b,scr
    lwz     memr,mS20b(r1)      # Fetch S20b
    xor     eAa,eAa,eBa
    rotlw   L2b,L2b,scr
    add     Aa,Aa,L2a
    xor     eAb,eAb,eBb
    add     Ab,Ab,L2b
    rotlwi  Aa,Aa,3
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     scr,Aa,L2a

    rotlwi  Ab,Ab,3
    lwz     memr,mS21a(r1)      # Fetch S21a
    rotlw   eAb,eAb,eBb
    add     eAa,eAa,Aa
    add     L0a,L0a,scr
    add     eAb,eAb,Ab
    rotlw   L0a,L0a,scr
    add     scr,Ab,L2b
    add     Aa,Aa,memr
    add     L0b,L0b,scr
    lwz     memr,mS21b(r1)      # Fetch S21b
    xor     eBa,eBa,eAa
    rotlw   L0b,L0b,scr
    add     Aa,Aa,L0a
    xor     eBb,eBb,eAb
    add     Ab,Ab,L0b
    rotlwi  Aa,Aa,3
    add     Ab,Ab,memr
    rotlw   eBa,eBa,eAa
    add     scr,Aa,L0a

    rotlwi  Ab,Ab,3
    lwz     memr,mS22a(r1)      # Fetch S22a
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,Aa
    add     L1a,L1a,scr
    add     eBb,eBb,Ab
    rotlw   L1a,L1a,scr
    add     scr,Ab,L0b
    add     Aa,Aa,memr
    add     L1b,L1b,scr
    lwz     memr,mS22b(r1)      # Fetch S22b
    xor     eAa,eAa,eBa
    rotlw   L1b,L1b,scr
    add     Aa,Aa,L1a
    xor     eAb,eAb,eBb
    add     Ab,Ab,L1b
    rotlwi  Aa,Aa,3
    add     Ab,Ab,memr
    rotlw   eAa,eAa,eBa
    add     scr,Aa,L1a

    rotlwi  Ab,Ab,3
    lwz     memr,mS23a(r1)      # Fetch S23a
    rotlw   eAb,eAb,eBb
    add     eAa,eAa,Aa
    add     L2a,L2a,scr
    add     eAb,eAb,Ab
    rotlw   L2a,L2a,scr
    add     scr,Ab,L1b
    add     Aa,Aa,memr
    add     L2b,L2b,scr
    lwz     memr,mS23b(r1)      # Fetch S23b
    xor     eBa,eBa,eAa
    rotlw   L2b,L2b,scr
    add     Aa,Aa,L2a
    xor     eBb,eBb,eAb
    add     Ab,Ab,L2b
    rotlwi  Aa,Aa,3
    add     Ab,Ab,memr
    rotlw   eBa,eBa,eAa
    add     scr,Aa,L2a

    # Iteration 24 - produces cypher_lo

    rotlwi  Ab,Ab,3
    lwz     memr,mS24a(r1)      # Fetch S24a
    rotlw   eBb,eBb,eAb
    add     eBa,eBa,Aa
    add     L0a,L0a,scr
    add     eBb,eBb,Ab
    rotlw   L0a,L0a,scr
    add     scr,Ab,L2b
    add     Aa,Aa,memr
    add     L0b,L0b,scr
    lwz     memr,mS24b(r1)      # Fetch S24b
    xor     eAa,eAa,eBa
    rotlw   L0b,L0b,scr
    add     Aa,Aa,L0a
    lwz     r16,cypher_lo(r1)
    xor     eAb,eAb,eBb
    rotlwi  Aa,Aa,3
    add     Ab,Ab,L0b
    rotlw   eAa,eAa,eBa
    add     Ab,Ab,memr

    rotlwi  Ab,Ab,3
    add     eAa,eAa,Aa      # cypher_lo is in eAa
    rotlw   eAb,eAb,eBb
    cmpw    eAa,r16

    add     eAb,eAb,Ab      # cypher_lo in eAb
    bne+    notfounda

    # Registers r17-r31 used to contain Snna values - we can now use
    # these for workspace

    # Found something on pipeline a - first word matches

    # Fill in "check" portion of RC5_72UnitWork
    lwz     r28,check_count(r1)
    addi    r28,r28,1
    stw     r28,check_count(r1)
    lmw     r29,L0_hi(r1)       # 3 words, r29-r31
    stmw    r29,check_hi(r1)

    # Check whole 64-bit block
    lwz     memr,mS25a(r1)      # Fetch S25a
    add     scr,Aa,L0a
    lwz     r18,cypher_hi(r1)
    add     L1a,L1a,scr
    xor     eBa,eBa,eAa
    rotlw   L1a,L1a,scr
    add     Aa,Aa,memr
    rotlw   eBa,eBa,eAa
    add     Aa,Aa,L1a
    rotlwi  Aa,Aa,3

    add     eBa,eBa,Aa

    cmpw    r18,eBa
    bne+    notfounda

    # Whole block matches for this key...

    lwz     r7,ILCP(r1)     # Get pointer to initial key count
    mfctr   r8              # Get remaining loop count
    lwz     r9,0(r7)        # Get initial key count
    slwi    r8,r8,1         # Multiply loop count by pipeline count
    subf    r8,r8,r9        # Difference = number of keys unsuccessfully tried
                            # = index of successful key
    stw     r8,0(r7)        # Return number in in/out parameter

    # Return
    li      r3,RESULT_FOUND

    b       ruf_exit

    .align 3
notfounda:
    cmpw    eAb,r16
    addi    r20,r20,2       # Increment key_hi by pipeline count
    bne+    notfoundb

    # Half a match found on pipeline b

    # Fill in "check" portion of RC5_72UnitWork
    lwz     r28,check_count(r1)
    addi    r28,r28,1
    stw     r28,check_count(r1)
    lmw     r29,L0_hi(r1)
    addi    r29,r29,1
    stmw    r29,check_hi(r1)

    # Thorough check
    lwz     memr,mS25b(r1)      # Fetch S25b
    add     scr,Ab,L0b
    lwz     r18,cypher_hi(r1)
    add     L1b,L1b,scr
    xor     eBb,eBb,eAb
    rotlw   L1b,L1b,scr
    add     Ab,Ab,L1b
    rotlw   eBb,eBb,eAb
    add     Ab,Ab,memr
    rotlwi  Ab,Ab,3
    add     eBb,eBb,Ab

    cmpw    r18,eBb
    bne+    notfoundb

    lwz     r7,ILCP(r1)     # Get pointer to initial key count
    mfctr   r8              # Get remaining loop count
    lwz     r9,0(r7)        # Get initial key count
    add     r8,r8,r8        # Multiply loop count by pipeline count
    subf    r8,r8,r9        # Difference = number of keys unsuccessfully tried
                            # = index of successful key
    addi    r8,r8,1         # Add one for pipeline b
    stw     r8,0(r7)        # Return number in in/out parameter

    # Return
    li      r3,RESULT_FOUND

    b       ruf_exit

    .align 3
notfoundb:

    # Mangle-increment the key by pipeline count (done earlier)
    # Can't store the key until both keys verified as wrong
    stb     r20,L0_hi+3(r1) # Just store the byte of key
    andi.   r19,r20,0x0100  # Check for carry (target reg just a dummy)
    bc      8,2,mainloop    # No carry && ctr != 0 -> jump to start of loop

    # If that didn't branch, we have counter == 0, or a carry, or both.
    beq-    exitloop        # If there is no carry but we didn't follow
                            # the last jump, counter must have expired

    # Therefore at this point, there *is* a carry to deal with, but we
    # still don't know whether the counter ran out. Therefore, nudge the
    # counter back up (so it can be decremented and tested again)
    # and deal with the carry, which is the correct thing to do before
    # returning if the counter has expired.

    # So L0_hi carried, but the mainloop expects L0_hi in r20
    # Need to clear the carry bit
    li      r5,L0_mid
    mfctr   r7
    lwbrx   r4,r5,r1        # Get byte-reversed key_mid
    addi    r7,r7,1
    andi.   r20,r20,0xFEFF  # Use the load latency to do carry clearing
    addic.  r4,r4,1         # If result is 0, it carried
    mtctr   r7
    stwbrx  r4,r5,r1
    bne+    l1chg

    li      r5,L0_lo
    lwbrx   r4,r5,r1
    addi    r4,r4,1         # Not concerned with carries here
    stwbrx  r4,r5,r1

    # Fall through to l0chg

l0chg:
    # Calculate L0-constants
    lwz     r30,L0_lo(r1)   # r30=L0.lo=L0

    lis     r31,P0QR3@h     # r31=P0QR3
    ori     r31,r31,P0QR3@l

    add     r30,r30,r31

    rotlwi  r30,r30,29      # r30=(L0+P0QR3)<<(P0QR3)=L0X

    stw     r30,L0X(r1)

    lis     r28,PR3Q@h
    ori     r28,r28,PR3Q@l
    add     r28,r28,r30
    rotlwi  r28,r28,3       # r28=AX

    stw     r28,AX(r1)

    add     r29,r30,r28     # r29=L0XAX
    stw     r29,L0XAX(r1)

    lis     r27,P2Q@h
    ori     r27,r27,P2Q@l
    add     r27,r28,r27     # r27=AXP2Q

    stw     r27,AXP2Q(r1)

l1chg:
    # --- No assumptions about registers left from l0chg ---
    #     Calculate L1-constants

    lwz     r31,L0XAX(r1)
    lis     r26,P3Q@h
    lwz     r30,L0_mid(r1)  # r30=L0_mid=L1
    ori     r26,r26,P3Q@l
    add     r30,r30,r31
    lwz     r29,AXP2Q(r1)
    rotlw   r30,r30,r31     # r30=L1X

    stw     r30,L1X(r1)

    add     r28,r29,r30
    rotlwi  r28,r28,3       # r28=AXP2QR

    stw     r28,AXP2QR(r1)

    add     r27,r30,r28     # r27=L1P2QR

    stw     r27,L1P2QR(r1)


    add     r26,r26,r28     # r26=P32QR

    stw     r26,P32QR(r1)

loopend:
    bdnz+   mainloop

exitloop:
    # Nothing found; clean up and exit
    li      r3,RESULT_NOTHING

    # No need to fiddle with the iterations arg, as we have done the
    # number requested (assuming it was a multiple of 2)

ruf_exit:
    # Transfer copy of UnitWork to original
    lmw     r21,UW_copy(r1)
    lwz     r4,UWP(r1)
    stmw    r21,0(r4)

    lwz     r5,save_ret(r1)
    mtlr    r5                  # Set up return address
    lmw     r13,save_regs(r1)
    la      r1,var_size(r1)
 
    blr
