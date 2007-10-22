# Copyright distributed.net 1997-2003 - All Rights Reserved
# For use in distributed.net projects only.
# Any other distribution or use of this source violates copyright.
#
# RC5-72 core for PowerPC 603e
# Written by Malcolm Howell <coreblimey@rottingscorpion.com>
# 24th January 2003
#
# Based in part on "Lintilla" core for RC5-64 by Dan Oetting
#
# $Id: r72-603e-mh-1-addi.gas.s,v 1.3 2007/10/22 16:48:35 jlawson Exp $
#
# $Log: r72-603e-mh-1-addi.gas.s,v $
# Revision 1.3  2007/10/22 16:48:35  jlawson
# overwrite head with contents of release-2-90xx
#
# Revision 1.1.2.2  2003/04/05 09:53:54  oliver
# support coff and a.out output formats, as well as elf
#
# Revision 1.1.2.1  2003/04/03 22:24:00  oliver
# new core from Malcolm Howell (1-pipe lintilla-alike), with a 604e
# optimized version from Roberto Ragusa - fastest cores in testing on
# everything except a G4.
#

gcc2_compiled.:

.section    .text

.globl  rc5_72_unit_func_mh603e_addi	# elf
.globl  .rc5_72_unit_func_mh603e_addi	# coff
.globl  _rc5_72_unit_func_mh603e_addi	# a.out

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

.set    L0n,r29
.set    L1n,r30
.set    L2n,r31

.set    scrn,r6     # scrn is used in addi, so can't be r0!

.set    L0,r26
.set    L1,r27
.set    L2,r28

.set    scr,r0
.set    mem,r25

.set    eA,r3
.set    eB,r4
.set    A,r5

.set    S07,r7
.set    S08,r8
.set    S09,r9
.set    S10,r10
.set    S11,r11
.set    S12,r12
.set    S13,r13
.set    S14,r14
.set    S15,r15
.set    S16,r16
.set    S17,r17
.set    S18,r18
.set    S19,r19
.set    S20,r20
.set    S21,r21
.set    S22,r22
.set    S23,r23
.set    S24,r24
.set    S25,r25

.set    PnQ,r29
.set    rQ,r30

# Stack variables (and constants)
.set    LC,4
.set    ILCP,8
.set    UWP,12
.set    L0XS1X,16
.set    S1XP2Q,20
.set    mP0QR3,24
.set    S1X,28
.set    S2X,32
.set    L1XS2X,36
.set    S2XP3Q,40
.set    L0X,44
.set    L1X,48
.set    mS00,52
.set    mS01,mS00 + 4
.set    mS02,mS00 + 8
.set    mS03,mS00 + 12
.set    mS04,mS00 + 16
.set    mS05,mS00 + 20
.set    mS06,mS00 + 24
.set    mS07,mS00 + 28
.set    mS08,mS00 + 32
.set    mS09,mS00 + 36
.set    mS10,mS00 + 40
.set    mS11,mS00 + 44
.set    mS12,mS00 + 48
.set    mS13,mS00 + 52
.set    mS14,mS00 + 56
.set    mS03n,mS00 + 60
.set    mS04n,mS00 + 64
.set    mS05n,mS00 + 68
.set    mS06n,mS00 + 72
.set    UW_copy,mS06n + 4
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
.set    P5Q,0xCEF6B200
.set    P6Q,0x6D2E2BB9
.set    P7Q,0x0B65A572
.set    P8Q,0xA99D1F2B
.set    P9Q,0x47D498E4
.set    P10Q,0xE60C129D
.set    P11Q,0x84438C56
.set    P12Q,0x227B060F
.set    P13Q,0xC0B27FC8
.set    P14Q,0x5EE9F981
.set    P15Q,0xFD21733A
.set    P16Q,0x9B58ECF3
.set    P17Q,0x399066AC
.set    P18Q,0xD7C7E065
.set    P19Q,0x75FF5A1E
.set    P20Q,0x1436D3D7
.set    P21Q,0xB26E4D90
.set    P22Q,0x50A5C749
.set    P23Q,0xEEDD4102
.set    P24Q,0x8D14BABB
.set    P25Q,0x2B4C3474

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

# ---- Main function: rc5_72_unit_func_mh603e ----
# Arguments: r3 = RC5_72UnitWork *
#            r4 = u32 * iterations
#           (r5 = void * memblk -- unused)

rc5_72_unit_func_mh603e_addi:	# elf
.rc5_72_unit_func_mh603e_addi:	# coff
_rc5_72_unit_func_mh603e_addi:	# a.out
    stwu    r1,-var_size(r1)
    stmw    r13,save_regs(r1)

    # Save parameters
    stw     r3,UWP(r1)
    stw     r4,ILCP(r1)

    # Copy UnitWork onto stack
    lmw     r21,0(r3)
    stmw    r21,UW_copy(r1)

    lwz     r6,0(r4)    # Get initial loop count
    cmplwi  r6,0        # Make sure it's not zero!
    beq     nothing_found

    # Do first key round 1 & store constants and LUT as we go
    lis     A,P@h
    lwz     L0,L0_lo(r1)
    ori     A,A,P@l
    lis     rQ,Q@h
    lwz     L1,L0_mid(r1)
    ori     rQ,rQ,Q@l

    add     PnQ,A,rQ        # PnQ = P + Q
    rotlwi  A,A,3

    stw     A,mP0QR3(r1)

    add     L0,L0,A
    rotlw   L0,L0,A

    add     A,A,L0          # Round 1.1
    stw     L0,L0X(r1)

    add     A,A,PnQ
    rotlwi  A,A,3

    add     scr,A,L0
    stw     A,S1X(r1)

    add     PnQ,PnQ,rQ
    add     L1,L1,scr
    stw     scr,L0XS1X(r1)
    rotlw   L1,L1,scr

    add     A,A,PnQ         # Round 1.2
    stw     L1,L1X(r1)

    stw     A,S1XP2Q(r1)
    add     A,A,L1
    lwz     L2n,L0_hi(r1)   # Need to start main loop with L2n = key_hi
    rotlwi  A,A,3

    stw     A,S2X(r1)

    add     scr,A,L1
    add     PnQ,PnQ,rQ
    add     L2,L2n,scr
    stw     scr,L1XS2X(r1)
    rotlw   L2,L2,scr

    add     A,A,PnQ         # Round 1.3
    stw     A,S2XP3Q(r1)
    add     A,A,L2
    rotlwi  A,A,3

    add     scr,A,L2
    stw     A,mS03n(r1)
    add     L0,L0,scr
    add     PnQ,PnQ,rQ
    rotlw   L0,L0,scr

    add     A,A,PnQ         # Round 1.4
    add     A,A,L0
    rotlwi  A,A,3

    add     scr,A,L0
    stw     A,mS04n(r1)
    add     L1,L1,scr
    add     PnQ,PnQ,rQ
    rotlw   L1,L1,scr

    add     A,A,PnQ         # Round 1.5
    add     A,A,L1
    rotlwi  A,A,3

    add     scr,A,L1
    stw     A,mS05n(r1)
    add     L2,L2,scr
    add     PnQ,PnQ,rQ
    rotlw   L2,L2,scr

    add     A,A,PnQ         # Round 1.6
    add     A,A,L2
    rotlwi  A,A,3

    add     scr,A,L2
    stw     A,mS06n(r1)
    add     L0,L0,scr
    add     PnQ,PnQ,rQ
    rotlw   L0,L0,scr

    add     A,A,PnQ         # Round 1.7
    add     A,A,L0
    rotlwi  S07,A,3

    add     scr,S07,L0
    add     L1,L1,scr
    add     PnQ,PnQ,rQ
    rotlw   L1,L1,scr

    add     S08,S07,PnQ         # Round 1.8
    add     S08,S08,L1
    rotlwi  S08,S08,3

    add     scr,S08,L1
    add     L2,L2,scr
    add     PnQ,PnQ,rQ
    rotlw   L2,L2,scr

    add     S09,S08,PnQ         # Round 1.9
    add     S09,S09,L2
    rotlwi  S09,S09,3

    add     scr,S09,L2
    add     L0,L0,scr
    add     PnQ,PnQ,rQ
    rotlw   L0,L0,scr

    add     S10,S09,PnQ         # Round 1.10
    add     S10,S10,L0
    rotlwi  S10,S10,3

    add     scr,S10,L0
    add     L1,L1,scr
    add     PnQ,PnQ,rQ
    rotlw   L1,L1,scr

    add     S11,S10,PnQ         # Round 1.11
    add     S11,S11,L1
    rotlwi  S11,S11,3

    add     scr,S11,L1
    add     L2,L2,scr
    add     PnQ,PnQ,rQ
    rotlw   L2,L2,scr

    add     S12,S11,PnQ         # Round 1.12
    add     S12,S12,L2
    rotlwi  S12,S12,3

    add     scr,S12,L2
    add     L0,L0,scr
    add     PnQ,PnQ,rQ
    rotlw   L0,L0,scr

    add     S13,S12,PnQ         # Round 1.13
    add     S13,S13,L0
    rotlwi  S13,S13,3

    add     scr,S13,L0
    add     L1,L1,scr
    add     PnQ,PnQ,rQ
    rotlw   L1,L1,scr

    add     S14,S13,PnQ         # Round 1.14
    add     S14,S14,L1
    rotlwi  S14,S14,3

    add     scr,S14,L1
    add     L2,L2,scr
    add     PnQ,PnQ,rQ
    rotlw   L2,L2,scr

    add     S15,S14,PnQ         # Round 1.15
    add     S15,S15,L2
    rotlwi  S15,S15,3

    add     scr,S15,L2
    add     L0,L0,scr
    add     PnQ,PnQ,rQ
    rotlw   L0,L0,scr

    add     S16,S15,PnQ         # Round 1.16
    add     S16,S16,L0
    rotlwi  S16,S16,3

    add     scr,S16,L0
    add     L1,L1,scr
    add     PnQ,PnQ,rQ
    rotlw   L1,L1,scr

    add     S17,S16,PnQ         # Round 1.17
    add     S17,S17,L1
    rotlwi  S17,S17,3

    add     scr,S17,L1
    add     L2,L2,scr
    add     PnQ,PnQ,rQ
    rotlw   L2,L2,scr

    add     S18,S17,PnQ         # Round 1.18
    add     S18,S18,L2
    rotlwi  S18,S18,3

    add     scr,S18,L2
    add     L0,L0,scr
    add     PnQ,PnQ,rQ
    rotlw   L0,L0,scr

    add     S19,S18,PnQ         # Round 1.19
    add     S19,S19,L0
    rotlwi  S19,S19,3

    add     scr,S19,L0
    add     L1,L1,scr
    add     PnQ,PnQ,rQ
    rotlw   L1,L1,scr

    add     S20,S19,PnQ         # Round 1.20
    add     S20,S20,L1
    rotlwi  S20,S20,3

    add     scr,S20,L1
    add     L2,L2,scr
    add     PnQ,PnQ,rQ
    rotlw   L2,L2,scr

    add     S21,S20,PnQ         # Round 1.21
    add     S21,S21,L2
    rotlwi  S21,S21,3

    add     scr,S21,L2
    add     L0,L0,scr
    add     PnQ,PnQ,rQ
    rotlw   L0,L0,scr

    add     S22,S21,PnQ         # Round 1.22
    add     S22,S22,L0
    rotlwi  S22,S22,3

    add     scr,S22,L0
    add     L1,L1,scr
    add     PnQ,PnQ,rQ
    rotlw   L1,L1,scr

    add     S23,S22,PnQ         # Round 1.23
    add     S23,S23,L1
    rotlwi  S23,S23,3

    add     scr,S23,L1
    add     L2,L2,scr
    add     PnQ,PnQ,rQ
    rotlw   L2,L2,scr

    add     S24,S23,PnQ         # Round 1.24
    add     S24,S24,L2
    rotlwi  S24,S24,3

    add     scr,S24,L2
    add     L0,L0,scr
    add     PnQ,PnQ,rQ
    rotlw   L0,L0,scr

    add     S25,S24,PnQ         # Round 1.25
    add     S25,S25,L0
    rotlwi  S25,S25,3

    lwz     r3,mP0QR3(r1)
    add     scr,S25,L0
    lwz     r4,S1X(r1)
    add     L1,L1,scr
    lwz     r5,S2X(r1)
    rotlw   L1,L1,scr

    b       outerloop

    .align 5

outerloop:
    # Reaching here, we must have the loop count in r6
    # r3-r28, except r6, contain the state for the upcoming round 2
    # r31 holds key_hi - 1, this gets incremented just before round 1
    # r0,r29,r30 are available

    # Work out the loop count as follows:
    # t = min(LC, 255 - key_hi)
    # ctr = t
    # LC -= t

    # Usually we come back here with key_hi = -1, so do 256 keys
    # Potentially if the function is called with key_hi = 255 it will
    # mess up this algorithm, but I'll rely on the outside knowledge that
    # the starting key is always aligned to other cores' pipeline counts,
    # currently 24 (but at the very least, it's even)

    subfic  r29,r31,255     # 255 - key_hi, always 1 <= r29 <= 256
    cmplw   r6,r29          # *Unsigned* comparison
    bge+    use_keyhi       # If LC >= r29 (inner loop count), continue

    mr      r29,r6          # Else, make inner loop count = LC

use_keyhi:
    subf    r6,r29,r6
    mtctr   r29
    stw     r6,LC(r1)
    nop                     # Pad loop to double-word boundary on cache line

mainloop:
    # Registers:
    # r3-r5:    S0X=P0QR3, S1X, S2X for round 2
    # r7-r25:   S07-S25 from round 1, ready for round 2
    # r26-r28:  L0-L2, ready for round 2
    # r31:      L2n; contains key_hi - 1
    # r0 is scr
    # r6 is scrn

    # Round 2.0
        # Key increment & Round 1.2

    add     r3,S25,r3           # r3 was S00 = P0QR3
        lwz     scrn,L1XS2X(r1)
    add     r3,r3,L1
        addi    L2n,L2n,1
    rotlwi  r3,r3,3             # r3 = S00
        stw     L2n,L0_hi(r1)
    add     scr,r3,L1
        add     L2n,L2n,scrn
        rotlw   L2n,L2n,scrn
    add     L2,L2,scr

    rotlw   L2,L2,scr
    add     r4,r3,r4            # Round 2.1
    add     r4,r4,L2
        lwz     r30,S2XP3Q(r1)  # Round 1.3
    rotlwi  r4,r4,3
    stw     r3,mS00(r1)
        add     r30,r30,L2n     # Use r30 (L1n) to find S03n
    add     scr,r4,L2
    add     L0,L0,scr
    lwz     r3,mS03n(r1)        # Get S03 from previous round 1

    rotlw   L0,L0,scr
    add     r5,r4,r5            # Round 2.2 - r5 = S02
        rotlwi  r30,r30,3
    add     r5,r5,L0
    rotlwi  r5,r5,3
    stw     r4,mS01(r1)
    add     scr,r5,L0
        lwz     L0n,L0X(r1)
    add     L1,L1,scr
    lwz     r4,mS04n(r1)        # Get S04 from previous round 1

    rotlw   L1,L1,scr
    add     r3,r5,r3            # Round 2.3 - r3 = S03
        add     scrn,r30,L2n
    add     r3,r3,L1
    rotlwi  r3,r3,3
        stw     r30,mS03n(r1)
    add     scr,r3,L1
    stw     r5,mS02(r1)
        add     L0n,L0n,scrn
    add     L2,L2,scr

    rotlw   L2,L2,scr
    add     r4,r3,r4            # Round 2.4 - r4 = S04
        rotlw   L0n,L0n,scrn
    add     r4,r4,L2
    rotlwi  r4,r4,3
    lwz     r5,mS05n(r1)
    add     scr,r4,L2
        addis   r30,r30,P4Q@ha  # Round 1.4
    add     L0,L0,scr
    stw     r3,mS03(r1)

    rotlw   L0,L0,scr
    add     r5,r4,r5            # Round 2.5 - r5 = S05
    add     r5,r5,L0
    stw     r4,mS04(r1)
    rotlwi  r5,r5,3
    lwz     r4,mS06n(r1)
        addi     r3,r30,P4Q@l   # Now use r3 as An
    add     scr,r5,L0
        add     r3,r3,L0n
    add     L1,L1,scr

    rotlw   L1,L1,scr
    add     r4,r5,r4            # Round 2.6 - r4 = S06
        rotlwi  r3,r3,3
    add     r4,r4,L1
    rotlwi  r4,r4,3
        lwz     L1n,L1X(r1)
    add     scr,r4,L1
    stw     r5,mS05(r1)
    add     L2,L2,scr
    stw     r4,mS06(r1)

    rotlw   L2,L2,scr
    add     S07,S07,r4          # Round 2.7 - r7 = S07 already set!
        add     scrn,r3,L0n
    add     S07,S07,L2
    rotlwi  S07,S07,3
        stw     r3,mS04n(r1)    
        add     L1n,L1n,scrn
    add     scr,S07,L2
        rotlw   L1n,L1n,scrn
    add     L0,L0,scr

    rotlw   L0,L0,scr
        addis   r3,r3,P5Q@ha    # Round 1.5
    add     S08,S08,S07         # Round 2.8
        addi    r3,r3,P5Q@l
    add     S08,S08,L0
    # Stall
    rotlwi  S08,S08,3
        add     r3,r3,L1n
        rotlwi  r5,r3,3         # Keep S05n in r5 for a while...
    add     scr,S08,L0
    add     L1,L1,scr
    stw     S07,mS07(r1)

    rotlw   L1,L1,scr
    add     S09,S09,S08         # Round 2.9
        add     scrn,r5,L1n
    add     S09,S09,L1
    rotlwi  S09,S09,3
        add     L2n,L2n,scrn
        rotlw   L2n,L2n,scrn
    add     scr,S09,L1
    add     L2,L2,scr
        addis   r3,r5,P6Q@ha    # Round 1.6

    rotlw   L2,L2,scr
    add     S10,S10,S09         # Round 2.10
        add     r3,r3,L2n
    add     S10,S10,L2
    rotlwi  S10,S10,3
        addi    r3,r3,P6Q@l
        rotlwi  r3,r3,3
    add     scr,S10,L2
    add     L0,L0,scr
        stw     r3,mS06n(r1)

    rotlw   L0,L0,scr
    add     S11,S11,S10         # Round 2.11
        add     scrn,r3,L2n
    add     S11,S11,L0
    rotlwi  S11,S11,3
        add     L0n,L0n,scrn
        rotlw   L0n,L0n,scrn
    add     scr,S11,L0
    add     L1,L1,scr
        addis   S07,r3,P7Q@ha   # Round 1.7

    rotlw   L1,L1,scr
    add     S12,S12,S11         # Round 2.12
        add     S07,S07,L0n
    add     S12,S12,L1
    rotlwi  S12,S12,3
        addi    S07,S07,P7Q@l
        rotlwi  S07,S07,3
    add     scr,S12,L1
    add     L2,L2,scr
    stw     S08,mS08(r1)

    rotlw   L2,L2,scr
    add     S13,S13,S12         # Round 2.13
        add     scrn,S07,L0n
    add     S13,S13,L2
    rotlwi  S13,S13,3
        add     L1n,L1n,scrn
        rotlw   L1n,L1n,scrn
    add     scr,S13,L2
    add     L0,L0,scr
        addis   S08,S07,P8Q@ha    # Round 1.8

    rotlw   L0,L0,scr
    add     S14,S14,S13         # Round 2.14
        add     S08,S08,L1n
    add     S14,S14,L0
    rotlwi  S14,S14,3
        addi    S08,S08,P8Q@l
        rotlwi  S08,S08,3
    add     scr,S14,L0
    add     L1,L1,scr
    stw     S09,mS09(r1)

    rotlw   L1,L1,scr
    add     S15,S15,S14         # Round 2.15
        add     scrn,S08,L1n
    add     S15,S15,L1
    rotlwi  S15,S15,3
        add     L2n,L2n,scrn
        rotlw   L2n,L2n,scrn
    add     scr,S15,L1
    add     L2,L2,scr
        addis   S09,S08,P9Q@ha  # Round 1.9

    rotlw   L2,L2,scr
    add     S16,S16,S15         # Round 2.16
        add     S09,S09,L2n
    add     S16,S16,L2
    rotlwi  S16,S16,3
        addi    S09,S09,P9Q@l
        rotlwi  S09,S09,3
    add     scr,S16,L2
    add     L0,L0,scr
    stw     S10,mS10(r1)

    rotlw   L0,L0,scr
    add     S17,S17,S16         # Round 2.17
        add     scrn,S09,L2n
    add     S17,S17,L0
    rotlwi  S17,S17,3
        add     L0n,L0n,scrn
        rotlw   L0n,L0n,scrn
    add     scr,S17,L0
    add     L1,L1,scr
        addis   S10,S09,P10Q@ha # Round 1.10

    rotlw   L1,L1,scr
    add     S18,S18,S17         # Round 2.18
        add     S10,S10,L0n
    add     S18,S18,L1
    rotlwi  S18,S18,3
        addi    S10,S10,P10Q@l
        rotlwi  S10,S10,3
    add     scr,S18,L1
    add     L2,L2,scr
    stw     S11,mS11(r1)

    rotlw   L2,L2,scr
    add     S19,S19,S18         # Round 2.19
        add     scrn,S10,L0n
    add     S19,S19,L2
    rotlwi  S19,S19,3
        add     L1n,L1n,scrn
        rotlw   L1n,L1n,scrn
    add     scr,S19,L2
    add     L0,L0,scr
        addis   S11,S10,P11Q@ha # Round 1.11

    rotlw   L0,L0,scr
    add     S20,S20,S19         # Round 2.20
        add     S11,S11,L1n
    add     S20,S20,L0
    rotlwi  S20,S20,3
        addi    S11,S11,P11Q@l
        rotlwi  S11,S11,3
    add     scr,S20,L0
    add     L1,L1,scr
    stw     S12,mS12(r1)

    rotlw   L1,L1,scr
    add     S21,S21,S20         # Round 2.21
        add     scrn,S11,L1n
    add     S21,S21,L1
    rotlwi  S21,S21,3
        add     L2n,L2n,scrn
        rotlw   L2n,L2n,scrn
    add     scr,S21,L1
    add     L2,L2,scr
        addis   S12,S11,P12Q@ha # Round 1.12

    rotlw   L2,L2,scr
    add     S22,S22,S21         # Round 2.22
        add     S12,S12,L2n
    add     S22,S22,L2
    rotlwi  S22,S22,3
        addi    S12,S12,P12Q@l
        rotlwi  S12,S12,3
    add     scr,S22,L2
    add     L0,L0,scr
    stw     S13,mS13(r1)

    rotlw   L0,L0,scr
    add     S23,S23,S22         # Round 2.23
        add     scrn,S12,L2n
    add     S23,S23,L0
    rotlwi  S23,S23,3
        add     L0n,L0n,scrn
        rotlw   L0n,L0n,scrn
    add     scr,S23,L0
    add     L1,L1,scr
        addis   S13,S12,P13Q@ha # Round 1.13

    rotlw   L1,L1,scr
    add     S24,S24,S23         # Round 2.24
        add     S13,S13,L0n
    add     S24,S24,L1
    rotlwi  S24,S24,3
        addi    S13,S13,P13Q@l
        rotlwi  S13,S13,3
    add     scr,S24,L1
    add     L2,L2,scr
    stw     S14,mS14(r1)

    rotlw   L2,L2,scr
    add     S25,S25,S24         # Round 2.25
    add     S25,S25,L2
        stw   r5,mS05n(r1)      # Left over from way back
    rotlwi  S25,S25,3
    lwz     A,mS00(r1)          # Note A = r5
    add     scr,S25,L2
    lwz     eA,plain_lo(r1)
        add     scrn,S13,L0n
    add     L0,L0,scr


    rotlw   L0,L0,scr
    add     A,S25,A             # Round 3.0 - S25 finished, r25 = mem now
        add     L1n,L1n,scrn
    add     A,A,L0
    rotlwi  A,A,3
    lwz     eB,plain_hi(r1)
        rotlw   L1n,L1n,scrn
    add     scr,A,L0
    add     L1,L1,scr
    lwz     mem,mS01(r1)


    rotlw   L1,L1,scr
    add     eA,eA,A
    add     A,A,mem             # Round 3.1
        addis   S14,S13,P14Q@ha # Round 1.14
    add     A,A,L1
        add     S14,S14,L1n
    rotlwi  A,A,3
        addi    S14,S14,P14Q@l

        rotlwi  S14,S14,3
    lwz     mem,mS02(r1)
        add     scrn,S14,L1n
    add     scr,A,L1
        add     L2n,L2n,scrn
    add     L2,L2,scr
        rotlw   L2n,L2n,scrn        # Finished Round 1.14
    add     eB,eB,A

# Register recap:
# r3,r4:    eA,eB
# r5,r0:    A,scr
# r6:       scrn, available for a while.
# r7-r14:   S07-S14 from round 1. Preserve these.
# r15-r24:  S15-S24 from round 2 (S25 was discarded).
# r25:      mem, used to look up S00-S14 in round 3.
# r26-r28:  L0-L2.
# r29-r31:  L0n-L2n from round 1. Preserve these.

    # Begin round 3 repetitions
    rotlw   L2,L2,scr
    add     A,A,mem         # Round 3.2
    xor     eA,eA,eB
    add     A,A,L2
    rotlwi  A,A,3
    lwz     mem,mS03(r1)
    rotlw   eA,eA,eB
    add     scr,A,L2
    add     eA,eA,A
    add     L0,L0,scr

    rotlw   L0,L0,scr
    add     A,A,mem         # Round 3.3
    xor     eB,eB,eA
    add     A,A,L0
    rotlwi  A,A,3
    lwz     mem,mS04(r1)
    rotlw   eB,eB,eA
    add     scr,A,L0
    add     eB,eB,A
    add     L1,L1,scr

    rotlw   L1,L1,scr
    add     A,A,mem         # Round 3.4
    xor     eA,eA,eB
    add     A,A,L1
    rotlwi  A,A,3
    lwz     mem,mS05(r1)
    rotlw   eA,eA,eB
    add     scr,A,L1
    add     eA,eA,A
    add     L2,L2,scr

    rotlw   L2,L2,scr
    add     A,A,mem         # Round 3.5
    xor     eB,eB,eA
    add     A,A,L2
    rotlwi  A,A,3
    lwz     mem,mS06(r1)
    rotlw   eB,eB,eA
    add     scr,A,L2
    add     eB,eB,A
    add     L0,L0,scr

    rotlw   L0,L0,scr
    add     A,A,mem         # Round 3.6
    xor     eA,eA,eB
    add     A,A,L0
    rotlwi  A,A,3
    lwz     mem,mS07(r1)
    rotlw   eA,eA,eB
    add     scr,A,L0
    add     eA,eA,A
    add     L1,L1,scr

    rotlw   L1,L1,scr
    add     A,A,mem         # Round 3.7
    xor     eB,eB,eA
    add     A,A,L1
    rotlwi  A,A,3
    lwz     mem,mS08(r1)
    rotlw   eB,eB,eA
    add     scr,A,L1
    add     eB,eB,A
    add     L2,L2,scr

    rotlw   L2,L2,scr
    add     A,A,mem         # Round 3.8
    xor     eA,eA,eB
    add     A,A,L2
    rotlwi  A,A,3
    lwz     mem,mS09(r1)
    rotlw   eA,eA,eB
    add     scr,A,L2
    add     eA,eA,A
    add     L0,L0,scr

    rotlw   L0,L0,scr
    add     A,A,mem         # Round 3.9
    xor     eB,eB,eA
    add     A,A,L0
    rotlwi  A,A,3
    lwz     mem,mS10(r1)
    rotlw   eB,eB,eA
    add     scr,A,L0
    add     eB,eB,A
    add     L1,L1,scr

    rotlw   L1,L1,scr
    add     A,A,mem         # Round 3.10
    xor     eA,eA,eB
    add     A,A,L1
    rotlwi  A,A,3
    lwz     mem,mS11(r1)
    rotlw   eA,eA,eB
    add     scr,A,L1
    add     eA,eA,A
    add     L2,L2,scr

    rotlw   L2,L2,scr
    add     A,A,mem         # Round 3.11
    xor     eB,eB,eA
    add     A,A,L2
    rotlwi  A,A,3
    lwz     mem,mS12(r1)
    rotlw   eB,eB,eA
    add     scr,A,L2
    add     eB,eB,A
    add     L0,L0,scr

    rotlw   L0,L0,scr
    add     A,A,mem         # Round 3.12
    xor     eA,eA,eB
    add     A,A,L0
    rotlwi  A,A,3
    lwz     mem,mS13(r1)
    rotlw   eA,eA,eB
    add     scr,A,L0
    add     eA,eA,A
    add     L1,L1,scr

    rotlw   L1,L1,scr
    add     A,A,mem         # Round 3.13
    xor     eB,eB,eA
    add     A,A,L1
    rotlwi  A,A,3
    lwz     mem,mS14(r1)
    rotlw   eB,eB,eA
    add     scr,A,L1
    add     eB,eB,A
    add     L2,L2,scr

    rotlw   L2,L2,scr
    add     A,A,mem         # Round 3.14
    xor     eA,eA,eB
    add     A,A,L2
    rotlwi  A,A,3
        addis     scrn,S14,P15Q@ha  # Round 1.15
    rotlw   eA,eA,eB
    add     scr,A,L2
    add     eA,eA,A
    add     L0,L0,scr

    rotlw   L0,L0,scr
        addi    scrn,scrn,P15Q@l
    add     A,S15,A             # Round 3.15
        add     scrn,scrn,L2n
        rotlwi  S15,scrn,3
    add     A,A,L0
    rotlwi  A,A,3
        add     scrn,S15,L2n
    xor     eB,eB,eA
        add     L0n,L0n,scrn
        rotlw   L0n,L0n,scrn
    add     scr,A,L0
    rotlw   eB,eB,eA
        addis   scrn,S15,P16Q@ha    # Round 1.16
    add     eB,eB,A
    add     L1,L1,scr

    rotlw   L1,L1,scr
        addi    scrn,scrn,P16Q@l
    add     A,S16,A             # Round 3.16
        add     scrn,scrn,L0n
        rotlwi  S16,scrn,3
    add     A,A,L1
    rotlwi  A,A,3
        add     scrn,S16,L0n
    xor     eA,eA,eB
        add     L1n,L1n,scrn
        rotlw   L1n,L1n,scrn
    add     scr,A,L1
    rotlw   eA,eA,eB
        addis   scrn,S16,P17Q@ha    # Round 1.17
    add     eA,eA,A
    add     L2,L2,scr

    rotlw   L2,L2,scr
        addi    scrn,scrn,P17Q@l
    add     A,S17,A             # Round 3.17
        add     scrn,scrn,L1n
        rotlwi  S17,scrn,3
    add     A,A,L2
    rotlwi  A,A,3
        add     scrn,S17,L1n
    xor     eB,eB,eA
        add     L2n,L2n,scrn
        rotlw   L2n,L2n,scrn
    add     scr,A,L2
    rotlw   eB,eB,eA
        addis   scrn,S17,P18Q@ha    # Round 1.18
    add     eB,eB,A
    add     L0,L0,scr

    rotlw   L0,L0,scr
        addi    scrn,scrn,P18Q@l
    add     A,S18,A             # Round 3.18
        add     scrn,scrn,L2n
        rotlwi  S18,scrn,3
    add     A,A,L0
    rotlwi  A,A,3
        add     scrn,S18,L2n
    xor     eA,eA,eB
        add     L0n,L0n,scrn
        rotlw   L0n,L0n,scrn
    add     scr,A,L0
    rotlw   eA,eA,eB
        addis   scrn,S18,P19Q@ha    # Round 1.19
    add     eA,eA,A
    add     L1,L1,scr

    rotlw   L1,L1,scr
        addi    scrn,scrn,P19Q@l
    add     A,S19,A             # Round 3.19
        add     scrn,scrn,L0n
        rotlwi  S19,scrn,3
    add     A,A,L1
    rotlwi  A,A,3
        add     scrn,S19,L0n
    xor     eB,eB,eA
        add     L1n,L1n,scrn
        rotlw   L1n,L1n,scrn
    add     scr,A,L1
    rotlw   eB,eB,eA
        addis   scrn,S19,P20Q@ha   # Round 1.20
    add     eB,eB,A
    add     L2,L2,scr

    rotlw   L2,L2,scr
        addi    scrn,scrn,P20Q@l
    add     A,S20,A             # Round 3.20
        add     scrn,scrn,L1n
        rotlwi  S20,scrn,3
    add     A,A,L2
    rotlwi  A,A,3
        add     scrn,S20,L1n
    xor     eA,eA,eB
        add     L2n,L2n,scrn
        rotlw   L2n,L2n,scrn
    add     scr,A,L2
    rotlw   eA,eA,eB
        addis   scrn,S20,P21Q@ha   # Round 1.21
    add     eA,eA,A
    add     L0,L0,scr

    rotlw   L0,L0,scr
        addi    scrn,scrn,P21Q@l
    add     A,S21,A             # Round 3.21
        add     scrn,scrn,L2n
        rotlwi  S21,scrn,3
    add     A,A,L0
    rotlwi  A,A,3
        add     scrn,S21,L2n
    xor     eB,eB,eA
        add     L0n,L0n,scrn
        rotlw   L0n,L0n,scrn
    add     scr,A,L0
    rotlw   eB,eB,eA
        addis   scrn,S21,P22Q@ha    # Round 1.22
    add     eB,eB,A
    add     L1,L1,scr

    rotlw   L1,L1,scr
        addi    scrn,scrn,P22Q@l
    add     A,S22,A             # Round 3.22
        add     scrn,scrn,L0n
        rotlwi  S22,scrn,3
    add     A,A,L1
    rotlwi  A,A,3
        add     scrn,S22,L0n
    xor     eA,eA,eB
        add     L1n,L1n,scrn
        rotlw   L1n,L1n,scrn
    add     scr,A,L1
    rotlw   eA,eA,eB
        addis   scrn,S22,P23Q@ha    # Round 1.23
    add     eA,eA,A
    add     L2,L2,scr

    rotlw   L2,L2,scr
        addi    scrn,scrn,P23Q@l
    add     A,S23,A             # Round 3.23
        add     scrn,scrn,L1n
        rotlwi  S23,scrn,3
    add     A,A,L2
    rotlwi  A,A,3
        add     scrn,S23,L1n
    xor     eB,eB,eA
        add     L2n,L2n,scrn
    add     scr,A,L2
        rotlw   L2,L2n,scrn         # Transfer L2n into L2
    rotlw   eB,eB,eA
        addis   scrn,S23,P24Q@ha    # Round 1.24
    add     eB,eB,A
    add     L0,L0,scr

    rotlw   L0,L0,scr
        addi    scrn,scrn,P24Q@l
    add     A,S24,A             # Round 3.24
        add     scrn,scrn,L2    # L2n already transferred to L2
        rotlwi  S24,scrn,3
    add     A,A,L0              # Finished with L0
    rotlwi  A,A,3
        add     scrn,S24,L2     
    xor     eA,eA,eB
        add     L0,L0n,scrn     # Transfer L0n into L0
 
        rotlw   L0,L0,scrn
        addis   S25,S24,P25Q@ha
        addi    S25,S25,P25Q@l
    lwz     scr,cypher_lo(r1)
    rotlw   eA,eA,eB
        add     S25,S25,L0
        rotlwi  S25,S25,3
    add     eA,eA,A             # And that's all for round 3

    # Get set up for Round 2
    cmplw   scr,eA
    lwz     r3,mP0QR3(r1)
        add     scrn,S25,L0
        lwz     L2n,L0_hi(r1)
        add     L1,L1n,scrn     # Transfer L1n into L1
    lwz     r4,S1X(r1)
        rotlw   L1,L1,scrn
    lwz     r5,S2X(r1)

    bdnzf+  eq,mainloop

    # Reaching here, some of the following are true:
    # eA = cypher_lo so we must check eB (this is iff CR0:eq is set);
    # ctr has reached zero and LC has too: time to exit;
    # Most likely: ctr == 0, LC > 0: redo constants and return to outer loop

    # Registers:
    # S07-S25 (r7-r25) are holding round 1 values and must be preserved
    # r3-r5 are holding S00-S02 for round 2, so should be preserved
    #       (but could be reloaded)
    # r26-r28 are holding L0-L2 and must be preserved
    # r31 holds L0_hi, which may be reloaded
    # r0=scr,r6=scrn,r29-r30 are free

    lwz     r6,LC(r1)

    beq-    keycheck
    # If cypher_hi also matches, or ctr != 0, keycheck doesn't return here

    # Return here if keycheck finds ctr is 0

ctrzero:
    # As cypher_lo didn't match, now know that ctr is 0
    # After this point, we don't need to preserve r31 as it will be
    # set to -1 before returning to outerloop

    li      r0,L0_mid
    cmplwi  r6,0            # Is overall loop count == 0?
    lwbrx   r30,r1,r0      # Get byte-reversed key_mid

    beq-    nothing_found   # Loop count 0 -> time to leave
    # (The key in UnitWork is the one which just did round 1.
    #  This work is wasted and the same key is used to begin the next
    #  function call, hence, no adjustments needed.)

    # Now we mangle increment key_mid and redo constants
    # Leave -1 (0xffffffff) in key_hi, so it increments to 0 on next loop

    addic.  r30,r30,1
    stwbrx  r30,r1,r0
    bne+    l1calc          # Not zero -> L0_mid didn't carry -> do l1calc

    # Otherwise, increment L0 too (in practice, I don't think this happens)
    li      r0,L0_lo
    lwbrx   r30,r1,r0
    addi    r30,r30,1
    stwbrx  r30,r1,r0

    # Calculate L0- and L1-constants
    # (Uses regs r0, r29-r31)
    # Note that we store r30=L0.lo byte-reversed but read it back normally
    # DO NOT "optimise" this by removing the lwz!
    # (By all means optimise it by doing a fast byte-reversal in registers.)

    lwz     r30,L0_lo(r1)   # r30=L0.lo=L0
    lwz     r31,mP0QR3(r1)  # r31=P0QR3

    add     r30,r30,r31

    rotlwi  r30,r30,29      # r30=(L0+P0QR3)<<(P0QR3)=L0X

    addi    r31,r30,PR3Q@l
    stw     r30,L0X(r1)

    addis   r31,r31,PR3Q@ha
    rotlwi  r31,r31,3       # r31=S1X

    add     r29,r30,r31     # r29=L0XS1X
    stw     r31,S1X(r1)
    addi    r30,r31,P2Q@l
    stw     r29,L0XS1X(r1)

    addis   r30,r30,P2Q@ha  # r30=S1XP2Q
    stw     r30,S1XP2Q(r1)

    # L1-constants
    # Note this is usually reached after just incrementing L0_mid,
    # so don't assume anything about regs from the L0 code above

l1calc:
    lwz     r29,L0XS1X(r1)
    lwz     r31,L0_mid(r1)
    lwz     r30,S1XP2Q(r1)
    add     r31,r31,r29
    rotlw   r31,r31,r29     # r31=L1X

    add     r30,r30,r31
    stw     r31,L1X(r1)

    rotlwi  r30,r30,3       # r30=S2X
    add     r31,r30,r31     # r31=L1XS2X
    stw     r30,S2X(r1)

    addi    r30,r30,P3Q@l
    stw     r31,L1XS2X(r1)

    addis   r30,r30,P3Q@ha  # r30=S2XP3Q
    stw     r30,S2XP3Q(r1)

    li      r31,-1      # Leave -1 in key_hi

    b       outerloop

nothing_found:
    li      r3,RESULT_NOTHING

ruf_exit:
    # Restore stack copy of UnitWork to original
    # Skip first 4 words as plaintext, cyphertext never change
    lwz     r5,UWP(r1)
    lmw     r25,UW_copy + 16(r1)
    stmw    r25,16(r5)

    lmw     r13,save_regs(r1)
    la      r1,var_size(r1)
    blr

keycheck:
    # This gets called whenever eA = cypher_lo
    # It updates the RC5_72UnitWork.check fields, then redoes the key to find
    # eB (yes, this core contains 2 separate checking routines)
    # If eB matches cypher_hi, exit with success (filling in the iter count)
    # If not, check the ctr value
    # If ctr != 0, go back to start of mainloop
    # If ctr == 0, jump to ctrzero (as if eA hadn't matched)

.set    Sr,r24  # S[] read pointer
.set    Sw,r23  # S[] write pointer

# keycheck uses 236 bytes of stack:
# 4 bytes = old SP
# 204 bytes = 26 + 25 words of S
# 28 bytes = 7 preserved registers

.set    kcstack,236
.set    kcsave_regs,208

    # Preserve registers (we only use 12, plus one to preserve ctr,
    # and r0, r3-r5,r29,r30 don't need preserving)
    stwu    r1,-kcstack(r1)
    stw     r21,kcsave_regs(r1)     # ctr (preserved)
    stw     r22,kcsave_regs+4(r1)   # loop counts for keycheck
    stw     r23,kcsave_regs+8(r1)   # Sw
    stw     r24,kcsave_regs+12(r1)  # Sr
    stw     r26,kcsave_regs+16(r1)  # L0
    stw     r27,kcsave_regs+20(r1)  # L1
    stw     r28,kcsave_regs+24(r1)  # L2
    mfctr   r21

    la      Sr,0(r1)    # Set up pointers to S (actually 1 word before start)
    mr      Sw,Sr

    lis     PnQ,P2Q@h   # Init P+nQ (to P+2Q) and Q
    ori     PnQ,PnQ,P2Q@l
    lis     rQ,Q@h
    ori     rQ,rQ,Q@l

    # Fill in check fields of RC5_72UnitWork
    # Need to mangle-decrement key for check fields and extended test

    li      r3,kcstack + L0_lo
    lwbrx   L0,r3,r1
    li      r4,kcstack + L0_mid
    lwbrx   L1,r4,r1
    lwz     L2,kcstack + L0_hi(r1)

    cmplwi  L2,0
    subi    L2,L2,1
    bne     check_store

    # If L2 was 0 before decrement, it carried.
    # So, clean it up and decrement L0_mid
    andi.   L2,L2,0xff

    cmplwi  L1,0
    subi    L1,L1,1
    bne     check_store

    subi    L0,L0,1

check_store:
    li      r3,kcstack + check_lo
    stwbrx  L0,r3,r1
    li      r4,kcstack + check_mid
    stwbrx  L1,r4,r1
    stw     L2,kcstack + check_hi(r1)

    lwz     r22,kcstack + check_count(r1)
    addi    r22,r22,1
    stw     r22,kcstack + check_count(r1)

    # Refetch correct key in appropriate byte order
    lwz     L0,kcstack + check_lo(r1)
    lwz     L1,kcstack + check_mid(r1)
    lwz     L2,kcstack + check_hi(r1)

    # Round 1, Iteration 0, second half
    lwz     scr,kcstack + mP0QR3(r1)  # scr=P0QR3, the value of A and S[0] too
    add     L0,L0,scr
    rotlw   L0,L0,scr

    # Iteration 1
    lis     A,PR3Q@h        # Init A to P<<3 + P + Q
    ori     A,A,PR3Q@l

    add     A,A,L0
    rotlwi  A,A,3
    stwu    A,4(Sw)

    add     scr,A,L0
    add     L1,L1,scr
    rotlw   L1,L1,scr

    # Iter 2
    # Now A,L0-2 are initialised, we can jump into loop

    li      r22,8
    mtctr   r22     # Set loop counter

    b       round1entry

round1loop:

    # Iters 2,5,8,11,14,17,20,23
    add     PnQ,PnQ,rQ

round1entry:    # During iteration 2
    add     A,A,PnQ
    add     A,A,L1
    rotlwi  A,A,3
    stwu    A,4(Sw)

    add     scr,A,L1
    add     L2,L2,scr
    rotlw   L2,L2,scr

    # Iteration 3,6,9,12,15,18,21,24
    add     PnQ,PnQ,rQ

    add     A,A,L2
    add     A,A,PnQ
    rotlwi  A,A,3
    stwu    A,4(Sw)

    add     scr,A,L2
    add     L0,L0,scr
    rotlw   L0,L0,scr

    #Iteration 4,7,10,13,16,19,22,25
    add     PnQ,PnQ,rQ

    add     A,A,PnQ
    add     A,A,L0
    rotlwi  A,A,3
    stwu    A,4(Sw)

    add     scr,A,L0
    add     L1,L1,scr
    rotlw   L1,L1,scr

    bdnz    round1loop

    # --- Round 2 ---

    # Iter 0 - S[0] is not in memory; it is always P0QR3

    lwz     scr,kcstack + mP0QR3(r1)

    # Jump into loop
    li      r22,9
    mtctr   r22

    b       round2entry

round2loop:
    stwu    A,4(Sw)

    add     scr,A,L0
    add     L1,L1,scr
    rotlw   L1,L1,scr

    # Iter 0,3,6,9,12,15,18,21,24
    lwzu    scr,4(Sr)

round2entry:
    add     A,A,L1
    add     A,A,scr
    rotlwi  A,A,3
    stwu    A,4(Sw)

    add     scr,A,L1
    add     L2,L2,scr
    rotlw   L2,L2,scr

    # Iter 1,4,7,10,13,16,19,22,25
    lwzu    scr,4(Sr)
    add     A,A,L2
    add     A,A,scr
    rotlwi  A,A,3
    stwu    A,4(Sw)

    add     scr,A,L2
    add     L0,L0,scr
    rotlw   L0,L0,scr

    # Iter 2,5,8,11,14,17,20,23 and round 3 iter 0
    lwzu    scr,4(Sr)
    add     A,A,L0
    add     A,A,scr
    rotlwi  A,A,3

    bdnz    round2loop

    # --- Round 3 / Encryption ---
    lwz     eA,kcstack + plain_lo(r1)
    lwz     eB,kcstack + plain_hi(r1)

    add     eA,eA,A

    add     scr,A,L0
    add     L1,L1,scr
    rotlw   L1,L1,scr

    # Iter 1

    lwzu    scr,4(Sr)
    add     A,A,L1
    add     A,A,scr
    rotlwi  A,A,3

    add     eB,eB,A

    li      r22,4
    mtctr   r22

round3loop:
    add     scr,A,L1
    add     L2,L2,scr
    rotlw   L2,L2,scr

    # Iter 2,8,14,20
    lwzu    scr,4(Sr)
    add     A,A,L2
    xor     eA,eA,eB
    add     A,A,scr
    rotlwi  A,A,3
    rotlw   eA,eA,eB
    add     eA,eA,A

    add     scr,A,L2
    add     L0,L0,scr
    rotlw   L0,L0,scr

    # Iter 3,9,15,21
    lwzu    scr,4(Sr)
    add     A,A,L0
    xor     eB,eB,eA
    add     A,A,scr
    rotlwi  A,A,3
    rotlw   eB,eB,eA
    add     eB,eB,A

    add     scr,A,L0
    add     L1,L1,scr
    rotlw   L1,L1,scr

    # Iter 4,10,16,22
    lwzu    scr,4(Sr)
    add     A,A,L1
    xor     eA,eA,eB
    add     A,A,scr
    rotlwi  A,A,3
    rotlw   eA,eA,eB
    add     eA,eA,A

    add     scr,A,L1
    add     L2,L2,scr
    rotlw   L2,L2,scr

    # Iter 5,11,17,23
    lwzu    scr,4(Sr)
    add     A,A,L2
    xor     eB,eB,eA
    add     A,A,scr
    rotlwi  A,A,3
    rotlw   eB,eB,eA
    add     eB,eB,A

    add     scr,A,L2
    add     L0,L0,scr
    rotlw   L0,L0,scr

    # Iter 6,12,18,24
    lwzu    scr,4(Sr)
    add     A,A,L0
    xor     eA,eA,eB
    add     A,A,scr
    rotlwi  A,A,3
    rotlw   eA,eA,eB
    add     eA,eA,A

    add     scr,A,L0
    add     L1,L1,scr
    rotlw   L1,L1,scr

    # Iter 7,13,19,25
    lwzu    scr,4(Sr)
    add     A,A,L1
    xor     eB,eB,eA
    add     A,A,scr
    rotlwi  A,A,3
    rotlw   eB,eB,eA
    add     eB,eB,A

    bdnz    round3loop

    # Now got ciphertext in eA, eB
    # Actually, we already know that eA matches, so just check eB
    lwz     L1,kcstack + cypher_hi(r1)

    cmplw   eB,L1

    beq     success_exit

    # If they didn't match, we need to return to the main loop
    # Exactly where we return depends on whether ctr == 0

    cmplwi  r21,0       # ctr was preserved in r21

    # Restore regs
    mtctr   r21
    lwz     r21,kcsave_regs(r1)
    lwz     r22,kcsave_regs+4(r1)
    lwz     r23,kcsave_regs+8(r1)
    lwz     r24,kcsave_regs+12(r1)
    lwz     r26,kcsave_regs+16(r1)
    lwz     r27,kcsave_regs+20(r1)
    lwz     r28,kcsave_regs+24(r1)

    la      r1,kcstack(r1)

    lwz     r3,mP0QR3(r1)
    lwz     r4,S1X(r1)
    lwz     r5,S2X(r1)

    bne+    mainloop    # If ctr != 0, go back to start of mainloop

    b       ctrzero     # If ctr is 0, go to handling code after loop

success_exit:
    # This is where both words of cyphertext match
    # Need to fill in the iteration count, straighten out the stack pointer,
    # and return success.

    la      r1,kcstack(r1)      # Return SP to normal place

    lwz     r4,ILCP(r1)
    lwz     r7,0(r4)            # Get initial iter count in r7

    # LC is in r6 already, and ctr is in r21

    subi    r7,r7,1             # Final value would be 1 too high after bdnzf
    add     r6,r6,r21           # Loops still to do = LC + ctr

    subf    r7,r6,r7            # r7 is now loops completed before success
    stw     r7,0(r4)

    li      r3,RESULT_FOUND
    b       ruf_exit
