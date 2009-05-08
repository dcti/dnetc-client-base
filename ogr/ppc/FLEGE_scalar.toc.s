#
# Copyright distributed.net 1999-2008 - All Rights Reserved
# For use in distributed.net projects only.
# Any other distribution or use of this source violates copyright.
#
# PPC Scalar core - 256 bits OGR core for PowerPC processors.
# Code designed for G3 (MPC750)
# Written by Didier Levet <kakace@distributed.net>
#
# $Id: FLEGE_scalar.toc.s,v 1.3 2009/05/08 16:32:52 kakace Exp $
#
#============================================================================
# Special notes :
# - The code extensively use simplified mnemonics.
# - Use a custom stack frame (leaf procedure).
# - LR register not used nor saved in caller's stack.
# - CTR, CR0, CR1, CR7, GPR0 and GPR3-GPR12 are volatile (not preserved).
#
#============================================================================


    .globl    .cycle_ppc_scalar_256     # coff
    .globl    cycle_ppc_scalar_256      # elf


# OrgNgLevel dependencies (offsets)
.set          Levels_list,        0     # list bitmap
.set          Levels_dist,        32    # dist bitmap
.set          Levels_comp,        64    # comp bitmap
.set          mark,               96
.set          limit,              100
.set          SIZEOF_LEVEL,       104
.set          NextLev_dist,       Levels_dist+SIZEOF_LEVEL # dist bitmap (next level)


# OgrNgState dependencies (offsets)
.set          MaxLength,          0     # oState->max
.set          MaxDepth,           8     # oState->maxdepthm1
.set          MidSegA,            12    # oState->half_depth
.set          MidSegB,            16    # oState->half_depth2
.set          StopDepth,          24    # oState->stopdepth
.set          Depth,              28    # oState->depth
.set          Levels,             32    # oState->Levels[]


# rlwinm arguments
.set          CHOOSE_BITS,        16
.set          SH,                 CHOOSE_BITS+6
.set          MB,                 32-CHOOSE_BITS-6
.set          ME,                 31-6


#============================================================================
# Custom stack frame

.set          FIRST_NV_GPR, 13          # Save r13..r31
.set          GPRSaveArea, (32-FIRST_NV_GPR) * 4

.set          localTop, 48

.set          oState, 16                # struct OgrNgState*
.set          pNodes, 20                # int* pNodes
.set          gpr2, 24
.set          maxLenM1, 28              # oState->max - 1
.set          pLevelA, 32               # &Levels[half_depth]
.set          FrameSize, (localTop + GPRSaveArea + 15) & (-16)

#============================================================================
# Register aliases

.set          r0,0
.set          r1,1
.set          RTOC,2                    # AIX
.set          r3,3
.set          r4,4
.set          r5,5
.set          r6,6
.set          r7,7
.set          r8,8
.set          r9,9
.set          r10,10
.set          r11,11
.set          r12,12
.set          r13,13
.set          r14,14
.set          r15,15
.set          r16,16
.set          r17,17
.set          r18,18
.set          r19,19
.set          r20,20
.set          r21,21
.set          r22,22
.set          r23,23
.set          r24,24
.set          r25,25
.set          r26,26
.set          r27,27
.set          r28,28
.set          r29,29
.set          r30,30
.set          r31,31


#============================================================================
# int cycle_ppc_scalar_256(struct OgrNgState* oState (r3)
#                          int*               pNodes (r4)
#                          const u16*         pChoose (r5)

    # Add new TOC entries
    .toc      

T.sw_addr:    .tc flege_scalar_core[TC], L_switch_cases-64

    .csect    .cycle_ppc_scalar_256[DS]

    # Set the TOC anchor
cycle_ppc_scalar_256:
    .long     .cycle_ppc_scalar_256, TOC[tc0], 0

    .csect    flege_scalar_core[PR]
    .align    4

.cycle_ppc_scalar_256:
    mr        r10,r1                    # Caller's stack pointer
    clrlwi    r12,r1,27                 # keep the low-order 4 bits
    subfic    r12,r12,-FrameSize        # Frame size, including padding
    stwux     r1,r1,r12

    # Save non-volatile registers
    stmw      r13,-GPRSaveArea(r10)     # Save GPRs


#============================================================================
# Core engine initialization - Registers allocation :
#   r0  := c0neg or temp3
#   r2  := JumpTable (const)
#   r3  := level
#   r4  := nodes
#   r5  := const u16* pChoose[]
#   r6  := Depth
#   r7  := mark
#   r8  := limit
#   r9  := half_depth (const)
#   r10 := half_depth2 (const)
#   r11 := stopdepth (const)
#   r12 := maxdepthm1 (const)
#   r13 := dist0 or newbit
#   r14 := temp1
#   r15 := temp2
#   r16 := list0
#   r17 := list1
#   r18 := list2
#   r19 := list3
#   r20 := list4
#   r21 := list5
#   r22 := list6
#   r23 := list7
#   r24 := comp0
#   r25 := comp1
#   r26 := comp2
#   r27 := comp3
#   r28 := comp4
#   r29 := comp5
#   r30 := comp6
#   r31 := comp7

    # Initialize local variables.
    stw       r3,oState(r1)             # store oState
    stw       r4,pNodes(r1)             # store pNodes
    stw       r2,gpr2(r1)               # store GPR2
    lwz       r9,MidSegA(r3)            # oState->half_depth
    lwz       r10,MidSegB(r3)           # oState->half_depth2
    lwz       r23,MaxLength(r3)         # oState->max
    lwz       r6,Depth(r3)              # oState->depth
    lwz       r11,StopDepth(r3)         # oState->stopdepth
    lwz       r12,MaxDepth(r3)          # oState->maxdepthm1
    lwz       r4,0(r4)                  # Load nodes count
    subi      r23,r23,1                 # max -= 1
    stw       r23,maxLenM1(r1)

    lwz       r2,T.sw_addr(RTOC)        # Jump table base address

    # Compute the base pointer to access pre-computed limits
    # This pointer always points to pChoose[0][Depth]
    add       r14,r6,r6                 # = depth * 2
    add       r5,r5,r14                 # &pChoose[0][Depth]

    # Compute the pointer to Levels[half_depth]
    mulli     r14,r9,SIZEOF_LEVEL       # half_depth * SIZEOF_LEVEL
    add       r14,r3,r14                # + oState
    addi      r14,r14,Levels            # + Levels = &Levels[half_depth]
    stw       r14,pLevelA(r1)

    # Compute the pointer to Levels[Depth]
    mulli     r14,r6,SIZEOF_LEVEL       # Depth * SIZEOF_LEVEL
    add       r14,r14,r3                # + oState
    addi      r3,r14,Levels             # + Levels = &Levels[Depth]

    cmpw      r6,r12                    # Depth == MaxDepth ?

    # Load state.
    lwz       r7,mark(r3)               # Levels[Depth].mark
    lwz       r8,limit(r3)              # Levels[Depth].limit
    lwz       r16,Levels_list+0(r3)     # list[0]
    lwz       r17,Levels_list+4(r3)     # list[1]
    lwz       r18,Levels_list+8(r3)     # list[2]
    lwz       r19,Levels_list+12(r3)    # list[3]
    lwz       r20,Levels_list+16(r3)    # list[4]
    lwz       r21,Levels_list+20(r3)    # list[5]
    lwz       r22,Levels_list+24(r3)    # list[6]
    lwz       r23,Levels_list+28(r3)    # list[7]
    lwz       r24,Levels_comp+0(r3)     # comp[0]
    lwz       r25,Levels_comp+4(r3)     # comp[1]
    lwz       r26,Levels_comp+8(r3)     # comp[2]
    lwz       r27,Levels_comp+12(r3)    # comp[3]
    lwz       r28,Levels_comp+16(r3)    # comp[4]
    lwz       r29,Levels_comp+20(r3)    # comp[5]
    lwz       r30,Levels_comp+24(r3)    # comp[6]
    lwz       r31,Levels_comp+28(r3)    # comp[7]
    li        r13,0                     # newbit = 0
    not       r0,r24                    # ~comp0
    srwi      r0,r0,1                   # (~comp0) >> 1
    beq       L_MainLoop                # Depth == MaxDepth

    li        r13,1                     # newbit = 1
    b         L_MainLoop


#============================================================================

    .align    4
    nop       

L_UpLevel:
    # Backtrack to the preceeding level.
    # r3  := pLevel
    # r6  := depth
    # r11 := stopdepth
    # r10 := half_depth2 = MidSegB
    # r5  := pChoose
    subi      r3,r3,SIZEOF_LEVEL        # --Levels
    subi      r6,r6,1                   # --Depth
    lwz       r24,Levels_comp(r3)       # comp0
    cmpw      r6,r11                    # Depth > StopDepth
    lwz       r8,limit(r3)              # limit = Levels[Depth].limit
    subi      r5,r5,2                   # --pChoose
    lwz       r7,mark(r3)               # mark = Levels[Depth].mark
    not       r0,r24                    # c0neg = ~comp0
    lwz       r16,Levels_list+0(r3)     # list[0]
    lwz       r17,Levels_list+4(r3)     # list[1]
    lwz       r18,Levels_list+8(r3)     # list[2]
    lwz       r19,Levels_list+12(r3)    # list[3]
    lwz       r20,Levels_list+16(r3)    # list[4]
    lwz       r21,Levels_list+20(r3)    # list[5]
    lwz       r22,Levels_list+24(r3)    # list[6]
    lwz       r23,Levels_list+28(r3)    # list[7]
    srwi      r0,r0,1                   # c0neg = (~comp0) >> 1
    lwz       r25,Levels_comp+4(r3)     # comp[1]
    lwz       r26,Levels_comp+8(r3)     # comp[2]
    lwz       r27,Levels_comp+12(r3)    # comp[3]
    lwz       r28,Levels_comp+16(r3)    # comp[4]
    lwz       r29,Levels_comp+20(r3)    # comp[5]
    lwz       r30,Levels_comp+24(r3)    # comp[6]
    lwz       r31,Levels_comp+28(r3)    # comp[7]
    li        r13,0                     # newbit = 0
    ble-      L_exit                    # Depth <= StopDepth

L_MainLoop:
    #  r0  := (~comp0) >> 1
    #  r8  := limit
    #  r7  := mark
    #  r24 := comp0
    #  r2  := JumpTable
    #  r13 := newbit
    cntlzw    r14,r0                    # diff
    cmpwi     cr1,r24,-1                # comp0 == 0xFFFFFFFF ?
    add       r7,r7,r14                 # mark += diff
    slwi      r15,r14,7                 # = offset to the corresponding 'case' block
    cmpw      r7,r8                     # mark > limit ?
    add       r15,r2,r15                # 'case' address
    bgt-      L_UpLevel                 # Go back to preceding mark.
    mtctr     r15
    cmpw      r6,r12                    # depth == maxdepthm1 ?
    rotlw     r24,r24,r14               # rotate comp0
    bctr                                # Go shift the bitmaps

    .align    4
L_SwitchCase:
    # Shift bitmaps by 'diff' bits.
    # r14 := diff
    # cr0 := depth == maxdepthm1
    # cr1 := comp0 == 0xFFFFFFFF
    # diff == 1
    rotlwi    r25,r25,1                 # rotate comp1
    rotlwi    r26,r26,1                 # rotate comp2
    rotlwi    r27,r27,1                 # rotate comp3
    rotlwi    r28,r28,1                 # rotate comp4
    rotlwi    r29,r29,1                 # rotate comp5
    rotlwi    r30,r30,1                 # rotate comp6
    rotrwi    r16,r16,1                 # rotate list0
    rotrwi    r17,r17,1                 # rotate list1
    rotrwi    r18,r18,1                 # rotate list2
    rotrwi    r19,r19,1                 # rotate list3
    rotrwi    r20,r20,1                 # rotate list4
    rotrwi    r21,r21,1                 # rotate list5
    rotrwi    r22,r22,1                 # rotate list6
    rotrwi    r23,r23,1                 # rotate list7
    rlwimi    r24,r25,0,31,31           # comp0 = comp0:comp1 << 1
    rlwimi    r25,r26,0,31,31           # comp1 = comp1:comp2 << 1
    rlwimi    r26,r27,0,31,31           # comp2 = comp2:comp3 << 1
    rlwimi    r27,r28,0,31,31           # comp3 = comp3:comp4 << 1
    rlwimi    r28,r29,0,31,31           # comp4 = comp4:comp5 << 1
    rlwimi    r29,r30,0,31,31           # comp5 = comp5:comp6 << 1
    rlwimi    r30,r31,1,31,31           # comp6 = comp6:comp7 << 1
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,0             # list7 = list6:list7 >> 1
    rlwimi    r22,r21,0,0,0             # list6 = list5:list6 >> 1
    rlwimi    r21,r20,0,0,0             # list5 = list4:list5 >> 1
    rlwimi    r20,r19,0,0,0             # list4 = list3:list4 >> 1
    rlwimi    r19,r18,0,0,0             # list3 = list2:list3 >> 1
    rlwimi    r18,r17,0,0,0             # list2 = list1:list2 >> 1
    rlwimi    r17,r16,0,0,0             # list1 = list0:list1 >> 1
    rlwimi    r16,r13,31,0,0            # list0 = newbit:list0 >> 1
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 2
    rotlwi    r25,r25,2                 # rotate comp1
    rotlwi    r26,r26,2                 # rotate comp2
    rotlwi    r27,r27,2                 # rotate comp3
    rotlwi    r28,r28,2                 # rotate comp4
    rotlwi    r29,r29,2                 # rotate comp5
    rotlwi    r30,r30,2                 # rotate comp6
    rotrwi    r16,r16,2                 # rotate list0
    rotrwi    r17,r17,2                 # rotate list1
    rotrwi    r18,r18,2                 # rotate list2
    rotrwi    r19,r19,2                 # rotate list3
    rotrwi    r20,r20,2                 # rotate list4
    rotrwi    r21,r21,2                 # rotate list5
    rotrwi    r22,r22,2                 # rotate list6
    rotrwi    r23,r23,2                 # rotate list7
    rlwimi    r24,r25,0,30,31           # comp0 = comp0:comp1 << 2
    rlwimi    r25,r26,0,30,31           # comp1 = comp1:comp2 << 2
    rlwimi    r26,r27,0,30,31           # comp2 = comp2:comp3 << 2
    rlwimi    r27,r28,0,30,31           # comp3 = comp3:comp4 << 2
    rlwimi    r28,r29,0,30,31           # comp4 = comp4:comp5 << 2
    rlwimi    r29,r30,0,30,31           # comp5 = comp5:comp6 << 2
    rlwimi    r30,r31,2,30,31           # comp6 = comp6:comp7 << 2
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,1             # list7 = list6:list7 >> 2
    rlwimi    r22,r21,0,0,1             # list6 = list5:list6 >> 2
    rlwimi    r21,r20,0,0,1             # list5 = list4:list5 >> 2
    rlwimi    r20,r19,0,0,1             # list4 = list3:list4 >> 2
    rlwimi    r19,r18,0,0,1             # list3 = list2:list3 >> 2
    rlwimi    r18,r17,0,0,1             # list2 = list1:list2 >> 2
    rlwimi    r17,r16,0,0,1             # list1 = list0:list1 >> 2
    rlwimi    r16,r13,30,0,1            # list0 = newbit:list0 >> 2
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 3
    rotlwi    r25,r25,3                 # rotate comp1
    rotlwi    r26,r26,3                 # rotate comp2
    rotlwi    r27,r27,3                 # rotate comp3
    rotlwi    r28,r28,3                 # rotate comp4
    rotlwi    r29,r29,3                 # rotate comp5
    rotlwi    r30,r30,3                 # rotate comp6
    rotrwi    r16,r16,3                 # rotate list0
    rotrwi    r17,r17,3                 # rotate list1
    rotrwi    r18,r18,3                 # rotate list2
    rotrwi    r19,r19,3                 # rotate list3
    rotrwi    r20,r20,3                 # rotate list4
    rotrwi    r21,r21,3                 # rotate list5
    rotrwi    r22,r22,3                 # rotate list6
    rotrwi    r23,r23,3                 # rotate list7
    rlwimi    r24,r25,0,29,31           # comp0 = comp0:comp1 << 3
    rlwimi    r25,r26,0,29,31           # comp1 = comp1:comp2 << 3
    rlwimi    r26,r27,0,29,31           # comp2 = comp2:comp3 << 3
    rlwimi    r27,r28,0,29,31           # comp3 = comp3:comp4 << 3
    rlwimi    r28,r29,0,29,31           # comp4 = comp4:comp5 << 3
    rlwimi    r29,r30,0,29,31           # comp5 = comp5:comp6 << 3
    rlwimi    r30,r31,3,29,31           # comp6 = comp6:comp7 << 3
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,2             # list7 = list6:list7 >> 3
    rlwimi    r22,r21,0,0,2             # list6 = list5:list6 >> 3
    rlwimi    r21,r20,0,0,2             # list5 = list4:list5 >> 3
    rlwimi    r20,r19,0,0,2             # list4 = list3:list4 >> 3
    rlwimi    r19,r18,0,0,2             # list3 = list2:list3 >> 3
    rlwimi    r18,r17,0,0,2             # list2 = list1:list2 >> 3
    rlwimi    r17,r16,0,0,2             # list1 = list0:list1 >> 3
    rlwimi    r16,r13,29,0,2            # list0 = newbit:list0 >> 3
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 4
    rotlwi    r25,r25,4                 # rotate comp1
    rotlwi    r26,r26,4                 # rotate comp2
    rotlwi    r27,r27,4                 # rotate comp3
    rotlwi    r28,r28,4                 # rotate comp4
    rotlwi    r29,r29,4                 # rotate comp5
    rotlwi    r30,r30,4                 # rotate comp6
    rotrwi    r16,r16,4                 # rotate list0
    rotrwi    r17,r17,4                 # rotate list1
    rotrwi    r18,r18,4                 # rotate list2
    rotrwi    r19,r19,4                 # rotate list3
    rotrwi    r20,r20,4                 # rotate list4
    rotrwi    r21,r21,4                 # rotate list5
    rotrwi    r22,r22,4                 # rotate list6
    rotrwi    r23,r23,4                 # rotate list7
    rlwimi    r24,r25,0,28,31           # comp0 = comp0:comp1 << 4
    rlwimi    r25,r26,0,28,31           # comp1 = comp1:comp2 << 4
    rlwimi    r26,r27,0,28,31           # comp2 = comp2:comp3 << 4
    rlwimi    r27,r28,0,28,31           # comp3 = comp3:comp4 << 4
    rlwimi    r28,r29,0,28,31           # comp4 = comp4:comp5 << 4
    rlwimi    r29,r30,0,28,31           # comp5 = comp5:comp6 << 4
    rlwimi    r30,r31,4,28,31           # comp6 = comp6:comp7 << 4
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,3             # list7 = list6:list7 >> 4
    rlwimi    r22,r21,0,0,3             # list6 = list5:list6 >> 4
    rlwimi    r21,r20,0,0,3             # list5 = list4:list5 >> 4
    rlwimi    r20,r19,0,0,3             # list4 = list3:list4 >> 4
    rlwimi    r19,r18,0,0,3             # list3 = list2:list3 >> 4
    rlwimi    r18,r17,0,0,3             # list2 = list1:list2 >> 4
    rlwimi    r17,r16,0,0,3             # list1 = list0:list1 >> 4
    rlwimi    r16,r13,28,0,3            # list0 = newbit:list0 >> 4
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 5
    rotlwi    r25,r25,5                 # rotate comp1
    rotlwi    r26,r26,5                 # rotate comp2
    rotlwi    r27,r27,5                 # rotate comp3
    rotlwi    r28,r28,5                 # rotate comp4
    rotlwi    r29,r29,5                 # rotate comp5
    rotlwi    r30,r30,5                 # rotate comp6
    rotrwi    r16,r16,5                 # rotate list0
    rotrwi    r17,r17,5                 # rotate list1
    rotrwi    r18,r18,5                 # rotate list2
    rotrwi    r19,r19,5                 # rotate list3
    rotrwi    r20,r20,5                 # rotate list4
    rotrwi    r21,r21,5                 # rotate list5
    rotrwi    r22,r22,5                 # rotate list6
    rotrwi    r23,r23,5                 # rotate list7
    rlwimi    r24,r25,0,27,31           # comp0 = comp0:comp1 << 5
    rlwimi    r25,r26,0,27,31           # comp1 = comp1:comp2 << 5
    rlwimi    r26,r27,0,27,31           # comp2 = comp2:comp3 << 5
    rlwimi    r27,r28,0,27,31           # comp3 = comp3:comp4 << 5
    rlwimi    r28,r29,0,27,31           # comp4 = comp4:comp5 << 5
    rlwimi    r29,r30,0,27,31           # comp5 = comp5:comp6 << 5
    rlwimi    r30,r31,5,27,31           # comp6 = comp6:comp7 << 5
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,4             # list7 = list6:list7 >> 5
    rlwimi    r22,r21,0,0,4             # list6 = list5:list6 >> 5
    rlwimi    r21,r20,0,0,4             # list5 = list4:list5 >> 5
    rlwimi    r20,r19,0,0,4             # list4 = list3:list4 >> 5
    rlwimi    r19,r18,0,0,4             # list3 = list2:list3 >> 5
    rlwimi    r18,r17,0,0,4             # list2 = list1:list2 >> 5
    rlwimi    r17,r16,0,0,4             # list1 = list0:list1 >> 5
    rlwimi    r16,r13,27,0,4            # list0 = newbit:list0 >> 5
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 6
    rotlwi    r25,r25,6                 # rotate comp1
    rotlwi    r26,r26,6                 # rotate comp2
    rotlwi    r27,r27,6                 # rotate comp3
    rotlwi    r28,r28,6                 # rotate comp4
    rotlwi    r29,r29,6                 # rotate comp5
    rotlwi    r30,r30,6                 # rotate comp6
    rotrwi    r16,r16,6                 # rotate list0
    rotrwi    r17,r17,6                 # rotate list1
    rotrwi    r18,r18,6                 # rotate list2
    rotrwi    r19,r19,6                 # rotate list3
    rotrwi    r20,r20,6                 # rotate list4
    rotrwi    r21,r21,6                 # rotate list5
    rotrwi    r22,r22,6                 # rotate list6
    rotrwi    r23,r23,6                 # rotate list7
    rlwimi    r24,r25,0,26,31           # comp0 = comp0:comp1 << 6
    rlwimi    r25,r26,0,26,31           # comp1 = comp1:comp2 << 6
    rlwimi    r26,r27,0,26,31           # comp2 = comp2:comp3 << 6
    rlwimi    r27,r28,0,26,31           # comp3 = comp3:comp4 << 6
    rlwimi    r28,r29,0,26,31           # comp4 = comp4:comp5 << 6
    rlwimi    r29,r30,0,26,31           # comp5 = comp5:comp6 << 6
    rlwimi    r30,r31,6,26,31           # comp6 = comp6:comp7 << 6
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,5             # list7 = list6:list7 >> 6
    rlwimi    r22,r21,0,0,5             # list6 = list5:list6 >> 6
    rlwimi    r21,r20,0,0,5             # list5 = list4:list5 >> 6
    rlwimi    r20,r19,0,0,5             # list4 = list3:list4 >> 6
    rlwimi    r19,r18,0,0,5             # list3 = list2:list3 >> 6
    rlwimi    r18,r17,0,0,5             # list2 = list1:list2 >> 6
    rlwimi    r17,r16,0,0,5             # list1 = list0:list1 >> 6
    rlwimi    r16,r13,26,0,5            # list0 = newbit:list0 >> 6
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 7
    rotlwi    r25,r25,7                 # rotate comp1
    rotlwi    r26,r26,7                 # rotate comp2
    rotlwi    r27,r27,7                 # rotate comp3
    rotlwi    r28,r28,7                 # rotate comp4
    rotlwi    r29,r29,7                 # rotate comp5
    rotlwi    r30,r30,7                 # rotate comp6
    rotrwi    r16,r16,7                 # rotate list0
    rotrwi    r17,r17,7                 # rotate list1
    rotrwi    r18,r18,7                 # rotate list2
    rotrwi    r19,r19,7                 # rotate list3
    rotrwi    r20,r20,7                 # rotate list4
    rotrwi    r21,r21,7                 # rotate list5
    rotrwi    r22,r22,7                 # rotate list6
    rotrwi    r23,r23,7                 # rotate list7
    rlwimi    r24,r25,0,25,31           # comp0 = comp0:comp1 << 7
    rlwimi    r25,r26,0,25,31           # comp1 = comp1:comp2 << 7
    rlwimi    r26,r27,0,25,31           # comp2 = comp2:comp3 << 7
    rlwimi    r27,r28,0,25,31           # comp3 = comp3:comp4 << 7
    rlwimi    r28,r29,0,25,31           # comp4 = comp4:comp5 << 7
    rlwimi    r29,r30,0,25,31           # comp5 = comp5:comp6 << 7
    rlwimi    r30,r31,7,25,31           # comp6 = comp6:comp7 << 7
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,6             # list7 = list6:list7 >> 7
    rlwimi    r22,r21,0,0,6             # list6 = list5:list6 >> 7
    rlwimi    r21,r20,0,0,6             # list5 = list4:list5 >> 7
    rlwimi    r20,r19,0,0,6             # list4 = list3:list4 >> 7
    rlwimi    r19,r18,0,0,6             # list3 = list2:list3 >> 7
    rlwimi    r18,r17,0,0,6             # list2 = list1:list2 >> 7
    rlwimi    r17,r16,0,0,6             # list1 = list0:list1 >> 7
    rlwimi    r16,r13,25,0,6            # list0 = newbit:list0 >> 7
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 8
    rotlwi    r25,r25,8                 # rotate comp1
    rotlwi    r26,r26,8                 # rotate comp2
    rotlwi    r27,r27,8                 # rotate comp3
    rotlwi    r28,r28,8                 # rotate comp4
    rotlwi    r29,r29,8                 # rotate comp5
    rotlwi    r30,r30,8                 # rotate comp6
    rotrwi    r16,r16,8                 # rotate list0
    rotrwi    r17,r17,8                 # rotate list1
    rotrwi    r18,r18,8                 # rotate list2
    rotrwi    r19,r19,8                 # rotate list3
    rotrwi    r20,r20,8                 # rotate list4
    rotrwi    r21,r21,8                 # rotate list5
    rotrwi    r22,r22,8                 # rotate list6
    rotrwi    r23,r23,8                 # rotate list7
    rlwimi    r24,r25,0,24,31           # comp0 = comp0:comp1 << 8
    rlwimi    r25,r26,0,24,31           # comp1 = comp1:comp2 << 8
    rlwimi    r26,r27,0,24,31           # comp2 = comp2:comp3 << 8
    rlwimi    r27,r28,0,24,31           # comp3 = comp3:comp4 << 8
    rlwimi    r28,r29,0,24,31           # comp4 = comp4:comp5 << 8
    rlwimi    r29,r30,0,24,31           # comp5 = comp5:comp6 << 8
    rlwimi    r30,r31,8,24,31           # comp6 = comp6:comp7 << 8
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,7             # list7 = list6:list7 >> 8
    rlwimi    r22,r21,0,0,7             # list6 = list5:list6 >> 8
    rlwimi    r21,r20,0,0,7             # list5 = list4:list5 >> 8
    rlwimi    r20,r19,0,0,7             # list4 = list3:list4 >> 8
    rlwimi    r19,r18,0,0,7             # list3 = list2:list3 >> 8
    rlwimi    r18,r17,0,0,7             # list2 = list1:list2 >> 8
    rlwimi    r17,r16,0,0,7             # list1 = list0:list1 >> 8
    rlwimi    r16,r13,24,0,7            # list0 = newbit:list0 >> 8
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 9
    rotlwi    r25,r25,9                 # rotate comp1
    rotlwi    r26,r26,9                 # rotate comp2
    rotlwi    r27,r27,9                 # rotate comp3
    rotlwi    r28,r28,9                 # rotate comp4
    rotlwi    r29,r29,9                 # rotate comp5
    rotlwi    r30,r30,9                 # rotate comp6
    rotrwi    r16,r16,9                 # rotate list0
    rotrwi    r17,r17,9                 # rotate list1
    rotrwi    r18,r18,9                 # rotate list2
    rotrwi    r19,r19,9                 # rotate list3
    rotrwi    r20,r20,9                 # rotate list4
    rotrwi    r21,r21,9                 # rotate list5
    rotrwi    r22,r22,9                 # rotate list6
    rotrwi    r23,r23,9                 # rotate list7
    rlwimi    r24,r25,0,23,31           # comp0 = comp0:comp1 << 9
    rlwimi    r25,r26,0,23,31           # comp1 = comp1:comp2 << 9
    rlwimi    r26,r27,0,23,31           # comp2 = comp2:comp3 << 9
    rlwimi    r27,r28,0,23,31           # comp3 = comp3:comp4 << 9
    rlwimi    r28,r29,0,23,31           # comp4 = comp4:comp5 << 9
    rlwimi    r29,r30,0,23,31           # comp5 = comp5:comp6 << 9
    rlwimi    r30,r31,9,23,31           # comp6 = comp6:comp7 << 9
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,8             # list7 = list6:list7 >> 9
    rlwimi    r22,r21,0,0,8             # list6 = list5:list6 >> 9
    rlwimi    r21,r20,0,0,8             # list5 = list4:list5 >> 9
    rlwimi    r20,r19,0,0,8             # list4 = list3:list4 >> 9
    rlwimi    r19,r18,0,0,8             # list3 = list2:list3 >> 9
    rlwimi    r18,r17,0,0,8             # list2 = list1:list2 >> 9
    rlwimi    r17,r16,0,0,8             # list1 = list0:list1 >> 9
    rlwimi    r16,r13,23,0,8            # list0 = newbit:list0 >> 9
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 10
    rotlwi    r25,r25,10                # rotate comp1
    rotlwi    r26,r26,10                # rotate comp2
    rotlwi    r27,r27,10                # rotate comp3
    rotlwi    r28,r28,10                # rotate comp4
    rotlwi    r29,r29,10                # rotate comp5
    rotlwi    r30,r30,10                # rotate comp6
    rotrwi    r16,r16,10                # rotate list0
    rotrwi    r17,r17,10                # rotate list1
    rotrwi    r18,r18,10                # rotate list2
    rotrwi    r19,r19,10                # rotate list3
    rotrwi    r20,r20,10                # rotate list4
    rotrwi    r21,r21,10                # rotate list5
    rotrwi    r22,r22,10                # rotate list6
    rotrwi    r23,r23,10                # rotate list7
    rlwimi    r24,r25,0,22,31           # comp0 = comp0:comp1 << 10
    rlwimi    r25,r26,0,22,31           # comp1 = comp1:comp2 << 10
    rlwimi    r26,r27,0,22,31           # comp2 = comp2:comp3 << 10
    rlwimi    r27,r28,0,22,31           # comp3 = comp3:comp4 << 10
    rlwimi    r28,r29,0,22,31           # comp4 = comp4:comp5 << 10
    rlwimi    r29,r30,0,22,31           # comp5 = comp5:comp6 << 10
    rlwimi    r30,r31,10,22,31          # comp6 = comp6:comp7 << 10
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,9             # list7 = list6:list7 >> 10
    rlwimi    r22,r21,0,0,9             # list6 = list5:list6 >> 10
    rlwimi    r21,r20,0,0,9             # list5 = list4:list5 >> 10
    rlwimi    r20,r19,0,0,9             # list4 = list3:list4 >> 10
    rlwimi    r19,r18,0,0,9             # list3 = list2:list3 >> 10
    rlwimi    r18,r17,0,0,9             # list2 = list1:list2 >> 10
    rlwimi    r17,r16,0,0,9             # list1 = list0:list1 >> 10
    rlwimi    r16,r13,22,0,9            # list0 = newbit:list0 >> 10
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 11
    rotlwi    r25,r25,11                # rotate comp1
    rotlwi    r26,r26,11                # rotate comp2
    rotlwi    r27,r27,11                # rotate comp3
    rotlwi    r28,r28,11                # rotate comp4
    rotlwi    r29,r29,11                # rotate comp5
    rotlwi    r30,r30,11                # rotate comp6
    rotrwi    r16,r16,11                # rotate list0
    rotrwi    r17,r17,11                # rotate list1
    rotrwi    r18,r18,11                # rotate list2
    rotrwi    r19,r19,11                # rotate list3
    rotrwi    r20,r20,11                # rotate list4
    rotrwi    r21,r21,11                # rotate list5
    rotrwi    r22,r22,11                # rotate list6
    rotrwi    r23,r23,11                # rotate list7
    rlwimi    r24,r25,0,21,31           # comp0 = comp0:comp1 << 11
    rlwimi    r25,r26,0,21,31           # comp1 = comp1:comp2 << 11
    rlwimi    r26,r27,0,21,31           # comp2 = comp2:comp3 << 11
    rlwimi    r27,r28,0,21,31           # comp3 = comp3:comp4 << 11
    rlwimi    r28,r29,0,21,31           # comp4 = comp4:comp5 << 11
    rlwimi    r29,r30,0,21,31           # comp5 = comp5:comp6 << 11
    rlwimi    r30,r31,11,21,31          # comp6 = comp6:comp7 << 11
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,10            # list7 = list6:list7 >> 11
    rlwimi    r22,r21,0,0,10            # list6 = list5:list6 >> 11
    rlwimi    r21,r20,0,0,10            # list5 = list4:list5 >> 11
    rlwimi    r20,r19,0,0,10            # list4 = list3:list4 >> 11
    rlwimi    r19,r18,0,0,10            # list3 = list2:list3 >> 11
    rlwimi    r18,r17,0,0,10            # list2 = list1:list2 >> 11
    rlwimi    r17,r16,0,0,10            # list1 = list0:list1 >> 11
    rlwimi    r16,r13,21,0,10           # list0 = newbit:list0 >> 11
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 12
    rotlwi    r25,r25,12                # rotate comp1
    rotlwi    r26,r26,12                # rotate comp2
    rotlwi    r27,r27,12                # rotate comp3
    rotlwi    r28,r28,12                # rotate comp4
    rotlwi    r29,r29,12                # rotate comp5
    rotlwi    r30,r30,12                # rotate comp6
    rotrwi    r16,r16,12                # rotate list0
    rotrwi    r17,r17,12                # rotate list1
    rotrwi    r18,r18,12                # rotate list2
    rotrwi    r19,r19,12                # rotate list3
    rotrwi    r20,r20,12                # rotate list4
    rotrwi    r21,r21,12                # rotate list5
    rotrwi    r22,r22,12                # rotate list6
    rotrwi    r23,r23,12                # rotate list7
    rlwimi    r24,r25,0,20,31           # comp0 = comp0:comp1 << 12
    rlwimi    r25,r26,0,20,31           # comp1 = comp1:comp2 << 12
    rlwimi    r26,r27,0,20,31           # comp2 = comp2:comp3 << 12
    rlwimi    r27,r28,0,20,31           # comp3 = comp3:comp4 << 12
    rlwimi    r28,r29,0,20,31           # comp4 = comp4:comp5 << 12
    rlwimi    r29,r30,0,20,31           # comp5 = comp5:comp6 << 12
    rlwimi    r30,r31,12,20,31          # comp6 = comp6:comp7 << 12
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,11            # list7 = list6:list7 >> 12
    rlwimi    r22,r21,0,0,11            # list6 = list5:list6 >> 12
    rlwimi    r21,r20,0,0,11            # list5 = list4:list5 >> 12
    rlwimi    r20,r19,0,0,11            # list4 = list3:list4 >> 12
    rlwimi    r19,r18,0,0,11            # list3 = list2:list3 >> 12
    rlwimi    r18,r17,0,0,11            # list2 = list1:list2 >> 12
    rlwimi    r17,r16,0,0,11            # list1 = list0:list1 >> 12
    rlwimi    r16,r13,20,0,11           # list0 = newbit:list0 >> 12
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 13
    rotlwi    r25,r25,13                # rotate comp1
    rotlwi    r26,r26,13                # rotate comp2
    rotlwi    r27,r27,13                # rotate comp3
    rotlwi    r28,r28,13                # rotate comp4
    rotlwi    r29,r29,13                # rotate comp5
    rotlwi    r30,r30,13                # rotate comp6
    rotrwi    r16,r16,13                # rotate list0
    rotrwi    r17,r17,13                # rotate list1
    rotrwi    r18,r18,13                # rotate list2
    rotrwi    r19,r19,13                # rotate list3
    rotrwi    r20,r20,13                # rotate list4
    rotrwi    r21,r21,13                # rotate list5
    rotrwi    r22,r22,13                # rotate list6
    rotrwi    r23,r23,13                # rotate list7
    rlwimi    r24,r25,0,19,31           # comp0 = comp0:comp1 << 13
    rlwimi    r25,r26,0,19,31           # comp1 = comp1:comp2 << 13
    rlwimi    r26,r27,0,19,31           # comp2 = comp2:comp3 << 13
    rlwimi    r27,r28,0,19,31           # comp3 = comp3:comp4 << 13
    rlwimi    r28,r29,0,19,31           # comp4 = comp4:comp5 << 13
    rlwimi    r29,r30,0,19,31           # comp5 = comp5:comp6 << 13
    rlwimi    r30,r31,13,19,31          # comp6 = comp6:comp7 << 13
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,12            # list7 = list6:list7 >> 13
    rlwimi    r22,r21,0,0,12            # list6 = list5:list6 >> 13
    rlwimi    r21,r20,0,0,12            # list5 = list4:list5 >> 13
    rlwimi    r20,r19,0,0,12            # list4 = list3:list4 >> 13
    rlwimi    r19,r18,0,0,12            # list3 = list2:list3 >> 13
    rlwimi    r18,r17,0,0,12            # list2 = list1:list2 >> 13
    rlwimi    r17,r16,0,0,12            # list1 = list0:list1 >> 13
    rlwimi    r16,r13,19,0,12           # list0 = newbit:list0 >> 13
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 14
    rotlwi    r25,r25,14                # rotate comp1
    rotlwi    r26,r26,14                # rotate comp2
    rotlwi    r27,r27,14                # rotate comp3
    rotlwi    r28,r28,14                # rotate comp4
    rotlwi    r29,r29,14                # rotate comp5
    rotlwi    r30,r30,14                # rotate comp6
    rotrwi    r16,r16,14                # rotate list0
    rotrwi    r17,r17,14                # rotate list1
    rotrwi    r18,r18,14                # rotate list2
    rotrwi    r19,r19,14                # rotate list3
    rotrwi    r20,r20,14                # rotate list4
    rotrwi    r21,r21,14                # rotate list5
    rotrwi    r22,r22,14                # rotate list6
    rotrwi    r23,r23,14                # rotate list7
    rlwimi    r24,r25,0,18,31           # comp0 = comp0:comp1 << 14
    rlwimi    r25,r26,0,18,31           # comp1 = comp1:comp2 << 14
    rlwimi    r26,r27,0,18,31           # comp2 = comp2:comp3 << 14
    rlwimi    r27,r28,0,18,31           # comp3 = comp3:comp4 << 14
    rlwimi    r28,r29,0,18,31           # comp4 = comp4:comp5 << 14
    rlwimi    r29,r30,0,18,31           # comp5 = comp5:comp6 << 14
    rlwimi    r30,r31,14,18,31          # comp6 = comp6:comp7 << 14
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,13            # list7 = list6:list7 >> 14
    rlwimi    r22,r21,0,0,13            # list6 = list5:list6 >> 14
    rlwimi    r21,r20,0,0,13            # list5 = list4:list5 >> 14
    rlwimi    r20,r19,0,0,13            # list4 = list3:list4 >> 14
    rlwimi    r19,r18,0,0,13            # list3 = list2:list3 >> 14
    rlwimi    r18,r17,0,0,13            # list2 = list1:list2 >> 14
    rlwimi    r17,r16,0,0,13            # list1 = list0:list1 >> 14
    rlwimi    r16,r13,18,0,13           # list0 = newbit:list0 >> 14
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 15
    rotlwi    r25,r25,15                # rotate comp1
    rotlwi    r26,r26,15                # rotate comp2
    rotlwi    r27,r27,15                # rotate comp3
    rotlwi    r28,r28,15                # rotate comp4
    rotlwi    r29,r29,15                # rotate comp5
    rotlwi    r30,r30,15                # rotate comp6
    rotrwi    r16,r16,15                # rotate list0
    rotrwi    r17,r17,15                # rotate list1
    rotrwi    r18,r18,15                # rotate list2
    rotrwi    r19,r19,15                # rotate list3
    rotrwi    r20,r20,15                # rotate list4
    rotrwi    r21,r21,15                # rotate list5
    rotrwi    r22,r22,15                # rotate list6
    rotrwi    r23,r23,15                # rotate list7
    rlwimi    r24,r25,0,17,31           # comp0 = comp0:comp1 << 15
    rlwimi    r25,r26,0,17,31           # comp1 = comp1:comp2 << 15
    rlwimi    r26,r27,0,17,31           # comp2 = comp2:comp3 << 15
    rlwimi    r27,r28,0,17,31           # comp3 = comp3:comp4 << 15
    rlwimi    r28,r29,0,17,31           # comp4 = comp4:comp5 << 15
    rlwimi    r29,r30,0,17,31           # comp5 = comp5:comp6 << 15
    rlwimi    r30,r31,15,17,31          # comp6 = comp6:comp7 << 15
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,14            # list7 = list6:list7 >> 15
    rlwimi    r22,r21,0,0,14            # list6 = list5:list6 >> 15
    rlwimi    r21,r20,0,0,14            # list5 = list4:list5 >> 15
    rlwimi    r20,r19,0,0,14            # list4 = list3:list4 >> 15
    rlwimi    r19,r18,0,0,14            # list3 = list2:list3 >> 15
    rlwimi    r18,r17,0,0,14            # list2 = list1:list2 >> 15
    rlwimi    r17,r16,0,0,14            # list1 = list0:list1 >> 15
    rlwimi    r16,r13,17,0,14           # list0 = newbit:list0 >> 15
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 16
    rotlwi    r25,r25,16                # rotate comp1
    rotlwi    r26,r26,16                # rotate comp2
    rotlwi    r27,r27,16                # rotate comp3
    rotlwi    r28,r28,16                # rotate comp4
    rotlwi    r29,r29,16                # rotate comp5
    rotlwi    r30,r30,16                # rotate comp6
    rotrwi    r16,r16,16                # rotate list0
    rotrwi    r17,r17,16                # rotate list1
    rotrwi    r18,r18,16                # rotate list2
    rotrwi    r19,r19,16                # rotate list3
    rotrwi    r20,r20,16                # rotate list4
    rotrwi    r21,r21,16                # rotate list5
    rotrwi    r22,r22,16                # rotate list6
    rotrwi    r23,r23,16                # rotate list7
    rlwimi    r24,r25,0,16,31           # comp0 = comp0:comp1 << 16
    rlwimi    r25,r26,0,16,31           # comp1 = comp1:comp2 << 16
    rlwimi    r26,r27,0,16,31           # comp2 = comp2:comp3 << 16
    rlwimi    r27,r28,0,16,31           # comp3 = comp3:comp4 << 16
    rlwimi    r28,r29,0,16,31           # comp4 = comp4:comp5 << 16
    rlwimi    r29,r30,0,16,31           # comp5 = comp5:comp6 << 16
    rlwimi    r30,r31,16,16,31          # comp6 = comp6:comp7 << 16
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,15            # list7 = list6:list7 >> 16
    rlwimi    r22,r21,0,0,15            # list6 = list5:list6 >> 16
    rlwimi    r21,r20,0,0,15            # list5 = list4:list5 >> 16
    rlwimi    r20,r19,0,0,15            # list4 = list3:list4 >> 16
    rlwimi    r19,r18,0,0,15            # list3 = list2:list3 >> 16
    rlwimi    r18,r17,0,0,15            # list2 = list1:list2 >> 16
    rlwimi    r17,r16,0,0,15            # list1 = list0:list1 >> 16
    rlwimi    r16,r13,16,0,15           # list0 = newbit:list0 >> 16
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 17
    rotlwi    r25,r25,17                # rotate comp1
    rotlwi    r26,r26,17                # rotate comp2
    rotlwi    r27,r27,17                # rotate comp3
    rotlwi    r28,r28,17                # rotate comp4
    rotlwi    r29,r29,17                # rotate comp5
    rotlwi    r30,r30,17                # rotate comp6
    rotrwi    r16,r16,17                # rotate list0
    rotrwi    r17,r17,17                # rotate list1
    rotrwi    r18,r18,17                # rotate list2
    rotrwi    r19,r19,17                # rotate list3
    rotrwi    r20,r20,17                # rotate list4
    rotrwi    r21,r21,17                # rotate list5
    rotrwi    r22,r22,17                # rotate list6
    rotrwi    r23,r23,17                # rotate list7
    rlwimi    r24,r25,0,15,31           # comp0 = comp0:comp1 << 17
    rlwimi    r25,r26,0,15,31           # comp1 = comp1:comp2 << 17
    rlwimi    r26,r27,0,15,31           # comp2 = comp2:comp3 << 17
    rlwimi    r27,r28,0,15,31           # comp3 = comp3:comp4 << 17
    rlwimi    r28,r29,0,15,31           # comp4 = comp4:comp5 << 17
    rlwimi    r29,r30,0,15,31           # comp5 = comp5:comp6 << 17
    rlwimi    r30,r31,17,15,31          # comp6 = comp6:comp7 << 17
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,16            # list7 = list6:list7 >> 17
    rlwimi    r22,r21,0,0,16            # list6 = list5:list6 >> 17
    rlwimi    r21,r20,0,0,16            # list5 = list4:list5 >> 17
    rlwimi    r20,r19,0,0,16            # list4 = list3:list4 >> 17
    rlwimi    r19,r18,0,0,16            # list3 = list2:list3 >> 17
    rlwimi    r18,r17,0,0,16            # list2 = list1:list2 >> 17
    rlwimi    r17,r16,0,0,16            # list1 = list0:list1 >> 17
    rlwimi    r16,r13,15,0,16           # list0 = newbit:list0 >> 17
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 18
    rotlwi    r25,r25,18                # rotate comp1
    rotlwi    r26,r26,18                # rotate comp2
    rotlwi    r27,r27,18                # rotate comp3
    rotlwi    r28,r28,18                # rotate comp4
    rotlwi    r29,r29,18                # rotate comp5
    rotlwi    r30,r30,18                # rotate comp6
    rotrwi    r16,r16,18                # rotate list0
    rotrwi    r17,r17,18                # rotate list1
    rotrwi    r18,r18,18                # rotate list2
    rotrwi    r19,r19,18                # rotate list3
    rotrwi    r20,r20,18                # rotate list4
    rotrwi    r21,r21,18                # rotate list5
    rotrwi    r22,r22,18                # rotate list6
    rotrwi    r23,r23,18                # rotate list7
    rlwimi    r24,r25,0,14,31           # comp0 = comp0:comp1 << 18
    rlwimi    r25,r26,0,14,31           # comp1 = comp1:comp2 << 18
    rlwimi    r26,r27,0,14,31           # comp2 = comp2:comp3 << 18
    rlwimi    r27,r28,0,14,31           # comp3 = comp3:comp4 << 18
    rlwimi    r28,r29,0,14,31           # comp4 = comp4:comp5 << 18
    rlwimi    r29,r30,0,14,31           # comp5 = comp5:comp6 << 18
    rlwimi    r30,r31,18,14,31          # comp6 = comp6:comp7 << 18
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,17            # list7 = list6:list7 >> 18
    rlwimi    r22,r21,0,0,17            # list6 = list5:list6 >> 18
    rlwimi    r21,r20,0,0,17            # list5 = list4:list5 >> 18
    rlwimi    r20,r19,0,0,17            # list4 = list3:list4 >> 18
    rlwimi    r19,r18,0,0,17            # list3 = list2:list3 >> 18
    rlwimi    r18,r17,0,0,17            # list2 = list1:list2 >> 18
    rlwimi    r17,r16,0,0,17            # list1 = list0:list1 >> 18
    rlwimi    r16,r13,14,0,17           # list0 = newbit:list0 >> 18
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 19
    rotlwi    r25,r25,19                # rotate comp1
    rotlwi    r26,r26,19                # rotate comp2
    rotlwi    r27,r27,19                # rotate comp3
    rotlwi    r28,r28,19                # rotate comp4
    rotlwi    r29,r29,19                # rotate comp5
    rotlwi    r30,r30,19                # rotate comp6
    rotrwi    r16,r16,19                # rotate list0
    rotrwi    r17,r17,19                # rotate list1
    rotrwi    r18,r18,19                # rotate list2
    rotrwi    r19,r19,19                # rotate list3
    rotrwi    r20,r20,19                # rotate list4
    rotrwi    r21,r21,19                # rotate list5
    rotrwi    r22,r22,19                # rotate list6
    rotrwi    r23,r23,19                # rotate list7
    rlwimi    r24,r25,0,13,31           # comp0 = comp0:comp1 << 19
    rlwimi    r25,r26,0,13,31           # comp1 = comp1:comp2 << 19
    rlwimi    r26,r27,0,13,31           # comp2 = comp2:comp3 << 19
    rlwimi    r27,r28,0,13,31           # comp3 = comp3:comp4 << 19
    rlwimi    r28,r29,0,13,31           # comp4 = comp4:comp5 << 19
    rlwimi    r29,r30,0,13,31           # comp5 = comp5:comp6 << 19
    rlwimi    r30,r31,19,13,31          # comp6 = comp6:comp7 << 19
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,18            # list7 = list6:list7 >> 19
    rlwimi    r22,r21,0,0,18            # list6 = list5:list6 >> 19
    rlwimi    r21,r20,0,0,18            # list5 = list4:list5 >> 19
    rlwimi    r20,r19,0,0,18            # list4 = list3:list4 >> 19
    rlwimi    r19,r18,0,0,18            # list3 = list2:list3 >> 19
    rlwimi    r18,r17,0,0,18            # list2 = list1:list2 >> 19
    rlwimi    r17,r16,0,0,18            # list1 = list0:list1 >> 19
    rlwimi    r16,r13,13,0,18           # list0 = newbit:list0 >> 19
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 20
    rotlwi    r25,r25,20                # rotate comp1
    rotlwi    r26,r26,20                # rotate comp2
    rotlwi    r27,r27,20                # rotate comp3
    rotlwi    r28,r28,20                # rotate comp4
    rotlwi    r29,r29,20                # rotate comp5
    rotlwi    r30,r30,20                # rotate comp6
    rotrwi    r16,r16,20                # rotate list0
    rotrwi    r17,r17,20                # rotate list1
    rotrwi    r18,r18,20                # rotate list2
    rotrwi    r19,r19,20                # rotate list3
    rotrwi    r20,r20,20                # rotate list4
    rotrwi    r21,r21,20                # rotate list5
    rotrwi    r22,r22,20                # rotate list6
    rotrwi    r23,r23,20                # rotate list7
    rlwimi    r24,r25,0,12,31           # comp0 = comp0:comp1 << 20
    rlwimi    r25,r26,0,12,31           # comp1 = comp1:comp2 << 20
    rlwimi    r26,r27,0,12,31           # comp2 = comp2:comp3 << 20
    rlwimi    r27,r28,0,12,31           # comp3 = comp3:comp4 << 20
    rlwimi    r28,r29,0,12,31           # comp4 = comp4:comp5 << 20
    rlwimi    r29,r30,0,12,31           # comp5 = comp5:comp6 << 20
    rlwimi    r30,r31,20,12,31          # comp6 = comp6:comp7 << 20
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,19            # list7 = list6:list7 >> 20
    rlwimi    r22,r21,0,0,19            # list6 = list5:list6 >> 20
    rlwimi    r21,r20,0,0,19            # list5 = list4:list5 >> 20
    rlwimi    r20,r19,0,0,19            # list4 = list3:list4 >> 20
    rlwimi    r19,r18,0,0,19            # list3 = list2:list3 >> 20
    rlwimi    r18,r17,0,0,19            # list2 = list1:list2 >> 20
    rlwimi    r17,r16,0,0,19            # list1 = list0:list1 >> 20
    rlwimi    r16,r13,12,0,19           # list0 = newbit:list0 >> 20
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 21
    rotlwi    r25,r25,21                # rotate comp1
    rotlwi    r26,r26,21                # rotate comp2
    rotlwi    r27,r27,21                # rotate comp3
    rotlwi    r28,r28,21                # rotate comp4
    rotlwi    r29,r29,21                # rotate comp5
    rotlwi    r30,r30,21                # rotate comp6
    rotrwi    r16,r16,21                # rotate list0
    rotrwi    r17,r17,21                # rotate list1
    rotrwi    r18,r18,21                # rotate list2
    rotrwi    r19,r19,21                # rotate list3
    rotrwi    r20,r20,21                # rotate list4
    rotrwi    r21,r21,21                # rotate list5
    rotrwi    r22,r22,21                # rotate list6
    rotrwi    r23,r23,21                # rotate list7
    rlwimi    r24,r25,0,11,31           # comp0 = comp0:comp1 << 21
    rlwimi    r25,r26,0,11,31           # comp1 = comp1:comp2 << 21
    rlwimi    r26,r27,0,11,31           # comp2 = comp2:comp3 << 21
    rlwimi    r27,r28,0,11,31           # comp3 = comp3:comp4 << 21
    rlwimi    r28,r29,0,11,31           # comp4 = comp4:comp5 << 21
    rlwimi    r29,r30,0,11,31           # comp5 = comp5:comp6 << 21
    rlwimi    r30,r31,21,11,31          # comp6 = comp6:comp7 << 21
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,20            # list7 = list6:list7 >> 21
    rlwimi    r22,r21,0,0,20            # list6 = list5:list6 >> 21
    rlwimi    r21,r20,0,0,20            # list5 = list4:list5 >> 21
    rlwimi    r20,r19,0,0,20            # list4 = list3:list4 >> 21
    rlwimi    r19,r18,0,0,20            # list3 = list2:list3 >> 21
    rlwimi    r18,r17,0,0,20            # list2 = list1:list2 >> 21
    rlwimi    r17,r16,0,0,20            # list1 = list0:list1 >> 21
    rlwimi    r16,r13,11,0,20           # list0 = newbit:list0 >> 21
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 22
    rotlwi    r25,r25,22                # rotate comp1
    rotlwi    r26,r26,22                # rotate comp2
    rotlwi    r27,r27,22                # rotate comp3
    rotlwi    r28,r28,22                # rotate comp4
    rotlwi    r29,r29,22                # rotate comp5
    rotlwi    r30,r30,22                # rotate comp6
    rotrwi    r16,r16,22                # rotate list0
    rotrwi    r17,r17,22                # rotate list1
    rotrwi    r18,r18,22                # rotate list2
    rotrwi    r19,r19,22                # rotate list3
    rotrwi    r20,r20,22                # rotate list4
    rotrwi    r21,r21,22                # rotate list5
    rotrwi    r22,r22,22                # rotate list6
    rotrwi    r23,r23,22                # rotate list7
    rlwimi    r24,r25,0,10,31           # comp0 = comp0:comp1 << 22
    rlwimi    r25,r26,0,10,31           # comp1 = comp1:comp2 << 22
    rlwimi    r26,r27,0,10,31           # comp2 = comp2:comp3 << 22
    rlwimi    r27,r28,0,10,31           # comp3 = comp3:comp4 << 22
    rlwimi    r28,r29,0,10,31           # comp4 = comp4:comp5 << 22
    rlwimi    r29,r30,0,10,31           # comp5 = comp5:comp6 << 22
    rlwimi    r30,r31,22,10,31          # comp6 = comp6:comp7 << 22
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,21            # list7 = list6:list7 >> 22
    rlwimi    r22,r21,0,0,21            # list6 = list5:list6 >> 22
    rlwimi    r21,r20,0,0,21            # list5 = list4:list5 >> 22
    rlwimi    r20,r19,0,0,21            # list4 = list3:list4 >> 22
    rlwimi    r19,r18,0,0,21            # list3 = list2:list3 >> 22
    rlwimi    r18,r17,0,0,21            # list2 = list1:list2 >> 22
    rlwimi    r17,r16,0,0,21            # list1 = list0:list1 >> 22
    rlwimi    r16,r13,10,0,21           # list0 = newbit:list0 >> 22
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 23
    rotlwi    r25,r25,23                # rotate comp1
    rotlwi    r26,r26,23                # rotate comp2
    rotlwi    r27,r27,23                # rotate comp3
    rotlwi    r28,r28,23                # rotate comp4
    rotlwi    r29,r29,23                # rotate comp5
    rotlwi    r30,r30,23                # rotate comp6
    rotrwi    r16,r16,23                # rotate list0
    rotrwi    r17,r17,23                # rotate list1
    rotrwi    r18,r18,23                # rotate list2
    rotrwi    r19,r19,23                # rotate list3
    rotrwi    r20,r20,23                # rotate list4
    rotrwi    r21,r21,23                # rotate list5
    rotrwi    r22,r22,23                # rotate list6
    rotrwi    r23,r23,23                # rotate list7
    rlwimi    r24,r25,0,9,31            # comp0 = comp0:comp1 << 23
    rlwimi    r25,r26,0,9,31            # comp1 = comp1:comp2 << 23
    rlwimi    r26,r27,0,9,31            # comp2 = comp2:comp3 << 23
    rlwimi    r27,r28,0,9,31            # comp3 = comp3:comp4 << 23
    rlwimi    r28,r29,0,9,31            # comp4 = comp4:comp5 << 23
    rlwimi    r29,r30,0,9,31            # comp5 = comp5:comp6 << 23
    rlwimi    r30,r31,23,9,31           # comp6 = comp6:comp7 << 23
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,22            # list7 = list6:list7 >> 23
    rlwimi    r22,r21,0,0,22            # list6 = list5:list6 >> 23
    rlwimi    r21,r20,0,0,22            # list5 = list4:list5 >> 23
    rlwimi    r20,r19,0,0,22            # list4 = list3:list4 >> 23
    rlwimi    r19,r18,0,0,22            # list3 = list2:list3 >> 23
    rlwimi    r18,r17,0,0,22            # list2 = list1:list2 >> 23
    rlwimi    r17,r16,0,0,22            # list1 = list0:list1 >> 23
    rlwimi    r16,r13,9,0,22            # list0 = newbit:list0 >> 23
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 24
    rotlwi    r25,r25,24                # rotate comp1
    rotlwi    r26,r26,24                # rotate comp2
    rotlwi    r27,r27,24                # rotate comp3
    rotlwi    r28,r28,24                # rotate comp4
    rotlwi    r29,r29,24                # rotate comp5
    rotlwi    r30,r30,24                # rotate comp6
    rotrwi    r16,r16,24                # rotate list0
    rotrwi    r17,r17,24                # rotate list1
    rotrwi    r18,r18,24                # rotate list2
    rotrwi    r19,r19,24                # rotate list3
    rotrwi    r20,r20,24                # rotate list4
    rotrwi    r21,r21,24                # rotate list5
    rotrwi    r22,r22,24                # rotate list6
    rotrwi    r23,r23,24                # rotate list7
    rlwimi    r24,r25,0,8,31            # comp0 = comp0:comp1 << 24
    rlwimi    r25,r26,0,8,31            # comp1 = comp1:comp2 << 24
    rlwimi    r26,r27,0,8,31            # comp2 = comp2:comp3 << 24
    rlwimi    r27,r28,0,8,31            # comp3 = comp3:comp4 << 24
    rlwimi    r28,r29,0,8,31            # comp4 = comp4:comp5 << 24
    rlwimi    r29,r30,0,8,31            # comp5 = comp5:comp6 << 24
    rlwimi    r30,r31,24,8,31           # comp6 = comp6:comp7 << 24
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,23            # list7 = list6:list7 >> 24
    rlwimi    r22,r21,0,0,23            # list6 = list5:list6 >> 24
    rlwimi    r21,r20,0,0,23            # list5 = list4:list5 >> 24
    rlwimi    r20,r19,0,0,23            # list4 = list3:list4 >> 24
    rlwimi    r19,r18,0,0,23            # list3 = list2:list3 >> 24
    rlwimi    r18,r17,0,0,23            # list2 = list1:list2 >> 24
    rlwimi    r17,r16,0,0,23            # list1 = list0:list1 >> 24
    rlwimi    r16,r13,8,0,23            # list0 = newbit:list0 >> 24
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 25
    rotlwi    r25,r25,25                # rotate comp1
    rotlwi    r26,r26,25                # rotate comp2
    rotlwi    r27,r27,25                # rotate comp3
    rotlwi    r28,r28,25                # rotate comp4
    rotlwi    r29,r29,25                # rotate comp5
    rotlwi    r30,r30,25                # rotate comp6
    rotrwi    r16,r16,25                # rotate list0
    rotrwi    r17,r17,25                # rotate list1
    rotrwi    r18,r18,25                # rotate list2
    rotrwi    r19,r19,25                # rotate list3
    rotrwi    r20,r20,25                # rotate list4
    rotrwi    r21,r21,25                # rotate list5
    rotrwi    r22,r22,25                # rotate list6
    rotrwi    r23,r23,25                # rotate list7
    rlwimi    r24,r25,0,7,31            # comp0 = comp0:comp1 << 25
    rlwimi    r25,r26,0,7,31            # comp1 = comp1:comp2 << 25
    rlwimi    r26,r27,0,7,31            # comp2 = comp2:comp3 << 25
    rlwimi    r27,r28,0,7,31            # comp3 = comp3:comp4 << 25
    rlwimi    r28,r29,0,7,31            # comp4 = comp4:comp5 << 25
    rlwimi    r29,r30,0,7,31            # comp5 = comp5:comp6 << 25
    rlwimi    r30,r31,25,7,31           # comp6 = comp6:comp7 << 25
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,24            # list7 = list6:list7 >> 25
    rlwimi    r22,r21,0,0,24            # list6 = list5:list6 >> 25
    rlwimi    r21,r20,0,0,24            # list5 = list4:list5 >> 25
    rlwimi    r20,r19,0,0,24            # list4 = list3:list4 >> 25
    rlwimi    r19,r18,0,0,24            # list3 = list2:list3 >> 25
    rlwimi    r18,r17,0,0,24            # list2 = list1:list2 >> 25
    rlwimi    r17,r16,0,0,24            # list1 = list0:list1 >> 25
    rlwimi    r16,r13,7,0,24            # list0 = newbit:list0 >> 25
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 26
    rotlwi    r25,r25,26                # rotate comp1
    rotlwi    r26,r26,26                # rotate comp2
    rotlwi    r27,r27,26                # rotate comp3
    rotlwi    r28,r28,26                # rotate comp4
    rotlwi    r29,r29,26                # rotate comp5
    rotlwi    r30,r30,26                # rotate comp6
    rotrwi    r16,r16,26                # rotate list0
    rotrwi    r17,r17,26                # rotate list1
    rotrwi    r18,r18,26                # rotate list2
    rotrwi    r19,r19,26                # rotate list3
    rotrwi    r20,r20,26                # rotate list4
    rotrwi    r21,r21,26                # rotate list5
    rotrwi    r22,r22,26                # rotate list6
    rotrwi    r23,r23,26                # rotate list7
    rlwimi    r24,r25,0,6,31            # comp0 = comp0:comp1 << 26
    rlwimi    r25,r26,0,6,31            # comp1 = comp1:comp2 << 26
    rlwimi    r26,r27,0,6,31            # comp2 = comp2:comp3 << 26
    rlwimi    r27,r28,0,6,31            # comp3 = comp3:comp4 << 26
    rlwimi    r28,r29,0,6,31            # comp4 = comp4:comp5 << 26
    rlwimi    r29,r30,0,6,31            # comp5 = comp5:comp6 << 26
    rlwimi    r30,r31,26,6,31           # comp6 = comp6:comp7 << 26
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,25            # list7 = list6:list7 >> 26
    rlwimi    r22,r21,0,0,25            # list6 = list5:list6 >> 26
    rlwimi    r21,r20,0,0,25            # list5 = list4:list5 >> 26
    rlwimi    r20,r19,0,0,25            # list4 = list3:list4 >> 26
    rlwimi    r19,r18,0,0,25            # list3 = list2:list3 >> 26
    rlwimi    r18,r17,0,0,25            # list2 = list1:list2 >> 26
    rlwimi    r17,r16,0,0,25            # list1 = list0:list1 >> 26
    rlwimi    r16,r13,6,0,25            # list0 = newbit:list0 >> 26
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 27
    rotlwi    r25,r25,27                # rotate comp1
    rotlwi    r26,r26,27                # rotate comp2
    rotlwi    r27,r27,27                # rotate comp3
    rotlwi    r28,r28,27                # rotate comp4
    rotlwi    r29,r29,27                # rotate comp5
    rotlwi    r30,r30,27                # rotate comp6
    rotrwi    r16,r16,27                # rotate list0
    rotrwi    r17,r17,27                # rotate list1
    rotrwi    r18,r18,27                # rotate list2
    rotrwi    r19,r19,27                # rotate list3
    rotrwi    r20,r20,27                # rotate list4
    rotrwi    r21,r21,27                # rotate list5
    rotrwi    r22,r22,27                # rotate list6
    rotrwi    r23,r23,27                # rotate list7
    rlwimi    r24,r25,0,5,31            # comp0 = comp0:comp1 << 27
    rlwimi    r25,r26,0,5,31            # comp1 = comp1:comp2 << 27
    rlwimi    r26,r27,0,5,31            # comp2 = comp2:comp3 << 27
    rlwimi    r27,r28,0,5,31            # comp3 = comp3:comp4 << 27
    rlwimi    r28,r29,0,5,31            # comp4 = comp4:comp5 << 27
    rlwimi    r29,r30,0,5,31            # comp5 = comp5:comp6 << 27
    rlwimi    r30,r31,27,5,31           # comp6 = comp6:comp7 << 27
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,26            # list7 = list6:list7 >> 27
    rlwimi    r22,r21,0,0,26            # list6 = list5:list6 >> 27
    rlwimi    r21,r20,0,0,26            # list5 = list4:list5 >> 27
    rlwimi    r20,r19,0,0,26            # list4 = list3:list4 >> 27
    rlwimi    r19,r18,0,0,26            # list3 = list2:list3 >> 27
    rlwimi    r18,r17,0,0,26            # list2 = list1:list2 >> 27
    rlwimi    r17,r16,0,0,26            # list1 = list0:list1 >> 27
    rlwimi    r16,r13,5,0,26            # list0 = newbit:list0 >> 27
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 28
    rotlwi    r25,r25,28                # rotate comp1
    rotlwi    r26,r26,28                # rotate comp2
    rotlwi    r27,r27,28                # rotate comp3
    rotlwi    r28,r28,28                # rotate comp4
    rotlwi    r29,r29,28                # rotate comp5
    rotlwi    r30,r30,28                # rotate comp6
    rotrwi    r16,r16,28                # rotate list0
    rotrwi    r17,r17,28                # rotate list1
    rotrwi    r18,r18,28                # rotate list2
    rotrwi    r19,r19,28                # rotate list3
    rotrwi    r20,r20,28                # rotate list4
    rotrwi    r21,r21,28                # rotate list5
    rotrwi    r22,r22,28                # rotate list6
    rotrwi    r23,r23,28                # rotate list7
    rlwimi    r24,r25,0,4,31            # comp0 = comp0:comp1 << 28
    rlwimi    r25,r26,0,4,31            # comp1 = comp1:comp2 << 28
    rlwimi    r26,r27,0,4,31            # comp2 = comp2:comp3 << 28
    rlwimi    r27,r28,0,4,31            # comp3 = comp3:comp4 << 28
    rlwimi    r28,r29,0,4,31            # comp4 = comp4:comp5 << 28
    rlwimi    r29,r30,0,4,31            # comp5 = comp5:comp6 << 28
    rlwimi    r30,r31,28,4,31           # comp6 = comp6:comp7 << 28
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,27            # list7 = list6:list7 >> 28
    rlwimi    r22,r21,0,0,27            # list6 = list5:list6 >> 28
    rlwimi    r21,r20,0,0,27            # list5 = list4:list5 >> 28
    rlwimi    r20,r19,0,0,27            # list4 = list3:list4 >> 28
    rlwimi    r19,r18,0,0,27            # list3 = list2:list3 >> 28
    rlwimi    r18,r17,0,0,27            # list2 = list1:list2 >> 28
    rlwimi    r17,r16,0,0,27            # list1 = list0:list1 >> 28
    rlwimi    r16,r13,4,0,27            # list0 = newbit:list0 >> 28
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 29
    rotlwi    r25,r25,29                # rotate comp1
    rotlwi    r26,r26,29                # rotate comp2
    rotlwi    r27,r27,29                # rotate comp3
    rotlwi    r28,r28,29                # rotate comp4
    rotlwi    r29,r29,29                # rotate comp5
    rotlwi    r30,r30,29                # rotate comp6
    rotrwi    r16,r16,29                # rotate list0
    rotrwi    r17,r17,29                # rotate list1
    rotrwi    r18,r18,29                # rotate list2
    rotrwi    r19,r19,29                # rotate list3
    rotrwi    r20,r20,29                # rotate list4
    rotrwi    r21,r21,29                # rotate list5
    rotrwi    r22,r22,29                # rotate list6
    rotrwi    r23,r23,29                # rotate list7
    rlwimi    r24,r25,0,3,31            # comp0 = comp0:comp1 << 29
    rlwimi    r25,r26,0,3,31            # comp1 = comp1:comp2 << 29
    rlwimi    r26,r27,0,3,31            # comp2 = comp2:comp3 << 29
    rlwimi    r27,r28,0,3,31            # comp3 = comp3:comp4 << 29
    rlwimi    r28,r29,0,3,31            # comp4 = comp4:comp5 << 29
    rlwimi    r29,r30,0,3,31            # comp5 = comp5:comp6 << 29
    rlwimi    r30,r31,29,3,31           # comp6 = comp6:comp7 << 29
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,28            # list7 = list6:list7 >> 29
    rlwimi    r22,r21,0,0,28            # list6 = list5:list6 >> 29
    rlwimi    r21,r20,0,0,28            # list5 = list4:list5 >> 29
    rlwimi    r20,r19,0,0,28            # list4 = list3:list4 >> 29
    rlwimi    r19,r18,0,0,28            # list3 = list2:list3 >> 29
    rlwimi    r18,r17,0,0,28            # list2 = list1:list2 >> 29
    rlwimi    r17,r16,0,0,28            # list1 = list0:list1 >> 29
    rlwimi    r16,r13,3,0,28            # list0 = newbit:list0 >> 29
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 30
    rotlwi    r25,r25,30                # rotate comp1
    rotlwi    r26,r26,30                # rotate comp2
    rotlwi    r27,r27,30                # rotate comp3
    rotlwi    r28,r28,30                # rotate comp4
    rotlwi    r29,r29,30                # rotate comp5
    rotlwi    r30,r30,30                # rotate comp6
    rotrwi    r16,r16,30                # rotate list0
    rotrwi    r17,r17,30                # rotate list1
    rotrwi    r18,r18,30                # rotate list2
    rotrwi    r19,r19,30                # rotate list3
    rotrwi    r20,r20,30                # rotate list4
    rotrwi    r21,r21,30                # rotate list5
    rotrwi    r22,r22,30                # rotate list6
    rotrwi    r23,r23,30                # rotate list7
    rlwimi    r24,r25,0,2,31            # comp0 = comp0:comp1 << 30
    rlwimi    r25,r26,0,2,31            # comp1 = comp1:comp2 << 30
    rlwimi    r26,r27,0,2,31            # comp2 = comp2:comp3 << 30
    rlwimi    r27,r28,0,2,31            # comp3 = comp3:comp4 << 30
    rlwimi    r28,r29,0,2,31            # comp4 = comp4:comp5 << 30
    rlwimi    r29,r30,0,2,31            # comp5 = comp5:comp6 << 30
    rlwimi    r30,r31,30,2,31           # comp6 = comp6:comp7 << 30
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,29            # list7 = list6:list7 >> 30
    rlwimi    r22,r21,0,0,29            # list6 = list5:list6 >> 30
    rlwimi    r21,r20,0,0,29            # list5 = list4:list5 >> 30
    rlwimi    r20,r19,0,0,29            # list4 = list3:list4 >> 30
    rlwimi    r19,r18,0,0,29            # list3 = list2:list3 >> 30
    rlwimi    r18,r17,0,0,29            # list2 = list1:list2 >> 30
    rlwimi    r17,r16,0,0,29            # list1 = list0:list1 >> 30
    rlwimi    r16,r13,2,0,29            # list0 = newbit:list0 >> 30
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 31
    rotlwi    r25,r25,31                # rotate comp1
    rotlwi    r26,r26,31                # rotate comp2
    rotlwi    r27,r27,31                # rotate comp3
    rotlwi    r28,r28,31                # rotate comp4
    rotlwi    r29,r29,31                # rotate comp5
    rotlwi    r30,r30,31                # rotate comp6
    rotrwi    r16,r16,31                # rotate list0
    rotrwi    r17,r17,31                # rotate list1
    rotrwi    r18,r18,31                # rotate list2
    rotrwi    r19,r19,31                # rotate list3
    rotrwi    r20,r20,31                # rotate list4
    rotrwi    r21,r21,31                # rotate list5
    rotrwi    r22,r22,31                # rotate list6
    rotrwi    r23,r23,31                # rotate list7
    rlwimi    r24,r25,0,1,31            # comp0 = comp0:comp1 << 31
    rlwimi    r25,r26,0,1,31            # comp1 = comp1:comp2 << 31
    rlwimi    r26,r27,0,1,31            # comp2 = comp2:comp3 << 31
    rlwimi    r27,r28,0,1,31            # comp3 = comp3:comp4 << 31
    rlwimi    r28,r29,0,1,31            # comp4 = comp4:comp5 << 31
    rlwimi    r29,r30,0,1,31            # comp5 = comp5:comp6 << 31
    rlwimi    r30,r31,31,1,31           # comp6 = comp6:comp7 << 31
    slw       r31,r31,r14               # shift comp7
    rlwimi    r23,r22,0,0,30            # list7 = list6:list7 >> 31
    rlwimi    r22,r21,0,0,30            # list6 = list5:list6 >> 31
    rlwimi    r21,r20,0,0,30            # list5 = list4:list5 >> 31
    rlwimi    r20,r19,0,0,30            # list4 = list3:list4 >> 31
    rlwimi    r19,r18,0,0,30            # list3 = list2:list3 >> 31
    rlwimi    r18,r17,0,0,30            # list2 = list1:list2 >> 31
    rlwimi    r17,r16,0,0,30            # list1 = list0:list1 >> 31
    rlwimi    r16,r13,1,0,30            # list0 = newbit:list0 >> 31
    beq-      L_exit                    # Ruler found
    b         L_UpdateLevel

    # diff == 32
    mr        r24,r25                   # comp0 = comp1
    mr        r25,r26                   # comp1 = comp2
    mr        r26,r27                   # comp2 = comp3
    mr        r27,r28                   # comp3 = comp4
    mr        r28,r29                   # comp4 = comp5
    mr        r29,r30                   # comp5 = comp6
    mr        r30,r31                   # comp6 = comp7
    not       r0,r24                    # ~comp0
    li        r31,0                     # comp7 = 0
    srwi      r0,r0,1                   # (~comp0) >> 1)
    mr        r23,r22                   # list7 = list6
    mr        r22,r21                   # list6 = list5
    mr        r21,r20                   # list5 = list4
    mr        r20,r19                   # list4 = list3
    mr        r19,r18                   # list3 = list2
    mr        r18,r17                   # list2 = list1
    mr        r17,r16                   # list1 = list0
    mr        r16,r13                   # list0 = newbit
    li        r13,0                     # newbit = 0
    beq+      cr1,L_MainLoop            # comp0 == 0xFFFFFFFF
    beq-      L_exit                    # Ruler found

    nop                                 # For alignment purpose
    nop       
    nop       

L_UpdateLevel:
    lwz       r14,Levels_dist+16(r3)    # load dist4
    addi      r6,r6,1                   # ++Depth
    lwz       r15,Levels_dist+20(r3)    # load dist5
    cmplw     cr1,r6,r10                # Depth <= MidSegB ?
    lwz       r0,Levels_dist+24(r3)     # load dist6
    cmplw     cr7,r6,r9                 # Depth > MidSegA ?
    lwz       r13,Levels_dist+28(r3)    # load dist7
    subic.    r4,r4,1                   # --nodes <= 0 ?
    stw       r20,Levels_list+16(r3)    # store list4
    or        r14,r14,r20               # dist4 |= list4
    stw       r21,Levels_list+20(r3)    # store list5
    or        r15,r15,r21               # dist5 |= list5
    stw       r22,Levels_list+24(r3)    # store list6
    or        r0,r0,r22                 # dist6 |= list6
    stw       r23,Levels_list+28(r3)    # store list7
    or        r13,r13,r23               # dist7 |= list7
    stw       r28,Levels_comp+16(r3)    # store comp4
    or        r28,r28,r14               # comp4 |=dist4
    stw       r29,Levels_comp+20(r3)    # store comp5
    or        r29,r29,r15               # comp5 |=dist5
    stw       r30,Levels_comp+24(r3)    # store comp6
    or        r30,r30,r0                # comp6 |=dist6
    stw       r31,Levels_comp+28(r3)    # store comp7
    or        r31,r31,r13               # comp7 |=dist7
    # That's not fast, but we ran out of registers...
    stw       r14,NextLev_dist+16(r3)   # store dist4
    stw       r15,NextLev_dist+20(r3)   # store dist5
    stw       r0,NextLev_dist+24(r3)    # store dist6
    stw       r13,NextLev_dist+28(r3)   # store dist7
    lwz       r13,Levels_dist+0(r3)     # load dist0
    lwz       r14,Levels_dist+4(r3)     # load dist1
    lwz       r15,Levels_dist+8(r3)     # load dist2
    lwz       r0,Levels_dist+12(r3)     # load dist3
    stw       r24,Levels_comp+0(r3)     # store comp0
    or        r13,r13,r16               # dist0 |= list0
    stw       r25,Levels_comp+4(r3)     # store comp1
    or        r14,r14,r17               # dist1 |= list1
    stw       r26,Levels_comp+8(r3)     # store comp2
    or        r15,r15,r18               # dist2 |= list2
    stw       r27,Levels_comp+12(r3)    # store comp3
    or        r0,r0,r19                 # dist3 |= list3
    stw       r13,NextLev_dist+0(r3)    # store dist0
    or        r24,r24,r13               # comp0 |= dist0
    stw       r14,NextLev_dist+4(r3)    # store dist1
    or        r25,r25,r14               # comp1 |= dist1
    stw       r15,NextLev_dist+8(r3)    # store dist2
    or        r26,r26,r15               # comp2 |= dist2
    stw       r0,NextLev_dist+12(r3)    # store dist3
    or        r27,r27,r0                # comp3 |= dist3
    stw       r16,Levels_list+0(r3)     # store list0
    addi      r5,r5,2                   # ++pChoose
    stw       r17,Levels_list+4(r3)     # store list1
    rlwinm    r14,r13,SH,MB,ME          # 32*2*(dist0 >> CHOOSE_BITS)
    stw       r18,Levels_list+8(r3)     # store list2
    not       r0,r24                    # c0neg = ~comp0
    stw       r19,Levels_list+12(r3)    # store list3
    srwi      r0,r0,1                   # c0neg = (~comp0) >> 1
    stw       r7,mark(r3)               # Store the current mark
    addi      r3,r3,SIZEOF_LEVEL        # ++Levels
    lhzx      r8,r14,r5                 # load the limit
    bgt+      cr1,L_LoopBack            # Depth > MidSegB
    ble-      cr7,L_LoopBack            # Depth <= MidSegA

L_GetLimit:
    # Compute the limit within the middle segment.
    # This part is only executed when depth == half_depth2 or
    # depth == half_depth2-1 (for odd rulers).
    # cr0 := nodes <= 0
    # cr1 := Depth == MidSegB
    # r8  := limit
    # r13 := dist0

    lwz       r14,pLevelA(r1)           # &Levels[MidSegA]
    lwz       r15,maxLenM1(r1)          # maxlen - 1
    lwz       r14,mark(r14)             # Levels[MidSegA].mark
    sub       r14,r15,r14               # temp
    beq       cr1,L_MinFunc             # Depth == MidSegB

    # Compute middle mark limit (depth < MidSegB)
    not       r15,r13                   # ~dist0
    cntlzw    r15,r15                   # FFZ(dist0)
    addi      r15,r15,1
    sub       r14,r14,r15               # temp -= FFZ(dist0)

L_MinFunc:
    subfc     r15,r8,r14
    subfe     r14,r14,r14
    and       r15,r15,r14
    add       r8,r8,r15                 # limit = min(temp, limit)

L_LoopBack:
    # cr0 := nodes <= 0
    li        r13,1                     # newbit = 1
    stw       r8,limit(r3)              # store the limit
    bgt+      L_MainLoop                # nodes > 0

L_exit:
    # Save state
    stw       r16,Levels_list+0(r3)     # store list0
    stw       r24,Levels_comp+0(r3)     # store comp0
    stw       r17,Levels_list+4(r3)     # store list1
    stw       r25,Levels_comp+4(r3)     # store comp1
    stw       r18,Levels_list+8(r3)     # store list2
    stw       r26,Levels_comp+8(r3)     # store comp2
    stw       r19,Levels_list+12(r3)    # store list3
    stw       r27,Levels_comp+12(r3)    # store comp3
    stw       r20,Levels_list+16(r3)    # store list4
    stw       r28,Levels_comp+16(r3)    # store comp4
    stw       r21,Levels_list+20(r3)    # store list5
    stw       r29,Levels_comp+20(r3)    # store comp5
    stw       r22,Levels_list+24(r3)    # store list6
    stw       r30,Levels_comp+24(r3)    # store comp6
    stw       r23,Levels_list+28(r3)    # store list7
    stw       r31,Levels_comp+28(r3)    # store comp7

    stw       r7,mark(r3)               # Store the current mark
    lwz       r8,pNodes(r1)             # reload pNodes
    lwz       r14,0(r8)
    sub       r14,r14,r4                # -= nodes
    stw       r14,0(r8)                 # *pNodes
    lwz       r2,gpr2(r1)               # restore GPR2
    mr        r3, r6                    # return actual depth


#============================================================================
# Epilog

    # Restore non-volatile registers
    lwz       r5,0(r1)                  # Obtains caller's stack pointer
    lmw       r13,-GPRSaveArea(r5)      # Restore GPRs
    mr        r1,r5
    blr       

