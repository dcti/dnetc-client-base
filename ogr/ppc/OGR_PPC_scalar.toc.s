;#
;# Scalar OGR core for PowerPC processors.
;# Code loosely scheduled for MPC750 (G3)
;# Written by Didier Levet (kakace@distributed.net)
;#
;# Copyright 2004 distributed.net, All Rights Reserved
;# For use in distributed.net projects only.
;# Any other distribution or use of this source violates copyright.
;#
;#============================================================================
;# Special notes :
;# - The code extensively use simplified mnemonics.
;# - Source code compatible with GAS and Apple's AS.
;# - Built-in implementation of found_one().
;# - Use a custom stack frame (leaf procedure).
;# - LR register not used nor saved in caller's stack.
;# - CTR, CR0, CR1, GPR0 and GPR3-GPR12 are volatile (not preserved).
;#
;# $Id: OGR_PPC_scalar.toc.s,v 1.1.2.1 2004/08/08 20:18:22 kakace Exp $
;#
;#============================================================================


    .globl    .cycle_ppc_scalar         ;# coff
    .globl    cycle_ppc_scalar          ;# elf


;# Bitmaps dependencies (offsets)
.set          LEVEL_LIST,         0     ;# list[] bitmap
.set          LEVEL_COMP,         40    ;# comp[] bitmap
.set          LEVEL_DIST,         20    ;# dist[] bitmap


;# Structure members dependencies (offsets)
.set          STATE_DEPTH,        28
.set          STATE_LEVELS,       32
.set          SIZEOF_LEVEL,       68
.set          STATE_MAX,          0     ;# OGR[stub.marks-1] = max length
.set          STATE_MAXDEPTHM1,   8     ;# stub.marks - 1
.set          STATE_STARTDEPTH,   24    ;# workstub->stub.length
.set          STATE_HALFDEPTH,    16    ;# first mark of the middle segment
.set          STATE_HALFDEPTH2,   20    ;# last mark of the middle segment
.set          STATE_HALFLENGTH,   12    ;# maximum position of the middle segment
.set          LEVEL_MARK,         60
.set          LEVEL_LIMIT,        64
.set          SIZEOF_OGR,         4
.set          OGR_SHIFT,          2


;# Constants
.set          CORE_S_SUCCESS,     2
.set          CORE_S_CONTINUE,    1
.set          CORE_S_OK,          0
.set          BITMAP_LENGTH,      160   ;# bitmap : 5 * 32-bits
.set          DIST_SHIFT,         20
.set          DIST_BITS,          32-DIST_SHIFT


;# Parameters for rlwinm (choose addressing)
.set          DIST_SH, DIST_BITS
.set          DIST_MB, DIST_SHIFT
.set          DIST_ME, 31


;#============================================================================
;# Custom stack frame

.set          FIRST_NV_GPR, 13          ;# Save r13..r31
.set          GPRSaveArea, (32-FIRST_NV_GPR) * 4

.set          aStorage, 4               ;# Private storage area
.set          aDiffs, 32                ;# diffs control array
.set          localTop, 1056
.set          FrameSize, (localTop + GPRSaveArea + 15) & (-16)

.set          oState, aStorage + 12
.set          pNodes, aStorage + 16


;#============================================================================
;# Register aliases (GAS). Ignored by Apple's AS

;.set         r0,0
;.set         r1,1
;.set         r3,3
;.set         r4,4
;.set         r5,5
;.set         r6,6
;.set         r7,7
;.set         r8,8
;.set         r9,9
;.set         r10,10
;.set         r11,11
;.set         r12,12
;.set         r13,13
;.set         r14,14
;.set         r15,15
;.set         r16,16
;.set         r17,17
;.set         r18,18
;.set         r19,19
;.set         r20,20
;.set         r21,21
;.set         r22,22
;.set         r23,23
;.set         r24,24
;.set         r25,25
;.set         r26,26
;.set         r27,27
;.set         r28,28
;.set         r29,29
;.set         r30,30
;.set         r31,31


;#============================================================================
;# int cycle_ppc_scalar(void *state (r3)
;#                      int *pnodes (r4)
;#                      const unsigned char *choose (r5)
;#                      const int *OGR (r6)

    ;# add new TOC entry
    .toc      
    .csect    .cycle_ppc_scalar[DS]
cycle_ppc_scalar:

    ;# set the TOC anchor
    .long     .cycle_ppc_scalar, TOC[tc0], 0

    .csect    .text[PR]
    .align    4

.cycle_ppc_scalar:                      ;# coff
    mr        r10,r1                    ;# Caller's stack pointer
    clrlwi    r12,r1,27                 ;# keep the low-order 4 bits
    subfic    r12,r12,-FrameSize        ;# Frame size, including padding
    stwux     r1,r1,r12

    ;# Save non-volatile registers
    stmw      r13,-GPRSaveArea(r10)     ;# Save GPRs


;#============================================================================
;# Core engine initialization - Registers allocation :
;#   r0  := tmp0 / c0neg
;#   r3  := tmp4 / retval
;#   r4  := &oState->Levels[depth]
;#   r5  := &choose[3+remdepth]
;#   r6  := &OGR[remdepth]
;#   r7  := tmp1 / stateptr
;#   r8  := tmp2 / limit
;#   r9  := tmp3 / newbit
;#   r10 := nodes
;#   r11 := oState->max (const)
;#   r12 := &oState->Levels[halfdepth] (const)
;#   r13 := oState->half_depth2 (const)
;#   r14 := depth
;#   r15 := remdepth
;#   r16 := mark
;#   r17-r21 := dist[0]...dist[4]
;#   r22-r26 := list[0]...list[4]
;#   r27-r31 := comp[0]...comp[4]

    stw       r3,oState(r1)             ;# Save oState
    stw       r4,pNodes(r1)             ;# Save pNodes
    lwz       r14,STATE_DEPTH(r3)       ;# oState->depth
    addi      r14,r14,1                 ;# ++depth
    li        r10,0                     ;# nodes = 0
    lwz       r7,STATE_STARTDEPTH(r3)   ;# oState->startdepth
    lwz       r13,STATE_HALFDEPTH2(r3)  ;# oState->half_depth2
    lwz       r12,STATE_HALFDEPTH(r3)   ;# oState->half_depth
    lwz       r15,STATE_MAXDEPTHM1(r3)  ;# oState->maxdepthm1
    lwz       r11,STATE_MAX(r3)         ;# oState->max

    mulli     r12,r12,SIZEOF_LEVEL
    addi      r4,r3,STATE_LEVELS        ;# &oState->Levels[0]
    mulli     r8,r14,SIZEOF_LEVEL
    add       r12,r4,r12                ;# &oState->Levels[halfdepth]
    add       r4,r4,r8                  ;# &oState->Levels[depth]

    sub       r15,r15,r14               ;# remdepth = maxdepthm1 - depth
    sub       r13,r13,r7                ;# halfdepth2 -= startdepth
    sub       r14,r14,r7                ;# depth -= startdepth
    slwi      r7,r15,OGR_SHIFT          ;# remdepth * sizeof(int)
    lwz       r16,LEVEL_MARK(r4)        ;# level->mark
    add       r5,r5,r15                 ;# &choose[remdepth]
    add       r6,r6,r7                  ;# &OGR[remdepth]

    ;# SETUP_TOP_STATE
    lwz       r27,LEVEL_COMP+0(r4)      ;# comp0 bitmap
    lwz       r22,LEVEL_LIST+0(r4)      ;# list0 bitmap
    lwz       r17,LEVEL_DIST+0(r4)      ;# dist0 bitmap
    lwz       r28,LEVEL_COMP+4(r4)      ;# comp1 bitmap
    lwz       r23,LEVEL_LIST+4(r4)      ;# list1 bitmap
    lwz       r18,LEVEL_DIST+4(r4)      ;# dist1 bitmap
    lwz       r29,LEVEL_COMP+8(r4)      ;# comp2 bitmap
    lwz       r24,LEVEL_LIST+8(r4)      ;# list2 bitmap
    lwz       r19,LEVEL_DIST+8(r4)      ;# dist2 bitmap
    lwz       r30,LEVEL_COMP+12(r4)     ;# comp3 bitmap
    lwz       r25,LEVEL_LIST+12(r4)     ;# list3 bitmap
    lwz       r20,LEVEL_DIST+12(r4)     ;# dist3 bitmap
    lwz       r31,LEVEL_COMP+16(r4)     ;# comp4 bitmap
    lwz       r26,LEVEL_LIST+16(r4)     ;# list4 bitmap
    lwz       r21,LEVEL_DIST+16(r4)     ;# dist4 bitmap
    rlwinm    r7,r17,DIST_SH+2,DIST_MB-2,DIST_ME-2
    rlwinm    r0,r17,DIST_SH+3,DIST_MB-3,DIST_ME-3
    add       r0,r7,r0                  ;# (dist0 >> DIST_SHIFT) * DIST_BITS
    lbzx      r8,r5,r0                  ;# choose(dist0 >> ttmDISTBITS, remdepth)
    b         L_comp_limit

    .align    4
L_push_level: 
    ;# PUSH_LEVEL_UPDATE_STATE(lev)
    stw       r22,LEVEL_LIST+0(r4)      ;# store list0
    or        r17,r17,r22               ;# dist0 |= list0
    stw       r27,LEVEL_COMP+0(r4)      ;# store comp0
    or        r27,r27,r17               ;# comp0 |= dist0
    stw       r23,LEVEL_LIST+4(r4)      ;# store list1
    or        r18,r18,r23               ;# dist1 |= list1
    stw       r28,LEVEL_COMP+4(r4)      ;# store comp1
    or        r28,r28,r18               ;# comp1 |= dist1
    stw       r24,LEVEL_LIST+8(r4)      ;# store list2
    or        r19,r19,r24               ;# dist2 |= list2
    stw       r29,LEVEL_COMP+8(r4)      ;# store comp2
    or        r29,r29,r19               ;# comp2 |= dist2
    stw       r25,LEVEL_LIST+12(r4)     ;# store list3
    or        r20,r20,r25               ;# dist3 |= list3
    stw       r30,LEVEL_COMP+12(r4)     ;# store comp3
    or        r30,r30,r20               ;# comp3 |= dist3
    stw       r26,LEVEL_LIST+16(r4)     ;# store list4
    or        r21,r21,r26               ;# dist4 |= list4
    stw       r31,LEVEL_COMP+16(r4)     ;# store comp4
    or        r31,r31,r21               ;# comp4 |= dist4
    addi      r14,r14,1                 ;# ++depth
    rlwinm    r8,r17,DIST_SH+2,DIST_MB-2,DIST_ME-2
    subi      r5,r5,1                   ;# --choose
    rlwinm    r0,r17,DIST_SH+3,DIST_MB-3,DIST_ME-3
    subi      r6,r6,SIZEOF_OGR          ;# --ogr
    add       r8,r8,r0                  ;# (dist0 >> DIST_SHIFT) * DIST_BITS
    addi      r4,r4,SIZEOF_LEVEL        ;# ++lev
    lbzx      r8,r8,r5                  ;# choose(dist0 >> ttmDISTBITS, remdepth)
    subi      r15,r15,1                 ;# --remdepth

L_comp_limit: 
    ;# Compute the maximum position (limit)
    ;# r8 = choose(dist >> ttmDISTBITS, maxdepthm1 - depth)
    ;# "depth <= halfdepth" is equivalent to "&lev[depth] <= &lev[halfdepth]"
    cmplw     r4,r12                    ;# depth <= oState->half_depth ?
    not       r0,r27                    ;# c0neg = ~comp0
    cmpw      cr1,r14,r13               ;# depth <= oState->half_depth2 ?
    srwi      r0,r0,1                   ;# prepare c0neg for cntlzw
    li        r9,1                      ;# newbit = 1
    ble-      L_left_side               ;# depth <= oState->half_depth
    addi      r10,r10,1                 ;# ++nodes
    sub       r8,r11,r8                 ;# limit = oState->max - choose[...]
    bgt+      cr1,L_stay                ;# depth > oState->half_depth2

    ;# oState->half_depth < depth <= oState->half_depth2
    lwz       r7,LEVEL_MARK(r12)        ;# levHalfDepth->mark
    sub       r7,r11,r7                 ;# temp = oState->max - levHalfDepth->mark
    cmpw      r8,r7                     ;# limit < temp ?
    blt       L_stay

    subi      r8,r7,1                   ;# limit = temp - 1
    b         L_stay

    .align    4
L_left_side:  
    ;# depth <= oState->half_depth
    lwz       r3,pNodes(r1)
    lwz       r7,oState(r1)             ;# oState ptr
    lwz       r3,0(r3)                  ;# max nodes = *pnodes
    lwz       r8,0(r6)                  ;# OGR[remdepth]
    cmpw      r10,r3                    ;# nodes >= max nodes ?
    lwz       r7,STATE_HALFLENGTH(r7)   ;# oState->half_length
    sub       r8,r11,r8                 ;# limit = oState->max - OGR[...]
    cmpw      cr1,r8,r7                 ;# limit > half_length ?
    bge-      L_exit_loop_CONT          ;# nodes >= max nodes : exit

    addi      r10,r10,1                 ;# ++nodes
    ble       cr1,L_stay                ;# limit <= oState->half_length

    mr        r8,r7                     ;# otherwise limit = half_length
    b         L_stay

    .align    4
L_shift32:                              ;# r9 = newbit
    cmpwi     cr1,r27,-1                ;# comp0 == -1 (i.e. s == 33) ?
    mr        r27,r28                   ;# comp0 = comp1
    mr        r28,r29                   ;# comp1 = comp2
    mr        r29,r30                   ;# comp2 = comp3
    mr        r30,r31                   ;# comp3 = comp4
    not       r0,r27                    ;# c0neg = ~comp0
    li        r31,0                     ;# comp4 = 0
    cmpwi     r15,0                     ;# remdepth == 0 ?
    srwi      r0,r0,1                   ;# c0neg >>= 1
    mr        r26,r25                   ;# list4 = list3
    mr        r25,r24                   ;# list3 = list2
    mr        r24,r23                   ;# list2 = list1
    mr        r23,r22                   ;# list1 = list0
    mr        r22,r9                    ;# list0 = newbit
    li        r9,0                      ;# newbit = 0
    beq+      cr1,L_stay                ;# shift again (s > 32)
    b         L_update_lev              ;# s == 32

    .align    4
L_up_level:   
    ;# POP_LEVEL(lev) : Restore the bitmaps then iterate
    subi      r4,r4,SIZEOF_LEVEL        ;# --lev
    subic.    r14,r14,1                 ;# --depth. Also set CR0
    lwz       r16,LEVEL_MARK(r4)        ;# Load mark
    li        r9,0                      ;# newbit = 0
    lwz       r22,LEVEL_LIST(r4)        ;# list0 = lev->list[0]
    addi      r6,r6,SIZEOF_OGR          ;# ++ogr
    lwz       r23,LEVEL_LIST+4(r4)      ;# list1 = lev->list[1]
    andc      r17,r17,r22               ;# dist0 &= ~list0
    lwz       r24,LEVEL_LIST+8(r4)      ;# list2 = lev->list[2]
    andc      r18,r18,r23               ;# dist1 &= ~list1
    lwz       r25,LEVEL_LIST+12(r4)     ;# list3 = lev->list[3]
    andc      r19,r19,r24               ;# dist2 &= ~list2
    lwz       r26,LEVEL_LIST+16(r4)     ;# list4 = lev->list[4]
    andc      r20,r20,r25               ;# dist3 &= ~list3
    lwz       r27,LEVEL_COMP(r4)        ;# comp0 = lev->comp[0]
    andc      r21,r21,r26               ;# dist4 &= ~list4
    lwz       r28,LEVEL_COMP+4(r4)      ;# comp1 = lev->comp[1]
    addi      r15,r15,1                 ;# ++remdepth
    lwz       r29,LEVEL_COMP+8(r4)      ;# comp2 = lev->comp[2]
    not       r0,r27                    ;# c0neg = ~comp0
    lwz       r30,LEVEL_COMP+12(r4)     ;# comp3 = lev->comp[3]
    srwi      r0,r0,1                   ;# Prepare c0neg
    lwz       r31,LEVEL_COMP+16(r4)     ;# comp4 = lev->comp[4]
    addi      r5,r5,1                   ;# ++choose
    lwz       r8,LEVEL_LIMIT(r4)        ;# Load limit
    ble-      L_exit_loop_OK            ;# depth <= 0 : exit

L_stay:
    ;# r0 = (~comp0) >> 1, so that cntlzw returns a value in the range [1;32]
    ;# r8 = limit
    ;# r9 = newbit
    cntlzw    r3,r0                     ;# s = Find first bit set [1;32]
    cmpwi     cr1,r0,0                  ;# Pre-check for case #32
    add       r16,r16,r3                ;# mark += s
    subfic    r0,r3,32                  ;# ss = 32 - s
    cmpw      r16,r8                    ;# mark > limit ?
    slw       r7,r9,r0                  ;# temp1 = newbit << ss
    bgt-      L_up_level                ;# Go back to the preceding mark
    stw       r8,LEVEL_LIMIT(r4)        ;# lev->limit = limit
    beq-      cr1,L_shift32             ;# s == 32
    slw       r27,r27,r3                ;# comp0 <<= s
    slw       r8,r22,r0                 ;# temp2 = list0 << ss
    srw       r22,r22,r3                ;# list0 >>= s
    slw       r9,r23,r0                 ;# temp3 = list1 << ss
    srw       r23,r23,r3                ;# list1 >>= s
    or        r22,r22,r7                ;# list0 |= temp1
    slw       r7,r24,r0                 ;# temp1 = list2 << ss
    srw       r24,r24,r3                ;# list2 >>= s
    or        r23,r23,r8                ;# list1 |= temp2
    slw       r8,r25,r0                 ;# temp2 = list3 << ss
    srw       r25,r25,r3                ;# list3 >>= s
    or        r24,r24,r9                ;# list2 |= temp3
    srw       r9,r28,r0                 ;# temp3 = comp1 >> ss
    srw       r26,r26,r3                ;# list4 >>= s
    or        r25,r25,r7                ;# list3 |= temp1
    srw       r7,r29,r0                 ;# temp1 = comp2 >> ss
    cmpwi     r15,0                     ;# remdepth == 0 ?
    slw       r28,r28,r3                ;# comp1 <<= s
    or        r26,r26,r8                ;# list4 |= temp2
    srw       r8,r30,r0                 ;# temp2 = comp3 >> ss
    or        r27,r27,r9                ;# comp0 |= temp3
    slw       r29,r29,r3                ;# comp2 <<= s
    or        r28,r28,r7                ;# comp1 |= temp1
    srw       r9,r31,r0                 ;# temp3 = comp4 >> ss
    or        r29,r29,r8                ;# comp2 |= temp2
    slw       r30,r30,r3                ;# comp3 <<= s
    not       r0,r27                    ;# c0neg = ~comp0
    slw       r31,r31,r3                ;# comp4 <<= s
    or        r30,r30,r9                ;# comp3 |= temp3
    srwi      r0,r0,1                   ;# c0neg >>= 1
    li        r9,0                      ;# newbit = 0

L_update_lev:                           ;# cr0 := remdepth == 0 ?
    stw       r16,LEVEL_MARK(r4)        ;# store mark position
    bne+      L_push_level              ;# go deeper

    ;# Last mark placed : verify the Golombness = found_one()
    ;# (This part is seldom used)

    ;# Backup volatile registers
    stw       r4,aStorage+0(r1)         ;# &oState->Levels[depth]
    stw       r5,aStorage+4(r1)         ;# &choose[3+remdepth]
    stw       r6,aStorage+8(r1)         ;# &OGR[remdepth]

    ;# Reset the diffs array
    srwi      r8,r11,3                  ;# maximum2 = oState->max / 8
    lwz       r4,oState(r1)
    addi      r7,r1,aDiffs-4            ;# diffs array
    lwz       r6,STATE_MAXDEPTHM1(r4)
    addi      r8,r8,1
    mtctr     r8
    li        r0,0

L_clrloop:    
    stwu      r0,4(r7)                  ;# diffs[k] = 0
    bdnz      L_clrloop

    addi      r4,r4,STATE_LEVELS        ;# &oState->Level[0]
    li        r9,1                      ;# Initial depth
    addi      r3,r4,SIZEOF_LEVEL        ;# levels[i=1]

L_iLoop:      
    lwz       r7,LEVEL_MARK(r3)         ;# levels[i].mark
    mr        r5,r4                     ;# levels[j=0]

L_jLoop:      
    lwz       r8,LEVEL_MARK(r5)         ;# levels[j].mark
    addi      r5,r5,SIZEOF_LEVEL        ;# ++j
    sub       r8,r7,r8                  ;# diffs = levels[i].mark - levels[j].mark
    cmpwi     r8,BITMAP_LENGTH          ;# diffs <= BITMAPS * 32 ?
    add       r0,r8,r8                  ;# 2*diffs
    cmpw      cr1,r0,r11                ;# 2*diff <= maximum ?
    addi      r0,r1,aDiffs              ;# &diffs[0]
    ble       L_next_i                  ;# diffs <= BITMAPS * 32 : break
    bgt       cr1,L_next_j              ;# diffs > maximum : continue

    lbzux     r0,r8,r0                  ;# diffs[diffs]
    cmpwi     r0,0                      ;# diffs[diffs] != 0 ?
    li        r0,1
    bne       L_not_golomb              ;# retval = CORE_S_CONTINUE
    stb       r0,0(r8)                  ;# Update the array

L_next_j:     
    cmplw     r5,r3                     ;# &diffs[j] < &diffs[i] ?
    blt       L_jLoop
L_next_i:     
    addi      r9,r9,1                   ;# ++i
    addi      r3,r3,SIZEOF_LEVEL
    cmpw      r9,r6                     ;# i <= maxdepthm1 ?
    ble       L_iLoop

    li        r3,CORE_S_SUCCESS         ;# Ruler is Golomb
    ;# Restore volatile registers
    lwz       r4,aStorage+0(r1)         ;# &oState->Levels[depth]
    lwz       r5,aStorage+4(r1)         ;# &choose[3+remdepth]
    lwz       r6,aStorage+8(r1)         ;# &OGR[remdepth]
    b         L_save_state              ;# Found it !

L_not_golomb: 
    ;# Restore volatile registers
    lwz       r4,aStorage+0(r1)         ;# &oState->Levels[depth]
    lwz       r5,aStorage+4(r1)         ;# &choose[3+remdepth]
    lwz       r6,aStorage+8(r1)         ;# &OGR[remdepth]
    ;# Restore clobbered regiters
    not       r0,r27                    ;# c0neg = ~comp0
    li        r9,0                      ;# newbit = 0
    srwi      r0,r0,1                   ;# Prepare c0neg
    lwz       r8,LEVEL_LIMIT(r4)        ;# Reload the limit
    b         L_stay                    ;# Not Golomb : iterate

L_exit_loop_OK: 
    li        r3,CORE_S_OK
    b         L_save_state

L_exit_loop_CONT: 
    li        r3,CORE_S_CONTINUE

L_save_state: 
    stw       r27,LEVEL_COMP+0(r4)      ;# comp0 bitmap
    stw       r17,LEVEL_DIST+0(r4)      ;# dist0 bitmap
    stw       r22,LEVEL_LIST+0(r4)      ;# list0 bitmap
    stw       r28,LEVEL_COMP+4(r4)      ;# comp1 bitmap
    stw       r18,LEVEL_DIST+4(r4)      ;# dist1 bitmap
    stw       r23,LEVEL_LIST+4(r4)      ;# list1 bitmap
    stw       r29,LEVEL_COMP+8(r4)      ;# comp2 bitmap
    stw       r19,LEVEL_DIST+8(r4)      ;# dist2 bitmap
    stw       r24,LEVEL_LIST+8(r4)      ;# list2 bitmap
    stw       r30,LEVEL_COMP+12(r4)     ;# comp3 bitmap
    stw       r20,LEVEL_DIST+12(r4)     ;# dist3 bitmap
    stw       r25,LEVEL_LIST+12(r4)     ;# list3 bitmap
    stw       r31,LEVEL_COMP+16(r4)     ;# comp4 bitmap
    stw       r21,LEVEL_DIST+16(r4)     ;# dist4 bitmap
    stw       r26,LEVEL_LIST+16(r4)     ;# list4 bitmap
    stw       r16,LEVEL_MARK(r4)
    lwz       r7,oState(r1)
    lwz       r8,pNodes(r1)
    lwz       r9,STATE_STARTDEPTH(r7)
    subi      r14,r14,1                 ;# --depth
    add       r14,r14,r9                ;# depth += startdepth
    stw       r10,0(r8)                 ;# Store node count
    stw       r14,STATE_DEPTH(r7)

;#============================================================================
;# Epilog

    ;# Restore non-volatile registers
    lwz       r5,0(r1)                  ;# Obtains caller's stack pointer
    lmw       r13,-GPRSaveArea(r5)      ;# Restore GPRs
    mr        r1,r5
    blr       

