;#
;# Hybrid Scalar/Vector OGR core for PowerPC processors.
;# Code scheduled for MPC744x/745x (G4+)
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
;# $Id: OGR_PPC_hybrid.gas.s,v 1.1.2.5 2004/08/15 14:43:26 kakace Exp $
;#
;#============================================================================


    .text     
    .align    4
    .globl    _cycle_ppc_hybrid         ;# a.out
    .globl    cycle_ppc_hybrid          ;# elf


;# Bitmaps dependencies (offsets)
.set          LEVEL_DISTV,        32    ;# distV vector
.set          LEVEL_LISTV,        0     ;# listV vector
.set          LEVEL_COMPV,        16    ;# compV vector
.set          LEVEL_DIST0,        56    ;# dist0 scalar
.set          LEVEL_LIST0,        60    ;# list0 scalar
.set          LEVEL_COMP0,        52    ;# comp0 scalar


;# Structure members dependencies (offsets)
.set          STATE_DEPTH,        28
.set          STATE_LEVELS,       32
.set          SIZEOF_LEVEL,       80
.set          STATE_MAX,          0     ;# OGR[stub.marks-1] = max length
.set          STATE_MAXDEPTHM1,   8     ;# stub.marks - 1
.set          STATE_STARTDEPTH,   24    ;# workstub->stub.length
.set          STATE_HALFDEPTH,    16    ;# first mark of the middle segment
.set          STATE_HALFDEPTH2,   20    ;# last mark of the middle segment
.set          STATE_HALFLENGTH,   12    ;# maximum position of the middle segment
.set          LEVEL_MARK,         64
.set          LEVEL_LIMIT,        48
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

.set          aDiffs, 16                ;# diffs control array
.set          wVRSave,-(GPRSaveArea+4)
.set          localTop, 1040
.set          FrameSize, (localTop + GPRSaveArea + 15) & (-16)


;#============================================================================
;# Register aliases (GAS). Ignored by Apple's AS

;.set         r0,0
;.set         r1,1
;.set         RTOC,2                    ;# Alias for AIX/COFF
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

;.set         v0,0
;.set         v1,1
;.set         v2,2
;.set         v3,3
;.set         v4,4
;.set         v5,5
;.set         v6,6
;.set         v7,7
;.set         v8,8
;.set         VRsave,0x100


;#============================================================================
;# int cycle_ppc_hybrid(void *state (r3)
;#                      int *pnodes (r4)
;#                      const unsigned char *choose (r5)
;#                      const int *OGR (r6)
;#                      const vector unsigned char Varray[] (r7)

cycle_ppc_hybrid:                       ;# elf
_cycle_ppc_hybrid:                      ;# a.out
    mr        r10,r1                    ;# Caller's stack pointer
    clrlwi    r12,r1,27                 ;# keep the low-order 4 bits
    subfic    r12,r12,-FrameSize        ;# Frame size, including padding
    stwux     r1,r1,r12

    ;# Save non-volatile registers
    stmw      r13,-GPRSaveArea(r10)     ;# Save GPRs

    mfspr     r11,VRsave
    oris      r12,r11,0xff80            ;# Use vector v0...v8
    stw       r11,wVRSave(r10)
    mtspr     VRsave,r12


;#============================================================================
;# Core engine initialization - Registers allocation :
;#   r4  := int *pnodes (const)
;#   r5  := &choose[3+remdepth]
;#   r6  := &OGR[remdepth]
;#   r7  := &Varray[0] (const)
;#   r13 := struct State *oState (const)
;#   r14 := &oState->Levels[depth]
;#   r15 := &oState->Levels[halfdepth]
;#   r16 := depth
;#   r17 := nodes
;#   r18 := oState->max (const)
;#   r19 := remdepth
;#   r20 := oState->half_depth2 (const)
;#   r21 := mark
;#   r22 := nodes max (*pnodes, const)
;#   r23 := c0neg
;#   r24 := limit
;#   r25 := offsetof(Level, list0) (const)
;#   r26 := offsetof(Level, distV) (const)
;#   r27 := offsetof(Level, listV) (const)
;#   r28 := offsetof(Level, compV) (const)
;#   r29 := dist[0]
;#   r30 := list[0]
;#   r31 := comp[0]
;#   v0  := zeroes (0, 0, 0, 0) (const)
;#   v1  := ones (0xff, 0xff, 0xff, 0xff) (const)
;#   v6  := dist bitmap
;#   v7  := list bitmap
;#   v8  := comp bitmap

    mr        r13,r3                    ;# Copy oState
    lwz       r16,STATE_DEPTH(r3)       ;# oState->depth
    addi      r16,r16,1                 ;# ++depth

    addi      r14,r3,STATE_LEVELS       ;# &oState->Levels[0]
    lwz       r3,STATE_HALFDEPTH(r13)   ;# oState->half_depth
    mulli     r3,r3,SIZEOF_LEVEL
    add       r15,r14,r3                ;# &oState->Levels[half_depth]
    mulli     r3,r16,SIZEOF_LEVEL
    add       r14,r14,r3                ;# &oState->Levels[depth]

    li        r17,0                     ;# nodes = 0
    lwz       r21,LEVEL_MARK(r14)       ;# level->mark

    lwz       r3,STATE_STARTDEPTH(r13)  ;# oState->startdepth
    lwz       r18,STATE_MAX(r13)        ;# oState->max
    lwz       r19,STATE_MAXDEPTHM1(r13) ;# oState->maxdepthm1
    lwz       r20,STATE_HALFDEPTH2(r13) ;# oState->half_depth2
    sub       r19,r19,r16               ;# remdepth = maxdepthm1 - depth
    sub       r20,r20,r3                ;# halfdepth2 -= startdepth
    sub       r16,r16,r3                ;# depth -= startdepth

    lwz       r22,0(r4)                 ;# nodes max = *pnodes
    slwi      r3,r19,OGR_SHIFT          ;# remdepth * sizeof(int)
    add       r5,r5,r19                 ;# &choose[remdepth]
    add       r6,r6,r3                  ;# &OGR[remdepth]

    ;# SETUP_TOP_STATE
    ;# - Initialize vector constants required to shift bitmaps
    ;# - Load current state
    vsubuwm   v0,v0,v0                  ;# zeroes = vector (0)
    vnor      v1,v0,v0                  ;# ones = vector (-1)
    li        r27,LEVEL_LISTV
    li        r26,LEVEL_DISTV
    li        r28,LEVEL_COMPV
    li        r25,LEVEL_LIST0
    lwz       r29,LEVEL_DIST0(r14)      ;# dist0 bitmap
    lwz       r31,LEVEL_COMP0(r14)      ;# comp0 bitmap
    lwz       r30,LEVEL_LIST0(r14)      ;# list0 bitmap
    lvx       v6,r26,r14                ;# dist vector
    lvx       v8,r28,r14                ;# comp vector
    lvx       v7,0,r14                  ;# list vector
    rlwinm    r3,r29,DIST_SH+2,DIST_MB-2,DIST_ME-2
    rlwinm    r0,r29,DIST_SH+3,DIST_MB-3,DIST_ME-3
    add       r0,r3,r0                  ;# (dist0 >> DIST_SHIFT) * DIST_BITS
    lbzx      r24,r5,r0                 ;# choose(dist0 >> ttmDISTBITS, remdepth)
    b         L_comp_limit
    nop       

    .align    4
L_push_level: 
    ;# PUSH_LEVEL_UPDATE_STATE(lev)
    addi      r3,r14,SIZEOF_LEVEL       ;# lev+1
    vor       v6,v6,v7                  ;# distV |= listV
    stw       r30,LEVEL_LIST0+SIZEOF_LEVEL(r14) ;# (lev+1)->list0 = list0
    or        r29,r29,r30               ;# dist0 |= list0
    vor       v8,v8,v6                  ;# compV |= distV
    stw       r31,LEVEL_COMP0(r14)      ;# store comp0
    rlwinm    r8,r29,DIST_SH+2,DIST_MB-2,DIST_ME-2
    subi      r6,r6,SIZEOF_OGR          ;# --ogr
    stvx      v8,r28,r3                 ;# (lev+1)->compV.v = compV
    rlwinm    r0,r29,DIST_SH+3,DIST_MB-3,DIST_ME-3
    or        r31,r31,r29               ;# comp0 |= dist0
    stw       r24,LEVEL_LIMIT(r14)      ;# store limit
    add       r0,r8,r0                  ;# (dist0 >> DIST_SHIFT) * DIST_BITS
    subi      r5,r5,1                   ;# --choose
    lbzx      r24,r5,r0                 ;# choose(dist0 >> ttmDISTBITS, remdepth)
    subi      r19,r19,1                 ;# --remdepth
    addi      r16,r16,1                 ;# ++depth
    stvx      v7,0,r14                  ;# store listV
    addi      r14,r14,SIZEOF_LEVEL      ;# ++lev

L_comp_limit: 
    ;# Compute the maximum position (limit)
    ;# "depth <= halfdepth" is equivalent to "&lev[depth] <= &lev[halfdepth]"
    cmplw     r14,r15                   ;# depth <= oState->half_depth ?
    not       r23,r31                   ;# c0neg = ~comp0
    li        r12,1                     ;# newbit = 1
    cmpw      cr1,r16,r20               ;# depth <= oState->half_depth2 ?
    srwi      r23,r23,1                 ;# prepare c0neg for cntlzw
    ble-      L_left_side               ;# depth <= oState->half_depth
    addi      r17,r17,1                 ;# ++nodes
    sub       r24,r18,r24               ;# limit = oState->max - choose[...]
    bgt+      cr1,L_stay                ;# depth > oState->half_depth2

    ;# oState->half_depth < depth <= oState->half_depth2
    lwz       r3,LEVEL_MARK(r15)        ;# levHalfDepth->mark
    sub       r3,r18,r3                 ;# temp = oState->max - levHalfDepth->mark
    cmpw      r24,r3                    ;# limit < temp ?
    blt       L_stay

    subi      r24,r3,1                  ;# limit = temp - 1
    b         L_stay

L_left_side:                            ;# depth <= oState->half_depth
    lwz       r8,0(r6)                  ;# OGR[remdepth]
    cmpw      r17,r22                   ;# nodes >= *pnodes ?
    lwz       r9,STATE_HALFLENGTH(r13)  ;# oState->half_length
    sub       r24,r18,r8                ;# limit = oState->max - OGR[...]
    li        r3,CORE_S_CONTINUE        ;# pre-load
    cmpw      cr1,r24,r9                ;# limit > half_length ?
    bge-      L_exit_loop               ;# nodes >= pnodes : exit

    addi      r17,r17,1                 ;# ++nodes
    ble       cr1,L_stay                ;# limit <= oState->half_length

    mr        r24,r9                    ;# otherwise limit = half_length
    b         L_stay

    .align    4
L_shift32:    
    ;# r8  = comp1 = compV.u[0]
    ;# r30 = (list0 >> 32) | (newbit << 0) = list0 = newbit
    ;# v4  = listV1 = INT_TO_VEC(list0)
    cmpwi     cr1,r31,-1                ;# comp0 == -1 (i.e. s == 33) ?
    vsldoi    v8,v8,v0,4                ;# compV = (compV:zeroes) << 32
    cmpwi     r19,0                     ;# remdepth == 0 ?
    stw       r30,LEVEL_LIST0(r14)      ;# lev->list0 = list0
    not       r23,r8                    ;# c0neg = ~comp1
    mr        r31,r8                    ;# comp0 = comp1
    vsldoi    v7,v4,v7,12               ;# listV = (listV1:listV) << 96
    srwi      r23,r23,1                 ;# Prepare c0neg
    stvx      v8,r28,r14                ;# lev->compV = compV
    li        r12,0                     ;# newbit = 0
    beq+      cr1,L_stay                ;# shift again (s > 32)
    ;# else s == 32
    stw       r21,LEVEL_MARK(r14)       ;# store mark
    bne+      L_push_level              ;# remdepth > 0
    b         L_check_Golomb            ;# last mark : Check Golombness

    .align    4
L_up_level:   
    ;# POP_LEVEL(lev) : Restore the bitmaps then iterate
    lwz       r31,LEVEL_COMP0-SIZEOF_LEVEL(r14) ;# comp0 = (lev-1)->comp0
    subi      r14,r14,SIZEOF_LEVEL      ;# --lev
    subic.    r16,r16,1                 ;# --depth. Also set CR0
    lvx       v7,0,r14                  ;# Load listV
    addi      r5,r5,1                   ;# ++choose
    addi      r19,r19,1                 ;# ++remdepth
    lwz       r30,LEVEL_LIST0(r14)      ;# Load list0
    addi      r6,r6,SIZEOF_OGR          ;# ++ogr
    li        r3,CORE_S_OK
    lvx       v8,r28,r14                ;# Load compV
    not       r23,r31                   ;# c0neg = ~comp0
    li        r12,0                     ;# newbit = 0
    lwz       r24,LEVEL_LIMIT(r14)      ;# Load limit
    srwi      r23,r23,1                 ;# Prepare c0neg
    vandc     v6,v6,v7                  ;# distV &= ~listV
    lwz       r21,LEVEL_MARK(r14)       ;# Load mark
    andc      r29,r29,r30               ;# dist0 &= ~list0
    ble-      L_exit_loop               ;# depth <= 0 : exit

L_stay:       
    ;# r23 = (~comp0) >> 1, so that cntlzw returns a value in the range [1;32]
    ;# r12 = newbit
    cntlzw    r0,r23                    ;# Find first bit set
    cmpwi     cr1,r23,0                 ;# Pre-check for case #32
    add       r21,r21,r0                ;# mark += s
    slwi      r11,r0,4                  ;# temp = s * 16
    lvewx     v4,r25,r14                ;# listV1 = vec_lde(list0)
    cmpw      r21,r24                   ;# mark > limit ?
    subfic    r3,r0,32                  ;# ss = 32 - s
    lvx       v2,r11,r7                 ;# Vs = Varray[s]
    bgt-      L_up_level                ;# Go back to the preceding mark
    srw       r30,r30,r0                ;# list0 >>= s
    lwz       r8,LEVEL_COMPV(r14)       ;# comp1 = lev->compV.u[0]
    slw       r12,r12,r3                ;# newbit <<= ss
    or        r30,r30,r12               ;# list0 |= newbit
    beq-      cr1,L_shift32             ;# s == 32

    cmpwi     r19,0                     ;# remdepth == 0 ?
    vslo      v8,v8,v2                  ;# compV = vec_slo(compV, Vs)
    stw       r30,LEVEL_LIST0(r14)      ;# lev->list0 = list0
    vsubuwm   v3,v0,v2                  ;# Vss = zeroes - Vs
    vsldoi    v4,v4,v7,12               ;# listV1 = (listV1:listV) << 96
    stw       r21,LEVEL_MARK(r14)       ;# store mark
    vslw      v5,v1,v2                  ;# bmV = vec_sl(ones, Vs)
    vsl       v8,v8,v2                  ;# compV <<= Vs
    slw       r31,r31,r0                ;# comp0 <<= s
    srw       r8,r8,r3                  ;# comp1 >>= ss
    vsel      v7,v4,v7,v5               ;# listV = vec_sel(listV1, listV, bmV)
    stvx      v8,r28,r14                ;# lev->compV = compV
    or        r31,r31,r8                ;# comp0 |= comp1
    vrlw      v7,v7,v3                  ;# listV = vec_rl(listV, Vss)
    li        r12,0                     ;# newbit = 0
    bne+      L_push_level              ;# remdepth > 0

L_check_Golomb: 
    ;# Last mark placed : verify the Golombness = found_one()
    ;# This part is seldom used.

    ;# Reset the diffs array
    srwi      r8,r18,5                  ;# maximum2 = oState->max / 32
    li        r3,aDiffs                 ;# diffs array
    addi      r8,r8,1
    mtctr     r8
    lwz       r23,STATE_MAXDEPTHM1(r13) ;# oState->maxdepthm1

L_clrloop:    
    stvx      v0,r3,r1
    addi      r3,r3,16
    bdnz      L_clrloop

    addi      r12,r13,STATE_LEVELS      ;# &oState->Level[0]
    li        r9,1                      ;# Initial depth
    addi      r10,r12,SIZEOF_LEVEL      ;# levels[i=1]

L_iLoop:      
    lwz       r3,LEVEL_MARK(r10)        ;# levels[i].mark
    mr        r11,r12                   ;# levels[j=0]

L_jLoop:      
    lwz       r8,LEVEL_MARK(r11)        ;# levels[j].mark
    addi      r11,r11,SIZEOF_LEVEL      ;# ++j
    sub       r8,r3,r8                  ;# diffs = levels[i].mark - levels[j].mark
    cmpwi     r8,BITMAP_LENGTH          ;# diffs <= BITMAPS * 32 ?
    add       r0,r8,r8                  ;# 2*diffs
    cmpw      cr1,r0,r18                ;# 2*diff <= maximum ?
    addi      r0,r1,aDiffs              ;# &diffs[0]
    ble       L_next_i                  ;# diffs <= BITMAPS * 32 : break
    bgt       cr1,L_next_j              ;# diffs > maximum : continue

    lbzux     r0,r8,r0                  ;# diffs[diffs]
    cmpwi     r0,0                      ;# diffs[diffs] != 0 ?
    ori       r0,r0,1
    bne       L_not_golomb              ;# retval = CORE_S_CONTINUE
    stb       r0,0(r8)                  ;# Update the array

L_next_j:     
    cmplw     r11,r10                   ;# &diffs[j] < &diffs[i] ?
    blt       L_jLoop
L_next_i:     
    addi      r9,r9,1                   ;# ++i
    addi      r10,r10,SIZEOF_LEVEL
    cmpw      r9,r23                    ;# i <= maxdepthm1 ?
    ble       L_iLoop

    li        r3,CORE_S_SUCCESS         ;# Ruler is Golomb
    b         L_exit_loop               ;# Found it !

L_not_golomb: 
    not       r23,r31                   ;# c0neg = ~comp0
    li        r12,0                     ;# newbit = 0
    srwi      r23,r23,1                 ;# Prepare c0neg
    b         L_stay                    ;# Not Golomb : iterate

L_exit_loop:  
    ;# SAVE_FINAL_STATE(oState,lev)
    stvx      v7,0,r14                  ;# Store listV
    stvx      v6,r26,r14                ;# Store distV
    stvx      v8,r28,r14                ;# Store compV
    stw       r30,LEVEL_LIST0(r14)      ;# Store list0
    stw       r29,LEVEL_DIST0(r14)      ;# Store dist0
    stw       r31,LEVEL_COMP0(r14)      ;# Store comp0
    stw       r21,LEVEL_MARK(r14)       ;# Store mark
    lwz       r8,STATE_STARTDEPTH(r13)
    subi      r16,r16,1                 ;# --depth
    add       r16,r16,r8                ;# depth += startdepth
    stw       r17,0(r4)                 ;# Store node count
    stw       r16,STATE_DEPTH(r13)


;#============================================================================
;# Epilog

    ;# Restore non-volatile registers
    lwz       r5,0(r1)                  ;# Obtains caller's stack pointer
    lmw       r13,-GPRSaveArea(r5)      ;# Restore GPRs
    lwz       r6,wVRSave(r5)            ;# Restore VRsave
    mtspr     VRsave,r6
    mr        r1,r5
    blr       

