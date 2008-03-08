;#
;# Copyright distributed.net 1999-2008 - All Rights Reserved
;# For use in distributed.net projects only.
;# Any other distribution or use of this source violates copyright.
;#
;# Hybrid Scalar/Vector - 256 bits OGR core for PowerPC processors.
;# Code scheduled for MPC744x/745x (G4+)
;# Written by Didier Levet
;#
;#============================================================================
;# Special notes :
;# - The code extensively use simplified mnemonics.
;# - Use a custom stack frame (leaf procedure).
;# - LR register not used nor saved in caller's stack.
;# - CTR, CR0, CR1, CR7, GPR0 and GPR3-GPR12 are volatile (not preserved).
;#
;#============================================================================


    .text     
    .align    4
    .globl    _cycle_ppc_hybrid_256     ;# a.out
    .globl    cycle_ppc_hybrid_256      ;# elf


;# OrgNgLevel dependencies (offsets)
.set          Levels_listV0,      0     ;# list vectors
.set          Levels_distV0,      32    ;# dist vectors
.set          Levels_compV0,      64    ;# comp vectors
.set          mark,               96
.set          limit,              100
.set          SIZEOF_LEVEL,       112


;# OgrNgState dependencies (offsets)
.set          MaxLength,          0     ;# max
.set          MaxDepth,           8     ;# maxdepthm1
.set          MidSegA,            12    ;# half_depth
.set          MidSegB,            16    ;# hald_depth2
.set          StopDepth,          24    ;# stopdepth
.set          Depth,              28    ;# depth
.set          Levels,             32


;# rlwinm arguments
.set          CHOOSE_BITS,        16
.set          SH,                 CHOOSE_BITS+6
.set          MB,                 32-CHOOSE_BITS-6
.set          ME,                 31-6


;#============================================================================
;# Custom stack frame

.set          FIRST_NV_GPR, 13          ;# Save r13..r31
.set          GPRSaveArea, (32-FIRST_NV_GPR) * 4

.set          wVRSave,-(GPRSaveArea+4)
.set          localTop, 16
.set          FrameSize, (localTop + GPRSaveArea + 15) & (-16)

;#============================================================================
;# Register aliases (GAS). Ignored by Apple's AS

;.set         r0,0
;.set         r1,1
;.set         r2,2
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
;.set         v9,9
;.set         v10,10
;.set         v11,11
;.set         v12,12
;.set         v13,13
;.set         VRsave, 0x100


;#============================================================================
;# int cycle_ppc_hybrid_256(struct OgrNgState* oState (r3)
;#                          int*               pNodes (r4)
;#                          const u16*         pChoose (r5)
;#                          const void*        pShift (r6)

_cycle_ppc_hybrid_256:
cycle_ppc_hybrid_256:
    mr        r10,r1                    ;# Caller's stack pointer
    clrlwi    r12,r1,27                 ;# keep the low-order 4 bits
    subfic    r12,r12,-FrameSize        ;# Frame size, including padding
    stwux     r1,r1,r12

    ;# Save non-volatile registers
    stmw      r13,-GPRSaveArea(r10)     ;# Save GPRs

    mfspr     r11,VRsave
    oris      r12,r11,0xfffe            ;# Use vector v0...v14
    stw       r11,wVRSave(r10)
    mtspr     VRsave,r12


;#============================================================================
;# Core engine initialization - Registers allocation :
;#   r3  := OgrNgState* oState (const)
;#   r4  := int* pNodes (const)
;#   r5  := const u16* pChoose[]
;#   r6  := const void* pShift[] (const)
;#   r7  := level
;#   r8  := Depth
;#   r9  := MaxDepth (const)
;#   r11 := MidSegA (const)
;#   r12 := MidSegB (const)
;#   r13 := MaxLength (const)
;#   r10 := StopDepth (const)
;#   r14 := comp0
;#   r15 := list0
;#   r16 := dist0
;#   r17 := newbit
;#   r18 := mark
;#   r19 := limit
;#   r20 := &Levels[half_depth]
;#   r21 := nodes
;#   v0  := zero (0, 0, 0, 0) (const)
;#   v1  := ones (0xff, 0xff, 0xff, 0xff) (const)
;#   v2  := vNewbit
;#   v9  := distV0
;#   v10 := listV0
;#   v11 := compV0
;#   v12 := distV1
;#   v13 := listV1
;#   v14 := compV1
;#   v3  := shift_l
;#   v4  := shift_r
;#   v5  := mask_l
;#   v6  := mask_r
;#   v7  := temp1
;#   v8  := temp2

    ;# Load core's parameters
    lwz       r21,0(r4)                 ;# Load nodes count
    lwz       r11,MidSegA(r3)           ;# half_depth
    lwz       r12,MidSegB(r3)           ;# half_depth2
    lwz       r13,MaxLength(r3)         ;# max
    lwz       r8,Depth(r3)              ;# depth
    lwz       r10,StopDepth(r3)         ;# stopdepth
    lwz       r9,MaxDepth(r3)           ;# maxdepthm1

    ;# Compute the pointer to Levels[Depth]
    mulli     r28,r8,SIZEOF_LEVEL       ;# Depth * SIZEOF_LEVEL
    add       r28,r28,r3
    addi      r7,r28,Levels             ;# Levels[Depth]

    ;# The pointer and offsets to the various vectors.
    li        r22,Levels_distV0         ;# index of distV0
    li        r23,Levels_distV0+16      ;# index of distV1
    li        r24,Levels_listV0         ;# index of listV0
    li        r25,Levels_listV0+16      ;# index of listV1
    li        r26,Levels_compV0         ;# index of compV0
    li        r27,Levels_compV0+16      ;# index of compV1

    ;# Compute the base pointer to access pre-computed limits
    ;# This pointer always points to pChoose[0][Depth]
    add       r29,r8,r8
    add       r5,r5,r29                 ;# &pChoose[0][Depth]

    ;# The maximum length is only used to compute a limit, and one is
    ;# always substracted to it.
    addi      r13,r13,-1                ;# --MaxLength

    ;# Initialize the vectors.
    vspltisw  v0,0                      ;# V_ZERO (constant)
    vspltisw  v1,-1                     ;# V_ONES (constant)
    vspltisw  v2,0                      ;# vNewbit = 0
    lvx       v9,r22,r7                 ;# distV0
    lvx       v10,r24,r7                ;# listV0
    lvx       v11,r26,r7                ;# compV0
    lvx       v12,r23,r7                ;# distV1
    lvx       v13,r25,r7                ;# listV1
    lvx       v14,r27,r7                ;# compV1

    ;# Compute the pointer to Levels[half_depth]
    mulli     r28,r11,SIZEOF_LEVEL      ;# MidSegA * SIZEOF_LEVEL
    add       r28,r28,r3
    addi      r20,r28,Levels            ;# = &Levels[MidSegA]

    lwz       r14,Levels_compV0(r7)     ;# comp0
    lwz       r15,Levels_listV0(r7)     ;# list0
    lwz       r16,Levels_distV0(r7)     ;# dist0

    not       r0,r14                    ;# ~comp0
    srwi      r0,r0,1                   ;# (~comp0) >> 1
    cmpw      r8,r9                     ;# Depth == MaxDepth ?

    lwz       r18,mark(r7)              ;# Levels[Depth].mark
    lwz       r19,limit(r7)             ;# Levels[Depth].limit
    li        r17,0                     ;# newbit = 0
    li        r31,-2                    ;# comp0_max = 0xFFFFFFFE
    beq-      L_MainLoop                ;# Depth == MaxDepth

    li        r17,1                     ;# newbit = 1
    vspltisw  v2,1                      ;# vNewbit = 1
    b         L_MainLoop


;#============================================================================

    .align    4
    nop       


L_UpLevel:
    ;# Backtrack...
    subi      r7,r7,SIZEOF_LEVEL        ;# --Levels
    li        r17,0                     ;# newbit = 0
    subi      r8,r8,1                   ;# --Depth
    lwz       r14,Levels_compV0(r7)     ;# comp0
    cmpw      r8,r10                    ;# Depth > StopDepth
    lwz       r15,Levels_listV0(r7)     ;# list0
    subi      r5,r5,2                   ;# --pChoose
    lvx       v10,r24,r7                ;# listV0
    vspltisw  v2,0                      ;# vNewbit = 0
    lvx       v13,r25,r7                ;# listV1
    not       r0,r14                    ;# v0neg = ~comp0
    lwz       r18,mark(r7)              ;# mark = Levels[Depth].mark
    andc      r16,r16,r15               ;# dist0 &= ~list0
    srwi      r0,r0,1                   ;# v0neg = (~comp0) >> 1
    lwz       r19,limit(r7)             ;# limit = Levels[Depth].limit
    vandc     v9,v9,v10                 ;# distV0 &= ~listV0
    lvx       v11,r26,r7                ;# compV0
    vandc     v12,v12,v13               ;# distV1 &= ~listV1
    lvx       v14,r27,r7                ;# compV1
    ble-      L_exit                    ;# Depth <= StopDepth


L_MainLoop:
;#  r0 := (~comp0) >> 1
    cntlzw    r28,r0                    ;# diff
    cmplw     cr1,r14,r31               ;# comp0 >= 0xFFFFFFFE ?
    add       r18,r18,r28               ;# mark += diff
    slwi      r30,r28,4                 ;# diff * 16
    cmpw      r18,r19                   ;# mark > limit ?
    subfic    r29,r28,32                ;# ss = 32 - diff
    lvx       v3,r6,r30                 ;# shift_l = pShift[diff]
    bgt-      L_UpLevel                 ;# Go back to preceding mark.

    srw       r15,r15,r28               ;# list0 >>= diff
    lwz       r30,Levels_compV0+4(r7)   ;# load comp1
    slw       r17,r17,r29               ;# newbit <<= ss
    or        r15,r15,r17               ;# list0 |= newbit
    bge       cr1,L_Shift32             ;# diff >= 32

    ;# Shift bitmaps by 'diff' bits.
    ;# r28 := diff
    ;# r29 := ss
    ;# r30 := comp1
    ;# r15 := list0 = (list0 >> diff) | (newbit << ss)
    ;# v3 := shift_l
    cmpw      r8,r9                     ;# Depth == MaxDepth ?
    vsldoi    v7,v11,v14,4              ;# compV0::compV1 << 4
    vsubuwm   v4,v0,v3                  ;# shift_r
    vsldoi    v8,v10,v13,12             ;# listV0::listV1 << 12
    vslo      v14,v14,v3
    vsrw      v5,v1,v3                  ;# mask_l
    vsl       v14,v14,v3                ;# = compV1
    vsel      v11,v7,v11,v5
    vsldoi    v7,v2,v10,12              ;# vNewbit::listV1 << 12
    vslw      v6,v1,v3                  ;# mask_r
    vrlw      v11,v11,v3                ;# = compV0
    slw       r14,r14,r28               ;# comp0 <<= diff
    vsel      v13,v8,v13,v6
    srw       r30,r30,r29               ;# comp1 >>= ss
    vsel      v10,v7,v10,v6
    stvx      v11,r26,r7                ;# store compV0
    vrlw      v13,v13,v4                ;# = listV1
    or        r14,r14,r30               ;# comp0 |= comp1
    vrlw      v10,v10,v4                ;# = listV0
    beq-      L_exit                    ;# Ruler found

L_UpdateLevel:
    vor       v9,v9,v10                 ;# distV0 |= listV0
    addi      r8,r8,1                   ;# ++Depth
    stvx      v10,r24,r7                ;# store listV0
    vor       v11,v11,v9                ;# compV0 |= distV0
    cmplw     cr1,r8,r12                ;# Depth <= MidSegB ?
    stvx      v14,r27,r7                ;# store compV1
    vor       v12,v12,v13               ;# distV1 |= listV1
    or        r16,r16,r15               ;# dist0 |= list0
    stvx      v13,r25,r7                ;# store listV1
    addi      r5,r5,2                   ;# ++pChoose
    rlwinm    r28,r16,SH,MB,ME          ;# 32*2*(dist0 >> CHOOSE_BITS)
    stw       r18,mark(r7)              ;# Store the current mark
    vor       v14,v14,v12               ;# compV1 |= distV1
    cmplw     cr7,r8,r11                ;# Depth > MidSegA ?
    lhzx      r19,r28,r5                ;# load the limit
    addi      r7,r7,SIZEOF_LEVEL        ;# ++Levels
    or        r14,r14,r16               ;# comp0 |= dist0
    subic.    r21,r21,1                 ;# --nodes <= 0 ?
    vspltisw  v2,1                      ;# vNewbit = 1
    not       r0,r14                    ;# v0neg = ~comp0
    stvx      v11,r26,r7                ;# store compV0 (next level)
    li        r17,1                     ;# newbit = 1
    srwi      r0,r0,1                   ;# v0neg = (~comp0) >> 1
    bgt+      cr1,L_CheckCnt            ;# Depth > MidSegB
    ble+      cr7,L_CheckCnt            ;# Depth <= MidSegA


L_GetLimit:
    ;# Compute the limit within the middle segment.
    ;# cr1 := Depth == MidSegB
    ;# r19 := limit

    lwz       r30,mark(r20)             ;# Levels[MidSegA].mark
    sub       r28,r13,r30               ;# temp
    not       r29,r16                   ;# ~dist0
    beq       cr1,L_adjust              ;# Depth == MidSegB

    cntlzw    r29,r29                   ;# FFS(dist0)
    addi      r29,r29,1
    sub       r28,r28,r29               ;# temp -= FFS(dist0)

L_adjust:                               ;# Compute : limit = min(temp, limit)
    subfc     r29,r19,r28
    subfe     r28,r28,r28
    and       r29,r29,r28
    add       r19,r19,r29


L_CheckCnt:
    ;# cr0 := nodes <= 0

    stw       r19,limit(r7)             ;# store the limit
    bgt+      L_MainLoop                ;# nodes > 0

    b         L_exit


    .align    4
L_Shift32:
    ;# Shift bitmaps by 32 bits.
    ;# cr1 := comp0 >= 0xFFFFFFFE
    ;# r30 := comp1
    cmpw      r8,r9                     ;# Depth == MaxDepth ?
    vsldoi    v11,v11,v14,4             ;# = compV0
    mr        r14,r30                   ;# comp0 = comp1
    vsldoi    v13,v10,v13,12            ;# = listV1
    li        r17,0                     ;# newbit = 0
    not       r0,r14                    ;# ~comp0
    vsldoi    v10,v2,v10,12             ;# = listV0
    vor       v2,v0,v0                  ;# vNewbit = 0
    srwi      r0,r0,1                   ;# (~comp0) >> 1
    vsldoi    v14,v14,v0,4              ;# = compV1
    stvx      v11,r26,r7                ;# store = compV0
    bgt+      cr1,L_MainLoop            ;# Shift count > 32
    bne+      L_UpdateLevel             ;# Depth != MaxDepth


L_exit:
    ;# Save state
    stw       r18,mark(r7)              ;# Store the current mark
    stvx      v9,r22,r7                 ;# distV0
    stvx      v10,r24,r7                ;# listV0
    stvx      v11,r26,r7                ;# compV0
    stvx      v12,r23,r7                ;# distV1
    stvx      v13,r25,r7                ;# listV1
    stvx      v14,r27,r7                ;# compV1
    lwz       r28,0(r4)                 ;# *pNodes
    sub       r28,r28,r21               ;# -= nodes
    stw       r28,0(r4)                 ;# *pNodes
    mr        r3, r8                    ;# return actual depth


;#============================================================================
;# Epilog

    ;# Restore non-volatile registers
    lwz       r5,0(r1)                  ;# Obtains caller's stack pointer
    lmw       r13,-GPRSaveArea(r5)      ;# Restore GPRs
    lwz       r6,wVRSave(r5)            ;# Restore VRsave
    mtspr     VRsave,r6
    mr        r1,r5
    blr       

