;#
;# Scalar OGR core for PowerPC processors.
;# Code scheduled for MPC750 (G3)
;# Written by Didier Levet (kakace@distributed.net)
;#
;# Copyright 2004 distributed.net, All Rights Reserved
;# For use in distributed.net projects only.
;# Any other distribution or use of this source violates copyright.
;#
;#============================================================================
;# Platform/assembler specific implementation :
;# - External symbols ("cycle_ppc_scalar" and "found_one") have no
;#   underscore prefix.
;# - Some simplified mnemonics are used.
;# - ha16() and lo16() directives have been replaced with @ha and @l.
;# - Stack frame based upon Mac OS/AIX ABI.
;# - CTR, CR0, CR1, GPR0 and GPR3-GPR12 are volatile (not saved).
;#
;# $Id: OGR_PPC_scalar.gas.s,v 1.1.2.1 2004/07/14 01:11:47 piru Exp $
;#
;#============================================================================


    .text                               
    .align    5                         
    .globl    cycle_ppc_scalar         


;# Bitmaps dependencies (offsets)
.set          LEVEL_LIST          ,56   ;# list[] bitmap
.set          LEVEL_COMP          ,96   ;# comp[] bitmap
.set          STATE_DIST          ,224  ;# dist[] bitmap


;# Constants
.set          CORE_S_CONTINUE, 1        
.set          CORE_S_OK, 0              
.set          DIST_SHIFT, 20            
.set          DIST_BITS, 12             

;# Parameters for rlwinm (choose addressing)
.set          DIST_SH, 32-DIST_SHIFT    
.set          DIST_MB, 32-DIST_BITS     
.set          DIST_ME, 31               


;# Structure members dependencies (offsets)
.set          STATE_DEPTH,        192   
.set          STATE_LEVELS,       256   
.set          SIZEOF_LEVEL,       128   
.set          LEVEL_SHIFT,        7     ;# sizeof(struct Level) is a multiple of 2
.set          STATE_MAX,          0     ;# OGR[stub.marks-1] = length max
.set          STATE_MAXDEPTHM1,   8     ;# stub.marks - 1
.set          STATE_STARTDEPTH,   188   ;# workstub->stub.length
.set          STATE_HALFDEPTH,    16    
.set          STATE_HALFDEPTH2,   20    
.set          STATE_HALFLENGTH,   12    
.set          LEVEL_CNT2,         120   
.set          LEVEL_LIMIT,        124   
.set          SIZEOF_OGR,         4     
.set          OGR_SHIFT,          2     


;#============================================================================
;# Register aliases (GAS)

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


;#============================================================================
;# Mac OS/AIX stack frame
.set          FIRST_NV_GPR, 13          ;# Save r13..r31
.set          GPRSaveArea, (32-FIRST_NV_GPR) * 4 

.set          wSaveLR, 8                ;# Save LR
.set          aStorage, 28              ;# Private storage area
.set          localTop, 80              
.set          FrameSize, (localTop + GPRSaveArea + 15) & (-16) 

.set          oState, aStorage + 36     
.set          pNodes, aStorage + 40     


;#============================================================================
;# int cycle_ppc_scalar(void *state (r3)
;#                      int *pnodes (r4)
;#                      const unsigned char *choose (r5)
;#                      const int *OGR (r6)

cycle_ppc_scalar:                      
    ;# Allocate the new stack frame
    mflr      r9                        
    stw       r9,wSaveLR(r1)            ;# ...in caller's stack
    mr        r8,r1                     ;# Caller's stack pointer
    clrlwi    r11,r1,27                 ;# keep the low-order 4 bits
    subfic    r11,r11,-FrameSize        ;# Frame size, including padding
    stwux     r1,r1,r11                 

    ;# Save non-volatile registers
    stmw      r13,-GPRSaveArea(r8)      ;# Save GPRs


;#============================================================================
;# Core engine initialization - Registers allocation :
;#   r3  := &oState->Levels[depth]
;#   r5  := &choose[3] (const)
;#   r6  := &OGR[0] (const)
;#   r4  := max nodes (const)
;#   r9  := depth
;#   r10 := nodes
;#   r11 := oStateMax (const)
;#   r12 := &oState->Levels[halfdepth] (const)
;#   r13 := oStateHaldDepth2 (const)
;#   r14 := startdepth (const)
;#   r15 := maxdepthm1 (const)
;#   r16 := cnt2
;#   r17-r21 := dist[0]...dist[4]
;#   r22-r26 := list[0]...list[4]
;#   r27-r31 := comp[0]...comp[4]

    lwz       r9,STATE_DEPTH(r3)        ;# oState->depth
    li        r10,0                     ;# nodes = 0
    lwz       r12,STATE_HALFDEPTH(r3)   ;# oState->half_depth
    addi      r9,r9,1                   ;# ++depth
    lwz       r15,STATE_MAXDEPTHM1(r3)  ;# oState->maxdepthm1
    lwz       r13,STATE_HALFDEPTH2(r3)  ;# oState->half_depth2
    lwz       r11,STATE_MAX(r3)         ;# oStateMax
    sub       r0,r15,r9                 ;# remainingDepth = maxdepthm1 - depth
    lwz       r14,STATE_STARTDEPTH(r3)  ;# oState->startdepth
    slwi      r7,r0,OGR_SHIFT           ;# remainingDepth * sizeof(int)
    stw       r3,oState(r1)             ;# Save oState
    addi      r3,r3,STATE_LEVELS        ;# &oState->Levels[0]
    stw       r4,pNodes(r1)             ;# Save pNodes
    add       r5,r5,r0                  ;# &choose[remainingDepth]
    slwi      r8,r9,LEVEL_SHIFT         
    slwi      r12,r12,LEVEL_SHIFT       
    lwz       r4,0(r4)                  ;# max nodes = *pnodes
    add       r6,r6,r7                  ;# &OGR[remainingDepth]
    add       r12,r3,r12                ;# &oState->Levels[halfdepth]
    add       r3,r3,r8                  ;# &oState->Levels[depth]

    ;# SETUP_TOP_STATE
    lwz       r7,oState(r1)             
    lwz       r27,LEVEL_COMP+0(r3)      ;# comp0 bitmap
    lwz       r22,LEVEL_LIST+0(r3)      ;# list0 bitmap
    lwz       r17,STATE_DIST+0(r7)      ;# dist0 bitmap
    lwz       r28,LEVEL_COMP+4(r3)      ;# comp1 bitmap
    lwz       r23,LEVEL_LIST+4(r3)      ;# list1 bitmap
    lwz       r18,STATE_DIST+4(r7)      ;# dist1 bitmap
    lwz       r29,LEVEL_COMP+8(r3)      ;# comp2 bitmap
    lwz       r24,LEVEL_LIST+8(r3)      ;# list2 bitmap
    lwz       r19,STATE_DIST+8(r7)      ;# dist2 bitmap
    lwz       r30,LEVEL_COMP+12(r3)     ;# comp3 bitmap
    lwz       r25,LEVEL_LIST+12(r3)     ;# list3 bitmap
    lwz       r20,STATE_DIST+12(r7)     ;# dist3 bitmap
    lwz       r31,LEVEL_COMP+16(r3)     ;# comp4 bitmap
    lwz       r26,LEVEL_LIST+16(r3)     ;# list4 bitmap
    lwz       r21,STATE_DIST+16(r7)     ;# dist4 bitmap
    lwz       r16,LEVEL_CNT2(r3)        ;# level->cnt2
    rlwinm    r7,r17,DIST_SH+2,DIST_MB-2,DIST_ME-2 
    rlwinm    r0,r17,DIST_SH+3,DIST_MB-3,DIST_ME-3 
    add       r0,r7,r0                  ;# (dist0 >> DIST_SHIFT) * DIST_BITS
    lbzx      r8,r5,r0                  ;# choose(dist0 >> ttmDISTBITS, remainingDepth)
    b         L_comp_limit              
    nop                                 

    .align    4                         
L_comp_limit:                           
    ;# Compute the maximum position (limit)
    ;# r8 := choose(dist >> ttmDISTBITS, maxdepthm1 - depth)
    ;# "depth <= halfdepth" is equivalent to "&lev[depth] <= &lev[halfdepth]"
    cmpw      r3,r12                    ;# depth <= oStateHalfDepth ?
    not       r0,r27                    ;# c0neg = ~comp0
    cmpw      cr1,r9,r13                ;# depth <= oStateHalfDepth2 ?
    srwi      r0,r0,1                   ;# prepare c0neg for cntlzw
    ble       L_left_side               ;# depth <= oStateHalfDepth
    addi      r10,r10,1                 ;# ++nodes
    sub       r8,r11,r8                 ;# limit = oStateMax - choose[...]
    bgt       cr1,L_stay_1              ;# depth > oStateHalfDepth2

    ;# oStateHalfDepth < depth <= oStateHalfDepth2
    lwz       r7,LEVEL_CNT2(r12)        ;# levHalfDepth->cnt2
    sub       r7,r11,r7                 ;# temp = oStateMax - levHalfDepth->cnt2
    cmpw      r8,r7                     ;# limit < temp ?
    blt       L_stay_1                  

    subi      r8,r7,1                   ;# limit = temp - 1
    b         L_stay_1                  

L_left_side:                            
    ;# depth <= oStateHalfDepth
    lwz       r7,oState(r1)             ;# oState ptr
    cmpw      r10,r4                    ;# nodes >= max nodes ?
    lwz       r8,0(r6)                  ;# OGR[remainingDepth]
    lwz       r7,STATE_HALFLENGTH(r7)   ;# oState->half_length
    sub       r8,r11,r8                 ;# limit = oStateMax - OGR[...]
    cmpw      cr1,r8,r7                 ;# limit > half_length ?
    bge       L_exit_loop_CONT          ;# nodes >= max nodes : exit

    addi      r10,r10,1                 ;# ++nodes
    ble       cr1,L_stay_1              ;# limit <= oState->half_length

    mr        r8,r7                     ;# otherwise limit = half_length

L_stay_1:                               ;# newbit == 1
    ;# r0 = (~comp0) >> 1, so that cntlzw returns a value in the range [1;32]
    cntlzw    r0,r0                     ;# Find first bit set
    cmpwi     cr1,r27,-1                ;# Pre-check comp0 for case #32
    add       r16,r16,r0                ;# cnt2 += firstbit
    slwi      r7,r0,6                   ;# firstbit * 64 = Offset of each 'case' block
    cmpw      r16,r8                    ;# cnt2 > limit ?
    addis     r7,r7,(L_switch_cases-64)@ha
    bgt       L_up_level                ;# Go back to the preceding mark

    addi      r7,r7,(L_switch_cases-64)@l 
    rotlw     r27,r27,r0                ;# rotate comp0
    mtctr     r7                        
    rotlw     r28,r28,r0                ;# rotate comp1
    li        r7,1                      ;# newbit = 1
    rotlw     r29,r29,r0                ;# rotate comp2
    cmpw      r9,r15                    ;# depth == maxdepthm1 ?
    rotlw     r30,r30,r0                ;# rotate comp3
    bctr                                ;# Jump to "case firstbit:"

    .align    4                         
L_up_level:                             
    ;# POP_LEVEL(lev) : Restore the bitmaps then iterate
    subi      r3,r3,SIZEOF_LEVEL        ;# --lev
    lwz       r16,LEVEL_CNT2(r3)        ;# Load cnt2
    subi      r9,r9,1                   ;# --depth
    lwz       r22,LEVEL_LIST(r3)        ;# list0 = lev->list[0]
    addi      r6,r6,SIZEOF_OGR          ;# ++ogr
    lwz       r23,LEVEL_LIST+4(r3)      ;# list1 = lev->list[1]
    andc      r17,r17,r22               ;# dist0 &= ~list0
    lwz       r24,LEVEL_LIST+8(r3)      ;# list2 = lev->list[2]
    andc      r18,r18,r23               ;# dist1 &= ~list1
    lwz       r25,LEVEL_LIST+12(r3)     ;# list3 = lev->list[3]
    andc      r19,r19,r24               ;# dist2 &= ~list2
    lwz       r26,LEVEL_LIST+16(r3)     ;# list4 = lev->list[4]
    andc      r20,r20,r25               ;# dist3 &= ~list3
    lwz       r27,LEVEL_COMP(r3)        ;# comp0 = lev->comp[0]
    andc      r21,r21,r26               ;# dist4 &= ~list4
    lwz       r28,LEVEL_COMP+4(r3)      ;# comp1 = lev->comp[1]
    cmpw      r9,r14                    ;# depth <= startdepth ?
    lwz       r29,LEVEL_COMP+8(r3)      ;# comp2 = lev->comp[2]
    not       r0,r27                    ;# c0neg = ~comp0
    lwz       r30,LEVEL_COMP+12(r3)     ;# comp3 = lev->comp[3]
    srwi      r0,r0,1                   ;# Prepare c0neg
    lwz       r31,LEVEL_COMP+16(r3)     ;# comp4 = lev->comp[4]
    addi      r5,r5,1                   ;# ++choose
    lwz       r8,LEVEL_LIMIT(r3)        ;# Load limit
    ble-      L_exit_loop_OK            ;# depth <= startdepth : exit
    nop                                 

L_stay_0:                               ;# newbit == 0
    ;# r0 = (~comp0) >> 1, so that cntlzw returns a value in the range [1;32]
    cntlzw    r0,r0                     ;# Find first bit set
    cmpwi     cr1,r27,-1                ;# Pre-check comp0 for case #32
    add       r16,r16,r0                ;# cnt2 += firstbit
    slwi      r7,r0,6                   ;# firstbit * 64 = Offset of each 'case' block
    cmpw      r16,r8                    ;# cnt2 > limit ?
    addis     r7,r7,(L_switch_cases-64)@ha
    bgt       L_up_level                ;# Go back to the preceding mark

    addi      r7,r7,(L_switch_cases-64)@l
    rotlw     r27,r27,r0                ;# rotate comp0
    mtctr     r7                        
    rotlw     r28,r28,r0                ;# rotate comp1
    li        r7,0                      ;# newbit = 0
    rotlw     r29,r29,r0                ;# rotate comp2
    cmpw      r9,r15                    ;# depth == maxdepthm1 ?
    rotlw     r30,r30,r0                ;# rotate comp3
    bctr                                ;# Jump to "case firstbit:"

    .align    6                         ;# Align to a 64 bytes boundary
L_switch_cases:                           
    ;# Case 1:
    rotrwi    r26,r26,1                 ;# rotate list4
    rotrwi    r25,r25,1                 ;# rotate list3
    rotrwi    r24,r24,1                 ;# rotate list2
    rotrwi    r23,r23,1                 ;# rotate list1
    rotrwi    r22,r22,1                 ;# rotate list0
    rlwimi    r27,r28,0,31,31           ;# comp0 = comp0:comp1 << 1
    rlwimi    r28,r29,0,31,31           ;# comp1 = comp1:comp2 << 1
    rlwimi    r29,r30,0,31,31           ;# comp2 = comp2:comp3 << 1
    rlwimi    r30,r31,1,31,31           ;# comp3 = comp3:comp4 << 1
    slwi      r31,r31,1                 ;# shift comp4
    rlwimi    r26,r25,0,0,0             ;# list4 = list3:list4 >> 1
    rlwimi    r25,r24,0,0,0             ;# list3 = list2:list3 >> 1
    rlwimi    r24,r23,0,0,0             ;# list2 = list1:list2 >> 1
    rlwimi    r23,r22,0,0,0             ;# list1 = list0:list1 >> 1
    rlwimi    r22,r7,31,0,0             ;# list0 = newbit:list0 >> 1
    b         L_end_switch              

    ;# Case 2:
    rotrwi    r26,r26,2                 ;# rotate list4
    rotrwi    r25,r25,2                 ;# rotate list3
    rotrwi    r24,r24,2                 ;# rotate list2
    rotrwi    r23,r23,2                 ;# rotate list1
    rotrwi    r22,r22,2                 ;# rotate list0
    rlwimi    r27,r28,0,30,31           ;# comp0 = comp0:comp1 << 2
    rlwimi    r28,r29,0,30,31           ;# comp1 = comp1:comp2 << 2
    rlwimi    r29,r30,0,30,31           ;# comp2 = comp2:comp3 << 2
    rlwimi    r30,r31,2,30,31           ;# comp3 = comp3:comp4 << 2
    slwi      r31,r31,2                 ;# shift comp4
    rlwimi    r26,r25,0,0,1             ;# list4 = list3:list4 >> 2
    rlwimi    r25,r24,0,0,1             ;# list3 = list2:list3 >> 2
    rlwimi    r24,r23,0,0,1             ;# list2 = list1:list2 >> 2
    rlwimi    r23,r22,0,0,1             ;# list1 = list0:list1 >> 2
    rlwimi    r22,r7,30,0,1             ;# list0 = newbit:list0 >> 2
    b         L_end_switch              

    ;# Case 3:
    rotrwi    r26,r26,3                 ;# rotate list4
    rotrwi    r25,r25,3                 ;# rotate list3
    rotrwi    r24,r24,3                 ;# rotate list2
    rotrwi    r23,r23,3                 ;# rotate list1
    rotrwi    r22,r22,3                 ;# rotate list0
    rlwimi    r27,r28,0,29,31           ;# comp0 = comp0:comp1 << 3
    rlwimi    r28,r29,0,29,31           ;# comp1 = comp1:comp2 << 3
    rlwimi    r29,r30,0,29,31           ;# comp2 = comp2:comp3 << 3
    rlwimi    r30,r31,3,29,31           ;# comp3 = comp3:comp4 << 3
    slwi      r31,r31,3                 ;# shift comp4
    rlwimi    r26,r25,0,0,2             ;# list4 = list3:list4 >> 3
    rlwimi    r25,r24,0,0,2             ;# list3 = list2:list3 >> 3
    rlwimi    r24,r23,0,0,2             ;# list2 = list1:list2 >> 3
    rlwimi    r23,r22,0,0,2             ;# list1 = list0:list1 >> 3
    rlwimi    r22,r7,29,0,2             ;# list0 = newbit:list0 >> 3
    b         L_end_switch              

    ;# Case 4:
    rotrwi    r26,r26,4                 ;# rotate list4
    rotrwi    r25,r25,4                 ;# rotate list3
    rotrwi    r24,r24,4                 ;# rotate list2
    rotrwi    r23,r23,4                 ;# rotate list1
    rotrwi    r22,r22,4                 ;# rotate list0
    rlwimi    r27,r28,0,28,31           ;# comp0 = comp0:comp1 << 4
    rlwimi    r28,r29,0,28,31           ;# comp1 = comp1:comp2 << 4
    rlwimi    r29,r30,0,28,31           ;# comp2 = comp2:comp3 << 4
    rlwimi    r30,r31,4,28,31           ;# comp3 = comp3:comp4 << 4
    slwi      r31,r31,4                 ;# shift comp4
    rlwimi    r26,r25,0,0,3             ;# list4 = list3:list4 >> 4
    rlwimi    r25,r24,0,0,3             ;# list3 = list2:list3 >> 4
    rlwimi    r24,r23,0,0,3             ;# list2 = list1:list2 >> 4
    rlwimi    r23,r22,0,0,3             ;# list1 = list0:list1 >> 4
    rlwimi    r22,r7,28,0,3             ;# list0 = newbit:list0 >> 4
    b         L_end_switch              

    ;# Case 5:
    rotrwi    r26,r26,5                 ;# rotate list4
    rotrwi    r25,r25,5                 ;# rotate list3
    rotrwi    r24,r24,5                 ;# rotate list2
    rotrwi    r23,r23,5                 ;# rotate list1
    rotrwi    r22,r22,5                 ;# rotate list0
    rlwimi    r27,r28,0,27,31           ;# comp0 = comp0:comp1 << 5
    rlwimi    r28,r29,0,27,31           ;# comp1 = comp1:comp2 << 5
    rlwimi    r29,r30,0,27,31           ;# comp2 = comp2:comp3 << 5
    rlwimi    r30,r31,5,27,31           ;# comp3 = comp3:comp4 << 5
    slwi      r31,r31,5                 ;# shift comp4
    rlwimi    r26,r25,0,0,4             ;# list4 = list3:list4 >> 5
    rlwimi    r25,r24,0,0,4             ;# list3 = list2:list3 >> 5
    rlwimi    r24,r23,0,0,4             ;# list2 = list1:list2 >> 5
    rlwimi    r23,r22,0,0,4             ;# list1 = list0:list1 >> 5
    rlwimi    r22,r7,27,0,4             ;# list0 = newbit:list0 >> 5
    b         L_end_switch              

    ;# Case 6:
    rotrwi    r26,r26,6                 ;# rotate list4
    rotrwi    r25,r25,6                 ;# rotate list3
    rotrwi    r24,r24,6                 ;# rotate list2
    rotrwi    r23,r23,6                 ;# rotate list1
    rotrwi    r22,r22,6                 ;# rotate list0
    rlwimi    r27,r28,0,26,31           ;# comp0 = comp0:comp1 << 6
    rlwimi    r28,r29,0,26,31           ;# comp1 = comp1:comp2 << 6
    rlwimi    r29,r30,0,26,31           ;# comp2 = comp2:comp3 << 6
    rlwimi    r30,r31,6,26,31           ;# comp3 = comp3:comp4 << 6
    slwi      r31,r31,6                 ;# shift comp4
    rlwimi    r26,r25,0,0,5             ;# list4 = list3:list4 >> 6
    rlwimi    r25,r24,0,0,5             ;# list3 = list2:list3 >> 6
    rlwimi    r24,r23,0,0,5             ;# list2 = list1:list2 >> 6
    rlwimi    r23,r22,0,0,5             ;# list1 = list0:list1 >> 6
    rlwimi    r22,r7,26,0,5             ;# list0 = newbit:list0 >> 6
    b         L_end_switch              

    ;# Case 7:
    rotrwi    r26,r26,7                 ;# rotate list4
    rotrwi    r25,r25,7                 ;# rotate list3
    rotrwi    r24,r24,7                 ;# rotate list2
    rotrwi    r23,r23,7                 ;# rotate list1
    rotrwi    r22,r22,7                 ;# rotate list0
    rlwimi    r27,r28,0,25,31           ;# comp0 = comp0:comp1 << 7
    rlwimi    r28,r29,0,25,31           ;# comp1 = comp1:comp2 << 7
    rlwimi    r29,r30,0,25,31           ;# comp2 = comp2:comp3 << 7
    rlwimi    r30,r31,7,25,31           ;# comp3 = comp3:comp4 << 7
    slwi      r31,r31,7                 ;# shift comp4
    rlwimi    r26,r25,0,0,6             ;# list4 = list3:list4 >> 7
    rlwimi    r25,r24,0,0,6             ;# list3 = list2:list3 >> 7
    rlwimi    r24,r23,0,0,6             ;# list2 = list1:list2 >> 7
    rlwimi    r23,r22,0,0,6             ;# list1 = list0:list1 >> 7
    rlwimi    r22,r7,25,0,6             ;# list0 = newbit:list0 >> 7
    b         L_end_switch              

    ;# Case 8:
    rotrwi    r26,r26,8                 ;# rotate list4
    rotrwi    r25,r25,8                 ;# rotate list3
    rotrwi    r24,r24,8                 ;# rotate list2
    rotrwi    r23,r23,8                 ;# rotate list1
    rotrwi    r22,r22,8                 ;# rotate list0
    rlwimi    r27,r28,0,24,31           ;# comp0 = comp0:comp1 << 8
    rlwimi    r28,r29,0,24,31           ;# comp1 = comp1:comp2 << 8
    rlwimi    r29,r30,0,24,31           ;# comp2 = comp2:comp3 << 8
    rlwimi    r30,r31,8,24,31           ;# comp3 = comp3:comp4 << 8
    slwi      r31,r31,8                 ;# shift comp4
    rlwimi    r26,r25,0,0,7             ;# list4 = list3:list4 >> 8
    rlwimi    r25,r24,0,0,7             ;# list3 = list2:list3 >> 8
    rlwimi    r24,r23,0,0,7             ;# list2 = list1:list2 >> 8
    rlwimi    r23,r22,0,0,7             ;# list1 = list0:list1 >> 8
    rlwimi    r22,r7,24,0,7             ;# list0 = newbit:list0 >> 8
    b         L_end_switch              

    ;# Case 9:
    rotrwi    r26,r26,9                 ;# rotate list4
    rotrwi    r25,r25,9                 ;# rotate list3
    rotrwi    r24,r24,9                 ;# rotate list2
    rotrwi    r23,r23,9                 ;# rotate list1
    rotrwi    r22,r22,9                 ;# rotate list0
    rlwimi    r27,r28,0,23,31           ;# comp0 = comp0:comp1 << 9
    rlwimi    r28,r29,0,23,31           ;# comp1 = comp1:comp2 << 9
    rlwimi    r29,r30,0,23,31           ;# comp2 = comp2:comp3 << 9
    rlwimi    r30,r31,9,23,31           ;# comp3 = comp3:comp4 << 9
    slwi      r31,r31,9                 ;# shift comp4
    rlwimi    r26,r25,0,0,8             ;# list4 = list3:list4 >> 9
    rlwimi    r25,r24,0,0,8             ;# list3 = list2:list3 >> 9
    rlwimi    r24,r23,0,0,8             ;# list2 = list1:list2 >> 9
    rlwimi    r23,r22,0,0,8             ;# list1 = list0:list1 >> 9
    rlwimi    r22,r7,23,0,8             ;# list0 = newbit:list0 >> 9
    b         L_end_switch              

    ;# Case 10:
    rotrwi    r26,r26,10                ;# rotate list4
    rotrwi    r25,r25,10                ;# rotate list3
    rotrwi    r24,r24,10                ;# rotate list2
    rotrwi    r23,r23,10                ;# rotate list1
    rotrwi    r22,r22,10                ;# rotate list0
    rlwimi    r27,r28,0,22,31           ;# comp0 = comp0:comp1 << 10
    rlwimi    r28,r29,0,22,31           ;# comp1 = comp1:comp2 << 10
    rlwimi    r29,r30,0,22,31           ;# comp2 = comp2:comp3 << 10
    rlwimi    r30,r31,10,22,31          ;# comp3 = comp3:comp4 << 10
    slwi      r31,r31,10                ;# shift comp4
    rlwimi    r26,r25,0,0,9             ;# list4 = list3:list4 >> 10
    rlwimi    r25,r24,0,0,9             ;# list3 = list2:list3 >> 10
    rlwimi    r24,r23,0,0,9             ;# list2 = list1:list2 >> 10
    rlwimi    r23,r22,0,0,9             ;# list1 = list0:list1 >> 10
    rlwimi    r22,r7,22,0,9             ;# list0 = newbit:list0 >> 10
    b         L_end_switch              

    ;# Case 11:
    rotrwi    r26,r26,11                ;# rotate list4
    rotrwi    r25,r25,11                ;# rotate list3
    rotrwi    r24,r24,11                ;# rotate list2
    rotrwi    r23,r23,11                ;# rotate list1
    rotrwi    r22,r22,11                ;# rotate list0
    rlwimi    r27,r28,0,21,31           ;# comp0 = comp0:comp1 << 11
    rlwimi    r28,r29,0,21,31           ;# comp1 = comp1:comp2 << 11
    rlwimi    r29,r30,0,21,31           ;# comp2 = comp2:comp3 << 11
    rlwimi    r30,r31,11,21,31          ;# comp3 = comp3:comp4 << 11
    slwi      r31,r31,11                ;# shift comp4
    rlwimi    r26,r25,0,0,10            ;# list4 = list3:list4 >> 11
    rlwimi    r25,r24,0,0,10            ;# list3 = list2:list3 >> 11
    rlwimi    r24,r23,0,0,10            ;# list2 = list1:list2 >> 11
    rlwimi    r23,r22,0,0,10            ;# list1 = list0:list1 >> 11
    rlwimi    r22,r7,21,0,10            ;# list0 = newbit:list0 >> 11
    b         L_end_switch              

    ;# Case 12:
    rotrwi    r26,r26,12                ;# rotate list4
    rotrwi    r25,r25,12                ;# rotate list3
    rotrwi    r24,r24,12                ;# rotate list2
    rotrwi    r23,r23,12                ;# rotate list1
    rotrwi    r22,r22,12                ;# rotate list0
    rlwimi    r27,r28,0,20,31           ;# comp0 = comp0:comp1 << 12
    rlwimi    r28,r29,0,20,31           ;# comp1 = comp1:comp2 << 12
    rlwimi    r29,r30,0,20,31           ;# comp2 = comp2:comp3 << 12
    rlwimi    r30,r31,12,20,31          ;# comp3 = comp3:comp4 << 12
    slwi      r31,r31,12                ;# shift comp4
    rlwimi    r26,r25,0,0,11            ;# list4 = list3:list4 >> 12
    rlwimi    r25,r24,0,0,11            ;# list3 = list2:list3 >> 12
    rlwimi    r24,r23,0,0,11            ;# list2 = list1:list2 >> 12
    rlwimi    r23,r22,0,0,11            ;# list1 = list0:list1 >> 12
    rlwimi    r22,r7,20,0,11            ;# list0 = newbit:list0 >> 12
    b         L_end_switch              

    ;# Case 13:
    rotrwi    r26,r26,13                ;# rotate list4
    rotrwi    r25,r25,13                ;# rotate list3
    rotrwi    r24,r24,13                ;# rotate list2
    rotrwi    r23,r23,13                ;# rotate list1
    rotrwi    r22,r22,13                ;# rotate list0
    rlwimi    r27,r28,0,19,31           ;# comp0 = comp0:comp1 << 13
    rlwimi    r28,r29,0,19,31           ;# comp1 = comp1:comp2 << 13
    rlwimi    r29,r30,0,19,31           ;# comp2 = comp2:comp3 << 13
    rlwimi    r30,r31,13,19,31          ;# comp3 = comp3:comp4 << 13
    slwi      r31,r31,13                ;# shift comp4
    rlwimi    r26,r25,0,0,12            ;# list4 = list3:list4 >> 13
    rlwimi    r25,r24,0,0,12            ;# list3 = list2:list3 >> 13
    rlwimi    r24,r23,0,0,12            ;# list2 = list1:list2 >> 13
    rlwimi    r23,r22,0,0,12            ;# list1 = list0:list1 >> 13
    rlwimi    r22,r7,19,0,12            ;# list0 = newbit:list0 >> 13
    b         L_end_switch              

    ;# Case 14:
    rotrwi    r26,r26,14                ;# rotate list4
    rotrwi    r25,r25,14                ;# rotate list3
    rotrwi    r24,r24,14                ;# rotate list2
    rotrwi    r23,r23,14                ;# rotate list1
    rotrwi    r22,r22,14                ;# rotate list0
    rlwimi    r27,r28,0,18,31           ;# comp0 = comp0:comp1 << 14
    rlwimi    r28,r29,0,18,31           ;# comp1 = comp1:comp2 << 14
    rlwimi    r29,r30,0,18,31           ;# comp2 = comp2:comp3 << 14
    rlwimi    r30,r31,14,18,31          ;# comp3 = comp3:comp4 << 14
    slwi      r31,r31,14                ;# shift comp4
    rlwimi    r26,r25,0,0,13            ;# list4 = list3:list4 >> 14
    rlwimi    r25,r24,0,0,13            ;# list3 = list2:list3 >> 14
    rlwimi    r24,r23,0,0,13            ;# list2 = list1:list2 >> 14
    rlwimi    r23,r22,0,0,13            ;# list1 = list0:list1 >> 14
    rlwimi    r22,r7,18,0,13            ;# list0 = newbit:list0 >> 14
    b         L_end_switch              

    ;# Case 15:
    rotrwi    r26,r26,15                ;# rotate list4
    rotrwi    r25,r25,15                ;# rotate list3
    rotrwi    r24,r24,15                ;# rotate list2
    rotrwi    r23,r23,15                ;# rotate list1
    rotrwi    r22,r22,15                ;# rotate list0
    rlwimi    r27,r28,0,17,31           ;# comp0 = comp0:comp1 << 15
    rlwimi    r28,r29,0,17,31           ;# comp1 = comp1:comp2 << 15
    rlwimi    r29,r30,0,17,31           ;# comp2 = comp2:comp3 << 15
    rlwimi    r30,r31,15,17,31          ;# comp3 = comp3:comp4 << 15
    slwi      r31,r31,15                ;# shift comp4
    rlwimi    r26,r25,0,0,14            ;# list4 = list3:list4 >> 15
    rlwimi    r25,r24,0,0,14            ;# list3 = list2:list3 >> 15
    rlwimi    r24,r23,0,0,14            ;# list2 = list1:list2 >> 15
    rlwimi    r23,r22,0,0,14            ;# list1 = list0:list1 >> 15
    rlwimi    r22,r7,17,0,14            ;# list0 = newbit:list0 >> 15
    b         L_end_switch              

    ;# Case 16:
    rotrwi    r26,r26,16                ;# rotate list4
    rotrwi    r25,r25,16                ;# rotate list3
    rotrwi    r24,r24,16                ;# rotate list2
    rotrwi    r23,r23,16                ;# rotate list1
    rotrwi    r22,r22,16                ;# rotate list0
    rlwimi    r27,r28,0,16,31           ;# comp0 = comp0:comp1 << 16
    rlwimi    r28,r29,0,16,31           ;# comp1 = comp1:comp2 << 16
    rlwimi    r29,r30,0,16,31           ;# comp2 = comp2:comp3 << 16
    rlwimi    r30,r31,16,16,31          ;# comp3 = comp3:comp4 << 16
    slwi      r31,r31,16                ;# shift comp4
    rlwimi    r26,r25,0,0,15            ;# list4 = list3:list4 >> 16
    rlwimi    r25,r24,0,0,15            ;# list3 = list2:list3 >> 16
    rlwimi    r24,r23,0,0,15            ;# list2 = list1:list2 >> 16
    rlwimi    r23,r22,0,0,15            ;# list1 = list0:list1 >> 16
    rlwimi    r22,r7,16,0,15            ;# list0 = newbit:list0 >> 16
    b         L_end_switch              

    ;# Case 17:
    rotrwi    r26,r26,17                ;# rotate list4
    rotrwi    r25,r25,17                ;# rotate list3
    rotrwi    r24,r24,17                ;# rotate list2
    rotrwi    r23,r23,17                ;# rotate list1
    rotrwi    r22,r22,17                ;# rotate list0
    rlwimi    r27,r28,0,15,31           ;# comp0 = comp0:comp1 << 17
    rlwimi    r28,r29,0,15,31           ;# comp1 = comp1:comp2 << 17
    rlwimi    r29,r30,0,15,31           ;# comp2 = comp2:comp3 << 17
    rlwimi    r30,r31,17,15,31          ;# comp3 = comp3:comp4 << 17
    slwi      r31,r31,17                ;# shift comp4
    rlwimi    r26,r25,0,0,16            ;# list4 = list3:list4 >> 17
    rlwimi    r25,r24,0,0,16            ;# list3 = list2:list3 >> 17
    rlwimi    r24,r23,0,0,16            ;# list2 = list1:list2 >> 17
    rlwimi    r23,r22,0,0,16            ;# list1 = list0:list1 >> 17
    rlwimi    r22,r7,15,0,16            ;# list0 = newbit:list0 >> 17
    b         L_end_switch              

    ;# Case 18:
    rotrwi    r26,r26,18                ;# rotate list4
    rotrwi    r25,r25,18                ;# rotate list3
    rotrwi    r24,r24,18                ;# rotate list2
    rotrwi    r23,r23,18                ;# rotate list1
    rotrwi    r22,r22,18                ;# rotate list0
    rlwimi    r27,r28,0,14,31           ;# comp0 = comp0:comp1 << 18
    rlwimi    r28,r29,0,14,31           ;# comp1 = comp1:comp2 << 18
    rlwimi    r29,r30,0,14,31           ;# comp2 = comp2:comp3 << 18
    rlwimi    r30,r31,18,14,31          ;# comp3 = comp3:comp4 << 18
    slwi      r31,r31,18                ;# shift comp4
    rlwimi    r26,r25,0,0,17            ;# list4 = list3:list4 >> 18
    rlwimi    r25,r24,0,0,17            ;# list3 = list2:list3 >> 18
    rlwimi    r24,r23,0,0,17            ;# list2 = list1:list2 >> 18
    rlwimi    r23,r22,0,0,17            ;# list1 = list0:list1 >> 18
    rlwimi    r22,r7,14,0,17            ;# list0 = newbit:list0 >> 18
    b         L_end_switch              

    ;# Case 19:
    rotrwi    r26,r26,19                ;# rotate list4
    rotrwi    r25,r25,19                ;# rotate list3
    rotrwi    r24,r24,19                ;# rotate list2
    rotrwi    r23,r23,19                ;# rotate list1
    rotrwi    r22,r22,19                ;# rotate list0
    rlwimi    r27,r28,0,13,31           ;# comp0 = comp0:comp1 << 19
    rlwimi    r28,r29,0,13,31           ;# comp1 = comp1:comp2 << 19
    rlwimi    r29,r30,0,13,31           ;# comp2 = comp2:comp3 << 19
    rlwimi    r30,r31,19,13,31          ;# comp3 = comp3:comp4 << 19
    slwi      r31,r31,19                ;# shift comp4
    rlwimi    r26,r25,0,0,18            ;# list4 = list3:list4 >> 19
    rlwimi    r25,r24,0,0,18            ;# list3 = list2:list3 >> 19
    rlwimi    r24,r23,0,0,18            ;# list2 = list1:list2 >> 19
    rlwimi    r23,r22,0,0,18            ;# list1 = list0:list1 >> 19
    rlwimi    r22,r7,13,0,18            ;# list0 = newbit:list0 >> 19
    b         L_end_switch              

    ;# Case 20:
    rotrwi    r26,r26,20                ;# rotate list4
    rotrwi    r25,r25,20                ;# rotate list3
    rotrwi    r24,r24,20                ;# rotate list2
    rotrwi    r23,r23,20                ;# rotate list1
    rotrwi    r22,r22,20                ;# rotate list0
    rlwimi    r27,r28,0,12,31           ;# comp0 = comp0:comp1 << 20
    rlwimi    r28,r29,0,12,31           ;# comp1 = comp1:comp2 << 20
    rlwimi    r29,r30,0,12,31           ;# comp2 = comp2:comp3 << 20
    rlwimi    r30,r31,20,12,31          ;# comp3 = comp3:comp4 << 20
    slwi      r31,r31,20                ;# shift comp4
    rlwimi    r26,r25,0,0,19            ;# list4 = list3:list4 >> 20
    rlwimi    r25,r24,0,0,19            ;# list3 = list2:list3 >> 20
    rlwimi    r24,r23,0,0,19            ;# list2 = list1:list2 >> 20
    rlwimi    r23,r22,0,0,19            ;# list1 = list0:list1 >> 20
    rlwimi    r22,r7,12,0,19            ;# list0 = newbit:list0 >> 20
    b         L_end_switch              

    ;# Case 21:
    rotrwi    r26,r26,21                ;# rotate list4
    rotrwi    r25,r25,21                ;# rotate list3
    rotrwi    r24,r24,21                ;# rotate list2
    rotrwi    r23,r23,21                ;# rotate list1
    rotrwi    r22,r22,21                ;# rotate list0
    rlwimi    r27,r28,0,11,31           ;# comp0 = comp0:comp1 << 21
    rlwimi    r28,r29,0,11,31           ;# comp1 = comp1:comp2 << 21
    rlwimi    r29,r30,0,11,31           ;# comp2 = comp2:comp3 << 21
    rlwimi    r30,r31,21,11,31          ;# comp3 = comp3:comp4 << 21
    slwi      r31,r31,21                ;# shift comp4
    rlwimi    r26,r25,0,0,20            ;# list4 = list3:list4 >> 21
    rlwimi    r25,r24,0,0,20            ;# list3 = list2:list3 >> 21
    rlwimi    r24,r23,0,0,20            ;# list2 = list1:list2 >> 21
    rlwimi    r23,r22,0,0,20            ;# list1 = list0:list1 >> 21
    rlwimi    r22,r7,11,0,20            ;# list0 = newbit:list0 >> 21
    b         L_end_switch              

    ;# Case 22:
    rotrwi    r26,r26,22                ;# rotate list4
    rotrwi    r25,r25,22                ;# rotate list3
    rotrwi    r24,r24,22                ;# rotate list2
    rotrwi    r23,r23,22                ;# rotate list1
    rotrwi    r22,r22,22                ;# rotate list0
    rlwimi    r27,r28,0,10,31           ;# comp0 = comp0:comp1 << 22
    rlwimi    r28,r29,0,10,31           ;# comp1 = comp1:comp2 << 22
    rlwimi    r29,r30,0,10,31           ;# comp2 = comp2:comp3 << 22
    rlwimi    r30,r31,22,10,31          ;# comp3 = comp3:comp4 << 22
    slwi      r31,r31,22                ;# shift comp4
    rlwimi    r26,r25,0,0,21            ;# list4 = list3:list4 >> 22
    rlwimi    r25,r24,0,0,21            ;# list3 = list2:list3 >> 22
    rlwimi    r24,r23,0,0,21            ;# list2 = list1:list2 >> 22
    rlwimi    r23,r22,0,0,21            ;# list1 = list0:list1 >> 22
    rlwimi    r22,r7,10,0,21            ;# list0 = newbit:list0 >> 22
    b         L_end_switch              

    ;# Case 23:
    rotrwi    r26,r26,23                ;# rotate list4
    rotrwi    r25,r25,23                ;# rotate list3
    rotrwi    r24,r24,23                ;# rotate list2
    rotrwi    r23,r23,23                ;# rotate list1
    rotrwi    r22,r22,23                ;# rotate list0
    rlwimi    r27,r28,0,9,31            ;# comp0 = comp0:comp1 << 23
    rlwimi    r28,r29,0,9,31            ;# comp1 = comp1:comp2 << 23
    rlwimi    r29,r30,0,9,31            ;# comp2 = comp2:comp3 << 23
    rlwimi    r30,r31,23,9,31           ;# comp3 = comp3:comp4 << 23
    slwi      r31,r31,23                ;# shift comp4
    rlwimi    r26,r25,0,0,22            ;# list4 = list3:list4 >> 23
    rlwimi    r25,r24,0,0,22            ;# list3 = list2:list3 >> 23
    rlwimi    r24,r23,0,0,22            ;# list2 = list1:list2 >> 23
    rlwimi    r23,r22,0,0,22            ;# list1 = list0:list1 >> 23
    rlwimi    r22,r7,9,0,22             ;# list0 = newbit:list0 >> 23
    b         L_end_switch              

    ;# Case 24:
    rotrwi    r26,r26,24                ;# rotate list4
    rotrwi    r25,r25,24                ;# rotate list3
    rotrwi    r24,r24,24                ;# rotate list2
    rotrwi    r23,r23,24                ;# rotate list1
    rotrwi    r22,r22,24                ;# rotate list0
    rlwimi    r27,r28,0,8,31            ;# comp0 = comp0:comp1 << 24
    rlwimi    r28,r29,0,8,31            ;# comp1 = comp1:comp2 << 24
    rlwimi    r29,r30,0,8,31            ;# comp2 = comp2:comp3 << 24
    rlwimi    r30,r31,24,8,31           ;# comp3 = comp3:comp4 << 24
    slwi      r31,r31,24                ;# shift comp4
    rlwimi    r26,r25,0,0,23            ;# list4 = list3:list4 >> 24
    rlwimi    r25,r24,0,0,23            ;# list3 = list2:list3 >> 24
    rlwimi    r24,r23,0,0,23            ;# list2 = list1:list2 >> 24
    rlwimi    r23,r22,0,0,23            ;# list1 = list0:list1 >> 24
    rlwimi    r22,r7,8,0,23             ;# list0 = newbit:list0 >> 24
    b         L_end_switch              

    ;# Case 25:
    rotrwi    r26,r26,25                ;# rotate list4
    rotrwi    r25,r25,25                ;# rotate list3
    rotrwi    r24,r24,25                ;# rotate list2
    rotrwi    r23,r23,25                ;# rotate list1
    rotrwi    r22,r22,25                ;# rotate list0
    rlwimi    r27,r28,0,7,31            ;# comp0 = comp0:comp1 << 25
    rlwimi    r28,r29,0,7,31            ;# comp1 = comp1:comp2 << 25
    rlwimi    r29,r30,0,7,31            ;# comp2 = comp2:comp3 << 25
    rlwimi    r30,r31,25,7,31           ;# comp3 = comp3:comp4 << 25
    slwi      r31,r31,25                ;# shift comp4
    rlwimi    r26,r25,0,0,24            ;# list4 = list3:list4 >> 25
    rlwimi    r25,r24,0,0,24            ;# list3 = list2:list3 >> 25
    rlwimi    r24,r23,0,0,24            ;# list2 = list1:list2 >> 25
    rlwimi    r23,r22,0,0,24            ;# list1 = list0:list1 >> 25
    rlwimi    r22,r7,7,0,24             ;# list0 = newbit:list0 >> 25
    b         L_end_switch              

    ;# Case 26:
    rotrwi    r26,r26,26                ;# rotate list4
    rotrwi    r25,r25,26                ;# rotate list3
    rotrwi    r24,r24,26                ;# rotate list2
    rotrwi    r23,r23,26                ;# rotate list1
    rotrwi    r22,r22,26                ;# rotate list0
    rlwimi    r27,r28,0,6,31            ;# comp0 = comp0:comp1 << 26
    rlwimi    r28,r29,0,6,31            ;# comp1 = comp1:comp2 << 26
    rlwimi    r29,r30,0,6,31            ;# comp2 = comp2:comp3 << 26
    rlwimi    r30,r31,26,6,31           ;# comp3 = comp3:comp4 << 26
    slwi      r31,r31,26                ;# shift comp4
    rlwimi    r26,r25,0,0,25            ;# list4 = list3:list4 >> 26
    rlwimi    r25,r24,0,0,25            ;# list3 = list2:list3 >> 26
    rlwimi    r24,r23,0,0,25            ;# list2 = list1:list2 >> 26
    rlwimi    r23,r22,0,0,25            ;# list1 = list0:list1 >> 26
    rlwimi    r22,r7,6,0,25             ;# list0 = newbit:list0 >> 26
    b         L_end_switch              

    ;# Case 27:
    rotrwi    r26,r26,27                ;# rotate list4
    rotrwi    r25,r25,27                ;# rotate list3
    rotrwi    r24,r24,27                ;# rotate list2
    rotrwi    r23,r23,27                ;# rotate list1
    rotrwi    r22,r22,27                ;# rotate list0
    rlwimi    r27,r28,0,5,31            ;# comp0 = comp0:comp1 << 27
    rlwimi    r28,r29,0,5,31            ;# comp1 = comp1:comp2 << 27
    rlwimi    r29,r30,0,5,31            ;# comp2 = comp2:comp3 << 27
    rlwimi    r30,r31,27,5,31           ;# comp3 = comp3:comp4 << 27
    slwi      r31,r31,27                ;# shift comp4
    rlwimi    r26,r25,0,0,26            ;# list4 = list3:list4 >> 27
    rlwimi    r25,r24,0,0,26            ;# list3 = list2:list3 >> 27
    rlwimi    r24,r23,0,0,26            ;# list2 = list1:list2 >> 27
    rlwimi    r23,r22,0,0,26            ;# list1 = list0:list1 >> 27
    rlwimi    r22,r7,5,0,26             ;# list0 = newbit:list0 >> 27
    b         L_end_switch              

    ;# Case 28:
    rotrwi    r26,r26,28                ;# rotate list4
    rotrwi    r25,r25,28                ;# rotate list3
    rotrwi    r24,r24,28                ;# rotate list2
    rotrwi    r23,r23,28                ;# rotate list1
    rotrwi    r22,r22,28                ;# rotate list0
    rlwimi    r27,r28,0,4,31            ;# comp0 = comp0:comp1 << 28
    rlwimi    r28,r29,0,4,31            ;# comp1 = comp1:comp2 << 28
    rlwimi    r29,r30,0,4,31            ;# comp2 = comp2:comp3 << 28
    rlwimi    r30,r31,28,4,31           ;# comp3 = comp3:comp4 << 28
    slwi      r31,r31,28                ;# shift comp4
    rlwimi    r26,r25,0,0,27            ;# list4 = list3:list4 >> 28
    rlwimi    r25,r24,0,0,27            ;# list3 = list2:list3 >> 28
    rlwimi    r24,r23,0,0,27            ;# list2 = list1:list2 >> 28
    rlwimi    r23,r22,0,0,27            ;# list1 = list0:list1 >> 28
    rlwimi    r22,r7,4,0,27             ;# list0 = newbit:list0 >> 28
    b         L_end_switch              

    ;# Case 29:
    rotrwi    r26,r26,29                ;# rotate list4
    rotrwi    r25,r25,29                ;# rotate list3
    rotrwi    r24,r24,29                ;# rotate list2
    rotrwi    r23,r23,29                ;# rotate list1
    rotrwi    r22,r22,29                ;# rotate list0
    rlwimi    r27,r28,0,3,31            ;# comp0 = comp0:comp1 << 29
    rlwimi    r28,r29,0,3,31            ;# comp1 = comp1:comp2 << 29
    rlwimi    r29,r30,0,3,31            ;# comp2 = comp2:comp3 << 29
    rlwimi    r30,r31,29,3,31           ;# comp3 = comp3:comp4 << 29
    slwi      r31,r31,29                ;# shift comp4
    rlwimi    r26,r25,0,0,28            ;# list4 = list3:list4 >> 29
    rlwimi    r25,r24,0,0,28            ;# list3 = list2:list3 >> 29
    rlwimi    r24,r23,0,0,28            ;# list2 = list1:list2 >> 29
    rlwimi    r23,r22,0,0,28            ;# list1 = list0:list1 >> 29
    rlwimi    r22,r7,3,0,28             ;# list0 = newbit:list0 >> 29
    b         L_end_switch              

    ;# Case 30:
    rotrwi    r26,r26,30                ;# rotate list4
    rotrwi    r25,r25,30                ;# rotate list3
    rotrwi    r24,r24,30                ;# rotate list2
    rotrwi    r23,r23,30                ;# rotate list1
    rotrwi    r22,r22,30                ;# rotate list0
    rlwimi    r27,r28,0,2,31            ;# comp0 = comp0:comp1 << 30
    rlwimi    r28,r29,0,2,31            ;# comp1 = comp1:comp2 << 30
    rlwimi    r29,r30,0,2,31            ;# comp2 = comp2:comp3 << 30
    rlwimi    r30,r31,30,2,31           ;# comp3 = comp3:comp4 << 30
    slwi      r31,r31,30                ;# shift comp4
    rlwimi    r26,r25,0,0,29            ;# list4 = list3:list4 >> 30
    rlwimi    r25,r24,0,0,29            ;# list3 = list2:list3 >> 30
    rlwimi    r24,r23,0,0,29            ;# list2 = list1:list2 >> 30
    rlwimi    r23,r22,0,0,29            ;# list1 = list0:list1 >> 30
    rlwimi    r22,r7,2,0,29             ;# list0 = newbit:list0 >> 30
    b         L_end_switch              

    ;# Case 31:
    rotrwi    r26,r26,31                ;# rotate list4
    rotrwi    r25,r25,31                ;# rotate list3
    rotrwi    r24,r24,31                ;# rotate list2
    rotrwi    r23,r23,31                ;# rotate list1
    rotrwi    r22,r22,31                ;# rotate list0
    rlwimi    r27,r28,0,1,31            ;# comp0 = comp0:comp1 << 31
    rlwimi    r28,r29,0,1,31            ;# comp1 = comp1:comp2 << 31
    rlwimi    r29,r30,0,1,31            ;# comp2 = comp2:comp3 << 31
    rlwimi    r30,r31,31,1,31           ;# comp3 = comp3:comp4 << 31
    slwi      r31,r31,31                ;# shift comp4
    rlwimi    r26,r25,0,0,30            ;# list4 = list3:list4 >> 31
    rlwimi    r25,r24,0,0,30            ;# list3 = list2:list3 >> 31
    rlwimi    r24,r23,0,0,30            ;# list2 = list1:list2 >> 31
    rlwimi    r23,r22,0,0,30            ;# list1 = list0:list1 >> 31
    rlwimi    r22,r7,1,0,30             ;# list0 = newbit:list0 >> 31
    b         L_end_switch              

    ;# Case 32: COMP_LEFT_LIST_RIGHT_32(lev)
    ;# cr1 := (comp0 == -1)
    mr        r27,r28                   ;# comp0 = comp1
    not       r0,r28                    ;# c0neg = ~comp1
    mr        r28,r29                   ;# comp1 = comp2
    srwi      r0,r0,1                   ;# prepare c0neg for cntlzw
    mr        r29,r30                   ;# comp2 = comp3
    mr        r30,r31                   ;# comp3 = comp4
    li        r31,0                     ;# comp4 = 0
    mr        r26,r25                   ;# list4 = list3
    mr        r25,r24                   ;# list3 = list2
    mr        r24,r23                   ;# list2 = list1
    mr        r23,r22                   ;# list1 = list0
    mr        r22,r7                    ;# list0 = newbit
    beq       cr1,L_stay_0              ;# comp0 == -1 : rescan
    nop                                 

L_end_switch:                           
    ;# cr0 := (depth == maxdepthm1)
    stw       r16,LEVEL_CNT2(r3)        ;# store cnt2
    beq-      L_check_Golomb            ;# Last mark placed : check for Golombness

    ;# PUSH_LEVEL_UPDATE_STATE(lev)
    stw       r22,LEVEL_LIST+0(r3)      ;# store list0
    or        r17,r17,r22               ;# dist0 |= list0
    stw       r27,LEVEL_COMP+0(r3)      ;# store comp0
    or        r27,r27,r17               ;# comp0 |= dist0
    stw       r23,LEVEL_LIST+4(r3)      ;# store list1
    or        r18,r18,r23               ;# dist1 |= list1
    stw       r28,LEVEL_COMP+4(r3)      ;# store comp1
    or        r28,r28,r18               ;# comp1 |= dist1
    stw       r24,LEVEL_LIST+8(r3)      ;# store list2
    or        r19,r19,r24               ;# dist2 |= list2
    stw       r29,LEVEL_COMP+8(r3)      ;# store comp2
    or        r29,r29,r19               ;# comp2 |= dist2
    stw       r25,LEVEL_LIST+12(r3)     ;# store list3
    or        r20,r20,r25               ;# dist3 |= list3
    stw       r30,LEVEL_COMP+12(r3)     ;# store comp3
    or        r30,r30,r20               ;# comp3 |= dist3
    stw       r26,LEVEL_LIST+16(r3)     ;# store list4
    or        r21,r21,r26               ;# dist4 |= list4
    stw       r31,LEVEL_COMP+16(r3)     ;# store comp4
    or        r31,r31,r21               ;# comp4 |= dist4
    stw       r8,LEVEL_LIMIT(r3)        ;# store limit
    addi      r9,r9,1                   ;# ++depth
    rlwinm    r8,r17,14,18,29           ;# (dist0 >> DIST_SHIFT) * 4
    subi      r5,r5,1                   ;# --choose
    rlwinm    r0,r17,15,17,28           ;# (dist0 >> DIST_SHIFT) * 8
    subi      r6,r6,SIZEOF_OGR          ;# --ogr
    add       r8,r8,r0                  ;# (dist0 >> DIST_SHIFT) * DIST_BITS
    addi      r3,r3,SIZEOF_LEVEL        ;# ++lev
    lbzx      r8,r8,r5                  ;# choose(dist0 >> ttmDISTBITS, remainingDepth)
    b         L_comp_limit              

L_check_Golomb:                           
    ;# Last mark placed : verify the Golombness
    ;# Backup volatile registers
    stw       r3,aStorage+0(r1)         
    stw       r4,aStorage+4(r1)         
    stw       r5,aStorage+8(r1)         
    stw       r6,aStorage+12(r1)        
    stw       r8,aStorage+16(r1)        
    stw       r9,aStorage+20(r1)        
    stw       r10,aStorage+24(r1)       
    stw       r11,aStorage+28(r1)       
    stw       r12,aStorage+32(r1)       
    lwz       r3,oState(r1)             
    bl        found_one                 ;# retval = found_one(oState)
    cmpwi     r3,CORE_S_CONTINUE        ;# retval == CORE_S_CONTINUE ?
    not       r0,r27                    ;# c0neg = ~comp0
    mr        r7,r3                     ;# backup result code
    srwi      r0,r0,1                   ;# Prepare c0neg
    ;# Restore volatile registers
    lwz       r3,aStorage+0(r1)         
    lwz       r4,aStorage+4(r1)         
    lwz       r5,aStorage+8(r1)         
    lwz       r6,aStorage+12(r1)        
    lwz       r8,aStorage+16(r1)        
    lwz       r9,aStorage+20(r1)        
    lwz       r10,aStorage+24(r1)       
    lwz       r11,aStorage+28(r1)       
    lwz       r12,aStorage+32(r1)       
    beq       L_stay_0                  ;# Not Golomb : iterate

    mr        r0,r7                     ;# Restore result code
    b         L_save_state              

L_exit_loop_OK:                           
    li        r0,CORE_S_OK              
    b         L_save_state              

L_exit_loop_CONT:                           
    li        r0,CORE_S_CONTINUE        

L_save_state:                           
    lwz       r7,oState(r1)             
    lwz       r8,pNodes(r1)             
    stw       r27,LEVEL_COMP+0(r3)      ;# comp0 bitmap
    stw       r17,STATE_DIST+0(r7)      ;# dist0 bitmap
    stw       r22,LEVEL_LIST+0(r3)      ;# list0 bitmap
    stw       r28,LEVEL_COMP+4(r3)      ;# comp1 bitmap
    stw       r18,STATE_DIST+4(r7)      ;# dist1 bitmap
    stw       r23,LEVEL_LIST+4(r3)      ;# list1 bitmap
    stw       r29,LEVEL_COMP+8(r3)      ;# comp2 bitmap
    stw       r19,STATE_DIST+8(r7)      ;# dist2 bitmap
    stw       r24,LEVEL_LIST+8(r3)      ;# list2 bitmap
    stw       r30,LEVEL_COMP+12(r3)     ;# comp3 bitmap
    stw       r20,STATE_DIST+12(r7)     ;# dist3 bitmap
    stw       r25,LEVEL_LIST+12(r3)     ;# list3 bitmap
    stw       r31,LEVEL_COMP+16(r3)     ;# comp4 bitmap
    stw       r21,STATE_DIST+16(r7)     ;# dist4 bitmap
    stw       r26,LEVEL_LIST+16(r3)     ;# list4 bitmap
    stw       r16,LEVEL_CNT2(r3)        
    subi      r9,r9,1                   ;# --depth
    stw       r10,0(r8)                 ;# Store node count
    mr        r3,r0                     ;# retval
    stw       r9,STATE_DEPTH(r7)        

;#============================================================================
;# Epilog

    ;# Restore non-volatile registers
    lwz       r5,0(r1)                  ;# Obtains caller's stack pointer
    lmw       r13,-GPRSaveArea(r5)      ;# Restore GPRs
    lwz       r6,wSaveLR(r5)            ;# ...from caller's stack
    mtlr      r6                        
    mr        r1,r5                     
    blr                                 

