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
;# Platform/assembler specific implementation :
;# - External symbols ("cycle_ppc_hybrid" and "found_one") have an
;#   underscore prefix.
;# - Some simplified mnemonics are used.
;# - hi16() and lo16() directives might have to be replaced.
;# - Stack frame based upon Mac OS/AIX ABI.
;# - CTR, CR0, CR1, GPR0 and GPR3-GPR12 are volatile (not saved).
;#
;# $Id: OGR_PPC_hybrid.osx.s,v 1.1.2.1 2004/07/13 19:01:54 kakace Exp $
;#
;#============================================================================


    .text                               
    .align    5                         
    .globl    _cycle_ppc_hybrid         


;# Bitmaps dependencies (offsets)
.set          STATE_DISTV,        208   ;# dist vector
.set          STATE_DIST0,        200   ;# dist[0] scalar
.set          LEVEL_LISTV0,       0     ;# list[0] vector
.set          LEVEL_LISTV1,       16    ;# list[1] vector
.set          LEVEL_COMPV,        32    ;# comp vector
.set          LEVEL_LIST0,        48    ;# list[0] scalar
.set          LEVEL_COMP0,        52    ;# comp[0] scalar


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
;.set         v14,14                    
;.set         v15,15                    
;.set         v16,16                    
;.set         v17,17                    
;.set         v18,18                    
;.set         v19,19                    
;.set         v20,20                    
;.set         v21,21                    
;.set         v22,22                    
;.set         v23,23                    
;.set         v24,24                    
;.set         v25,25                    
;.set         v26,26                    
;.set         v27,27                    
;.set         VRsave,0x100              


;#============================================================================
;# Mac OS/AIX stack frame
.set          FIRST_NV_GPR, 13          ;# Save r13..r31
.set          GPRSaveArea, (32-FIRST_NV_GPR) * 4 
.set          FIRST_NV_VR, 20           ;# Save v20..
.set          VRSaveArea, (28 - FIRST_NV_VR) * 16 

.set          wSaveLR, 8                ;# Save LR
.set          aStorage, 28              ;# Private storage area
.set          wVRSave,-(GPRSaveArea+4)  
.set          aVectorArea, 48           ;# Vectors save area
.set          localTop, aVectorArea + 4 + VRSaveArea 
.set          FrameSize, (localTop + GPRSaveArea + 15) & (-16) 


;#============================================================================
;# int cycle_ppc_hybrid(void *state (r3)
;#                      int *pnodes (r4)
;#                      const unsigned char *choose (r5)
;#                      const int *OGR (r6)

_cycle_ppc_hybrid:                      
    ;# Allocate the new stack frame
    mflr      r9                        
    stw       r9,wSaveLR(r1)            ;# ...in caller's stack
    mr        r8,r1                     ;# Caller's stack pointer
    clrlwi    r11,r1,27                 ;# keep the low-order 4 bits
    subfic    r11,r11,-FrameSize        ;# Frame size, including padding
    stwux     r1,r1,r11                 

    ;# Save non-volatile registers
    stmw      r13,-GPRSaveArea(r8)      ;# Save GPRs

    ;# Save vector registers
    mfspr     r10,VRsave                
    oris      r11,r10,0xffff            ;# Use vector v0...v27
    stw       r10,wVRSave(r8)           
    ori       r11,r11,0xfff0            
    mtspr     VRsave,r11                

    li        r11,aVectorArea           
    stvx      v27,r11,r1                
    addi      r11,r11,16                
    stvx      v26,r11,r1                
    addi      r11,r11,16                
    stvx      v25,r11,r1                
    addi      r11,r11,16                
    stvx      v24,r11,r1                
    addi      r11,r11,16                
    stvx      v23,r11,r1                
    addi      r11,r11,16                
    stvx      v22,r11,r1                
    addi      r11,r11,16                
    stvx      v21,r11,r1                
    addi      r11,r11,16                
    stvx      v20,r11,r1                


;#============================================================================
;# Core engine initialization - Registers allocation :
;#   r3  := retval
;#   r4  := int *pnodes (const)
;#   r5  := &choose[3] (const)
;#   r6  := &OGR[0] (const)
;#   r13 := struct State *oState (const)
;#   r14 := &oState->Levels[depth]
;#   r15 := &oState->Levels[halfdepth]
;#   r16 := depth
;#   r17 := nodes
;#   r18 := oStateMax (const)
;#   r19 := oStateHalfDepth (const)
;#   r20 := oStateHaldDepth2 (const)
;#   r21 := cnt2
;#   r22 := nodes max (*pnodes)
;#   r23 := c0neg
;#   r24 := newbit
;#   r25 := limit
;#   r26 := startdepth (const)
;#   r27 := pointer to switch_cases-64
;#   r28 := maxdepthm1 (const)
;#   r29 := dist[0]
;#   r30 := list[0]
;#   r31 := comp[0]
;#   v1 -v7  := Shift constants (bytes)
;#   v11-v23 := Shift constants (words)
;#   v0  := ZEROS (0, 0, 0, 0)
;#   v20 := ZEROBIT (0, 0, 1, 0)
;#   v19 := ONES (0xff, 0xff, 0xff, 0xff)
;#   v24 := dist bitmap
;#   v25 := list[0] bitmap
;#   v26 := list[1] bitmap
;#   v27 := comp bitmap

    lwz       r16,STATE_DEPTH(r3)       ;# oState->depth
    mr        r13,r3                    ;# Copy oState
    lis       r27,hi16(L_switch_cases-64) 
    lwz       r19,STATE_HALFDEPTH(r3)   ;# oState->half_depth
    addi      r14,r3,STATE_LEVELS       ;# &oState->Levels[0]
    ori       r27,r27,lo16(L_switch_cases-64) 
    lwz       r28,STATE_MAXDEPTHM1(r3)  ;# oState->maxdepthm1
    li        r17,0                     ;# nodes = 0
    addi      r16,r16,1                 ;# ++depth
    lwz       r20,STATE_HALFDEPTH2(r13) ;# oState->half_depth2
    li        r3,CORE_S_CONTINUE        ;# retval
    slwi      r7,r19,LEVEL_SHIFT        
    lwz       r18,STATE_MAX(r13)        ;# oStateMax
    slwi      r8,r16,LEVEL_SHIFT        
    sub       r0,r28,r16                ;# remainingDepth = maxdepthm1 - depth
    lwz       r22,0(r4)                 ;# nodes max = *pnodes
    add       r15,r14,r7                ;# &oState->Levels[half_depth]
    slwi      r7,r0,OGR_SHIFT           ;# remainingDepth * sizeof(int)
    add       r14,r14,r8                ;# &oState->Levels[depth]
    lwz       r26,STATE_STARTDEPTH(r13) ;# oState->startdepth
    add       r5,r5,r0                  ;# &choose[remainingDepth]
    add       r6,r6,r7                  ;# &OGR[remainingDepth]

    ;# SETUP_TOP_STATE
    ;# - Initialize vector constants required to shift bitmaps
    ;# - Load current state
    lwz       r31,LEVEL_COMP0(r14)      ;# comp0 bitmap
    vspltisb  v1,1                      ;# VSHIFT_B1
    li        r9,STATE_DISTV            
    lwz       r29,STATE_DIST0(r13)      ;# dist0 bitmap
    vspltisb  v2,2                      ;# VSHIFT_B2
    li        r12,LEVEL_COMPV           
    lwz       r30,LEVEL_LIST0(r14)      ;# list0 bitmap
    vspltisb  v3,3                      ;# VSHIFT_B3
    lwz       r21,LEVEL_CNT2(r14)       ;# level->cnt2
    vspltisb  v4,4                      ;# VSHIFT_B4
    li        r11,LEVEL_LISTV1          
    lvx       v24,r9,r13                ;# dist vector
    vspltisb  v5,5                      ;# VSHIFT_B5
    lvx       v27,r12,r14               ;# comp vector
    vspltisb  v6,6                      ;# VSHIFT_B6
    lvx       v25,0,r14                 ;# listV0 vector
    vspltisb  v7,7                      ;# VSHIFT_B7
    lvx       v26,r11,r14               ;# listV1 vector
    vspltisw  v11,1                     ;# VSHIFT_L1
    vsubuwm   v0,v0,v0                  ;# ZEROS
    vspltisw  v18,8                     ;# VSHIFT_L8
    vnor      v19,v0,v0                 ;# ONES
    vspltisw  v12,2                     ;# VSHIFT_L2
    vadduwm   v21,v18,v18               ;# VSHIFT_L16
    vspltisw  v13,3                     ;# VSHIFT_L3
    vadduwm   v22,v21,v18               ;# VSHIFT_L24
    vmrglw    v20,v0,v11                ;# = (0, 1, 0, 1)
    vadduwm   v23,v21,v21               ;# VSHIFT_L32
    vmrglw    v20,v20,v0                ;# = (0, 0, 1, 0)
    vspltisw  v14,4                     ;# VSHIFT_L4
    rlwinm    r7,r29,DIST_SH+2,DIST_MB-2,DIST_ME-2 
    rlwinm    r0,r29,DIST_SH+3,DIST_MB-3,DIST_ME-3 
    vspltisw  v15,5                     ;# VSHIFT_L5
    vor       v25,v25,v20               ;# listV0 |= ZEROBIT
    add       r0,r7,r0                  ;# (dist0 >> DIST_SHIFT) * DIST_BITS
    vspltisw  v16,6                     ;# VSHIFT_L6
    lbzx      r25,r5,r0                 ;# choose(dist0 >> ttmDISTBITS, remainingDepth)
    vspltisw  v17,7                     ;# VSHIFT_L7
    b         L_comp_limit              
    nop                                 

    .align    4                         
L_comp_limit:                           
    ;# Compute the maximum position (limit)
    cmpw      r16,r19                   ;# depth <= oStateHalfDepth ?
    not       r23,r31                   ;# c0neg = ~comp0
    li        r24,1                     ;# newbit = 1
    cmpw      cr1,r16,r20               ;# depth <= oStateHalfDepth2 ?
    srwi      r23,r23,1                 ;# prepare c0neg for cntlzw
    ble       L_left_side               ;# depth <= oStateHalfDepth
    addi      r17,r17,1                 ;# ++nodes
    sub       r25,r18,r25               ;# limit = oStateMax - choose[...]
    bgt       cr1,L_stay                ;# depth > oStateHalfDepth2

    ;# oStateHalfDepth < depth <= oStateHalfDepth2
    lwz       r7,LEVEL_CNT2(r15)        ;# levHalfDepth->cnt2
    sub       r7,r18,r7                 ;# temp = oStateMax - levHalfDepth->cnt2
    cmpw      r25,r7                    ;# limit < temp ?
    blt       L_stay                    

    subi      r25,r7,1                  ;# limit = temp - 1
    b         L_stay                    

L_left_side:                            
    ;# depth <= oStateHalfDepth
    lwz       r7,0(r6)                  ;# OGR[remainingDepth]
    cmpw      r17,r22                   ;# nodes >= *pnodes ?
    lwz       r8,STATE_HALFLENGTH(r13)  ;# oState->half_length
    sub       r25,r18,r7                ;# limit = oStateMax - OGR[...]
    cmpw      cr1,r25,r8                ;# limit > half_length ?
    bge       L_exit_loop               ;# nodes >= pnodes : exit

    addi      r17,r17,1                 ;# ++nodes
    ble       cr1,L_stay                ;# limit <= oState->half_length
    mr        r25,r8                    ;# otherwise limit = half_length

L_stay:                                 
    ;# r0 = (~comp0) >> 1, so that cntlzw returns a value in the range [1;32]
    cntlzw    r0,r23                    ;# Find first bit set
    cmpwi     cr1,r31,-1                ;# Pre-check comp0 for case #32
    slwi      r7,r0,6                   ;# firstbit * 64 = Offset of each 'case' block
    add       r21,r21,r0                ;# cnt2 += firstbit
    slw       r31,r31,r0                ;# comp0 <<= firstbit
    add       r7,r7,r27                 ;# jump address
    cmpw      r21,r25                   ;# cnt2 > limit ?
    mtctr     r7                        
    lwz       r8,LEVEL_COMPV(r14)       ;# comp1 = lev->compV.u[0]
    srw       r30,r30,r0                ;# list0 >>= firstbit
    bgt       L_up_level                ;# Go back to the preceding mark
    bctr                                ;# Jump to 'case firstbit:'

    .align    6                         ;# Align to a 64 bytes boundary
L_switch_cases:                           
    ;# Case 1:
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v11                ;# maskV = vector (0xFFFFFFFE)
    rlwimi    r31,r8,1,31,31            ;# comp0 = (comp0:comp1) << 1
    vsl       v27,v27,v1                ;# compV <<= 1
    vsubuwm   v10,v0,v11                ;# shiftV = vector (-1)
    rlwimi    r30,r24,31,0,0            ;# list0 |= newbit << 31
    vsr       v25,v25,v1                ;# listV0 >>= 1
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    li        r24,0                     ;# newbit = 0
    vrlw      v26,v26,v10               ;# rotate listV1 (== right shift)
    b         L_end_switch              
    .align    6                         ;# Align to a 64 bytes boundary

    ;# Case 2:
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v12                ;# maskV = vector (0xFFFFFFFC)
    rlwimi    r31,r8,2,30,31            ;# comp0 = (comp0:comp1) << 2
    vsl       v27,v27,v2                ;# compV <<= 2
    vsubuwm   v10,v0,v12                ;# shiftV = vector (-2)
    rlwimi    r30,r24,30,0,1            ;# list0 |= newbit << 30
    vsr       v25,v25,v2                ;# listV0 >>= 2
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    li        r24,0                     ;# newbit = 0
    vrlw      v26,v26,v10               ;# rotate listV1 (== right shift)
    b         L_end_switch              
    .align    6                         ;# Align to a 64 bytes boundary

    ;# Case 3:
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v13                ;# maskV = vector (0xFFFFFFF8)
    rlwimi    r31,r8,3,29,31            ;# comp0 = (comp0:comp1) << 3
    vsl       v27,v27,v3                ;# compV <<= 3
    vsubuwm   v10,v0,v13                ;# shiftV = vector (-3)
    rlwimi    r30,r24,29,0,2            ;# list0 |= newbit << 29
    vsr       v25,v25,v3                ;# listV0 >>= 3
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    li        r24,0                     ;# newbit = 0
    vrlw      v26,v26,v10               ;# rotate listV1 (== right shift)
    b         L_end_switch              
    .align    6                         ;# Align to a 64 bytes boundary

    ;# Case 4:
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v14                ;# maskV = vector (0xFFFFFFF0)
    rlwimi    r31,r8,4,28,31            ;# comp0 = (comp0:comp1) << 4
    vsl       v27,v27,v4                ;# compV <<= 4
    vsubuwm   v10,v0,v14                ;# shiftV = vector (-4)
    rlwimi    r30,r24,28,0,3            ;# list0 |= newbit << 28
    vsr       v25,v25,v4                ;# listV0 >>= 4
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    li        r24,0                     ;# newbit = 0
    vrlw      v26,v26,v10               ;# rotate listV1 (== right shift)
    b         L_end_switch              
    .align    6                         ;# Align to a 64 bytes boundary

    ;# Case 5:
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v15                ;# maskV = vector (0xFFFFFFE0)
    rlwimi    r31,r8,5,27,31            ;# comp0 = (comp0:comp1) << 5
    vsl       v27,v27,v5                ;# compV <<= 5
    vsubuwm   v10,v0,v15                ;# shiftV = vector (-5)
    rlwimi    r30,r24,27,0,4            ;# list0 |= newbit << 27
    vsr       v25,v25,v5                ;# listV0 >>= 5
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    li        r24,0                     ;# newbit = 0
    vrlw      v26,v26,v10               ;# rotate listV1 (== right shift)
    b         L_end_switch              
    .align    6                         ;# Align to a 64 bytes boundary

    ;# Case 6:
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v16                ;# maskV = vector (0xFFFFFFC0)
    rlwimi    r31,r8,6,26,31            ;# comp0 = (comp0:comp1) << 6
    vsl       v27,v27,v6                ;# compV <<= 6
    vsubuwm   v10,v0,v16                ;# shiftV = vector (-6)
    rlwimi    r30,r24,26,0,5            ;# list0 |= newbit << 26
    vsr       v25,v25,v6                ;# listV0 >>= 6
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    li        r24,0                     ;# newbit = 0
    vrlw      v26,v26,v10               ;# rotate listV1 (== right shift)
    b         L_end_switch              
    .align    6                         ;# Align to a 64 bytes boundary

    ;# Case 7:
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v17                ;# maskV = vector (0xFFFFFF80)
    rlwimi    r31,r8,7,25,31            ;# comp0 = (comp0:comp1) << 7
    vsl       v27,v27,v7                ;# compV <<= 7
    vsubuwm   v10,v0,v17                ;# shiftV = vector (-7)
    rlwimi    r30,r24,25,0,6            ;# list0 |= newbit << 25
    vsr       v25,v25,v7                ;# listV0 >>= 7
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    li        r24,0                     ;# newbit = 0
    vrlw      v26,v26,v10               ;# rotate listV1 (== right shift)
    b         L_end_switch              
    .align    6                         ;# Align to a 64 bytes boundary

    ;# Case 8:
    vslo      v27,v27,v18               ;# compV <<= 8
    rlwimi    r31,r8,8,24,31            ;# comp0 = (comp0:comp1) << 8
    rlwimi    r30,r24,24,0,7            ;# list0 |= newbit << 24
    vsldoi    v26,v25,v26,15            ;# listV1 = (listV0:listV1) >> 8
    li        r24,0                     ;# newbit = 0
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsro      v25,v25,v18               ;# listV0 >>= 8
    b         L_end_switch              
    .align    6                         ;# Align to a 64 bytes boundary

    ;# Case 9:
    vslo      v27,v27,v18               ;# compV <<= 8
    vadduwm   v10,v18,v11               ;# shiftV = vector (9)
    rlwimi    r31,r8,9,23,31            ;# comp0 = (comp0:comp1) << 9
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFFFFE00)
    rlwimi    r30,r24,23,0,8            ;# list0 |= newbit << 23
    vsl       v27,v27,v1                ;# compV <<= 1
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-9)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v18               ;# listV0 >>= 8
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v1                ;# listV0 >>= 1
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 10:
    vslo      v27,v27,v18               ;# compV <<= 8
    vadduwm   v10,v18,v12               ;# shiftV = vector (10)
    rlwimi    r31,r8,10,22,31           ;# comp0 = (comp0:comp1) << 10
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFFFFC00)
    rlwimi    r30,r24,22,0,9            ;# list0 |= newbit << 22
    vsl       v27,v27,v2                ;# compV <<= 2
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-10)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v18               ;# listV0 >>= 8
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v2                ;# listV0 >>= 2
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 11:
    vslo      v27,v27,v18               ;# compV <<= 8
    vadduwm   v10,v18,v13               ;# shiftV = vector (11)
    rlwimi    r31,r8,11,21,31           ;# comp0 = (comp0:comp1) << 11
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFFFF800)
    rlwimi    r30,r24,21,0,10           ;# list0 |= newbit << 21
    vsl       v27,v27,v3                ;# compV <<= 3
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-11)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v18               ;# listV0 >>= 8
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v3                ;# listV0 >>= 3
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 12:
    vslo      v27,v27,v18               ;# compV <<= 8
    vadduwm   v10,v18,v14               ;# shiftV = vector (12)
    rlwimi    r31,r8,12,20,31           ;# comp0 = (comp0:comp1) << 12
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFFFF000)
    rlwimi    r30,r24,20,0,11           ;# list0 |= newbit << 20
    vsl       v27,v27,v4                ;# compV <<= 4
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-12)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v18               ;# listV0 >>= 8
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v4                ;# listV0 >>= 4
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 13:
    vslo      v27,v27,v18               ;# compV <<= 8
    vadduwm   v10,v18,v15               ;# shiftV = vector (13)
    rlwimi    r31,r8,13,19,31           ;# comp0 = (comp0:comp1) << 13
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFFFE000)
    rlwimi    r30,r24,19,0,12           ;# list0 |= newbit << 19
    vsl       v27,v27,v5                ;# compV <<= 5
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-13)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v18               ;# listV0 >>= 8
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v5                ;# listV0 >>= 5
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 14:
    vslo      v27,v27,v18               ;# compV <<= 8
    vadduwm   v10,v18,v16               ;# shiftV = vector (14)
    rlwimi    r31,r8,14,18,31           ;# comp0 = (comp0:comp1) << 14
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFFFC000)
    rlwimi    r30,r24,18,0,13           ;# list0 |= newbit << 18
    vsl       v27,v27,v6                ;# compV <<= 6
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-14)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v18               ;# listV0 >>= 8
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v6                ;# listV0 >>= 6
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 15:
    vslo      v27,v27,v18               ;# compV <<= 8
    vadduwm   v10,v18,v17               ;# shiftV = vector (15)
    rlwimi    r31,r8,15,17,31           ;# comp0 = (comp0:comp1) << 15
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFFF8000)
    rlwimi    r30,r24,17,0,14           ;# list0 |= newbit << 17
    vsl       v27,v27,v7                ;# compV <<= 7
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-15)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v18               ;# listV0 >>= 8
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v7                ;# listV0 >>= 7
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 16:
    vslo      v27,v27,v21               ;# compV <<= 16
    rlwimi    r31,r8,16,16,31           ;# comp0 = (comp0:comp1) << 16
    rlwimi    r30,r24,16,0,15           ;# list0 |= newbit << 16
    vsldoi    v26,v25,v26,14            ;# listV1 = (listV0:listV1) >> 16
    li        r24,0                     ;# newbit = 0
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsro      v25,v25,v21               ;# listV0 >>= 16
    b         L_end_switch              
    .align    6                         ;# Align to a 64 bytes boundary

    ;# Case 17:
    vslo      v27,v27,v21               ;# compV <<= 16
    vadduwm   v10,v21,v11               ;# shiftV = vector (17)
    rlwimi    r31,r8,17,15,31           ;# comp0 = (comp0:comp1) << 17
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFFE0000)
    rlwimi    r30,r24,15,0,16           ;# list0 |= newbit << 15
    vsl       v27,v27,v1                ;# compV <<= 1
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-17)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v21               ;# listV0 >>= 16
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v1                ;# listV0 >>= 1
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 18:
    vslo      v27,v27,v21               ;# compV <<= 16
    vadduwm   v10,v21,v12               ;# shiftV = vector (18)
    rlwimi    r31,r8,18,14,31           ;# comp0 = (comp0:comp1) << 18
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFFC0000)
    rlwimi    r30,r24,14,0,17           ;# list0 |= newbit << 14
    vsl       v27,v27,v2                ;# compV <<= 2
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-18)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v21               ;# listV0 >>= 16
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v2                ;# listV0 >>= 2
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 19:
    vslo      v27,v27,v21               ;# compV <<= 16
    vadduwm   v10,v21,v13               ;# shiftV = vector (19)
    rlwimi    r31,r8,19,13,31           ;# comp0 = (comp0:comp1) << 19
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFF80000)
    rlwimi    r30,r24,13,0,18           ;# list0 |= newbit << 13
    vsl       v27,v27,v3                ;# compV <<= 3
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-19)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v21               ;# listV0 >>= 16
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v3                ;# listV0 >>= 3
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 20:
    vslo      v27,v27,v21               ;# compV <<= 16
    vadduwm   v10,v21,v14               ;# shiftV = vector (20)
    rlwimi    r31,r8,20,12,31           ;# comp0 = (comp0:comp1) << 20
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFF00000)
    rlwimi    r30,r24,12,0,19           ;# list0 |= newbit << 12
    vsl       v27,v27,v4                ;# compV <<= 4
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-20)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v21               ;# listV0 >>= 16
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v4                ;# listV0 >>= 4
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 21:
    vslo      v27,v27,v21               ;# compV <<= 16
    vadduwm   v10,v21,v15               ;# shiftV = vector (21)
    rlwimi    r31,r8,21,11,31           ;# comp0 = (comp0:comp1) << 21
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFE00000)
    rlwimi    r30,r24,11,0,20           ;# list0 |= newbit << 11
    vsl       v27,v27,v5                ;# compV <<= 5
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-21)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v21               ;# listV0 >>= 16
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v5                ;# listV0 >>= 5
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 22:
    vslo      v27,v27,v21               ;# compV <<= 16
    vadduwm   v10,v21,v16               ;# shiftV = vector (22)
    rlwimi    r31,r8,22,10,31           ;# comp0 = (comp0:comp1) << 22
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFFC00000)
    rlwimi    r30,r24,10,0,21           ;# list0 |= newbit << 10
    vsl       v27,v27,v6                ;# compV <<= 6
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-22)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v21               ;# listV0 >>= 16
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v6                ;# listV0 >>= 6
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 23:
    vslo      v27,v27,v21               ;# compV <<= 16
    vadduwm   v10,v21,v17               ;# shiftV = vector (23)
    rlwimi    r31,r8,23,9,31            ;# comp0 = (comp0:comp1) << 23
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFF800000)
    rlwimi    r30,r24,9,0,22            ;# list0 |= newbit << 9
    vsl       v27,v27,v7                ;# compV <<= 7
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-23)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v21               ;# listV0 >>= 16
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v7                ;# listV0 >>= 7
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 24:
    vslo      v27,v27,v22               ;# compV <<= 24
    rlwimi    r31,r8,24,8,31            ;# comp0 = (comp0:comp1) << 24
    rlwimi    r30,r24,8,0,23            ;# list0 |= newbit << 8
    vsldoi    v26,v25,v26,13            ;# listV1 = (listV0:listV1) >> 24
    li        r24,0                     ;# newbit = 0
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsro      v25,v25,v22               ;# listV0 >>= 24
    b         L_end_switch              
    .align    6                         ;# Align to a 64 bytes boundary

    ;# Case 25:
    vslo      v27,v27,v22               ;# compV <<= 24
    vadduwm   v10,v22,v11               ;# shiftV = vector (25)
    rlwimi    r31,r8,25,7,31            ;# comp0 = (comp0:comp1) << 25
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFE000000)
    rlwimi    r30,r24,7,0,24            ;# list0 |= newbit << 7
    vsl       v27,v27,v1                ;# compV <<= 1
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-25)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v22               ;# listV0 >>= 24
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v1                ;# listV0 >>= 1
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 26:
    vslo      v27,v27,v22               ;# compV <<= 24
    vadduwm   v10,v22,v12               ;# shiftV = vector (26)
    rlwimi    r31,r8,26,6,31            ;# comp0 = (comp0:comp1) << 26
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xFC000000)
    rlwimi    r30,r24,6,0,25            ;# list0 |= newbit << 6
    vsl       v27,v27,v2                ;# compV <<= 2
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-26)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v22               ;# listV0 >>= 24
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v2                ;# listV0 >>= 2
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 27:
    vslo      v27,v27,v22               ;# compV <<= 24
    vadduwm   v10,v22,v13               ;# shiftV = vector (27)
    rlwimi    r31,r8,27,5,31            ;# comp0 = (comp0:comp1) << 27
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xF8000000)
    rlwimi    r30,r24,5,0,26            ;# list0 |= newbit << 5
    vsl       v27,v27,v3                ;# compV <<= 3
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-27)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v22               ;# listV0 >>= 24
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v3                ;# listV0 >>= 3
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 28:
    vslo      v27,v27,v22               ;# compV <<= 24
    vadduwm   v10,v22,v14               ;# shiftV = vector (28)
    rlwimi    r31,r8,28,4,31            ;# comp0 = (comp0:comp1) << 28
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xF0000000)
    rlwimi    r30,r24,4,0,27            ;# list0 |= newbit << 4
    vsl       v27,v27,v4                ;# compV <<= 4
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-28)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v22               ;# listV0 >>= 24
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v4                ;# listV0 >>= 4
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 29:
    vslo      v27,v27,v22               ;# compV <<= 24
    vadduwm   v10,v22,v15               ;# shiftV = vector (29)
    rlwimi    r31,r8,29,3,31            ;# comp0 = (comp0:comp1) << 29
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xE0000000)
    rlwimi    r30,r24,3,0,28            ;# list0 |= newbit << 3
    vsl       v27,v27,v5                ;# compV <<= 5
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-29)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v22               ;# listV0 >>= 24
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v5                ;# listV0 >>= 5
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 30:
    vslo      v27,v27,v22               ;# compV <<= 24
    vadduwm   v10,v22,v16               ;# shiftV = vector (30)
    rlwimi    r31,r8,30,2,31            ;# comp0 = (comp0:comp1) << 30
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0xC0000000)
    rlwimi    r30,r24,2,0,29            ;# list0 |= newbit << 2
    vsl       v27,v27,v6                ;# compV <<= 6
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-30)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v22               ;# listV0 >>= 24
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v6                ;# listV0 >>= 6
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 31:
    vslo      v27,v27,v22               ;# compV <<= 24
    vadduwm   v10,v22,v17               ;# shiftV = vector (31)
    rlwimi    r31,r8,31,1,31            ;# comp0 = (comp0:comp1) << 31
    vsldoi    v8,v25,v26,12             ;# tempV = (listV0:listV1) >> 32
    vslw      v9,v19,v10                ;# maskV = vector (0x80000000)
    rlwimi    r30,r24,1,0,30            ;# list0 |= newbit << 1
    vsl       v27,v27,v7                ;# compV <<= 7
    vsubuwm   v10,v0,v10                ;# shiftV = vector (-31)
    li        r24,0                     ;# newbit = 0
    vsro      v25,v25,v22               ;# listV0 >>= 24
    vsel      v26,v8,v26,v9             ;# listV1 = select(tempV, listV1, maskV)
    stvx      v27,r12,r14               ;# lev->compV.v = compV
    vsr       v25,v25,v7                ;# listV0 >>= 7
    vrlw      v26,v26,v10               ;# rotate listV1 (== shift right)
    b         L_end_switch              
    nop                                 ;# 16 instructions (64 bytes)

    ;# Case 32: COMP_LEFT_LIST_RIGHT_32(lev)
    ;# cr1 := (comp0 == -1)
    mr        r31,r8                    ;# comp0 = comp1
    vslo      v27,v27,v23               ;# compV <<= 32
    not       r23,r8                    ;# c0neg = ~comp1
    vsldoi    v26,v25,v26,12            ;# listV1 = (listV0:listV1) >> 32
    mr        r30,r24                   ;# list0 = newbit
    stvx      v27,r12,r14               ;# store compV
    srwi      r23,r23,1                 ;# prepare c0neg for cntlzw
    vsro      v25,v25,v23               ;# listV0 >>= 32
    li        r24,0                     ;# newbit = 0
    beq       cr1,L_stay                ;# comp0 == -1 : rescan
    nop                                 
    nop                                 

L_end_switch:                           
    cmpw      r16,r28                   ;# depth == maxdepthm1 ?
    addi      r7,r12,SIZEOF_LEVEL       
    stw       r21,LEVEL_CNT2(r14)       ;# store cnt2
    beq-      L_check_Golomb            ;# Last mark placed : check for Golombness

    ;# PUSH_LEVEL_UPDATE_STATE(lev)
    vor       v24,v24,v26               ;# distV |= listV1
    stvx      v26,r11,r14               ;# store listV1
    vor       v27,v27,v24               ;# compV |= distV
    stvx      v25,0,r14                 ;# store listV0
    vor       v25,v25,v20               ;# listV0 |= ZEROBIT
    or        r29,r29,r30               ;# dist0 |= list0
    stvx      v27,r7,r14                ;# (lev+1)->compV.v = compV
    addi      r16,r16,1                 ;# ++depth
    subi      r5,r5,1                   ;# --choose
    stw       r30,LEVEL_LIST0(r14)      ;# store list0
    rlwinm    r7,r29,14,18,29           ;# (dist0 >> DIST_SHIFT) * 4
    rlwinm    r0,r29,15,17,28           ;# (dist0 >> DIST_SHIFT) * 8
    stw       r31,LEVEL_COMP0(r14)      ;# store comp0
    or        r31,r31,r29               ;# comp0 |= dist0
    add       r0,r7,r0                  ;# (dist0 >> DIST_SHIFT) * DIST_BITS
    stw       r25,LEVEL_LIMIT(r14)      ;# store limit
    subi      r6,r6,SIZEOF_OGR          ;# --ogr
    addi      r14,r14,SIZEOF_LEVEL      ;# ++lev
    lbzx      r25,r5,r0                 ;# choose(dist0 >> ttmDISTBITS, remainingDepth)
    b         L_comp_limit              

L_check_Golomb:                           
    ;# Last mark placed : verify the Golombness
    not       r23,r31                   ;# c0neg = ~comp0
    stw       r4,aStorage(r1)           ;# Backup registers
    mr        r3,r13                    
    stw       r5,aStorage+4(r1)         
    srwi      r23,r23,1                 ;# Prepare c0neg
    stw       r6,aStorage+8(r1)         
    bl        _found_one                ;# retval = found_one(oState)
    cmpwi     r3,CORE_S_CONTINUE        ;# retval == CORE_S_CONTINUE ?
    ;# Restore unsaved registers
    vspltisw  v11,1                     ;# VSHIFT_L1 = vector (1)
    lwz       r6,aStorage+8(r1)         ;# ogr
    li        r9,STATE_DISTV            
    vspltisb  v1,1                      ;# VSHIFT_B1 = vector (1)
    lwz       r5,aStorage+4(r1)         ;# choose
    li        r12,LEVEL_COMPV           
    vspltisb  v4,4                      ;# VSHIFT_B4 = vector (4)
    lwz       r4,aStorage(r1)           ;# pnodes
    vspltisw  v12,2                     ;# VSHIFT_L2 = vector (2)
    vadduwm   v2,v1,v1                  ;# VSHIFT_B2 = vector (2)
    li        r11,LEVEL_LISTV1          
    vspltisw  v13,3                     ;# VSHIFT_L3 = vector (3)
    vadduwm   v5,v4,v1                  ;# VSHIFT_B5 = vector (5)
    vspltisw  v14,4                     ;# VSHIFT_L4 = vector (4)
    vadduwm   v3,v2,v1                  ;# VSHIFT_B3 = vector (3)
    vspltisw  v15,5                     ;# VSHIFT_L5 = vector (5)
    vadduwm   v6,v4,v2                  ;# VSHIFT_B6 = vector (6)
    vspltisw  v16,6                     ;# VSHIFT_L6 = vector (6)
    vadduwm   v7,v4,v3                  ;# VSHIFT_B7 = vector (7)
    vspltisw  v17,7                     ;# VSHIFT_L7 = vector (7)
    vsubuwm   v0,v0,v0                  ;# ZEROS = vector (0)
    vspltisw  v18,8                     ;# VSHIFT_L8 = vector (8)
    vnor      v19,v0,v0                 ;# ONES = vector (-1)
    beq       L_stay                    ;# Not Golomb : iterate
    b         L_exit_loop               

    .align    4                         
L_up_level:                             
    ;# POP_LEVEL(lev) : Restore the bitmaps then iterate
    subi      r16,r16,1                 ;# --depth
    lwz       r31,LEVEL_COMP0-SIZEOF_LEVEL(r14) ;# comp0 = (lev-1)->comp0
    subi      r14,r14,SIZEOF_LEVEL      ;# --lev
    lvx       v26,r11,r14               ;# Load listV1
    addi      r6,r6,SIZEOF_OGR          ;# ++ogr
    addi      r5,r5,1                   ;# ++choose
    lwz       r30,LEVEL_LIST0(r14)      ;# Load list0
    li        r24,0                     ;# newbit = 0
    cmpw      r16,r26                   ;# depth <= startdepth ?
    lvx       v27,r12,r14               ;# Load compV
    not       r23,r31                   ;# c0neg = ~comp0
    lvx       v25,0,r14                 ;# Load listV0
    vandc     v24,v24,v26               ;# distV &= ~listV1
    lwz       r25,LEVEL_LIMIT(r14)      ;# Load limit
    andc      r29,r29,r30               ;# dist0 &= ~list0
    srwi      r23,r23,1                 ;# Prepare c0neg
    lwz       r21,LEVEL_CNT2(r14)       ;# Load cnt2
    bgt+      L_stay                    ;# depth > startdepth : iterate

    li        r3,CORE_S_OK              

L_exit_loop:                            
    ;# SAVE_FINAL_STATE(oState,lev)
    stvx      v25,0,r14                 ;# Store listV0
    stvx      v26,r11,r14               ;# Store listV1
    stvx      v24,r9,r13                ;# Store distV
    stvx      v27,r12,r14               ;# Store compV
    stw       r30,LEVEL_LIST0(r14)      ;# Store list0
    stw       r29,STATE_DIST0(r13)      ;# Store dist0
    stw       r31,LEVEL_COMP0(r14)      ;# Store comp0
    stw       r21,LEVEL_CNT2(r14)       ;# Store cnt2
    subi      r16,r16,1                 ;# --depth
    stw       r17,0(r4)                 ;# Store node count
    stw       r16,STATE_DEPTH(r13)      


;#============================================================================
;# Epilog

    ;# Restore vector registers
    li        r8,aVectorArea            
    lvx       v27,r8,r1                 
    addi      r8,r8,16                  
    lvx       v26,r8,r1                 
    addi      r8,r8,16                  
    lvx       v25,r8,r1                 
    addi      r8,r8,16                  
    lvx       v24,r8,r1                 
    addi      r8,r8,16                  
    lvx       v23,r8,r1                 
    addi      r8,r8,16                  
    lvx       v22,r8,r1                 
    addi      r8,r8,16                  
    lvx       v21,r8,r1                 
    addi      r8,r8,16                  
    lvx       v20,r8,r1                 

    ;# Restore non-volatile registers
    lwz       r5,0(r1)                  ;# Obtains caller's stack pointer
    lmw       r13,-GPRSaveArea(r5)      ;# Restore GPRs
    lwz       r7,wVRSave(r5)            ;# Restore VRsave
    mtspr     VRsave,r7                 
    lwz       r6,wSaveLR(r5)            ;# ...from caller's stack
    mtlr      r6                        
    mr        r1,r5                     
    blr                                 

