; Copyright distributed.net 1997-2003 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; PowerPC RC5-72 core
; 2-pipeline version, optimised for 603e
;
; Written by Malcolm Howell <coreblimey@rottingscorpion.com>
; 4th Jan 2003
;
; $Id: r72-ppc-mh-2.osx.s,v 1.2 2003/09/12 23:08:52 mweiser Exp $

gcc2_compiled.:

	.text

	.globl     _rc5_72_unit_func_ppc_mh_2  ; Exported

; Stack variables
	.set    save_ret,4      ; Function return address
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
	.set    var_size,(save_regs + 76 + 15) & -16

; Constants
	.set    P,0xb7e15163
	.set    Q,0x9e3779b9
	.set    P0QR3,0xbf0a8b1d    ; P<<3
	.set    PR3Q,0x15235639     ; P0QR3 + P + Q
	.set    P2Q,0xf45044d5      ; P+2Q
	.set    P3Q,0x9287be8e      ; P+3Q
	.set    P4Q,0x30bf3847      ; P+4Q

	.set    RESULT_NOTHING,1
	.set    RESULT_FOUND,2

; RC5_72UnitWork struct offsets
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

; Args: r3=RC5_72UnitWork *, r4=u32 * iteration_count

_rc5_72_unit_func_ppc_mh_2:
    stwu    r1,-var_size(r1)
    stmw    r13,save_regs(r1)
    lwz     r7,0(r4)            ; Get initial key count
    mflr    r6
    srwi    r7,r7,1             ; Divide by pipeline count
    stw     r3,UWP(r1)          ; Save params
    stw     r4,ILCP(r1)

    ; The loop count is incremented by one, as we start by jumping to the
    ; end of the loop, where the constants are calculated.

    addi    r7,r7,1
    stw     r6,save_ret(r1)     ; Save return address
    mtctr   r7

    ; Make copy of UnitWork on stack
    lmw     r21,0(r3)       ; 11 words, r21-r31
    stmw    r21,UW_copy(r1)

    ; Some constants are expected in registers on entry
    lis     r23,hi16(P4Q)
    lwz     r20,L0_hi(r1)

    b       l0chg

.align  3

mainloop:
    ; Begin round 1, halfway into iteration 2
    ; At start of loop:

    ; r20 contains L0_hi
    ; r23 contains P4Q@h

    lwz     r28,L1P2QR(r1)      ; Get L1P2QR into scratch reg r28
    lis     r0,hi16(Q)
    ori     r3,r23,lo16(P4Q)    ; P4Q@h is preloaded into r23
    addi    r15,r20,1

    lwz     r21,P32QR(r1)
    add     r15,r15,r28
    rotlw   r15,r15,r28
    add     r11,r20,r28
    rotlw   r11,r11,r28
    add     r12,r15,r21

    ; 3b: L0 = ROTL(L0 + L2 + A, L2 + A)

    lwz     r30,L0X(r1)
    rotlwi  r12,r12,3
    ori     r0,r0,lo16(Q)
    add     r17,r11,r21
    rotlwi  r17,r17,3
    add     r6,r15,r12
    stw     r12,mS03b(r1)
    add     r13,r6,r30
    add     r7,r11,r17   		; Use r7 as a second scratch register

    ; 4a: S[04] = A = ROTL(A + L0 + P+4Q, 3)

    rotlw   r13,r13,r6
    add     r18,r17,r3
    add     r9,r7,r30
    add     r12,r12,r3

    rotlw   r9,r9,r7
    add     r12,r12,r13

    ; 4b: L1 = ROTL(L1 + A + L0, A + L0)

    lwz     r31,L1X(r1)
    rotlwi  r12,r12,3
    add     r18,r18,r9
    add     r6,r12,r13      ; r6 = A + L0
    stw     r12,mS04b(r1)
    add     r14,r6,r31     ; L1 = L1X + A + L0

    ; Now pipe a is waiting to rotate A (to find S04), and pipe b has
    ; done S04 already and is waiting to rotlw L1 to find the new L1

    rotlw   r14,r14,r6
    add     r3,r3,r0          ; Advance to P + 5Q
    rotlwi  r18,r18,3         ; Find r18
    add     r12,r12,r14
    add     r6,r18,r9
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r10,r31,r6
    stw     r12,mS05b(r1)        ; Store S05b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r19,r18,r3
    add     r15,r15,r6
    add     r19,r19,r10

    rotlw   r15,r15,r6
    add     r3,r3,r0          ; Advance to P + 6Q
    rotlwi  r19,r19,3         ; Find r19
    add     r12,r12,r15           ; Start iter 6 on b
    add     r6,r19,r10
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS06b(r1)        ; Store S06b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r20,r19,r3       ; Start iter 6 on a
    add     r13,r13,r6
    add     r20,r20,r11

    rotlw   r13,r13,r6
    add     r3,r3,r0          ; P + 7Q
    rotlwi  r20,r20,3         ; Find r20
    add     r12,r12,r13           ; Start iter 7 on b
    add     r6,r20,r11
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS07b(r1)        ; Store S07b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r21,r20,r3
    add     r14,r14,r6
    add     r21,r21,r9

    rotlw   r14,r14,r6
    add     r3,r3,r0          ; P + 8Q
    rotlwi  r21,r21,3         ; Find r21
    add     r12,r12,r14           ; Start iter 8 on b
    add     r6,r21,r9
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS08b(r1)        ; Store S08b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r22,r21,r3
    add     r15,r15,r6
    add     r22,r22,r10

    rotlw   r15,r15,r6
    add     r3,r3,r0          ; P + 9Q
    rotlwi  r22,r22,3         ; Find r22
    add     r12,r12,r15           ; Start iter 9 on b
    add     r6,r22,r10
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS09b(r1)        ; Store S09b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r23,r22,r3       ; Start iter 9 on a
    add     r13,r13,r6
    add     r23,r23,r11

    rotlw   r13,r13,r6
    add     r3,r3,r0          ; P + 10Q
    rotlwi  r23,r23,3         ; Find r23
    add     r12,r12,r13           ; Start iter 10 on b
    add     r6,r23,r11
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS10b(r1)        ; Store S10b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r24,r23,r3       ; Start iter 10 on a
    add     r14,r14,r6
    add     r24,r24,r9

    rotlw   r14,r14,r6
    add     r3,r3,r0          ; P + 11Q
    rotlwi  r24,r24,3         ; Find r24
    add     r12,r12,r14           ; Start iter 11 on b
    add     r6,r24,r9
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS11b(r1)        ; Store S11b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r25,r24,r3       ; Start iter 11 on a
    add     r15,r15,r6
    add     r25,r25,r10

    rotlw   r15,r15,r6
    add     r3,r3,r0          ; P + 12Q
    rotlwi  r25,r25,3         ; Find r25
    add     r12,r12,r15           ; Start iter 12 on b
    add     r6,r25,r10
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS12b(r1)        ; Store S12b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r26,r25,r3       ; Start iter 12 on a
    add     r13,r13,r6
    add     r26,r26,r11

    rotlw   r13,r13,r6
    add     r3,r3,r0          ; P + 13Q
    rotlwi  r26,r26,3         ; Find r26
    add     r12,r12,r13           ; Start iter 13 on b
    add     r6,r26,r11
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS13b(r1)        ; Store S13b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r27,r26,r3       ; Start iter 13 on a
    add     r14,r14,r6
    add     r27,r27,r9

    rotlw   r14,r14,r6
    add     r3,r3,r0          ; P + 14Q
    rotlwi  r27,r27,3         ; Find r27
    add     r12,r12,r14           ; Start iter 14 on b
    add     r6,r27,r9
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS14b(r1)        ; Store S14b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r28,r27,r3       ; Start iter 14 on a
    add     r15,r15,r6
    add     r28,r28,r10

    rotlw   r15,r15,r6
    add     r3,r3,r0          ; P + 15Q
    rotlwi  r28,r28,3         ; Find r28
    add     r12,r12,r15           ; Start iter 15 on b
    add     r6,r28,r10
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS15b(r1)        ; Store S15b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r29,r28,r3       ; Start iter 15 on a
    add     r13,r13,r6
    add     r29,r29,r11

    rotlw   r13,r13,r6
    add     r3,r3,r0          ; P + 16Q
    rotlwi  r29,r29,3         ; Find r29
    add     r12,r12,r13           ; Start iter 16 on b
    add     r6,r29,r11
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS16b(r1)        ; Store S16b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r30,r29,r3       ; Start iter 16 on a
    add     r14,r14,r6
    add     r30,r30,r9

    rotlw   r14,r14,r6
    add     r3,r3,r0          ; P + 17Q
    rotlwi  r30,r30,3         ; Find r30
    add     r12,r12,r14           ; Start iter 17 on b
    add     r6,r30,r9
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS17b(r1)        ; Store S17b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r31,r30,r3       ; Start iter 17 on a
    add     r15,r15,r6
    add     r31,r31,r10

    ; Transition iteration: now exhausted Snna regs, start using r8
    ; and storing results in mem

    rotlw   r15,r15,r6
    add     r3,r3,r0          ; P + 18Q
    rotlwi  r31,r31,3         ; Find r31
    add     r12,r12,r15           ; Start iter 18 on b
    add     r6,r31,r10
    add     r12,r12,r3
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS18b(r1)        ; Store S18b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r8,r31,r3         ; Start iter 18 on a
    add     r13,r13,r6
    add     r8,r8,r11

    rotlw   r13,r13,r6
    add     r3,r3,r0          ; P + 19Q
    rotlwi  r8,r8,3             ; Find S18a
    add     r12,r12,r13           ; Start iter 19 on b
    add     r6,r8,r11
    add     r12,r12,r3
    stw     r8,mS18a(r1)
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS19b(r1)        ; Store S19b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r14,r14,r6
    add     r8,r8,r3           ; Start iter 19 on a

    rotlw   r14,r14,r6
    add     r8,r8,r9
    rotlwi  r8,r8,3             ; Find S19a
    add     r3,r3,r0          ; P + 20Q
    add     r12,r12,r14           ; Start iter 20 on b
    stw     r8,mS19a(r1)
    add     r12,r12,r3
    add     r6,r8,r9
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS20b(r1)        ; Store S20b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r8,r8,r3           ; Start iter 20 on a
    add     r15,r15,r6
    add     r8,r8,r10

    rotlw   r15,r15,r6
    add     r3,r3,r0          ; P + 21Q
    rotlwi  r8,r8,3             ; Find S20a
    add     r12,r12,r15           ; Start iter 21 on b
    add     r6,r8,r10
    add     r12,r12,r3
    stw     r8,mS20a(r1)
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS21b(r1)    ; Store S21b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r8,r8,r3         ; Start iter 21 on a
    add     r13,r13,r6

    rotlw   r13,r13,r6
    add     r8,r8,r11
    rotlwi  r8,r8,3             ; Find S21a
    add     r3,r3,r0          ; P + 22Q
    add     r12,r12,r13           ; Start iter 22 on b
    stw     r8,mS21a(r1)
    add     r12,r12,r3
    add     r6,r8,r11
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS22b(r1)    ; Store S22b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r8,r8,r3           ; Start iter 22 on a
    add     r14,r14,r6
    add     r8,r8,r9

    rotlw   r14,r14,r6
    add     r3,r3,r0          ; P + 23Q
    rotlwi  r8,r8,3             ; Find S22a
    add     r12,r12,r14           ; Start iter 23 on b
    add     r6,r8,r9
    add     r12,r12,r3
    stw     r8,mS22a(r1)
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS23b(r1)        ; Store S23b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r8,r8,r3           ; Start iter 23 on a
    add     r15,r15,r6

    rotlw   r15,r15,r6
    add     r8,r8,r10
    rotlwi  r8,r8,3             ; Find S23a
    add     r12,r12,r15           ; Start iter 24 on b
    add     r3,r3,r0          ; P + 24Q
    stw     r8,mS23a(r1)
    add     r12,r12,r3
    add     r6,r8,r10
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS24b(r1)        ; Store S24b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r8,r8,r3           ; Start iter 24 on a
    add     r13,r13,r6
    add     r8,r8,r11

    rotlwi  r8,r8,3             ; Find S24a
    add     r3,r3,r0          ; P + 25Q
    rotlw   r13,r13,r6
    stw     r8,mS24a(r1)
    add     r12,r12,r13           ; Start iter 25 on b
    add     r6,r8,r11
    add     r12,r12,r3
    add     r9,r9,r6
    rotlwi  r12,r12,3
    add     r8,r8,r3           ; Start iter 25 on a
    add     r7,r12,r13           ; Use r7 as a scratch reg
    stw     r12,mS25b(r1)        ; Store S25b
    rotlw   r9,r9,r6
    add     r14,r14,r7
    rotlw   r14,r14,r7          ; Finished round 1 on b
    add     r8,r8,r9

    ; ---- ROUND 2 begins ----
    ; (Somewhere near here)

; Recap on registers:

; r8-r11:   r8, r9-r11
; r12-r15:  r12, r13-r15
; r17-r31:  r17-r31
; r4,r5,r16:still unused, waiting for r4-r16

; r0,r3:    Were r3, r0; now waiting for r0,r3
; r7:       r7, still unused (except for occasional temp usage)
; r6:       r6

    ; Iteration 0a: S[00] = A = ROTL(A + L1 + P0QR3,3)

    rotlwi  r8,r8,3             ; Find S25a
    add     r12,r12,r14
    lis     r7,hi16(P0QR3)
    stw     r8,mS25a(r1)
    ori     r7,r7,lo16(P0QR3)   ; Both S[00] values are P0QR3
    add     r6,r8,r9

    add     r12,r12,r7
    add     r10,r10,r6
    rotlw   r10,r10,r6         ; Finish round 1 on a
    add     r8,r8,r7
    rotlwi  r0,r12,3
    add     r8,r8,r10

    rotlwi  r4,r8,3
    add     r6,r0,r14
    add     r15,r15,r6
    lwz     r7,AX(r1)

    ; 0b: L2 = ROTL(L2 + A + L1, A + L1)

    rotlw   r15,r15,r6
    add     r6,r4,r10
    add     r11,r11,r6
    add     r3,r0,r15
    rotlw   r11,r11,r6
    add     r3,r3,r7

    ; 1a: S[01] = A = ROTL(A + L2 + AX,3)

    rotlwi  r3,r3,3
    add     r5,r4,r11

    ; 1b: L0 = ROTL(L0 + A + L2, A + L2)

    add     r6,r3,r15
    add     r5,r5,r7
    add     r13,r13,r6
    lwz     r7,AXP2QR(r1)
    rotlw   r13,r13,r6
    add     r12,r3,r13

    ; 2a: S[02] = A = ROTL(A + L0 + AXP2QR,3)

    rotlwi  r5,r5,3
    ; Oops, stalled again
    add     r6,r5,r11
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS02b(r1)       ; Store S02b in memory

    ; 2b: L1 = ROTL(L1 + A + L0, A + L0)

    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r16,r5,r9
    add     r14,r14,r6
    add     r16,r16,r7

#---- Start the round 2 repeating code

    rotlw   r14,r14,r6
    lwz     r7,mS03b(r1)          ; Fetch S03b from round 1
    rotlwi  r16,r16,3             ; Calculate r16
    add     r12,r12,r14               ; Begin iter 3 on pipe b
    add     r6,r16,r9
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS03b(r1)            ; Store S03b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r17,r16,r17          ; Begin iter 3 on pipe a
    add     r15,r15,r6
    add     r17,r17,r10

    rotlw   r15,r15,r6
    lwz     r7,mS04b(r1)          ; Fetch S04b from round 1
    rotlwi  r17,r17,3             ; Calculate r17
    add     r12,r12,r15               ; Begin iter 4 on pipe b
    add     r6,r17,r10
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS04b(r1)            ; Store S04b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r18,r17,r18          ; Begin iter 4 on pipe a
    add     r13,r13,r6
    add     r18,r18,r11

    rotlw   r13,r13,r6
    lwz     r7,mS05b(r1)          ; Fetch S05b from round 1
    rotlwi  r18,r18,3             ; Calculate r18
    add     r12,r12,r13               ; Begin iter 5 on pipe b
    add     r6,r18,r11
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS05b(r1)            ; Store S05b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r19,r18,r19          ; Begin iter 5 on pipe a
    add     r14,r14,r6
    add     r19,r19,r9

    rotlw   r14,r14,r6
    lwz     r7,mS06b(r1)          ; Fetch S06b from round 1
    rotlwi  r19,r19,3             ; Calculate r19
    add     r12,r12,r14               ; Begin iter 6 on pipe b
    add     r6,r19,r9
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS06b(r1)            ; Store S06b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r20,r19,r20          ; Begin iter 6 on pipe a
    add     r15,r15,r6
    add     r20,r20,r10

    rotlw   r15,r15,r6
    lwz     r7,mS07b(r1)          ; Fetch S07b from round 1
    rotlwi  r20,r20,3             ; Calculate r20
    add     r12,r12,r15               ; Begin iter 7 on pipe b
    add     r6,r20,r10
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS07b(r1)            ; Store S07b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r21,r20,r21          ; Begin iter 7 on pipe a
    add     r13,r13,r6
    add     r21,r21,r11

    rotlw   r13,r13,r6
    lwz     r7,mS08b(r1)          ; Fetch S08b from round 1
    rotlwi  r21,r21,3             ; Calculate r21
    add     r12,r12,r13               ; Begin iter 8 on pipe b
    add     r6,r21,r11
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS08b(r1)            ; Store S08b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r22,r21,r22          ; Begin iter 8 on pipe a
    add     r14,r14,r6
    add     r22,r22,r9

    rotlw   r14,r14,r6
    lwz     r7,mS09b(r1)          ; Fetch S09b from round 1
    rotlwi  r22,r22,3             ; Calculate r22
    add     r12,r12,r14               ; Begin iter 9 on pipe b
    add     r6,r22,r9
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS09b(r1)            ; Store S09b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r23,r22,r23          ; Begin iter 9 on pipe a
    add     r15,r15,r6
    add     r23,r23,r10

    rotlw   r15,r15,r6
    lwz     r7,mS10b(r1)          ; Fetch S10b from round 1
    rotlwi  r23,r23,3             ; Calculate r23
    add     r12,r12,r15               ; Begin iter 10 on pipe b
    add     r6,r23,r10
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS10b(r1)            ; Store S10b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r24,r23,r24          ; Begin iter 10 on pipe a
    add     r13,r13,r6
    add     r24,r24,r11

    rotlw   r13,r13,r6
    lwz     r7,mS11b(r1)          ; Fetch S11b from round 1
    rotlwi  r24,r24,3             ; Calculate r24
    add     r12,r12,r13               ; Begin iter 11 on pipe b
    add     r6,r24,r11
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS11b(r1)            ; Store S11b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r25,r24,r25          ; Begin iter 11 on pipe a
    add     r14,r14,r6
    add     r25,r25,r9

    rotlw   r14,r14,r6
    lwz     r7,mS12b(r1)          ; Fetch S12b from round 1
    rotlwi  r25,r25,3             ; Calculate r25
    add     r12,r12,r14               ; Begin iter 12 on pipe b
    add     r6,r25,r9
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS12b(r1)            ; Store S12b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r26,r25,r26          ; Begin iter 12 on pipe a
    add     r15,r15,r6
    add     r26,r26,r10

    rotlw   r15,r15,r6
    lwz     r7,mS13b(r1)          ; Fetch S13b from round 1
    rotlwi  r26,r26,3             ; Calculate r26
    add     r12,r12,r15               ; Begin iter 13 on pipe b
    add     r6,r26,r10
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS13b(r1)            ; Store S13b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r27,r26,r27          ; Begin iter 13 on pipe a
    add     r13,r13,r6
    add     r27,r27,r11

    rotlw   r13,r13,r6
    lwz     r7,mS14b(r1)          ; Fetch S14b from round 1
    rotlwi  r27,r27,3             ; Calculate r27
    add     r12,r12,r13               ; Begin iter 14 on pipe b
    add     r6,r27,r11
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS14b(r1)            ; Store S14b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r28,r27,r28          ; Begin iter 14 on pipe a
    add     r14,r14,r6
    add     r28,r28,r9

    rotlw   r14,r14,r6
    lwz     r7,mS15b(r1)          ; Fetch S15b from round 1
    rotlwi  r28,r28,3             ; Calculate r28
    add     r12,r12,r14               ; Begin iter 15 on pipe b
    add     r6,r28,r9
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS15b(r1)            ; Store S15b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r29,r28,r29          ; Begin iter 15 on pipe a
    add     r15,r15,r6
    add     r29,r29,r10

    rotlw   r15,r15,r6
    lwz     r7,mS16b(r1)          ; Fetch S16b from round 1
    rotlwi  r29,r29,3             ; Calculate r29
    add     r12,r12,r15               ; Begin iter 16 on pipe b
    add     r6,r29,r10
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS16b(r1)            ; Store S16b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r30,r29,r30          ; Begin iter 16 on pipe a
    add     r13,r13,r6
    add     r30,r30,r11

    rotlw   r13,r13,r6
    lwz     r7,mS17b(r1)          ; Fetch S17b from round 1
    rotlwi  r30,r30,3             ; Calculate r30
    add     r12,r12,r13               ; Begin iter 17 on pipe b
    add     r6,r30,r11
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS17b(r1)            ; Store S17b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r31,r30,r31          ; Begin iter 17 on pipe a
    add     r14,r14,r6
    add     r31,r31,r9

    ; This block fetches S18a from memory
    rotlw   r14,r14,r6
    lwz     r7,mS18b(r1)          ; Fetch S18b from round 1
    rotlwi  r31,r31,3             ; Calculate r31
    add     r12,r12,r14               ; Begin iter 18 on pipe b
    add     r6,r31,r9
    add     r12,r12,r7
    lwz     r7,mS18a(r1)          ; Fetch S18a
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS18b(r1)            ; Store S18b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r8,r31,r7            ; Begin iter 18 on pipe a
    add     r15,r15,r6

    ; This block stores S18a and fetches S19a
    add     r8,r8,r10
    rotlw   r15,r15,r6
    lwz     r7,mS19b(r1)          ; Fetch S19b from round 1
    rotlwi  r8,r8,3                 ; Calculate S18a
    add     r12,r12,r15               ; Begin iter 19 on pipe b
    add     r6,r8,r10
    stw     r8,mS18a(r1)            ; Store S18a
    add     r12,r12,r7
    lwz     r7,mS19a(r1)          ; Fetch S19a
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS19b(r1)            ; Store S19b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r8,r8,r7              ; Begin iter 19 on pipe a
    add     r13,r13,r6

    add     r8,r8,r11
    rotlw   r13,r13,r6
    lwz     r7,mS20b(r1)          ; Fetch S20b from round 1
    rotlwi  r8,r8,3                 ; Calculate S19a
    add     r12,r12,r13               ; Begin iter 20 on pipe b
    add     r6,r8,r11
    stw     r8,mS19a(r1)            ; Store S19a
    add     r12,r12,r7
    lwz     r7,mS20a(r1)          ; Fetch S20a
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS20b(r1)            ; Store S20b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r8,r8,r7              ; Begin iter 20 on pipe a
    add     r14,r14,r6

    add     r8,r8,r9
    rotlw   r14,r14,r6
    lwz     r7,mS21b(r1)          ; Fetch S21b from round 1
    rotlwi  r8,r8,3                 ; Calculate S20a
    add     r12,r12,r14               ; Begin iter 21 on pipe b
    add     r6,r8,r9
    stw     r8,mS20a(r1)            ; Store S20a
    add     r12,r12,r7
    lwz     r7,mS21a(r1)          ; Fetch S21a
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS21b(r1)            ; Store S21b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r8,r8,r7              ; Begin iter 21 on pipe a
    add     r15,r15,r6

    add     r8,r8,r10
    rotlw   r15,r15,r6
    lwz     r7,mS22b(r1)          ; Fetch S22b from round 1
    rotlwi  r8,r8,3                 ; Calculate S21a
    add     r12,r12,r15               ; Begin iter 22 on pipe b
    add     r6,r8,r10
    stw     r8,mS21a(r1)            ; Store S21a
    add     r12,r12,r7
    lwz     r7,mS22a(r1)          ; Fetch S22a
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS22b(r1)            ; Store S22b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r8,r8,r7              ; Begin iter 22 on pipe a
    add     r13,r13,r6

    add     r8,r8,r11
    rotlw   r13,r13,r6
    lwz     r7,mS23b(r1)          ; Fetch S23b from round 1
    rotlwi  r8,r8,3                 ; Calculate S22a
    add     r12,r12,r13               ; Begin iter 23 on pipe b
    add     r6,r8,r11
    stw     r8,mS22a(r1)            ; Store S22a
    add     r12,r12,r7
    lwz     r7,mS23a(r1)          ; Fetch S23a
    rotlwi  r12,r12,3
    add     r9,r9,r6
    stw     r12,mS23b(r1)            ; Store S23b
    rotlw   r9,r9,r6
    add     r6,r12,r13
    add     r8,r8,r7              ; Begin iter 23 on pipe a
    add     r14,r14,r6

    add     r8,r8,r9
    rotlw   r14,r14,r6
    lwz     r7,mS24b(r1)          ; Fetch S24b from round 1
    rotlwi  r8,r8,3                 ; Calculate S23a
    add     r12,r12,r14               ; Begin iter 24 on pipe b
    add     r6,r8,r9
    stw     r8,mS23a(r1)            ; Store S23a
    add     r12,r12,r7
    lwz     r7,mS24a(r1)          ; Fetch S24a
    rotlwi  r12,r12,3
    add     r10,r10,r6
    stw     r12,mS24b(r1)            ; Store S24b
    rotlw   r10,r10,r6
    add     r6,r12,r14
    add     r8,r8,r7              ; Begin iter 24 on pipe a
    add     r15,r15,r6

    add     r8,r8,r10
    rotlw   r15,r15,r6
    lwz     r7,mS25b(r1)          ; Fetch S25b from round 1
    rotlwi  r8,r8,3                 ; Calculate S24a
    add     r12,r12,r15               ; Begin iter 25 on pipe b
    add     r6,r8,r10
    stw     r8,mS24a(r1)            ; Store S24a
    add     r12,r12,r7
    lwz     r7,mS25a(r1)          ; Fetch S25a
    rotlwi  r12,r12,3
    add     r11,r11,r6
    stw     r12,mS25b(r1)            ; Store S25b
    rotlw   r11,r11,r6
    add     r6,r12,r15
    add     r8,r8,r7              ; Begin iter 25 on pipe a
    add     r13,r13,r6

    rotlw   r13,r13,r6             ; End of round 2 on pipe b
    add     r8,r8,r11
    rotlwi  r8,r8,3                 ; Calculate S25a
    add     r12,r12,r0              ; r0=r0 now becomes r0
    lwz     r7,plain_lo(r1)
    add     r6,r8,r11
    add     r12,r12,r13
    add     r9,r9,r6
    stw     r8,mS25a(r1)            ; Store S25a
    rotlw   r9,r9,r6             ; End of round 2 on pipe a

; --- End of round 2

; Registers:

; r0,r3:        r0,r3
; r4,r5:        r4,r5
; r6,r7:        available (r6 and r7)
; r8-r15:       A,L0-L2
; r16-r31:      r16-r31

    ; ---- ROUND 3 / ENCRYPTION ----

    rotlwi  r12,r12,3
    add     r8,r8,r4  ; r4=r4 now becomes r4
    add     r6,r12,r13
    add     r8,r8,r9
    rotlwi  r8,r8,3
    add     r14,r14,r6
    rotlw   r14,r14,r6
    add     r4,r7,r8  ; eA = plain_lo + S[00]
    add     r6,r8,r9
    add     r0,r7,r12  ; Finish with r7

    ; L1 = ROTL(L1 + A + L0, A + L0)

    add     r10,r10,r6
    add     r8,r8,r5  ; r5=r5 becomes r5
    rotlw   r10,r10,r6
    lwz     r7,plain_hi(r1)

    ; Iteration 1

    add     r12,r12,r3  ; r3=r3 becomes r3
    add     r8,r8,r10
    rotlwi  r8,r8,3
    add     r12,r12,r14

    rotlwi  r12,r12,3
    add     r6,r8,r10

    add     r5,r7,r8 ; eB = plain_hi + S[01]
    add     r11,r11,r6
    rotlw   r11,r11,r6
    add     r3,r7,r12

    ; L2 = ROTL(L2 + A + L1, A + L1)

    add     r6,r12,r14
    add     r16,r16,r8
    add     r15,r15,r6
    add     r16,r16,r11

    ; After a nifty bit of overtaking, pipe a is now ahead of pipe b

    rotlw   r15,r15,r6
    lwz     r7,mS02b(r1)      ; Fetch S02b
    rotlwi  r16,r16,3         ; Calculate r16
    add     r12,r12,r15           ; Begin iteration 2 on b
    xor     r4,r4,r5
    add     r6,r16,r11
    xor     r0,r0,r3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r9,r9,r6
    rotlwi  r12,r12,3
    add     r4,r4,r16
    rotlw   r9,r9,r6
    add     r17,r17,r16      ; Begin iteration 3 on a
    rotlw   r0,r0,r3
    add     r6,r12,r15
    xor     r5,r5,r4
    add     r17,r17,r9
    add     r13,r13,r6
    lwz     r7,mS03b(r1)      ; Fetch S03b
    rotlw   r13,r13,r6
    add     r0,r0,r12
    rotlwi  r17,r17,3         ; Calculate r17
    add     r12,r12,r13           ; Begin iteration 3 on b
    xor     r3,r3,r0
    add     r6,r17,r9
    rotlw   r5,r5,r4
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r10,r10,r6
    rotlw   r3,r3,r0
    add     r5,r5,r17
    rotlw   r10,r10,r6
    add     r18,r18,r17      ; Begin iteration 4 on a
    add     r6,r12,r13
    add     r3,r3,r12          ; Add S03b to text B
    add     r14,r14,r6
    add     r18,r18,r10

    rotlw   r14,r14,r6
    lwz     r7,mS04b(r1)      ; Fetch S04b
    rotlwi  r18,r18,3         ; Calculate r18
    add     r12,r12,r14           ; Begin iteration 4 on b
    xor     r4,r4,r5
    add     r6,r18,r10
    xor     r0,r0,r3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r11,r11,r6
    rotlwi  r12,r12,3
    add     r4,r4,r18
    rotlw   r11,r11,r6
    add     r19,r19,r18      ; Begin iteration 5 on a
    rotlw   r0,r0,r3
    add     r6,r12,r14
    xor     r5,r5,r4
    add     r19,r19,r11
    add     r15,r15,r6
    lwz     r7,mS05b(r1)      ; Fetch S05b
    rotlw   r15,r15,r6
    add     r0,r0,r12
    rotlwi  r19,r19,3         ; Calculate r19
    add     r12,r12,r15           ; Begin iteration 5 on b
    xor     r3,r3,r0
    add     r6,r19,r11
    rotlw   r5,r5,r4
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r9,r9,r6
    rotlw   r3,r3,r0
    add     r5,r5,r19
    rotlw   r9,r9,r6
    add     r20,r20,r19      ; Begin iteration 6 on a
    add     r6,r12,r15
    add     r3,r3,r12
    add     r13,r13,r6
    add     r20,r20,r9

    rotlw   r13,r13,r6
    lwz     r7,mS06b(r1)      ; Fetch S06b
    rotlwi  r20,r20,3         ; Calculate r20
    add     r12,r12,r13           ; Begin iteration 6 on b
    xor     r4,r4,r5
    add     r6,r20,r9
    xor     r0,r0,r3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r10,r10,r6
    rotlwi  r12,r12,3
    add     r4,r4,r20
    rotlw   r10,r10,r6
    add     r21,r21,r20      ; Begin iteration 7 on a
    rotlw   r0,r0,r3
    add     r6,r12,r13
    xor     r5,r5,r4
    add     r21,r21,r10
    add     r14,r14,r6
    lwz     r7,mS07b(r1)      ; Fetch S07b
    rotlw   r14,r14,r6
    add     r0,r0,r12
    rotlwi  r21,r21,3         ; Calculate r21
    add     r12,r12,r14           ; Begin iteration 7 on b
    xor     r3,r3,r0
    add     r6,r21,r10
    rotlw   r5,r5,r4
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r11,r11,r6
    rotlw   r3,r3,r0
    add     r5,r5,r21
    rotlw   r11,r11,r6
    add     r22,r22,r21      ; Begin iteration 8 on a
    add     r6,r12,r14
    add     r3,r3,r12
    add     r15,r15,r6
    add     r22,r22,r11

    rotlw   r15,r15,r6
    lwz     r7,mS08b(r1)      ; Fetch S08b
    rotlwi  r22,r22,3         ; Calculate r22
    add     r12,r12,r15           ; Begin iteration 8 on b
    xor     r4,r4,r5
    add     r6,r22,r11
    xor     r0,r0,r3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r9,r9,r6
    rotlwi  r12,r12,3
    add     r4,r4,r22
    rotlw   r9,r9,r6
    add     r23,r23,r22      ; Begin iteration 9 on a
    rotlw   r0,r0,r3
    add     r6,r12,r15
    xor     r5,r5,r4
    add     r23,r23,r9
    add     r13,r13,r6
    lwz     r7,mS09b(r1)      ; Fetch S09b
    rotlw   r13,r13,r6
    add     r0,r0,r12
    rotlwi  r23,r23,3         ; Calculate r23
    add     r12,r12,r13           ; Begin iteration 9 on b
    xor     r3,r3,r0
    add     r6,r23,r9
    rotlw   r5,r5,r4
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r10,r10,r6
    rotlw   r3,r3,r0
    add     r5,r5,r23
    rotlw   r10,r10,r6
    add     r24,r24,r23      ; Begin iteration 10 on a
    add     r6,r12,r13
    add     r3,r3,r12
    add     r14,r14,r6
    add     r24,r24,r10

    rotlw   r14,r14,r6
    lwz     r7,mS10b(r1)      ; Fetch S10b
    rotlwi  r24,r24,3         ; Calculate r24
    add     r12,r12,r14           ; Begin iteration 10 on b
    xor     r4,r4,r5
    add     r6,r24,r10
    xor     r0,r0,r3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r11,r11,r6
    rotlwi  r12,r12,3
    add     r4,r4,r24
    rotlw   r11,r11,r6
    add     r25,r25,r24      ; Begin iteration 11 on a
    rotlw   r0,r0,r3
    add     r6,r12,r14
    xor     r5,r5,r4
    add     r25,r25,r11
    add     r15,r15,r6
    lwz     r7,mS11b(r1)      ; Fetch S11b
    rotlw   r15,r15,r6
    add     r0,r0,r12
    rotlwi  r25,r25,3         ; Calculate r25
    add     r12,r12,r15           ; Begin iteration 11 on b
    xor     r3,r3,r0
    add     r6,r25,r11
    rotlw   r5,r5,r4
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r9,r9,r6
    rotlw   r3,r3,r0
    add     r5,r5,r25
    rotlw   r9,r9,r6
    add     r26,r26,r25      ; Begin iteration 12 on a
    add     r6,r12,r15
    add     r3,r3,r12
    add     r13,r13,r6
    add     r26,r26,r9

    rotlw   r13,r13,r6
    lwz     r7,mS12b(r1)      ; Fetch S12b
    rotlwi  r26,r26,3         ; Calculate r26
    add     r12,r12,r13           ; Begin iteration 12 on b
    xor     r4,r4,r5
    add     r6,r26,r9
    xor     r0,r0,r3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r10,r10,r6
    rotlwi  r12,r12,3
    add     r4,r4,r26
    rotlw   r10,r10,r6
    add     r27,r27,r26      ; Begin iteration 13 on a
    rotlw   r0,r0,r3
    add     r6,r12,r13
    xor     r5,r5,r4
    add     r27,r27,r10
    add     r14,r14,r6
    lwz     r7,mS13b(r1)      ; Fetch S13b
    rotlw   r14,r14,r6
    add     r0,r0,r12
    rotlwi  r27,r27,3         ; Calculate r27
    add     r12,r12,r14           ; Begin iteration 13 on b
    xor     r3,r3,r0
    add     r6,r27,r10
    rotlw   r5,r5,r4
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r11,r11,r6
    rotlw   r3,r3,r0
    add     r5,r5,r27
    rotlw   r11,r11,r6
    add     r28,r28,r27      ; Begin iteration 14 on a
    add     r6,r12,r14
    add     r3,r3,r12
    add     r15,r15,r6
    add     r28,r28,r11

    rotlw   r15,r15,r6
    lwz     r7,mS14b(r1)      ; Fetch S14b
    rotlwi  r28,r28,3         ; Calculate r28
    add     r12,r12,r15           ; Begin iteration 14 on b
    xor     r4,r4,r5
    add     r6,r28,r11
    xor     r0,r0,r3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r9,r9,r6
    rotlwi  r12,r12,3
    add     r4,r4,r28
    rotlw   r9,r9,r6
    add     r29,r29,r28      ; Begin iteration 15 on a
    rotlw   r0,r0,r3
    add     r6,r12,r15
    xor     r5,r5,r4
    add     r29,r29,r9
    add     r13,r13,r6
    lwz     r7,mS15b(r1)      ; Fetch S15b
    rotlw   r13,r13,r6
    add     r0,r0,r12
    rotlwi  r29,r29,3         ; Calculate r29
    add     r12,r12,r13           ; Begin iteration 15 on b
    xor     r3,r3,r0
    add     r6,r29,r9
    rotlw   r5,r5,r4
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r10,r10,r6
    rotlw   r3,r3,r0
    add     r5,r5,r29
    rotlw   r10,r10,r6
    add     r30,r30,r29      ; Begin iteration 16 on a
    add     r6,r12,r13
    add     r3,r3,r12
    add     r14,r14,r6
    add     r30,r30,r10

    rotlw   r14,r14,r6
    lwz     r7,mS16b(r1)      ; Fetch S16b
    rotlwi  r30,r30,3         ; Calculate r30
    add     r12,r12,r14           ; Begin iteration 16 on b
    xor     r4,r4,r5
    add     r6,r30,r10
    xor     r0,r0,r3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r11,r11,r6
    rotlwi  r12,r12,3
    add     r4,r4,r30
    lwz     r20,L0_hi(r1)       ; Load some useful data in advance
    lis     r23,hi16(P4Q)       ; Load half of r3, ready for next mainloop
    rotlw   r11,r11,r6
    add     r31,r31,r30      ; Begin iteration 17 on a
    rotlw   r0,r0,r3
    add     r6,r12,r14
    xor     r5,r5,r4
    add     r31,r31,r11
    add     r15,r15,r6
    lwz     r7,mS17b(r1)      ; Fetch S17b
    rotlw   r15,r15,r6
    add     r0,r0,r12
    rotlwi  r31,r31,3         ; Calculate r31
    add     r12,r12,r15           ; Begin iteration 17 on b
    xor     r3,r3,r0
    add     r6,r31,r11
    rotlw   r5,r5,r4
    add     r12,r12,r7

    rotlwi  r12,r12,3
    lwz     r7,mS18a(r1)      ; Fetch S18a
    rotlw   r3,r3,r0
    add     r5,r5,r31
    add     r9,r9,r6
    add     r3,r3,r12
    rotlw   r9,r9,r6
    add     r6,r12,r15
    add     r8,r31,r7
    add     r13,r13,r6
    lwz     r7,mS18b(r1)      ; Fetch S18b
    xor     r4,r4,r5
    rotlw   r13,r13,r6
    add     r8,r8,r9
    xor     r0,r0,r3
    add     r12,r12,r13
    rotlwi  r8,r8,3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r6,r8,r9

    rotlwi  r12,r12,3
    lwz     r7,mS19a(r1)      ; Fetch S19a
    rotlw   r0,r0,r3
    add     r4,r4,r8
    add     r10,r10,r6
    add     r0,r0,r12
    rotlw   r10,r10,r6
    add     r6,r12,r13
    add     r8,r8,r7
    add     r14,r14,r6
    lwz     r7,mS19b(r1)      ; Fetch S19b
    xor     r5,r5,r4
    rotlw   r14,r14,r6
    add     r8,r8,r10
    xor     r3,r3,r0
    add     r12,r12,r14
    rotlwi  r8,r8,3
    add     r12,r12,r7
    rotlw   r5,r5,r4
    add     r6,r8,r10

    rotlwi  r12,r12,3
    lwz     r7,mS20a(r1)      ; Fetch S20a
    rotlw   r3,r3,r0
    add     r5,r5,r8
    add     r11,r11,r6
    add     r3,r3,r12
    rotlw   r11,r11,r6
    add     r6,r12,r14
    add     r8,r8,r7
    add     r15,r15,r6
    lwz     r7,mS20b(r1)      ; Fetch S20b
    xor     r4,r4,r5
    rotlw   r15,r15,r6
    add     r8,r8,r11
    xor     r0,r0,r3
    add     r12,r12,r15
    rotlwi  r8,r8,3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r6,r8,r11

    rotlwi  r12,r12,3
    lwz     r7,mS21a(r1)      ; Fetch S21a
    rotlw   r0,r0,r3
    add     r4,r4,r8
    add     r9,r9,r6
    add     r0,r0,r12
    rotlw   r9,r9,r6
    add     r6,r12,r15
    add     r8,r8,r7
    add     r13,r13,r6
    lwz     r7,mS21b(r1)      ; Fetch S21b
    xor     r5,r5,r4
    rotlw   r13,r13,r6
    add     r8,r8,r9
    xor     r3,r3,r0
    add     r12,r12,r13
    rotlwi  r8,r8,3
    add     r12,r12,r7
    rotlw   r5,r5,r4
    add     r6,r8,r9

    rotlwi  r12,r12,3
    lwz     r7,mS22a(r1)      ; Fetch S22a
    rotlw   r3,r3,r0
    add     r5,r5,r8
    add     r10,r10,r6
    add     r3,r3,r12
    rotlw   r10,r10,r6
    add     r6,r12,r13
    add     r8,r8,r7
    add     r14,r14,r6
    lwz     r7,mS22b(r1)      ; Fetch S22b
    xor     r4,r4,r5
    rotlw   r14,r14,r6
    add     r8,r8,r10
    xor     r0,r0,r3
    add     r12,r12,r14
    rotlwi  r8,r8,3
    add     r12,r12,r7
    rotlw   r4,r4,r5
    add     r6,r8,r10

    rotlwi  r12,r12,3
    lwz     r7,mS23a(r1)      ; Fetch S23a
    rotlw   r0,r0,r3
    add     r4,r4,r8
    add     r11,r11,r6
    add     r0,r0,r12
    rotlw   r11,r11,r6
    add     r6,r12,r14
    add     r8,r8,r7
    add     r15,r15,r6
    lwz     r7,mS23b(r1)      ; Fetch S23b
    xor     r5,r5,r4
    rotlw   r15,r15,r6
    add     r8,r8,r11
    xor     r3,r3,r0
    add     r12,r12,r15
    rotlwi  r8,r8,3
    add     r12,r12,r7
    rotlw   r5,r5,r4
    add     r6,r8,r11

    ; Iteration 24 - produces cypher_lo

    rotlwi  r12,r12,3
    lwz     r7,mS24a(r1)      ; Fetch S24a
    rotlw   r3,r3,r0
    add     r5,r5,r8
    add     r9,r9,r6
    add     r3,r3,r12
    rotlw   r9,r9,r6
    add     r6,r12,r15
    add     r8,r8,r7
    add     r13,r13,r6
    lwz     r7,mS24b(r1)      ; Fetch S24b
    xor     r4,r4,r5
    rotlw   r13,r13,r6
    add     r8,r8,r9
    lwz     r16,cypher_lo(r1)
    xor     r0,r0,r3
    rotlwi  r8,r8,3
    add     r12,r12,r13
    rotlw   r4,r4,r5
    add     r12,r12,r7

    rotlwi  r12,r12,3
    add     r4,r4,r8      ; cypher_lo is in r4
    rotlw   r0,r0,r3
    cmpw    r4,r16

    add     r0,r0,r12      ; cypher_lo in r0
    bne+    notfounda

    ; Registers r17-r31 used to contain Snna values - we can now use
    ; these for workspace

    ; Found something on pipeline a - first word matches

    ; Fill in "check" portion of RC5_72UnitWork
    lwz     r28,check_count(r1)
    addi    r28,r28,1
    stw     r28,check_count(r1)
    lmw     r29,L0_hi(r1)       ; 3 words, r29-r31
    stmw    r29,check_hi(r1)

    ; Check whole 64-bit block
    lwz     r7,mS25a(r1)      ; Fetch S25a
    add     r6,r8,r9
    lwz     r18,cypher_hi(r1)
    add     r10,r10,r6
    xor     r5,r5,r4
    rotlw   r10,r10,r6
    add     r8,r8,r7
    rotlw   r5,r5,r4
    add     r8,r8,r10
    rotlwi  r8,r8,3

    add     r5,r5,r8

    cmpw    r18,r5
    bne+    notfounda

    ; Whole block matches for this key...

    lwz     r7,ILCP(r1)     ; Get pointer to initial key count
    mfctr   r8              ; Get remaining loop count
    lwz     r9,0(r7)        ; Get initial key count
    slwi    r8,r8,1         ; Multiply loop count by pipeline count
    subf    r8,r8,r9        ; Difference = number of keys unsuccessfully tried
                            ; = index of successful key
    stw     r8,0(r7)        ; Return number in in/out parameter

    ; Return
    li      r3,RESULT_FOUND

    b       ruf_exit

    .align 3
notfounda:
    cmpw    r0,r16
    addi    r20,r20,2       ; Increment key_hi by pipeline count
    bne+    notfoundb

    ; Half a match found on pipeline b

    ; Fill in "check" portion of RC5_72UnitWork
    lwz     r28,check_count(r1)
    addi    r28,r28,1
    stw     r28,check_count(r1)
    lmw     r29,L0_hi(r1)
    addi    r29,r29,1
    stmw    r29,check_hi(r1)

    ; Thorough check
    lwz     r7,mS25b(r1)      ; Fetch S25b
    add     r6,r12,r13
    lwz     r18,cypher_hi(r1)
    add     r14,r14,r6
    xor     r3,r3,r0
    rotlw   r14,r14,r6
    add     r12,r12,r14
    rotlw   r3,r3,r0
    add     r12,r12,r7
    rotlwi  r12,r12,3
    add     r3,r3,r12

    cmpw    r18,r3
    bne+    notfoundb

    lwz     r7,ILCP(r1)     ; Get pointer to initial key count
    mfctr   r8              ; Get remaining loop count
    lwz     r9,0(r7)        ; Get initial key count
    add     r8,r8,r8        ; Multiply loop count by pipeline count
    subf    r8,r8,r9        ; Difference = number of keys unsuccessfully tried
                            ; = index of successful key
    addi    r8,r8,1         ; Add one for pipeline b
    stw     r8,0(r7)        ; Return number in in/out parameter

    ; Return
    li      r3,RESULT_FOUND

    b       ruf_exit

    .align 3
notfoundb:

    ; Mangle-increment the key by pipeline count (done earlier)
    ; Can't store the key until both keys verified as wrong
    stb     r20,L0_hi+3(r1) ; Just store the byte of key
    andi.   r19,r20,0x0100  ; Check for carry (target reg just a dummy)
    bc      8,2,mainloop    ; No carry && ctr != 0 -> jump to start of loop

    ; If that didn't branch, we have counter == 0, or a carry, or both.
    beq-    exitloop        ; If there is no carry but we didn't follow
                            ; the last jump, counter must have expired

    ; Therefore at this point, there *is* a carry to deal with, but we
    ; still don't know whether the counter ran out. Therefore, nudge the
    ; counter back up (so it can be decremented and tested again)
    ; and deal with the carry, which is the correct thing to do before
    ; returning if the counter has expired.

    ; So L0_hi carried, but the mainloop expects L0_hi in r20
    ; Need to clear the carry bit
    li      r5,L0_mid
    mfctr   r7
    lwbrx   r4,r5,r1        ; Get byte-reversed key_mid
    addi    r7,r7,1
    andi.   r20,r20,0xFEFF  ; Use the load latency to do carry clearing
    addic.  r4,r4,1         ; If result is 0, it carried
    mtctr   r7
    stwbrx  r4,r5,r1
    bne+    l1chg

    li      r5,L0_lo
    lwbrx   r4,r5,r1
    addi    r4,r4,1         ; Not concerned with carries here
    stwbrx  r4,r5,r1

    ; Fall through to l0chg

l0chg:
    ; Calculate L0-constants
    lwz     r30,L0_lo(r1)   ; r30=L0.lo=L0

    lis     r31,hi16(P0QR3) ; r31=P0QR3
    ori     r31,r31,lo16(P0QR3)

    add     r30,r30,r31

    rotlwi  r30,r30,29      ; r30=(L0+P0QR3)<<(P0QR3)=L0X

    stw     r30,L0X(r1)

    lis     r28,hi16(PR3Q)
    ori     r28,r28,lo16(PR3Q)
    add     r28,r28,r30
    rotlwi  r28,r28,3       ; r28=AX

    stw     r28,AX(r1)

    add     r29,r30,r28     ; r29=L0XAX
    stw     r29,L0XAX(r1)

    lis     r27,hi16(P2Q)
    ori     r27,r27,lo16(P2Q)
    add     r27,r28,r27     ; r27=AXP2Q

    stw     r27,AXP2Q(r1)

l1chg:
    ; --- No assumptions about registers left from l0chg ---
    ;     Calculate L1-constants

    lwz     r31,L0XAX(r1)
    lis     r26,hi16(P3Q)
    lwz     r30,L0_mid(r1)  ; r30=L0_mid=L1
    ori     r26,r26,lo16(P3Q)
    add     r30,r30,r31
    lwz     r29,AXP2Q(r1)
    rotlw   r30,r30,r31     ; r30=L1X

    stw     r30,L1X(r1)

    add     r28,r29,r30
    rotlwi  r28,r28,3       ; r28=AXP2QR

    stw     r28,AXP2QR(r1)

    add     r27,r30,r28     ; r27=L1P2QR

    stw     r27,L1P2QR(r1)


    add     r26,r26,r28     ; r26=P32QR

    stw     r26,P32QR(r1)

loopend:
    bdnz+   mainloop

exitloop:
    ; Nothing found; clean up and exit
    li      r3,RESULT_NOTHING

    ; No need to fiddle with the iterations arg, as we have done the
    ; number requested (assuming it was a multiple of 2)

ruf_exit:
    ; Transfer copy of UnitWork to original
    lmw     r21,UW_copy(r1)
    lwz     r4,UWP(r1)
    stmw    r21,0(r4)

    lwz     r5,save_ret(r1)
    mtlr    r5                  ; Set up return address
    lmw     r13,save_regs(r1)
    la      r1,var_size(r1)
 
    blr

