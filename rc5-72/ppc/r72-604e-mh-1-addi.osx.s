; Copyright distributed.net 1997-2003 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; RC5-72 core for PowerPC 604e
; Written by Malcolm Howell <coreblimey@rottingscorpion.com>
; 24th January 2003
; Optimized for 604e by Roberto Ragusa <r.ragusa@libero.it>
; 23th March 2003
;
; Based in part on "Lintilla" core for RC5-64 by Dan Oetting
;
; $Id: r72-604e-mh-1-addi.osx.s,v 1.2 2003/09/12 23:08:52 mweiser Exp $
;
; $Log: r72-604e-mh-1-addi.osx.s,v $
; Revision 1.2  2003/09/12 23:08:52  mweiser
; add new files from release-2-90xx
;
; Revision 1.1.2.2  2003/05/09 12:34:41  mfeiri
; Fixed various issues that Apples gas couldnt handle
;
; Revision 1.1.2.1  2003/04/03 22:24:00  oliver
; new core from Malcolm Howell (1-pipe lintilla-alike), with a 604e
; optimized version from Roberto Ragusa - fastest cores in testing on
; everything except a G4.
;

gcc2_compiled.:

	.text

	.globl     _rc5_72_unit_func_mh604e_addi 

; Register aliases

;;;; DELETED

; Stack variables (and constants)
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
.set    mr7,mS00 + 28
.set    mr8,mS00 + 32
.set    mr9,mS00 + 36
.set    mr10,mS00 + 40
.set    mr11,mS00 + 44
.set    mr12,mS00 + 48
.set    mr13,mS00 + 52
.set    mr14,mS00 + 56
.set    mS03n,mS00 + 60
.set    mS04n,mS00 + 64
.set    mS05n,mS00 + 68
.set    mS06n,mS00 + 72
.set    UW_copy,mS06n + 4
.set    save_regs,UW_copy + 44
.set    var_size,save_regs + 76

; Constants
.set    P,0xb7e15163
.set    Q,0x9e3779b9
.set    P0QR3,0xbf0a8b1d    ; P<<3
.set    PR3Q,0x15235639     ; P0QR3 + P + Q
.set    P2Q,0xf45044d5      ; P+2Q
.set    P3Q,0x9287be8e      ; P+3Q
.set    P4Q,0x30bf3847      ; P+4Q
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

; ---- Main function: r72_unit_func_mh603e ----
; Arguments: r3 = RC5_72UnitWork *
;            r4 = u32 * iterations
;           (r5 = void * r25blk -- unused)

_rc5_72_unit_func_mh604e_addi:
    stwu    r1,-var_size(r1)
    stmw    r13,save_regs(r1)

    ; Save parameters
    stw     r3,UWP(r1)
    stw     r4,ILCP(r1)

    ; Copy UnitWork onto stack
    lmw     r21,0(r3)
    stmw    r21,UW_copy(r1)

    lwz     r6,0(r4)    ; Get initial loop count
    cmplwi  r6,0        ; Make sure it's not zero!
    beq     nothing_found

    ; Do first key round 1 & store constants and LUT as we go
    lis     r5,hi16(P)
    lwz     r26,L0_lo(r1)
    ori     r5,r5,lo16(P)
    lis     r30,hi16(Q)
    lwz     r27,L0_mid(r1)
    ori     r30,r30,lo16(Q)

    add     r29,r5,r30        ; r29 = P + Q
    rotlwi  r5,r5,3

    stw     r5,mP0QR3(r1)

    add     r26,r26,r5
    rotlw   r26,r26,r5

    add     r5,r5,r26          ; Round 1.1
    stw     r26,L0X(r1)

    add     r5,r5,r29
    rotlwi  r5,r5,3

    add     r0,r5,r26
    stw     r5,S1X(r1)

    add     r29,r29,r30
    add     r27,r27,r0
    stw     r0,L0XS1X(r1)
    rotlw   r27,r27,r0

    add     r5,r5,r29         ; Round 1.2
    stw     r27,L1X(r1)

    stw     r5,S1XP2Q(r1)
    add     r5,r5,r27
    lwz     r31,L0_hi(r1)   ; Need to start main loop with r31 = key_hi
    rotlwi  r5,r5,3

    stw     r5,S2X(r1)

    add     r0,r5,r27
    add     r29,r29,r30
    add     r28,r31,r0
    stw     r0,L1XS2X(r1)
    rotlw   r28,r28,r0

    add     r5,r5,r29         ; Round 1.3
    stw     r5,S2XP3Q(r1)
    add     r5,r5,r28
    rotlwi  r5,r5,3

    add     r0,r5,r28
    stw     r5,mS03n(r1)
    add     r26,r26,r0
    add     r29,r29,r30
    rotlw   r26,r26,r0

    add     r5,r5,r29         ; Round 1.4
    add     r5,r5,r26
    rotlwi  r5,r5,3

    add     r0,r5,r26
    stw     r5,mS04n(r1)
    add     r27,r27,r0
    add     r29,r29,r30
    rotlw   r27,r27,r0

    add     r5,r5,r29         ; Round 1.5
    add     r5,r5,r27
    rotlwi  r5,r5,3

    add     r0,r5,r27
    stw     r5,mS05n(r1)
    add     r28,r28,r0
    add     r29,r29,r30
    rotlw   r28,r28,r0

    add     r5,r5,r29         ; Round 1.6
    add     r5,r5,r28
    rotlwi  r5,r5,3

    add     r0,r5,r28
    stw     r5,mS06n(r1)
    add     r26,r26,r0
    add     r29,r29,r30
    rotlw   r26,r26,r0

    add     r5,r5,r29         ; Round 1.7
    add     r5,r5,r26
    rotlwi  r7,r5,3

    add     r0,r7,r26
    add     r27,r27,r0
    add     r29,r29,r30
    rotlw   r27,r27,r0

    add     r8,r7,r29         ; Round 1.8
    add     r8,r8,r27
    rotlwi  r8,r8,3

    add     r0,r8,r27
    add     r28,r28,r0
    add     r29,r29,r30
    rotlw   r28,r28,r0

    add     r9,r8,r29         ; Round 1.9
    add     r9,r9,r28
    rotlwi  r9,r9,3

    add     r0,r9,r28
    add     r26,r26,r0
    add     r29,r29,r30
    rotlw   r26,r26,r0

    add     r10,r9,r29         ; Round 1.10
    add     r10,r10,r26
    rotlwi  r10,r10,3

    add     r0,r10,r26
    add     r27,r27,r0
    add     r29,r29,r30
    rotlw   r27,r27,r0

    add     r11,r10,r29         ; Round 1.11
    add     r11,r11,r27
    rotlwi  r11,r11,3

    add     r0,r11,r27
    add     r28,r28,r0
    add     r29,r29,r30
    rotlw   r28,r28,r0

    add     r12,r11,r29         ; Round 1.12
    add     r12,r12,r28
    rotlwi  r12,r12,3

    add     r0,r12,r28
    add     r26,r26,r0
    add     r29,r29,r30
    rotlw   r26,r26,r0

    add     r13,r12,r29         ; Round 1.13
    add     r13,r13,r26
    rotlwi  r13,r13,3

    add     r0,r13,r26
    add     r27,r27,r0
    add     r29,r29,r30
    rotlw   r27,r27,r0

    add     r14,r13,r29         ; Round 1.14
    add     r14,r14,r27
    rotlwi  r14,r14,3

    add     r0,r14,r27
    add     r28,r28,r0
    add     r29,r29,r30
    rotlw   r28,r28,r0

    add     r15,r14,r29         ; Round 1.15
    add     r15,r15,r28
    rotlwi  r15,r15,3

    add     r0,r15,r28
    add     r26,r26,r0
    add     r29,r29,r30
    rotlw   r26,r26,r0

    add     r16,r15,r29         ; Round 1.16
    add     r16,r16,r26
    rotlwi  r16,r16,3

    add     r0,r16,r26
    add     r27,r27,r0
    add     r29,r29,r30
    rotlw   r27,r27,r0

    add     r17,r16,r29         ; Round 1.17
    add     r17,r17,r27
    rotlwi  r17,r17,3

    add     r0,r17,r27
    add     r28,r28,r0
    add     r29,r29,r30
    rotlw   r28,r28,r0

    add     r18,r17,r29         ; Round 1.18
    add     r18,r18,r28
    rotlwi  r18,r18,3

    add     r0,r18,r28
    add     r26,r26,r0
    add     r29,r29,r30
    rotlw   r26,r26,r0

    add     r19,r18,r29         ; Round 1.19
    add     r19,r19,r26
    rotlwi  r19,r19,3

    add     r0,r19,r26
    add     r27,r27,r0
    add     r29,r29,r30
    rotlw   r27,r27,r0

    add     r20,r19,r29         ; Round 1.20
    add     r20,r20,r27
    rotlwi  r20,r20,3

    add     r0,r20,r27
    add     r28,r28,r0
    add     r29,r29,r30
    rotlw   r28,r28,r0

    add     r21,r20,r29         ; Round 1.21
    add     r21,r21,r28
    rotlwi  r21,r21,3

    add     r0,r21,r28
    add     r26,r26,r0
    add     r29,r29,r30
    rotlw   r26,r26,r0

    add     r22,r21,r29         ; Round 1.22
    add     r22,r22,r26
    rotlwi  r22,r22,3

    add     r0,r22,r26
    add     r27,r27,r0
    add     r29,r29,r30
    rotlw   r27,r27,r0

    add     r23,r22,r29         ; Round 1.23
    add     r23,r23,r27
    rotlwi  r23,r23,3

    add     r0,r23,r27
    add     r28,r28,r0
    add     r29,r29,r30
    rotlw   r28,r28,r0

    add     r24,r23,r29         ; Round 1.24
    add     r24,r24,r28
    rotlwi  r24,r24,3

    add     r0,r24,r28
    add     r26,r26,r0
    add     r29,r29,r30
    rotlw   r26,r26,r0

    add     r25,r24,r29         ; Round 1.25
    add     r25,r25,r26
    rotlwi  r25,r25,3

    lwz     r3,mP0QR3(r1)
    add     r0,r25,r26
    lwz     r4,S1X(r1)
    add     r27,r27,r0
    lwz     r5,S2X(r1)
    rotlw   r27,r27,r0

    b       outerloop

    .align 5

outerloop:
    ; Reaching here, we must have the loop count in r6
    ; r3-r28, except r6, contain the state for the upcoming round 2
    ; r31 holds key_hi - 1, this gets incremented just before round 1
    ; r0,r29,r30 are available

    ; Work out the loop count as follows:
    ; t = min(LC, 255 - key_hi)
    ; ctr = t
    ; LC -= t

    ; Usually we come back here with key_hi = -1, so do 256 keys
    ; Potentially if the function is called with key_hi = 255 it will
    ; mess up this algorithm, but I'll rely on the outside knowledge that
    ; the starting key is always aligned to other cores' pipeline counts,
    ; currently 24 (but at the very least, it's even)

    subfic  r29,r31,255     ; 255 - key_hi, always 1 <= r29 <= 256
    cmplw   r6,r29          ; *Unsigned* comparison
    bge+    use_keyhi       ; If LC >= r29 (inner loop count), continue

    mr      r29,r6          ; Else, make inner loop count = LC

use_keyhi:
    subf    r6,r29,r6
    mtctr   r29
    stw     r6,LC(r1)
    nop                     ; Pad loop to double-word boundary on cache line

mainloop:
    ; Registers:
    ; r3-r5:    S0X=P0QR3, S1X, S2X for round 2
    ; r7-r25:   r7-r25 from round 1, ready for round 2
    ; r26-r28:  L0-r28, ready for round 2
    ; r31:      r31; contains key_hi - 1
    ; r0 is r0
    ; r6 is r6

    ; Round 2.0
        ; Key increment & Round 1.2

    add     r3,r25,r3           ; r3 was S00 = P0QR3
        lwz     r6,L1XS2X(r1)
    add     r3,r3,r27
        addi    r31,r31,1
    rotlwi  r3,r3,3             ; r3 = S00
        stw     r31,L0_hi(r1)
    add     r0,r3,r27
        add     r31,r31,r6
        rotlw   r31,r31,r6
    add     r28,r28,r0

    rotlw   r28,r28,r0
    add     r4,r3,r4            ; Round 2.1
    add     r4,r4,r28
        lwz     r30,S2XP3Q(r1)  ; Round 1.3
    rotlwi  r4,r4,3
    stw     r3,mS00(r1)
        add     r30,r30,r31     ; Use r30 (r30) to find S03n
    add     r0,r4,r28
    add     r26,r26,r0
    lwz     r3,mS03n(r1)        ; Get S03 from previous round 1

    rotlw   r26,r26,r0
    add     r5,r4,r5            ; Round 2.2 - r5 = S02
        rotlwi  r30,r30,3
    add     r5,r5,r26
    rotlwi  r5,r5,3
    stw     r4,mS01(r1)
    add     r0,r5,r26
        lwz     r29,L0X(r1)
    add     r27,r27,r0
    lwz     r4,mS04n(r1)        ; Get S04 from previous round 1

    rotlw   r27,r27,r0
    add     r3,r5,r3            ; Round 2.3 - r3 = S03
        add     r6,r30,r31
    add     r3,r3,r27
    rotlwi  r3,r3,3
        stw     r30,mS03n(r1)
    add     r0,r3,r27
    stw     r5,mS02(r1)
        add     r29,r29,r6
    add     r28,r28,r0

    rotlw   r28,r28,r0
    add     r4,r3,r4            ; Round 2.4 - r4 = S04
        rotlw   r29,r29,r6
    add     r4,r4,r28
    rotlwi  r4,r4,3
    lwz     r5,mS05n(r1)
    add     r0,r4,r28
        addis   r30,r30,ha16(P4Q)  ; Round 1.4
    add     r26,r26,r0
    stw     r3,mS03(r1)

    rotlw   r26,r26,r0
    add     r5,r4,r5            ; Round 2.5 - r5 = S05
    add     r5,r5,r26
    stw     r4,mS04(r1)
    rotlwi  r5,r5,3
    lwz     r4,mS06n(r1)
        addi     r3,r30,lo16(P4Q)   ; Now use r3 as An
    add     r0,r5,r26
        add     r3,r3,r29
    add     r27,r27,r0

    rotlw   r27,r27,r0
    add     r4,r5,r4            ; Round 2.6 - r4 = S06
        rotlwi  r3,r3,3
    add     r4,r4,r27
    rotlwi  r4,r4,3
        lwz     r30,L1X(r1)
    add     r0,r4,r27
    stw     r5,mS05(r1)
    add     r28,r28,r0
    stw     r4,mS06(r1)
        stw     r3,mS04n(r1)    

    rotlw   r28,r28,r0
    add     r7,r7,r4          ; Round 2.7 - r7 = r7 already set!
        add     r6,r3,r29
    add     r7,r7,r28
    rotlwi  r7,r7,3
        add     r30,r30,r6
    add     r0,r7,r28
        rotlw   r30,r30,r6
    add     r26,r26,r0

    rotlw   r26,r26,r0
        addis   r3,r3,ha16(P5Q)    ; Round 1.5
    add     r8,r8,r7         ; Round 2.8
        addi    r3,r3,lo16(P5Q)
    add     r8,r8,r26
    ; Stall
    rotlwi  r8,r8,3
        add     r3,r3,r30
        rotlwi  r5,r3,3         ; Keep S05n in r5 for a while...
    add     r0,r8,r26
    add     r27,r27,r0

    rotlw   r27,r27,r0
    add     r9,r9,r8         ; Round 2.9
        add     r6,r5,r30
    add     r9,r9,r27
    rotlwi  r9,r9,3
        add     r31,r31,r6
        rotlw   r31,r31,r6
    add     r0,r9,r27
    add     r28,r28,r0
        addis   r3,r5,ha16(P6Q)    ; Round 1.6

    rotlw   r28,r28,r0
    add     r10,r10,r9         ; Round 2.10
        add     r3,r3,r31
    add     r10,r10,r28
    rotlwi  r10,r10,3
        addi    r3,r3,lo16(P6Q)
        rotlwi  r3,r3,3
    add     r0,r10,r28
    add     r26,r26,r0

    rotlw   r26,r26,r0
    add     r11,r11,r10         ; Round 2.11
        add     r6,r3,r31
    add     r11,r11,r26
    rotlwi  r11,r11,3
        add     r29,r29,r6
        rotlw   r29,r29,r6
    add     r0,r11,r26
    add     r27,r27,r0
        stw     r3,mS06n(r1)
    stw     r7,mr7(r1)
    stw     r8,mr8(r1)
    stw     r9,mr9(r1)
    stw     r10,mr10(r1)
        addis   r7,r3,ha16(P7Q)   ; Round 1.7

    rotlw   r27,r27,r0
    add     r12,r12,r11         ; Round 2.12
        add     r7,r7,r29
    add     r12,r12,r27
    rotlwi  r12,r12,3
        addi    r7,r7,lo16(P7Q)
        rotlwi  r7,r7,3
    add     r0,r12,r27
    add     r28,r28,r0

    rotlw   r28,r28,r0
    add     r13,r13,r12         ; Round 2.13
        add     r6,r7,r29
    add     r13,r13,r28
    rotlwi  r13,r13,3
        add     r30,r30,r6
        rotlw   r30,r30,r6
    add     r0,r13,r28
    add     r26,r26,r0
        addis   r8,r7,ha16(P8Q)    ; Round 1.8

    rotlw   r26,r26,r0
    add     r14,r14,r13         ; Round 2.14
        add     r8,r8,r30
    add     r14,r14,r26
    rotlwi  r14,r14,3
        addi    r8,r8,lo16(P8Q)
        rotlwi  r8,r8,3
    add     r0,r14,r26
    add     r27,r27,r0

    rotlw   r27,r27,r0
    add     r15,r15,r14         ; Round 2.15
        add     r6,r8,r30
    add     r15,r15,r27
    rotlwi  r15,r15,3
        add     r31,r31,r6
        rotlw   r31,r31,r6
    add     r0,r15,r27
    add     r28,r28,r0
        addis   r9,r8,ha16(P9Q)  ; Round 1.9

    rotlw   r28,r28,r0
    add     r16,r16,r15         ; Round 2.16
        add     r9,r9,r31
    add     r16,r16,r28
    rotlwi  r16,r16,3
        addi    r9,r9,lo16(P9Q)
        rotlwi  r9,r9,3
    add     r0,r16,r28
    add     r26,r26,r0

    rotlw   r26,r26,r0
    add     r17,r17,r16         ; Round 2.17
        add     r6,r9,r31
    add     r17,r17,r26
    rotlwi  r17,r17,3
        add     r29,r29,r6
        rotlw   r29,r29,r6
    add     r0,r17,r26
    add     r27,r27,r0
        addis   r10,r9,ha16(P10Q) ; Round 1.10

    rotlw   r27,r27,r0
    add     r18,r18,r17         ; Round 2.18
        add     r10,r10,r29
    add     r18,r18,r27
    rotlwi  r18,r18,3
        addi    r10,r10,lo16(P10Q)
        rotlwi  r10,r10,3
    add     r0,r18,r27
    add     r28,r28,r0
    stw     r11,mr11(r1)
    stw     r12,mr12(r1)
    stw     r13,mr13(r1)
    stw     r14,mr14(r1)
        stw   r5,mS05n(r1)      ; Left over from way back

    rotlw   r28,r28,r0
    add     r19,r19,r18         ; Round 2.19
        add     r6,r10,r29
    add     r19,r19,r28
    rotlwi  r19,r19,3
        add     r30,r30,r6
        rotlw   r30,r30,r6
    add     r0,r19,r28
    add     r26,r26,r0
        addis   r11,r10,ha16(P11Q) ; Round 1.11

    rotlw   r26,r26,r0
    add     r20,r20,r19         ; Round 2.20
        add     r11,r11,r30
    add     r20,r20,r26
    rotlwi  r20,r20,3
        addi    r11,r11,lo16(P11Q)
        rotlwi  r11,r11,3
    add     r0,r20,r26
    add     r27,r27,r0

    rotlw   r27,r27,r0
    add     r21,r21,r20         ; Round 2.21
        add     r6,r11,r30
    add     r21,r21,r27
    rotlwi  r21,r21,3
        add     r31,r31,r6
        rotlw   r31,r31,r6
    add     r0,r21,r27
    add     r28,r28,r0
        addis   r12,r11,ha16(P12Q) ; Round 1.12

    rotlw   r28,r28,r0
    add     r22,r22,r21         ; Round 2.22
        add     r12,r12,r31
    add     r22,r22,r28
    rotlwi  r22,r22,3
        addi    r12,r12,lo16(P12Q)
        rotlwi  r12,r12,3
    add     r0,r22,r28
    add     r26,r26,r0

    rotlw   r26,r26,r0
    add     r23,r23,r22         ; Round 2.23
        add     r6,r12,r31
    add     r23,r23,r26
    rotlwi  r23,r23,3
        add     r29,r29,r6
        rotlw   r29,r29,r6
    add     r0,r23,r26
    add     r27,r27,r0
        addis   r13,r12,ha16(P13Q) ; Round 1.13

    rotlw   r27,r27,r0
    add     r24,r24,r23         ; Round 2.24
        add     r13,r13,r29
    add     r24,r24,r27
    rotlwi  r24,r24,3
        addi    r13,r13,lo16(P13Q)
        rotlwi  r13,r13,3
    add     r0,r24,r27
    add     r28,r28,r0

    rotlw   r28,r28,r0
    add     r25,r25,r24         ; Round 2.25
    add     r25,r25,r28
    rotlwi  r25,r25,3
    lwz     r5,mS00(r1)          ; Note A = r5
    add     r0,r25,r28
    lwz     r3,plain_lo(r1)
        add     r6,r13,r29
    add     r26,r26,r0


    rotlw   r26,r26,r0
    add     r5,r25,r5             ; Round 3.0 - r25 finished, r25 = r25 now
        add     r30,r30,r6
    add     r5,r5,r26
    rotlwi  r5,r5,3
    lwz     r4,plain_hi(r1)
        rotlw   r30,r30,r6
    add     r0,r5,r26
    add     r27,r27,r0
    lwz     r25,mS01(r1)


    rotlw   r27,r27,r0
    add     r3,r3,r5
    add     r5,r5,r25             ; Round 3.1
        addis   r14,r13,ha16(P14Q) ; Round 1.14
    add     r5,r5,r27
        add     r14,r14,r30
    rotlwi  r5,r5,3
        addi    r14,r14,lo16(P14Q)

        rotlwi  r14,r14,3
    lwz     r25,mS02(r1)
        add     r6,r14,r30
    add     r0,r5,r27
        add     r31,r31,r6
    add     r28,r28,r0
        rotlw   r31,r31,r6        ; Finished Round 1.14
    add     r4,r4,r5

; Register recap:
; r3,r4:    r3,r4
; r5,r0:    r5,r0
; r6:       r6, available for a while.
; r7-r14:   r7-r14 from round 1. Preserve these.
; r15-r24:  r15-r24 from round 2 (r25 was discarded).
; r25:      r25, used to look up S00-r14 in round 3.
; r26-r28:  L0-L2.
; r29-r31:  r29-r31 from round 1. Preserve these.

    ; Begin round 3 repetitions
    rotlw   r28,r28,r0
    add     r5,r5,r25         ; Round 3.2
    xor     r3,r3,r4
    add     r5,r5,r28
    rotlwi  r5,r5,3
    lwz     r25,mS03(r1)
    rotlw   r3,r3,r4
    add     r0,r5,r28
    add     r3,r3,r5
    add     r26,r26,r0

    rotlw   r26,r26,r0
    add     r5,r5,r25         ; Round 3.3
    xor     r4,r4,r3
    add     r5,r5,r26
    rotlwi  r5,r5,3
    lwz     r25,mS04(r1)
    rotlw   r4,r4,r3
    add     r0,r5,r26
    add     r4,r4,r5
    add     r27,r27,r0

    rotlw   r27,r27,r0
    add     r5,r5,r25         ; Round 3.4
    xor     r3,r3,r4
    add     r5,r5,r27
    rotlwi  r5,r5,3
    lwz     r25,mS05(r1)
    rotlw   r3,r3,r4
    add     r0,r5,r27
    add     r3,r3,r5
    add     r28,r28,r0

    rotlw   r28,r28,r0
    add     r5,r5,r25         ; Round 3.5
    xor     r4,r4,r3
    add     r5,r5,r28
    rotlwi  r5,r5,3
    lwz     r25,mS06(r1)
    rotlw   r4,r4,r3
    add     r0,r5,r28
    add     r4,r4,r5
    add     r26,r26,r0

    rotlw   r26,r26,r0
    add     r5,r5,r25         ; Round 3.6
    xor     r3,r3,r4
    add     r5,r5,r26
    rotlwi  r5,r5,3
    lwz     r25,mr7(r1)
    rotlw   r3,r3,r4
    add     r0,r5,r26
    add     r3,r3,r5
    add     r27,r27,r0

    rotlw   r27,r27,r0
    add     r5,r5,r25         ; Round 3.7
    xor     r4,r4,r3
    add     r5,r5,r27
    rotlwi  r5,r5,3
    lwz     r25,mr8(r1)
    rotlw   r4,r4,r3
    add     r0,r5,r27
    add     r4,r4,r5
    add     r28,r28,r0

    rotlw   r28,r28,r0
    add     r5,r5,r25         ; Round 3.8
    xor     r3,r3,r4
    add     r5,r5,r28
    rotlwi  r5,r5,3
    lwz     r25,mr9(r1)
    rotlw   r3,r3,r4
    add     r0,r5,r28
    add     r3,r3,r5
    add     r26,r26,r0

    rotlw   r26,r26,r0
    add     r5,r5,r25         ; Round 3.9
    xor     r4,r4,r3
    add     r5,r5,r26
    rotlwi  r5,r5,3
    lwz     r25,mr10(r1)
    rotlw   r4,r4,r3
    add     r0,r5,r26
    add     r4,r4,r5
    add     r27,r27,r0

    rotlw   r27,r27,r0
    add     r5,r5,r25         ; Round 3.10
    xor     r3,r3,r4
    add     r5,r5,r27
    rotlwi  r5,r5,3
    lwz     r25,mr11(r1)
    rotlw   r3,r3,r4
    add     r0,r5,r27
    add     r3,r3,r5
    add     r28,r28,r0

    rotlw   r28,r28,r0
    add     r5,r5,r25         ; Round 3.11
    xor     r4,r4,r3
    add     r5,r5,r28
    rotlwi  r5,r5,3
    lwz     r25,mr12(r1)
    rotlw   r4,r4,r3
    add     r0,r5,r28
    add     r4,r4,r5
    add     r26,r26,r0

    rotlw   r26,r26,r0
    add     r5,r5,r25         ; Round 3.12
    xor     r3,r3,r4
    add     r5,r5,r26
    rotlwi  r5,r5,3
    lwz     r25,mr13(r1)
    rotlw   r3,r3,r4
    add     r0,r5,r26
    add     r3,r3,r5
    add     r27,r27,r0

    rotlw   r27,r27,r0
    add     r5,r5,r25         ; Round 3.13
    xor     r4,r4,r3
    add     r5,r5,r27
    rotlwi  r5,r5,3
    lwz     r25,mr14(r1)
    rotlw   r4,r4,r3
    add     r0,r5,r27
    add     r4,r4,r5
    add     r28,r28,r0

    rotlw   r28,r28,r0
    add     r5,r5,r25         ; Round 3.14
    xor     r3,r3,r4
    add     r5,r5,r28
    rotlwi  r5,r5,3
        addis     r6,r14,ha16(P15Q)  ; Round 1.15
    rotlw   r3,r3,r4
    add     r0,r5,r28
    add     r3,r3,r5
    add     r26,r26,r0

    rotlw   r26,r26,r0
        addi    r6,r6,lo16(P15Q)
    add     r5,r15,r5             ; Round 3.15
        add     r6,r6,r31
        rotlwi  r15,r6,3
    add     r5,r5,r26
    rotlwi  r5,r5,3
        add     r6,r15,r31
    xor     r4,r4,r3
        add     r29,r29,r6
        rotlw   r29,r29,r6
    add     r0,r5,r26
    rotlw   r4,r4,r3
        addis   r6,r15,ha16(P16Q)    ; Round 1.16
    add     r4,r4,r5
    add     r27,r27,r0

    rotlw   r27,r27,r0
        addi    r6,r6,lo16(P16Q)
    add     r5,r16,r5             ; Round 3.16
        add     r6,r6,r29
        rotlwi  r16,r6,3
    add     r5,r5,r27
    rotlwi  r5,r5,3
        add     r6,r16,r29
    xor     r3,r3,r4
        add     r30,r30,r6
        rotlw   r30,r30,r6
    add     r0,r5,r27
    rotlw   r3,r3,r4
        addis   r6,r16,ha16(P17Q)    ; Round 1.17
    add     r3,r3,r5
    add     r28,r28,r0

    rotlw   r28,r28,r0
        addi    r6,r6,lo16(P17Q)
    add     r5,r17,r5             ; Round 3.17
        add     r6,r6,r30
        rotlwi  r17,r6,3
    add     r5,r5,r28
    rotlwi  r5,r5,3
        add     r6,r17,r30
    xor     r4,r4,r3
        add     r31,r31,r6
        rotlw   r31,r31,r6
    add     r0,r5,r28
    rotlw   r4,r4,r3
        addis   r6,r17,ha16(P18Q)    ; Round 1.18
    add     r4,r4,r5
    add     r26,r26,r0

    rotlw   r26,r26,r0
        addi    r6,r6,lo16(P18Q)
    add     r5,r18,r5             ; Round 3.18
        add     r6,r6,r31
        rotlwi  r18,r6,3
    add     r5,r5,r26
    rotlwi  r5,r5,3
        add     r6,r18,r31
    xor     r3,r3,r4
        add     r29,r29,r6
        rotlw   r29,r29,r6
    add     r0,r5,r26
    rotlw   r3,r3,r4
        addis   r6,r18,ha16(P19Q)    ; Round 1.19
    add     r3,r3,r5
    add     r27,r27,r0

    rotlw   r27,r27,r0
        addi    r6,r6,lo16(P19Q)
    add     r5,r19,r5             ; Round 3.19
        add     r6,r6,r29
        rotlwi  r19,r6,3
    add     r5,r5,r27
    rotlwi  r5,r5,3
        add     r6,r19,r29
    xor     r4,r4,r3
        add     r30,r30,r6
        rotlw   r30,r30,r6
    add     r0,r5,r27
    rotlw   r4,r4,r3
        addis   r6,r19,ha16(P20Q)   ; Round 1.20
    add     r4,r4,r5
    add     r28,r28,r0

    rotlw   r28,r28,r0
        addi    r6,r6,lo16(P20Q)
    add     r5,r20,r5             ; Round 3.20
        add     r6,r6,r30
        rotlwi  r20,r6,3
    add     r5,r5,r28
    rotlwi  r5,r5,3
        add     r6,r20,r30
    xor     r3,r3,r4
        add     r31,r31,r6
        rotlw   r31,r31,r6
    add     r0,r5,r28
    rotlw   r3,r3,r4
        addis   r6,r20,ha16(P21Q)   ; Round 1.21
    add     r3,r3,r5
    add     r26,r26,r0

    rotlw   r26,r26,r0
        addi    r6,r6,lo16(P21Q)
    add     r5,r21,r5             ; Round 3.21
        add     r6,r6,r31
        rotlwi  r21,r6,3
    add     r5,r5,r26
    rotlwi  r5,r5,3
        add     r6,r21,r31
    xor     r4,r4,r3
        add     r29,r29,r6
        rotlw   r29,r29,r6
    add     r0,r5,r26
    rotlw   r4,r4,r3
        addis   r6,r21,ha16(P22Q)    ; Round 1.22
    add     r4,r4,r5
    add     r27,r27,r0

    rotlw   r27,r27,r0
        addi    r6,r6,lo16(P22Q)
    add     r5,r22,r5             ; Round 3.22
        add     r6,r6,r29
        rotlwi  r22,r6,3
    add     r5,r5,r27
    rotlwi  r5,r5,3
        add     r6,r22,r29
    xor     r3,r3,r4
        add     r30,r30,r6
        rotlw   r30,r30,r6
    add     r0,r5,r27
    rotlw   r3,r3,r4
        addis   r6,r22,ha16(P23Q)    ; Round 1.23
    add     r3,r3,r5
    add     r28,r28,r0

    rotlw   r28,r28,r0
        addi    r6,r6,lo16(P23Q)
    add     r5,r23,r5             ; Round 3.23
        add     r6,r6,r30
        rotlwi  r23,r6,3
    add     r5,r5,r28
    rotlwi  r5,r5,3
        add     r6,r23,r30
    xor     r4,r4,r3
        add     r31,r31,r6
    add     r0,r5,r28
        rotlw   r28,r31,r6         ; Transfer r31 into r28
    rotlw   r4,r4,r3
        addis   r6,r23,ha16(P24Q)    ; Round 1.24
    add     r4,r4,r5
    add     r26,r26,r0

    rotlw   r26,r26,r0
        addi    r6,r6,lo16(P24Q)
    add     r5,r24,r5             ; Round 3.24
        add     r6,r6,r28    ; r31 already transferred to r28
        rotlwi  r24,r6,3
    add     r5,r5,r26              ; Finished with r26
    rotlwi  r5,r5,3
        add     r6,r24,r28     
    xor     r3,r3,r4
        add     r26,r29,r6     ; Transfer r29 into r26
 
        rotlw   r26,r26,r6
        addis   r25,r24,ha16(P25Q)
        addi    r25,r25,lo16(P25Q)
    lwz     r0,cypher_lo(r1)
    rotlw   r3,r3,r4
        add     r25,r25,r26
        rotlwi  r25,r25,3
    add     r3,r3,r5             ; And that's all for round 3

    ; Get set up for Round 2
    cmplw   r0,r3
    lwz     r3,mP0QR3(r1)
        add     r6,r25,r26
        lwz     r31,L0_hi(r1)
        add     r27,r30,r6     ; Transfer r30 into r27
    lwz     r4,S1X(r1)
        rotlw   r27,r27,r6
    lwz     r5,S2X(r1)

    bdnzf+  eq,mainloop

    ; Reaching here, some of the following are true:
    ; r3 = cypher_lo so we must check r4 (this is iff CR0:eq is set);
    ; ctr has reached zero and LC has too: time to exit;
    ; Most likely: ctr == 0, LC > 0: redo constants and return to outer loop

    ; Registers:
    ; r7-r25 (r7-r25) are holding round 1 values and must be preserved
    ; r3-r5 are holding S00-S02 for round 2, so should be preserved
    ;       (but could be reloaded)
    ; r26-r28 are holding L0-r28 and must be preserved
    ; r31 holds L0_hi, which may be reloaded
    ; r0=r0,r6=r6,r29-r30 are free

    lwz     r6,LC(r1)

    beq-    keycheck
    ; If cypher_hi also matches, or ctr != 0, keycheck doesn't return here

    ; Return here if keycheck finds ctr is 0

ctrzero:
    ; As cypher_lo didn't match, now know that ctr is 0
    ; After this point, we don't need to preserve r31 as it will be
    ; set to -1 before returning to outerloop

    li      r0,L0_mid
    cmplwi  r6,0            ; Is overall loop count == 0?
    lwbrx   r30,r1,r0      ; Get byte-reversed key_mid

    beq-    nothing_found   ; Loop count 0 -> time to leave
    ; (The key in UnitWork is the one which just did round 1.
    ;  This work is wasted and the same key is used to begin the next
    ;  function call, hence, no adjustments needed.)

    ; Now we mangle increment key_mid and redo constants
    ; Leave -1 (0xffffffff) in key_hi, so it increments to 0 on next loop

    addic.  r30,r30,1
    stwbrx  r30,r1,r0
    bne+    l1calc          ; Not zero -> L0_mid didn't carry -> do l1calc

    ; Otherwise, increment r26 too (in practice, I don't think this happens)
    li      r0,L0_lo
    lwbrx   r30,r1,r0
    addi    r30,r30,1
    stwbrx  r30,r1,r0

    ; Calculate L0- and L1-constants
    ; (Uses regs r0, r29-r31)
    ; Note that we store r30=L0.lo byte-reversed but read it back normally
    ; DO NOT "optimise" this by removing the lwz!
    ; (By all means optimise it by doing a fast byte-reversal in registers.)

    lwz     r30,L0_lo(r1)   ; r30=L0.lo=r26
    lwz     r31,mP0QR3(r1)  ; r31=P0QR3

    add     r30,r30,r31

    rotlwi  r30,r30,29      ; r30=(L0+P0QR3)<<(P0QR3)=L0X

    addi    r31,r30,lo16(PR3Q)
    stw     r30,L0X(r1)

    addis   r31,r31,ha16(PR3Q)
    rotlwi  r31,r31,3       ; r31=S1X

    add     r29,r30,r31     ; r29=L0XS1X
    stw     r31,S1X(r1)
    addi    r30,r31,lo16(P2Q)
    stw     r29,L0XS1X(r1)

    addis   r30,r30,ha16(P2Q)  ; r30=S1XP2Q
    stw     r30,S1XP2Q(r1)

    ; L1-constants
    ; Note this is usually reached after just incrementing L0_mid,
    ; so don't assume anything about regs from the r26 code above

l1calc:
    lwz     r29,L0XS1X(r1)
    lwz     r31,L0_mid(r1)
    lwz     r30,S1XP2Q(r1)
    add     r31,r31,r29
    rotlw   r31,r31,r29     ; r31=L1X

    add     r30,r30,r31
    stw     r31,L1X(r1)

    rotlwi  r30,r30,3       ; r30=S2X
    add     r31,r30,r31     ; r31=L1XS2X
    stw     r30,S2X(r1)

    addi    r30,r30,lo16(P3Q)
    stw     r31,L1XS2X(r1)

    addis   r30,r30,ha16(P3Q)  ; r30=S2XP3Q
    stw     r30,S2XP3Q(r1)

    li      r31,-1      ; Leave -1 in key_hi

    b       outerloop

nothing_found:
    li      r3,RESULT_NOTHING

ruf_exit:
    ; Restore stack copy of UnitWork to original
    ; Skip first 4 words as plaintext, cyphertext never change
    lwz     r5,UWP(r1)
    lmw     r25,UW_copy + 16(r1)
    stmw    r25,16(r5)

    lmw     r13,save_regs(r1)
    la      r1,var_size(r1)
    blr

keycheck:
    ; This gets called whenever r3 = cypher_lo
    ; It updates the RC5_72UnitWork.check fields, then redoes the key to find
    ; r4 (yes, this core contains 2 separate checking routines)
    ; If r4 matches cypher_hi, exit with success (filling in the iter count)
    ; If not, check the ctr value
    ; If ctr != 0, go back to start of mainloop
    ; If ctr == 0, jump to ctrzero (as if r3 hadn't matched)

; keycheck uses 236 bytes of stack:
; 4 bytes = old SP
; 204 bytes = 26 + 25 words of S
; 28 bytes = 7 preserved registers

.set    kcstack,236
.set    kcsave_regs,208

    ; Preserve registers (we only use 12, plus one to preserve ctr,
    ; and r0, r3-r5,r29,r30 don't need preserving)
    stwu    r1,-kcstack(r1)
    stw     r21,kcsave_regs(r1)     ; ctr (preserved)
    stw     r22,kcsave_regs+4(r1)   ; loop counts for keycheck
    stw     r23,kcsave_regs+8(r1)   ; r23
    stw     r24,kcsave_regs+12(r1)  ; r24
    stw     r26,kcsave_regs+16(r1)  ; r26
    stw     r27,kcsave_regs+20(r1)  ; r27
    stw     r28,kcsave_regs+24(r1)  ; r28
    mfctr   r21

    la      r24,0(r1)    ; Set up pointers to S (actually 1 word before start)
    mr      r23,r24

    lis     r29,hi16(P2Q)   ; Init P+nQ (to P+2Q) and Q
    ori     r29,r29,lo16(P2Q)
    lis     r30,hi16(Q)
    ori     r30,r30,lo16(Q)

    ; Fill in check fields of RC5_72UnitWork
    ; Need to mangle-decrement key for check fields and extended test

    li      r3,kcstack + L0_lo
    lwbrx   r26,r3,r1
    li      r4,kcstack + L0_mid
    lwbrx   r27,r4,r1
    lwz     r28,kcstack + L0_hi(r1)

    cmplwi  r28,0
    subi    r28,r28,1
    bne     check_store

    ; If r28 was 0 before decrement, it carried.
    ; So, clean it up and decrement L0_mid
    andi.   r28,r28,0xff

    cmplwi  r27,0
    subi    r27,r27,1
    bne     check_store

    subi    r26,r26,1

check_store:
    li      r3,kcstack + check_lo
    stwbrx  r26,r3,r1
    li      r4,kcstack + check_mid
    stwbrx  r27,r4,r1
    stw     r28,kcstack + check_hi(r1)

    lwz     r22,kcstack + check_count(r1)
    addi    r22,r22,1
    stw     r22,kcstack + check_count(r1)

    ; Refetch correct key in appropriate byte order
    lwz     r26,kcstack + check_lo(r1)
    lwz     r27,kcstack + check_mid(r1)
    lwz     r28,kcstack + check_hi(r1)

    ; Round 1, Iteration 0, second half
    lwz     r0,kcstack + mP0QR3(r1)  ; r0=P0QR3, the value of A and S[0] too
    add     r26,r26,r0
    rotlw   r26,r26,r0

    ; Iteration 1
    lis     r5,hi16(PR3Q)        ; Init A to P<<3 + P + Q
    ori     r5,r5,lo16(PR3Q)

    add     r5,r5,r26
    rotlwi  r5,r5,3
    stwu    r5,4(r23)

    add     r0,r5,r26
    add     r27,r27,r0
    rotlw   r27,r27,r0

    ; Iter 2
    ; Now r5,L0-2 are initialised, we can jump into loop

    li      r22,8
    mtctr   r22     ; Set loop counter

    b       round1entry

round1loop:

    ; Iters 2,5,8,11,14,17,20,23
    add     r29,r29,r30

round1entry:    ; During iteration 2
    add     r5,r5,r29
    add     r5,r5,r27
    rotlwi  r5,r5,3
    stwu    r5,4(r23)

    add     r0,r5,r27
    add     r28,r28,r0
    rotlw   r28,r28,r0

    ; Iteration 3,6,9,12,15,18,21,24
    add     r29,r29,r30

    add     r5,r5,r28
    add     r5,r5,r29
    rotlwi  r5,r5,3
    stwu    r5,4(r23)

    add     r0,r5,r28
    add     r26,r26,r0
    rotlw   r26,r26,r0

    ;Iteration 4,7,10,13,16,19,22,25
    add     r29,r29,r30

    add     r5,r5,r29
    add     r5,r5,r26
    rotlwi  r5,r5,3
    stwu    r5,4(r23)

    add     r0,r5,r26
    add     r27,r27,r0
    rotlw   r27,r27,r0

    bdnz    round1loop

    ; --- Round 2 ---

    ; Iter 0 - S[0] is not in r25ory; it is always P0QR3

    lwz     r0,kcstack + mP0QR3(r1)

    ; Jump into loop
    li      r22,9
    mtctr   r22

    b       round2entry

round2loop:
    stwu    r5,4(r23)

    add     r0,r5,r26
    add     r27,r27,r0
    rotlw   r27,r27,r0

    ; Iter 0,3,6,9,12,15,18,21,24
    lwzu    r0,4(r24)

round2entry:
    add     r5,r5,r27
    add     r5,r5,r0
    rotlwi  r5,r5,3
    stwu    r5,4(r23)

    add     r0,r5,r27
    add     r28,r28,r0
    rotlw   r28,r28,r0

    ; Iter 1,4,7,10,13,16,19,22,25
    lwzu    r0,4(r24)
    add     r5,r5,r28
    add     r5,r5,r0
    rotlwi  r5,r5,3
    stwu    r5,4(r23)

    add     r0,r5,r28
    add     r26,r26,r0
    rotlw   r26,r26,r0

    ; Iter 2,5,8,11,14,17,20,23 and round 3 iter 0
    lwzu    r0,4(r24)
    add     r5,r5,r26
    add     r5,r5,r0
    rotlwi  r5,r5,3

    bdnz    round2loop

    ; --- Round 3 / Encryption ---
    lwz     r3,kcstack + plain_lo(r1)
    lwz     r4,kcstack + plain_hi(r1)

    add     r3,r3,r5

    add     r0,r5,r26
    add     r27,r27,r0
    rotlw   r27,r27,r0

    ; Iter 1

    lwzu    r0,4(r24)
    add     r5,r5,r27
    add     r5,r5,r0
    rotlwi  r5,r5,3

    add     r4,r4,r5

    li      r22,4
    mtctr   r22

round3loop:
    add     r0,r5,r27
    add     r28,r28,r0
    rotlw   r28,r28,r0

    ; Iter 2,8,14,20
    lwzu    r0,4(r24)
    add     r5,r5,r28
    xor     r3,r3,r4
    add     r5,r5,r0
    rotlwi  r5,r5,3
    rotlw   r3,r3,r4
    add     r3,r3,r5

    add     r0,r5,r28
    add     r26,r26,r0
    rotlw   r26,r26,r0

    ; Iter 3,9,15,21
    lwzu    r0,4(r24)
    add     r5,r5,r26
    xor     r4,r4,r3
    add     r5,r5,r0
    rotlwi  r5,r5,3
    rotlw   r4,r4,r3
    add     r4,r4,r5

    add     r0,r5,r26
    add     r27,r27,r0
    rotlw   r27,r27,r0

    ; Iter 4,10,16,22
    lwzu    r0,4(r24)
    add     r5,r5,r27
    xor     r3,r3,r4
    add     r5,r5,r0
    rotlwi  r5,r5,3
    rotlw   r3,r3,r4
    add     r3,r3,r5

    add     r0,r5,r27
    add     r28,r28,r0
    rotlw   r28,r28,r0

    ; Iter 5,11,17,23
    lwzu    r0,4(r24)
    add     r5,r5,r28
    xor     r4,r4,r3
    add     r5,r5,r0
    rotlwi  r5,r5,3
    rotlw   r4,r4,r3
    add     r4,r4,r5

    add     r0,r5,r28
    add     r26,r26,r0
    rotlw   r26,r26,r0

    ; Iter 6,12,18,24
    lwzu    r0,4(r24)
    add     r5,r5,r26
    xor     r3,r3,r4
    add     r5,r5,r0
    rotlwi  r5,r5,3
    rotlw   r3,r3,r4
    add     r3,r3,r5

    add     r0,r5,r26
    add     r27,r27,r0
    rotlw   r27,r27,r0

    ; Iter 7,13,19,25
    lwzu    r0,4(r24)
    add     r5,r5,r27
    xor     r4,r4,r3
    add     r5,r5,r0
    rotlwi  r5,r5,3
    rotlw   r4,r4,r3
    add     r4,r4,r5

    bdnz    round3loop

    ; Now got ciphertext in r3, r4
    ; Actually, we already know that r3 matches, so just check r4
    lwz     r27,kcstack + cypher_hi(r1)

    cmplw   r4,r27

    beq     success_exit

    ; If they didn't match, we need to return to the main loop
    ; Exactly where we return depends on whether ctr == 0

    cmplwi  r21,0       ; ctr was preserved in r21

    ; Restore regs
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

    bne+    mainloop    ; If ctr != 0, go back to start of mainloop

    b       ctrzero     ; If ctr is 0, go to handling code after loop

success_exit:
    ; This is where both words of cyphertext match
    ; Need to fill in the iteration count, straighten out the stack pointer,
    ; and return success.

    la      r1,kcstack(r1)      ; Return SP to normal place

    lwz     r4,ILCP(r1)
    lwz     r7,0(r4)            ; Get initial iter count in r7

    ; LC is in r6 already, and ctr is in r21

    subi    r7,r7,1             ; Final value would be 1 too high after bdnzf
    add     r6,r6,r21           ; Loops still to do = LC + ctr

    subf    r7,r6,r7            ; r7 is now loops completed before success
    stw     r7,0(r4)

    li      r3,RESULT_FOUND
    b       ruf_exit
