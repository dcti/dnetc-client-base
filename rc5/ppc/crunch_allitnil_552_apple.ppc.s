;
; $Log: crunch_allitnil_552_apple.ppc.s,v $
; Revision 1.1.2.1  2000/06/13 01:28:09  mfeiri
; A version of allitnil that is fully compatible with Apples version of gas. Thanks a lot to pmack for his help!
;
; Revision 1.3.2.2  2000/01/03 14:34:00  patrick
;
; GCC on AIX needs the ;defines. Please keep em.
;
; Revision 1.3.2.1  1999/12/06 09:56:55  myshkin
; Minor changes to conform to gas syntr30 (version 980114).
;
; Revision 1.3  1999/04/08 18:46:23  patrick
;
; bug fixes (double rgister usage and other stuff). Did this ever compile ?
;
; Revision 1.2  1998/06/14 10:30:30  friedbait
; 'Log' keyword added.
;
;
; allitnil -- the 601 companion for lintilla
 .file	"crunch-ppc.cpp"
gcc2_compiled.:
; .csect	.text[PR]
; .align 8
;if (CLIENT_OS == OS_AIX)
 .globl .crunch_allitnil
;else
 .globl _crunch_allitnil
;endif
; .type  crunch_allitnil,@function

; strategy: straight code in inner loop, 256 key cycle, load S0_n from
; pre-calculated constants
;			runtime calculation of round 1 constants for key.lo
;				r0	r1	r2	r3	r4	...	r24	r25	r26	r27	r28	r29	r30	r31
; registers:	r0	r1	r2	r3	r4	...	24	25	r26	r27	r28 --	r30	r31
;							r3	r4	r6	r7	r9	r10	r11	r30	r31

;if (CLIENT_OS == OS_AIX)
.crunch_allitnil:
;else
_crunch_allitnil:
;endif

.set plain_hi,0     ; plaintext, already mixed with IV
.set plain_lo,4
.set cypher_hi,8    ; target cyphertext
.set cypher_lo,12
.set L0_hi,16       ; key, changes with every unit * PIPELINE_COUNT
.set L0_lo,20

; register name aliases
.set r0, 0
.set r1, 1
.set r2, 2
.set r3, 3
.set r4, 4
.set r5, 5
.set r6, 6
.set r7, 7
.set r8, 8
.set r9, 9
.set r10, 10
.set r11, 11
.set r12, 12
.set r13, 13
.set r14, 14
.set r15, 15
.set r16, 16
.set r17, 17
.set r18, 18
.set r19, 19
.set r20, 20
.set r21, 21
.set r22, 22
.set r23, 23
.set r24, 24
.set r25, 25
.set r26, 26
.set r27, 27
.set r28, 28
.set r29, 29
.set r30, 30
.set r31, 31

.set r1, r1

.set iterations, 28
.set work_r3, 24		
.set save_RTOC, -80
.set count, -84
.set P_0, -88
.set P_1, -92
; .set C_0, -96
.set C_1, -100
.set L0_0, -104
.set L0_1, -108
.set Sr_0, -112
.set Sr_1, -116
.set con0, -120
.set con1, -124
.set con2, -128
.set S0_3, -132
.set S0_4, -136
.set S0_5, -140
.set S0_6, -144
.set S0_7, -148
.set S0_8, -152
.set S0_9, -156
.set S0_10, -160
.set S0_11, -164
.set S0_12, -168
.set S0_13, -172
.set S0_14, -176
.set S0_15, -180
.set S0_16, -184
.set S0_17, -188
.set S0_18, -192
.set S0_19, -196
.set S0_20, -200
.set S0_21, -204
.set S0_22, -208
.set S0_23, -212
.set S0_24, -216
.set S0_25, -220

.set r0, r0
.set r2, r2
.set r3, r3
.set r3, r3
.set r4, r4
.set r4, r4
.set r5, r5
;.set Lt0, r5
.set r6, r6
.set r6, r6
.set r7, r7
.set r7, r7
.set r8, r8
.set r9, r9
.set r9, r9
.set r10, r10
.set r10, r10
.set r11, r11
.set r11, r11
.set r12, r12
.set r012, r12
.set r13, r13
.set r14, r14
.set r15, r15
.set r16, r16
.set r17, r17
.set r18, r18
.set r19, r19
.set r20, r20
.set r21, r21
.set r22, r22
.set r23, r23
.set r24, r24
.set r25, r25
.set r26, r26
.set r27, r27
.set r28, r28
.set r29, r29
.set r30, r30
.set r30, r30
.set r31, r31
.set r31, r31

 stmw    r13,-76(r1)

 stw	r2,save_RTOC(r1)
 stw     r3,work_r3(r1); 24(r1)	; work
 stw     r4,iterations(r1);	28(r1)	; iterations
 stw     r4,count(r1);		-148(r1);save_iterations

; Save work parameters on the local stack
 lwz	r0,plain_hi(r3)
 stw	r0,P_1(r1)
 lwz	r0,plain_lo(r3)
 stw	r0,P_0(r1)
 lwz	r0,cypher_hi(r3)
 stw	r0,C_1(r1)
 lwz	r29,cypher_lo(r3)
;	stw	r0,r29(r1)
 lwz	r28,L0_hi(r3)
;	stw	r28,L0_1(r1)
 lwz	r11,L0_lo(r3)
;	stw	r11,L0_0(r1)

; Precompute the round 0 constants
 lis     r9,0xB7E1
 lis     r10,0x9E37
 addi    r9,r9,0x5163	;r9 = P;
 addi    r10,r10,0x79B9	;r10 = Q;
 add     r9,r9,r10
 add     r9,r9,r10
 add     r9,r9,r10
 stw	r9,S0_3(r1)
 add     r9,r9,r10
 stw	r9,S0_4(r1)
 add     r9,r9,r10
 stw	r9,S0_5(r1)
 add     r9,r9,r10
 stw	r9,S0_6(r1)
 add     r9,r9,r10
 stw	r9,S0_7(r1)
 add     r9,r9,r10
 stw	r9,S0_8(r1)
 add     r9,r9,r10
 stw	r9,S0_9(r1)
 add     r9,r9,r10
 stw	r9,S0_10(r1)
 add     r9,r9,r10
 stw	r9,S0_11(r1)
 add     r9,r9,r10
 stw	r9,S0_12(r1)
 add     r9,r9,r10
 stw	r9,S0_13(r1)
 add     r9,r9,r10
 stw	r9,S0_14(r1)
 add     r9,r9,r10
 stw	r9,S0_15(r1)
 add     r9,r9,r10
 stw	r9,S0_16(r1)
 add     r9,r9,r10
 stw	r9,S0_17(r1)
 add     r9,r9,r10
 stw	r9,S0_18(r1)
 add     r9,r9,r10
 stw	r9,S0_19(r1)
 add     r9,r9,r10
 stw	r9,S0_20(r1)
 add     r9,r9,r10
 stw	r9,S0_21(r1)
 add     r9,r9,r10
 stw	r9,S0_22(r1)
 add     r9,r9,r10
 stw	r9,S0_23(r1)
 add     r9,r9,r10
 stw	r9,S0_24(r1)
 add     r9,r9,r10
 stw	r9,S0_25(r1)

start:	; Return here to recompute the r11 invariants
 lis     r9,0xB7E1
 lis     r10,0x9E37
 addi    r9,r9,0x5163	;r9 = P;
 addi    r10,r10,0x79B9	;r10 = Q;

 lwz	r3,work_r3(r1)		; probably already loaded
 lwz	r11,L0_lo(r3)

; round 1.0 even
 rotlwi  r30,r9,3
 stw	r30,Sr_0(r1);		; Sr_0 = ...
 add     r26,r11,r30
 rotlw	r26,r26,r30
 stw	r26,con0(r1)		; con0 = Lr_0
 add     r9,r9,r10

; round 1.1 odd
 add     r0,r9,r30
 add     r0,r0,r26
 rotlwi  r31,r0,3
 stw	r31,Sr_1(r1)		; Sr_1 = rotl3(...)
 add     r012,r31,r26
 stw	r012,con1(r1)		; con1 =  Sr_1 + Lr_0
 add     r27,r28,r012
 rotlw	r27,r27,r012
 add     r9,r9,r10

; round 1.2 even
 add     r0,r9,r31
 stw	r0,con2(r1)		; con2 = S0_2 + Sr_1


reloop:
; registers: r4, r28

; compute the loop count until the next event
 srwi	r0,r28,24		; shift right logical 24, mask 0xFF
 subfic	r0,r0,256
 cmpw	r0,r4
 ble	label1
 mr	r0,r4
label1:
;// save remaining count, setup r4
 sub	r4,r4,r0
 stw	r4,count(r1)
 mtctr	r0

loop:
;// Uses saved constants

 lwz     r2,con2(r1)	;	 = S0_2 + Sr_1
; round 1.1 odd
 add     r27,r28,r012
 rotlw	r27,r27,r012

 lwz	r3,S0_3(r1)
; round 1.2 even
 add     r2,r2,r27
 rotlwi  r2,r2,3
 add     r0,r2,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r4,S0_4(r1)
; round 1.3 odd
 add     r3,r3,r2
 add     r3,r3,r26
 rotlwi  r3,r3,3
 add     r0,r3,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r5,S0_5(r1)
; round 1.4 even
 add     r4,r4,r3
 add     r4,r4,r27
 rotlwi  r4,r4,3
 add     r0,r4,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r6,S0_6(r1)
; round 1.5 odd
 add     r5,r5,r4
 add     r5,r5,r26
 rotlwi  r5,r5,3
 add     r0,r5,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r7,S0_7(r1)
; round 1.6 even
 add     r6,r6,r5
 add     r6,r6,r27
 rotlwi  r6,r6,3
 add     r0,r6,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r8,S0_8(r1)
; round 1.7 odd
 add     r7,r7,r6
 add     r7,r7,r26
 rotlwi  r7,r7,3
 add     r0,r7,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r9,S0_9(r1)
; round 1.8 even
 add     r8,r8,r7
 add     r8,r8,r27
 rotlwi  r8,r8,3
 add     r0,r8,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r10,S0_10(r1)
; round 1.9 odd
 add     r9,r9,r8
 add     r9,r9,r26
 rotlwi  r9,r9,3
 add     r0,r9,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r11,S0_11(r1)
; round 1.10 even
 add     r10,r10,r9
 add     r10,r10,r27
 rotlwi  r10,r10,3
 add     r0,r10,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r12,S0_12(r1)
; round 1.11 odd
 add     r11,r11,r10
 add     r11,r11,r26
 rotlwi  r11,r11,3
 add     r0,r11,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r13,S0_13(r1)
; round 1.12 even
 add     r12,r12,r11
 add     r12,r12,r27
 rotlwi  r12,r12,3
 add     r0,r12,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r14,S0_14(r1)
; round 1.13 odd
 add     r13,r13,r12
 add     r13,r13,r26
 rotlwi  r13,r13,3
 add     r0,r13,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r15,S0_15(r1)
; round 1.14 even
 add     r14,r14,r13
 add     r14,r14,r27
 rotlwi  r14,r14,3
 add     r0,r14,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r16,S0_16(r1)
; round 1.15 odd
 add     r15,r15,r14
 add     r15,r15,r26
 rotlwi  r15,r15,3
 add     r0,r15,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r17,S0_17(r1)
; round 1.16 even
 add     r16,r16,r15
 add     r16,r16,r27
 rotlwi  r16,r16,3
 add     r0,r16,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r18,S0_18(r1)
; round 1.17 odd
 add     r17,r17,r16
 add     r17,r17,r26
 rotlwi  r17,r17,3
 add     r0,r17,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r19,S0_19(r1)
; round 1.18 even
 add     r18,r18,r17
 add     r18,r18,r27
 rotlwi  r18,r18,3
 add     r0,r18,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r20,S0_20(r1)
; round 1.19 odd
 add     r19,r19,r18
 add     r19,r19,r26
 rotlwi  r19,r19,3
 add     r0,r19,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r21,S0_21(r1)
; round 1.20 even
 add     r20,r20,r19
 add     r20,r20,r27
 rotlwi  r20,r20,3
 add     r0,r20,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r22,S0_22(r1)
; round 1.21 odd
 add     r21,r21,r20
 add     r21,r21,r26
 rotlwi  r21,r21,3
 add     r0,r21,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r23,S0_23(r1)
; round 1.22 even
 add     r22,r22,r21
 add     r22,r22,r27
 rotlwi  r22,r22,3
 add     r0,r22,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r24,S0_24(r1)
; round 1.23 odd
 add     r23,r23,r22
 add     r23,r23,r26
 rotlwi  r23,r23,3
 add     r0,r23,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r25,S0_25(r1)
; round 1.24 even
 add     r24,r24,r23
 add     r24,r24,r27
 rotlwi  r24,r24,3
 add     r0,r24,r27
 add     r26,r26,r0
 rotlw	r26,r26,r0

 lwz	r30,Sr_0(r1)
; round 1.25 odd
 add     r25,r25,r24
 add     r25,r25,r26
 rotlwi  r25,r25,3
 add     r0,r25,r26
 add     r27,r27,r0
 rotlw	r27,r27,r0

 lwz	r31,Sr_1(r1)
; round 2.0 even
 add     r30,r30,r25
 add     r30,r30,r27
 rotlwi  r30,r30,3
 add     r0,r30,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.1 odd
 add     r31,r31,r30
 add     r31,r31,r26
 rotlwi  r31,r31,3
 add     r0,r31,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.2 even
 add     r2,r2,r31
 add     r2,r2,r27
 rotlwi  r2,r2,3
 add     r0,r2,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.3 odd
 add     r3,r3,r2
 add     r3,r3,r26
 rotlwi  r3,r3,3
 add     r0,r3,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.4 even
 add     r4,r4,r3
 add     r4,r4,r27
 rotlwi  r4,r4,3
 add     r0,r4,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.5 odd
 add     r5,r5,r4
 add     r5,r5,r26
 rotlwi  r5,r5,3
 add     r0,r5,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.6 even
 add     r6,r6,r5
 add     r6,r6,r27
 rotlwi  r6,r6,3
 add     r0,r6,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.7 odd
 add     r7,r7,r6
 add     r7,r7,r26
 rotlwi  r7,r7,3
 add     r0,r7,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.8 even
 add     r8,r8,r7
 add     r8,r8,r27
 rotlwi  r8,r8,3
 add     r0,r8,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.9 odd
 add     r9,r9,r8
 add     r9,r9,r26
 rotlwi  r9,r9,3
 add     r0,r9,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.10 even
 add     r10,r10,r9
 add     r10,r10,r27
 rotlwi  r10,r10,3
 add     r0,r10,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.11 odd
 add     r11,r11,r10
 add     r11,r11,r26
 rotlwi  r11,r11,3
 add     r0,r11,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.12 even
 add     r12,r12,r11
 add     r12,r12,r27
 rotlwi  r12,r12,3
 add     r0,r12,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.13 odd
 add     r13,r13,r12
 add     r13,r13,r26
 rotlwi  r13,r13,3
 add     r0,r13,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.14 even
 add     r14,r14,r13
 add     r14,r14,r27
 rotlwi  r14,r14,3
 add     r0,r14,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.15 odd
 add     r15,r15,r14
 add     r15,r15,r26
 rotlwi  r15,r15,3
 add     r0,r15,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.16 even
 add     r16,r16,r15
 add     r16,r16,r27
 rotlwi  r16,r16,3
 add     r0,r16,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.17 odd
 add     r17,r17,r16
 add     r17,r17,r26
 rotlwi  r17,r17,3
 add     r0,r17,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.18 even
 add     r18,r18,r17
 add     r18,r18,r27
 rotlwi  r18,r18,3
 add     r0,r18,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.19 odd
 add     r19,r19,r18
 add     r19,r19,r26
 rotlwi  r19,r19,3
 add     r0,r19,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.20 even
 add     r20,r20,r19
 add     r20,r20,r27
 rotlwi  r20,r20,3
 add     r0,r20,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.21 odd
 add     r21,r21,r20
 add     r21,r21,r26
 rotlwi  r21,r21,3
 add     r0,r21,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.22 even
 add     r22,r22,r21
 add     r22,r22,r27
 rotlwi  r22,r22,3
 add     r0,r22,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.23 odd
 add     r23,r23,r22
 add     r23,r23,r26
 rotlwi  r23,r23,3
 add     r0,r23,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; round 2.24 even
 add     r24,r24,r23
 add     r24,r24,r27
 rotlwi  r24,r24,3
 add     r0,r24,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

; round 2.25 odd
 add     r25,r25,r24
 add     r25,r25,r26
 rotlwi  r25,r25,3
 add     r0,r25,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

; registers in use: r30 ... r25, r26, r27, r28
; we might save a load if we throw away r25!

; round 3.0 even
 add     r30,r30,r25
 add     r30,r30,r27
 rotlwi  r30,r30,3
 add     r0,r30,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0

 lwz	r0,P_0(r1)
; round 3.1 odd
 add     r31,r31,r30
 add     r31,r31,r26
 rotlwi  r31,r31,3
 add	r30,r0,r30
 add     r0,r31,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0

 lwz	r0,P_1(r1)
; round 3.2 even
 add     r2,r2,r31
 add     r2,r2,r27
 rotlwi  r2,r2,3
 add	r31,r0,r31
 add     r0,r2,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r2

; round 3.3 odd
 add     r3,r3,r2
 add     r3,r3,r26
 rotlwi  r3,r3,3
 add     r0,r3,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r3

; round 3.4 even
 add     r4,r4,r3
 add     r4,r4,r27
 rotlwi  r4,r4,3
 add     r0,r4,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r4

; round 3.5 odd
 add     r5,r5,r4
 add     r5,r5,r26
 rotlwi  r5,r5,3
 add     r0,r5,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r5

; round 3.6 even
 add     r6,r6,r5
 add     r6,r6,r27
 rotlwi  r6,r6,3
 add     r0,r6,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r6

; round 3.7 odd
 add     r7,r7,r6
 add     r7,r7,r26
 rotlwi  r7,r7,3
 add     r0,r7,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r7

; round 3.8 even
 add     r8,r8,r7
 add     r8,r8,r27
 rotlwi  r8,r8,3
 add     r0,r8,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r8

; round 3.9 odd
 add     r9,r9,r8
 add     r9,r9,r26
 rotlwi  r9,r9,3
 add     r0,r9,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r9

; round 3.10 even
 add     r10,r10,r9
 add     r10,r10,r27
 rotlwi  r10,r10,3
 add     r0,r10,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r10

; round 3.11 odd
 add     r11,r11,r10
 add     r11,r11,r26
 rotlwi  r11,r11,3
 add     r0,r11,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r11

; round 3.12 even
 add     r12,r12,r11
 add     r12,r12,r27
 rotlwi  r12,r12,3
 add     r0,r12,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r12

; round 3.13 odd
 add     r13,r13,r12
 add     r13,r13,r26
 rotlwi  r13,r13,3
 add     r0,r13,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r13

; round 3.14 even
 add     r14,r14,r13
 add     r14,r14,r27
 rotlwi  r14,r14,3
 add     r0,r14,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r14

; round 3.15 odd
 add     r15,r15,r14
 add     r15,r15,r26
 rotlwi  r15,r15,3
 add     r0,r15,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r15

; round 3.16 even
 add     r16,r16,r15
 add     r16,r16,r27
 rotlwi  r16,r16,3
 add     r0,r16,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r16

; round 3.17 odd
 add     r17,r17,r16
 add     r17,r17,r26
 rotlwi  r17,r17,3
 add     r0,r17,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r17

; round 3.18 even
 add     r18,r18,r17
 add     r18,r18,r27
 rotlwi  r18,r18,3
 add     r0,r18,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r18

; round 3.19 odd
 add     r19,r19,r18
 add     r19,r19,r26
 rotlwi  r19,r19,3
 add     r0,r19,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r19

; round 3.20 even
 add     r20,r20,r19
 add     r20,r20,r27
 rotlwi  r20,r20,3
 add     r0,r20,r27
 add     r26,r0,r26
 rotlw	r26,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r20

; round 3.21 odd
 add     r21,r21,r20
 add     r21,r21,r26
 rotlwi  r21,r21,3
 add     r0,r21,r26
 add     r27,r0,r27
 rotlw	r27,r27,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r21

; round 3.22 even
 add     r22,r22,r21
 add     r22,r22,r27
 rotlwi  r22,r22,3
 add     r0,r22,r27
 add     r26,r0,r26
 rotlw	r6,r26,r0
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r22

; round 3.23 odd
 add     r23,r23,r22
 add     r23,r23,r6
 rotlwi  r23,r23,3
 add     r0,r23,r6
 add     r7,r0,r27
 rotlw	r7,r7,r0
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r23

;	lwz	r0,r29(r1)
; round 3.24 even
 add     r24,r24,r23
 add     r24,r24,r7
 rotlwi  r24,r24,3
 xor	r30,r30,r31
 rotlw	r30,r30,r31
 add	r30,r30,r24

; Preserve these registers to finish 3.24 and 3.25
; r30, r31, r24, r25, r6, r7, r28

 cmpw	r29,r30

 addis	r28,r28,256	; Increment high byte of key

; --fill from round 1
; round 1.0 even
 lwz     r012,con1(r1)	; = r31 + r26
 lwz	r26,con0(r1)

 bdnzf	2,loop

 lwz	r4,count(r1)
 mfctr	r0
 add	r4,r4,r0

 bne	label2
; round 3.24 (continued)
 add     r0,r24,r7
 add     r6,r0,r6
 rotlw	r6,r6,r0
 lwz	r0,C_1(r1)
; round 3.25 odd
 add     r25,r25,r24
 add     r25,r25,r6
 rotlwi  r25,r25,3
 xor	r31,r31,r30
 rotlw	r31,r31,r30
 add	r31,r31,r25
 cmpw	r0,r31
 bne	label2

;!!!! Found it !!!!
; undo the last key increment and return the result
 addi	r4,r4,1
 subis	r28,r28,256
 b	exit

label2:

; registers: r4, r28, temps for round 1
; check for rollovers, update key in work record
 srwi.   r0,r28,24	;// logical shift right ??
 bne	label4		;// high bits of r28 != 0

 lwz	r3,work_r3(r1)
 li	r7,L0_hi
 lwbrx	r28,r3,r7	; L0_hi reversed
 ori	r28,r28,0xff
 addic.	r28,r28,1
 stwbrx	r28,r3,r7
;	bnz	label3	
 bne	label3
 li	r6,L0_lo
 lwbrx	r11,r3,r6	; L0_lo reversed
 addi	r11,r11,1
 stwbrx	r11,r3,r6
label3:
 lwz	r28,L0_hi(r3)

label4:
; check if done
 cmpwi	r4,0
 beq	exit

;// be sure the registers are setup for relooping!!
 cmpwi	r28,0
 beq	start	; recalc r11 constants
 b	reloop	; just restart the loop

exit:
; save the last key tested in the work record
 lwz	r3,work_r3(r1)
 stw	r28,L0_hi(r3)

; return the count of keys tested
 lwz	r3,iterations(r1)
 sub	r3,r3,r4

 lwz	r2,save_RTOC(r1)
 lmw     r13,-76(r1)
 blr

