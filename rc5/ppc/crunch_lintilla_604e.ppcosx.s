;
; $Log: crunch_lintilla_604e.ppcosx.s,v $
; Revision 1.1.2.1  2001/08/07 05:20:43  mfeiri
; Renamed from crunch_lintilla_296-604_a.ppc.s to crunch_lintilla_604e.ppcosx.s
;
; Revision 1.1.2.3  2000/06/20 22:11:08  mfeiri
; The compiler used in MacOSX doesnt support ".balignl"
;
; Revision 1.1.2.2  2000/06/20 18:18:57  mfeiri
; Applied the alignment changes by Oliver. It starts getting difficult to keep things in sync...
;
; Revision 1.1.2.1  2000/06/13 01:30:22  mfeiri
; A modified version of crunch_lintilla_296_apple.ppc.s that just includes the optimizations for 604e CPUs. Please keep filenames under 32 characters to circumvent problems for MacOS.
;
; Revision 1.3.2.2  2000/01/03 14:34:37  patrick
;
; EGCS on AIX needs the defines. Please keep em.
;
; Revision 1.3.2.1  1999/12/06 09:56:55  myshkin
; Minor changes to conform to gas syntax (version 980114).
;
; Revision 1.3  1999/04/08 18:49:22  patrick
;
; bug fixes (double rgister usage and other stuff). Did this ever compile ?
;
; Revision 1.2  1998/06/14 10:30:36  friedbait
; 'Log' keyword added.
;
;
 .file	"crunch-ppc.cpp"
gcc2_compiled.:
; .csect        .text[PR]
; .align 8
;if (CLIENT_OS == OS_AIX)
 .globl .crunch_lintilla_604e
;else
 .globl _crunch_lintilla_604e
;endif
; .type  crunch_lintilla_604e,@function
;if (CLIENT_OS == OS_AIX)
.crunch_lintilla_604e:
;else
_crunch_lintilla_604e:
;endif

; standard register aliases
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

; offsets into the RC5UnitWork structure
.set plain_hi,0     ; plaintext, already mixed with IV
.set plain_lo,4
.set cypher_hi,8    ; target cyphertext
.set cypher_lo,12
.set L0_hi,16       ; key, changes with every unit * PIPELINE_COUNT
.set L0_lo,20

; Local_Frame struct: stack frame and local variables
.set frame_new,0	; save r1
.set save_CR,4		; save CR
.set save_RTOC,20	; save RTOC

; local variables
;.set param1, 24	; 24(r1) pass parameters here
;.set param2, 28	; 28(r1) ...
.set count, 32
.set P_0, 36
.set P_1, 40
.set C_0, 44
.set C_1, 48
.set L0_0, 52
.set L0_1, 56
.set Sr_0, 60
.set Sr_1, 64
.set con0, 68
.set con1, 72
.set con2, 76
.set S0_3, 80
.set S0_4, 84
.set S0_5, 88
.set S0_6, 92
.set S0_7, 96
.set S0_8, 100
.set S0_9, 104
.set S0_10, 108
.set S0_11, 112
.set S0_12, 116
.set S0_13, 120
.set S0_14, 124
.set S0_15, 128
.set S0_16, 132
.set S0_17, 136
.set S0_18, 140
.set S0_19, 144
.set S0_20, 148
.set S0_21, 152
.set S0_22, 156
.set S0_23, 160
.set S0_24, 164
.set S0_25, 168
.set Ss_2, 172
.set Ss_3, 176
.set Ss_4, 180
.set Ss_5, 184
.set Ss_6, 188
.set Ss_7, 192
.set Ss_8, 196
.set Ss_9, 200
.set Ss_10, 204
.set Ss_11, 208
.set Ss_12, 212
.set Ss_13, 216
.set Ss_14, 220

; old stack frame
   ; -76+frame(r1)	saved GPR13-31
.set frame_old, 300	; 0+frame(r1)	old r1 pointed here
   ; 4+frame(r1)	save CR
   ; 8+frame(r1)	save LR
   ; 12+frame(r1)	reserved
   ; 16+frame(r1)	reserved
   ; 20+frame(r1)	callers RTOC
; calling parameter storage
.set work_ptr, 324	; 24+frame(r1)
.set iterations, 328	; 28+frame(r1)

; register aliases

.set r0, 0
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
.set r25, 25	; Note register overloading
.set r25, 25
.set r26, r26
.set r27, 27
.set r28, 28
.set r28, 28
.set r29, 29
.set r29, 29
.set r30, 30
.set r30, 30
.set r30, 30
.set r31, 31
.set r31, 31
.set r31, 31

prolog:
 stmw	r13,-76(r1)
 stwu	r1,frame_new-frame_old(r1)

 stw	r2,save_RTOC(r1)
 stw     r3,work_ptr(r1)		; work
 stw     r4,iterations(r1)	; initial iterations
 stw     r4,count(r1)		; remaining iterations

; Save work parameters on the local stack
 lwz	r0,plain_hi(r3)
 stw	r0,P_1(r1)
 lwz	r0,plain_lo(r3)
 stw	r0,P_0(r1)
 lwz	r0,cypher_hi(r3)
 stw	r0,C_1(r1)
 lwz	r0,cypher_lo(r3)
 stw	r0,C_0(r1)
 lwz	r27,L0_hi(r3)
 stw	r27,L0_1(r1)
 lwz	r26,L0_lo(r3)
 stw	r26,L0_0(r1)

start:
;Round 1 is rolled out of the loop to fill the pipe and setup constants.
;
 lis     r31,0xB7E1
 lis     r30,0x9E37
 addi    r31,r31,0x5163	;r31 = P;
 addi    r30,r30,0x79B9	;r30 = Q;

; Round 1.0 even
 rotlwi  r25,r31,3
 stw	r25,Sr_0(r1);		; Sr_0 = ...
 lwz	r26,L0_0(r1)
 add     r26,r26,r25
 rotlw	r26,r26,r25
 stw	r26,con0(r1)		; con0 = Lr_0
 add     r31,r31,r30

; Round 1.1 odd
 add     r0,r31,r25
 add     r0,r0,r26	
 rotlwi  r25,r0,3
 stw	r25,Sr_1(r1)		; Sr_1 = rotl3(...)
 add     r0,r25,r26
 stw	r0,con1(r1)		; con1 =  Sr_1 + Lr_0
 lwz	r27,L0_1(r1)
 add     r27,r27,r0
 rotlw	r27,r27,r0
 add     r31,r31,r30

; Round 1.2 even
 add     r0,r31,r25
 stw	r0,con2(r1)		; con2 = S0_2 + Sr_1
 add     r0,r0,r27
 rotlwi  r2,r0,3
 add     r0,r2,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_3(r1)
 rotlw	r26,r26,r0

; Round 1.3 odd
 add     r0,r31,r2
 add     r0,r0,r26	
 rotlwi  r3,r0,3
 add     r0,r3,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_4(r1)
 rotlw	r27,r27,r0

; Round 1.4 even
 add     r0,r31,r3
 add     r0,r0,r27
 rotlwi  r4,r0,3
 add     r0,r4,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_5(r1)
 rotlw	r26,r26,r0

; Round 1.5 odd
 add     r0,r31,r4
 add     r0,r0,r26	
 rotlwi  r5,r0,3
 add     r0,r5,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_6(r1)
 rotlw	r27,r27,r0

; Round 1.6 even
 add     r0,r31,r5
 add     r0,r0,r27
 rotlwi  r6,r0,3
 add     r0,r6,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_7(r1)
 rotlw	r26,r26,r0

; Round 1.7 odd
 add     r0,r31,r6
 add     r0,r0,r26	
 rotlwi  r7,r0,3
 add     r0,r7,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_8(r1)
 rotlw	r27,r27,r0

; Round 1.8 even
 add     r0,r31,r7
 add     r0,r0,r27
 rotlwi  r8,r0,3
 add     r0,r8,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_9(r1)
 rotlw	r26,r26,r0

; Round 1.9 odd
 add     r0,r31,r8
 add     r0,r0,r26	
 rotlwi  r9,r0,3
 add     r0,r9,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_10(r1)
 rotlw	r27,r27,r0

; Round 1.10 even
 add     r0,r31,r9
 add     r0,r0,r27
 rotlwi  r10,r0,3
 add     r0,r10,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_11(r1)
 rotlw	r26,r26,r0

; Round 1.11 odd
 add     r0,r31,r10
 add     r0,r0,r26	
 rotlwi  r11,r0,3
 add     r0,r11,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_12(r1)
 rotlw	r27,r27,r0

; Round 1.12 even
 add     r0,r31,r11
 add     r0,r0,r27
 rotlwi  r12,r0,3
 add     r0,r12,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_13(r1)
 rotlw	r26,r26,r0

; Round 1.13 odd
 add     r0,r31,r12
 add     r0,r0,r26	
 rotlwi  r13,r0,3
 add     r0,r13,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_14(r1)
 rotlw	r27,r27,r0

; Round 1.14 even
 add     r0,r31,r13
 add     r0,r0,r27
 rotlwi  r14,r0,3
 add     r0,r14,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_15(r1)
 rotlw	r26,r26,r0

; Round 1.15 odd
 add     r0,r31,r14
 add     r0,r0,r26	
 rotlwi  r15,r0,3
 add     r0,r15,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_16(r1)
 rotlw	r27,r27,r0

; Round 1.16 even
 add     r0,r31,r15
 add     r0,r0,r27
 rotlwi  r16,r0,3
 add     r0,r16,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_17(r1)
 rotlw	r26,r26,r0

; Round 1.17 odd
 add     r0,r31,r16
 add     r0,r0,r26	
 rotlwi  r17,r0,3
 add     r0,r17,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_18(r1)
 rotlw	r27,r27,r0

; Round 1.18 even
 add     r0,r31,r17
 add     r0,r0,r27
 rotlwi  r18,r0,3
 add     r0,r18,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_19(r1)
 rotlw	r26,r26,r0

; Round 1.19 odd
 add     r0,r31,r18
 add     r0,r0,r26	
 rotlwi  r19,r0,3
 add     r0,r19,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_20(r1)
 rotlw	r27,r27,r0

; Round 1.20 even
 add     r0,r31,r19
 add     r0,r0,r27
 rotlwi  r20,r0,3
 add     r0,r20,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_21(r1)
 rotlw	r26,r26,r0

; Round 1.21 odd
 add     r0,r31,r20
 add     r0,r0,r26	
 rotlwi  r21,r0,3
 add     r0,r21,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_22(r1)
 rotlw	r27,r27,r0

; Round 1.22 even
 add     r0,r31,r21
 add     r0,r0,r27
 rotlwi  r22,r0,3
 add     r0,r22,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_23(r1)
 rotlw	r26,r26,r0

; Round 1.23 odd
 add     r0,r31,r22
 add     r0,r0,r26	
 rotlwi  r23,r0,3
 add     r0,r23,r26
 add     r31,r31,r30
 add     r27,r27,r0
 stw	r31,S0_24(r1)
 rotlw	r27,r27,r0

; Round 1.24 even
 add     r0,r31,r23
 add     r0,r0,r27
 rotlwi  r24,r0,3
 add     r0,r24,r27
 add     r31,r31,r30
 add     r26,r26,r0
 stw	r31,S0_25(r1)
 rotlw	r28,r26,r0

; Round 1.25 odd
 add     r0,r31,r24
 add     r0,r0,r28	
 rotlwi  r25,r0,3
 add     r0,r25,r28
 add     r27,r27,r0
 rotlw	r29,r27,r0

reloop:

 lwz	r27,L0_1(r1)	; key.hi

; registers r0, r26, r30, r31 are free
;//	compute the inner loop count to next rollover
;	cnt = 255-((key1>>24)&0xff)
 srwi	r0,r27,24	; shift right logical 24, mask 0xFF
 subfic	r0,r0,255
;	if cnt==0 // handle hi byte rollover
;	{
  cmpwi	r0,0
  bne	label2
;		key1=rb(rb(key1)+1)
  li	r26,L0_1
  lwbrx	r27,r1,r26
  addic.	r27,r27,1
  stwbrx	r27,r1,r26
;		cnt = 256
  li	r0,256
  lwz	r27,L0_1(r1)
  subis	r27,r27,256
;		if key1 == 0 // finish last key before rollover
;			cnt = 1
  bne	label2
  li	r0,1
;	}

; limit loop to remaining count
label2:
 lwz	r30,count(r1)
 cmpw	r0,r30
 ble	label3
 mr	r0,r30
label3:	
 sub	r30,r30,r0
 stw	r30,count(r1)
 mtctr	r0

 lwz	r30,Sr_0(r1)
 lwz	r31,Sr_1(r1)

loop:

;	r0	r1	r2	r3	...	r23	r24	r25	r26	r27	r28	r29	r30	r31
;	r0	r1	r2	r3	...			r25 ?	r27	r28	r29	r30	r31

 ;2.0
 add     r30,r30,r25
   ;1-0
 add     r30,r30,r29
   lwz	r26,con1(r1)	; const Lr_0 + Sr_1

 rotlwi  r30,r30,3
   addis	r27,r27,256

 add     r0,r30,r29
   stw	r27,L0_1(r1)	; key.hi

 add     r28,r0,r28
   ;1-1
   add     r27,r27,r26

 rotlw	r28,r28,r0

;	r0	r1	r2	r3	...	r23	r24	r25	r26	r27	r28	r29	r30	r31
;	r0	r1	r2	r3	...		   r25	r26	r27	r28	r29	Ss0	r31

; Round 2.1 odd
 add     r31,r31,r30

 add     r0,r31,r28
   rotlw	r27,r27,r26

 rotlwi  r31,r0,3
 add	r29,r28,r29

 add     r0,r31,r28		
 add     r29,r31,r29

 rotlw	r29,r29,r0

;	r0	r1	r2	r3	...	r23	r24	r25	r26	r27	r28	r29	r30	r31
;	r0	r1	r2	r3	...		   r25	---	key	r28	r29	Ss0	Ss1

; Round 2.2 even
 add     r2,r2,r31

 add     r2,r2,r29
   lwz	r26,con2(r1)	; const S0_2 + Sr_1

 rotlwi  r2,r2,3
 add	r28,r28,r29

 add     r0,r2,r29
 add     r28,r2,r28

 rotlw	r28,r28,r0


;	r0	r1	r2	r3	r4	...	r24	r25	r26	r27	r28	r29	r30	r31
;	r0	r1	Ss2	Ss3	r4	...	   r25	c2	r27	r28	r29	Ss0	Ss1

; Round 2.3 odd
 add     r3,r3,r2
   ;1.2
 add     r3,r3,r28
 stw	r2,Ss_2(r1)

 rotlwi  r3,r3,3
 stw	r3,Ss_3(r1)
   add     r2,r26,r27

 add     r0,r3,r28	
   rotlwi  r2,r2,3

 add     r29,r0,r29
   lwz	r26,con0(r1)	; const Lr_0

 rotlw	r29,r29,r0


; Round 2.4 even
 add     r4,r4,r3

 add     r4,r4,r29

   add     r0,r2,r27
   lwz     r3,S0_3(r1)

   add     r26,r26,r0
 rotlwi  r4,r4,3

   rotlw	r26,r26,r0
 add     r0,r4,r29

 add     r28,r0,r28

 rotlw	r28,r28,r0

; Round 2.5 odd		round 1.3 odd
 add     r5,r5,r4
 add     r5,r5,r28
   add     r3,r3,r2
 rotlwi  r5,r5,3
   add     r3,r3,r26
 add     r0,r5,r28		
   rotlwi  r3,r3,3
 add     r29,r0,r29
   stw	r4,Ss_4(r1)	
   stw	r5,Ss_5(r1)
 rotlw	r29,r29,r0

; Round 2.6 even
 add     r6,r6,r5
 add     r6,r6,r29
   add     r0,r3,r26
 rotlwi  r6,r6,3
   add     r27,r27,r0
   rotlw	r27,r27,r0
 add     r0,r6,r29
 add     r28,r0,r28
   lwz	r4,S0_4(r1)
 rotlw	r28,r28,r0

; Round 2.7 odd		round 1.4 even
 add     r7,r7,r6
 add     r7,r7,r28
   add     r4,r4,r3
 rotlwi  r7,r7,3
   add     r4,r4,r27
   rotlwi  r4,r4,3
 add     r0,r7,r28	
 add     r29,r0,r29
 rotlw	r29,r29,r0

; Round 2.8 even
 add     r8,r8,r7
 add     r8,r8,r29
   add     r0,r4,r27
 rotlwi  r8,r8,3
   add     r26,r26,r0
   rotlw	r26,r26,r0
 add     r0,r8,r29
 add     r28,r0,r28
   lwz	r5,S0_5(r1)
 rotlw	r28,r28,r0

; Round 2.9 odd		round 1.5 odd
 add     r9,r9,r8
 add     r9,r9,r28
   add     r5,r5,r4
 rotlwi  r9,r9,3
   add     r5,r5,r26
   rotlwi  r5,r5,3
 add     r0,r9,r28		
 add     r29,r0,r29
   stw	r6,Ss_6(r1)
   stw	r7,Ss_7(r1)
   stw	r8,Ss_8(r1)
   stw	r9,Ss_9(r1)   
 rotlw	r29,r29,r0

; Round 2.10 even
 add     r10,r10,r9
 add     r10,r10,r29
   add     r0,r5,r26
 rotlwi  r10,r10,3
   add     r27,r27,r0
   rotlw	r27,r27,r0
 add     r0,r10,r29
 add     r28,r0,r28
   lwz	r6,S0_6(r1)
 rotlw	r28,r28,r0

; Round 2.11 odd	round 1.6 even
 add     r11,r11,r10
 add     r11,r11,r28
   add     r6,r6,r5
 rotlwi  r11,r11,3
   add     r6,r6,r27
   rotlwi  r6,r6,3
 add     r0,r11,r28		
 add     r29,r0,r29
 rotlw	r29,r29,r0

; Round 2.12 even
 add     r12,r12,r11
 add     r12,r12,r29
   add     r0,r6,r27
 rotlwi  r12,r12,3
   add     r26,r26,r0
   rotlw	r26,r26,r0
 add     r0,r12,r29
 add     r28,r0,r28
   lwz	r7,S0_7(r1)
 rotlw	r28,r28,r0

; Round 2.13 odd	round 1.7 odd
 add     r13,r13,r12
 add     r13,r13,r28
   add     r7,r7,r6
 rotlwi  r13,r13,3
   add     r7,r7,r26
   rotlwi  r7,r7,3
 add     r0,r13,r28		
 add     r29,r0,r29
 rotlw	r29,r29,r0

; Round 2.14 even
 add     r14,r14,r13
 add     r14,r14,r29
   add     r0,r7,r26
 rotlwi  r14,r14,3
   add     r27,r27,r0
   rotlw	r27,r27,r0
 add     r0,r14,r29
 add     r28,r0,r28
   lwz	r8,S0_8(r1)
 rotlw	r28,r28,r0

; Round 2.15 odd	round 1.8 even
 add     r15,r15,r14
 add     r15,r15,r28
   add     r8,r8,r7
 rotlwi  r15,r15,3
   add     r8,r8,r27
   rotlwi  r8,r8,3
 add     r0,r15,r28		
 add     r29,r0,r29
 rotlw	r29,r29,r0

; Round 2.16 even
 add     r16,r16,r15
 add     r16,r16,r29
   add     r0,r8,r27
 rotlwi  r16,r16,3
   add     r26,r26,r0
   rotlw	r26,r26,r0
 add     r0,r16,r29
 add     r28,r0,r28
   lwz	r9,S0_9(r1)
 rotlw	r28,r28,r0

; Round 2.17 odd	round 1.9 odd
 add     r17,r17,r16
 add     r17,r17,r28
   add     r9,r9,r8
 rotlwi  r17,r17,3
   add     r9,r9,r26
   rotlwi  r9,r9,3
 add     r0,r17,r28		
 add     r29,r0,r29
   stw	r10,Ss_10(r1)
   stw	r11,Ss_11(r1)
   stw	r12,Ss_12(r1)
   stw	r13,Ss_13(r1)
   stw	r14,Ss_14(r1)
 rotlw	r29,r29,r0

; Round 2.18 even
 add     r18,r18,r17
 add     r18,r18,r29
   add     r0,r9,r26
 rotlwi  r18,r18,3
   add     r27,r27,r0
   rotlw	r27,r27,r0
 add     r0,r18,r29
 add     r28,r0,r28
   lwz	r10,S0_10(r1)
 rotlw	r28,r28,r0

; Round 2.19 odd	round 1.10 even
 add     r19,r19,r18
 add     r19,r19,r28
   add     r10,r10,r9
 rotlwi  r19,r19,3
   add     r10,r10,r27
   rotlwi  r10,r10,3
 add     r0,r19,r28		
 add     r29,r0,r29
 rotlw	r29,r29,r0

; Round 2.20 even
 add     r20,r20,r19
 add     r20,r20,r29
   add     r0,r10,r27
 rotlwi  r20,r20,3
   add     r26,r26,r0
   rotlw	r26,r26,r0
 add     r0,r20,r29
 add     r28,r0,r28
   lwz	r11,S0_11(r1)
 rotlw	r28,r28,r0

; Round 2.21 odd	round 1.11 odd
 add     r21,r21,r20
 add     r21,r21,r28
   add     r11,r11,r10
 rotlwi  r21,r21,3
   add     r11,r11,r26
   rotlwi  r11,r11,3
 add     r0,r21,r28		
 add     r29,r0,r29
 rotlw	r29,r29,r0

; Round 2.22 even
 add     r22,r22,r21
 add     r22,r22,r29
   add     r0,r11,r26
 rotlwi  r22,r22,3
   add     r27,r27,r0
   rotlw	r27,r27,r0
 add     r0,r22,r29
 add     r28,r0,r28
   lwz	r12,S0_12(r1)
 rotlw	r28,r28,r0

; Round 2.23 odd	round 1.12 even
 add     r23,r23,r22
 add     r23,r23,r28
   add     r12,r12,r11
 rotlwi  r23,r23,3
   add     r12,r12,r27
   rotlwi  r12,r12,3
 add     r0,r23,r28		
 add     r29,r0,r29
 rotlw	r29,r29,r0

; Round 2.24 even
 add     r24,r24,r23
 add     r24,r24,r29
   add     r0,r12,r27
 rotlwi  r24,r24,3
   add     r26,r26,r0
   rotlw	r26,r26,r0
 add     r0,r24,r29
 add     r28,r0,r28
   lwz	r13,S0_13(r1)
 rotlw	r28,r28,r0

; Round 2.25 odd	round 1.13 odd
 add     r25,r25,r24

 add     r25,r25,r28
   add     r13,r13,r12

 rotlwi  r25,r25,3
   add     r13,r13,r26

   rotlwi  r13,r13,3
 add     r0,r25,r28	

 add     r29,r0,r29

 rotlw	r29,r29,r0

;//	stw	r25,Ss_25(r1)	; Ss_25 will most often never get read

; Registers:
;	r0	r1	r2	...	r13	r14	...	r25	r26	r27	r28	r29	r30	r31
;	r0	r1	r2	...r13	Ss14...Ss25	r26	r27	r28	r29	Ss0	Ss1
;						r14	r25			r28	r29	r30	r31

; Note register lineup r25->r25, r28->r28, r29->r29, r30->r30, r31->r31

; Round 3.0 even
 add     r25,r25,r30	; r25 is r25
 add     r25,r25,r29
   lwz	r30,P_0(r1)
 rotlwi  r25,r25,3
   add     r14,r13,r26
 add     r0,r25,r29
   add     r27,r27,r14
 add	r28,r0,r28
   rotlw	r27,r27,r14
 rotlw	r28,r28,r0

; Round 3.1 odd
 add     r14,r25,r31

 add     r14,r14,r28
   lwz	r31,P_1(r1)

   add	r30,r30,r25	;r30_0
 rotlwi  r25,r14,3

 add     r0,r25,r28
   lwz	r14,Ss_2(r1)

 add	r29,r0,r29
;	addi	r10,r10,0	;--fill???--

 rotlw	r29,r29,r0

; Round 3.2 even
 add     r14,r25,r14

 add     r14,r14,r29
   add	r31,r31,r25	;r31_0

 rotlwi  r25,r14,3
 addi	r10,r10,0	;--fill--

 add     r0,r25,r29
   lwz	r14,Ss_3(r1)

 add     r28,r28,r0
   xor	r30,r30,r31

 rotlw	r28,r28,r0

; Round 3.3 odd
 add     r14,r25,r14
 add     r14,r14,r28
   rotlw	r30,r30,r31
   add	r30,r30,r25	;r30_2
 rotlwi  r25,r14,3
 add     r0,r25,r28
   lwz	r14,Ss_4(r1)
   xor	r31,r31,r30
 add     r29,r29,r0
 rotlw	r29,r29,r0

; Round 3.4 even
 add     r14,r25,r14
 add     r14,r14,r29
   rotlw	r31,r31,r30
   add	r31,r31,r25
 rotlwi  r25,r14,3
 add     r0,r25,r29
   lwz	r14,Ss_5(r1)
   xor	r30,r30,r31
 add     r28,r28,r0
 rotlw	r28,r28,r0


; Round 3.5 odd
 add     r14,r25,r14
 add     r14,r14,r28
   rotlw	r30,r30,r31
   add	r30,r30,r25
 rotlwi  r25,r14,3
 add     r0,r25,r28
   lwz	r14,Ss_6(r1)
   xor	r31,r31,r30
 add     r29,r29,r0
 rotlw	r29,r29,r0

; Round 3.6 even
 add     r14,r25,r14
 add     r14,r14,r29
   rotlw	r31,r31,r30
   add	r31,r31,r25
 rotlwi  r25,r14,3
 add     r0,r25,r29
   lwz	r14,Ss_7(r1)
 add     r28,r28,r0
   xor	r30,r30,r31
 rotlw	r28,r28,r0

; Round 3.7 odd
 add     r14,r25,r14
 add     r14,r14,r28
   rotlw	r30,r30,r31
   add	r30,r30,r25
 rotlwi  r25,r14,3
 add     r0,r25,r28
   lwz	r14,Ss_8(r1)
 add     r29,r29,r0
   xor	r31,r31,r30
 rotlw	r29,r29,r0

; Round 3.8 even
 add     r14,r25,r14
 add     r14,r14,r29
   rotlw	r31,r31,r30
   add	r31,r31,r25
 rotlwi  r25,r14,3
 add     r0,r25,r29
   lwz	r14,Ss_9(r1)
 add     r28,r28,r0
   xor	r30,r30,r31
 rotlw	r28,r28,r0

; Round 3.9 odd
 add     r14,r25,r14
 add     r14,r14,r28
   rotlw	r30,r30,r31
   add	r30,r30,r25
 rotlwi  r25,r14,3
 add     r0,r25,r28
   lwz	r14,Ss_10(r1)
 add     r29,r29,r0
   xor	r31,r31,r30
 rotlw	r29,r29,r0

; Round 3.10 even
 add     r14,r25,r14
 add     r14,r14,r29
   rotlw	r31,r31,r30
   add	r31,r31,r25
 rotlwi  r25,r14,3
 add     r0,r25,r29
   lwz	r14,Ss_11(r1)
   xor	r30,r30,r31
 add     r28,r28,r0
 rotlw	r28,r28,r0

; Round 3.11 odd
 add     r14,r25,r14
 add     r14,r14,r28
   rotlw	r30,r30,r31
   add	r30,r30,r25
 rotlwi  r25,r14,3
 add     r0,r25,r28
   lwz	r14,Ss_12(r1)
   xor	r31,r31,r30
 add     r29,r29,r0
 rotlw	r29,r29,r0

; Round 3.12 even
 add     r14,r25,r14
 add     r14,r14,r29
   rotlw	r31,r31,r30
   add	r31,r31,r25
 rotlwi  r25,r14,3
 add     r0,r25,r29
   lwz	r14,Ss_13(r1)
   xor	r30,r30,r31
 add     r28,r28,r0
 rotlw	r28,r28,r0

; Round 3.13 odd
 add     r14,r25,r14
 add     r14,r14,r28
   rotlw	r30,r30,r31
   add	r30,r30,r25	;r30_12
 rotlwi  r25,r14,3
 add     r0,r25,r28
   lwz	r14,Ss_14(r1)	;r14 is r14!
   xor	r31,r31,r30
 add     r29,r29,r0

   rotlw	r31,r31,r30
   add	r31,r31,r25
 rotlw	r29,r29,r0

; Round 1.14/3.14 even
 add     r25,r25,r14
 lwz     r14,S0_14(r1)
 xor	r30,r30,r31
 add     r25,r25,r29
 rotlwi  r25,r25,3
 add     r14,r14,r13
 add     r14,r14,r27
 add     r0,r25,r29
 rotlwi  r14,r14,3
 add     r28,r28,r0
 rotlw	r28,r28,r0
 add     r0,r14,r27
 rotlw	r30,r30,r31
 add     r26,r26,r0
 rotlw	r26,r26,r0
 add	r30,r30,r25

; Round 1.15/3.15 odd
 add     r25,r25,r15
 lwz     r15,S0_15(r1)
 xor	r31,r31,r30
 add     r25,r25,r28
 rotlwi  r25,r25,3
 add     r15,r15,r14
 add     r15,r15,r26
 add     r0,r25,r28
 rotlwi  r15,r15,3
 add     r29,r29,r0
 rotlw	r29,r29,r0
 add     r0,r15,r26
 rotlw	r31,r31,r30
 add     r27,r27,r0
 rotlw	r27,r27,r0
 add	r31,r31,r25

; Round 1.16/3.16 even
 add     r25,r25,r16
 lwz     r16,S0_16(r1)
 xor	r30,r30,r31
 add     r25,r25,r29
 rotlwi  r25,r25,3
 add     r16,r16,r15
 add     r16,r16,r27
 add     r0,r25,r29
 rotlwi  r16,r16,3
 add     r28,r28,r0
 rotlw	r28,r28,r0
 add     r0,r16,r27
 rotlw	r30,r30,r31
 add     r26,r26,r0
 rotlw	r26,r26,r0
 add	r30,r30,r25

; Round 1.17/3.17 odd
 add     r25,r25,r17
 lwz     r17,S0_17(r1)
 xor	r31,r31,r30
 add     r25,r25,r28
 rotlwi  r25,r25,3
 add     r17,r17,r16
 add     r17,r17,r26
 add     r0,r25,r28
 rotlwi  r17,r17,3
 add     r29,r29,r0
 rotlw	r29,r29,r0
 add     r0,r17,r26
 rotlw	r31,r31,r30
 add     r27,r27,r0
 rotlw	r27,r27,r0
 add	r31,r31,r25

; Round 1.18/3.18 even
 add     r25,r25,r18
 lwz     r18,S0_18(r1)
 xor	r30,r30,r31
 add     r25,r25,r29
 rotlwi  r25,r25,3
 add     r18,r18,r17
 add     r18,r18,r27
 add     r0,r25,r29
 rotlwi  r18,r18,3
 add     r28,r28,r0
 rotlw	r28,r28,r0
 add     r0,r18,r27
 rotlw	r30,r30,r31
 add     r26,r26,r0
 rotlw	r26,r26,r0
 add	r30,r30,r25

; Round 1.19/3.19 odd
 add     r25,r25,r19
 lwz     r19,S0_19(r1)
 xor	r31,r31,r30
 add     r25,r25,r28
 rotlwi  r25,r25,3
 add     r19,r19,r18
 add     r19,r19,r26
 add     r0,r25,r28
 rotlwi  r19,r19,3
 add     r29,r29,r0
 rotlw	r29,r29,r0
 add     r0,r19,r26
 rotlw	r31,r31,r30
 add     r27,r27,r0
 rotlw	r27,r27,r0
 add	r31,r31,r25

; Round 1.20/3.20 even
 add     r25,r25,r20
 lwz     r20,S0_20(r1)
 xor	r30,r30,r31
 add     r25,r25,r29
 rotlwi  r25,r25,3
 add     r20,r20,r19
 add     r20,r20,r27
 add     r0,r25,r29
 rotlwi  r20,r20,3
 add     r28,r28,r0
 rotlw	r28,r28,r0
 add     r0,r20,r27
 rotlw	r30,r30,r31
 add     r26,r26,r0
 rotlw	r26,r26,r0
 add	r30,r30,r25

; Round 1.21/3.21 odd
 add     r25,r25,r21
 lwz     r21,S0_21(r1)
 xor	r31,r31,r30
 add     r25,r25,r28
 rotlwi  r25,r25,3
 add     r21,r21,r20
 add     r21,r21,r26
 add     r0,r25,r28
 rotlwi  r21,r21,3
 add     r29,r29,r0
 rotlw	r29,r29,r0
 add     r0,r21,r26
 rotlw	r31,r31,r30
 add     r27,r27,r0
 rotlw	r27,r27,r0
 add	r31,r31,r25

; Round 1.22/3.22 even
 add     r25,r25,r22
 lwz     r22,S0_22(r1)
 xor	r30,r30,r31
 add     r25,r25,r29
 rotlwi  r25,r25,3
 add     r22,r22,r21
 add     r22,r22,r27
 add     r0,r25,r29
 rotlwi  r22,r22,3
 add     r28,r28,r0
 rotlw	r28,r28,r0
 add     r0,r22,r27
 rotlw	r30,r30,r31
 add     r26,r26,r0
 rotlw	r26,r26,r0
 add	r30,r30,r25

; Round 1.23/3.23 odd
 add     r25,r25,r23
 lwz     r23,S0_23(r1)

 xor	r31,r31,r30
 add     r25,r25,r28

 rotlwi  r25,r25,3
 add     r23,r23,r22

 add     r23,r23,r26
 add     r0,r25,r28

 rotlwi  r23,r23,3
 add     r29,r29,r0

 rotlw	r29,r29,r0
 add     r0,r23,r26

 rotlw	r31,r31,r30
 add     r27,r27,r0

 rotlw	r27,r27,r0
 add	r31,r31,r25


;	r0	r1	r2	r3	...	r23	r24	r25	r26	r27	r28	r29	r30	r31
;	r0	r1	r2	r3	...	r23		r26	r27	---	r29
;							Ss24 r25					r30	r31
; Round 1.24/3.24 even
 add     r25,r25,r24
 lwz     r24,S0_24(r1)

 xor	r30,r30,r31
 add     r25,r25,r29

 add     r24,r24,r23
 rotlwi  r25,r25,3

 add     r24,r24,r27
 rotlw	r30,r30,r31

 rotlwi  r24,r24,3
 add	r31,r30,r25

 add     r0,r24,r27
 lwz     r25,S0_25(r1)

 add     r28,r26,r0		;switch to r28/r29 registers
 lwz	r29,C_0(r1)

 rotlw	r28,r28,r0

;	r0	r1	r2	r3	...	r23	r24	r25	r26	r27	r28	r29	r30	r31
;	r0	r1	r2	r3	...			r25---	r27	r28	C_0	---	tr30

; Round 1.25 odd
 add     r25,r25,r24
 add     r25,r25,r28
   lwz	r30,Sr_0(r1)
   cmplw	r29,r31
 rotlwi  r25,r25,3
 add     r0,r25,r28
   lwz	r31,Sr_1(r1)
 add     r29,r0,r27
   lwz	r27,L0_1(r1)	; key.hi
 rotlw	r29,r29,r0

;	r0	r1	r2	r3	...	r23	r24	r25	r26	r27	r28	r29	r30	r31
;	r0	r1	r2	r3	...			r25 ?	r27	r28	r29	r30	r31

 bdnzf	2,loop
;// free registers:
;// r0 (r0), r26 (r26)
;// r27 (r27), r30 (r30), r31 (r31) can be reloaded

;//registers available:
;//	r0=r0,r26=r26,r27=r27,r30,r31

 lwz	r30,count(r1)

;	if key found
 bne	label5
;		decrement key_hi (byte reversed)
 li	r26,L0_1
 lwbrx	r27,r1,r26
 subi	r27,r27,1
 stwbrx	r27,r1,r26	;//also save in work?
 lwz	r27,L0_1(r1)

;		increment remaining count
;		add unused loop count
 mfctr	r0
 add	r30,r30,r0
 addi	r30,r30,1
;		goto exit
 b	exit
;
label5:	
;	if key_hi==0x0000
;		increment key_lo
 cmpwi	r27,0
 bne	label6
 li	r31,L0_0
 lwbrx	r26,r1,r31
 addi	r26,r26,1
 stwbrx	r26,r1,r31	;// also save in work record

label6:	
;	if remaining count == 0 goto exit
 cmpwi	r30,0
 beq	exit

;	if key_hi==0x0000
 cmpwi	r27,0; same as earlier
 beq	start	;// round 1 was invalid! (should never happen in real client)
;carry registers to reloop: r27,cnt
 b	reloop

;//exit conditions:
;//  r30 = remiaining count
;//  r27 = key.hi

exit:
;// save the last key tested in the work record
 lwz	r4,work_ptr(r1)
;	WITH RC5UnitWork ; base register r4
 lwz	r26,L0_0(r1)
 stw	r26,L0_lo(r4)
;	lwz	r27,L0_1(r1)
 stw	r27,L0_hi(r4)
;	ENDWITH
;// return count of keys tested
 lwz	r3,iterations(r1)
 sub	r3,r3,r30

 lwz	r2,save_RTOC(r1)
 addi	r1,r1,frame_old-frame_new
 lmw     r13,-76(r1)
 blr

;	ENDWITH
;	END

