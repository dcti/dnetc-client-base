#// IMPORTANT: This function only tests the low word of the encryption result.
#// If a hit is returned you must verify that the high word is also correct
#// by calling another cracking function.
.file	"crunch-ppc.cpp"
gcc2_compiled.:
.section	".text"
.align 2
.globl crunch_lintilla
.type	 crunch_lintilla,@function
.long 0x4008
crunch_lintilla:


# offsets into the RC5UnitWork structure
.set plain_hi,0     # plaintext, already mixed with IV
.set plain_lo,4
.set cypher_hi,8    # target cyphertext
.set cypher_lo,12
.set L0_hi,16       # key, changes with every unit * PIPELINE_COUNT
.set L0_lo,20


# register name aliases
.set SP, r1
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

.set iterations, 28
.set work_ptr, 24		
.set save_RTOC, -80
.set count, -84
.set P_0, -88
.set P_1, -92
.set C_0, -96
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

.set tmp, r0
.set Sr2, r2
.set Sr3, r3
.set Sr4, r4
.set Sr5, r5
.set Sr6, r6
.set Sr7, r7
.set Sr8, r8
.set Sr9, r9
.set Sr10, r10
.set Sr11, r11
.set Sr12, r12
.set Sr13, r13
.set Sr14, r14
.set Sr15, r15
.set Sr16, r16
.set Sr17, r17
.set Sr18, r18
.set Sr19, r19
.set Sr20, r20
.set Sr21, r21
.set Sr22, r22
.set Sr23, r23
.set Sr24, r24
.set Sr25, r25
.set At, r25
.set Lr0, r26
.set Lr1, r27
.set Lt0, r28
.set Ls0, r28
.set S0r, r28
.set Lt1, r29
.set Ls1, r29
.set Qr, r29
.set Sr0, r30
.set Ax, r30
.set Sr1, r31
.set Bx, r31

stmw     r13,-76(SP)

stw	 r2,save_RTOC(SP)
stw      r3,work_ptr(SP)
stw      r4,iterations(SP)
stw      r4,count(SP)

lwz		tmp,plain_hi(r3)
stw		tmp,P_1(SP)
lwz		tmp,plain_lo(r3)
stw		tmp,P_0(SP)
lwz		tmp,cypher_hi(r3)
stw		tmp,C_1(SP)
lwz		tmp,cypher_lo(r3)
stw		tmp,C_0(SP)
lwz		Lr1,L0_hi(r3)
stw		Lr1,L0_1(SP)
lwz		Lr0,L0_lo(r3)
stw		Lr0,L0_0(SP)

# compute the maximum count before rollover
# limit count to iterations

#count = ~L0_0
#if count > iterations-1
#	count = iterations-1
#count++	# roll over maximum count to 0 (2^32)
li	r5,L0_1
lwbrx	r5,SP,r5
not	r5,r5
subi	r4,r4,1
cmplw	r5,r4
ble	label1
mr	r5,r4
label1:
addi	r5,r5,1
stw	r5,count(SP)
mtctr	r5

start:
#Round 1 is rolled out of the loop to fill the pipe and setup constants.
#
lis      S0r,0xB7E1
lis      Qr,0x9E37
addi     S0r,S0r,0x5163	#S0r = P#
addi     Qr,Qr,0x79B9	#Qr = Q#

# round 1.0 even
rotlwi  Sr0,S0r,3
stw	Sr0,Sr_0(SP)#		# Sr_0 = ...
lwz	Lr0,L0_0(SP)
add     Lr0,Lr0,Sr0
rotlw	Lr0,Lr0,Sr0
stw	Lr0,con0(SP)		# con0 = Lr_0
add     S0r,S0r,Qr

# round 1.1 odd
add     tmp,S0r,Sr0
add     tmp,tmp,Lr0
rotlwi  Sr1,tmp,3
stw	Sr1,Sr_1(SP)		# Sr_1 = rotl3(...)
add     tmp,Sr1,Lr0
stw	tmp,con1(SP)		# con1 =  Sr_1 + Lr_0
lwz	Lr1,L0_1(SP)
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add     S0r,S0r,Qr

# round 1.2 even
add     tmp,S0r,Sr1
stw	tmp,con2(SP)		# con2 = S0_2 + Sr_1
add     tmp,tmp,Lr1
rotlwi  Sr2,tmp,3
add     tmp,Sr2,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_3(SP)
rotlw	Lr0,Lr0,tmp

# round 1.3 odd
add     tmp,S0r,Sr2
add     tmp,tmp,Lr0
rotlwi  Sr3,tmp,3
add     tmp,Sr3,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_4(SP)
rotlw	Lr1,Lr1,tmp

# round 1.4 even
add     tmp,S0r,Sr3
add     tmp,tmp,Lr1
rotlwi  Sr4,tmp,3
add     tmp,Sr4,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_5(SP)
rotlw	Lr0,Lr0,tmp

# round 1.5 odd
add     tmp,S0r,Sr4
add     tmp,tmp,Lr0
rotlwi  Sr5,tmp,3
add     tmp,Sr5,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_6(SP)
rotlw	Lr1,Lr1,tmp

# round 1.6 even
add     tmp,S0r,Sr5
add     tmp,tmp,Lr1
rotlwi  Sr6,tmp,3
add     tmp,Sr6,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_7(SP)
rotlw	Lr0,Lr0,tmp

# round 1.7 odd
add     tmp,S0r,Sr6
add     tmp,tmp,Lr0
rotlwi  Sr7,tmp,3
add     tmp,Sr7,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_8(SP)
rotlw	Lr1,Lr1,tmp

# round 1.8 even
add     tmp,S0r,Sr7
add     tmp,tmp,Lr1
rotlwi  Sr8,tmp,3
add     tmp,Sr8,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_9(SP)
rotlw	Lr0,Lr0,tmp

# round 1.9 odd
add     tmp,S0r,Sr8
add     tmp,tmp,Lr0
rotlwi  Sr9,tmp,3
add     tmp,Sr9,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_10(SP)
rotlw	Lr1,Lr1,tmp

# round 1.10 even
add     tmp,S0r,Sr9
add     tmp,tmp,Lr1
rotlwi  Sr10,tmp,3
add     tmp,Sr10,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_11(SP)
rotlw	Lr0,Lr0,tmp

# round 1.11 odd
add     tmp,S0r,Sr10
add     tmp,tmp,Lr0
rotlwi  Sr11,tmp,3
add     tmp,Sr11,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_12(SP)
rotlw	Lr1,Lr1,tmp

# round 1.12 even
add     tmp,S0r,Sr11
add     tmp,tmp,Lr1
rotlwi  Sr12,tmp,3
add     tmp,Sr12,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_13(SP)
rotlw	Lr0,Lr0,tmp

# round 1.13 odd
add     tmp,S0r,Sr12
add     tmp,tmp,Lr0
rotlwi  Sr13,tmp,3
add     tmp,Sr13,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_14(SP)
rotlw	Lr1,Lr1,tmp

# round 1.14 even
add     tmp,S0r,Sr13
add     tmp,tmp,Lr1
rotlwi  Sr14,tmp,3
add     tmp,Sr14,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_15(SP)
rotlw	Lr0,Lr0,tmp

# round 1.15 odd
add     tmp,S0r,Sr14
add     tmp,tmp,Lr0
rotlwi  Sr15,tmp,3
add     tmp,Sr15,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_16(SP)
rotlw	Lr1,Lr1,tmp

# round 1.16 even
add     tmp,S0r,Sr15
add     tmp,tmp,Lr1
rotlwi  Sr16,tmp,3
add     tmp,Sr16,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_17(SP)
rotlw	Lr0,Lr0,tmp

# round 1.17 odd
add     tmp,S0r,Sr16
add     tmp,tmp,Lr0
rotlwi  Sr17,tmp,3
add     tmp,Sr17,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_18(SP)
rotlw	Lr1,Lr1,tmp

# round 1.18 even
add     tmp,S0r,Sr17
add     tmp,tmp,Lr1
rotlwi  Sr18,tmp,3
add     tmp,Sr18,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_19(SP)
rotlw	Lr0,Lr0,tmp

# round 1.19 odd
add     tmp,S0r,Sr18
add     tmp,tmp,Lr0
rotlwi  Sr19,tmp,3
add     tmp,Sr19,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_20(SP)
rotlw	Lr1,Lr1,tmp

# round 1.20 even
add     tmp,S0r,Sr19
add     tmp,tmp,Lr1
rotlwi  Sr20,tmp,3
add     tmp,Sr20,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_21(SP)
rotlw	Lr0,Lr0,tmp

# round 1.21 odd
add     tmp,S0r,Sr20
add     tmp,tmp,Lr0
rotlwi  Sr21,tmp,3
add     tmp,Sr21,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_22(SP)
rotlw	Lr1,Lr1,tmp

# round 1.22 even
add     tmp,S0r,Sr21
add     tmp,tmp,Lr1
rotlwi  Sr22,tmp,3
add     tmp,Sr22,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_23(SP)
rotlw	Lr0,Lr0,tmp

# round 1.23 odd
add     tmp,S0r,Sr22
add     tmp,tmp,Lr0
rotlwi  Sr23,tmp,3
add     tmp,Sr23,Lr0
add     S0r,S0r,Qr
add     Lr1,Lr1,tmp
stw	S0r,S0_24(SP)
rotlw	Lr1,Lr1,tmp

# round 1.24 even
add     tmp,S0r,Sr23
add     tmp,tmp,Lr1
rotlwi  Sr24,tmp,3
add     tmp,Sr24,Lr1
add     S0r,S0r,Qr
add     Lr0,Lr0,tmp
stw	S0r,S0_25(SP)
rotlw	Lr0,Lr0,tmp

# round 1.25 odd
add     tmp,S0r,Sr24
add     tmp,tmp,Lr0
rotlwi  Sr25,tmp,3
add     tmp,Sr25,Lr0
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp

#// Note:
#// Round 2 of the main loop contains 26 fill instructions
#// that could be used to generate the stored constants for
#// the next iteration so we could use the full count in cnt.
#// We should definatly use this channel to do the reverse byte
#// increment of L0_1 each round.

fix_key:
#// If the count in L0_1 will roll over into L0_0 in the next itteration
#// the stored constants will need to be fixed. An easy alternative is to
#// jump back to the start and do round 1 over with the correct key.
#// A better solution would be to not allow the rollover in the first place!

lwz		Sr0,Sr_0(SP)

loop:

# round 2.0 even
#	-fill??-
add     Sr0,Sr0,Sr25
add     tmp,Sr0,Lr1
lwz	Sr1,Sr_1(SP)
rotlwi  Sr0,tmp,3
add	Ls0,Lr0,Lr1
add     tmp,Sr0,Lr1
add     Ls0,Sr0,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.1 odd
add     Sr1,Sr1,Sr0
add     tmp,Sr1,Ls0
#	-fill-
rotlwi  Sr1,tmp,3
add	Ls1,Lr1,Ls0
add     tmp,Sr1,Ls0
add     Ls1,Sr1,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.2 even
add     Sr2,Sr2,Sr1
add     tmp,Sr2,Ls1
#	-fill-
rotlwi  Sr2,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr2,Ls1
add     Ls0,Sr2,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.3 odd
add     Sr3,Sr3,Sr2
add     tmp,Sr3,Ls0
#	-fill-
rotlwi  Sr3,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr3,Ls0
add     Ls1,Sr3,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.4 even
add     Sr4,Sr4,Sr3
add     tmp,Sr4,Ls1
#	-fill-
rotlwi  Sr4,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr4,Ls1
add     Ls0,Sr4,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.5 odd
add     Sr5,Sr5,Sr4
add     tmp,Sr5,Ls0
#	-fill-
rotlwi  Sr5,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr5,Ls0
add     Ls1,Sr5,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.6 even
add     Sr6,Sr6,Sr5
add     tmp,Sr6,Ls1
#	-fill-
rotlwi  Sr6,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr6,Ls1
add     Ls0,Sr6,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.7 odd
add     Sr7,Sr7,Sr6
add     tmp,Sr7,Ls0
#	-fill-
rotlwi  Sr7,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr7,Ls0
add     Ls1,Sr7,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.8 even
add     Sr8,Sr8,Sr7
add     tmp,Sr8,Ls1
#	-fill-
rotlwi  Sr8,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr8,Ls1
add     Ls0,Sr8,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.9 odd
add     Sr9,Sr9,Sr8
add     tmp,Sr9,Ls0
#	-fill-
rotlwi  Sr9,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr9,Ls0
add     Ls1,Sr9,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.10 even
add     Sr10,Sr10,Sr9
add     tmp,Sr10,Ls1
#	-fill-
rotlwi  Sr10,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr10,Ls1
add     Ls0,Sr10,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.11 odd
add     Sr11,Sr11,Sr10
add     tmp,Sr11,Ls0
#	-fill-
rotlwi  Sr11,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr11,Ls0
add     Ls1,Sr11,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.12 even
add     Sr12,Sr12,Sr11
add     tmp,Sr12,Ls1
#	-fill-
rotlwi  Sr12,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr12,Ls1
add     Ls0,Sr12,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.13 odd
add     Sr13,Sr13,Sr12
add     tmp,Sr13,Ls0
#	-fill-
rotlwi  Sr13,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr13,Ls0
add     Ls1,Sr13,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.14 even
add     Sr14,Sr14,Sr13
add     tmp,Sr14,Ls1
#	-fill-
rotlwi  Sr14,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr14,Ls1
add     Ls0,Sr14,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.15 odd
add     Sr15,Sr15,Sr14
add     tmp,Sr15,Ls0
#	-fill-
rotlwi  Sr15,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr15,Ls0
add     Ls1,Sr15,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.16 even
add     Sr16,Sr16,Sr15
add     tmp,Sr16,Ls1
li	Lr0,L0_1
rotlwi  Sr16,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr16,Ls1
add     Ls0,Sr16,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.17 odd
add     Sr17,Sr17,Sr16
add     tmp,Sr17,Ls0
lwbrx	Lr1,SP,Lr0
rotlwi  Sr17,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr17,Ls0
add     Ls1,Sr17,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.18 even
add     Sr18,Sr18,Sr17
add     tmp,Sr18,Ls1
addi	Lr1,Lr1,1
rotlwi  Sr18,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr18,Ls1
add     Ls0,Sr18,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.19 odd
add     Sr19,Sr19,Sr18
add     tmp,Sr19,Ls0
stwbrx	Lr1,SP,Lr0
rotlwi  Sr19,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr19,Ls0
add     Ls1,Sr19,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.20 even
add     Sr20,Sr20,Sr19
add     tmp,Sr20,Ls1
lwz	Lr1,L0_1(SP)	# key.hi
rotlwi  Sr20,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr20,Ls1
add     Ls0,Sr20,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.21 odd
add     Sr21,Sr21,Sr20
add     tmp,Sr21,Ls0
#	-fill-
lwz	Lr0,con1(SP)	# const Lr_0 + Sr_1
rotlwi  Sr21,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr21,Ls0
add     Ls1,Sr21,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.22 even
add     Sr22,Sr22,Sr21
add     tmp,Sr22,Ls1
add     Lr1,Lr1,Lr0
rotlwi  Sr22,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr22,Ls1
add     Ls0,Sr22,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.23 odd
add     Sr23,Sr23,Sr22
add     tmp,Sr23,Ls0
rotlw	Lr1,Lr1,Lr0
rotlwi  Sr23,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr23,Ls0
add     Ls1,Sr23,Ls1
rotlw	Ls1,Ls1,tmp

# round 2.24 even
add     Sr24,Sr24,Sr23
add     tmp,Sr24,Ls1
lwz	Lr0,con0(SP)	# const Lr_0
rotlwi  Sr24,tmp,3
add	Ls0,Ls0,Ls1
add     tmp,Sr24,Ls1
add     Ls0,Sr24,Ls0
rotlw	Ls0,Ls0,tmp

# round 2.25 odd
add     Sr25,Sr25,Sr24
add     tmp,Sr25,Ls0
#	-fill-
rotlwi  Sr25,tmp,3
add	Ls1,Ls1,Ls0
add     tmp,Sr25,Ls0
add     Ls1,Sr25,Ls1
rotlw	Ls1,Ls1,tmp
#//	stw		Sr25,Ss_25(SP)	# Ss_25 will most often never get read

# Note register lineup Sr25->At, Ls0->Lt0, Ls1->Lt1, Sr0->Ax, Sr1->Bx
# Registers:	r0	r2	r3	...	r25	r26	r27	r28
#	r29	r30	r31
# usage			tmp	S2	S3		S25	Lr0	Lr1
#	Ls0	Ls1	S0	S1
#								At
#		Lt0	Lt1	Ax	Bx

#// fix the holes in rounds 3.0,3.1
# round 3.0 even
add     At,Sr25,Sr0
lwz	Ax,P_0(SP)
add     At,At,Ls1
#		-fill-
add	Lt0,Ls0,Ls1
rotlwi  At,At,3
add     tmp,At,Ls1
add	Lt0,Lt0,At
rotlw	Lt0,Lt0,tmp
add	Ax,Ax,At

# round 3.1 odd
add     At,At,Sr1
lwz	Bx,P_1(SP)
add     At,At,Lt0
#	-fill-
add	Lt1,Lt1,Lt0
rotlwi  At,At,3
add     tmp,At,Lt0
add	Lt1,Lt1,At
rotlw	Lt1,Lt1,tmp
add	Bx,Bx,At

# round 1.2/3.2 even
add     At,At,Sr2
lwz	Sr2,con2(SP)	# const S0_2 + Sr_1
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr2,Sr2,Lr1
add     tmp,At,Lt1
rotlwi  Sr2,Sr2,3
add     Lt0,Lt0,tmp
#	-fill-
rotlw	Lt0,Lt0,tmp
add     tmp,Sr2,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.3/3.3 odd
add     At,At,Sr3
lwz     Sr3,S0_3(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr3,Sr3,Sr2
add     Sr3,Sr3,Lr0
add     tmp,At,Lt0
rotlwi  Sr3,Sr3,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr3,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.4/3.4 even
add     At,At,Sr4
lwz     Sr4,S0_4(SP)
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr4,Sr4,Sr3
add     Sr4,Sr4,Lr1
add     tmp,At,Lt1
rotlwi  Sr4,Sr4,3
add     Lt0,Lt0,tmp
rotlw	Lt0,Lt0,tmp
add     tmp,Sr4,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.5/3.5 odd
add     At,At,Sr5
lwz     Sr5,S0_5(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr5,Sr5,Sr4
add     Sr5,Sr5,Lr0
add     tmp,At,Lt0
rotlwi  Sr5,Sr5,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr5,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.6/3.6 even
add     At,At,Sr6
lwz     Sr6,S0_6(SP)
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr6,Sr6,Sr5
add     Sr6,Sr6,Lr1
add     tmp,At,Lt1
rotlwi  Sr6,Sr6,3
add     Lt0,Lt0,tmp
rotlw	Lt0,Lt0,tmp
add     tmp,Sr6,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.7/3.7 odd
add     At,At,Sr7
lwz     Sr7,S0_7(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr7,Sr7,Sr6
add     Sr7,Sr7,Lr0
add     tmp,At,Lt0
rotlwi  Sr7,Sr7,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr7,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.8/3.8 even
add     At,At,Sr8
lwz     Sr8,S0_8(SP)
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr8,Sr8,Sr7
add     Sr8,Sr8,Lr1
add     tmp,At,Lt1
rotlwi  Sr8,Sr8,3
add     Lt0,Lt0,tmp
rotlw	Lt0,Lt0,tmp
add     tmp,Sr8,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.9/3.9 odd
add     At,At,Sr9
lwz     Sr9,S0_9(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr9,Sr9,Sr8
add     Sr9,Sr9,Lr0
add     tmp,At,Lt0
rotlwi  Sr9,Sr9,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr9,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.10/3.10 even
add     At,At,Sr10
lwz     Sr10,S0_10(SP)
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr10,Sr10,Sr9
add     Sr10,Sr10,Lr1
add     tmp,At,Lt1
rotlwi  Sr10,Sr10,3
add     Lt0,Lt0,tmp
rotlw	Lt0,Lt0,tmp
add     tmp,Sr10,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.11/3.11 odd
add     At,At,Sr11
lwz     Sr11,S0_11(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr11,Sr11,Sr10
add     Sr11,Sr11,Lr0
add     tmp,At,Lt0
rotlwi  Sr11,Sr11,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr11,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.12/3.12 even
add     At,At,Sr12
lwz     Sr12,S0_12(SP)
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr12,Sr12,Sr11
add     Sr12,Sr12,Lr1
add     tmp,At,Lt1
rotlwi  Sr12,Sr12,3
add     Lt0,Lt0,tmp
rotlw	Lt0,Lt0,tmp
add     tmp,Sr12,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.13/3.13 odd
add     At,At,Sr13
lwz     Sr13,S0_13(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr13,Sr13,Sr12
add     Sr13,Sr13,Lr0
add     tmp,At,Lt0
rotlwi  Sr13,Sr13,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr13,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.14/3.14 even
add     At,At,Sr14
lwz     Sr14,S0_14(SP)
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr14,Sr14,Sr13
add     Sr14,Sr14,Lr1
add     tmp,At,Lt1
rotlwi  Sr14,Sr14,3
add     Lt0,Lt0,tmp
rotlw	Lt0,Lt0,tmp
add     tmp,Sr14,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.15/3.15 odd
add     At,At,Sr15
lwz     Sr15,S0_15(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr15,Sr15,Sr14
add     Sr15,Sr15,Lr0
add     tmp,At,Lt0
rotlwi  Sr15,Sr15,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr15,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.16/3.16 even
add     At,At,Sr16
lwz     Sr16,S0_16(SP)
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr16,Sr16,Sr15
add     Sr16,Sr16,Lr1
add     tmp,At,Lt1
rotlwi  Sr16,Sr16,3
add     Lt0,Lt0,tmp
rotlw	Lt0,Lt0,tmp
add     tmp,Sr16,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.17/3.17 odd
add     At,At,Sr17
lwz     Sr17,S0_17(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr17,Sr17,Sr16
add     Sr17,Sr17,Lr0
add     tmp,At,Lt0
rotlwi  Sr17,Sr17,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr17,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.18/3.18 even
add     At,At,Sr18
lwz     Sr18,S0_18(SP)
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr18,Sr18,Sr17
add     Sr18,Sr18,Lr1
add     tmp,At,Lt1
rotlwi  Sr18,Sr18,3
add     Lt0,Lt0,tmp
rotlw	Lt0,Lt0,tmp
add     tmp,Sr18,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.19/3.19 odd
add     At,At,Sr19
lwz     Sr19,S0_19(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr19,Sr19,Sr18
add     Sr19,Sr19,Lr0
add     tmp,At,Lt0
rotlwi  Sr19,Sr19,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr19,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.20/3.20 even
add     At,At,Sr20
lwz     Sr20,S0_20(SP)
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr20,Sr20,Sr19
add     Sr20,Sr20,Lr1
add     tmp,At,Lt1
rotlwi  Sr20,Sr20,3
add     Lt0,Lt0,tmp
rotlw	Lt0,Lt0,tmp
add     tmp,Sr20,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.21/3.21 odd
add     At,At,Sr21
lwz     Sr21,S0_21(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr21,Sr21,Sr20
add     Sr21,Sr21,Lr0
add     tmp,At,Lt0
rotlwi  Sr21,Sr21,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr21,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.22/3.22 even
add     At,At,Sr22
lwz     Sr22,S0_22(SP)
xor	Ax,Ax,Bx
add     At,At,Lt1
rotlwi  At,At,3
add     Sr22,Sr22,Sr21
add     Sr22,Sr22,Lr1
add     tmp,At,Lt1
rotlwi  Sr22,Sr22,3
add     Lt0,Lt0,tmp
rotlw	Lt0,Lt0,tmp
add     tmp,Sr22,Lr1
rotlw	Ax,Ax,Bx
add     Lr0,Lr0,tmp
rotlw	Lr0,Lr0,tmp
add	Ax,Ax,At

# round 1.23/3.23 odd
add     At,At,Sr23
lwz     Sr23,S0_23(SP)
xor	Bx,Bx,Ax
add     At,At,Lt0
rotlwi  At,At,3
add     Sr23,Sr23,Sr22
add     Sr23,Sr23,Lr0
add     tmp,At,Lt0
rotlwi  Sr23,Sr23,3
add     Lt1,Lt1,tmp
rotlw	Lt1,Lt1,tmp
add     tmp,Sr23,Lr0
rotlw	Bx,Bx,Ax
add     Lr1,Lr1,tmp
rotlw	Lr1,Lr1,tmp
add	Bx,Bx,At

# round 1.24/3.24p even
add     At,At,Sr24
lwz     Sr24,S0_24(SP)
add     At,At,Lt1
xor	Ax,Ax,Bx
add     Sr24,Sr24,Sr23
rotlwi  At,At,3
add     Sr24,Sr24,Lr1
rotlw	Ax,Ax,Bx
rotlwi  Sr24,Sr24,3
add	Ax,Ax,At
lwz	Lt0,C_0(SP)
add     tmp,Sr24,Lr1
lwz     Sr25,S0_25(SP)
add     Lr0,Lr0,tmp
cmpw	Ax,Lt0
rotlw	Lr0,Lr0,tmp


#// Note: all round 3 registers are now available
# round 1.25
add     Sr25,Sr25,Sr24
lwz	Sr0,Sr_0(SP)
add     Sr25,Sr25,Lr0
#	-fill-
rotlwi  Sr25,Sr25,3
add	Lr1,Lr1,Lr0
add     tmp,Sr25,Lr0
add     Lr1,Lr1,Sr25
rotlw	Lr1,Lr1,tmp
#	-fill?-

bdnzf	2,loop

#// return the count of keys tested
mfctr	r3	# residual count
lwz	r4,count(SP)
sub	r3,r4,r3	# iterations run

li	r5,L0_1
lwbrx	Lr1,SP,r5	# L0_hi reversed
li	r6,L0_0
lwbrx	Lr0,SP,r6	# L0_lo reversed

#// If we got a hit
#	bnz	label2
bt	0, label2
subi	Lr1,Lr1,1	#//undo the last increment of the key in L0_1
subi	r3,r3,1		#//decrement the iterations done
b		exit
label2:

#// If L0_1 == 0x00000000 // end of block reached
cmpwi	Lr1,0
bne	exit	#// note: count != iterations is a bug except at rollover
addi	Lr0,Lr0,1	# increment L0_0 (byte reversed of course)
#// Test for key range overlapping the block boundary
lwz		r4,iterations(SP)
cmpw	r3,r4
beq	exit
#//	fix the count and do it again
subf	r3,r3,r4		# iterations - count -> ctr
mtctr	r3
stw	r4,count(SP)	# iterations -> count
li	r5,L0_1
stwbrx	Lr1,SP,r5		# set the current key
li	r6,L0_0
stwbrx	Lr0,SP,r6
b	start

exit:
#// save the last key tested in the work record
lwz	r4,work_ptr(SP)
li	r5,L0_lo
stwbrx	Lr0,r4,r5
li	r5,L0_hi
stwbrx	Lr1,r4,r5

lwz		r2,save_RTOC(SP)
lmw      r13,-76(SP)
blr

