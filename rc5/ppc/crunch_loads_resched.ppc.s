# GNU C++ 2.7.2
# -O6 -fdefer-pop -fomit-frame-pointer -fcse-follow-jumps -fcse-skip-blocks
# -fexpensive-optimizations -fthread-jumps -fstrength-reduce -funroll-loops
# -fpeephole -fforce-mem -ffunction-cse -finline-functions -finline
# -fcaller-saves -fpcc-struct-return -frerun-cse-after-loop -fschedule-insns
# -fschedule-insns2 -fcommon -fgnu-linker -Wunused -Wswitch -mpowerpc
# -mnew-mnemonics

	.file	"crunch_loads_resched.cpp"
gcc2_compiled.:

initKS:
        .long -1089828067
        .long 354637369
        .long -196066091
        .long -1836597618
        .long 817838151
        .long -822693376
        .long 1831742393
        .long 191210866
        .long -1449320661
        .long 1205115108
        .long -435416419
        .long -2075947946
        .long 578487823
        .long -1062043704
        .long 1592392065
        .long -48139462
        .long -1688670989
        .long 965764780
        .long -674766747
        .long 1979669022
        .long 339137495
        .long -1301394032
        .long 1353041737
        .long -287489790
        .long -1928021317
        .long 726414452

	.globl crunch_loads_resched
	.type	 crunch_loads_resched,@function
#	.long 0x4008
crunch_loads_resched:

	
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


# S-box offsets
.set dataptr, r3
.set s0, r4
.set s1, s0+1
.set s2, s1+1
.set s3, s2+1
.set s4, s3+1
.set s5, s4+1
.set s6, s5+1
.set s7, s6+1
.set s8, s7+1
.set s9, s8+1
.set s10, s9+1
.set s11, s10+1
.set s12, s11+1
.set s13, s12+1
.set s14, s13+1
.set s15, s14+1
.set s16, s15+1
.set s17, s16+1
.set s18, s17+1
.set s19, s18+1
.set s20, s19+1
.set s21, s20+1
.set s22, s21+1
.set s23, s22+1
.set s24, s23+1
.set s25, r2
.set initptr, s24+1	

# other register assignments
.set t1, r0
.set rA, initptr
.set rB, s0
.set key0, r30
.set key1, r31
.set keylo, key0
.set keyhi, key1
.set targetA, s1
.set targetB, s2

prologue:	
		  stw		r31,-4(SP)
		  stw		r30,-8(SP)
		  stw		r29,-12(SP)
		  stw		r28,-16(SP)
		  stw		r27,-20(SP)
		  stw		r26,-24(SP)
		  stw		r25,-28(SP)
		  stw		r24,-32(SP)
		  stw		r23,-36(SP)
		  stw		r22,-40(SP)
		  stw		r21,-44(SP)
		  stw		r20,-48(SP)
		  stw		r19,-52(SP)
		  stw		r18,-56(SP)
		  stw		r17,-60(SP)
		  stw		r16,-64(SP)
		  stw		r15,-68(SP)
		  stw		r14,-72(SP)
		  stw		r13,-76(SP)
		  stw		r2,-80(SP)
		  ## CR at -84?
		  # input iterations at -88
		  # initptr at -92
		  
		  stw		r4,-88(SP)
		  mtctr	r4
		
#		lwz		initptr,initKS(rtoc)
		addis		9,0,initKS@ha
		addi		initptr,9,initKS@l
		
		lwz		keylo,L0_lo(dataptr)
		lwz		keyhi,L0_hi(dataptr)
		stw		initptr,-92(SP)
		
checkkey:	
pass1:	
		lwz		s0,0(initptr)
		lis		s1,0x1523
		add		key0,key0,s0
		ori		s1,s1,0x5639
		rlwnm	key0,key0,s0,0,31
pass1a:
		lwz		s2,2*4(initptr)		
		
		add		s1,s1,key0
		add		key1,key1,key0
		rlwinm	s1,s1,3,0,31
		lwz		s3,3*4(initptr)
		add		t1,key0,s1
		add		key1,key1,s1
		add		s2,s2,s1
		rlwnm	key1,key1,t1,0,31
		
		add		s2,s2,key1
		add		key0,key0,key1
		rlwinm	s2,s2,3,0,31
		lwz		s4,4*4(initptr)
		add		t1,key1,s2
		add		key0,key0,s2
		add		s3,s3,s2
		rlwnm	key0,key0,t1,0,31
		
		add		s3,s3,key0
		add		key1,key1,key0
		rlwinm	s3,s3,3,0,31
		lwz		s5,5*4(initptr)
		add		t1,key0,s3
		add		key1,key1,s3
		add		s4,s4,s3
		rlwnm	key1,key1,t1,0,31
		
		add		s4,s4,key1
		add		key0,key0,key1
		rlwinm	s4,s4,3,0,31
		lwz		s6,6*4(initptr)
		add		t1,key1,s4
		add		key0,key0,s4
		add		s5,s5,s4
		rlwnm	key0,key0,t1,0,31
		
		add		s5,s5,key0
		add		key1,key1,key0
		rlwinm	s5,s5,3,0,31
		lwz		s7,7*4(initptr)
		add		t1,key0,s5
		add		key1,key1,s5
		add		s6,s6,s5
		rlwnm	key1,key1,t1,0,31
		
		add		s6,s6,key1
		add		key0,key0,key1
		rlwinm	s6,s6,3,0,31
		lwz		s8,8*4(initptr)
		add		t1,key1,s6
		add		key0,key0,s6
		add		s7,s7,s6
		rlwnm	key0,key0,t1,0,31
		
		add		s7,s7,key0
		add		key1,key1,key0
		rlwinm	s7,s7,3,0,31
		lwz		s9,9*4(initptr)
		add		t1,key0,s7
		add		key1,key1,s7
		add		s8,s8,s7
		rlwnm	key1,key1,t1,0,31
		
		add		s8,s8,key1
		add		key0,key0,key1
		rlwinm	s8,s8,3,0,31
		lwz		s10,10*4(initptr)
		add		t1,key1,s8
		add		key0,key0,s8
		add		s9,s9,s8
		rlwnm	key0,key0,t1,0,31
		
		add		s9,s9,key0
		add		key1,key1,key0
		rlwinm	s9,s9,3,0,31
		lwz		s11,11*4(initptr)
		add		t1,key0,s9
		add		key1,key1,s9
		add		s10,s10,s9
		rlwnm	key1,key1,t1,0,31
		
		add		s10,s10,key1
		add		key0,key0,key1
		rlwinm	s10,s10,3,0,31
		lwz		s12,12*4(initptr)
		add		t1,key1,s10
		add		key0,key0,s10
		add		s11,s11,s10
		rlwnm	key0,key0,t1,0,31
		
		add		s11,s11,key0
		add		key1,key1,key0
		rlwinm	s11,s11,3,0,31
		lwz		s13,13*4(initptr)
		add		t1,key0,s11
		add		key1,key1,s11
		add		s12,s12,s11
		rlwnm	key1,key1,t1,0,31
		
		add		s12,s12,key1
		add		key0,key0,key1
		rlwinm	s12,s12,3,0,31
		lwz		s14,14*4(initptr)
		add		t1,key1,s12
		add		key0,key0,s12
		add		s13,s13,s12
		rlwnm	key0,key0,t1,0,31
		
		add		s13,s13,key0
		add		key1,key1,key0
		rlwinm	s13,s13,3,0,31
		lwz		s15,15*4(initptr)
		add		t1,key0,s13
		add		key1,key1,s13
		add		s14,s14,s13
		rlwnm	key1,key1,t1,0,31
		
		add		s14,s14,key1
		add		key0,key0,key1
		rlwinm	s14,s14,3,0,31
		lwz		s16,16*4(initptr)
		add		t1,key1,s14
		add		key0,key0,s14
		add		s15,s15,s14
		rlwnm	key0,key0,t1,0,31
		
		add		s15,s15,key0
		add		key1,key1,key0
		rlwinm	s15,s15,3,0,31
		lwz		s17,17*4(initptr)
		add		t1,key0,s15
		add		key1,key1,s15
		add		s16,s16,s15
		rlwnm	key1,key1,t1,0,31
		
		add		s16,s16,key1
		add		key0,key0,key1
		rlwinm	s16,s16,3,0,31
		lwz		s18,18*4(initptr)
		add		t1,key1,s16
		add		key0,key0,s16
		add		s17,s17,s16
		rlwnm	key0,key0,t1,0,31
		
		add		s17,s17,key0
		add		key1,key1,key0
		rlwinm	s17,s17,3,0,31
		lwz		s19,19*4(initptr)
		add		t1,key0,s17
		add		key1,key1,s17
		add		s18,s18,s17
		rlwnm	key1,key1,t1,0,31
		
		add		s18,s18,key1
		add		key0,key0,key1
		rlwinm	s18,s18,3,0,31
		lwz		s20,20*4(initptr)
		add		t1,key1,s18
		add		key0,key0,s18
		add		s19,s19,s18
		rlwnm	key0,key0,t1,0,31
		
		add		s19,s19,key0
		add		key1,key1,key0
		rlwinm	s19,s19,3,0,31
		lwz		s21,21*4(initptr)
		add		t1,key0,s19
		add		key1,key1,s19
		add		s20,s20,s19
		rlwnm	key1,key1,t1,0,31
		
		add		s20,s20,key1
		add		key0,key0,key1
		rlwinm	s20,s20,3,0,31
		lwz		s22,22*4(initptr)
		add		t1,key1,s20
		add		key0,key0,s20
		add		s21,s21,s20
		rlwnm	key0,key0,t1,0,31
		
		add		s21,s21,key0
		add		key1,key1,key0
		rlwinm	s21,s21,3,0,31
		lwz		s23,23*4(initptr)
		add		t1,key0,s21
		add		key1,key1,s21
		add		s22,s22,s21
		rlwnm	key1,key1,t1,0,31
		
		add		s22,s22,key1
		add		key0,key0,key1
		rlwinm	s22,s22,3,0,31
		lwz		s24,24*4(initptr)
		add		t1,key1,s22
		add		key0,key0,s22
		add		s23,s23,s22
		rlwnm	key0,key0,t1,0,31
		
		add		s23,s23,key0
		add		key1,key1,key0
		rlwinm	s23,s23,3,0,31
		lwz		s25,25*4(initptr)
		add		t1,key0,s23
		add		key1,key1,s23
		add		s24,s24,s23
		rlwnm	key1,key1,t1,0,31
		
		add		s24,s24,key1
		add		key0,key0,key1
		rlwinm	s24,s24,3,0,31
		add		t1,key1,s24
		add		key0,key0,s24
		add		s25,s25,s24
		rlwnm	key0,key0,t1,0,31
		
		add		s25,s25,key0
		add		key1,key1,key0
		rlwinm	s25,s25,3,0,31
		add		t1,key0,s25
		add		key1,key1,s25
		add		s0,s0,s25
		rlwnm	key1,key1,t1,0,31
		
pass2:	
		add		s0,s0,key1
		add		key0,key0,key1
		rlwinm	s0,s0,3,0,31
		add		t1,key1,s0
		add		key0,key0,s0
		add		s1,s1,s0
		rlwnm	key0,key0,t1,0,31
		
		add		s1,s1,key0
		add		key1,key1,key0
		rlwinm	s1,s1,3,0,31
		add		t1,key0,s1
		add		key1,key1,s1
		add		s2,s2,s1
		rlwnm	key1,key1,t1,0,31
		
		add		s2,s2,key1
		add		key0,key0,key1
		rlwinm	s2,s2,3,0,31
		add		t1,key1,s2
		add		key0,key0,s2
		add		s3,s3,s2
		rlwnm	key0,key0,t1,0,31
		
		add		s3,s3,key0
		add		key1,key1,key0
		rlwinm	s3,s3,3,0,31
		add		t1,key0,s3
		add		key1,key1,s3
		add		s4,s4,s3
		rlwnm	key1,key1,t1,0,31
		
		add		s4,s4,key1
		add		key0,key0,key1
		rlwinm	s4,s4,3,0,31
		add		t1,key1,s4
		add		key0,key0,s4
		add		s5,s5,s4
		rlwnm	key0,key0,t1,0,31
		
		add		s5,s5,key0
		add		key1,key1,key0
		rlwinm	s5,s5,3,0,31
		add		t1,key0,s5
		add		key1,key1,s5
		add		s6,s6,s5
		rlwnm	key1,key1,t1,0,31
		
		add		s6,s6,key1
		add		key0,key0,key1
		rlwinm	s6,s6,3,0,31
		add		t1,key1,s6
		add		key0,key0,s6
		add		s7,s7,s6
		rlwnm	key0,key0,t1,0,31
		
		add		s7,s7,key0
		add		key1,key1,key0
		rlwinm	s7,s7,3,0,31
		add		t1,key0,s7
		add		key1,key1,s7
		add		s8,s8,s7
		rlwnm	key1,key1,t1,0,31
		
		add		s8,s8,key1
		add		key0,key0,key1
		rlwinm	s8,s8,3,0,31
		add		t1,key1,s8
		add		key0,key0,s8
		add		s9,s9,s8
		rlwnm	key0,key0,t1,0,31
		
		add		s9,s9,key0
		add		key1,key1,key0
		rlwinm	s9,s9,3,0,31
		add		t1,key0,s9
		add		key1,key1,s9
		add		s10,s10,s9
		rlwnm	key1,key1,t1,0,31
		
		add		s10,s10,key1
		add		key0,key0,key1
		rlwinm	s10,s10,3,0,31
		add		t1,key1,s10
		add		key0,key0,s10
		add		s11,s11,s10
		rlwnm	key0,key0,t1,0,31
		
		add		s11,s11,key0
		add		key1,key1,key0
		rlwinm	s11,s11,3,0,31
		add		t1,key0,s11
		add		key1,key1,s11
		add		s12,s12,s11
		rlwnm	key1,key1,t1,0,31
		
		add		s12,s12,key1
		add		key0,key0,key1
		rlwinm	s12,s12,3,0,31
		add		t1,key1,s12
		add		key0,key0,s12
		add		s13,s13,s12
		rlwnm	key0,key0,t1,0,31
		
		add		s13,s13,key0
		add		key1,key1,key0
		rlwinm	s13,s13,3,0,31
		add		t1,key0,s13
		add		key1,key1,s13
		add		s14,s14,s13
		rlwnm	key1,key1,t1,0,31
		
		add		s14,s14,key1
		add		key0,key0,key1
		rlwinm	s14,s14,3,0,31
		add		t1,key1,s14
		add		key0,key0,s14
		add		s15,s15,s14
		rlwnm	key0,key0,t1,0,31
		
		add		s15,s15,key0
		add		key1,key1,key0
		rlwinm	s15,s15,3,0,31
		add		t1,key0,s15
		add		key1,key1,s15
		add		s16,s16,s15
		rlwnm	key1,key1,t1,0,31
		
		add		s16,s16,key1
		add		key0,key0,key1
		rlwinm	s16,s16,3,0,31
		add		t1,key1,s16
		add		key0,key0,s16
		add		s17,s17,s16
		rlwnm	key0,key0,t1,0,31
		
		add		s17,s17,key0
		add		key1,key1,key0
		rlwinm	s17,s17,3,0,31
		add		t1,key0,s17
		add		key1,key1,s17
		add		s18,s18,s17
		rlwnm	key1,key1,t1,0,31
		
		add		s18,s18,key1
		add		key0,key0,key1
		rlwinm	s18,s18,3,0,31
		add		t1,key1,s18
		add		key0,key0,s18
		add		s19,s19,s18
		rlwnm	key0,key0,t1,0,31
		
		add		s19,s19,key0
		add		key1,key1,key0
		rlwinm	s19,s19,3,0,31
		add		t1,key0,s19
		add		key1,key1,s19
		add		s20,s20,s19
		rlwnm	key1,key1,t1,0,31
		
		add		s20,s20,key1
		add		key0,key0,key1
		rlwinm	s20,s20,3,0,31
		add		t1,key1,s20
		add		key0,key0,s20
		add		s21,s21,s20
		rlwnm	key0,key0,t1,0,31
		
		add		s21,s21,key0
		add		key1,key1,key0
		rlwinm	s21,s21,3,0,31
		add		t1,key0,s21
		add		key1,key1,s21
		add		s22,s22,s21
		rlwnm	key1,key1,t1,0,31
		
		add		s22,s22,key1
		add		key0,key0,key1
		rlwinm	s22,s22,3,0,31
		add		t1,key1,s22
		add		key0,key0,s22
		add		s23,s23,s22
		rlwnm	key0,key0,t1,0,31
		
		add		s23,s23,key0
		add		key1,key1,key0
		rlwinm	s23,s23,3,0,31
		add		t1,key0,s23
		add		key1,key1,s23
		add		s24,s24,s23
		rlwnm	key1,key1,t1,0,31
		
		add		s24,s24,key1
		add		key0,key0,key1
		rlwinm	s24,s24,3,0,31
		add		t1,key1,s24
		add		key0,key0,s24
		add		s25,s25,s24
		rlwnm	key0,key0,t1,0,31
		
		add		s25,s25,key0
		add		key1,key1,key0
		rlwinm	s25,s25,3,0,31
			lwz		rA,plain_lo(dataptr)
		add		t1,key0,s25
		add		key1,key1,s25
		add		s0,s0,s25
		rlwnm	key1,key1,t1,0,31
pass3:	
		add		s0,s0,key1
		add		key0,key0,key1
		rlwinm	s0,s0,3,0,31
		add		t1,key1,s0
		add		key0,key0,s0
		add		rA,rA,s0
		add		s1,s1,s0
		rlwnm	key0,key0,t1,0,31
		
		lwz		rB,plain_hi(dataptr)
		
		add		s1,s1,key0
		add		key1,key1,key0
		rlwinm	s1,s1,3,0,31
		add		t1,key0,s1
		add		key1,key1,s1
		add		s2,s2,s1
		add		rB,rB,s1
		rlwnm	key1,key1,t1,0,31
		
		xor		rA,rA,rB
		add		s2,s2,key1
		rlwinm	s2,s2,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s2,key1
		add		key0,key0,t1
		add		rA,rA,s2
		rlwnm	key0,key0,t1,0,31
		
		add		s3,s3,s2
		xor		rB,rB,rA
		add		s3,s3,key0
		rlwinm	s3,s3,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s3,key0
		add		key1,key1,t1
		add		rB,rB,s3
		rlwnm	key1,key1,t1,0,31
		
		add		s4,s4,s3
		xor		rA,rA,rB
		add		s4,s4,key1
		rlwinm	s4,s4,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s4,key1
		add		key0,key0,t1
		add		rA,rA,s4
		rlwnm	key0,key0,t1,0,31
		
		add		s5,s5,s4
		xor		rB,rB,rA
		add		s5,s5,key0
		rlwinm	s5,s5,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s5,key0
		add		key1,key1,t1
		add		rB,rB,s5
		rlwnm	key1,key1,t1,0,31
		
		add		s6,s6,s5
		xor		rA,rA,rB
		add		s6,s6,key1
		rlwinm	s6,s6,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s6,key1
		add		key0,key0,t1
		add		rA,rA,s6
		rlwnm	key0,key0,t1,0,31
		
		add		s7,s7,s6
		xor		rB,rB,rA
		add		s7,s7,key0
		rlwinm	s7,s7,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s7,key0
		add		key1,key1,t1
		add		rB,rB,s7
		rlwnm	key1,key1,t1,0,31
		
		add		s8,s8,s7
		xor		rA,rA,rB
		add		s8,s8,key1
		rlwinm	s8,s8,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s8,key1
		add		key0,key0,t1
		add		rA,rA,s8
		rlwnm	key0,key0,t1,0,31
		
		add		s9,s9,s8
		xor		rB,rB,rA
		add		s9,s9,key0
		rlwinm	s9,s9,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s9,key0
		add		key1,key1,t1
		add		rB,rB,s9
		rlwnm	key1,key1,t1,0,31
		
		add		s10,s10,s9
		xor		rA,rA,rB
		add		s10,s10,key1
		rlwinm	s10,s10,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s10,key1
		add		key0,key0,t1
		add		rA,rA,s10
		rlwnm	key0,key0,t1,0,31
		
		add		s11,s11,s10
		xor		rB,rB,rA
		add		s11,s11,key0
		rlwinm	s11,s11,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s11,key0
		add		key1,key1,t1
		add		rB,rB,s11
		rlwnm	key1,key1,t1,0,31
		
		add		s12,s12,s11
		xor		rA,rA,rB
		add		s12,s12,key1
		rlwinm	s12,s12,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s12,key1
		add		key0,key0,t1
		add		rA,rA,s12
		rlwnm	key0,key0,t1,0,31
		
		add		s13,s13,s12
		xor		rB,rB,rA
		add		s13,s13,key0
		rlwinm	s13,s13,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s13,key0
		add		key1,key1,t1
		add		rB,rB,s13
		rlwnm	key1,key1,t1,0,31
		
		add		s14,s14,s13
		xor		rA,rA,rB
		add		s14,s14,key1
		rlwinm	s14,s14,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s14,key1
		add		key0,key0,t1
		add		rA,rA,s14
		rlwnm	key0,key0,t1,0,31
		
		add		s15,s15,s14
		xor		rB,rB,rA
		add		s15,s15,key0
		rlwinm	s15,s15,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s15,key0
		add		key1,key1,t1
		add		rB,rB,s15
		rlwnm	key1,key1,t1,0,31
		
		add		s16,s16,s15
		xor		rA,rA,rB
		add		s16,s16,key1
		rlwinm	s16,s16,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s16,key1
		add		key0,key0,t1
		add		rA,rA,s16
		rlwnm	key0,key0,t1,0,31
		
		add		s17,s17,s16
		xor		rB,rB,rA
		add		s17,s17,key0
		rlwinm	s17,s17,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s17,key0
		add		key1,key1,t1
		add		rB,rB,s17
		rlwnm	key1,key1,t1,0,31
		
		add		s18,s18,s17
		xor		rA,rA,rB
		add		s18,s18,key1
		rlwinm	s18,s18,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s18,key1
		add		key0,key0,t1
		add		rA,rA,s18
		rlwnm	key0,key0,t1,0,31
		
		add		s19,s19,s18
		xor		rB,rB,rA
		add		s19,s19,key0
		rlwinm	s19,s19,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s19,key0
		add		key1,key1,t1
		add		rB,rB,s19
		rlwnm	key1,key1,t1,0,31
		
		add		s20,s20,s19
		xor		rA,rA,rB
		add		s20,s20,key1
		rlwinm	s20,s20,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s20,key1
		add		key0,key0,t1
		add		rA,rA,s20
		rlwnm	key0,key0,t1,0,31
		
		add		s21,s21,s20
		xor		rB,rB,rA
		add		s21,s21,key0
		rlwinm	s21,s21,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s21,key0
		add		key1,key1,t1
		add		rB,rB,s21
		rlwnm	key1,key1,t1,0,31
		
		add		s22,s22,s21
		xor		rA,rA,rB
		add		s22,s22,key1
		rlwinm	s22,s22,3,0,31
		rlwnm	rA,rA,rB,0,31
		add		t1,s22,key1
		add		key0,key0,t1
		add		rA,rA,s22
		rlwnm	key0,key0,t1,0,31
		
		add		s23,s23,s22
		xor		rB,rB,rA
		add		s23,s23,key0
		lwz		targetA,cypher_lo(dataptr)
		rlwinm	s23,s23,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		t1,s23,key0
		add		key1,key1,t1
		add		rB,rB,s23
		rlwnm	key1,key1,t1,0,31
		
		add		s24,s24,s23
		xor		rA,rA,rB
		add		s24,s24,key1
		rlwnm	rA,rA,rB,0,31
		rlwinm	s24,s24,3,0,31
		add		rA,rA,s24
		cmplw	cr0,rA,targetA
		add		t1,s24,key1
		bne+	cr0,notfound
		
		add		key0,key0,t1
		rlwnm	key0,key0,t1,0,31
		
		
		add		s25,s25,s24
		xor		rB,rB,rA
		add		s25,s25,key0
		rlwinm	s25,s25,3,0,31
		rlwnm	rB,rB,rA,0,31
		add		rB,rB,s25
		
		lwz		targetB,cypher_hi(dataptr)
		
		cmplw	cr1,rB,targetB
		beq-	cr1,keyfound
		
notfound:	
		lwz		keyhi,L0_hi(dataptr)
		addis	keyhi,keyhi,0x100
		rlwinm.	t1,keyhi,0,0,7
		lwz		keylo,L0_lo(dataptr)
		bne+	cycle

                addis	keyhi,keyhi,1
		rlwinm.	t1,keyhi,0,8,15
		rlwinm	keyhi,keyhi,0,8,31
		bne+	cycle
		
		addi	keyhi,keyhi,0x100
		rlwinm.	t1,keyhi,0,16,23
		rlwinm	keyhi,keyhi,0,16,31
		bne+	cycle
		
		addi	keyhi,keyhi,1
		rlwinm.	keyhi,keyhi,0,24,31
		bne+	cycle
		
		addis	keylo,keylo,0x100
		rlwinm.	t1,keylo,0,0,7
		bne+	cycle
		
		addis	keylo,keylo,1
		rlwinm.	t1,keylo,0,8,15
		rlwinm	keylo,keylo,0,8,31
		bne+	cycle
		
		addi	keylo,keylo,0x100
		rlwinm.	t1,keylo,0,16,23
		rlwinm	keylo,keylo,0,16,31
		bne+	cycle
		
		addi	keylo,keylo,1
		rlwinm	keylo,keylo,0,24,31
		
cycle:
		stw		keyhi,L0_hi(dataptr)
		stw		keylo,L0_lo(dataptr)
		lwz		initptr,-92(SP)
		bdnz	checkkey
		
keyfound:	
epilogue:	
		lwz		r4,-88(SP)
		mfctr	r5
		subf	r3,r5,r4
		
		lwz		r31,-4(SP)
		lwz		r30,-8(SP)
		lwz		r29,-12(SP)
		lwz		r28,-16(SP)
		lwz		r27,-20(SP)
		lwz		r26,-24(SP)
		lwz		r25,-28(SP)
		lwz		r24,-32(SP)
		lwz		r23,-36(SP)
		lwz		r22,-40(SP)
		lwz		r21,-44(SP)
		lwz		r20,-48(SP)
		lwz		r19,-52(SP)
		lwz		r18,-56(SP)
		lwz		r17,-60(SP)
		lwz		r16,-64(SP)
		lwz		r15,-68(SP)
		lwz		r14,-72(SP)
		lwz		r13,-76(SP)
		lwz		r2,-80(SP)
		
		blr

.Lfe1:
	.size	 crunch_loads_resched,.Lfe1-crunch_loads_resched
	.ident	"GCC: (GNU) 2.7.2"

.Lfe2:
	.size	 crunch_loads_resched,.Lfe2-crunch_loads_resched
	.ident	"GCC: (GNU) 2.7.2"
