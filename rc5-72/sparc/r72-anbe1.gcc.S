! Copyright distributed.net 2003 - All Rights Reserved
! For use in distributed.net projects only.
! Any other distribution or use of this source violates copyright.
!
! Author: Andreas Beckmann <andreasb@distributed.net>

.section	".rodata"

Id_tag:
	.ascii	"@(#)$Id: r72-anbe1.gcc.S,v 1.1.2.2 2003/04/19 17:33:54 andreasb Exp $\000"


.section	".text"

.global r72_anbe1_s
r72_anbe1_s:

	set	Id_tag, %o0
	retl
	nop


#define PIPELINES	1
#define RESULT_NOTHING	1
#define RESULT_FOUND	2


! extern "C" s32 rc5_72_unit_func_anbe_1( RC5_72UnitWork *, u32 *, void *);
.global rc5_72_unit_func_anbe_1
rc5_72_unit_func_anbe_1:

! Input:
! %i0 = RC5_72UnitWork *
! %i1 = u32* iterations
! %i2 = (unused)

! Output:
! [rc5_72unitwork]
! [iterations]
! %i0 = RESULT_{NOTHING|FOUND}

#define r72unitwork_plain_hi	0
#define r72unitwork_plain_lo	4
#define r72unitwork_cypher_hi	8
#define r72unitwork_cypher_lo	12
#define r72unitwork_L0_hi	16
#define r72unitwork_L0_mid	20
#define r72unitwork_L0_lo	24
#define r72unitwork_check_count	28
#define r72unitwork_check_hi	32
#define r72unitwork_check_mid	36
#define r72unitwork_check_lo	40
#define sizeof_r72unitwork	44


#define default_stackframe	(4*(16+1+6))

#define stack			%sp+default_stackframe

! stack layout:

! <<<=== %fp points here
!   <--- stack copy of r72unitwork (%fp relative, dword aligned)
#define s_save_r72unitworkP	%fp-4
#define s_uw(offset)		%fp-4-sizeof_r72unitwork+(offset)
#define s_save_iterationsP	%fp-4-sizeof_r72unitwork-4
#define fp_relative_size	(4+sizeof_r72unitwork+4)
!   <--? here could be a filler for dword alignment
#define stackframe_size		(default_stackframe+fp_relative_size+4*(26))
! s_S1[26]
#define s_S1(i)			stack+0+(4*i)
#define bar			stack+0
! <<<=== 'stack' points here
! (default stackframe 6+1+16 words)
! <<<=== %sp points here


	save	%sp,-(stackframe_size+(stackframe_size % 8)),%sp
	st	%i0,[s_save_r72unitworkP]
	ld	[%i0+r72unitwork_plain_hi],%l0
	ld	[%i0+r72unitwork_plain_lo],%l1
	ld	[%i0+r72unitwork_cypher_hi],%l2
	ld	[%i0+r72unitwork_cypher_lo],%l3
	ld	[%i0+r72unitwork_check_count],%l4
	ld	[%i0+r72unitwork_check_hi],%l5
	ld	[%i0+r72unitwork_check_mid],%l6
	ld	[%i0+r72unitwork_check_lo],%l7
	st	%l0,[s_uw(r72unitwork_plain_hi)]
	st	%l1,[s_uw(r72unitwork_plain_lo)]
	st	%l2,[s_uw(r72unitwork_cypher_hi)]
	st	%l3,[s_uw(r72unitwork_cypher_lo)]
	st	%l4,[s_uw(r72unitwork_check_count)]
	st	%l5,[s_uw(r72unitwork_check_hi)]
	st	%l6,[s_uw(r72unitwork_check_mid)]
	st	%l7,[s_uw(r72unitwork_check_lo)]
	st	%i1,[s_save_iterationsP]

#define P		0xB7E15163
#define Q		0x9E3779B9
#define P_ROTL3		0xBF0A8B1D

#define RGiterations	%i5
#define RGLlo		%l3
#define RGLmid		%l4
#define	RGLhi		%l5

#define RGSinit		%o7

#define RCP		%o0
#define RCP_ROTL3	%o1
#define RCQ		%o2


#define R1SS		%g1
#define R1L(i)		R1L_##i
#define R1L_0		%l0
#define R1L_1		%l1
#define R1L_2		%l2
#define R1Lpred(i)	R1L##i##pred
#define R1L0pred	R1L_2
#define R1L1pred	R1L_0
#define R1L2pred	R1L_1

#define R1A		%l6
#define R1B		%l7

#define T1a		%o4
#define	T1b		%o5


#define R2SS		%g2
#define R2L(i)		R2L_##i
#define R2L_0		%g3
#define R2L_1		%g4
#define R2L_2		%g5
#define R2Lpred(i)	R2L##i##pred
#define R2L0pred	R2L_2
#define R2L1pred	R2L_0
#define R2L2pred	R2L_1

#define R2A		%g6
#define R2B		%g7

#define T2a		%i3
#define	T2b		%i4


	ld	[%i1],RGiterations
	ld	[%i0+r72unitwork_L0_hi],RGLhi
	ld	[%i0+r72unitwork_L0_mid],RGLmid
	ld	[%i0+r72unitwork_L0_lo],RGLlo

	set	P,RCP
	set	Q,RCQ
	set	P_ROTL3,RCP_ROTL3

	mov	RCP,RGSinit
	mov	RCP_ROTL3,R1SS
	mov	RGLlo,R1L(0)

mainloop:

	!mov	RGLlo,R1L(0)
	mov	RGLmid,R1L(1)
	mov	RGLhi,R1L(2)

! ROUND 1 - key init and round 1

! #define ROTL_BLOCK_j1(i,j) \
!    S[i] = ROTL3(S[i]+(S[i-1]+L[0])); \
!    L[1] = ROTL(L[1]+(S[i]+L[0]),(S[i]+L[0])); 

! Sinit = P
! case i=0
!! S(0) = P
!! S(0) = SS = ROTL3(S(0))
! S(0) = SS = P_ROTL3
! L(0) =      ROTL(L(0)+SS, SS)
!! L(0) =     ROTL(L(0)+SS, P_ROTL3 & 0x1f
! case i>0
! S(i) = P+i*Q
! S(i) = SS = ROTL3(S(i)+(SS+Lpred(i)))
! L(i) =      ROTL(L(i)+(SS+Lpred(i)),(SS+Lpred(i))))

! Sinit	+= Q
! SS	+= Lpred(i)
! SS	+= Sinit
! SS	 = ROTL3(SS)
! S(i)	 = SS
! tmp	 = SS+Lpred(i)
! L(i)	+= tmp
! L(i)   = ROTL(L(i), tmp)

#define ROUND1(i,j)				\
/* ! ROUND 1 (i,j) */				\
	add	RGSinit,RCQ,RGSinit		;\
	add	R1SS,R1Lpred(j),R1SS		;\
	add	R1SS,RGSinit,R1SS		;\
	sll	R1SS,3,T1a			;\
	srl	R1SS,29,R1SS			;\
	or	R1SS,T1a,R1SS			;\
	st	R1SS,[s_S1(i)]			;\
	add	R1SS,R1Lpred(j),T1a		;\
	add	R1L(j),T1a,R1L(j)		;\
	sll	R1L(j),T1a,T1b			;\
	neg	T1a				;\
	srl	R1L(j),T1a,R1L(j)		;\
	or	R1L(j),T1b,R1L(j)		;\


	!mov	RCP,RGSinit

	/* ROUND1(0,0) */
	!mov	RCP_ROTL3,R1SS
	st	R1SS,[s_S1(0)]
	add	R1L(0),R1SS,R1L(0)
	sll	R1L(0),(P_ROTL3 % 32),T1a
	srl	R1L(0),(32 - (P_ROTL3 % 32)),R1L(0)
	or	R1L(0),T1a,R1L(0)
	
	ROUND1(1,1)
	ROUND1(2,2)
	ROUND1(3,0)
	ROUND1(4,1)
	ROUND1(5,2)
	ROUND1(6,0)
	ROUND1(7,1)
	ROUND1(8,2)
	ROUND1(9,0)
	ROUND1(10,1)
	ROUND1(11,2)
	ROUND1(12,0)
	ROUND1(13,1)
	ROUND1(14,2)
	ROUND1(15,0)
	ROUND1(16,1)
	ROUND1(17,2)
	ROUND1(18,0)
	ROUND1(19,1)
	ROUND1(20,2)
	ROUND1(21,0)
	ROUND1(22,1)
	ROUND1(23,2)
	ROUND1(24,0)
	ROUND1(25,1)

! ROUND 2 - key expansion

! #define ROTL_BLOCK_j1(i,j) \
!    S[i] = ROTL3(S[i]+(S[i-1]+L[0])); \
!    L[1] = ROTL(L[1]+(S[i]+L[0]),(S[i]+L[0]));

! S(i) = SS = ROTL3(S(i)+(SS+Lpred(i)))
! L(i) =      ROTL(L(i)+(SS+Lpred(i)),(SS+Lpred(i))))

! tmp    = S(i)
! SS    += Lpred(i)
! SS    += tmp
! SS     = ROTL3(SS)
! S(i)   = SS
! tmp    = SS+Lpred(i)
! L(i)  += tmp
! L(i)   = ROTL(L(i), tmp)

#define ROUND2(i,j)				\
/* ! ROUND 2 (i,j) */				\
	ld	[s_S1(i)],T1a			;\
	add	R1SS,R1Lpred(j),R1SS		;\
	add	R1SS,T1a,R1SS			;\
	sll	R1SS,3,T1a			;\
	srl	R1SS,29,R1SS			;\
	or	R1SS,T1a,R1SS			;\
	st	R1SS,[s_S1(i)]			;\
	add	R1SS,R1Lpred(j),T1a		;\
	add	R1L(j),T1a,R1L(j)		;\
	sll	R1L(j),T1a,T1b			;\
	neg	T1a				;\
	srl	R1L(j),T1a,R1L(j)		;\
	or	R1L(j),T1b,R1L(j)		;\

	ROUND2(0,2)
	ROUND2(1,0)
	ROUND2(2,1)
	ROUND2(3,2)
	ROUND2(4,0)
	ROUND2(5,1)
	ROUND2(6,2)
	ROUND2(7,0)
	ROUND2(8,1)
	ROUND2(9,2)
	ROUND2(10,0)
	ROUND2(11,1)
	ROUND2(12,2)
	ROUND2(13,0)
	ROUND2(14,1)
	ROUND2(15,2)
	ROUND2(16,0)
	ROUND2(17,1)
	ROUND2(18,2)
	ROUND2(19,0)
	ROUND2(20,1)
	ROUND2(21,2)
	ROUND2(22,0)
	ROUND2(23,1)
	ROUND2(24,2)
	ROUND2(25,0)

! ROUND 3 - key expansion and encryption
! do not save expanded key, not referenced after encryption

#define ROUND3(i,j)				\
/* ! ROUND 3 (i,j) */				\
	ld	[s_S1(i)],T1a			;\
	add	R1SS,R1Lpred(j),R1SS		;\
	add	R1SS,T1a,R1SS			;\
	sll	R1SS,3,T1a			;\
	srl	R1SS,29,R1SS			;\
	or	R1SS,T1a,R1SS			;\
	/* !st	R1SS,[s_S1(i)]	*/		;\
	add	R1SS,R1Lpred(j),T1a		;\
	add	R1L(j),T1a,R1L(j)		;\
	sll	R1L(j),T1a,T1b			;\
	neg	T1a				;\
	srl	R1L(j),T1a,R1L(j)		;\
	or	R1L(j),T1b,R1L(j)		;\

#define ROUND3S(i,j)				\
/* ! ROUND 3 (i,j) */				\
	ld	[s_S1(i)],T1a			;\
	add	R1SS,R1Lpred(j),R1SS		;\
	add	R1SS,T1a,R1SS			;\
	sll	R1SS,3,T1a			;\
	srl	R1SS,29,R1SS			;\
	or	R1SS,T1a,R1SS			;\
	/* !st	R1SS,[s_S1(i)]	*/		;\

#define ROUND3L(i,j)				\
/* ! ROUND 3 (i,j) */				\
	add	R1SS,R1Lpred(j),T1a		;\
	add	R1L(j),T1a,R1L(j)		;\
	sll	R1L(j),T1a,T1b			;\
	neg	T1a				;\
	srl	R1L(j),T1a,R1L(j)		;\
	or	R1L(j),T1b,R1L(j)		;\


! #define FINAL_BLOCK(i) \
!    A = ROTL(A^B,B)+S[2*i]; \
!    B = ROTL(B^A,A)+S[2*i+1];

! i even:
! A  = A^B
! A  = ROTL(A,B)
! A += SS
! i odd:
! B  = B^A
! B  = ROTL(B,A)
! B += SS

#define ENCRYPTION_EVEN(i)			\
/* ! ENCRYPTION EVEN (i) */			\
	xor	R1A,R1B,R1A			;\
	sll	R1A,R1B,T1a			;\
	neg	R1B,T1b				;\
	srl	R1A,T1b,R1A			;\
	or	R1A,T1a,R1A			;\
	add	R1A,R1SS,R1A			;\

#define ENCRYPTION_ODD(i)			\
/* ! ENCRYPTION ODD (i)	*/			\
	xor	R1B,R1A,R1B			;\
	sll	R1B,R1A,T1a			;\
	neg	R1A,T1b				;\
	srl	R1B,T1b,R1B			;\
	or	R1B,T1a,R1B			;\
	add	R1B,R1SS,R1B			;\



	ld	[s_uw(r72unitwork_plain_lo)],R1A
	ld	[s_uw(r72unitwork_plain_hi)],R1B

	ROUND3(0,1)
	add	R1A,R1SS,R1A

	ROUND3(1,2)
	add	R1B,R1SS,R1B

	ROUND3(2,0)
	ENCRYPTION_EVEN(2)
	ROUND3(3,1)
	ENCRYPTION_ODD(3)
	ROUND3(4,2)
	ENCRYPTION_EVEN(4)
	ROUND3(5,0)
	ENCRYPTION_ODD(5)
	ROUND3(6,1)
	ENCRYPTION_EVEN(6)
	ROUND3(7,2)
	ENCRYPTION_ODD(7)
	ROUND3(8,0)
	ENCRYPTION_EVEN(8)
	ROUND3(9,1)
	ENCRYPTION_ODD(9)
	ROUND3(10,2)
	ENCRYPTION_EVEN(10)
	ROUND3(11,0)
	ENCRYPTION_ODD(11)
	ROUND3(12,1)
	ENCRYPTION_EVEN(12)
	ROUND3(13,2)
	ENCRYPTION_ODD(13)
	ROUND3(14,0)
	ENCRYPTION_EVEN(14)
	ROUND3(15,1)
	ENCRYPTION_ODD(15)
	ROUND3(16,2)
	ENCRYPTION_EVEN(16)
	ROUND3(17,0)
	ENCRYPTION_ODD(17)
	ROUND3(18,1)
	ENCRYPTION_EVEN(18)
	ROUND3(19,2)
	ENCRYPTION_ODD(19)
	ROUND3(20,0)
	ENCRYPTION_EVEN(20)
	ROUND3(21,1)
	ENCRYPTION_ODD(21)
	ROUND3(22,2)
	ENCRYPTION_EVEN(22)
	ROUND3(23,0)
	ENCRYPTION_ODD(23)
	ROUND3S(24,1)
	ENCRYPTION_EVEN(24)
/* ! delayed after cypher_lo test	
	ROUND3L(24,1)
	ROUND3(25,2)
	ENCRYPTION_ODD(25)
*/




#define RGcypher_lo RGSinit

	ld	[s_uw(r72unitwork_cypher_lo)],RGcypher_lo
test_pipe_1:
	cmp	R1A,RGcypher_lo
	be	partial_success
	 mov	RCP,RGSinit
	
increment_key:
	add	RGLhi,PIPELINES,RGLhi
	andcc	RGLhi,0xff,RGLhi
	bz	complex_increment
	 mov	RCP_ROTL3,R1SS
increment_key_done:

	subcc	RGiterations,PIPELINES,RGiterations
	bnz	mainloop
	 mov	RGLlo,R1L(0)
	b	finish
	 mov	RESULT_NOTHING,%i0


complex_increment:
#define RI0x01000000 R1A
#define RI0xFF000000 R1B
#define RI0x00010000 T1a
//#define RI0x00FFFFFF
#define RI0x00FF0000 R1L(0)
#define RI0x0000FFFF R1L(1)
#define RI0x0000FF00 R1L(2)

	set	0x01000000,RI0x01000000
	set	0xFF000000,RI0xFF000000
	add	RGLmid,RI0x01000000,RGLmid
	btst	RI0xFF000000,RGLmid
	bnz	increment_key_done
	 set	0x00010000,RI0x00010000
	!set	0x00FFFFFF,RI0x00FFFFFF			!!!
	!sub	RI0x01000000,1,RI0x00FFFFFF
	add	RGLmid,RI0x00010000,RGLmid
	set	0x00FF0000,RI0x00FF0000
	!and	RGLmid,RI0x00FFFFFF,RGLmid
	andn	RGLmid,RI0xFF000000,RGLmid
	btst	RI0x00FF0000,RGLmid
	bnz	increment_key_done
	 sub	RI0x00010000,1,RI0x0000FFFF
	add	RGLmid,0x0100,RGLmid
	!set	0x0000FF00,RI0x0000FF00			!!!
	andn	RI0x0000FFFF,0xff,RI0x0000FF00
	and	RGLmid,RI0x0000FFFF,RGLmid
	btst	RI0x0000FF00,RGLmid
	bnz,a	increment_key_done
	 nop
	add	RGLmid,0x01,RGLmid
	andcc	RGLmid,0xff,RGLmid
	bnz,a	increment_key_done
	 nop
	add	RGLlo,RI0x01000000,RGLlo
	btst	RI0xFF000000,RGLlo
	bnz,a	increment_key_done
	 nop
	add	RGLlo,RI0x00010000,RGLlo
	!and	RGLlo,RI0x00FFFFFF,RGLlo
	andn	RGLlo,RI0xFF000000,RGLlo
	btst	RI0x00FF0000,RGLlo
	bnz,a	increment_key_done
	 nop
	add	RGLlo,0x0100,RGLlo
	and	RGLlo,RI0x0000FFFF,RGLlo
	btst	RI0x0000FF00,RGLlo
	bnz,a	increment_key_done
	 nop
	add	RGLlo,0x01,RGLlo
	and	RGLlo,0xff,RGLlo
	b	increment_key_done
	nop


partial_success:
	ROUND3L(24,1)
	ROUND3(25,2)
	ENCRYPTION_ODD(25)
	!ld	[s_uw(r72unitwork_cypher_lo)],RGcypher_lo
	!cmp	R1A,RGcypher_lo
	!be	test_pipe_1_lo_success
	! nop
test_pipe_1_lo_success:
	ld	[s_uw(r72unitwork_check_count)],T1a
	inc	T1a
	st	RGLhi,[s_uw(r72unitwork_check_hi)]
	st	RGLmid,[s_uw(r72unitwork_check_mid)]
	st	RGLlo,[s_uw(r72unitwork_check_lo)]
	st	T1a,[s_uw(r72unitwork_check_count)]
	ld	[s_uw(r72unitwork_cypher_hi)],T1b
	cmp	R1B,T1b
	bne	increment_key
	 mov	RCP,RGSinit
test_pipe_1_lohi_success:
	ld	[s_save_iterationsP],T1a
	ld	[T1a],T1b
	sub	T1b,RGiterations,T1b
	st	T1b,[T1a]
	!b	finish
	mov	RESULT_FOUND,%i0


finish:

! copy r72unitwork from Stack back into supplied memory
	ld	[s_save_r72unitworkP],%g1
	st	RGLhi,[%g1+r72unitwork_L0_hi]
	st	RGLmid,[%g1+r72unitwork_L0_mid]
	st	RGLlo,[%g1+r72unitwork_L0_lo]
	ld	[s_uw(r72unitwork_check_count)],%l0
	ld	[s_uw(r72unitwork_check_hi)],%l1
	ld	[s_uw(r72unitwork_check_mid)],%l2
	ld	[s_uw(r72unitwork_check_lo)],%l3
	st	%l0,[%g1+r72unitwork_check_count]
	st	%l1,[%g1+r72unitwork_check_hi]
	st	%l2,[%g1+r72unitwork_check_mid]
	st	%l3,[%g1+r72unitwork_check_lo]
	

	ret
	restore
