! Copyright distributed.net 2003 - All Rights Reserved
! For use in distributed.net projects only.
! Any other distribution or use of this source violates copyright.
!
! Author: Andreas Beckmann <andreasb@distributed.net>

.section	".rodata"

Id_tag:
	.ascii	"@(#)$Id: r72-anbe2.gcc.S,v 1.1.2.8 2003/04/19 17:37:15 andreasb Exp $\000"


.section	".text"

.global r72_anbe2_s
r72_anbe2_s:

	set	Id_tag, %o0
	retl
	nop


#define PIPELINES	2
#define RESULT_NOTHING	1
#define RESULT_FOUND	2


! extern "C" s32 rc5_72_unit_func_anbe_2( RC5_72UnitWork *, u32 *, void *);
.global rc5_72_unit_func_anbe_2
rc5_72_unit_func_anbe_2:

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
!   <--- stack copy of r72unitwork (%fp relative)
#define s_uw(offset)		%fp-sizeof_r72unitwork+(offset)
#define s_save_r72unitworkP	%fp-sizeof_r72unitwork-4
#define s_save_iterationsP	%fp-sizeof_r72unitwork-8
#define s_CS0			s_save_iterationsP-4
#define s_GS1			s_CS0-4
#define s_GS2			s_GS1-4
#define fp_relative_size	(sizeof_r72unitwork+2*4+3*4)
!   <--? here could be a filler for dword alignment
#define stackframe_size		(default_stackframe+fp_relative_size+4*(1+2*26+26))
! s_Sinit[26]
#define s_Sinit(i)		stack+4+8*26+(4*i)
! s_S#[26][2]
#define s_S2(i)			s_S1(i)+4
#define s_S1(i)			stack+4+8*(((i)+26) % 26)
! <- dword aligned
#define s_save_i7		stack+0
! <<<=== 'stack' points here
! (default stackframe 6+1+16 words)
! <<<=== %sp points here


	save	%sp,-(stackframe_size+(stackframe_size % 8)),%sp
	st	%i7,[s_save_i7]

	st	%i0,[s_save_r72unitworkP]
	! copy r72unitwork to stack
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

#define pred3(j)	_pred3(j)
#define _pred3(j)	pred3_##j
#define pred3_0		2
#define pred3_1		0
#define pred3_2		1

#define P		0xB7E15163
#define Q		0x9E3779B9
#define P_ROTL3		0xBF0A8B1D


! about registers
! do not use %sp, %fp, %g0, %g5, %g6, %g7
! save before use: %i7
! all other registers are available for use (assuming no function calls)

! free registers: none

#define RGiterations	%i2
#define RGLlo		%o5		// rarely used
#define RGLmid		%l4		// rarely used
#define	RGLhi		%l5


//#define RCS0		%o0
//#define RGS1		%o1
//#define RGS2		%o2
#define tRCS0		R1A//%o0
#define tRGS1		R1B//%o1
#define tRGS2		R2A//%o2
#define RGL0		%o3
#define RGL1		%o4
#define RGL1S2		%i0
#define RGS2S3		%i1

#define RCP_ROTL3	tRCS0
#define RCP		tRGS1
#define RCQ		tRGS2
#define RGSinit		RGL1


#define R1SS		%l0		// dw align0
#define R1L(j)		_R1L(j)
#define _R1L(j)		R1L_##j
#define R1L_0		%i3
#define R1L_1		%i4
#define R1L_2		%i5
#define R1Lpred(j)	R1L(pred3(j))

#define R1A		%l6
#define R1B		%l7

#define T1a		%l2		// dw align0
#define	T1b		%g1


#define R2SS		%l1		// dw align1
#define R2L(j)		_R2L(j)
#define _R2L(j)		R2L_##j
#define R2L_0		%g3
#define R2L_1		%g4
#define R2L_2		%o0//%g5
#define R2Lpred(j)	R2L(pred3(j))

#define R2A		%o1//%g6
#define R2B		%o2//%g7

#define T2a		%l3		// dw align1
#define	T2b		%g2

#define T1x		%o7
#define Tc		T1x
#define T2x		%i7


	ld	[%i1],RGiterations
	ld	[%i0+r72unitwork_L0_hi],RGLhi
	ld	[%i0+r72unitwork_L0_mid],RGLmid
	ld	[%i0+r72unitwork_L0_lo],RGLlo

	set	P,RCP
	set	Q,RCQ
	set	P_ROTL3,RCP_ROTL3
	st	tRCS0,[s_CS0]

! #define KEY_INIT(i) S[i] = P + i*Q;

#define KEY_INIT(i) 				\
	add	RGSinit,RCQ,RGSinit		;\
	st	RGSinit,[s_Sinit(i)]		;\

	mov	RCP,RGSinit
	st	RGSinit,[s_Sinit(0)]
	
        KEY_INIT(1)
        KEY_INIT(2)
        KEY_INIT(3)
        KEY_INIT(4)
        KEY_INIT(5)
        KEY_INIT(6)
        KEY_INIT(7)
        KEY_INIT(8)
        KEY_INIT(9)
        KEY_INIT(10)
        KEY_INIT(11)
        KEY_INIT(12)
        KEY_INIT(13)
        KEY_INIT(14)
        KEY_INIT(15)
        KEY_INIT(16)
        KEY_INIT(17)
        KEY_INIT(18)
        KEY_INIT(19)
        KEY_INIT(20)
        KEY_INIT(21)
        KEY_INIT(22)
        KEY_INIT(23)
        KEY_INIT(24)
        KEY_INIT(25)

#undef RCP
#undef RCQ
#undef RCP_ROTL3
#undef RGSinit

! precalculate nearly constant s1, s2, l0, l1

#define CALCULATE_RGlo_DEPS 				\
							\
	/* ROUND1(0,0) */				\
/*	!! S[0] = ROTL3(S[0]);
	!mov	RCP_ROTL3,RCS0
	!st	RCS0,[s_S#(0)]
*/	/* !! L[0] = ROTL(L[0]+S[0],S[0]); */		\
	ld	[s_CS0], tRCS0				;\
	add	RGLlo,tRCS0,R1L(0)			;\
	sll	R1L(0),(P_ROTL3 % 32),T1a		;\
	srl	R1L(0),(32 - (P_ROTL3 % 32)),R1L(0)	;\
	or	R1L(0),T1a,RGL0				;\
							\
	/* ROUND1S(1,1) */				\
	/* !! S[1] = ROTL3(S[1]+(S[0]+L[0])); */	\
	ld	[s_Sinit(1)],Tc			/* precalc S[0]+S[1] ? */	;\
	add	tRCS0,RGL0,R1SS				;\
	add	R1SS,Tc,R1SS			/* precalc S[0]+S[1] ? */	;\
	sll	R1SS,3,T1a				;\
	srl	R1SS,29,R1SS				;\
	or	R1SS,T1a,tRGS1				;\
	st	tRGS1,[s_GS1] /*###*/			;\

#define CALCULATE_RGmid_DEPS				\
							\
	ld	[s_GS1],tRGS1 /*###*/			;\
	/* ROUND1L(1,1) */				\
	/* !! L[1] = ROTL(L[1]+(S[1]+L[0]),(S[1]+L[0]));*/	\
	add	tRGS1,RGL0,T1a				;\
	add	RGLmid,T1a,R1L(1)			;\
	sll	R1L(1),T1a,T1b				;\
	neg	T1a					;\
	srl	R1L(1),T1a,R1L(1)			;\
	or	R1L(1),T1b,RGL1				;\
							\
	/* ROUND1S(2,2) */				\
	/* !! S[2] = ROTL3(S[2]+(S[1]+L[1])); */	\
	ld	[s_Sinit(2)],Tc				;\
	add	tRGS1,RGL1,R1SS				;\
	add	R1SS,Tc,R1SS				;\
	sll	R1SS,3,T1a				;\
	srl	R1SS,29,R1SS				;\
	or	R1SS,T1a,tRGS2				;\
	ld	[s_Sinit(3)],Tc				;\
	add	RGL1,tRGS2,RGL1S2			;\
	add	tRGS2,Tc,RGS2S3				;\
	st	tRGS2,[s_GS2] /*###*/			;\

CALCULATE_RGlo_DEPS
CALCULATE_RGmid_DEPS


INLINE_ROUND1_L_i2_j2:
/*2-2*/	add	RGLhi,RGL1S2,Tc /* from BeginOfLoop */

mainloop:

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
!! L(0) =     ROTL(L(0)+SS, P_ROTL3 & 0x1f)
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


#define ROUND1_FIRST_i3_j0(i,j,inR1SS,inR2SS)  ROUND1_S_i3_j0(i,j,inR1SS,inR2SS,pred3(pred3(j)),pred3(j))
#define ROUND1_FIRST(i,j)                      ROUND1_S(i,j,R1L(pred3(j)),R2L(pred3(j)),pred3(pred3(j)),pred3(j))
#define ROUND1_prec(i,j,inR1Lj,inR2Lj)         ROUND1_L_S(i,j,inR1Lj,inR2Lj,pred3(pred3(j)),pred3(j))
#define ROUND1(i,j)                            ROUND1_L_S(i,j,R1L(pred3(j)),R2L(pred3(j)),pred3(pred3(j)),pred3(j))
#define ROUND1_LAST(i,j)                       ROUND1_L(i,j,R1L(pred3(j)),R2L(pred3(j)),pred3(pred3(j)),pred3(j))

#define ROUND1_S_i3_j0(i,j,inR1SS,inR2SS,Lpred,Lcurr)	\
ROUND1_S_i3_j0_i##i##_j##j:				\
	/*ld	[s_Sinit(i)],Tc*/		;\
						\
	/*add	inR1SS,Tc,R1SS*/	/***/	;\
	/*add	inR2SS,Tc,R2SS*/	/***/	;\
	/*add	R1SS,R1L(Lcurr),R1SS*/	/***/	;\
	add	RGS2S3,R1L(Lcurr),R1SS	/***/	;\
						\
	sll	R1SS,3,T1a			;\
	/*add	R2SS,R2L(Lcurr),R2SS*/	/***/	;\
	add	RGS2S3,R2L(Lcurr),R2SS	/***/	;\
						\
	srl	R1SS,29,R1SS			;\
						\
	sll	R2SS,3,T2a			;\
	or	R1SS,T1a,R1SS			;\
						\
	srl	R2SS,29,R2SS			;\
/*Ln*/	add	R1SS,R1L(Lcurr),T1a /* from next */ ;\
						\
/*S*/	/* or	R2SS,T2a,R2SS */ /* to next */	;\
						\
/*S*/	/* st	R1SS,[s_S1(i)] */ /* to next */	;\
/*S*/	/* st	R2SS,[s_S2(i)] */ /* to next */	;\

/* unused */
#define ROUND1_S(i,j,inR1Lj,inR2Lj,Lpred,Lcurr) \
ROUND1_S_i##i##_j##j:				\
/*S*/	ld	[s_Sinit(i)],Tc			;\
						\
/*S*/	add	R1SS,R1L(Lcurr),R1SS		;\
/*S*/	add	R2SS,R2L(Lcurr),R2SS		;\
/*S*/	add	R1SS,Tc,R1SS			;\
/*S*/	sll	R1SS,3,T1a			;\
/*S*/	add	R2SS,Tc,R2SS			;\
/*S*/	sll	R2SS,3,T2a			;\
/*S*/	srl	R1SS,29,R1SS			;\
/*S*/	srl	R2SS,29,R2SS			;\
/*S*/	or	R1SS,T1a,R1SS			;\
/*S*/	/* or	R2SS,T2a,R2SS */ /* to next */	;\
						\
/*S*/	/* st	R1SS,[s_S1(i)] */ /* to next */	;\
/*S*/	/* st	R2SS,[s_S2(i)] */ /* to next */	;\
/*Ln*/	add	R1SS,R1L(Lcurr),T1a /* from next */ ;\

#define ROUND1_L_S(i,j,inR1Lj,inR2Lj,Lpred,Lcurr) \
ROUND1_L_S_i##i##_j##j:				\
						\
/*Sp*/	or	R2SS,T2a,R2SS /* from prev */	;\
/*L*/	/* add	R1SS,R1L(Lpred),T1a */ /* to prev */ ;\
/*L*/	/* add	R1L(Lcurr),T1a,R1L(Lcurr) */	;\
/*L*/	add	inR1Lj,T1a,R1L(Lcurr)		;\
/*Sp*/	st	R1SS,[s_S1(i-1)] /* from prev */;\
						\
/*L*/	add	R2SS,R2L(Lpred),T2a		;\
/*L*/	neg	T1a,T1b				;\
/*Sp*/	st	R2SS,[s_S2(i-1)] /* from prev */;\
						\
/*L*/	sll	R1L(Lcurr),T1a,T1a		;\
/*L*/	/* add	R2L(Lcurr),T2a,R2L(Lcurr) */	;\
/*L*/	add	inR2Lj,T2a,R2L(Lcurr)		;\
						\
/*L*/	srl	R1L(Lcurr),T1b,R1L(Lcurr)	;\
/*L*/	neg	T2a,T2b				;\
						\
/*L*/	sll	R2L(Lcurr),T2a,T2a		;\
/*L*/	or	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*S*/	ld	[s_Sinit(i)],Tc			;\
						\
/*L*/	srl	R2L(Lcurr),T2b,R2L(Lcurr)	;\
/*S*/	add	R1SS,R1L(Lcurr),R1SS		;\
						\
/*L*/	or	R2L(Lcurr),T2a,R2L(Lcurr)	;\
/*S*/	add	R1SS,Tc,R1SS			;\
						\
/*S*/	sll	R1SS,3,T1a			;\
/*S*/	add	R2SS,R2L(Lcurr),R2SS		;\
						\
/*S*/	srl	R1SS,29,R1SS			;\
/*S*/	add	R2SS,Tc,R2SS			;\
						\
/*S*/	sll	R2SS,3,T2a			;\
/*S*/	or	R1SS,T1a,R1SS			;\
						\
/*S*/	srl	R2SS,29,R2SS			;\
/*Ln*/	add	R1SS,R1L(Lcurr),T1a /* from next */ ;\
						\
/*S*/	/* or	R2SS,T2a,R2SS */ /* to next */	;\
						\
/*S*/	/* st	R1SS,[s_S1(i)] */ /* to next */	;\
/*S*/	/* st	R2SS,[s_S2(i)] */ /* to next */	;\

#define ROUND1_L(i,j,inR1Lj,inR2Lj,Lpred,Lcurr) \
ROUND1_L_i##i##_j##j:				\
						\
/*Sp*/	or	R2SS,T2a,R2SS /* from prev */	;\
/*L*/	/* add	R1SS,R1L(Lpred),T1a */ /* to prev */ ;\
/*L*/	/* add	R1L(Lcurr),T1a,R1L(Lcurr) */	;\
/*L*/	add	inR1Lj,T1a,R1L(Lcurr)		;\
/*Sp*/	st	R1SS,[s_S1(i-1)] /* from prev */;\
						\
/*L*/	add	R2SS,R2L(Lpred),T2a		;\
/*L*/	neg	T1a,T1b				;\
/*Sp*/	st	R2SS,[s_S2(i-1)] /* from prev */;\
						\
/*L*/	sll	R1L(Lcurr),T1a,T1a		;\
/*L*/	/* add	R2L(Lcurr),T2a,R2L(Lcurr) */	;\
/*L*/	add	inR2Lj,T2a,R2L(Lcurr)		;\
						\
/*L*/	srl	R1L(Lcurr),T1b,R1L(Lcurr)	;\
/*L*/	neg	T2a,T2b				;\
						\
/*L*/	sll	R2L(Lcurr),T2a,T2a		;\
/*L*/	or	R1L(Lcurr),T1a,R1L(Lcurr)	;\
						\
/*L*/	srl	R2L(Lcurr),T2b,R2L(Lcurr)	;\
						\
/*L*/	or	R2L(Lcurr),T2a,R2L(Lcurr)	;\


	/* ROUND1L(2,2) */
!INLINE_ROUND1_L_i2_j2:
	!! L[2] = ROTL(L[2]+(S[2]+L[1]),(S[2]+L[1]));
/*
	!add	RGS2,RGL1,T1a			// PRECALC!
	!!!add	RGS2,RGL1,T2a
	add	RGLhi,RGL1S2,R1L(2)
	!!!add	RGLhi,T2a,R2L(2)
	!!!inc	R2L(2)
	add	R1L(2),1,R2L(2)
	sll	R1L(2),RGL1S2,T1b
	neg	RGL1S2,T2a
	sll	R2L(2),RGL1S2,T2b
	!neg	T2a
	srl	R1L(2),T2a,R1L(2)
	srl	R2L(2),T2a,R2L(2)
	or	R1L(2),T1b,R1L(2)
	or	R2L(2),T2b,R2L(2)
*/
/*2-2*/	/* add	RGLhi,RGL1S2,Tc */ /* to EndOfLoop */

/*2-2*/	sll	Tc,RGL1S2,T1b
/*2-2*/	neg	RGL1S2,T2a			// constant, but slot is free anyway

/*2-2*/	srl	Tc,T2a,R1L(2)
/*2-2*/	add	Tc,1,R2L(2)

/*2-2*/	sll	R2L(2),RGL1S2,T2b
/*2-2*/	or	R1L(2),T1b,R1L(2)

/*2-2*/	srl	R2L(2),T2a,R2L(2)
/*3-0*/	add	RGS2S3,R1L(2),R1SS /* from below */

/*2-2*/	/* or	R2L(2),T2b,R2L(2) */ /* to below */

INLINE_ROUND1_S_i3_j0:

/*3-0*/	/* add	RGS2S3,R1L(2),R1SS */ /* to up */

/*3-0*/	sll	R1SS,3,T1a
/*2-2*/	or	R2L(2),T2b,R2L(2) /* from up */

/*3-0*/	srl	R1SS,29,R1SS
/*3-0*/	add	RGS2S3,R2L(2),R2SS

/*3-0*/	sll	R2SS,3,T2a
/*3-0*/	or	R1SS,T1a,R1SS

/*3-0*/	srl	R2SS,29,R2SS
/*Ln*/	add	R1SS,R1L(2),T1a /* from next */

/*S*/	/* or	R2SS,T2a,R2SS */ /* to next */

/*S*/	/* st	R1SS,[s_S1(i)] */ /* to next */
/*S*/	/* st	R2SS,[s_S2(i)] */ /* to next */

/*	ROUND1_FIRST_i3_j0(3,0,RGS2,RGS2) */ /* has been inlined */
	ROUND1_prec(4,1,RGL0,RGL0)
	ROUND1_prec(5,2,RGL1,RGL1)
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
	/* ROUND1_LAST(26,2) */ /* inlined */

INLINE_ROUND1_L_i26_j2:
#define Lpred pred3(pred3(2))
#define Lcurr pred3(2)
#define i 26

/*Sp*/	or	R2SS,T2a,R2SS /* from prev */
/*L*/	/* add	R1SS,R1L(Lpred),T1a */ /* to prev */
/*L*/	add	R1L(Lcurr),T1a,R1L(Lcurr)
/*L*/	/* add	inR1Lj,T1a,R1L(Lcurr) */
/*Sp*/	st	R1SS,[s_S1(i-1)] /* from prev */

/*L*/	add	R2SS,R2L(Lpred),T2a
/*L*/	neg	T1a,T1b
/*Sp*/	st	R2SS,[s_S2(i-1)] /* from prev */

/*L*/	sll	R1L(Lcurr),T1a,T1a
/*L*/	add	R2L(Lcurr),T2a,R2L(Lcurr)
/*L*/	/* add	inR2Lj,T2a,R2L(Lcurr) */

/*L*/	srl	R1L(Lcurr),T1b,R1L(Lcurr)
/*L*/	neg	T2a,T2b

/*L*/	sll	R2L(Lcurr),T2a,T2a
/*L*/	or	R1L(Lcurr),T1a,R1L(Lcurr)
	ld	[s_CS0], Tc /*###*/

/*L*/	srl	R2L(Lcurr),T2b,R2L(Lcurr)

/*L*/	/* or	R2L(Lcurr),T2a,R2L(Lcurr) */ /* to BeginOfRound2 */
#undef Lpred
#undef Lcurr
#undef i

! ############################################################################

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

#define ROUND2_START_prec(i,j,SiPrec) ROUND2_START_S_prec(i,j,SiPrec,pred3(pred3(j)),pred3(j))
#define ROUND2prec(i,j,SiPrec)        ROUND2_L_S_prec(i,j,SiPrec,pred3(pred3(j)),pred3(j))
#define ROUND2(i,j)                   ROUND2_L_S(i,j,pred3(pred3(j)),pred3(j))
#define ROUND2_END(i,j)               ROUND2_END_L(i,j,pred3(pred3(j)),pred3(j))

#define ROUND2_START_S_prec(i,j,SiPrec,Lpred,Lcurr) \
ROUND2_START_S_PREC_i##i##_j##j:		\
/*S*/	/*ld	[s_S1(i)],T1x */		;\
/*S*/	/*ld	[s_S2(i)],T2x */		;\
						\
/*S*/	add	R1SS,R1L(Lcurr),R1SS		;\
						\
/*S*/	add	R1SS,SiPrec,R1SS		;\
						\
/*S*/	sll	R1SS,3,T1a			;\
/*S*/	add	R2SS,R2L(Lcurr),R2SS		;\
						\
/*S*/	srl	R1SS,29,R1SS			;\
/*S*/	add	R2SS,SiPrec,R2SS		;\
						\
/*S*/	sll	R2SS,3,T2a			;\
/*S*/	or	R1SS,T1a,R1SS			;\
						\
/*S*/	srl	R2SS,29,R2SS			;\
/*Ln*/	add	R1SS,R1L(Lcurr),T1a /* from next */ ;\
						\
/*S*/	/* or	R2SS,T2a,R2SS */ /* to next */	;\
						\
/*S*/	/* st	R1SS,[s_S1(i)] */ /* to next */	;\
/*S*/	/* st	R2SS,[s_S2(i)] */ /* to next */	;\

#define ROUND2_L_S_prec(i,j,SiPrec,Lpred,Lcurr)	\
ROUND2_L_S_prec_i##i##_j##j:			\
						\
/*L*/	/* add	R1SS,R1L(Lpred),T1a */ /* to prev */ ;\
						\
/*Sp*/	or	R2SS,T2a,R2SS /* from prev */	;\
/*L*/	add	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*Sp*/	st	R1SS,[s_S1(i-1)] /* from prev */;\
						\
/*L*/	add	R2SS,R2L(Lpred),T2a		;\
/*L*/	neg	T1a,T1b				;\
/*Sp*/	st	R2SS,[s_S2(i-1)] /* from prev */;\
						\
/*L*/	sll	R1L(Lcurr),T1a,T1a		;\
/*L*/	add	R2L(Lcurr),T2a,R2L(Lcurr)	;\
						\
/*L*/	srl	R1L(Lcurr),T1b,R1L(Lcurr)	;\
/*L*/	neg	T2a,T2b				;\
						\
/*L*/	sll	R2L(Lcurr),T2a,T2a		;\
/*L*/	or	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*S*/	/*ld	[s_S1(i)],T1x */	/****/	;\
						\
/*L*/	srl	R2L(Lcurr),T2b,R2L(Lcurr)	;\
/*S*/	add	R1SS,R1L(Lcurr),R1SS		;\
						\
/*L*/	or	R2L(Lcurr),T2a,R2L(Lcurr)	;\
/*S*/	add	R1SS,SiPrec,R1SS	/****/	;\
/*S*/	/*ld	[s_S2(i)],T2x */	/****/	;\
						\
/*S*/	sll	R1SS,3,T1a			;\
/*S*/	add	R2SS,R2L(Lcurr),R2SS		;\
						\
/*S*/	srl	R1SS,29,R1SS			;\
/*S*/	add	R2SS,SiPrec,R2SS	/****/	;\
						\
/*S*/	sll	R2SS,3,T2a			;\
/*S*/	or	R1SS,T1a,R1SS			;\
						\
/*S*/	srl	R2SS,29,R2SS			;\
/*Ln*/	add	R1SS,R1L(Lcurr),T1a /* from next */ ;\
						\
/*S*/	/* or	R2SS,T2a,R2SS */ /* to next */	;\
						\
/*S*/	/* st	R1SS,[s_S1(i)] */ /* to next */	;\
/*S*/	/* st	R2SS,[s_S2(i)] */ /* to next */	;\

#define ROUND2_L_S(i,j,Lpred,Lcurr)		\
ROUND2_L_S_i##i##_j##j:				\
						\
/*L*/	/* add	R1SS,R1L(Lpred),T1a */ /* to prev */ ;\
						\
/*Sp*/	or	R2SS,T2a,R2SS /* from prev */	;\
/*L*/	add	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*Sp*/	st	R1SS,[s_S1(i-1)] /* from prev */;\
						\
/*L*/	add	R2SS,R2L(Lpred),T2a		;\
/*L*/	neg	T1a,T1b				;\
/*Sp*/	st	R2SS,[s_S2(i-1)] /* from prev */;\
						\
/*L*/	sll	R1L(Lcurr),T1a,T1a		;\
/*L*/	add	R2L(Lcurr),T2a,R2L(Lcurr)	;\
						\
/*L*/	srl	R1L(Lcurr),T1b,R1L(Lcurr)	;\
/*L*/	neg	T2a,T2b				;\
						\
/*L*/	sll	R2L(Lcurr),T2a,T2a		;\
/*L*/	or	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*S*/	ld	[s_S1(i)],T1x			;\
						\
/*L*/	srl	R2L(Lcurr),T2b,R2L(Lcurr)	;\
/*S*/	add	R1SS,R1L(Lcurr),R1SS		;\
						\
/*L*/	or	R2L(Lcurr),T2a,R2L(Lcurr)	;\
/*S*/	add	R1SS,T1x,R1SS			;\
/*S*/	ld	[s_S2(i)],T2x			;\
						\
/*S*/	sll	R1SS,3,T1a			;\
/*S*/	add	R2SS,R2L(Lcurr),R2SS		;\
						\
/*S*/	srl	R1SS,29,R1SS			;\
/*S*/	add	R2SS,T2x,R2SS			;\
						\
/*S*/	sll	R2SS,3,T2a			;\
/*S*/	or	R1SS,T1a,R1SS			;\
						\
/*S*/	srl	R2SS,29,R2SS			;\
/*Ln*/	add	R1SS,R1L(Lcurr),T1a /* from next */ ;\
						\
/*S*/	/* or	R2SS,T2a,R2SS */ /* to next */	;\
						\
/*S*/	/* st	R1SS,[s_S1(i)] */ /* to next */	;\
/*S*/	/* st	R2SS,[s_S2(i)] */ /* to next */	;\

#define ROUND2_END_L(i,j,Lpred,Lcurr)		\
ROUND2_END_L_i##i##_j##j:				\
						\
/*L*/	/* add	R1SS,R1L(Lpred),T1a */ /* to prev */ ;\
						\
/*Sp*/	or	R2SS,T2a,R2SS /* from prev */	;\
/*L*/	add	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*Sp*/	st	R1SS,[s_S1(i-1)] /* from prev */;\
						\
/*L*/	add	R2SS,R2L(Lpred),T2a		;\
/*L*/	neg	T1a,T1b				;\
/*Sp*/	st	R2SS,[s_S2(i-1)] /* from prev */;\
						\
/*L*/	sll	R1L(Lcurr),T1a,T1a		;\
/*L*/	add	R2L(Lcurr),T2a,R2L(Lcurr)	;\
						\
/*L*/	srl	R1L(Lcurr),T1b,R1L(Lcurr)	;\
/*L*/	neg	T2a,T2b				;\
						\
/*L*/	sll	R2L(Lcurr),T2a,T2a		;\
/*L*/	or	R1L(Lcurr),T1a,R1L(Lcurr)	;\
						\
/*L*/	srl	R2L(Lcurr),T2b,R2L(Lcurr)	;\
						\
/*L*/	or	R2L(Lcurr),T2a,R2L(Lcurr)	;\


INLINE_ROUND2_START_S_PREC_i0_j2:
#define Lpred pred3(pred3(2))
#define Lcurr pred3(2)
#define i 0

/*S*/	/*ld	[s_S1(i)],T1x */
/*S*/	/*ld	[s_S2(i)],T2x */

/*S*/	add	R1SS,R1L(Lcurr),R1SS

/*L*/	or	R2L(Lcurr),T2a,R2L(Lcurr) /* from EndOfRound1 */
///*S*/	add	R1SS,SiPrec,R1SS
///*S*/	add	R1SS,tRCS0,R1SS /*###*/
/*S*/	add	R1SS,Tc,R1SS /*###*/

/*S*/	sll	R1SS,3,T1a
/*S*/	add	R2SS,R2L(Lcurr),R2SS

/*S*/	srl	R1SS,29,R1SS
///*S*/	add	R2SS,SiPrec,R2SS
///*S*/	add	R2SS,tRCS0,R2SS /*###*/
/*S*/	add	R2SS,Tc,R2SS /*###*/

/*S*/	sll	R2SS,3,T2a
/*S*/	or	R1SS,T1a,R1SS

/*S*/	srl	R2SS,29,R2SS
/*Ln*/	add	R1SS,R1L(Lcurr),T1a /* from next */

/*S*/	/* or	R2SS,T2a,R2SS */ /* to next */

/*S*/	/* st	R1SS,[s_S1(i)] */ /* to next */
/*S*/	/* st	R2SS,[s_S2(i)] */ /* to next */

#undef Lpred
#undef Lcurr
#undef i

	/* ROUND2_START_prec(0,2,RCS0) */ /* inlined */
	ld	[s_GS1],Tc /*###*/
	//ROUND2prec(1,0,RGS1)
	ROUND2prec(1,0,Tc)
	ld	[s_GS2],Tc /*###*/
	//ROUND2prec(2,1,RGS2)
	ROUND2prec(2,1,Tc)
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
	/* ROUND2_END(26,1) */ /* inlined */

INLINE_ROUND2_END_L_i26_j1:
#define Lpred pred3(pred3(1))
#define Lcurr pred3(1)
#define i 26

/*L*/	/* add	R1SS,R1L(Lpred),T1a */ /* to prev */ ;\

/*Sp*/	or	R2SS,T2a,R2SS /* from prev */	;\
/*L*/	add	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*Sp*/	st	R1SS,[s_S1(i-1)] /* from prev */;\

/*L*/	add	R2SS,R2L(Lpred),T2a		;\
/*L*/	neg	T1a,T1b				;\
/*Sp*/	st	R2SS,[s_S2(i-1)] /* from prev */;\

/*L*/	sll	R1L(Lcurr),T1a,T1a		;\
/*L*/	add	R2L(Lcurr),T2a,R2L(Lcurr)	;\

/*L*/	srl	R1L(Lcurr),T1b,R1L(Lcurr)	;\
/*L*/	neg	T2a,T2b				;\

/*L*/	sll	R2L(Lcurr),T2a,T2a		;\
/*L*/	or	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*S0*/	ld	[s_S1(0)],T1x /* from BeginOfRound3 */

/*L*/	srl	R2L(Lcurr),T2b,R2L(Lcurr)	;\
/*S0*/	add	R1SS,R1L(pred3(1)),R1SS /* from BeginOfRound3 */
/*S0*/	ld	[s_S2(0)],T2x /* from BeginOfRound3 */

/*L*/	or	R2L(Lcurr),T2a,R2L(Lcurr)	;\

#undef Lpred
#undef Lcurr
#undef i

! ############################################################################

round3_encryption:
! ROUND 3 - key expansion and encryption
! do not save expanded key, not referenced after encryption

! #define ROTL_BLOCK_j1(i,j) \
!    S[i] = ROTL3(S[i]+(S[i-1]+L[0])); \
!    L[1] = ROTL(L[1]+(S[i]+L[0]),(S[i]+L[0]));

! S(i) = SS = ROTL3(S(i)+(SS+Lpred(i)))
! L(i) =      ROTL(L(i)+(SS+Lpred(i)),(SS+Lpred(i))))

! tmp    = S(i)
! SS    += Lpred(i)
! SS    += tmp
! SS     = ROTL3(SS)
//S(i)   = SS
! tmp    = SS+Lpred(i)
! L(i)  += tmp
! L(i)   = ROTL(L(i), tmp)

! ROUND 3 - key expansion and encryption
!
! ROUND3EVEN(i,j) = ROUND3(i,j,A,B)
! ROUND3ODD (i,j) = ROUND3(i,j,B,A)
! ROUND3(i,j,A,B) =
!    S[i] = ROTL3(S[i]+(S[i-1]+Lpred[j]));
!    L[j] = ROTL(L[j]+(S[i]+Lpred[j]),(S[i]+Lpred[j]));
!    A    = ROTL(A^B,B) + S[i]
! lets reorder this a bit ...
!  * A depends on S[i] (in A's last operation)
!  * L[j] depends on S[i] (in L's first operation)
!  * S[i] depends on Lpred[j] (in S's first or second operation
!  * we don't need to calculate L[j] for i=25
!  * this ordering allows better instruction sceduling because the S[i]->L[j]
!    dependency is no longer in adjacent operations
! ROUND3(i,j,A,B) =
!    L[j-1] = ROTL(L[j-1]+(S[i-1]+Lpred[j-1]),(S[i-1]+Lpred[j-1]));
!    S[i]   = ROTL3(S[i]+(S[i-1]+Lpred[j]));
!    A      = ROTL(A^B,B) + S[i]
! Note: L[j-1] actually is L[(j-1) % 3]
! Note: there is a special A calculation for i=0 and i=1 (A initialization)
! Note: ROUND3(0,...) just needs to calculate S[0]
! lets translate this into some other variable names
! ROUND3(i,j,A,B) = ROUND3_LSA(i,j,A,B,Lpred[j],Lpred[j-1])
! ROUND3_LSA(i,j,A,B,Lprd,Lcur) =
!    Lcur   = ROTL(Lcur+(SS+Lprd),(SS+Lprd));
!    SS     = ROTL3(S[i]+SS+Lcur);
!    A      = ROTL(A^B,B) + SS
! split into basic oprations:
! ROUND3_LSA(i,j,A,B,Lprd,Lcur) =
!    T1     = SS+Lprd
!    Lcur   = Lcur+T1
!    Lcur   = ROTL(Lcur,T1)
!    T2     = S[i]
!    SS     = SS+T2
!    SS     = SS+Lcur
!    SS     = ROTL3(SS)
!    A      = A^B
!    A      = ROTL(A^B,B)
!    A      = A+SS
!

#define ROUND3_i0(i,j)  ROUND3_S(i,j,pred3(pred3(j)),pred3(j))
#define ROUND3(i,j)     ROUND3_L_S(i,j,pred3(pred3(j)),pred3(j))
#define ROUND3EVEN(i,j) ROUND3_L_S_A(i,j,A,B,pred3(pred3(j)),pred3(j))
#define ROUND3ODD(i,j)  ROUND3_L_S_A(i,j,B,A,pred3(pred3(j)),pred3(j))

#define ROUND3_L_S_A(i,j,A,B,Lpred,Lcurr)	\
ROUND3_L_S_A_i##i##_j##j:			\
/* ! ROUND 3 (i,j) */				\
/*L*/	add	R1SS,R1L(Lpred),T1a		;\
	STARTSLOT_R2SS				;\
/*L*/	add	R2SS,R2L(Lpred),T2a		;\
/*S*/	ld	[s_S1(i)],T1x			;\
						\
/*L*/	add	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*L*/	neg	T1a,T1b				;\
/*S*/	ld	[s_S2(i)],T2x			;\
						\
/*L*/	sll	R1L(Lcurr),T1a,T1a		;\
/*L*/	add	R2L(Lcurr),T2a,R2L(Lcurr)	;\
						\
/*L*/	srl	R1L(Lcurr),T1b,R1L(Lcurr)	;\
/*L*/	neg	T2a,T2b				;\
						\
/*L*/	sll	R2L(Lcurr),T2a,T2a		;\
/*L*/	or	R1L(Lcurr),T1a,R1L(Lcurr)	;\
						\
/*L*/	srl	R2L(Lcurr),T2b,R2L(Lcurr)	;\
/*S*/	add	R1SS,T1x,R1SS			;\
						\
/*S*/	add	R2SS,T2x,R2SS			;\
/*S*/	add	R1SS,R1L(Lcurr),R1SS		;\
						\
/*S*/	sll	R1SS,3,T1x			;\
/*L*/	or	R2L(Lcurr),T2a,R2L(Lcurr)	;\
						\
/*S*/	srl	R1SS,29,R1SS			;\
/*S*/	add	R2SS,R2L(Lcurr),R2SS		;\
						\
/*S*/	sll	R2SS,3,T2x			;\
/*A*/	xor	R1##A,R1##B,R1##A		;\
						\
/*S*/	srl	R2SS,29,R2SS			;\
/*A*/	neg	R1##B,T1b			;\
						\
/*A*/	sll	R1##A,R1##B,T1a			;\
/*A*/	xor	R2##A,R2##B,R2##A		;\
						\
/*A*/	srl	R1##A,T1b,R1##A			;\
/*A*/	neg	R2##B,T2b			;\
						\
/*A*/	sll	R2##A,R2##B,T2a			;\
/*S*/	or	R1SS,T1x,R1SS			;\
/*S*/	/*st	R1SS,[s_S1(i)]*/		;\
						\
/*A*/	srl	R2##A,T2b,R2##A			;\
/*S*/	or	R2SS,T2x,R2SS			;\
	PRELOAD_SLOT				;\
						\
/*A*/	or	R1##A,T1a,R1##A			;\
/*A*/	or	R2##A,T2a,R2##A			;\
/*S*/	/*st	R2SS,[s_S2(i)]*/		;\
						\
/*A*/	add	R1##A,R1SS,R1##A		;\
/*A*/	add	R2##A,R2SS,R2##A		;\

#define ROUND3_L_S(i,j,Lpred,Lcurr)		\
ROUND3_L_S_i##i##_j##j:				\
/*L*/	add	R1SS,R1L(Lpred),T1a		;\
/*L*/	add	R2SS,R2L(Lpred),T2a		;\
/*S*/	ld	[s_S1(i)],T1x			;\
						\
/*L*/	add	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*L*/	add	R2L(Lcurr),T2a,R2L(Lcurr)	;\
/*S*/	ld	[s_S2(i)],T2x			;\
						\
/*L*/	sll	R1L(Lcurr),T1a,T1b		;\
/*L*/	neg	T1a				;\
						\
/*L*/	srl	R1L(Lcurr),T1a,R1L(Lcurr)	;\
/*S*/	add	R1SS,T1x,R1SS			;\
						\
/*L*/	sll	R2L(Lcurr),T2a,T2b		;\
/*L*/	neg	T2a				;\
						\
/*L*/	srl	R2L(Lcurr),T2a,R2L(Lcurr)	;\
/*L*/	or	R1L(Lcurr),T1b,R1L(Lcurr)	;\
						\
/*S*/	add	R2SS,T2x,R2SS			;\
/*S*/	add	R1SS,R1L(Lcurr),R1SS		;\
						\
/*S*/	sll	R1SS,3,T1a			;\
/*L*/	or	R2L(Lcurr),T2b,R2L(Lcurr)	;\
						\
/*S*/	srl	R1SS,29,R1SS			;\
/*S*/	add	R2SS,R2L(Lcurr),R2SS		;\
						\
/*S*/	sll	R2SS,3,T2a			;\
/*S*/	or	R1SS,T1a,R1SS			;\
						\
/*S*/	srl	R2SS,29,R2SS			;\
		ENDSLOT_R1SS			;\
						\
/*S*/	or	R2SS,T2a,R2SS			;\

#define ROUND3_S(i,j,Lpred,Lcurr)		\
ROUND3_S_i##i##_j##j:				\
/*S*/	ld	[s_S1(i)],T1a			;\
/*S*/	ld	[s_S2(i)],T2a			;\
/*S*/	add	R1SS,R1L(Lcurr),R1SS		;\
/*S*/	add	R2SS,R2L(Lcurr),R2SS		;\
/*S*/	add	R1SS,T1a,R1SS			;\
/*S*/	sll	R1SS,3,T1a			;\
/*S*/	add	R2SS,T2a,R2SS			;\
/*S*/	sll	R2SS,3,T2a			;\
/*S*/	srl	R1SS,29,R1SS			;\
/*S*/	srl	R2SS,29,R2SS			;\
/*S*/	or	R1SS,T1a,R1SS			;\
/*S*/	or	R2SS,T2a,R2SS			;\


// inlined ROUND3_i0(0,1), ROUND3(1,2), R#A, R#B initialization
// for better sceduling

INLINE_ROUND3_i0_j1___i1_j2:
//	ROUND3_i0(0,1)
//	ROUND3(1,2)
#define Lpred pred3(pred3(2))
#define Lcurr pred3(2)

/*S0*/	/* ld	[s_S1(0)],T1x */ /* to EndOfRound2 */
/*S0*/	/* ld	[s_S2(0)],T2x */ /* to EndOfRound2 */

/*S0*/	/* add	R1SS,R1L(pred3(1)),R1SS */ /* to EndOfRound2 */

/*S0*/	add	R1SS,T1x,R1SS
	ld	[s_uw(r72unitwork_plain_lo)],R2A

/*S0*/	sll	R1SS,3,T1a
/*S0*/	add	R2SS,R2L(pred3(1)),R2SS
	ld	[s_uw(r72unitwork_plain_hi)],R2B

/*S0*/	srl	R1SS,29,R1SS
/*S0*/	add	R2SS,T2x,R2SS

/*S0*/	sll	R2SS,3,T2a
/*S0*/	or	R1SS,T1a,R1SS

/*S0*/	srl	R2SS,29,R2SS
	add	R2A,R1SS,R1A

/*S0*/	or	R2SS,T2a,R2SS
/*L*/	add	R1SS,R1L(Lpred),T1a

	add	R2A,R2SS,R2A
/*L*/	add	R2SS,R2L(Lpred),T2a
/*S*/	ld	[s_S1(1)],T1x

/*L*/	add	R1L(Lcurr),T1a,R1L(Lcurr)
/*L*/	add	R2L(Lcurr),T2a,R2L(Lcurr)
/*S*/	ld	[s_S2(1)],T2x

/*L*/	sll	R1L(Lcurr),T1a,T1b
/*L*/	neg	T1a

/*L*/	srl	R1L(Lcurr),T1a,R1L(Lcurr)
/*S*/	add	R1SS,T1x,R1SS

/*L*/	sll	R2L(Lcurr),T2a,T2b
/*L*/	neg	T2a

/*L*/	srl	R2L(Lcurr),T2a,R2L(Lcurr)
/*L*/	or	R1L(Lcurr),T1b,R1L(Lcurr)

/*S*/	add	R2SS,T2x,R2SS
/*S*/	add	R1SS,R1L(Lcurr),R1SS

/*S*/	sll	R1SS,3,T1a
/*L*/	or	R2L(Lcurr),T2b,R2L(Lcurr)

/*S*/	srl	R1SS,29,R1SS
/*S*/	add	R2SS,R2L(Lcurr),R2SS

/*S*/	sll	R2SS,3,T2a
/*S*/	or	R1SS,T1a,R1SS

/*S*/	srl	R2SS,29,R2SS
	add	R2B,R1SS,R1B

/*S*/	or	R2SS,T2a,R2SS
	//add	R2B,R2SS,R2B // moved into ROUND3EVEN(2,0)
#define STARTSLOT_R2SS	add	R2B,R2SS,R2B

#undef Lpred
#undef Lcurr

#define PRELOAD_SLOT // nothing
	ROUND3EVEN(2,0)
#undef STARTSLOT_R2SS
#define STARTSLOT_R2SS // clear
	ROUND3ODD (3,1)
	ROUND3EVEN(4,2)
	ROUND3ODD (5,0)
	ROUND3EVEN(6,1)
	ROUND3ODD (7,2)
	ROUND3EVEN(8,0)
	ROUND3ODD (9,1)
	ROUND3EVEN(10,2)
	ROUND3ODD (11,0)
	ROUND3EVEN(12,1)
	ROUND3ODD (13,2)
	ROUND3EVEN(14,0)
	ROUND3ODD (15,1)
	ROUND3EVEN(16,2)
	ROUND3ODD (17,0)
	ROUND3EVEN(18,1)
	ROUND3ODD (19,2)
	ROUND3EVEN(20,0)
	ROUND3ODD (21,1)
	ROUND3EVEN(22,2)
	ROUND3ODD (23,0)
#define RGcypher_lo T1x
#undef PRELOAD_SLOT
#define PRELOAD_SLOT ld	[s_uw(r72unitwork_cypher_lo)],RGcypher_lo
	ROUND3EVEN(24,1)
#undef PRELOAD_SLOT
#define PRELOAD_SLOT // nothing
/* ! calculation of cypher_hi delayed after successful cypher_lo test	
	ROUND3ODD (25,2)
*/

	// ld	[s_uw(r72unitwork_cypher_lo)],RGcypher_lo /* moved to PRELOAD_SLOT */
test_pipe_1:
	cmp	R1A,RGcypher_lo
	add	RGLhi,PIPELINES,RGLhi /* from further down */
	be	partial_success
	 cmp	R2A,RGcypher_lo
	be	partial_success
	 andcc	RGLhi,0xff,RGLhi /* from further down */
	
increment_key:
	/* add	RGLhi,PIPELINES,RGLhi */    /* moved up */
	/* andcc	RGLhi,0xff,RGLhi */ /* moved up and repeated in delay
	                                       slots from partial_success: */
	bz	complex_increment
	 subcc	RGiterations,PIPELINES,RGiterations /* from further down */
increment_key_done:

	/* subcc	RGiterations,PIPELINES,RGiterations */ /* moved up */
	bnz	mainloop
/*2-2*/	 add	RGLhi,RGL1S2,Tc /* from BeginOfLoop */
end_of_mainloop:
	b	finish
	 mov	RESULT_NOTHING,%i0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

complex_increment:
#define RI0x01000000 R2A
#define RI0xFF000000 R2B
#define RI0x00010000 T2a
//#define RI0x00FFFFFF
#define RI0x00FF0000 R2L(0)
#define RI0x0000FFFF R2L(1)
#define RI0x0000FF00 R2L(2)

	! increment RGLmid
	set	0x01000000,RI0x01000000
	set	0xFF000000,RI0xFF000000
	add	RGLmid,RI0x01000000,RGLmid
	btst	RI0xFF000000,RGLmid
	bnz	update_RGLmid
	 set	0x00010000,RI0x00010000
	!set	0x00FFFFFF,RI0x00FFFFFF			!!!
	!sub	RI0x01000000,1,RI0x00FFFFFF
	add	RGLmid,RI0x00010000,RGLmid
	set	0x00FF0000,RI0x00FF0000
	!and	RGLmid,RI0x00FFFFFF,RGLmid
	andn	RGLmid,RI0xFF000000,RGLmid
	btst	RI0x00FF0000,RGLmid
	bnz	update_RGLmid
	 sub	RI0x00010000,1,RI0x0000FFFF
	add	RGLmid,0x0100,RGLmid
	!set	0x0000FF00,RI0x0000FF00			!!!
	andn	RI0x0000FFFF,0xff,RI0x0000FF00
	and	RGLmid,RI0x0000FFFF,RGLmid
	btst	RI0x0000FF00,RGLmid
	bnz,a	update_RGLmid
	 nop
	add	RGLmid,0x01,RGLmid
	andcc	RGLmid,0xff,RGLmid
	bnz,a	update_RGLmid
	 nop

	! increment RGLlo
	add	RGLlo,RI0x01000000,RGLlo
	btst	RI0xFF000000,RGLlo
	bnz,a	update_RGLlo
	 nop
	add	RGLlo,RI0x00010000,RGLlo
	!and	RGLlo,RI0x00FFFFFF,RGLlo
	andn	RGLlo,RI0xFF000000,RGLlo
	btst	RI0x00FF0000,RGLlo
	bnz,a	update_RGLlo
	 nop
	add	RGLlo,0x0100,RGLlo
	and	RGLlo,RI0x0000FFFF,RGLlo
	btst	RI0x0000FF00,RGLlo
	bnz,a	update_RGLlo
	 nop
	add	RGLlo,0x01,RGLlo
	and	RGLlo,0xff,RGLlo
	!b	update_RGLlo
	! nop

update_RGLlo:
	! update precalculated values RGL0, RGS1, RGL1, RGS2

	! first calculate RGL0, RGS1
	CALCULATE_RGlo_DEPS
	! and fall through to calculate RGL1, RGS2

update_RGLmid:
	! update precalculated values RGL1, RGS2

	CALCULATE_RGmid_DEPS
	b	increment_key_done
	 subcc	RGiterations,0,RGiterations
	 !nop

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

partial_success:
	ROUND3ODD(25,2)
	ld	[s_uw(r72unitwork_cypher_lo)],RGcypher_lo
	cmp	R1A,RGcypher_lo
	bne	test_pipe_2_lo_success
	 ld	[s_uw(r72unitwork_cypher_hi)],T1b
test_pipe_1_lo_success:
	ld	[s_uw(r72unitwork_check_count)],T1a
	inc	T1a
	st	RGLmid,[s_uw(r72unitwork_check_mid)]
	st	T1a,[s_uw(r72unitwork_check_count)]
	add	RGLhi,-PIPELINES+0,T2a
	st	RGLlo,[s_uw(r72unitwork_check_lo)]
	!st	RGLhi,[s_uw(r72unitwork_check_hi)]
	st	T2a,[s_uw(r72unitwork_check_hi)]
	!ld	[s_uw(r72unitwork_cypher_hi)],T1b
	cmp	R1B,T1b
	be	test_pipe_1_lohi_success
	 cmp	R2A,RGcypher_lo
	bne	increment_key
	 andcc	RGLhi,0xff,RGLhi
test_pipe_2_lo_success:
	ld	[s_uw(r72unitwork_check_count)],T2a
	inc	T2a
	st	RGLmid,[s_uw(r72unitwork_check_mid)]
	st	T2a,[s_uw(r72unitwork_check_count)]
	add	RGLhi,-PIPELINES+1,T2a
	st	RGLlo,[s_uw(r72unitwork_check_lo)]
	!st	RGLhi,[s_uw(r72unitwork_check_hi)]
	st	T2a,[s_uw(r72unitwork_check_hi)]
	!ld	[s_uw(r72unitwork_cypher_hi)],T2b
	ld	[s_uw(r72unitwork_cypher_hi)],T1b
	cmp	R2B,T1b
	bne	increment_key
	 andcc	RGLhi,0xff,RGLhi
test_pipe_2_lohi_success:
	ld	[s_save_iterationsP],T2a
	ld	[T2a],T2b
	sub	T2b,RGiterations,T2b
	inc	T2b
	st	T2b,[T2a]
	b	finish
	 mov	RESULT_FOUND,%i0
test_pipe_1_lohi_success:
	ld	[s_save_iterationsP],T1a
	ld	[T1a],T1b
	sub	T1b,RGiterations,T1b
	st	T1b,[T1a]
	!b	finish
	 mov	RESULT_FOUND,%i0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
	
	ld	[s_save_i7],%i7

	ret
	restore
