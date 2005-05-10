; RC5-72 Assembly version - AMD Duron/Athlon/Athlon XP optimized version
; integer/mmx mixed version - 2 pipe
; Vyacheslav Chupyatov - goteam@mail.ru - 26/04/2003
; For use by distributed.net. 
;
; based on r72-dg2 and RC5-64 'RG/HB re-pair II' cores
; $Id: r72-go2.asm,v 1.1.2.8 2005/05/10 19:32:22 snikkel Exp $

%define P	  0xB7E15163
%define Q	  0x9E3779B9
%define S_not(N)  (P+Q*(N))


[SECTION _DATA FLAT USE32 align=16 CLASS=DATA]

incr		dd	2,3
S_not_3		dd	S_not(3),S_not(3)
S_not_4		dd	S_not(4),S_not(4)
S_not_5		dd	S_not(5),S_not(5)
S_not_6		dd	S_not(6),S_not(6)
S_not_7		dd	S_not(7),S_not(7)
S_not_8		dd	S_not(8),S_not(8)
S_not_9		dd	S_not(9),S_not(9)
S_not_10	dd	S_not(10),S_not(10)
S_not_11	dd	S_not(11),S_not(11)
S_not_12	dd	S_not(12),S_not(12)
S_not_13	dd	S_not(13),S_not(13)
S_not_14	dd	S_not(14),S_not(14)
S_not_15	dd	S_not(15),S_not(15)
S_not_16	dd	S_not(16),S_not(16)


%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_72_unit_func_go_2]
[GLOBAL rc5_72_unit_func_go_2_]
[GLOBAL rc5_72_unit_func_go_2]


%define RESULT_NOTHING	1
%define RESULT_FOUND	2

%assign work_size 0

%macro defidef 2
    %define %1 esp+%2
%endmacro

%macro defwork 1-2 1
    defidef %1,work_size
    %assign work_size work_size+4*(%2)
%endmacro

defwork work_L,3
defwork work_S,52
defwork	dummy
defwork RC5_72_L0hi
defwork work_pre1_r1
defwork work_pre2_r1
defwork work_pre3_r1
defwork work_pre4_r1
defwork work_P_0
defwork work_P_1
defwork work_C_0
defwork work_C_1
defwork work_pre5_r1
defwork work_iterations
defwork save_ebx
defwork save_esi
defwork save_edi
defwork save_ebp
defwork	S_ext,52


%define RC5_72UnitWork_plainhi	eax+0
%define RC5_72UnitWork_plainlo	eax+4
%define RC5_72UnitWork_cipherhi eax+8
%define RC5_72UnitWork_cipherlo eax+12
%define RC5_72UnitWork_L0hi	eax+16
%define RC5_72UnitWork_L0mid	eax+20
%define RC5_72UnitWork_L0lo	eax+24
%define RC5_72UnitWork_CMCcount eax+28
%define RC5_72UnitWork_CMChi	eax+32
%define RC5_72UnitWork_CMCmid	eax+36
%define RC5_72UnitWork_CMClo	eax+40

%define RC5_72UnitWork		esp+work_size+4
%define iterations		esp+work_size+8

%define S1(N)			[work_S+((N)*4*2)]
%define S2(N)			[work_S+((N)*4*2+4)]
%define L2(N)			[work_L+((N)*4)]
%define S1_alt(N)		[S_ext+((N)*4*2)]
%define S2_alt(N)		[S_ext+((N)*4*2+4)]
%define	S(N)			[work_S+(N)*8]

%macro k7nop 1
    %if %1>3
	%error k7nop max 3
    %endif
    %if %1>2
	rep
    %endif
    %if %1>1
	rep
    %endif
    nop
%endmacro

%macro k7align 1
    %assign unal ($-startseg)&(%1-1)
    %if unal
	%assign fill %1 - unal
	%if fill<=9
	    %rep 3
		%if fill>3
		    k7nop 3
		    %assign fill fill-3
		%else
		    k7nop fill
		    %exitrep
		%endif
	    %endrep
	%else
	    jmp short %%alend
	    align %1
	    %%alend:
	%endif
    %endif
%endmacro

; register allocation for the key setup blocks
%define A1	eax
%define	A2	edx

%define eA1	edx
%define eB1	ebp

%define B1_2	ebx
%define B1_1    edi
%define B1_0    esi
%define B2	ebp

; register allocation for the mmx helper
%define B3_0	mm0
%define	B3_1	mm1
%define	B3_2	mm2
%define	A3	mm4
%define	A3_H	mm6


align 16
startseg:
_rc5_72_unit_func_go_2:
rc5_72_unit_func_go_2_:
rc5_72_unit_func_go_2:
;------ integer stream -------- mmx helper ----------
				emms
				mov	eax,0x1f
				movd	mm3,eax

	sub	esp, work_size
	mov	[save_ebp], ebp

	mov	eax, [RC5_72UnitWork]
	mov	[save_ebx], ebx
	mov	[save_esi], esi

	mov	[save_edi], edi
	mov	ebx, [RC5_72UnitWork_plainlo]
	mov	edi, [RC5_72UnitWork_plainhi]

	mov	esi, [RC5_72UnitWork_cipherlo]
	mov	ecx, [RC5_72UnitWork_cipherhi]
	mov	edx, [iterations]

	mov	[work_P_0], ebx
	mov	[work_P_1], edi
	mov	ebx, [RC5_72UnitWork_L0hi]

	mov	[work_C_0], esi
	mov	[work_C_1], ecx
	mov	edi, [edx]

	shr	edi, 1
	mov	ebp, [RC5_72UnitWork_L0lo]

	mov	[work_iterations], edi
	mov	B1_0, [RC5_72UnitWork_L0mid]
	mov	[RC5_72_L0hi],ebx

k7align 16
key_setup_1_bigger_loop:
	add	B2,0xBF0A8B1D
	rol	B2,0x1D
	mov	A1,0x15235639

	mov	ecx,B2					;L(0)
	add	A1,B2					;S[1]+S[0]+L[0]
	mov	[work_pre1_r1],B2

	rol	A1,3					;S[1]
	add	B1_0,B2					;mid+L(0)

	add	ecx,A1					;L[0]+S[1]
	add	B1_0,A1					;mid+L(0)+S[1]
	mov	[work_pre2_r1],A1

	add	A1,S_not(2)				;S[1]+S_not(2)
	rol	B1_0,cl					;L(1)

	add	A1,B1_0					;S[1]+S_not(2)+L[1]

	mov	ecx,B1_0 				;L[1]
	rol	A1,3					;S[2]
	add	ecx,A1					;L[1]+S[2]
	mov	[work_pre5_r1],ecx				

	mov	[work_pre3_r1],B1_0
	mov	[work_pre4_r1],A1
	
;-----
	mov	B2,B1_2					;Lhi

	inc	B2					;Lhi+1
	add	A1,S_not(3)				;S[2]+S_not(3)
	add	B1_2,ecx				;Lhi+L[1]+S[2]

	mov	A2,A1
	rol	B1_2,cl 				;L1(2)
	add	B2,ecx					;Lhi+1+L[1]+S[2]

	add	A1,B1_2 				;S1[2]+S_not(3)+L1[2]
	rol	B2,cl 					;L2(2)
	mov	ecx,B1_2

	rol	A1,3					;S1[3]
	mov	L2(2),B2

	add	ecx,A1
	mov	B1_0,[work_pre1_r1]			;L1(0)

;------------------Round 1(4)-------------------------
	add	A2,B2	 				;S2[3]+L2[2]+S_not(3)
	mov	S1(3),A1	
	add	B1_0,ecx				;L1[2]+S1[3]+L1[0]

	rol	A2,3			     		;S2[3]
	add	A1,S_not(4)		     		;S1[3]+S_not(4)

	rol	B1_0,cl 		     		;L1[0]
	lea	ecx,[A2+B2]		     		;S2[3]+L2[2]
	mov	B2, [work_pre1_r1]			;L2(0)

	mov	S2(3),A2
	add	A1,B1_0 		     		;S1[3]+L1[0]+S_not(4)
	mov	B1_1,[work_pre3_r1]	     		;L1(1)

	add	B2,ecx			     		;L2[2]+S2[3]+L2[0]
	rol	A1,3			     		;S1[4]
	add	A2,S_not(4)		     		;S2[3]+S_not(4)

	rol	B2,cl 			     		;L2[0]
	lea	ecx,[A1+B1_0]		     		;S1[4]+L1[0]
	mov	L2(0),B2

	;---
	add	A2,B2	 		     		;S2[3]+L2[0]+S_not(4)
	mov	S1(4),A1
	add	A1,S_not(5)		     		;S1[4]+S_not(5)

	add	B1_1,ecx		     		;L1[1]+S1[4]+L1[0]
	rol	A2,3			     		;S2[4]

	rol	B1_1,cl 		     		;L1[1]
	lea	ecx,[A2+B2]		 		;S2[4]+L2[0]

	mov	S2(4),A2
	add	A1,B1_1 		     		;S1[4]+L1[1]+S_not(5)

	rol	A1,3			     		;S1[5]
	mov	B2,[work_pre3_r1]			;L2(1)
	mov	S1(5),A1

	add	B2,ecx			     		;L2[1]+S2[4]+L2[0]
	add	A2,S_not(5)		     		;S2[4]+S_not(5)

	rol	B2,cl 			     		;L2[1]
	lea	ecx,[A1+B1_1]		 		;S1[5]+L1[1]
	mov	L2(1),B2
;---------------------Round 1(6)-------------------
	add	A2,B2
	add	A1,S_not(6)

	rol	A2,3
	add	B1_2,ecx

	mov	S2(5),A2
	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	add	A2,S_not(6)

	add	B2,ecx
	rol	A1,3

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
	;---
	add	A2,B2
	mov	S1(6),A1
	rol	A2,3

	add	A1,S_not(7)
	add	B1_0,ecx

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	mov	S2(6),A2
	add	A2,S_not(7)
	add	A1,B1_0

	add	B2,ecx
	rol	A1,3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
;---------------------Round 1(8)-------------------
	add	A2,B2
	mov	S1(7),A1
	add	A1,S_not(8)

	rol	A2,3
	add	B1_1,ecx

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)
	mov	S2(7),A2

	add	A2,S_not(8)
	add	A1,B1_1

	add	B2,ecx
	rol	A1,3

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;--- 
	add	A2,B2
	mov	S1(8),A1
	add	A1,S_not(9)

	rol	A2,3
	add	B1_2,ecx

	rol	B1_2,cl	
	lea	ecx,[A2+B2]
	mov	B2,L2(2)
	mov	S2(8),A2

	add	A1,B1_2
	add	A2,S_not(9)

	add	B2,ecx
	rol	A1,3

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 1(10)-------------------
	add	A2,B2
	mov	S1(9),A1
	add	A1,S_not(10)

	add	B1_0,ecx
	rol	A2,3

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2(9),A2
	add	A2,S_not(10)

	add	B2,ecx
	rol	A1,3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
	;---
	add	A2,B2
	mov	S1(10),A1
	add	A1,S_not(11)

	rol	A2,3
	mov	S2(10),A2
	add	B1_1,ecx

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	add	A2,S_not(11)

	rol	A1,3
	add	B2,ecx

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 1(12)-------------------
	add	A2,B2
	mov	S1(11),A1
	add	A1,S_not(12)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2(11),A2
	add	A2,S_not(12)

	add	B2,ecx
	rol	A1,3

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
	;---
	add	A2,B2
	mov	S1(12),A1
	add	A1,S_not(13)

	rol	A2,3
	mov	S2(12),A2
	add	B1_0,ecx

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	add	A2,S_not(13)

	rol	A1,3
	add	B2,ecx

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2

;---------------------Round 1(14)-------------------
	add	A2,B2
	mov	S1(13),A1
	add	A1,S_not(14)

	add	B1_1,ecx
	rol	A2,3

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2(13),A2
	add	A2,S_not(14)

	add	B2,ecx
	rol	A1,3

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add	A2,B2
	mov	S1(14),A1
	add	A1,S_not(15)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2(14),A2

	rol	A1,3
	add	A2,S_not(15)
	add	B2,ecx

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 1(16)-------------------
	add	A2,B2
	mov	S1(15),A1
	add	A1,S_not(16)

	rol	A2,3
	add	B1_0,ecx
 
	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2(15),A2

	add	A2,S_not(16)
	add	B2,ecx
	rol	A1,3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
	mov	S1(16),A1
	;---
k7align 16
key_setup_1_inner_loop:
	add	A2,B2
	add	A1,S_not(17)
					pshufw	B3_2,[RC5_72_L0hi],0x44 

	add	B1_1,ecx
	rol	A2,3
					pshufw	B3_2,[RC5_72_L0hi],0x44 

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	S2(16),A2

	add	A1,B1_1
	mov	B2,L2(1)
					pshufw	A3,[work_pre4_r1],0x44 

	rep	add	B2,ecx
	rep	rol	A1,3
	add	A2,S_not(17)

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 1(18)-------------------
	add	A2,B2
	mov	S1(17),A1
	add	A1,S_not(18)

	add	B1_2,ecx
	rol	A2,3
					pshufw	mm5,[work_pre5_r1],0x44 

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2(17),A2
	add	A2,S_not(18)

	add	B2,ecx
	rol	A1,3
					paddd	B3_2,[incr]

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
	;---
	add	A2,B2
	mov	S1(18),A1
	add	A1,S_not(19)

	add	B1_0,ecx
	rol	A2,3
					pshufw	mm7,mm5,0xee

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	S2(18),A2

	add	A1,B1_0
	mov	B2,L2(0)
	add	A2,S_not(19)

	add	B2,ecx
	rol	A1,3
					paddd	A3,[S_not_3]

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
;---------------------Round 1(20)-------------------
	add	A2,B2
	mov	S1(19),A1
	add	A1,S_not(20)

	add	B1_1,ecx
	rol	A2,3
					paddd	B3_2,mm5

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2(19),A2
	add	A2,S_not(20)

	rol	A1,3
	add	B2,ecx
					pand	mm5,mm3

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add	A2,B2
	mov	S1(20),A1
	add	A1,S_not(21)

	add	B1_2,ecx
	rol	A2,3
					pshufw	A3_H,B3_2,0xee

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2(20),A2
	add	A2,S_not(21)

	rol	A1,3
	add	B2,ecx
					pand	mm7,mm3

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 1(22)-------------------
	add	A2,B2
	mov	S1(21),A1
	add	A1,S_not(22)

	add	B1_0,ecx
	rep	rol	A2,3
					punpckldq	B3_2,B3_2

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	S2(21),A2

	add	A1,B1_0
					psllq	A3_H,mm7
	mov	B2,L2(0)

	rol	A1,3
	add	B2,ecx
	add	A2,S_not(22)

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	S1(22),A1
	;---
	add	A2,B2
	mov	L2(0),B2
	add	A1,S_not(23)

	add	B1_1,ecx
	rol	A2,3
					psllq	B3_2,mm5

	rep	rol	B1_1,cl
	lea	ecx,[A2+B2]
	rep	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2(22),A2
	add	A2,S_not(23)

	rol	A1,3
	rep	add	B2,ecx
					punpckhdq	B3_2,A3_H

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 1(24)-------------------
	add	A2,B2
	mov	S1(23),A1
	add	A1,S_not(24)

	add	B1_2,ecx
	rep	rol	A2,3
					paddd	A3,B3_2

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	rep	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2(23),A2
					movq	mm5,B3_2

	rol	A1,3
	add	B2,ecx
	add	A2,S_not(24)		

	rol	B2,cl				
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
	;---
	add	A2,B2
	mov	S1(24),A1
	add	A1,S_not(25)

	add	B1_0,ecx
	rol	A2,3
					movq	A3_H,A3

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2(24),A2
                                        pslld	A3_H,3

	rol	A1,3
	add	B2,ecx
	add	A2,S_not(25)

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
;---------------------Round 1(Last)-------------------
	add	A2,B2
	mov	S1(25),A1
	add	A1,0xBF0A8B1D				;S(0)

	add	B1_1,ecx
	rol	A2,3					;S2(25)
					psrld	A3,29

	rol	B1_1,cl 				;L1(1)
	lea	ecx,[A2+B2]
					pshufw	B3_0,[work_pre1_r1],0x44

	add	A1,B1_1
	mov	B2,L2(1)
	mov	S2(25),A2

	rep	rol	A1,3
	add	B2,ecx
	add	A2,0xBF0A8B1D				;S(0)

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add	A2,B2
	mov	S1(0),A1
	add	A1,[work_pre2_r1]			;S[1]

	add	B1_2,ecx
	rol	A2,3
					por	A3,A3_H

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	rep	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(0),A2
	add	A2,[work_pre2_r1]

	rol	A1,3
	add	B2,ecx
					paddd	mm5,A3

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 2(2)-------------------
	add	A2,B2
	mov	S1(1),A1
	add	A1,[work_pre4_r1]			;S(2)

	rol	A2,3					;S2(25)
	add	B1_0,ecx
					paddd	B3_0,mm5

	rol	B1_0,cl 				;L1(1)
	rep	lea	ecx,[A2+B2]
	rep	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(1),A2
	add	A2,[work_pre4_r1]			;S(2)

	rol	A1,3
	add	B2,ecx
					pshufw	mm7,mm5,0xee

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	rep	mov	L2(0),B2
	;---
	add	A2,B2
	mov	S1(2),A1
	add	A1,S1(3)

	rep	add	B1_1,ecx
	rol	A2,3
					pand	mm5,mm3

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(2),A2
	add	A2,S2(3)

	rep	add	B2,ecx
	rep	rol	A1,3
					pshufw	A3_H,B3_0,0xee

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 2(4)-------------------
	rep	add	A2,B2
	mov	S1_alt(3),A1
	add	A1,S1(4)

	add	B1_2,ecx
	rol	A2,3
                                        movq	S(3),A3

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(3),A2
	add	A2,S2(4)

	rol	A1,3
	add	B2,ecx
					paddd	A3,[S_not_4]

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
	;---
	add 	A2,B2
	mov	S1_alt(4),A1
	add	A1,S1(5)

	rep	add	B1_0,ecx
	rol	A2,3
					pand	mm7,mm3

	rol 	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add 	A1,B1_0
	mov	S2_alt(4),A2
	add	A2,S2(5)

	rol	A1,3
	add	B2,ecx
					punpckldq	B3_0,B3_0

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
;---------------------Round 2(6)-------------------
	add	A2,B2
	mov	S1_alt(5),A1
	add	A1,S1(6)

	add	B1_1,ecx
	rol	A2,3
					psllq	A3_H,mm7

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(5),A2
	add	A2,S2(6)

	add	B2,ecx
	rol	A1,3
					psllq	B3_0,mm5

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add 	A2,B2
	mov	S1_alt(6),A1
	add	A1,S1(7)

	rep	add	B1_2,ecx
	rep	rol	A2,3
					punpckhdq	B3_0,A3_H

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(6),A2
	add	A2,S2(7)

	rol	A1,3
	rep	add	B2,ecx
				        paddd	A3,B3_0

	rep	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 2(8)-------------------
	add 	A2,B2
	mov	S1(7),A1
	add	A1,S1(8)

	add	B1_0,ecx
	rol	A2,3
                                        pshufw	B3_1,[work_pre3_r1],0x44

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	rep	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(7),A2
	add	A2,S2(8)

	add 	B2,ecx
	rep	rol	A1,3
					movq	A3_H,A3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
	;---
	add	A2,B2
	mov	S1(8),A1
	add	A1,S1(9)

	add	B1_1,ecx
	rol	A2,3
					pslld	A3_H,3

	rep	rol 	B1_1,cl
	rep	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(8),A2
	add	A2,S2(9)

	rol	A1,3
	add	B2,ecx
					psrld	A3,29

	rep	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 2(10)-------------------
	add	A2,B2
	mov	S1(9),A1
	add	A1,S1(10)

	add	B1_2,ecx
	rol	A2,3
					por	A3,A3_H

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(9),A2
	add	A2,S2(10)

	rol	A1,3
	add	B2,ecx
					movq	mm5,B3_0

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	rep	mov	L2(2),B2
	;---
	add	A2,B2
	mov	S1(10),A1
	add	A1,S1(11)

	rol	A2,3
	add	B1_0,ecx
					paddd	mm5,A3

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(10),A2
	add	A2,S2(11)

	rol	A1,3
	add	B2,ecx
					movq	S(4),A3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
;---------------------Round 2(12)-------------------
	add	A2,B2
	mov	S1(11),A1
					paddd	B3_1,mm5

	add	B1_1,ecx
	rep	rol	A2,3
	rep	add	A1,S1(12)

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(11),A2
	add	A2,S2(12)

	rol	A1,3
	add	B2,ecx
					paddd	A3,[S_not_5]

	rol 	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add	A2,B2
	mov	S1(12),A1
	add	A1,S1(13)

	add	B1_2,ecx
	rep	rol	A2,3
					pshufw	mm7,mm5,0xee

	rep	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(12),A2
	add	A2,S2(13)

	rol	A1,3
	rep	add	B2,ecx
					pand	mm5,mm3

	rep	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 2(14)-------------------
	add	A2,B2
	mov	S1(13),A1
	add	A1,S1(14)

	add	B1_0,ecx
	rol	A2,3
					pand	mm7,mm3

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(13),A2
	add	A2,S2(14)

	rol	A1,3
	add	B2,ecx
					pshufw	A3_H,B3_1,0xee

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	rep	mov	L2(0),B2
	;---
	add	A2,B2
	mov	S1(14),A1
	add	A1,S1(15)

	add	B1_1,ecx
	rol	A2,3
					punpckldq	B3_1,B3_1

	rol 	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(14),A2
	add	A2,S2(15)

	rol	A1,3
	add	B2,ecx
					psllq	A3_H,mm7

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 2(16)-------------------
	add	A2,B2
	mov	S1(15),A1
	add	A1,S1(16)

	add	B1_2,ecx
	rol	A2,3
					psllq	B3_1,mm5

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(15),A2
	add	A2,S2(16)

	rol	A1,3
	add	B2,ecx
					punpckhdq	B3_1,A3_H

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	S1(16),A1
	;---
	rep	add	A2,B2
	mov	L2(2),B2
					paddd	A3,B3_1

	add	B1_0,ecx
	rol	A2,3
	add	A1,S1(17)

	rep	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(16),A2
	add	A2,S2(17)

	rol	A1,3
	add	B2,ecx
					movq	A3_H,A3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
;---------------------Round 2(18)-------------------
	add	A2,B2
	mov	S1(17),A1
	add	A1,S1(18)

	add	B1_1,ecx
	rol	A2,3
					pslld	A3_H,3

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	rep	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(17),A2
	add	A2,S2(18)

	rol	A1,3
	rep	add	B2,ecx
					psrld	A3,29

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add	A2,B2
	mov	S1(18),A1
	add	A1,S1(19)

	rep	add	B1_2,ecx
	rep	rol	A2,3
					por	A3,A3_H

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(18),A2
	add	A2,S2(19)

	add	B2,ecx
	rol	A1,3
					movq	S(5),A3

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 2(20)-------------------
	add	A2,B2
	mov	S1(19),A1
	add	A1,S1(20)

	add	B1_0,ecx
	rol	A2,3
					movq	mm5,B3_1

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	rep	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(19),A2
	add	A2,S2(20)

	add	B2,ecx
	rol	A1,3
					paddd	mm5,A3	

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
	;---
	add	A2,B2
	mov	S1(20),A1
	add	A1,S1(21)

	add	B1_1,ecx
	rol	A2,3
					paddd	A3,[S_not_6]

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	S2_alt(20),A2

	add	A1,B1_1
	mov	B2,L2(1)
	add	A2,S2(21)

	rol	A1,3
	add	B2,ecx
					paddd	B3_2,mm5

	rep	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 2(22)-------------------
	add	A2,B2
	mov	S1(21),A1
	add	A1,S1(22)

	add	B1_2,ecx
	rol	A2,3
					pshufw	mm7,mm5,0xee

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(21),A2
	add	A2,S2(22)

	rep	rol 	A1,3
	rep	add	B2,ecx
					pand	mm5,mm3

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
	;---
	add	A2,B2
	mov	S1(22),A1
	add	A1,S1(23)

	rol	A2,3
	add	B1_0,ecx
					pshufw	A3_H,B3_2,0xee

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	rep	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(22),A2
	add	A2,S2(23)

	add	B2,ecx
	rol	A1,3
					pand	mm7,mm3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
;---------------------Round 2(24)-------------------
	add	A2,B2
	mov	S1(23),A1
	add	A1,S1(24)

	rol	A2,3
	add	B1_1,ecx
					punpckldq	B3_2,B3_2

	rol 	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(23),A2
	add	A2,S2(24)

	rol	A1,3
	add	B2,ecx
					psllq	A3_H,mm7

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add	A2,B2
	mov	S1(24),A1
	add	A1,S1(25)

	add	B1_2,ecx
	rol	A2,3
					psllq	B3_2,mm5

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(24),A2
					punpckhdq	B3_2,A3_H

	rep	add	B2,ecx
	rol	A1,3
	add	A2,S2(25)

	rep	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2

	add	A2,B2 				;S2[24]+S2[25]+L2[2]
	mov	S1(25),A1
					paddd	A3,B3_2

;---------------------Round 2-Last-------------
	add	B1_0,ecx				;L1[0]+S1[25]+L1[2]
	rol	A2,3					;S2[25]	
	add	A1,S1(0)				;S1[25]+S1[0]

	rep	rol	B1_0,cl 				;L1[0]
	rep	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0 				;S1[25]+S1[0]+L1[0]
	mov	S2_alt(25),A2
					movq	A3_H,A3

	rol	A1,3					;S1[0]
	add	B2,ecx					;L2[0]+S2[25]+L2[2]
	add	A2,S2_alt(0)				;S2[25]+S2[0]

	rol	B2,cl					;L2[0]
	lea	ecx,[A1+B1_0]				;S1[0]+L1[0]
					pslld	A3_H,3

	add	A2,B2 					;S2[25]+L2[0]+S2[0]
	mov	L2(0),B2
					psrld	A3,29
	;---
	rep	add 	B1_1,ecx				;L1[1]+S1[0]+L1[0]
	rol	A2,3					;S2[0]
	mov	S2_alt(0),A2
	
	rol	B1_1,cl 				;L1[1]
	lea	ecx,[B2+A2]
	mov	B2,L2(1)

	mov	eA1,[work_P_0]
	add	eA1,A1					;eA1
	add	A1,S1(1)				;S1[0]+S1[1]

	add 	B2,ecx					;L2[1]+S2[0]+L2[0]
					por	A3,A3_H
					movq	mm5,B3_2

	rep	rol	B2,cl 					;L2[1]
	mov	L2(1),B2				;key #2
	add 	A1,B1_1 				;S1[0]+S1[1]+L1[1]

	mov	eB1,[work_P_1]
	mov	ecx,B1_1
	rol	A1,3					;S1(1)

;----
	add	ecx,A1
	add	eB1,A1					;eB1
	add	A1,S1(2)

	xor	eA1,eB1
	add	B1_2,ecx
					movq	S(6),A3

	rol	B1_2,cl	
	;start ROUND3 mixed with encryption
	mov	ecx,eB1
					paddd	mm5,A3

	add	A1,B1_2
	rol	eA1,cl
					paddd	B3_0,mm5

	rol	A1,3		       
	rep	mov 	ecx,B1_2
					paddd	A3,[S_not_7]

	rep	add	eA1,A1
	rep	add	ecx,A1
	add	A1,S1_alt(3)

	add	B1_0,ecx
	xor	eB1,eA1
					pshufw	mm7,mm5,0xee

	rol	B1_0,cl 	       
	mov	ecx,eA1
					pand	mm5,mm3

	add	A1,B1_0
	rol	eB1,cl
	mov	ecx,B1_0

	rep	rol 	A1,3		       
					pshufw	A3_H,B3_0,0xee
					pand	mm7,mm3

	add	eB1,A1	
	add	ecx,A1
	add	A1,S1_alt(4)
	
	xor	eA1,eB1
	add	B1_1,ecx
					punpckldq	B3_0,B3_0

	rol	B1_1,cl
	mov	ecx,eB1
					psllq	A3_H,mm7

	add	A1,B1_1
	rol	eA1,cl
	mov	ecx,B1_1

	rep	rol	A1,3
					psllq	B3_0,mm5
					punpckhdq	B3_0,A3_H

	add	eA1,A1
	add	ecx,A1
	add	A1,S1_alt(5)

	rep	xor	eB1,eA1
	add	B1_2,ecx
					paddd	A3,B3_0

	rol	B1_2,cl
	mov	ecx,eA1	
					movq	mm5,B3_0

	rol	eB1,cl
	mov	ecx,B1_2
	add	A1,B1_2

	rol	A1,3
					pshufw	A3_H,A3,0xee
					punpckldq	A3,A3

	rep	add	ecx,A1
	rep	add	eB1,A1
	add	A1,S1_alt(6)

	add	B1_0,ecx
	xor	eA1,eB1
					psllq	A3_H,3

	rol	B1_0,cl
;------------Round 3(1)------------
	mov	ecx,eB1
					psllq	A3,3

	rep	add	A1,B1_0
	rep	rol	eA1,cl
	rep	mov	ecx,B1_0

	rol	A1,3		       
					punpckhdq	A3,A3_H
					paddd	mm5,A3

	add	ecx,A1
	add	eA1,A1
					paddd	B3_1,mm5

	add 	B1_1,ecx
	xor	eB1,eA1
	add	A1,S1(7)

	rol	B1_1,cl 	       
	mov	ecx,eA1
					movq	S(7),A3

	rol	eB1,cl
	add	A1,B1_1
					pshufw	mm7,mm5,0xee

	rep	rol	A1,3		       
	rep	mov	ecx,B1_1
                                        paddd	A3,[S_not_8]

	add 	ecx,A1
	add	eB1,A1
	add	A1,S1(8)

	add	B1_2,ecx
	xor	eA1,eB1
					pand	mm5,mm3

	rol	B1_2,cl
;---
	mov	ecx,eB1
					pshufw	A3_H,B3_1,0xee

	add	A1,B1_2
	rol	eA1,cl
	mov	ecx,B1_2

	rol	A1,3
					pand	mm7,mm3
					punpckldq	B3_1,B3_1

	rep	add	ecx,A1
	add	eA1,A1
	add	A1,S1(9)

	add	B1_0,ecx
	xor	eB1,eA1
					psllq	A3_H,mm7

	rol	B1_0,cl
	mov	ecx,eA1
					psllq	B3_1,mm5

	add	A1,B1_0
	rol	eB1,cl
	mov	ecx,B1_0

	rol	A1,3
					punpckhdq	B3_1,A3_H
					paddd	A3,B3_1

	add	ecx,A1
	add	eB1,A1
	add	A1,S1(10)

	rep	add	B1_1,ecx
	xor	eA1,eB1
					movq	A3_H,A3

	rol	B1_1,cl
;-------
	mov	ecx,eB1
					pslld	A3_H,3

	rep	add	A1,B1_1
	rol	eA1,cl
	mov	ecx,B1_1		

	rol	A1,3
					psrld	A3,29
					por	A3,A3_H

	add	ecx,A1
	add	eA1,A1
	add	A1,S1(11)

	add	B1_2,ecx
	xor	eB1,eA1
					movq	mm5,B3_1

	rep	rol	B1_2,cl
	mov	ecx,eA1
					paddd	mm5,A3

	add	A1,B1_2
	rol	eB1,cl
	mov	ecx,B1_2

	rep	rol	A1,3		       
					paddd	B3_2,mm5
					movq	S(8),A3

	add	ecx,A1
	add	eB1,A1
					pshufw	mm7,mm5,0xee

	add	B1_0,ecx
	xor	eA1,eB1
	add	A1,S1(12)

	rol	B1_0,cl
;------------Round 3(4)------------
	mov	ecx,eB1
					pshufw	A3_H,B3_2,0xee

	rep	add	A1,B1_0
	rol	eA1,cl
	mov	ecx,B1_0

	rol	A1,3		       
					pand	mm5,mm3
					pand	mm7,mm3

	rep	add	ecx,A1
	add	eA1,A1
	add	A1,S1(13)

	add	B1_1,ecx
	xor	eB1,eA1
					paddd	A3,[S_not_9]

	rol	B1_1,cl 	       
	mov	ecx,eA1
					punpckldq	B3_2,B3_2

	add	A1,B1_1
	rol	eB1,cl
	mov	ecx,B1_1

	rol	A1,3		       
					psllq	A3_H,mm7
					psllq	B3_2,mm5

	add	ecx,A1
	add	eB1,A1
	rep	add	A1,S1(14)

	add	B1_2,ecx
	xor	eA1,eB1
					punpckhdq	B3_2,A3_H

	rep	rol	B1_2,cl
;---
	rep	mov	ecx,eB1
					paddd	A3,B3_2

	add	A1,B1_2
	rol	eA1,cl
	mov	ecx,B1_2

	rol	A1,3
					movq	A3_H,A3
					psrld	A3,29

	add	ecx,A1
	add	eA1,A1
					pslld	A3_H,3

	add	B1_0,ecx
	xor	eB1,eA1
	add	A1,S1(15)

	rol	B1_0,cl
	mov	ecx,eA1
					movq	mm5,B3_2

	add	A1,B1_0
	rol	eB1,cl
	mov	ecx,B1_0

	rol	A1,3
					por	A3,A3_H
					movq	S(9),A3

	rep	add	eB1,A1
	add	ecx,A1
	add	A1,S1(16)

	rep	add	B1_1,ecx
	rep	xor	eA1,eB1
					paddd	mm5,A3

	rol	B1_1,cl
;-------
	mov	ecx,eB1
					paddd	A3,[S_not_10]

	rep	add	A1,B1_1
	rol	eA1,cl
	mov	ecx,B1_1

	rol	A1,3
					paddd	B3_0,mm5
					pshufw	mm7,mm5,0xee

	add	ecx,A1
	add	eA1,A1
	add	A1,S1(17)

	add	B1_2,ecx
	xor	eB1,eA1
					pand	mm5,mm3

	rol	B1_2,cl
	mov	ecx,eA1
					pshufw	A3_H,B3_0,0xee

	add	A1,B1_2
	rol	eB1,cl
	mov	ecx,B1_2

	rep	rol	A1,3		       
					pand	mm7,mm3
					punpckldq	B3_0,B3_0

	add	eB1,A1
	add	ecx,A1
	add	A1,S1(18)

	add	B1_0,ecx
	xor	eA1,eB1
					psllq	A3_H,mm7

	rol	B1_0,cl
	mov	ecx,eB1
					psllq	B3_0,mm5
;------------Round 3(7)------------
	add	A1,B1_0
	rol	eA1,cl
	mov	ecx,B1_0

	rol	A1,3		       
					punpckhdq	B3_0,A3_H
					paddd	A3,B3_0

	add	eA1,A1
	add	ecx,A1
	add	A1,S1(19)

	xor	eB1,eA1
	add	B1_1,ecx
					movq	A3_H,A3

	rol	B1_1,cl
	mov	ecx,eA1
					pslld	A3_H,3

	add	A1,B1_1
	rol	eB1,cl
	mov	ecx,B1_1

	rol	A1,3		       
					psrld	A3,29
					por	A3,A3_H

	add	ecx,A1
	rep	add	eB1,A1
					movq	mm5,B3_0

	add	B1_2,ecx
	xor	eA1,eB1
	add	A1,S1(20)

	rep	rol	B1_2,cl
;---
	mov	ecx,eB1
					paddd	mm5,A3

	add	A1,B1_2
	rol	eA1,cl
	mov	ecx,B1_2

	rol	A1,3
					movq	S(10),A3
					paddd	A3,[S_not_11]

	add	ecx,A1
	add	eA1,A1
	add	A1,S1(21)
	
	add	B1_0,ecx
	xor	eB1,eA1
					paddd	B3_1,mm5

	rep	rol	B1_0,cl
	rep	mov	ecx,eA1
					pshufw	mm7,mm5,0xee

	add	A1,B1_0
	rol	eB1,cl
	mov	ecx,B1_0

	rol	A1,3
					pand	mm5,mm3
					pshufw	A3_H,B3_1,0xee

	add	ecx,A1
	add	eB1,A1
	add	A1,S1(22)

	add	B1_1,ecx
	xor	eA1,eB1
                                        pand	mm7,mm3

	rol	B1_1,cl
;-------
	mov	ecx,eB1
					punpckldq	B3_1,B3_1

	rep	add	A1,B1_1
	rol	eA1,cl
	mov	ecx,B1_1

	rol	A1,3
					psllq	A3_H,mm7
					psllq	B3_1,mm5

	rep	add	ecx,A1
	add	eA1,A1
					punpckhdq	B3_1,A3_H

	add	B1_2,ecx        
	xor	eB1,eA1
	add	A1,S1(23)

	rol	B1_2,cl
	mov	ecx,eA1
                                        paddd	A3,B3_1

	rep	add	A1,B1_2
	rep	rol	eB1,cl
	rep	mov	ecx,B1_2

	rol	A1,3		      
					movq	A3_H,A3
					pslld	A3_H,3

	rep	add	ecx,A1
	rep	add	eB1,A1
	add	A1,S1(24)

	add	B1_0,ecx
	xor	eA1,eB1
					psrld	A3,29

	rol	B1_0,cl
;------------Round 3(10)------------
	mov	ecx,eB1
					por	A3,A3_H

	add	A1,B1_0
	rol	eA1,cl
					movq	mm5,B3_1

	rep	rol	A1,3		       
	rep	add	eA1,A1
					movq	S(11),A3

	cmp	eA1,[work_C_0]
					paddd	mm5,A3
	je	near _checkKey1High_k7_mixed

;Finished Key #1, move data from mmx to integer pipe
;Load mmx unit with round 1 data for the next pair of keys
_Key2Round3_k7_mixed:
	mov	eA1,[work_P_0]
	mov	A1,S2_alt(0)
	add	eA1,A1

	mov	B1_1,L2(1)
	add	A1,S2_alt(1)
					paddd	B3_2,mm5

	mov	ecx,B1_1
	mov	eB1,[work_P_1]
	mov	B1_2,L2(2)

	add	A1,B1_1
	mov	B1_0,L2(0)
					pshufw	mm7,mm5,0xee

	rol	A1,3
					pand	mm5,mm3
					pshufw	A3_H,B3_2,0xee

	add	eB1,A1
	add	ecx,A1
	add	A1,S2_alt(2)

	add	B1_2,ecx
	xor	eA1,eB1
					punpckldq	B3_2,B3_2

	rol	B1_2,cl
	mov	ecx,eB1
					pand	mm7,mm3

	add	A1,B1_2
	rol	eA1,cl
	mov	ecx,B1_2

	rep	rol	A1,3
					psllq	B3_2,mm5
					psllq	A3_H,mm7

	add	eA1,A1
	add	ecx,A1
	add	A1,S2_alt(3)

	rep	add	B1_0,ecx
	xor	eB1,eA1
					paddd	A3,[S_not_12]
					
					
	rol	B1_0,cl
	mov	ecx,eA1
					punpckhdq	B3_2,A3_H

	add	A1,B1_0
	rol	eB1,cl
	mov	ecx,B1_0

	rol	A1,3
					paddd	A3,B3_2
					movq	A3_H,A3

	rep	add	ecx,A1
	add	eB1,A1
	add	A1,S2_alt(4)

	add	B1_1,ecx
	xor	eA1,eB1
					movq	mm5,B3_2

	rol	B1_1,cl
	mov	ecx,eB1
					psrld	A3,29

	add	A1,B1_1
	rol	eA1,cl
	mov	ecx,B1_1

	rol	A1,3
					pslld	A3_H,3
					por	A3,A3_H

	add	eA1,A1
	rep	add	ecx,A1
					paddd	mm5,A3

	add	B1_2,ecx
	xor	eB1,eA1
	add	A1,S2_alt(5)

	rep	rol	B1_2,cl
	mov	ecx,eA1
					movq	S(12),A3

	add	A1,B1_2
	rol	eB1,cl
	mov	ecx,B1_2

	rol	A1,3
					paddd	A3,[S_not_13]
					paddd	B3_0,mm5

	add	eB1,A1
	add	ecx,A1
					pshufw	mm7,mm5,0xee

	add	B1_0,ecx
	xor	eA1,eB1
	add	A1,S2_alt(6)

	rep	rol	B1_0,cl
	mov	ecx,eB1
					pand	mm5,mm3

	add	A1,B1_0
	rol	eA1,cl
	mov	ecx,B1_0

	rol	A1,3
					pshufw	A3_H,B3_0,0xee
					punpckldq	B3_0,B3_0

	add	eA1,A1
	add	ecx,A1
	add	A1,S2_alt(7)

	add	B1_1,ecx
	xor	eB1,eA1
					pand	mm7,mm3

	rep	rol	B1_1,cl
	rep	mov	ecx,eA1
					psllq	B3_0,mm5

	rol	eB1,cl
	add	A1,B1_1
	mov	ecx,B1_1

	rol	A1,3
					psllq	A3_H,mm7
					punpckhdq	B3_0,A3_H
					
	rep	add	eB1,A1
	add	ecx,A1
					paddd	A3,B3_0

	xor	eA1,eB1
	add	B1_2,ecx
	add	A1,S2_alt(8)

	rol	B1_2,cl
	mov	ecx,eB1
					movq	A3_H,A3

	rol	eA1,cl
	rep	add	A1,B1_2
					psrld	A3,29
;---

	rol	A1,3
	mov	ecx,B1_2	
					pslld	A3_H,3

	add	ecx,A1
	add	eA1,A1
					por	A3,A3_H

	add	B1_0,ecx
	xor	eB1,eA1
	add	A1,S2_alt(9)

	rep	rol	B1_0,cl
	mov	ecx,eA1
					movq	mm5,B3_0

	add	A1,B1_0
	rol	eB1,cl
	mov	ecx,B1_0

	rep	rol	A1,3
					paddd	mm5,A3
					paddd	B3_1,mm5

	add	ecx,A1
	add	eB1,A1
	add	A1,S2_alt(10)

	add	B1_1,ecx
	xor	eA1,eB1
					movq	S(13),A3

	rol	B1_1,cl
;-------
	rep	mov	ecx,eB1
					pshufw	mm7,mm5,0xee


	add	A1,B1_1
	rol	eA1,cl
					paddd	A3,[S_not_14]

	mov	ecx,B1_1
	rol	A1,3
					pand	mm5,mm3

	add	ecx,A1
	add	eA1,A1
	add	A1,S2_alt(11)

	add	B1_2,ecx
	xor	eB1,eA1
					pshufw	A3_H,B3_1,0xee

	rol	B1_2,cl
	mov	ecx,eA1
					punpckldq	B3_1,B3_1

	add	A1,B1_2
	rol	eB1,cl
					pand	mm7,mm3

	rep	mov	ecx,B1_2
	rep	rol	A1,3		       
					psllq	B3_1,mm5

	add	ecx,A1
	add	eB1,A1
	add	A1,S2_alt(12)

	add	B1_0,ecx
	xor	eA1,eB1
					psllq	A3_H,mm7

	rol	B1_0,cl
;------------Round 3(4) - key 2------------
	mov	ecx,eB1
					punpckhdq	B3_1,A3_H

	rep	add	A1,B1_0
	rep	rol	eA1,cl
					paddd	A3,B3_1

	mov	ecx,B1_0
	rol	A1,3
					movq	A3_H,A3

	add	ecx,A1
	add	eA1,A1
	add	A1,S2_alt(13)

	add	B1_1,ecx
	xor	eB1,eA1
					movq	mm5,B3_1

	rep	rol	B1_1,cl 	       
	rep	mov	ecx,eA1
					psrld	A3,29

	add	A1,B1_1
	rol	eB1,cl
	mov	ecx,B1_1

	rol	A1,3		       
					pslld	A3_H,3
					por	A3,A3_H

	add	ecx,A1
	add	eB1,A1
	add	A1,S2_alt(14)

	add	B1_2,ecx
	xor	eA1,eB1
					movq	S(14),A3

	rol	B1_2,cl
	;---
	mov	ecx,eB1
					paddd	mm5,A3

	add	A1,B1_2
	rol	eA1,cl
	mov	ecx,B1_2

	rol	A1,3
					paddd	B3_2,mm5
					pshufw	mm7,mm5,0xee

	add	ecx,A1
	add	eA1,A1
	add	A1,S2_alt(15)

	rep	add	B1_0,ecx
	rep	xor	eB1,eA1
					paddd	A3,[S_not_15]

	rol	B1_0,cl
	mov	ecx,eA1
					pand	mm5,mm3

	add	A1,B1_0
	rol	eB1,cl
	mov	ecx,B1_0

	rol	A1,3
					pshufw	A3_H,B3_2,0xee
					punpckldq	B3_2,B3_2

	add	ecx,A1
	add	eB1,A1
	add	A1,S2_alt(16)

	add	B1_1,ecx
	xor	eA1,eB1
					pand	mm7,mm3

	rol	B1_1,cl
	;-------
	mov	ecx,eB1
					psllq	B3_2,mm5

	rep	add	A1,B1_1
	rep	rol	eA1,cl
	rep	mov	ecx,B1_1

	rep	rol	A1,3
					psllq	A3_H,mm7
					punpckhdq	B3_2,A3_H

	rep	add	ecx,A1
	rep	add	eA1,A1
					movq	L2(1),B3_2	;L2(2)
			
	add	B1_2,ecx
	xor	eB1,eA1
	add	A1,S2_alt(17)

	rol	B1_2,cl
	mov	ecx,eA1
					movq	L2(0),B3_1	;L2(1)

	add	A1,B1_2
	rol	eB1,cl
	mov	ecx,B1_2

	rol	A1,3		       
					paddd	A3,B3_2
					movq	A3_H,A3

	add	ecx,A1
	rep	add	eB1,A1
					psrld	A3,29

	add	B1_0,ecx
	xor	eA1,eB1
	add	A1,S2_alt(18)
;------------Round 3(7) - key 2------------
	rep	rol	B1_0,cl
	mov	ecx,eB1
					pslld	A3_H,3

	add	A1,B1_0                 
	rol	eA1,cl
	mov	ecx,B1_0

	rol	A1,3		       
					por	A3,A3_H
					movq	mm5,B3_2

	add	ecx,A1
	add	eA1,A1
	add	A1,S2_alt(19)

	rep	add	B1_1,ecx
	rep	xor	eB1,eA1
					movq	S(15),A3

	rep	rol	B1_1,cl 	       
	mov	ecx,eA1
					paddd	mm5,A3

	add	A1,B1_1
	rol	eB1,cl
	mov	ecx,B1_1
					
	rol	A1,3		       
					paddd	B3_0,mm5
					pshufw	mm7,mm5,0xee

	add	ecx,A1
	add	eB1,A1
	add	A1,S2_alt(20)

	add	B1_2,ecx
	xor	eA1,eB1
					pand	mm5,mm3

	rol	B1_2,cl
	;---
	mov	ecx,eB1
					pshufw	A3_H,B3_0,0xee

	add	A1,B1_2
	rol	eA1,cl
	mov	ecx,B1_2

	rol	A1,3
					punpckldq	B3_0,B3_0
					pand	mm7,mm3

	rep	add	ecx,A1
	add	eA1,A1
	add	A1,S2_alt(21)

	add	B1_0,ecx
	xor	eB1,eA1
					psllq	B3_0,mm5

	rol	B1_0,cl
	mov	ecx,eA1
					paddd	A3,[S_not_16]

	rep	add	A1,B1_0
	rol	eB1,cl
	mov	ecx,B1_0

	rol	A1,3
					psllq	A3_H,mm7
					punpckhdq	B3_0,A3_H

	add	ecx,A1
	add	eB1,A1
	add	A1,S2_alt(22)

	add	B1_1,ecx
	xor	eA1,eB1
					punpckhdq	A3_H,A3_H

	rep	rol	B1_1,cl
	;-------
	mov	ecx,eB1
					movq	S(16),A3

	add	A1,B1_1
	rol	eA1,cl
	mov	ecx,B1_1

	rol	A1,3
					movd	L2(0),A3_H
					paddd	A3,B3_0

	rep	add	ecx,A1
	rep	add	eA1,A1
	add	A1,S2_alt(23)

	add	B1_2,ecx
	xor	eB1,eA1
					punpckldq	A3,A3

	rol	B1_2,cl
	mov	ecx,eA1
					movd	S1(0),B3_0

	rep	add	A1,B1_2
	rep	rol	eB1,cl
	rep	mov	ecx,B1_2

	rol	A1,3		  
					psllq	A3,3
					punpckhdq	A3,A3

	rep	add	ecx,A1
	rep	add	eB1,A1
					movd	S1(16),A3

	add	B1_0,ecx
	xor	eA1,eB1
	add	A1,S2_alt(24)

	rol	B1_0,cl
;------------Round 3(10) - key 2------------
	mov	ecx,eB1
					movd	S1(1),B3_1

	add	A1,B1_0
					movd	S2(0),B3_2

	rol	eA1,cl
	rol	A1,3		       

	add	eA1,A1

	cmp	eA1,[work_C_0]
	je	short _checkKey2High_k7_mixed

_NextKey:
	dec	dword [work_iterations]

	jz	near finished_Found_nothing
	add	byte [RC5_72_L0hi],2

	mov	B1_0,S1(0)
	mov	A1,S1(16)
	mov	A2,S2(16)
	mov	B1_1,S1(1)

	mov	B2,L2(0)
	lea	ecx,[A1+B1_0]
	mov	B1_2,S2(0)

	jnc	key_setup_1_inner_loop
	
	mov	eax, [RC5_72UnitWork]
	mov	B1_0, [RC5_72UnitWork_L0mid]
	mov	B2, [RC5_72UnitWork_L0lo]
	mov	ebx, [RC5_72_L0hi]
	bswap	B1_0
	bswap	B2
	adc	B1_0,0
	adc	B2,0
	bswap	B1_0
	bswap	B2
	mov	[RC5_72UnitWork_L0mid],B1_0
	mov	[RC5_72UnitWork_L0lo],B2
	jmp	key_setup_1_bigger_loop

k7align 16
_checkKey2High_k7_mixed:
	lea	ecx,[A1+B1_0]

	add	A1,S2_alt(25)

	add	B1_1,ecx
	xor	eB1,eA1

	rol	B1_1,cl
	mov	ecx,eA1

	add	A1,B1_1
	rol	eB1,cl

	rol	A1,3

	add	eB1,A1

	mov	eax, [RC5_72UnitWork]
	mov	edx, [RC5_72UnitWork_L0mid]
	mov	esi, [RC5_72UnitWork_L0lo]
	mov	ebx, [RC5_72_L0hi]

        inc     dword [RC5_72UnitWork_CMCcount]
	inc	ebx
	cmp	eB1,[work_C_1]
        mov     [RC5_72UnitWork_CMChi], ebx
        mov     [RC5_72UnitWork_CMCmid], edx
        mov     [RC5_72UnitWork_CMClo], esi
	jne	_NextKey

	mov	ecx, [work_iterations]
	mov	esi, [iterations]

	shl	ecx, 1

	dec	ecx

	sub	[esi], ecx
	mov	eax, RESULT_FOUND

	jmp	finished

k7align 16
_checkKey1High_k7_mixed:
	lea	ecx,[B1_0+A1]
	add	A1,S1(25)

	add	B1_1,ecx
	xor	eB1,eA1

	rol	B1_1,cl
	mov	ecx,eA1

	add	A1,B1_1
	rol	eB1,cl

	rol	A1,3

	add	eB1,A1
	mov	eax, [RC5_72UnitWork]
	mov	edx, [RC5_72UnitWork_L0mid]
	mov	esi, [RC5_72UnitWork_L0lo]
	mov	ebx, [RC5_72_L0hi]
        inc     dword [RC5_72UnitWork_CMCcount]
	cmp	eB1,[work_C_1]
        mov     [RC5_72UnitWork_CMChi], ebx
        mov     [RC5_72UnitWork_CMCmid], edx
        mov     [RC5_72UnitWork_CMClo], esi
	jne	_Key2Round3_k7_mixed

	mov	ecx, [work_iterations]
	mov	esi, [iterations]

	shl	ecx, 1

	sub	[esi], ecx
	mov	eax, RESULT_FOUND

	jmp	finished

finished_Found_nothing:
	mov	eax, [RC5_72UnitWork]
	mov	ebx,[RC5_72_L0hi]
	add	bl,2
	mov	[RC5_72UnitWork_L0hi],ebx
	mov	edx, [RC5_72UnitWork_L0mid]
	mov	esi, [RC5_72UnitWork_L0lo]
	bswap	edx
	bswap	esi
	adc	edx,0
	adc	esi,0
	bswap	edx
	bswap	esi
	mov	[RC5_72UnitWork_L0mid],edx
	mov	[RC5_72UnitWork_L0lo],esi
	mov	eax, RESULT_NOTHING
finished:
	mov	ebx, [save_ebx]
	mov	esi, [save_esi]

	mov	edi, [save_edi]
	mov	ebp, [save_ebp]
	add	esp, work_size

	emms
	ret
