; RC5-72 Assembly version - AMD Duron/Athlon/Athlon XP optimized version
; integer/mmx mixed version - 2 pipe
; Vyacheslav Chupyatov - goteam@mail.ru - 26/04/2003
; For use by distributed.net. 
;
; based on r72-dg2 and RC5-64 'RG/HB re-pair II' cores
; $Id: r72-go2.asm,v 1.1.2.2 2003/05/22 00:16:34 andreasb Exp $

%define P	  0xB7E15163
%define Q	  0x9E3779B9
%define S_not(N)  (P+Q*(N))


%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _DATA FLAT USE32 align=16 CLASS=DATA]
%else
[SECTION .data]
%endif

incr		dd	1,1
incr2		dd	2,2
S_not_4		dd	S_not(4),S_not(4)
S_not_5		dd	S_not(5),S_not(5)
S_not_3		dd	S_not(3)


%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_72_unit_func_go_2]
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

defwork work_L1,3
defwork work_s1,26
defwork work_L2,3
defwork work_s2,26
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

%define S1(N)			[work_s1+((N)*4)]
%define S2(N)			[work_s2+((N)*4)]
%define L1(N)			[work_L1+((N)*4)]
%define L2(N)			[work_L2+((N)*4)]

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
	%endif
    %endif
    %%alend:
%endmacro

; register allocation for the key setup blocks
%define A1	eax
%define	A2	edx

%define eA1	edx
%define eB1	edi
%define B1_0	esi

%define B1_1	ebx
%define B2_1    esi
%define B1_2	ebp
%define B2_2    edi

; register allocation for the mmx helper
%define L2_0	mm4
%define L2_2	mm2
%define	L2_1	mm1
%define	eA2	mm6
%define	A3	mm0
%define	eB2	mm7


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
	mov	edx, [RC5_72UnitWork_L0mid]
	mov	esi, [RC5_72UnitWork_L0lo]

	mov	[work_iterations], edi
	mov	[RC5_72_L0hi],ebx

k7align 16
key_setup_1_bigger_loop:
	add	B2_1,0xBF0A8B1D
	rol	B2_1,0x1D
	mov	A1,0x15235639

	mov	ecx,B2_1				;L(0)
	add	A1,B2_1					;S[1]+S[0]+L[0]
	mov	[work_pre1_r1],B2_1

	rol	A1,3					;S[1]
	add	edx,B2_1				;mid+L(0)

	add	ecx,A1					;L[0]+S[1]
	add	edx,A1					;mid+L(0)+S[1]
	mov	[work_pre2_r1],A1

	add	A1,S_not(2)				;S[1]+S_not(2)
	rol	edx,cl					;L(1)

	add	A1,edx					;S[1]+S_not(2)+L[1]

	mov	ecx,edx 				;L[1]
	rol	A1,3					;S[2]
	add	ecx,A1					;L[1]+S[2]
	mov	[work_pre5_r1],ecx				

	mov	[work_pre3_r1],edx
	mov	[work_pre4_r1],A1

	mov	ecx,[work_pre5_r1]
	mov	B2_1,B1_1				;Lhi
	mov	A1,[work_pre4_r1]

	inc	B2_1					;Lhi+1
	add	A1,S_not(3)				;S[2]+S_not(3)
	add	B1_1,ecx				;Lhi+L[1]+S[2]

	mov	A2,A1
	rol	B1_1,cl 				;L1(2)
	add	B2_1,ecx				;Lhi+1+L[1]+S[2]

	add	A1,B1_1 				;S1[2]+S_not(3)+L1[2]
	rol	B2_1,cl 				;L2(2)
	mov	ecx,B1_1

	mov	L1(2),B1_1
	rol	A1,3					;S1[3]
	mov	L2(2),B2_1

	add	ecx,A1
	mov	B1_2,[work_pre1_r1]			;L1(0)

;------------------Round 1(4)-------------------------
	add	A2,B2_1 				;S2[3]+L2[2]+S_not(3)
	mov	S1(3),A1
	add	B1_2,ecx				;L1[2]+S1[3]+L1[0]

	add	A1,S_not(4)		     		;S1[3]+S_not(4)
	rol	A2,3			     		;S2[3]
	mov	B2_2, [work_pre1_r1]			;L2(0)

	rol	B1_2,cl 		     		;L1[0]
	lea	ecx,[A2+B2_1]		     		;S2[3]+L2[2]
	mov	S2(3),A2

	mov	L1(0),B1_2
	add	A1,B1_2 		     		;S1[3]+L1[0]+S_not(4)
	mov	B1_1,[work_pre3_r1]	     		;L1(1)

	add	B2_2,ecx		     		;L2[2]+S2[3]+L2[0]
	rol	A1,3			     		;S1[4]
	add	A2,S_not(4)		     		;S2[3]+S_not(4)

	rol	B2_2,cl 		     		;L2[0]
	lea	ecx,[A1+B1_2]		     		;S1[4]+L1[0]
	mov	L2(0),B2_2

	;---
	add	A2,B2_2 		     		;S2[3]+L2[0]+S_not(4)
	mov	S1(4),A1
	add	A1,S_not(5)		     		;S1[4]+S_not(5)

	add	B1_1,ecx		     		;L1[1]+S1[4]+L1[0]
	rol	A2,3			     		;S2[4]

	rol	B1_1,cl 		     		;L1[1]
	lea	ecx,[A2+B2_2]		 		;S2[4]+L2[0]

	mov	L1(1),B1_1
	mov	S2(4),A2
	add	A1,B1_1 		     		;S1[4]+L1[1]+S_not(5)

	rol	A1,3			     		;S1[5]
	mov	B2_1,[work_pre3_r1]			;L2(1)

	mov	S1(5),A1
	add	A2,S_not(5)		     		;S2[4]+S_not(5)
	add	B2_1,ecx			     	;L2[1]+S2[4]+L2[0]

				movd	eB2,[work_P_1]
	add	B1_1,A1

k7align 16
key_setup_1_inner_loop:
	mov	B1_2,L1(2)
	rol	B2_1,cl 		     		;L2[1]
	mov	ecx,B1_1
;---------------------Round 1(6)-------------------
	mov	L2(1),B2_1
	add	A2,B2_1
	rol	A2,3

	add	B1_2,ecx
	add	A1,S_not(6)
	mov	B2_2,L2(2)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(5),A2

	mov	B1_1,L1(0)
	mov	L1(2),B1_2
	add	A1,B1_2

	add	A2,S_not(6)
	add	B2_2,ecx
	rep rep rol	A1,3

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2_2
	;---
	add	A2,B2_2
	mov	S1(6),A1
	add	A1,S_not(7)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(0)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	B1_2,L1(1)

	mov	S2(6),A2
	add	A2,S_not(7)
	add	A1,B1_1

	rep mov	L1(0),B1_1
	rep rep rol	A1,3
	rep rep add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(0),B2_1
;---------------------Round 1(8)-------------------
	rep add	A2,B2_1
	mov	S1(7),A1
	add	A1,S_not(8)

	mov	B2_2,L2(1)
	rol	A2,3
	rep rep add	B1_2,ecx

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(7),A2

	add	A1,B1_2
	add	A2,S_not(8)
	mov	B1_1,L1(2)

	mov	L1(1),B1_2
	add	B2_2,ecx
	rol	A1,3

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(1),B2_2
	;--- 
	add	A2,B2_2
	mov	S1(8),A1
	add	A1,S_not(9)

	rol	A2,3
	add	B1_1,ecx
	mov	B2_1,L2(2)

	rol	B1_1,cl	
	lea	ecx,[A2+B2_2]
	mov	S2(8),A2

	rep add	A1,B1_1
	mov	B1_2,L1(0)
	mov	L1(2),B1_1

	add	A2,S_not(9)
	add	B2_1,ecx
	rol	A1,3

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(2),B2_1
;---------------------Round 1(10)-------------------
	add	A2,B2_1
	mov	S1(9),A1
	add	B1_2,ecx

	add	A1,S_not(10)
	rol	A2,3
	mov	B2_2,L2(0)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	B1_1,L1(1)

	rep rep add	A1,B1_2
	mov	S2(9),A2
	add	A2,S_not(10)

	add	B2_2,ecx
	rol	A1,3
	mov	L1(0),B1_2

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(0),B2_2
	;---
	add	A2,B2_2
	mov	S1(10),A1
	add	A1,S_not(11)

	rol	A2,3
	mov	S2(10),A2
	add	B1_1,ecx

	mov	B2_1,L2(1)
	rol	B1_1,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_1
	mov	L1(1),B1_1
	rol	A1,3

	add	A2,S_not(11)
	add	B2_1,ecx
	mov	B1_2,L1(2)

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2_1
;---------------------Round 1(12)-------------------
	add	A2,B2_1
	mov	S1(11),A1
	add	A1,S_not(12)

	add	B1_2,ecx
	mov	B2_2,L2(2)
	rol	A2,3

	rep rep rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	B1_1,L1(0)

	add	A1,B1_2
	mov	S2(11),A2
	mov	L1(2),B1_2

	add	B2_2,ecx
	rol	A1,3
	add	A2,S_not(12)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2_2
	;---
	mov	S1(12),A1
	add	A1,S_not(13)
	add	A2,B2_2

	rol	A2,3
	mov	B2_1,L2(0)
	add	B1_1,ecx

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(12),A2

	add	A1,B1_1
	add	A2,S_not(13)
	mov	B1_2,L1(1)

	rol	A1,3
	add	B2_1,ecx
	mov	L1(0),B1_1

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(0),B2_1
;---------------------Round 1(14)-------------------
	add	A2,B2_1
	mov	S1(13),A1
	add	A1,S_not(14)

	add	B1_2,ecx
	rol	A2,3
	mov	B2_2,L2(1)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	B1_1,L1(2)

	add	A1,B1_2
	mov	S2(13),A2
	add	A2,S_not(14)

	add	B2_2,ecx
	rol	A1,3
	mov	L1(1),B1_2

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(1),B2_2
	;---
	add	A2,B2_2
	mov	S1(14),A1
	add	A1,S_not(15)

	rep add	B1_1,ecx
	rol	A2,3
	mov	S2(14),A2

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	L1(2),B1_1

	add	A1,B1_1
	mov	B2_1,L2(2)
	mov	B1_2,L1(0)

	rol	A1,3
	add	A2,S_not(15)
	add	B2_1,ecx

	rep rol	B2_1,cl
	rep lea	ecx,[A1+B1_1]
	rep mov	L2(2),B2_1
;---------------------Round 1(16)-------------------
	add	A2,B2_1
	mov	S1(15),A1
	add	A1,S_not(16)

	rol	A2,3
	add	B1_2,ecx
	mov	B2_2,L2(0)

	mov	S2(15),A2
	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	L1(0),B1_2
	add	A2,S_not(16)

	mov	B1_1,L1(1)
	add	B2_2,ecx
	rol	A1,3

	rep rol	B2_2,cl
	rep lea	ecx,[A1+B1_2]
	rep mov	L2(0),B2_2
	;---
	add	A2,B2_2
	mov	S1(16),A1
	add	A1,S_not(17)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(1)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(16),A2

	add	A1,B1_1
	mov	B1_2,L1(2)
	mov	L1(1),B1_1

	rep add	B2_1,ecx
	rol	A1,3
	add	A2,S_not(17)

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2_1
;---------------------Round 1(18)-------------------
	add	A2,B2_1
	mov	S1(17),A1
	mov	B2_2,L2(2)

	add	B1_2,ecx
	rol	A2,3
	rol	B1_2,cl

	add	A1,S_not(18)
	lea	ecx,[A2+B2_1]
	mov	S2(17),A2

	add	A1,B1_2
	mov	B1_1,L1(0)
	add	A2,S_not(18)

	mov	L1(2),B1_2
	add	B2_2,ecx
	rol	A1,3

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2_2
	;---
	add	A2,B2_2
	mov	S1(18),A1
	add	A1,S_not(19)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(0)

	mov	S2(18),A2
	rol	B1_1,cl
	lea	ecx,[A2+B2_2]

	add	A2,S_not(19)
	add	A1,B1_1
	mov	B1_2,L1(1)

	mov	L1(0),B1_1
	add	B2_1,ecx
	rol	A1,3

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(0),B2_1
;---------------------Round 1(20)-------------------
	rep add	A2,B2_1
	mov	S1(19),A1
	add	A1,S_not(20)

	add	B1_2,ecx
	rol	A2,3
	mov	B2_2,L2(1)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(19),A2

	add	A1,B1_2
	mov	L1(1),B1_2
	add	A2,S_not(20)

	rol	A1,3
	add	B2_2,ecx
	mov	B1_1,L1(2)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(1),B2_2
	;---
	add	A2,B2_2
	mov	S1(20),A1
	add	B1_1,ecx

	add	A1,S_not(21)
	rol	A2,3
	mov	B2_1,L2(2)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(20),A2

	add	A1,B1_1
	mov	L1(2),B1_1
	mov	B1_2,L1(0)

	rol	A1,3
	add	B2_1,ecx
	add	A2,S_not(21)

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(2),B2_1
;---------------------Round 1(22)-------------------
	add	A2,B2_1
	mov	S1(21),A1
	mov	B2_2,L2(0)

	add	B1_2,ecx
	rol	A2,3
	rep rep add	A1,S_not(22)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(21),A2

	add	A1,B1_2
	mov	L1(0),B1_2
	mov	B1_1,L1(1)

	rol	A1,3
	add	B2_2,ecx
	add	A2,S_not(22)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(0),B2_2
	;---
	add	A2,B2_2
	mov	S1(22),A1
	add	A1,S_not(23)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(1)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(22),A2

	add	A1,B1_1
	mov	B1_2,L1(2)
	mov	L1(1),B1_1

	rol	A1,3
	add	B2_1,ecx
	rep rep add	A2,S_not(23)

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2_1
;---------------------Round 1(24)-------------------
	add	A2,B2_1
	mov	S1(23),A1
	add	A1,S_not(24)

	add	B1_2,ecx
	rol	A2,3
	mov	B2_2,L2(2)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(23),A2

	add	A1,B1_2
	rep rep mov	B1_1,L1(0)
	rep mov	L1(2),B1_2

	add	B2_2,ecx
	rol	A1,3
	add	A2,S_not(24)		

	rol	B2_2,cl				
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2_2
	;---
	add	A2,B2_2
	mov	S1(24),A1
	add	A1,S_not(25)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(0)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(24),A2

	add	A1,B1_1
	mov	L1(0),B1_1
	mov	B1_2,L1(1)

	rol	A1,3
	add	B2_1,ecx
	add	A2,S_not(25)

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(0),B2_1
;---------------------Round 1(Last)-------------------
	add	A2,B2_1
	mov	S1(25),A1
	add	A1,0xBF0A8B1D				;S(0)

	add	B1_2,ecx
	rol	A2,3					;S2(25)
	mov	B2_2,L2(1)

	rol	B1_2,cl 				;L1(1)
	mov	B1_1,L1(2)
	lea	ecx,[A2+B2_1]

	mov	S2(25),A2
	add	A1,B1_2
	add	A2,0xBF0A8B1D				;S(0)

	mov	L1(1),B1_2
	rol	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(1),B2_2
	;---
	add	A2,B2_2
	mov	S1(0),A1
	add	A1,[work_pre2_r1]			;S[1]

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(2)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	L1(2),B1_1

	add	A1,B1_1
	mov	S2(0),A2
	add	A2,[work_pre2_r1]

	rol	A1,3
	add	B2_1,ecx
	mov	B1_2,L1(0)

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(2),B2_1
;---------------------Round 2(2)-------------------
	add	A2,B2_1
	mov	S1(1),A1
	add	A1,[work_pre4_r1]			;S(2)

	rol	A2,3					;S2(25)
	add	B1_2,ecx
	mov	B2_2,L2(0)

	rep rol	B1_2,cl 				;L1(1)
	lea	ecx,[A2+B2_1]
	rep rep mov	L1(0),B1_2

	add	A1,B1_2
	mov	S2(1),A2
	mov	B1_1,L1(1)

	add	A2,[work_pre4_r1]			;S(2)
	rol	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	add	A2,B2_2
	;---
	mov	L2(0),B2_2
	mov	S1(2),A1
	add	A1,S1(3)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(1)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	L1(1),B1_1

	add	A1,B1_1
	mov	S2(2),A2
	mov	B1_2,L1(2)

	add	A2,S2(3)
	add	B2_1,ecx
	rol	A1,3

	rol	B2_1,cl
	mov	S1(3),A1
	lea	ecx,[A1+B1_1]
;---------------------Round 2(4)-------------------
	mov	L2(1),B2_1
	add	A2,B2_1
	rep add	A1,S1(4)

	add	B1_2,ecx
	rol	A2,3
	mov	B2_2,L2(2)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(3),A2

	add	A1,B1_2
	mov	B1_1,L1(0)
	add	A2,S2(4)

	rol	A1,3
	add	B2_2,ecx
	mov	L1(2),B1_2

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2_2
	;---
	add	A2,B2_2
	mov	S1(4),A1
	add	A1,S1(5)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(0)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(4),A2

	add	A1,B1_1
	add	A2,S2(5)
	mov	L1(0),B1_1

	rol	A1,3
	add	B2_1,ecx
	mov	B1_2,L1(1)

	rol	B2_1,cl
	mov	B2_2,L2(1)
	lea	ecx,[A1+B1_1]
;---------------------Round 2(6)-------------------
	mov	L2(0),B2_1
	add	A2,B2_1
	mov	S1(5),A1

	add	B1_2,ecx
	add	A1,S1(6)
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	L1(1),B1_2

	add	A1,B1_2
	mov	S2(5),A2
	mov	B1_1,L1(2)

	add	B2_2,ecx
	rol	A1,3
	add	A2,S2(6)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	rep mov	L2(1),B2_2
	;---
	add	A2,B2_2
	mov	S1(6),A1
	add	A1,S1(7)

	add	B1_1,ecx
	rol	A2,3
	mov	S2(6),A2

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	B2_1,L2(2)

	add	A1,B1_1
	add	A2,S2(7)
	mov	B1_2,L1(0)

	rol	A1,3
	mov	L1(2),B1_1
	add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(2),B2_1
;---------------------Round 2(8)-------------------
	add	A2,B2_1
	mov	S1(7),A1
	add	A1,S1(8)

	add	B1_2,ecx
	rol	A2,3
	rep mov	B2_2,L2(0)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(7),A2

	add	A1,B1_2
	add	A2,S2(8)
	mov	B1_1,L1(1)

	rol	A1,3
	add	B2_2,ecx
	rep rep mov	L1(0),B1_2

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(0),B2_2
	;---
	mov	S1(8),A1
	add	A1,S1(9)
	add	A2,B2_2

	add	B1_1,ecx
	mov	B2_1,L2(1)
	rep rol	A2,3

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(8),A2

	add	A1,B1_1
	add	A2,S2(9)
	mov	L1(1),B1_1

	rol	A1,3
	mov	B1_2,L1(2)
	rep add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	S1(9),A1
;---------------------Round 2(10)-------------------
	add	A2,B2_1
	add	A1,S1(10)
	mov	B2_2,L2(2)

	mov	L2(1),B2_1
	add	B1_2,ecx
	rep rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	L1(2),B1_2

	add	A1,B1_2
	mov	S2(9),A2
	add	A2,S2(10)

	rol	A1,3
	add	B2_2,ecx
	rep rep mov	B1_1,L1(0)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2_2
	;---
	add	A2,B2_2
	mov	S1(10),A1
	add	A1,S1(11)

	rol	A2,3
	add	B1_1,ecx
	rep mov	B2_1,L2(0)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	L1(0),B1_1

	add	A1,B1_1
	mov	S2(10),A2
	mov	B1_2,L1(1)

	add	A2,S2(11)
	rol	A1,3
	rep rep add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	S1(11),A1
;---------------------Round 2(12)-------------------
	add	A2,B2_1
	mov	B2_2,L2(1)
	mov	L2(0),B2_1

	add	A1,S1(12)
	add	B1_2,ecx
	rep rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(11),A2

	add	A1,B1_2
	mov	L1(1),B1_2
	mov	B1_1,L1(2)

	rol	A1,3
	add	B2_2,ecx
	add	A2,S2(12)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	B2_1,L2(2)
	;---
	add	A2,B2_2
	mov	S1(12),A1
	add	A1,S1(13)

	add	B1_1,ecx
	rol	A2,3
	mov	L2(1),B2_2

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(12),A2

	add	A1,B1_1
	mov	B1_2,L1(0)
	mov	L1(2),B1_1

	add	A2,S2(13)
	rol	A1,3
	rep add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(2),B2_1
;---------------------Round 2(14)-------------------
	add	A2,B2_1
	mov	B2_2,L2(0)
	mov	S1(13),A1

	add	B1_2,ecx
	rol	A2,3
	add	A1,S1(14)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(13),A2

	add	A1,B1_2
	add	A2,S2(14)
	mov	L1(0),B1_2

	rol	A1,3
	add	B2_2,ecx
	mov	B1_1,L1(1)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(0),B2_2
	;---
	mov	S1(14),A1
	add	A2,B2_2
	add	A1,S1(15)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(1)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(14),A2

	mov	L1(1),B1_1
	add	A1,B1_1
	add	A2,S2(15)

	rol	A1,3
	add	B2_1,ecx
	mov	B1_2,L1(2)

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2_1
;---------------------Round 2(16)-------------------
	mov	S1(15),A1
	add	A2,B2_1
	add	A1,S1(16)

	add	B1_2,ecx
	rol	A2,3
	mov	B2_2,L2(2)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	B1_1,L1(0)

	add	A1,B1_2
	mov	S2(15),A2
	mov	L1(2),B1_2

	rol	A1,3
	add	B2_2,ecx
	add	A2,S2(16)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2_2
	;---
	add	A2,B2_2
	mov	S1(16),A1
	add	A1,S1(17)

	add	B1_1,ecx
	rol	A2,3
	rep mov	B2_1,L2(0)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(16),A2

	add	A1,B1_1
	mov	L1(0),B1_1
	mov	B1_2,L1(1)

	add	A2,S2(17)
	rol	A1,3
	add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(0),B2_1
;---------------------Round 2(18)-------------------
	add	A2,B2_1
	mov	S1(17),A1
	add	A1,S1(18)

	mov	B2_2,L2(1)
	add	B1_2,ecx
	rep rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(17),A2

	add	A1,B1_2
	mov	L1(1),B1_2
	mov	B1_1,L1(2)

	rol	A1,3
	add	B2_2,ecx
	add	A2,S2(18)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	rep mov	L2(1),B2_2
	;---
	add	A2,B2_2
	mov	S1(18),A1
	add	A1,S1(19)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(2)

	rep rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	B1_2,L1(0)

	add	A1,B1_1
	mov	L1(2),B1_1
	mov	S2(18),A2

	add	A2,S2(19)
	add	B2_1,ecx
	rep rol	A1,3

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(2),B2_1
;---------------------Round 2(20)-------------------
	rep add	A2,B2_1
	mov	S1(19),A1
	add	A1,S1(20)

	mov	B2_2,L2(0)
	rep add	B1_2,ecx
	rep rep rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(19),A2

	add	A1,B1_2
	add	A2,S2(20)
	rep rep mov	B1_1,L1(1)

	add	B2_2,ecx
	rol	A1,3
	mov	L1(0),B1_2

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(0),B2_2
	;---
	add	A2,B2_2
	mov	S1(20),A1
	add	A1,S1(21)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(1)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(20),A2

	add	A1,B1_1
	mov	L1(1),B1_1
	add	A2,S2(21)

	rol	A1,3
	add	B2_1,ecx
	mov	B1_2,L1(2)

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2_1
;---------------------Round 2(22)-------------------
	add	A2,B2_1
	mov	S1(21),A1
	add	A1,S1(22)

	add	B1_2,ecx
	rol	A2,3
	mov	B2_2,L2(2)

	rep rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(21),A2

	rep add	A1,B1_2
	mov	B1_1,L1(0)
	add	A2,S2(22)

	mov	L1(2),B1_2
	rol	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	mov	L2(2),B2_2
	lea	ecx,[A1+B1_2]
	;---
	rep add	A2,B2_2
	mov	S1(22),A1
	add	A1,S1(23)

	add	B1_1,ecx
	rol	A2,3
	mov	B2_1,L2(0)

	rep rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(22),A2

	add	A1,B1_1
	add	A2,S2(23)
	mov	B1_2,L1(1)

	mov	L1(0),B1_1
	add	B2_1,ecx
	rol	A1,3

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	mov	L2(0),B2_1
;---------------------Round 2(24)-------------------
	add	A2,B2_1
	mov	S1(23),A1
	add	B1_2,ecx

	rol	A2,3
	add	A1,S1(24)
	mov	B2_2,L2(1)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
	mov	S2(23),A2

	add	A1,B1_2
	rep rep add	A2,S2(24)
	rep mov	L1(1),B1_2

	rol	A1,3
	add	B2_2,ecx
	mov	B1_1,L1(2)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(1),B2_2
	;---
	rep add	A2,B2_2
	mov	S1(24),A1
	add	A1,S1(25)

	add	B1_1,ecx
	rol	A2,3
	rep mov	B2_1,L2(2)

	rol	B1_1,cl
	lea	ecx,[A2+B2_2]
	mov	S2(24),A2

	add	A1,B1_1
	add	A2,S2(25)
				movd	eA2,[work_P_0]

	add	B2_1,ecx
	rol	A1,3
	mov	L1(2),B1_1

	mov	B1_2,L1(0)
	rol	B2_1,cl
	lea	ecx,[A1+B1_1]

	add	A2,B2_1 				;S2[24]+S2[25]+L2[2]
	mov	L2(2),B2_1
	mov	S1(25),A1
;---------------------Round 2-Last-------------
	add	B1_2,ecx				;L1[0]+S1[25]+L1[2]
	rol	A2,3					;S2[25]	
	add	A1,S1(0)				;S1[25]+S1[0]

	rol	B1_2,cl 				;L1[0]
	mov	ecx,B2_1
	mov	B2_2,L2(0)

	add	ecx,A2
	add	A1,B1_2 				;S1[25]+S1[0]+L1[0]
	add	B2_2,ecx				;L2[0]+S2[25]+L2[2]

	mov	S2(25),A2
	rol	A1,3					;S1[0]
	add	A2,S2(0)				;S2[25]+S2[0]

	rol	B2_2,cl					;L2[0]
	mov	L1(0),B1_2
	mov	B1_1,L1(1)

	lea	ecx,[A1+B1_2]				;S1[0]+L1[0]
				movd	L2_2,L2(2)
	add	A2,B2_2 				;S2[25]+L2[0]+S2[0]
	;---
	mov	L2(0),B2_2
	add	B1_1,ecx				;L1[1]+S1[0]+L1[0]
	rol	A2,3					;S2[0]
	
	mov	B2_1,L2(1)
	rol	B1_1,cl 				;L1[1]
	mov	ecx,B2_2

	add	ecx,A2
	mov	S2(0),A2
	mov	eA1,[work_P_0]

	add	B2_1,ecx				;L2[1]+S2[0]+L2[0]
	add	eA1,A1					;eA1
	add	A1,S1(1)				;S1[0]+S1[1]

	mov	B1_2,L1(2)
				movd	mm5,S2(1)	;start mmx round 3 - key 2
				movd	L2_0,L2(0)

	rep rol	B2_1,cl 				;L2[1]
	mov	ecx,B1_1
				movd 	A3,S2(0)

	mov	eB1,[work_P_1]
	add	A1,B1_1 				;S1[0]+S1[1]+L1[1]
;----
	mov	L2(1),B2_1				;key #2

	rol	A1,3					;S1(1)
				paddd	eA2,A3
				paddd	A3,mm5

	add	eB1,A1					;eB1
	add	ecx,A1
	add	A1,S1(2)

	xor	eA1,eB1
				movd	L2_1,L2(1)
	add	B1_2,ecx

	rol	B1_2,cl	
	;start ROUND3 mixed with encryption
	 mov	ecx,eB1
				paddd	A3,L2_1

	add	A1,B1_2
	rep rol	eA1,cl
	rep rep mov	B1_0,L1(0)

	mov	ecx,B1_2
	rol	A1,3		       
				punpckldq A3,A3

				punpckldq eB2,eB2
				movq	mm5,L2_1
	add	eA1,A1

				psllq	A3,3
	add	ecx,A1
	add	A1,S1(3)

				punpckldq L2_0,L2_0
	rep add	B1_0,ecx
	xor	eB1,eA1

	rol	B1_0,cl 	       
	mov	ecx,eA1
				punpckhdq A3,A3		

	add	A1,B1_0
	rol	eB1,cl
	mov	ecx,B1_0

	rol A1,3		       
				paddd	eB2,A3
				paddd	mm5,A3

	add	eB1,A1	
	add	ecx,A1
	add	A1,S1(4)
	
				paddd	A3,S2(2)
	xor	eA1,eB1
	rep add	B1_1,ecx

				paddd	L2_2,mm5
	rol	B1_1,cl
	mov	ecx,eB1

	add	A1,B1_1
	rol	eA1,cl
				pand	mm5,mm3

				punpckldq L2_2,L2_2
	rol	A1,3
				pxor	eA2,eB2


	add	eA1,A1
	lea	ecx,[A1+B1_1]
				psllq	L2_2,mm5

	rep rep xor	eB1,eA1
				movq	mm5,eB2
	add	A1,S1(5)

	add	B1_2,ecx
				punpckhdq L2_2,L2_2
				punpckldq A3,A3

	rol	B1_2,cl
	mov	ecx,eA1	
				pand	mm5,mm3

	rol	eB1,cl
	add	A1,B1_2
				punpckldq eA2,eA2

	rol	A1,3
	mov	ecx,B1_2
				paddd	A3,L2_2

	add	ecx,A1
	add	eB1,A1
				psllq	eA2,mm5

	add	A1,S1(6)
	add	B1_0,ecx
				psllq	A3,3


	xor	eA1,eB1
	rol	B1_0,cl
;------------Round 3(1)------------
	mov	ecx,eB1

	add	A1,B1_0
				punpckhdq A3,A3
	rol	eA1,cl

	rep rep rol	A1,3		       
				punpckhdq eA2,eA2
	rep rep mov	ecx,B1_0

	add	ecx,A1
	add	eA1,A1
	add	A1,S1(7)

	add	B1_1,ecx
	rep rep xor	eB1,eA1
				movq	mm5,L2_2

				paddd	eA2,A3
	rol	B1_1,cl 	       
	mov	ecx,eA1

	add	A1,B1_1
	rep rep rol	eB1,cl
				paddd	mm5,A3

	rol	A1,3		       
	mov	ecx,B1_1
				paddd	A3,S2(2)

	add	ecx,A1
	add	eB1,A1
	add	A1,S1(8)

	rep rep add	B1_2,ecx
	rep rep xor	eA1,eB1
				paddd	L2_0,mm5


	rol	B1_2,cl
;---
	mov	ecx,eB1
				pand	mm5,mm3

				pxor	eB2,eA2		
	add	A1,B1_2
	rep rep rol	eA1,cl

	mov	ecx,B1_2
	rol	A1,3
				psllq	L2_0,mm5

	add	ecx,A1
				movq	mm5,eA2
	add	eA1,A1

	add	A1,S1(9)
	add	B1_0,ecx
				punpckhdq L2_0,L2_0

	xor	eB1,eA1
	rep rol	B1_0,cl
				punpckhdq A3,A3
	
	mov	ecx,eA1
	add	A1,B1_0
				pand	mm5,mm3

	rol	A1,3
	rol	eB1,cl
	rep rep mov	ecx,B1_0

	add	ecx,A1
	add	eB1,A1
				paddd	A3,L2_0

	add	A1,S1(10)
				psllq	eB2,mm5
	add	B1_1,ecx

	xor	eA1,eB1
				movq	mm5,L2_0
				psllq	A3,3

	rol	B1_1,cl
;-------
	mov	ecx,eB1
				punpckhdq eB2,eB2

	add	A1,B1_1
	rol	eA1,cl
				punpckhdq A3,A3

	rol	A1,3
	rep mov	ecx,B1_1		
				paddd	mm5,A3

	add	ecx,A1
	add	eA1,A1
	add	A1,S1(11)

	add	B1_2,ecx
	rep xor	eB1,eA1
				paddd	eB2,A3

	rol	B1_2,cl
	mov	ecx,eA1
				paddd	A3,S2(4)

	add	A1,B1_2
	rol	eB1,cl
				paddd	L2_1,mm5


	rol	A1,3		       
	mov	ecx,B1_2
				pand	mm5,mm3

	add	ecx,A1
				punpckldq L2_1,L2_1
				punpckldq A3,A3

	add	eB1,A1
	add	A1,S1(12)
	add	B1_0,ecx

	xor	eA1,eB1
				pxor	eA2,eB2
	rol	B1_0,cl

;------------Round 3(4)------------
	mov	ecx,eB1
	add	A1,B1_0
				psllq	L2_1,mm5

	rol	A1,3		       
	rol	eA1,cl
	mov	ecx,B1_0

	add	ecx,A1
	add	eA1,A1
	add	A1,S1(13)

	add	B1_1,ecx
	xor	eB1,eA1
				movq	mm5,eB2

	rol	B1_1,cl 	       
	mov	ecx,eA1
				punpckhdq L2_1,L2_1

	add	A1,B1_1
	rol	eB1,cl
	mov	ecx,B1_1

	rep rol	A1,3		       
				pand	mm5,mm3
				paddd	A3,L2_1

	add	ecx,A1
	add	eB1,A1
				psllq	eA2,mm5

	add	A1,S1(14)
	add	B1_2,ecx
				psllq	A3,3

	rep rep		xor	eA1,eB1
	rol	B1_2,cl
;---
	rep mov	ecx,eB1

	add	A1,B1_2
	rol	eA1,cl
				punpckhdq eA2,eA2

	rol	A1,3
	rep mov	ecx,B1_2
				punpckhdq A3,A3

	add	ecx,A1
	add	eA1,A1
				movq	mm5,L2_1

	add	A1,S1(15)
	add	B1_0,ecx
	rep xor	eB1,eA1

	rol	B1_0,cl
				paddd	eA2,A3
				paddd	mm5,A3

	add	A1,B1_0
	mov	ecx,eA1
				paddd	A3,S2(4)

	rep rep rol	eB1,cl
	rep rep rol	A1,3
				paddd	L2_2,mm5

	mov	ecx,B1_0
	add	eB1,A1
				pand	mm5,mm3

	add	ecx,A1
	add	A1,S1(16)
				pxor	eB2,eA2

				psllq	L2_2,mm5
	add	B1_1,ecx
	xor	eA1,eB1

				movq	mm5,eA2
				punpckhdq A3,A3
	rol	B1_1,cl

;-------
	mov	ecx,eB1
				punpckhdq L2_2,L2_2
				pand	mm5,mm3

	add	A1,B1_1
	rep rol	eA1,cl
	rep rep mov	ecx,B1_1

	rol	A1,3
				paddd	A3,L2_2
				psllq	eB2,mm5

	add	ecx,A1
	add	eA1,A1
				punpckhdq eB2,eB2	

	add	A1,S1(17)
	add	B1_2,ecx
	xor	eB1,eA1

	rol	B1_2,cl
				psllq	A3,3
	mov	ecx,eA1

	add	A1,B1_2
	rol	eB1,cl
				movq	mm5,L2_2

	rol	A1,3		       
				punpckhdq A3,A3
	rep mov	ecx,B1_2

	add	eB1,A1
	add	ecx,A1
	add	A1,S1(18)

	add	B1_0,ecx
				paddd	eB2,A3
				paddd	mm5,A3

	xor	eA1,eB1
	rol	B1_0,cl
	mov	ecx,eB1

;------------Round 3(7)------------
	add	A1,B1_0
				paddd	L2_0,mm5
				pand	mm5,mm3

	rol	A1,3		       
	rol	eA1,cl
				paddd	A3,S2(6)

	mov	ecx,B1_0
				pxor	eA2,eB2
	add	eA1,A1

	add	ecx,A1
	add	A1,S1(19)
				psllq	L2_0,mm5

	xor	eB1,eA1
	add	B1_1,ecx
				movq	mm5,eB2

				punpckldq A3,A3
				punpckhdq L2_0,L2_0
	rol	B1_1,cl

	rep mov	ecx,eA1
				pand	mm5,mm3
				paddd	A3,L2_0

	add	A1,B1_1
				psllq	eA2,mm5
	rol	eB1,cl

	rol	A1,3		       
	mov	ecx,B1_1
				psllq	A3,3

	add	ecx,A1
				punpckhdq eA2,eA2
	add	eB1,A1

	add	A1,S1(20)
	add	B1_2,ecx
				punpckhdq A3,A3

	xor	eA1,eB1
	rol	B1_2,cl
				movq	mm5,L2_0
;---
	rep mov	ecx,eB1
				paddd	eA2,A3
	add	A1,B1_2

	rol	eA1,cl
	mov	ecx,B1_2
	rol	A1,3

	add	ecx,A1
	add	eA1,A1
				paddd	mm5,A3
	
	add	B1_0,ecx
	add	A1,S1(21)
				paddd	L2_1,mm5

	rol	B1_0,cl
				paddd	A3,S2(6)
				pand	mm5,mm3

	xor	eB1,eA1
	mov	ecx,eA1
	add	A1,B1_0

				pxor	eB2,eA2
	rol	eB1,cl
				psllq	L2_1,mm5

	rep rep rol	A1,3
				movq	mm5,eA2
	rep rep mov	ecx,B1_0

	add	ecx,A1
				punpckhdq A3,A3
				punpckhdq L2_1,L2_1

	add	eB1,A1
	add	B1_1,ecx
	add	A1,S1(22)

	xor	eA1,eB1
				pand	mm5,mm3
				paddd	A3,L2_1

	rol	B1_1,cl
;-------
	mov	ecx,eB1
				psllq	eB2,mm5

	add	A1,B1_1
	rol	eA1,cl
				psllq	A3,3

	rol	A1,3
				punpckhdq eB2,eB2
	rep mov	ecx,B1_1

	add	ecx,A1
	add	eA1,A1
				punpckhdq A3,A3

	add	A1,S1(23)
	add	B1_2,ecx        
				movq	mm5,L2_1
	
	xor	eB1,eA1
				paddd	eB2,A3
	rol	B1_2,cl

	mov	ecx,eA1
	rep add	A1,B1_2
				movd	L1(0),eB2

				paddd	mm5,A3
				paddd	A3,S2(8)
	rol	eB1,cl

	rol	A1,3		      
	mov	ecx,B1_2
				pxor	eA2,eB2

	add	ecx,A1
	rep rep add	eB1,A1
	rep add	A1,S1(24)

	add	B1_0,ecx
				paddd	L2_2,mm5
				pand	mm5,mm3

				psllq	L2_2,mm5
	xor	eA1,eB1
				movd	L1(2),A3	

;------------Round 3(10)------------
	rol	B1_0,cl
	mov	ecx,eB1
				movd	L1(1),eA2

	rep add	A1,B1_0
	rol	eA1,cl
				punpckhdq L2_2,L2_2

	rol	A1,3		       
				movd	L2(2),L2_2		
				movd	L2(0),L2_0

	add	eA1,A1
	cmp	eA1,[work_C_0]
				movd	L2(1),L2_1		

	je	near _checkKey1High_k7_mixed

;Finished Key #1, move data from mmx to integer pipe
;Load mmx unit with round 1 data for the next pair of keys
_Key2Round3_k7_mixed:
	mov	eB1,L1(0)
	mov	A1,L1(2)

	mov	ecx,eB1
	mov	eA1,L1(1)
				movd	L2_2,[RC5_72_L0hi]

	mov	B1_2,L2(2)
	rep rol	eA1,cl
				movd	mm5,[work_pre5_r1]

	mov	B1_0,L2(0)
				paddd	L2_2,[incr2]
				punpckldq mm5,mm5

	add	A1,B1_2
;---
	rep rep mov	ecx,B1_2
	mov	B1_1,L2(1)

	rol	A1,3
				movd	A3,[work_pre4_r1]
				punpckldq L2_2,L2_2

	add	ecx,A1
	add	eA1,A1
	add	A1,S2(9)

	add	B1_0,ecx
	xor	eB1,eA1
				movq	L2_1,L2_2

	rol	B1_0,cl
	mov	ecx,eA1
				paddd	A3,[S_not_3]

	rep add	A1,B1_0
	rol	eB1,cl
				paddd	L2_1,mm5

	rol	A1,3
	mov	ecx,B1_0
				paddd	L2_2,[incr]

	add	ecx,A1
	add	eB1,A1
	add	A1,S2(10)

	add	B1_1,ecx
	xor	eA1,eB1
	rol	B1_1,cl

				punpckldq A3,A3
;-------
	mov	ecx,eB1
				paddd	L2_2,mm5

	add	A1,B1_1
	rol	eA1,cl
				pand	mm5,mm3

	rol	A1,3
	mov	ecx,B1_1
				movq	mm7,A3

	add	ecx,A1
	rep rep add	eA1,A1
				psllq	L2_1,mm5

	add	A1,S2(11)
	rep rep add	B1_2,ecx
	rep rep xor	eB1,eA1

	rol	B1_2,cl
				psllq	L2_2,mm5
	mov	ecx,eA1

	rep rep add	A1,B1_2
	rep rol	eB1,cl
				punpckhdq L2_1,L2_1

	mov	ecx,B1_2
	rol	A1,3		       
				punpckhdq L2_2,L2_2

	add	ecx,A1
	add	eB1,A1
	add	A1,S2(12)

	add	B1_0,ecx
	xor	eA1,eB1
				paddd	A3,L2_1

	rol	B1_0,cl
;------------Round 3(4) - key 2------------
	mov	ecx,eB1
	add	A1,B1_0

				paddd	mm7,L2_2
	rol	eA1,cl
				movd	L1(2),L2_1

	rol	A1,3
				psllq	A3,3
	mov	ecx,B1_0

	add	ecx,A1
	add	eA1,A1
	add	A1,S2(13)

	add	B1_1,ecx
	xor	eB1,eA1
				psllq	mm7,3

	rol	B1_1,cl 	       
	mov	ecx,eA1
				punpckhdq A3,A3

	add	A1,B1_1
	rol	eB1,cl
				punpckhdq mm7,mm7

	rep rep mov	ecx,B1_1
	rep rep		rol	A1,3		       
				paddd	L2_1,A3

	rep add	ecx,A1
	add	eB1,A1
	add	A1,S2(14)

	add	B1_2,ecx
	xor	eA1,eB1
				movq	mm5,L2_1

	rol	B1_2,cl
	;---
	mov	ecx,eB1
				paddd	L2_1,[work_pre1_r1-4]

	add	A1,B1_2
				movd	L2(2),L2_2
				pand	mm5,mm3

	rol	eA1,cl
	mov	ecx,B1_2
				punpckhdq L2_1,L2_1

	rol	A1,3
				movd	S1(3),A3
				paddd	L2_2,mm7

	rep add	ecx,A1
	add	eA1,A1
	add	A1,S2(15)

	add	B1_0,ecx
	rep rep xor	eB1,eA1
				psllq	L2_1,mm5

	rol	B1_0,cl
				movq	mm5,L2_2
				paddd	A3,[S_not_4]

	mov	ecx,eA1
	add	A1,B1_0
				punpckhdq L2_1,L2_1

	rol	eB1,cl
	rol	A1,3
	mov	ecx,B1_0

	add	ecx,A1
	rep rep add	eB1,A1
	rep rep add	A1,S2(16)

				paddd	L2_2,[work_pre1_r1-4]
	add	B1_1,ecx
	xor	eA1,eB1

	rep rep rol	B1_1,cl
	;-------
	rep rep mov	ecx,eB1
				paddd	A3,L2_1

	add	A1,B1_1
	rol	eA1,cl
				movd	S2(3),mm7		

	mov	ecx,B1_1
	rol	A1,3
				punpckhdq L2_2,L2_2

	add	ecx,A1
	add	eA1,A1
	add	A1,S2(17)
			
	add	B1_2,ecx
	xor	eB1,eA1
				paddd	mm7,[S_not_4]

	rol	B1_2,cl
	mov	ecx,eA1
	add	A1,B1_2

				psllq	A3,3
	rol	eB1,cl
	mov	ecx,B1_2

	rol	A1,3		       
				pand	mm5,mm3
				punpckhdq A3,A3

	add	ecx,A1
	add	eB1,A1
	add	A1,S2(18)

	add	B1_0,ecx
	xor	eA1,eB1
				movd	L1(0),L2_1

	;------------Round 3(7) - key 2------------
	rol	B1_0,cl
	mov	ecx,eB1
				psllq	L2_2,mm5

	add	A1,B1_0
	rol	eA1,cl
				movd	S1(4),A3

	rep rep mov	ecx,B1_0
	rep rep rol	A1,3		       
				paddd	L2_1,A3

	add	ecx,A1
	add	eA1,A1
	add	A1,S2(19)

	add	B1_1,ecx
				movq	mm5,L2_1
				paddd	L2_1,[work_pre3_r1-4]

	xor	eB1,eA1
	rol	B1_1,cl 	       
	mov	ecx,eA1

	add	A1,B1_1
	rol	eB1,cl
				punpckhdq L2_2,L2_2

	mov	ecx,B1_1
	rol	A1,3		       
				movd	L2(0),L2_2

	add	ecx,A1
	add	eB1,A1
	add	A1,S2(20)

	add	B1_2,ecx
	xor	eA1,eB1
	rol	B1_2,cl

				punpckhdq L2_1,L2_1		
	;---
	mov	ecx,eB1
				paddd	mm7,L2_2

	add	A1,B1_2
	rep rol	eA1,cl
				pand	mm5,mm3
		
	mov ecx,B1_2
	rol	A1,3
				paddd	A3,[S_not_5]

	add	ecx,A1
	add	eA1,A1
	add	A1,S2(21)

	add	B1_0,ecx
	xor	eB1,eA1
				psllq	mm7,3

	rol	B1_0,cl
	mov	ecx,eA1
				psllq	L2_1,mm5

	add	A1,B1_0
	rol	eB1,cl
				punpckhdq mm7,mm7

	rol	A1,3
	mov	ecx,B1_0
				punpckhdq L2_1,L2_1

	add	ecx,A1
	add	eB1,A1
	add	A1,S2(22)

	add	B1_1,ecx
	xor	eA1,eB1
				movd	mm5,L2(0)

	rol	B1_1,cl
	;-------
	mov	ecx,eB1
				paddd	A3,L2_1

	add	A1,B1_1
	rep rol eA1,cl
				paddd	mm5,mm7

				movd	S2(4),mm7
	mov	ecx,B1_1
	rol	A1,3

	add	ecx,A1
	add	eA1,A1
	add	A1,S2(23)

	add	B1_2,ecx
	xor	eB1,eA1
				psllq	A3,3

	rol	B1_2,cl
	mov	ecx,eA1
				movd	S1(0),mm5

	add	A1,B1_2
	rol	eB1,cl
				punpckhdq A3,A3

	mov	ecx,B1_2
	rol	A1,3		  
				paddd	mm7,[S_not_5]

	add	ecx,A1
	add	eB1,A1
	add	A1,S2(24)

	add	B1_0,ecx
	rep xor	eA1,eB1
				paddd	mm5,[work_pre3_r1]

	rol	B1_0,cl
;------------Round 3(10) - key 2------------
	mov	ecx,eB1
				movd	L1(1),L2_1

	add	A1,B1_0
				movd	S1(5),A3
				paddd	A3,L2_1
	rol	A1,3		       
	rol	eA1,cl
				movd	S2(0),mm7

	add	eA1,A1
				movd	S1(1),mm5
				movd	eB2,[work_P_1]
				movd	S2(1),A3


	cmp	eA1,[work_C_0]
	je	short _checkKey2High_k7_mixed
_NextKey:
	dec	dword [work_iterations]
	mov	ecx,S1(0)
	jz	near finished_Found_nothing

	add	byte [RC5_72_L0hi],2
	mov	B1_1,S2(1)		
	mov	B2_1,S1(1)

	mov	A2,S2(0)
	mov	A1,S1(5)
	jnc	key_setup_1_inner_loop
	
	mov	eax, [RC5_72UnitWork]
	mov	edx, [RC5_72UnitWork_L0mid]
	mov	esi, [RC5_72UnitWork_L0lo]
	mov	ebx, [RC5_72_L0hi]
	bswap	edx
	bswap	esi
	adc	edx,0
	adc	esi,0
	bswap	edx
	bswap	esi
	mov	[RC5_72UnitWork_L0mid],edx
	mov	[RC5_72UnitWork_L0lo],esi
	jmp	key_setup_1_bigger_loop

k7align 16
_checkKey2High_k7_mixed:
	lea	ecx,[A1+B1_0]

	add	A1,S2(25)

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
