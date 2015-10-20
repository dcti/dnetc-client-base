; RC5-72 integer/mmx+ mixed version - 2 pipe b
; Vyacheslav Chupyatov - goteam@mail.ru - 2003-2009
; For use by distributed.net. 

%define P	  0xB7E15163
%define Q	  0x9E3779B9
%define S_not(N)  ((P+Q*(N)) & 0xFFFFFFFF)

%ifdef __OMF__ 		;  Borland and Watcom compilers/linkers
[SECTION _DATA FLAT USE32 align=16 CLASS=DATA]
%else
[SECTION .data align=16]
%endif


incr1		dd	2,3
incr2		dd	4,5
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
S_not_17	dd	S_not(17),S_not(17)
S_not_18	dd	S_not(18),S_not(18)
S_not_19	dd	S_not(19),S_not(19)


%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_72_unit_func_go_2b]
[GLOBAL rc5_72_unit_func_go_2b]


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
defwork work_iterations
defwork work_S,52
defwork	S3,2
defwork RC5_72_L0hi
defwork work_C_1
defwork work_pre1_r1
defwork	iterations_addr
defwork work_pre2_r1
defwork work_P_0
defwork work_pre3_r1
defwork work_P_1
defwork work_pre4_r1
defwork work_C_0
defwork work_pre5_r1
defwork save_ebx
defwork save_esi
defwork save_edi
defwork save_ebp
defwork	S_ext,52
defwork	unitwork_addr
defwork save_esp


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

; NASM needs help to generate optimized nop padding
%ifdef __NASM_VER__
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
%else
    %define k7align align
%endif

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
_rc5_72_unit_func_go_2b:
rc5_72_unit_func_go_2b:
;------ integer stream -------- mmx helper ----------
				mov	eax,0x1f
	mov	ecx, esp
	sub	esp, work_size

				movd	mm3,eax
				movd	xmm3,eax

	lea	eax, [RC5_72UnitWork]
	and	esp,-64
	mov	[save_esp], ecx

	mov	[save_ebp], ebp
	lea	ecx,[eax+4]
	mov	eax, [eax]

	mov	[unitwork_addr],eax
	mov	[save_ebx], ebx
	mov	[save_esi], esi

	mov	[save_edi], edi
	mov	ebx, [RC5_72UnitWork_plainlo]
	mov	edi, [RC5_72UnitWork_plainhi]

	mov	esi, [RC5_72UnitWork_cipherlo]
	mov	[iterations_addr],ecx
	mov	edx, [ecx]
	mov	ecx, [RC5_72UnitWork_cipherhi]

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
					pshufw	B3_2,[RC5_72_L0hi],0x44 

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
					paddb	B3_2,[incr1]
 
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
	add	A2,B2

	rol	A2,3
	mov	S2(16),A2
	;---
	add	B1_1,ecx
	rol	B1_1,cl

	lea	ecx,[A2+B2]
	mov	B2,L2(1)
	add	B2,ecx
	rol	B2,cl
	mov	L2(1),B2
	lea	A1,[A1+B1_1+S_not(17)]
	lea	A2,[A2+B2+S_not(17)]
	rol	A1,3
	rol	A2,3
	mov	S1(17),A1
	mov	S2(17),A2

	lea	ecx,[A1+B1_1]
	add	A1,S_not(18)
;---------------------Round 1(18)-------------------
	add	B1_2,ecx

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	B2,ecx

	rol	B2,cl

	add	A2,S_not(18)
	mov	L2(2),B2
	add	A1,B1_2
	add	A2,B2
	rol	A1,3
	rol	A2,3
	mov	S1(18),A1
	mov	S2(18),A2

	lea	ecx,[A1+B1_2]
	;---                            
	add	B1_0,ecx

	rol	B1_0,cl

	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	B2,ecx
	rol	B2,cl
	mov	L2(0),B2

 	lea	A1,[A1+B1_0+S_not(19)]
 	lea	A2,[A2+B2+S_not(19)]
k7align 64
key_setup_1_inner_loop:
 	rep 	rol	A1,3
 	rol	A2,3
					pshufw	mm5,[work_pre5_r1],0x44 

 	rep	lea	ecx,[A1+B1_0]
					paddd	B3_2,mm5
					pshufw	A3,[work_pre4_r1],0x44 
;---------------------Round 1(20)-------------------
 	add	B1_1,ecx
	mov	S1(19),A1
					pand	mm5,mm3

	rol	B1_1,cl
 	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	lea	A1,[A1+B1_1+S_not(20)]
					paddd	A3,[S_not_3]
					pshufw	A3_H,B3_2,0xee

	mov	S2(19),A2
	add	B2,ecx
					punpckldq	B3_2,B3_2

	rol	A1,3
	rol	B2,cl
					psllq	A3_H,mm5

	lea	ecx,[A1+B1_1]
	lea	A2,[A2+B2+S_not(20)]
	mov	L2(1),B2
	;---
	add	B1_2,ecx
	mov	S1(20),A1
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	lea	A1,[A1+B1_2+S_not(21)]
					psllq	B3_2,mm5
					punpckhdq	B3_2,A3_H

	mov	S2(20),A2
	add	B2,ecx
					paddd	A3,B3_2

	rol	A1,3
	rol	B2,cl
					movq	mm5,B3_2

	lea	ecx,[A1+B1_2]
	lea	A2,[A2+B2+S_not(21)]
	mov	L2(2),B2
;---------------------Round 1(22)-------------------
	add	B1_0,ecx
	mov	S1(21),A1
	rol	A2,3

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	lea	A1,[A1+B1_0+S_not(22)]
					movq	A3_H,A3
					psrld	A3,29

	mov	S2(21),A2
	add	B2,ecx
                                        pslld	A3_H,3

	rol	A1,3
	rol	B2,cl
					por	A3,A3_H

	lea	ecx,[A1+B1_0]
	lea	A2,[A2+B2+S_not(22)]
	mov	L2(0),B2
	;---
	add	B1_1,ecx
	rol	A2,3
	mov	S1(22),A1

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	lea	A1,[A1+B1_1+S_not(23)]
					pshufw	B3_0,[work_pre1_r1],0x44
					paddd	mm5,A3

	mov	S2(22),A2
	add	B2,ecx
					paddd	B3_0,mm5

	rol	A1,3
	rol	B2,cl
					pshufw	mm7,mm5,0xee

	lea	ecx,[A1+B1_1]
	lea	A2,[A2+B2+S_not(23)]
	mov	L2(1),B2
;---------------------Round 1(24)-------------------
	add	B1_2,ecx
	rol	A2,3
	mov	S1(23),A1

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	lea	A1,[A1+B1_2+S_not(24)]
					pand	mm5,mm3
                                        movq	[S3],A3

	add	B2,ecx
	mov	S2(23),A2
					pshufw	A3_H,B3_0,0xee

	rol	A1,3
	rol	B2,cl				
					paddd	A3,[S_not_4]

	lea	ecx,[A1+B1_2]
	lea	A2,[A2+B2+S_not(24)]
	mov	L2(2),B2
	;---
	add	B1_0,ecx
	rol	A2,3
	mov	S1(24),A1

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	lea	A1,[A1+B1_0+S_not(25)]
					pand	mm7,mm3
					punpckldq	B3_0,B3_0

	mov	S2(24),A2
	add	B2,ecx
					psllq	A3_H,mm7

	rol	A1,3
	rol	B2,cl
					psllq	B3_0,mm5

	lea	ecx,[A1+B1_0]
	lea	A2,[A2+B2+S_not(25)]
	mov	L2(0),B2
;---------------------Round 1(Last)-------------------
	add	B1_1,ecx
	mov	S1(25),A1
	rol	A2,3					;S2(25)

	rol	B1_1,cl 				;L1(1)
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	lea	A1,[A1+B1_1+0xBF0A8B1D]
					punpckhdq	B3_0,A3_H
				        paddd	A3,B3_0

	mov	S2(25),A2
	add	B2,ecx
					movq	A3_H,A3

	rol	A1,3
	rol	B2,cl
					psrld	A3,29

	lea	ecx,[A1+B1_1]
	lea	A2,[A2+B2+0xBF0A8B1D]
	mov	L2(1),B2
	;---
	mov	S1(0),A1
	add	A1,[work_pre2_r1]			;S[1]
					pslld	A3_H,3

	add	B1_2,ecx
	rol	A2,3
					movq	mm5,B3_0

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(0),A2
	add	A2,[work_pre2_r1]

	rol	A1,3
	add	B2,ecx
					por	A3,A3_H

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 2(2)-------------------
	add	A2,B2
	mov	S1(1),A1
	add	A1,[work_pre4_r1]			;S(2)

	rol	A2,3					;S2(25)
	add	B1_0,ecx
                                        pshufw	B3_1,[work_pre3_r1],0x44

	rol	B1_0,cl 				;L1(1)
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(1),A2
	add	A2,[work_pre4_r1]			;S(2)

	rol	A1,3
	add	B2,ecx
					paddd	mm5,A3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
	;---
	add	A2,B2
	mov	S1(2),A1
	add	A1,S1(3)

	add	B1_1,ecx
	rol	A2,3
					paddd	B3_1,mm5

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(2),A2
	add	A2,S2(3)

	add	B2,ecx
	rol	A1,3
					pshufw	mm7,mm5,0xee

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 2(4)-------------------
	add	A2,B2
	mov	S1_alt(3),A1
	add	A1,S1(4)

	add	B1_2,ecx
	rol	A2,3
					pand	mm5,mm3

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(3),A2
	add	A2,S2(4)

	rol	A1,3
	add	B2,ecx
					pand	mm7,mm3

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
	;---
	add 	A2,B2
	mov	S1_alt(4),A1
	add	A1,S1(5)

	add	B1_0,ecx
	rol	A2,3
					pshufw	A3_H,B3_1,0xee

	rol 	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add 	A1,B1_0
	mov	S2_alt(4),A2
	add	A2,S2(5)

	rol	A1,3
	add	B2,ecx
					punpckldq	B3_1,B3_1

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
					psllq	B3_1,mm5

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add 	A2,B2
	mov	S1_alt(6),A1
	add	A1,S1(7)

	add	B1_2,ecx
	rol	A2,3
					punpckhdq	B3_1,A3_H

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(6),A2
	add	A2,S2(7)

	rol	A1,3
	add	B2,ecx
					movq	A3_H,A3

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 2(8)-------------------
	add 	A2,B2
	mov	S1_alt(7),A1
	add	A1,S1(8)

	add	B1_0,ecx
	rol	A2,3
					paddd	A3,[S_not_5]

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(7),A2
					movq	S(4),A3_H

	add 	B2,ecx
	rol	A1,3
	add	A2,S2(8)

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
	;---
	add	A2,B2
	mov	S1_alt(8),A1
	add	A1,S1(9)

	add	B1_1,ecx
	rol	A2,3
					paddd	A3,B3_1

	rol 	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(8),A2
	add	A2,S2(9)

	rol	A1,3
	add	B2,ecx
					movq	A3_H,A3

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 2(10)-------------------
	add	A2,B2
	mov	S1(9),A1
	add	A1,S1(10)

	add	B1_2,ecx
	rol	A2,3
					pslld	A3_H,3

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(9),A2
	add	A2,S2(10)

	rol	A1,3
	add	B2,ecx
					psrld	A3,29

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
	;---
	add	A2,B2
	mov	S1(10),A1
	add	A1,S1(11)

	rol	A2,3
	add	B1_0,ecx
					por	A3,A3_H

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(10),A2
	add	A2,S2(11)

	rol	A1,3
	add	B2,ecx
					movq	S(5),A3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
;---------------------Round 2(12)-------------------
	add	A2,B2
	mov	S1(11),A1
	add	A1,S1(12)

	add	B1_1,ecx
	rol	A2,3
					movq	mm5,B3_1

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(11),A2
	add	A2,S2(12)

	rol	A1,3
	add	B2,ecx
					paddd	mm5,A3	

	rol 	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add	A2,B2
	mov	S1(12),A1
	add	A1,S1(13)

	add	B1_2,ecx
	rol	A2,3
					paddd	A3,[S_not_6]

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(12),A2
	add	A2,S2(13)

	rol	A1,3
	add	B2,ecx
					paddd	B3_2,mm5

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 2(14)-------------------
	add	A2,B2
	mov	S1(13),A1
	add	A1,S1(14)

	add	B1_0,ecx
	rol	A2,3
					pshufw	mm7,mm5,0xee

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(13),A2
	add	A2,S2(14)

	rol	A1,3
	add	B2,ecx
					pand	mm5,mm3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
	;---
	add	A2,B2
	mov	S1(14),A1
	add	A1,S1(15)

	add	B1_1,ecx
	rol	A2,3
					pshufw	A3_H,B3_2,0xee

	rol 	B1_1,cl
	lea	ecx,[A2+B2]
	mov	S2_alt(14),A2

	add	A1,B1_1
	mov	B2,L2(1)
					pand	mm7,mm3

	rol	A1,3
	add	B2,ecx
	add	A2,S2(15)

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 2(16)-------------------
	add	A2,B2
	mov	S1(15),A1
	add	A1,S1(16)

	add	B1_2,ecx
	rol	A2,3
					punpckldq	B3_2,B3_2

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(15),A2
	add	A2,S2(16)

	rol	A1,3
	add	B2,ecx
					psllq	A3_H,mm7

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
	;---
	add	A2,B2
	mov	S1(16),A1
	add	A1,S1(17)

	add	B1_0,ecx
	rol	A2,3
					psllq	B3_2,mm5

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(16),A2
	add	A2,S2(17)

	rol	A1,3
	add	B2,ecx
					punpckhdq	B3_2,A3_H

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
;---------------------Round 2(18)-------------------
	add	A2,B2
	mov	S1(17),A1
	add	A1,S1(18)

	add	B1_1,ecx
	rol	A2,3
					paddd	A3,B3_2

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(17),A2
	add	A2,S2(18)

	rol	A1,3
	add	B2,ecx
					movq	A3_H,A3

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add	A2,B2
	mov	S1(18),A1
	add	A1,S1(19)

	add	B1_2,ecx
	rol	A2,3
					psrld	A3,29

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(18),A2
	add	A2,S2(19)

	add	B2,ecx
	rol	A1,3
					pslld	A3_H,3

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
;---------------------Round 2(20)-------------------
	add	A2,B2
	mov	S1(19),A1
	add	A1,S1(20)

	add	B1_0,ecx
	rol	A2,3
					por	A3,A3_H

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(19),A2
	add	A2,S2(20)

	add	B2,ecx
	rol	A1,3
					movq	mm5,B3_2

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
	;---
	add	A2,B2
	mov	S1(20),A1
	add	A1,S1(21)

	add	B1_1,ecx
	rol	A2,3
					movq	S(6),A3

	rol	B1_1,cl
	lea	ecx,[A2+B2]
	mov	S2_alt(20),A2

	add	A1,B1_1
	mov	B2,L2(1)
	add	A2,S2(21)

	rol	A1,3
	add	B2,ecx
					paddd	mm5,A3

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
;---------------------Round 2(22)-------------------
	add	A2,B2
	mov	S1(21),A1
	add	A1,S1(22)

	add	B1_2,ecx
	rol	A2,3
					paddd	B3_0,mm5

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(21),A2
	add	A2,S2(22)

	rol 	A1,3
	add	B2,ecx
					pshufw	mm7,mm5,0xee

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2
	;---
	add	A2,B2
	mov	S1(22),A1
	add	A1,S1(23)

	rol	A2,3
	add	B1_0,ecx
					paddd	A3,[S_not_7]

	rol	B1_0,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(0)

	add	A1,B1_0
	mov	S2_alt(22),A2
	add	A2,S2(23)

	add	B2,ecx
	rol	A1,3
					pand	mm5,mm3

	rol	B2,cl
	lea	ecx,[A1+B1_0]
	mov	L2(0),B2
;---------------------Round 2(24)-------------------
	add	A2,B2
	mov	S1(23),A1
	add	A1,S1(24)

	rol	A2,3
	add	B1_1,ecx
					pshufw	A3_H,B3_0,0xee

	rol 	B1_1,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2_alt(23),A2
	add	A2,S2(24)

	rol	A1,3
	add	B2,ecx
					pand	mm7,mm3

	rol	B2,cl
	lea	ecx,[A1+B1_1]
	mov	L2(1),B2
	;---
	add	A2,B2
	mov	S1(24),A1
	add	A1,S1(25)

	add	B1_2,ecx
	rol	A2,3
					punpckldq	B3_0,B3_0

	rol	B1_2,cl
	lea	ecx,[A2+B2]
	mov	B2,L2(2)

	add	A1,B1_2
	mov	S2_alt(24),A2
					psllq	A3_H,mm7

	add	B2,ecx
	rol	A1,3
	add	A2,S2(25)

	rol	B2,cl
	lea	ecx,[A1+B1_2]
	mov	L2(2),B2

	add	A2,B2 					;S2[24]+S2[25]+L2[2]
	mov	S1(25),A1
	add	B1_0,ecx				;L1[0]+S1[25]+L1[2]
;---------------------Round 2-Last-------------
	rol	A2,3					;S2[25]	
	add	A1,S1(0)				;S1[25]+S1[0]
	rol	B1_0,cl 				;L1[0]

	lea	ecx,[A2+B2]
	mov	B2,L2(0)
	add	A1,B1_0 				;S1[25]+S1[0]+L1[0]

	rol	A1,3					;S1[0]
	add	B2,ecx					;L2[0]+S2[25]+L2[2]
	mov	S2_alt(25),A2

	add	A2,S2_alt(0)				;S2[25]+S2[0]
	rol	B2,cl					;L2[0]
	lea	ecx,[A1+B1_0]				;S1[0]+L1[0]

	add	A2,B2 					;S2[25]+L2[0]+S2[0]
	mov	L2(0),B2
					psllq	B3_0,mm5
	;---
	add 	B1_1,ecx				;L1[1]+S1[0]+L1[0]
	rol	A2,3					;S2[0]
	mov	S2_alt(0),A2
	
	rol	B1_1,cl 				;L1[1]
	lea	ecx,[B2+A2]
	mov	eA1,[work_P_0]

	mov	B2,L2(1)
	add	eA1,A1					;eA1
	add	A1,S1(1)				;S1[0]+S1[1]

	add 	B2,ecx					;L2[1]+S2[0]+L2[0]
	add 	A1,B1_1 				;S1[0]+S1[1]+L1[1]
					punpckhdq	B3_0,A3_H

	rol	B2,cl 					;L2[1]
	rol	A1,3					;S1(1)
					paddd	A3,B3_0

	mov	L2(1),B2				;key #2
	mov	eB1,[work_P_1]
;----
	lea	ecx,[A1+B1_1]

	add	eB1,A1					;eB1
	add	A1,S1(2)
					movq	mm5,B3_0

	xor	eA1,eB1
	add	B1_2,ecx
					movq	A3_H,A3

	rol	B1_2,cl	
	;start ROUND3 mixed with encryption
	mov	ecx,eB1
					psrld	A3,29

	add	A1,B1_2
	rol	eA1,cl
					pslld	A3_H,3

	rol	A1,3		       
					por	A3,A3_H
					paddd	mm5,A3

	add	eA1,A1
	lea	ecx,[B1_2+A1]
	add	A1,S1_alt(3)

	add	B1_0,ecx
	xor	eB1,eA1
					paddd	B3_1,mm5

	rol	B1_0,cl 	       
	mov	ecx,eA1
					pshufw	mm7,mm5,0xee

	add	A1,B1_0
	rol	eB1,cl
					pand	mm5,mm3

	rol 	A1,3		       
					pshufw	A3_H,B3_1,0xee
					pand	mm7,mm3

	add	eB1,A1	
	lea	ecx,[B1_0+A1]
	add	A1,S1_alt(4)
	
	xor	eA1,eB1
	add	B1_1,ecx
					movq	S(7),A3

	rol	B1_1,cl
	mov	ecx,eB1
                                        paddd	A3,[S_not_8]

	add	A1,B1_1
	rol	eA1,cl
					punpckldq	B3_1,B3_1

	rol	A1,3
					psllq	A3_H,mm7
					psllq	B3_1,mm5

	add	eA1,A1
	lea	ecx,[A1+B1_1]
	add	A1,S1_alt(5)

	xor	eB1,eA1
	add	B1_2,ecx
					punpckhdq	B3_1,A3_H

	rol	B1_2,cl
	mov	ecx,eA1	
					paddd	A3,B3_1

	add	A1,B1_2
	rol	eB1,cl
					movq	A3_H,A3

	rol	A1,3
					psrld	A3,29
					pslld	A3_H,3

	lea	ecx,[A1+B1_2]
	add	eB1,A1
					movq	mm5,B3_1

	add	B1_0,ecx
	xor	eA1,eB1
	add	A1,S1_alt(6)

	rol	B1_0,cl
;------------Round 3(1)------------
	mov	ecx,eB1
					por	A3,A3_H

	add	A1,B1_0
	rol	eA1,cl
					paddd	mm5,A3

	rol	A1,3		       
					paddd	B3_2,mm5
					movq	S(8),A3

	lea	ecx,[A1+B1_0]
	add	eA1,A1
	add	A1,S1_alt(7)

	add 	B1_1,ecx
	xor	eB1,eA1
					pshufw	mm7,mm5,0xee

	rol	B1_1,cl 	       
	mov	ecx,eA1
					pshufw	A3_H,B3_2,0xee

	rol	eB1,cl
	add	A1,B1_1
					pand	mm5,mm3

	rol	A1,3		       
					pand	mm7,mm3
					paddd	A3,[S_not_9]

	lea	ecx,[A1+B1_1]
	add	eB1,A1
	add	A1,S1_alt(8)

	add	B1_2,ecx
	xor	eA1,eB1
					punpckldq	B3_2,B3_2

	rol	B1_2,cl
;---
	mov	ecx,eB1
					psllq	A3_H,mm7

	add	A1,B1_2
	rol	eA1,cl
					psllq	B3_2,mm5

	rol	A1,3
					punpckhdq	B3_2,A3_H
					paddd	A3,B3_2

	lea	ecx,[A1+B1_2]
	add	eA1,A1
	add	A1,S1(9)

	add	B1_0,ecx
	xor	eB1,eA1
					movq	A3_H,A3

	rol	B1_0,cl
	mov	ecx,eA1
					psrld	A3,29

	add	A1,B1_0
	rol	eB1,cl
					pslld	A3_H,3

	rol	A1,3
					movq	mm5,B3_2
					por	A3,A3_H

	lea	ecx,[A1+B1_0]
	add	eB1,A1
	add	A1,S1(10)

	add	B1_1,ecx
	xor	eA1,eB1
					paddd	mm5,A3

	rol	B1_1,cl
;-------
	mov	ecx,eB1
					movq	S(9),A3

	add	A1,B1_1
	rol	eA1,cl
					paddd	B3_0,mm5

	rol	A1,3
					paddd	A3,[S_not_10]
					pshufw	mm7,mm5,0xee

	lea	ecx,[A1+B1_1]
	add	eA1,A1
	add	A1,S1(11)

	add	B1_2,ecx
	xor	eB1,eA1
					pand	mm5,mm3

	rol	B1_2,cl
	mov	ecx,eA1
					pshufw	A3_H,B3_0,0xee

	add	A1,B1_2
	rol	eB1,cl
					pand	mm7,mm3

	rol	A1,3		       
					punpckldq	B3_0,B3_0
					psllq	A3_H,mm7

	lea	ecx,[A1+B1_2]
	add	eB1,A1
	add	A1,S1(12)

	add	B1_0,ecx
	xor	eA1,eB1
					psllq	B3_0,mm5

	rol	B1_0,cl
;------------Round 3(4)------------
	mov	ecx,eB1
					punpckhdq	B3_0,A3_H

	add	A1,B1_0
	rol	eA1,cl
					paddd	A3,B3_0

	rol	A1,3		       
					movq	A3_H,A3
					psrld	A3,29

	lea	ecx,[A1+B1_0]
	add	eA1,A1
	add	A1,S1(13)

	add	B1_1,ecx
	xor	eB1,eA1
					pslld	A3_H,3

	rol	B1_1,cl 	       
	mov	ecx,eA1
					movq	mm5,B3_0

	add	A1,B1_1
	rol	eB1,cl
					por	A3,A3_H

	rol	A1,3		       
					paddd	mm5,A3
					movq	S(10),A3

	lea	ecx,[A1+B1_1]
	add	eB1,A1
	add	A1,S1(14)

	add	B1_2,ecx
	xor	eA1,eB1
					paddd	A3,[S_not_11]

	rol	B1_2,cl
;---
	mov	ecx,eB1
					paddd	B3_1,mm5

	add	A1,B1_2
	rol	eA1,cl
					pshufw	mm7,mm5,0xee

	rol	A1,3
					pand	mm5,mm3
					pshufw	A3_H,B3_1,0xee

	lea	ecx,[A1+B1_2]
	add	eA1,A1
	add	A1,S1(15)

	add	B1_0,ecx
	xor	eB1,eA1
                                        pand	mm7,mm3

	rol	B1_0,cl
	mov	ecx,eA1
					punpckldq	B3_1,B3_1

	add	A1,B1_0
	rol	eB1,cl
					psllq	A3_H,mm7

	rol	A1,3
					psllq	B3_1,mm5
					punpckhdq	B3_1,A3_H

	add	eB1,A1
	lea	ecx,[A1+B1_0]
	add	A1,S1(16)

	add	B1_1,ecx
	xor	eA1,eB1
                                        paddd	A3,B3_1

	rol	B1_1,cl
;-------
	mov	ecx,eB1
					movq	A3_H,A3

	add	A1,B1_1
	rol	eA1,cl
					psrld	A3,29

	rol	A1,3
					pslld	A3_H,3
					por	A3,A3_H

	lea	ecx,[A1+B1_1]
	add	eA1,A1
	add	A1,S1(17)

	add	B1_2,ecx
	xor	eB1,eA1
					movq	mm5,B3_1

	rol	B1_2,cl
	mov	ecx,eA1
					movq	S(11),A3

	add	A1,B1_2
	rol	eB1,cl
					paddd	mm5,A3

	rol	A1,3		       
					pshufw	mm7,mm5,0xee
					paddd	B3_2,mm5

	add	eB1,A1
	lea	ecx,[A1+B1_2]
	add	A1,S1(18)

	add	B1_0,ecx
	xor	eA1,eB1
					pand	mm5,mm3

	rol	B1_0,cl
	mov	ecx,eB1
					pshufw	A3_H,B3_2,0xee
;------------Round 3(7)------------

	add	A1,B1_0
	rol	eA1,cl
					punpckldq	B3_2,B3_2

	rol	A1,3		       
	mov	ecx,B1_0
					pand	mm7,mm3

	add	eA1,A1
	add	ecx,A1
	add	A1,S1(19)

	xor	eB1,eA1
	add	B1_1,ecx
					psllq	B3_2,mm5

	rol	B1_1,cl
	mov	ecx,eA1
					psllq	A3_H,mm7

	add	A1,B1_1
	rol	eB1,cl
					paddd	A3,[S_not_12]

	rol	A1,3		       
					punpckhdq	B3_2,A3_H
					paddd	A3,B3_2					

	lea	ecx,[A1+B1_1]
	add	eB1,A1
					movq	A3_H,A3

	add	B1_2,ecx
	xor	eA1,eB1
	add	A1,S1(20)

	rol	B1_2,cl
;---
	mov	ecx,eB1
					movq	mm5,B3_2

	add	A1,B1_2
	rol	eA1,cl
					psrld	A3,29

	rol	A1,3
					pslld	A3_H,3
					por	A3,A3_H

	lea	ecx,[A1+B1_2]
	add	eA1,A1
	add	A1,S1(21)
	
	add	B1_0,ecx
	xor	eB1,eA1
					paddd	mm5,A3

	rol	B1_0,cl
	mov	ecx,eA1
					movq	S(12),A3

	add	A1,B1_0
	rol	eB1,cl
					paddd	A3,[S_not_13]

	rol	A1,3
					paddd	B3_0,mm5
					pshufw	mm7,mm5,0xee

	lea	ecx,[A1+B1_0]
	add	eB1,A1
	add	A1,S1(22)

	add	B1_1,ecx
	xor	eA1,eB1
					pand	mm5,mm3

	rol	B1_1,cl
;-------
	mov	ecx,eB1
					pshufw	A3_H,B3_0,0xee

	add	A1,B1_1
	rol	eA1,cl
					punpckldq	B3_0,B3_0
	
	rol	A1,3
					psllq	B3_0,mm5
					pand	mm7,mm3

	lea	ecx,[A1+B1_1]
	add	eA1,A1
	add	A1,S1(23)

	add	B1_2,ecx        
	xor	eB1,eA1
					psllq	A3_H,mm7

	rol	B1_2,cl
	mov	ecx,eA1
					punpckhdq	B3_0,A3_H

	add	A1,B1_2
	rol	eB1,cl
					paddd	A3,B3_0

	rol	A1,3		      
					movq	A3_H,A3
					psrld	A3,29

	lea	ecx,[A1+B1_2]
	add	eB1,A1
	add	A1,S1(24)

	add	B1_0,ecx
	xor	eA1,eB1
					pslld	A3_H,3

	rol	B1_0,cl
;------------Round 3(10)------------
	mov	ecx,eB1
					por	A3,A3_H

	add	A1,B1_0
	rol	eA1,cl
					movq	mm5,B3_0

	rol	A1,3		       
	add	eA1,A1
					paddd	mm5,A3

	cmp	eA1,[work_C_0]
					paddd	B3_1,mm5
	je	near _checkKey1High_k7_mixed

_Key2Round3_k7_mixed:
	mov	eA1,[work_P_0]
	mov	A1,S2_alt(0)
	add	eA1,A1

	mov	B1_1,L2(1)
	add	A1,S2_alt(1)
					pshufw	mm7,mm5,0xee

	mov	eB1,[work_P_1]
	mov	B1_2,L2(2)
					pand	mm5,mm3

	add	A1,B1_1
	mov	B1_0,L2(0)
					movq	S(13),A3

	rol	A1,3
					paddd	A3,[S_not_14]
					pshufw	A3_H,B3_1,0xee

	add	eB1,A1
	lea	ecx,[A1+B1_1]
	add	A1,S2_alt(2)

	add	B1_2,ecx
	xor	eA1,eB1
					punpckldq	B3_1,B3_1

	rol	B1_2,cl
	mov	ecx,eB1
					pand	mm7,mm3

	add	A1,B1_2
	rol	eA1,cl
					psllq	B3_1,mm5

	rol	A1,3
					psllq	A3_H,mm7
					punpckhdq	B3_1,A3_H

	add	eA1,A1
	lea	ecx,[A1+B1_2]
	add	A1,S2_alt(3)

	add	B1_0,ecx
	xor	eB1,eA1
					paddd	A3,B3_1
					
	rol	B1_0,cl
	mov	ecx,eA1
					movq	A3_H,A3

	add	A1,B1_0
	rol	eB1,cl
					psrld	A3,29

	rol	A1,3
					pslld	A3_H,3
					movq	mm5,B3_1

	lea	ecx,[A1+B1_0]
	add	eB1,A1
					por	A3,A3_H

	add	B1_1,ecx
	xor	eA1,eB1
	add	A1,S2_alt(4)

	rol	B1_1,cl
	mov	ecx,eB1
					movq	S(14),A3

	add	A1,B1_1
	rol	eA1,cl
					paddd	mm5,A3

	rol	A1,3
					paddd	B3_2,mm5
					pshufw	mm7,mm5,0xee

	add	eA1,A1
	lea	ecx,[A1+B1_1]
					paddd	A3,[S_not_15]

	add	B1_2,ecx
	xor	eB1,eA1
	add	A1,S2_alt(5)

	rol	B1_2,cl
	mov	ecx,eA1
					pand	mm5,mm3

	add	A1,B1_2
	rol	eB1,cl
					pshufw	A3_H,B3_2,0xee

	rol	A1,3
					punpckldq	B3_2,B3_2
					pand	mm7,mm3

	add	eB1,A1
	lea	ecx,[A1+B1_2]
					psllq	B3_2,mm5

	add	B1_0,ecx
	xor	eA1,eB1
	add	A1,S2_alt(6)

	rol	B1_0,cl
	mov	ecx,eB1
					psllq	A3_H,mm7

	add	A1,B1_0
	rol	eA1,cl
					punpckhdq	B3_2,A3_H

	rol	A1,3
					paddd	A3,B3_2
					movq	A3_H,A3

	add	eA1,A1
	lea	ecx,[A1+B1_0]
	add	A1,S2_alt(7)

	add	B1_1,ecx
	xor	eB1,eA1
					psrld	A3,29

	rol	B1_1,cl
	mov	ecx,eA1
					pslld	A3_H,3

	rol	eB1,cl
	add	A1,B1_1
	mov	ecx,B1_1

	rol	A1,3
					por	A3,A3_H
					movq	mm5,B3_2
					
	add	eB1,A1
	add	ecx,A1
					movq	S(15),A3

	xor	eA1,eB1
	add	B1_2,ecx
	add	A1,S2_alt(8)

	rol	B1_2,cl
	mov	ecx,eB1
					paddd	mm5,A3

	rol	eA1,cl
	add	A1,B1_2
					paddd	B3_0,mm5
;---

	rol	A1,3
					pshufw	mm7,mm5,0xee
					pand	mm5,mm3

	lea	ecx,[A1+B1_2]
	add	eA1,A1
	add	A1,S2_alt(9)

	add	B1_0,ecx
	xor	eB1,eA1
					pshufw	A3_H,B3_0,0xee

	rol	B1_0,cl
	mov	ecx,eA1
					punpckldq	B3_0,B3_0

	add	A1,B1_0
	rol	eB1,cl
					pand	mm7,mm3

	rol	A1,3
					psllq	B3_0,mm5
					paddd	A3,[S_not_16]

	lea	ecx,[A1+B1_0]
	add	eB1,A1
	add	A1,S2_alt(10)

	add	B1_1,ecx
	xor	eA1,eB1
					psllq	A3_H,mm7

	rol	B1_1,cl
;-------
	mov	ecx,eB1
					punpckhdq	B3_0,A3_H

	add	A1,B1_1
	rol	eA1,cl
	mov	ecx,B1_1

	rol	A1,3
					paddd	A3,B3_0
					movq	A3_H,A3

	add	ecx,A1
	add	eA1,A1
	add	A1,S2_alt(11)

	add	B1_2,ecx
	xor	eB1,eA1
					psrld	A3,29

	rol	B1_2,cl
	mov	ecx,eA1
					pslld	A3_H,3

	add	A1,B1_2
	rol	eB1,cl
					por	A3,A3_H

	rol	A1,3		       
					movq	S(16),A3
					movq	mm5,B3_0

	lea	ecx,[A1+B1_2]
	add	eB1,A1
	add	A1,S2_alt(12)

	add	B1_0,ecx
	xor	eA1,eB1
					paddd	mm5,A3

	rol	B1_0,cl
;------------Round 3(4) - key 2------------
	mov	ecx,eB1
					paddd	A3,[S_not_17]

	add	A1,B1_0
	rol	eA1,cl
					paddd	B3_1,mm5

	rol	A1,3
					pshufw	mm7,mm5,0xee
					pand	mm5,mm3

	lea	ecx,[A1+B1_0]
	add	eA1,A1
	add	A1,S2_alt(13)

	add	B1_1,ecx
	xor	eB1,eA1
					pshufw	A3_H,B3_1,0xee

	rol	B1_1,cl 	       
	mov	ecx,eA1
					punpckldq	B3_1,B3_1

	add	A1,B1_1
	rol	eB1,cl
					pand	mm7,mm3

	rol	A1,3		       
					psllq	B3_1,mm5
					psllq	A3_H,mm7

	lea	ecx,[A1+B1_1]
	add	eB1,A1
	add	A1,S2_alt(14)

	add	B1_2,ecx
	xor	eA1,eB1
					punpckhdq	B3_1,A3_H

	rol	B1_2,cl
	;---
	mov	ecx,eB1
					paddd	A3,B3_1

	add	A1,B1_2
	rol	eA1,cl
					movd	S1(1),B3_1

	rol	A1,3
					movq	A3_H,A3
					psrld	A3,29

	lea	ecx,[A1+B1_2]
	add	eA1,A1
	add	A1,S2_alt(15)

	add	B1_0,ecx
	xor	eB1,eA1
					pslld	A3_H,3

	rol	B1_0,cl
	mov	ecx,eA1
					por	A3,A3_H

	add	A1,B1_0
	rol	eB1,cl
					movq	mm5,B3_1

	rol	A1,3
					movq	S1(17),A3
					paddd	mm5,A3

	lea	ecx,[A1+B1_0]
	add	eB1,A1
	add	A1,S2_alt(16)

	add	B1_1,ecx
	xor	eA1,eB1
					paddd	B3_2,mm5

	rol	B1_1,cl
	;-------
	mov	ecx,eB1
					pshufw	mm7,mm5,0xee

	add	A1,B1_1
	rol	eA1,cl
					pand	mm5,mm3

	rol	A1,3
					pshufw	A3_H,B3_2,0xee
					punpckldq	B3_2,B3_2

	lea	ecx,[A1+B1_1]
	add	eA1,A1
	add	A1,S2_alt(17)
			
	add	B1_2,ecx
	xor	eB1,eA1
					pand	mm7,mm3

	rol	B1_2,cl
	mov	ecx,eA1
					psllq	B3_2,mm5

	add	A1,B1_2
	rol	eB1,cl
					psllq	A3_H,mm7

	rol	A1,3		       
					punpckhdq	B3_2,A3_H
					paddd	A3,[S_not_18]

	lea	ecx,[A1+B1_2]
	add	eB1,A1
	add	A1,S2_alt(18)

	add	B1_0,ecx
	xor	eA1,eB1
					movq	L2(0),B3_1	;L2(1)
;------------Round 3(7) - key 2------------
	rol	B1_0,cl
	mov	ecx,eB1
                                        paddd	A3,B3_2

	add	A1,B1_0                 
	rol	eA1,cl
					movq	A3_H,A3

	rol	A1,3		       
					psrld	A3,29
					pslld	A3_H,3

	lea	ecx,[A1+B1_0]
	add	eA1,A1
	add	A1,S2_alt(19)

	add	B1_1,ecx
	xor	eB1,eA1
					por	A3,A3_H

	rol	B1_1,cl 	       
	mov	ecx,eA1
					movq	mm5,B3_2

	add	A1,B1_1
	rol	eB1,cl
					movq	S(18),A3

	rol	A1,3		       
					paddd	mm5,A3	
					paddd	A3,[S_not_19]

	lea	ecx,[A1+B1_1]
	add	eB1,A1
	add	A1,S2_alt(20)

	add	B1_2,ecx
	xor	eA1,eB1
					paddd	B3_0,mm5

	rol	B1_2,cl
	;---
	mov	ecx,eB1
					pshufw	mm7,mm5,0xee

	add	A1,B1_2
	rol	eA1,cl
					pand	mm5,mm3

	rol	A1,3
					pshufw	A3_H,B3_0,0xee
					punpckldq	B3_0,B3_0

	lea	ecx,[A1+B1_2]
	add	eA1,A1
	add	A1,S2_alt(21)

	add	B1_0,ecx
	xor	eB1,eA1
					pand	mm7,mm3

	rol	B1_0,cl
	mov	ecx,eA1
					psllq	B3_0,mm5

	add	A1,B1_0
	rol	eB1,cl
					psllq	A3_H,mm7

	rol	A1,3
					punpckhdq	B3_0,A3_H
					movd	S1(0),B3_0

	lea	ecx,[A1+B1_0]
	add	eB1,A1
	add	A1,S2_alt(22)

	add	B1_1,ecx
	xor	eA1,eB1
					paddd	A3,B3_0

	rol	B1_1,cl
	;-------
	mov	ecx,eB1
					movd	S1(19),A3

	add	A1,B1_1
	rol	eA1,cl
					punpckhdq	A3,A3

	rol	A1,3
					movd	S2(19),A3
					punpckhdq B3_0,B3_0

	lea	ecx,[A1+B1_1]
	add	eA1,A1
	add	A1,S2_alt(23)

	add	B1_2,ecx
	xor	eB1,eA1
					movd	L2(0),B3_0

	rol	B1_2,cl
	mov	ecx,eA1
					movd	S1(2),B3_2

	add	A1,B1_2
	rol	eB1,cl
					punpckhdq B3_2,B3_2

	rol	A1,3		  
					movd	L2(2),B3_2
					pshufw	B3_2,[RC5_72_L0hi],0x44 

	lea	ecx,[A1+B1_2]
	add	eB1,A1
					movq	mm5,[S3]

	add	B1_0,ecx
	xor	eA1,eB1
	add	A1,S2_alt(24)

	rol	B1_0,cl
;------------Round 3(10) - key 2------------
	mov	ecx,eB1
					paddb	B3_2,[incr2]

	rol	eA1,cl
	add	A1,B1_0
					movq	S(3),mm5

	rol	A1,3		        
	add	eA1,A1

	cmp	eA1,[work_C_0]
	je	near _checkKey2High_k7_mixed

_NextKey:
	dec	dword [work_iterations]

	jz	near finished_Found_nothing

	mov	B1_0,S1(0)
	mov	A2,S2(19)
	mov	A1,S1(19)
	mov	B1_1,S1(1)

	mov	B2,L2(0)
	mov	B1_2,S1(2)

	add	byte [RC5_72_L0hi],2
	jnc	key_setup_1_inner_loop
	
	mov	eax, [unitwork_addr]
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

	mov	eax, [unitwork_addr]
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
	mov	esi, [iterations_addr]
	mov	esi, [esi]

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
	mov	eax, [unitwork_addr]
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
	mov	esi, [iterations_addr]
	mov	esi, [esi]

	shl	ecx, 1

	sub	[esi], ecx
	mov	eax, RESULT_FOUND

	jmp	finished

finished_Found_nothing:
	mov	eax, [unitwork_addr]
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
	mov	esp, [save_esp]

	emms
	ret
