; RC5-72 Assembly version - P4E/Core 2 optimized version - 2 pipe
; Vyacheslav Chupyatov - goteam@mail.ru - 26/04/2003,07/03/2007,03/04/2008
; For use by distributed.net. 

%define P	  0xB7E15163
%define Q	  0x9E3779B9
%define S_not(N)  ((P+Q*(N)) & 0xFFFFFFFF)

%ifdef __OMF__ 		;  Borland and Watcom compilers/linkers
[SECTION _DATA FLAT USE32 align=16 CLASS=DATA]
%else
[SECTION .data align=16]
%endif

%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_72_unit_func_go_2a]
[GLOBAL rc5_72_unit_func_go_2a]


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
defwork work_S1,26
defwork work_S2,26
defwork work_pre5_r1
defwork RC5_72_L0hi
defwork work_pre4_r1
defwork work_pre2_r1
defwork work_pre3_r1
defwork work_pre1_r1
defwork work_L3,3


defwork work_iterations

defwork work_C_1
defwork	iterations_addr
defwork work_P_0
defwork work_P_1
defwork work_C_0
defwork save_ebx
defwork save_esi
defwork save_edi
defwork save_ebp
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

%define S1(N)			[work_S1+((N)*4)]
%define S2(N)			[work_S2+((N)*4)]

%define L2(N)			[work_L+((N)*4)]
%define L3(N)			[work_L3+((N)*4)]


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

%macro KEY_SETUP_ROUND 4
	%if %4==1
		add	A1,S_not(%1)
	%else
		%if %4==2
			%if %1==1
				add	A1,[work_pre2_r1]
			%else 
				%if %1==2
					add	A1,[work_pre4_r1]
				%else
					add	A1,S1(%1)
				%endif 
			%endif
		%else
			add	A1,S1(%1)
		%endif
	%endif
	add	A1,B1_%2
	rol	A1,3
	mov	S1(%1),A1

	%if	(%4<3)||(%1<25)
		lea	ecx,[A1+B1_%2]
		add	B1_%3,ecx
		rol	B1_%3,cl
	%endif
	;---------------
	%if %4==1
		add	A2,S_not(%1)
	%else
		%if %4==2
			%if %1==1
				add	A2,[work_pre2_r1]
			%else 
				%if %1==2
					add	A2,[work_pre4_r1]
				%else
					add	A2,S2(%1)
				%endif 
			%endif
		%else
			add	A2,S2(%1)
		%endif
	%endif
	add	A2,B2
	rol	A2,3
	mov	S2(%1),A2
	
	%if	(%4<3)||(%1<25)
		lea	ecx,[A2+B2]
		%if (%1==3)&&(%4==1)
			mov	B2,[work_pre1_r1]
		%else
			%if (%1==4)&&(%4==1)
				mov	B2,[work_pre3_r1]
			%else
				mov	B2,L2(%3)
			%endif
		%endif
		add	B2,ecx
		rol	B2,cl
		%if	(%4<3)||(%1<22)
			mov	L2(%3),B2
		%endif
	%endif

%endmacro

%macro ENC_ROUND 4
	mov	ecx,eB1
	xor	eA1,eB1
	rol	eA1,cl
	add	eA1,S%2(%1)
	%if %1!=24 
		mov	ecx,eA1
		xor	eB1,eA1
		rol	eB1,cl
		add	eB1,S%2(%1+1)
	%endif

	mov	ecx,eB2
	xor	eA2,eB2
	rol	eA2,cl
	add	eA2,S%3(%1)
	%if %1!=24 
		mov	ecx,eA2
		xor	eB2,eA2
		rol	eB2,cl
		add	eB2,S%3(%1+1)
	%endif
%endmacro

%macro ENC_HELPER_PREPARE1 0
	mov	ecx,[work_pre5_r1]
	mov	A3,[work_pre4_r1]

	mov	B3,[RC5_72_L0hi]

	lea	B3,[B3+ecx+2]
	
	rol	B3,cl
	mov	L3(2),B3
%endmacro

%macro ENC_HELPER_PREPARE2 0
	mov	ecx,[work_pre5_r1]
	mov	A3,[work_pre4_r1]

	mov	B3,[RC5_72_L0hi]

	lea	B3,[B3+ecx+3]
	
	rol	B3,cl
	mov	L2(2),B3
%endmacro

%macro ENC_ROUND_HELPER2 4
	lea	A3,[A3+B3+S_not(%1)]
	rol	A3,3
	mov	S2(%1),A3
	
	lea	ecx,[A3+B3]
	%if %1==3
		mov	B3,[work_pre1_r1]
	%else
		%if %1==4
			mov	B3,[work_pre3_r1]
		%else
			mov	B3,L2(%3)
		%endif
	%endif
	add	B3,ecx
	rol	B3,cl
	mov	L2(%3),B3
%endmacro

%macro ENC_ROUND_HELPER1 4
	lea	A3,[A3+B3+S_not(%1)]
	rol	A3,3
	mov	S1(%1),A3
	
	lea	ecx,[A3+B3]
	%if %1==3
		mov	B3,[work_pre1_r1]
	%else
		%if %1==4
			mov	B3,[work_pre3_r1]
		%else
			mov	B3,L3(%3)
		%endif
	%endif
	add	B3,ecx
	rol	B3,cl
	mov	L3(%3),B3
%endmacro


; register allocation for the key setup blocks
%define A1	eax
%define	A2	edx
%define	A3	edi

%define B1_2	ebx
%define B1_1    edi
%define B1_0    esi
%define B2	ebp
%define B3	ebp

%define eA1	eax
%define eA2	edx
%define eB1	ebx
%define	eB2	esi


align 16
startseg:
_rc5_72_unit_func_go_2a:
rc5_72_unit_func_go_2a:
	mov	ecx, esp
	sub	esp, work_size


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

	shr	edi, 1                 ; 2 потока
	mov	[work_iterations], edi
	mov	B2, [RC5_72UnitWork_L0lo]

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
key_setup_1_inner_loop:
	mov	ecx,[work_pre5_r1]
	mov	A1,[work_pre4_r1]

	lea	B2,[B1_2+ecx+1]
	add	B1_2,ecx
	mov	A2,A1
	
	rol	B1_2,cl

	rol	B2,cl
	mov	L2(2),B2
	
	mov	B1_0,[work_pre1_r1]			;L1(0)
	mov	B1_1,[work_pre3_r1]	     		;L1(1)
	;----------начало R1-------------
	KEY_SETUP_ROUND  3,2,0,1			
	KEY_SETUP_ROUND  4,0,1,1			
	KEY_SETUP_ROUND  5,1,2,1
	KEY_SETUP_ROUND  6,2,0,1
	KEY_SETUP_ROUND  7,0,1,1
k7align 64
key_setup_1_inner_loop2:
	KEY_SETUP_ROUND  8,1,2,1
	KEY_SETUP_ROUND  9,2,0,1
	KEY_SETUP_ROUND 10,0,1,1
	KEY_SETUP_ROUND 11,1,2,1
	KEY_SETUP_ROUND 12,2,0,1
	KEY_SETUP_ROUND 13,0,1,1
	KEY_SETUP_ROUND 14,1,2,1
	KEY_SETUP_ROUND 15,2,0,1
	KEY_SETUP_ROUND 16,0,1,1
	KEY_SETUP_ROUND 17,1,2,1
	KEY_SETUP_ROUND 18,2,0,1
	KEY_SETUP_ROUND 19,0,1,1
	KEY_SETUP_ROUND 20,1,2,1
	KEY_SETUP_ROUND 21,2,0,1
	KEY_SETUP_ROUND 22,0,1,1
	KEY_SETUP_ROUND 23,1,2,1
	KEY_SETUP_ROUND 24,2,0,1
	KEY_SETUP_ROUND 25,0,1,1


	lea	A1,[A1+B1_1+0xBF0A8B1D]
	rol	A1,3
	mov	S1(0),A1

	add	A2,0xBF0A8B1D				;S(0)
	add	A2,B2
	rol	A2,3
	mov	S2(0),A2

	lea	ecx,[A1+B1_1]
	add	B1_2,ecx
	rol	B1_2,cl

	lea	ecx,[A2+B2]
	mov	B2,L2(2)
	add	B2,ecx
	rol	B2,cl
	mov	L2(2),B2

	KEY_SETUP_ROUND  1,2,0,2
	KEY_SETUP_ROUND  2,0,1,2
 	KEY_SETUP_ROUND  3,1,2,2
 	KEY_SETUP_ROUND  4,2,0,2
 	KEY_SETUP_ROUND  5,0,1,2
 	KEY_SETUP_ROUND  6,1,2,2
 	KEY_SETUP_ROUND  7,2,0,2
 	KEY_SETUP_ROUND  8,0,1,2
 	KEY_SETUP_ROUND  9,1,2,2
 	KEY_SETUP_ROUND 10,2,0,2
 	KEY_SETUP_ROUND 11,0,1,2
 	KEY_SETUP_ROUND 12,1,2,2
 	KEY_SETUP_ROUND 13,2,0,2
 	KEY_SETUP_ROUND 14,0,1,2
 	KEY_SETUP_ROUND 15,1,2,2
 	KEY_SETUP_ROUND 16,2,0,2
 	KEY_SETUP_ROUND 17,0,1,2
 	KEY_SETUP_ROUND 18,1,2,2
 	KEY_SETUP_ROUND 19,2,0,2
 	KEY_SETUP_ROUND 20,0,1,2
 	KEY_SETUP_ROUND 21,1,2,2
 	KEY_SETUP_ROUND 22,2,0,2
 	KEY_SETUP_ROUND 23,0,1,2
 	KEY_SETUP_ROUND 24,1,2,2
 	KEY_SETUP_ROUND 25,2,0,2

	;-------Round 3
 	KEY_SETUP_ROUND  0,0,1,3
 	KEY_SETUP_ROUND  1,1,2,3
 	KEY_SETUP_ROUND  2,2,0,3
 	KEY_SETUP_ROUND  3,0,1,3
 	KEY_SETUP_ROUND  4,1,2,3
 	KEY_SETUP_ROUND  5,2,0,3
 	KEY_SETUP_ROUND  6,0,1,3
 	KEY_SETUP_ROUND  7,1,2,3
 	KEY_SETUP_ROUND  8,2,0,3
 	KEY_SETUP_ROUND  9,0,1,3
 	KEY_SETUP_ROUND 10,1,2,3
 	KEY_SETUP_ROUND 11,2,0,3
 	KEY_SETUP_ROUND 12,0,1,3
 	KEY_SETUP_ROUND 13,1,2,3
 	KEY_SETUP_ROUND 14,2,0,3
 	KEY_SETUP_ROUND 15,0,1,3
 	KEY_SETUP_ROUND 16,1,2,3
 	KEY_SETUP_ROUND 17,2,0,3
 	KEY_SETUP_ROUND 18,0,1,3
 	KEY_SETUP_ROUND 19,1,2,3
 	KEY_SETUP_ROUND 20,2,0,3
 	KEY_SETUP_ROUND 21,0,1,3
 	KEY_SETUP_ROUND 22,1,2,3
 	KEY_SETUP_ROUND 23,2,0,3
 	KEY_SETUP_ROUND 24,0,1,3
 	KEY_SETUP_ROUND 25,1,2,3

;--------------------Encryption (Setup) Key 1,2,3--------
	mov	eA1,[work_P_0]
	mov	eB1,[work_P_1]
	
	mov	eA2,eA1

	add	eA1,S1(0)
	add	eA2,S2(0)

	mov	eB2,eB1

	add	eB1,S1(1)
	add	eB2,S2(1)

;--------------------Encryption (2)----------------------
	ENC_ROUND  2,1,2,2
		ENC_HELPER_PREPARE2
	ENC_ROUND  4,1,2,2
		ENC_ROUND_HELPER2  3,2,0,1
	ENC_ROUND  6,1,2,2
		ENC_ROUND_HELPER2  4,0,1,1			
	ENC_ROUND  8,1,2,2
		ENC_ROUND_HELPER2  5,1,2,1
	ENC_ROUND 10,1,2,2
		ENC_ROUND_HELPER2  6,2,0,1
	ENC_ROUND 12,1,2,2
		ENC_ROUND_HELPER2  7,0,1,1
	ENC_ROUND 14,1,2,2
		ENC_HELPER_PREPARE1
	ENC_ROUND 16,1,2,2
		ENC_ROUND_HELPER1  3,2,0,1
	ENC_ROUND 18,1,2,2
		ENC_ROUND_HELPER1  4,0,1,1			
	ENC_ROUND 20,1,2,2
		ENC_ROUND_HELPER1  5,1,2,1
	ENC_ROUND 22,1,2,2
		ENC_ROUND_HELPER1  6,2,0,1
	ENC_ROUND 24,1,2,2
		ENC_ROUND_HELPER1  7,0,1,1
                 
	cmp	eA1,[work_C_0]	
	je	near _checkKey1High_k7_mixed

Key1_Finished:

	cmp	eA2,[work_C_0]	
	je	near _checkKey2High_k7_mixed

Key2_Finished:

	mov	ebx,[RC5_72_L0hi]
	dec	dword [work_iterations]

	jz	near finished_Found_nothing

	add	bl,2
	movzx	ebx,bl
	mov	[RC5_72_L0hi],ebx

	mov	A1,A3
	mov	B1_2,L3(2)
	mov	A2,S2(7)
	mov	B1_0,L3(0)
	mov	B1_1,B3
	mov	B2,L2(1)

	jnc	key_setup_1_inner_loop2
	
	mov	ebx,[RC5_72_L0hi]

	mov	eax, [unitwork_addr]
	mov	B1_0, [RC5_72UnitWork_L0mid]
	mov	B2, [RC5_72UnitWork_L0lo]
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
_checkKey1High_k7_mixed:
		mov	ecx,eA1
		xor	eB1,eA1
		rol	eB1,cl
		add	eB1,S1(25)

	mov	eax, [unitwork_addr]
        inc     dword [RC5_72UnitWork_CMCcount]

	cmp	eB1,[work_C_1]

	mov	ebx, [RC5_72UnitWork_L0mid]
        mov     [RC5_72UnitWork_CMCmid], ebx

	mov	ebx, [RC5_72UnitWork_L0lo]
        mov     [RC5_72UnitWork_CMClo], ebx

	mov	ebx, [RC5_72_L0hi]
        mov     [RC5_72UnitWork_CMChi], ebx

	jne	Key1_Finished

	mov	ecx, [work_iterations]
	mov	esi, [iterations_addr]
	mov	esi, [esi]

	shl	ecx, 1

	sub	[esi], ecx
	mov	eax, RESULT_FOUND

	jmp	finished

k7align 16
_checkKey2High_k7_mixed:
		mov	ecx,eA2
		xor	eB2,eA2
		rol	eB2,cl
		add	eB2,S2(25)

	mov	eax, [unitwork_addr]
        inc     dword [RC5_72UnitWork_CMCcount]

	mov	edx, [RC5_72_L0hi]
	inc	edx

	cmp	eB2,[work_C_1]

        mov     [RC5_72UnitWork_CMChi], edx
	mov	edx, [RC5_72UnitWork_L0mid]
	mov	esi, [RC5_72UnitWork_L0lo]

        mov     [RC5_72UnitWork_CMCmid], edx
        mov     [RC5_72UnitWork_CMClo], esi
	jne	Key2_Finished

	mov	ecx, [work_iterations]
	mov	esi, [iterations_addr]
	mov	esi, [esi]

	shl	ecx, 1
	dec	ecx

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

	ret
