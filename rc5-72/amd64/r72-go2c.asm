; Copyright distributed.net 1997-2011 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
; Vyacheslav Chupyatov - goteam@mail.ru - 07/03/2011
; For use by distributed.net. 

[SECTION .text]
BITS 64

[GLOBAL _rc5_72_unit_func_go_2c]
[GLOBAL rc5_72_unit_func_go_2c]

%define P	  0xB7E15163
%define Q	  0x9E3779B9
%define S_not(N)  (P+Q*(N))

%define RESULT_NOTHING 1
%define RESULT_FOUND   2

%assign work_size 0

%macro defidef 2
    %define %1 rsp+%2
%endmacro

%macro defwork 1-2 1
    defidef %1,work_size
    %assign work_size work_size+4*(%2)
%endmacro

; local storage variables
defwork work_S1,26
defwork work_S2,26
defwork work_pre1_r1
defwork work_pre2_r1
defwork work_pre3_r1
defwork work_pre4_r1
defwork work_pre5_r1
defwork RC5_72_L0hi
defwork work_L3,3
defwork work_S3,26
defwork work_P_0
defwork work_P_1
defwork work_C_0
defwork work_C_1
defwork work_iterations
defwork save_rbx,2
defwork save_rbp,2
%ifdef _WINDOWS
defwork save_rsi,2
defwork save_rdi,2
%endif
defwork save_r12,2
defwork save_r13,2
defwork save_r14,2
defwork save_r15,2
defwork RC5_72UnitWork,2	; 1st argument (64-bit pointer), passed in rdi
defwork iterations_addr,2	; 2nd argument (64-bit pointer), passed in rsi

;; offsets within the parameter structure
%define RC5_72UnitWork_plainhi  rax+0
%define RC5_72UnitWork_plainlo  rax+4
%define RC5_72UnitWork_cipherhi rax+8
%define RC5_72UnitWork_cipherlo rax+12
%define RC5_72UnitWork_L0hi     rax+16
%define RC5_72UnitWork_L0mid    rax+20
%define RC5_72UnitWork_L0lo     rax+24
%define RC5_72UnitWork_CMCcount rax+28
%define RC5_72UnitWork_CMChi    rax+32
%define RC5_72UnitWork_CMCmid   rax+36
%define RC5_72UnitWork_CMClo    rax+40


%define S1(N)                   [work_S1+((N)*4)]
%define S2(N)                   [work_S2+((N)*4)]
%define S3(N)                   [work_S3+((N)*4)]
%define L3(N)                   [work_L3+((N)*4)]


%define	A1 	eax
%define A2 	edx
%define A3 	r10d

%define B1_0	r9d
%define B1_1	r8d
%define B1_2	ebx

%define B2_0	esi
%define	B2_1	edi
%define B2_2	r14d

%define B3_0	r11d
%define B3_1	r12d
%define B3_2	r13d

%define eA1	r15d
%define	eB1	r11d
%define eA2	r12d
%define	eB2	r13d


align 16
startseg:
_rc5_72_unit_func_go_2c:
rc5_72_unit_func_go_2c:

        sub     rsp, work_size

%ifdef _WINDOWS
        mov     [RC5_72UnitWork],rcx ; 1st argument is passed in rcx
        mov     [iterations_addr],rdx ; 2nd argument is passwd in rdx
        mov     rax, rcx	; rax points to RC5_72UnitWork

        ;; Windows requires that rsi and rdi also be preserved by callee!
        mov     [save_rsi], rsi
        mov     [save_rdi], rdi
        mov     rsi,rdx         ; rsi points to iterations
%else
        ;; Linux, FreeBSD, and other UNIX platforms
        mov     [RC5_72UnitWork],rdi ; 1st argument is passed in rdi
        mov     [iterations_addr],rsi ; 2nd argument is passwd in rsi
        mov     rax, rdi	; rax points to RC5_72UnitWork
%endif

        ;; rbp, rbx, and r12 thru r15 must be preserved by callee!
        mov     [save_rbp], rbp
        mov     [save_rbx], rbx
        mov     [save_r12], r12
        mov     [save_r13], r13
        mov     [save_r14], r14
        mov     [save_r15], r15

        mov     edx, [RC5_72UnitWork_plainlo]
        mov     edi, [RC5_72UnitWork_plainhi]

        mov     ebx, [RC5_72UnitWork_cipherlo]
        mov     ecx, [RC5_72UnitWork_cipherhi]

        mov     [work_P_0], edx
        mov     [work_P_1], edi

        mov     [work_C_0], ebx
        mov     [work_C_1], ecx
        mov     edi, [rsi]	; rsi points to iterations

        mov     B1_2, [RC5_72UnitWork_L0hi]
        shr     edi, 1
        mov     B1_1, [RC5_72UnitWork_L0mid]
        mov     B1_0, [RC5_72UnitWork_L0lo]

        mov     [work_iterations], edi
 	mov	[RC5_72_L0hi],B1_2

align 16
key_setup_1_bigger_loop:

	add	B1_0,0xBF0A8B1D
	rol	B1_0,0x1D
	mov	A1,0x15235639

	mov	ecx,B1_0				;L(0)
	add	A1,B1_0					;S[1]+S[0]+L[0]
	mov	[work_pre1_r1],B1_0

	rol	A1,3					;S[1]
	add	B1_1,B1_0				;mid+L(0)

	add	ecx,A1					;L[0]+S[1]
	add	B1_1,A1					;mid+L(0)+S[1]
	mov	[work_pre2_r1],A1

	add	A1,S_not(2)				;S[1]+S_not(2)
	rol	B1_1,cl					;L(1)

	add	A1,B1_1					;S[1]+S_not(2)+L[1]

	mov	ecx,B1_1 				;L[1]
	rol	A1,3					;S[2]
	add	ecx,A1					;L[1]+S[2]
	mov	[work_pre5_r1],ecx				

	mov	[work_pre3_r1],B1_1
	mov	[work_pre4_r1],A1

key_setup_1_inner_loop:
	lea	B2_2,[B1_2+1]				;Lhi
	add	A1,S_not(3)				;S[2]+S_not(3)
	add	B1_2,ecx				;Lhi+L[1]+S[2]

	rol	B1_2,cl 				;L1(2)
	add	B2_2,ecx				;Lhi+1+L[1]+S[2]

	rol	B2_2,cl					;L2(2)
	lea	A2,[A1+B2_2]
	add	A1,B1_2 				;S1[2]+S_not(3)+L1[2]

	rol	A1,3					;S1[3]
	mov	B1_0,[work_pre1_r1]			;L1(0)
	mov	S1(3),A1	

	lea	ecx,[A1+B1_2]

;------------------Round 1(4)-------------------------
	add	B1_0,ecx				;L1[2]+S1[3]+L1[0]

	rol	A2,3			     		;S2[3]
	add	A1,S_not(4)		     		;S1[3]+S_not(4)

	rol	B1_0,cl 		     		;L1[0]
	lea	ecx,[A2+B2_2]		     		;S2[3]+L2[2]
	mov	B2_0, [work_pre1_r1]			;L2(0)

	mov	S2(3),A2
	add	A1,B1_0 		     		;S1[3]+L1[0]+S_not(4)
	mov	B1_1,[work_pre3_r1]	     		;L1(1)

	add	B2_0,ecx		     		;L2[2]+S2[3]+L2[0]
	rol	A1,3			     		;S1[4]
	add	A2,S_not(4)		     		;S2[3]+S_not(4)

	rol	B2_0,cl			     		;L2[0]
	lea	ecx,[A1+B1_0]		     		;S1[4]+L1[0]

	;---
	add	A2,B2_0	 		     		;S2[3]+L2[0]+S_not(4)
	mov	S1(4),A1
	add	A1,S_not(5)		     		;S1[4]+S_not(5)

	add	B1_1,ecx		     		;L1[1]+S1[4]+L1[0]
	rol	A2,3			     		;S2[4]

	rol	B1_1,cl 		     		;L1[1]
	lea	ecx,[A2+B2_0]		 		;S2[4]+L2[0]

	mov	S2(4),A2
	add	A1,B1_1 		     		;S1[4]+L1[1]+S_not(5)

	rol	A1,3			     		;S1[5]
	mov	B2_1,[work_pre3_r1]			;L2(1)
	mov	S1(5),A1

	add	B2_1,ecx			     		;L2[1]+S2[4]+L2[0]
	add	A2,S_not(5)		     		;S2[4]+S_not(5)

	rol	B2_1,cl 			     		;L2[1]
	lea	ecx,[A1+B1_1]		 		;S1[5]+L1[1]
;---------------------Round 1(6)-------------------
	add	A2,B2_1
	add	A1,S_not(6)

	rol	A2,3
	add	B1_2,ecx

	mov	S2(5),A2
	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	add	A2,S_not(6)

	add	B2_2,ecx
	rol	A1,3

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	;---
	add	A2,B2_2
	mov	S1(6),A1
	rol	A2,3

	add	A1,S_not(7)
	add	B1_0,ecx

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	mov	S2(6),A2
	add	A2,S_not(7)
	add	A1,B1_0

	add	B2_0,ecx
	rol	A1,3

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
;---------------------Round 1(8)-------------------
	add	A2,B2_0
	mov	S1(7),A1
	add	A1,S_not(8)

	rol	A2,3
	add	B1_1,ecx

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]
	mov	S2(7),A2

	add	A2,S_not(8)
	add	A1,B1_1

	add	B2_1,ecx
	rol	A1,3

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	;--- 
	add	A2,B2_1
	mov	S1(8),A1
	add	A1,S_not(9)

	rol	A2,3
	add	B1_2,ecx

	rol	B1_2,cl	
	lea	ecx,[A2+B2_1]
	mov	S2(8),A2

	add	A1,B1_2
	add	A2,S_not(9)

	add	B2_2,ecx
	rol	A1,3

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
;---------------------Round 1(10)-------------------
	add	A2,B2_2
	mov	S1(9),A1
	add	A1,S_not(10)

	add	B1_0,ecx
	rol	A2,3

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	mov	S2(9),A2
	add	A2,S_not(10)

	add	B2_0,ecx
	rol	A1,3

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
	;---
	add	A2,B2_0
	mov	S1(10),A1
	add	A1,S_not(11)

	rol	A2,3
	mov	S2(10),A2
	add	B1_1,ecx

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]

	add	A1,B1_1
	add	A2,S_not(11)

	rol	A1,3
	add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
;---------------------Round 1(12)-------------------
	add	A2,B2_1
	mov	S1(11),A1
	add	A1,S_not(12)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(11),A2
	add	A2,S_not(12)

	add	B2_2,ecx
	rol	A1,3

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	;---
	add	A2,B2_2
	mov	S1(12),A1
	add	A1,S_not(13)

	rol	A2,3
	mov	S2(12),A2
	add	B1_0,ecx

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	add	A2,S_not(13)

	rol	A1,3
	add	B2_0,ecx

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]

;---------------------Round 1(14)-------------------
	add	A2,B2_0
	mov	S1(13),A1
	add	A1,S_not(14)

	add	B1_1,ecx
	rol	A2,3

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]
;	mov	B2,L2(1)

	add	A1,B1_1
	mov	S2(13),A2
	add	A2,S_not(14)

	add	B2_1,ecx
	rol	A1,3

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]

	;---
	add	A2,B2_1
	mov	S1(14),A1
	add	A1,S_not(15)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(14),A2

	rol	A1,3
	add	A2,S_not(15)
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
;---------------------Round 1(16)-------------------
	add	A2,B2_2
	mov	S1(15),A1
	add	A1,S_not(16)

	rol	A2,3
	add	B1_0,ecx
 
	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	mov	S2(15),A2

	add	A2,S_not(16)
	add	B2_0,ecx
	rol	A1,3

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]

	mov	S1(16),A1
	add	A2,B2_0
	;---
	add	B1_1,ecx
	add	A1,S_not(17)
	rol	A2,3

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]

	add	A1,B1_1
	mov	S2(16),A2
	add	A2,S_not(17)

	add	B2_1,ecx
	rol	A1,3
	mov	S1(17),A1

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
;---------------------Round 1(18)-------------------
	add	A2,B2_1
	add	A1,S_not(18)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(17),A2
	add	A2,S_not(18)

	add	B2_2,ecx
	rol	A1,3

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	;---
	add	A2,B2_2
	mov	S1(18),A1
	add	A1,S_not(19)

	add	B1_0,ecx
	rol	A2,3

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]
	mov	S2(18),A2

	add	A1,B1_0
	add	A2,S_not(19)

	add	B2_0,ecx
	rol	A1,3

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
;---------------------Round 1(20)-------------------
	add	A2,B2_0
	mov	S1(19),A1
	add	A1,S_not(20)

	add	B1_1,ecx
	rol	A2,3

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]

	add	A1,B1_1
	mov	S2(19),A2
	add	A2,S_not(20)

	rol	A1,3
	add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	;---
	add	A2,B2_1
	mov	S1(20),A1
	add	A1,S_not(21)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(20),A2
	add	A2,S_not(21)

	rol	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
;---------------------Round 1(22)-------------------
	add	A2,B2_2
	mov	S1(21),A1
	add	A1,S_not(22)

	add	B1_0,ecx
	rol	A2,3

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]
	mov	S2(21),A2

	add	A1,B1_0

	rol	A1,3
	add	B2_0,ecx
	add	A2,S_not(22)

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
	mov	S1(22),A1
	;---
	add	A2,B2_0
	add	A1,S_not(23)

	add	B1_1,ecx
	rol	A2,3

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]

	add	A1,B1_1

	add	B2_1,ecx
	mov	S2(22),A2
	rol	A1,3

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	add	A2,S_not(23)
;---------------------Round 1(24)-------------------
	add	A2,B2_1
	mov	S1(23),A1
	add	A1,S_not(24)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(23),A2

	rol	A1,3
	add	B2_2,ecx
	add	A2,S_not(24)		

	rol	B2_2,cl				
	lea	ecx,[A1+B1_2]
	;---
	add	A2,B2_2
	mov	S1(24),A1
	add	A1,S_not(25)

	add	B1_0,ecx
	rol	A2,3

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	mov	S2(24),A2

	rol	A1,3
	add	B2_0,ecx
	add	A2,S_not(25)

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
;---------------------Round 1(Last)-------------------
	add	A2,B2_0
	mov	S1(25),A1
	add	A1,0xBF0A8B1D				;S(0)

	rol	A2,3					;S2(25)
	add	B1_1,ecx

	rol	B1_1,cl 				;L1(1)
	lea	ecx,[A2+B2_0]

	add	A1,B1_1
	mov	S2(25),A2

	rol	A1,3
	add	B2_1,ecx
	add	A2,0xBF0A8B1D				;S(0)

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	;---
	add	A2,B2_1
	mov	S1(0),A1
	add	A1,[work_pre2_r1]			;S[1]

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(0),A2
	add	A2,[work_pre2_r1]

	rol	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
;---------------------Round 2(2)-------------------
	add	A2,B2_2
	mov	S1(1),A1
	add	A1,[work_pre4_r1]			;S(2)

	rol	A2,3					;S2(25)
	add	B1_0,ecx

	rol	B1_0,cl 				;L1(1)
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	mov	S2(1),A2
	add	A2,[work_pre4_r1]			;S(2)

	rol	A1,3
	add	B2_0,ecx

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
	;---
	add	A2,B2_0
	mov	S1(2),A1
	add	A1,S1(3)

	add	B1_1,ecx
	rol	A2,3

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]

	add	A1,B1_1
	mov	S2(2),A2
	add	A2,S2(3)

	add	B2_1,ecx
	rol	A1,3

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
;---------------------Round 2(4)-------------------
	add	A2,B2_1
	mov	S1(3),A1
	add	A1,S1(4)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(3),A2
	add	A2,S2(4)

	rol	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	;---
	add 	A2,B2_2
	mov	S1(4),A1
	add	A1,S1(5)

	add	B1_0,ecx
	rol	A2,3

	rol 	B1_0,cl
	lea	ecx,[A2+B2_2]

	add 	A1,B1_0
	mov	S2(4),A2
	add	A2,S2(5)

	rol	A1,3
	add	B2_0,ecx

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
;---------------------Round 2(6)-------------------
	add	A2,B2_0
	mov	S1(5),A1
	add	A1,S1(6)

	add	B1_1,ecx
	rol	A2,3

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]

	add	A1,B1_1
	mov	S2(5),A2
	add	A2,S2(6)

	add	B2_1,ecx
	rol	A1,3

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	;---
	add 	A2,B2_1
	mov	S1(6),A1
	add	A1,S1(7)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(6),A2
	add	A2,S2(7)

	rol	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
;---------------------Round 2(8)-------------------
	add 	A2,B2_2
	mov	S1(7),A1
	add	A1,S1(8)

	add	B1_0,ecx
	rol	A2,3

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	mov	S2(7),A2

	add 	B2_0,ecx
	rol	A1,3
	add	A2,S2(8)

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
	;---
	add	A2,B2_0
	mov	S1(8),A1
	add	A1,S1(9)

	add	B1_1,ecx
	rol	A2,3

	rol 	B1_1,cl
	lea	ecx,[A2+B2_0]

	add	A1,B1_1
	mov	S2(8),A2
	add	A2,S2(9)

	rol	A1,3
	add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
;---------------------Round 2(10)-------------------
	add	A2,B2_1
	mov	S1(9),A1
	add	A1,S1(10)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(9),A2
	add	A2,S2(10)

	rol	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	;---
	add	A2,B2_2
	mov	S1(10),A1
	add	A1,S1(11)

	rol	A2,3
	add	B1_0,ecx

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	mov	S2(10),A2
	add	A2,S2(11)

	rol	A1,3
	add	B2_0,ecx

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
;---------------------Round 2(12)-------------------
	add	A2,B2_0
	mov	S1(11),A1

	add	B1_1,ecx
	rol	A2,3
	add	A1,S1(12)

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]

	add	A1,B1_1
	mov	S2(11),A2
	add	A2,S2(12)

	rol	A1,3
	add	B2_1,ecx

	rol 	B2_1,cl
	lea	ecx,[A1+B1_1]
	;---
	add	A2,B2_1
	mov	S1(12),A1
	add	A1,S1(13)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(12),A2
	add	A2,S2(13)

	rol	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
;---------------------Round 2(14)-------------------
	add	A2,B2_2
	mov	S1(13),A1
	add	A1,S1(14)

	add	B1_0,ecx
	rol	A2,3

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	mov	S2(13),A2
	add	A2,S2(14)

	rol	A1,3
	add	B2_0,ecx

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
	;---
	add	A2,B2_0
	mov	S1(14),A1
	add	A1,S1(15)

	add	B1_1,ecx
	rol	A2,3

	rol 	B1_1,cl
	lea	ecx,[A2+B2_0]
	mov	S2(14),A2

	add	A1,B1_1

	rol	A1,3
	add	B2_1,ecx
	add	A2,S2(15)

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
;---------------------Round 2(16)-------------------
	add	A2,B2_1
	mov	S1(15),A1
	add	A1,S1(16)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(15),A2
	add	A2,S2(16)

	rol	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	mov	S1(16),A1
	;---
	add	A2,B2_2

	add	B1_0,ecx
	rol	A2,3
	add	A1,S1(17)

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	mov	S2(16),A2
	add	A2,S2(17)

	rol	A1,3
	add	B2_0,ecx

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
;---------------------Round 2(18)-------------------
	add	A2,B2_0
	mov	S1(17),A1
	add	A1,S1(18)

	add	B1_1,ecx
	rol	A2,3

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]

	add	A1,B1_1
	mov	S2(17),A2
	add	A2,S2(18)

	rol	A1,3
	add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	;---
	add	A2,B2_1
	mov	S1(18),A1
	add	A1,S1(19)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(18),A2
	add	A2,S2(19)

	add	B2_2,ecx
	rol	A1,3

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
;---------------------Round 2(20)-------------------
	add	A2,B2_2
	mov	S1(19),A1
	add	A1,S1(20)

	add	B1_0,ecx
	rol	A2,3

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	mov	S2(19),A2
	add	A2,S2(20)

	add	B2_0,ecx
	rol	A1,3

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
	;---
	add	A2,B2_0
	mov	S1(20),A1

	add	B1_1,ecx
	rol	A2,3
	add	A1,S1(21)

	rol	B1_1,cl
	lea	ecx,[A2+B2_0]
	mov	S2(20),A2

	add	A1,B1_1
	add	A2,S2(21)

	rol	A1,3
	add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
;---------------------Round 2(22)-------------------
	add	A2,B2_1
	mov	S1(21),A1
	add	A1,S1(22)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(21),A2
	add	A2,S2(22)

	rol 	A1,3
	add	B2_2,ecx

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]
	;---
	add	A2,B2_2
	mov	S1(22),A1
	add	A1,S1(23)

	rol	A2,3
	add	B1_0,ecx

	rol	B1_0,cl
	lea	ecx,[A2+B2_2]

	add	A1,B1_0
	mov	S2(22),A2
	add	A2,S2(23)

	add	B2_0,ecx
	rol	A1,3

	rol	B2_0,cl
	lea	ecx,[A1+B1_0]
;---------------------Round 2(24)-------------------
	add	A2,B2_0
	mov	S1(23),A1
	add	A1,S1(24)

	rol	A2,3
	add	B1_1,ecx

	rol 	B1_1,cl
	lea	ecx,[A2+B2_0]

	add	A1,B1_1
	mov	S2(23),A2
	add	A2,S2(24)

	rol	A1,3
	add	B2_1,ecx

	rol	B2_1,cl
	lea	ecx,[A1+B1_1]
	;---
	add	A2,B2_1
	mov	S1(24),A1
	add	A1,S1(25)

	add	B1_2,ecx
	rol	A2,3

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]

	add	A1,B1_2
	mov	S2(24),A2

	add	B2_2,ecx
	rol	A1,3
	add	A2,S2(25)

	rol	B2_2,cl
	lea	ecx,[A1+B1_2]

	add	A2,B2_2 				;S2[24]+S2[25]+L2[2]
	mov	S1(25),A1

;---------------------Round 2-Last-------------
	add	B1_0,ecx				;L1[0]+S1[25]+L1[2]
	rol	A2,3					;S2[25]	
	add	A1,S1(0)				;S1[25]+S1[0]

	rol	B1_0,cl 				;L1[0]
	lea	ecx,[A2+B2_2]

	add	A1,B1_0 				;S1[25]+S1[0]+L1[0]
	mov	S2(25),A2

	rol	A1,3					;S1[0]
	add	B2_0,ecx				;L2[0]+S2[25]+L2[2]
	add	A2,S2(0)				;S2[25]+S2[0]

	rol	B2_0,cl					;L2[0]
	lea	ecx,[A1+B1_0]				;S1[0]+L1[0]

	add	A2,B2_0					;S2[25]+L2[0]+S2[0]
	;---
	add 	B1_1,ecx				;L1[1]+S1[0]+L1[0]
	rol	A2,3					;S2[0]

	mov	eA1,[work_P_0]
	rol	B1_1,cl 				;L1[1]
	lea	ecx,[A2+B2_0]

	mov	eB1,[work_P_1]
	add	B2_1,ecx
	mov	eA2,eA1

	add	eA2,A2
	add	eA1,A1					;eA1
	add	A1,S1(1)

	add	A2,S2(1)
	rol	B2_1,cl
;----
	add	A1,B1_1
	add	A2,B2_1

	rol	A1,3
	rol	A2,3

	lea	ecx,[A1+B1_1]
	lea	eB2,[eB1+A2]

	add	B1_2,ecx
	add	eB1,A1
	add	A1,S1(2)

	rol	B1_2,cl
	lea	ecx,[A2+B2_1]
;---------------------Round 3(2)+Encryption-------------------
	xor	eA1,eB1
			add	B2_2,ecx

			rol	B2_2,cl
	mov	ecx,eB1
			add	A2,S2(2)

	add	A1,B1_2
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_2

			rol	eA2,cl
	lea	ecx,[A1+B1_2]
			rol	A2,3

	add	eA1,A1
	add	B1_0,ecx
			add	eA2,A2

	rol	B1_0,cl
			lea	ecx,[A2+B2_2]
	;nop
;----
	add	A1,S1(3)
	xor	eB1,eA1
			add	B2_0,ecx

			rol	B2_0,cl
	mov	ecx,eA1
			add	A2,S2(3)

	add	A1,B1_0
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_0

			rol	eB2,cl
	lea	ecx,[A1+B1_0]
			rol	A2,3

	add	eB1,A1
	add	B1_1,ecx
			add	eB2,A2

	rol	B1_1,cl
			lea	ecx,[A2+B2_0]
	;nop
;---------------------Round 3(4)+Encryption-------------------
	add	A1,S1(4)
	xor	eA1,eB1
			add	B2_1,ecx

			rol	B2_1,cl
	mov	ecx,eB1
			add	A2,S2(4)

	add	A1,B1_1
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_1

			rol	eA2,cl
	lea	ecx,[A1+B1_1]
			rol	A2,3

	add	eA1,A1
	add	B1_2,ecx
			add	eA2,A2

	rol	B1_2,cl
			lea	ecx,[A2+B2_1]
	;nop
;----
	add	A1,S1(5)
	xor	eB1,eA1
			add	B2_2,ecx

			rol	B2_2,cl
	mov	ecx,eA1
			add	A2,S2(5)

	add	A1,B1_2
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_2

			rol	eB2,cl
	lea	ecx,[A1+B1_2]
			rol	A2,3

	add	eB1,A1
	add	B1_0,ecx
			add	eB2,A2

	rol	B1_0,cl
			lea	ecx,[A2+B2_2]
	;nop
;---------------------Round 3(6)+Encryption-------------------
	add	A1,S1(6)
	xor	eA1,eB1
			add	B2_0,ecx

			rol	B2_0,cl
	mov	ecx,eB1
			add	A2,S2(6)

	add	A1,B1_0
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_0

			rol	eA2,cl
	lea	ecx,[A1+B1_0]
			rol	A2,3

	add	eA1,A1
	add	B1_1,ecx
			add	eA2,A2

	rol	B1_1,cl
			lea	ecx,[A2+B2_0]
	;nop
;----
	add	A1,S1(7)
	xor	eB1,eA1
			add	B2_1,ecx

			rol	B2_1,cl
	mov	ecx,eA1
			add	A2,S2(7)

	add	A1,B1_1
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_1

			rol	eB2,cl
	lea	ecx,[A1+B1_1]
			rol	A2,3

	add	eB1,A1
	add	B1_2,ecx
			add	eB2,A2

	rol	B1_2,cl
			lea	ecx,[A2+B2_1]
	;nop
;---------------------Round 3(8)+Encryption-------------------
	add	A1,S1(8)
	xor	eA1,eB1
			add	B2_2,ecx

			rol	B2_2,cl
	mov	ecx,eB1
			add	A2,S2(8)

	add	A1,B1_2
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_2

			rol	eA2,cl
	lea	ecx,[A1+B1_2]
			rol	A2,3

	add	eA1,A1
	add	B1_0,ecx
			add	eA2,A2

	rol	B1_0,cl
			lea	ecx,[A2+B2_2]
	;nop
;----
	add	A1,S1(9)
	xor	eB1,eA1
			add	B2_0,ecx

			rol	B2_0,cl
	mov	ecx,eA1
			add	A2,S2(9)

	add	A1,B1_0
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_0

			rol	eB2,cl
	lea	ecx,[A1+B1_0]
			rol	A2,3

	add	eB1,A1
	add	B1_1,ecx
			add	eB2,A2

	rol	B1_1,cl
			lea	ecx,[A2+B2_0]
	;nop
;---------------------Round 3(10)+Encryption-------------------
	add	A1,S1(10)
	xor	eA1,eB1
			add	B2_1,ecx

			rol	B2_1,cl
	mov	ecx,eB1
			add	A2,S2(10)

	add	A1,B1_1
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_1

			rol	eA2,cl
	lea	ecx,[A1+B1_1]
			rol	A2,3

	add	eA1,A1
	add	B1_2,ecx
			add	eA2,A2

	rol	B1_2,cl
			lea	ecx,[A2+B2_1]
	;nop
;----
	add	A1,S1(11)
	xor	eB1,eA1
			add	B2_2,ecx

			rol	B2_2,cl
	mov	ecx,eA1
			add	A2,S2(11)

	add	A1,B1_2
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_2

			rol	eB2,cl
	lea	ecx,[A1+B1_2]
			rol	A2,3

	add	eB1,A1
	add	B1_0,ecx
			add	eB2,A2

	rol	B1_0,cl
			lea	ecx,[A2+B2_2]
	;nop
;---------------------Round 3(12)+Encryption-------------------
	add	A1,S1(12)
	xor	eA1,eB1
			add	B2_0,ecx

			rol	B2_0,cl
	mov	ecx,eB1
			add	A2,S2(12)

	add	A1,B1_0
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_0

			rol	eA2,cl
	lea	ecx,[A1+B1_0]
			rol	A2,3

	add	eA1,A1
	add	B1_1,ecx
			add	eA2,A2

	rol	B1_1,cl
			lea	ecx,[A2+B2_0]
	;nop
;----
	add	A1,S1(13)
	xor	eB1,eA1
			add	B2_1,ecx

			rol	B2_1,cl
	mov	ecx,eA1
			add	A2,S2(13)

	add	A1,B1_1
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_1

			rol	eB2,cl
	lea	ecx,[A1+B1_1]
			rol	A2,3

	add	eB1,A1
	add	B1_2,ecx
			add	eB2,A2

	rol	B1_2,cl
			lea	ecx,[A2+B2_1]
	;nop
;---------------------Round 3(14)+Encryption-------------------
	add	A1,S1(14)
	xor	eA1,eB1
			add	B2_2,ecx

			rol	B2_2,cl
	mov	ecx,eB1
			add	A2,S2(14)

	add	A1,B1_2
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_2

			rol	eA2,cl
	lea	ecx,[A1+B1_2]
			rol	A2,3

	add	eA1,A1
	add	B1_0,ecx
			add	eA2,A2

	rol	B1_0,cl
			lea	ecx,[A2+B2_2]
	;nop
;----
	add	A1,S1(15)
	xor	eB1,eA1
			add	B2_0,ecx

			rol	B2_0,cl
	mov	ecx,eA1
			add	A2,S2(15)

	add	A1,B1_0
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_0

			rol	eB2,cl
	lea	ecx,[A1+B1_0]
			rol	A2,3

	add	eB1,A1
	add	B1_1,ecx
			add	eB2,A2

	rol	B1_1,cl
			lea	ecx,[A2+B2_0]
	;nop
;---------------------Round 3(16)+Encryption-------------------
	add	A1,S1(16)
	xor	eA1,eB1
			add	B2_1,ecx

			rol	B2_1,cl
	mov	ecx,eB1
			add	A2,S2(16)

	add	A1,B1_1
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_1

			rol	eA2,cl
	lea	ecx,[A1+B1_1]
			rol	A2,3

	add	eA1,A1
	add	B1_2,ecx
			add	eA2,A2

	rol	B1_2,cl
			lea	ecx,[A2+B2_1]
	;nop
;----
	add	A1,S1(17)
	xor	eB1,eA1
			add	B2_2,ecx

			rol	B2_2,cl
	mov	ecx,eA1
			add	A2,S2(17)

	add	A1,B1_2
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_2

			rol	eB2,cl
	lea	ecx,[A1+B1_2]
			rol	A2,3

	add	eB1,A1
	add	B1_0,ecx
			add	eB2,A2

	rol	B1_0,cl
			lea	ecx,[A2+B2_2]
	;nop
;---------------------Round 3(18)+Encryption-------------------
	add	A1,S1(18)
	xor	eA1,eB1
			add	B2_0,ecx

			rol	B2_0,cl
	mov	ecx,eB1
			add	A2,S2(18)

	add	A1,B1_0
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_0

			rol	eA2,cl
	lea	ecx,[A1+B1_0]
			rol	A2,3

	add	eA1,A1
	add	B1_1,ecx
			add	eA2,A2

	rol	B1_1,cl
			lea	ecx,[A2+B2_0]
	;nop
;----
	add	A1,S1(19)
	xor	eB1,eA1
			add	B2_1,ecx

			rol	B2_1,cl
	mov	ecx,eA1
			add	A2,S2(19)

	add	A1,B1_1
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_1

			rol	eB2,cl
	lea	ecx,[A1+B1_1]
			rol	A2,3

	add	eB1,A1
	add	B1_2,ecx
			add	eB2,A2

	rol	B1_2,cl
			lea	ecx,[A2+B2_1]
	;nop
;---------------------Round 3(20)+Encryption-------------------
	add	A1,S1(20)
	xor	eA1,eB1
			add	B2_2,ecx

			rol	B2_2,cl
	mov	ecx,eB1
			add	A2,S2(20)

	add	A1,B1_2
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_2

			rol	eA2,cl
	lea	ecx,[A1+B1_2]
			rol	A2,3

	add	eA1,A1
	add	B1_0,ecx
			add	eA2,A2

	rol	B1_0,cl
			lea	ecx,[A2+B2_2]
	;nop
;----
	add	A1,S1(21)
	xor	eB1,eA1
			add	B2_0,ecx

			rol	B2_0,cl
	mov	ecx,eA1
			add	A2,S2(21)

	add	A1,B1_0
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_0

			rol	eB2,cl
	lea	ecx,[A1+B1_0]
			rol	A2,3

	add	eB1,A1
	add	B1_1,ecx
			add	eB2,A2

	rol	B1_1,cl
			lea	ecx,[A2+B2_0]
	;nop
;---------------------Round 3(22)+Encryption-------------------
	add	A1,S1(22)
	xor	eA1,eB1
			add	B2_1,ecx

			rol	B2_1,cl
	mov	ecx,eB1
			add	A2,S2(22)

	add	A1,B1_1
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_1

			rol	eA2,cl
	lea	ecx,[A1+B1_1]
			rol	A2,3

	add	eA1,A1
	add	B1_2,ecx
			add	eA2,A2

	rol	B1_2,cl
			lea	ecx,[A2+B2_1]
	;nop
;----
	add	A1,S1(23)
	xor	eB1,eA1
			add	B2_2,ecx

			rol	B2_2,cl
	mov	ecx,eA1
			add	A2,S2(23)

	add	A1,B1_2
	rol	eB1,cl
			xor	eB2,eA2

			mov	ecx,eA2
	rol	A1,3
			add	A2,B2_2

			rol	eB2,cl
	lea	ecx,[A1+B1_2]
			rol	A2,3

	add	eB1,A1
	add	B1_0,ecx
			add	eB2,A2

	rol	B1_0,cl
			lea	ecx,[A2+B2_2]
	;nop
;---------------------Round 3(24)+Encryption-------------------
	add	A1,S1(24)
	xor	eA1,eB1
			add	B2_0,ecx

			rol	B2_0,cl
	mov	ecx,eB1
			add	A2,S2(24)

	add	A1,B1_0
	rol	eA1,cl
			xor	eA2,eB2

			mov	ecx,eB2
	rol	A1,3
			add	A2,B2_0

			rol	eA2,cl
			rol	A2,3

	add	eA1,A1
			add	eA2,A2
;----------------------------------------------------
	cmp	eA1,[work_C_0]
	je	near _checkKey1High

CheckKey2:
	cmp	eA2,[work_C_0]
	je	near _checkKey2High

_NextKey:
	mov	ebx,[RC5_72_L0hi]
	dec	dword [work_iterations]
	jz	near finished_Found_nothing

	add	bl,2
	movzx	ebx,bl
	mov	[RC5_72_L0hi],ebx

	mov	A1,[work_pre4_r1]
	mov	ecx,[work_pre5_r1]

	jnc	key_setup_1_inner_loop

	mov	rax, [RC5_72UnitWork]
        mov     B1_1, [RC5_72UnitWork_L0mid]
        mov     B1_0, [RC5_72UnitWork_L0lo]
	
	bswap	B1_1
	bswap	B1_0

	adc	B1_1,0
	adc	B1_0,0

	bswap	B1_1
	bswap	B1_0
	mov	[RC5_72UnitWork_L0mid],B1_1
	mov	[RC5_72UnitWork_L0lo],B1_0
	jmp	key_setup_1_bigger_loop

align 16
_checkKey1High:
	lea	ecx,[A1+B1_0]
	add	B1_1,ecx
	rol	B1_1,cl
	add	A1,S1(25)
	xor	eB1,eA1
	mov	ecx,eA1
	add	A1,B1_1
	rol	eB1,cl
	rol	A1,3
	add	eB1,A1

	mov	rax, [RC5_72UnitWork]
	mov	B1_1, [RC5_72UnitWork_L0mid]
	mov	B1_0, [RC5_72UnitWork_L0lo]
	mov	B1_2, [RC5_72_L0hi]

        inc     dword [RC5_72UnitWork_CMCcount]
	cmp	eB1,[work_C_1]

        mov     [RC5_72UnitWork_CMChi], B1_2
        mov     [RC5_72UnitWork_CMCmid], B1_1
        mov     [RC5_72UnitWork_CMClo], B1_0

	jne	CheckKey2

	mov	ecx, [work_iterations]
	mov	rsi, [iterations_addr]

	shl	ecx, 1

	sub	[rsi], ecx
	mov	eax, RESULT_FOUND

	jmp	finished

_checkKey2High:
	lea	ecx,[A2+B2_0]
	add	B2_1,ecx
	rol	B2_1,cl
	add	A2,S2(25)
	xor	eB2,eA2
	mov	ecx,eA2
	add	A2,B2_1
	rol	eB2,cl
	rol	A2,3
	add	eB2,A2

	mov	rax, [RC5_72UnitWork]
	mov	B1_1, [RC5_72UnitWork_L0mid]
	mov	B1_0, [RC5_72UnitWork_L0lo]
	mov	B1_2, [RC5_72_L0hi]

        inc     dword [RC5_72UnitWork_CMCcount]
	inc	B1_2
	cmp	eB2,[work_C_1]

        mov     [RC5_72UnitWork_CMChi], B1_2
        mov     [RC5_72UnitWork_CMCmid], B1_1
        mov     [RC5_72UnitWork_CMClo], B1_0

	jne	_NextKey

	mov	ecx, [work_iterations]
	mov	rsi, [iterations_addr]

	shl	ecx, 1
	dec	ecx

	sub	[rsi], ecx
	mov	eax, RESULT_FOUND

	jmp	finished


finished_Found_nothing:
	mov	rax, [RC5_72UnitWork]
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

%ifdef _WINDOWS
        ;; Windows requires that rsi and rdi also be restored.
        mov     rsi, [save_rsi]
        mov     rdi, [save_rdi]
%endif
        ;; rbp, rbx, and r12 thru r15 must be restored
        mov     rbp, [save_rbp]
        mov     rbx, [save_rbx]
        mov     r12, [save_r12]
        mov     r13, [save_r13]
        mov     r14, [save_r14]
        mov     r15, [save_r15]


        add     rsp, work_size

        ret
	
