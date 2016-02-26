; Copyright distributed.net 1997-2003 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; AMD64 (x86_64) 2-pipe core by:
;     Steven Nikkel <snikkel@distributed.net>
;     Jeff Lawson <bovine@distributed.net>
;
; Based off the r72-ss2.asm core, by Ianos Gnatiuc <ssianky@hotmail.com>
; (r72-ss2 was based on r72-dg2 and r72ansi2 cores)
; $Id: r72-snjl2.asm,v 1.2 2007/10/22 16:48:35 jlawson Exp $

%error This code lacks at least Windows prologue and must be not used without modifications and testing

[SECTION .text]
BITS 64

[GLOBAL _rc5_72_unit_func_snjl2]
[GLOBAL rc5_72_unit_func_snjl2]

;------------------------------------------------

%define RESULT_NOTHING  1
%define RESULT_FOUND    2
%define PIPES           2

;; magic constants for the RC5 algorithm
%define P         0xB7E15163
%define Q         0x9E3779B9
%define S_not(N)  (P+Q*(N))
%define S0_ROL3   0xBF0A8B1D

;------------------------------------------------

%assign work_size 0

%macro defidef 2
    %define %1 rsp+%2
%endmacro

%macro defwork 1-2 1
    defidef %1,work_size
    %assign work_size work_size+4*(%2)
%endmacro

;; local storage variables
defwork work_S1,26		; array accessed via S1() macro.
defwork work_iter,1
defwork work_L0hi,1
defwork work_L0mid,1
defwork work_L0lo,1
defwork save_rbx,2
defwork save_rbp,2
defwork save_r12,2
defwork save_r13,2
defwork save_r14,2
defwork save_r15,2
defwork work_Clo,1
defwork work_Chi,1
defwork work_Plo,1
defwork work_Phi,1
defwork S1_ROL3,1
defwork S2_ROL3,1
;defwork L0_ROL,1
;defwork L1_ROL,1
defwork work_S2,26		; array accessed via S2() macro.
defwork S2_L1,1
defwork RC5_72UnitWork,2	; 1st argument (64-bit pointer), passed in rdi
defwork iterations,2		; 2nd argument (64-bit pointer), passed in rsi

;; macros to access into arrays.
%define S1(N)       [work_S1 + (N)*4] 
%define S2(N)       [work_S2 + (N)*4]
	
;; offsets within the parameter structure (rax contains base address).
%define RC5_72UnitWork_plainhi  rax +  0
%define RC5_72UnitWork_plainlo  rax +  4
%define RC5_72UnitWork_cipherhi rax +  8
%define RC5_72UnitWork_cipherlo rax + 12
%define RC5_72UnitWork_L0hi     rax + 16
%define RC5_72UnitWork_L0mid    rax + 20
%define RC5_72UnitWork_L0lo     rax + 24
%define RC5_72UnitWork_CMCcount rax + 28
%define RC5_72UnitWork_CMChi    rax + 32
%define RC5_72UnitWork_CMCmid   rax + 36
%define RC5_72UnitWork_CMClo    rax + 40

;; 32-bit values saved in the extended general-purpose registers.
%define L1_0    r8d 
%define L1_1    r9d
%define L1_2    r10d   
%define L2_0    r11d  
%define L2_1    r12d  
%define L2_2    r13d  
%define L0_ROL	r14d
%define L1_ROL	r15d

;------------------------------------------------

%macro KEYSETUP_BLOCK_CONSTANTS 2
;---
    add  esi, eax
    mov  ecx, esi
    lea  ebx, [ebx + edi + S_not(%1)]
    add esi, L1_%2

    rol  esi, cl
    rol  ebx, 3

    mov L1_%2,esi
    mov  S2(%1), ebx
;---
    add  edi, ebx

    mov  ecx, edi
    lea  eax, [eax + esi + S_not(%1+1)]
    add edi, L2_%2

    rol  edi, cl
    rol  eax, 3

    mov  L2_%2, edi
    mov  S1(%1+1), eax
;---
%endmacro

;------------------------------------------------

%macro KEYSETUP_BLOCK 2
    %if %1==25
        %assign next 0
    %else
        %assign next %1 + 1
    %endif
;---
    add  esi, eax
    add  ebx, edi
    add  eax, S1(next)

    mov  ecx, esi
    add  esi, L1_%2

    rol  esi, cl
    rol  ebx, 3

    mov  L1_%2, esi
    mov  S2(%1), ebx
;---
    add  edi, ebx
    add  eax, esi
    add  ebx, S2(next)

    mov  ecx, edi
    add edi, L2_%2

    rol  edi, cl
    rol  eax, 3

    mov  L2_%2, edi
    mov  S1(next), eax
;---
%endmacro

;------------------------------------------------

%macro ENCRYPTION_BLOCK 1
    mov  ecx, esi
    xor  eax, esi
    xor  ebx, edi

    rol  eax, cl
    mov  ecx, edi
    add  eax, S1(2*%1)

    rol  ebx, cl
    add  ebx, S2(2*%1)


    xor  esi, eax
    mov  ecx, eax
    xor  edi, ebx

    rol  esi, cl
    mov  ecx, ebx
    add  esi, S1(2*%1 + 1)

    rol  edi, cl
    add  edi, S2(2*%1 + 1)
%endmacro

;------------------------------------------------

align 16
startseg:
rc5_72_unit_func_snjl2:
_rc5_72_unit_func_snjl2:

    sub  rsp, work_size
    mov  [RC5_72UnitWork],rdi ; 1st argument is passed in rdi
    mov  [iterations],rsi ; 2nd argument is passed in rsi

    mov  rax, rdi	; rax points to RC5_72UnitWork

;; rbp, rbx, and r12 thru r15 must be preserved!
    mov  [save_rbp], rbp
    mov  [save_rbx], rbx
    mov  [save_r12], r12
    mov  [save_r13], r13
    mov  [save_r14], r14
    mov  [save_r15], r15   

    mov  ecx, [rsi]      	; iterations

    mov  ebx, [RC5_72UnitWork_plainhi]
    mov  edx, [RC5_72UnitWork_plainlo]
    mov  esi, [RC5_72UnitWork_cipherhi]
    mov  edi, [RC5_72UnitWork_cipherlo]

    mov  [work_Phi],  ebx
    mov  [work_Plo],  edx
    mov  [work_Chi],  esi
    mov  [work_Clo],  edi
    mov  [work_iter], ecx

;    mov  ebx, [RC5_72UnitWork_L0lo]
    mov  L1_0, [RC5_72UnitWork_L0lo]
;    mov  esi, [RC5_72UnitWork_L0mid]
    mov  L1_1, [RC5_72UnitWork_L0mid]
;    mov  edx, [RC5_72UnitWork_L0hi]
    mov  L1_2, [RC5_72UnitWork_L0hi]

;    bswap ebx
     bswap L1_0
;    bswap esi
     bswap L1_1

;    mov  [work_L0lo],  ebx
;    mov  [work_L0mid], esi
    mov  [work_L0lo], L1_0
    mov  [work_L0mid], L1_1


;;TODO:	edx,ebp are unused from this point on

LOOP_LO_SS_2:

    ; L[0] = rc5_72unitwork->L0.lo;
    ; S[0] = S0_ROL3
    ; L[0] = ROTL(L[0] + S[0], S[0]);

;    mov  eax, [work_L0lo]
    mov L1_0, [work_L0lo]
;    bswap eax
    bswap L1_0

;    add  eax, S0_ROL3
;    rol  eax, 0x1D
    add  L1_0, S0_ROL3
    rol  L1_0, 0x1D

;    mov  [L0_ROL], eax
    mov  L0_ROL, L1_0

    align 16
LOOP_MID_SS_2:

    ; L[1] = rc5_72unitwork->L0.mid;
    ; S[1] = ROTL3(S_not(1) + S[0] + L[0]);
    ; L[1] = ROTL(L[1] + S[1] + L[0], S[1] + L[0]);

;    mov  eax, [L0_ROL]
    mov  eax, L0_ROL
    mov  ebx, [work_L0mid]
;    mov  ecx, eax
    mov ecx, L0_ROL
    bswap ebx

    add  eax, S_not(1) + S0_ROL3
    rol  eax, 3

    add  ecx, eax
    mov  [S1_ROL3], eax

    add  ebx, ecx
    rol  ebx, cl
;    mov  [L1_ROL], ebx
    mov  L1_ROL, ebx

    ; S[2] = ROTL3(S_not(2) + S[1] + L[1])

    lea  eax, [eax + ebx + S_not(2)]
    rol  eax, 3

    ; ECX = S[2] + L[1]

    add  ebx, eax

    mov  [S2_ROL3], eax
    mov  [S2_L1], ebx

    align 16
LOOP_HI_SS_2:

    ; S[0] = S0_ROL3
    ; S[1] = S1_ROL3
    ; S[2] = S2_ROL3

    ; L[0] = L0_ROL
    ; L[1] = L1_ROL

    ; L1[2] = rc5_72unitwork->L0.hi
    ; L2[2] = rc5_72unitwork->L0.hi + 1

    ; ECX = S2_L1

    ; L1[2] = ROTL(L1[2] + ECX, CL)
    ; L2[2] = ROTL(L2[2] + ECX, CL)

    ; ECX - shiftreg
    ; EAX - S1  ESI - L1
    ; EBX - S2  EDI - L2
;    mov  esi, edx
    mov esi, L1_2
;    mov  eax, [L0_ROL]
    mov  eax, L0_ROL
;    lea  edi, [edx+1]
    lea  edi, [L1_2+1]
    mov  ecx, [S2_L1]
;    mov  ebx, [L1_ROL]
    mov  ebx, L1_ROL

    mov  L1_0, eax
    mov  L2_0, eax
    add  esi, ecx
    add  edi, ecx

    mov  L1_1, ebx
    mov  L2_1, ebx
    rol  esi, cl
    rol  edi, cl

    mov  ebx, [S2_ROL3]
    mov  ecx, [S1_ROL3]

    mov  L2_2, edi
    mov  L1_2, esi
    lea  eax, [ebx + esi + S_not(3)]
    mov  S1(2), ebx
    mov  S2(2), ebx
    rol  eax, 3
    mov  S1(1), ecx
    mov  S2(1), ecx
    mov  S1(3), eax

    KEYSETUP_BLOCK_CONSTANTS  3, 0
    KEYSETUP_BLOCK_CONSTANTS  4, 1
    KEYSETUP_BLOCK_CONSTANTS  5, 2
    KEYSETUP_BLOCK_CONSTANTS  6, 0
    KEYSETUP_BLOCK_CONSTANTS  7, 1
    KEYSETUP_BLOCK_CONSTANTS  8, 2
    KEYSETUP_BLOCK_CONSTANTS  9, 0
    KEYSETUP_BLOCK_CONSTANTS 10, 1
    KEYSETUP_BLOCK_CONSTANTS 11, 2
    KEYSETUP_BLOCK_CONSTANTS 12, 0
    KEYSETUP_BLOCK_CONSTANTS 13, 1
    KEYSETUP_BLOCK_CONSTANTS 14, 2
    KEYSETUP_BLOCK_CONSTANTS 15, 0
    KEYSETUP_BLOCK_CONSTANTS 16, 1
    KEYSETUP_BLOCK_CONSTANTS 17, 2
    KEYSETUP_BLOCK_CONSTANTS 18, 0
    KEYSETUP_BLOCK_CONSTANTS 19, 1
    KEYSETUP_BLOCK_CONSTANTS 20, 2
    KEYSETUP_BLOCK_CONSTANTS 21, 0
    KEYSETUP_BLOCK_CONSTANTS 22, 1
    KEYSETUP_BLOCK_CONSTANTS 23, 2
    KEYSETUP_BLOCK_CONSTANTS 24, 0

;   KEYSETUP_BLOCK_CONSTANTS 25, 1
;---
    add  esi, eax
    mov  ecx, esi
    lea  ebx, [ebx + edi + S_not(25)]
    add  esi, L1_1

    rol  esi, cl
    rol  ebx, 3

    mov  L1_1, esi
    mov  S2(25), ebx
;---
    add  edi, ebx
    lea  eax, [eax + esi + S0_ROL3]

    mov  ecx, edi
    add  edi, L2_1

    rol  edi, cl
    rol  eax, 3

    add  ebx, S0_ROL3
    mov  L2_1, edi
    mov  S1(0), eax
;---

    KEYSETUP_BLOCK  0, 2
    KEYSETUP_BLOCK  1, 0
    KEYSETUP_BLOCK  2, 1
    KEYSETUP_BLOCK  3, 2
    KEYSETUP_BLOCK  4, 0
    KEYSETUP_BLOCK  5, 1
    KEYSETUP_BLOCK  6, 2
    KEYSETUP_BLOCK  7, 0
    KEYSETUP_BLOCK  8, 1
    KEYSETUP_BLOCK  9, 2
    KEYSETUP_BLOCK 10, 0
    KEYSETUP_BLOCK 11, 1
    KEYSETUP_BLOCK 12, 2
    KEYSETUP_BLOCK 13, 0
    KEYSETUP_BLOCK 14, 1
    KEYSETUP_BLOCK 15, 2
    KEYSETUP_BLOCK 16, 0
    KEYSETUP_BLOCK 17, 1
    KEYSETUP_BLOCK 18, 2
    KEYSETUP_BLOCK 19, 0
    KEYSETUP_BLOCK 20, 1
    KEYSETUP_BLOCK 21, 2
    KEYSETUP_BLOCK 22, 0
    KEYSETUP_BLOCK 23, 1
    KEYSETUP_BLOCK 24, 2
    KEYSETUP_BLOCK 25, 0

    KEYSETUP_BLOCK  0, 1
    KEYSETUP_BLOCK  1, 2
    KEYSETUP_BLOCK  2, 0
    KEYSETUP_BLOCK  3, 1
    KEYSETUP_BLOCK  4, 2
    KEYSETUP_BLOCK  5, 0
    KEYSETUP_BLOCK  6, 1
    KEYSETUP_BLOCK  7, 2
    KEYSETUP_BLOCK  8, 0
    KEYSETUP_BLOCK  9, 1
    KEYSETUP_BLOCK 10, 2
    KEYSETUP_BLOCK 11, 0
    KEYSETUP_BLOCK 12, 1
    KEYSETUP_BLOCK 13, 2
    KEYSETUP_BLOCK 14, 0
    KEYSETUP_BLOCK 15, 1
    KEYSETUP_BLOCK 16, 2
    KEYSETUP_BLOCK 17, 0
    KEYSETUP_BLOCK 18, 1
    KEYSETUP_BLOCK 19, 2
    KEYSETUP_BLOCK 20, 0
    KEYSETUP_BLOCK 21, 1
    KEYSETUP_BLOCK 22, 2
    KEYSETUP_BLOCK 23, 0
    KEYSETUP_BLOCK 24, 1

;   KEYSETUP_BLOCK 25, 2
;---

    mov  eax, [work_Plo]
    add  ebx, edi

    mov  esi, [work_Phi]
    rol  ebx, 3

    mov  edi, esi
    mov  S2(25), ebx

    mov  ebx, eax

    add  eax, S1(0)
    add  esi, S1(1)

    add  ebx, S2(0)
    add  edi, S2(1)

    ENCRYPTION_BLOCK 1
    ENCRYPTION_BLOCK 2
    ENCRYPTION_BLOCK 3
    ENCRYPTION_BLOCK 4
    ENCRYPTION_BLOCK 5
    ENCRYPTION_BLOCK 6
    ENCRYPTION_BLOCK 7
    ENCRYPTION_BLOCK 8
    ENCRYPTION_BLOCK 9
    ENCRYPTION_BLOCK 10
    ENCRYPTION_BLOCK 11
    ENCRYPTION_BLOCK 12


TEST_KEY_1_SS_2:
;---
    cmp  eax, [work_Clo]
    jne  short TEST_KEY_2_SS_2

    mov  rax, [RC5_72UnitWork]

    mov  ecx, [work_L0mid]
    inc  dword [RC5_72UnitWork_CMCcount]
    bswap ecx
    mov  [RC5_72UnitWork_CMCmid], ecx

    mov  ecx, [work_L0lo]
;    mov  [RC5_72UnitWork_CMChi], edx
    mov  [RC5_72UnitWork_CMChi], L1_2
    bswap ecx
    mov  [RC5_72UnitWork_CMClo], ecx
;---
    cmp  esi, [work_Chi]
    jne  short TEST_KEY_2_SS_2

    mov  rcx, [iterations]
    mov  ebx, [work_iter]
    mov  esi, RESULT_FOUND
    sub  [rcx], ebx

    jmp  LOOP_EXIT_SS_2


    align 16
TEST_KEY_2_SS_2:
;---
    cmp  ebx, [work_Clo]
    jne  INC_KEY_HI_SS_2

    mov  rax, [RC5_72UnitWork]
    inc  dword [RC5_72UnitWork_CMCcount]

;    inc  edx
    inc L1_2
    mov  ebx, [work_L0mid]
    mov  ecx, [work_L0lo]
    bswap ebx
    bswap ecx
;    mov  [RC5_72UnitWork_CMChi], edx
    mov  [RC5_72UnitWork_CMChi], L1_2
    mov  [RC5_72UnitWork_CMCmid], ebx
    mov  [RC5_72UnitWork_CMClo], ecx
;    dec edx
    dec L1_2
;---
    cmp  edi, [work_Chi]
    jne  short INC_KEY_HI_SS_2

    mov  ebx, [work_iter]
    mov  rcx, [iterations]
    dec  ebx
    mov  esi, RESULT_FOUND
    sub  [rcx], ebx

    jmp  short LOOP_EXIT_SS_2


    align 16
INC_KEY_HI_SS_2:

    add  dl, BYTE PIPES
    jc   INC_KEY_MID_SS_2

    sub  dword [work_iter], BYTE PIPES
    jnz  LOOP_HI_SS_2

    mov  esi, RESULT_NOTHING
    jmp  short LOOP_EXIT_SS_2


INC_KEY_MID_SS_2:

    adc  dword [work_L0mid], BYTE 0
    jc   INC_KEY_LO_SS_2

    sub  dword [work_iter], BYTE PIPES
    jnz  LOOP_MID_SS_2

    mov  esi, RESULT_NOTHING
    jmp  short LOOP_EXIT_SS_2


INC_KEY_LO_SS_2:

    adc  dword [work_L0lo], BYTE 0
;!  jc   LOOP_EXIT_NOTHING_SS_2

    sub  dword [work_iter], BYTE PIPES
    jnz  LOOP_LO_SS_2

    mov  esi, RESULT_NOTHING


LOOP_EXIT_SS_2:

    mov  rax, [RC5_72UnitWork]

    mov  ebx, [work_L0mid]
    mov  ecx, [work_L0lo]

    bswap ebx
    bswap ecx

;    mov  [RC5_72UnitWork_L0hi],  edx
    mov  [RC5_72UnitWork_L0hi], L1_2
    mov  [RC5_72UnitWork_L0mid], ebx
    mov  [RC5_72UnitWork_L0lo],  ecx

    mov  eax, esi

;; rbp, rbx, and r12 thru r15 must be restored
    mov  rbp, [save_rbp]
    mov  rbx, [save_rbx]
    mov  r12, [save_r12]
    mov  r13, [save_r13]
    mov  r14, [save_r14]
    mov  r15, [save_r15]

    add  rsp, work_size

    ret
