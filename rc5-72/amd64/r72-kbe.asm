; Copyright distributed.net 1997-2004 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; AMD64 (x86_64) 3-pipe core by:
;     Didier Levet <kakace@distributed.net>
;     Jeff Lawson <bovine@distributed.net>
;     Steven Nikkel <snikkel@distributed.net>
;
; $Id: r72-kbe.asm,v 1.1.2.1 2005/06/05 19:31:15 snikkel Exp $

[SECTION .text]
BITS 64

[GLOBAL _rc5_72_unit_func_kbe]
[GLOBAL rc5_72_unit_func_kbe]

%define P         0xB7E15163
%define Q         0x9E3779B9
%define _S(N)     (P+Q*(N))

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

;; local storage variables
defwork work_L1,3
defwork work_L2,3
defwork work_L3,3
defwork work_S1,52
defwork work_S2,52
defwork work_S3,52
defwork work_cached_L1_0,1
defwork work_cached_L2_0,1
defwork work_cached_L3_0,1
defwork work_cached_L1_1,1
defwork work_cached_L2_1,1
defwork work_cached_L3_1,1
defwork work_P_0
defwork work_P_1
defwork work_C_0
defwork work_C_1
defwork work_L0lo,3
defwork work_L0mid,3
defwork work_L0hi,3
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
defwork iterations,2		; 2nd argument (64-bit pointer), passed in rsi

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

;; convenience aliases
%define S1(N)                   [work_S1+((N)*4)]
%define S2(N)                   [work_S2+((N)*4)]
%define S3(N)                   [work_S3+((N)*4)]
%define L1(N)                   [work_L1+((N)*4)]
%define L2(N)                   [work_L2+((N)*4)]
%define L3(N)                   [work_L3+((N)*4)]
%define cached_L1_0             [work_cached_L1_0]
%define cached_L2_0             [work_cached_L2_0]
%define cached_L3_0             [work_cached_L3_0]
%define cached_L1_1             [work_cached_L1_1]
%define cached_L2_1             [work_cached_L2_1]
%define cached_L3_1             [work_cached_L3_1]
%define L0lo(N)                 [work_L0lo+((N)*4)]
%define L0mid(N)                [work_L0mid+((N)*4)]
%define L0hi(N)                 [work_L0hi+((N)*4)]
%define CACHED_S0               0xBF0A8B1D            ; = ROTL3(P)

%define temp       eax 
%define temp2      ebx
%define a1         edx
%define a2         esi
%define a3         edi
%define l1         ebp
%define shiftreg   ecx
%define shiftcount cl
%define l2         r8d 
%define l3         r9d
%define c1         r10d   
%define c2         r11d  
%define c3         r12d  
%define d1         r13d  
%define d2         r14d  
%define d3         r15d  

;; aliases
%define l1_0    d1
%define l2_0    d2
%define l3_0    d3
%define l1_1    l1
%define l2_1    l2
%define l3_1    l3
%define l1_2    c1
%define l2_2    c2
%define l3_2    c3


; KEYSETUP_1  S, L, B
%macro KEYSETUP_1 3
    add   l1_%2, a1
    lea   shiftreg, [a1 + l1_%3]
    mov   S2(%1-1), a2

    add   l1_%2, l1_%3
    add   l2_%2, a2
    rol   a3, 3

    rol   l1_%2, shiftcount
    lea   shiftreg, [a2 + l2_%3]
    mov   S3(%1-1), a3

    add   a1, l1_%2
    add   l2_%2, l2_%3
    add   l3_%2, a3

    rol   l2_%2, shiftcount
    lea   shiftreg, [a3 + l3_%3]
    add   a1, _S(%1)

    add   l3_%2, l3_%3
    add   a2, _S(%1)
    rol   a1, 3

    rol   l3_%2, shiftcount
    add   a2, l2_%2
    add   a3, _S(%1)

    rol   a2, 3
    add   a3, l3_%2
    mov   S1(%1), a1
%endmacro


; KEYSETUP_2  S, L, B
%macro KEYSETUP_2 3
    add   l1_%2, a1
    mov   shiftreg, a1
    add   a1, S1(%1-26)

    add   l1_%2, l1_%3
    add   shiftreg, l1_%3
    mov   S2(%1-1), a2

    rol   l1_%2, shiftcount
    mov   shiftreg, a2
    add   a2, S2(%1-26)

    rol   a3, 3
    add   shiftreg, l2_%3
    mov   temp, S3(%1-26)

    add   l2_%2, shiftreg
    add   l3_%2, a3
    mov   S3(%1-1), a3

    rol   l2_%2, shiftcount
    mov   shiftreg, a3
    add   l3_%2, l3_%3

    add   shiftreg, l3_%3
    add   a1, l1_%2
    add   a2, l2_%2

    rol   l3_%2, shiftcount
    add   a3, temp
    rol   a1, 3

    rol   a2, 3
    add   a3, l3_%2
    mov   S1(%1), a1
%endmacro


; ENCRYPT_C  S, L
%macro ENCRYPT_C 2
    lea    temp2, [a1 + l1]
    rol    d2, shiftcount
    mov    l1, L1(%2)

    add    a1, S1(%1)
    mov    shiftreg, c3
    xor    d3, c3

    add    d2, a2
    mov    temp, L2(%2)
    rol    d3, shiftcount

    add    l2, a2
    add    a2, S2(%1)
    rol    a3, 3

    add    l1, temp2
    mov    shiftreg, temp2
    mov    temp2, L3(%2)

    rol    l1, shiftcount
    mov    shiftreg, l2
    add    d3, a3

    add    a1, l1
    add    l3, a3
    mov    L1(%2), l1

    add    l2, temp
    rol    a1, 3
    add    a3, S3(%1)

    xor    c1, d1
    rol    l2, shiftcount
    mov    shiftreg, l3

    add    l3, temp2
    add    a2, l2
    mov    L2(%2), l2

    rol    l3, shiftcount
    mov    shiftreg, d1
    rol    a2, 3

    rol    c1, shiftcount
    mov    shiftreg, d2
    mov    L3(%2), l3

    xor    c2, d2
    add    c1, a1
    add    a3, l3
%endmacro


; ENCRYPT_D  S, L
%macro ENCRYPT_D 2
    lea    temp2, [a1 + l1]
    rol    c2, shiftcount
    mov    l1, L1(%2)

    add    a1, S1(%1)
    mov    shiftreg, d3
    xor    c3, d3

    add    c2, a2
    mov    temp, L2(%2)
    rol    c3, shiftcount

    add    l2, a2
    add    a2, S2(%1)
    rol    a3, 3

    add    l1, temp2
    mov    shiftreg, temp2
    mov    temp2, L3(%2)

    rol    l1, shiftcount
    mov    shiftreg, l2
    add    c3, a3

    add    a1, l1
    add    l3, a3
    mov    L1(%2), l1

    add    l2, temp
    rol    a1, 3
    add    a3, S3(%1)

    xor    d1, c1
    rol    l2, shiftcount
    mov    shiftreg, l3

    add    l3, temp2
    add    a2, l2
    mov    L2(%2), l2

    rol    l3, shiftcount
    mov    shiftreg, c1
    rol    a2, 3

    rol    d1, shiftcount
    mov    shiftreg, c2
    mov    L3(%2), l3

    xor    d2, c2
    add    d1, a1
    add    a3, l3
%endmacro


; CHECK_HI pipe,S,label
%macro CHECK_HI 3
    cmp     c%1, [work_C_0]
    jne     short %3

    mov   shiftreg, c%1
    xor   d%1, c%1
    rol   d%1, shiftcount
    lea   shiftreg, [a%1 + l%1]
    mov   l%1, L%1(1)
    add   a%1, S%1(%2)
    add   l%1, shiftreg
    rol   l%1, shiftcount
    add   a%1, l%1
    rol   a%1, 3
    add   d%1, a%1

    mov   a%1, L0lo(%1 - 1)
    mov   l%1, L0mid(%1 - 1)
    mov   c%1, L0hi(%1 - 1)
    inc   dword [RC5_72UnitWork_CMCcount]
    mov   [RC5_72UnitWork_CMClo], a%1
    mov   [RC5_72UnitWork_CMCmid], l%1
    mov   [RC5_72UnitWork_CMChi], c%1

    cmp   d%1, [work_C_1]
    jne   short %3

    mov   ecx, 1-%1
    jmp   finished_found
%endmacro


align 16
startseg:
rc5_72_unit_func_kbe:
rc5_72_unit_func_kbe_:
_rc5_72_unit_func_kbe:

        sub     rsp, work_size

%ifdef _WINDOWS
        mov     [RC5_72UnitWork],rcx ; 1st argument is passed in rcx
        mov     [iterations],rdx ; 2nd argument is passwd in rdx
        mov     rax, rcx	; rax points to RC5_72UnitWork

        ;; Windows requires that rsi and rdi also be preserved by callee!
        mov     [save_rsi], rsi
        mov     [save_rdi], rdi
        mov     rsi,rdx         ; rsi points to iterations
%else
        ;; Linux, FreeBSD, and other UNIX platforms
        mov     [RC5_72UnitWork],rdi ; 1st argument is passed in rdi
        mov     [iterations],rsi ; 2nd argument is passwd in rsi
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
        mov     edx, [RC5_72UnitWork_L0hi]

        mov     [work_C_0], ebx
        mov     [work_C_1], ecx
        mov     edi, [rsi]	; rsi points to iterations

        imul    edi, 2863311531
        mov     ecx, [RC5_72UnitWork_L0mid]
        mov     ebx, [RC5_72UnitWork_L0lo]

        mov     [work_iterations], edi

  ;; NOTE - L0hi is always an even multiple of 4.
        mov     L0hi(0), edx
        mov     L0mid(0), ecx
        mov     L0lo(0), ebx

        add     dl, 1               ; ++L0hi (key #2)
        mov     L0hi(1), edx
        mov     L0mid(1), ecx
        mov     L0lo(1), ebx

        add     dl, 1               ; ++L0hi (key #3)
        mov     L0hi(2), edx
        mov     L0mid(2), ecx
        mov     L0lo(2), ebx

        mov     a1, CACHED_S0
        mov     S1(0), a1
        mov     S2(0), a1
        mov     S3(0), a1

align 16
outer_loop:
    mov    l1_0, L0lo(0)
    mov    l2_0, L0lo(1)
    mov    l3_0, L0lo(2)
    add    l1_0, CACHED_S0
    add    l2_0, CACHED_S0
    add    l3_0, CACHED_S0
    rol    l1_0, 0x1D
    rol    l2_0, 0x1D
    rol    l3_0, 0x1D
    mov    cached_L1_0, l1_0
    mov    cached_L2_0, l2_0
    mov    cached_L3_0, l3_0

    mov    a1, P+Q+CACHED_S0
    mov    a2, a1
    mov    a3, a1
    add    a1, l1_0
    add    a2, l2_0
    add    a3, l3_0
    rol    a1, 3
    rol    a2, 3
    rol    a3, 3
    mov    S1(1), a1
    mov    S2(1), a2
    mov    S3(1), a3

    mov    l1_1, L0mid(0)
    mov    l2_1, L0mid(1)
    mov    l3_1, L0mid(2)
    mov    shiftreg, a1
    add    l1_1, a1
    mov    l1_2, L0hi(0)
    add    shiftreg, l1_0
    add    l1_1, l1_0
    mov    l2_2, L0hi(1)
    add    l2_1, a2
    add    l3_1, a3
    mov    l3_2, L0hi(2)
    rol    l1_1, shiftcount
    mov    shiftreg, a2
    add    l2_1, l2_0
    add    l3_1, l3_0
    add    shiftreg, l2_0
    add    a1, l1_1
    rol    l2_1, shiftcount
    mov    shiftreg, a3
    add    a1, _S(2)
    add    shiftreg, l3_0
    rol    a1, 3
    add    a2, _S(2)
    rol    l3_1, shiftcount
    add    a2, l2_1
    add    a3, _S(2)
    rol    a2, 3
    add    a3, l3_1
    mov    S1(2), a1
    rol    a3, 3
    mov    S2(2), a2
    mov    S3(2), a3
    mov    cached_L1_1, l1_1
    mov    cached_L2_1, l2_1
    mov    cached_L3_1, l3_1

align 16
inner_loop:
    ; KEYSETUP_1(3,2,1)  // L[2], S[3]
    lea    shiftreg, [a1 + l1_1]
    add    l1_2, a1
    mov    l1_0, cached_L1_0

    add    l2_2, a2
    mov    l2_0, cached_L2_0
    add    l1_2, l1_1

    add    l3_2, a3
    mov    l3_0, cached_L3_0
    add    l2_2, l2_1

    rol    l1_2, shiftcount
    lea    shiftreg, [a2 + l2_1]
    add    a1, _S(3)

    add    l3_2, l3_1
    add    a1, l1_2
    add    a2, _S(3)

    rol    l2_2, shiftcount
    lea    shiftreg, [a3 + l3_1]
    rol    a1, 3

    rol    l3_2, shiftcount
    add    a2, l2_2
    add    a3, _S(3)

    rol    a2, 3
    add    a3, l3_2
    mov    S1(3), a1

    ;          S,L,B
    KEYSETUP_1 4,0,2
    KEYSETUP_1 5,1,0
    KEYSETUP_1 6,2,1
    KEYSETUP_1 7,0,2
    KEYSETUP_1 8,1,0
    KEYSETUP_1 9,2,1
    KEYSETUP_1 10,0,2
    KEYSETUP_1 11,1,0
    KEYSETUP_1 12,2,1
    KEYSETUP_1 13,0,2
    KEYSETUP_1 14,1,0
    KEYSETUP_1 15,2,1
    KEYSETUP_1 16,0,2
    KEYSETUP_1 17,1,0
    KEYSETUP_1 18,2,1
    KEYSETUP_1 19,0,2
    KEYSETUP_1 20,1,0
    KEYSETUP_1 21,2,1
    KEYSETUP_1 22,0,2
    KEYSETUP_1 23,1,0
    KEYSETUP_1 24,2,1
    KEYSETUP_1 25,0,2

    KEYSETUP_2 26,1,0
    KEYSETUP_2 27,2,1
    KEYSETUP_2 28,0,2
    KEYSETUP_2 29,1,0
    KEYSETUP_2 30,2,1
    KEYSETUP_2 31,0,2
    KEYSETUP_2 32,1,0
    KEYSETUP_2 33,2,1
    KEYSETUP_2 34,0,2
    KEYSETUP_2 35,1,0
    KEYSETUP_2 36,2,1
    KEYSETUP_2 37,0,2
    KEYSETUP_2 38,1,0
    KEYSETUP_2 39,2,1
    KEYSETUP_2 40,0,2
    KEYSETUP_2 41,1,0
    KEYSETUP_2 42,2,1
    KEYSETUP_2 43,0,2
    KEYSETUP_2 44,1,0
    KEYSETUP_2 45,2,1
    KEYSETUP_2 46,0,2
    KEYSETUP_2 47,1,0
    KEYSETUP_2 48,2,1
    KEYSETUP_2 49,0,2
    KEYSETUP_2 50,1,0

    ; KEYSETUP_2(51,2,1)   // L[2], S[25]
    add    l1_2, a1
    mov    shiftreg, a1
    add    a1, S1(51-26)

    add    l1_2, l1_1
    add    shiftreg, l1_1
    mov    S2(51-1), a2

    rol    l1_2, shiftcount
    mov    shiftreg, a2
    add    a2, S2(51-26)

    rol    a3, 3
    add    shiftreg, l2_1
    mov    temp, S3(51-26)

    add    l2_2, shiftreg
    add    l3_2, a3
    mov    S3(51-1), a3

    rol    l2_2, shiftcount
    mov    shiftreg, a3
    mov    L1(2), l1_2

    add    l3_2, l3_1
    add    shiftreg, l3_1
    mov    L2(2), l2_2

    rol    l3_2, shiftcount
    add    a1, l1_2
    add    a2, l2_2

    add    a3, temp
    rol    a1, 3
    mov    L3(2), l3_2

    rol    a2, 3
    add    a3, l3_2
    mov    S1(51), a1

    ; KEYSETUP_2(26,0,2)   // L[0], S[0]
    add    l1_0, a1
    mov    shiftreg, a1
    add    a1, S1(26)

    add    l1_0, l1_2
    add    shiftreg, l1_2
    mov    S2(26-1), a2

    rol    l1_0, shiftcount
    mov    shiftreg, a2
    add    a2, S2(26)

    rol    a3, 3
    add    shiftreg, l2_2
    mov    temp, S3(26)

    add    l2_0, shiftreg
    add    l3_0, a3
    mov    S3(26-1), a3

    rol    l2_0, shiftcount
    mov    shiftreg, a3
    mov    L1(0), l1_0

    add    l3_0, l3_2
    add    shiftreg, l3_2
    mov    L2(0), l2_0

    rol    l3_0, shiftcount
    add    a1, l1_0
    mov    c3, [work_P_0]   ; c3 alias l3_2

    add    a2, l2_0
    add    a3, temp
    mov    L3(0), l3_0

    rol    a1, 3
    rol    a2, 3
    add    a3, l3_0

    add    l1_1, a1
    lea    shiftreg, [a1 + l1_0]
    rol    a3, 3

    ; KEYSETUP_2(27,1,0)   // L[1], S[1]
    add    l1_1, l1_0
    lea    c1, [c3 + a1]
    add    a1, S1(27)

    rol    l1_1, shiftcount
    mov    shiftreg, a2
    lea    c2, [c3 + a2]

    add    c3, a3
    add    shiftreg, l2_0
    add    a2, S2(27)

    add    l2_1, shiftreg
    add    l3_1, a3
    mov    temp, S3(27)

    rol    l2_1, shiftcount
    mov    shiftreg, a3
    mov    L1(1), l1_1

    add    l3_1, l3_0
    add    shiftreg, l3_0
    mov    L2(1), l2_1

    rol    l3_1, shiftcount
    add    a1, l1_1
    mov    d3, [work_P_1]   ; d3 alias l3_0

    add    a2, l2_1
    rol    a1, 3
    mov    L3(1), l3_1

    rol    a2, 3
    add    a3, temp
    mov    S1(27), a1

    ; ENCRYPT_C(28,2)
    mov    d1,a1
    lea    temp2, [a1 + l1]
    mov    l1, L1(2)

    add    a1, S1(28)
    add    a3, l3_1
    mov    d2,d3

    add    d1,d3
    mov    temp, L2(2)
    add    d2,a2

    add    l2, a2
    rol    a3, 3
    add    a2, S2(28)

    add    l1, temp2
    mov    shiftreg, temp2
    mov    temp2, L3(2)

    rol    l1, shiftcount
    mov    shiftreg, l2
    add    d3,a3

    add    a1, l1
    add    l3, a3
    mov    L1(2), l1

    add    l2, temp
    rol    a1, 3
    add    a3, S3(28)

    xor    c1, d1
    rol    l2, shiftcount
    mov    shiftreg, l3

    add    l3, temp2
    add    a2, l2
    mov    L2(2), l2

    rol    l3, shiftcount
    mov    shiftreg, d1
    rol    a2, 3

    rol    c1, shiftcount
    mov    shiftreg, d2
    mov    L3(2), l3

    xor    c2, d2
    add    c1, a1
    add    a3, l3

    ENCRYPT_D 29,0
    ENCRYPT_C 30,1
    ENCRYPT_D 31,2
    ENCRYPT_C 32,0
    ENCRYPT_D 33,1
    ENCRYPT_C 34,2
    ENCRYPT_D 35,0
    ENCRYPT_C 36,1
    ENCRYPT_D 37,2
    ENCRYPT_C 38,0
    ENCRYPT_D 39,1
    ENCRYPT_C 40,2
    ENCRYPT_D 41,0
    ENCRYPT_C 42,1
    ENCRYPT_D 43,2
    ENCRYPT_C 44,0
    ENCRYPT_D 45,1
    ENCRYPT_C 46,2
    ENCRYPT_D 47,0
    ENCRYPT_C 48,1
    ENCRYPT_D 49,2
    ENCRYPT_C 50,0

    rol    c2, shiftcount
    mov    shiftreg, d3
    xor    c3, d3

    rol    a3, 3
    rol    c3, shiftcount
    mov    ebx, L0hi(0)          ; prefetch key#1 (hi)

    add    c2, a2
    add    c3, a3
    mov    rax, [RC5_72UnitWork] ; 64-bit pointer
    ;; *bx and *ax are aliased by temp and temp2, respectively. None of them
    ;; is modified in the CHECK_HI macro.

test_key_1:
    CHECK_HI 1,51,test_key_2

align 16
test_key_2:
    CHECK_HI 2,25,test_key_3

align 16
test_key_3:
    CHECK_HI 3,25,inc_key

align 16
inc_key:
    cmp   bl, 0xFB                ; key#1 (hi)
    lea   l1_2, [ebx + 3]
    lea   l2_2, [ebx + 4]
    lea   l3_2, [ebx + 5]

    mov   a1, S1(2)
    mov   a2, S2(2)
    mov   a3, S3(2)
    jae   complex_incr

    mov   l1_1, cached_L1_1
    mov   l2_1, cached_L2_1
    mov   l3_1, cached_L3_1

    dec   dword [work_iterations]

    mov   L0hi(0), l1_2
    mov   L0hi(1), l2_2
    mov   L0hi(2), l3_2
    mov   [RC5_72UnitWork_L0hi], l1_2

    jnz   inner_loop

    xor   eax, eax
    jmp   finished

align 16
complex_incr:
    mov     edx, L0hi(0)
    mov     ecx, L0mid(0)
    mov     ebx, L0lo(0)

    add     dl, 3
    bswap   ecx
    bswap   ebx
    adc     ecx, BYTE 0
    adc     ebx, BYTE 0
    bswap   ecx
    bswap   ebx

    mov     L0hi(0), edx
    mov     L0mid(0), ecx
    mov     L0lo(0), ebx
    mov     [RC5_72UnitWork_L0hi], edx
    mov     [RC5_72UnitWork_L0mid], ecx
    mov     [RC5_72UnitWork_L0lo], ebx

    add     dl, 1
    bswap   ecx
    bswap   ebx
    adc     ecx, BYTE 0
    adc     ebx, BYTE 0
    bswap   ecx
    bswap   ebx

    mov     L0hi(1), edx
    mov     L0mid(1), ecx
    mov     L0lo(1), ebx

    add     dl, 1
    bswap   ecx
    bswap   ebx
    adc     ecx, BYTE 0
    adc     ebx, BYTE 0
    dec     dword [work_iterations]
    bswap   ecx
    bswap   ebx

    mov     L0hi(2), edx
    mov     L0mid(2), ecx
    mov     L0lo(2), ebx

    jnz     outer_loop

    xor     eax, eax
    jmp     short finished

finished_found:
        mov     rsi, [iterations] ; 64-bit pointer
        add     ecx, [work_iterations]
        add     ecx, [work_iterations]
        add     ecx, [work_iterations]
        sub     [rsi], ecx

        xor     eax, eax
        inc     eax
finished:
        inc     eax

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
