; Copyright distributed.net 1997-2004 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; AMD64 (x86_64) 3-pipe core by:
;     Steven Nikkel <snikkel@distributed.net>
;     Jeff Lawson <bovine@distributed.net>
;
; Based off of the r72-dg3.asm core by Décio Luiz Gazzoni Filho.
; $Id: r72-snjl.asm,v 1.1.2.7 2004/06/27 20:32:40 jlawson Exp $

[SECTION .text]
BITS 64

[GLOBAL _rc5_72_unit_func_snjl]
[GLOBAL rc5_72_unit_func_snjl]

%define P         0xB7E15163
%define Q         0x9E3779B9
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

;; local storage variables
defwork work_L3,1
defwork work_s1,26
defwork work_s2,26
defwork work_s3,26
defwork work_backup_L1,3
defwork work_backup_L2,3
defwork work_backup_L3,3
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
%define S1(N)                   [work_s1+((N)*4)]
%define S2(N)                   [work_s2+((N)*4)]
%define S3(N)                   [work_s3+((N)*4)]
%define L3_2			[work_L3]
%define L1backup(N)             [work_backup_L1+((N)*4)]
%define L2backup(N)             [work_backup_L2+((N)*4)]
%define L3backup(N)             [work_backup_L3+((N)*4)]
%define A1         eax 
%define A2         ebx
%define A3         edx
%define B1         esi
%define B2         edi
%define B3         ebp
%define shiftreg   ecx
%define shiftcount cl
%define L1_0    r8d 
%define L1_1    r9d
%define L1_2    r10d   
%define L2_0    r11d  
%define L2_1    r12d  
%define L2_2    r13d  
%define L3_0    r14d  
%define L3_1    r15d  


%macro KEYSETUP_BLOCK 3
        rol     A1, 3
        rol     A2, 3
        rol     A3, 3

        lea     shiftreg, [A1 + B1]
        add     B1, L1_%3
        mov     S1(%2), A1

        add     B1, A1
        rol     B1, shiftcount
%ifidn %1,S
        add     A1, S1(%2+1)
%else
        lea     A1, [A1 + B1 + S_not(%2+1)]
%endif

        add     B2, A2
        mov     shiftreg, B2
        add     B2, L2_%3

        add     B3, A3
        mov     S2(%2), A2
        mov     S3(%2), A3

        rol     B2, shiftcount
%ifidn %1,S
        add     A2, S2(%2+1)
        add     A1, B1
%else
        lea     A2, [A2 + B2 + S_not(%2+1)]
%endif

        mov     L1_%3, B1
        mov     shiftreg, B3
        add     B3, L3_%3

        rol     B3, shiftcount
%ifidn %1,S
        add     A3, S3(%2+1)
        add     A2, B2
%else
        lea     A3, [A3 + B3 + S_not(%2+1)]
%endif

        mov     L2_%3, B2
        mov     L3_%3, B3
%ifidn %1,S
        add     A3, B3
%endif
%endmacro

%macro ENCRYPTION_BLOCK 1
        mov     shiftreg, B1
        xor     A1, B1
        xor     A2, B2

        rol     A1, shiftcount
        mov     shiftreg, B2
        add     A1, S1(2*%1)

        xor     A3, B3
        rol     A2, shiftcount
        mov     shiftreg, B3

        add     A2, S2(2*%1)
        rol     A3, shiftcount
        mov     shiftreg, A1

        add     A3, S3(2*%1)
        xor     B1, A1
        xor     B2, A2

        rol     B1, shiftcount
        mov     shiftreg, A2
        add     B1, S1(2*%1+1)

        xor     B3, A3
        rol     B2, shiftcount
        mov     shiftreg, A3

        add     B2, S2(2*%1+1)
        rol     B3, shiftcount

        add     B3, S3(2*%1+1)
%endmacro


align 16
startseg:
rc5_72_unit_func_snjl:
rc5_72_unit_func_snjl_:
_rc5_72_unit_func_snjl:

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
        mov     L1_2, edx
        mov     L1_1, ecx 
        mov     L1_0, ebx

        mov     L1backup(2), edx
        mov     L1backup(1), ecx
        mov     L1backup(0), ebx

        add     dl, 1
        bswap   ecx
        bswap   ebx

        adc     ecx, BYTE 0
        adc     ebx, BYTE 0

        bswap   ecx
        bswap   ebx

        mov     L2_2, edx
        mov     L2_1, ecx
        mov     L2_0, ebx

        mov     L2backup(2), edx
        mov     L2backup(1), ecx
        mov     L2backup(0), ebx

        add     dl, 1
        bswap   ecx
        bswap   ebx

        adc     ecx, BYTE 0
        adc     ebx, BYTE 0

        bswap   ecx
        bswap   ebx

        mov     L3_2, edx
        mov     L3_1, ecx
        mov     L3_0, ebx

        mov     L3backup(2), edx
        mov     L3backup(1), ecx
        mov     L3backup(0), ebx


align 16
key_setup_1:
        mov     B1, L1_0
        mov     B2, L2_0
        mov     B3, L3_0

        mov     A1, 0xBF0A8B1D ; 0xBF0A8B1D is S[0]
        mov     A2, A1
        mov     A3, A1

        mov     S1(0), A1
        mov     S2(0), A1
        mov     S3(0), A1

        add     B1, A1
        add     B2, A1
        add     B3, A1

        rol     B1, 0x1D       ; 0x1D are least significant bits of S[0]
        rol     B2, 0x1D
        rol     B3, 0x1D

        lea     A1, [A1 + B1 + S_not(1)]
        lea     A2, [A2 + B2 + S_not(1)]
        lea     A3, [A3 + B3 + S_not(1)]

        mov     L1_0, B1
        mov     L2_0, B2
        mov     L3_0, B3

        KEYSETUP_BLOCK S_not,1,1
        KEYSETUP_BLOCK S_not,2,2
        KEYSETUP_BLOCK S_not,3,0
        KEYSETUP_BLOCK S_not,4,1
        KEYSETUP_BLOCK S_not,5,2
        KEYSETUP_BLOCK S_not,6,0
        KEYSETUP_BLOCK S_not,7,1
        KEYSETUP_BLOCK S_not,8,2
        KEYSETUP_BLOCK S_not,9,0
        KEYSETUP_BLOCK S_not,10,1
        KEYSETUP_BLOCK S_not,11,2
        KEYSETUP_BLOCK S_not,12,0
        KEYSETUP_BLOCK S_not,13,1
        KEYSETUP_BLOCK S_not,14,2
        KEYSETUP_BLOCK S_not,15,0
        KEYSETUP_BLOCK S_not,16,1
        KEYSETUP_BLOCK S_not,17,2
        KEYSETUP_BLOCK S_not,18,0
        KEYSETUP_BLOCK S_not,19,1
        KEYSETUP_BLOCK S_not,20,2
        KEYSETUP_BLOCK S_not,21,0
        KEYSETUP_BLOCK S_not,22,1
        KEYSETUP_BLOCK S_not,23,2
        KEYSETUP_BLOCK S_not,24,0

        rol     A1, 3
        rol     A2, 3
        rol     A3, 3

        lea     shiftreg, [A1 + B1]

        add     B1, L1_1
        mov     S1(25), A1

        mov     S2(25), A2
        mov     S3(25), A3
        add     B1, A1

        add     A1, S1(0)
        add     B2, A2
        add     B3, A3

        rol     B1, shiftcount
        mov     shiftreg, B2
        add     B2, L2_1

        add     A2, S2(0)
        add     A1, B1
        mov     L1_1, B1

        rol     B2, shiftcount
        mov     shiftreg, B3
        add     B3, L3_1

        add     A3, S3(0)
        add     A2, B2
        mov     L2_1, B2

        rol     B3, shiftcount

        mov     L3_1, B3
        add     A3, B3

key_setup_2:

        KEYSETUP_BLOCK S,0,2
        KEYSETUP_BLOCK S,1,0
        KEYSETUP_BLOCK S,2,1
        KEYSETUP_BLOCK S,3,2
        KEYSETUP_BLOCK S,4,0
        KEYSETUP_BLOCK S,5,1
        KEYSETUP_BLOCK S,6,2
        KEYSETUP_BLOCK S,7,0
        KEYSETUP_BLOCK S,8,1
        KEYSETUP_BLOCK S,9,2
        KEYSETUP_BLOCK S,10,0
        KEYSETUP_BLOCK S,11,1
        KEYSETUP_BLOCK S,12,2
        KEYSETUP_BLOCK S,13,0
        KEYSETUP_BLOCK S,14,1
        KEYSETUP_BLOCK S,15,2
        KEYSETUP_BLOCK S,16,0
        KEYSETUP_BLOCK S,17,1
        KEYSETUP_BLOCK S,18,2
        KEYSETUP_BLOCK S,19,0
        KEYSETUP_BLOCK S,20,1
        KEYSETUP_BLOCK S,21,2
        KEYSETUP_BLOCK S,22,0
        KEYSETUP_BLOCK S,23,1
        KEYSETUP_BLOCK S,24,2

        rol     A1, 3
        rol     A2, 3
        rol     A3, 3

        lea     shiftreg, [A1 + B1]
        add     B1, L1_0
        mov     S1(25), A1

        mov     S2(25), A2
        mov     S3(25), A3
        add     B1, A1

        add     A1, S1(0)
        add     B2, A2
        add     B3, A3

        rol     B1, shiftcount
        mov     shiftreg, B2
        add     B2, L2_0

        add     A2, S2(0)
        add     A1, B1
        mov     L1_0, B1

        rol     B2, shiftcount
        mov     shiftreg, B3
        add     B3, L3_0

        add     A3, S3(0)
        add     A2, B2
        mov     L2_0, B2

        rol     B3, shiftcount

        mov     L3_0, B3
        add     A3, B3

key_setup_3:

        KEYSETUP_BLOCK S,0,1
        KEYSETUP_BLOCK S,1,2
        KEYSETUP_BLOCK S,2,0
        KEYSETUP_BLOCK S,3,1
        KEYSETUP_BLOCK S,4,2
        KEYSETUP_BLOCK S,5,0
        KEYSETUP_BLOCK S,6,1
        KEYSETUP_BLOCK S,7,2
        KEYSETUP_BLOCK S,8,0
        KEYSETUP_BLOCK S,9,1
        KEYSETUP_BLOCK S,10,2
        KEYSETUP_BLOCK S,11,0
        KEYSETUP_BLOCK S,12,1
        KEYSETUP_BLOCK S,13,2
        KEYSETUP_BLOCK S,14,0
        KEYSETUP_BLOCK S,15,1
        KEYSETUP_BLOCK S,16,2
        KEYSETUP_BLOCK S,17,0
        KEYSETUP_BLOCK S,18,1
        KEYSETUP_BLOCK S,19,2
        KEYSETUP_BLOCK S,20,0
        KEYSETUP_BLOCK S,21,1
        KEYSETUP_BLOCK S,22,2
        KEYSETUP_BLOCK S,23,0
        KEYSETUP_BLOCK S,24,1

        rol     A1, 3
        rol     A2, 3
        rol     A3, 3

        mov     S1(25), A1
        mov     S2(25), A2
        mov     S3(25), A3


;    A1 = rc5_72unitwork->plain.lo + S1[0];
;    A2 = rc5_72unitwork->plain.lo + S2[0];
;    B1 = rc5_72unitwork->plain.hi + S1[1];
;    B2 = rc5_72unitwork->plain.hi + S2[1];

encryption:

        mov     A1, [work_P_0]
        mov     B1, [work_P_1]
        mov     A2, A1

        mov     A3, A1
        mov     B2, B1
        mov     B3, B1

        add     A1, S1(0)
        add     A2, S2(0)
        add     A3, S3(0)

        add     B1, S1(1)
        add     B2, S2(1)
        add     B3, S3(1)

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

test_key_1:
        cmp     A1, [work_C_0]
        mov     rax, [RC5_72UnitWork] ; 64-bit pointer

        jne     short test_key_2

        inc     dword [RC5_72UnitWork_CMCcount]

        mov     ecx, L1backup(2)
        mov     [RC5_72UnitWork_CMChi], ecx

        mov     ecx, L1backup(1)
        mov     [RC5_72UnitWork_CMCmid], ecx

        mov     ecx, L1backup(0)
        mov     [RC5_72UnitWork_CMClo], ecx

        cmp     B1, [work_C_1]
        jne     short test_key_2

        xor     ecx, ecx

        jmp     finished_found

align 16
test_key_2:
        cmp     A2, [work_C_0]

        jne     short test_key_3

        mov     esi, L2backup(2)
        mov     ecx, L2backup(1)
        mov     ebx, L2backup(0)

        inc     dword [RC5_72UnitWork_CMCcount]

        mov     [RC5_72UnitWork_CMChi], esi
        mov     [RC5_72UnitWork_CMCmid], ecx
        mov     [RC5_72UnitWork_CMClo], ebx

        cmp     B2, [work_C_1]
        jne     short test_key_3

        xor     ecx, ecx
        dec     ecx

        jmp     finished_found

align 16
test_key_3:
        cmp     A3, [work_C_0]
        mov     edx, [RC5_72UnitWork_L0hi]

        jne     short inc_key

        mov     esi, L3backup(2)
        mov     ecx, L3backup(1)
        mov     ebx, L3backup(0)

        inc     dword [RC5_72UnitWork_CMCcount]

        mov     [RC5_72UnitWork_CMChi], esi
        mov     [RC5_72UnitWork_CMCmid], ecx
        mov     [RC5_72UnitWork_CMClo], ebx

        cmp     B3, [work_C_1]

        jne     short inc_key

        mov     ecx, -2

        jmp     finished_found


align 16
inc_key:
        cmp     dl, 0xFD
        mov     ecx, [RC5_72UnitWork_L0mid]
        mov     ebx, [RC5_72UnitWork_L0lo]

        jae     complex_incr

        add     dl, 3

        mov     [RC5_72UnitWork_L0hi], edx
        mov     L1_2, edx
        mov     L1backup(2), edx
        inc     edx

        mov     L2_2, edx
        mov     L2backup(2), edx
        inc     edx

        mov     L3_2, edx
        mov     L3backup(2), edx
        dec     dword [work_iterations]

        mov     L1_1, ecx
        mov     L2_1, ecx
        mov     L3_1, ecx

        mov     L1_0, ebx
        mov     L2_0, ebx
        mov     L3_0, ebx

        jnz     key_setup_1

        xor     eax, eax
        jmp     finished

align 16
complex_incr:
        add     dl, 3
        bswap   ecx
        bswap   ebx

        adc     ecx, BYTE 0
        adc     ebx, BYTE 0

        bswap   ecx
        bswap   ebx

        mov     L1_2, edx
        mov     L1_1, ecx
        mov     L1_0, ebx

        mov     L1backup(2), edx
        mov     L1backup(1), ecx
        mov     L1backup(0), ebx

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

        mov     L2_2, edx
        mov     L2_1, ecx
        mov     L2_0, ebx

        mov     L2backup(2), edx
        mov     L2backup(1), ecx
        mov     L2backup(0), ebx

        add     dl, 1
        bswap   ecx
        bswap   ebx

        adc     ecx, BYTE 0
        adc     ebx, BYTE 0
        dec     dword [work_iterations]

        bswap   ecx
        bswap   ebx

        mov     L3_2, edx
        mov     L3_1, ecx
        mov     L3_0, ebx

        mov     L3backup(2), edx
        mov     L3backup(1), ecx
        mov     L3backup(0), ebx

        jnz     key_setup_1

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
