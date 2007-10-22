; Copyright distributed.net 1997-2002 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; Author: Décio Luiz Gazzoni Filho <acidblood@distributed.net>
; $Id: r72-dgp4.asm,v 1.7 2007/10/22 16:48:36 jlawson Exp $

%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_72_unit_func_dg_p4]
[GLOBAL rc5_72_unit_func_dg_p4]

%define P         0xB7E15163
%define Q         0x9E3779B9
%define S_not(N)  (P+Q*(N))

%define RESULT_NOTHING 1
%define RESULT_FOUND   2

%assign work_size 0

%macro defidef 2
    %define %1 esp+%2
%endmacro

%macro defwork 1-2 1
    defidef %1,work_size
    %assign work_size work_size+4*(%2)
%endmacro

defwork save_esp
defwork save_ebp
defwork save_ebx
defwork save_esi
defwork save_edi
defwork work_iterations
defwork work_P_0
defwork work_P_1
defwork work_C_0
defwork work_C_1
defwork work_L1,3
defwork work_L2,3
defwork work_s,52

%define RC5_72UnitWork_plainhi  eax+0
%define RC5_72UnitWork_plainlo  eax+4
%define RC5_72UnitWork_cipherhi eax+8
%define RC5_72UnitWork_cipherlo eax+12
%define RC5_72UnitWork_L0hi     eax+16
%define RC5_72UnitWork_L0mid    eax+20
%define RC5_72UnitWork_L0lo     eax+24
%define RC5_72UnitWork_CMCcount eax+28
%define RC5_72UnitWork_CMChi    eax+32
%define RC5_72UnitWork_CMCmid   eax+36
%define RC5_72UnitWork_CMClo    eax+40

%define RC5_72UnitWork          esp+work_size+4
%define iterations              esp+work_size+8

%define S1(N)                   [work_s+((N)*8)+0]
%define S2(N)                   [work_s+((N)*8)+4]
%define L1(N)                   [work_L1+((N)*4)]
%define L2(N)                   [work_L2+((N)*4)]


; register allocation for the key setup blocks
%define A1         eax
%define A2         edx
%define B1         ebx
%define B2         ebp
%define L1_0       esi
%define L2_0       edi
%define shiftreg   ecx
%define shiftcount cl

; register allocation for the encryption block is mostly the same
%define S1_next    esi
%define S2_next    edi

;       L1[j]   S2[i]
;       S1[i+1] L2[j]

%macro KEYSETUP_BLOCK_CONSTANTS_j0 1
        lea     shiftreg, [A1 + B1]
        add     B1, A1
        add     A2, S_not(%1)

        add     B1, L1_0
        add     A2, B2

        rol     B1, shiftcount
        rol     A2, 3

        lea     shiftreg, [A2 + B2]
        add     B2, A2
        add     A1, S_not(%1+1)

        mov     L1_0, B1
        add     B2, L2_0
        add     A1, B1

        mov     S2(%1), A2
        rol     B2, shiftcount
        rol     A1, 3

        mov     L2_0, B2
        mov     S1(%1+1), A1
%endmacro

%macro KEYSETUP_BLOCK_CONSTANTS_j1 1
        lea     shiftreg, [A1 + B1]
        add     B1, A1
        add     A2, S_not(%1)

        add     B1, L1(1)
        add     A2, B2

        rol     B1, shiftcount
        rol     A2, 3

        lea     shiftreg, [A2 + B2]
        add     B2, A2
        add     A1, S_not(%1+1)

        mov     L1(1), B1
        add     B2, L2(1)
        add     A1, B1

        mov     S2(%1), A2
        rol     B2, shiftcount
        rol     A1, 3

        mov     L2(1), B2
        mov     S1(%1+1), A1
%endmacro

%macro KEYSETUP_BLOCK_CONSTANTS_j2 1
        lea     shiftreg, [A1 + B1]
        add     B1, A1
        add     A2, S_not(%1)

        add     B1, L1(2)
        add     A2, B2

        rol     B1, shiftcount
        rol     A2, 3

        lea     shiftreg, [A2 + B2]
        add     B2, A2
        add     A1, S_not(%1+1)

        mov     L1(2), B1
        add     B2, L2(2)
        add     A1, B1

        mov     S2(%1), A2
        rol     B2, shiftcount
        rol     A1, 3

        mov     L2(2), B2
        mov     S1(%1+1), A1
%endmacro

;       L1[j]   S2[i]
;       S1[i+1] L2[j]

%macro KEYSETUP_BLOCK_j0 1
        add     B1, A1
        add     A2, B2
        add     A1, S1(%1+1)

        mov     shiftreg, B1
        add     B1, L1_0
        rol     A2, 3

        rol     B1, shiftcount
        mov     S2(%1), A2
        add     B2, A2

        add     A1, B1
        add     A2, S2(%1+1)
        mov     shiftreg, B2

        add     B2, L2_0
        rol     A1, 3

        rol     B2, shiftcount
        mov     S1(%1+1), A1

        mov     L1_0, B1
        mov     L2_0, B2
%endmacro

%macro KEYSETUP_BLOCK_j1 1
        add     B1, A1
        add     A2, B2
        add     A1, S1(%1+1)

        mov     shiftreg, B1
        add     B1, L1(1)
        rol     A2, 3

        rol     B1, shiftcount
        mov     S2(%1), A2
        add     B2, A2

        add     A1, B1
        add     A2, S2(%1+1)
        mov     shiftreg, B2

        add     B2, L2(1)
        rol     A1, 3

        rol     B2, shiftcount
        mov     S1(%1+1), A1

        mov     L1(1), B1
        mov     L2(1), B2
%endmacro

%macro KEYSETUP_BLOCK_j2 1
        add     B1, A1
        add     A2, B2
        add     A1, S1(%1+1)

        mov     shiftreg, B1
        add     B1, L1(2)
        rol     A2, 3

        rol     B1, shiftcount
        mov     S2(%1), A2
        add     B2, A2

        add     A1, B1
        add     A2, S2(%1+1)
        mov     shiftreg, B2

        add     B2, L2(2)
        rol     A1, 3

        rol     B2, shiftcount
        mov     S1(%1+1), A1

        mov     L1(2), B1
        mov     L2(2), B2
%endmacro

%macro ENCRYPTION_BLOCK 1
        mov     shiftreg, B1
        xor     A1, B1
        xor     A2, B2

        rol     A1, shiftcount
        mov     shiftreg, B2

        add     A1, S1(2*%1)
        rol     A2, shiftcount

        add     A2, S2(2*%1)
        mov     shiftreg, A1
        xor     B1, A1

        rol     B1, shiftcount
        mov     shiftreg, A2
        xor     B2, A2

        add     B1, S1(2*%1+1)
        rol     B2, shiftcount

        add     B2, S2(2*%1+1)
%endmacro


align 16
startseg:
rc5_72_unit_func_dg_p4:
_rc5_72_unit_func_dg_p4:

;	mov     ecx, esp
;      page 91, opt manual 64-byte align esp
;      
;      mov     [save_esp], esp

        sub     esp, work_size

;	move the eax and edx loads above the stack mangling

        mov     eax, [RC5_72UnitWork]

        mov     [save_ebp], ebp
        mov     [save_ebx], ebx
        mov     [save_esi], esi

        mov     [save_edi], edi
        mov     ebx, [RC5_72UnitWork_plainlo]
        mov     ecx, [RC5_72UnitWork_plainhi]

        mov     esi, [RC5_72UnitWork_cipherlo]
        mov     edi, [RC5_72UnitWork_cipherhi]
        mov     edx, [iterations]

        mov     [work_P_0], ebx
        mov     [work_P_1], ecx
        mov     ebx, [RC5_72UnitWork_L0hi]

        mov     [work_C_0], esi
        mov     [work_C_1], edi
        mov     ebp, [edx]

        shr     ebp, 1
        mov     ecx, [RC5_72UnitWork_L0mid]
        mov     esi, [RC5_72UnitWork_L0lo]

        mov     [work_iterations], ebp
        mov     L1(2), ebx

        inc     ebx

        mov     L2(2), ebx

        mov     L1(1), ecx
        mov     L2(1), ecx

        mov     L1(0), esi
        mov     L2(0), esi

key_setup_1:
        mov     B1, L1(0)
        mov     A1, 0xBF0A8B1D ; 0xBF0A8B1D is S[0]
        mov     B2, L2(0)

;       S1[0]

        add     B1, A1
        mov     A2, A1
        mov     S1(0), A1

        rol     B1, 0x1D       ; 0x1D are least significant bits of S[0]
        mov     S2(0), A2

;       L1[0]   S2[0]

        mov     L1_0, B1
        add     A1, S_not(1)
        add     B2, A2

        add     A1, B1
        rol     B2, 0x1D

        mov     L2_0, B2
        rol     A1, 3

        mov     S1(1), A1

;       S1[1]   L2[0]

;       L1[1]   S2[1]
;       S1[2]   L2[1]
;       ...
        KEYSETUP_BLOCK_CONSTANTS_j1 1
        KEYSETUP_BLOCK_CONSTANTS_j2 2
        KEYSETUP_BLOCK_CONSTANTS_j0 3
        KEYSETUP_BLOCK_CONSTANTS_j1 4
        KEYSETUP_BLOCK_CONSTANTS_j2 5
        KEYSETUP_BLOCK_CONSTANTS_j0 6
        KEYSETUP_BLOCK_CONSTANTS_j1 7
        KEYSETUP_BLOCK_CONSTANTS_j2 8
        KEYSETUP_BLOCK_CONSTANTS_j0 9
        KEYSETUP_BLOCK_CONSTANTS_j1 10
        KEYSETUP_BLOCK_CONSTANTS_j2 11
        KEYSETUP_BLOCK_CONSTANTS_j0 12
        KEYSETUP_BLOCK_CONSTANTS_j1 13
        KEYSETUP_BLOCK_CONSTANTS_j2 14
        KEYSETUP_BLOCK_CONSTANTS_j0 15
        KEYSETUP_BLOCK_CONSTANTS_j1 16
        KEYSETUP_BLOCK_CONSTANTS_j2 17
        KEYSETUP_BLOCK_CONSTANTS_j0 18
        KEYSETUP_BLOCK_CONSTANTS_j1 19
        KEYSETUP_BLOCK_CONSTANTS_j2 20
        KEYSETUP_BLOCK_CONSTANTS_j0 21
        KEYSETUP_BLOCK_CONSTANTS_j1 22
        KEYSETUP_BLOCK_CONSTANTS_j2 23
        KEYSETUP_BLOCK_CONSTANTS_j0 24
;       ...
;       L1[0]  S2[24]
;       S1[25] L2[0]

;       L1[1]  S2[25]
;       S1[0]  L2[1]

key_setup_2:
        lea     shiftreg, [A1 + B1]
        add     B1, A1
        add     A2, S_not(25)

        add     B1, L1(1)
        add     A2, B2

        rol     B1, shiftcount
        rol     A2, 3

        mov     L1(1), B1
        mov     S2(25), A2

        add     B2, A2
        add     A1, 0xBF0A8B1D
        add     A2, S2(0)

        mov     shiftreg, B2
        add     B2, L2(1)
        add     A1, B1

        rol     B2, shiftcount
        rol     A1, 3

        mov     L2(1), B2
        mov     S1(0), A1

;       L1[2]   S2[0]
;       S1[1]   L2[2]
;       ...
        KEYSETUP_BLOCK_j2 0
        KEYSETUP_BLOCK_j0 1
        KEYSETUP_BLOCK_j1 2
        KEYSETUP_BLOCK_j2 3
        KEYSETUP_BLOCK_j0 4
        KEYSETUP_BLOCK_j1 5
        KEYSETUP_BLOCK_j2 6
        KEYSETUP_BLOCK_j0 7
        KEYSETUP_BLOCK_j1 8
        KEYSETUP_BLOCK_j2 9
        KEYSETUP_BLOCK_j0 10
        KEYSETUP_BLOCK_j1 11
        KEYSETUP_BLOCK_j2 12
        KEYSETUP_BLOCK_j0 13
        KEYSETUP_BLOCK_j1 14
        KEYSETUP_BLOCK_j2 15
        KEYSETUP_BLOCK_j0 16
        KEYSETUP_BLOCK_j1 17
        KEYSETUP_BLOCK_j2 18
        KEYSETUP_BLOCK_j0 19
        KEYSETUP_BLOCK_j1 20
        KEYSETUP_BLOCK_j2 21
        KEYSETUP_BLOCK_j0 22
        KEYSETUP_BLOCK_j1 23
        KEYSETUP_BLOCK_j2 24
;       ...
;       L1[2]   S2[24]
;       S1[25]  L1[2]

;       L1[0]   S2[25]
;       S1[0]   L2[0]

key_setup_3:
        add     B1, A1
        add     A2, B2
        add     A1, S1(0)

        mov     shiftreg, B1
        add     B1, L1_0
        rol     A2, 3

        rol     B1, shiftcount
        mov     S2(25), A2
        add     B2, A2

        add     A1, B1
        add     A2, S2(0)
        mov     shiftreg, B2

        add     B2, L2_0
        rol     A1, 3

        rol     B2, shiftcount
        mov     S1(0), A1

        mov     L1_0, B1
        mov     L2_0, B2

;       L1[1]   S2[0]
;       S1[1]   L2[1]
;       ...
        KEYSETUP_BLOCK_j1 0
        KEYSETUP_BLOCK_j2 1
        KEYSETUP_BLOCK_j0 2
        KEYSETUP_BLOCK_j1 3
        KEYSETUP_BLOCK_j2 4
        KEYSETUP_BLOCK_j0 5
        KEYSETUP_BLOCK_j1 6
        KEYSETUP_BLOCK_j2 7
        KEYSETUP_BLOCK_j0 8
        KEYSETUP_BLOCK_j1 9
        KEYSETUP_BLOCK_j2 10
        KEYSETUP_BLOCK_j0 11
        KEYSETUP_BLOCK_j1 12
        KEYSETUP_BLOCK_j2 13
        KEYSETUP_BLOCK_j0 14
        KEYSETUP_BLOCK_j1 15
        KEYSETUP_BLOCK_j2 16
        KEYSETUP_BLOCK_j0 17
        KEYSETUP_BLOCK_j1 18
        KEYSETUP_BLOCK_j2 19
        KEYSETUP_BLOCK_j0 20
        KEYSETUP_BLOCK_j1 21
        KEYSETUP_BLOCK_j2 22
        KEYSETUP_BLOCK_j0 23
        KEYSETUP_BLOCK_j1 24
;       ...
;       L[1]    S2[24]
;       S1[25]  L2[1]

;               S2[25]

        mov     A1, [work_P_0]
        add     A2, B2

        rol     A2, 3

        mov     S2(25), A2

;    A1 = rc5_72unitwork->plain.lo + S1[0];
;    A2 = rc5_72unitwork->plain.lo + S2[0];
;    B1 = rc5_72unitwork->plain.hi + S1[1];
;    B2 = rc5_72unitwork->plain.hi + S2[1];

encryption:
        add     B2, A2
        mov     A2, [work_P_0]

        mov     shiftreg, B2
        add     B2, L1(2)
        mov     B1, [work_P_1]

        rol     B2, shiftcount
        add     A1, S1(0)

        mov     L2(2), B2
        mov     B2, [work_P_1]
        add     A2, S2(0)

        add     B1, S1(1)
        add     B2, S2(1)

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


;    if (A1 == rc5_72unitwork->cypher.lo)
;    {
;      ++rc5_72unitwork->check.count;
;      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi;
;      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
;      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
;      if (B1 == rc5_72unitwork->cypher.hi)
;      {
;        *iterations -= (kiter + 1)*2;
;        return RESULT_FOUND;
;      }
;    }

test_key_1_0:
        cmp     A1, [work_C_0]
        mov     eax, [RC5_72UnitWork]
        je	near test_key_1_1
        
test_key_2_0:        
        cmp     A2, [work_C_0]
        mov     ebx, [RC5_72UnitWork_L0hi]
        mov     ecx, [RC5_72UnitWork_L0mid]
        mov     edx, [RC5_72UnitWork_L0lo]
        je	near test_key_2_1

inc_key:
        bswap   ecx		; 1
        bswap   edx		; 2
        movzx	esi, bl		; 1

        and	ebx, 0xffffff00	; 1
        mov	ebp, 1		; 2
        add	esi, 2		; 2

        xor	edi, edi	; 3
        and	esi, 0xff	; 3

        cmove	edi, ebp	; 4
        or	ebx, esi	; 3

        xor	esi, esi	; 4
        mov     [RC5_72UnitWork_L0hi], ebx	; 4, 5
        add	ecx, edi	; 8

        bswap   ecx		; 9
        cmovc	esi, ebp	; 8

        add     edx, esi	; 9
        bswap   edx		; 10
        mov     L1(2), ebx	; 7, 7

        add     ebx, 1		; 8
        sub     dword [work_iterations], 1	; 8, 8
        mov     L2(2), ebx	; 9, 9

        mov     L1(1), ecx	; 16, 16
        mov     L2(1), ecx	; 16, 17
        mov     L1(0), edx	; 17, 17

        mov     L2(0), edx	; 18, 18
        mov     [RC5_72UnitWork_L0mid], ecx
        mov     [RC5_72UnitWork_L0lo], edx

        jnz     key_setup_1

        mov     eax, RESULT_NOTHING
        jmp	restore_stuff
        
test_key_1_1:
        add     dword [RC5_72UnitWork_CMCcount], 1

        mov     ecx, [RC5_72UnitWork_L0hi]
        mov     esi, [RC5_72UnitWork_L0mid]
        mov     edi, [RC5_72UnitWork_L0lo]

        cmp     B1, [work_C_1]
        mov     [RC5_72UnitWork_CMChi], ecx
        mov     [RC5_72UnitWork_CMCmid], esi
        mov     [RC5_72UnitWork_CMClo], edi
        jne     test_key_2_0

        mov     ecx, [work_iterations]
        mov     esi, [iterations]
        mov     edi, [esi]
        add     ecx, ecx
        sub     edi, ecx
        mov     [esi], edi
        mov     eax, RESULT_FOUND
        
        jmp	restore_stuff

test_key_2_1:
        add     dword [RC5_72UnitWork_CMCcount], 1

        cmp     B2, [work_C_1]
        mov     [RC5_72UnitWork_CMChi], ebx
        mov     [RC5_72UnitWork_CMCmid], ecx
        mov     [RC5_72UnitWork_CMClo], edx
        jne     inc_key

        mov     ecx, [work_iterations]
        mov     esi, [iterations]
        mov     edi, [esi]
        lea		ecx, [ecx+ecx-1]
        sub     edi, ecx
        mov     [esi], edi
        mov     eax, RESULT_FOUND

restore_stuff:        
        mov     ebp, [save_ebp]
        mov     ebx, [save_ebx]
        mov     esi, [save_esi]
        mov     edi, [save_edi]
        add     esp, work_size

        ret
