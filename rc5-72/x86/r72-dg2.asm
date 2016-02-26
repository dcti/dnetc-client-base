; Copyright distributed.net 1997-2003 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; Author: Décio Luiz Gazzoni Filho <acidblood@distributed.net>
; $Id: r72-dg2.asm,v 1.17 2009/01/05 22:45:21 mfeiri Exp $

%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_72_unit_func_dg_2]
[GLOBAL rc5_72_unit_func_dg_2]

%define P         0xB7E15163
%define Q         0x9E3779B9
%define S_not(N)  ((P+Q*(N)) & 0xFFFFFFFF)

%define RESULT_NOTHING 1
%define RESULT_FOUND   2

%assign work_size 0

%macro defidef 2
    %define %1 esp+%2
%endmacro

%macro fedifed 2
    %define %1 ebp+%2-232
%endmacro

%macro defwork 1-2 1
    defidef %1,work_size
    fedifed %1_ebp,work_size
    %assign work_size work_size+4*(%2)
%endmacro

defwork work_L1,3
defwork work_s1,26
defwork work_L2,3
defwork work_s2,26
defwork work_P_0
defwork work_P_1
defwork work_C_0
defwork work_C_1
defwork work_iterations
defwork save_ebx
defwork save_esi
defwork save_edi
defwork save_ebp

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

%define RC5_72UnitWork          ebp+(work_size-232)+4
%define iterations              ebp+(work_size-232)+8

%define S1(N)                   [work_s1+((N)*4)]
%define S2(N)                   [ebp-26*4+((N)*4)]
%define L1(N)                   [work_L1+((N)*4)]
%define L2(N)                   [work_L2+((N)*4)]

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
        %endif
    %endif
    %%alend:
%endmacro
%else
    %define k7align align
%endif

; register allocation for the key setup blocks
%define A1         eax
%define A2         edx
%define B1         ebx
%define B2         esi
%define shiftreg   ecx
%define shiftcount cl

;       L1[j]   S2[i]
;       S1[i+1] L2[j]

%macro KEYSETUP_BLOCK_CONSTANTS 2
        add     B1, A1
        lea     A2, [A2 + B2 + S_not(%1)]

        mov     shiftreg, B1
        add     B1, L1(%2)

        rol     B1, shiftcount
        rol     A2, 3

        mov     L1(%2), B1
        mov     S2(%1), A2


        add     B2, A2
        lea     A1, [A1 + B1 + S_not(%1+1)]

        mov     shiftreg, B2
        add     B2, L2(%2)

        rol     B2, shiftcount
        rol     A1, 3

        mov     L2(%2), B2
        mov     S1(%1+1), A1
%endmacro

;       L1[j]   S2[i]
;       S1[i+1] L2[j]

%macro KEYSETUP_BLOCK 2
        add     B1, A1
        add     A2, B2
        add     A1, S1(%1+1)

        mov     shiftreg, B1
        add     B1, L1(%2)
        rol     A2, 3

        rol     B1, shiftcount
        mov     S2(%1), A2

        add     B2, A2
        add     A1, B1
        add     A2, S2(%1+1)

        mov     shiftreg, B2
        add     B2, L2(%2)
        rol     A1, 3

        rol     B2, shiftcount
        mov     S1(%1+1), A1

        mov     L1(%2), B1
        mov     L2(%2), B2
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
rc5_72_unit_func_dg_2:
_rc5_72_unit_func_dg_2:

	mov	[esp-4], ebp
	lea	ebp, [esp-work_size+232]
        sub     esp, work_size

        mov     eax, [RC5_72UnitWork]
        mov     [save_ebx_ebp], ebx
        mov     [save_esi_ebp], esi

        mov     [save_edi_ebp], edi
        mov     esi, [RC5_72UnitWork_plainlo]
        mov     edi, [RC5_72UnitWork_plainhi]

        mov     ebx, [RC5_72UnitWork_cipherlo]
        mov     ecx, [RC5_72UnitWork_cipherhi]
        mov     edx, [iterations]

        mov     [work_P_0_ebp], esi
        mov     [work_P_1_ebp], edi
        mov     esi, [RC5_72UnitWork_L0hi]

        mov     [work_C_0_ebp], ebx
        mov     [work_C_1_ebp], ecx
        mov     edi, [edx]

        shr     edi, 1
        mov     ecx, [RC5_72UnitWork_L0mid]
        mov     ebx, [RC5_72UnitWork_L0lo]

        mov     [work_iterations_ebp], edi
        mov     L1(2), esi

        inc     esi

        mov     L2(2), esi

        mov     L1(1), ecx
        mov     L2(1), ecx

        mov     L1(0), ebx
        mov     L2(0), ebx

k7align 16
key_setup_1:
        mov     A2, 0xBF0A8B1D ; 0xBF0A8B1D is S[0]
        add     B1, 0xBF0A8B1D

;       S1[0]

        rol     B1, 0x1D       ; 0x1D are least significant bits of S[0]
        mov     S1(0), A2
        mov     S2(0), A2

        mov     B2, B1
        lea     A1, [A2 + B1 + S_not(1)]

;       L1[0]   S2[0]

        rol     A1, 3
        mov     L1(0), B1

        mov     L2(0), B2
        mov     S1(1), A1

;       S1[1]   L2[0]

;       L1[1]   S2[1]
;       S1[2]   L2[1]
;       ...
        KEYSETUP_BLOCK_CONSTANTS 1,1
        KEYSETUP_BLOCK_CONSTANTS 2,2
        KEYSETUP_BLOCK_CONSTANTS 3,0
        KEYSETUP_BLOCK_CONSTANTS 4,1
        KEYSETUP_BLOCK_CONSTANTS 5,2
        KEYSETUP_BLOCK_CONSTANTS 6,0
        KEYSETUP_BLOCK_CONSTANTS 7,1
        KEYSETUP_BLOCK_CONSTANTS 8,2
        KEYSETUP_BLOCK_CONSTANTS 9,0
        KEYSETUP_BLOCK_CONSTANTS 10,1
        KEYSETUP_BLOCK_CONSTANTS 11,2
        KEYSETUP_BLOCK_CONSTANTS 12,0
        KEYSETUP_BLOCK_CONSTANTS 13,1
        KEYSETUP_BLOCK_CONSTANTS 14,2
        KEYSETUP_BLOCK_CONSTANTS 15,0
        KEYSETUP_BLOCK_CONSTANTS 16,1
        KEYSETUP_BLOCK_CONSTANTS 17,2
        KEYSETUP_BLOCK_CONSTANTS 18,0
        KEYSETUP_BLOCK_CONSTANTS 19,1
        KEYSETUP_BLOCK_CONSTANTS 20,2
        KEYSETUP_BLOCK_CONSTANTS 21,0
        KEYSETUP_BLOCK_CONSTANTS 22,1
        KEYSETUP_BLOCK_CONSTANTS 23,2
        KEYSETUP_BLOCK_CONSTANTS 24,0
;       ...
;       L1[0]  S2[24]
;       S1[25] L2[0]

;       L1[1]  S2[25]
;       S1[0]  L2[1]

key_setup_2:
        add     B1, A1
        add     A2, S_not(25)
        mov     shiftreg, B1

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
        KEYSETUP_BLOCK 0,2
        KEYSETUP_BLOCK 1,0
        KEYSETUP_BLOCK 2,1
        KEYSETUP_BLOCK 3,2
        KEYSETUP_BLOCK 4,0
        KEYSETUP_BLOCK 5,1
        KEYSETUP_BLOCK 6,2
        KEYSETUP_BLOCK 7,0
        KEYSETUP_BLOCK 8,1
        KEYSETUP_BLOCK 9,2
        KEYSETUP_BLOCK 10,0
        KEYSETUP_BLOCK 11,1
        KEYSETUP_BLOCK 12,2
        KEYSETUP_BLOCK 13,0
        KEYSETUP_BLOCK 14,1
        KEYSETUP_BLOCK 15,2
        KEYSETUP_BLOCK 16,0
        KEYSETUP_BLOCK 17,1
        KEYSETUP_BLOCK 18,2
        KEYSETUP_BLOCK 19,0
        KEYSETUP_BLOCK 20,1
        KEYSETUP_BLOCK 21,2
        KEYSETUP_BLOCK 22,0
        KEYSETUP_BLOCK 23,1
        KEYSETUP_BLOCK 24,2
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
        add     B1, L1(0)
        rol     A2, 3

        rol     B1, shiftcount
        mov     S2(25), A2

        add     B2, A2
        add     A1, B1
        add     A2, S2(0)

        mov     shiftreg, B2
        add     B2, L2(0)
        rol     A1, 3

        rol     B2, shiftcount
        mov     S1(0), A1

        mov     L1(0), B1
        mov     L2(0), B2

;       L1[1]   S2[0]
;       S1[1]   L2[1]
;       ...
        KEYSETUP_BLOCK 0,1
        KEYSETUP_BLOCK 1,2
        KEYSETUP_BLOCK 2,0
        KEYSETUP_BLOCK 3,1
        KEYSETUP_BLOCK 4,2
        KEYSETUP_BLOCK 5,0
        KEYSETUP_BLOCK 6,1
        KEYSETUP_BLOCK 7,2
        KEYSETUP_BLOCK 8,0
        KEYSETUP_BLOCK 9,1
        KEYSETUP_BLOCK 10,2
        KEYSETUP_BLOCK 11,0
        KEYSETUP_BLOCK 12,1
        KEYSETUP_BLOCK 13,2
        KEYSETUP_BLOCK 14,0
        KEYSETUP_BLOCK 15,1
        KEYSETUP_BLOCK 16,2
        KEYSETUP_BLOCK 17,0
        KEYSETUP_BLOCK 18,1
        KEYSETUP_BLOCK 19,2
        KEYSETUP_BLOCK 20,0
        KEYSETUP_BLOCK 21,1
        KEYSETUP_BLOCK 22,2
        KEYSETUP_BLOCK 23,0
        KEYSETUP_BLOCK 24,1
;       ...
;       L[1]    S2[24]
;       S1[25]  L2[1]

;               S2[25]

        mov     A1, [work_P_0_ebp]
        add     A2, B2

        rol     A2, 3

        mov     S2(25), A2

;    A1 = rc5_72unitwork->plain.lo + S1[0];
;    A2 = rc5_72unitwork->plain.lo + S2[0];
;    B1 = rc5_72unitwork->plain.hi + S1[1];
;    B2 = rc5_72unitwork->plain.hi + S2[1];

encryption:
        add     B2, A2
        mov     A2, [work_P_0_ebp]

        mov     shiftreg, B2
        add     B2, L2(0)
        mov     B1, [work_P_1_ebp]

        rol     B2, shiftcount
        add     A1, S1(0)

        mov     L2(2), B2
        mov     B2, [work_P_1_ebp]
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

test_key_1:
        cmp     A1, [work_C_0_ebp]
        mov     eax, [RC5_72UnitWork]

        jne     short test_key_2

        inc     dword [RC5_72UnitWork_CMCcount]

        mov     ecx, [RC5_72UnitWork_L0hi]
        mov     esi, [RC5_72UnitWork_L0mid]
        mov     edi, [RC5_72UnitWork_L0lo]

        cmp     B1, [work_C_1_ebp]

        mov     [RC5_72UnitWork_CMChi], ecx
        mov     [RC5_72UnitWork_CMCmid], esi
        mov     [RC5_72UnitWork_CMClo], edi

        jne     short test_key_2

        xor     ecx, ecx

;        mov     eax, RESULT_FOUND

        jmp     short finished_found

;    if (A2 == rc5_72unitwork->cypher.lo)
;    {
;      ++rc5_72unitwork->check.count;
;      rc5_72unitwork->check.hi  = rc5_72unitwork->L0.hi + 0x01;
;      rc5_72unitwork->check.mid = rc5_72unitwork->L0.mid;
;      rc5_72unitwork->check.lo  = rc5_72unitwork->L0.lo;
;      if (B2 == rc5_72unitwork->cypher.hi)
;      {
;        *iterations -= (kiter + 1)*2 - 1;
;        return RESULT_FOUND;
;      }
;    }

k7align 8
test_key_2:
        cmp     A2, [work_C_0_ebp]
        mov     edx, [RC5_72UnitWork_L0hi]
        mov     ecx, [RC5_72UnitWork_L0mid]

        mov     ebx, [RC5_72UnitWork_L0lo]
        jne     short inc_key

        inc     dword [RC5_72UnitWork_CMCcount]
        inc     edx

        mov     [RC5_72UnitWork_CMChi], edx
        mov     [RC5_72UnitWork_CMCmid], ecx
        mov     [RC5_72UnitWork_CMClo], ebx

        dec     edx
        cmp     B2, [work_C_1_ebp]
        jne     short inc_key

        xor     ecx, ecx
        dec     ecx

;        mov     eax, RESULT_FOUND

        jmp     short finished_found

k7align 16
inc_key:
        add     dl, 2
        bswap   ecx
        bswap   ebx

        mov     [RC5_72UnitWork_L0hi], edx
        mov     L1(2), edx
        adc     ecx, BYTE 0

        bswap   ecx
        adc     ebx, BYTE 0
        inc     edx

        bswap   ebx
        dec     dword [work_iterations_ebp]
        mov     L2(2), edx

        mov     L1(1), ecx
        mov     L2(1), ecx

        mov     [RC5_72UnitWork_L0mid], ecx
        mov     [RC5_72UnitWork_L0lo], ebx

        jnz     key_setup_1

        xor     eax, eax
;        mov     eax, RESULT_NOTHING
        jmp     short finished

finished_found:
        xor     eax, eax
        inc     eax

        mov     esi, [iterations]

        add     ecx, [work_iterations_ebp]
        add     ecx, [work_iterations_ebp]

        sub     [esi], ecx

finished:
        inc     eax
        mov     ebx, [save_ebx_ebp]
        mov     esi, [save_esi_ebp]

        mov     edi, [save_edi_ebp]
        mov     ebp, [save_ebp_ebp]
        add     esp, work_size

        ret
