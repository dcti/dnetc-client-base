; Copyright distributed.net 1997-2002 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; Author: Décio Luiz Gazzoni Filho <acidblood@distributed.net>
; $Id: r72-dg2.asm,v 1.9 2002/10/23 04:11:33 acidblood Exp $

%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
[SECTION .text]
%endif

[GLOBAL rc5_72_unit_func_dg_2_]
[GLOBAL _rc5_72_unit_func_dg_2]
[GLOBAL rc5_72_unit_func_dg_2]

%define P         0xB7E15163
%define Q         0x9E3779B9
%define S_not(N)  (P+Q*(N))

%define RESULT_NOTHING 1
%define RESULT_FOUND   2

%assign work_size 0

%macro defidef 2
    %define %1 esp+%2
%endmacro

%macro fedifed 2
    %define %1 ebp+%2-128
%endmacro

%macro defwork 1-2 1
    defidef %1,work_size
    fedifed %1_ebp,work_size
    %assign work_size work_size+4*(%2)
%endmacro

defwork work_L1,3
defwork work_L2,3
defwork work_s1,26
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

%define RC5_72UnitWork          esp+work_size+4
%define iterations              esp+work_size+8

%define S1(N)                   [work_s1+((N)*4)]
%define S2(N)                   [ebp+((N)*4)]
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
%define A1         eax
%define A2         edx
%define B1         ebx
%define B2         esi
%define shiftreg   ecx
%define shiftcount cl

;       L1[j]   S2[i]
;       S1[i+1] L2[j]

%macro KEYSETUP_BLOCK_CONSTANTS_j0 1
        lea     shiftreg, [A1 + B1]
        add     B1, A1
        lea     A2, [A2 + B2 + S_not(%1)]

        add     B1, L1(0)

        rol     B1, shiftcount
        rol     A2, 3

        mov     L1(0), B1
        mov     S2(%1), A2


        lea     shiftreg, [A2 + B2]
        add     B2, A2
        lea     A1, [A1 + B1 + S_not(%1+1)]

        add     B2, L2(0)

        rol     B2, shiftcount
        rol     A1, 3

        mov     L2(0), B2
        mov     S1(%1+1), A1
%endmacro

%macro KEYSETUP_BLOCK_CONSTANTS_j1 1
        lea     shiftreg, [A1 + B1]
        add     B1, A1
        lea     A2, [A2 + B2 + S_not(%1)]

        add     B1, L1(1)

        rol     B1, shiftcount
        rol     A2, 3

        mov     L1(1), B1
        mov     S2(%1), A2


        lea     shiftreg, [A2 + B2]
        add     B2, A2
        lea     A1, [A1 + B1 + S_not(%1+1)]

        add     B2, L2(1)

        rol     B2, shiftcount
        rol     A1, 3

        mov     L2(1), B2
        mov     S1(%1+1), A1
%endmacro

%macro KEYSETUP_BLOCK_CONSTANTS_j2 1
        lea     shiftreg, [A1 + B1]
        add     B1, A1
        lea     A2, [A2 + B2 + S_not(%1)]

        add     B1, L1(2)

        rol     B1, shiftcount
        rol     A2, 3

        mov     L1(2), B1
        mov     S2(%1), A2


        lea     shiftreg, [A2 + B2]
        add     B2, A2
        lea     A1, [A1 + B1 + S_not(%1+1)]

        add     B2, L2(2)

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
        add     B1, L1(0)
        rol     A2, 3

        rol     B1, shiftcount
        mov     S2(%1), A2

        add     B2, A2
        add     A1, B1
        add     A2, S2(%1+1)

        mov     shiftreg, B2
        add     B2, L2(0)
        rol     A1, 3

        rol     B2, shiftcount
        mov     S1(%1+1), A1

        mov     L1(0), B1
        mov     L2(0), B2
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
rc5_72_unit_func_dg_2:
rc5_72_unit_func_dg_2_:
_rc5_72_unit_func_dg_2:

        sub     esp, work_size
        mov     [save_ebp], ebp
        lea     ebp, S1(26)

        mov     eax, [RC5_72UnitWork]
        mov     [save_ebx], ebx
        mov     [save_esi], esi

        mov     [save_edi], edi
        mov     esi, [RC5_72UnitWork_plainlo]
        mov     edi, [RC5_72UnitWork_plainhi]

        mov     ebx, [RC5_72UnitWork_cipherlo]
        mov     ecx, [RC5_72UnitWork_cipherhi]
        mov     edx, [iterations]

        mov     [work_P_0_ebp], esi
        mov     [work_P_1_ebp], edi
        mov     edi, [RC5_72UnitWork_L0hi]

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

        mov     ecx, [work_iterations_ebp]
        mov     esi, [iterations]

        mov     edi, [esi]

        shl     ecx, 1
        mov     ebx, [save_ebx]

        sub     edi, ecx

        mov     [esi], edi
        mov     esi, [save_esi]

        mov     edi, [save_edi]
        mov     ebp, [save_ebp]
        mov     eax, RESULT_FOUND
        add     esp, work_size

        ret

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

k7align 16
test_key_2:
        cmp     A2, [work_C_0_ebp]
        mov     edx, [RC5_72UnitWork_L0hi]
        mov     ecx, [RC5_72UnitWork_L0mid]

        mov     ebx, [RC5_72UnitWork_L0lo]
        jne     short inc_key

        inc     dword [RC5_72UnitWork_CMCcount]

        cmp     B2, [work_C_1_ebp]

        mov     [RC5_72UnitWork_CMChi], edx
        mov     [RC5_72UnitWork_CMCmid], ecx
        mov     [RC5_72UnitWork_CMClo], ebx

        jne     short inc_key

        mov     ecx, [work_iterations_ebp]
        mov     esi, [iterations]

        mov     edi, [esi]

        shl     ecx, 1

        sub     ecx, 1
        mov     ebx, [save_ebx]

        sub     edi, ecx

        mov     [esi], edi
        mov     esi, [save_esi]

        mov     edi, [save_edi]
        mov     ebp, [save_ebp]
        mov     eax, RESULT_FOUND
        add     esp, work_size

        ret

k7align 16
inc_key:
        add     dl, 2
        bswap   ecx
        bswap   ebx

        mov     [RC5_72UnitWork_L0hi], edx
        mov     L1(2), edx
        adc     ecx, 0

        bswap   ecx
        adc     ebx, 0
        inc     edx

        bswap   ebx
        dec     dword [work_iterations_ebp]
        mov     L2(2), edx

        mov     L1(1), ecx
        mov     L2(1), ecx
        mov     L1(0), ebx

        mov     L2(0), ebx
        mov     [RC5_72UnitWork_L0mid], ecx
        mov     [RC5_72UnitWork_L0lo], ebx

        jnz     key_setup_1

finished:
        mov     ebx, [save_ebx]
        mov     esi, [save_esi]

        mov     edi, [save_edi]
        mov     ebp, [save_ebp]
        mov     eax, RESULT_NOTHING
        add     esp, work_size

        ret
