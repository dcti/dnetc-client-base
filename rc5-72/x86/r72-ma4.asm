; Copyright distributed.net 1997-2003 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
;  (requires SSE2 instruction support)
; MMX Extentions: Matt Andrews <andrewsm@ufl.edu> 2003/09/17
; Author: Slawomir Piotrowski <sgp@telsatgp.com.pl>
; Version 1.0    2003/09/08  23:53
;
; Based on dg-3 by Décio Luiz Gazzoni Filho <acidblood@distributed.net>
; $Id: r72-ma4.asm,v 1.3 2007/10/22 16:48:36 jlawson Exp $

; SIMD Core Idea
; The only areas of the CPU that former cores have not utilized are the
; FPU and the MMX_* units.  I have yet to figure out a way to effectively
; use the FPU for key crunching (it doesn't even have bit shift instructions)
; but the MMX_ALU has all the necessary ops but rotate.  Theoretically,
; we should be able to decode a key in the MMX_ALU simultaneously with keys
; being decoded using the regular ALU.  We simulate a rotate instruction by
; copying the 32-bit value we are working on to the lo and hi dwords of
; a quadword.  Then when we shift the quadword left, the bits shift into the
; top dword from the bottom dword.

; Having the stack quadword aligned could possibly increase performance, but since
; we use EBP for computation this would have to be done by the caller.

; Register usage:
; xmm0 -- L4(0)
; xmm1 -- L4(1)
; xmm2 -- L4(2)
; xmm3 -- B4
; xmm4 -- A4
; xmm5 -- temp data
; xmm6 -- shiftcount
; xmm7 -- shift mask

%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_72_unit_func_ma_4]
[GLOBAL rc5_72_unit_func_ma_4]

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

defwork work_l,3*3
defwork work_s,4*26
defwork work_backup_l,4*3
defwork temp, 4
defwork work_P_0
defwork work_P_1
defwork work_C_0
defwork work_C_1
defwork work_iterations
defwork save_ebx
defwork save_esi
defwork save_edi
defwork save_ebp
defwork save_esp

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

%define S1(N)                   [work_s+((N)*4*4)]
%define S2(N)                   [work_s+((N)*4*4)+4]
%define S3(N)                   [work_s+((N)*4*4)+8]
%define S4(N)                   [work_s+((N)*4*4)+12]
%define L1(N)                   [work_l+((N)*3*4)]
%define L2(N)                   [work_l+((N)*3*4)+4]
%define L3(N)                   [work_l+((N)*3*4)+8]
;%define L4(N)                   [work_l+((N)*4*4)+12]
%define L1backup(N)             [work_backup_l+((N)*4*4)]
%define L2backup(N)             [work_backup_l+((N)*4*4)+4]
%define L3backup(N)             [work_backup_l+((N)*4*4)+8]
%define L4backup(N)             [work_backup_l+((N)*4*4)+12]

%define A1         eax
%define A2         ebx
%define A3         edx
%define B1         esi
%define B2         edi
%define B3         ebp
%define shiftreg   ecx
%define shiftcount cl


%define P         0xB7E15163
%define Q         0x9E3779B9
%define S_not(N)  (P+Q*(N))

; Having a quadword-based value table provides considerable speedups for the SIMD extentions
   align 8
   InitTable  dd 0xb7e15163, 0xb7e15163
              dd 0x5618cb1c, 0x5618cb1c
              dd 0xf45044d5, 0xf45044d5
              dd 0x9287be8e, 0x9287be8e
              dd 0x30bf3847, 0x30bf3847
              dd 0xcef6b200, 0xcef6b200
              dd 0x6d2e2bb9, 0x6d2e2bb9
              dd 0x0b65a572, 0x0b65a572
              dd 0xa99d1f2b, 0xa99d1f2b
              dd 0x47d498e4, 0x47d498e4
              dd 0xe60c129d, 0xe60c129d
              dd 0x84438c56, 0x84438c56
              dd 0x227b060f, 0x227b060f
              dd 0xc0b27fc8, 0xc0b27fc8
              dd 0x5ee9f981, 0x5ee9f981
              dd 0xfd21733a, 0xfd21733a
              dd 0x9b58ecf3, 0x9b58ecf3
              dd 0x399066ac, 0x399066ac
              dd 0xd7c7e065, 0xd7c7e065
              dd 0x75ff5a1e, 0x75ff5a1e
              dd 0x1436d3d7, 0x1436d3d7
              dd 0xb26e4d90, 0xb26e4d90
              dd 0x50a5c749, 0x50a5c749
              dd 0xeedd4102, 0xeedd4102
              dd 0x8d14babb, 0x8d14babb
              dd 0x2b4c3474, 0x2b4c3474


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

%macro KEYSETUP_BLOCK 3
        movdqa  xmm6, xmm3

        mov     S1(%2), A1
        movd    S4(%2), xmm4        ; KEY 4
        mov     S2(%2), A2
        mov     S3(%2), A3

        lea     shiftreg, [A1+B1]   ; KEY 1
        paddd   xmm6, xmm4          ; KEY 4
        add     B1, L1(%3)          ; KEY 1
        pand    xmm6, xmm7          ; KEY 4
        add     B1, A1              ; KEY 1
        rol     B1, shiftcount      ; KEY 1
        paddd   xmm3, xmm%3         ; KEY 4

%ifidn %1,S
    movd    xmm5, S4(%2+1)          ; KEY 4
%else
    movq    xmm5, [InitTable + (%2 + 1) * 8]    ; KEY 4
%endif

        lea     shiftreg, [A2+B2]   ; KEY 2
        paddd   xmm3, xmm4          ; KEY 4
        add     B2, L2(%3)          ; KEY 2
        psllq   xmm3, xmm6          ; KEY 4
        add     B2, A2              ; KEY 2

%ifidn %1,S
        add     A1, S1(%2+1)        ; KEY 1
%else
        add     A1, S_not(%2+1)     ; KEY 1
%endif

        rol     B2, shiftcount      ; KEY 2

        lea     shiftreg, [A3+B3]   ; KEY 3
        add     B3, L3(%3)          ; KEY 3
        pshuflw xmm3, xmm3, 0xEE    ; KEY 4
        add     B3, A3              ; KEY 3

%ifidn %1,S
        add     A2, S2(%2+1)        ; KEY 2
%else
        add     A2, S_not(%2+1)     ; KEY 2
%endif

        rol     B3, shiftcount      ; KEY 3


        add     A1, B1
        mov     L1(%3), B1
        rol     A1, 3

        paddd   xmm4, xmm5

        add     A2, B2

%ifid %1, S
        pshuflw xmm4, xmm4, 0x44
%endif

        mov     L2(%3), B2

        paddd   xmm4, xmm3

        rol     A2, 3               ; rol1   (2)

        movdqa  xmm%3, xmm3

%ifidn %1,S                         ; count2 (3)
        add     A3, S3(%2+1)
%else
        add     A3, S_not(%2+1)
%endif
        add     A3, B3
        psllq   xmm4, 3
        mov     L3(%3), B3

        rol     A3, 3               ; rol1   (3)

        pshuflw xmm4, xmm4, 0xEE
%endmacro

%macro KEYSETUP_BLOCK_PRE 1
        psllq   xmm4, 3
        rol     A1, 3
        rol     A2, 3
        pshuflw xmm4, xmm4, 0xEE
        rol     A3, 3

%ifnidn %1,-1
        mov     L1(%1), B1
        mov     L2(%1), B2
        mov     L3(%1), B3
        movdqa  xmm%1, xmm3
%endif
%endmacro

%macro KEYSETUP_BLOCK_POST 1
%endmacro


%macro ENCRYPTION_BLOCK 1
        mov     shiftreg, B1
        xor     A1, B1
        movdqa  xmm6, xmm3
        xor     A2, B2
        rol     A1, shiftcount

        movd    xmm5, S4(2*%1)
        mov     shiftreg, B2
        xor     A3, B3
        rol     A2, shiftcount

        pand    xmm6, xmm7
        mov     shiftreg, B3
        pxor    xmm4, xmm3
        rol     A3, shiftcount

        psllq   xmm4, xmm6
        add     A1, S1(2*%1)
        add     A2, S2(2*%1)
        pshuflw xmm4, xmm4, 0xEE
        add     A3, S3(2*%1)
        paddd   xmm4, xmm5
        pshufd  xmm4, xmm4, 0xE0

        mov     shiftreg, A1
        xor     B1, A1
        movdqa  xmm6, xmm4
        xor     B2, A2
        rol     B1, shiftcount
        pand    xmm6, xmm7

        mov     shiftreg, A2
        pxor    xmm3, xmm4
        xor     B3, A3
        psllq   xmm3, xmm6
        rol     B2, shiftcount

        ; We could move S3(2*%1 + 1) quadword here, and then we wouldn't need the
        ; pshuflw xmm3, xmm3, 0xEE below.  However, this will only improve time if
        ; the entire quadword happens to be in cache, which from testing doesn't
        ; appear to often be the case.
        movd    xmm5, S4(2*%1 + 1)

        mov     shiftreg, A3
        add     B1, S1(2*%1+1)
        rol     B3, shiftcount

        pshuflw xmm3, xmm3, 0xEE
        add     B2, S2(2*%1+1)
        paddd   xmm3, xmm5

        add     B3, S3(2*%1+1)
        pshuflw xmm3, xmm3, 0x44
%endmacro

%macro  DEBUG_CHECK_AB 0
        movd    A1, xmm4
        movd    A2, xmm3
        cmp     A3, A1
        cmp     B3, A2
        int 3
%endmacro

align 16

startseg:
rc5_72_unit_func_ma_4:
_rc5_72_unit_func_ma_4:

        sub     esp, work_size

        mov     eax, [RC5_72UnitWork]
        mov     [save_ebp], ebp
        mov     [save_ebx], ebx
        mov     [save_esi], esi
        mov     [save_edi], edi

        mov     esi, [RC5_72UnitWork_plainlo]
        mov     edi, [RC5_72UnitWork_plainhi]

        mov     ebx, [RC5_72UnitWork_cipherlo]
        mov     ecx, [RC5_72UnitWork_cipherhi]

        mov     edx, [iterations]

        mov     [work_P_0], esi
        mov     [work_P_1], edi
        mov     esi, [RC5_72UnitWork_L0hi]

        mov     [work_C_0], ebx
        mov     [work_C_1], ecx
        mov     edi, [edx]

        mov     ecx, [RC5_72UnitWork_L0mid]
        mov     ebx, [RC5_72UnitWork_L0lo]

        mov     [work_iterations], edi
        mov     L1(2), esi
        mov     L1backup(2), esi

        add     esi, 1

        mov     L2(2), esi
        mov     L2backup(2), esi

        add     esi, 1

        mov     L3(2), esi
        mov     L3backup(2), esi

        add     esi, 1
        movd    xmm2, esi
        mov     L4backup(2), esi
        pshuflw xmm2, xmm2, 0x44

        mov     L1(1), ecx
        mov     L2(1), ecx
        mov     L3(1), ecx
        movd    xmm1, ecx
        mov     L1backup(1), ecx
        mov     L2backup(1), ecx
        mov     L3backup(1), ecx
        mov     L4backup(1), ecx
        pshuflw xmm1, xmm1, 0x44

        mov     L1(0), ebx
        mov     L2(0), ebx
        mov     L3(0), ebx
        movd    xmm0, ebx
        mov     L1backup(0), ebx
        mov     L2backup(0), ebx
        mov     L3backup(0), ebx
        mov     L4backup(0), ebx
        pshuflw xmm0, xmm0, 0x44

        mov     dword [temp], 0x0000001F
        movd    xmm7, [temp]
k7align 16
key_setup_1:
        mov     B1, L1(0)
        mov     A1, 0xBF0A8B1D ; 0xBF0A8B1D is S[0]
        add     B1, 0xBF0A8B1D
        mov     A2, A1
        ror     B1, 3
        mov     B2, B1
        mov     B3, B1
        movd    xmm3, B1

        mov     A3, A1
        movd    xmm4, A1

        mov     S1(0), A1
        mov     S2(0), A1
        pshuflw xmm3, xmm3, 0x44
        mov     S3(0), A1
        mov     S4(0), A1

        ;mov        dword [temp], S_not(1)
        ;movd   xmm5, [temp]
        movq    xmm5, [InitTable + 8]
        lea     A1, [A1 + B1 + S_not(1)]
        paddd   xmm4, xmm3
        lea     A2, [A2 + B2 + S_not(1)]
        paddd   xmm4, xmm5
        lea     A3, [A3 + B3 + S_not(1)]
        pshufd  xmm4, xmm4, 0xE0

        KEYSETUP_BLOCK_PRE 0
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
        KEYSETUP_BLOCK_POST 0

        add     B1, A1

        mov     S1(25), A1
        mov     S2(25), A2
        mov     S3(25), A3
        movd    S4(25), xmm4

        paddd   xmm3, xmm4

        mov     shiftreg, B1
        movdqa  xmm6, xmm3
        add     B1, L1(1)

        add     A1, S1(0)
        rol     B1, shiftcount

        add     B2, A2
        pand    xmm6, xmm7
        paddd   xmm3, xmm1
        movd    xmm5, S4(0)
        mov     shiftreg, B2


        paddd   xmm4, xmm5

        add     B2, L2(1)
        pshuflw xmm4, xmm4, 0x44
        add     B3, A3

        rol     B2, shiftcount


        mov     shiftreg, B3
        psllq   xmm3, xmm6
        add     B3, L3(1)
        pshuflw xmm3, xmm3, 0xEE
        rol     B3, shiftcount

        paddd   xmm4, xmm3
        add     A2, S2(0)
        add     A3, S3(0)
        add     A1, B1
        mov     L1(1), B1
        add     A2, B2
        mov     L2(1), B2
        add     A3, B3
        mov     L3(1), B3

        movdqa  xmm1, xmm3

key_setup_2:

        KEYSETUP_BLOCK_PRE -1
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
        KEYSETUP_BLOCK_POST 2

        add     B1, A1

        mov     S1(25), A1
        mov     S2(25), A2
        mov     S3(25), A3
        movd    S4(25), xmm4

        movd    xmm5, S4(0)
        mov     shiftreg, B1
        paddd   xmm3, xmm4
        movdqa  xmm6, xmm3
        paddd   xmm3, xmm0
        add     B1, L1(0)

        add     A1, S1(0)
        pand    xmm6, xmm7
        paddd   xmm4, xmm5
        rol     B1, shiftcount

        add     B2, A2
        pshufd  xmm4, xmm4, 0xE0
        mov     shiftreg, B2

        add     B2, L2(0)

        psllq   xmm3, xmm6
        add     B3, A3

        rol     B2, shiftcount
        mov     shiftreg, B3
        pshufd  xmm3, xmm3, 0xE5
        add     B3, L3(0)

        rol     B3, shiftcount

        add     A2, S2(0)
        add     A3, S3(0)

        paddd   xmm4, xmm3
        add     A1, B1
        mov     L1(0), B1
        add     A2, B2
        mov     L2(0), B2
        add     A3, B3
        mov     L3(0), B3
        movdqa  xmm0, xmm3

key_setup_3:

        KEYSETUP_BLOCK_PRE -1
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
        KEYSETUP_BLOCK_POST 1

        mov     S1(25), A1
        mov     S2(25), A2
        mov     S3(25), A3
        movd    S4(25), xmm4

;    A1 = rc5_72unitwork->plain.lo + S1[0];
;    A2 = rc5_72unitwork->plain.lo + S2[0];
;    B1 = rc5_72unitwork->plain.hi + S1[1];
;    B2 = rc5_72unitwork->plain.hi + S2[1];

encryption:

        mov     A1, [work_P_0]
        mov     B1, [work_P_1]
        movd    xmm4, A1
        mov     A2, A1
        mov     A3, A1

        movd    xmm3, B1
        mov     B2, B1
        movd    xmm5, S4(0)
        mov     B3, B1

        paddd   xmm4, xmm5
        add     A1, S1(0)
        add     A2, S2(0)
        pshuflw xmm4, xmm4, 0x44
        add     A3, S3(0)

        movd    xmm5, S4(1)
        add     B1, S1(1)
        paddd   xmm3, xmm5
        add     B2, S2(1)
        pshuflw xmm3, xmm3, 0x44
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

        ;DEBUG_CHECK_AB

test_key:
        cmp     A1, [work_C_0]
        mov     eax, [RC5_72UnitWork]
        je      near test_key_1
back_after_key_1:
        cmp     A2, [work_C_0]
        je      near test_key_2
back_after_key_2:
        cmp     A3, [work_C_0]
        je      near test_key_3
back_after_key_3:
        movd    A2, xmm4
        cmp     A2, [work_C_0]
        je      near test_key_4
back_after_key_4:
        mov     edx, [RC5_72UnitWork_L0hi]

inc_key:
        xor edi, edi
        movzx esi, dl
        mov ecx, 1
        and edx, 0xFFFFFF00
        add esi, 4
        and esi, 0xFF
        cmove edi, ecx
        or edx, esi
        mov [RC5_72UnitWork_L0hi], edx

        mov ebx, [RC5_72UnitWork_L0mid]
        mov ecx, [RC5_72UnitWork_L0lo]
        bswap ebx
        bswap ecx
        add ebx, edi
        adc ecx, 0
        bswap ebx
        bswap ecx
        mov [RC5_72UnitWork_L0mid], ebx
        mov [RC5_72UnitWork_L0lo], ecx

        mov     L1(2), edx
        mov     L1backup(2), edx
        add     edx, 1
        mov     L2(2), edx
        mov     L2backup(2), edx
        add     edx, 1
        mov     L3(2), edx
        mov     L3backup(2), edx
        add     edx, 1
        movd    xmm2, edx
        mov     L4backup(2), edx
        pshuflw xmm2, xmm2, 0x44

        mov     L1(1), ebx
        mov     L2(1), ebx
        mov     L3(1), ebx
        movd    xmm1, ebx
        mov     L1backup(1), ebx
        mov     L2backup(1), ebx
        mov     L3backup(1), ebx
        mov     L4backup(1), ebx
        pshuflw xmm1, xmm1, 0x44

        mov     L1(0), ecx
        mov     L2(0), ecx
        mov     L3(0), ecx
        movd    xmm0, ecx
        mov     L1backup(0), ecx
        mov     L2backup(0), ecx
        mov     L3backup(0), ecx
        mov     L4backup(0), ecx
        pshuflw xmm0, xmm0, 0x44

        sub dword [work_iterations], 4

        ja      key_setup_1

        mov     eax, RESULT_NOTHING
finished:
        mov     ebx, [save_ebx]
        mov     esi, [save_esi]

        mov     edi, [save_edi]
        mov     ebp, [save_ebp]
        add     esp, work_size

        ret
test_key_1:
        add     dword [RC5_72UnitWork_CMCcount], 1
        mov     ecx, L1backup(2)
        mov     [RC5_72UnitWork_CMChi], ecx
        mov     ecx, L1backup(1)
        mov     [RC5_72UnitWork_CMCmid], ecx
        mov     ecx, L1backup(0)
        mov     [RC5_72UnitWork_CMClo], ecx

        cmp     B1, [work_C_1]
        jne     back_after_key_1

        mov     ecx, [work_iterations]
        mov     esi, [iterations]

        sub     [esi], ecx
        mov     eax, RESULT_FOUND
        jmp     finished

test_key_2:
        mov     esi, L2backup(2)
        mov     ecx, L2backup(1)
        mov     ebx, L2backup(0)
        add     dword [RC5_72UnitWork_CMCcount], 1
        mov     [RC5_72UnitWork_CMChi], esi
        mov     [RC5_72UnitWork_CMCmid], ecx
        mov     [RC5_72UnitWork_CMClo], ebx

        cmp     B2, [work_C_1]
        jne     back_after_key_2

        mov     ecx, [work_iterations]
        mov     esi, [iterations]

        sub     ecx, 1
        sub     [esi], ecx
        mov     eax, RESULT_FOUND
        jmp     finished

test_key_3:
        mov     esi, L3backup(2)
        mov     ecx, L3backup(1)
        mov     ebx, L3backup(0)
        add     dword [RC5_72UnitWork_CMCcount], 1
        mov     [RC5_72UnitWork_CMChi], esi
        mov     [RC5_72UnitWork_CMCmid], ecx
        mov     [RC5_72UnitWork_CMClo], ebx

        cmp     B3, [work_C_1]
        jne     back_after_key_3

        mov     ecx, [work_iterations]
        mov     esi, [iterations]

        sub     ecx, 2
        sub     [esi], ecx
        mov     eax, RESULT_FOUND
        jmp     finished
test_key_4:
        movd    B3, xmm3
        mov     esi, L4backup(2)
        mov     ecx, L4backup(1)
        mov     ebx, L4backup(0)
        add     dword [RC5_72UnitWork_CMCcount], 1
        mov     [RC5_72UnitWork_CMChi], esi
        mov     [RC5_72UnitWork_CMCmid], ecx
        mov     [RC5_72UnitWork_CMClo], ebx

        cmp     B3, [work_C_1]
        jne     back_after_key_4

        mov     ecx, [work_iterations]
        mov     esi, [iterations]

        sub     ecx, 3
        sub     [esi], ecx
        mov     eax, RESULT_FOUND
        jmp     finished
