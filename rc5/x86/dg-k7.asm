; $Id: dg-k7.asm,v 1.1.2.1 2002/03/04 18:41:13 acidblood Exp $
; Slight improvements by Décio Luiz Gazzoni Filho <acidblood@distributed.net>
; Measured speedup relative to hb-k7, average of 3 benchmarks: 1.6%
; Code size reduction: about 10%
; Changes:
; -have ESP point at middle of allocated space for temp variables; that way,
;  most [ESP+offset] memory accesses have displacements encoded as 8-bit
;  signed immediates, instead of the full 32-bit addressing mode.
; -incrementing code at end of loop was replaced by a BSWAP-based approach.
;
; AMD K7 optimized version
; from Holger Böhring <HBoehring@hboehring.de>
; based on Cx6x86 core
;
; performance 2106/1830 with client v2.8008-459
; comparitive benchmark with core from rc5-k7ss.asm:
; [Jun 26 23:07:28 UTC] RC5: using core #6 (RG/SS ath).
; [Jun 26 23:07:47 UTC] Benchmark for RC5 core #6 (RG/SS ath)
;                       0.00:00:16.10 [2,018,617.09 keys/sec]
; [Jun 26 23:07:47 UTC] RC5: using core #7 (RG/HB ath).
; [Jun 26 23:08:05 UTC] Benchmark for RC5 core #7 (RG/HB ath)
;                       0.00:00:16.42 [2,106,921.64 keys/sec]
;
; Initial release
; based on routines for 6x86 so partial credit goes to whoever wrote that
; performance increment: 12.45 %
;    (medium of 30 loops alternating new and old core,1<<16 iterations/lo
;    higest prio,timer rtdsc)
; test-configuration: AMD K7-600 (detected:K7-1), Asus K7M, 60ns DRAM, N
; example: 2094KKeys/1832KKeys with -benchmark rc5, client v2.8002-446
; eg. 14.3 % speedup


%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _TEXT USE32 align=16]
%else
[SECTION .text]
%endif

[GLOBAL _rc5_unit_func_k7]
[GLOBAL rc5_unit_func_k7]

; The S0 values for key expansion round 1 are constants.

%define P         0xB7E15163
%define Q         0x9E3779B9
%define S_not(N)  (P+Q*(N))

;#define S0_ROTL3  _(((P<<3) | (P>>29)))
%define S0_ROTL3 0xbf0a8b1d
;#define FIRST_ROTL _((S0_ROTL3 & 0x1f))
%define FIRST_ROTL 0x1d
;#define S1_S0_ROTL3 _((S_not(1) + S0_ROTL3))
%define S1_S0_ROTL3 0x15235639


;  Offsets to access work_struct fields.
%assign work_size        0

%macro predefwork 1-2 1
    %assign work_size work_size+4*(%2)
%endmacro

%macro enddefwork 0
    %assign opt_work_size work_size/2
    %assign work_size opt_work_size-work_size
%endmacro

; macros to define the workspace
%macro defidef 2
    %define %1 esp+%2
%endmacro

%macro defwork 1-2 1
    defidef %1,work_size
    %assign work_size work_size+4*(%2)
%endmacro

; Locals
predefwork work_key2_edi
predefwork work_key2_esi
predefwork work_key_hi
predefwork work_key_lo
predefwork RC5UnitWork
predefwork timeslice
predefwork work_s1,26
predefwork work_s2,26
predefwork work_P_0
predefwork work_P_1
predefwork work_C_0
predefwork work_C_1
predefwork work_iterations
predefwork work_pre1_r1
predefwork work_pre2_r1
predefwork work_pre3_r1
predefwork save_ebp
predefwork save_edi
predefwork save_esi
predefwork save_ebx

enddefwork

defwork save_ebp
defwork work_key2_esi
defwork work_key_hi
defwork work_key_lo
defwork RC5UnitWork
defwork timeslice
defwork work_s1,26
defwork work_s2,26
defwork work_P_0
defwork work_P_1
defwork work_C_0
defwork work_C_1
defwork work_iterations
defwork work_pre1_r1
defwork work_pre2_r1
defwork work_pre3_r1
defwork work_key2_edi
defwork save_edi
defwork save_esi
defwork save_ebx


; Offsets to access RC5UnitWork fields
%define RC5UnitWork_plainhi   eax+0
%define RC5UnitWork_plainlo   eax+4
%define RC5UnitWork_cipherhi  eax+8
%define RC5UnitWork_cipherlo  eax+12
%define RC5UnitWork_L0hi      eax+16
%define RC5UnitWork_L0lo      eax+20

%define RC5UnitWork_t   esp+work_size+4
%define timeslice_t     esp+work_size+8


  ; A1   = %eax  A2   = %ebp
  ; Llo1 = %ebx  Llo2 = %esi
  ; Lhi1 = %edx  Lhi2 = %edi

%define S1(N) [work_s1+((N)*4)]
%define S2(N) [work_s2+((N)*4)]

; ------------------------------------------------------------------
; S1(N) = A1 = ROTL3 (A1 + Lhi1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Lhi2 + S_not(N));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro ROUND_1_EVEN 1
        mov     S1(%1), eax

        add     ebp,edi
        lea     ecx,[eax+edx]

        add     eax,S_not(%1+1)
        rol     ebp,3
        add     ebx,ecx

        mov     S2(%1),ebp
        rol     ebx,cl

        lea     ecx,[ebp+edi]

        add     eax,ebx
        add     esi,ecx

        rol     eax,3
        add     ebp,S_not(%1+1)
        rol     esi,cl
%endmacro


; S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_1_ODD 1
        mov     S1(%1),eax

        add     ebp, esi
        lea     ecx,[eax+ebx]
        add     eax,S_not(%1+1)

        rol     ebp,3
        add     edx,ecx

        mov     S2(%1),ebp
        rol     edx, cl

        lea     ecx,[ebp+esi]
        add     eax,edx
        add     edi,ecx

        rol     eax,3
        add     ebp,S_not(%1+1)
        rol     edi,cl
%endmacro

; ------------------------------------------------------------------
; S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
; S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro ROUND_2_EVEN 1
        add     ebp,edi

        mov     S1(%1),eax
        lea     ecx,[eax+edx]

        rol     ebp,3
        add     eax,S1(%1+1)
        add     ebx,ecx

        mov     S2(%1),ebp
        rol     ebx,cl

        lea     ecx,[ebp+edi]

        add     ebp,S2(%1+1)
        add     eax,ebx
        add     esi,ecx

        rol     eax,3
        rol     esi,cl
%endmacro

; S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
; S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_2_ODD 1
        add     ebp,esi
        mov     S1(%1),eax
        lea     ecx,[eax+ebx]

        rol     ebp,3
        add     eax,S1(%1+1)
        add     edx,ecx

        mov     S2(%1),ebp
        rol     edx,cl
        lea     ecx,[ebp+esi]

        add     eax,edx
        add     edi,ecx

        add     ebp,S2(%1+1)
        rol     eax,3
        rol     edi,cl
%endmacro

; ------------------------------------------------------------------
; even
; eA = ROTL (eA ^ eB, eB) + (A = ROTL3 (A + L1 + S(N)));
; L0 = ROTL (L0 + A + L1, A + L1);
; odd
; eB = ROTL (eA ^ eB, eA) + (A = ROTL3 (A + L0 + S(N+1)));
; L1 = ROTL (L1 + A + L0, A + L0);

; A  = %eax  eA = %esi
; L0 = %ebx  eB = %edi
; L1 = %edx  .. = %ebp
; %%ebp is either &S1 or &S2

%define S3(N) [ebp+(N)*4]

%macro ROUND_3_EVEN_AND_ODD 1
        mov     ecx,edi
        add     eax,edx
        rol     esi,cl

        rol     eax,3
        mov     ecx,edx

        add     esi,eax
        add     ecx,eax
        add     eax,S3(%1+1)

        add     ebx,ecx
        xor     edi,esi

        rol     ebx,cl

        mov     ecx,esi

        add     eax,ebx
        rol     edi,cl

        rol     eax,3
        mov     ecx,ebx

        add     edi,eax

        add     ecx,eax
        add     eax,S3(%1+2)

        add     edx,ecx
        xor     esi,edi

        rol     edx,cl
%endmacro

%macro ROUND 3
%assign i %2
%rep %3
    %if %1==3
        %if (i&1)^1
            ROUND_3_EVEN_AND_ODD i
        %endif
    %elif %1==2
        %if i&1
            ROUND_2_ODD i
        %else
            ROUND_2_EVEN i
        %endif
    %elif %1==1
        %if i&1
            ROUND_1_ODD i
        %else
            ROUND_1_EVEN i
        %endif
    %else
        %error round must be range 1..3
    %endif
    %assign i i+1
    %if i>%3
        %exitrep
    %endif
%endrep
%endmacro

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

; ------------------------------------------------------------------
; rc5_unit will get passed an RC5WorkUnit to complete
;
; Returns number of keys checked before a possible good key is found, or
; timeslice*PIPELINE_COUNT if no keys are 'good' keys.
; (ie:      if (result == timeslice*PIPELINE_COUNT) NOTHING_FOUND
;      else if (result < timeslice*PIPELINE_COUNT) SOMETHING_FOUND at result+1
;      else SOMETHING_WENT_WRONG... )

align 16
startseg:                                           ; used by k7align
rc5_unit_func_k7:
_rc5_unit_func_k7:
;u32 rc5_unit_func_k7( RC5UnitWork * rc5unitwork, u32 timeslice )

        sub esp, opt_work_size ; set up stack

        mov     [save_ebp], ebp ; save registers
        mov     [save_edi], edi
        mov     [save_esi], esi
        mov     [save_ebx], ebx

        mov     ebp, [timeslice_t]

        mov     eax, [RC5UnitWork_t] ; load pointer to rc5unitwork into eax

;    work.iterations = timeslice;
        mov     [timeslice], ebp
        mov     [work_iterations], ebp
        mov     [RC5UnitWork], eax

        ; load parameters
        mov     ebx, [RC5UnitWork_L0lo]                 ; ebx = l0 = Llo1
        mov     edx, [RC5UnitWork_L0hi]                 ; edx = l1 = Lhi1
        mov     esi, ebx                                ; esi = l2 = Llo2
        lea     edi, [0x01000000+edx]                   ; edi = l3 = lhi2
        mov     [work_key_lo], ebx
        mov     [work_key_hi], edx

        ; Save other parameters
        ; (it's faster to do so, since we will only load 1 value
        ; each time in RC5_ROUND_3xy, instead of two if we save
        ; only the pointer to the RC5 struct)
        mov     ebp, [RC5UnitWork_plainlo]
        mov     ecx, [RC5UnitWork_plainhi]
        mov     [work_P_0], ebp
        mov     [work_P_1], ecx
        mov     ebp, [RC5UnitWork_cipherlo]
        mov     ecx, [RC5UnitWork_cipherhi]
        mov     [work_C_0], ebp
        mov     [work_C_1], ecx

        ; Pre-calculate things. Assume work.key_lo won't change it this loop
        ; (it's pretty safe to assume that, because we're working on 28 bits
        ; blocks)
        ; It means also that %%ebx == %%esi (Llo1 == Llo2)

k7align 16
_bigger_loop_k7:
        add     ebx, S0_ROTL3
        rol     ebx, FIRST_ROTL
        mov     [work_pre1_r1],ebx

        lea     eax,[S1_S0_ROTL3+ebx]
        rol     eax,3
        mov     [work_pre2_r1],eax

        lea     ecx, [eax+ebx]
        mov     [work_pre3_r1], ecx

;k7align 16
_loaded_k7:
    ; ------------------------------
    ; Begin round 1 of key expansion
    ; ------------------------------

        mov     ecx, [work_pre3_r1]
        mov     eax, [work_pre2_r1]

        add     edx, ecx
        add     edi, ecx
        mov     ebp, eax

        mov     ebx, [work_pre1_r1]

        mov     S1(1),eax
        ;mov     S2(1),ebp               ; --> att in round 2 !

        mov     esi, ebx
        add     eax,S_not(2)
        rol     edx,cl
        add     ebp,S_not(2)
        rol     edi,cl
        add     eax,edx
        rol     eax,3

        ROUND   1,2,24

;_round_1_last:
        mov     S1(25),eax

        add     ebp, esi
        lea     ecx,[eax+ebx]

        add     eax,S0_ROTL3
        rol     ebp,3
        add     edx,ecx

        mov     S2(25),ebp
        rol     edx,cl

        lea     ecx,[ebp+esi]
        add     eax,edx
        add     edi,ecx

        rol     eax,3
        add     ebp,S0_ROTL3
        rol     edi,cl

;_end_round1_k7:
    ; ------------------------------
    ; Begin round 2 of key expansion
    ; ------------------------------
        add     ebp,edi

        mov     S1(0),eax
        lea     ecx,[eax+edx]

        add     eax,S1(1)
        rol     ebp,3
        add     ebx,ecx

        mov     S2(0),ebp
        rol     ebx,cl

        lea     ecx,[ebp+edi]

        add     ebp,S1(1)              ;!!att S2(1) is not filled
        add     eax,ebx
        add     esi,ecx

        rol     eax,3
        rol     esi,cl

        ROUND   2,1,24

;_round_2_last:
        add     ebp,esi
        mov     S1(25),eax
        lea     ecx,[eax+ebx]

        mov     [work_key2_esi],esi
        rol     ebp,3
        add     edx,ecx

        mov     S2(25),ebp
        rol     edx,cl
        lea     ecx,[ebp+esi]

        add     eax,S1(0)
        add     edi,ecx

        mov     esi,[work_P_0]
        rol     edi,cl


;_end_round2_k7:
        mov     [work_key2_edi], edi

    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------
    ; (first key)

        lea     ebp, S1(0)              ;

        ; A  = %eax  eA = %esi
        ; L0 = %ebx  eB = %edi
        ; L1 = %edx  .. = %ebp


        ; A = ROTL3(S00 + A + L1);
        ; eA = P_0 + A;
        ; L0 = ROTL(L0 + A + L1, A + L1);

        add     eax,edx
        mov     ecx,edx

        rol     eax,3

        add     esi,eax
        add     ecx,eax
        add     eax,S3(1)

        add     ebx,ecx
        mov     edi,[work_P_1]

        rol     ebx,cl

        ; A = ROTL3(S01 + A + L0);
        ; eB = P_1 + A;
        ; L1 = ROTL(L1 + A + L0, A + L0);

        add     eax,ebx
        mov     ecx,ebx

        rol     eax,3

        add     edi,eax
        add     ecx,eax
        add     eax,S3(2)

        add     edx,ecx
        xor     esi,edi

        rol     edx,cl

        ROUND   3,2,23

        ; early exit

;_end_round3_1_k7:
        ;A = ROTL3(S24 + A + L1);
        ;eA = ROTL(eA ^ eB, eB) + A;

        add     eax,edx
        rol     eax,3

        mov     ecx,edi
        rol     esi,cl

        add     esi,eax

        cmp     esi,[work_C_0]
        je      near _checkKey1High_k7

__exit_1_k7:
    ; ---------------------------------------------------- */
    ; Begin round 3 of key expansion mixed with encryption */
    ; ---------------------------------------------------- */
    ; (second key)                                          */

        ; A  = %eax  eA = %esi
        ; L0 = %ebx  eB = %edi
        ; L1 = %edx  .. = %ebp


        ; A = ROTL3(S00 + A + L1);
        ; eA = P_0 + A;
        ; L0 = ROTL(L0 + A + L1, A + L1);

        mov     eax,S2(25)
        mov     edx,[work_key2_edi]
        lea     ebp,S2(0)

        add     eax,edx
        add     eax,S2(0)

        mov     ebx,[work_key2_esi]
        rol     eax,3

        mov     esi,[work_P_0]
        lea     ecx,[eax+edx]
        add     esi,eax

        add     ebx,ecx
        rol     ebx,cl

        ; A = ROTL3(S01 + A + L0);
        ; eB = P_1 + A;
        ; L1 = ROTL(L1 + A + L0, A + L0);

        add     eax,S3(1)
        add     eax,ebx

        mov     edi,[work_P_1]
        rol     eax,3

        mov     ecx,ebx
        add     edi,eax
        add     ecx,eax

        add     edx,ecx
        rol     edx,cl

        add     eax,S3(2)
        xor     esi,edi

        ROUND   3,2,23

        ; late exit

;_end_round3_2_k7:
        ;A = ROTL3(S24 + A + L1);
        ;eA = ROTL(eA ^ eB, eB) + A;
        add     eax,edx
        rol     eax,3

        mov     ecx,edi
        rol     esi,cl

        add     esi,eax

        cmp     esi,[work_C_0]
        je      near _checkKey2High_k7

__exit_2_k7:
        mov     edx,[work_key_hi]
        bswap   edx
        add     edx,2
        bswap   edx
        lea     edi,[edx+0x01000000]
        jc      short _next_inc_k7

;k7align 4
_next_iter_k7:
        mov     [work_key_hi],edx
        dec     dword [work_iterations]
        jg      near _loaded_k7

        mov     eax,[RC5UnitWork]                   ; pointer to rc5unitwork
        mov     ebx,[work_key_lo]
        mov     [RC5UnitWork_L0hi],edx              ; (used by caller)
        mov     [RC5UnitWork_L0lo],ebx              ; Update real data
        jmp     _full_exit_k7

k7align 16
_next_inc_k7:
        mov     ebx, [work_key_lo]

        bswap   ebx
        inc     ebx
        bswap   ebx

;k7align 4
_next_iter2_k7:
        mov     [work_key_hi],edx
        mov     [work_key_lo],ebx
        mov     esi, ebx
        dec     dword [work_iterations]
        jg      near _bigger_loop_k7

        mov     eax, [RC5UnitWork]                  ; pointer to rc5unitwork
        mov     [RC5UnitWork_L0lo],ebx              ; Update real data
        mov     [RC5UnitWork_L0hi],edx              ; (used by caller)
        jmp     short _full_exit_k7

k7align 16
_checkKey1High_k7:
        ;L0 = ROTL(L0 + A + L1, A + L1);
        ;A = ROTL3(S25 + A + L0);
        ;eB = ROTL(eB ^ eA, eA) + A;

        mov     ecx,edx
        add     ecx,eax
        add     eax,S3(25)

        add     ebx,ecx
        xor     edi,esi

        rol     ebx,cl

        mov     ecx,esi
        add     eax,ebx

        rol     edi,cl
        rol     eax,3
        add     edi,eax

        cmp     edi,[work_C_1]
        jne     near __exit_1_k7
        jmp     _full_exit_k7

k7align 16
_checkKey2High_k7:
        ;L0 = ROTL(L0 + A + L1, A + L1);
        ;A = ROTL3(S25 + A + L0);
        ;eB = ROTL(eB ^ eA, eA) + A;

        mov     ecx,edx
        add     ecx,eax
        add     eax,S3(25)

        add     ebx,ecx
        xor     edi,esi

        rol     ebx,cl

        mov     ecx,esi
        add     eax,ebx

        rol     edi,cl
        rol     eax,3
        add     edi,eax

        cmp     edi, [work_C_1]
        jne     near __exit_2_k7

        mov     ebp, [timeslice]
        sub     ebp, [work_iterations]
        lea     eax, [ebp*2+1]
        jmp     short _rest_reg_k7

k7align 16
_full_exit_k7:
        mov     eax, [timeslice]
        sub     eax, [work_iterations]
        shl     eax,1

;    return (timeslice - work.iterations) * 2 + work.add_iter;
_rest_reg_k7:
        mov ebx, [save_ebx]
        mov esi, [save_esi]
        mov edi, [save_edi]
        mov ebp, [save_ebp]

        add esp, opt_work_size

        ret


