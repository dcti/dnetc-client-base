; Self-modifying core for i386/i486 with adaptations for use with DPMI hosts
; by Bruce Ford, based in part on Remi's smc core.
; Slight improvement over RG smc core. 104 kkeys/s on a 486/DX4-100
;
; $Id: brf-smc.asm,v 1.1.2.2 2001/04/14 13:41:39 cyp Exp $

%macro calign 1  ; code align macro (arg = 'align to')
                 ; [nasm's integral 'align' statement blindly inserts 'nop']
  %assign sz  0
  %if %1 > 0
    %%szx equ ($ - $$)
    %assign sz (%%szx & (%1 - 1))
    %if sz != 0
      %assign sz %1 - sz
    %endif
  %endif
  %assign edinext 0
  %rep %1
    %assign edinext 0
    %if sz >= 7
      db 0x8D,0xB4,0x26,0x00,0x00,0x00,0x00  ; lea       esi,[esi]
      %assign sz sz-7
      %assign edinext 1
    %elif sz >= 6 && edinext != 0
      db 0x8d,0xBf,0x00,0x00,0x00,0x00       ; lea       edi,[edi]
      %assign edinext 0
      %assign sz sz-6
    %elif sz >= 6
      db 0x8D,0xB6,0x00,0x00,0x00,0x00       ; lea       esi,[esi]
      %assign edinext 1
      %assign sz sz-6
    %elif sz >= 4   
      db 0x8D,0x74,0x26,0x00                 ; lea       esi,[esi]
      %assign sz sz-4
      %assign edinext 1
    %elif sz >= 3 && edinext != 0
      db 0x8d,0x7f,0x00                      ; lea       edi,[edi]
      %assign sz sz-3
      %assign edinext 0
    %elif sz >= 3
      db 0x8D,0x76,0x00                      ; lea       esi,[esi] 
      %assign sz sz-3
      %assign edinext 1
    %elif sz >= 2 && edinext != 0
      db 0x8d,0x3f                           ; lea       edi,[edi]
      %assign sz sz-2
      %assign edinext 0
    %elif sz >= 2
      ;db 0x8D,0x36                          ; gas 2.7: lea esi,[esi]     
      mov esi,esi                            ; gas 2.9: mov esi,esi   
      %assign sz sz-2
      %assign edinext 1
    %elif sz >= 1
      nop
      %assign sz sz-1
    %else 
      %exitrep
    %endif
  %endrep  
%endmacro


%define work_size       56

%define RC5UnitWork     esp+work_size+4
%define timeslice       esp+work_size+8


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

%define save_ebp        esp+0
%define save_edi        esp+4
%define save_esi        esp+8
%define save_ebx        esp+12
%define save_ds         esp+16

%define work_add_iter   esp+20
%define work_key2_edi   esp+4+20
%define work_key2_esi   esp+8+20
%define work_key_hi     esp+12+20
%define work_key_lo     esp+16+20
%define work_iterations esp+20+20
%define work_pre1_r1    esp+24+20
%define work_pre2_r1    esp+28+20
%define work_pre3_r1    esp+32+20

; Offsets to access RC5UnitWork fields

%define RC5UnitWork_plainhi   ecx+0
%define RC5UnitWork_plainlo   ecx+4
%define RC5UnitWork_cipherhi  ecx+8
%define RC5UnitWork_cipherlo  ecx+12
%define RC5UnitWork_L0hi      ecx+16
%define RC5UnitWork_L0lo      ecx+20


  ; A1   = %eax  A2   = %ebp
  ; Llo1 = %ebx  Llo2 = %esi
  ; Lhi1 = %edx  Lhi2 = %edi

; ------------------------------------------------------------------
; S1(N) = A1 = ROTL3 (A1 + Lhi1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Lhi2 + S_not(N));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro ROUND_1_EVEN 1
        lea     eax, [S_not(%1)+eax+edx]        ; 2 : 3
        lea     ebp, [S_not(%1)+ebp+edi]        ; 2 : 3
        rol     eax, 3                          ; 2 : 3
        mov     [_modif486_r2_S1_%1+3], eax     ; 1 : 2
        rol     ebp, 3                          ; 2 : 3
        mov     [_modif486_r2_S2_%1+3], ebp     ; 1 : 2
        lea     ecx, [eax+edx]                  ; 2 : 2
        add     ebx, ecx                        ; 1 : 2
        rol     ebx, cl                         ; 3 : 3
        lea     ecx, [ebp+edi]                  ; 2 : 2
        add     esi, ecx                        ; 1 : 2
        rol     esi, cl                         ; 3 : 3     sum = 22 : 30
%endmacro
; S1(N) = A1 = ROTL3 (A1 + Llo1 + S_not(N));
; S2(N) = A2 = ROTL3 (A2 + Llo2 + S_not(N));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_1_ODD 1
        lea     eax, [S_not(%1)+eax+ebx]        ; 2 : 3
        lea     ebp, [S_not(%1)+ebp+esi]        ; 2 : 3
        rol     eax, 3                          ; 2 : 3
        mov     [_modif486_r2_S1_%1+3], eax     ; 1 : 2
        rol     ebp, 3                          ; 2 : 3
        mov     [_modif486_r2_S2_%1+3], ebp     ; 1 : 2
        lea     ecx, [eax+ebx]                  ; 2 : 2
        add     edx, ecx                        ; 1 : 2
        rol     edx, cl                         ; 3 : 3
        lea     ecx, [ebp+esi]                  ; 2 : 2
        add     edi, ecx                        ; 1 : 2
        rol     edi, cl                         ; 3 : 3     sum = 22 : 30
%endmacro

%macro  ROUND_1_EVEN_AND_ODD 2
        ROUND_1_EVEN %1
        ROUND_1_ODD  %2
%endmacro

; ------------------------------------------------------------------
; S1N = A1 = ROTL3 (A1 + Lhi1 + S1N);
; S2N = A2 = ROTL3 (A2 + Lhi2 + S2N);
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; Llo2 = ROTL (Llo2 + A2 + Lhi2, A2 + Lhi2);
%macro  ROUND_2_EVEN 1
        rol     eax, 3                          ; 2 : 3
        mov     [_modif486_r3_S1_%1+3], eax     ; 1 : 2
        lea     ecx, [eax+edx]                  ; 2 : 2
_modif486_r2_S2_%1:
        lea     ebp, [ebp+edi+0x90abcdef]       ; 2 : 3
        add     ebx, ecx                        ; 1 : 2
        rol     ebx, cl                         ; 3 : 3
        rol     ebp, 3                          ; 2 : 3
        mov     [_modif486_r3_S2_%1+3], ebp     ; 1 : 2
        lea     ecx, [ebp+edi]                  ; 2 : 2
        add     esi, ecx                        ; 1 : 2
        rol     esi, cl                         ; 3 : 3     sum = 20 : 27
%endmacro

; S1N = A1 = ROTL3 (A1 + Llo1 + S1N);
; S2N = A2 = ROTL3 (A2 + Llo2 + S2N);
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);
; Lhi2 = ROTL (Lhi2 + A2 + Llo2, A2 + Llo2);
%macro ROUND_2_ODD 2
_modif486_r2_S1_%1:
        lea     eax, [eax+ebx+0x12345678]       ; 2 : 3
        rol     eax, 3                          ; 2 : 3
        mov     [_modif486_r3_S1_%1+3], eax     ; 1 : 2
        lea     ecx, [eax+ebx]                  ; 2 : 2
        add     edx, ecx                        ; 1 : 2
_modif486_r2_S2_%1:
        lea     ebp, [ebp+esi+0x90abcdef]       ; 2 : 3
        rol     edx, cl                         ; 3 : 3
        rol     ebp, 3                          ; 2 : 3
        mov     [_modif486_r3_S2_%1+3], ebp     ; 1 : 2
        lea     ecx, [ebp+esi]                  ; 2 : 2
        add     edi, ecx                        ; 1 : 2
_modif486_r2_S1_%2:
        lea     eax, [eax+edx+0x12345678]       ; 2 : 3
        rol     edi, cl                         ; 3 : 3     sum = 24 : 33
%endmacro

%macro  ROUND_2_EVEN_AND_ODD 3
        ROUND_2_EVEN %1
        ROUND_2_ODD  %2, %3
%endmacro

%macro ROUND_2_ODD_LAST 1
_modif486_r2_S1_%1:
        lea     eax, [eax+ebx+0x12345678]       ; 2 : 3
        rol     eax, 3                          ; 2 : 3
        mov     [_modif486_r3_S1_%1+3], eax     ; 1 : 2
        lea     ecx, [eax+ebx]                  ; 2 : 2
        add     edx, ecx                        ; 1 : 2
_modif486_r2_S2_%1:
        lea     ebp, [ebp+esi+0x90abcdef]       ; 2 : 3
        rol     edx, cl                         ; 3 : 3
        rol     ebp, 3                          ; 2 : 3
        mov     [_modif486_r3_S2_%1+3], ebp     ; 1 : 2
        lea     ecx, [ebp+esi]                  ; 2 : 2
        add     edi, ecx                        ; 1 : 2
        rol     edi, cl                         ; 3 : 3     sum = 22 : 30
%endmacro

%macro  ROUND_2_EVEN_AND_ODD_LAST 2
        ROUND_2_EVEN     %1
        ROUND_2_ODD_LAST %2
%endmacro

; ------------------------------------------------------------------
; It's faster to do 1 key at a time with round3 and encryption mixed
; than to do 2 keys at once but round3 and encryption separated
; Too bad x86 hasn't more registers ...

; eA1 = ROTL (eA1 ^ eB1, eB1) + (A1 = ROTL3 (A1 + Lhi1 + S1(N)));
; Llo1 = ROTL (Llo1 + A1 + Lhi1, A1 + Lhi1);
; eB1 = ROTL (eA1 ^ eB1, eA1) + (A1 = ROTL3 (A1 + Llo1 + S1(N)));
; Lhi1 = ROTL (Lhi1 + A1 + Llo1, A1 + Llo1);

; A  = %eax  eA = %esi
; L0 = %ebx  eB = %edi
; L1 = %edx  .. = %ebp

%macro ROUND_3_EVEN_AND_ODD 3
        mov     ecx, edi                        ; 1 : 2
_modif486_r3_%{1}_%{2}:
        lea     ebp, [ebp+edx+0x12345678]       ; 2 : 3
        rol     ebp, 3                          ; 2 : 3
        xor     esi, edi                        ; 1 : 2
        rol     esi, cl                         ; 3 : 3
        add     esi, ebp                        ; 1 : 2
        lea     ecx, [ebp+edx]                  ; 2 : 2
        add     ebx, ecx                        ; 1 : 2
        rol     ebx, cl                         ; 3 : 3

        mov     ecx, esi                        ; 1 : 2
        xor     edi, esi                        ; 1 : 2
_modif486_r3_%{1}_%{3}:
        lea     ebp, [ebp+ebx+0x12345678]       ; 2 : 3
        rol     ebp, 3                          ; 2 : 3
        rol     edi, cl                         ; 3 : 3
        add     edi, ebp                        ; 1 : 2
        lea     ecx, [ebp+ebx]                  ; 2 : 2
        add     edx, ecx                        ; 1 : 2
        rol     edx, cl                         ; 3 : 3     sum = 32 : 46
%endmacro

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _TEXT FLAT USE32 ALIGN=16 CLASS=CODE]
%else
[SECTION .text]
%endif

%ifdef USE_DPMI
[GLOBAL smc_dpmi_ds_alias_alloc]
[GLOBAL _smc_dpmi_ds_alias_alloc]
[GLOBAL smc_dpmi_ds_alias_free]
[GLOBAL _smc_dpmi_ds_alias_free]
extern rc5_unit_func_486  ; where to jump to if no DS alias

align 4
dpmi_ds_alias_ref_count dd 0
dpmi_ds_alias           dw 0

align 4
_smc_dpmi_ds_alias_alloc:
smc_dpmi_ds_alias_alloc:
        xor     eax, eax
        mov     ax, [cs:dpmi_ds_alias]
        or      ax, ax
        jnz     _dsala1 
;       Alias the code selector to the data selector using DPMI
        push    ebx
        xor     ebx, ebx
        mov     bx, cs
        mov     eax, 0x0a
        int     0x31
        pop     ebx
        jc      _dsala2
_dsala1:push    ds
        mov     ds,ax
        mov     [dpmi_ds_alias],ax
        inc     dword [dpmi_ds_alias_ref_count]
        pop     ds
_dsala2:mov     eax,[cs:dpmi_ds_alias_ref_count]
        ret

_smc_dpmi_ds_alias_free:
smc_dpmi_ds_alias_free:
        mov     ax, [cs:dpmi_ds_alias]
        or      ax, ax
        jz      _dsalf2 
        push    ds
        mov     ds, ax
        mov     eax,[dpmi_ds_alias_ref_count]
        dec     eax
        mov     [dpmi_ds_alias_ref_count],eax
        jnz     _dsalf1
        xchg    ax, [dpmi_ds_alias] 
        push    ebx
        mov     ebx, eax
        mov     eax, 1
        int     0x31
        pop     ebx 
        xor     eax,eax
_dsalf1:pop     ds 
_dsalf2:ret
%endif

; ------------------------------------------------------------------
; rc5_unit will get passed an RC5WorkUnit to complete
; this is where all the actually work occurs, this is where you optimize.
; assembly gurus encouraged.
; Returns number of keys checked before a possible good key is found, or
; timeslice*PIPELINE_COUNT if no keys are 'good' keys.
; (ie:      if (result == timeslice*PIPELINE_COUNT) NOTHING_FOUND
;      else if (result < timeslice*PIPELINE_COUNT) SOMETHING_FOUND at result+1
;      else SOMETHING_GET_WRONG... )

[GLOBAL _rc5_unit_func_486_smc]
[GLOBAL rc5_unit_func_486_smc]

align 4
_rc5_unit_func_486_smc:
rc5_unit_func_486_smc:
;u32 rc5_unit_func_486_smc( RC5UnitWork * rc5unitwork, u32 timeslice )

        sub esp, work_size                      ; set up stack

%ifdef USE_DPMI
        mov  ax,[cs:dpmi_ds_alias]
        or   ax,ax
        jnz  _have_dpmi_ds
        add  esp, work_size
        jmp  rc5_unit_func_486 
_have_dpmi_ds:
        mov  [save_ds], ds
        mov  ds,ax
%endif

        mov [save_ebp], ebp                     ; save registers
        mov [save_edi], edi
        mov [save_esi], esi
        mov [save_ebx], ebx

        mov ebp, [timeslice]

        mov dword [work_add_iter], 0x00000000

        mov [work_iterations], ebp


        mov     ecx, [RC5UnitWork]              ; load pointer to rc5unitwork into ecx

        ; load parameters
        mov     ebx, [RC5UnitWork_L0lo]         ; ebx = l0 = Llo1
        mov     edx, [RC5UnitWork_L0hi]         ; edx = l1 = Lhi1
        mov     esi, ebx                        ; esi = l2 = Llo2
        lea     edi, [0x01000000+edx]           ; edi = l3 = lhi2
        mov     [work_key_lo], ebx
        mov     [work_key_hi], edx

        ; Save other parameters
        ; (it's faster to do so, since we will only load 1 value
        ; each time in RC5_ROUND_3xy, instead of two if we save
        ; only the pointer to the RC5 struct)
        mov     ebp, [RC5UnitWork_plainlo]
        mov     [_modif486_work_P_0_1+2], ebp
        mov     [_modif486_work_P_0_2+2], ebp
        mov     ebp, [RC5UnitWork_plainhi]
        mov     [_modif486_work_P_1_1+2], ebp
        mov     [_modif486_work_P_1_2+2], ebp
        mov     ebp, [RC5UnitWork_cipherlo]
        mov     [_modif486_work_C_0_1+2], ebp
        mov     [_modif486_work_C_0_2+2], ebp
        mov     ebp, [RC5UnitWork_cipherhi]
        mov     [_modif486_work_C_1_1+2], ebp
        mov     [_modif486_work_C_1_2+2], ebp

        ; Pre-calculate things. Assume work.key_lo won't change in this loop
        ; (it's pretty safe to assume that, because we're working on 28 bits
        ; blocks)
        ; It means also that %%ebx == %%esi (Llo1 == Llo2)

calign 4
_bigger_loop_486:
        add     ebx, S0_ROTL3                   ; 1
        rol     ebx, FIRST_ROTL                 ; 3
        mov     [work_pre1_r1], ebx             ; 1

        lea     eax, [S1_S0_ROTL3+ebx]          ; 1
        rol     eax, 3                          ; 2
        mov     [work_pre2_r1], eax             ; 1

        lea     ecx, [eax+ebx]                  ; 2
        mov     [work_pre3_r1], ecx             ; 1

        lea     esi,[esi+1]                     ; 1         Alignment 

calign 4
_loaded_486:
    ; ------------------------------
    ; Begin round 1 of key expansion
    ; ------------------------------

        mov     ebx, [work_pre1_r1]             ; 1 : 4
        mov     esi, ebx                        ; 1 : 2
        mov     eax, [work_pre2_r1]             ; 1 : 4
        mov     ebp, eax                        ; 1 : 2

        mov     ecx, [work_pre3_r1]             ; 1 : 4
        add     edx, ecx                        ; 1 : 2
        rol     edx, cl                         ; 3 : 3
        add     edi, ecx                        ; 1 : 2
        rol     edi, cl                         ; 3 : 3     sum = 13 : 26

        ROUND_1_EVEN_AND_ODD  2, 3              ; 44 : 60   ( 12 times )
        ROUND_1_EVEN_AND_ODD  4, 5
        ROUND_1_EVEN_AND_ODD  6, 7
        ROUND_1_EVEN_AND_ODD  8, 9
        ROUND_1_EVEN_AND_ODD 10,11
        ROUND_1_EVEN_AND_ODD 12,13
        ROUND_1_EVEN_AND_ODD 14,15
        ROUND_1_EVEN_AND_ODD 16,17
        ROUND_1_EVEN_AND_ODD 18,19
        ROUND_1_EVEN_AND_ODD 20,21
        ROUND_1_EVEN_AND_ODD 22,23
        ROUND_1_EVEN_AND_ODD 24,25


    ; ------------------------------
    ; Begin round 2 of key expansion
    ; ------------------------------

_end_round1_486:
        lea     eax, [S0_ROTL3+eax+edx]         ; 2 : 3
        lea     ebp, [S0_ROTL3+ebp+edi]         ; 2 : 3
        rol     eax, 3                          ; 2 : 3
        mov     [_modif486_r3_S1_0+3], eax      ; 1 : 2
        rol     ebp, 3                          ; 2 : 3
        mov     [_modif486_r3_S2_0+3], ebp      ; 1 : 2

        mov     ecx, eax                        ; 1 : 2
        add     ecx, edx                        ; 1 : 2
        add     ebx, ecx                        ; 1 : 2
        rol     ebx, cl                         ; 3 : 3
        lea     ecx, [ebp+edi]                  ; 2 : 2
        add     esi, ecx                        ; 1 : 2
        rol     esi, cl                         ; 3 : 3

        mov     ecx, [work_pre2_r1]             ; 1 : 4
        add     eax, ebx                        ; 1 : 2
        add     eax, ecx                        ; 1 : 2
        add     ebp, esi                        ; 1 : 2
        add     ebp, ecx                        ; 1 : 2
        rol     eax, 3                          ; 2 : 3
        mov     [_modif486_r3_S1_1+3], eax      ; 1 : 2
        rol     ebp, 3                          ; 2 : 3
        mov     [_modif486_r3_S2_1+3], ebp      ; 1 : 2
        lea     ecx, [eax+ebx]                  ; 2 : 2
        add     edx, ecx                        ; 1 : 2
        rol     edx, cl                         ; 3 : 3
        lea     ecx, [ebp+esi]                  ; 2 : 2
        add     edi, ecx                        ; 1 : 2
_modif486_r2_S1_2:
        lea     eax, [eax+edx+0x12345678]       ; 2 : 3
        rol     edi, cl                         ; 3 : 3     sum = 47 : 71

        ROUND_2_EVEN_AND_ODD       2, 3, 4      ; 44 : 60   ( 11 times )
        ROUND_2_EVEN_AND_ODD       4, 5, 6
        ROUND_2_EVEN_AND_ODD       6, 7, 8
        ROUND_2_EVEN_AND_ODD       8, 9,10
        ROUND_2_EVEN_AND_ODD      10,11,12
        ROUND_2_EVEN_AND_ODD      12,13,14
        ROUND_2_EVEN_AND_ODD      14,15,16
        ROUND_2_EVEN_AND_ODD      16,17,18
        ROUND_2_EVEN_AND_ODD      18,19,20
        ROUND_2_EVEN_AND_ODD      20,21,22
        ROUND_2_EVEN_AND_ODD      22,23,24
        ROUND_2_EVEN_AND_ODD_LAST 24,25         ; 42 : 57

_end_round2_486:
        mov     [work_key2_esi], esi            ; 1 : 2
        mov     [work_key2_edi], edi            ; 1 : 2

    ; ----------------------------------------------------
    ; Begin round 3 of key expansion mixed with encryption
    ; ----------------------------------------------------
    ; (first key)

        ; .. = %eax  eA = %esi
        ; L0 = %ebx  eB = %edi
        ; L1 = %edx  A  = %ebp

        mov     ecx, edx                        ; 1 : 2
_modif486_r3_S1_0:
        lea     ebp, [eax+edx+0x12345678]       ; 2 : 3     A = ROTL3(S00 + A + L1);
        rol     ebp, 3                          ; 2 : 3
        add     ecx, ebp                        ; 1 : 2     L0 = ROTL(L0 + A + L1, A + L1);
_modif486_work_P_0_1:
        lea     esi, [ebp+0x12345678]           ; 1 : 2     eA = P_0 + A;
        add     ebx, ecx                        ; 1 : 2
        rol     ebx, cl                         ; 3 : 3

        mov     ecx, ebx                        ; 1 : 2
_modif486_r3_S1_1:
        lea     ebp, [ebp+ebx+0x90abcdef]       ; 2 : 3     A = ROTL3(S01 + A + L0);
        rol     ebp, 3                          ; 2 : 3
        add     ecx, ebp                        ; 1 : 2     L1 = ROTL(L1 + A + L0, A + L0);
_modif486_work_P_1_1:
        lea     edi, [ebp+0x12345678]           ; 1 : 2     eB = P_1 + A;
        add     edx, ecx                        ; 1 : 2
        rol     edx, cl                         ; 3 : 3     sum = 24 : 38

        ROUND_3_EVEN_AND_ODD S1, 2, 3           ; 32 : 46   ( 11 times )
        ROUND_3_EVEN_AND_ODD S1, 4, 5
        ROUND_3_EVEN_AND_ODD S1, 6, 7
        ROUND_3_EVEN_AND_ODD S1, 8, 9
        ROUND_3_EVEN_AND_ODD S1,10,11
        ROUND_3_EVEN_AND_ODD S1,12,13
        ROUND_3_EVEN_AND_ODD S1,14,15
        ROUND_3_EVEN_AND_ODD S1,16,17
        ROUND_3_EVEN_AND_ODD S1,18,19
        ROUND_3_EVEN_AND_ODD S1,20,21
        ROUND_3_EVEN_AND_ODD S1,22,23

        ; early exit
_end_round3_1_486:
        mov     ecx, edi                        ; 1 : 2
_modif486_r3_S1_24:
        lea     ebp, [ebp+edx+0x12345678]       ; 2 : 3     A = ROTL3(S24 + A + L1);
        rol     ebp, 3                          ; 2 : 3
        xor     esi, edi                        ; 1 : 2     eA = ROTL(eA ^ eB, eB) + A
        rol     esi, cl                         ; 3 : 3
        add     esi, ebp                        ; 1 : 2

_modif486_work_C_0_1:
        cmp     esi, 0x12345678                 ; 1 : 2
        je near test_C_1_1                      ; 1 : 3    sum = 12 : 20

__exit_1_486:
    ; Restore 2nd key parameters */
        mov     ebp, [_modif486_r3_S2_25+3]     ; 1 : 4
        mov     edx, [work_key2_edi]            ; 1 : 4
        mov     ebx, [work_key2_esi]            ; 1 : 4

    ; ---------------------------------------------------- */
    ; Begin round 3 of key expansion mixed with encryption */
    ; ---------------------------------------------------- */
    ; (second key)                                          */

        ; .. = %eax  eA = %esi
        ; L0 = %ebx  eB = %edi
        ; L1 = %edx  A  = %ebp

        mov     ecx, edx                        ; 1 : 2
_modif486_r3_S2_0:
        lea     ebp, [ebp+edx+0x12345678]       ; 2 : 3     A = ROTL3(S00 + A + L1);
        rol     ebp, 3                          ; 2 : 3
        add     ecx, ebp                        ; 1 : 2     L0 = ROTL(L0 + A + L1, A + L1);
_modif486_work_P_0_2:
        lea     esi, [ebp+0x12345678]           ; 1 : 2     eA = P_0 + A;
        add     ebx, ecx                        ; 1 : 2
        rol     ebx, cl                         ; 3 : 3

        mov     ecx, ebx                        ; 1 : 2
_modif486_r3_S2_1:
        lea     ebp, [ebp+ebx+0x90abcdef]       ; 2 : 3     A = ROTL3(S01 + A + L0);
        rol     ebp, 3                          ; 2 : 3
        add     ecx, ebp                        ; 1 : 2     L1 = ROTL(L1 + A + L0, A + L0);
_modif486_work_P_1_2:
        lea     edi, [ebp+0x12345678]           ; 1 : 2     eB = P_1 + A;
        add     edx, ecx                        ; 1 : 2
        rol     edx, cl                         ; 3 : 3     sum = 25 : 46

        ROUND_3_EVEN_AND_ODD S2, 2, 3           ; 32 : 46   ( 11 times )
        ROUND_3_EVEN_AND_ODD S2, 4, 5
        ROUND_3_EVEN_AND_ODD S2, 6, 7
        ROUND_3_EVEN_AND_ODD S2, 8, 9
        ROUND_3_EVEN_AND_ODD S2,10,11
        ROUND_3_EVEN_AND_ODD S2,12,13
        ROUND_3_EVEN_AND_ODD S2,14,15
        ROUND_3_EVEN_AND_ODD S2,16,17
        ROUND_3_EVEN_AND_ODD S2,18,19
        ROUND_3_EVEN_AND_ODD S2,20,21
        ROUND_3_EVEN_AND_ODD S2,22,23

        ; early exit
_end_round3_2_486:
        mov     ecx, edi                        ; 1 : 2
_modif486_r3_S2_24:
        lea     ebp, [ebp+edx+0x12345678]       ; 2 : 3     A = ROTL3(S24 + A + L1);
        rol     ebp, 3                          ; 2 : 3
        xor     esi, edi                        ; 1 : 2     eA = ROTL(eA ^ eB, eB) + A
        rol     esi, cl                         ; 3 : 3
        add     esi, ebp                        ; 1 : 2

_modif486_work_C_0_2:
        cmp     esi, 0x12345678                 ; 1 : 2
        je near test_C_1_2                      ; 1 : 3    sum = 12 : 20

__exit_2_486:
        mov     edx, [work_key_hi]              ; 2 : 4

; Jumps not taken are faster
        add     edx, 0x02000000                 ; 1 : 2
        jc near _next_inc_486                   ; 1 : 3

_next_iter_486:
        mov     [work_key_hi], edx              ; 1 : 2
        lea     edi, [0x01000000+edx]           ; 1 : 2
        dec     dword [work_iterations]         ; 3 : 6
        jg near _loaded_486                     ; 16?: 11   sum = 26 : 30

                                                ; Processor    i486 : i386
                                                ;              -----------
                                                ; Total clocks 1916 : 2700
                                                ;      per key  958 : 1350

        mov     ecx, [RC5UnitWork]              ; pointer to rc5unitwork
        mov     ebx, [work_key_lo]
        mov     [RC5UnitWork_L0lo], ebx         ; update real data
        mov     [RC5UnitWork_L0hi], edx         ; (used by caller)
        jmp     _full_exit_486

calign 4
_next_iter2_486:
        mov     [work_key_lo], ebx
        mov     [work_key_hi], edx
        lea     edi, [0x01000000+edx]
        mov     esi, ebx
        dec     dword [work_iterations]
        jg near _bigger_loop_486
        mov     ecx, [RC5UnitWork]              ; pointer to rc5unitwork
        mov     [RC5UnitWork_L0lo], ebx         ; update real data
        mov     [RC5UnitWork_L0hi], edx         ; (used by caller)
        jmp     _full_exit_486

calign 4
_next_inc_486:
        add     edx, 0x00010000
        test    edx, 0x00FF0000
        jnz near _next_iter_486

        add     edx, 0xFF000100
        test    edx, 0x0000FF00
        jnz near _next_iter_486

        add     edx, 0xFFFF0001
        test    edx, 0x000000FF
        jnz near _next_iter_486


        mov     ebx, [work_key_lo]

        sub     edx, 0x00000100
        add     ebx, 0x01000000
        jnc near _next_iter2_486

        add     ebx, 0x00010000
        test    ebx, 0x00FF0000
        jnz near _next_iter2_486

        add     ebx, 0xFF000100
        test    ebx, 0x0000FF00
        jnz near _next_iter2_486

        add     ebx, 0xFFFF0001
        test    ebx, 0x000000FF
        jnz near _next_iter2_486

        ; Moo !
        ; We have just finished checking the last key
        ; of the rc5-64 keyspace...
        ; Not much to do here, since we have finished the block ...
        jmp     _full_exit_486

        ; Test of second half of ciphertext for key 1
        calign 4 ; No alignment needed.  Just lucky.
test_C_1_1:
        mov     ecx, ebp                        ; 1
        add     ecx, edx                        ; 1     L0 = ROTL(L0 + A + L1, A + L1);
        add     ebx, ecx                        ; 1
        rol     ebx, cl                         ; 3
        mov     ecx, esi                        ; 1     eB = ROTL(eB ^ eA, eA) + A
_modif486_r3_S1_25:
        lea     ebp, [ebp+ebx+0x90abcdef]       ; 2     A = ROTL3(S25 + A + L0);
        rol     ebp, 3                          ; 2

        xor     edi, esi                        ; 1
        rol     edi, cl                         ; 3
        add     edi, ebp                        ; 1

_modif486_work_C_1_1:
        cmp     edi, 0x12345678                 ; 1
        je near _full_exit_486
        jmp     __exit_1_486


        ; Test of second half of ciphertext for key 2
        nop                                     ; Alignment
test_C_1_2:
        mov     ecx, ebp                        ; 1
        add     ecx, edx                        ; 1     L0 = ROTL(L0 + A + L1, A + L1);
        add     ebx, ecx                        ; 1
        rol     ebx, cl                         ; 3
        mov     ecx, esi                        ; 1
_modif486_r3_S2_25:
        lea     ebp, [ebp+ebx+0x90abcdef]       ; 1     A = ROTL3(S25 + A + L0);
        rol     ebp, 3                          ; 2
        xor     edi, esi                        ; 1     eB = ROTL(eB ^ eA, eA) + A
        rol     edi, cl                         ; 3
        add     edi, ebp                        ; 1

_modif486_work_C_1_2:
        cmp     edi, 0x12345678                 ; 1
        jne near __exit_2_486
        mov     dword [work_add_iter], 1

        lea     esi, [esi+1]                    ; Alignment

calign 4
_full_exit_486:
        mov     ebp, [timeslice]
        sub     ebp, [work_iterations]
        mov     edx, [work_add_iter]

        lea     eax, [edx+ebp*2]

;    return (timeslice - work.iterations) * 2 + work.add_iter;


        mov     ebx, [save_ebx]
        mov     esi, [save_esi]
        mov     edi, [save_edi]
        mov     ebp, [save_ebp]
%ifdef USE_DPMI
        mov     ds,  [save_ds]
%endif

        add     esp, work_size                  ; restore stack pointer

        ret
