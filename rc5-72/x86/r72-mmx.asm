;---------------------------------------------------------------------
;
;  RC5-72 core using MMX instructions
;  This core based on the jp-mmx core (RC5-64) by Jason Papadopoulos.
;
;  Author: Carsten Haustein (chaus@cs.uni-potsdam.de)
;
;---------------------------------------------------------------------
; $Id: r72-mmx.asm,v 1.3 2009/01/05 22:45:21 mfeiri Exp $


;
; some aliases
;
%define RC5_72UnitWork  esp+work_size+20
%define iterations      esp+work_size+24

%define RESULT_NOTHING  1
%define RESULT_FOUND    2

%define RC5_72UnitWork_plainhi   esi+0
%define RC5_72UnitWork_plainlo   esi+4
%define RC5_72UnitWork_cipherhi  esi+8
%define RC5_72UnitWork_cipherlo  esi+12
%define RC5_72UnitWork_L0hi      esi+16
%define RC5_72UnitWork_L0mid     esi+20
%define RC5_72UnitWork_L0lo      esi+24
%define RC5_72UnitWork_CMCcount  esi+28
%define RC5_72UnitWork_CMChi     esi+32
%define RC5_72UnitWork_CMCmid    esi+36
%define RC5_72UnitWork_CMClo     esi+40

;
; local data
;
%define align_value     8                 ; local data quadword aligned
;
%define L1              ebp+0             ; Arrange elements as "structure of
%define L2              L1+16             ; arrays"
%define L3              L2+16
%define STable          L3+16             ; Needs to be divisible by 8
%define L1_Init         STable+416        ; 26 elements * 16 byte
%define L2_Init         L1_Init+16
%define L3_Init         L2_Init+16
%define work_P_0        L3_Init+16
%define work_P_1        work_P_0+8
%define work_C_0        work_P_1+8
%define work_C_1        work_C_0+8
%define work_size       544+align_value   ; extra bytes for alignment


;
;macros for the RC5-72 MMX core
;
%macro keytable_round_1 5
   ; %1 where keytable[i] comes from
   ; %2 where keytable[i] goes to
   ; %3 where L[i] comes from
   ; %4 where L[i] goes to
   ; %5 where L[i+1] comes from

                          ; P   0     1     2     3     4     5     6     7
                          ;--------------------------------------------------
                          ;   A1:A0 x0:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f

   movq mm6, mm2          ; U A1:A0 x0:x0 +1:+0 x1:x0 A3:A2 B3:B2 +1:+0 00:1f
   punpckhdq mm3, mm3     ; V A1:A0 x0:x0 +1:+0 x1:x1 A3:A2 B3:B2 +1:+0 00:1f

   pand mm6, mm7          ; U A1:A0 x0:x0 +1:+0 x1:x1 A3:A2 B3:B2 00:s0 00:1f
   punpckhdq mm2, mm2     ; V A1:A0 x0:x0 +1:+1 x1:x1 A3:A2 B3:B2 00:s0 00:1f

   paddd mm5, mm4         ; V A1:A0 x0:x0 +1:+1 x1:x1 A3:A2 +3:+2 00:s0 00:1f
   psllq mm1, mm6         ; U A1:A0 B0:?? +1:+1 x1:x1 A3:A2 +3:+2       00:1f

   movq mm6, mm5          ; U A1:A0 B0:?? +1:+1 x1:x1 A3:A2 +3:+2 +3:+2 00:1f
   pand mm2, mm7          ; V A1:A0 B0:?? 00:s1 x1:x1 A3:A2 +3:+2 +3:+2 00:1f

   paddd mm5, [%3+8]      ; U A1:A0 B0:?? 00:s1 x1:x1 A3:A2 x3:x2 +3:+2 00:1f
   psllq mm3, mm2         ; V A1:A0 B0:??       B1:?? A3:A2 x3:x2 +3:+2 00:1f

   movq mm2, mm6          ; U A1:A0 B0:?? +3:+2 B1:?? A3:A2 x3:x2 +3:+2 00:1f
   punpckhdq mm1, mm3     ; V A1:A0 B1:B0 +3:+2       A3:A2 x3:x2 +3:+2 00:1f

   movq mm3, mm5          ; U A1:A0 B1:B0 +3:+2 x3:x2 A3:A2 x3:x2 +3:+2 00:1f
   punpckldq mm5, mm5     ; V A1:A0 B1:B0 +3:+2 x3:x2 A3:A2 x2:x2 +3:+2 00:1f

   pand mm2, mm7          ; U A1:A0 B1:B0 00:s2 x3:x2 A3:A2 x2:x2 +3:+2 00:1f
   punpckhdq mm6, mm6     ; V A1:A0 B1:B0 00:s2 x3:x2 A3:A2 x2:x2 +3:+3 00:1f

   paddd mm0, [%1]        ; U +1:+0 B1:B0 00:s2 x3:x2 A3:A2 x2:x2 +3:+3 00:1f
   punpckhdq mm3, mm3     ; V +1:+0 B1:B0 00:s2 x3:x3 A3:A2 x2:x2 +3:+3 00:1f

   pand mm6, mm7          ; U +1:+0 B1:B0 00:s2 x3:x3 A3:A2 x2:x2 00:s3 00:1f
   psllq mm5, mm2         ; V +1:+0 B1:B0       x3:x3 A3:A2 B2:?? 00:s3 00:1f

   paddd mm0, mm1         ; U x1:x0 B1:B0       x3:x3 A3:A2 B2:?? 00:s3 00:1f
   psllq mm3, mm6         ; V x1:x0 B1:B0       B3:?? A3:A2 B2:??       00:1f

   movq mm2, mm0          ; U x1:x0 B1:B0 x1:x0 B3:?? A3:A2 B2:??       00:1f
   pslld mm0, 3           ; V l1:l0 B1:B0 x1:x0 B3:?? A3:A2 B2:??       00:1f

   movq [%4], mm1         ; U l1:l0 B1:B0 x1:x0 B3:?? A3:A2 B2:??       00:1f
   punpckhdq mm5, mm3     ; V l1:l0 B1:B0 x1:x0       A3:A2 B3:B2       00:1f

   paddd mm4, [%1]        ; U l1:l0 B1:B0 x1:x0       +3:+2 B3:B2       00:1f
   psrld mm2, 29          ; V l1:l0 B1:B0 r1:r0       +3:+2 B3:B2       00:1f

   movq [%4+8], mm5       ; U l1:l0 B1:B0 r1:r0       +3:+2 B3:B2       00:1f
   paddd mm4, mm5         ; U l1:l0 B1:B0 r1:r0       x3:x2 B3:B2       00:1f

   por mm0, mm2           ; U A1:A0 B1:B0             x3:x2 B3:B2       00:1f
   movq mm6, mm4          ; V A1:A0 B1:B0             x3:x2 B3:B2 x3:x2 00:1f

   paddd mm1, mm0         ; U A1:A0 +1:+0             x3:x2 B3:B2 x3:x2 00:1f
   pslld mm4, 3           ; V A1:A0 +1:+0             l3:l2 B3:B2 x3:x2 00:1f

   movq mm2, mm1          ; U A1:A0 +1:+0 +1:+0       l3:l2 B3:B2 x3:x2 00:1f
   psrld mm6, 29          ; V A1:A0 +1:+0 +1:+0       l3:l2 B3:B2 r3:r2 00:1f

   paddd mm1, [%5]        ; U A1:A0 x1:x0 +1:+0       l3:l2 B3:B2 r3:r2 00:1f
   por mm4, mm6           ; V A1:A0 x1:x0 +1:+0       A3:A2 B3:B2       00:1f

   movq [%2], mm0         ; U A1:A0 x1:x0 +1:+0       A3:A2 B3:B2       00:1f
   movq mm3, mm1          ; V A1:A0 x1:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f

   movq [%2+8], mm4       ; U A1:A0 x1:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f
   punpckldq mm1, mm1     ; V A1:A0 x0:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f
%endmacro

%macro keytable_round 3
   ; %1 where keytable[i] comes from and goes to
   ; %2 where L[i] comes from and goes to
   ; %3 where L[i+1] comes from

                          ; P   0     1     2     3     4     5     6     7
                          ;--------------------------------------------------
                          ;   A1:A0 x0:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f

   movq mm6, mm2          ; U A1:A0 x0:x0 +1:+0 x1:x0 A3:A2 B3:B2 +1:+0 00:1f
   punpckhdq mm3, mm3     ; V A1:A0 x0:x0 +1:+0 x1:x1 A3:A2 B3:B2 +1:+0 00:1f

   pand mm6, mm7          ; U A1:A0 x0:x0 +1:+0 x1:x1 A3:A2 B3:B2 00:s0 00:1f
   punpckhdq mm2, mm2     ; V A1:A0 x0:x0 +1:+1 x1:x1 A3:A2 B3:B2 00:s0 00:1f

   paddd mm5, mm4         ; V A1:A0 x0:x0 +1:+1 x1:x1 A3:A2 +3:+2 00:s0 00:1f
   psllq mm1, mm6         ; U A1:A0 B0:?? +1:+1 x1:x1 A3:A2 +3:+2       00:1f

   movq mm6, mm5          ; U A1:A0 B0:?? +1:+1 x1:x1 A3:A2 +3:+2 +3:+2 00:1f
   pand mm2, mm7          ; V A1:A0 B0:?? 00:s1 x1:x1 A3:A2 +3:+2 +3:+2 00:1f

   paddd mm5, [%2+8]      ; U A1:A0 B0:?? 00:s1 x1:x1 A3:A2 x3:x2 +3:+2 00:1f
   psllq mm3, mm2         ; V A1:A0 B0:??       B1:?? A3:A2 x3:x2 +3:+2 00:1f

   movq mm2, mm6          ; U A1:A0 B0:?? +3:+2 B1:?? A3:A2 x3:x2 +3:+2 00:1f
   punpckhdq mm1, mm3     ; V A1:A0 B1:B0 +3:+2       A3:A2 x3:x2 +3:+2 00:1f

   movq mm3, mm5          ; U A1:A0 B1:B0 +3:+2 x3:x2 A3:A2 x3:x2 +3:+2 00:1f
   punpckldq mm5, mm5     ; V A1:A0 B1:B0 +3:+2 x3:x2 A3:A2 x2:x2 +3:+2 00:1f

   pand mm2, mm7          ; U A1:A0 B1:B0 00:s2 x3:x2 A3:A2 x2:x2 +3:+2 00:1f
   punpckhdq mm6, mm6     ; V A1:A0 B1:B0 00:s2 x3:x2 A3:A2 x2:x2 +3:+3 00:1f

   paddd mm0, [%1]        ; U +1:+0 B1:B0 00:s2 x3:x2 A3:A2 x2:x2 +3:+3 00:1f
   punpckhdq mm3, mm3     ; V +1:+0 B1:B0 00:s2 x3:x3 A3:A2 x2:x2 +3:+3 00:1f

   pand mm6, mm7          ; U +1:+0 B1:B0 00:s2 x3:x3 A3:A2 x2:x2 00:s3 00:1f
   psllq mm5, mm2         ; V +1:+0 B1:B0       x3:x3 A3:A2 B2:?? 00:s3 00:1f

   paddd mm0, mm1         ; U x1:x0 B1:B0       x3:x3 A3:A2 B2:?? 00:s3 00:1f
   psllq mm3, mm6         ; V x1:x0 B1:B0       B3:?? A3:A2 B2:??       00:1f

   movq mm2, mm0          ; U x1:x0 B1:B0 x1:x0 B3:?? A3:A2 B2:??       00:1f
   pslld mm0, 3           ; V l1:l0 B1:B0 x1:x0 B3:?? A3:A2 B2:??       00:1f

   movq [%2], mm1         ; U l1:l0 B1:B0 x1:x0 B3:?? A3:A2 B2:??       00:1f
   punpckhdq mm5, mm3     ; V l1:l0 B1:B0 x1:x0       A3:A2 B3:B2       00:1f

   paddd mm4, [%1+8]      ; U l1:l0 B1:B0 x1:x0       +3:+2 B3:B2       00:1f
   psrld mm2, 29          ; V l1:l0 B1:B0 r1:r0       +3:+2 B3:B2       00:1f

   movq [%2+8], mm5       ; U l1:l0 B1:B0 r1:r0       +3:+2 B3:B2       00:1f
   paddd mm4, mm5         ; U l1:l0 B1:B0 r1:r0       x3:x2 B3:B2       00:1f

   por mm0, mm2           ; U A1:A0 B1:B0             x3:x2 B3:B2       00:1f
   movq mm6, mm4          ; V A1:A0 B1:B0             x3:x2 B3:B2 x3:x2 00:1f

   paddd mm1, mm0         ; U A1:A0 +1:+0             x3:x2 B3:B2 x3:x2 00:1f
   pslld mm4, 3           ; V A1:A0 +1:+0             l3:l2 B3:B2 x3:x2 00:1f

   movq mm2, mm1          ; U A1:A0 +1:+0 +1:+0       l3:l2 B3:B2 x3:x2 00:1f
   psrld mm6, 29          ; V A1:A0 +1:+0 +1:+0       l3:l2 B3:B2 r3:r2 00:1f

   paddd mm1, [%3]        ; U A1:A0 x1:x0 +1:+0       l3:l2 B3:B2 r3:r2 00:1f
   por mm4, mm6           ; V A1:A0 x1:x0 +1:+0       A3:A2 B3:B2       00:1f

   movq [%1], mm0         ; U A1:A0 x1:x0 +1:+0       A3:A2 B3:B2       00:1f
   movq mm3, mm1          ; V A1:A0 x1:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f

   movq [%1+8], mm4       ; U A1:A0 x1:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f
   punpckldq mm1, mm1     ; V A1:A0 x0:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f
%endmacro

%macro encryption_round 5
                          ; P  %1    %2    mm2   mm3   %3    %4    mm6   mm7
                          ;--------------------------------------------------
                          ;   a0:?? B1:B0 x3:x2 a1:?? x3:x2 B3:B2 B3:B2 00:1f

   punpckhdq mm%1, mm3    ; U a1:a0 B1:B0 x3:x2       x3:x2 B3:B2 B3:B2 00:1f
   movq mm3, mm6          ; V a1:a0 B1:B0 x3:x2 B3:B2 x3:x2 B3:B2 B3:B2 00:1f

   punpckhdq mm6, mm6     ; U a1:a0 B1:B0 x3:x2 B3:B2 x3:x2 B3:B2 B3:B3 00:1f
   pand mm3, mm7          ; V a1:a0 B1:B0 x3:x2 00:s2 x3:x2 B3:B2 B3:B3 00:1f

   paddd mm%1, [%5]       ; U A1:A0 B1:B0 x3:x2 00:s2 x3:x2 B3:B2 B3:B3 00:1f
   punpckldq mm%3, mm%3   ; V A1:A0 B1:B0 x3:x2 00:s2 x2:x2 B3:B2 B3:B3 00:1f

   pand mm6, mm7          ; U A1:A0 x1:x0 x3:x2 00:s2 x2:x2 B3:B2 00:s3 00:1f
   punpckhdq mm2, mm2     ; V A1:A0 x1:x0 x3:x3 00:s2 x2:x2 B3:B2 00:s3 00:1f

   pxor mm%2, mm%1        ; U A1:A0 x1:x0 x3:x3 00:s2 x2:x2 B3:B2 00:s3 00:1f
   psllq mm%3, mm3        ; V A1:A0 x1:x0 x3:x3       a2:?? B3:B2 00:s3 00:1f

   movq mm3, mm%2         ; U A1:A0 x1:x0 x3:x3 x1:x0 a2:?? B3:B2 00:s3 00:1f
   psllq mm2, mm6         ; V A1:A0 x1:x0 a3:?? x1:x0 a2:?? B3:B2       00:1f

   movq mm6, mm%1         ; U A1:A0 x1:x0 a3:?? x1:x0 a2:?? B3:B2 A1:A0 00:1f
   punpckhdq mm%3, mm2    ; V A1:A0 x1:x0       x1:x0 a3:a2 B3:B2 A1:A0 00:1f

   movq mm2, mm%1         ; U A1:A0 x1:x0 A1:A0 x1:x0 a3:a2 B3:B2 A1:A0 00:1f
   punpckhdq mm6, mm6     ; V A1:A0 x1:x0 A1:A0 x1:x0 a3:a2 B3:B2 A1:A1 00:1f

   paddd mm%3, [%5+8]     ; U A1:A0 x1:x0 A1:A0 x1:x0 A3:A2 B3:B2 A1:A1 00:1f
   punpckldq mm%2, mm%2   ; V A1:A0 x0:x0 A1:A0 x1:x0 A3:A2 B3:B2 A1:A1 00:1f

   pand mm2, mm7          ; U A1:A0 x0:x0 00:s0 x1:x0 A3:A2 B3:B2 A1:A1 00:1f
   pand mm6, mm7          ; V A1:A0 x0:x0 00:s0 x1:x0 A3:A2 B3:B2 00:s1 00:1f

   punpckhdq mm3, mm3     ; U A1:A0 x0:x0       x1:x1 A3:A2 x3:x2 00:s1 00:1f
   pxor mm%4, mm%3        ; V A1:A0 x0:x0       x1:x1 A3:A2 x3:x2 00:s1 00:1f

   psllq mm%2, mm2        ; U A1:A0 a0:??       x1:x1 A3:A2 B3:B2 00:s1 00:1f
   movq mm2, mm%4         ; V A1:A0 a0:?? x3:x2 x1:x1 A3:A2 x3:x2 00:s1 00:1f

   psllq mm3, mm6         ; U A1:A0 a0:?? x3:x2 a1:?? A3:A2 x3:x2       00:1f
   movq mm6, mm%3         ; V A1:A0 a0:?? x3:x2 a1:?? A3:A2 x3:x2 A3:A2 00:1f
%endmacro


[GLOBAL _rc5_72_unit_func_mmx]
[GLOBAL rc5_72_unit_func_mmx]

%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _DATA USE32 ALIGN=16 CLASS=DATA]
%else
[SECTION .data]
%endif

   align 8
   S_Init       dd 0xbf0a8b1d, 0xbf0a8b1d ;S0 (is constant (0xb7e15163 <<< 3))
                dd 0x5618cb1c, 0x5618cb1c ;S1
                dd 0xf45044d5, 0xf45044d5 ;S2
                dd 0x9287be8e, 0x9287be8e ;S3
                dd 0x30bf3847, 0x30bf3847 ;S4
                dd 0xcef6b200, 0xcef6b200 ;S5
                dd 0x6d2e2bb9, 0x6d2e2bb9 ;S6
                dd 0x0b65a572, 0x0b65a572 ;S7
                dd 0xa99d1f2b, 0xa99d1f2b ;S8
                dd 0x47d498e4, 0x47d498e4 ;S9
                dd 0xe60c129d, 0xe60c129d ;S10
                dd 0x84438c56, 0x84438c56 ;S11
                dd 0x227b060f, 0x227b060f ;S12
                dd 0xc0b27fc8, 0xc0b27fc8 ;S13
                dd 0x5ee9f981, 0x5ee9f981 ;S14
                dd 0xfd21733a, 0xfd21733a ;S15
                dd 0x9b58ecf3, 0x9b58ecf3 ;S16
                dd 0x399066ac, 0x399066ac ;S17
                dd 0xd7c7e065, 0xd7c7e065 ;S18
                dd 0x75ff5a1e, 0x75ff5a1e ;S19
                dd 0x1436d3d7, 0x1436d3d7 ;S20
                dd 0xb26e4d90, 0xb26e4d90 ;S21
                dd 0x50a5c749, 0x50a5c749 ;S22
                dd 0xeedd4102, 0xeedd4102 ;S23
                dd 0x8d14babb, 0x8d14babb ;S24
                dd 0x2b4c3474, 0x2b4c3474 ;S25


%ifdef __OMF__ ; Watcom and Borland compilers/linkers
[SECTION _TEXT USE32 ALIGN=16 CLASS=CODE]
%else
[SECTION .text]
%endif

_rc5_72_unit_func_mmx:
rc5_72_unit_func_mmx:
   push ebx
   push esi

   push edi
   push ebp

   sub esp, work_size
   pcmpeqd mm7, mm7           ; create shift mask in mm7 (mm7 = -1:-1)

   lea ebp, [esp+align_value]
   psrlq mm7, 59              ; mm7 = 00000000:0000001f

   and ebp, -align_value      ; make local data aligned

; 32 bit register assigment
; eax - test ciphertext high, temporary, result
; ebx - key incrementing
; ecx - test ciphertext low, temporary
; edx - temporary
; esi - pointer to RC5UnitWork struct
; edi - iterations
; ebp - Quadword aligned pointer to work area

; Access parameters
   mov edx, [iterations]
   mov esi, [RC5_72UnitWork]

; Load parameters

   movd mm6, [RC5_72UnitWork_L0lo]
   mov edi, [edx]             ; work.iterations = *iterations

   movd mm5, [RC5_72UnitWork_L0mid]
   shr edi, 2                 ; we're testing 4 keys/loop

   mov ebx, [RC5_72UnitWork_L0hi]

; Save other parameters
   movd mm0, [RC5_72UnitWork_plainlo]
   movd mm1, [RC5_72UnitWork_plainhi]

   movd mm2, [RC5_72UnitWork_cipherlo]
   punpckldq mm0, mm0

   movd mm3, [RC5_72UnitWork_cipherhi]
   punpckldq mm1, mm1

   movq [work_P_0], mm0
   punpckldq mm2, mm2

   movq [work_P_1], mm1
   punpckldq mm3, mm3

   movq [work_C_0], mm2
   movq [work_C_1], mm3

align 8
change_lo:
   punpckldq mm6, mm6
   movq [L1_Init], mm6
   punpckldq mm5, mm5
   movq [L1_Init+8], mm6
   movq [L2_Init], mm5
   movq [L2_Init+8], mm5

mainloop:
   mov [L3_Init], ebx
   lea edx, [ebx+0x01]

   ;
   ; First two rounds are done seperately, since all pipes start with equal
   ; values. Thus several calculations have to be done only once. Even some
   ; hacks regarding the rol-operations only work because of this. Be
   ; careful in changing this code.
   ;
                          ;             KEY SETUP STARTS HERE
                          ; P   0     1     2     3     4     5     6     7
                          ;--------------------------------------------------
                          ;                                             00:1f

   movq mm0, [S_Init]     ; U A1:A0                                     00:1f
   movq mm2, mm7          ; V A1:A0       00:1f                         00:1f

   movq mm1, [L1_Init]    ; U A1:A0 +1:+0 00:1f                         00:1f
   pand mm2, mm0          ; V A1:A0 +1:+0 00:s1                         00:1f

   movq [STable], mm0     ; U A1:A0 +1:+0 00:s1                         00:1f
   paddd mm1, mm0         ; V A1:A0 B1:B0 00:s1                         00:1f

   movq [STable+8], mm0   ; U A1:A0 B1:B0 00:s1                         00:1f
   psllq mm1, mm2         ; V A1:A0 B0:??                               00:1f

   paddd mm0, [S_Init+8]  ; U +1:+0 B0:??                               00:1f
   punpckhdq mm1, mm1     ; V +1:+0 B1:B0                               00:1f

   mov [L3_Init+4], edx   ; U
   add bl, BYTE 0x02      ; V

   movq [L1], mm1         ; U +1:+0 B1:B0                               00:1f
   paddd mm0, mm1         ; V x1:x0 B1:B0                               00:1f

   mov [L3_Init+8], ebx   ; U
   add dl, BYTE 0x02      ; V

   movq mm2, mm0          ; U x1:x0 B1:B0 x1:x0                         00:1f
   pslld mm0, 3           ; V l1:l0 B1:B0 x1:x0                         00:1f

   movq [L1+8], mm1       ; U l1:l0 B1:B0                               00:1f
   psrld mm2, 29          ; V l1:l0 B1:B0 r1:r0                         00:1f

   mov [L3_Init+12], edx  ; U
   por mm0, mm2           ; V A1:A0 B1:B0                               00:1f

   movq [STable+16], mm0  ; U A1:A0 B1:B0                               00:1f
   paddd mm1, mm0         ; V A1:A0 +1:+0                               00:1f

   movq [STable+24], mm0  ; U A1:A0 +1:+0                               00:1f
   movq mm2, mm1          ; V A1:A0 +1:+0 +1:+0                         00:1f

   paddd mm1, [L2_Init]   ; U A1:A0 x1:x0 +1:+0                         00:1f
   pand mm2, mm7          ; V A1:A0 B0:?? 00:s1                         00:1f

   paddd mm0, [S_Init+16] ; U +1:+0 B0:?? 00:s1                         00:1f
   psllq mm1, mm2         ; V +1:+0 B0:??                               00:1f

   add bl, BYTE 0x02      ; U
   punpckhdq mm1, mm1     ; V +1:+0 B1:B0                               00:1f

   movq [L2], mm1         ; U l1:l0 B1:B0                               00:1f
   paddd mm0, mm1         ; V x1:x0 B1:B0                               00:1f

   movq mm2, mm0          ; U x1:x0 B1:B0 x1:x0                         00:1f
   pslld mm0, 3           ; V l1:l0 B1:B0 x1:x0                         00:1f

   movq [L2+8], mm1       ; U l1:l0 B1:B0 x1:x0                         00:1f
   psrld mm2, 29          ; V l1:l0 B1:B0 r1:r0                         00:1f

   movq mm5, mm1          ; U A1:A0 B1:B0 r1:r0             B3:B2       00:1f
   por mm0, mm2           ; V A1:A0 B1:B0                   B3:B2       00:1f

   movq [STable+32], mm0  ; U A1:A0 x1:x0 +1:+0             B3:B2       00:1f
   paddd mm1, mm0         ; V A1:A0 +1:+0                   B3:B2       00:1f

   movq [STable+40], mm0  ; U A1:A0 +1:+0 +1:+0             B3:B2       00:1f
   movq mm2, mm1          ; V A1:A0 +1:+0 +1:+0             B3:B2       00:1f

   paddd mm1, [L3_Init]   ; U A1:A0 x1:x0 +1:+0             B3:B2       00:1f
   movq mm4, mm0          ; V A1:A0 x1:x0 +1:+0       A3:A2 B3:B2       00:1f

   movq mm3, mm1          ; U A1:A0 x1:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f
   punpckldq mm1, mm1     ; V A1:A0 x0:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f

   keytable_round_1  S_Init+24 , STable+48 , L3_Init, L3, L1
   keytable_round_1  S_Init+32 , STable+64 , L1, L1, L2
   keytable_round_1  S_Init+40 , STable+80 , L2, L2, L3
   keytable_round_1  S_Init+48 , STable+96 , L3, L3, L1
   keytable_round_1  S_Init+56 , STable+112, L1, L1, L2
   keytable_round_1  S_Init+64 , STable+128, L2, L2, L3
   keytable_round_1  S_Init+72 , STable+144, L3, L3, L1
   keytable_round_1  S_Init+80 , STable+160, L1, L1, L2
   keytable_round_1  S_Init+88 , STable+176, L2, L2, L3
   keytable_round_1  S_Init+96 , STable+192, L3, L3, L1
   keytable_round_1  S_Init+104, STable+208, L1, L1, L2
   keytable_round_1  S_Init+112, STable+224, L2, L2, L3
   keytable_round_1  S_Init+120, STable+240, L3, L3, L1
   keytable_round_1  S_Init+128, STable+256, L1, L1, L2
   keytable_round_1  S_Init+136, STable+272, L2, L2, L3
   keytable_round_1  S_Init+144, STable+288, L3, L3, L1
   keytable_round_1  S_Init+152, STable+304, L1, L1, L2
   keytable_round_1  S_Init+160, STable+320, L2, L2, L3
   keytable_round_1  S_Init+168, STable+336, L3, L3, L1
   keytable_round_1  S_Init+176, STable+352, L1, L1, L2
   keytable_round_1  S_Init+184, STable+368, L2, L2, L3
   keytable_round_1  S_Init+192, STable+384, L3, L3, L1
   keytable_round_1  S_Init+200, STable+400, L1, L1, L2

   ;
   ; Round 2
   ;
   keytable_round  STable    , L2, L3
   keytable_round  STable+16 , L3, L1
   keytable_round  STable+32 , L1, L2
   keytable_round  STable+48 , L2, L3
   keytable_round  STable+64 , L3, L1
   keytable_round  STable+80 , L1, L2
   keytable_round  STable+96 , L2, L3
   keytable_round  STable+112, L3, L1
   keytable_round  STable+128, L1, L2
   keytable_round  STable+144, L2, L3
   keytable_round  STable+160, L3, L1
   keytable_round  STable+176, L1, L2
   keytable_round  STable+192, L2, L3
   keytable_round  STable+208, L3, L1
   keytable_round  STable+224, L1, L2
   keytable_round  STable+240, L2, L3
   keytable_round  STable+256, L3, L1
   keytable_round  STable+272, L1, L2
   keytable_round  STable+288, L2, L3
   keytable_round  STable+304, L3, L1
   keytable_round  STable+320, L1, L2
   keytable_round  STable+336, L2, L3
   keytable_round  STable+352, L3, L1
   keytable_round  STable+368, L1, L2
   keytable_round  STable+384, L2, L3
   keytable_round  STable+400, L3, L1

   ;
   ; Round 3
   ;
   keytable_round  STable    , L1, L2
   keytable_round  STable+16 , L2, L3
   keytable_round  STable+32 , L3, L1
   keytable_round  STable+48 , L1, L2
   keytable_round  STable+64 , L2, L3
   keytable_round  STable+80 , L3, L1
   keytable_round  STable+96 , L1, L2
   keytable_round  STable+112, L2, L3
   keytable_round  STable+128, L3, L1
   keytable_round  STable+144, L1, L2
   keytable_round  STable+160, L2, L3
   keytable_round  STable+176, L3, L1
   keytable_round  STable+192, L1, L2
   keytable_round  STable+208, L2, L3
   keytable_round  STable+224, L3, L1
   keytable_round  STable+240, L1, L2
   keytable_round  STable+256, L2, L3
   keytable_round  STable+272, L3, L1
   keytable_round  STable+288, L1, L2
   keytable_round  STable+304, L2, L3
   keytable_round  STable+320, L3, L1
   keytable_round  STable+336, L1, L2
   keytable_round  STable+352, L2, L3
   keytable_round  STable+368, L3, L1
   keytable_round  STable+384, L1, L2

                          ; P   0     1     2     3     4     5     6     7
                          ;--------------------------------------------------
                          ;   A1:A0 x0:x0 +1:+0 x1:x0 A3:A2 B3:B2       00:1f

   movq mm6, mm2          ; U A1:A0 x0:x0 +1:+0 x1:x0 A3:A2 B3:B2 +1:+0 00:1f
   punpckhdq mm2, mm2     ; V A1:A0 x0:x0 00:+1 x1:x0 A3:A2 B3:B2 +1:+1 00:1f

   paddd mm5, mm4         ; U A1:A0 x0:x0 00:+1 x1:x0 A3:A2 +3:+2 +1:+0 00:1f
   punpckhdq mm3, mm3     ; V A1:A0 x0:x0 00:+1 x1:x1 A3:A2 +3:+2 +1:+0 00:1f

   pand mm6, mm7          ; U A1:A0 x0:x0 00:+0 x1:x0 A3:A2 +3:+2 00:s0 00:1f
   pand mm2, mm7          ; V A1:A0 x0:x0 00:s1 x1:x0 A3:A2 +3:+2 00:s0 00:1f

   psllq mm1, mm6         ; U A1:A0 B0:?? 00:s1 x1:x1 A3:A2 +3:+2       00:1f
   movq mm6, mm5          ; V A1:A0 B0:?? 00:s1 x1:x1 A3:A2 +3:+2 +3:+2 00:1f

   paddd mm5, [L2+8]      ; U A1:A0 B0:?? 00:s1 x1:x1 A3:A2 x3:x2 +3:+2 00:1f
   psllq mm3, mm2         ; V A1:A0 B0:??       B1:?? A3:A2 x3:x2 +3:+2 00:1f

   movq mm2, mm6          ; U A1:A0 B0:?? +3:+2 B1:?? A3:A2 x3:x2 +3:+2 00:1f
   punpckhdq mm6, mm6     ; V A1:A0 B0:?? +3:+2 B1:?? A3:A2 x3:x2 00:+3 00:1f

   punpckhdq mm1, mm3     ; U A1:A0 B1:B0 +3:+2       A3:A2 x3:x2 00:+3 00:1f
   movq mm3, mm5          ; V A1:A0 B1:B0 +3:+2 x3:x2 A3:A2 x3:x2 00:+3 00:1f

   punpckldq mm5, mm5     ; U A1:A0 B1:B0 +3:+2 x3:x2 A3:A2 x2:x2 00:+3 00:1f
   pand mm6, mm7          ; V A1:A0 B1:B0 +3:+2 x3:x2 A3:A2 x2:x2 00:s3 00:1f

   paddd mm0, [STable+400]; U +1:+0 B1:B0 +3:+2 x3:x2 A3:A2 x2:x2 00:s3 00:1f
   punpckhdq mm3, mm3     ; V +1:+0 B1:B0 +3:+2 x3:x3 A3:A2 x2:x2 00:s3 00:1f

   pand mm2, mm7          ; U +1:+0 B1:B0 00:s2 x3:x3 A3:A2 x2:x2 00:s3 00:1f
   psllq mm3, mm6         ; V +1:+0 B1:B0 00:s2 B3:?? A3:A2 x2:x2       00:1f

   paddd mm0, mm1         ; U x1:x0 B1:B0 00:s2 B3:?? A3:A2 x2:x2       00:1f
   psllq mm5, mm2         ; V x1:x0             B3:?? A3:A2 B2:??       00:1f

   paddd mm4, [STable+408]; U x1:x0             B3:?? +3:+2 B2:??       00:1f
   punpckhdq mm5, mm3     ; U x1:x0                   +3:+2 B3:B2       00:1f

   movq mm2, mm0          ; V x1:x0       x1:x0       +3:+2 B3:B2       00:1f
   pslld mm0, 3           ; V l1:l0       x1:x0       +3:+2 B3:B2       00:1f

   paddd mm4, mm5         ; U l1:l0       x1:x0       x3:x2 B3:B2       00:1f
   psrld mm2, 29          ; V l1:l0       r1:r0       x3:x2             00:1f

   movq mm6, mm4          ; U l1:l0       r1:r0       x3:x2       x3:x2 00:1f
   por mm0, mm2           ; V A1:A0                   x3:x2       x3:x2 00:1f

   movq mm1, [work_P_1]   ; U A1:A0 b1:b0             x3:x2       x3:x2 00:1f
   pslld mm4, 3           ; V A1:A0 b1:b0             l3:l2       x3:x2 00:1f

   movq [STable+400], mm0 ; U A1:A0 b1:b0             l3:l2       x3:x2 00:1f
   psrld mm6, 29          ; V A1:A0 b1:b0             l3:l2       r3:r2 00:1f

   movq mm0, [work_P_0]   ; U a1:a0 b1:b0             l3:l2       r3:r2 00:1f
   por mm4, mm6           ; V a1:a0 b1:b0             A3:A2             00:1f

   movq [STable+408], mm4 ; U a1:a0 b1:b0             A3:A2             00:1f
   movq mm5, mm1          ; V a1:a0 b1:b0                   b3:b2       00:1f

                          ;---------------------ENCRYPTION STARTS HERE
   paddd mm1, [STable+16] ; U a1:a0 B1:B0                               00:1f
   movq mm4, mm0          ; V a1:a0 B1:B0             a3:a2 b3:b2       00:1f

   paddd mm0, [STable]    ; U A1:A0 B1:B0             a3:a2 b3:b2       00:1f
   movq mm2, mm1          ; V A1:A0 B1:B0 B1:B0       a3:a2 b3:b2       00:1f

   paddd mm4, [STable+8]  ; U A1:A0 B1:B0 B1:B0       A3:A2 b3:b2       00:1f
   pxor mm0, mm1          ; V x1:x0 B1:B0 B1:B0       A3:A2 b3:b2       00:1f

   movq mm6, mm2          ; U x1:x0 B1:B0 B1:B0       A3:A2 b3:b2 B1:B0 00:1f
   punpckhdq mm2, mm2     ; V x1:x0 B1:B0 00:B1       A3:A2 b3:b2 B1:B0 00:1f

   movq mm3, mm0          ; U x1:x0 B1:B0 00:B1 x1:x0 A3:A2 b3:b2 B1:B0 00:1f
   paddd mm5, [STable+24] ; V x1:x0 B1:B0 00:B1 x1:x0 A3:A2 B3:B2 B1:B0 00:1f

   punpckldq mm0, mm0     ; U x0:x0 B1:B0 00:B1 x1:x0 A3:A2 B3:B2 B1:B0 00:1f
   pand mm6, mm7          ; V x0:x0 B1:B0 00:B1 x1:x0 A3:A2 B3:B2 00:s0 00:1f

   pand mm2, mm7          ; U x0:x0 B1:B0 00:s1 x1:x0 A3:A2 B3:B2 00:s0 00:1f
   psllq mm0, mm6         ; V b0:?? B1:B0 00:s1 x1:x0 A3:A2 B3:B2       00:1f

   pxor mm4, mm5          ; U b0:?? B1:B0 00:s1 x1:x0 x3:x2 B3:B2       00:1f
   punpckhdq mm3, mm3     ; V b0:?? B1:B0 00:s1 x1:x1 x3:x2 B3:B2       00:1f

   movq mm6, mm5          ; U b0:?? B1:B0 00:s1 x1:x1 x3:x2 B3:B2 B3:B2 00:1f
   psllq mm3, mm2         ; V b0:?? B1:B0       b1:?? x3:x2 B3:B2 B3:B2 00:1f

   movq mm2, mm4          ; U b0:?? B1:B0 x3:x2 b1:?? x3:x2 B3:B2 B3:B2 00:1f

   encryption_round  0, 1, 4, 5, STable+32
   encryption_round  1, 0, 5, 4, STable+48
   encryption_round  0, 1, 4, 5, STable+64
   encryption_round  1, 0, 5, 4, STable+80
   encryption_round  0, 1, 4, 5, STable+96
   encryption_round  1, 0, 5, 4, STable+112
   encryption_round  0, 1, 4, 5, STable+128
   encryption_round  1, 0, 5, 4, STable+144
   encryption_round  0, 1, 4, 5, STable+160
   encryption_round  1, 0, 5, 4, STable+176
   encryption_round  0, 1, 4, 5, STable+192
   encryption_round  1, 0, 5, 4, STable+208
   encryption_round  0, 1, 4, 5, STable+224
   encryption_round  1, 0, 5, 4, STable+240
   encryption_round  0, 1, 4, 5, STable+256
   encryption_round  1, 0, 5, 4, STable+272
   encryption_round  0, 1, 4, 5, STable+288
   encryption_round  1, 0, 5, 4, STable+304
   encryption_round  0, 1, 4, 5, STable+320
   encryption_round  1, 0, 5, 4, STable+336
   encryption_round  0, 1, 4, 5, STable+352
   encryption_round  1, 0, 5, 4, STable+368
   encryption_round  0, 1, 4, 5, STable+384

                          ; P   0     1     2     3     4     5     6     7
                          ;--------------------------------------------------
                          ;   A1:A0 a0:?? x3:x2 a1:?? A3:A2 x3:x2 A3:A2 00:1f

   punpckhdq mm1, mm3     ; U A1:A0 a1:a0 x3:x2       A3:A2 x3:x2 A3:A2 00:1f
   movq mm3, mm6          ; V A1:A0 a1:a0 x3:x2 A3:A2 A3:A2 x3:x2 A3:A2 00:1f

   paddd mm1, [STable+400]; U A1:A0 B1:B0 x3:x2 A3:A2 A3:A2 x3:x2 A3:A2 00:1f
   punpckldq mm5, mm5     ; V A1:A0 B1:B0 x3:x2 A3:A2 A3:A2 x2:x2 A3:A2 00:1f

   pcmpeqd mm0, [work_C_0]; U ??:?? B1:B0 x3:x2 A3:A2 A3:A2 x2:x2 A3:A2 00:1f
   punpckhdq mm2, mm2     ; V ??:?? B1:B0 x3:x3 A3:A2 A3:A2 x2:x2 A3:A2 00:1f

   pcmpeqd mm1, [work_C_1]; U ??:?? ??:?? x3:x3 A3:A2 A3:A2 x2:x2 A3:A2 00:1f
   psrlq mm0, 24          ; "pack" comparison results in a single word and
                          ; test 8-bit registers later on
   psrlq mm1, 24
   movd ecx, mm0

   test cl, cl            ; Check A0
   jnz near check_B0

check_A1:
   test ch, ch            ; Check A1
   jnz near check_B1

check_A2:
   punpckhdq mm6, mm6     ; U             x3:x2 A3:A2 A3:A2 x2:x2 A3:A3 00:1f
   pand mm3, mm7          ; V             x3:x3 00:s2 A3:A2 x2:x2 A3:A3 00:1f

   pand mm6, mm7          ; U             x3:x3 00:s2 A3:A2 x2:x2 00:s3 00:1f
   psllq mm5, mm3         ; V             x3:x3       A3:A2 a2:?? 00:s3 00:1f

   pcmpeqd mm4, [work_C_0]; U             x3:x3       ??:?? a2:?? 00:s3 00:1f
   psllq mm2, mm6         ; V             a3:??             a2:??       00:1f

   punpckhdq mm5, mm2     ; U                               a3:a2       00:1f

   paddd mm5, [STable+408]; U                               B3:B2       00:1f
   psrlq mm4, 24          ; V

   pcmpeqd mm5, [work_C_1]; U                               ??:??       00:1f

   movd ecx, mm4          ; U
   psrlq mm5, 24          ; V

   test cl, cl            ; Check A2
   jnz near check_B2

check_A3:
   test ch, ch            ; Check A3
   jnz near check_B3

incr_key:
   dec edi
   jz  near exit_nothing

   test ebx, ebx          ; Overflow while incrementing high-byte of the key?
   jnz mainloop

   mov edx, [L2_Init]     ; Adjust the other bytes, too.
   mov eax, [L1_Init]

   bswap edx
   bswap eax

   add edx, BYTE 1        ; Don't use inc edx, since we need the carry flag.
   adc eax, BYTE 0

   bswap edx
   bswap eax

   movd mm5, edx
   movd mm6, eax

   jmp change_lo

align 8
check_B0:
   mov  edx, [L1_Init]
   mov  eax, [L2_Init]

   mov  [RC5_72UnitWork_CMClo], edx
   mov  [RC5_72UnitWork_CMCmid], eax

   lea  edx, [ebx-4]
   movd eax, mm1

   mov  [RC5_72UnitWork_CMChi], edx
   inc  DWORD [RC5_72UnitWork_CMCcount]

   test al, al            ; Check B0
   jz   near check_A1

   shl  edi, 2
   mov  ebx, edx
   jmp  near exit_found


align 8
check_B1:
   mov  edx, [L1_Init]
   mov  eax, [L2_Init]

   mov  [RC5_72UnitWork_CMClo], edx
   mov  [RC5_72UnitWork_CMCmid], eax

   lea  edx, [ebx-3]
   movd eax, mm1

   mov  [RC5_72UnitWork_CMChi], edx
   inc  DWORD [RC5_72UnitWork_CMCcount]

   test ah, ah            ; Check B1
   jz near check_A2

   lea  edi, [4*edi-1]
   mov  ebx, edx
   jmp  short exit_found


align 8
check_B2:
   mov  edx, [L1_Init]
   mov  eax, [L2_Init]

   mov  [RC5_72UnitWork_CMClo], edx
   mov  [RC5_72UnitWork_CMCmid], eax

   lea  edx, [ebx-2]
   movd eax, mm5

   mov  [RC5_72UnitWork_CMChi], edx
   inc  DWORD [RC5_72UnitWork_CMCcount]

   test al, al            ; Check B2
   jz  near check_A3

   lea  edi, [4*edi-2]
   mov  ebx, edx
   jmp  short exit_found

align 8
check_B3:
   mov  edx, [L1_Init]
   mov  eax, [L2_Init]

   mov  [RC5_72UnitWork_CMClo], edx
   mov  [RC5_72UnitWork_CMCmid], eax

   lea  edx, [ebx-1]
   movd eax, mm5

   mov  [RC5_72UnitWork_CMChi], edx
   inc  DWORD [RC5_72UnitWork_CMCcount]

   test ah, ah            ; Check B3
   jz  near incr_key

   lea  edi, [4*edi-3]
   mov  ebx, edx

align 8
exit_found:
   mov  edx, [iterations]
   mov  eax, RESULT_FOUND
   sub  [edx], edi
   jmp  short finish

align 8
exit_nothing:
   mov  eax, RESULT_NOTHING

finish:
   test ebx, ebx
   mov edx, [L2_Init]

   mov ecx, [L1_Init]
   mov [RC5_72UnitWork_L0hi], ebx

   jnz short key_updated

   bswap edx
   bswap ecx

   add edx, BYTE 1
   adc ecx, BYTE 0

   bswap edx
   bswap ecx

key_updated:
   mov [RC5_72UnitWork_L0mid], edx
   mov [RC5_72UnitWork_L0lo], ecx

   add esp, work_size
   emms

   pop ebp
   pop edi
   pop esi
   pop ebx

   ret
