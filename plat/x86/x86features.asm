;
; Copyright distributed.net 2003-2004 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; x86 Processor feature identification for distributed.net effort
;
; Taken from Loki_Utils
; http://www.icculus.org
;
; 1997-98 by H. Dietz and R. Fisher
; Bug fixes and SSE detection by Sam Lantinga
;
; Modified for distributed.net by Steven Nikkel, Nov 2003
;
; $Id: x86features.asm,v 1.1.2.16 2004/11/23 16:13:01 snikkel Exp $
;
; return u32

%ifndef __OMF__
  %ifdef OS2
    %define __OMF__
  %endif
%endif
                      
%ifdef __OMF__   ; Watcom+OS/2 or Borland+Win32
[SECTION _DATA CLASS=DATA USE32 PUBLIC ALIGN=16]
[SECTION _TEXT CLASS=CODE USE32 PUBLIC ALIGN=16]
%define __DATASECT__ [SECTION _DATA]
%define __CODESECT__ [SECTION _TEXT]
%else
%define __DATASECT__ [SECTION .data]
%define __CODESECT__ [SECTION .text]
%endif

global          x86features,_x86features

%define CPU_F_MMX             00000100h
%define CPU_F_CYRIX_MMX_PLUS  00000200h
%define CPU_F_AMD_MMX_PLUS    00000400h
%define CPU_F_3DNOW           00000800h
%define CPU_F_3DNOW_PLUS      00001000h
%define CPU_F_SSE             00002000h
%define CPU_F_SSE2            00004000h
%define CPU_F_SSE3            00008000h
%define CPU_F_HYPERTHREAD     00010000h
%define CPU_F_AMD64           00020000h
%define CPU_F_EM64T           00040000h

__CODESECT__
_x86features:            
x86features: 

  push ebx
  push ecx
  push edx
  push esi
  push edi

  mov esi, 0h

  ; See if CPUID instruction is supported ...
  ; ... Get copies of EFLAGS into eax and ecx
  pushf
  pop eax
  mov ecx, eax

  ; ... Toggle the ID bit in one copy and store
  ;     to the EFLAGS reg
  xor eax, 200000h
  push eax
  popf

  ; ... Get the (hopefully modified) EFLAGS
  pushf
  pop eax

  ; ... Compare and test result
  cmp ecx, eax
  je near NotSupported

Standard:
  mov eax, 1h
  cpuid
MMX_test:
  mov edi, eax
  and edi, 00000FFFh
  cmp edi, 00000545h    ; Skip Pentium with buggy MMX
  je SSE_test
  test edx, 00800000h   ; Test for MMX
  jz SSE_test           ; MMX Not supported
  or esi, CPU_F_MMX     ; MMX Supported
SSE_test:
  test edx, 02000000h   ; Test for SSE
  jz SSE2_test          ; SSE Not supported
  or esi, CPU_F_SSE     ; SSE Supported
SSE2_test:
  test edx, 04000000h   ; Test for SSE2
  jz SSE3_test          ; SSE2 Not supported
  or esi, CPU_F_SSE2    ; SSE2 Supported
SSE3_test:
  test ecx, 00000001h   ; Test for SSE3
  jz HT_test            ; SSE3 Not supported
  or esi, CPU_F_SSE3    ; SSE3 Supported
HT_test:
  test edx, 10000000h   ; Test for Hyper-Threading support
  jz TryExtended
  and ebx, 00FF0000h
  cmp ebx, 00010000h    ; Check if Hyper-Threading enabled
  je TryExtended
  or esi, CPU_F_HYPERTHREAD ; Hyper-Threading supported and enabled

  jmp TryExtended

TryExtended:
  ; See if extended CPUID is supported
  mov eax, 80000000h
  cpuid
  cmp eax, 80000000h
  jl near NotSupported

  ; Get standard CPUID information, and
  ; go to a specific vendor section
  mov eax, 0h
  cpuid

; Check for Intel
TryIntel:
  cmp ebx, 756E6547h
  jne TryAMD
  cmp edx, 49656E69h
  jne TryAMD
  cmp ecx, 6C65746Eh
  jne TryAMD
  jmp Intel

; Check for AMD
TryAMD:
  cmp ebx, 68747541h
  jne TryCyrix
  cmp edx, 69746E65h
  jne TryCyrix
  cmp ecx, 444D4163h
  jne TryCyrix
  jmp AMD

; Check for Cyrix
TryCyrix:
  cmp ebx, 69727943h
  jne near NotSupported
  cmp edx, 736E4978h
  jne near NotSupported
  cmp ecx, 64616574h
  jne near NotSupported
  jmp Cyrix

Cyrix:
  ; Extended CPUID supported, so get extended features
  mov eax, 80000001h
  cpuid
  
CYRIXMMXPLUS_test:  
  test edx, 01000000h  ; Test for Cyrix Ext'd MMX
  jz Return
  or esi, CPU_F_CYRIX_MMX_PLUS           ; Cyrix EMMX supported

  jmp Return

AMD:
  ; Extended CPUID supported, so get extended features
  mov eax, 80000001h
  cpuid

AMDMMX_test:
  test edx, 00800000h   ; Test for MMX
  jz AMDMMXPlus_test
  or esi, CPU_F_MMX     ; MMX Supported
AMDMMXPlus_test:
  test edx, 00400000h   ; Test for AMD Ext'd MMX
  jz ThreeDNow_test
  or esi, CPU_F_AMD_MMX_PLUS  ; AMD EMMX supported
ThreeDNow_test:
  test edx, 80000000h   ; Test for 3DNow!
  jz ThreeDNowPlus_test
  or esi, CPU_F_3DNOW   ; 3DNow! also supported
ThreeDNowPlus_test:
  test edx, 40000000h   ; Test for 3DNow!+
  jz AMD64_test
  or esi, CPU_F_3DNOW_PLUS  ; 3DNow!+ also supported
AMD64_test:
  test edx, 20000000h   ; Test for AMD64
  jz Return   
  or esi, CPU_F_AMD64   ; AMD64 supported

  jmp Return

Intel:
  ; Extended CPUID supported, so get extended features
  mov eax, 80000001h
  cpuid

EM64T_test:
  test edx, 20000000h   ; Test for EM64T
  jz Return
  or esi, CPU_F_EM64T   ; EM64T supported

  jmp Return

; Nothing supported
NotSupported:
Return:
  mov eax, esi

  pop edi
  pop esi
  pop edx
  pop ecx
  pop ebx

  ret
