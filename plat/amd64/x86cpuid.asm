;
; Copyright distributed.net 1998-2009 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; x86 Processor identification for rc5 distributed.net effort
; Written by Didier Levet <kakace@distributed.net>, mostly derived from
; x86ident.asm written by Cyrus Patel.
;
; $Id: x86cpuid.asm,v 1.3 2009/12/27 13:52:31 andreasb Exp $
;
; correctly identifies almost every 386+ processor with the
; following exceptions:
; a) The code to use Cyrix Device ID registers for differentiate between a 
;    Cyrix 486, 6x86, etc is in place but disabled. Even so, it would not
;    detect correctly if CPUID is enabled _and_ reassigned.
; b) It may not correctly identify a Rise CPU if cpuid is disabled/reassigned
; c) It identifies an Intel 486 as an AMD 486 (chips are identical) and not
;    vice-versa because the Intel:0400 CPUID is actually a real CPUID value.
;
; x86getid: return i32
;    -1 : The cpuid instruction is supported and shall be used to identify
;         the processor.
;   oth : ID code for processors that don't support the cpuid instruction.
;     0x10003000 = Intel 80386
;     0x10004020 = Intel 80486SX
;     0x6000500F = NexGen i386
;     0x40004000 = AMD i486
;     0x50004040 = Cyrix Media GX
;     0x50004090 = Cyrix 5x86
;     0x50005020 = Cyrix 6x86
;     0x50005040 = Cyrix MediaGX MMX / GXm
;     0x50006000 = Cyrix 6x86mx (M2)
;
;   The ID code is constructed as follows :
;     bits [31, 28] : Vendor code
;     bits [27, 20] : Brand ID field
;     bits [19, 12] : Family
;     bits [11,  4] : Model
;     bits [ 3,  0] : Stepping
;
;-------------------------------------------------------------------------------

BITS 64

; Vendor identifiers :

%define VENDOR_UNKNOWN     00000000h
%define VENDOR_INTEL       10000000h
%define VENDOR_TRANSMETA   20000000h
%define VENDOR_NSC         30000000h
%define VENDOR_AMD         40000000h
%define VENDOR_CYRIX       50000000h
%define VENDOR_NEXGEN      60000000h
%define VENDOR_CENTAUR     70000000h
%define VENDOR_UMC         80000000h
%define VENDOR_RISE        90000000h
%define VENDOR_SIS         A0000000h

; Predefined cpu ID codes :

%define INTEL_i386   10003000h          ; family=3, model=0, stepping=0
%define INTEL_486SX  10004020h          ; family=4, model=2, stepping=0
%define NEXGEN_i386  6000500Fh          ; family=5, model=0, stepping=15
%define AMD_i486     40004000h          ; family=4, model=0, stepping=0

;-------------------------------------------------------------------------------

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

global          x86getid,_x86getid
global          x86cpuid,_x86cpuid
global          x86xgetbv,_x86xgetbv
global          x86ident_haveioperm, _x86ident_haveioperm

__DATASECT__

; We define this variable for the sake of source compatibility with the 32-bit
; version of x86cpuid, but we don't actually use it in the 64-bit version.
_x86ident_haveioperm:                   ; do we have permission to do in/out?
x86ident_haveioperm     dd 0            ; we do on win9x (not NT), win16, dos


;----------------------------------------------------------------------

__CODESECT__

;fastcall input calling convetion:
; rdx = infos
; ecx = function
_x86cpuid:      ; x86cpuid(u32 function, union PageInfos *infos)
x86cpuid:       push    rbx
                push    rsi
                mov     eax, ecx   ; Function number
                mov     rsi, rdx   ; PageInfos *
                xor     ebx, ebx        ; Reset output registers in case they
                xor     ecx, ecx        ; are not modified by the cpuid instr
                xor     edx, edx        ; for some reason...
                cpuid
                mov     [rsi+0], ebx    ; Registers are ordered in such a way
                mov     [rsi+4], edx    ; that the brand string is correctly
                mov     [rsi+8], ecx    ; formatted.
                mov     [rsi+12], eax
                pop     rsi
                pop     rbx
                ret                     ; returns eax


;----------------------------------------------------------------------

; This function always reports that CPUID is available, since we are
; known to be running in 64-bit mode.
_x86getid:
x86getid:       mov     eax, -1         ; cpuid instruction is supported.
                ret

;----------------------------------------------------------------------

_x86xgetbv:     ; x86xgetbv(u32 function, union PageInfos *infos)
x86xgetbv:      push    rdx            ; save address, function number already in ECX
                db      0fh, 01h, 0d0h ; xgetbv
                pop     rcx            ; PageInfos *
                mov     [rcx+4 ], edx
                mov     [rcx+12], eax
                ret                    ; return copy of eax
